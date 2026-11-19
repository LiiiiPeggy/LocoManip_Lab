# Copyright (c) 2025-2026, Junjie Zhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bisect why the D455 fixed-view check saw no marker.

Three questions, answered from one run:

1. does the marker prim actually move when ``XformPrimView.set_world_poses`` is called?
   (read the pose back, and from USD as well)
2. does the camera really look along the direction the mount chain predicts?
   (place a marker on the camera's *measured* forward axis and look for it)
3. is the pre-computed expected pixel reachable at all?
   (project the marker centre with the camera's own intrinsics)

Frames are written for each case so a wrong answer is visible, not inferred.

Run:
    python scripts/ranger_cr10/debug_d455_frame.py
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import yaml

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, enable_cameras=True).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.sim import PinholeCameraCfg  # noqa: E402
from pxr import UsdGeom  # noqa: E402

import LeggedManip_Lab.tasks  # noqa: E402
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.simple_collision_env_cfg import (  # noqa: E402
    RangerCr10TeleopSimpleCollisionEnvCfg,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "outputs/stage413/b/d455_views/debug")
os.makedirs(OUT, exist_ok=True)

IMG_W, IMG_H, F_PX = 640, 480, 372.0
FOCAL_MM, HAPERTURE_MM = 18.6, 32.0
MARKER = 0.06

# Same chain as validate_d455_views.py
LINK6_CAM_POS = np.array([0.0131, -0.0121, 0.1849])  # replaced below by the composed value


def quat_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q, v):
    return quat_mul(quat_mul(q, np.array([0.0, *v])), quat_conj(q))[1:]


def compose(p1, q1, p2, q2):
    return np.asarray(p1) + quat_rotate(np.asarray(q1), np.asarray(p2)), quat_mul(np.asarray(q1), np.asarray(q2))


_mid_p, _mid_q = compose(np.array([0.0, 0.10, 0.14]), quat_from_rpy(0.785398, -1.570796, -2.356194),
                         np.array([0.0, -0.059, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
LINK6_CAM_POS, LINK6_CAM_QUAT = compose(
    _mid_p, _mid_q, np.zeros(3), quat_from_rpy(-1.5707963267948966, 0.0, -1.5707963267948966)
)

cfg = RangerCr10TeleopSimpleCollisionEnvCfg()
cfg.seed = 42
cfg.scene.num_envs = 1
cfg.observations.policy.enable_corruption = False
cfg.commands.ee_pose.debug_vis = False

cfg.scene.d455_camera = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/cr10_Link6/d455_color",
    update_period=0.0,
    update_latest_camera_pose=True,
    width=IMG_W,
    height=IMG_H,
    data_types=["rgb"],
    offset=CameraCfg.OffsetCfg(
        pos=tuple(float(v) for v in LINK6_CAM_POS),
        rot=tuple(float(v) for v in LINK6_CAM_QUAT),
        convention="ros",
    ),
    spawn=PinholeCameraCfg(focal_length=FOCAL_MM, focus_distance=0.35,
                           horizontal_aperture=HAPERTURE_MM,
                           clipping_range=(0.02, 100.0), lock_camera=False),
)
cfg.scene.probe = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/probe",
    spawn=sim_utils.CuboidCfg(
        size=(MARKER, MARKER, MARKER),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        collision_props=None,
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -50.0)),
)

torch.manual_seed(42)
np.random.seed(42)
env = gym.make("RANGER-CR10-TELEOP-SIMPLECOLL", cfg=cfg)
u = env.unwrapped
env.reset()

robot = u.scene["robot"]
camera = u.scene["d455_camera"]
probe = u.scene["probe"]
if hasattr(probe, "_sync_usd_on_fabric_write"):
    probe._sync_usd_on_fabric_write = True

stage = sim_utils.get_current_stage()


def usd_pose(path):
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return None
    xf = UsdGeom.Xformable(prim)
    m = xf.ComputeLocalToWorldTransform(0.0)
    t = m.ExtractTranslation()
    return [float(t[0]), float(t[1]), float(t[2])]


def set_probe(pos_w, quat_w):
    position = torch.tensor(pos_w, dtype=torch.float32, device=u.device).unsqueeze(0)
    orientation = torch.tensor(quat_w, dtype=torch.float32, device=u.device).unsqueeze(0)
    probe.set_world_poses(positions=position, orientations=orientation)


def warm(steps=3):
    zero = torch.zeros(1, u.action_manager.total_action_dim, device=u.device)
    for _ in range(steps):
        u.step(zero)


def grab(name):
    u.sim.render()
    frame = camera.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)
    from PIL import Image
    Image.fromarray(frame).save(os.path.join(OUT, f"{name}.png"))
    rgb = frame[..., :3].astype(np.int16)
    mask = (rgb[..., 0] > 50) & (rgb[..., 0] > 1.6 * rgb[..., 1]) & (rgb[..., 0] > 1.6 * rgb[..., 2])
    count = int(mask.sum())
    if count:
        ys, xs = np.nonzero(mask)
        return {"file": f"{name}.png", "red_pixels": count,
                "u": float(xs.mean()), "v": float(ys.mean()),
                "bbox_px": int(max(xs.max() - xs.min(), ys.max() - ys.min()) + 1)}
    return {"file": f"{name}.png", "red_pixels": 0, "u": None, "v": None, "bbox_px": None}


report = {"link6_cam_pos": LINK6_CAM_POS.tolist(), "link6_cam_quat": LINK6_CAM_QUAT.tolist()}

warm(3)

# -- 1. does the write take effect? ------------------------------------------------------
link6_pos = robot.data.body_pos_w[0, robot.find_bodies("cr10_Link6")[0][0]].detach().cpu().numpy()
link6_quat = robot.data.body_quat_w[0, robot.find_bodies("cr10_Link6")[0][0]].detach().cpu().numpy()
print("[dbg] link6 world pos", link6_pos.tolist(), flush=True)
print("[dbg] link6 world quat", link6_quat.tolist(), flush=True)

# -- 2. camera pose as measured by the sensor -------------------------------------------
cam_pos = camera.data.pos_w[0].detach().cpu().numpy()
cam_quat_world = camera.data.quat_w_world[0].detach().cpu().numpy()
# Camera.data quat_w_world is in the "world" convention: forward +X, up +Z.
fwd_world = quat_rotate(cam_quat_world, np.array([1.0, 0.0, 0.0]))
up_world = quat_rotate(cam_quat_world, np.array([0.0, 0.0, 1.0]))
derived_cam = link6_pos + quat_rotate(link6_quat, LINK6_CAM_POS)
report["camera"] = {
    "pos_w_measured": cam_pos.tolist(),
    "quat_w_measured": cam_quat_world.tolist(),
    "pos_w_derived_from_link6": derived_cam.tolist(),
    "mount_error_m": float(np.linalg.norm(cam_pos - derived_cam)),
    "forward_world": fwd_world.tolist(),
    "up_world": up_world.tolist(),
    "intrinsic": camera.data.intrinsic_matrices[0].detach().cpu().numpy().tolist(),
}
print("[dbg] camera pos_w   ", cam_pos.tolist(), flush=True)
print("[dbg] camera derived ", derived_cam.tolist(), flush=True)
print("[dbg] forward_world  ", fwd_world.tolist(), flush=True)
print("[dbg] intrinsics     ", camera.data.intrinsic_matrices[0].detach().cpu().numpy().tolist(), flush=True)

# -- 3. three probes: measured forward, predicted optical axis, and reversed --------------
pred_axis = quat_rotate(quat_mul(link6_quat, LINK6_CAM_QUAT), np.array([0.0, 0.0, 1.0]))
cases = {
    "probe_measured_forward": cam_pos + 0.35 * fwd_world,
    "probe_predicted_axis": cam_pos + 0.35 * pred_axis,
    "probe_reversed": cam_pos - 0.35 * pred_axis,
}
report["probes"] = {}
for name, pos in cases.items():
    set_probe(pos, cam_quat_world)
    warm(2)
    usd = usd_pose(f"/World/envs/env_0/probe")
    got = probe.get_world_poses()
    got_pos = got[0][0].detach().cpu().numpy().tolist() if isinstance(got, tuple) else None
    shot = grab(name)
    report["probes"][name] = {
        "target_world": list(map(float, pos)),
        "usd_position_after_write": usd,
        "view_position_after_write": got_pos,
        "shot": shot,
    }
    print(f"[dbg] {name}: target {np.round(pos, 3).tolist()} usd {None if usd is None else np.round(usd, 3).tolist()} -> {shot}", flush=True)

with open(os.path.join(OUT, "debug_report.json"), "w") as handle:
    json.dump(report, handle, indent=1)
print(json.dumps(report, indent=1)[:6000], flush=True)
os._exit(0)
