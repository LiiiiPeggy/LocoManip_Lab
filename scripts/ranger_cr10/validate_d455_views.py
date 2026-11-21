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

"""Small frozen-inference D455 fixed-viewpoint observation check.

One robot in the SIMPLECOLL v2 environment, a D455 RGB camera rigidly mounted at the URDF's
``d455_color_optical_frame``, and three pre-registered TCP command poses that stay inside the
trained command box.  Each view is a fresh episode: reset, drive to the commanded pose with
``model_9999`` (deterministic mean action), then hold while the D455 frames are measured.

The target is not part of the training scene, so each view gets a temporary, known-size solid
marker.  The marker is placed on that view's *expected* optical axis, and it is placed in the
**chassis frame** (moved to world after every reset), because the command it is derived from is
chassis-relative and ``reset_base`` randomises the chassis pose.  This means the expected pixel
is the principal point by construction: the run tests reachability, camera consistency and frame
stability, not an independent estimate of the mount.

No training, no parameter updates, no change to any default config.  The camera and the markers
exist only in this script's in-memory evaluation scene.

Pre-registered acceptance (fixed before any run; plan section 48):
    arrival   <= ARRIVAL_TIMEOUT steps at pos <= POS_TOL and orientation <= ORI_TOL
    hold      >= HOLD_STEPS frames, target visible in >= VISIBLE_MIN_RATIO of them,
              centroid offset mean <= PIXEL_OFFSET_MAX px, centroid std <= PIXEL_JITTER_MAX px
    camera    axis error <= AXIS_MAX_DEG; mount consistency (Link6-derived vs Camera.data) <=
              MOUNT_CONSISTENCY_MAX
    episode   no base_contact / bad_orientation termination, no NaN

Run:
    python scripts/ranger_cr10/validate_d455_views.py \
        --checkpoint models/ranger_cr10/model_9999.pt \
        --out-dir outputs/stage413/b/d455_views
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", default="models/ranger_cr10/model_9999.pt")
ap.add_argument("--out-dir", default="outputs/stage413/b/d455_views")
ap.add_argument("--views", default="front,side,top")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--num_envs", type=int, default=1)
ap.add_argument(
    "--marker-mode",
    choices=("world", "hold"),
    default="world",
    help="world = pre-registered protocol: the marker is fixed in the world from the first warm "
         "step, so chassis drift shows up as pixel offset. hold = diagnostic: the marker is "
         "re-registered at the first hold frame, which isolates aim accuracy plus jitter from "
         "the chassis drift that happened before the hold started.",
)
ap.add_argument(
    "--detail-view",
    default="",
    help="Name of one view (e.g. 'top') for which every hold frame also records the base world "
         "pose, the D455 optical-frame world pose, the target's actual world pose, the detected "
         "centroid and the projection of the target centre with the measured camera. Every view "
         "is still executed, so the reset/command random sequence matches the default order.",
)
ap.add_argument(
    "--yaw-log",
    action="store_true",
    help="For --detail-view, also record the command-to-motion chain every step from the warmup "
         "onwards: upper velocity command, raw base action, the action term's post-tanh command, "
         "the post-branch (vx, wz), steering and wheel targets, wheel speeds, base yaw and yaw "
         "rate, EE tracking error and the detected centroid.",
)
ap.add_argument(
    "--zero-wz-after-arrival",
    action="store_true",
    help="Intervention: from the control step after the arrival decision, zero only the wz that "
         "reaches the base controller, leaving vx and the six arm joints under normal policy "
         "control. The chassis is not locked.",
)
ap.add_argument(
    "--deploy-filter",
    action="store_true",
    help="Evaluation-only switch for the deployment zero-velocity filter "
         "(scripts/ranger_cr10/base_command_filter.py), used in the same order as "
         "mujoco/deploy/deploy_mujoco/ranger_cr10/ranger_cr10_deploy.py: deadband the command, "
         "let the policy observe the deadbanded value, then filter_action before the sim step. "
         "Defaults to off, which is the unfiltered baseline.",
)
ap.add_argument(
    "--filter-config",
    default="mujoco/deploy/deploy_mujoco/ranger_cr10/config.yaml",
    help="Deployment YAML whose 'filter' block is read; the effective values are written to the "
         "result JSON. Relative paths resolve against the repository root.",
)
ap.add_argument(
    "--marker-world",
    default="",
    help="JSON written by an earlier run: reuse that run's registered target world pose instead of "
         "registering a new one from the current chassis pose. Needed because the filter changes "
         "the chassis motion from the first step, which would otherwise move the A1 target.",
)
args = ap.parse_args()

app = AppLauncher(headless=True, enable_cameras=True).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.sim import PinholeCameraCfg  # noqa: E402
from isaaclab.utils.math import convert_camera_frame_orientation_convention  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import LeggedManip_Lab.tasks  # noqa: E402
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.simple_collision_env_cfg import (  # noqa: E402
    RangerCr10TeleopSimpleCollisionEnvCfg,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.pose_command_teleop import (  # noqa: E402
    TCP_DEFAULT_POS_B,
    TCP_DEFAULT_QUAT_B,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.tcp import (  # noqa: E402
    TCP_ANCHOR_BODY,
    TCP_OFFSET_POS,
    TCP_OFFSET_QUAT,
    tcp_pose_b,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(ROOT, args.checkpoint)
OUT_DIR = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)

# ---------------------------------------------------------------------------------------
# Pre-registered thresholds and view definition.  Nothing here depends on a run result.
# ---------------------------------------------------------------------------------------
WARMUP_STEPS = 5
ARRIVAL_TIMEOUT = 80
HOLD_STEPS = 60
POS_TOL = 0.03                 # m
ORI_TOL = 0.10                 # rad
VISIBLE_MIN_RATIO = 0.90
PIXEL_OFFSET_MAX = 20.0        # px, mean centroid distance from the principal point
PIXEL_JITTER_MAX = 5.0         # px, std of the centroid over the hold window
AXIS_MAX_DEG = 10.0
MOUNT_CONSISTENCY_MAX = 0.005  # m, Link6-derived camera position vs Camera.data.pos_w
MIN_VISIBLE_PIXELS = 30

# D455 colour camera, nominal and deliberately conservative: 640x480, square pixels, f = 372 px.
# VFOV then equals the real colour camera's 65.6 deg; HFOV is 81.3 deg versus the real ~86 deg,
# i.e. narrower, which can only make the visibility test stricter.
# The camera model, the three view definitions and the target geometry are shared with
# validate_d455_continuous.py; see d455_eval_common.py.
from d455_eval_common import (  # noqa: E402
    ARRIVAL_TIMEOUT,
    AXIS_MAX_DEG,
    DEFAULT_AXIS_ERR_DEG,
    DEFAULT_TCP_POS,
    DEFAULT_TCP_QUAT,
    F_PX,
    FOCAL_LENGTH_MM,
    HOLD_STEPS,
    HORIZONTAL_APERTURE_MM,
    IMG_H,
    IMG_W,
    LINK6_CAM_POS,
    LINK6_CAM_QUAT,
    MARKER_DISTANCE_M,
    MARKER_SIZE_M,
    MIN_VISIBLE_PIXELS,
    MOUNT_CONSISTENCY_MAX,
    ORI_TOL,
    PIXEL_JITTER_MAX,
    PIXEL_OFFSET_MAX,
    POS_TOL,
    VISIBLE_MIN_RATIO,
    VIEWS,
    WARMUP_STEPS,
    axis_angle_deg,
    compose,
    detect,
    invert,
    project_point,
    quat_conj,
    quat_from_rpy,
    quat_mul,
    quat_rotate,
    tcp_to_camera,
    view_geometry,
)

SELECTED = [name.strip() for name in args.views.split(",") if name.strip()]
GEOMETRY = {name: view_geometry(name) for name in SELECTED}

# ---------------------------------------------------------------------------------------
# Evaluation scene: camera + one temporary marker per view.
# ---------------------------------------------------------------------------------------
cfg = RangerCr10TeleopSimpleCollisionEnvCfg()
cfg.seed = args.seed
cfg.scene.num_envs = args.num_envs
cfg.commands.ee_pose.debug_vis = False
cfg.observations.policy.enable_corruption = False

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
    spawn=PinholeCameraCfg(
        focal_length=FOCAL_LENGTH_MM,
        focus_distance=MARKER_DISTANCE_M,
        horizontal_aperture=HORIZONTAL_APERTURE_MM,
        clipping_range=(0.02, 100.0),
        # the camera is mounted on a moving link; do not lock it to the viewport transform
        lock_camera=False,
    ),
)

for _name, _geom in GEOMETRY.items():
    setattr(
        cfg.scene,
        f"marker_{_name}",
        AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/marker_{_name}",
            spawn=sim_utils.CuboidCfg(
                size=(MARKER_SIZE_M, MARKER_SIZE_M, MARKER_SIZE_M),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tuple(_geom["marker_color"])
                ),
                # Visual only: no rigid body and no collider. With RigidBodyAPI applied, PhysX
                # owns the transform and pushes its (unchanged) pose back every step, which
                # silently undoes any runtime xform write.
                rigid_props=None,
                collision_props=None,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
        ),
    )

torch.manual_seed(args.seed)
np.random.seed(args.seed)
env = gym.make("RANGER-CR10-TELEOP-SIMPLECOLL", cfg=cfg)
u = env.unwrapped
with open(os.path.join(os.path.dirname(CKPT), "params", "agent.yaml")) as handle:
    train_cfg = yaml.safe_load(handle)
train_cfg["device"] = str(u.device)
train_cfg["seed"] = args.seed
wrapper = RslRlVecEnvWrapper(env, clip_actions=train_cfg.get("clip_actions"))
runner = OnPolicyRunner(wrapper, train_cfg, log_dir=None, device=u.device)
runner.load(CKPT)
policy = runner.get_inference_policy(device=u.device)

robot = u.scene["robot"]
camera = u.scene["d455_camera"]
contact_sensor = u.scene["contact_forces"]
body_names = list(contact_sensor.body_names)
wheel_rows = {i for i, n in enumerate(body_names) if n.endswith("_wheel_link")}
non_wheel_rows = [i for i in range(len(body_names)) if i not in wheel_rows]
anchor_id = robot.find_bodies(TCP_ANCHOR_BODY)[0][0]
base_id = robot.find_bodies("base_link")[0][0]
dt = u.step_dt
term_names = list(u.termination_manager.active_terms)
markers = {name: u.scene[f"marker_{name}"] for name in SELECTED}
# XformPrimView writes to Fabric when fabric is enabled, and its constructor default does not
# mirror those writes into USD. Ask it to mirror, so the prim is at the same place for the RTX
# renderer and for any USD consumer.
for _view in markers.values():
    if hasattr(_view, "_sync_usd_on_fabric_write"):
        _view._sync_usd_on_fabric_write = True

_current = {
    "pos": torch.tensor(GEOMETRY[SELECTED[0]]["tcp_pos_b"], dtype=torch.float32),
    "quat": torch.tensor(GEOMETRY[SELECTED[0]]["tcp_quat_b"], dtype=torch.float32),
}


def apply_current_command() -> None:
    ee = u.command_manager.get_term("ee_pose")
    ee.pose_command_b[:, :3] = _current["pos"].to(u.device)
    ee.pose_command_b[:, 3:] = _current["quat"].to(u.device)
    vel = u.command_manager.get_term("base_velocity")
    # Deploy order: the policy observes the *deadbanded* chassis command, so write the filtered
    # value here rather than a literal zero. With the operator command at exactly zero the value
    # is zero either way, but the observation must come from the same path the filter uses.
    vel.vel_command_b[:, 0] = filter_state["vx"]
    vel.vel_command_b[:, 1] = 0.0
    vel.vel_command_b[:, 2] = filter_state["wz"]
    vel.is_standing_env[:] = True


def _reset_commands(env_ids=None):
    apply_current_command()
    return {}


def _compute_commands(dt):
    apply_current_command()


u.command_manager.reset = _reset_commands
u.command_manager.compute = _compute_commands

captured: dict = {}
_orig_reward = u.reward_manager.compute


def _capture(dt, _original=_orig_reward):
    out = _original(dt)
    captured["terms"] = {name: int(u.termination_manager.get_term(name).sum().item()) for name in term_names}
    captured["dones"] = int(
        (u.termination_manager.terminated | u.termination_manager.time_outs).sum().item()
    )
    forces = contact_sensor.data.net_forces_w
    if forces is not None and non_wheel_rows:
        mag = torch.linalg.norm(forces[:, non_wheel_rows], dim=-1)
        hit = mag > 100.0
        captured["contact_env_steps"] = int(hit.any(dim=-1).sum().item())
        captured["contact_body_events"] = int(hit.sum().item())
        captured["contact_by_body"] = {
            body_names[idx]: int(hit[:, local].sum().item())
            for local, idx in enumerate(non_wheel_rows)
            if int(hit[:, local].sum().item()) > 0
        }
    else:
        captured["contact_env_steps"] = 0
        captured["contact_body_events"] = 0
        captured["contact_by_body"] = {}
    return out


u.reward_manager.compute = _capture


def base_pose():
    """Chassis (base_link) pose in world; the command frame is base_link, not the articulation root."""
    return (robot.data.body_pos_w[0, base_id].detach().cpu().numpy(),
            robot.data.body_quat_w[0, base_id].detach().cpu().numpy())


def _set_marker_pose(marker, position_np, orientation_np):
    position = torch.tensor(position_np, dtype=torch.float32, device=u.device).unsqueeze(0)
    orientation = torch.tensor(orientation_np, dtype=torch.float32, device=u.device).unsqueeze(0)
    # AssetBaseCfg resolves to an XformPrimView in this Isaac Lab version (not a RigidObject),
    # so poses go through the view API; keep the rigid-object path as a fallback.
    if hasattr(marker, "set_world_poses"):
        marker.set_world_poses(positions=position, orientations=orientation)
    elif hasattr(marker, "write_root_pose_to_sim"):
        marker.write_root_pose_to_sim(torch.cat([position, orientation], dim=-1))
    else:
        raise RuntimeError("marker asset exposes neither set_world_poses nor write_root_pose_to_sim")


# -- reusable world targets, one per view ------------------------------------------
# Two input shapes are accepted:
#   * a previous run's result JSON: {"views": {name: {"marker_registered_world": [x, y, z]}}}
#     -> position only; the marker keeps the chassis orientation at placement time
#   * a preset: {"targets": {name: {"pos": [...], "quat": [w, x, y, z]}}} or {name: [...]}
#     -> position and orientation; a cube's silhouette depends on its orientation, so a paired
#        comparison between two runs must use the same quaternion, not each run's chassis attitude.
# The active view is looked up by name; a missing view is a hard error, never a silent fallback
# to another view's target.
preset_targets: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
preset_source = None
if args.marker_world:
    _mw_path = Path(args.marker_world) if Path(args.marker_world).is_absolute() else Path(ROOT) / args.marker_world
    _mw = json.loads(_mw_path.read_text())
    preset_source = str(_mw_path)
    if isinstance(_mw, dict) and "targets" in _mw:
        for _name, _spec in _mw["targets"].items():
            if isinstance(_spec, dict):
                preset_targets[_name] = (np.array(_spec["pos"], dtype=float),
                                         None if _spec.get("quat") is None else np.array(_spec["quat"], dtype=float))
            else:
                preset_targets[_name] = (np.array(_spec, dtype=float), None)
    elif isinstance(_mw, dict) and "views" in _mw:
        for _name, _view in _mw["views"].items():
            for _key in ("marker_registered_world", "marker_pos_world_at_last_hold",
                         "expected_marker_pos_world_at_reset"):
                if _view.get(_key):
                    preset_targets[_name] = (np.array(_view[_key], dtype=float), None)
                    break
    else:
        raise SystemExit(f"{_mw_path}: expected a results JSON with 'views' or a preset with 'targets'")
    if not preset_targets:
        raise SystemExit(f"{_mw_path}: no per-view target found")
    print(f"[target] loaded {sorted(preset_targets)} from {preset_source}", flush=True)

# Per-view lookups resolved at the start of each view.
fixed_marker_world = None
fixed_marker_quat = None


def place_markers(active: str):
    """Place the active view's marker at base_pose * marker_pos_b and park the others far away.

    With ``--marker-world`` the active marker is placed at the supplied world point instead, so
    two runs that move the chassis differently still observe the same target. The other views'
    markers would otherwise sit inside this view's field of view and could occlude the target;
    parking them 100 m below the floor keeps them out of the frame without deleting them.
    """
    base_pos, base_quat = base_pose()
    for name, geom in GEOMETRY.items():
        if name == active:
            if fixed_marker_world is not None:
                world_pos = fixed_marker_world
                world_quat = base_quat if fixed_marker_quat is None else fixed_marker_quat
            else:
                world_pos = base_pos + quat_rotate(base_quat, np.array(geom["marker_pos_b"]))
                world_quat = base_quat
        else:
            world_pos = base_pos + np.array([0.0, 0.0, -100.0])
            world_quat = base_quat
        _set_marker_pose(markers[name], world_pos, world_quat)
        readback = marker_world(name)
        print(
            f"[place] {name}: target={np.round(world_pos, 3).tolist()} "
            f"quat={np.round(world_quat, 4).tolist()} readback={np.round(readback, 3).tolist()}",
            flush=True,
        )


def camera_pose_b():
    """Measured camera pose in the chassis frame, from Camera.data (world -> base)."""
    base_pos, base_quat = base_pose()
    cam_pos = camera.data.pos_w[0].detach().cpu().numpy()
    cam_quat_world = camera.data.quat_w_world[0].detach().cpu().numpy()
    cam_quat_ros = convert_camera_frame_orientation_convention(
        torch.tensor(cam_quat_world, dtype=torch.float32).unsqueeze(0), origin="world", target="ros"
    )[0].numpy()
    qc = quat_conj(base_quat)
    return quat_rotate(qc, cam_pos - base_pos), quat_mul(qc, cam_quat_ros), cam_pos


def camera_pose_w_ros():
    """Camera pose in the world frame, expressed in the ROS optical convention."""
    pos = camera.data.pos_w[0].detach().cpu().numpy()
    quat = convert_camera_frame_orientation_convention(
        camera.data.quat_w_world, origin="world", target="ros"
    )[0].detach().cpu().numpy()
    return pos, quat


def marker_world(name: str) -> np.ndarray:
    poses = markers[name].get_world_poses()
    return poses[0][0].detach().cpu().numpy()


def read_frame() -> np.ndarray:
    if "rgb" not in camera.data.output:
        camera.update(dt)
    if "rgb" not in camera.data.output:
        raise RuntimeError(
            "camera has no 'rgb' buffer after update; the render products are not producing frames "
            "(check that the app was launched with enable_cameras=True)"
        )
    return camera.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)


def save_frame(frame: np.ndarray, path: str) -> str:
    try:
        from PIL import Image
        Image.fromarray(frame).save(path)
        return "png(PIL)"
    except Exception:
        pass
    try:
        import torchvision.utils as vutils
        vutils.save_image(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0, path)
        return "png(torchvision)"
    except Exception:
        np.save(path.replace(".png", ".npy"), frame)
        return "npy"


os.makedirs(OUT_DIR, exist_ok=True)
# -- deployment zero-velocity filter (evaluation only, default off) -------------------
deploy_filter = None
filter_effective = {}
if args.deploy_filter:
    _cfg_path = Path(args.filter_config) if Path(args.filter_config).is_absolute() else Path(ROOT) / args.filter_config
    _deploy_cfg = yaml.safe_load(_cfg_path.read_text())
    _fcfg = _deploy_cfg["filter"]
    sys.path.insert(0, str(Path(ROOT) / "scripts" / "ranger_cr10"))
    from base_command_filter import BaseCommandFilter  # noqa: E402

    deploy_filter = BaseCommandFilter(
        vx_deadband=float(_fcfg["vx_deadband"]) if _fcfg.get("enabled", True) else 0.0,
        wz_deadband=float(_fcfg["wz_deadband"]) if _fcfg.get("enabled", True) else 0.0,
        lock_output=bool(_fcfg.get("lock_output", False)) and bool(_fcfg.get("enabled", True)),
        hysteresis=float(_fcfg.get("hysteresis", 0.0)),
    )
    filter_effective = {
        "config_file": str(_cfg_path),
        "enabled_in_config": bool(_fcfg.get("enabled", False)),
        "vx_deadband": deploy_filter.vx_deadband,
        "wz_deadband": deploy_filter.wz_deadband,
        "lock_output": deploy_filter.lock_output,
        "hysteresis": deploy_filter.hysteresis,
        "enabled": deploy_filter.enabled,
        "call_order": "filter_command -> policy(obs with deadbanded command) -> filter_action -> env.step",
    }

# The upper velocity command is exactly zero in this evaluation, so the deadbanded value is also
# zero; it is still computed through the deployment path so the log proves the path ran.
filter_state = {"vx": 0.0, "wz": 0.0}
last_step = {"raw": None, "used": None}


def policy_step(obs_now):
    """One control step in deployment order: deadband -> policy -> filter_action -> sim step."""
    vx_t = torch.zeros(1, device=u.device)
    wz_t = torch.zeros(1, device=u.device)
    if deploy_filter is not None:
        vx_t, wz_t = deploy_filter.filter_command(vx_t, wz_t)
        filter_state["vx"] = float(vx_t[0].item())
        filter_state["wz"] = float(wz_t[0].item())
    with torch.no_grad():
        raw = policy(obs_now)
        used = raw if deploy_filter is None else deploy_filter.filter_action(raw, vx_t, wz_t)
        obs_next, _rew, _done, _extra = wrapper.step(used)
    last_step["raw"] = raw
    last_step["used"] = used
    return obs_next

results = {
    "checkpoint": CKPT,
    "checkpoint_sha256": hashlib.sha256(open(CKPT, "rb").read()).hexdigest(),
    "env_id": "RANGER-CR10-TELEOP-SIMPLECOLL",
    "env_asset": "ranger_cr10_simple_v2_flat.usd, self-collisions ON",
    "inference": "deterministic mean, observation corruption off",
    "seed": args.seed,
    "camera": {
        "prim_path": "{ENV_NAMESPACE}/Robot/cr10_Link6/d455_color",
        "mount_source": "ranger_cr10_simple_v2.urdf: cr10_Link6 -> gripper_base_link (identity) -> "
                        "d455_link -> d455_color_frame -> d455_color_optical_frame",
        "link6_to_camera_pos_m": LINK6_CAM_POS.tolist(),
        "link6_to_camera_quat_wxyz": LINK6_CAM_QUAT.tolist(),
        "convention": "ros (optical: +Z forward, +X right, +Y down)",
        "image_wh": [IMG_W, IMG_H],
        "focal_length_mm": FOCAL_LENGTH_MM,
        "horizontal_aperture_mm": HORIZONTAL_APERTURE_MM,
        "f_px": F_PX,
        "hfov_deg": math.degrees(2 * math.atan(IMG_W / 2 / F_PX)),
        "vfov_deg": math.degrees(2 * math.atan(IMG_H / 2 / F_PX)),
        "intrinsics_note": "square pixels; VFOV matches the D455 colour camera, HFOV is narrower "
                           "than the real ~86 deg, which only makes visibility stricter",
        "default_pose_axis_vs_base_x_deg": DEFAULT_AXIS_ERR_DEG,
    },
    "protocol": {
        "warmup_steps": WARMUP_STEPS,
        "arrival_timeout_steps": ARRIVAL_TIMEOUT,
        "hold_steps": HOLD_STEPS,
        "pos_tol_m": POS_TOL,
        "ori_tol_rad": ORI_TOL,
        "visible_min_ratio": VISIBLE_MIN_RATIO,
        "pixel_offset_max_px": PIXEL_OFFSET_MAX,
        "pixel_jitter_max_px": PIXEL_JITTER_MAX,
        "axis_max_deg": AXIS_MAX_DEG,
        "mount_consistency_max_m": MOUNT_CONSISTENCY_MAX,
        "min_visible_pixels": MIN_VISIBLE_PIXELS,
        "declared_before_run": True,
        "marker_frame": "chassis frame (markers are moved to world after every reset)",
        "marker_mode": args.marker_mode,
        "marker_mode_note": "world = pre-registered protocol (fixed world target); hold = diagnostic "
                            "re-registration at the first hold frame",
        "deploy_filter": filter_effective if deploy_filter is not None else {"enabled": False},
    },
    "views": {},
}

detail_records: list[dict] = []
detail_intrinsics = None
yaw_records: list[dict] = []

# -- command-to-motion chain access ------------------------------------------------
base_term = u.action_manager.get_term("base_vel")
base_cfg = base_term.cfg
wz_state = {"suppress": False, "wz_at_suppression": None}
_original_apply = base_term.apply_actions


def _apply_with_optional_wz_zero():
    """Optionally zero only the wz that reaches the base controller.

    ``apply_actions`` reads the post-tanh command and converts (vx, wz) into steering and wheel
    targets, so zeroing the second component here removes the turn request while leaving vx and
    every arm joint command untouched. The chassis stays under physics, not locked.
    """
    wz_state["wz_at_suppression"] = float(base_term._command[0, 1].item())
    if wz_state["suppress"]:
        base_term._command[:, 1] = 0.0
    _original_apply()


base_term.apply_actions = _apply_with_optional_wz_zero




def record_chain(phase: str, steps_now: int, det, pos_err: float, target_marked: str = "") -> None:
    """One row of the command-to-motion chain, for the detail view only.

    The command the operator sends is ``[vx, vy, wz]``; vy is unused by this platform, so the
    recorded pair is items 0 and 2. The chassis yaw rate is the world-frame Z component of the
    ``base_link`` angular velocity, taken as a scalar *before* any conversion — the earlier
    version indexed the batch and then called ``.item()`` on the remaining 3-vector, which raised
    at the first recorded step.
    """
    if not args.yaw_log or not yaw_records_allowed[0]:
        return
    bp, bq = base_pose()
    yaw = math.degrees(math.atan2(*(quat_rotate(bq, np.array([1.0, 0.0, 0.0]))[[1, 0]])))
    yaw_rate_dps = float(robot.data.body_ang_vel_w[0, base_id, 2].item()) * 180.0 / math.pi
    vel = u.command_manager.get_term("base_velocity")
    upper = vel.vel_command_b[0].detach().cpu().numpy()   # [vx, vy, wz]
    cmd = base_term._command[0].detach().cpu().numpy()
    vx, wz = float(cmd[0]), float(cmd[1])
    abs_wz = abs(wz)
    radius = (abs(vx) / abs_wz) if abs_wz > 1e-4 else 1e6
    spinning = abs_wz > 1e-4 and (radius < float(base_cfg.min_turn_radius) or abs(vx) < 1e-3)
    if spinning:
        wz = float(np.clip(wz, -float(base_cfg.max_spin_rate), float(base_cfg.max_spin_rate)))
        vx = 0.0
    raw = last_step["raw"]
    used = last_step["used"]
    raw_np = None if raw is None else raw[0].detach().cpu().numpy()
    used_np = None if used is None else used[0].detach().cpu().numpy()
    wheel_sign = torch.as_tensor(base_cfg.wheel_sign, device=u.device, dtype=torch.float32)
    wheel_actual = robot.data.joint_vel[0, base_term._wheel_ids]
    wheel_target = base_term._wheel_target[0]
    yaw_records.append({
        "phase": phase,
        "steps_since_reset": int(steps_now),
        "sim_time_s": float(getattr(u.sim, "current_time", steps_now * u.step_dt)),
        "target_marked_at": target_marked,
        "upper_cmd_vx_wz": [float(upper[0]), float(upper[2])],
        "upper_cmd_vy": float(upper[1]),
        "upper_cmd_standing": bool(vel.is_standing_env[0].item()),
        "filter_enabled": deploy_filter is not None,
        "filter_effective_vx_wz": [filter_state["vx"], filter_state["wz"]],
        "action_base_raw": None if raw_np is None else [float(raw_np[0]), float(raw_np[1])],
        "action_base_filtered": None if used_np is None else [float(used_np[0]), float(used_np[1])],
        "arm_actions_identical_before_after_filter": (
            None if raw_np is None or used_np is None else bool(np.allclose(raw_np[2:], used_np[2:], atol=0.0))
        ),
        "term_command_vx_wz": [float(cmd[0]), float(cmd[1])],
        "term_command_after_spin": [float(vx), float(wz)],
        "wz_zeroed": bool(wz_state["suppress"]),
        "steering_target": [float(v) for v in base_term._steering_target[0].detach().cpu().numpy()],
        "wheel_target": [float(v) for v in wheel_target.detach().cpu().numpy()],
        "wheel_target_signed": [float(v) for v in (wheel_target * wheel_sign).detach().cpu().numpy()],
        "steering_actual": [float(v) for v in robot.data.joint_pos[0, base_term._steering_ids].detach().cpu().numpy()],
        "wheel_actual": [float(v) for v in wheel_actual.detach().cpu().numpy()],
        "wheel_actual_signed": [float(v) for v in (wheel_actual * wheel_sign).detach().cpu().numpy()],
        "base_pos_w": [float(v) for v in bp],
        "base_yaw_deg": yaw,
        "base_yaw_rate_dps": yaw_rate_dps,
        "ee_pos_err_m": float(pos_err),
        "u": det.get("u"),
        "v": det.get("v"),
    })


yaw_records_allowed = [False]

for name in SELECTED:
    geom = GEOMETRY[name]
    _current["pos"] = torch.tensor(geom["tcp_pos_b"], dtype=torch.float32)
    _current["quat"] = torch.tensor(geom["tcp_quat_b"], dtype=torch.float32)
    obs, _ = wrapper.reset()
    u.episode_length_buf[:] = 0          # a view is shorter than an episode; no timeout mid-hold
    base_pos, base_quat = base_pose()
    base_pos_at_reset = base_pos.copy()
    expected_cam_world = base_pos + quat_rotate(base_quat, np.array(geom["expected_camera_pos_b"]))
    expected_marker_world = base_pos + quat_rotate(base_quat, np.array(geom["marker_pos_b"]))

    frames_dir = os.path.join(OUT_DIR, name)
    os.makedirs(frames_dir, exist_ok=True)
    saved, fail_saved = [], []
    hold_records = []
    arrival_step = None
    steps = 0
    marker_replaces = 0
    wz_state["suppress"] = False          # do not leak the post-arrival switch across views
    marker_registered_world = None
    marker_registered_quat = None

    yaw_records_allowed[0] = bool(args.detail_view) and name == args.detail_view and args.yaw_log

    # Reset state of this view, recorded before any step, so two runs can be shown to start from
    # the same place even though their trajectories diverge immediately.
    _init_bp, _init_bq = base_pose()
    view_initial_state = {
        "base_pos_w": [float(v) for v in _init_bp],
        "base_quat_w": [float(v) for v in _init_bq],
        "joint_pos_digest": hashlib.sha256(
            robot.data.joint_pos[0].detach().cpu().numpy().round(6).tobytes()
        ).hexdigest(),
        "tcp_pos_b": [
            float(v) for v in tcp_pose_b(robot, anchor_id, base_id)[0][0].detach().cpu().numpy()
        ],
    }

    target_mark = ""
    # Resolve this view's preset before anything is placed: a missing view must fail here, not
    # silently fall back to another view's target.
    if args.marker_world:
        if name not in preset_targets:
            raise SystemExit(
                f"--marker-world {preset_source} has no target for view '{name}'; "
                f"available views: {sorted(preset_targets)}"
            )
        fixed_marker_world, fixed_marker_quat = preset_targets[name]
        fixed_marker_world = np.array(fixed_marker_world, dtype=float)

    for _warm in range(WARMUP_STEPS):
        obs = policy_step(obs)
        steps += 1
        if _warm == 0:
            # The first step after reset re-applies each extra asset's init_state, so a marker
            # placed before it is silently put back at (0, 0, -10). Place after that step, from
            # the base pose at that moment: the target is a fixed world point from here on, which
            # is what "observe a fixed object" means, and any base drift then shows up as a pixel
            # offset instead of being hidden.
            base_pos, base_quat = base_pose()
            if fixed_marker_world is not None:
                expected_marker_world = np.array(fixed_marker_world, dtype=float)
                target_mark = "reused-from-file"
            else:
                expected_marker_world = base_pos + quat_rotate(base_quat, np.array(geom["marker_pos_b"]))
                target_mark = "registered-from-chassis"
            place_markers(name)
            marker_registered_world = np.array(marker_world(name), dtype=float)
            marker_registered_quat = (
                base_quat if (fixed_marker_world is None or fixed_marker_quat is None) else fixed_marker_quat
            )
            print(f"[view {name}] target {target_mark} at {np.round(marker_registered_world, 4).tolist()} "
                  f"quat={np.round(marker_registered_quat, 4).tolist()}",
                  flush=True)
        if yaw_records_allowed[0]:
            tcp_pos_now, _ = tcp_pose_b(robot, anchor_id, base_id)
            record_chain(
                "warmup", steps,
                detect(read_frame(), geom["marker_color"]),
                float(torch.linalg.norm(tcp_pos_now[0] - torch.tensor(geom["tcp_pos_b"], device=u.device)).item()),
                target_mark,
            )

    while steps < WARMUP_STEPS + ARRIVAL_TIMEOUT:
        obs = policy_step(obs)
        steps += 1
        tcp_pos, tcp_quat = tcp_pose_b(robot, anchor_id, base_id)
        pos_err = float(torch.linalg.norm(tcp_pos[0] - torch.tensor(geom["tcp_pos_b"], device=u.device)).item())
        q_cur = tcp_quat[0].detach().cpu().numpy()
        ori_err = 2.0 * math.acos(min(1.0, abs(float(np.dot(q_cur, np.array(geom["tcp_quat_b"]))))))
        if yaw_records_allowed[0]:
            record_chain("arrival", steps, detect(read_frame(), geom["marker_color"]), pos_err, target_mark)
        if pos_err <= POS_TOL and ori_err <= ORI_TOL:
            arrival_step = steps
            break

    if args.marker_mode == "hold" and arrival_step is not None:
        # Diagnostic protocol: register the target against the chassis as it is at the end of the
        # arrival phase, *before* the first hold frame is rendered, so the offset measures aim
        # plus jitter rather than the drift that happened while the arm was still moving.
        base_pos_now, base_quat_now = base_pose()
        expected_marker_world = base_pos_now + quat_rotate(base_quat_now, np.array(geom["marker_pos_b"]))
        place_markers(name)
        marker_replaces += 1
        base_pos_at_reset = base_pos_now.copy()
        print(f"[view {name}] marker re-registered at hold start (diagnostic mode)", flush=True)

    if args.zero_wz_after_arrival and arrival_step is not None:
        # Intervention starts at the control step after the arrival decision, i.e. the first hold
        # step. Only wz is removed; vx and the six arm joints keep their normal policy commands.
        wz_state["suppress"] = True
        print(f"[view {name}] wz suppressed from the first hold step", flush=True)

    for hold_idx in range(HOLD_STEPS if arrival_step is not None else 0):
        obs = policy_step(obs)
        steps += 1
        frame = read_frame()
        det = detect(frame, geom["marker_color"])
        tcp_pos, tcp_quat = tcp_pose_b(robot, anchor_id, base_id)
        pos_err = float(torch.linalg.norm(tcp_pos[0] - torch.tensor(geom["tcp_pos_b"], device=u.device)).item())
        q_cur = tcp_quat[0].detach().cpu().numpy()
        ori_err = 2.0 * math.acos(min(1.0, abs(float(np.dot(q_cur, np.array(geom["tcp_quat_b"]))))))
        cam_pos_b, cam_quat_b, cam_pos_w = camera_pose_b()
        axis_err = axis_angle_deg(cam_quat_b, np.array(geom["expected_camera_quat_b"]))
        if yaw_records_allowed[0]:
            record_chain("hold", steps, det, pos_err, target_mark)
        if hold_idx == 0 and args.marker_mode == "world":
            drift = float(np.linalg.norm(marker_world(name) - expected_marker_world))
            if drift > 1e-3:
                place_markers(name)
                marker_replaces += 1
                print(f"[view {name}] marker re-placed at hold 0 (drift {drift * 1000:.2f} mm)", flush=True)
        if hold_idx in (0, 1, HOLD_STEPS - 1):
            intr = camera.data.intrinsic_matrices[0].detach().cpu().numpy()
            cp, cq = camera_pose_w_ros()
            mw = marker_world(name)
            print(
                f"[view {name}] hold {hold_idx}: marker_w={np.round(mw, 3).tolist()} "
                f"expected={np.round(expected_marker_world, 3).tolist()} "
                f"cam_w={np.round(cp, 3).tolist()} proj={project_point(cp, cq, intr, mw)} "
                f"proj_expected={project_point(cp, cq, intr, expected_marker_world)}",
                flush=True,
            )

        if name == args.detail_view:
            # Full pose record for one view only, so the motion can be reproduced offline:
            # the target is fixed in the world, the chassis moves, and the camera moves relative
            # to the chassis because the arm moves. Everything needed to recompose those four
            # cases is captured here, at the same instant as the rendered frame.
            intr = camera.data.intrinsic_matrices[0].detach().cpu().numpy()
            cam_w_pos, cam_w_quat = camera_pose_w_ros()
            target_w = marker_world(name)
            base_w_pos, base_w_quat = base_pose()
            detail_records.append({
                "hold_index": int(hold_idx),
                "steps_since_reset": int(steps),
                "arrival_step": arrival_step,
                "sim_time_s": float(getattr(u.sim, "current_time", steps * u.step_dt)),
                "base_pos_w": base_w_pos.tolist(),
                "base_quat_w": base_w_quat.tolist(),
                "cam_pos_w": cam_w_pos.tolist(),
                "cam_quat_w_ros": cam_w_quat.tolist(),
                "target_pos_w": target_w.tolist(),
                "target_registered_w": expected_marker_world.tolist(),
                "u": det["u"],
                "v": det["v"],
                "du": None if det["u"] is None else float(det["u"] - IMG_W / 2.0),
                "dv": None if det["v"] is None else float(det["v"] - IMG_H / 2.0),
                "pixels": det["count"],
                "extent_px": det["extent_px"],
                "visible": bool(
                    det["count"] >= MIN_VISIBLE_PIXELS and det["extent_px"] is not None
                    and det["extent_px"] < 0.5 * IMG_W
                ),
                "projected_center_uv": project_point(cam_w_pos, cam_w_quat, intr, target_w),
            })
            detail_intrinsics = intr.tolist()

        link6_w = robot.data.body_pos_w[0, anchor_id].detach().cpu().numpy()
        link6_q = robot.data.body_quat_w[0, anchor_id].detach().cpu().numpy()
        derived_cam_w = link6_w + quat_rotate(link6_q, LINK6_CAM_POS)
        mount_err = float(np.linalg.norm(cam_pos_w - derived_cam_w))

        visible = (det["count"] >= MIN_VISIBLE_PIXELS and det["extent_px"] is not None
                   and det["extent_px"] < 0.5 * IMG_W)
        hold_records.append({
            "hold_index": hold_idx,
            "pos_err_m": pos_err,
            "ori_err_rad": ori_err,
            "cam_pos_err_m": float(np.linalg.norm(cam_pos_b - np.array(geom["expected_camera_pos_b"]))),
            "cam_axis_err_deg": axis_err,
            "mount_consistency_m": mount_err,
            "visible": bool(visible),
            "pixels": det["count"],
            "u": det["u"],
            "v": det["v"],
            "extent_px": det["extent_px"],
            "du": None if det["u"] is None else det["u"] - IMG_W / 2.0,
            "dv": None if det["v"] is None else det["v"] - IMG_H / 2.0,
            "term_counts": dict(captured.get("terms", {})),
            "contact_env_steps": int(captured.get("contact_env_steps", 0)),
            "contact_body_events": int(captured.get("contact_body_events", 0)),
            # The base velocity command is zero, but the policy still owns the chassis
            # channels; drift is recorded (not part of the acceptance rule) so a pixel
            # offset can be told apart from a stale target. Yaw matters more than the
            # translation for a target 0.35 m away: 0.35 m * sin(1 deg) = 6 mm = 6.5 px.
            "base_drift_xy_m": float(np.linalg.norm(base_pose()[0][:2] - base_pos_at_reset[:2])),
            "base_pos_w": [float(v) for v in base_pose()[0]],
            "base_yaw_deg": float(np.degrees(math.atan2(
                *(quat_rotate(base_pose()[1], np.array([1.0, 0.0, 0.0]))[[1, 0]])
            ))),
        })
        if not saved:
            path = os.path.join(frames_dir, f"{name}_hold_000.png")
            saved.append({"path": path, "format": save_frame(frame, path)})
        if not visible and len(fail_saved) < 3:
            path = os.path.join(frames_dir, f"{name}_fail_hold_{hold_idx:03d}.png")
            fail_saved.append({"path": path, "format": save_frame(frame, path)})

    ratios = [math.hypot(r["du"], r["dv"]) for r in hold_records if r["du"] is not None]
    us = [r["u"] for r in hold_records if r["u"] is not None]
    vs = [r["v"] for r in hold_records if r["v"] is not None]
    visible_count = sum(1 for r in hold_records if r["visible"])
    base_contact = sum(r["term_counts"].get("base_contact", 0) for r in hold_records)
    bad_orientation = sum(r["term_counts"].get("bad_orientation", 0) for r in hold_records)
    summary = {
        "geometry": geom,
        "expected_camera_pos_world_at_reset": expected_cam_world.tolist(),
        "expected_marker_pos_world_at_reset": expected_marker_world.tolist(),
        "marker_pos_world_at_last_hold": marker_world(name).tolist(),
        "marker_registered_world": None if marker_registered_world is None else marker_registered_world.tolist(),
        "marker_registration": "reused-from-file" if fixed_marker_world is not None else "registered-from-chassis",
        "marker_registered_quat": None if marker_registered_quat is None else [float(v) for v in marker_registered_quat],
        "initial_state": view_initial_state,
        "marker_replaces_in_hold": marker_replaces,
        "arrival_step": arrival_step,
        "arrived": arrival_step is not None,
        "steps_used": steps,
        "hold_frames": len(hold_records),
        "visible_frames": visible_count,
        "visible_ratio": (visible_count / len(hold_records)) if hold_records else None,
        "pixel_offset_mean_px": float(np.mean(ratios)) if ratios else None,
        "pixel_offset_p95_px": float(np.percentile(ratios, 95)) if ratios else None,
        "centroid_std_px": {"u": float(np.std(us)) if us else None,
                            "v": float(np.std(vs)) if vs else None},
        "ee_pos_err_mean_m": float(np.mean([r["pos_err_m"] for r in hold_records])) if hold_records else None,
        "ee_pos_err_std_m": float(np.std([r["pos_err_m"] for r in hold_records])) if hold_records else None,
        "ee_ori_err_mean_rad": float(np.mean([r["ori_err_rad"] for r in hold_records])) if hold_records else None,
        "camera_pos_err_mean_m": float(np.mean([r["cam_pos_err_m"] for r in hold_records])) if hold_records else None,
        "camera_axis_err_mean_deg": float(np.mean([r["cam_axis_err_deg"] for r in hold_records])) if hold_records else None,
        "mount_consistency_max_m": float(np.max([r["mount_consistency_m"] for r in hold_records])) if hold_records else None,
        "base_drift_xy_mean_m": float(np.mean([r["base_drift_xy_m"] for r in hold_records])) if hold_records else None,
        "base_drift_xy_max_m": float(np.max([r["base_drift_xy_m"] for r in hold_records])) if hold_records else None,
        "base_contact_events": int(base_contact),
        "bad_orientation_events": int(bad_orientation),
        "contact_env_steps": int(sum(r["contact_env_steps"] for r in hold_records)),
        "contact_body_events": int(sum(r["contact_body_events"] for r in hold_records)),
        "saved_frames": saved,
        "failure_frames": fail_saved,
        "hold_records": hold_records,
    }
    summary["acceptance"] = {
        "arrived": bool(summary["arrived"]),
        "visible_ratio_ok": summary["visible_ratio"] is not None and summary["visible_ratio"] >= VISIBLE_MIN_RATIO,
        "pixel_offset_ok": summary["pixel_offset_mean_px"] is not None and summary["pixel_offset_mean_px"] <= PIXEL_OFFSET_MAX,
        "pixel_jitter_ok": (summary["centroid_std_px"]["u"] is not None
                            and max(summary["centroid_std_px"]["u"], summary["centroid_std_px"]["v"]) <= PIXEL_JITTER_MAX),
        "camera_pose_ok": (summary["camera_pos_err_mean_m"] is not None
                           and summary["camera_pos_err_mean_m"] <= 0.05
                           and summary["camera_axis_err_mean_deg"] <= AXIS_MAX_DEG),
        "mount_consistency_ok": (summary["mount_consistency_max_m"] is not None
                                 and summary["mount_consistency_max_m"] <= MOUNT_CONSISTENCY_MAX),
        "no_bad_termination": base_contact == 0 and bad_orientation == 0,
    }
    summary["accepted"] = all(summary["acceptance"].values())
    results["views"][name] = summary
    print(json.dumps({k: v for k, v in summary.items() if k != "hold_records"}, indent=1)[:4000], flush=True)

results["summary"] = {
    "views": len(results["views"]),
    "accepted": [n for n, v in results["views"].items() if v["accepted"]],
    "failed": [n for n, v in results["views"].items() if not v["accepted"]],
}
with open(os.path.join(OUT_DIR, "d455_view_results.json"), "w") as handle:
    json.dump(results, handle, indent=1)
print(json.dumps(results["summary"], indent=1), flush=True)
print(f"wrote {os.path.join(OUT_DIR, 'd455_view_results.json')}", flush=True)

if yaw_records:
    yaw_path = os.path.join(OUT_DIR, f"d455_yaw_chain_{args.detail_view}.json")
    with open(yaw_path, "w") as handle:
        json.dump({
            "view": args.detail_view,
            "control_dt_s": float(u.step_dt),
            "wz_suppressed_after_arrival": bool(args.zero_wz_after_arrival),
            "base_term_cfg": {
                "clip": [list(v) for v in base_cfg.clip],
                "max_spin_rate": float(base_cfg.max_spin_rate),
                "min_turn_radius": float(base_cfg.min_turn_radius),
                "max_steer_angle": float(base_cfg.max_steer_angle),
                "wheel_radius": float(base_cfg.wheel_radius),
            },
            "columns": ["phase", "steps_since_reset", "sim_time_s", "upper_cmd_vx_wz",
                        "action_base_raw", "term_command_vx_wz", "term_command_after_spin",
                        "wz_zeroed", "steering_target", "wheel_target", "steering_actual",
                        "wheel_actual", "base_yaw_deg", "base_yaw_rate_dps", "ee_pos_err_m", "u", "v"],
            "records": yaw_records,
        }, handle, indent=1)
    print(f"wrote {yaw_path}", flush=True)

if detail_records:
    detail = {
        "view": args.detail_view,
        "marker_mode": args.marker_mode,
        "control_dt_s": float(u.step_dt),
        "intrinsics": detail_intrinsics,
        "image_wh": [IMG_W, IMG_H],
        "arrival_rule": "arrival is decided after stepping and checking the pose; the first hold "
                        "frame is read one control step later",
        "arrival_step": results["views"][args.detail_view]["arrival_step"],
        "first_hold_frame_step": detail_records[0]["steps_since_reset"],
        "records": detail_records,
    }
    path = os.path.join(OUT_DIR, f"d455_detail_{args.detail_view}.json")
    with open(path, "w") as handle:
        json.dump(detail, handle, indent=1)
    print(f"wrote {path}", flush=True)
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
