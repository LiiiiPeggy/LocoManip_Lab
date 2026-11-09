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

"""Compare PhysX contact onset with the actual-shape distance on two blocking paths.

This is a small 2-env diagnostic, not a replacement for the 64-env validation. It repeats the
same 120-step joint-space interpolation used by the frozen blocking test for:

* env 0: Link2 vs box_link left enclosure;
* env 1: Link3 vs Link6.

At each control step it records the actual arm joints, the exact contact body pair and force from
``RigidContactView``, and the PhysX-consistent proxy distance. The comparison is about contact
offset/rest offset semantics: PhysX may generate contact points before geometric zero separation;
that is not a geometry error and the offsets are not changed to force numeric agreement.

Run:
    python scripts/ranger_cr10/check_phys_proxy_contact_timing.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default="outputs/stage413/b/simple_collision_v2/phys_proxy_contact_timing.json")
args = ap.parse_args()

app = AppLauncher(headless=True).app

import ast  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402

import LeggedManip_Lab.tasks  # noqa: E402
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.simple_collision_env_cfg import (  # noqa: E402
    RangerCr10TeleopSimpleCollisionEnvCfg,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from phys_proxy import load_shapes, proxy_group_distance, shapes_by_group  # noqa: E402

TARGET_FILE = os.path.join(ROOT, "outputs/stage413/b/simple_collision_v2/targets_v2.json")
PROXY_SCENE = os.path.join(ROOT, "mujoco/robots/ranger_cr10/scene_simple_v2_physproxy.xml")

cfg = RangerCr10TeleopSimpleCollisionEnvCfg()
cfg.scene.num_envs = 2
cfg.commands.ee_pose.debug_vis = False
cfg.commands.ee_pose.resampling_time_range = (1e6, 1e6)
cfg.commands.ee_pose.curriculum_enabled = False
cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
cfg.events.randomize_actuator_gains = None
cfg.terminations.base_contact = None
cfg.terminations.bad_orientation = None
cfg.terminations.time_out = None
cfg.actions.joint_pos.use_default_offset = False
cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True

env = gym.make("RANGER-CR10-TELEOP-SIMPLECOLL", cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
env.reset(seed=0)
num_envs = u.num_envs
dt = u.physics_dt
arm_cols = [robot.joint_names.index(f"cr10_joint{i}") for i in range(1, 7)]
scale = cfg.actions.joint_pos.scale
total_action_dim = u.action_manager.total_action_dim


def hold_action(q):
    action = torch.zeros(num_envs, total_action_dim, device=u.device)
    action[:, 2:] = q / scale
    return action


def read_pairs(view, sensor_names, filter_names):
    matrix = view.get_contact_force_matrix(dt)
    matrix = matrix.reshape(view.sensor_count, view.filter_count, 3).cpu()
    forces, points, normals, separations, counts, starts = view.get_contact_data(dt)
    forces = forces.reshape(-1).cpu()
    points = points.reshape(-1, 3).cpu()
    normals = normals.reshape(-1, 3).cpu()
    separations = separations.reshape(-1).cpu()
    counts = counts.reshape(view.sensor_count, view.filter_count).cpu()
    starts = starts.reshape(view.sensor_count, view.filter_count).cpu()
    pairs = {}
    for si, sensor_name in enumerate(sensor_names):
        for fi, filter_name in enumerate(filter_names):
            force = matrix[si, fi]
            if float(torch.linalg.norm(force)) < 1e-3:
                continue
            start = int(starts[si, fi])
            count = int(counts[si, fi])
            items = []
            for index in range(start, min(start + count, len(forces))):
                items.append({
                    "force_n": float(forces[index]),
                    "separation_m": float(separations[index]),
                    "position": [float(x) for x in points[index]],
                    "normal": [float(x) for x in normals[index]],
                })
            key = tuple(sorted((sensor_name, filter_name)))
            entry = pairs.setdefault(key, {"pair": list(key), "force_n": 0.0, "n_points": 0,
                                           "min_separation_m": float("inf")})
            entry["force_n"] += float(torch.linalg.norm(force))
            entry["n_points"] += count
            if items:
                entry["min_separation_m"] = min(entry["min_separation_m"],
                                                min(item["separation_m"] for item in items))
    return list(pairs.values())


physics_view = SimulationManager.get_physics_sim_view()
detail_views = {}
for env_id in range(num_envs):
    paths = [f"/World/envs/env_{env_id}/Robot/{name}" for name in robot.body_names]
    view = physics_view.create_rigid_contact_view(
        f"/World/envs/env_{env_id}/Robot/*", paths, max_contact_data_count=256
    )
    sensor_names = [str(name) for name in view.sensor_names]
    raw_filter = view.filter_names[0] if view.filter_names else []
    filter_names = list(ast.literal_eval(raw_filter)) if isinstance(raw_filter, str) else list(raw_filter)
    detail_views[env_id] = (view, sensor_names, filter_names)

try:
    contact_offsets = robot.root_physx_view.get_contact_offsets().detach().cpu().tolist()
except Exception as error:  # noqa: BLE001
    contact_offsets = f"unavailable: {error}"
try:
    rest_offsets = robot.root_physx_view.get_rest_offsets().detach().cpu().tolist()
except Exception as error:  # noqa: BLE001
    rest_offsets = f"unavailable: {error}"

targets = json.load(open(TARGET_FILE))["selected"]
blocks = {0: ("box_left", "cr10_Link2", "box_left"),
          1: ("arm_link3_link6", "cr10_Link3", "cr10_Link6")}
starts = robot.data.joint_pos[:, arm_cols].detach().clone()
target_q = {env_id: torch.tensor(targets[key]["q"], dtype=torch.float32, device=u.device)
            for env_id, (key, _, _) in blocks.items()}

shapes = shapes_by_group(load_shapes())
import mujoco  # noqa: E402

proxy_model = mujoco.MjModel.from_xml_path(PROXY_SCENE)
proxy_data = mujoco.MjData(proxy_model)
rows = {env_id: [] for env_id in blocks}
steps = 120
for step in range(steps):
    alpha = (step + 1) / steps
    q_cmd = hold_action(starts).clone()
    for env_id in blocks:
        q_cmd[env_id, 2:] = (starts[env_id] * (1.0 - alpha) + target_q[env_id] * alpha) / scale
    env.step(q_cmd)
    for env_id, (case, arm, part) in blocks.items():
        actual = robot.data.joint_pos[env_id, arm_cols].detach().cpu().numpy()
        view, sensor_names, filter_names = detail_views[env_id]
        pairs = read_pairs(view, sensor_names, filter_names)
        mujoco.mj_resetDataKeyframe(proxy_model, proxy_data, proxy_model.key("home").id)
        for index, value in enumerate(actual):
            joint = mujoco.mj_name2id(proxy_model, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{index + 1}")
            proxy_data.qpos[proxy_model.jnt_qposadr[joint]] = float(value)
        mujoco.mj_forward(proxy_model, proxy_data)
        result = proxy_group_distance(proxy_model, proxy_data, shapes, arm, part)
        rows[env_id].append({
            "step": step,
            "q": [float(x) for x in actual],
            "target_error_rad": float(np.max(np.abs(actual - target_q[env_id].detach().cpu().numpy()))),
            "proxy_distance_m": result["distance"],
            "proxy_signed_m": result["signed"],
            "proxy_status": result["status"],
            "proxy_shape_a": result["shape_a"],
            "proxy_shape_b": result["shape_b"],
            "pairs": pairs,
            "contact": bool(pairs),
        })

summary = {}
for env_id, (case, arm, part) in blocks.items():
    data = rows[env_id]
    first_contact = next((row for row in data if row["contact"]), None)
    first_force = next((row for row in data if any(pair["force_n"] > 100.0 for pair in row["pairs"])), None)
    first_zero = next((row for row in data if row["proxy_distance_m"] <= 1e-4), None)
    first_margin = next((row for row in data if row["proxy_distance_m"] <= 0.02), None)
    summary[str(env_id)] = {
        "case": case,
        "target_pair": [arm, part],
        "first_contact_any": None if first_contact is None else first_contact["step"],
        "first_contact_force_gt_100n": None if first_force is None else first_force["step"],
        "first_proxy_zero_or_contact_tol": None if first_zero is None else first_zero["step"],
        "first_proxy_margin_20mm": None if first_margin is None else first_margin["step"],
        "first_contact_pairs": None if first_contact is None else first_contact["pairs"],
        "first_zero_distance_m": None if first_zero is None else first_zero["proxy_distance_m"],
        "first_contact_proxy_distance_m": None if first_contact is None else first_contact["proxy_distance_m"],
        "first_contact_target_error_rad": None if first_contact is None else first_contact["target_error_rad"],
        "note": "PhysX contact generation and contact forces are not the same predicate as geometric "
                "zero distance; contact offset can create points before zero separation.",
    }

result = {
    "envs": 2,
    "physics_dt_s": dt,
    "control_dt_s": dt * cfg.decimation,
    "contact_offsets": contact_offsets,
    "rest_offsets": rest_offsets,
    "targets": {str(env_id): {"case": key, "q": targets[key]["q"]} for env_id, (key, _, _) in blocks.items()},
    "summary": summary,
    "rows": {str(env_id): rows[env_id] for env_id in rows},
}
os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
with open(os.path.join(ROOT, args.out), "w") as handle:
    json.dump(result, handle, indent=1)
print(json.dumps({key: result[key] for key in ("physics_dt_s", "control_dt_s", "contact_offsets",
                                               "rest_offsets", "summary")}, indent=1))
print(f"wrote {os.path.join(ROOT, args.out)}")
