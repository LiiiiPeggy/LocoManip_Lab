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

"""Small sinusoidal arm motion on the v2 candidate, with body-pair contact recording.

The arm starts at the reset home pose and follows +-0.08 rad sinusoids for 240 steps.
The base is not commanded. This checks that normal joint motion does not create
self-contacts now that the body-level filters are applied.

Run:
    python scripts/ranger_cr10/check_simple_joint_motion_v2.py \
        --self-collisions on --out outputs/stage413/b/simple_collision_v2/motion_on.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--self-collisions", choices=("on", "off"), default="on")
ap.add_argument("--steps", type=int, default=240)
ap.add_argument("--amplitude", type=float, default=0.08)
ap.add_argument("--period", type=int, default=100)
ap.add_argument("--out", required=True)
args = ap.parse_args()

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402

import LeggedManip_Lab.tasks  # noqa: E402
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.simple_collision_env_cfg import (  # noqa: E402
    RangerCr10TeleopSimpleCollisionEnvCfg,
)

cfg = RangerCr10TeleopSimpleCollisionEnvCfg()
cfg.scene.num_envs = 1
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
cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = args.self_collisions == "on"

env = gym.make("RANGER-CR10-TELEOP-SIMPLECOLL", cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
env.reset()

body_paths = [f"/World/envs/env_0/Robot/{name}" for name in robot.body_names]
physics_view = SimulationManager.get_physics_sim_view()
contact_view = physics_view.create_rigid_contact_view(
    "/World/envs/env_0/Robot/*", body_paths, max_contact_data_count=512
)
sensor_names = [str(name) for name in contact_view.sensor_names]
_raw_filter = contact_view.filter_names[0] if contact_view.filter_names else []
filter_names = list(_raw_filter) if not isinstance(_raw_filter, str) else ast.literal_eval(_raw_filter)
arm_idx = [robot.joint_names.index(f"cr10_joint{i}") for i in range(1, 7)]
start = robot.data.joint_pos[:, arm_idx].clone()
scale = cfg.actions.joint_pos.scale
base0 = robot.data.root_pos_w.clone()
dt = u.physics_dt


def read_contacts():
    matrix = contact_view.get_contact_force_matrix(dt)
    matrix = matrix.reshape(contact_view.sensor_count, contact_view.filter_count, 3).cpu()
    forces, points, normals, separations, counts, starts = contact_view.get_contact_data(dt)
    separations = separations.reshape(-1).cpu()
    points = points.reshape(-1, 3).cpu()
    counts = counts.reshape(contact_view.sensor_count, contact_view.filter_count).cpu()
    starts = starts.reshape(contact_view.sensor_count, contact_view.filter_count).cpu()
    out = []
    for sensor_index, sensor_name in enumerate(sensor_names):
        for filter_index, filter_name in enumerate(filter_names):
            force = matrix[sensor_index, filter_index]
            if float(torch.linalg.norm(force)) < 1e-3:
                continue
            start_index = int(starts[sensor_index, filter_index])
            count = int(counts[sensor_index, filter_index])
            out.append(
                {
                    "pair": sorted((sensor_name, filter_name)),
                    "force_n": float(torch.linalg.norm(force)),
                    "min_separation_m": float(separations[start_index : start_index + count].min().item())
                    if count
                    else None,
                    "position": [float(x) for x in points[start_index]] if count else None,
                }
            )
    return out


max_force = 0.0
contacts = {}
max_tracking = 0.0
max_arm_drift = 0.0
for step in range(args.steps):
    phase = 2.0 * np.pi * step / args.period
    offset = args.amplitude * np.sin(phase)
    q_cmd = start[0] + offset
    action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
    action[0, 2:] = q_cmd / scale
    env.step(action)
    actual = robot.data.joint_pos[0, arm_idx]
    max_tracking = max(max_tracking, float(torch.max(torch.abs(actual - q_cmd)).item()))
    max_arm_drift = max(max_arm_drift, float(torch.max(torch.abs(actual - start[0])).item()))
    for pair in read_contacts():
        max_force = max(max_force, pair["force_n"])
        key = "|".join(pair["pair"])
        entry = contacts.setdefault(
            key,
            {"pair": pair["pair"], "first_step": step, "max_force_n": 0.0, "min_separation_m": float("inf"),
             "first_position": pair["position"]},
        )
        entry["max_force_n"] = max(entry["max_force_n"], pair["force_n"])
        if pair["min_separation_m"] is not None:
            entry["min_separation_m"] = min(entry["min_separation_m"], pair["min_separation_m"])

result = {
    "self_collisions": args.self_collisions,
    "steps": args.steps,
    "amplitude_rad": args.amplitude,
    "period_steps": args.period,
    "max_arm_drift_rad": max_arm_drift,
    "max_tracking_error_rad": max_tracking,
    "base_displacement_m": float((robot.data.root_pos_w - base0).norm().item()),
    "max_contact_force_n": max_force,
    "contact_pairs": sorted(contacts.values(), key=lambda item: item["max_force_n"], reverse=True),
    "note": "small sinusoidal arm targets around the home pose; base command zero; terminations disabled; "
            "contacts are body-level PhysX pairs",
}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as handle:
    json.dump(result, handle, indent=1)
print(json.dumps({k: result[k] for k in result if k != "contact_pairs"}, indent=1), flush=True)
print("contact_pairs", result["contact_pairs"][:8], flush=True)
print(f"wrote {args.out}", flush=True)
env.close()
