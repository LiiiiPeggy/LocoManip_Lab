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

"""Continuous resets and moving-base box_link follow on the v2 candidate.

The two box_link colliders are merged into ``base_link``. This test:
  * resets five times at the home pose and measures arm drift/base height;
  * commands vx=0.2 for 40 steps while holding the arm, records the base motion,
    the arm drift and any body-pair contact, and computes the world centres of the
    two box_link boxes from the base transform and their fixed local offsets.

Run:
    python scripts/ranger_cr10/check_simple_reset_base_v2.py \
        --out outputs/stage413/b/simple_collision_v2/reset_base.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", required=True)
ap.add_argument("--resets", type=int, default=5)
ap.add_argument("--hold-steps", type=int, default=10)
ap.add_argument("--base-steps", type=int, default=40)
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

# Fixed box centres in the base_link frame: box_link is at z=+0.09 and its two
# collision boxes are authored in the box_link frame. The right plate collision was
# removed per user request; the second collider is now the top accessory.
BOX_LEFT_LOCAL = np.array([-0.3269, 0.0, 0.09 + 0.12725])
BOX_TOP_LOCAL = np.array([-0.436, 0.0, 0.09 + 0.31855])

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
cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True

env = gym.make("RANGER-CR10-TELEOP-SIMPLECOLL", cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
body_paths = [f"/World/envs/env_0/Robot/{name}" for name in robot.body_names]
physics_view = SimulationManager.get_physics_sim_view()
contact_view = physics_view.create_rigid_contact_view(
    "/World/envs/env_0/Robot/*", body_paths, max_contact_data_count=256
)
dt = u.physics_dt
scale = cfg.actions.joint_pos.scale
arm_idx = [robot.joint_names.index(f"cr10_joint{i}") for i in range(1, 7)]


def hold_action(q):
    action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
    action[:, 2:] = q / scale
    return action


def any_contact():
    matrix = contact_view.get_contact_force_matrix(dt)
    matrix = matrix.reshape(contact_view.sensor_count, contact_view.filter_count, 3)
    return float(torch.linalg.norm(matrix).item()) > 1e-3


reset_rows = []
for index in range(args.resets):
    env.reset()
    start = robot.data.joint_pos[:, arm_idx].clone()
    for _ in range(args.hold_steps):
        env.step(hold_action(start))
    reset_rows.append(
        {
            "reset": index,
            "arm_drift_rad": float(torch.max(torch.abs(robot.data.joint_pos[:, arm_idx] - start)).item()),
            "base_z": float(robot.data.root_pos_w[0, 2]),
            "contact": any_contact(),
        }
    )

# Moving-base test: hold the arm, command forward velocity.
env.reset()
hold = robot.data.joint_pos[:, arm_idx].clone()
base0 = robot.data.root_pos_w.clone()
rows = []
for step in range(args.base_steps):
    action = hold_action(hold)
    action[0, 0] = 0.2
    env.step(action)
    quat = robot.data.root_quat_w[0].detach().cpu().numpy()  # (w, x, y, z)
    w, x, y, z = quat
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    base = robot.data.root_pos_w[0].detach().cpu().numpy()
    rows.append(
        {
            "step": step,
            "base": base.tolist(),
            "box_left_world": (base + rotation @ BOX_LEFT_LOCAL).tolist(),
            "box_top_world": (base + rotation @ BOX_TOP_LOCAL).tolist(),
            "arm_drift_rad": float(torch.max(torch.abs(robot.data.joint_pos[:, arm_idx] - hold)).item()),
            "contact": any_contact(),
        }
    )

result = {
    "resets": reset_rows,
    "base_motion": rows,
    "base_displacement_m": float((robot.data.root_pos_w - base0).norm().item()),
    "box_left_local_offset": BOX_LEFT_LOCAL.tolist(),
    "box_top_local_offset": BOX_TOP_LOCAL.tolist(),
    "note": "box_link colliders are merged into base_link; their local offsets are fixed and the world "
            "centres are computed with the measured base transform. The right plate collision was "
            "removed per user request; the second collider is the top accessory. Contacts are body-level "
            "PhysX pairs.",
}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as handle:
    json.dump(result, handle, indent=1)
print(
    json.dumps(
        {
            "resets": reset_rows,
            "base_displacement_m": result["base_displacement_m"],
            "last_base": rows[-1]["base"],
            "last_box_left": rows[-1]["box_left_world"],
            "last_box_top": rows[-1]["box_top_world"],
            "any_contact": any(row["contact"] for row in rows),
            "max_arm_drift_rad": max(row["arm_drift_rad"] for row in rows),
        },
        indent=1,
    ),
    flush=True,
)
print(f"wrote {args.out}", flush=True)
env.close()
