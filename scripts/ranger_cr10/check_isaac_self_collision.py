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

"""Stage 4.9: can Isaac Lab see the arm hitting the lidar, and is enabling it safe?

``enabled_self_collisions`` was False for every stage before this one, so Isaac -- the
simulator the policy is trained in -- could not report the arm passing through the chassis,
the lidar or the wrist camera. Turning the flag on is not free: adjacent links overlap at
their joints, and a permanent contact at the default pose would be a penalty the policy can
never remove. This script answers both questions by *behaviour* rather than by contact counts,
which keeps it independent of how a contact sensor is wired:

* hold the authored default pose for 100 steps -> if enabling the flag is safe, the arm sits
  still (the residual joint motion is the same as with the flag off);
* ask the arm for the joint configuration that puts Link2 inside the lidar -> with the flag on
  the arm should be *stopped short* of the request; with it off it reaches it.

Run it twice and compare:

    python scripts/ranger_cr10/check_isaac_self_collision.py --self-collisions off
    python scripts/ranger_cr10/check_isaac_self_collision.py --self-collisions on
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--self-collisions", choices=("on", "off"), required=True)
args = ap.parse_args()

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import LeggedManip_Lab.tasks  # noqa: E402,F401
import torch  # noqa: E402
from LeggedManip_Lab.assets.ranger_cr10.ranger_cr10_articulation_cfg import ARM_JOINT_NAMES  # noqa: E402
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.teleop_env_cfg import (  # noqa: E402
    RangerCr10TeleopEnvCfg,
)

# The MuJoCo fixture that puts Link2 inside the lidar (see check_self_collision.py).
PENETRATING = (-3.554, 1.453, 0.229, 1.720, 0.184, 0.701)


def main() -> None:
    cfg = RangerCr10TeleopEnvCfg()
    cfg.scene.num_envs = 1
    cfg.commands.ee_pose.debug_vis = False
    cfg.commands.ee_pose.resampling_time_range = (1e6, 1e6)
    cfg.commands.ee_pose.curriculum_enabled = False
    cfg.events.reset_robot_joints = None
    cfg.events.randomize_actuator_gains = None
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = args.self_collisions == "on"
    scale = cfg.actions.joint_pos.scale
    env = gym.make("RANGER-CR10-TELEOP-WBC", cfg=cfg)
    u = env.unwrapped
    robot = u.scene["robot"]
    env.reset()
    zero = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
    start = robot.data.joint_pos[:, robot.joint_names.index(ARM_JOINT_NAMES[0]) :][:, :6].clone()

    for _ in range(100):
        env.step(zero)
    rest = robot.data.joint_pos[:, robot.joint_names.index(ARM_JOINT_NAMES[0]) :][:, :6]
    drift = (rest - start).abs().max().item()

    target = torch.tensor(PENETRATING, device=u.device).unsqueeze(0)
    act = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
    act[:, 2:] = (target - start) / scale
    trace = []
    for _ in range(120):
        env.step(act)
        trace.append(robot.data.joint_pos[:, robot.joint_names.index(ARM_JOINT_NAMES[0]) :][:, :6].clone())
    reached = trace[-1][0].tolist()
    err = max(abs(a - b) for a, b in zip(reached, PENETRATING))
    # the first joint that would enter the lidar stops moving in a different place with the
    # flag on; report the per-joint error so the difference is visible, not just the max
    per_joint = [round(a - b, 3) for a, b in zip(reached, PENETRATING)]
    print(f"\n===== enabled_self_collisions = {args.self_collisions} =====")
    print(f"  默认位姿 100 步残余运动 (max |dq|): {drift:.4f} rad")
    print(f"  驱动的关节误差 (最大): {err:.3f} rad   逐关节: {per_joint}")
    env.close()


if __name__ == "__main__":
    main()
