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

"""Stage 4.11: does the safety term fire at the right places *in Isaac*, not just on paper?

`check_clearance_proxy.py` proves the distance the term uses is the distance physics has, but
it does that with numpy and MuJoCo. This runs the same term through the real environment, which
is where a frame or a body-name mistake would show up, and answers three questions:

  * does the authored pose read what MuJoCo says it reads (~14 mm, `Link3 <-> Link5`)?
  * does the penalty stay near zero there -- the pose must not pay for standing still;
  * does the penalty grow and the collision flag fire when the arm is driven into itself?

Usage:
    python scripts/ranger_cr10/check_safety_reward.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--fingers", type=float, default=0.6, help="action magnitude used to fold the arm in")
args = ap.parse_args()

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import LeggedManip_Lab.tasks  # noqa: E402,F401
import torch  # noqa: E402
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.teleop_env_cfg import (  # noqa: E402
    RangerCr10TeleopEnvCfg,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp import safety  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402


def report(env, label: str, action: torch.Tensor) -> None:
    u = env.unwrapped
    for _ in range(args.steps):
        env.step(action)
    summary = safety.clearance_summary(u, u.scene["robot"])
    penalty = safety.sensor_clearance_penalty(u, SceneEntityCfg("robot"), margin=0.015)
    print(f"\n[{label}]")
    print(f"  reward term           {penalty.mean().item():+.4f}")
    for key, value in summary.items():
        print(f"  {key:22s} min {value.min().item() * 1000:6.1f} mm  mean {value.mean().item() * 1000:6.1f} mm")
    self_collision = float((summary["self"] <= 0.0).float().mean())
    sensor_collision = float((summary["sensor"] <= 0.0).float().mean())
    print(f"  collision share       self {self_collision * 100:.0f}%  sensor {sensor_collision * 100:.0f}%")


def main() -> None:
    cfg = RangerCr10TeleopEnvCfg()
    cfg.scene.num_envs = 8
    cfg.commands.ee_pose.debug_vis = False
    cfg.commands.ee_pose.resampling_time_range = (1e6, 1e6)
    cfg.events.reset_robot_joints = None
    env = gym.make("RANGER-CR10-TELEOP-WBC", cfg=cfg)
    u = env.unwrapped
    env.reset()

    zero = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
    report(env, "authored pose, zero action", zero)

    # fold the wrist: the joint channels are [vx, wz, q1..q6] scaled by 0.25 rad
    fold = zero.clone()
    fold[:, 2:] = args.fingers
    report(env, f"arm driven with {args.fingers:+.2f} on every arm channel", fold)

    env.close()


if __name__ == "__main__":
    main()
