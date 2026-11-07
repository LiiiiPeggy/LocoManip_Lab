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

"""Hold the default pose on the simple-collision candidate and report stability/contacts."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--self-collisions", choices=("on", "off"), default="on")
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--out", default="outputs/stage413/b/simple_collision_validation")
args = ap.parse_args()

app = AppLauncher(headless=True).app

import json  # noqa: E402
import os  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import LeggedManip_Lab.tasks  # noqa: E402
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.simple_collision_env_cfg import (  # noqa: E402
    RangerCr10TeleopSimpleCollisionEnvCfg,
)


def main() -> None:
    cfg = RangerCr10TeleopSimpleCollisionEnvCfg()
    cfg.scene.num_envs = 1
    cfg.events.reset_robot_joints = None
    cfg.events.randomize_actuator_gains = None
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = args.self_collisions == "on"
    env = gym.make("RANGER-CR10-TELEOP-SIMPLECOLL", cfg=cfg)
    u = env.unwrapped
    robot = u.scene["robot"]
    sensor = u.scene["contact_forces"]
    env.reset()
    zero = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
    arm_idx = [robot.joint_names.index(f"cr10_joint{i}") for i in range(1, 7)]
    start = robot.data.joint_pos[:, arm_idx].clone()
    base0 = robot.data.root_pos_w.clone()
    max_drift = 0.0
    max_force = 0.0
    force_bodies = []
    for step in range(args.steps):
        env.step(zero)
        drift = (robot.data.joint_pos[:, arm_idx] - start).abs().max().item()
        max_drift = max(max_drift, drift)
        forces = sensor.data.net_forces_w
        if forces is not None and forces.numel():
            fmax = forces.norm(dim=-1).max().item()
            max_force = max(max_force, fmax)
            if fmax > 1e-3:
                idx = int(forces.norm(dim=-1).argmax())
                body = sensor.body_names[idx % len(sensor.body_names)]
                if body not in force_bodies:
                    force_bodies.append(body)
    final = robot.data.joint_pos[:, arm_idx]
    base1 = robot.data.root_pos_w
    result = dict(mode=args.self_collisions, steps=args.steps,
                  max_arm_drift_rad=max_drift,
                  final_arm_error_rad=(final - start).abs().max().item(),
                  base_displacement_m=(base1 - base0).norm().item(),
                  max_contact_force_n=max_force,
                  contact_bodies=force_bodies,
                  body_names=robot.body_names,
                  sensor_body_names=sensor.body_names,
                  note="zero action; reset randomisation disabled; simple-box candidate")
    out = os.path.join(args.out, f"stability_{args.self_collisions}.json")
    os.makedirs(args.out, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1), flush=True)
    env.close()


if __name__ == "__main__":
    main()
