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

"""Slow joint-target approach to box_link or a non-adjacent arm pair, self-collisions on/off."""

from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--case", choices=("box", "plate", "arm"), required=True)
ap.add_argument("--index", type=int, default=0)
ap.add_argument("--self-collisions", choices=("on", "off"), required=True)
ap.add_argument("--steps", type=int, default=120)
ap.add_argument("--out", default="outputs/stage413/b/simple_collision_validation")
args = ap.parse_args()

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
import torch  # noqa: E402
import LeggedManip_Lab.tasks  # noqa: E402
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.simple_collision_env_cfg import (  # noqa: E402
    RangerCr10TeleopSimpleCollisionEnvCfg,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
targets = json.load(open(os.path.join(ROOT, "outputs/stage413/b/simple_collision_validation/targets.json")))
target = targets[{"box": "box_targets", "plate": "box_plate_targets", "arm": "arm_targets"}[args.case]][args.index]
q_target = torch.tensor(target["q"], dtype=torch.float32)
pair = target["pair"].split("|")
mj = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "mujoco/robots/ranger_cr10/scene_simple.xml"))
md = mujoco.MjData(mj)
mj_names = {mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, i): i for i in range(mj.ngeom) if mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, i)}


def pair_distance(q):
    mujoco.mj_resetDataKeyframe(mj, md, mj.key("home").id)
    for i, v in enumerate(q, start=1):
        adr = mj.jnt_qposadr[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")]
        md.qpos[adr] = float(v)
    mujoco.mj_forward(mj, md)
    return float(mujoco.mj_geomDistance(mj, md, mj_names[pair[0]], mj_names[pair[1]], 0.2, None))


def main() -> None:
    cfg = RangerCr10TeleopSimpleCollisionEnvCfg()
    cfg.scene.num_envs = 1
    cfg.commands.ee_pose.debug_vis = False
    cfg.commands.ee_pose.resampling_time_range = (1e6, 1e6)
    cfg.commands.ee_pose.curriculum_enabled = False
    cfg.events.reset_robot_joints = None
    cfg.events.randomize_actuator_gains = None
    cfg.actions.joint_pos.use_default_offset = False
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = args.self_collisions == "on"
    env = gym.make("RANGER-CR10-TELEOP-SIMPLECOLL", cfg=cfg)
    u = env.unwrapped
    robot = u.scene["robot"]
    sensor = u.scene["contact_forces"]
    env.reset()
    idx = [robot.joint_names.index(f"cr10_joint{i}") for i in range(1, 7)]
    start = robot.data.joint_pos[:, idx].clone()
    scale = cfg.actions.joint_pos.scale
    base0 = robot.data.root_pos_w.clone()
    min_pair = 1e9
    max_force = 0.0
    force_bodies = []
    rows = []
    # settle at the start target for 10 steps, then interpolate slowly to the penetrating target
    for step in range(10):
        act = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
        act[:, 2:] = start / scale
        env.step(act)
    for step in range(args.steps):
        alpha = (step + 1) / args.steps
        q_cmd = start[0] * (1.0 - alpha) + q_target.to(u.device) * alpha
        act = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
        act[0, 2:] = q_cmd / scale
        env.step(act)
        actual = robot.data.joint_pos[0, idx].detach().cpu().numpy()
        d = pair_distance(actual)
        min_pair = min(min_pair, d)
        forces = sensor.data.net_forces_w
        if forces is not None and forces.numel():
            fmax = forces.norm(dim=-1).max().item()
            max_force = max(max_force, fmax)
            if fmax > 1e-3:
                body = sensor.body_names[int(forces.norm(dim=-1).argmax()) % len(sensor.body_names)]
                if body not in force_bodies:
                    force_bodies.append(body)
        if step % 10 == 0 or step == args.steps - 1:
            rows.append(dict(step=step, target=[float(x) for x in q_cmd], actual=actual.tolist(),
                             pair_distance_m=d, max_force_n=fmax if forces is not None and forces.numel() else 0.0))
    actual_final = robot.data.joint_pos[0, idx].detach().cpu().numpy()
    result = dict(case=args.case, index=args.index, self_collisions=args.self_collisions,
                  pair=target["pair"], target_q=target["q"], start_q=start[0].tolist(),
                  actual_final=actual_final.tolist(),
                  target_error_rad=float(max(abs(actual_final[i] - target["q"][i]) for i in range(6))),
                  min_pair_distance_m=min_pair,
                  final_pair_distance_m=pair_distance(actual_final),
                  base_displacement_m=(robot.data.root_pos_w - base0).norm().item(),
                  max_contact_force_n=max_force, contact_bodies=force_bodies,
                  trace=rows, note="slow joint-target interpolation; base command zero; pair distance from the simple-box MuJoCo model at the actual q")
    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    path = os.path.join(ROOT, args.out, f"block_{args.case}_{args.index}_{args.self_collisions}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: result[k] for k in result if k != "trace"}, indent=1), flush=True)
    env.close()


if __name__ == "__main__":
    main()
