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

"""Slow v2 joint-target approach with body-pair contact recording.

The test holds the home pose, disables reset randomisation and the contact/orientation
terminations, then interpolates the six arm joint targets slowly. It records:
  * the actual joint positions and the commanded target (they are not the same);
  * the target-pair distance from the v2 simple-box MuJoCo model at the actual q;
  * the actual PhysX body pairs, contact position, separation and force.

The target is only called blocked by the intended solid if the intended body pair
contacts; a growing target error alone is not evidence.

Run:
    python scripts/ranger_cr10/check_simple_collision_blocking_v2.py \
        --case box --steps 120 --self-collisions on \
        --out outputs/stage413/b/simple_collision_v2/block_box_on.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--case", choices=("box", "arm", "over", "plate"), required=True)
ap.add_argument("--index", type=int, default=0)
ap.add_argument("--steps", type=int, default=120)
ap.add_argument("--self-collisions", choices=("on", "off"), default="on")
ap.add_argument("--usd", default=None)
ap.add_argument("--out", required=True)
args = ap.parse_args()

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402

import LeggedManip_Lab.tasks  # noqa: E402
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.simple_collision_env_cfg import (  # noqa: E402
    RangerCr10TeleopSimpleCollisionEnvCfg,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARGET_FILE = os.path.join(ROOT, "outputs/stage413/b/simple_collision_v2/targets_v2.json")
OLD_TARGETS = os.path.join(ROOT, "outputs/stage413/b/simple_collision_validation/targets.json")

if args.case == "plate":
    old = json.load(open(OLD_TARGETS))
    target = old["box_plate_targets"][args.index]
elif args.case == "arm":
    target = json.load(open(TARGET_FILE))["selected"]["arm_link3_link6"]
elif args.case == "over":
    target = json.load(open(TARGET_FILE))["selected"]["over_box_clean"]
else:
    target = json.load(open(TARGET_FILE))["selected"]["box_left"]

if args.case == "plate":
    result = {
        "case": "plate",
        "status": "not_applicable",
        "reason": "the right mounting plate collision was removed per user request; it is visual only. "
                  "The old plate target is retained only as history and cannot be a collision test.",
        "old_target": target,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps(result, indent=1), flush=True)
    print(f"wrote {args.out}", flush=True)
    app.close()
    raise SystemExit(0)

q_target = torch.tensor(target["q"], dtype=torch.float32)

# MuJoCo pair distance at the actual joint positions (base fixed at home).
mj = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "mujoco/robots/ranger_cr10/scene_simple_v2.xml"))
md = mujoco.MjData(mj)
mj_names = {mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, i): i for i in range(mj.ngeom)}


def pair_distance(pair, q):
    mujoco.mj_resetDataKeyframe(mj, md, mj.key("home").id)
    for index, value in enumerate(q, start=1):
        address = mj.jnt_qposadr[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{index}")]
        md.qpos[address] = float(value)
    mujoco.mj_forward(mj, md)
    a, b = pair.split("|")
    return float(mujoco.mj_geomDistance(mj, md, mj_names[a], mj_names[b], 0.2, None))


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
cfg.actions.joint_pos.use_default_offset = False
cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = args.self_collisions == "on"
if args.usd is not None:
    cfg.scene.robot.spawn.usd_path = args.usd

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
rows = []


def read_contacts():
    matrix = contact_view.get_contact_force_matrix(dt)
    matrix = matrix.reshape(contact_view.sensor_count, contact_view.filter_count, 3).cpu()
    forces, points, normals, separations, counts, starts = contact_view.get_contact_data(dt)
    forces = forces.reshape(-1).cpu()
    points = points.reshape(-1, 3).cpu()
    normals = normals.reshape(-1, 3).cpu()
    separations = separations.reshape(-1).cpu()
    counts = counts.reshape(contact_view.sensor_count, contact_view.filter_count).cpu()
    starts = starts.reshape(contact_view.sensor_count, contact_view.filter_count).cpu()
    pairs = []
    for sensor_index, sensor_name in enumerate(sensor_names):
        for filter_index, filter_name in enumerate(filter_names):
            force = matrix[sensor_index, filter_index]
            if float(torch.linalg.norm(force)) < 1e-3:
                continue
            start_index = int(starts[sensor_index, filter_index])
            count = int(counts[sensor_index, filter_index])
            items = []
            for index in range(start_index, min(start_index + count, len(forces))):
                items.append(
                    {
                        "force_n": float(forces[index]),
                        "position": [float(x) for x in points[index]],
                        "normal": [float(x) for x in normals[index]],
                        "separation_m": float(separations[index]),
                    }
                )
            pairs.append(
                {
                    "pair": sorted((sensor_name, filter_name)),
                    "force_n": float(torch.linalg.norm(force)),
                    "n_points": count,
                    "min_separation_m": min((item["separation_m"] for item in items), default=None),
                    "position": items[0]["position"] if items else None,
                }
            )
    return pairs


for step in range(args.steps):
    alpha = (step + 1) / args.steps
    q_cmd = start[0] * (1.0 - alpha) + q_target.to(u.device) * alpha
    action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
    action[0, 2:] = q_cmd / scale
    env.step(action)
    actual = robot.data.joint_pos[0, arm_idx].detach().cpu().numpy()
    pairs = read_contacts()
    if step % 5 == 0 or step == args.steps - 1 or pairs:
        rows.append(
            {
                "step": step,
                "alpha": alpha,
                "commanded": [float(x) for x in q_cmd],
                "actual": actual.tolist(),
                "target_error_rad": float(np.max(np.abs(actual - np.asarray(target["q"])))),
                "pair_distance_m": pair_distance(target["pair"], actual),
                "base_pos": [float(x) for x in robot.data.root_pos_w[0]],
                "pairs": pairs,
            }
        )

actual_final = robot.data.joint_pos[0, arm_idx].detach().cpu().numpy()
first_contact = None
for row in rows:
    if row["pairs"]:
        first_contact = row
        break
contact_pairs = {}
for row in rows:
    for pair in row["pairs"]:
        key = "|".join(pair["pair"])
        entry = contact_pairs.setdefault(
            key,
            {
                "pair": pair["pair"],
                "first_step": row["step"],
                "first_alpha": row["alpha"],
                "max_force_n": 0.0,
                "min_separation_m": float("inf"),
                "first_position": pair["position"],
            },
        )
        entry["max_force_n"] = max(entry["max_force_n"], pair["force_n"])
        if pair["min_separation_m"] is not None:
            entry["min_separation_m"] = min(entry["min_separation_m"], pair["min_separation_m"])

result = {
    "case": args.case,
    "index": args.index,
    "self_collisions": args.self_collisions,
    "usd": args.usd or "env default",
    "target_pair_mujoco": target["pair"],
    "target_q": target["q"],
    "start_q": start[0].tolist(),
    "actual_final": actual_final.tolist(),
    "target_error_rad": float(np.max(np.abs(actual_final - np.asarray(target["q"])))),
    "final_pair_distance_m": pair_distance(target["pair"], actual_final),
    "min_pair_distance_m": min(
        (pair_distance(target["pair"], np.asarray(row["actual"])) for row in rows),
        default=float("inf"),
    ),
    "base_displacement_m": float((robot.data.root_pos_w - base0).norm().item()),
    "first_contact": first_contact,
    "contact_pairs": sorted(contact_pairs.values(), key=lambda item: item["first_step"]),
    "rows": rows,
    "note": "target pair distance is from the v2 MuJoCo boxes at the actual q; PhysX contacts are "
            "body-level and cannot separate the merged base_link box_link/lidar/chassis colliders; "
            "the contact position is used for that separation",
}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as handle:
    json.dump(result, handle, indent=1)
print(
    json.dumps(
        {
            "case": args.case,
            "self_collisions": args.self_collisions,
            "target_pair": target["pair"],
            "target_error_rad": result["target_error_rad"],
            "final_pair_distance_m": result["final_pair_distance_m"],
            "min_pair_distance_m": result["min_pair_distance_m"],
            "base_displacement_m": result["base_displacement_m"],
            "first_contact": first_contact,
            "contact_pairs": result["contact_pairs"][:6],
        },
        indent=1,
    ),
    flush=True,
)
print(f"wrote {args.out}", flush=True)
env.close()
