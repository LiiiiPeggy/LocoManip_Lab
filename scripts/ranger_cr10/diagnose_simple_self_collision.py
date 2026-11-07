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

"""Record the actual PhysX body-pair contacts during a default-pose hold.

The ContactSensor only reports net force per body, which cannot distinguish a
wheel-ground contact from an arm-box_link contact once ``box_link`` is merged into
``base_link``. This script creates a RigidContactView with every robot body as both
sensor and filter. The force matrix and contact data then identify the body pair,
contact position, normal, separation and impulse-derived force.

Protocol notes:
  * base pose randomisation and joint reset scaling are disabled so on/off start from
    the same home pose;
  * ``base_contact``/``bad_orientation`` terminations are disabled so a bad contact
    cannot teleport the robot through an automatic reset (the earlier 0.5 m
    "displacement" was repeated resets, not motion);
  * the arm holds the reset joint targets, not zero.

Run:
    python scripts/ranger_cr10/diagnose_simple_self_collision.py \
        --self-collisions on --steps 30 --usd <flat.usd> \
        --out outputs/stage413/b/simple_collision_v2/diag_on.json
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--self-collisions", choices=("on", "off"), default="on")
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--usd", default=None)
ap.add_argument("--out", default="outputs/stage413/b/simple_collision_v2/diag.json")
args = ap.parse_args()

app = AppLauncher(headless=True).app

import json  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
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
# Same initial state for on/off: home joint pose, fixed base pose, no gain scaling.
cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
cfg.events.randomize_actuator_gains = None
# No automatic reset from a contact; the point is to observe the contact.
cfg.terminations.base_contact = None
cfg.terminations.bad_orientation = None
cfg.terminations.time_out = None
cfg.actions.joint_pos.use_default_offset = False
cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = args.self_collisions == "on"
if args.usd is not None:
    cfg.scene.robot.spawn.usd_path = args.usd

env = gym.make("RANGER-CR10-TELEOP-SIMPLECOLL", cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
sensor = u.scene["contact_forces"]
env.reset()

# Build the pair view: every robot body is a sensor and every robot body is a filter.
body_paths = [f"/World/envs/env_0/Robot/{name}" for name in robot.body_names]
physics_view = SimulationManager.get_physics_sim_view()
contact_view = physics_view.create_rigid_contact_view(
    "/World/envs/env_0/Robot/*",
    body_paths,
    max_contact_data_count=512,
)
sensor_paths = [str(path) for path in contact_view.sensor_paths]
sensor_names = [str(name) for name in contact_view.sensor_names]
# RigidContactView.filter_names returns one string per sensor, each listing all filters.
# All sensors share the same filter list, so parse the first one to recover the column order.
import ast  # noqa: E402

_raw_filter = contact_view.filter_names[0] if contact_view.filter_names else []
filter_names = list(_raw_filter) if not isinstance(_raw_filter, str) else ast.literal_eval(_raw_filter)
print(
    json.dumps(
        {
            "sensor_count": contact_view.sensor_count,
            "filter_count": contact_view.filter_count,
            "sensor_names": sensor_names,
            "filter_names": filter_names,
        }
    ),
    flush=True,
)

arm_idx = [robot.joint_names.index(f"cr10_joint{i}") for i in range(1, 7)]
start = robot.data.joint_pos[:, arm_idx].clone()
scale = cfg.actions.joint_pos.scale
hold = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
hold[:, 2:] = start / scale
dt = u.physics_dt
base0 = robot.data.root_pos_w.clone()

step_records = []
for step in range(args.steps):
    env.step(hold)
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
            for index in range(start_index, start_index + count):
                if index >= len(forces):
                    break
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
                    "sensor": sensor_name,
                    "filter": filter_name,
                    "force_n": float(torch.linalg.norm(force)),
                    "force_vector_n": [float(x) for x in force],
                    "n_points": count,
                    "min_separation_m": min((item["separation_m"] for item in items), default=None),
                    "first_point": items[0]["position"] if items else None,
                    "first_normal": items[0]["normal"] if items else None,
                }
            )
    step_records.append(
        {
            "step": step,
            "base_pos": [float(x) for x in robot.data.root_pos_w[0]],
            "base_vel": [float(x) for x in robot.data.root_vel_w[0]],
            "arm_q": [float(x) for x in robot.data.joint_pos[0, arm_idx]],
            "episode_length": int(u.episode_length_buf[0]),
            "pairs": sorted(pairs, key=lambda item: item["force_n"], reverse=True),
            "net_force_by_body": {
                sensor.body_names[index]: float(torch.linalg.norm(sensor.data.net_forces_w[0, index]).item())
                for index in range(len(sensor.body_names))
            },
        }
    )

aggregate = {}
for row in step_records:
    for pair in row["pairs"]:
        key = "|".join(sorted((pair["sensor"], pair["filter"])))
        entry = aggregate.setdefault(
            key,
            {
                "pair": sorted((pair["sensor"], pair["filter"])),
                "first_step": row["step"],
                "n_steps": 0,
                "max_force_n": 0.0,
                "min_separation_m": float("inf"),
                "first_point": pair["first_point"],
                "first_normal": pair["first_normal"],
            },
        )
        entry["n_steps"] += 1
        entry["max_force_n"] = max(entry["max_force_n"], pair["force_n"])
        if pair["min_separation_m"] is not None:
            entry["min_separation_m"] = min(entry["min_separation_m"], pair["min_separation_m"])

max_arm_drift = max(
    (float(torch.max(torch.abs(torch.tensor(row["arm_q"]) - start[0].cpu())).item()) for row in step_records),
    default=0.0,
)
result = {
    "self_collisions": args.self_collisions,
    "usd": args.usd or "env default",
    "steps": args.steps,
    "physics_dt_s": dt,
    "env_dt_s": dt * cfg.decimation,
    "sensor_paths": sensor_paths,
    "sensor_names": sensor_names,
    "filter_names": filter_names,
    "sensor_count": contact_view.sensor_count,
    "filter_count": contact_view.filter_count,
    "max_arm_drift_rad": max_arm_drift,
    "base_displacement_m": float((robot.data.root_pos_w - base0).norm().item()),
    "aggregate": sorted(aggregate.values(), key=lambda item: item["max_force_n"], reverse=True),
    "step_records": step_records,
    "note": "force_n is the normal contact force returned by PhysX (impulse / physics_dt) for a "
            "body pair; positions and normals are the first contact point for that pair; "
            "min_separation is the PhysX separation (negative means penetration)",
}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as handle:
    json.dump(result, handle, indent=1)
print(
    json.dumps(
        {
            "self_collisions": args.self_collisions,
            "base_displacement_m": result["base_displacement_m"],
            "aggregate": result["aggregate"][:15],
        },
        indent=1,
    ),
    flush=True,
)
print(f"wrote {args.out}", flush=True)
env.close()
