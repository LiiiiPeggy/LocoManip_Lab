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

"""64-env pure-simulation validation for the frozen v2 SIMPLECOLL candidate.

No PPO, no policy update, no default deployment change. The action semantics are
explicit: ``use_default_offset=False`` means a zero arm action targets zero, not the
current pose, so every phase builds an explicit hold command. The gripper is not in
the action vector; its actual and target positions are recorded.

Phases:
  * all-env hold 200 control steps;
  * all-env reset then hold;
  * partial reset of 8 envs then hold, checking the other envs do not move;
  * small sinusoidal arm motion (240 steps);
  * base vx=0.2 for 40 steps;
  * env 0 reproduces the box-left blocking target, env 1 the Link3-Link6 target,
    all other envs hold;
  * warmed 50-step timing.

Contact evidence is body-level net force for all envs plus exact per-body pair views
for a few detail envs. A pass does not mean "no contacts anywhere": wheel-ground
contacts are normal, and the two blocking envs must produce their intended contacts.

Run:
    python scripts/ranger_cr10/check_simple_collision_64env.py --num_envs 64 \
        --out outputs/stage413/b/simple_collision_v2/validation_64env.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--num_envs", type=int, default=64)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--out", default="outputs/stage413/b/simple_collision_v2/validation_64env.json")
args = ap.parse_args()

app = AppLauncher(headless=True).app

import importlib.metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import PhysxSchema, UsdPhysics  # noqa: E402

import LeggedManip_Lab.tasks  # noqa: E402
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.simple_collision_env_cfg import (  # noqa: E402
    RangerCr10TeleopSimpleCollisionEnvCfg,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARGET_FILE = os.path.join(ROOT, "outputs/stage413/b/simple_collision_v2/targets_v2.json")
CONTACT_THRESHOLD_N = 100.0

cfg = RangerCr10TeleopSimpleCollisionEnvCfg()
cfg.scene.num_envs = args.num_envs
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

torch.manual_seed(args.seed)
np.random.seed(args.seed)
env = gym.make("RANGER-CR10-TELEOP-SIMPLECOLL", cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
sensor = u.scene["contact_forces"]
env.reset(seed=args.seed)

num_envs = u.num_envs
device = u.device
dt = u.physics_dt
control_dt = dt * cfg.decimation
arm_cols = [robot.joint_names.index(f"cr10_joint{i}") for i in range(1, 7)]
gripper_col = robot.joint_names.index("gripper_finger1_joint")
body_names = list(sensor.body_names)
wheel_rows = [i for i, name in enumerate(body_names) if name.endswith("_wheel_link")]
base_row = body_names.index("base_link")
non_wheel_rows = [i for i in range(len(body_names)) if i not in wheel_rows]
scale = cfg.actions.joint_pos.scale
total_action_dim = u.action_manager.total_action_dim


def hold_action(q=None):
    """Explicit hold; zero action would drive the arm to zero (use_default_offset=False)."""
    if q is None:
        q = robot.data.joint_pos[:, arm_cols]
    q = q.to(device)
    action = torch.zeros(num_envs, total_action_dim, device=device)
    action[:, 2:] = q / scale
    return action


def base_state():
    return {
        "pos": robot.data.root_pos_w.detach().cpu().clone(),
        "quat": robot.data.root_quat_w.detach().cpu().clone(),
        "vel": robot.data.root_vel_w.detach().cpu().clone(),
        "q": robot.data.joint_pos.detach().cpu().clone(),
        "qd": robot.data.joint_vel.detach().cpu().clone(),
    }


def finite_ok():
    tensors = [
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        robot.data.root_vel_w,
        robot.data.joint_pos,
        robot.data.joint_vel,
        sensor.data.net_forces_w,
    ]
    return all(bool(torch.isfinite(t).all()) for t in tensors if t is not None)


def force_matrix():
    forces = sensor.data.net_forces_w
    if forces is None:
        return torch.zeros(num_envs, len(body_names), device=device)
    return torch.linalg.norm(forces, dim=-1)


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
                items.append(
                    {
                        "force_n": float(forces[index]),
                        "position": [float(x) for x in points[index]],
                        "normal": [float(x) for x in normals[index]],
                        "separation_m": float(separations[index]),
                    }
                )
            key = tuple(sorted((sensor_name, filter_name)))
            entry = pairs.setdefault(
                key,
                {"pair": list(key), "force_n": 0.0, "n_points": 0, "min_separation_m": float("inf"),
                 "first_position": None, "first_normal": None},
            )
            entry["force_n"] += float(torch.linalg.norm(force))
            entry["n_points"] += count
            if items:
                entry["min_separation_m"] = min(entry["min_separation_m"], min(i["separation_m"] for i in items))
                if entry["first_position"] is None:
                    entry["first_position"] = items[0]["position"]
                    entry["first_normal"] = items[0]["normal"]
    return list(pairs.values())


physics_view = SimulationManager.get_physics_sim_view()
detail_envs = sorted({i for i in (0, 1, 2, 3) if i < num_envs})
detail_views = {}
for env_id in detail_envs:
    paths = [f"/World/envs/env_{env_id}/Robot/{name}" for name in robot.body_names]
    view = physics_view.create_rigid_contact_view(
        f"/World/envs/env_{env_id}/Robot/*", paths, max_contact_data_count=256
    )
    sensor_names = [str(name) for name in view.sensor_names]
    raw_filter = view.filter_names[0] if view.filter_names else []
    if isinstance(raw_filter, str):
        import ast

        filter_names = list(ast.literal_eval(raw_filter))
    else:
        filter_names = list(raw_filter)
    detail_views[env_id] = (view, sensor_names, filter_names)

# Per-env runtime asset check: self-collision flag and body-filtered prims.
stage = u.sim.stage
env_asset_checks = {}
for env_id in range(num_envs):
    env_asset_checks[env_id] = {
        "self_collisions": None,
        "rigid_bodies": 0,
        "rigid_bodies_with_filters": 0,
        "filtered_targets": 0,
    }
for prim in stage.Traverse():
    path = prim.GetPath().pathString
    if "/Robot/" not in path:
        continue
    parts = path.split("/")
    try:
        env_id = int(parts[3].replace("env_", ""))
    except (IndexError, ValueError):
        continue
    if env_id not in env_asset_checks:
        continue
    if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
        api = PhysxSchema.PhysxArticulationAPI(prim)
        attr = api.GetEnabledSelfCollisionsAttr()
        env_asset_checks[env_id]["self_collisions"] = bool(attr.Get()) if attr and attr.HasValue() else None
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        env_asset_checks[env_id]["rigid_bodies"] += 1
        rel = prim.GetRelationship("physics:filteredPairs")
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI) and rel and rel.GetTargets():
            env_asset_checks[env_id]["rigid_bodies_with_filters"] += 1
            env_asset_checks[env_id]["filtered_targets"] += len(rel.GetTargets())
env_asset_summary = dict(
    unique=len({json.dumps(value, sort_keys=True) for value in env_asset_checks.values()}),
    self_collisions_all_true=all(value["self_collisions"] is True for value in env_asset_checks.values()),
    all_have_filters=all(value["rigid_bodies_with_filters"] > 0 for value in env_asset_checks.values()),
    min_bodies=min(value["rigid_bodies"] for value in env_asset_checks.values()),
    min_filtered_bodies=min(value["rigid_bodies_with_filters"] for value in env_asset_checks.values()),
)

# MuJoCo pair distance at an actual arm q (base fixed at home).
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


def track_phase(steps, action_fn, *, detail_pairs=False, stop_on_nonfinite=False):
    """Run a phase and collect per-env max forces, non-finite counts and detail pairs."""
    max_force = torch.zeros(num_envs, len(body_names), device=device)
    first_over = torch.full((num_envs, len(body_names)), -1, dtype=torch.long, device=device)
    nonfinite = 0
    records = []
    for step in range(steps):
        action = action_fn(step)
        env.step(action)
        forces = force_matrix()
        max_force = torch.maximum(max_force, forces)
        over = (forces > CONTACT_THRESHOLD_N) & (first_over < 0)
        first_over[over] = step
        if not finite_ok():
            nonfinite += 1
            if stop_on_nonfinite:
                records.append({"step": step, "non_finite": True})
                break
        if detail_pairs and step % 5 == 0:
            detail = {}
            for env_id, (view, sensor_names, filter_names) in detail_views.items():
                detail[env_id] = read_pairs(view, sensor_names, filter_names)
            records.append({"step": step, "detail_pairs": detail})
    return max_force, first_over, nonfinite, records


def abnormal_from(max_force, first_over, allowed_bodies=()):
    """Non-wheel bodies above the threshold; ignore explicitly allowed bodies."""
    allowed = set(wheel_rows) | {body_names.index(name) for name in allowed_bodies if name in body_names}
    out = []
    for env_id in range(num_envs):
        for row in range(len(body_names)):
            if row in allowed:
                continue
            if float(max_force[env_id, row]) > CONTACT_THRESHOLD_N:
                out.append(
                    {
                        "env": env_id,
                        "body": body_names[row],
                        "max_force_n": float(max_force[env_id, row]),
                        "first_step": int(first_over[env_id, row]),
                    }
                )
    return out


def env_stats(q_before):
    displacement = (robot.data.root_pos_w.detach().cpu() - q_before["pos"]).norm(dim=-1)
    drift = (robot.data.joint_pos[:, arm_cols].detach().cpu() - q_before["q"][:, arm_cols]).abs().max(dim=-1).values
    quat = robot.data.root_quat_w.detach().cpu()
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return {
        "base_displacement_m": displacement.tolist(),
        "arm_drift_rad": drift.tolist(),
        "base_quat_wxyz_final": quat.tolist(),
        "base_yaw_rad_final": yaw.tolist(),
        "worst_base_env": int(displacement.argmax()),
        "worst_base_m": float(displacement.max()),
        "worst_arm_env": int(drift.argmax()),
        "worst_arm_rad": float(drift.max()),
    }


# ---------------------------------------------------------------- all-env hold
start = base_state()
hold = hold_action(start["q"][:, arm_cols])
initial_arm_target = robot.data.joint_pos_target[:, arm_cols].detach().cpu().clone()
initial_gripper_target = robot.data.joint_pos_target[:, gripper_col].detach().cpu().clone()
initial_gripper_actual = robot.data.joint_pos[:, gripper_col].detach().cpu().clone()
max_force, first_over, nonfinite, _ = track_phase(args.steps, lambda step: hold)
hold_stats = env_stats(start)
hold_abnormal = abnormal_from(max_force, first_over)
hold_contacts = {
    "max_force_by_body": {body_names[row]: float(max_force[:, row].max()) for row in range(len(body_names))},
    "max_wheel_force_n": float(max_force[:, wheel_rows].max()) if wheel_rows else 0.0,
    "max_non_wheel_force_by_env": [float(max_force[env_id, non_wheel_rows].max()) for env_id in range(num_envs)],
    "max_wheel_force_by_env": [float(max_force[env_id, wheel_rows].max()) for env_id in range(num_envs)],
}
hold_arm_target_error = (
    robot.data.joint_pos[:, arm_cols].detach().cpu() - start["q"][:, arm_cols]
).abs().max(dim=-1).values
hold_arm_target = robot.data.joint_pos_target[:, arm_cols].detach().cpu()
hold_arm_actual = robot.data.joint_pos[:, arm_cols].detach().cpu()
action_check = {
    "base_action": [0.0, 0.0],
    "base_actual_velocity_mean": robot.data.root_vel_w[:, :3].detach().cpu().mean(dim=0).tolist(),
    "arm_target_initial": initial_arm_target[0].tolist(),
    "arm_actual_initial": start["q"][0, arm_cols].tolist(),
    "arm_target_error_max_after_hold": float(hold_arm_target_error.max()),
    "arm_target_range": [float(hold_arm_target.min()), float(hold_arm_target.max())],
    "arm_actual_range": [float(hold_arm_actual.min()), float(hold_arm_actual.max())],
    "gripper_target_initial": initial_gripper_target[0].item(),
    "gripper_actual_initial": initial_gripper_actual[0].item(),
    "gripper_target_range": [float(initial_gripper_target.min()), float(initial_gripper_target.max())],
    "gripper_actual_range": [float(robot.data.joint_pos[:, gripper_col].min()),
                             float(robot.data.joint_pos[:, gripper_col].max())],
    "note": "explicit hold; zero action is not a hold because use_default_offset=False",
}
all_reset_start = base_state()

# ---------------------------------------------------------------- all-env reset
env.reset()
post_all_reset = base_state()
reset_base_z_immediate = post_all_reset["pos"][:, 2].tolist()
hold = hold_action(robot.data.joint_pos[:, arm_cols])
max_force, first_over, nonfinite_reset, _ = track_phase(20, lambda step: hold)
reset_base_z = robot.data.root_pos_w[:, 2].detach().cpu().tolist()
reset_arm_offset = (
    robot.data.joint_pos[:, arm_cols].detach().cpu() - start["q"][:, arm_cols]
).abs().max(dim=-1).values
reset_abnormal = abnormal_from(max_force, first_over)

# ---------------------------------------------------------------- partial reset
partial_ids = sorted({0, 7, 13, 21, 34, 42, 55, 63} & set(range(num_envs)))
pre_partial = base_state()
u.reset(env_ids=torch.tensor(partial_ids, dtype=torch.int64, device=device))
post_reset = base_state()
hold = hold_action(post_reset["q"][:, arm_cols])
max_force, first_over, nonfinite_partial, _ = track_phase(20, lambda step: hold)
partial_after = base_state()
partial_nonreset = [i for i in range(num_envs) if i not in partial_ids]
partial_nonreset_delta = (
    (partial_after["pos"][partial_nonreset] - pre_partial["pos"][partial_nonreset]).norm(dim=-1)
    if partial_nonreset else torch.zeros(0)
)
partial_reset_z = post_reset["pos"][partial_ids, 2]
partial_reset_arm = (post_reset["q"][partial_ids][:, arm_cols] - start["q"][partial_ids][:, arm_cols]).abs().max(dim=-1).values
partial_abnormal = abnormal_from(max_force, first_over)

# ---------------------------------------------------------------- small joint motion
motion_start = base_state()
motion_hold = motion_start["q"][:, arm_cols].clone()


def motion_action(step):
    phase = 2.0 * np.pi * step / 100.0
    q_cmd = motion_hold + 0.08 * np.sin(phase)
    return hold_action(q_cmd)


max_force, first_over, nonfinite_motion, _ = track_phase(240, motion_action)
motion_stats = env_stats(motion_start)
motion_abnormal = abnormal_from(max_force, first_over)

# ---------------------------------------------------------------- base motion
base_start = base_state()
base_hold = base_start["q"][:, arm_cols].clone()


def base_action(step):
    action = hold_action(base_hold)
    action[:, 0] = 0.2
    return action


max_force, first_over, nonfinite_base, _ = track_phase(40, base_action)
base_stats = env_stats(base_start)
base_abnormal = abnormal_from(max_force, first_over)
base_velocity = robot.data.root_vel_w[:, 0].detach().cpu().tolist()

# ---------------------------------------------------------------- blocking envs
targets = json.load(open(TARGET_FILE))["selected"]
blocking = {}
blocking_envs = {0: "box_left", 1: "arm_link3_link6"}
blocking_envs = {env_id: key for env_id, key in blocking_envs.items() if env_id < num_envs}
if blocking_envs:
    ids = sorted(blocking_envs)
    u.reset(env_ids=torch.tensor(ids, dtype=torch.int64, device=device))
    block_start = base_state()
    starts = block_start["q"][:, arm_cols].clone().to(device)
    targets_q = {env_id: torch.tensor(targets[key]["q"], dtype=torch.float32, device=device)
                 for env_id, key in blocking_envs.items()}
    block_pairs = {env_id: [] for env_id in ids}
    block_max_force = torch.zeros(num_envs, len(body_names), device=device)
    block_first_over = torch.full((num_envs, len(body_names)), -1, dtype=torch.long, device=device)
    block_steps = 120
    block_rows = {env_id: [] for env_id in ids}
    for step in range(block_steps):
        alpha = (step + 1) / block_steps
        q_cmd = hold_action(starts).clone()
        for env_id in ids:
            q_cmd[env_id, 2:] = (starts[env_id] * (1.0 - alpha) + targets_q[env_id] * alpha) / scale
        env.step(q_cmd)
        forces = force_matrix()
        block_max_force = torch.maximum(block_max_force, forces)
        over = (forces > CONTACT_THRESHOLD_N) & (block_first_over < 0)
        block_first_over[over] = step
        for env_id in ids:
            view, sensor_names, filter_names = detail_views[env_id]
            pairs = read_pairs(view, sensor_names, filter_names)
            if pairs:
                block_pairs[env_id].append({"step": step, "pairs": pairs})
            actual = robot.data.joint_pos[env_id, arm_cols].detach().cpu().numpy()
            if step % 5 == 0 or pairs:
                block_rows[env_id].append(
                    {
                        "step": step,
                        "target_error_rad": float(np.max(np.abs(actual - targets_q[env_id].detach().cpu().numpy()))),
                        "pair_distance_m": pair_distance(targets[blocking_envs[env_id]]["pair"], actual),
                        "base_pos": robot.data.root_pos_w[env_id].detach().cpu().tolist(),
                        "pairs": pairs,
                    }
                )
    for env_id in ids:
        key = blocking_envs[env_id]
        actual = robot.data.joint_pos[env_id, arm_cols].detach().cpu().numpy()
        pairs = block_pairs[env_id]
        first = next((row for row in block_rows[env_id] if row["pairs"]), None)
        blocking[env_id] = {
            "case": key,
            "target_pair": targets[key]["pair"],
            "target_error_rad": float(np.max(np.abs(actual - targets_q[env_id].detach().cpu().numpy()))),
            "final_pair_distance_m": pair_distance(targets[key]["pair"], actual),
            "min_pair_distance_m": min((row["pair_distance_m"] for row in block_rows[env_id]), default=float("inf")),
            "base_displacement_m": float((robot.data.root_pos_w[env_id].detach().cpu() - block_start["pos"][env_id]).norm()),
            "first_contact": first,
            "contact_steps": len(pairs),
            "last_pairs": pairs[-1] if pairs else [],
        }
    expected = {0: {"base_link", "cr10_Link2"}, 1: {"cr10_Link3", "cr10_Link6"}}
    block_abnormal = [
        item
        for item in abnormal_from(block_max_force, block_first_over)
        if not (item["env"] in blocking_envs and item["body"] in expected.get(item["env"], set()))
    ]
else:
    block_abnormal = []

# ---------------------------------------------------------------- timing
env.reset()
warm = hold_action(robot.data.joint_pos[:, arm_cols])
for _ in range(10):
    env.step(warm)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    env.step(warm)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
control_step_s = elapsed / 50.0
physics_step_s = control_step_s / cfg.decimation
throughput_env_steps_s = num_envs / control_step_s
throughput_physics_steps_s = num_envs * cfg.decimation / control_step_s
peak_allocated_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
free_mem, total_mem = torch.cuda.mem_get_info() if torch.cuda.is_available() else (None, None)

baseline = {}
for name in ("hold_on_200.json", "motion_on.json", "reset_base.json", "block_box_on.json",
             "block_arm_front_on.json"):
    path = os.path.join(ROOT, "outputs/stage413/b/simple_collision_v2", name)
    if os.path.exists(path):
        baseline[name] = json.load(open(path))

result = {
    "num_envs": num_envs,
    "seed": args.seed,
    "steps_hold": args.steps,
    "detail_envs": detail_envs,
    "detail_filter_names": {env_id: names for env_id, (_, _, names) in detail_views.items()},
    "env_asset_checks": env_asset_checks,
    "env_asset_summary": env_asset_summary,
    "physics_dt_s": dt,
    "decimation": cfg.decimation,
    "control_dt_s": control_dt,
    "action_semantics": {
        "use_default_offset": False,
        "zero_action_holds_current_target": False,
        "hold_action": "zeros; arm = current_q / scale; base = 0",
        "arm_scale": scale,
        "total_action_dim": total_action_dim,
        "gripper_in_action": False,
        "gripper_joint": "gripper_finger1_joint (mimic linkage, target stays default)",
    },
    "frozen_asset": {
        "usd": "source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10_simple_v2/ranger_cr10_simple_v2_flat.usd",
        "sha256": "683cd45d17bce99002703d61d11531c96d8532ee6a9c453523d78c7b412c9739",
        "rigid_bodies": 23,
        "colliders": 30,
        "body_filter_pairs": 77,
        "self_collisions": True,
        "chassis": "previous version: URDF/PhysX unscaled hull, MuJoCo AABB box",
    },
    "action_check": action_check,
    "hold": {
        "stats": hold_stats,
        "max_force_by_body": hold_contacts["max_force_by_body"],
        "max_wheel_force_n": hold_contacts["max_wheel_force_n"],
        "abnormal_contacts": hold_abnormal,
        "non_finite_steps": nonfinite,
        "gripper_pos_range": [float(robot.data.joint_pos[:, gripper_col].min()),
                              float(robot.data.joint_pos[:, gripper_col].max())],
        "gripper_target_range": [float(robot.data.joint_pos_target[:, gripper_col].min()),
                                 float(robot.data.joint_pos_target[:, gripper_col].max())],
    },
    "all_reset": {
        "base_z_immediate_range": [float(min(reset_base_z_immediate)), float(max(reset_base_z_immediate))],
        "base_z_after_20_steps_range": [float(min(reset_base_z)), float(max(reset_base_z))],
        "arm_offset_max_rad": float(reset_arm_offset.max()),
        "abnormal_contacts": reset_abnormal,
        "non_finite_steps": nonfinite_reset,
    },
    "partial_reset": {
        "reset_envs": partial_ids,
        "reset_base_z": [float(x) for x in partial_reset_z],
        "reset_arm_offset_max_rad": float(partial_reset_arm.max()) if len(partial_reset_arm) else 0.0,
        "non_reset_max_displacement_m": float(partial_nonreset_delta.max()) if len(partial_nonreset_delta) else 0.0,
        "abnormal_contacts": partial_abnormal,
        "non_finite_steps": nonfinite_partial,
    },
    "motion": {
        "stats": motion_stats,
        "abnormal_contacts": motion_abnormal,
        "non_finite_steps": nonfinite_motion,
    },
    "base_motion": {
        "stats": base_stats,
        "abnormal_contacts": base_abnormal,
        "non_finite_steps": nonfinite_base,
        "mean_vx_m_s": float(np.mean(base_velocity)),
    },
    "blocking": {
        "envs": blocking,
        "abnormal_contacts_other_envs": block_abnormal,
    },
    "performance": {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_memory_gb": total_mem / 1e9 if total_mem else None,
        "peak_allocated_gb": peak_allocated_gb,
        "free_memory_gb_after": free_mem / 1e9 if free_mem else None,
        "isaaclab_version": importlib.metadata.version("isaaclab"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "control_step_s": control_step_s,
        "physics_step_s_estimate": physics_step_s,
        "throughput_env_steps_s": throughput_env_steps_s,
        "throughput_physics_steps_s": throughput_physics_steps_s,
        "fps_definition": "control FPS = env control steps per second across all envs; "
                          "per-env control FPS is the same rate for each env",
    },
    "single_env_baseline": baseline,
    "verdict": {
        "all_env_hold": "pass" if not hold_abnormal and nonfinite == 0 else "fail",
        "all_reset": "pass" if not reset_abnormal and nonfinite_reset == 0 else "fail",
        "partial_reset": "pass" if not partial_abnormal and nonfinite_partial == 0
                          and float(partial_nonreset_delta.max() if len(partial_nonreset_delta) else 0.0) < 0.05
                          else "fail",
        "motion": "pass" if not motion_abnormal and nonfinite_motion == 0 else "fail",
        "base_motion": "pass" if not base_abnormal and nonfinite_base == 0 else "fail",
        "blocking": "see per-env first_contact; expected pairs are base_link|cr10_Link2 and "
                    "cr10_Link3|cr10_Link6",
        "env_assets": "pass" if env_asset_summary["self_collisions_all_true"]
                      and env_asset_summary["all_have_filters"] else "fail",
    },
}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as handle:
    json.dump(result, handle, indent=1)
print(json.dumps({
    "num_envs": num_envs,
    "hold": result["hold"]["stats"],
    "hold_abnormal": hold_abnormal[:8],
    "all_reset": result["all_reset"],
    "partial_reset": {k: result["partial_reset"][k] for k in
                      ("reset_envs", "reset_base_z", "reset_arm_offset_max_rad",
                       "non_reset_max_displacement_m")},
    "motion": result["motion"]["stats"],
    "base_motion": result["base_motion"]["stats"],
    "blocking": blocking,
    "performance": result["performance"],
    "verdict": result["verdict"],
    "env_asset_summary": env_asset_summary,
}, indent=1), flush=True)
print(f"wrote {args.out}", flush=True)
env.close()
