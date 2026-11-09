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

"""Frozen-policy integration rollout for the independent B-distance reward.

No PPO, no parameter update, no default deployment change. The independent SIMPLECOLL config is
used as authored (terminations and randomisation stay enabled); only the command resampling is
frozen for a deterministic 200-step sequence and policy observation corruption is disabled for
inference. The policy is model_9999, loaded through RSL-RL's inference policy.

Usage:
    python scripts/ranger_cr10/rollout_phys_proxy_policy.py --num_envs 64 --steps 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import yaml

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--num_envs", type=int, default=64)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument(
    "--checkpoint",
    default="logs/rsl_rl/ranger_cr10_teleop/2026-11-04_06-57-27_stage410f_ppo/model_9999.pt",
)
ap.add_argument("--out", default="outputs/stage413/b/simple_collision_v2/phys_reward_rollout.json")
args = ap.parse_args()

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import LeggedManip_Lab.tasks  # noqa: E402
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.simple_collision_env_cfg import (  # noqa: E402
    RangerCr10TeleopSimpleCollisionEnvCfg,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.phys_proxy_reward import (  # noqa: E402
    phys_proxy_snapshot,
    reset_cache,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.pose_command_teleop import (  # noqa: E402
    TCP_DEFAULT_POS_B,
    TCP_DEFAULT_QUAT_B,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT = os.path.join(ROOT, args.checkpoint)
OUT = os.path.join(ROOT, args.out)

cfg = RangerCr10TeleopSimpleCollisionEnvCfg()
cfg.seed = args.seed
cfg.scene.num_envs = args.num_envs
cfg.commands.ee_pose.debug_vis = False
# The policy is frozen inference: no observation noise. Terminations and randomisation are
# deliberately kept at the training config; only command resampling is frozen for the sequence.
cfg.observations.policy.enable_corruption = False
cfg.commands.ee_pose.resampling_time_range = (1e6, 1e6)
cfg.commands.base_velocity.resampling_time_range = (1e6, 1e6)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
env = gym.make("RANGER-CR10-TELEOP-SIMPLECOLL", cfg=cfg)
u = env.unwrapped
with open(os.path.join(os.path.dirname(CHECKPOINT), "params", "agent.yaml")) as handle:
    train_cfg = yaml.safe_load(handle)
train_cfg["device"] = str(u.device)
train_cfg["seed"] = args.seed
wrapper = RslRlVecEnvWrapper(env, clip_actions=train_cfg.get("clip_actions"))
runner = OnPolicyRunner(wrapper, train_cfg, log_dir=None, device=u.device)
runner.load(CHECKPOINT)
policy = runner.get_inference_policy(device=u.device)

robot = u.scene["robot"]
contact_sensor = u.scene["contact_forces"]
contact_body_names = list(contact_sensor.body_names)
wheel_rows = [i for i, name in enumerate(contact_body_names) if name.endswith("_wheel_link")]
non_wheel_rows = [i for i in range(len(contact_body_names)) if i not in wheel_rows]
ee_cmd = u.command_manager.get_term("ee_pose")
vel_cmd = u.command_manager.get_term("base_velocity")
dt = u.step_dt
default_pos = torch.tensor(TCP_DEFAULT_POS_B, device=u.device)
default_quat = torch.tensor(TCP_DEFAULT_QUAT_B, device=u.device)

checkpoint_state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
actor_input = int(checkpoint_state["actor_state_dict"]["mlp.0.weight"].shape[1])
obs, _ = wrapper.reset()
obs_dim = int(obs["policy"].shape[-1])
action_dim = int(u.action_manager.total_action_dim)

HOLD = (0.06, -0.05, 0.04)
PHASES = [
    (0, 50, 0.0, 0.0, HOLD),
    (50, 100, 0.25, 0.25, HOLD),
    (100, 150, 0.0, 0.0, (0.10, 0.0, 0.0)),
    (150, 200, -0.25, 0.0, (-0.10, 0.12, 0.0)),
]


def set_commands(vx: float, wz: float, offset) -> None:
    vel_cmd.is_standing_env[:] = False
    vel_cmd.vel_command_b[:, 0] = vx
    vel_cmd.vel_command_b[:, 1] = 0.0
    vel_cmd.vel_command_b[:, 2] = wz
    ee_cmd.pose_command_b[:, :3] = default_pos + torch.tensor(offset, device=u.device)
    ee_cmd.pose_command_b[:, 3:] = default_quat
    vel_cmd.compute(0.0)


def phase_of(step: int):
    for start, end, vx, wz, offset in PHASES:
        if start <= step < end:
            return vx, wz, offset
    return 0.0, 0.0, (0.0, 0.0, 0.0)


term_names = list(u.reward_manager.active_terms)
term_index = {name: index for index, name in enumerate(term_names)}
watch_terms = [name for name in (
    "arm_sensor_clearance", "end_effector_position_tracking_exp", "end_effector_orientation_tracking",
    "track_lin_vel_xy_exp", "track_ang_vel_z_exp", "action_rate_l2", "dof_torques_l2",
) if name in term_index]
term_sums = {name: 0.0 for name in watch_terms}
term_abs = {name: 0.0 for name in watch_terms}
safety_steps = []
link46_distances = []
link46_all_distances = []
pair_stats = {}
negative_abs_all = 0.0
link46_safety = []
link46_min_count = 0
link46_below20 = 0
link46_contact = 0
query_failures = 0
nan_steps = 0
dones_total = 0
term_done_counts = {name: 0 for name in u.termination_manager.active_terms}
explicit_resets = 0
step_times = []
pose_errors = []
orientation_errors = []
vel_errors_xy = []
vel_errors_yaw = []
actual_contact_env_steps = 0
actual_contact_body_events = 0
actual_contact_by_body = {}

obs, _ = wrapper.reset()
reset_cache(u)
for step in range(args.steps):
    if step == 100:
        obs, _ = wrapper.reset()
        reset_cache(u)
        explicit_resets += 1
    vx, wz, offset = phase_of(step)
    set_commands(vx, wz, offset)
    # no_grad, not inference_mode: inference tensors cannot be updated in place by the
    # environment's reset path at step 100 (RuntimeError in write_root_link_pose_to_sim).
    with torch.no_grad():
        actions = policy(obs)
        start = time.perf_counter()
        obs, rewards, dones, extras = wrapper.step(actions)
        elapsed = time.perf_counter() - start
    step_times.append(elapsed)
    forces = contact_sensor.data.net_forces_w
    if forces is not None and non_wheel_rows:
        magnitudes = torch.linalg.norm(forces[:, non_wheel_rows], dim=-1)
        hit = magnitudes > 100.0
        actual_contact_env_steps += int(hit.any(dim=-1).sum().item())
        actual_contact_body_events += int(hit.sum().item())
        for local, body_index in enumerate(non_wheel_rows):
            count = int(hit[:, local].sum().item())
            if count:
                name = contact_body_names[body_index]
                actual_contact_by_body[name] = actual_contact_by_body.get(name, 0) + count
    dones_total += int(dones.sum().item())
    for name in u.termination_manager.active_terms:
        term_done_counts[name] += int(u.termination_manager.get_term(name).sum().item())
    if not bool(torch.isfinite(rewards).all()):
        nan_steps += 1
    safety_idx = term_index.get("arm_sensor_clearance")
    for name in watch_terms:
        value = u.reward_manager._step_reward[:, term_index[name]]
        term_sums[name] += float(value.sum().item()) * dt
        term_abs[name] += float(value.abs().sum().item()) * dt
    if safety_idx is not None:
        safety = u.reward_manager._step_reward[:, safety_idx] * dt
        safety_steps.extend(safety.detach().cpu().tolist())
    negative_abs_all += float((-u.reward_manager._step_reward.clamp(max=0.0) * dt).sum().item())
    snapshot, failures, records = phys_proxy_snapshot(u, SceneEntityCfg("robot"))
    query_failures += failures
    link46 = next(item for item in snapshot if item["pair"] == ["cr10_Link4", "cr10_Link6"])
    link46_distances.append(link46["mean_m"])
    link46_below20 += link46["count_margin_20mm"]
    link46_contact += link46["count_contact"]
    for env_index, env_records in enumerate(records):
        link46_record = next(item for item in env_records if item["pair"] == ["cr10_Link4", "cr10_Link6"])
        link46_all_distances.append(link46_record["distance"])
        min_record = min(env_records, key=lambda item: item["distance"])
        for record in env_records:
            key = tuple(record["pair"])
            entry = pair_stats.setdefault(key, {"pair": list(key), "below20": 0, "contact": 0,
                                                "min_count": 0, "safety_sum": 0.0})
            if record["distance"] <= 0.02:
                entry["below20"] += 1
            if record["distance"] <= 1e-6:
                entry["contact"] += 1
            if record["pair"] == min_record["pair"]:
                entry["min_count"] += 1
                if safety_idx is not None:
                    entry["safety_sum"] += float(
                        u.reward_manager._step_reward[env_index, safety_idx].item()
                    ) * dt
        if min_record["pair"] == ["cr10_Link4", "cr10_Link6"]:
            link46_min_count += 1
            distance = min_record["distance"]
            if safety_idx is not None:
                link46_safety.append(float(u.reward_manager._step_reward[env_index, safety_idx].item()) * dt)
            if distance <= 0.02:
                link46_below20 += 0
    if hasattr(ee_cmd, "metrics"):
        if "position_error" in ee_cmd.metrics:
            pose_errors.append(float(ee_cmd.metrics["position_error"].mean().item()))
        if "orientation_error" in ee_cmd.metrics:
            orientation_errors.append(float(ee_cmd.metrics["orientation_error"].mean().item()))
    if hasattr(vel_cmd, "metrics"):
        if "error_vel_xy" in vel_cmd.metrics:
            vel_errors_xy.append(float(vel_cmd.metrics["error_vel_xy"].mean().item()))
        if "error_vel_yaw" in vel_cmd.metrics:
            vel_errors_yaw.append(float(vel_cmd.metrics["error_vel_yaw"].mean().item()))
    if not bool(torch.isfinite(obs["policy"]).all()) or not bool(torch.isfinite(robot.data.joint_pos).all()):
        nan_steps += 1

summary = {
    "checkpoint": args.checkpoint,
    "num_envs": args.num_envs,
    "steps": args.steps,
    "seed": args.seed,
    "action_dim": action_dim,
    "obs_dim": obs_dim,
    "actor_input_dim": actor_input,
    "policy_loaded": True,
    "normalization": {"actor_obs_normalization": bool(train_cfg["actor"].get("obs_normalization", False))},
    "reward_terms_active": term_names,
    "safety_term": {
        "name": "arm_sensor_clearance",
        "func_module": getattr(u.reward_manager._term_cfgs[term_index["arm_sensor_clearance"]].func, "__module__", None),
        "func_name": getattr(u.reward_manager._term_cfgs[term_index["arm_sensor_clearance"]].func, "__name__", None),
        "weight": float(u.reward_manager._term_cfgs[term_index["arm_sensor_clearance"]].weight),
        "params": {
            key: (value.name if hasattr(value, "name") else value)
            for key, value in u.reward_manager._term_cfgs[term_index["arm_sensor_clearance"]].params.items()
        },
    },
    "terminations_active": list(u.termination_manager.active_terms),
    "events_active": list(u.event_manager.active_terms),
    "command_resampling_frozen": True,
    "observation_corruption": False,
    "randomization": "training-config events left enabled",
    "explicit_resets": explicit_resets,
    "dones_total": dones_total,
    "termination_counts": term_done_counts,
    "nan_or_inf_steps": nan_steps,
    "query_failures": query_failures,
    "step_time_s": {
        "mean": float(np.mean(step_times)),
        "p95": float(np.percentile(step_times, 95)),
        "min": float(np.min(step_times)),
        "max": float(np.max(step_times)),
    },
    "safety_reward": {
        "steps": len(safety_steps),
        "sum": float(np.sum(safety_steps)),
        "mean_per_env_step": float(np.mean(safety_steps)) if safety_steps else None,
        "min": float(np.min(safety_steps)) if safety_steps else None,
        "share_of_negative_abs": None,
    },
    "watch_term_sums": term_sums,
    "watch_term_abs_sums": term_abs,
    "pose_error": {
        "mean": float(np.mean(pose_errors)) if pose_errors else None,
        "p95": float(np.percentile(pose_errors, 95)) if pose_errors else None,
    },
    "orientation_error": {
        "mean": float(np.mean(orientation_errors)) if orientation_errors else None,
        "p95": float(np.percentile(orientation_errors, 95)) if orientation_errors else None,
    },
    "velocity_error": {
        "xy_mean": float(np.mean(vel_errors_xy)) if vel_errors_xy else None,
        "yaw_mean": float(np.mean(vel_errors_yaw)) if vel_errors_yaw else None,
    },
    "actual_physx_contacts": {
        "nonwheel_body_net_force_gt_100n_env_steps": actual_contact_env_steps,
        "nonwheel_body_events": actual_contact_body_events,
        "by_body": actual_contact_by_body,
        "note": "body-level net force, not a pair-specific contact view; wheel-ground contacts are excluded",
    },
    "link4_link6": {
        "steps_with_margin_20mm_env_steps": link46_below20,
        "contact_env_steps": link46_contact,
        "min_pair_env_steps": link46_min_count,
        "safety_contribution_sum": float(np.sum(link46_safety)) if link46_safety else 0.0,
        "distance_step_means": link46_distances,
        "distance_env_steps_quantiles_m": (
            [float(x) for x in np.percentile(link46_all_distances, [0, 10, 25, 50, 75, 90, 100])]
            if link46_all_distances else None
        ),
        "distance_env_steps": len(link46_all_distances),
    },
    "phases": PHASES,
}
if negative_abs_all > 0:
    summary["safety_reward"]["share_of_negative_abs"] = (
        abs(term_sums.get("arm_sensor_clearance", 0.0)) / negative_abs_all
    )
summary["top_pairs"] = sorted(
    pair_stats.values(), key=lambda item: (abs(item["safety_sum"]), item["below20"]), reverse=True
)[:10]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as handle:
    json.dump(summary, handle, indent=1)
print(json.dumps(summary, indent=1)[:12000], flush=True)
print(f"wrote {OUT}", flush=True)
sys.stdout.flush()
sys.stderr.flush()
# Isaac Sim shutdown can take minutes after the artifact is written; exit cleanly without
# relying on the native close callback.
os._exit(0)
