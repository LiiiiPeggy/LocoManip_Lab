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

"""Frozen-policy rollout under the *training* command distribution.

The fixed HOLD/offset evaluation in ``rollout_phys_proxy_policy.py`` answers a different
question from training: it freezes both command terms and holds the tool orientation at the
default.  This script keeps the training command generation (4--8 s resampling, stage-3
workspace box, replay box, standing fraction) and the training reset/startup randomisation, so
the Link4--Link6 marginal rate can be compared with the training log on the same distribution.

Deliberate inference-only changes, recorded in the artifact:
    * ``observations.policy.enable_corruption = False`` (deterministic inference, same for both
      policies; training itself had corruption enabled);
    * the policy is the deterministic mean from ``get_inference_policy`` (training sampled from
      the Gaussian);
    * random initial episode lengths are set exactly as ``OnPolicyRunner.learn`` does, unless
      ``--no-random-initial-episode-length`` is passed.

One checkpoint per process.  Run the original and candidate with the same seed so the reset,
command samples and random initial episode lengths are drawn from the same RNG stream; the
artifact stores per-step command digests so accidental divergence is measurable.

Usage:
    python scripts/ranger_cr10/rollout_phys_proxy_training_dist.py \
        --checkpoint <original.pt> --out outputs/stage413/b/dist_explain/train_dist_original.json
"""

from __future__ import annotations

import argparse
import hashlib
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
ap.add_argument("--out", required=True)
ap.add_argument("--random-initial-episode-length", action=argparse.BooleanOptionalAction, default=True)
args = ap.parse_args()

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat, quat_inv, quat_mul  # noqa: E402
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
CHECKPOINT = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(ROOT, args.checkpoint)
OUT = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)

cfg = RangerCr10TeleopSimpleCollisionEnvCfg()
cfg.seed = args.seed
cfg.scene.num_envs = args.num_envs
cfg.commands.ee_pose.debug_vis = False
# Inference protocol, not a training change: no observation noise, deterministic mean action.
cfg.observations.policy.enable_corruption = False

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


def digest_tensor(*tensors) -> str:
    hasher = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().numpy()
        hasher.update(np.round(array.astype(np.float64), 6).tobytes())
    return hasher.hexdigest()


def command_pose() -> torch.Tensor:
    return ee_cmd.pose_command_b.detach().clone()


def relative_euler_and_angle() -> tuple[torch.Tensor, torch.Tensor]:
    quat = ee_cmd.pose_command_b[:, 3:]
    relative = quat_mul(quat_inv(default_quat.expand_as(quat)), quat)
    roll, pitch, yaw = euler_xyz_from_quat(relative)
    angle = 2.0 * torch.acos(torch.clamp(relative[:, 0].abs(), 0.0, 1.0))
    return torch.stack((roll, pitch, yaw), dim=-1), angle


def command_digest() -> str:
    return digest_tensor(ee_cmd.pose_command_b, vel_cmd.vel_command_b)


obs, _ = wrapper.reset()
if args.random_initial_episode_length:
    u.episode_length_buf = torch.randint_like(u.episode_length_buf, high=int(u.max_episode_length))
reset_cache(u)
initial_digest = digest_tensor(
    robot.data.joint_pos,
    robot.data.root_pose_w,
    ee_cmd.pose_command_b,
    vel_cmd.vel_command_b,
    u.episode_length_buf,
)

term_names = list(u.termination_manager.active_terms)
term_index = {name: index for index, name in enumerate(u.reward_manager.active_terms)}
watch_terms = [name for name in (
    "arm_sensor_clearance",
    "end_effector_position_tracking_exp",
    "end_effector_orientation_tracking",
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "action_rate_l2",
    "dof_torques_l2",
) if name in term_index]
term_sums = {name: 0.0 for name in watch_terms}
term_abs = {name: 0.0 for name in watch_terms}
safety_idx = term_index.get("arm_sensor_clearance")
if safety_idx is None:
    raise RuntimeError("arm_sensor_clearance is not an active reward term")

pair_stats: dict[tuple[str, str], dict] = {}
records: list[dict] = []
step_summaries: list[dict] = []
step_digests: list[str] = []
actual_contact_env_steps = 0
actual_contact_body_events = 0
actual_contact_by_body: dict[str, int] = {}
query_failures = 0
nan_steps = 0
dones_total = 0
term_done_counts = {name: 0 for name in term_names}
explicit_resets = 0
step_times: list[float] = []
resample_events = 0
previous_counter = ee_cmd.command_counter.clone()
command_age = torch.zeros(u.num_envs, dtype=torch.long, device=u.device)

for step in range(args.steps):
    with torch.no_grad():
        actions = policy(obs)
        start = time.perf_counter()
        obs, rewards, dones, extras = wrapper.step(actions)
        elapsed = time.perf_counter() - start
    step_times.append(elapsed)
    step_digests.append(command_digest())
    dones_total += int(dones.sum().item())
    for name in term_names:
        term_done_counts[name] += int(u.termination_manager.get_term(name).sum().item())

    forces = contact_sensor.data.net_forces_w
    step_contact_envs = 0
    if forces is not None and non_wheel_rows:
        magnitudes = torch.linalg.norm(forces[:, non_wheel_rows], dim=-1)
        hit = magnitudes > 100.0
        step_contact_envs = int(hit.any(dim=-1).sum().item())
        actual_contact_env_steps += step_contact_envs
        actual_contact_body_events += int(hit.sum().item())
        for local, body_index in enumerate(non_wheel_rows):
            count = int(hit[:, local].sum().item())
            if count:
                name = contact_body_names[body_index]
                actual_contact_by_body[name] = actual_contact_by_body.get(name, 0) + count

    safety = u.reward_manager._step_reward[:, safety_idx] * dt
    for name in watch_terms:
        value = u.reward_manager._step_reward[:, term_index[name]]
        term_sums[name] += float(value.sum().item()) * dt
        term_abs[name] += float(value.abs().sum().item()) * dt

    snapshot, failures, env_records = phys_proxy_snapshot(u, SceneEntityCfg("robot"))
    query_failures += failures
    if not bool(torch.isfinite(rewards).all()):
        nan_steps += 1
    if not bool(torch.isfinite(robot.data.joint_pos).all()):
        nan_steps += 1

    counter_now = ee_cmd.command_counter
    changed = counter_now != previous_counter
    resample_events += int(changed.sum().item())
    command_age = torch.where(changed, torch.zeros_like(command_age), command_age + 1)
    previous_counter = counter_now.clone()

    euler, angle = relative_euler_and_angle()
    offset = ee_cmd.pose_command_b[:, :3] - default_pos
    pos_error = ee_cmd.metrics["position_error"].detach().cpu()
    ori_error = ee_cmd.metrics["orientation_error"].detach().cpu()
    safety_cpu = safety.detach().cpu()
    offset_cpu = offset.detach().cpu()
    euler_cpu = euler.detach().cpu()
    angle_cpu = angle.detach().cpu()
    age_cpu = command_age.detach().cpu()
    time_left_cpu = ee_cmd.time_left.detach().cpu()
    episode_length_cpu = u.episode_length_buf.detach().cpu()
    done_cpu = dones.detach().cpu()

    step_link46_below = 0
    step_link46_contact = 0
    for env_index, env_pair_records in enumerate(env_records):
        link46 = next(item for item in env_pair_records if item["pair"] == ["cr10_Link4", "cr10_Link6"])
        below = bool(link46["distance"] <= 0.02)
        contact = bool(link46["distance"] <= 1e-6)
        step_link46_below += int(below)
        step_link46_contact += int(contact)
        min_record = min(env_pair_records, key=lambda item: item["distance"])
        for record in env_pair_records:
            key = tuple(record["pair"])
            entry = pair_stats.setdefault(
                key, {"pair": list(key), "below20": 0, "contact": 0, "min_count": 0, "safety_sum": 0.0}
            )
            if record["distance"] <= 0.02:
                entry["below20"] += 1
            if record["distance"] <= 1e-6:
                entry["contact"] += 1
            if record["pair"] == min_record["pair"]:
                entry["min_count"] += 1
                entry["safety_sum"] += float(safety_cpu[env_index].item())
        records.append({
            "step": step,
            "env": env_index,
            "link46_m": float(link46["distance"]),
            "below20": below,
            "contact": contact,
            "safety": float(safety_cpu[env_index].item()),
            "pos_err": float(pos_error[env_index].item()),
            "ori_err": float(ori_error[env_index].item()),
            "cmd_offset": [float(value) for value in offset_cpu[env_index].tolist()],
            "cmd_euler": [float(value) for value in euler_cpu[env_index].tolist()],
            "cmd_angle": float(angle_cpu[env_index].item()),
            "cmd_age": int(age_cpu[env_index].item()),
            "time_left": float(time_left_cpu[env_index].item()),
            "episode_length": int(episode_length_cpu[env_index].item()),
            "done": bool(done_cpu[env_index].item()),
        })

    step_summaries.append({
        "step": step,
        "step_time_s": elapsed,
        "link46_below20_envs": step_link46_below,
        "link46_contact_envs": step_link46_contact,
        "safety_sum": float(safety.sum().item()),
        "pos_err_mean": float(pos_error.mean().item()),
        "ori_err_mean": float(ori_error.mean().item()),
        "cmd_offset_mean": [float(value) for value in offset_cpu.mean(dim=0).tolist()],
        "cmd_euler_mean": [float(value) for value in euler_cpu.mean(dim=0).tolist()],
        "cmd_angle_mean": float(angle_cpu.mean().item()),
        "cmd_age_mean": float(age_cpu.float().mean().item()),
        "time_left_mean": float(time_left_cpu.mean().item()),
        "episode_length_mean": float(episode_length_cpu.float().mean().item()),
        "resample_envs": int(changed.sum().item()),
        "actual_contact_envs": step_contact_envs,
        "dones": int(dones.sum().item()),
    })
    if step % 20 == 0:
        print(
            f"step {step:3d}/{args.steps} link46<20 {step_link46_below:2d}/{u.num_envs} "
            f"safety {float(safety.sum().item()):+.4f} pos {float(pos_error.mean().item()):.4f} "
            f"angle {float(angle_cpu.mean().item()):.3f} age {float(age_cpu.float().mean().item()):.1f}",
            flush=True,
        )

snapshot, failures, last_records = phys_proxy_snapshot(u, SceneEntityCfg("robot"))

def ranges_of(term) -> dict:
    return {
        axis: [float(value) for value in getattr(term.cfg.ranges, axis)]
        for axis in ("pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw")
    }


summary = {
    "checkpoint": CHECKPOINT,
    "num_envs": args.num_envs,
    "steps": args.steps,
    "seed": args.seed,
    "initial_digest": initial_digest,
    "step_command_digests": step_digests,
    "protocol": {
        "command_generation": "training config: independent 4--8 s resampling, stage-3 box, replay box",
        "ee_ranges_actual": ranges_of(ee_cmd),
        "ee_resampling_time_range_s": [float(v) for v in ee_cmd.cfg.resampling_time_range],
        "ee_replay_ratio": float(ee_cmd.cfg.replay_ratio),
        "ee_stage_index": int(ee_cmd.stage_index),
        "ee_reachable_fraction": float(ee_cmd.reachable_fraction),
        "vel_ranges_actual": {
            axis: [float(value) for value in getattr(vel_cmd.cfg.ranges, axis)]
            for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z", "heading")
        },
        "vel_resampling_time_range_s": [float(v) for v in vel_cmd.cfg.resampling_time_range],
        "vel_rel_standing_envs": float(vel_cmd.cfg.rel_standing_envs),
        "episode_length_s": float(u.max_episode_length_s),
        "max_episode_length_steps": int(u.max_episode_length),
        "decimation": int(u.cfg.decimation),
        "control_dt_s": float(dt),
        "random_initial_episode_length": bool(args.random_initial_episode_length),
        "observation_corruption": False,
        "action_selection": "deterministic mean via OnPolicyRunner.get_inference_policy",
        "explicit_resets": explicit_resets,
        "terminations_active": list(term_names),
        "events_active": list(u.event_manager.active_terms),
    },
    "episode_counts": {
        "dones_total": dones_total,
        "termination_counts": term_done_counts,
        "resample_env_events": resample_events,
    },
    "nan_or_inf_steps": nan_steps,
    "query_failures": query_failures,
    "step_time_s": {
        "mean": float(np.mean(step_times)),
        "p95": float(np.percentile(step_times, 95)),
        "min": float(np.min(step_times)),
        "max": float(np.max(step_times)),
    },
    "safety_reward": {
        "sum": float(sum(item["safety_sum"] for item in step_summaries)),
        "per_env_step": float(sum(item["safety_sum"] for item in step_summaries) / (args.steps * args.num_envs)),
        "share_of_negative_abs": abs(term_sums.get("arm_sensor_clearance", 0.0)) / sum(term_abs.values())
        if sum(term_abs.values()) else None,
    },
    "watch_term_sums": term_sums,
    "watch_term_abs_sums": term_abs,
    "actual_physx_contacts": {
        "nonwheel_body_net_force_gt_100n_env_steps": actual_contact_env_steps,
        "nonwheel_body_events": actual_contact_body_events,
        "by_body": actual_contact_by_body,
        "note": "body-level net force, not a pair-specific contact view; wheel-ground contacts are excluded",
    },
    "link4_link6_overall": {
        "env_steps": len(records),
        "below20": sum(int(item["below20"]) for item in records),
        "contact": sum(int(item["contact"]) for item in records),
        "mean_m": float(np.mean([item["link46_m"] for item in records])),
        "quantiles_m": [
            float(x) for x in np.percentile(
                [item["link46_m"] for item in records], [0, 10, 25, 50, 75, 90, 100]
            )
        ],
    },
    "top_pairs": sorted(
        pair_stats.values(), key=lambda item: (abs(item["safety_sum"]), item["below20"]), reverse=True
    )[:10],
    "step_summaries": step_summaries,
    "records": records,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as handle:
    json.dump(summary, handle, indent=1)
print(json.dumps({key: summary[key] for key in (
    "checkpoint", "seed", "protocol", "episode_counts", "nan_or_inf_steps", "query_failures",
    "step_time_s", "safety_reward", "actual_physx_contacts", "link4_link6_overall", "top_pairs",
)}, indent=1)[:12000], flush=True)
print(f"wrote {OUT}", flush=True)
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
