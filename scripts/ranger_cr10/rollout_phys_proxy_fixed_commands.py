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

"""Frozen-policy evaluation with a pre-generated, policy-independent command plan.

The historical training-distribution rollout let the command manager resample on reset; a
policy-specific early termination therefore consumed command RNG and changed every later
command.  This script instead pre-generates the full command plan from ``(eval seed, env id,
global control step)``, applies it before each observation, and prevents reset from resampling
commands.  Initial and reset randomisation use a per-environment seed derived from
``(eval seed, env id, reset count)``; one environment's termination cannot change another
environment's command or reset stream.

The policy still uses the real terminations (base_contact, time_out, bad_orientation).  B
distance, tracking error, reward and contact are captured at reward time, before reset.  The
command plan is written as a hash + per-step digests; a step whose applied command does not
match the plan raises immediately.

Usage:
    python scripts/ranger_cr10/rollout_phys_proxy_fixed_commands.py \
        --num_envs 64 --steps 200 --seed 42 --checkpoint <ckpt.pt> --out <json>
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
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--random-initial-episode-length", action=argparse.BooleanOptionalAction, default=True)
ap.add_argument("--force-reset-step", type=int, default=-1)
ap.add_argument("--force-reset-env", type=int, default=0)
ap.add_argument(
    "--env",
    choices=("simplecoll", "teleop"),
    default="simplecoll",
    help="simplecoll = v2 box asset with self-collision ON (B reward); teleop = the original "
         "mesh asset with self-collision OFF and the 15 mm clearance term.",
)
args = ap.parse_args()

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    euler_xyz_from_quat,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import LeggedManip_Lab.tasks  # noqa: E402
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.simple_collision_env_cfg import (  # noqa: E402
    RangerCr10TeleopSimpleCollisionEnvCfg,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.teleop_env_cfg import (  # noqa: E402
    RangerCr10TeleopEnvCfg,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.phys_proxy_reward import (  # noqa: E402
    cached_pair_distances,
    phys_proxy_snapshot,
    reset_cache,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.pose_command_teleop import (  # noqa: E402
    TCP_DEFAULT_POS_B,
    TCP_DEFAULT_QUAT_B,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.tcp import tcp_pose_b  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(ROOT, args.checkpoint)
OUT = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)

ENV_SPECS = {
    "simplecoll": {
        "gym_id": "RANGER-CR10-TELEOP-SIMPLECOLL",
        "cfg": RangerCr10TeleopSimpleCollisionEnvCfg,
        "asset": "ranger_cr10_simple_v2_flat.usd (box collisions, self-collisions ON)",
        "clearance_term": "phys_proxy_clearance_penalty (B proxy, 44 pairs, 20 mm)",
    },
    "teleop": {
        "gym_id": "RANGER-CR10-TELEOP-WBC",
        "cfg": RangerCr10TeleopEnvCfg,
        "asset": "ranger_cr10.usd (original meshes, self-collisions OFF)",
        "clearance_term": "sensor_clearance_penalty (safety.py proxy, 15 mm margin)",
    },
}
ENV_NAME = args.env
ENV_SPEC = ENV_SPECS[ENV_NAME]

cfg = ENV_SPEC["cfg"]()
cfg.seed = args.seed
cfg.scene.num_envs = args.num_envs
cfg.commands.ee_pose.debug_vis = False
cfg.observations.policy.enable_corruption = False

torch.manual_seed(args.seed)
np.random.seed(args.seed)
env = gym.make(ENV_SPEC["gym_id"], cfg=cfg)
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
anchor_body_id = robot.find_bodies("cr10_Link6")[0][0]
base_body_id = robot.find_bodies("base_link")[0][0]
arm_joint_cols = [robot.joint_names.index(f"cr10_joint{i}") for i in range(1, 7)]

# Fixed phase rule, declared before any comparison. In priority order:
#   reset_transient: episode_length <= RESET_WINDOW (first 1.0 s of an episode, including the
#                    random initial episode lengths)
#   switch:          min(command age) <= SWITCH_WINDOW (0.5 s after an ee or velocity change)
#   hold:            everything else
RESET_WINDOW = 10
SWITCH_WINDOW = 5
PHASES = ("reset_transient", "switch", "hold")


def digest_tensor(*tensors) -> str:
    hasher = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().numpy()
        hasher.update(np.round(array.astype(np.float64), 6).tobytes())
    return hasher.hexdigest()


class CommandPlan:
    """Pre-generated command arrays indexed by (global control step, env id)."""

    def __init__(self, seed: int, steps: int, num_envs: int):
        rng = np.random.default_rng(seed ^ 0x6A09E667)
        self.steps = steps
        self.ee = torch.zeros(steps + 1, num_envs, 7)
        self.vel = torch.zeros(steps + 1, num_envs, 3)
        self.standing = torch.zeros(steps + 1, num_envs, dtype=torch.bool)
        default_pos_cpu = torch.tensor(TCP_DEFAULT_POS_B)
        default_quat_cpu = torch.tensor(TCP_DEFAULT_QUAT_B)
        ee_ranges = ee_cmd.cfg.ranges
        replay_ranges = ee_cmd.cfg.replay_ranges
        replay_ratio = float(ee_cmd.cfg.replay_ratio)
        vel_ranges = vel_cmd.cfg.ranges
        standing_fraction = float(vel_cmd.cfg.rel_standing_envs)

        def uniform(bounds):
            return float(rng.uniform(float(bounds[0]), float(bounds[1])))

        for env_id in range(num_envs):
            # End-effector command epochs.
            t = 0
            while t <= steps:
                interval = max(1, int(round(float(rng.uniform(4.0, 8.0)) / float(dt))))
                end = min(steps + 1, t + interval)
                ranges = replay_ranges if (replay_ranges is not None and rng.random() < replay_ratio) else ee_ranges
                offset = torch.tensor([
                    uniform(ranges.pos_x), uniform(ranges.pos_y), uniform(ranges.pos_z),
                ])
                euler = torch.tensor([
                    uniform(ranges.roll), uniform(ranges.pitch), uniform(ranges.yaw),
                ])
                offset_quat = quat_from_euler_xyz(euler[0:1], euler[1:2], euler[2:3])[0]
                target = torch.cat([default_pos_cpu + offset, quat_mul(offset_quat, default_quat_cpu)])
                self.ee[t:end, env_id] = target
                t = end

            # Base-velocity command epochs, independently resampled.
            t = 0
            while t <= steps:
                interval = max(1, int(round(float(rng.uniform(4.0, 8.0)) / float(dt))))
                end = min(steps + 1, t + interval)
                standing = bool(rng.random() < standing_fraction)
                vx = 0.0 if standing else uniform(vel_ranges.lin_vel_x)
                wz = 0.0 if standing else uniform(vel_ranges.ang_vel_z)
                self.standing[t:end, env_id] = standing
                self.vel[t:end, env_id] = torch.tensor([vx, 0.0, wz])
                t = end

        # Command age per env-step: 0 on the step the epoch changed, then +1. Derived from the
        # plan only, so it is independent of the policy and of resets.
        self.age_ee = torch.zeros(steps + 1, num_envs, dtype=torch.long)
        self.age_vel = torch.zeros(steps + 1, num_envs, dtype=torch.long)
        for t in range(1, steps + 1):
            changed_ee = (self.ee[t] != self.ee[t - 1]).any(dim=-1)
            changed_vel = (self.vel[t] != self.vel[t - 1]).any(dim=-1)
            self.age_ee[t] = torch.where(changed_ee, torch.zeros_like(self.age_ee[t]), self.age_ee[t - 1] + 1)
            self.age_vel[t] = torch.where(changed_vel, torch.zeros_like(self.age_vel[t]), self.age_vel[t - 1] + 1)

        self.ee = self.ee.to(u.device)
        self.vel = self.vel.to(u.device)
        self.standing = self.standing.to(u.device)
        self.age_ee = self.age_ee.to(u.device)
        self.age_vel = self.age_vel.to(u.device)
        self.step_digests = [
            digest_tensor(self.ee[t], self.vel[t]) for t in range(steps)
        ]
        self.plan_hash = digest_tensor(
            self.ee[:steps], self.vel[:steps], self.standing[:steps].float()
        )

    def apply(self, step: int, env_ids=None) -> None:
        index = min(int(step), self.steps)
        if env_ids is None:
            ee_cmd.pose_command_b[:] = self.ee[index]
            vel_cmd.vel_command_b[:] = self.vel[index]
            vel_cmd.is_standing_env[:] = self.standing[index]
        else:
            ids = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(env_ids, device=u.device, dtype=torch.long)
            ee_cmd.pose_command_b[ids] = self.ee[index, ids]
            vel_cmd.vel_command_b[ids] = self.vel[index, ids]
            vel_cmd.is_standing_env[ids] = self.standing[index, ids]


plan = CommandPlan(args.seed, args.steps, args.num_envs)
reset_counts = torch.zeros(args.num_envs, dtype=torch.long, device="cpu")
original_event_apply = u.event_manager.apply


def isolated_event_apply(mode, env_ids=None, dt=None, global_env_step_count=None, _original=original_event_apply):
    if mode != "reset":
        return _original(mode=mode, env_ids=env_ids, dt=dt, global_env_step_count=global_env_step_count)
    ids = list(range(args.num_envs)) if env_ids is None else [int(env_id) for env_id in env_ids]
    for env_id in ids:
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all()
        torch.manual_seed(args.seed * 1000003 + env_id * 7919 + int(reset_counts[env_id].item()) * 104729)
        _original(
            mode=mode,
            env_ids=torch.tensor([env_id], device=u.device, dtype=torch.long),
            dt=dt,
            global_env_step_count=global_env_step_count,
        )
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state_all(cuda_state)
        reset_counts[env_id] += 1
    reset_cache(u)
    return None


u.event_manager.apply = isolated_event_apply


def reset_commands(env_ids=None):
    plan.apply(int(u.common_step_counter), env_ids)
    return {}


def compute_commands(dt):
    plan.apply(int(u.common_step_counter))


u.command_manager.reset = reset_commands
u.command_manager.compute = compute_commands

captured: dict = {}
original_reward_compute = u.reward_manager.compute


def capture_reward(dt, _original=original_reward_compute):
    result = _original(dt)
    last = getattr(u, "_phys_proxy_last", None)
    if last is None or int(last.get("step", -1)) != int(u.common_step_counter):
        tensor, pairs, records, failures = cached_pair_distances(u, robot)
    else:
        tensor, pairs, records, failures = (
            last["tensor"], last["pairs"], last["records"], last.get("failures", 0),
        )
    tcp_pos_b, tcp_quat_b = tcp_pose_b(robot, anchor_body_id, base_body_id)
    pos_error = torch.linalg.norm(tcp_pos_b - ee_cmd.pose_command_b[:, :3], dim=-1)
    ori_error = quat_error_magnitude(tcp_quat_b, ee_cmd.pose_command_b[:, 3:])
    forces = contact_sensor.data.net_forces_w
    if forces is not None and non_wheel_rows:
        magnitudes = torch.linalg.norm(forces[:, non_wheel_rows], dim=-1)
        hit = magnitudes > 100.0
        contact_env_steps = int(hit.any(dim=-1).sum().item())
        contact_body_events = int(hit.sum().item())
        contact_by_body = {
            contact_body_names[body_index]: int(hit[:, local].sum().item())
            for local, body_index in enumerate(non_wheel_rows)
            if int(hit[:, local].sum().item()) > 0
        }
    else:
        contact_env_steps = 0
        contact_body_events = 0
        contact_by_body = {}
    captured.clear()
    captured.update({
        "step": int(u.common_step_counter),
        "reward": float(result.sum().item()),
        "step_reward": u.reward_manager._step_reward.detach().clone(),
        "safety": (u.reward_manager._step_reward[:, safety_idx] * dt).detach().clone(),
        "tensor": tensor.detach().clone(),
        "pairs": pairs,
        "records": records,
        "failures": int(failures),
        "pos_error": pos_error.detach().clone(),
        "ori_error": ori_error.detach().clone(),
        "joint_dev_abs_sum": (
            robot.data.joint_pos[:, arm_joint_cols] - robot.data.default_joint_pos[:, arm_joint_cols]
        ).abs().sum(dim=-1).detach().clone(),
        "joint_dev_abs_max": (
            robot.data.joint_pos[:, arm_joint_cols] - robot.data.default_joint_pos[:, arm_joint_cols]
        ).abs().amax(dim=-1).detach().clone(),
        "ee_command": ee_cmd.pose_command_b.detach().clone(),
        "vel_command": vel_cmd.vel_command_b.detach().clone(),
        "episode_length": u.episode_length_buf.detach().clone(),
        "termination_counts": {
            name: int(u.termination_manager.get_term(name).sum().item()) for name in term_names
        },
        "dones": int((u.termination_manager.terminated | u.termination_manager.time_outs).sum().item()),
        "done_mask": (u.termination_manager.terminated | u.termination_manager.time_outs).detach().clone(),
        "joint_finite": bool(torch.isfinite(robot.data.joint_pos).all().item()),
        "actual_contact_env_steps": contact_env_steps,
        "actual_contact_body_events": contact_body_events,
        "actual_contact_by_body": contact_by_body,
    })
    return result


u.reward_manager.compute = capture_reward

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
initial_command_digest = digest_tensor(ee_cmd.pose_command_b, vel_cmd.vel_command_b)
# Named initial state, so the two assets can be compared joint by joint rather than by a
# single hash that is expected to differ.
initial_joint_pos_by_name = {
    name: float(robot.data.joint_pos[0, index].item())
    for index, name in enumerate(robot.joint_names)
}
initial_arm_joint_pos = {
    f"cr10_joint{i}": float(robot.data.joint_pos[0, arm_joint_cols[i - 1]].item()) for i in range(1, 7)
}
initial_base_height_m = float(robot.data.root_pose_w[0, 2].item())
if initial_command_digest != plan.step_digests[0]:
    raise RuntimeError("initial command does not match the pre-generated plan")

term_names = list(u.termination_manager.active_terms)
term_index = {name: index for index, name in enumerate(u.reward_manager.active_terms)}
safety_idx = term_index.get("arm_sensor_clearance")
if safety_idx is None:
    raise RuntimeError("arm_sensor_clearance is not an active reward term")
watch_terms = [name for name in (
    "arm_sensor_clearance", "end_effector_position_tracking_exp",
    "end_effector_orientation_tracking", "track_lin_vel_xy_exp", "track_ang_vel_z_exp",
    "action_rate_l2", "dof_torques_l2",
) if name in term_index]
term_sums = {name: 0.0 for name in watch_terms}
term_abs = {name: 0.0 for name in watch_terms}

# Every active reward term, in the manager's own order and with its own weight. The weighted
# contribution is `_step_reward * dt`; the raw term value is that divided by the weight.
reward_term_names = list(u.reward_manager.active_terms)
reward_term_weights = {
    name: float(u.reward_manager._term_cfgs[reward_term_names.index(name)].weight)
    for name in reward_term_names
}
term_weighted_sums = {name: 0.0 for name in reward_term_names}
term_raw_sums = {name: 0.0 for name in reward_term_names}
term_env_steps = 0
total_reward_sum = 0.0
total_from_terms = 0.0
phase_env_steps = {phase: 0 for phase in PHASES}
phase_pos = {phase: 0.0 for phase in PHASES}
phase_ori = {phase: 0.0 for phase in PHASES}
phase_safety = {phase: 0.0 for phase in PHASES}
phase_reward = {phase: 0.0 for phase in PHASES}
phase_joint_dev = {phase: 0.0 for phase in PHASES}
phase_joint_dev_max = {phase: 0.0 for phase in PHASES}
phase_term_sums = {phase: {name: 0.0 for name in reward_term_names} for phase in PHASES}

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
step_times: list[float] = []
resample_events = 0

forced_reset = None
for step in range(args.steps):
    if step == args.force_reset_step:
        other_ids = [i for i in range(args.num_envs) if i != args.force_reset_env]
        state_before = digest_tensor(robot.data.joint_pos[other_ids], robot.data.root_pose_w[other_ids])
        new_obs, _ = u.reset(env_ids=[args.force_reset_env])
        reset_cache(u)
        state_after = digest_tensor(robot.data.joint_pos[other_ids], robot.data.root_pose_w[other_ids])
        obs = new_obs
        forced_reset = {
            "step": step, "env": args.force_reset_env,
            "cache_cleared": not hasattr(u, "_phys_proxy_last"),
            "other_env_state_unchanged": state_before == state_after,
            "command_digest_after_reset": digest_tensor(ee_cmd.pose_command_b, vel_cmd.vel_command_b),
            "expected_plan_digest": plan.step_digests[step],
        }
        obs = new_obs
    with torch.no_grad():
        actions = policy(obs)
        start = time.perf_counter()
        obs, rewards, dones, extras = wrapper.step(actions)
        elapsed = time.perf_counter() - start
    step_times.append(elapsed)
    if not captured or int(captured["step"]) != int(u.common_step_counter):
        raise RuntimeError(f"missing pre-reset capture at step {step}")
    command_digest = digest_tensor(captured["ee_command"], captured["vel_command"])
    if command_digest != plan.step_digests[step]:
        raise RuntimeError(f"applied command does not match plan at step {step}")
    step_digests.append(command_digest)
    if step + 1 < args.steps:
        resample_events += int(
            (plan.ee[step + 1] != plan.ee[step]).any(dim=-1).sum().item()
            + (plan.vel[step + 1] != plan.vel[step]).any(dim=-1).sum().item()
        )

    dones_total += int(captured["dones"])
    for name in term_names:
        term_done_counts[name] += int(captured["termination_counts"].get(name, 0))
    query_failures += int(captured["failures"])
    actual_contact_env_steps += int(captured["actual_contact_env_steps"])
    actual_contact_body_events += int(captured["actual_contact_body_events"])
    for body_name, count in captured["actual_contact_by_body"].items():
        actual_contact_by_body[body_name] = actual_contact_by_body.get(body_name, 0) + count
    if not bool(np.isfinite(captured["reward"])) or not bool(captured["joint_finite"]):
        nan_steps += 1

    safety = captured["safety"]
    step_reward = captured["step_reward"]
    for name in watch_terms:
        value = step_reward[:, term_index[name]]
        term_sums[name] += float(value.sum().item()) * dt
        term_abs[name] += float(value.abs().sum().item()) * dt

    # Every active term, weighted contribution and raw value, plus an independent sum used to
    # check that the accounted terms reproduce the total reward.
    weighted_columns = []
    for name in reward_term_names:
        column = step_reward[:, term_index[name]] * dt
        weighted_columns.append(column)
        term_weighted_sums[name] += float(column.sum().item())
        weight = reward_term_weights[name]
        if weight != 0.0:
            term_raw_sums[name] += float((column / weight / dt).sum().item())
    weighted_stack = torch.stack(weighted_columns, dim=-1)
    step_total_from_terms = float(weighted_stack.sum().item())
    total_reward_sum += float(captured["reward"])
    total_from_terms += step_total_from_terms

    # Fixed phase split, applied per env-step.
    episode_length_now = captured["episode_length"]
    age_ee_now = plan.age_ee[min(step, args.steps)]
    age_vel_now = plan.age_vel[min(step, args.steps)]
    command_age = torch.minimum(age_ee_now, age_vel_now)
    phase_masks = {
        "reset_transient": episode_length_now <= RESET_WINDOW,
        "switch": command_age <= SWITCH_WINDOW,
    }
    # Priority: reset_transient wins over switch, switch wins over hold.
    phase_masks["switch"] = phase_masks["switch"] & ~phase_masks["reset_transient"]
    phase_masks["hold"] = ~phase_masks["reset_transient"] & ~phase_masks["switch"]
    joint_dev = captured["joint_dev_abs_sum"]
    joint_dev_max = captured["joint_dev_abs_max"]
    for phase, mask in phase_masks.items():
        count = int(mask.sum().item())
        if count == 0:
            continue
        phase_env_steps[phase] += count
        phase_pos[phase] += float(captured["pos_error"][mask].sum().item())
        phase_ori[phase] += float(captured["ori_error"][mask].sum().item())
        phase_safety[phase] += float(captured["safety"][mask].sum().item())
        phase_reward[phase] += float(weighted_stack[mask].sum().item())
        phase_joint_dev[phase] += float(joint_dev[mask].sum().item())
        phase_joint_dev_max[phase] += float(joint_dev_max[mask].sum().item())
        for name in reward_term_names:
            phase_term_sums[phase][name] += float(weighted_stack[mask, term_index[name]].sum().item())
    term_env_steps += int(u.num_envs)

    distances = captured["tensor"].detach().cpu().numpy()
    pairs = captured["pairs"]
    env_records = captured["records"]
    link46_idx = next(
        idx for idx, pair in enumerate(pairs) if list(pair) == ["cr10_Link4", "cr10_Link6"]
    )
    link46 = distances[:, link46_idx]
    pos_error = captured["pos_error"].detach().cpu()
    ori_error = captured["ori_error"].detach().cpu()
    safety_cpu = safety.detach().cpu()
    ee_command = captured["ee_command"]
    offset = (ee_command[:, :3] - default_pos).detach().cpu()
    rel_quat = quat_mul(quat_inv(default_quat.expand_as(ee_command[:, 3:])), ee_command[:, 3:])
    roll, pitch, yaw = euler_xyz_from_quat(rel_quat)
    euler = torch.stack((roll, pitch, yaw), dim=-1).detach().cpu()
    angle = 2.0 * torch.acos(torch.clamp(rel_quat[:, 0].abs(), 0.0, 1.0)).detach().cpu()
    episode_length = captured["episode_length"].cpu()
    done_cpu = captured["done_mask"].detach().cpu()
    step_link46_below = 0
    step_link46_contact = 0
    for env_index, env_pair_records in enumerate(env_records):
        link46_record = next(item for item in env_pair_records if item["pair"] == ["cr10_Link4", "cr10_Link6"])
        below = bool(link46_record["distance"] <= 0.02)
        contact = bool(link46_record["distance"] <= 1e-6)
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
            "link46_m": float(link46_record["distance"]),
            "below20": below,
            "contact": contact,
            "safety": float(safety_cpu[env_index].item()),
            "pos_err": float(pos_error[env_index].item()),
            "ori_err": float(ori_error[env_index].item()),
            "cmd_offset": [float(value) for value in offset[env_index].tolist()],
            "cmd_euler": [float(value) for value in euler[env_index].tolist()],
            "cmd_angle": float(angle[env_index].item()),
            "cmd_age_ee": int(plan.age_ee[min(step, args.steps)][env_index].item()),
            "cmd_age_vel": int(plan.age_vel[min(step, args.steps)][env_index].item()),
            "phase": (
                "reset_transient"
                if bool(episode_length[env_index].item() <= RESET_WINDOW)
                else "switch"
                if int(min(plan.age_ee[min(step, args.steps)][env_index].item(),
                           plan.age_vel[min(step, args.steps)][env_index].item())) <= SWITCH_WINDOW
                else "hold"
            ),
            "joint_dev_abs_sum": float(captured["joint_dev_abs_sum"][env_index].item()),
            "joint_dev_abs_max": float(captured["joint_dev_abs_max"][env_index].item()),
            "episode_length": int(episode_length[env_index].item()),
            "done": bool(done_cpu[env_index].item()),
        })

    step_summaries.append({
        "step": step,
        "step_time_s": elapsed,
        "reward_sum": float(captured["reward"]),
        "link46_below20_envs": step_link46_below,
        "link46_contact_envs": step_link46_contact,
        "safety_sum": float(safety.sum().item()),
        "pos_err_mean": float(pos_error.mean().item()),
        "ori_err_mean": float(ori_error.mean().item()),
        "cmd_offset_mean": [float(value) for value in offset.mean(dim=0).tolist()],
        "cmd_euler_mean": [float(value) for value in euler.mean(dim=0).tolist()],
        "cmd_angle_mean": float(angle.mean().item()),
        "episode_length_mean": float(episode_length.float().mean().item()),
        "resample_envs": 0,
        "actual_contact_envs": int(captured["actual_contact_env_steps"]),
        "dones": int(captured["dones"]),
        "reset_events_after_initial": int(reset_counts.sum().item() - args.num_envs),
    })
    if step % 20 == 0:
        print(
            f"step {step:3d}/{args.steps} link46<20 {step_link46_below:2d}/{u.num_envs} "
            f"safety {float(safety.sum().item()):+.4f} pos {float(pos_error.mean().item()):.4f} "
            f"angle {float(angle.mean().item()):.3f}",
            flush=True,
        )


positive_total = sum(value for value in term_weighted_sums.values() if value > 0)
negative_total = -sum(value for value in term_weighted_sums.values() if value < 0)


def ranges_of(term) -> dict:
    return {axis: [float(value) for value in getattr(term.cfg.ranges, axis)]
            for axis in ("pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw")}


summary = {
    "checkpoint": CHECKPOINT,
    "checkpoint_sha256": hashlib.sha256(open(CHECKPOINT, "rb").read()).hexdigest(),
    "num_envs": args.num_envs,
    "steps": args.steps,
    "seed": args.seed,
    "env_name": ENV_NAME,
    "env_id": ENV_SPEC["gym_id"],
    "env_asset": ENV_SPEC["asset"],
    "env_clearance_term": ENV_SPEC["clearance_term"],
    "initial_joint_pos_by_name": initial_joint_pos_by_name,
    "initial_arm_joint_pos": initial_arm_joint_pos,
    "initial_base_height_m": initial_base_height_m,
    "robot_joint_names": list(robot.joint_names),
    "ee_anchor_body": "cr10_Link6",
    "ee_measure_frame": "base_link (TCP offset applied to the anchor body)",
    "control_dt_s": float(dt),
    "decimation": int(u.cfg.decimation),
    "initial_digest": initial_digest,
    "initial_command_digest": initial_command_digest,
    "command_plan_hash": plan.plan_hash,
    "command_plan_step_digests": plan.step_digests,
    "step_command_digests": step_digests,
    "command_consistency": {
        "initial_matches_plan": initial_command_digest == plan.step_digests[0],
        "all_steps_match_plan": step_digests == plan.step_digests,
        "steps_checked": args.steps,
        "envs_checked": args.num_envs,
    },
    "protocol": {
        "command_generation": "pre-generated independent plan: eval seed + env id + global control step; 4--8 s ee and velocity epochs; stage-3 box + replay box; standing fraction",
        "command_resampling_independent_of_policy": True,
        "reset_does_not_resample_commands": True,
        "initial_and_reset_randomization": "per-env seed = eval_seed*1000003 + env_id*7919 + reset_count*104729",
        "ee_ranges_actual": ranges_of(ee_cmd),
        "ee_replay_ratio": float(ee_cmd.cfg.replay_ratio),
        "vel_ranges_actual": {axis: [float(value) for value in getattr(vel_cmd.cfg.ranges, axis)]
                             for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z", "heading")},
        "vel_rel_standing_envs": float(vel_cmd.cfg.rel_standing_envs),
        "episode_length_s": float(u.max_episode_length_s),
        "max_episode_length_steps": int(u.max_episode_length),
        "control_dt_s": float(dt),
        "random_initial_episode_length": bool(args.random_initial_episode_length),
        "observation_corruption": False,
        "action_selection": "deterministic mean via OnPolicyRunner.get_inference_policy",
        "training_defaults_changed": False,
        "terminations_active": list(term_names),
        "events_active": list(u.event_manager.active_terms),
    },
    "episode_counts": {
        "dones_total": dones_total,
        "termination_counts": term_done_counts,
        "reset_counts_total": int(reset_counts.sum().item()),
        "reset_events_after_initial": int(reset_counts.sum().item() - args.num_envs),
        "reset_counts_by_env": reset_counts.tolist(),
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
    "reward_terms": {
        name: {
            "weight": reward_term_weights[name],
            "weighted_sum": term_weighted_sums[name],
            "raw_sum": term_raw_sums[name],
            "weighted_per_env_step": term_weighted_sums[name] / max(term_env_steps, 1),
            "raw_per_env_step": term_raw_sums[name] / max(term_env_steps, 1),
            "weighted_share_of_positive": (
                term_weighted_sums[name] / positive_total if term_weighted_sums[name] > 0 and positive_total > 0 else 0.0
            ),
            "weighted_share_of_negative": (
                -term_weighted_sums[name] / negative_total if term_weighted_sums[name] < 0 and negative_total > 0 else 0.0
            ),
        }
        for name in reward_term_names
    },
    "reward_consistency": {
        "term_env_steps": term_env_steps,
        "total_reward_sum": total_reward_sum,
        "total_from_terms": total_from_terms,
        "abs_difference": abs(total_reward_sum - total_from_terms),
        "relative_difference": (
            abs(total_reward_sum - total_from_terms) / abs(total_reward_sum) if total_reward_sum else None
        ),
    },
    "phase_rule": {
        "reset_transient": f"episode_length <= {RESET_WINDOW} control steps (first 1.0 s of an episode)",
        "switch": f"min(ee command age, velocity command age) <= {SWITCH_WINDOW} control steps",
        "hold": "neither of the above",
        "priority": "reset_transient > switch > hold, mutually exclusive",
        "declared_before_comparison": True,
    },
    "phase_split": {
        phase: {
            "env_steps": phase_env_steps[phase],
            "share": phase_env_steps[phase] / max(term_env_steps, 1),
            "pos_err_mean_m": phase_pos[phase] / phase_env_steps[phase] if phase_env_steps[phase] else None,
            "ori_err_mean_rad": phase_ori[phase] / phase_env_steps[phase] if phase_env_steps[phase] else None,
            "safety_per_env_step": phase_safety[phase] / phase_env_steps[phase] if phase_env_steps[phase] else None,
            "reward_per_env_step": phase_reward[phase] / phase_env_steps[phase] if phase_env_steps[phase] else None,
            "joint_dev_abs_sum_mean": phase_joint_dev[phase] / phase_env_steps[phase] if phase_env_steps[phase] else None,
            "joint_dev_abs_max_mean": phase_joint_dev_max[phase] / phase_env_steps[phase] if phase_env_steps[phase] else None,
            "term_weighted_per_env_step": {
                name: phase_term_sums[phase][name] / phase_env_steps[phase] if phase_env_steps[phase] else None
                for name in reward_term_names
            },
            "error_and_duration": (
                "N/A (no env-steps in this phase)" if phase_env_steps[phase] == 0 else "available"
            ),
        }
        for phase in PHASES
    },
    "arm_deviation_stats": {
        "joint_dev_abs_sum_mean": (
            sum(float(record["joint_dev_abs_sum"]) for record in records) / max(len(records), 1)
        ),
        "joint_dev_abs_max_mean": (
            sum(float(record["joint_dev_abs_max"]) for record in records) / max(len(records), 1)
        ),
        "note": "Sum and max of |q - q_default| over cr10_joint1..6, metres of joint angle (rad).",
    },
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
        "quantiles_m": [float(x) for x in np.percentile(
            [item["link46_m"] for item in records], [0, 10, 25, 50, 75, 90, 100]
        )],
    },
    "top_pairs": sorted(pair_stats.values(), key=lambda item: (abs(item["safety_sum"]), item["below20"]), reverse=True)[:10],
    "forced_reset": forced_reset,
    "step_summaries": step_summaries,
    "records": records,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as handle:
    json.dump(summary, handle, indent=1)
print(json.dumps({key: summary[key] for key in (
    "checkpoint", "checkpoint_sha256", "seed", "command_plan_hash", "command_consistency",
    "protocol", "episode_counts", "nan_or_inf_steps", "query_failures", "step_time_s",
    "safety_reward", "actual_physx_contacts", "link4_link6_overall", "top_pairs",
)}, indent=1)[:16000], flush=True)
print(f"wrote {OUT}", flush=True)
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
