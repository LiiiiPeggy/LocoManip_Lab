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

"""Budgeted short PPO fine-tune on the independent SIMPLECOLL B-reward config.

Loads model_9999 with the checkpoint's compatible agent.yaml, performs real PPO updates (no
no_grad around the update), and stops at ``--max_iterations`` or the wall-clock budget. A complete
actor/critic/optimizer checkpoint is saved at the last completed update. No default deployment,
default safety module, USD or model_9999 is overwritten.

Run:
    python scripts/ranger_cr10/train_phys_proxy_short.py --num_envs 64 --max_iterations 50 \
        --seed 20261108 --wall_minutes 60
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
import time
from datetime import datetime

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--num_envs", type=int, default=64)
ap.add_argument("--max_iterations", type=int, default=50)
ap.add_argument("--seed", type=int, default=20261108)
ap.add_argument("--wall_minutes", type=float, default=60.0)
ap.add_argument(
    "--checkpoint",
    default="logs/rsl_rl/ranger_cr10_teleop/2026-11-04_06-57-27_stage410f_ppo/model_9999.pt",
)
ap.add_argument("--run_name", default="physproxy_short_ppo")
ap.add_argument(
    "--baseline_run",
    default="logs/rsl_rl/ranger_cr10_teleop/2026-11-08_00-11-40_physproxy_short_ppo_50",
    help="Previous run whose behavior config is compared before training.",
)
args = ap.parse_args()

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import LeggedManip_Lab.tasks  # noqa: E402
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.simple_collision_env_cfg import (  # noqa: E402
    RangerCr10TeleopSimpleCollisionEnvCfg,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT = os.path.join(ROOT, args.checkpoint)
RUN_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_DIR = os.path.join(ROOT, "logs", "rsl_rl", "ranger_cr10_teleop", f"{RUN_STAMP}_{args.run_name}")
os.makedirs(os.path.join(LOG_DIR, "params"), exist_ok=True)
METRICS_PATH = os.path.join(LOG_DIR, "train_metrics.jsonl")
LR_KL_PATH = os.path.join(LOG_DIR, "lr_kl.jsonl")
CONFIG_PATH = os.path.join(LOG_DIR, "run_config.json")
CONFIG_DIFF_PATH = os.path.join(LOG_DIR, "config_diff.json")
SUMMARY_PATH = os.path.join(LOG_DIR, "run_summary.json")
BASELINE_RUN = os.path.join(ROOT, args.baseline_run)


class StopTraining(RuntimeError):
    pass


class Monitor:
    """Per-update metrics without changing the environment or reward."""

    def __init__(self, env, runner, wall_seconds: float, checkpoint_iter: int):
        self.env = env
        self.runner = runner
        self.wall_seconds = wall_seconds
        self.checkpoint_iter = int(checkpoint_iter)
        self.started = time.time()
        self.term_names = list(env.termination_manager.active_terms)
        self.term_counts = {name: 0 for name in self.term_names}
        self.completed = 0
        self.env_steps = 0
        self.b_failures = 0
        self.safety_sum = 0.0
        self.safety_abs = 0.0
        self.link46_below20 = 0
        self.link46_contact = 0
        self.link46_min_count = 0
        self.actual_contact_env_steps = 0
        self.actual_contact_body_events = 0
        self.track_pos = []
        self.track_ori = []
        self.vel_xy = []
        self.vel_yaw = []
        self.abort_reason = None
        self.consecutive_failures = 0
        self.updates = 0
        self.reward_manager = env.reward_manager
        self.safety_idx = self.reward_manager.active_terms.index("arm_sensor_clearance")
        self.ee_cmd = env.command_manager.get_term("ee_pose")
        self.vel_cmd = env.command_manager.get_term("base_velocity")
        self.sensor = env.scene["contact_forces"]
        self.body_names = list(self.sensor.body_names)
        self.wheel_rows = [i for i, name in enumerate(self.body_names) if name.endswith("_wheel_link")]
        self.non_wheel_rows = [i for i in range(len(self.body_names)) if i not in self.wheel_rows]
        self.control_dt = env.step_dt

    def after_step(self, observation, reward, terminated, truncated, extras):
        self.env_steps += self.env.num_envs
        dones = terminated | truncated
        self.completed += int(dones.sum().item())
        for name in self.term_names:
            self.term_counts[name] += int(self.env.termination_manager.get_term(name).sum().item())
        last = getattr(self.env, "_phys_proxy_last", None)
        if last is not None:
            self.b_failures += int(last.get("failures", 0))
            records = last.get("records", [])
            for env_records in records:
                link46 = next(
                    (record for record in env_records if record["pair"] == ["cr10_Link4", "cr10_Link6"]), None
                )
                if link46 is None:
                    continue
                if link46["distance"] <= 0.02:
                    self.link46_below20 += 1
                if link46["distance"] <= 1e-6:
                    self.link46_contact += 1
                minimum = min(env_records, key=lambda record: record["distance"])
                if minimum["pair"] == ["cr10_Link4", "cr10_Link6"]:
                    self.link46_min_count += 1
        safety = self.reward_manager._step_reward[:, self.safety_idx] * self.control_dt
        self.safety_sum += float(safety.sum().item())
        self.safety_abs += float(safety.abs().sum().item())
        forces = self.sensor.data.net_forces_w
        if forces is not None and self.non_wheel_rows:
            magnitudes = torch.linalg.norm(forces[:, self.non_wheel_rows], dim=-1)
            self.actual_contact_env_steps += int((magnitudes > 100.0).any(dim=-1).sum().item())
            self.actual_contact_body_events += int((magnitudes > 100.0).sum().item())
        if hasattr(self.ee_cmd, "metrics"):
            if "position_error" in self.ee_cmd.metrics:
                self.track_pos.append(float(self.ee_cmd.metrics["position_error"].mean().item()))
            if "orientation_error" in self.ee_cmd.metrics:
                self.track_ori.append(float(self.ee_cmd.metrics["orientation_error"].mean().item()))
        if hasattr(self.vel_cmd, "metrics"):
            if "error_vel_xy" in self.vel_cmd.metrics:
                self.vel_xy.append(float(self.vel_cmd.metrics["error_vel_xy"].mean().item()))
            if "error_vel_yaw" in self.vel_cmd.metrics:
                self.vel_yaw.append(float(self.vel_cmd.metrics["error_vel_yaw"].mean().item()))
        if not bool(torch.isfinite(reward).all()):
            self.abort_reason = "non_finite_reward"
            raise StopTraining(self.abort_reason)
        if not bool(torch.isfinite(self.env.scene["robot"].data.joint_pos).all()):
            self.abort_reason = "non_finite_joint_state"
            raise StopTraining(self.abort_reason)

    def consume(self, kwargs):
        self.updates += 1
        loss = {key: float(value) for key, value in kwargs.get("loss_dict", {}).items()}
        item = {
            "iteration": int(kwargs.get("it", -1)),
            "collect_time_s": float(kwargs.get("collect_time", 0.0)),
            "learn_time_s": float(kwargs.get("learn_time", 0.0)),
            "learning_rate": float(kwargs.get("learning_rate", 0.0)),
            "lr_schedule": getattr(self.runner.alg, "_lr_kl_last", None),
            "mean_std": float(kwargs.get("action_std").mean().item()) if kwargs.get("action_std") is not None else None,
            "loss": loss,
            "episode_return_mean": float(statistics.mean(self.runner.logger.rewbuffer))
            if len(self.runner.logger.rewbuffer) else None,
            "episode_length_mean": float(statistics.mean(self.runner.logger.lenbuffer))
            if len(self.runner.logger.lenbuffer) else None,
            "env_steps": self.env_steps,
            "completed_episodes": self.completed,
            "termination_counts": dict(self.term_counts),
            "b_query_failures": self.b_failures,
            "b_safety_sum": self.safety_sum,
            "b_safety_abs": self.safety_abs,
            "link4_link6_below20_env_steps": self.link46_below20,
            "link4_link6_contact_env_steps": self.link46_contact,
            "link4_link6_min_pair_env_steps": self.link46_min_count,
            "actual_nonwheel_contact_env_steps": self.actual_contact_env_steps,
            "actual_nonwheel_contact_body_events": self.actual_contact_body_events,
            "ee_position_error_mean": float(np.mean(self.track_pos)) if self.track_pos else None,
            "ee_orientation_error_mean": float(np.mean(self.track_ori)) if self.track_ori else None,
            "vel_xy_error_mean": float(np.mean(self.vel_xy)) if self.vel_xy else None,
            "vel_yaw_error_mean": float(np.mean(self.vel_yaw)) if self.vel_yaw else None,
        }
        if item["iteration"] == self.checkpoint_iter:
            item["lr_kl_first_update_minibatches"] = getattr(
                self.runner.alg, "_lr_kl_last_records", None
            )
        self.term_counts = {name: 0 for name in self.term_names}
        self.completed = 0
        self.env_steps = 0
        self.b_failures = 0
        self.safety_sum = 0.0
        self.safety_abs = 0.0
        self.link46_below20 = 0
        self.link46_contact = 0
        self.link46_min_count = 0
        self.actual_contact_env_steps = 0
        self.actual_contact_body_events = 0
        self.track_pos = []
        self.track_ori = []
        self.vel_xy = []
        self.vel_yaw = []
        return item


def _optimizer_state_dict_equal(left: dict, right: dict) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False
    if set(left["state"].keys()) != set(right["state"].keys()):
        return False
    for key in left["state"]:
        left_state = left["state"][key]
        right_state = right["state"][key]
        if set(left_state.keys()) != set(right_state.keys()):
            return False
        for name, value in left_state.items():
            other = right_state[name]
            if torch.is_tensor(value):
                if not torch.is_tensor(other) or not torch.equal(value, other):
                    return False
            elif value != other:
                return False
    if len(left["param_groups"]) != len(right["param_groups"]):
        return False
    for left_group, right_group in zip(left["param_groups"], right["param_groups"]):
        if set(left_group.keys()) != set(right_group.keys()):
            return False
        for name, value in left_group.items():
            if name == "params":
                if list(value) != list(right_group[name]):
                    return False
            elif value != right_group[name]:
                return False
    return True


def sync_scheduler_learning_rate(runner, checkpoint_data: dict) -> dict:
    """Set the adaptive scheduler's internal LR to the restored optimizer LR.

    The scheduler overwrites every optimizer param group with ``self.learning_rate``.  It can
    therefore only restore a single effective LR without changing behavior; multiple groups with
    different LRs are rejected explicitly instead of being averaged or silently unified.
    """
    alg = runner.alg
    live_groups = alg.optimizer.param_groups
    checkpoint_groups = checkpoint_data["optimizer_state_dict"]["param_groups"]
    if len(live_groups) != len(checkpoint_groups):
        raise RuntimeError(
            f"optimizer param-group count changed on load: live={len(live_groups)} "
            f"checkpoint={len(checkpoint_groups)}"
        )
    live_lrs = [float(group["lr"]) for group in live_groups]
    checkpoint_lrs = [float(group["lr"]) for group in checkpoint_groups]
    if any(abs(a - b) > 1e-15 for a, b in zip(live_lrs, checkpoint_lrs)):
        raise RuntimeError(
            f"restored optimizer LR differs from checkpoint: live={live_lrs} checkpoint={checkpoint_lrs}"
        )
    unique_lrs = {round(value, 15) for value in live_lrs}
    if len(unique_lrs) != 1:
        raise RuntimeError(
            "adaptive scheduler applies one internal LR to all param groups; refusing to "
            f"average different restored LRs: {live_lrs}"
        )
    restored_lr = live_lrs[0]
    optimizer_before = copy.deepcopy(alg.optimizer.state_dict())
    internal_before = float(alg.learning_rate)
    alg.learning_rate = restored_lr
    optimizer_after = alg.optimizer.state_dict()
    optimizer_unchanged = _optimizer_state_dict_equal(optimizer_before, optimizer_after)
    actor_unchanged = torch.equal(
        alg.actor.mlp[0].weight.detach().cpu(), checkpoint_data["actor_state_dict"]["mlp.0.weight"]
    )
    critic_unchanged = torch.equal(
        alg.critic.mlp[0].weight.detach().cpu(), checkpoint_data["critic_state_dict"]["mlp.0.weight"]
    )
    groups_synced = all(abs(float(group["lr"]) - restored_lr) <= 1e-15 for group in live_groups)
    precondition = (
        abs(float(alg.learning_rate) - restored_lr) <= 1e-15
        and groups_synced
        and optimizer_unchanged
        and actor_unchanged
        and critic_unchanged
    )
    report = {
        "checkpoint_optimizer_lr": checkpoint_lrs,
        "optimizer_lr_after_load": live_lrs,
        "scheduler_internal_lr_before_sync": internal_before,
        "scheduler_internal_lr_after_sync": float(alg.learning_rate),
        "param_group_count": len(live_groups),
        "param_group_lrs": live_lrs,
        "multiple_group_policy": "reject different LRs; scheduler overwrites all groups with one value",
        "optimizer_state_unchanged_by_sync": bool(optimizer_unchanged),
        "actor_state_unchanged_by_sync": bool(actor_unchanged),
        "critic_state_unchanged_by_sync": bool(critic_unchanged),
        "first_update_precondition_ok": bool(precondition),
    }
    if not precondition:
        raise RuntimeError(f"LR restore precondition failed: {report}")
    return report


def _aggregate_lr_kl(records: list[dict]) -> dict:
    if not records:
        return {
            "minibatches": 0,
            "kl_mean": None,
            "kl_max": None,
            "kl_min": None,
            "lr_before_first": None,
            "lr_after_first": None,
            "lr_min": None,
            "lr_max": None,
            "lr_final": None,
            "lr_changed_minibatches": 0,
        }
    kl_means = [record["kl_mean_used"] for record in records]
    lrs = [record["lr_before"] for record in records] + [
        record["lr_after"] for record in records if record["lr_after"] is not None
    ]
    return {
        "minibatches": len(records),
        "kl_mean": float(np.mean(kl_means)),
        "kl_max": float(max(record["kl_max"] for record in records)),
        "kl_min": float(min(record["kl_min"] for record in records)),
        "lr_before_first": records[0]["lr_before"],
        "lr_after_first": records[0]["lr_after"],
        "lr_min": float(min(lrs)),
        "lr_max": float(max(lrs)),
        "lr_final": records[-1]["lr_after"],
        "lr_changed_minibatches": sum(
            int(record["lr_after"] is not None and abs(record["lr_after"] - record["lr_before"]) > 1e-15)
            for record in records
        ),
        "desired_kl": records[0]["desired_kl"],
        "schedule": records[0]["schedule"],
        "sampling": "KL and pre-schedule LR at the actual adaptive call; post-schedule LR after optimizer.step",
    }


def install_lr_kl_logging(runner, lr_kl_file) -> None:
    """Observe PPO.update without changing data, loss, or update order."""
    alg = runner.alg
    original_update = alg.update
    next_iteration = int(runner.current_learning_iteration)

    def instrumented_update(*update_args, **update_kwargs):
        nonlocal next_iteration
        records: list[dict] = []
        pending: int | None = None
        original_kl = alg.actor.get_kl_divergence
        original_step = alg.optimizer.step
        iteration = next_iteration
        next_iteration += 1

        def kl_wrapper(old_params, new_params):
            nonlocal pending
            kl = original_kl(old_params, new_params)
            detached = kl.detach()
            record = {
                "iteration": iteration,
                "minibatch": len(records),
                "kl_mean_used": float(detached.mean().item()),
                "kl_max": float(detached.max().item()),
                "kl_min": float(detached.min().item()),
                "lr_before": float(alg.learning_rate),
                "param_lr_before": [float(group["lr"]) for group in alg.optimizer.param_groups],
                "desired_kl": float(alg.desired_kl) if alg.desired_kl is not None else None,
                "schedule": alg.schedule,
                "kl_sampled_at": "inside PPO.update after get_kl_divergence, before adaptive branch",
                "lr_after": None,
                "param_lr_after": None,
                "lr_sampled_at": "after optimizer.step",
            }
            records.append(record)
            pending = len(records) - 1
            return kl

        def step_wrapper(*step_args, **step_kwargs):
            nonlocal pending
            result = original_step(*step_args, **step_kwargs)
            if pending is not None:
                record = records[pending]
                record["lr_after"] = float(alg.learning_rate)
                record["param_lr_after"] = [float(group["lr"]) for group in alg.optimizer.param_groups]
                pending = None
            return result

        alg.actor.get_kl_divergence = kl_wrapper
        alg.optimizer.step = step_wrapper
        try:
            loss_dict = original_update(*update_args, **update_kwargs)
        finally:
            alg.actor.get_kl_divergence = original_kl
            alg.optimizer.step = original_step
        aggregate = _aggregate_lr_kl(records)
        alg._lr_kl_last = aggregate
        alg._lr_kl_last_records = records
        for record in records:
            lr_kl_file.write(json.dumps(record) + "\n")
        lr_kl_file.flush()
        return loss_dict

    alg.update = instrumented_update


def compare_behavior_config(new_config: dict) -> dict:
    baseline_config_path = os.path.join(BASELINE_RUN, "run_config.json")
    if not os.path.isfile(baseline_config_path):
        return {
            "baseline_run": BASELINE_RUN,
            "baseline_config_found": False,
            "behavior_diffs": {},
        }
    with open(baseline_config_path) as handle:
        baseline = json.load(handle)
    behavior_keys = (
        "num_envs",
        "seed",
        "num_steps_per_env_actual",
        "ppo",
        "actor",
        "critic",
        "terminations",
        "events",
        "reward_terms",
    )
    budget_keys = ("max_iterations", "max_updates", "wall_minutes")
    behavior_diffs = {
        key: {"baseline": baseline.get(key), "new": new_config.get(key)}
        for key in behavior_keys
        if baseline.get(key) != new_config.get(key)
    }
    budget_diffs = {
        key: {"baseline": baseline.get(key), "new": new_config.get(key)}
        for key in budget_keys
        if baseline.get(key) != new_config.get(key)
    }
    return {
        "baseline_run": BASELINE_RUN,
        "baseline_config_found": True,
        "behavior_diffs": behavior_diffs,
        "budget_diffs": budget_diffs,
        "intentional_changes": [
            "scheduler internal LR synchronized to restored optimizer LR before first update",
            "LR/KL logging added (lr_kl.jsonl, lr_schedule in train_metrics.jsonl)",
            "new independent output directory",
        ],
        "expected_behavior_diffs": [],
    }


def main() -> None:
    with open(os.path.join(os.path.dirname(CHECKPOINT), "params", "agent.yaml")) as handle:
        train_cfg = yaml.safe_load(handle)
    train_cfg["device"] = "cuda:0"
    train_cfg["seed"] = args.seed
    train_cfg["run_name"] = args.run_name
    train_cfg["save_interval"] = 10
    train_cfg["max_iterations"] = args.max_iterations
    # Keep the exact agent config next to the candidate checkpoint so the evaluation script can
    # load the same architecture/hyperparameters without guessing.
    with open(os.path.join(LOG_DIR, "params", "agent.yaml"), "w") as handle:
        yaml.safe_dump(train_cfg, handle, sort_keys=False)

    env_cfg = RangerCr10TeleopSimpleCollisionEnvCfg()
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.log_dir = LOG_DIR
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = gym.make("RANGER-CR10-TELEOP-SIMPLECOLL", cfg=env_cfg)
    u = env.unwrapped
    wrapper = RslRlVecEnvWrapper(env, clip_actions=train_cfg.get("clip_actions"))
    runner = OnPolicyRunner(wrapper, train_cfg, log_dir=LOG_DIR, device=u.device)
    runner.load(CHECKPOINT)
    runner.add_git_repo_to_log(__file__)

    checkpoint_data = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    lr_sync_report = sync_scheduler_learning_rate(runner, checkpoint_data)
    loaded_actor = torch.equal(
        runner.alg.actor.mlp[0].weight.detach().cpu(), checkpoint_data["actor_state_dict"]["mlp.0.weight"]
    )
    loaded_critic = torch.equal(
        runner.alg.critic.mlp[0].weight.detach().cpu(), checkpoint_data["critic_state_dict"]["mlp.0.weight"]
    )
    optimizer_states = len(runner.alg.optimizer.state_dict()["state"])
    load_report = {
        "checkpoint": CHECKPOINT,
        "checkpoint_iter": int(checkpoint_data["iter"]),
        "runner_iter_after_load": int(runner.current_learning_iteration),
        "actor_state_loaded": bool(loaded_actor),
        "critic_state_loaded": bool(loaded_critic),
        "optimizer_state_entries_loaded": optimizer_states,
        "load_cfg": "default (actor/critic/optimizer/iteration all loaded)",
        "lr_restore_sync": lr_sync_report,
    }
    print(json.dumps({
        "load_report": load_report,
        "num_steps_per_env_actual": train_cfg["num_steps_per_env"],
        "ppo": train_cfg["algorithm"],
        "max_iterations": args.max_iterations,
        "seed": args.seed,
    }, indent=1), flush=True)
    monitor = Monitor(u, runner, args.wall_minutes * 60.0, int(checkpoint_data["iter"]))
    original_step = u.step
    def monitored_step(action, _orig=original_step):
        output = _orig(action)
        monitor.after_step(*output)
        return output
    u.step = monitored_step

    original_log = runner.logger.log
    metrics_file = open(METRICS_PATH, "w")
    budget_exceeded = False
    def monitored_log(*log_args, **log_kwargs):
        nonlocal budget_exceeded
        original_log(*log_args, **log_kwargs)
        item = monitor.consume(log_kwargs)
        metrics_file.write(json.dumps(item) + "\n")
        metrics_file.flush()
        if item["b_query_failures"]:
            monitor.consecutive_failures += 1
        else:
            monitor.consecutive_failures = 0
        if monitor.consecutive_failures >= 3:
            monitor.abort_reason = "persistent_b_query_failures"
            raise StopTraining(monitor.abort_reason)
        if time.time() - monitor.started > args.wall_minutes * 60.0:
            budget_exceeded = True
            raise StopTraining("wall_clock_budget")
    runner.logger.log = monitored_log
    lr_kl_file = open(LR_KL_PATH, "w")
    install_lr_kl_logging(runner, lr_kl_file)

    config = {
        "command": " ".join(sys.argv),
        "checkpoint": CHECKPOINT,
        "log_dir": LOG_DIR,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "max_iterations": args.max_iterations,
        "max_updates": args.max_iterations,
        "wall_minutes": args.wall_minutes,
        "num_steps_per_env_actual": train_cfg["num_steps_per_env"],
        "ppo": train_cfg["algorithm"],
        "actor": train_cfg["actor"],
        "critic": train_cfg["critic"],
        "load_report": load_report,
        "lr_restore_sync": lr_sync_report,
        "lr_kl_path": LR_KL_PATH,
        "lr_kl_sampling": "actual PPO.update KL; LR before adaptive branch and after optimizer.step",
        "terminations": list(u.termination_manager.active_terms),
        "events": list(u.event_manager.active_terms),
        "reward_terms": list(u.reward_manager.active_terms),
    }
    config_diff = compare_behavior_config(config)
    config["config_diff"] = config_diff
    with open(CONFIG_PATH, "w") as handle:
        json.dump(config, handle, indent=1)
    with open(CONFIG_DIFF_PATH, "w") as handle:
        json.dump(config_diff, handle, indent=1)
    print(json.dumps({"config_diff": config_diff}, indent=1), flush=True)
    if config_diff.get("behavior_diffs"):
        lr_kl_file.close()
        raise RuntimeError(f"unexpected behavior config differences: {config_diff['behavior_diffs']}")

    stop_reason = "completed_updates"
    try:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    except StopTraining as error:
        stop_reason = str(error)
        if monitor.abort_reason is not None:
            stop_reason = monitor.abort_reason
        save_path = os.path.join(LOG_DIR, f"model_{runner.current_learning_iteration}_stopped.pt")
        runner.save(save_path)
    except Exception as error:  # noqa: BLE001
        stop_reason = f"exception:{type(error).__name__}:{error}"
        save_path = os.path.join(LOG_DIR, f"model_{runner.current_learning_iteration}_stopped.pt")
        runner.save(save_path)
        raise
    finally:
        metrics_file.close()
        lr_kl_file.close()

    candidates = [
        os.path.join(LOG_DIR, name) for name in os.listdir(LOG_DIR)
        if name.startswith("model_") and name.endswith(".pt")
    ]
    final_checkpoint = max(candidates, key=os.path.getmtime) if candidates else None
    summary = {
        "stop_reason": stop_reason,
        "budget_exceeded": budget_exceeded,
        "final_checkpoint": final_checkpoint,
        "final_iteration": int(runner.current_learning_iteration),
        "load_report": load_report,
        "lr_restore_sync": lr_sync_report,
        "config_diff": config_diff,
        "wall_time_s": time.time() - monitor.started,
        "training_time_s": time.time() - monitor.started,
        "num_updates_completed": monitor.updates,
        "metrics_file": METRICS_PATH,
        "lr_kl_file": LR_KL_PATH,
        "config_file": CONFIG_PATH,
        "config_diff_file": CONFIG_DIFF_PATH,
    }
    with open(SUMMARY_PATH, "w") as handle:
        json.dump(summary, handle, indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
