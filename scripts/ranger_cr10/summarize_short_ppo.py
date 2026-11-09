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

"""Summarize the budgeted short PPO run and the same-protocol before/after evaluation.

Run after the training and both rollout evaluations:
    python scripts/ranger_cr10/summarize_short_ppo.py --run-dir <run_dir> \
        --original-eval <original.json> --candidate-eval <candidate.json>
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--run-dir", required=True)
ap.add_argument("--original-eval", required=True)
ap.add_argument("--candidate-eval", required=True)
ap.add_argument("--out", default=str(ROOT / "outputs/stage413/b/simple_collision_v2/physproxy_short_ppo_summary.json"))
args = ap.parse_args()

run_dir = Path(args.run_dir)
lines = [json.loads(line) for line in (run_dir / "train_metrics.jsonl").read_text().splitlines() if line.strip()]
run_summary = json.loads((run_dir / "run_summary.json").read_text())
run_config = json.loads((run_dir / "run_config.json").read_text())
original = json.loads(Path(args.original_eval).read_text())
candidate = json.loads(Path(args.candidate_eval).read_text())

first = lines[: min(10, len(lines))]
last = lines[-min(10, len(lines)) :]
terms = sorted({name for line in lines for name in line["termination_counts"]})
termination_events = {name: sum(line["termination_counts"].get(name, 0) for line in lines) for name in terms}
completed = sum(line["completed_episodes"] for line in lines)
env_steps = sum(line["env_steps"] for line in lines)
training = {
    "run_dir": str(run_dir),
    "updates": len(lines),
    "first_iteration": lines[0]["iteration"] if lines else None,
    "last_iteration": lines[-1]["iteration"] if lines else None,
    "checkpoint_iter": run_summary["load_report"]["checkpoint_iter"],
    "final_iteration": run_summary["final_iteration"],
    "final_checkpoint": run_summary["final_checkpoint"],
    "stop_reason": run_summary["stop_reason"],
    "wall_time_s": run_summary["wall_time_s"],
    "num_updates_completed": run_summary["num_updates_completed"],
    "horizon_num_steps_per_env": run_config["num_steps_per_env_actual"],
    "seed": run_config["seed"],
    "learning_rate_configured": run_config["ppo"]["learning_rate"],
    "learning_rate_actual_first_last": [float(lines[0]["learning_rate"]), float(lines[-1]["learning_rate"])],
    "learning_rate_actual_min_max": [float(min(line["learning_rate"] for line in lines)),
                                     float(max(line["learning_rate"] for line in lines))],
    "ppo": run_config["ppo"],
    "collect_time_s": {"mean": float(np.mean([line["collect_time_s"] for line in lines])),
                       "p95": float(np.percentile([line["collect_time_s"] for line in lines], 95))},
    "learn_time_s": {"mean": float(np.mean([line["learn_time_s"] for line in lines])),
                     "p95": float(np.percentile([line["learn_time_s"] for line in lines], 95))},
    "env_steps": env_steps,
    "completed_episodes": completed,
    "termination_events": termination_events,
    "termination_rate_per_completed_episode": {
        name: (count / completed if completed else None) for name, count in termination_events.items()
    },
    "termination_rate_per_env_step": {
        name: (count / env_steps if env_steps else None) for name, count in termination_events.items()
    },
    "episode_return_first10_mean": float(np.mean([line["episode_return_mean"] for line in first])),
    "episode_return_last10_mean": float(np.mean([line["episode_return_mean"] for line in last])),
    "episode_return_per_step_first_last": [
        float(np.mean([line["episode_return_mean"] / max(line["episode_length_mean"], 1e-9) for line in first])),
        float(np.mean([line["episode_return_mean"] / max(line["episode_length_mean"], 1e-9) for line in last])),
    ],
    "episode_length_first10_mean": float(np.mean([line["episode_length_mean"] for line in first])),
    "episode_length_last10_mean": float(np.mean([line["episode_length_mean"] for line in last])),
    "loss_first_last": {
        key: [float(np.mean([line["loss"][key] for line in first])),
              float(np.mean([line["loss"][key] for line in last]))]
        for key in ("value", "surrogate", "entropy") if key in lines[0]["loss"]
    },
    "mean_std_first_last": [float(np.mean([line["mean_std"] for line in first])),
                            float(np.mean([line["mean_std"] for line in last]))],
    "b_query_failures": sum(line["b_query_failures"] for line in lines),
    "b_safety_sum": sum(line["b_safety_sum"] for line in lines),
    "link4_link6_below20_env_steps": sum(line["link4_link6_below20_env_steps"] for line in lines),
    "link4_link6_contact_env_steps": sum(line["link4_link6_contact_env_steps"] for line in lines),
    "link4_link6_min_pair_env_steps": sum(line["link4_link6_min_pair_env_steps"] for line in lines),
    "actual_nonwheel_contact_env_steps": sum(line["actual_nonwheel_contact_env_steps"] for line in lines),
    "actual_nonwheel_contact_body_events": sum(line["actual_nonwheel_contact_body_events"] for line in lines),
    "ee_position_error_mean": float(np.mean([line["ee_position_error_mean"] for line in lines])),
    "ee_orientation_error_mean": float(np.mean([line["ee_orientation_error_mean"] for line in lines])),
    "note": "RL loss dict exposes value/surrogate/entropy; the implementation does not return KL. "
            "Termination event counts are per-episode term signals; the separate env-step rate is "
            "labeled and is not an episode failure rate.",
}


def eval_row(data):
    return {
        "checkpoint": data["checkpoint"],
        "policy_loaded": data["policy_loaded"],
        "obs_dim": data["obs_dim"],
        "actor_input_dim": data["actor_input_dim"],
        "action_dim": data["action_dim"],
        "step_time_mean_s": data["step_time_s"]["mean"],
        "step_time_p95_s": data["step_time_s"]["p95"],
        "safety_sum": data["safety_reward"]["sum"],
        "safety_share_negative": data["safety_reward"]["share_of_negative_abs"],
        "ee_position_mean_m": data["pose_error"]["mean"],
        "ee_position_p95_m": data["pose_error"]["p95"],
        "ee_orientation_mean_rad": data["orientation_error"]["mean"],
        "ee_orientation_p95_rad": data["orientation_error"]["p95"],
        "vel_error_xy": data["velocity_error"]["xy_mean"],
        "vel_error_yaw": data["velocity_error"]["yaw_mean"],
        "nan_inf_steps": data["nan_or_inf_steps"],
        "b_query_failures": data["query_failures"],
        "dones_total": data["dones_total"],
        "termination_counts": data["termination_counts"],
        "actual_physx_contacts": data.get("actual_physx_contacts"),
        "link4_link6": data["link4_link6"],
        "top_pairs": data["top_pairs"],
    }


original_eval = eval_row(original)
candidate_eval = eval_row(candidate)
comparison = {
    "protocol": "same 64 env, 200 steps, seed 42, command sequence, reset and inference settings",
    "original": original_eval,
    "candidate": candidate_eval,
    "delta_candidate_minus_original": {
        "ee_position_mean_m": candidate_eval["ee_position_mean_m"] - original_eval["ee_position_mean_m"],
        "ee_position_p95_m": candidate_eval["ee_position_p95_m"] - original_eval["ee_position_p95_m"],
        "ee_orientation_mean_rad": candidate_eval["ee_orientation_mean_rad"] - original_eval["ee_orientation_mean_rad"],
        "safety_sum": candidate_eval["safety_sum"] - original_eval["safety_sum"],
        "link4_link6_below20_env_steps": candidate_eval["link4_link6"]["steps_with_margin_20mm_env_steps"]
        - original_eval["link4_link6"]["steps_with_margin_20mm_env_steps"],
        "link4_link6_contact_env_steps": candidate_eval["link4_link6"]["contact_env_steps"]
        - original_eval["link4_link6"]["contact_env_steps"],
        "dones_total": candidate_eval["dones_total"] - original_eval["dones_total"],
        "actual_physx_contact_env_steps": (
            (candidate_eval.get("actual_physx_contacts") or {}).get("nonwheel_body_net_force_gt_100n_env_steps", 0)
            - (original_eval.get("actual_physx_contacts") or {}).get("nonwheel_body_net_force_gt_100n_env_steps", 0)
        ),
        "actual_physx_contact_body_events": (
            (candidate_eval.get("actual_physx_contacts") or {}).get("nonwheel_body_events", 0)
            - (original_eval.get("actual_physx_contacts") or {}).get("nonwheel_body_events", 0)
        ),
    },
}
result = {"training": training, "comparison": comparison}
Path(args.out).write_text(json.dumps(result, indent=1))
print(json.dumps(result, indent=1)[:12000])
print(f"wrote {args.out}")
