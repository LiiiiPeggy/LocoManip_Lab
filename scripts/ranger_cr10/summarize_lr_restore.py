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

"""Summarize the LR-restore experiment: training LR/KL and the three-policy evaluation.

Usage:
    python scripts/ranger_cr10/summarize_lr_restore.py \
        --new-run-dir <run_dir> \
        --original-eval outputs/stage413/b/dist_explain/train_dist_original.json \
        --old-eval outputs/stage413/b/dist_explain/train_dist_candidate.json \
        --new-eval outputs/stage413/b/lr_restore/train_dist_new.json \
        --out outputs/stage413/b/lr_restore/lr_restore_comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--new-run-dir", required=True)
ap.add_argument("--old-run-dir", default=str(
    ROOT / "logs/rsl_rl/ranger_cr10_teleop/2026-11-08_00-11-40_physproxy_short_ppo_50"
))
ap.add_argument("--original-eval", default=str(
    ROOT / "outputs/stage413/b/dist_explain/train_dist_original.json"
))
ap.add_argument("--old-eval", default=str(
    ROOT / "outputs/stage413/b/dist_explain/train_dist_candidate.json"
))
ap.add_argument("--new-eval", required=True)
ap.add_argument("--out", default=str(ROOT / "outputs/stage413/b/lr_restore/lr_restore_comparison.json"))
args = ap.parse_args()


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def mean(values):
    return float(np.mean(values)) if values else None


def get_value(row: dict, key: str):
    value = row
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def first_last(rows: list[dict], key: str) -> list[float | None]:
    first = rows[: min(10, len(rows))]
    last = rows[-min(10, len(rows)) :]
    return [
        mean([value for row in first if (value := get_value(row, key)) is not None]),
        mean([value for row in last if (value := get_value(row, key)) is not None]),
    ]


def training_summary(run_dir: str, has_kl: bool) -> dict:
    run_dir = Path(run_dir)
    summary = load_json(run_dir / "run_summary.json")
    config = load_json(run_dir / "run_config.json")
    rows = load_jsonl(run_dir / "train_metrics.jsonl")
    out = {
        "run_dir": str(run_dir),
        "stop_reason": summary.get("stop_reason"),
        "final_checkpoint": summary.get("final_checkpoint"),
        "final_iteration": summary.get("final_iteration"),
        "wall_time_s": summary.get("wall_time_s"),
        "num_updates_completed": summary.get("num_updates_completed"),
        "load_report": summary.get("load_report"),
        "config_diff": summary.get("config_diff"),
        "first_iteration": rows[0]["iteration"] if rows else None,
        "last_iteration": rows[-1]["iteration"] if rows else None,
        "loss_value_first_last": first_last(rows, "loss.value"),
        "surrogate_first_last": first_last(rows, "loss.surrogate"),
        "entropy_first_last": first_last(rows, "loss.entropy"),
        "episode_return_per_step_first_last": [
            mean([r["episode_return_mean"] / max(r["episode_length_mean"], 1e-9)
                  for r in rows[: min(10, len(rows))] if r.get("episode_return_mean") is not None]),
            mean([r["episode_return_mean"] / max(r["episode_length_mean"], 1e-9)
                  for r in rows[-min(10, len(rows)):] if r.get("episode_return_mean") is not None]),
        ],
        "episode_length_first_last": first_last(rows, "episode_length_mean"),
        "safety_per_env_step_first_last": [
            mean([r["b_safety_sum"] / max(r["env_steps"], 1) for r in rows[: min(10, len(rows))]]),
            mean([r["b_safety_sum"] / max(r["env_steps"], 1) for r in rows[-min(10, len(rows)):]]),
        ],
        "link46_rate_first_last": [
            mean([r["link4_link6_below20_env_steps"] / max(r["env_steps"], 1)
                  for r in rows[: min(10, len(rows))]]),
            mean([r["link4_link6_below20_env_steps"] / max(r["env_steps"], 1)
                  for r in rows[-min(10, len(rows)):]]),
        ],
        "actual_contact_env_steps": sum(r["actual_nonwheel_contact_env_steps"] for r in rows),
        "termination_events": {
            key: sum(r["termination_counts"].get(key, 0) for r in rows)
            for key in rows[0]["termination_counts"]
        } if rows else {},
        "completed_episodes": sum(r["completed_episodes"] for r in rows),
        "env_steps": sum(r["env_steps"] for r in rows),
        "b_query_failures": sum(r["b_query_failures"] for r in rows),
        "lr_first_last": first_last(rows, "learning_rate"),
        "lr_min_max": [
            min(r["learning_rate"] for r in rows),
            max(r["learning_rate"] for r in rows),
        ] if rows else None,
        "kl_logging": "available" if has_kl else "missing in this historical run",
        "lr_kl_file": config.get("lr_kl_path"),
    }
    if has_kl:
        lr_kl_path = Path(config["lr_kl_path"])
        lr_rows = load_jsonl(lr_kl_path)
        first_iteration = rows[0]["iteration"] if rows else None
        first = [record for record in lr_rows if record["iteration"] == first_iteration]
        out["first_update_minibatches"] = first
        out["first_update_lr_path"] = (
            [first[0]["lr_before"]] + [record["lr_after"] for record in first] if first else []
        )
        out["first_update_kl_path"] = [record["kl_mean_used"] for record in first]
        out["kl_first_last"] = first_last(rows, "lr_schedule.kl_mean")
        out["kl_max_update"] = max(
            (r["lr_schedule"]["kl_max"] for r in rows if r.get("lr_schedule")), default=None
        )
        out["lr_min_update"] = min(
            (r["lr_schedule"]["lr_min"] for r in rows if r.get("lr_schedule")), default=None
        )
        out["lr_max_update"] = max(
            (r["lr_schedule"]["lr_max"] for r in rows if r.get("lr_schedule")), default=None
        )
        out["lr_final_update"] = rows[-1].get("lr_schedule", {}).get("lr_final") if rows else None
    return out


def eval_summary(data: dict) -> dict:
    items = data["records"]
    link = data["link4_link6_overall"]
    return {
        "checkpoint": data["checkpoint"],
        "below20": link["below20"],
        "env_steps": link["env_steps"],
        "below20_rate": link["below20"] / link["env_steps"],
        "contact": link["contact"],
        "distance_mean_m": link["mean_m"],
        "distance_p10_p50_p90_m": [link["quantiles_m"][1], link["quantiles_m"][3], link["quantiles_m"][5]],
        "ee_position_mean_m": mean([item["pos_err"] for item in items]),
        "ee_position_p95_m": float(np.percentile([item["pos_err"] for item in items], 95)),
        "ee_orientation_mean_rad": mean([item["ori_err"] for item in items]),
        "ee_orientation_p95_rad": float(np.percentile([item["ori_err"] for item in items], 95)),
        "safety_sum": data["safety_reward"]["sum"],
        "safety_per_env_step": data["safety_reward"]["per_env_step"],
        "actual_physx_contacts": data["actual_physx_contacts"],
        "dones_total": data["episode_counts"]["dones_total"],
        "termination_counts": data["episode_counts"]["termination_counts"],
        "nan_or_inf_steps": data["nan_or_inf_steps"],
        "query_failures": data["query_failures"],
        "step_time_mean_s": data["step_time_s"]["mean"],
        "top_pairs": data["top_pairs"],
    }


original = load_json(args.original_eval)
old = load_json(args.old_eval)
new = load_json(args.new_eval)
old_training = training_summary(args.old_run_dir, has_kl=False)
new_training = training_summary(args.new_run_dir, has_kl=True)

initial_digests = {
    "original": original["initial_digest"],
    "old_candidate": old["initial_digest"],
    "new_candidate": new["initial_digest"],
}
command_digest_matches = {
    "original_vs_old": sum(int(a == b) for a, b in zip(original["step_command_digests"], old["step_command_digests"])),
    "original_vs_new": sum(int(a == b) for a, b in zip(original["step_command_digests"], new["step_command_digests"])),
    "steps": len(original["step_command_digests"]),
}
results = {
    "protocol": "matched training command distribution, 64 env, 200 steps, seed 42, deterministic mean, corruption off",
    "pairing": {
        "initial_digests": initial_digests,
        "initial_digest_all_equal": len(set(initial_digests.values())) == 1,
        "command_digest_matches": command_digest_matches,
        "strict_state_pairing_claimed": False,
        "note": "Same initial digest and command digest are required for a valid comparison; "
                "state trajectories are not claimed to be frame-paired.",
    },
    "training": {
        "old": old_training,
        "new": new_training,
    },
    "evaluation": {
        "original": eval_summary(original),
        "old_candidate": eval_summary(old),
        "new_candidate": eval_summary(new),
    },
    "delta_new_minus_original": {},
    "delta_new_minus_old": {},
}
for key in (
    "below20_rate",
    "distance_mean_m",
    "ee_position_mean_m",
    "ee_position_p95_m",
    "ee_orientation_mean_rad",
    "ee_orientation_p95_rad",
    "safety_per_env_step",
    "dones_total",
):
    results["delta_new_minus_original"][key] = (
        results["evaluation"]["new_candidate"][key] - results["evaluation"]["original"][key]
    )
    results["delta_new_minus_old"][key] = (
        results["evaluation"]["new_candidate"][key] - results["evaluation"]["old_candidate"][key]
    )
for key in ("nonwheel_body_net_force_gt_100n_env_steps", "nonwheel_body_events"):
    results["delta_new_minus_original"][f"actual_{key}"] = (
        results["evaluation"]["new_candidate"]["actual_physx_contacts"][key]
        - results["evaluation"]["original"]["actual_physx_contacts"][key]
    )
    results["delta_new_minus_old"][f"actual_{key}"] = (
        results["evaluation"]["new_candidate"]["actual_physx_contacts"][key]
        - results["evaluation"]["old_candidate"]["actual_physx_contacts"][key]
    )

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(results, indent=1))
print(json.dumps(results, indent=1)[:24000])
print(f"wrote {args.out}")
