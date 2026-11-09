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

"""Summarize the long SIMPLECOLL PPO run and its stage/final evaluations.

Usage:
    python scripts/ranger_cr10/summarize_long_ppo.py \
        --run-dir <run_dir> --baseline-eval <json> \
        --stage-eval 100=<json> --stage-eval 250=<json> --stage-eval 500=<json> \
        --final-eval seed43_original=<json> --final-eval seed43_candidate=<json> \
        --final-eval seed44_original=<json> --final-eval seed44_candidate=<json> \
        --out outputs/stage413/b/long_ppo/long_ppo_summary.json
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
ap.add_argument("--baseline-eval", required=True)
ap.add_argument("--stage-eval", action="append", default=[])
ap.add_argument("--final-eval", action="append", default=[])
ap.add_argument("--out", default=str(ROOT / "outputs/stage413/b/long_ppo/long_ppo_summary.json"))
args = ap.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text())


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def first_last(rows, getter):
    first = rows[: min(10, len(rows))]
    last = rows[-min(10, len(rows)):]
    return [mean([getter(row) for row in first]), mean([getter(row) for row in last])]


def nested(row, *keys):
    value = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def training(run_dir: str) -> dict:
    run_dir = Path(run_dir)
    summary = load_json(run_dir / "run_summary.json")
    config = load_json(run_dir / "run_config.json")
    rows = load_jsonl(run_dir / "train_metrics.jsonl")
    lr_kl = load_jsonl(run_dir / "lr_kl.jsonl") if (run_dir / "lr_kl.jsonl").is_file() else []
    first_iteration = rows[0]["iteration"] if rows else None
    first = [row for row in lr_kl if row["iteration"] == first_iteration]
    return {
        "run_dir": str(run_dir),
        "stop_reason": summary.get("stop_reason"),
        "final_checkpoint": summary.get("final_checkpoint"),
        "checkpoint_source_iteration": summary.get("checkpoint_source_iteration"),
        "final_iteration": summary.get("final_iteration"),
        "new_updates_completed": summary.get("new_updates_completed"),
        "new_updates_formula": summary.get("new_updates_formula"),
        "wall_time_s": summary.get("wall_time_s"),
        "saved_checkpoints": summary.get("saved_checkpoints"),
        "load_report": summary.get("load_report"),
        "config_diff": summary.get("config_diff"),
        "env_steps": sum(row["env_steps"] for row in rows),
        "completed_episodes": sum(row["completed_episodes"] for row in rows),
        "termination_events": {
            key: sum(row["termination_counts"].get(key, 0) for row in rows)
            for key in rows[0]["termination_counts"]
        } if rows else {},
        "b_query_failures": sum(row["b_query_failures"] for row in rows),
        "actual_nonwheel_contact_env_steps": sum(row["actual_nonwheel_contact_env_steps"] for row in rows),
        "actual_nonwheel_contact_body_events": sum(row["actual_nonwheel_contact_body_events"] for row in rows),
        "value_loss_first_last": first_last(rows, lambda row: nested(row, "loss", "value")),
        "surrogate_first_last": first_last(rows, lambda row: nested(row, "loss", "surrogate")),
        "entropy_first_last": first_last(rows, lambda row: nested(row, "loss", "entropy")),
        "return_per_step_first_last": [
            first_last(rows, lambda row: row["episode_return_mean"] / max(row["episode_length_mean"], 1e-9))[0],
            first_last(rows, lambda row: row["episode_return_mean"] / max(row["episode_length_mean"], 1e-9))[1],
        ],
        "episode_length_first_last": first_last(rows, lambda row: row["episode_length_mean"]),
        "safety_per_env_step_first_last": [
            mean([row["b_safety_sum"] / max(row["env_steps"], 1) for row in rows[: min(10, len(rows))]]),
            mean([row["b_safety_sum"] / max(row["env_steps"], 1) for row in rows[-min(10, len(rows)):]]),
        ],
        "link46_rate_first_last": [
            mean([row["link4_link6_below20_env_steps"] / max(row["env_steps"], 1)
                  for row in rows[: min(10, len(rows))]]),
            mean([row["link4_link6_below20_env_steps"] / max(row["env_steps"], 1)
                  for row in rows[-min(10, len(rows)):]]),
        ],
        "lr_logged_first_last": first_last(rows, lambda row: row["learning_rate"]),
        "lr_kl_first_update": {
            "minibatches": len(first),
            "lr_before_first": first[0]["lr_before"] if first else None,
            "lr_after_path": [record["lr_after"] for record in first],
            "kl_mean_path": [record["kl_mean_used"] for record in first],
            "kl_mean": mean([record["kl_mean_used"] for record in first]),
            "kl_max": max((record["kl_max"] for record in first), default=None),
            "lr_min": min((record["lr_before"] for record in first), default=None),
            "lr_max": max((record["lr_after"] for record in first), default=None),
            "lr_final": first[-1]["lr_after"] if first else None,
        },
        "kl_per_update_mean_range": [
            min((nested(row, "lr_schedule", "kl_mean") for row in rows
                 if nested(row, "lr_schedule", "kl_mean") is not None), default=None),
            max((nested(row, "lr_schedule", "kl_mean") for row in rows
                 if nested(row, "lr_schedule", "kl_mean") is not None), default=None),
        ],
        "lr_per_update_min_range": [
            min((nested(row, "lr_schedule", "lr_min") for row in rows
                 if nested(row, "lr_schedule", "lr_min") is not None), default=None),
            max((nested(row, "lr_schedule", "lr_max") for row in rows
                 if nested(row, "lr_schedule", "lr_max") is not None), default=None),
        ],
    }


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
        "initial_digest": data.get("initial_digest"),
        "step_command_digests": data.get("step_command_digests"),
        "top_pairs": data["top_pairs"],
    }


def parse_pairs(values):
    out = {}
    for value in values:
        label, path = value.split("=", 1)
        out[label] = str(Path(path))
    return out


baseline = load_json(args.baseline_eval)
stages = {label: eval_summary(load_json(path)) for label, path in parse_pairs(args.stage_eval).items()}
final = {label: eval_summary(load_json(path)) for label, path in parse_pairs(args.final_eval).items()}
result = {
    "protocol": "matched training command distribution, 64 env, 200 steps, seed given per eval, "
                "deterministic mean, corruption off",
    "training": training(args.run_dir),
    "baseline_original": eval_summary(baseline),
    "stages": stages,
    "final": final,
    "pairing": {},
}
for label, stage in stages.items():
    result["pairing"][label] = {
        "initial_digest_matches_baseline": stage["initial_digest"] == result["baseline_original"]["initial_digest"],
        "command_digest_matches": sum(
            int(a == b) for a, b in zip(
                result["baseline_original"]["step_command_digests"], stage["step_command_digests"]
            )
        ),
        "steps": len(result["baseline_original"]["step_command_digests"]),
    }
for label, item in final.items():
    result["pairing"][label] = {
        "initial_digest_matches_baseline": item["initial_digest"] == result["baseline_original"]["initial_digest"],
    }
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(result, indent=1))
print(json.dumps(result, indent=1)[:24000])
print(f"wrote {args.out}")
