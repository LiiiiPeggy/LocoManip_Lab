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

"""Summarize the fixed-command three-policy evaluation.

Usage:
    python scripts/ranger_cr10/summarize_fixed_commands.py \
        --eval 42:original=<json> --eval 42:weight2=<json> --eval 42:weight0=<json> \
        --eval 43:original=<json> --eval 43:weight2=<json> --eval 43:weight0=<json> \
        --old 42:original=<json> --old 42:weight2=<json> --old 42:weight0=<json> \
        --out outputs/stage413/b/fixed_eval/fixed_commands_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--eval", action="append", default=[], help="seed:label=path")
ap.add_argument("--old", action="append", default=[], help="seed:label=path (historical, not paired)")
ap.add_argument("--out", default=str(ROOT / "outputs/stage413/b/fixed_eval/fixed_commands_summary.json"))
args = ap.parse_args()


def parse(values):
    out = {}
    for value in values:
        key, path = value.split("=", 1)
        seed, label = key.split(":", 1)
        out[(int(seed), label)] = str(Path(path))
    return out


def load(path):
    return json.loads(Path(path).read_text())


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def eval_summary(data, *, fixed=True):
    items = data["records"]
    link = data["link4_link6_overall"]
    out = {
        "checkpoint": data["checkpoint"],
        "checkpoint_sha256": data.get("checkpoint_sha256"),
        "seed": data.get("seed"),
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
        "initial_digest": data.get("initial_digest"),
        "command_plan_hash": data.get("command_plan_hash"),
        "step_command_digests": data.get("step_command_digests"),
    }
    if fixed:
        out["reset_counts"] = {
            "total": data["episode_counts"]["reset_counts_total"],
            "after_initial": data["episode_counts"]["reset_events_after_initial"],
        }
        out["command_consistency"] = data["command_consistency"]
    return out


def aggregate(rows):
    keys = (
        "below20_rate", "safety_per_env_step", "ee_position_mean_m", "ee_position_p95_m",
        "ee_orientation_mean_rad", "ee_orientation_p95_rad", "step_time_mean_s",
    )
    return {key: mean([row[key] for row in rows]) for key in keys}


evaluations = {key: eval_summary(load(path)) for key, path in parse(args.eval).items()}
old = {f"{seed}:{label}": eval_summary(load(path), fixed=False) for (seed, label), path in parse(args.old).items()}
seeds = sorted({seed for seed, _ in evaluations})
labels = ("original", "weight2", "weight0")
result = {
    "protocol": "pre-generated command plan, 64 env, 200 control steps, deterministic mean, corruption off, weight=2 evaluation",
    "by_seed": {},
    "aggregate_two_seeds": {},
    "old_seed42": old,
    "pairing": {},
}
for seed in seeds:
    result["by_seed"][str(seed)] = {label: evaluations.get((seed, label)) for label in labels}
    result["pairing"][str(seed)] = {
        "initial_digests": {label: evaluations[(seed, label)]["initial_digest"] for label in labels if (seed, label) in evaluations},
        "command_plan_hashes": {label: evaluations[(seed, label)]["command_plan_hash"] for label in labels if (seed, label) in evaluations},
        "initial_all_equal": len({evaluations[(seed, label)]["initial_digest"] for label in labels if (seed, label) in evaluations}) == 1,
        "plan_hash_all_equal": len({evaluations[(seed, label)]["command_plan_hash"] for label in labels if (seed, label) in evaluations}) == 1,
        "step_command_digests_all_equal": len({tuple(evaluations[(seed, label)]["step_command_digests"]) for label in labels if (seed, label) in evaluations}) == 1,
    }
for label in labels:
    rows = [result["by_seed"][str(seed)][label] for seed in seeds if result["by_seed"][str(seed)][label] is not None]
    result["aggregate_two_seeds"][label] = aggregate(rows)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(result, indent=1))
print(json.dumps(result, indent=1)[:24000])
print(f"wrote {args.out}")
