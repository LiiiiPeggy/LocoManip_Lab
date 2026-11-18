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

"""Attribute the fine-tuning tracking loss from the fixed-command evaluations.

Reads the six instrumented rollouts (3 policies x 2 seeds) and reports:

* every active reward term at a common scale (weight x dt per env-step, plus the raw term value);
* where tracking error sits: reset transient, command switch, or hold;
* error against command translation and rotation magnitude;
* the actual arm joint deviation from the default pose, next to the ``arm_deviation`` term.

The phase rule is fixed in the rollout script and reused here; it is never chosen per policy.

Usage:
    python scripts/ranger_cr10/analyze_fixed_eval.py \
        --eval 42:original=... --eval 42:weight2=... --eval 42:weight0=... \
        --eval 43:original=... --eval 43:weight2=... --eval 43:weight0=... \
        --out outputs/stage413/b/attribution/attribution_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--eval", action="append", default=[], help="seed:label=path")
ap.add_argument("--out", default=str(ROOT / "outputs/stage413/b/attribution/attribution_summary.json"))
args = ap.parse_args()

LABELS = ("original", "weight2", "weight0")


def parse(values):
    out = {}
    for value in values:
        key, path = value.split("=", 1)
        seed, label = key.split(":", 1)
        out[(int(seed), label)] = Path(path)
    return out


def load(path):
    return json.loads(Path(path).read_text())


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def phase_error_mass(records, phase, key):
    total = sum(float(item[key]) for item in records if item["phase"] == phase)
    return total


def binned(records, key, edges, scale=1.0):
    values = np.array([float(item[key]) * scale for item in records])
    out = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (values >= low) & (values < high)
        selected = [item for item, keep in zip(records, mask) if keep]
        out.append({
            "range": [low, high],
            "n": len(selected),
            "pos_mean": mean([float(item["pos_err"]) for item in selected]),
            "ori_mean": mean([float(item["ori_err"]) for item in selected]),
            "share_of_env_steps": len(selected) / max(len(records), 1),
        })
    return out


def summarize(seed, label, data):
    records = data["records"]
    terms = data["reward_terms"]
    return {
        "seed": seed,
        "label": label,
        "checkpoint": data["checkpoint"],
        "checkpoint_sha256": data["checkpoint_sha256"],
        "env_steps": len(records),
        "ee_position_mean_m": mean([item["pos_err"] for item in records]),
        "ee_orientation_mean_rad": mean([item["ori_err"] for item in records]),
        "safety_per_env_step": data["safety_reward"]["per_env_step"],
        "reward_per_env_step": data["reward_consistency"]["total_reward_sum"]
        / max(data["reward_consistency"]["term_env_steps"], 1),
        "terms": {name: value["weighted_per_env_step"] for name, value in terms.items()},
        "raw_terms": {name: value["raw_per_env_step"] for name, value in terms.items()},
        "weights": {name: value["weight"] for name, value in terms.items()},
        "phase_split": data["phase_split"],
        "phase_rule": data["phase_rule"],
        "arm_deviation_stats": data["arm_deviation_stats"],
        "reward_consistency": data["reward_consistency"],
        "command_plan_hash": data["command_plan_hash"],
        "initial_digest": data["initial_digest"],
        "bins_offset": binned(records, "offset_norm" if "offset_norm" in records[0] else "cmd_offset_norm", [0.0, 0.1, 0.2, 0.3, 0.6]),
        "bins_angle_deg": binned(records, "cmd_angle", [0.0, 15.0, 30.0, 45.0, 60.0, 180.0], scale=180.0 / math.pi),
    }


evaluations = {(seed, label): load(path) for (seed, label), path in parse(args.eval).items()}
seeds = sorted({seed for seed, _ in evaluations})

# The bins need |offset|; compute it from the recorded command offset.
summary = {}
for (seed, label), data in evaluations.items():
    for item in data["records"]:
        offset = item["cmd_offset"]
        item["cmd_offset_norm"] = float(math.sqrt(sum(value * value for value in offset)))
    summary[(seed, label)] = summarize(seed, label, data)

term_names = list(next(iter(summary.values()))["weights"].keys())
pairing = {}
for seed in seeds:
    available = [label for label in LABELS if (seed, label) in summary]
    pairing[str(seed)] = {
        "labels": available,
        "initial_digest_all_equal": len({summary[(seed, label)]["initial_digest"] for label in available}) == 1,
        "command_plan_hash_all_equal": len(
            {summary[(seed, label)]["command_plan_hash"] for label in available}
        ) == 1,
        "reward_consistency_max_abs_difference": max(
            summary[(seed, label)]["reward_consistency"]["abs_difference"] for label in available
        ),
        "reward_consistency_max_relative_difference": max(
            summary[(seed, label)]["reward_consistency"]["relative_difference"] or 0.0
            for label in available
        ),
    }

term_table = {}
for name in term_names:
    row = {"weight": summary[(seeds[0], LABELS[0])]["weights"][name]}
    for seed in seeds:
        for label in LABELS:
            if (seed, label) not in summary:
                continue
            row[f"seed{seed}_{label}"] = summary[(seed, label)]["terms"][name]
            row[f"seed{seed}_{label}_raw"] = summary[(seed, label)]["raw_terms"][name]
        if (seed, "original") in summary:
            for label in ("weight2", "weight0"):
                if (seed, label) in summary:
                    row[f"seed{seed}_{label}_minus_original"] = (
                        summary[(seed, label)]["terms"][name] - summary[(seed, "original")]["terms"][name]
                    )
    for label in ("weight2", "weight0"):
        deltas = [row[f"seed{seed}_{label}_minus_original"] for seed in seeds
                  if f"seed{seed}_{label}_minus_original" in row]
        row[f"{label}_minus_original_mean_over_seeds"] = mean(deltas)
    term_table[name] = row

phase_table = {}
for seed in seeds:
    phase_table[str(seed)] = {}
    for label in LABELS:
        if (seed, label) not in summary:
            continue
        split = summary[(seed, label)]["phase_split"]
        records = evaluations[(seed, label)]["records"]
        total_pos_mass = sum(phase_error_mass(records, phase, "pos_err") for phase in split)
        total_ori_mass = sum(phase_error_mass(records, phase, "ori_err") for phase in split)
        phase_table[str(seed)][label] = {
            phase: {
                **split[phase],
                "pos_error_mass_share": (
                    phase_error_mass(records, phase, "pos_err") / total_pos_mass if total_pos_mass else None
                ),
                "ori_error_mass_share": (
                    phase_error_mass(records, phase, "ori_err") / total_ori_mass if total_ori_mass else None
                ),
                "pos_mean_over_original_mean": None,
            }
            for phase in split
        }
        for phase in split:
            original_pos = phase_table[str(seed)].get("original", {}).get(phase, {}).get("pos_err_mean_m")
            value = phase_table[str(seed)][label][phase]["pos_err_mean_m"]
            phase_table[str(seed)][label][phase]["pos_mean_over_original_mean"] = (
                value / original_pos if original_pos and value is not None else None
            )

aggregate = {}
for label in LABELS:
    rows = [summary[(seed, label)] for seed in seeds if (seed, label) in summary]
    if not rows:
        continue
    aggregate[label] = {
        "seeds": len(rows),
        "ee_position_mean_m": mean([row["ee_position_mean_m"] for row in rows]),
        "ee_orientation_mean_rad": mean([row["ee_orientation_mean_rad"] for row in rows]),
        "safety_per_env_step": mean([row["safety_per_env_step"] for row in rows]),
        "reward_per_env_step": mean([row["reward_per_env_step"] for row in rows]),
        "terms": {name: mean([row["terms"][name] for row in rows]) for name in term_names},
        "raw_terms": {name: mean([row["raw_terms"][name] for row in rows]) for name in term_names},
        "arm_deviation_stats": {
            key: mean([row["arm_deviation_stats"][key] for row in rows])
            for key in rows[0]["arm_deviation_stats"]
            if key != "note"
        },
        "phase_split": {
            phase: {
                key: mean([row["phase_split"][phase][key] for row in rows])
                for key in ("share", "pos_err_mean_m", "ori_err_mean_rad", "safety_per_env_step",
                            "reward_per_env_step", "joint_dev_abs_sum_mean")
            }
            for phase in rows[0]["phase_split"]
        },
    }
aggregate_delta = {
    label: {
        "ee_position_mean_m": aggregate[label]["ee_position_mean_m"] - aggregate["original"]["ee_position_mean_m"],
        "ee_orientation_mean_rad": aggregate[label]["ee_orientation_mean_rad"]
        - aggregate["original"]["ee_orientation_mean_rad"],
        "safety_per_env_step": aggregate[label]["safety_per_env_step"] - aggregate["original"]["safety_per_env_step"],
        "reward_per_env_step": aggregate[label]["reward_per_env_step"] - aggregate["original"]["reward_per_env_step"],
        "terms": {name: aggregate[label]["terms"][name] - aggregate["original"]["terms"][name] for name in term_names},
    }
    for label in LABELS
    if label in aggregate and "original" in aggregate
}

result = {
    "protocol": "fixed pre-generated commands, 64 env, 200 control steps, deterministic mean, corruption off, weight=2 evaluation",
    "phase_rule": next(iter(summary.values()))["phase_rule"],
    "pairing": pairing,
    "per_seed": {f"{seed}_{label}": summary[(seed, label)] for (seed, label) in summary},
    "reward_terms": term_table,
    "phase_table": phase_table,
    "aggregate_two_seeds": aggregate,
    "aggregate_delta_vs_original": aggregate_delta,
    "notes": {
        "p95": "Per-seed P95 values are reported separately; a mean of per-seed P95 is labelled as "
               "'each seed's P95 averaged', never as a pooled-sample P95.",
        "reward_scale": "weighted_per_env_step = sum over env-steps of (_step_reward * dt) / env-steps; "
                        "raw_per_env_step = the same divided by the term weight.",
        "sign_convention": "Reward contributions add to the total; a more negative value is worse for "
                           "that term. Safety differences are signed contributions, not magnitudes.",
    },
}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(result, indent=1))
print(json.dumps({
    "pairing": pairing,
    "aggregate": aggregate,
    "aggregate_delta_vs_original": aggregate_delta,
}, indent=1)[:20000])
print(f"wrote {args.out}")
