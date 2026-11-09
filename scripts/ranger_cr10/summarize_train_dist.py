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

"""Compare the two frozen-policy rollouts under the training command distribution.

Run after ``rollout_phys_proxy_training_dist.py`` has produced the original and candidate
artifacts.  The output is the matched-distribution before/after table plus the command/age bins
needed to explain the training-vs-HOLD Link4--Link6 rate gap.

Usage:
    python scripts/ranger_cr10/summarize_train_dist.py \
        --original outputs/stage413/b/dist_explain/train_dist_original.json \
        --candidate outputs/stage413/b/dist_explain/train_dist_candidate.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--original", required=True)
ap.add_argument("--candidate", required=True)
ap.add_argument("--out", default=str(ROOT / "outputs/stage413/b/dist_explain/train_dist_comparison.json"))
args = ap.parse_args()


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


original = load(args.original)
candidate = load(args.candidate)


def rows(data: dict) -> list[dict]:
    return data["records"]


def rate(items: list[dict]) -> float | None:
    return sum(int(item["below20"]) for item in items) / len(items) if items else None


def summarize(data: dict) -> dict:
    items = rows(data)
    link = data["link4_link6_overall"]
    return {
        "checkpoint": data["checkpoint"],
        "seed": data["seed"],
        "env_steps": len(items),
        "below20": link["below20"],
        "below20_rate": link["below20"] / link["env_steps"],
        "contact": link["contact"],
        "distance_mean_m": link["mean_m"],
        "distance_p10_p50_p90_m": [link["quantiles_m"][1], link["quantiles_m"][3], link["quantiles_m"][5]],
        "ee_position_mean_m": float(np.mean([item["pos_err"] for item in items])),
        "ee_position_p95_m": float(np.percentile([item["pos_err"] for item in items], 95)),
        "ee_orientation_mean_rad": float(np.mean([item["ori_err"] for item in items])),
        "ee_orientation_p95_rad": float(np.percentile([item["ori_err"] for item in items], 95)),
        "safety_sum": data["safety_reward"]["sum"],
        "safety_per_env_step": data["safety_reward"]["per_env_step"],
        "safety_share_negative": data["safety_reward"]["share_of_negative_abs"],
        "watch_term_sums": data["watch_term_sums"],
        "actual_physx_contacts": data["actual_physx_contacts"],
        "dones_total": data["episode_counts"]["dones_total"],
        "termination_counts": data["episode_counts"]["termination_counts"],
        "resample_env_events": data["episode_counts"]["resample_env_events"],
        "nan_or_inf_steps": data["nan_or_inf_steps"],
        "query_failures": data["query_failures"],
        "step_time_mean_s": data["step_time_s"]["mean"],
        "top_pairs": data["top_pairs"],
        "protocol": data["protocol"],
    }


def bins(items: list[dict], key: str, edges: list[float]) -> list[dict]:
    values = np.array([float(item[key]) for item in items])
    out = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (values >= low) & (values < high)
        selected = [item for item, keep in zip(items, mask) if keep]
        out.append({
            "range": [low, high],
            "n": len(selected),
            "below20": sum(int(item["below20"]) for item in selected),
            "rate": rate(selected),
            "mean_safety": float(np.mean([item["safety"] for item in selected])) if selected else None,
            "mean_pos_err": float(np.mean([item["pos_err"] for item in selected])) if selected else None,
        })
    return out


def axis_bins(items: list[dict], axis: int, edges_deg: list[float]) -> list[dict]:
    values = np.degrees(np.array([item["cmd_euler"][axis] for item in items]))
    out = []
    for low, high in zip(edges_deg[:-1], edges_deg[1:]):
        mask = (values >= low) & (values < high)
        selected = [item for item, keep in zip(items, mask) if keep]
        out.append({
            "range_deg": [low, high],
            "n": len(selected),
            "below20": sum(int(item["below20"]) for item in selected),
            "rate": rate(selected),
        })
    return out


def hold_like(items: list[dict]) -> list[dict]:
    selected = []
    for item in items:
        x, y, z = item["cmd_offset"]
        near_hold = abs(x - 0.06) <= 0.05 and abs(y + 0.05) <= 0.05 and abs(z - 0.04) <= 0.05
        if near_hold and item["cmd_angle"] <= math.radians(10):
            selected.append(item)
    return selected


def observed_ranges(items: list[dict]) -> dict:
    return {
        "offset_min": [float(min(item["cmd_offset"][axis] for item in items)) for axis in range(3)],
        "offset_max": [float(max(item["cmd_offset"][axis] for item in items)) for axis in range(3)],
        "euler_min_deg": [float(np.degrees(min(item["cmd_euler"][axis] for item in items))) for axis in range(3)],
        "euler_max_deg": [float(np.degrees(max(item["cmd_euler"][axis] for item in items))) for axis in range(3)],
        "angle_p50_p95_deg": [
            float(np.degrees(np.percentile([item["cmd_angle"] for item in items], q))) for q in (50, 95)
        ],
    }


def pairing(orig: dict, cand: dict) -> dict:
    same_initial = orig["initial_digest"] == cand["initial_digest"]
    digest_matches = sum(
        int(a == b) for a, b in zip(orig["step_command_digests"], cand["step_command_digests"])
    )
    digest_pairs = zip(orig["step_command_digests"], cand["step_command_digests"])
    first_mismatch = next((index for index, (a, b) in enumerate(digest_pairs) if a != b), None)
    orig_map = {(item["step"], item["env"]): item for item in orig["records"]}
    cand_map = {(item["step"], item["env"]): item for item in cand["records"]}
    common = set(orig_map) & set(cand_map)
    command_equal = 0
    for key in common:
        a = orig_map[key]
        b = cand_map[key]
        same_offset = all(abs(x - y) <= 1e-6 for x, y in zip(a["cmd_offset"], b["cmd_offset"]))
        same_euler = all(abs(x - y) <= 1e-6 for x, y in zip(a["cmd_euler"], b["cmd_euler"]))
        command_equal += int(same_offset and same_euler)
    return {
        "initial_digest_match": bool(same_initial),
        "step_command_digest_matches": int(digest_matches),
        "steps": len(orig["step_command_digests"]),
        "first_step_command_mismatch": first_mismatch,
        "paired_env_steps": len(common),
        "paired_command_equal_env_steps": command_equal,
        "paired_command_equal_rate": command_equal / len(common) if common else None,
        "note": "same seed and same reset/command config; a mismatch after a termination is expected "
                "and is not treated as strict per-step pairing",
    }


orig_summary = summarize(original)
cand_summary = summarize(candidate)
orig_items = rows(original)
cand_items = rows(candidate)

angle_edges = [0.0, 2.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 180.0]
age_edges = [0.0, 1.0, 5.0, 20.0, 50.0, 10000.0]
episode_edges = [0.0, 20.0, 50.0, 100.0, 150.0, 200.0]
axis_edges = [-180.0, -60.0, -30.0, -10.0, 10.0, 30.0, 60.0, 180.0]

comparison = {
    "protocol": "training command distribution, 64 env, 200 steps, seed 42, deterministic inference, corruption off",
    "pairing": pairing(original, candidate),
    "original": orig_summary,
    "candidate": cand_summary,
    "delta_candidate_minus_original": {
        "below20_rate": cand_summary["below20_rate"] - orig_summary["below20_rate"],
        "distance_mean_m": cand_summary["distance_mean_m"] - orig_summary["distance_mean_m"],
        "ee_position_mean_m": cand_summary["ee_position_mean_m"] - orig_summary["ee_position_mean_m"],
        "ee_position_p95_m": cand_summary["ee_position_p95_m"] - orig_summary["ee_position_p95_m"],
        "ee_orientation_mean_rad": cand_summary["ee_orientation_mean_rad"] - orig_summary["ee_orientation_mean_rad"],
        "safety_per_env_step": cand_summary["safety_per_env_step"] - orig_summary["safety_per_env_step"],
        "dones_total": cand_summary["dones_total"] - orig_summary["dones_total"],
        "actual_physx_contact_env_steps": (
            cand_summary["actual_physx_contacts"]["nonwheel_body_net_force_gt_100n_env_steps"]
            - orig_summary["actual_physx_contacts"]["nonwheel_body_net_force_gt_100n_env_steps"]
        ),
        "actual_physx_contact_body_events": (
            cand_summary["actual_physx_contacts"]["nonwheel_body_events"]
            - orig_summary["actual_physx_contacts"]["nonwheel_body_events"]
        ),
    },
    "original_bins": {
        "command_angle_deg": bins(orig_items, "cmd_angle", [math.radians(x) for x in angle_edges]),
        "command_age_steps": bins(orig_items, "cmd_age", age_edges),
        "episode_length_steps": bins(orig_items, "episode_length", episode_edges),
        "roll_deg": axis_bins(orig_items, 0, axis_edges),
        "pitch_deg": axis_bins(orig_items, 1, axis_edges),
        "yaw_deg": axis_bins(orig_items, 2, axis_edges),
        "hold_like": {
            "n": len(hold_like(orig_items)),
            "below20": sum(int(item["below20"]) for item in hold_like(orig_items)),
            "rate": rate(hold_like(orig_items)),
        },
    },
    "candidate_bins": {
        "command_angle_deg": bins(cand_items, "cmd_angle", [math.radians(x) for x in angle_edges]),
        "command_age_steps": bins(cand_items, "cmd_age", age_edges),
        "episode_length_steps": bins(cand_items, "episode_length", episode_edges),
        "roll_deg": axis_bins(cand_items, 0, axis_edges),
        "pitch_deg": axis_bins(cand_items, 1, axis_edges),
        "yaw_deg": axis_bins(cand_items, 2, axis_edges),
        "hold_like": {
            "n": len(hold_like(cand_items)),
            "below20": sum(int(item["below20"]) for item in hold_like(cand_items)),
            "rate": rate(hold_like(cand_items)),
        },
    },
    "observed_command_ranges": {
        "original": observed_ranges(orig_items),
        "candidate": observed_ranges(cand_items),
    },
}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(comparison, indent=1))
print(json.dumps(comparison, indent=1)[:20000])
print(f"wrote {args.out}")
