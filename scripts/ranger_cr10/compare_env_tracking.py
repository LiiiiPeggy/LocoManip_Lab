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

"""Cross-environment tracking comparison for the three policies.

Compares the same three checkpoints in the two physical environments:

* ``simplecoll`` — v2 box asset, self-collisions ON, B-proxy 20 mm clearance term;
* ``teleop``     — original mesh asset, self-collisions OFF, safety.py 15 mm clearance term.

The v2 rollouts are reused from the attribution round; the teleop rollouts are new. Both were
produced by ``rollout_phys_proxy_fixed_commands.py`` with the same pre-generated command plan,
the same phase rule and the same pre-reset capture, so the ``hold``-phase errors are comparable
across assets. Only tracking is compared across environments; the clearance reward term itself
differs between the two configs, so its value is reported per environment and never differenced.

Usage:
    python scripts/ranger_cr10/compare_env_tracking.py \
        --v2-dir outputs/stage413/b/attribution \
        --teleop-dir outputs/stage413/b/cross_env \
        --out outputs/stage413/b/cross_env/cross_env_comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--v2-dir", default=str(ROOT / "outputs/stage413/b/attribution"))
ap.add_argument("--teleop-dir", default=str(ROOT / "outputs/stage413/b/cross_env"))
ap.add_argument("--seeds", default="42,43")
ap.add_argument("--out", default=str(ROOT / "outputs/stage413/b/cross_env/cross_env_comparison.json"))
args = ap.parse_args()

LABELS = ("original", "weight2", "weight0")
ENVS = ("simplecoll", "teleop")
SEEDS = [int(seed) for seed in args.seeds.split(",")]
PATTERNS = {
    "simplecoll": lambda seed, label: Path(args.v2_dir) / f"seed{seed}_{label}.json",
    "teleop": lambda seed, label: Path(args.teleop_dir) / f"teleop_seed{seed}_{label}.json",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def run_point(data: dict) -> dict:
    records = data["records"]
    split = data["phase_split"]
    hold = split["hold"]
    return {
        "checkpoint": data["checkpoint"],
        "checkpoint_sha256": data["checkpoint_sha256"],
        "env_name": data["env_name"],
        "env_id": data["env_id"],
        "env_asset": data["env_asset"],
        "env_clearance_term": data["env_clearance_term"],
        "command_plan_hash": data["command_plan_hash"],
        "initial_digest": data["initial_digest"],
        "initial_arm_joint_pos": data["initial_arm_joint_pos"],
        "initial_base_height_m": data["initial_base_height_m"],
        "command_consistency": data["command_consistency"],
        "control_dt_s": data["control_dt_s"],
        "hold_env_steps": hold["env_steps"],
        "hold_share": hold["share"],
        "hold_pos_err_mean_m": hold["pos_err_mean_m"],
        "hold_ori_err_mean_rad": hold["ori_err_mean_rad"],
        "rollout_pos_err_mean_m": mean([item["pos_err"] for item in records]),
        "rollout_ori_err_mean_rad": mean([item["ori_err"] for item in records]),
        "safety_per_env_step": data["safety_reward"]["per_env_step"],
        "actual_physx_contacts": data["actual_physx_contacts"],
        "dones_total": data["episode_counts"]["dones_total"],
        "termination_counts": data["episode_counts"]["termination_counts"],
        "nan_or_inf_steps": data["nan_or_inf_steps"],
        "query_failures": data["query_failures"],
        "step_time_mean_s": data["step_time_s"]["mean"],
        "joint_dev_abs_sum_mean": data["arm_deviation_stats"]["joint_dev_abs_sum_mean"],
    }


points = {}
missing = []
for env in ENVS:
    for seed in SEEDS:
        for label in LABELS:
            path = PATTERNS[env](seed, label)
            if not path.is_file():
                missing.append(str(path))
                continue
            points[f"{env}:{seed}:{label}"] = run_point(load(path))

# Command-plan consistency: the same seed must produce the same plan in both environments, and
# every rollout must have applied it for all steps.
plan_check = {}
for seed in SEEDS:
    hashes = {
        f"{env}:{label}": points[f"{env}:{seed}:{label}"]["command_plan_hash"]
        for env in ENVS if f"{env}:{seed}:original" in points
        for label in LABELS if f"{env}:{seed}:{label}" in points
    }
    steps_ok = {
        f"{env}:{label}": points[f"{env}:{seed}:{label}"]["command_consistency"]
        for env in ENVS
        for label in LABELS
        if f"{env}:{seed}:{label}" in points
    }
    plan_check[str(seed)] = {
        "plan_hashes": hashes,
        "plan_hash_all_equal_across_envs_and_policies": len(set(hashes.values())) == 1,
        "all_steps_match_plan": all(item["all_steps_match_plan"] for item in steps_ok.values()),
        "steps_checked": sorted({item["steps_checked"] for item in steps_ok.values()}),
        "envs_checked": sorted({item["envs_checked"] for item in steps_ok.values()}),
        "initial_digest_all_equal": len({
            points[f"{env}:{seed}:{label}"]["initial_digest"]
            for env in ENVS for label in LABELS if f"{env}:{seed}:{label}" in points
        }) == 1,
    }

# Cross-asset initial state, by joint name.
joint_diff = {}
if "simplecoll:42:original" in points and "teleop:42:original" in points:
    v2_joints = points["simplecoll:42:original"]["initial_arm_joint_pos"]
    teleop_joints = points["teleop:42:original"]["initial_arm_joint_pos"]
    joint_diff = {
        name: {
            "simplecoll": v2_joints.get(name),
            "teleop": teleop_joints.get(name),
            "abs_diff": abs((v2_joints.get(name) or 0.0) - (teleop_joints.get(name) or 0.0)),
        }
        for name in sorted(set(v2_joints) | set(teleop_joints))
    }
    joint_diff_max = max((item["abs_diff"] for item in joint_diff.values()), default=None)
    base_diff = abs(
        points["simplecoll:42:original"]["initial_base_height_m"]
        - points["teleop:42:original"]["initial_base_height_m"]
    )
else:
    joint_diff_max = None
    base_diff = None

# Main table: per environment and seed, the hold-phase errors and the increment against the
# same environment's model_9999.
main_table = {}
increments = {}
for env in ENVS:
    main_table[env] = {}
    increments[env] = {}
    for seed in SEEDS:
        row = {}
        for label in LABELS:
            key = f"{env}:{seed}:{label}"
            if key not in points:
                continue
            point = points[key]
            row[label] = {
                "hold_env_steps": point["hold_env_steps"],
                "hold_pos_err_mean_m": point["hold_pos_err_mean_m"],
                "hold_ori_err_mean_rad": point["hold_ori_err_mean_rad"],
                "rollout_pos_err_mean_m": point["rollout_pos_err_mean_m"],
                "rollout_ori_err_mean_rad": point["rollout_ori_err_mean_rad"],
            }
        reference = row.get("original")
        if reference:
            for label in ("weight2", "weight0"):
                if label in row:
                    row[label]["delta_hold_pos_mm"] = (
                        row[label]["hold_pos_err_mean_m"] - reference["hold_pos_err_mean_m"]
                    ) * 1000.0
                    row[label]["delta_hold_ori_mrad"] = (
                        row[label]["hold_ori_err_mean_rad"] - reference["hold_ori_err_mean_rad"]
                    ) * 1000.0
                    row[label]["ratio_hold_pos"] = (
                        row[label]["hold_pos_err_mean_m"] / reference["hold_pos_err_mean_m"]
                    )
                    row[label]["ratio_hold_ori"] = (
                        row[label]["hold_ori_err_mean_rad"] / reference["hold_ori_err_mean_rad"]
                    )
        main_table[env][str(seed)] = row
        increments[env][str(seed)] = {
            label: {
                "delta_hold_pos_mm": row.get(label, {}).get("delta_hold_pos_mm"),
                "delta_hold_ori_mrad": row.get(label, {}).get("delta_hold_ori_mrad"),
            }
            for label in ("weight2", "weight0")
        }

# Per-policy, per-seed increment averaged over the two seeds, and the env-to-env change of that
# increment. A larger degradation only in v2 means the policy is sensitive to the environment
# configuration; it does not identify a collision body.
increment_comparison = {}
for label in ("weight2", "weight0"):
    per_env = {}
    for env in ENVS:
        pos = [increments[env][str(seed)][label]["delta_hold_pos_mm"] for seed in SEEDS]
        ori = [increments[env][str(seed)][label]["delta_hold_ori_mrad"] for seed in SEEDS]
        per_env[env] = {
            "delta_hold_pos_mm_per_seed": pos,
            "delta_hold_ori_mrad_per_seed": ori,
            "mean_delta_hold_pos_mm": mean(pos),
            "mean_delta_hold_ori_mrad": mean(ori),
        }
    increment_comparison[label] = per_env
    if "simplecoll" in per_env and "teleop" in per_env:
        increment_comparison[label]["simplecoll_minus_teleop"] = {
            "delta_hold_pos_mm": per_env["simplecoll"]["mean_delta_hold_pos_mm"]
            - per_env["teleop"]["mean_delta_hold_pos_mm"],
            "delta_hold_ori_mrad": per_env["simplecoll"]["mean_delta_hold_ori_mrad"]
            - per_env["teleop"]["mean_delta_hold_ori_mrad"],
        }

result = {
    "protocol": "pre-generated policy-independent commands, 64 env, 200 control steps, deterministic mean, "
                "corruption off; identical plan, phase rule and pre-reset capture in both environments",
    "phase_rule_hold": "episode_length > 10 and min(ee command age, velocity command age) > 5 control steps",
    "environments": {
        "simplecoll": {
            "env_id": "RANGER-CR10-TELEOP-SIMPLECOLL",
            "asset": "ranger_cr10_simple_v2_flat.usd, self-collisions ON",
            "clearance_term": "phys_proxy_clearance_penalty, 44 B pairs, 20 mm margin",
            "source": "outputs/stage413/b/attribution (reused, same round as plan section 46)",
        },
        "teleop": {
            "env_id": "RANGER-CR10-TELEOP-WBC",
            "asset": "ranger_cr10.usd, self-collisions OFF",
            "clearance_term": "sensor_clearance_penalty (safety.py proxy), 15 mm margin",
            "source": "outputs/stage413/b/cross_env (new rollouts)",
        },
    },
    "command_plan_check": plan_check,
    "cross_asset_initial_state": {
        "arm_joint_by_name": joint_diff,
        "max_abs_arm_joint_difference_rad": joint_diff_max,
        "base_height_difference_m": base_diff,
        "initial_digest_all_equal": plan_check.get(str(SEEDS[0]), {}).get("initial_digest_all_equal"),
    },
    "main_table": main_table,
    "hold_increment_vs_same_env_original": increments,
    "increment_comparison": increment_comparison,
    "points": points,
    "missing": missing,
    "notes": {
        "cross_env_rule": "Only tracking is differenced across environments. The clearance reward "
                          "term, its margin and its distance source differ by design between the two "
                          "configs, so safety_per_env_step is reported per environment and never "
                          "differenced.",
        "attribution_rule": "A larger degradation confined to the v2 environment supports "
                            "'sensitive to the environment configuration'. Attributing it to a "
                            "specific collision body additionally requires the contact records and "
                            "the error to agree in time and in pair.",
        "per_seed": "Per-seed values are the comparison; any two-seed mean is labelled as such.",
    },
}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(result, indent=1))
print(json.dumps({
    "command_plan_check": plan_check,
    "cross_asset_initial_state": {key: value for key, value in result["cross_asset_initial_state"].items()
                                 if key != "arm_joint_by_name"},
    "main_table": main_table,
    "increment_comparison": increment_comparison,
    "missing": missing,
}, indent=1)[:20000])
print(f"wrote {args.out}")
