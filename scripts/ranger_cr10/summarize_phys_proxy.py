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

"""Assemble the PhysX-consistent proxy round summary."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/stage413/b/simple_collision_v2"
ASSET = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10_simple_v2/ranger_cr10_simple_v2_flat.usd"


def load(name):
    return json.loads((OUT / name).read_text())


eval_data = load("phys_proxy_eval.json")
targeted = load("phys_targeted_set.json")
targeted_rule = load("phys_targeted_rule.json")
timing = load("phys_proxy_contact_timing.json")
union = load("phys_proxy_union_regression.json")
mapping = load("collider_proxy_map.json")
env64 = load("validation_64env.json")
env1 = load("validation_1env_baseline.json")


def a_global(summary_key):
    rows = eval_data[summary_key]["rows"]
    return {
        "margin_confirmed_unsafe_by_arm_lidar": sum(1 for row in rows if row["a_margin"] == "unsafe"),
        "margin_unknown_global": sum(1 for row in rows if row["a_margin"] != "unsafe"),
        "tolerance_confirmed_unsafe_by_arm_lidar": sum(1 for row in rows if row["a_tol"] == "unsafe"),
        "tolerance_unknown_global": sum(1 for row in rows if row["a_tol"] != "unsafe"),
        "note": "global-safe is unknown because A does not query arm-arm, arm-box, arm-chassis or arm-d435",
    }


def targeted_stats():
    rows = targeted["rows"]
    def label(row, key, threshold):
        status = row[key + "_status"]
        distance = row[key + "_distance"]
        if status == "contact":
            return "unsafe"
        if not isinstance(distance, (int, float)):
            return "unknown"
        return "unsafe" if distance <= max(threshold, 1e-4) else "safe"

    def confusion(ref_key, pred_key):
        valid = [row for row in rows if row[ref_key] in ("safe", "unsafe") and row[pred_key] in ("safe", "unsafe")]
        return dict(
            TP=sum(1 for row in valid if row[ref_key] == "unsafe" and row[pred_key] == "unsafe"),
            FP=sum(1 for row in valid if row[ref_key] == "safe" and row[pred_key] == "unsafe"),
            FN=sum(1 for row in valid if row[ref_key] == "unsafe" and row[pred_key] == "safe"),
            TN=sum(1 for row in valid if row[ref_key] == "safe" and row[pred_key] == "safe"),
            valid=len(valid), unknown=len(rows) - len(valid),
        )

    for row in rows:
        row["_p_tol"] = label(row, "proxy_pair", 0.0)
        row["_p_margin"] = label(row, "proxy_pair", 0.02)
        row["_r_tol"] = label(row, "ref_pair", 0.0)
        row["_r_margin"] = label(row, "ref_pair", 0.02)
    errors = [row["error_m"] for row in rows if row["error_m"] is not None]
    bands = Counter(row["row_id"].split("_")[-1] for row in rows if row["source"] == "random_uniform")
    return dict(
        n=len(rows),
        rule=targeted["rule"],
        rule_sha256=targeted["rule_sha256"],
        sources=dict(Counter(row["source"] for row in rows)),
        random_band_counts=dict(bands),
        pair_b_tolerance=confusion("_r_tol", "_p_tol"),
        pair_b_margin=confusion("_r_margin", "_p_margin"),
        global_b_tolerance=confusion("ref_global_tol", "proxy_global_tol"),
        global_b_margin=confusion("ref_global_margin", "proxy_global_margin"),
        pair_a_tolerance=confusion("a_tol", "_p_tol"),
        pair_a_margin=confusion("a_margin", "_p_margin"),
        distance_error=dict(n=len(errors), max_abs_m=max(abs(value) for value in errors) if errors else None,
                            median_abs_m=float(sorted(abs(value) for value in errors)[len(errors) // 2])
                            if errors else None),
        proxy_contacts=sum(1 for row in rows if row["proxy_pair_status"] == "contact"),
        ref_contacts=sum(1 for row in rows if row["ref_pair_status"] == "contact"),
        proxy_capped=sum(1 for row in rows if row["proxy_capped"]),
        inactive_structural_rows=[row["row_id"] for row in rows if not row["active_pair"]],
        a_available_rows=sum(1 for row in rows if row["a_tol"] is not None),
        note="diagnostic set; 32 random rows are independent of A labels but share the same FK; "
             "the 42 fixed-margin rows are source rows, not independent evidence",
    )


def key_rows():
    rows = {row["row_id"]: row for row in eval_data["fixed"]["rows"]}
    out = {}
    for row_id in ("D0090", "D0106", "D0109", "D0157", "D0264", "D0252",
                   "P0021", "P0022", "P0023", "P0024", "D0249", "D0267"):
        row = rows[row_id]
        out[row_id] = {
            "arm": row["arm"],
            "a_tolerance": row["a_tol"], "a_margin": row["a_margin"],
            "a_reference_m": row["a_reference"],
            "b_pair_m": row["ref_pair_distance"],
            "b_pair_tolerance": row["ref_pair_tol"], "b_pair_margin": row["ref_pair_margin"],
            "proxy_pair_m": row["proxy_pair_distance"],
            "proxy_pair_tolerance": row["proxy_pair_tol"], "proxy_pair_margin": row["proxy_pair_margin"],
            "b_global_tolerance": row["ref_global_tol"], "b_global_margin": row["ref_global_margin"],
            "proxy_global_tolerance": row["proxy_global_tol"],
            "proxy_global_margin": row["proxy_global_margin"],
            "b_global_margin_winner": {
                "pair": row["ref_global_margin_winner"]["pair"],
                "distance_m": row["ref_global_margin_winner"]["distance"],
                "status": row["ref_global_margin_winner"]["status"],
                "shape_a": row["ref_global_margin_winner"]["shape_a"],
                "shape_b": row["ref_global_margin_winner"]["shape_b"],
            },
        }
    return out


performance = env64["performance"]
result = {
    "date": "2026-11-07",
    "physical_freeze": {
        "usd": str(ASSET.relative_to(ROOT)),
        "sha256": hashlib.sha256(ASSET.read_bytes()).hexdigest(),
        "bytes": ASSET.stat().st_size,
        "rigid_bodies": 23,
        "actual_colliders": 30,
        "body_filter_pairs": 77,
        "self_collisions": True,
        "chassis": "previous version: URDF/PhysX unscaled convex hull; the old MuJoCo AABB is not "
                   "used in the PhysX-consistent proxy",
        "box_link": "left enclosure + top accessory; right plate visual only",
    },
    "physical_64env": {
        "verdict": "pass under the current fixed initialization/control/test protocol",
        "scope": "64 env, home/hold/reset/partial-reset/motion/base/blocking; not extrapolated to "
                 "random initialization, arbitrary actions or all scales",
        "performance": {
            "gpu": performance["gpu"],
            "gpu_total_memory_gb": performance["gpu_total_memory_gb"],
            "process_gpu_memory_used_gb_approx": performance["gpu_total_memory_gb"] - performance["free_memory_gb_after"],
            "torch_allocator_peak_gb": performance["peak_allocated_gb"],
            "control_step_s": performance["control_step_s"],
            "amortised_physics_step_s": performance["physics_step_s_estimate"],
            "amortised_physics_note": "control_step_s / decimation; not an independently measured PhysX kernel time",
            "env_control_steps_s": performance["throughput_env_steps_s"],
            "per_env_control_hz": performance["throughput_env_steps_s"] / env64["num_envs"],
            "physics_steps_s_amortised": performance["throughput_physics_steps_s"],
            "single_env_baseline_control_step_s": env1["performance"]["control_step_s"],
            "single_env_baseline_amortised_physics_step_s": env1["performance"]["physics_step_s_estimate"],
            "note": "test-script throughput, not PPO training throughput; no viewer/debug queries in this run",
        },
        "rerun_this_round": False,
        "rerun_reason": "no simulation configuration changed; only offline proxy code and a 2-env "
                        "contact-timing diagnostic were added",
    },
    "definitions": {
        "per_pair": "prediction and reference must be the same logical pair; a link is the union minimum "
                    "over its atomic shapes, and the winning atomic subpair is kept",
        "global": "unsafe if any confirmed reference pair is unsafe; safe only if every pair in the declared "
                  "scope has a valid negative; otherwise unknown",
        "global_scope": "44 active internal pairs: non-adjacent arm pairs plus arm vs chassis/box_left/"
                        "box_top/cr10_base/lidar/d435, excluding the Link1 structural mount pairs",
        "a_reference": "original arm/lidar mesh reference; per-pair only, not a complete global reference",
        "b_reference": "frozen PhysX collider shapes (cube or convexHull), exact GJK query",
        "global_a": "a confirmed arm-lidar positive makes global A unsafe; an arm-lidar negative leaves "
                    "the other pairs unqueried, so global A is unknown, never a fabricated FP/TN",
        "thresholds_m": {"tolerance": 1e-4, "margin": 0.02},
    },
    "mapping": {
        "actual_colliders": mapping["actual_collider_count"],
        "proxy_relevant_colliders": mapping["proxy_relevant_collider_count"],
        "out_of_scope": mapping["out_of_scope_colliders"],
        "fixed_joint_merge": mapping["fixed_joint_merge"],
        "link_frame_note": mapping["link_frame_note"],
        "unsupported_shapes": mapping["unsupported_shapes"],
        "active_pairs": mapping["active_pairs"],
        "excluded_pairs": mapping["excluded_pairs"],
        "csv": str((OUT / "collider_proxy_map.csv").relative_to(ROOT)),
    },
    "union_regression": union,
    "fixed": {
        "summary": eval_data["fixed"]["summary"],
        "a_global": a_global("fixed"),
    },
    "new_local_perturbation": {
        "summary": eval_data["new"]["summary"],
        "a_global": a_global("new"),
        "note": "local perturbations of the fixed miss rows; not independent global safety evidence",
    },
    "replay_1800": {
        "summary": eval_data["replay"]["summary"],
        "a_global": a_global("replay"),
        "reference_reuse": {
            "poses_sha256": "31aa34b8e0878ad7cc7981f307a7ddfde335f9cce83f0e426892bc4d0e706528",
            "a_reference": "reused per-pair arm-lidar reference (0 contact, 37 exact, 10763 certified-culled)",
            "kinematics_check": "scene.xml vs scene_simple_v2.xml body poses over 50 operating poses: max 0",
            "b_reference": "new exact query on the actual collider shapes; not a reuse of A",
        },
        "global_margin_unsafe_poses": 691,
        "global_margin_winner": {"pair": ["cr10_Link4", "cr10_Link6"], "distance_range_m": [0.00255553, 0.02000000]},
        "global_margin_note": "the 1800-pose A reference is arm-lidar only; the 691 global alarms come "
                              "from the active non-adjacent Link4-Link6 pair under B, not from lidar",
    },
    "targeted_diagnostic": targeted_stats(),
    "contact_timing": {
        "physics_dt_s": timing["physics_dt_s"],
        "control_dt_s": timing["control_dt_s"],
        "contact_offsets_m_range": [min(timing["contact_offsets"][0]), max(timing["contact_offsets"][0])],
        "rest_offsets_m_unique": sorted(set(timing["rest_offsets"][0])),
        "summary": timing["summary"],
        "note": "contact offset generates contact points before geometric zero; the measured force onset "
                "is near geometric zero on these two paths but is not used to change any offset",
    },
    "key_rows": key_rows(),
    "decisions": {
        "single_env_physical": "pass",
        "sixty_four_env_physical": "pass under the fixed protocol; no extrapolation",
        "physx_consistent_proxy": "pass against the frozen collider shapes: zero per-pair and global "
                                  "confusion, union regression pass, distance error <= 8.7e-9 m in the "
                                  "targeted set and <= 7.2e-9 m in the fixed set",
        "original_mesh_proxy": "not pass: fixed per-pair tolerance 21 FN / margin 42 FN, new local "
                               "perturbation 15 FN / 13 FN; these are A-vs-B geometry differences, not "
                               "B-vs-PhysX errors. Global A remains unknown on safe rows",
        "reward_integration": "not done; safety.py, reward formula, weights and policy untouched",
        "training_readiness": "an independent training config may be prepared next round to protect the "
                              "frozen PhysX collider geometry (B); no PPO or default replacement now",
        "commit_push": False,
    },
    "limitations": [
        "64-env physical pass is limited to the fixed initialization/control/test protocol; it is not a "
        "random-initialization, arbitrary-action or all-scale guarantee",
        "B uses the source STL convex hull for a PhysX convexHull collider; vertex ordering may differ, "
        "but the distance check and the two contact paths show no material mismatch",
        "A covers only arm-lidar; global A cannot be classified on safe rows",
        "the 71-row set is a local perturbation set, not independent global evidence",
        "the targeted set is diagnostic, not an independent safety set",
        "wheels/steering are out of the internal-arm-obstacle scope and are not represented in the proxy",
        "Link1-vs-chassis/cr10_base is excluded from the reward scope as the joint1 mount; PhysX leaves "
        "it active because cr10_base is merged into base_link",
        "the proxy is geometric zero distance; PhysX contact offset (0.5-8.7 mm measured) can move contact "
        "generation earlier and is not tuned to force numeric agreement",
        "no unsupported relevant collider shape was found; no unknown B label remains",
    ],
    "artifacts": {
        "collider_geometry": str(OUT / "collider_geometry_v2.json"),
        "collider_map_json": str(OUT / "collider_proxy_map.json"),
        "collider_map_csv": str(OUT / "collider_proxy_map.csv"),
        "proxy_model": "mujoco/robots/ranger_cr10/ranger_cr10_simple_v2_physproxy.xml",
        "proxy_scene": "mujoco/robots/ranger_cr10/scene_simple_v2_physproxy.xml",
        "evaluation": str(OUT / "phys_proxy_eval.json"),
        "targeted_rule": str(OUT / "phys_targeted_rule.json"),
        "targeted_set": str(OUT / "phys_targeted_set.json"),
        "contact_timing": str(OUT / "phys_proxy_contact_timing.json"),
        "union_regression": str(OUT / "phys_proxy_union_regression.json"),
    },
}
(OUT / "phys_proxy_summary.json").write_text(json.dumps(result, indent=1))
print(json.dumps({
    "physical": result["physical_64env"]["verdict"],
    "fixed": result["fixed"]["summary"],
    "new": result["new_local_perturbation"]["summary"],
    "replay_global": result["replay_1800"]["summary"]["global_b_margin"],
    "targeted": result["targeted_diagnostic"],
    "union_passed": union["passed"],
    "decisions": result["decisions"],
}, indent=1)[:12000])
