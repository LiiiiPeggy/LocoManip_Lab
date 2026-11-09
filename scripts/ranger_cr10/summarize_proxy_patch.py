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

"""Assemble the 64-env and proxy-patch round summary."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/stage413/b/simple_collision_v2"


def load(name):
    path = OUT / name
    return json.loads(path.read_text()) if path.exists() else None


def counts(entry):
    if not entry:
        return None
    return {key: entry.get(key) for key in ("TP", "FP", "FN", "TN", "valid", "unknown", "miss_rate")}


v2 = load("distance_proxy_comparison_v2.json")
l3 = load("distance_proxy_patch_l3.json")
tight = load("distance_proxy_patch_l3_tight.json")
l3l6 = load("distance_proxy_patch_comparison.json")
new_set = load("proxy_new_set.json")
env64 = load("validation_64env.json")
env1 = load("validation_1env_baseline.json")
usd = load("usd_inspection_v2.json")
mimic = load("mimic_references.json")
filters = load("filters_v2.json")
fixed_ref = {row["row_id"]: row for row in json.loads(
    (ROOT / "outputs/stage413/b/radar_near_contact/v3/near_set_containment_v3.json").read_text())["rows"]}
new_ref = {row["row_id"]: row for row in (new_set or {}).get("rows", [])}
fixed_rows_csv = {row["row_id"]: row for row in csv.DictReader(open(OUT / "fixed_patch_rows.csv"))} \
    if (OUT / "fixed_patch_rows.csv").exists() else {}
new_rows_csv = {row["row_id"]: row for row in csv.DictReader(open(OUT / "new_patch_rows.csv"))} \
    if (OUT / "new_patch_rows.csv").exists() else {}

asset_path = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10_simple_v2/ranger_cr10_simple_v2_flat.usd"
result = {
    "date": "2026-11-07",
    "physical_freeze": {
        "usd": str(asset_path.relative_to(ROOT)),
        "sha256": hashlib.sha256(asset_path.read_bytes()).hexdigest() if asset_path.exists() else None,
        "bytes": asset_path.stat().st_size if asset_path.exists() else None,
        "rigid_bodies": usd["n_rigid_bodies"] if usd else None,
        "colliders": usd["n_colliders"] if usd else None,
        "body_filter_pairs": len(usd["body_filter_pair_counts"]) if usd else None,
        "self_collisions": usd["articulation"]["enabled_self_collisions"] if usd else None,
        "mimic_repaired": mimic["repaired"] if mimic else None,
        "chassis": "previous version: URDF/PhysX unscaled hull; MuJoCo AABB box",
        "box_link": "left enclosure + small top accessory; right plate visual only",
    },
    "sixty_four_env": {
        "verdict": env64["verdict"] if env64 else None,
        "num_envs": env64["num_envs"] if env64 else None,
        "hold_worst": env64["hold"]["stats"] if env64 else None,
        "hold_abnormal": env64["hold"]["abnormal_contacts"] if env64 else None,
        "all_reset": env64["all_reset"] if env64 else None,
        "partial_reset": env64["partial_reset"] if env64 else None,
        "motion": env64["motion"] if env64 else None,
        "base_motion": env64["base_motion"] if env64 else None,
        "blocking": env64["blocking"] if env64 else None,
        "env_asset_summary": env64["env_asset_summary"] if env64 else None,
        "performance": env64["performance"] if env64 else None,
        "single_env_baseline": {
            "hold_worst": env1["hold"]["stats"] if env1 else None,
            "all_reset": env1["all_reset"] if env1 else None,
            "partial_reset": env1["partial_reset"] if env1 else None,
            "motion": env1["motion"] if env1 else None,
            "base_motion": env1["base_motion"] if env1 else None,
            "blocking": env1["blocking"] if env1 else None,
            "performance": env1["performance"] if env1 else None,
        },
    },
    "proxy_v2_corrected": {
        "note": "per-link union min over atomic geoms; the earlier Link6 11 misses were the d455 geoms "
                "overwriting the Link6 main box, not true misses",
        "global_tolerance": counts(v2["near"]["new_tolerance"]) if v2 else None,
        "global_margin": counts(v2["near"]["new_margin"]) if v2 else None,
        "row_tolerance": counts(v2["near"]["new_row_tolerance"]) if v2 else None,
        "row_margin": counts(v2["near"]["new_row_margin"]) if v2 else None,
        "row_tolerance_fn_ids": v2["near"]["row_tolerance_fn_ids"] if v2 else None,
        "row_margin_fn_ids": v2["near"]["row_margin_fn_ids"] if v2 else None,
        "global_margin_fn_ids": v2["near"]["margin_fn_ids"] if v2 else None,
        "row_fn_global_winners": {
            "fixed": [
                {"row_id": rid, "arm": fixed_rows_csv.get(rid, {}).get("arm"),
                 "v2_row_m": float(fixed_rows_csv[rid]["v2_row_m"]),
                 "v2_global_winner": fixed_rows_csv[rid]["v2_global_winner"],
                 "v2_global_m": float(fixed_rows_csv[rid]["v2_global_m"])}
                for rid in (l3 or {}).get("fixed_lists", {}).get("v2_row_tolerance_fn", [])
                if rid in fixed_rows_csv
            ],
            "new": [
                {"row_id": rid, "arm": new_rows_csv.get(rid, {}).get("arm"),
                 "v2_row_m": float(new_rows_csv[rid]["v2_row_m"]),
                 "v2_global_winner": new_rows_csv[rid]["v2_global_winner"],
                 "v2_global_m": float(new_rows_csv[rid]["v2_global_m"])}
                for rid in (l3 or {}).get("new_lists", {}).get("v2_row_tolerance_fn", [])
                if rid in new_rows_csv
            ],
        },
        "p0021_p0024": [
            {"row_id": row["row_id"], "v2_row_m": row["v2_row_m"], "patch_row_m": row["patch_row_m"],
             "ref_tol": row["ref_tol"], "patch_row_tol": row["patch_row_tol"]}
            for row in (l3 or {}).get("fixed_p0021_p0024", [])
        ],
    },
    "patch_variants": {
        "l3_wide": {
            "model": "scene_simple_v2_proxy.xml",
            "box": {"center": [-0.585, 0.0, 0.010], "size": [0.060, 0.100, 0.080]},
            "fixed": {
                "row_tolerance": counts(l3["fixed"]["row_tolerance"]["patch"]),
                "row_margin": counts(l3["fixed"]["row_margin"]["patch"]),
                "global_tolerance": counts(l3["fixed"]["global_tolerance"]["patch"]),
                "global_margin": counts(l3["fixed"]["global_margin"]["patch"]),
            },
            "new": {
                "row_tolerance": counts(l3["new"]["row_tolerance"]["patch"]),
                "row_margin": counts(l3["new"]["row_margin"]["patch"]),
                "global_tolerance": counts(l3["new"]["global_tolerance"]["patch"]),
                "global_margin": counts(l3["new"]["global_margin"]["patch"]),
            },
            "fixed_lists": l3["fixed_lists"],
            "new_lists": l3["new_lists"],
        },
        "l3_tight": {
            "box": {"center": [-0.592, -0.012, -0.010], "size": [0.042, 0.060, 0.024]},
            "fixed": {
                "row_tolerance": counts(tight["fixed"]["row_tolerance"]["patch"]),
                "row_margin": counts(tight["fixed"]["row_margin"]["patch"]),
                "global_tolerance": counts(tight["fixed"]["global_tolerance"]["patch"]),
                "global_margin": counts(tight["fixed"]["global_margin"]["patch"]),
            },
            "new": {
                "row_tolerance": counts(tight["new"]["row_tolerance"]["patch"]),
                "row_margin": counts(tight["new"]["row_margin"]["patch"]),
                "global_tolerance": counts(tight["new"]["global_tolerance"]["patch"]),
                "global_margin": counts(tight["new"]["global_margin"]["patch"]),
            },
        },
        "l3_plus_l6": {
            "model": "scene_simple_v2_proxy_l3l6.xml",
            "link6_box": {"center": [0.0025, 0.0, -0.020], "size": [0.095, 0.080, 0.050]},
            "fixed": {
                "row_tolerance": counts(l3l6["fixed"]["row_tolerance"]["patch"]),
                "row_margin": counts(l3l6["fixed"]["row_margin"]["patch"]),
                "global_tolerance": counts(l3l6["fixed"]["global_tolerance"]["patch"]),
                "global_margin": counts(l3l6["fixed"]["global_margin"]["patch"]),
            },
            "new": {
                "row_tolerance": counts(l3l6["new"]["row_tolerance"]["patch"]),
                "row_margin": counts(l3l6["new"]["row_margin"]["patch"]),
                "global_tolerance": counts(l3l6["new"]["global_tolerance"]["patch"]),
                "global_margin": counts(l3l6["new"]["global_margin"]["patch"]),
            },
            "verdict": "no FN benefit over l3_wide; adds 5 fixed tolerance FP and 1 margin FP; not adopted",
        },
    },
    "added_false_positive_audit": {
        "l3_wide_fixed_tolerance": [
            {"row_id": rid, "bin": fixed_ref.get(rid, {}).get("bin"),
             "ref_distance_m": fixed_ref.get(rid, {}).get("reference_distance_m")}
            for rid in (l3 or {}).get("fixed_lists", {}).get("patch_added_row_tolerance_fp", [])
        ],
        "l3_wide_fixed_margin": [
            {"row_id": rid, "bin": fixed_ref.get(rid, {}).get("bin"),
             "ref_distance_m": fixed_ref.get(rid, {}).get("reference_distance_m")}
            for rid in (l3 or {}).get("fixed_lists", {}).get("patch_added_row_margin_fp", [])
        ],
        "l3_wide_new_tolerance": [
            {"row_id": rid, "bin": new_ref.get(rid, {}).get("bin"),
             "ref_distance_m": new_ref.get(rid, {}).get("reference_distance_m")}
            for rid in (l3 or {}).get("new_lists", {}).get("patch_added_row_tolerance_fp", [])
        ],
    },
    "new_set_rule": (new_set or {}).get("rule"),
    "decisions": {
        "single_env_physical": "pass",
        "sixty_four_env_physical": "pass" if env64 and all(
            value in ("pass", "see per-env first_contact; expected pairs are base_link|cr10_Link2 and "
                              "cr10_Link3|cr10_Link6")
            for value in env64["verdict"].values()
        ) else "fail",
        "reward_proxy": "not accepted; l3_wide removes all 20 fixed / 15 new row tolerance FN but adds "
                        "15 fixed / 3 new tolerance FP and 17 fixed margin FP (9 of the margin FP are "
                        "certified_culled >=30 mm); l3_tight reduces FP but still has 6 culled / 3 "
                        "20-30 mm margin FP and leaves 1 fixed / 3 new tolerance FN; Link6 patch is "
                        "unnecessary and adds FP",
        "training_readiness": "no; do not replace safety.py or start PPO in this round",
    },
}
(OUT / "proxy_patch_summary.json").write_text(json.dumps(result, indent=1))
print(json.dumps({
    "physical_freeze": result["physical_freeze"],
    "sixty_four_env": {"verdict": result["sixty_four_env"]["verdict"],
                       "performance": result["sixty_four_env"]["performance"]},
    "proxy_v2": {k: result["proxy_v2_corrected"][k] for k in
                 ("global_tolerance", "global_margin", "row_tolerance", "row_margin")},
    "patch_l3_wide": result["patch_variants"]["l3_wide"]["fixed"],
    "patch_l3_wide_new": result["patch_variants"]["l3_wide"]["new"],
    "decisions": result["decisions"],
}, indent=1))
print(f"wrote {OUT / 'proxy_patch_summary.json'}")
