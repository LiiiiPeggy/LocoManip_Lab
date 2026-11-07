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

"""Assemble the v2 validation summary and artifact manifest.

Run after the geometry, USD, stability, blocking and proxy scripts:
    python scripts/ranger_cr10/summarize_simple_collision_v2.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/stage413/b/simple_collision_v2"
ASSET = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10_simple_v2"


def load(name):
    path = OUT / name
    return json.loads(path.read_text()) if path.exists() else None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def brief_contacts(path):
    data = load(path)
    if data is None:
        return None
    return [
        {
            "pair": item["pair"],
            "max_force_n": item["max_force_n"],
            "min_separation_m": item["min_separation_m"],
            "first_point": item["first_point"],
        }
        for item in data.get("aggregate", [])[:8]
    ]


geometry = load("geometry_validation.json")
usd = load("usd_inspection_v2.json")
mimic = load("mimic_references.json")
filters = load("filters_v2.json")
hold_on = load("hold_on_200.json")
hold_off = load("hold_off_200.json")
motion_on = load("motion_on.json")
motion_off = load("motion_off.json")
reset_base = load("reset_base.json")
box_on = load("block_box_on.json")
box_off = load("block_box_off.json")
arm_on = load("block_arm_front_on.json")
arm_off = load("block_arm_front_off.json")
over_on = load("block_over_on.json")
plate_on = load("block_plate_on.json")
proxy = load("distance_proxy_comparison_v2.json")
link3 = load("link3_miss_localization.json")

summary = {
    "date": "2026-11-07",
    "candidate": {
        "env_id": "RANGER-CR10-TELEOP-SIMPLECOLL",
        "urdf": str((ASSET / "ranger_cr10_simple_v2.urdf").relative_to(ROOT)),
        "urdf_sha256": sha(ASSET / "ranger_cr10_simple_v2.urdf"),
        "config": str((ASSET / "config.yaml").relative_to(ROOT)),
        "config_sha256": sha(ASSET / "config.yaml"),
        "flat_usd": str((ASSET / "ranger_cr10_simple_v2_flat.usd").relative_to(ROOT)),
        "flat_usd_sha256": sha(ASSET / "ranger_cr10_simple_v2_flat.usd"),
        "flat_usd_bytes": (ASSET / "ranger_cr10_simple_v2_flat.usd").stat().st_size,
        "env_cfg_sha256": sha(
            ROOT
            / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/config/ranger_cr10/simple_collision_env_cfg.py"
        ),
        "tracked": False,
        "rebuild": "see assets/ranger_cr10_simple_v2/README.md",
        "screenshots": sorted(
            str(path.relative_to(OUT)) for path in (OUT / "screenshots").glob("*.png")
        ) if (OUT / "screenshots").exists() else [],
    },
    "three_layers": {
        "visualization": "mujoco/robots/ranger_cr10/scene_simple_v2.xml (v2 boxes); default scene.xml retained",
        "distance_proxy": "mdp/safety.py unchanged (old A boxes, no box_link); new simple proxy offline only",
        "physx": "ranger_cr10_simple_v2_flat.usd; self-collision true; body-level filtered pairs",
        "safety_sha256": sha(ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/safety.py"),
    },
    "usd": {
        "articulation": usd["articulation"] if usd else None,
        "n_rigid_bodies": usd["n_rigid_bodies"] if usd else None,
        "n_colliders": usd["n_colliders"] if usd else None,
        "n_rigid_bodies_with_filters": usd["n_rigid_bodies_with_filters"] if usd else None,
        "n_body_filter_pairs": len(usd.get("body_filter_pair_counts", {})) if usd else None,
        "n_broken_targets": len(usd["broken_targets"]) if usd else None,
        "box_link": "merged into base_link; left enclosure and top accessory colliders retained; "
                    "the right mounting plate collision was removed per user request (visual only)",
        "mimic_repaired": mimic["repaired"] if mimic else None,
        "mimic_broken_targets": mimic["broken_targets"] if mimic else None,
        "filters": {
            "mode": filters["mode"] if filters else None,
            "body_pairs": filters["body_pairs"] if filters else None,
            "directed_targets": filters["directed_targets"] if filters else None,
        },
    },
    "geometry": {
        "adjacent_default_m": geometry["adjacent_default"] if geometry else None,
        "pose_stats_v2_adjacent": {
            key: value
            for key, value in (geometry["pose_stats"]["v2"].items() if geometry else [])
            if key.startswith("cr10_Link") and key.count("cr10_Link") == 2
            and int(key.split("|")[1].replace("cr10_Link", "").split("_")[0])
            == int(key.split("|")[0].replace("cr10_Link", "").split("_")[0]) + 1
        },
        "pose_stats_v2_non_adjacent_min_m": {
            key: value
            for key, value in (geometry["pose_stats"]["v2"].items() if geometry else [])
            if key.startswith("cr10_Link") and key.count("cr10_Link") == 2
            and int(key.split("|")[1].replace("cr10_Link", "").split("_")[0])
            != int(key.split("|")[0].replace("cr10_Link", "").split("_")[0]) + 1
        },
        "box_link": geometry["box_link"] if geometry else None,
        "arm_coverage": geometry["arm_coverage"] if geometry else None,
    },
    "chassis": {
        "collision": "reverted to the previous version: URDF/PhysX unscaled ranger_base_link convex hull; "
                     "MuJoCo base_link_col is the AABB box",
        "hull_faces": 346,
        "physx_hull_volume_m3": 0.3331,
        "mujoco_aabb_volume_m3": 0.4926,
        "mujoco_aabb_extents_m": [1.302, 0.8694, 0.4351],
        "note": "the 0.98-scaled hull trial was superseded per user request; no chassis shrink is in the "
                "current candidate. The MuJoCo AABB remains an overfilling proxy for visualization/proxy; "
                "the PhysX chassis is the unscaled convex hull.",
    },
    "self_collision_cause": {
        "v1_broken_filters_contacts": brief_contacts("diag_v1_broken_filters.json"),
        "v1_collider_filterfix_contacts": brief_contacts("diag_v1_collider_filterfix.json"),
        "v1_body_filter_contacts": brief_contacts("diag_on_bodyfilter_v1.json"),
        "conclusion": "the v1 collider-level FilteredPairsAPI did not take effect in PhysX; the "
                      "chassis/wheel and gripper overlaps produced 47-85 kN contacts. Body-level "
                      "filters remove them. Collider-level targets also suffered last-pair-wins.",
    },
    "stability": {
        "hold_on": {"max_arm_drift_rad": hold_on["max_arm_drift_rad"],
                    "base_displacement_m": hold_on["base_displacement_m"],
                    "contact_pairs": hold_on["aggregate"]} if hold_on else None,
        "hold_off": {"max_arm_drift_rad": hold_off["max_arm_drift_rad"],
                     "base_displacement_m": hold_off["base_displacement_m"],
                     "contact_pairs": hold_off["aggregate"]} if hold_off else None,
        "motion_on": {key: motion_on[key] for key in ("max_arm_drift_rad", "max_tracking_error_rad",
                                                      "base_displacement_m", "max_contact_force_n",
                                                      "contact_pairs")} if motion_on else None,
        "motion_off": {key: motion_off[key] for key in ("max_arm_drift_rad", "max_tracking_error_rad",
                                                        "base_displacement_m", "max_contact_force_n",
                                                        "contact_pairs")} if motion_off else None,
        "resets": reset_base["resets"] if reset_base else None,
        "base_motion": {
            "base_displacement_m": reset_base["base_displacement_m"],
            "max_arm_drift_rad": max(row["arm_drift_rad"] for row in reset_base["base_motion"]),
            "any_contact": any(row["contact"] for row in reset_base["base_motion"]),
            "last_box_left_world": reset_base["base_motion"][-1]["box_left_world"],
            "last_box_top_world": reset_base["base_motion"][-1]["box_top_world"],
        } if reset_base else None,
    },
    "blocking": {
        "box_on": {"target_error_rad": box_on["target_error_rad"],
                   "final_pair_distance_m": box_on["final_pair_distance_m"],
                   "base_displacement_m": box_on["base_displacement_m"],
                   "first_contact": box_on["first_contact"],
                   "contact_pairs": box_on["contact_pairs"]} if box_on else None,
        "box_off": {"target_error_rad": box_off["target_error_rad"],
                    "final_pair_distance_m": box_off["final_pair_distance_m"],
                    "first_contact": box_off["first_contact"]} if box_off else None,
        "arm_on": {"target_error_rad": arm_on["target_error_rad"],
                   "final_pair_distance_m": arm_on["final_pair_distance_m"],
                   "base_displacement_m": arm_on["base_displacement_m"],
                   "first_contact": arm_on["first_contact"],
                   "contact_pairs": arm_on["contact_pairs"]} if arm_on else None,
        "arm_off": {"target_error_rad": arm_off["target_error_rad"],
                    "final_pair_distance_m": arm_off["final_pair_distance_m"],
                    "first_contact": arm_off["first_contact"]} if arm_off else None,
        "over_box_on": {"target_error_rad": over_on["target_error_rad"],
                        "final_pair_distance_m": over_on["final_pair_distance_m"],
                        "first_contact": over_on["first_contact"]} if over_on else None,
        "plate_on": (
            {"status": plate_on.get("status"), "reason": plate_on.get("reason")}
            if plate_on and plate_on.get("status") == "not_applicable"
            else {
                "target_error_rad": plate_on["target_error_rad"],
                "final_pair_distance_m": plate_on["final_pair_distance_m"],
                "first_contact": plate_on["first_contact"],
                "contact_pairs": plate_on["contact_pairs"],
            } if plate_on else None
        ),
        "plate_verdict": "the right mounting plate collision was removed per user request; the plate is "
                         "visual only. The old plate target is no longer a collision test and is N/A.",
    },
    "proxy": {
        "replay": proxy["replay"] if proxy else None,
        "global_tolerance": proxy["near"]["new_tolerance"] if proxy else None,
        "global_margin": proxy["near"]["new_margin"] if proxy else None,
        "row_tolerance": proxy["near"]["new_row_tolerance"] if proxy else None,
        "row_margin": proxy["near"]["new_row_margin"] if proxy else None,
        "margin_fn_ids": proxy["near"]["margin_fn_ids"] if proxy else None,
        "margin_fn_details": proxy["near"]["margin_fn_details"] if proxy else None,
        "p0021_p0024": proxy["near"]["p0021_p0024"] if proxy else None,
        "link3_miss_categories": {
            item["row_id"]: item["category"] for item in link3
        } if link3 else None,
        "correction": "the previous 1.19/4.18/5.18/18.62 mm values were the global minimum over all "
                      "arm-protected pairs, not the Link3-lidar pair. The Link3 box-lidar distance is "
                      "42-53 mm at P0021-P0024; the misses are at the distal joint4 retreat. The "
                      "20 mm margin false-safe rows are D0106, D0109, D0157, D0264.",
    },
    "verdict": {
        "single_env_self_collision_stability": "pass (on/off both 0 self-contacts; base settling 22-32 mm)",
        "solid_blocking": "pass for the left enclosure; non-adjacent Link3-Link6 pass; the right "
                          "plate collider was removed per user (visual only), so plate blocking is N/A",
        "open_space_above_box": "pass: Link2 crosses the v1 upper slab with v2 clearance 18.7 mm and no contact",
        "box_link_follow": "pass: both colliders merged into base_link; world centres follow the 0.342 m base motion",
        "distance_proxy": "not acceptable as a reward proxy: row-specific tolerance FN 31/81, margin FN 60/190; "
                          "global tolerance FN 0 and margin FN 4; all 20 Link3 misses are in the "
                          "distal joint4 retreat; 11 Link6 misses",
        "small_scale_parallel": "a small-scale self-collision-only parallel validation can proceed; do not "
                                "start PPO or replace the reward proxy. The default deployment, safety.py and "
                                "frozen assets remain untouched.",
    },
}

(OUT / "validation_summary_v2.json").write_text(json.dumps(summary, indent=1))

artifacts = sorted(path for path in OUT.rglob("*") if path.is_file())
manifest = {
    "created": "2026-11-07",
    "root": str(OUT.relative_to(ROOT)),
    "files": {
        str(path.relative_to(OUT)): {"sha256": sha(path), "bytes": path.stat().st_size}
        for path in artifacts
        if path.name != "RESULT_MANIFEST_V2.json"
    },
    "note": "v2 simple-collision evidence; generated USDs are untracked; default config/reward untouched",
}
(OUT / "RESULT_MANIFEST_V2.json").write_text(json.dumps(manifest, indent=1))
print(json.dumps(summary["verdict"], indent=1))
print(f"wrote {OUT / 'validation_summary_v2.json'}")
print(f"wrote {OUT / 'RESULT_MANIFEST_V2.json'} with {len(manifest['files'])} files")
