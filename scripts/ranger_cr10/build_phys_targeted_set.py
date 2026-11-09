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

"""Build a small diagnostic set for the PhysX-consistent proxy.

The rule is written to ``phys_targeted_rule.json`` before any candidate is selected. The set is
**diagnostic**, not an independent safety set:

* the 42 fixed rows whose arm-lidar reference is 20 mm-unsafe but whose physical collider pair is
  safe (the original-mesh margin misses);
* the Link6 atomic-union winners (main box, AG95, D455, one finger);
* random uniform and operating-pose samples screened by the actual-shape proxy for arm vs
  box_link/chassis/cr10-base/d435, one per distance band and pair.

No path is densely interpolated. The 342 fixed rows remain the development regression set and the
71 perturbation rows remain a local perturbation check.

Run:
    python scripts/ranger_cr10/build_phys_targeted_set.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from phys_proxy import (  # noqa: E402
    CAP,
    MARGIN,
    ROOT,
    active_pairs,
    load_shapes,
    proxy_group_distance,
    reference_group_distance,
    shapes_by_group,
)

OUT = ROOT / "outputs/stage413/b/simple_collision_v2"
RULE_OUT = OUT / "phys_targeted_rule.json"
SET_OUT = OUT / "phys_targeted_set.json"
FIXED_REF = ROOT / "outputs/stage413/b/radar_near_contact/v3/near_set_containment_v3.json"
NEW_REF = OUT / "proxy_new_set.json"
POSES = ROOT / "outputs/stage411/operating_poses.npz"
PROXY_SCENE = ROOT / "mujoco/robots/ranger_cr10/scene_simple_v2_physproxy.xml"
V2_SCENE = ROOT / "mujoco/robots/ranger_cr10/scene_simple_v2.xml"
SEED = 20261109
LIMITS = np.array([[-3.92, 0.94], [-1.57, 1.57], [-2.86, 2.86],
                   [-3.14, 3.14], [-3.14, 3.14], [-3.14, 3.14]])
TARGET_PAIRS = [
    ("cr10_Link1", "chassis"),
    ("cr10_Link1", "cr10_base"),
    ("cr10_Link1", "box_top"),
    ("cr10_Link2", "box_left"),
    ("cr10_Link2", "d435"),
    ("cr10_Link3", "box_left"),
    ("cr10_Link3", "box_top"),
    ("cr10_Link5", "chassis"),
    ("cr10_Link6", "box_left"),
]
BANDS = (("overlap", -np.inf, 0.0), ("0-5", 0.0, 0.005), ("5-10", 0.005, 0.010),
         ("10-20", 0.010, 0.020), ("20-30", 0.020, 0.030))
LINK6_ROWS = {"cr10_Link6_main": "D0240", "cr10_Link6_ag95": "D0249",
              "cr10_Link6_d455": "D0253", "cr10_Link6_finger": "D0284"}


def set_pose(model, data, q) -> None:
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for index, value in enumerate(q):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{index + 1}")
        data.qpos[model.jnt_qposadr[joint]] = float(value)
    mujoco.mj_forward(model, data)


def band_of(distance: float) -> str | None:
    for name, low, high in BANDS:
        if low < distance <= high or (name == "overlap" and distance <= 0.0):
            return name
    return None


def main() -> None:
    eval_data = json.loads((OUT / "phys_proxy_eval.json").read_text())
    fixed_summary = eval_data["fixed"]["summary"]
    margin_fn = fixed_summary["pair_a_margin_fn"]
    tolerance_fn = fixed_summary["pair_a_tolerance_fn"]
    rule = {
        "created": "2026-11-07",
        "seed": SEED,
        "purpose": "diagnostic coverage of joint4, Link6 multi-atom, and arm-box/chassis regions; "
                   "not independent global safety evidence",
        "fixed_development_set": str(FIXED_REF.relative_to(ROOT)),
        "local_perturbation_set": str(NEW_REF.relative_to(ROOT)),
        "fixed_margin_fn_ids": margin_fn,
        "fixed_tolerance_fn_ids": tolerance_fn,
        "link6_winner_rows": LINK6_ROWS,
        "random_uniform_count": 20000,
        "operating_poses": 1800,
        "target_pairs": [list(pair) for pair in TARGET_PAIRS],
        "bands_m": {name: [None if np.isinf(low) else float(low), float(high)]
                    for name, low, high in BANDS},
        "selection": {
            "per_pair_per_band": 1,
            "max_rows": 80,
            "min_linf_rad": 0.01,
            "prefer_band_centre": True,
            "dense_path_interpolation": False,
        },
        "note": "the fixed margin/tolerance lists are source rows, not independent evidence; "
                "the random/operating screen is independent of the A labels but shares the same FK",
    }
    RULE_OUT.write_text(json.dumps(rule, indent=1))
    print(f"wrote rule {RULE_OUT}", flush=True)

    shapes = shapes_by_group(load_shapes())
    proxy = mujoco.MjModel.from_xml_path(str(PROXY_SCENE))
    proxy_data = mujoco.MjData(proxy)
    v2 = mujoco.MjModel.from_xml_path(str(V2_SCENE))
    v2_data = mujoco.MjData(v2)

    fixed_rows = {row["row_id"]: row for row in json.loads(FIXED_REF.read_text())["rows"]}
    new_rows = json.loads(NEW_REF.read_text())["rows"]
    existing = np.array([row["q"] for row in fixed_rows.values()]
                        + [row["q"] for row in new_rows], dtype=np.float64)
    poses = np.load(POSES)["poses"]
    existing = np.vstack([existing, poses])

    selected = []
    seen_q = set()

    def add_row(row_id, arm, q, source, target_pair, note="", a_row=None):
        key = (tuple(np.round(q, 6)), tuple(target_pair))
        if key in seen_q:
            return
        seen_q.add(key)
        selected.append(dict(row_id=row_id, arm=arm, q=[float(value) for value in q],
                             source=source, pair=list(target_pair), note=note,
                             a_row=a_row))

    for row_id in margin_fn:
        row = fixed_rows[row_id]
        add_row(f"T_M_{row_id}", row["arm"], row["q"], "fixed_margin_fn",
                [row["arm"], "lidar"], "A margin unsafe, B pair safe", a_row=row)
    for label, row_id in LINK6_ROWS.items():
        row = fixed_rows[row_id]
        add_row(f"T_L6_{label}", row["arm"], row["q"], "link6_subfamily",
                [row["arm"], "lidar"], f"Link6 winner {label}", a_row=row)

    # Screen random and operating poses with the actual-shape proxy. A candidate is assigned to
    # the band of its target pair; the first candidate closest to the band centre is kept.
    pool = []
    rng = np.random.default_rng(SEED)
    for q in rng.uniform(LIMITS[:, 0], LIMITS[:, 1], size=(rule["random_uniform_count"], 6)):
        pool.append(("random_uniform", q))
    for index, q in enumerate(poses):
        pool.append((f"operating:{index}", q))

    candidates = {pair: {name: [] for name, _, _ in BANDS} for pair in TARGET_PAIRS}
    for source, q in pool:
        set_pose(proxy, proxy_data, q)
        if np.min(np.max(np.abs(existing - q), axis=1)) < rule["selection"]["min_linf_rad"]:
            continue
        for pair in TARGET_PAIRS:
            result = proxy_group_distance(proxy, proxy_data, shapes, pair[0], pair[1])
            band = band_of(result["distance"])
            if band is None:
                continue
            low, high = rule["bands_m"][band]
            centre = 0.0 if low is None else (low + high) / 2.0
            candidates[pair][band].append((abs(result["distance"] - centre), source, q.copy(),
                                           result["distance"], result["shape_a"], result["shape_b"]))

    for pair in TARGET_PAIRS:
        for band, _, _ in BANDS:
            if not candidates[pair][band]:
                continue
            candidates[pair][band].sort(key=lambda item: item[0])
            for _, source, q, distance, shape_a, shape_b in candidates[pair][band][:1]:
                add_row(f"T_{pair[0]}_{pair[1]}_{band}", pair[0], q, source, pair,
                        f"proxy {distance:.6f} winner {shape_a}|{shape_b}")
    print(f"selected {len(selected)}", flush=True)

    rows = []
    for item in selected:
        set_pose(proxy, proxy_data, item["q"])
        set_pose(v2, v2_data, item["q"])
        pair_result = proxy_group_distance(proxy, proxy_data, shapes, item["pair"][0], item["pair"][1])
        ref_pair = reference_group_distance(shapes, item["pair"][0], item["pair"][1], v2_data)
        records = []
        ref_records = []
        for first, second in active_pairs():
            records.append(dict(pair=[first, second],
                                **proxy_group_distance(proxy, proxy_data, shapes, first, second)))
            ref_records.append(dict(pair=[first, second],
                                    **reference_group_distance(shapes, first, second, v2_data)))
        proxy_global = min(records, key=lambda record: record["distance"])
        ref_global = min(ref_records, key=lambda record: record["distance"])
        def global_label(items, threshold):
            if any(item["distance"] <= max(threshold, 1e-4) for item in items):
                return "unsafe"
            return "safe" if all(np.isfinite(item["distance"]) for item in items) else "unknown"
        proxy_capped = pair_result["distance"] >= CAP - 1e-9
        ref_pair_capped = ref_pair["distance"] >= CAP
        error = (pair_result["distance"] - ref_pair["distance"]
                 if pair_result["status"] == "separated" and ref_pair["status"] == "separated"
                 and not proxy_capped and not ref_pair_capped else None)
        a_row = item.pop("a_row")
        a_tol = None if a_row is None else (
            "unsafe" if a_row.get("reference_contact") else
            "unsafe" if a_row.get("reference_status") == "exact" and a_row.get("reference_distance_m") is not None
            and float(a_row["reference_distance_m"]) <= 1e-4 else
            "safe" if a_row.get("reference_status") == "exact" and a_row.get("reference_distance_m") is not None else
            "safe" if a_row.get("reference_status") in ("culled", "certified_culled")
            and a_row.get("reference_lower_bound_m") is not None
            and float(a_row["reference_lower_bound_m"]) > 1e-4 else "unknown")
        a_margin = None if a_row is None else (
            "unsafe" if a_row.get("reference_contact") else
            "unsafe" if a_row.get("reference_status") == "exact" and a_row.get("reference_distance_m") is not None
            and float(a_row["reference_distance_m"]) <= 0.02 else
            "safe" if a_row.get("reference_status") == "exact" and a_row.get("reference_distance_m") is not None else
            "safe" if a_row.get("reference_status") in ("culled", "certified_culled")
            and a_row.get("reference_lower_bound_m") is not None
            and float(a_row["reference_lower_bound_m"]) > 0.02 else "unknown")
        rows.append(dict(item,
                         a_tol=a_tol, a_margin=a_margin,
                         a_reference=None if a_row is None else a_row.get("reference_distance_m"),
                         a_status=None if a_row is None else a_row.get("reference_status"),
                         proxy_pair_distance=pair_result["distance"],
                         proxy_pair_signed=pair_result["signed"],
                         proxy_pair_status=pair_result["status"],
                         proxy_pair_shape_a=pair_result["shape_a"],
                         proxy_pair_shape_b=pair_result["shape_b"],
                         ref_pair_distance=ref_pair["distance"],
                         ref_pair_status=ref_pair["status"],
                         ref_pair_shape_a=ref_pair["shape_a"],
                         ref_pair_shape_b=ref_pair["shape_b"],
                         proxy_global_distance=proxy_global["distance"],
                         proxy_global_pair=proxy_global["pair"],
                         proxy_global_status=proxy_global["status"],
                         ref_global_distance=ref_global["distance"],
                         ref_global_pair=ref_global["pair"],
                         ref_global_status=ref_global["status"],
                         proxy_global_tol=global_label(records, 0.0),
                         proxy_global_margin=global_label(records, MARGIN),
                         ref_global_tol=global_label(ref_records, 0.0),
                         ref_global_margin=global_label(ref_records, MARGIN),
                         active_pair=tuple(item["pair"]) in active_pairs(),
                         proxy_capped=proxy_capped,
                         error_m=error))

    result = {
        "rule": str(RULE_OUT.relative_to(ROOT)),
        "rule_sha256": __import__("hashlib").sha256(RULE_OUT.read_bytes()).hexdigest(),
        "n": len(rows),
        "rows": rows,
        "note": "diagnostic set; not an independent global safety set. A reference is only available "
                "for the arm-lidar source rows.",
    }
    SET_OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps({
        "n": len(rows),
        "by_source": {source: sum(1 for row in rows if row["source"] == source)
                      for source in sorted({row["source"] for row in rows})},
        "by_pair": {str(pair): sum(1 for row in rows if row["pair"] == list(pair))
                    for pair in TARGET_PAIRS},
        "out": str(SET_OUT),
    }, indent=1))


if __name__ == "__main__":
    main()
