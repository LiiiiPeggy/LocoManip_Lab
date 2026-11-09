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

"""Compare the old safety.py A proxy with the new simple-box distance proxy (offline only)."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "mujoco/robots/ranger_cr10/scene_simple_v2.xml"
REPLAY = ROOT / "outputs/stage413/b/radar_replay/rows.csv"
POSES = ROOT / "outputs/stage411/operating_poses.npz"
NEAR_V2 = ROOT / "outputs/stage413/b/radar_near_contact/v2/rows.csv"
NEAR_V3 = ROOT / "outputs/stage413/b/radar_near_contact/v3/near_set_containment_v3.json"
OUT = ROOT / "outputs/stage413/b/simple_collision_v2/distance_proxy_comparison_v2.json"

m = mujoco.MjModel.from_xml_path(str(MODEL))
d = mujoco.MjData(m)
name = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
idx = {name(i): i for i in range(m.ngeom) if name(i)}
adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)]
pose_array = np.load(POSES)["poses"]

ARM = {
    "cr10_Link1": ["cr10_Link1_col_simple"],
    "cr10_Link2": ["cr10_Link2_col_simple"],
    "cr10_Link3": ["cr10_Link3_col_simple"],
    "cr10_Link4": ["cr10_Link4_col_simple"],
    "cr10_Link5": ["cr10_Link5_col_simple"],
    "cr10_Link6": ["cr10_Link6_col_simple", "gripper_base_link_col0", "gripper_base_link_col1",
                   "gripper_base_link_col2", "d455_link_col"],
}
PROTECTED = ["base_link_col", "lidar_link0_col", "d435_link_col", "box_link_col_left", "box_link_col_top"]


def new_distances(q):
    """All arm-geom vs protected-geom distances at one pose.

    One entry per atomic geom: the union minimum per link must be taken explicitly,
    otherwise the last geom in the list silently wins and a Link6 main-box overlap
    can be replaced by the d455 distance.
    """
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    for a, v in zip(adr, q):
        d.qpos[a] = float(v)
    mujoco.mj_forward(m, d)
    out = []
    for arm, arm_geoms in ARM.items():
        for ga in arm_geoms:
            for gb in PROTECTED:
                out.append((float(mujoco.mj_geomDistance(m, d, idx[ga], idx[gb], 0.15, None)), arm, ga, gb))
    return out


def best_of(distances, arms=None, protected=None):
    candidates = [
        item
        for item in distances
        if (arms is None or item[1] in arms) and (protected is None or item[3] in protected)
    ]
    if not candidates:
        return float("nan"), None, None, None
    value, arm, geom, protected_geom = min(candidates, key=lambda item: item[0])
    return value, arm, geom, protected_geom


def label(distance, threshold):
    if distance <= max(threshold, 1e-4):
        return "unsafe"
    if np.isfinite(distance):
        return "safe"
    return "unknown"


def main():
    replay = list(csv.DictReader(open(REPLAY)))
    old_replay = [float(r["A_distance_m"]) for r in replay]
    new_replay = [best_of(new_distances(pose_array[int(r["pose"])]))[0] for r in replay]
    old_alarm = [label(x, 0.02) for x in old_replay]
    new_alarm = [label(x, 0.02) for x in new_replay]
    replay_rows = []
    for r, old, new, new_m in zip(replay, old_alarm, new_alarm, new_replay):
        replay_rows.append(dict(pose=int(r["pose"]), arm=r["arm"], old_m=float(r["A_distance_m"]),
                                new_m=float(new_m), old_label=old, new_label=new,
                                reference=r["reference_status"]))
    # near set
    v2 = {r["row_id"]: r for r in csv.DictReader(open(NEAR_V2))}
    near_ref = {r["row_id"]: r for r in json.loads(NEAR_V3.read_text())["rows"]}
    near_rows = []
    for row_id, ref in near_ref.items():
        q = json.loads(v2[row_id]["q"])
        old = float(v2[row_id]["A_distance_m"])
        distances = new_distances(q)
        new, new_arm, new_geom, new_protected = best_of(distances)
        new_row, row_arm, row_geom, row_protected = best_of(
            distances, arms=[ref["arm"]], protected=["lidar_link0_col"]
        )
        ref_dist = ref.get("reference_distance_m")
        near_rows.append(dict(
            row_id=row_id, arm=ref["arm"], bin=ref["bin"], old_m=old, new_m=new, new_row_m=new_row,
            new_winner=f"{new_arm}|{new_geom}|{new_protected}" if new_arm else None,
            new_row_winner=f"{row_arm}|{row_geom}|{row_protected}" if row_arm else None,
            old_tol=label(old, 0.0), new_tol=label(new, 0.0),
            new_row_tol=label(new_row, 0.0),
            old_margin=label(old, 0.02), new_margin=label(new, 0.02),
            new_row_margin=label(new_row, 0.02),
            ref_status=ref["reference_status"], ref_contact=ref["reference_contact"],
            ref_distance=ref_dist, ref_lower=ref.get("reference_lower_bound_m"),
        ))
    # reference tolerance labels
    def ref_tol(row):
        if row["ref_contact"]:
            return "unsafe"
        if row["ref_status"] == "exact" and row["ref_distance"] is not None:
            return "unsafe" if row["ref_distance"] <= 1e-4 else "safe"
        if row["ref_status"] == "certified_culled" and row["ref_lower"] is not None:
            return "safe" if row["ref_lower"] > 1e-4 else "unknown"
        return "unknown"

    def ref_margin(row):
        if row["ref_contact"]:
            return "unsafe"
        if row["ref_status"] == "exact" and row["ref_distance"] is not None:
            return "unsafe" if row["ref_distance"] <= 0.02 else "safe"
        if row["ref_status"] == "certified_culled" and row["ref_lower"] is not None:
            return "safe" if row["ref_lower"] > 0.02 else "unknown"
        return "unknown"

    def confusion(rows, ref_fn, key):
        valid = [r for r in rows if ref_fn(r) in ("safe", "unsafe") and r[key] in ("safe", "unsafe")]
        tp = sum(1 for r in valid if ref_fn(r) == "unsafe" and r[key] == "unsafe")
        fp = sum(1 for r in valid if ref_fn(r) == "safe" and r[key] == "unsafe")
        fn = sum(1 for r in valid if ref_fn(r) == "unsafe" and r[key] == "safe")
        tn = sum(1 for r in valid if ref_fn(r) == "safe" and r[key] == "safe")
        return dict(TP=tp, FP=fp, FN=fn, TN=tn, valid=len(valid), unknown=len(rows) - len(valid),
                    miss_rate=(fn / (tp + fn)) if (tp + fn) else None)

    near = dict(n=len(near_rows), old_tolerance=confusion(near_rows, ref_tol, "old_tol"),
                new_tolerance=confusion(near_rows, ref_tol, "new_tol"),
                new_row_tolerance=confusion(near_rows, ref_tol, "new_row_tol"),
                old_margin=confusion(near_rows, ref_margin, "old_margin"),
                new_margin=confusion(near_rows, ref_margin, "new_margin"),
                new_row_margin=confusion(near_rows, ref_margin, "new_row_margin"),
                new_tolerance_misses=[r for r in near_rows if ref_tol(r) == "unsafe" and r["new_tol"] == "safe"],
                new_row_tolerance_misses=[
                    r for r in near_rows if ref_tol(r) == "unsafe" and r["new_row_tol"] == "safe"
                ],
                old_tolerance_misses=[r for r in near_rows if ref_tol(r) == "unsafe" and r["old_tol"] == "safe"],
                new_margin_misses=[
                    dict(r, ref_margin=ref_margin(r))
                    for r in near_rows
                    if ref_margin(r) == "unsafe" and r["new_margin"] == "safe"
                ],
                new_row_margin_misses=[
                    dict(r, ref_margin=ref_margin(r))
                    for r in near_rows
                    if ref_margin(r) == "unsafe" and r["new_row_margin"] == "safe"
                ],
                old_margin_misses=[
                    dict(r, ref_margin=ref_margin(r))
                    for r in near_rows
                    if ref_margin(r) == "unsafe" and r["old_margin"] == "safe"
                ],
                new_false=[r for r in near_rows if ref_tol(r) == "safe" and r["new_tol"] == "unsafe"],
                new_row_false=[r for r in near_rows if ref_tol(r) == "safe" and r["new_row_tol"] == "unsafe"],
                removed_false=[r for r in near_rows if ref_tol(r) == "safe" and r["old_tol"] == "unsafe" and r["new_tol"] == "safe"],
                added_false=[r for r in near_rows if ref_tol(r) == "safe" and r["old_tol"] == "safe" and r["new_tol"] == "unsafe"],
                tolerance_fn_ids=sorted(
                    r["row_id"] for r in near_rows if ref_tol(r) == "unsafe" and r["new_tol"] == "safe"
                ),
                row_tolerance_fn_ids=sorted(
                    r["row_id"] for r in near_rows if ref_tol(r) == "unsafe" and r["new_row_tol"] == "safe"
                ),
                margin_fn_ids=sorted(
                    r["row_id"] for r in near_rows if ref_margin(r) == "unsafe" and r["new_margin"] == "safe"
                ),
                row_margin_fn_ids=sorted(
                    r["row_id"] for r in near_rows if ref_margin(r) == "unsafe" and r["new_row_margin"] == "safe"
                ),
                margin_fn_details=[
                    {
                        "row_id": r["row_id"], "arm": r["arm"], "bin": r["bin"],
                        "ref_margin": ref_margin(r), "ref_status": r["ref_status"],
                        "ref_distance": r["ref_distance"], "ref_lower": r["ref_lower"],
                        "old_m": r["old_m"], "new_m": r["new_m"], "new_row_m": r["new_row_m"],
                        "new_row_winner": r["new_row_winner"], "new_winner": r["new_winner"],
                    }
                    for r in near_rows
                    if ref_margin(r) == "unsafe" and r["new_margin"] == "safe"
                ],
                p0021_p0024=[
                    dict(r, ref_tol=ref_tol(r), ref_margin=ref_margin(r))
                    for r in near_rows
                    if r["row_id"] in ("P0021", "P0022", "P0023", "P0024")
                ],
                non_finite=dict(
                    old=sum(1 for r in near_rows if not np.isfinite(r["old_m"])),
                    new=sum(1 for r in near_rows if not np.isfinite(r["new_m"])),
                ))
    replay_summary = dict(
        n=len(replay_rows),
        old_margin_alarms=sum(1 for r in replay_rows if r["old_label"] == "unsafe"),
        new_margin_alarms=sum(1 for r in replay_rows if r["new_label"] == "unsafe"),
        removed_alarms=sum(1 for r in replay_rows if r["old_label"] == "unsafe" and r["new_label"] == "safe"),
        added_alarms=sum(1 for r in replay_rows if r["old_label"] == "safe" and r["new_label"] == "unsafe"),
        note="the 1800-pose replay has no per-pose ground-truth distance; these are proxy-to-proxy alarm "
             "differences, not verified false alarms. Only the near set carries reference labels.",
    )
    out = dict(replay=replay_summary, near=near,
               note="offline proxy comparison only; safety.py is unchanged; new proxy is not used for reward; "
                    "box proxies do not provide a strict surface-intersection predicate")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    near_fields = [
        "row_id", "arm", "bin", "ref_status", "ref_contact", "ref_distance", "ref_lower",
        "old_m", "new_m", "new_row_m", "new_winner", "new_row_winner",
        "old_tol", "new_tol", "new_row_tol", "old_margin", "new_margin", "new_row_margin",
        "ref_tol", "ref_margin",
    ]
    with (OUT.parent / "near_rows_v2.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=near_fields)
        writer.writeheader()
        for row in near_rows:
            writer.writerow(
                {
                    "row_id": row["row_id"], "arm": row["arm"], "bin": row["bin"],
                    "ref_status": row["ref_status"], "ref_contact": row["ref_contact"],
                    "ref_distance": row["ref_distance"], "ref_lower": row["ref_lower"],
                    "old_m": row["old_m"], "new_m": row["new_m"], "new_row_m": row["new_row_m"],
                    "new_winner": row["new_winner"], "new_row_winner": row["new_row_winner"],
                    "old_tol": row["old_tol"], "new_tol": row["new_tol"], "new_row_tol": row["new_row_tol"],
                    "old_margin": row["old_margin"], "new_margin": row["new_margin"],
                    "new_row_margin": row["new_row_margin"],
                    "ref_tol": ref_tol(row), "ref_margin": ref_margin(row),
                }
            )
    with (OUT.parent / "replay_rows_v2.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pose", "arm", "old_m", "new_m", "old_label", "new_label", "reference"])
        writer.writeheader()
        writer.writerows(replay_rows)
    print(
        json.dumps(
            {
                "replay": replay_summary,
                "near_counts": {
                    key: near[key]
                    for key in (
                        "n", "old_tolerance", "new_tolerance", "new_row_tolerance",
                        "old_margin", "new_margin", "new_row_margin", "non_finite",
                    )
                },
                "tolerance_fn_ids": near["tolerance_fn_ids"],
                "row_tolerance_fn_ids": near["row_tolerance_fn_ids"],
                "margin_fn_ids": near["margin_fn_ids"],
                "row_margin_fn_ids": near["row_margin_fn_ids"],
                "margin_fn_details": near["margin_fn_details"],
                "p0021_p0024": [
                    {
                        "row_id": row["row_id"],
                        "ref_tol": row["ref_tol"],
                        "ref_margin": row["ref_margin"],
                        "new_m": row["new_m"],
                        "new_row_m": row["new_row_m"],
                        "new_winner": row["new_winner"],
                        "new_row_winner": row["new_row_winner"],
                        "new_tol": row["new_tol"],
                        "new_row_tol": row["new_row_tol"],
                        "new_margin": row["new_margin"],
                        "new_row_margin": row["new_row_margin"],
                    }
                    for row in near["p0021_p0024"]
                ],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
