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
MODEL = ROOT / "mujoco/robots/ranger_cr10/scene_simple.xml"
REPLAY = ROOT / "outputs/stage413/b/radar_replay/rows.csv"
POSES = ROOT / "outputs/stage411/operating_poses.npz"
NEAR_V2 = ROOT / "outputs/stage413/b/radar_near_contact/v2/rows.csv"
NEAR_V3 = ROOT / "outputs/stage413/b/radar_near_contact/v3/near_set_containment_v3.json"
OUT = ROOT / "outputs/stage413/b/simple_collision_validation/distance_proxy_comparison.json"

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
PROTECTED = ["base_link_col", "lidar_link0_col", "d435_link_col", "box_link_col_left", "box_link_col_plate"]


def new_distance(q):
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    for a, v in zip(adr, q):
        d.qpos[a] = float(v)
    mujoco.mj_forward(m, d)
    best = 1e9
    for arm_geoms in ARM.values():
        for ga in arm_geoms:
            for gb in PROTECTED:
                x = mujoco.mj_geomDistance(m, d, idx[ga], idx[gb], 0.15, None)
                if x < best:
                    best = x
    return float(best)


def label(distance, threshold):
    if distance <= max(threshold, 1e-4):
        return "unsafe"
    if np.isfinite(distance):
        return "safe"
    return "unknown"


def main():
    replay = list(csv.DictReader(open(REPLAY)))
    old_replay = [float(r["A_distance_m"]) for r in replay]
    new_replay = [new_distance(pose_array[int(r["pose"])]) for r in replay]
    old_alarm = [label(x, 0.02) for x in old_replay]
    new_alarm = [label(x, 0.02) for x in new_replay]
    replay_rows = []
    for r, old, new in zip(replay, old_alarm, new_alarm):
        replay_rows.append(dict(pose=int(r["pose"]), arm=r["arm"], old_m=float(r["A_distance_m"]),
                                new_m=new_distance(pose_array[int(r["pose"])]), old_label=old, new_label=new,
                                reference=r["reference_status"]))
    # near set
    v2 = {r["row_id"]: r for r in csv.DictReader(open(NEAR_V2))}
    near_ref = {r["row_id"]: r for r in json.loads(NEAR_V3.read_text())["rows"]}
    near_rows = []
    for row_id, ref in near_ref.items():
        q = json.loads(v2[row_id]["q"])
        old = float(v2[row_id]["A_distance_m"])
        new = new_distance(q)
        ref_dist = ref.get("reference_distance_m")
        near_rows.append(dict(row_id=row_id, arm=ref["arm"], bin=ref["bin"], old_m=old, new_m=new,
                              old_tol=label(old, 0.0), new_tol=label(new, 0.0),
                              old_margin=label(old, 0.02), new_margin=label(new, 0.02),
                              ref_status=ref["reference_status"], ref_contact=ref["reference_contact"],
                              ref_distance=ref_dist, ref_lower=ref.get("reference_lower_bound_m")))
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
                old_margin=confusion(near_rows, ref_margin, "old_margin"),
                new_margin=confusion(near_rows, ref_margin, "new_margin"),
                new_misses=[r for r in near_rows if ref_tol(r) == "unsafe" and r["new_tol"] == "safe"],
                old_misses=[r for r in near_rows if ref_tol(r) == "unsafe" and r["old_tol"] == "safe"],
                new_false=[r for r in near_rows if ref_tol(r) == "safe" and r["new_tol"] == "unsafe"],
                removed_false=[r for r in near_rows if ref_tol(r) == "safe" and r["old_tol"] == "unsafe" and r["new_tol"] == "safe"],
                added_false=[r for r in near_rows if ref_tol(r) == "safe" and r["old_tol"] == "safe" and r["new_tol"] == "unsafe"])
    replay_summary = dict(n=len(replay_rows), old_margin_fp=sum(1 for r in replay_rows if r["old_label"] == "unsafe"),
                          new_margin_fp=sum(1 for r in replay_rows if r["new_label"] == "unsafe"),
                          removed=sum(1 for r in replay_rows if r["old_label"] == "unsafe" and r["new_label"] == "safe"),
                          added=sum(1 for r in replay_rows if r["old_label"] == "safe" and r["new_label"] == "unsafe"))
    out = dict(replay=replay_summary, near=near,
               note="offline proxy comparison only; safety.py is unchanged; new proxy is not used for reward; "
                    "box proxies do not provide a strict surface-intersection predicate")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:8000])


if __name__ == "__main__":
    main()
