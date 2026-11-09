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

"""Evaluate the PhysX-consistent distance proxy against two separate references.

A. the original mesh reference already frozen for the 342 fixed rows, 71 new rows and the
   10800 arm-lidar pair-steps of the 1800 operating poses;
B. the frozen PhysX collider shapes, queried exactly with GJK and represented by the MuJoCo
   proxy model with ``mj_geomDistance``.

The two references are never merged: a result against A is not a result against B. Global labels
are only computed where every active logical pair has a valid reference result; the arm-lidar-only
A reference therefore yields ``unknown`` for global-safe rows instead of a fabricated FP/TN.

Run:
    python scripts/ranger_cr10/evaluate_phys_proxy.py
"""

from __future__ import annotations

import csv
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
    TOL,
    active_pairs,
    label_from_distance,
    load_shapes,
    proxy_group_distance,
    reference_group_distance,
    shapes_by_group,
)

OUT = ROOT / "outputs/stage413/b/simple_collision_v2"
FIXED_REF = ROOT / "outputs/stage413/b/radar_near_contact/v3/near_set_containment_v3.json"
NEW_REF = OUT / "proxy_new_set.json"
REPLAY_ROWS = ROOT / "outputs/stage413/b/radar_replay/rows.csv"
POSES = ROOT / "outputs/stage411/operating_poses.npz"
PROXY_SCENE = ROOT / "mujoco/robots/ranger_cr10/scene_simple_v2_physproxy.xml"
V2_SCENE = ROOT / "mujoco/robots/ranger_cr10/scene_simple_v2.xml"


def set_pose(model, data, q) -> None:
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for index, value in enumerate(q):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{index + 1}")
        data.qpos[model.jnt_qposadr[joint]] = float(value)
    mujoco.mj_forward(model, data)


def a_label(row: dict, threshold: float) -> str:
    if bool(row.get("reference_contact", False)):
        return "unsafe"
    status = row.get("reference_status")
    distance = row.get("reference_distance_m")
    lower = row.get("reference_lower_bound_m")
    if status == "exact" and distance is not None:
        return "unsafe" if float(distance) <= max(threshold, TOL) else "safe"
    if status in ("culled", "certified_culled") and lower is not None:
        return "safe" if float(lower) > threshold else "unknown"
    return "unknown"


def b_label(distance: float, status: str, threshold: float) -> str:
    return label_from_distance(distance, threshold, status)


def all_reference(shapes, data):
    records = []
    for first, second in active_pairs():
        result = reference_group_distance(shapes, first, second, data)
        records.append(dict(pair=[first, second], **result))
    return records


def all_proxy(model, data, shapes):
    records = []
    for first, second in active_pairs():
        result = proxy_group_distance(model, data, shapes, first, second)
        records.append(dict(pair=[first, second], **result))
    return records


def label_records(records, threshold: float):
    unsafe = False
    unknown = False
    labelled = []
    for record in records:
        label = b_label(record["distance"], record["status"], threshold)
        unsafe = unsafe or label == "unsafe"
        unknown = unknown or label == "unknown"
        labelled.append(dict(record, label=label))
    label = "unsafe" if unsafe else ("unknown" if unknown else "safe")
    winner = min(labelled, key=lambda item: item["distance"])
    return dict(label=label, winner=winner, pairs=labelled)


def pair_record(records, first: str, second: str):
    for record in records:
        if record["pair"] == [first, second] or record["pair"] == [second, first]:
            return record
    raise KeyError((first, second))


def row_record(row_id: str, arm: str, q, a_row: dict, ref_model, ref_data, proxy_model, proxy_data,
               shapes) -> dict:
    set_pose(ref_model, ref_data, q)
    set_pose(proxy_model, proxy_data, q)
    ref_records = all_reference(shapes, ref_data)
    proxy_records = all_proxy(proxy_model, proxy_data, shapes)
    ref_pair = pair_record(ref_records, arm, "lidar")
    proxy_pair = pair_record(proxy_records, arm, "lidar")
    ref_global = label_records(ref_records, MARGIN)
    proxy_global = label_records(proxy_records, MARGIN)
    ref_global_tol = label_records(ref_records, 0.0)
    proxy_global_tol = label_records(proxy_records, 0.0)
    a_tol = a_label(a_row, 0.0)
    a_margin = a_label(a_row, MARGIN)
    return dict(
        row_id=row_id,
        arm=arm,
        q=[float(value) for value in q],
        a_tol=a_tol,
        a_margin=a_margin,
        a_reference=a_row.get("reference_distance_m"),
        a_status=a_row.get("reference_status"),
        a_contact=bool(a_row.get("reference_contact", False)),
        ref_pair_distance=ref_pair["distance"],
        ref_pair_status=ref_pair["status"],
        ref_pair_shape_a=ref_pair["shape_a"],
        ref_pair_shape_b=ref_pair["shape_b"],
        ref_pair_tol=b_label(ref_pair["distance"], ref_pair["status"], 0.0),
        ref_pair_margin=b_label(ref_pair["distance"], ref_pair["status"], MARGIN),
        proxy_pair_distance=proxy_pair["distance"],
        proxy_pair_signed=proxy_pair["signed"],
        proxy_pair_status=proxy_pair["status"],
        proxy_pair_shape_a=proxy_pair["shape_a"],
        proxy_pair_shape_b=proxy_pair["shape_b"],
        proxy_pair_tol=b_label(proxy_pair["distance"], proxy_pair["status"], 0.0),
        proxy_pair_margin=b_label(proxy_pair["distance"], proxy_pair["status"], MARGIN),
        ref_global_tol=ref_global_tol["label"],
        ref_global_margin=ref_global["label"],
        proxy_global_tol=proxy_global_tol["label"],
        proxy_global_margin=proxy_global["label"],
        ref_global_margin_winner=ref_global["winner"],
        proxy_global_margin_winner=proxy_global["winner"],
    )


def confusion(rows, ref_key, pred_key):
    valid = [row for row in rows if row[ref_key] in ("safe", "unsafe") and row[pred_key] in ("safe", "unsafe")]
    tp = sum(1 for row in valid if row[ref_key] == "unsafe" and row[pred_key] == "unsafe")
    fp = sum(1 for row in valid if row[ref_key] == "safe" and row[pred_key] == "unsafe")
    fn = sum(1 for row in valid if row[ref_key] == "unsafe" and row[pred_key] == "safe")
    tn = sum(1 for row in valid if row[ref_key] == "safe" and row[pred_key] == "safe")
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, valid=len(valid), unknown=len(rows) - len(valid),
                miss_rate=(fn / (tp + fn)) if (tp + fn) else None)


def ids(rows, ref_key, pred_key, ref_value, pred_value):
    return sorted(row["row_id"] for row in rows if row[ref_key] == ref_value and row[pred_key] == pred_value)


def distance_stats(rows):
    values = [
        row["proxy_pair_distance"] - row["ref_pair_distance"]
        for row in rows
        if row["ref_pair_status"] == "separated" and row["proxy_pair_status"] == "separated"
        and np.isfinite(row["ref_pair_distance"]) and np.isfinite(row["proxy_pair_distance"])
        and row["ref_pair_distance"] < CAP and row["proxy_pair_distance"] < CAP
    ]
    if not values:
        return dict(n=0)
    array = np.asarray(values)
    return dict(n=len(array), median_m=float(np.median(array)), p95_abs_m=float(np.percentile(np.abs(array), 95)),
                max_abs_m=float(np.max(np.abs(array))), min_m=float(np.min(array)), max_m=float(np.max(array)))


def summarize(rows):
    return dict(
        n=len(rows),
        pair_b_tolerance=confusion(rows, "ref_pair_tol", "proxy_pair_tol"),
        pair_b_margin=confusion(rows, "ref_pair_margin", "proxy_pair_margin"),
        pair_a_tolerance=confusion(rows, "a_tol", "proxy_pair_tol"),
        pair_a_margin=confusion(rows, "a_margin", "proxy_pair_margin"),
        global_b_tolerance=confusion(rows, "ref_global_tol", "proxy_global_tol"),
        global_b_margin=confusion(rows, "ref_global_margin", "proxy_global_margin"),
        pair_b_tolerance_fn=ids(rows, "ref_pair_tol", "proxy_pair_tol", "unsafe", "safe"),
        pair_b_tolerance_fp=ids(rows, "ref_pair_tol", "proxy_pair_tol", "safe", "unsafe"),
        pair_b_margin_fn=ids(rows, "ref_pair_margin", "proxy_pair_margin", "unsafe", "safe"),
        pair_b_margin_fp=ids(rows, "ref_pair_margin", "proxy_pair_margin", "safe", "unsafe"),
        pair_a_tolerance_fn=ids(rows, "a_tol", "proxy_pair_tol", "unsafe", "safe"),
        pair_a_tolerance_fp=ids(rows, "a_tol", "proxy_pair_tol", "safe", "unsafe"),
        pair_a_margin_fn=ids(rows, "a_margin", "proxy_pair_margin", "unsafe", "safe"),
        pair_a_margin_fp=ids(rows, "a_margin", "proxy_pair_margin", "safe", "unsafe"),
        global_b_tolerance_fn=ids(rows, "ref_global_tol", "proxy_global_tol", "unsafe", "safe"),
        global_b_tolerance_fp=ids(rows, "ref_global_tol", "proxy_global_tol", "safe", "unsafe"),
        global_b_margin_fn=ids(rows, "ref_global_margin", "proxy_global_margin", "unsafe", "safe"),
        global_b_margin_fp=ids(rows, "ref_global_margin", "proxy_global_margin", "safe", "unsafe"),
        distance_error=distance_stats(rows),
        a_global_status={
            "confirmed_unsafe": sum(1 for row in rows if row["a_tol"] == "unsafe"),
            "unknown_global": sum(1 for row in rows if row["a_tol"] != "unsafe"),
            "note": "A covers only arm-vs-lidar; global-safe cannot be claimed from it",
        },
        b_global_status={
            "unsafe": sum(1 for row in rows if row["ref_global_margin"] == "unsafe"),
            "safe": sum(1 for row in rows if row["ref_global_margin"] == "safe"),
            "unknown": sum(1 for row in rows if row["ref_global_margin"] == "unknown"),
        },
    )


def load_json_rows(path: Path):
    return json.loads(path.read_text())["rows"]


def main() -> None:
    shapes = shapes_by_group(load_shapes())
    ref_model = mujoco.MjModel.from_xml_path(str(V2_SCENE))
    ref_data = mujoco.MjData(ref_model)
    proxy_model = mujoco.MjModel.from_xml_path(str(PROXY_SCENE))
    proxy_data = mujoco.MjData(proxy_model)

    fixed_rows = []
    for row in load_json_rows(FIXED_REF):
        fixed_rows.append(row_record(row["row_id"], row["arm"], row["q"], row, ref_model, ref_data,
                                     proxy_model, proxy_data, shapes))
    print(f"fixed {len(fixed_rows)}", flush=True)

    new_ref = json.loads(NEW_REF.read_text())
    new_rows = []
    for row in new_ref["rows"]:
        new_rows.append(row_record(row["row_id"], row["arm"], row["q"], row, ref_model, ref_data,
                                   proxy_model, proxy_data, shapes))
    print(f"new {len(new_rows)}", flush=True)

    poses = np.load(POSES)["poses"]
    replay = {(int(row["pose"]), row["arm"]): row for row in csv.DictReader(open(REPLAY_ROWS))}
    replay_rows = []
    for pose_index in range(len(poses)):
        q = poses[pose_index]
        set_pose(ref_model, ref_data, q)
        set_pose(proxy_model, proxy_data, q)
        ref_records = all_reference(shapes, ref_data)
        proxy_records = all_proxy(proxy_model, proxy_data, shapes)
        ref_global = label_records(ref_records, MARGIN)
        proxy_global = label_records(proxy_records, MARGIN)
        ref_global_tol = label_records(ref_records, 0.0)
        proxy_global_tol = label_records(proxy_records, 0.0)
        for arm in ("cr10_Link1", "cr10_Link2", "cr10_Link3", "cr10_Link4", "cr10_Link5", "cr10_Link6"):
            pair_row = replay.get((pose_index, arm))
            if pair_row is None:
                continue
            ref_pair = pair_record(ref_records, arm, "lidar")
            proxy_pair = pair_record(proxy_records, arm, "lidar")
            a_data = dict(
                reference_contact=pair_row["reference_contact"] == "True",
                reference_status=pair_row["reference_status"],
                reference_distance_m=float(pair_row["reference_distance_m"])
                if pair_row["reference_distance_m"] else None,
                reference_lower_bound_m=float(pair_row["reference_lower_bound_m"])
                if pair_row["reference_lower_bound_m"] else None,
            )
            replay_rows.append(dict(
                row_id=f"R{pose_index:04d}_{arm.replace('cr10_', '')}",
                pose=pose_index, arm=arm, q=[float(value) for value in q],
                a_tol=a_label(a_data, 0.0), a_margin=a_label(a_data, MARGIN),
                a_reference=pair_row["reference_distance_m"],
                a_status=pair_row["reference_status"],
                a_contact=pair_row["reference_contact"] == "True",
                ref_pair_distance=ref_pair["distance"], ref_pair_status=ref_pair["status"],
                ref_pair_shape_a=ref_pair["shape_a"], ref_pair_shape_b=ref_pair["shape_b"],
                ref_pair_tol=b_label(ref_pair["distance"], ref_pair["status"], 0.0),
                ref_pair_margin=b_label(ref_pair["distance"], ref_pair["status"], MARGIN),
                proxy_pair_distance=proxy_pair["distance"], proxy_pair_signed=proxy_pair["signed"],
                proxy_pair_status=proxy_pair["status"], proxy_pair_shape_a=proxy_pair["shape_a"],
                proxy_pair_shape_b=proxy_pair["shape_b"],
                proxy_pair_tol=b_label(proxy_pair["distance"], proxy_pair["status"], 0.0),
                proxy_pair_margin=b_label(proxy_pair["distance"], proxy_pair["status"], MARGIN),
                ref_global_tol=ref_global_tol["label"], ref_global_margin=ref_global["label"],
                proxy_global_tol=proxy_global_tol["label"], proxy_global_margin=proxy_global["label"],
                ref_global_margin_winner=ref_global["winner"],
                proxy_global_margin_winner=proxy_global["winner"],
            ))
        if pose_index % 50 == 0:
            print(f"replay {pose_index}/{len(poses)}", flush=True)
    print(f"replay {len(replay_rows)}", flush=True)

    result = dict(
        definitions=dict(
            tolerance_m=TOL,
            margin_m=MARGIN,
            pair_scope="the row's arm group vs lidar_link0; Link6 group includes Link6, AG95, D455 and fingers",
            global_scope="all active internal pairs: non-adjacent arm pairs and arm vs the six protected parts",
            a_reference="original mesh reference (arm vs lidar only; not a complete global reference)",
            b_reference="frozen PhysX collider shapes, exact GJK convex-hull query",
            global_rule="unsafe if any reference pair is unsafe; safe only if every active pair is validly safe; "
                        "otherwise unknown",
        ),
        fixed=dict(summary=summarize(fixed_rows), rows=fixed_rows),
        new=dict(summary=summarize(new_rows), rows=new_rows),
        replay=dict(summary=summarize(replay_rows), rows=replay_rows),
    )
    out = OUT / "phys_proxy_eval.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({
        "fixed": result["fixed"]["summary"],
        "new": result["new"]["summary"],
        "replay": result["replay"]["summary"],
        "out": str(out),
    }, indent=1)[:12000], flush=True)


if __name__ == "__main__":
    main()
