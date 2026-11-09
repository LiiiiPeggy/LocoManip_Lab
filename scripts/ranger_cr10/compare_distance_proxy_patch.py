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

"""Compare old A, v2 simple proxy, and v2 + local patch boxes.

The patch boxes live only in `scene_simple_v2_proxy.xml`; the physical scene and
`ranger_cr10_simple_v2_flat.usd` are not changed. The distance for a link is the
minimum over all of its atomic geoms (main + patch), and the winning geom is kept.

Run:
    python scripts/ranger_cr10/compare_distance_proxy_patch.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
V2_SCENE = ROOT / "mujoco/robots/ranger_cr10/scene_simple_v2.xml"
PATCH_SCENE = ROOT / "mujoco/robots/ranger_cr10/scene_simple_v2_proxy.xml"
PATCH_SCENE_L3L6 = ROOT / "mujoco/robots/ranger_cr10/scene_simple_v2_proxy_l3l6.xml"
NEAR_V2 = ROOT / "outputs/stage413/b/radar_near_contact/v2/rows.csv"
NEAR_V3 = ROOT / "outputs/stage413/b/radar_near_contact/v3/near_set_containment_v3.json"
NEAR_ROWS = ROOT / "outputs/stage413/b/simple_collision_v2/near_rows_v2.csv"
NEW_SET = ROOT / "outputs/stage413/b/simple_collision_v2/proxy_new_set.json"
REPLAY = ROOT / "outputs/stage413/b/radar_replay/rows.csv"
POSES = ROOT / "outputs/stage411/operating_poses.npz"
SAFETY = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/safety.py"
OUT = ROOT / "outputs/stage413/b/simple_collision_v2/distance_proxy_patch_comparison.json"
TOL = 1e-4
MARGIN = 0.02

ARM_MAIN = {
    "cr10_Link1": ["cr10_Link1_col_simple"],
    "cr10_Link2": ["cr10_Link2_col_simple"],
    "cr10_Link3": ["cr10_Link3_col_simple"],
    "cr10_Link4": ["cr10_Link4_col_simple"],
    "cr10_Link5": ["cr10_Link5_col_simple"],
    "cr10_Link6": ["cr10_Link6_col_simple", "gripper_base_link_col0", "gripper_base_link_col1",
                   "gripper_base_link_col2", "d455_link_col"],
}
ARM_PATCH_L3 = {key: list(value) for key, value in ARM_MAIN.items()}
ARM_PATCH_L3["cr10_Link3"].append("cr10_Link3_col_patch")
ARM_PATCH = {key: list(value) for key, value in ARM_PATCH_L3.items()}
ARM_PATCH["cr10_Link6"].append("cr10_Link6_col_patch")

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--variant", choices=("l3", "l3l6"), default="l3")
ap.add_argument("--out", default=str(OUT))
args = ap.parse_args()
PROTECTED = ["base_link_col", "lidar_link0_col", "d435_link_col", "box_link_col_left", "box_link_col_top"]


def load(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i): i for i in range(model.ngeom)}
    addresses = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")]
                 for i in range(1, 7)]
    return model, data, names, addresses


MODELS = {"v2": load(V2_SCENE),
          "patch": load(PATCH_SCENE_L3L6 if args.variant == "l3l6" else PATCH_SCENE)}
ARM = {"v2": ARM_MAIN, "patch": ARM_PATCH_L3 if args.variant == "l3" else ARM_PATCH}

spec = importlib.util.spec_from_file_location("safety_proxy", SAFETY)
safety = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safety)


def set_q(model, data, addresses, q):
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for address, value in zip(addresses, q):
        data.qpos[address] = float(value)
    mujoco.mj_forward(model, data)


def best_of(entries, arm=None, protected=None):
    candidates = [
        entry for entry in entries
        if (arm is None or entry[1] == arm) and (protected is None or entry[3] in protected)
    ]
    if not candidates:
        return float("nan"), None, None, None
    return min(candidates, key=lambda entry: entry[0])


def distances(kind, q):
    model, data, names, addresses = MODELS[kind]
    set_q(model, data, addresses, q)
    out = []
    for arm, geoms in ARM[kind].items():
        for geom in geoms:
            for protected in PROTECTED:
                out.append((float(mujoco.mj_geomDistance(model, data, names[geom], names[protected], 0.2, None)),
                            arm, geom, protected))
    return out


def old_a_distances(q):
    """Old safety.py box proxy, global minimum, on the v2 kinematic model."""
    model, data, names, addresses = MODELS["v2"]
    set_q(model, data, addresses, q)
    bodies = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)}

    def world(body_name, center, quat):
        body = bodies[body_name]
        rotation = data.xmat[body].reshape(3, 3)
        position = data.xpos[body]
        world_center = torch.tensor(position + rotation @ np.asarray(center), dtype=torch.float32)
        body_quat = torch.tensor(data.xquat[body], dtype=torch.float32)
        local_quat = torch.tensor(quat if quat else [1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        return world_center, safety._quat_mul(body_quat, local_quat)

    arms = []
    for group, entries in safety.ARM_PARTS.items():
        for _, center, half, quat in entries:
            c, quat_world = world(group, center, quat)
            arms.append((c, quat_world, torch.tensor(half, dtype=torch.float32)))
    protected = []
    for _, (body, entries) in safety.PROTECTED_PARTS.items():
        for _, center, half, quat in entries:
            c, quat_world = world(body, center, quat)
            protected.append((c, quat_world, torch.tensor(half, dtype=torch.float32)))
    return min(float(safety._exact_distance(ca, qa, ha, cb, qb, hb))
               for ca, qa, ha in arms for cb, qb, hb in protected)


def label(distance, threshold):
    if not np.isfinite(distance):
        return "unknown"
    return "unsafe" if distance <= max(threshold, TOL) else "safe"


def ref_labels(row, from_new):
    contact = bool(row.get("reference_contact", False))
    status = row.get("reference_status")
    distance = row.get("reference_distance_m")
    lower = row.get("reference_lower_bound_m")
    if contact:
        return "unsafe", "unsafe"
    if status == "exact" and distance is not None:
        value = float(distance)
        return ("unsafe" if value <= TOL else "safe", "unsafe" if value <= MARGIN else "safe")
    if status in ("culled", "certified_culled") and lower is not None:
        value = float(lower)
        return ("safe" if value > TOL else "unknown", "safe" if value > MARGIN else "unknown")
    return "unknown", "unknown"


def confusion(rows, key_ref, key_candidate):
    valid = [row for row in rows if row[key_ref] in ("unsafe", "safe") and row[key_candidate] in ("unsafe", "safe")]
    tp = sum(1 for row in valid if row[key_ref] == "unsafe" and row[key_candidate] == "unsafe")
    fp = sum(1 for row in valid if row[key_ref] == "safe" and row[key_candidate] == "unsafe")
    fn = sum(1 for row in valid if row[key_ref] == "unsafe" and row[key_candidate] == "safe")
    tn = sum(1 for row in valid if row[key_ref] == "safe" and row[key_candidate] == "safe")
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, valid=len(valid), unknown=len(rows) - len(valid),
                positives=sum(1 for row in valid if row[key_ref] == "unsafe"),
                miss_rate=(fn / (tp + fn)) if (tp + fn) else None)


def evaluate_rows(rows, source):
    out = []
    for row in rows:
        entries_v2 = distances("v2", row["q"])
        entries_patch = distances("patch", row["q"])
        v2_global = best_of(entries_v2)
        patch_global = best_of(entries_patch)
        arm = row["arm"]
        v2_row = best_of(entries_v2, arm=arm, protected=["lidar_link0_col"])
        patch_row = best_of(entries_patch, arm=arm, protected=["lidar_link0_col"])
        ref_tol, ref_margin = ref_labels(row, source == "new")
        old_a = float(row["old_m"]) if row.get("old_m") not in (None, "", "None") else old_a_distances(row["q"])
        out.append(dict(
            row_id=row["row_id"], arm=arm, source=source,
            old_a_m=old_a,
            v2_global_m=v2_global[0], v2_global_winner="|".join(str(x) for x in v2_global[1:] if x),
            patch_global_m=patch_global[0], patch_global_winner="|".join(str(x) for x in patch_global[1:] if x),
            v2_row_m=v2_row[0], v2_row_winner="|".join(str(x) for x in v2_row[1:] if x),
            patch_row_m=patch_row[0], patch_row_winner="|".join(str(x) for x in patch_row[1:] if x),
            ref_tol=ref_tol, ref_margin=ref_margin,
            old_tol=label(old_a, 0.0), old_margin=label(old_a, MARGIN),
            v2_tol=label(v2_global[0], 0.0), v2_margin=label(v2_global[0], MARGIN),
            patch_tol=label(patch_global[0], 0.0), patch_margin=label(patch_global[0], MARGIN),
            v2_row_tol=label(v2_row[0], 0.0), v2_row_margin=label(v2_row[0], MARGIN),
            patch_row_tol=label(patch_row[0], 0.0), patch_row_margin=label(patch_row[0], MARGIN),
        ))
    return out


def summarize(rows, prefix):
    return {
        "n": len(rows),
        "global_tolerance": {
            "old": confusion(rows, "ref_tol", "old_tol"),
            "v2": confusion(rows, "ref_tol", "v2_tol"),
            "patch": confusion(rows, "ref_tol", "patch_tol"),
        },
        "global_margin": {
            "old": confusion(rows, "ref_margin", "old_margin"),
            "v2": confusion(rows, "ref_margin", "v2_margin"),
            "patch": confusion(rows, "ref_margin", "patch_margin"),
        },
        "row_tolerance": {
            "v2": confusion(rows, "ref_tol", "v2_row_tol"),
            "patch": confusion(rows, "ref_tol", "patch_row_tol"),
        },
        "row_margin": {
            "v2": confusion(rows, "ref_margin", "v2_row_margin"),
            "patch": confusion(rows, "ref_margin", "patch_row_margin"),
        },
        "non_finite": {
            "v2": sum(1 for row in rows if not np.isfinite(row["v2_global_m"])),
            "patch": sum(1 for row in rows if not np.isfinite(row["patch_global_m"])),
            "old": sum(1 for row in rows if not np.isfinite(row["old_a_m"])),
        },
    }


def lists(rows):
    def ids(pred):
        return sorted(row["row_id"] for row in rows if pred(row))

    return {
        "v2_row_tolerance_fn": ids(lambda row: row["ref_tol"] == "unsafe" and row["v2_row_tol"] == "safe"),
        "patch_row_tolerance_fn": ids(lambda row: row["ref_tol"] == "unsafe" and row["patch_row_tol"] == "safe"),
        "v2_row_margin_fn": ids(lambda row: row["ref_margin"] == "unsafe" and row["v2_row_margin"] == "safe"),
        "patch_row_margin_fn": ids(lambda row: row["ref_margin"] == "unsafe" and row["patch_row_margin"] == "safe"),
        "patch_added_row_tolerance_fp": ids(lambda row: row["ref_tol"] == "safe"
                                           and row["v2_row_tol"] == "safe" and row["patch_row_tol"] == "unsafe"),
        "patch_removed_row_tolerance_fp": ids(lambda row: row["ref_tol"] == "safe"
                                             and row["v2_row_tol"] == "unsafe" and row["patch_row_tol"] == "safe"),
        "patch_added_row_margin_fp": ids(lambda row: row["ref_margin"] == "safe"
                                        and row["v2_row_margin"] == "safe" and row["patch_row_margin"] == "unsafe"),
        "patch_removed_row_margin_fp": ids(lambda row: row["ref_margin"] == "safe"
                                          and row["v2_row_margin"] == "unsafe" and row["patch_row_margin"] == "safe"),
        "global_margin_fn": ids(lambda row: row["ref_margin"] == "unsafe" and row["patch_margin"] == "safe"),
        "global_margin_fn_old_ids": ids(lambda row: row["ref_margin"] == "unsafe" and row["old_margin"] == "safe"),
    }


def main():
    fixed_csv = {row["row_id"]: row for row in csv.DictReader(open(NEAR_ROWS))}
    fixed_ref = {row["row_id"]: row for row in json.loads(NEAR_V3.read_text())["rows"]}
    fixed_rows = []
    for row_id, ref in fixed_ref.items():
        row = dict(ref)
        row["old_m"] = fixed_csv[row_id]["old_m"]
        fixed_rows.append(row)
    fixed_eval = evaluate_rows(fixed_rows, "fixed")

    new = json.loads(NEW_SET.read_text())["rows"]
    new_eval = evaluate_rows(new, "new")

    replay = list(csv.DictReader(open(REPLAY)))
    poses = np.load(POSES)["poses"]
    replay_rows = []
    for row in replay:
        q = poses[int(row["pose"])]
        v2 = best_of(distances("v2", q))[0]
        patch = best_of(distances("patch", q))[0]
        replay_rows.append(dict(pose=int(row["pose"]), arm=row["arm"], old_a_m=float(row["A_distance_m"]),
                                v2_global_m=v2, patch_global_m=patch,
                                old_alarm=label(float(row["A_distance_m"]), MARGIN),
                                v2_alarm=label(v2, MARGIN), patch_alarm=label(patch, MARGIN)))

    fixed_summary = summarize(fixed_eval, "fixed")
    new_summary = summarize(new_eval, "new")
    result = {
        "fixed": fixed_summary,
        "fixed_lists": lists(fixed_eval),
        "fixed_p0021_p0024": [row for row in fixed_eval if row["row_id"] in ("P0021", "P0022", "P0023", "P0024")],
        "fixed_margin_global_rows": [row for row in fixed_eval
                                     if row["row_id"] in ("D0106", "D0109", "D0157", "D0264")],
        "new": new_summary,
        "new_lists": lists(new_eval),
        "replay": {
            "n": len(replay_rows),
            "old_alarms": sum(1 for row in replay_rows if row["old_alarm"] == "unsafe"),
            "v2_alarms": sum(1 for row in replay_rows if row["v2_alarm"] == "unsafe"),
            "patch_alarms": sum(1 for row in replay_rows if row["patch_alarm"] == "unsafe"),
            "v2_removed": sum(1 for row in replay_rows if row["old_alarm"] == "unsafe" and row["v2_alarm"] == "safe"),
            "v2_added": sum(1 for row in replay_rows if row["old_alarm"] == "safe" and row["v2_alarm"] == "unsafe"),
            "patch_added_vs_v2": sum(1 for row in replay_rows if row["v2_alarm"] == "safe" and row["patch_alarm"] == "unsafe"),
            "note": "replay has no per-pose ground truth; these are proxy alarm differences",
        },
        "patch_variant": args.variant,
        "patch_model": str((PATCH_SCENE_L3L6 if args.variant == "l3l6" else PATCH_SCENE).relative_to(ROOT)),
        "physical_asset_unchanged": True,
        "note": "patch boxes are proxy-only; the physical USD/URDF is not modified. Row distance is the union "
                "minimum over main+patch geoms for the row's arm; global is the minimum over all arms.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    for name, rows in (("fixed", fixed_eval), ("new", new_eval)):
        fields = list(rows[0].keys())
        with (out.parent / f"{name}_patch_rows.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({k: result[k] for k in ("fixed", "new", "replay")}, indent=1)[:12000])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
