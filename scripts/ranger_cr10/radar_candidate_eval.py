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

"""Evaluate one replacement candidate against D on a frozen near/holdout set.

The candidate replaces one arm group's single convex hull with its own convex pieces. Every
other group keeps D. The reference labels are read from the frozen near-set rows; they are never
recomputed here. Strict surface intersection, the 1e-4 m tolerance contact and the 20 mm margin
are separate tables; the 80 strict surface overlaps and the 81 tolerance positives are kept.

Run:
    python scripts/ranger_cr10/radar_candidate_eval.py --near <near_set.json> \
        --pieces <candidate_dir> --arm cr10_Link2 --out <dir>
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from candidate_eval import load_pieces  # noqa: E402
from collision_parts import box_mesh, constituents_of  # noqa: E402
from convex_distance import gjk_distance  # noqa: E402

ROOT = HERE.parents[1]
SCENE = ROOT / "mujoco/robots/ranger_cr10/scene.xml"
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes"
ARMS = ("cr10_Link1", "cr10_Link2", "cr10_Link3", "cr10_Link4", "cr10_Link5", "cr10_Link6")
LIDAR = "lidar_link0"
CONTACT_TOL = 1e-4
MARGIN = 0.02

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--near", required=True)
ap.add_argument("--pieces", required=True)
ap.add_argument("--arm", default="cr10_Link2")
ap.add_argument("--target", choices=("arm", "lidar"), default="arm")
ap.add_argument("--out", required=True)
args = ap.parse_args()

MODEL = mujoco.MjModel.from_xml_path(str(SCENE))
DATA = mujoco.MjData(MODEL)
BID = {mujoco.mj_id2name(MODEL, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(MODEL.nbody)}
ARM_ADR = [MODEL.jnt_qposadr[mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)]


def _load(part):
    if part.kind == "mesh":
        mesh = trimesh.load(str(MESH_DIR / part.stl), force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64) * part.scale
        faces = np.asarray(mesh.faces, dtype=np.int64)
    else:
        vertices, faces = box_mesh(part.centre, part.half, part.quat)
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
    return vertices, faces


ARM_HULL = {}
for arm in ARMS:
    ARM_HULL[arm] = []
    for part in constituents_of(arm):
        vertices, faces = _load(part)
        hull = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).convex_hull
        ARM_HULL[arm].append((part.name, np.asarray(hull.vertices, dtype=np.float64)))
_lidar_v, _lidar_f = _load(constituents_of(LIDAR)[0])
LIDAR_HULL = np.asarray(trimesh.Trimesh(vertices=_lidar_v, faces=_lidar_f, process=False).convex_hull.vertices,
                        dtype=np.float64)
CANDIDATE = load_pieces("lidar_link0" if args.target == "lidar" else args.arm, ROOT / args.pieces)
if not CANDIDATE:
    raise SystemExit(f"no pieces in {args.pieces}")


def set_q(q):
    mujoco.mj_resetDataKeyframe(MODEL, DATA, MODEL.key("home").id)
    for address, value in zip(ARM_ADR, q):
        DATA.qpos[address] = float(value)
    mujoco.mj_forward(MODEL, DATA)


def body_transform(name):
    body = BID[name]
    return np.array(DATA.xmat[body], dtype=np.float64).reshape(3, 3), np.array(DATA.xpos[body], dtype=np.float64)


def gjk(points_a, points_b):
    result = gjk_distance(np.asarray(points_a), np.asarray(points_b), tol=1e-8)
    if result.status == "intersecting":
        return 0.0, True, "intersecting"
    if result.status == "separated" and np.isfinite(result.distance):
        return float(result.distance), False, "separated"
    if result.status == "degenerate" and np.isfinite(result.distance):
        if result.distance > CONTACT_TOL:
            return float(result.distance), False, "degenerate_separated"
        return 0.0, True, "degenerate_contact"
    return float("nan"), False, result.status


def group_hull(arm, lidar_hull, pieces=None):
    r_arm, p_arm = body_transform(arm)
    r_lidar, p_lidar = body_transform(LIDAR)
    lidar_world = lidar_hull @ r_lidar.T + p_lidar
    best, best_name, best_status = float("inf"), "", "clear"
    if pieces is None:
        items = [(name, hull) for name, hull in ARM_HULL[arm]]
    else:
        items = pieces
    for name, points in items:
        world = np.asarray(points) @ r_arm.T + p_arm
        distance, contact, status = gjk(world, lidar_world)
        if contact:
            return 0.0, True, name, status
        if np.isfinite(distance) and distance < best:
            best, best_name, best_status = distance, name, status
    return (best, False, best_name, best_status) if np.isfinite(best) else (float("nan"), False, best_name, "unknown")


def group_lidar(arm, lidar_pieces):
    r_arm, p_arm = body_transform(arm)
    r_lidar, p_lidar = body_transform(LIDAR)
    best, best_name, best_status = float("inf"), "", "clear"
    for arm_name, arm_hull in ARM_HULL[arm]:
        arm_world = arm_hull @ r_arm.T + p_arm
        for lidar_name, lidar_hull in lidar_pieces:
            lidar_world = np.asarray(lidar_hull) @ r_lidar.T + p_lidar
            distance, contact, status = gjk(arm_world, lidar_world)
            if contact:
                return 0.0, True, f"{arm_name}|{lidar_name}", status
            if np.isfinite(distance) and distance < best:
                best, best_name, best_status = distance, f"{arm_name}|{lidar_name}", status
    return (best, False, best_name, best_status) if np.isfinite(best) else (float("nan"), False, best_name, "unknown")


def label_class(row):
    detail = row.get("reference_detail", "")
    if "intersecting:surface_intersection" in detail:
        return "strict_surface"
    if "containment" in detail:
        return "complete_containment"
    if row["reference_status"] == "exact" and row["reference_distance_m"] not in ("", "None"):
        if float(row["reference_distance_m"]) <= CONTACT_TOL:
            return "clearance_le_0_10"
    return "clearance_gt_0_10"


def confusion(rows, label_key, positive):
    valid = [row for row in rows if row[label_key] in ("unsafe", "safe")
             and row["ref_class"] in ("strict_surface", "complete_containment",
                                      "clearance_le_0_10", "clearance_gt_0_10")]
    tp = sum(1 for row in valid if positive(row) and row[label_key] == "unsafe")
    fp = sum(1 for row in valid if not positive(row) and row[label_key] == "unsafe")
    fn = sum(1 for row in valid if positive(row) and row[label_key] == "safe")
    tn = sum(1 for row in valid if not positive(row) and row[label_key] == "safe")
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, valid=len(valid),
                positives=sum(1 for row in valid if positive(row)),
                miss_rate=(fn / (tp + fn)) if (tp + fn) else None,
                unknown=len(rows) - len(valid))


def is_strict(row):
    return row["ref_class"] == "strict_surface"


def is_tolerance(row):
    return row["ref_class"] in ("strict_surface", "complete_containment", "clearance_le_0_10")


def is_margin(row):
    if is_tolerance(row):
        return True
    if row["reference_status"] == "exact" and row["reference_distance_m"] not in ("", "None"):
        return float(row["reference_distance_m"]) <= MARGIN
    return False


def contact_labels(distance, contact, status):
    tol_unsafe = bool(contact or (np.isfinite(distance) and distance <= CONTACT_TOL))
    strict_unsafe = bool(status == "intersecting")
    if tol_unsafe:
        tol = "unsafe"
    elif np.isfinite(distance):
        tol = "safe"
    else:
        tol = "unknown"
    if strict_unsafe:
        strict = "unsafe"
    elif np.isfinite(distance):
        strict = "safe"
    else:
        strict = "unknown"
    if tol_unsafe or (np.isfinite(distance) and distance <= MARGIN):
        margin = "unsafe"
    elif np.isfinite(distance):
        margin = "safe"
    else:
        margin = "unknown"
    return strict, tol, margin


def error_stats(rows, key):
    subset = [row for row in rows if row["ref_class"] == "clearance_gt_0_10"
              and row["reference_status"] == "exact" and row["reference_distance_m"] not in ("", "None")
              and np.isfinite(row[f"{key}_distance_m"])]
    if not subset:
        return dict(n=0, abs_p95_m=None, abs_max_m=None, signed_mean_m=None, optimistic_max_m=None)
    ref = np.array([float(row["reference_distance_m"]) for row in subset])
    cand = np.array([row[f"{key}_distance_m"] for row in subset])
    error = cand - ref
    return dict(n=len(subset), abs_p95_m=float(np.percentile(np.abs(error), 95)),
                abs_max_m=float(np.max(np.abs(error))), signed_mean_m=float(np.mean(error)),
                optimistic_max_m=float(np.max(np.maximum(error, 0.0))))


def main() -> None:
    near = json.loads(Path(args.near).read_text())
    rows = near["rows"]
    for row in rows:
        set_q(row["q"])
        base = group_hull(row["arm"], LIDAR_HULL)
        if args.target == "arm":
            cand = group_hull(row["arm"], LIDAR_HULL, CANDIDATE) if row["arm"] == args.arm else base
        else:
            cand = group_lidar(row["arm"], CANDIDATE)
        row.update(
            ref_class=label_class(row),
            baseline_distance_m=base[0], baseline_contact=bool(base[1]), baseline_status=base[2],
            candidate_distance_m=cand[0], candidate_contact=bool(cand[1]), candidate_status=cand[2],
            candidate_winner=cand[2] if (args.target == "lidar" or row["arm"] == args.arm) else base[2],
        )
        for key, distance, contact, status in (("baseline", base[0], base[1], base[3]),
                                                ("candidate", cand[0], cand[1], cand[3])):
            strict_label, tol_label, margin_label = contact_labels(distance, contact, status)
            row[f"{key}_strict_label"] = strict_label
            row[f"{key}_tolerance_label"] = tol_label
            row[f"{key}_margin_label"] = margin_label
    by_pair = {}
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        if not subset:
            continue
        entry = dict(n=len(subset), bins=dict(Counter(row["bin"] for row in subset)),
                     strict_surface_rows=sum(1 for row in subset if row["ref_class"] == "strict_surface"),
                     tolerance_rows=sum(1 for row in subset if row["ref_class"] in
                                        ("strict_surface", "complete_containment", "clearance_le_0_10")),
                     overlap_rows=sum(1 for row in subset if row["ref_class"] == "strict_surface"))
        for key in ("baseline", "candidate"):
            entry[key] = dict(
                strict_surface=confusion(subset, f"{key}_strict_label", is_strict),
                tolerance_contact=confusion(subset, f"{key}_tolerance_label", is_tolerance),
                margin_20mm=confusion(subset, f"{key}_margin_label", is_margin),
                error=error_stats(subset, key),
                error_by_bin={label: error_stats([row for row in subset if row["bin"] == label], key)
                              for label in sorted({row["bin"] for row in subset})},
                strict_false_contact=sum(1 for row in subset if row[f"{key}_strict_label"] == "unsafe"
                                         and row["ref_class"] != "strict_surface"),
                tolerance_false_contact=sum(1 for row in subset if row[f"{key}_tolerance_label"] == "unsafe"
                                            and not is_tolerance(row)),
                unknown=sum(1 for row in subset if row[f"{key}_tolerance_label"] == "unknown"
                            or row[f"{key}_margin_label"] == "unknown"),
            )
            # The strict table uses the strict contact label; the tolerance table uses the same
            # contact label but a different positive set. This is intentional and recorded.
        by_pair[arm] = entry
    summary = dict(candidate=args.arm, candidate_target=args.target, pieces_dir=args.pieces, pieces=len(CANDIDATE),
                   rows=len(rows), by_pair=by_pair,
                   note="candidate replaces the named group's single hull (arm or lidar) and every other group is D; "
                        "strict surface and 1e-4 tolerance contact are separate tables; 80/81 positives preserved")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    fieldnames = ["row_id", "arm", "bin", "ref_class", "reference_status", "reference_distance_m",
                  "baseline_distance_m", "candidate_distance_m", "baseline_status", "candidate_status",
                  "candidate_winner", "baseline_strict_label", "baseline_tolerance_label", "baseline_margin_label",
                  "candidate_strict_label", "candidate_tolerance_label", "candidate_margin_label"]
    with (out / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)
    print(json.dumps(dict(candidate=args.arm, candidate_target=args.target, pieces=len(CANDIDATE), rows=len(rows),
                          by_pair={arm: dict(n=entry["n"],
                                             base_p95=entry["baseline"]["error"]["abs_p95_m"],
                                             cand_p95=entry["candidate"]["error"]["abs_p95_m"],
                                             base_strict_fp=entry["baseline"]["strict_surface"]["FP"],
                                             cand_strict_fp=entry["candidate"]["strict_surface"]["FP"])
                                   for arm, entry in by_pair.items()}), indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
