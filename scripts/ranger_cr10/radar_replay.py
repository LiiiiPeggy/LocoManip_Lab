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

"""Stage 4.13b: replay the 1800 recorded operating poses for the three radar baselines.

The three representations are the ones frozen for this round:

    A  the deployed built-in reward proxy: arm box + lidar box (``safety.py``)
    C  arm box + the existing lidar convex hull
    D  atomic arm convex hulls + the existing lidar convex hull

The lidar hull is the convex hull of the raw ``lidar_link0.STL`` vertices. The arm group is
never re-hulled as one object: Link6 keeps its five atomic constituents (wrist, gripper base,
two jaws, D455). Object names come from the safety table and the constituent table, never from
the flattened column order.

Every pair-step gets an independent reference label from the original triangle meshes:

    exact             every triangle pair that could beat the query cap was evaluated
    certified_culled  the BVH lower bounds prove distance >= cap (0.05 m)
    failed            the query did not finish or a constituent could not be certified

A cap result is a lower bound, never an exact distance. A lower bound of 0.05 m is enough to
classify both the 0 mm and 20 mm thresholds, but it is not used for an error median, a P95 or a
nearest-point claim. The run is chunked and resumable; each chunk is written atomically and the
manifest records the complete coverage count.

Run:
    python scripts/ranger_cr10/radar_replay.py --out outputs/stage413/b/radar_replay
"""

from __future__ import annotations

import os

# Keep the reference and the GJK calls from fighting over BLAS threads. The earlier benchmark
# showed a 0.397 s wall value that was thread contention, not query cost.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import csv
import hashlib
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from collision_parts import CONSTITUENTS, box_mesh, constituents_of  # noqa: E402
from convex_distance import gjk_distance  # noqa: E402
from reference_geometry import MeshRef  # noqa: E402

ROOT = HERE.parents[1]
SCENE = ROOT / "mujoco/robots/ranger_cr10/scene.xml"
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes"
SAFETY = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/safety.py"
POSES = ROOT / "outputs/stage411/operating_poses.npz"

ARMS = ("cr10_Link1", "cr10_Link2", "cr10_Link3", "cr10_Link4", "cr10_Link5", "cr10_Link6")
LIDAR = "lidar_link0"
CONTACT_TOL = 1e-4  # metres; a reference value inside this band is numerically contact
MARGIN = 0.02
CAP = 0.05
BOX_CAP = 0.15

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default="outputs/stage413/b/radar_replay")
ap.add_argument("--poses", default=str(POSES))
ap.add_argument("--chunk", type=int, default=25)
ap.add_argument("--workers", type=int, default=10)
ap.add_argument("--restart", action="store_true", help="delete this run's chunks and recompute")
args = ap.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_safety():
    spec = importlib.util.spec_from_file_location("safety_replay", SAFETY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


safety = load_safety()
MODEL = mujoco.MjModel.from_xml_path(str(SCENE))
DATA = mujoco.MjData(MODEL)
BID = {mujoco.mj_id2name(MODEL, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(MODEL.nbody)}
ARM_ADR = [MODEL.jnt_qposadr[mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)]
POSE_ARRAY = np.load(Path(args.poses))["poses"]

# Real flattened names, parallel to safety.ARM_BODY_LIST / PROT_GROUPS. The index from
# SENSOR_PAIRS is a lookup into this table, not an inferred column label.
ARM_PART_NAME = [entry[0] for entries in safety.ARM_PARTS.values() for entry in entries]
PROT_PART_NAME = [entry[0] for _, entries in safety.PROTECTED_PARTS.values() for entry in entries]

# Column census, printed and stored in the manifest.
A_COLUMNS: dict[str, list[tuple[int, str]]] = {arm: [] for arm in ARMS}
for k, (i, p) in enumerate(safety.SENSOR_PAIRS):
    if safety.PROT_GROUPS[p] != LIDAR:
        continue
    arm = safety.ARM_BODY_LIST[i]
    if arm in A_COLUMNS:
        A_COLUMNS[arm].append((len(safety.SELF_PAIRS) + k, f"{ARM_PART_NAME[i]}|{PROT_PART_NAME[p]}"))

# The A baseline itself: the same safety.clearance call the reward uses.
A_BOX = {arm: [dict(name=entry[0], vertices=box_mesh(entry[1], entry[2], entry[3])[0]) for entry in safety.ARM_PARTS[arm]]
         for arm in ARMS}
LIDAR_BOX_BODY = safety.PROTECTED_PARTS[LIDAR][0]
LIDAR_BOX = [dict(name=entry[0], vertices=box_mesh(entry[1], entry[2], entry[3])[0])
             for entry in safety.PROTECTED_PARTS[LIDAR][1]]


def _load_mesh(part):
    if part.kind == "mesh":
        mesh = trimesh.load(str(MESH_DIR / part.stl), force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64) * part.scale
        faces = np.asarray(mesh.faces, dtype=np.int64)
    else:
        vertices, faces = box_mesh(part.centre, part.half, part.quat)
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
    return vertices, faces


# Static geometry, built once before the pool forks. A worker inherits read-only copies.
ARM_REF: dict[str, list[tuple[str, MeshRef]]] = {}
ARM_HULL: dict[str, list[tuple[str, np.ndarray]]] = {}
for arm in ARMS:
    ARM_REF[arm] = []
    ARM_HULL[arm] = []
    for part in constituents_of(arm):
        vertices, faces = _load_mesh(part)
        ARM_REF[arm].append((part.name, MeshRef(vertices, faces, name=f"{arm}/{part.name}")))
        if part.kind == "mesh":
            hull = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).convex_hull.vertices
        else:
            hull = vertices
        ARM_HULL[arm].append((part.name, np.asarray(hull, dtype=np.float64)))
_lidar_part = constituents_of(LIDAR)[0]
_lidar_vertices, _lidar_faces = _load_mesh(_lidar_part)
LIDAR_REF = MeshRef(_lidar_vertices, _lidar_faces, name=LIDAR)
LIDAR_HULL = np.asarray(trimesh.Trimesh(vertices=_lidar_vertices, faces=_lidar_faces, process=False).convex_hull.vertices,
                        dtype=np.float64)


def pose_transforms(q) -> None:
    mujoco.mj_resetDataKeyframe(MODEL, DATA, MODEL.key("home").id)
    for address, value in zip(ARM_ADR, q):
        DATA.qpos[address] = float(value)
    mujoco.mj_forward(MODEL, DATA)


def body_transform(name: str):
    body = BID[name]
    return np.array(DATA.xmat[body], dtype=np.float64).reshape(3, 3), np.array(DATA.xpos[body], dtype=np.float64)


def to_world(points, rotation, translation):
    return np.asarray(points, dtype=np.float64) @ rotation.T + translation


def gjk(points_a, points_b):
    """GJK with an explicit status. Non-convergence is never turned into a safe distance."""
    result = gjk_distance(np.asarray(points_a, dtype=np.float64), np.asarray(points_b, dtype=np.float64), tol=1e-8)
    if result.status == "intersecting":
        return 0.0, True, "intersecting"
    if result.status == "separated" and np.isfinite(result.distance):
        return float(result.distance), False, "separated"
    if result.status == "degenerate" and np.isfinite(result.distance):
        # Same convention as candidate_eval: a converged near-zero support is contact, a finite
        # positive value is separated; anything else is unknown.
        if result.distance > CONTACT_TOL:
            return float(result.distance), False, "degenerate_separated"
        return 0.0, True, "degenerate_contact"
    return float("nan"), False, result.status


def safety_asset():
    names = list(dict.fromkeys(safety.ARM_BODY_LIST + safety.PROT_BODY_LIST))
    position = torch.tensor(np.stack([DATA.xpos[BID[name]] for name in names])[None], dtype=torch.float32)
    quat = torch.tensor(np.stack([DATA.xquat[BID[name]] for name in names])[None], dtype=torch.float32)

    class Asset:
        body_names = names

        class data:  # noqa: N801
            body_pos_w = position
            body_quat_w = quat

    return Asset


def a_distances():
    """A: the deployed box proxy, once per pose, min over each group's flattened boxes."""
    asset = safety_asset()
    distances = safety.clearance(asset, cap=BOX_CAP)[0].detach().cpu().numpy()
    out = {}
    for arm in ARMS:
        columns = A_COLUMNS[arm]
        if not columns:
            out[arm] = (float("nan"), False, "", "no_columns")
            continue
        index, name = min(columns, key=lambda item: float(distances[item[0]]))
        value = float(distances[index])
        capped = bool(value >= BOX_CAP - 1e-6)
        out[arm] = (value, capped, name, "capped" if capped else "box")
    return out


def c_distance(arm: str):
    """C: atomic arm boxes against the lidar convex hull."""
    r_arm, p_arm = body_transform(arm)
    r_lidar, p_lidar = body_transform(LIDAR)
    lidar_world = to_world(LIDAR_HULL, r_lidar, p_lidar)
    best = float("inf")
    best_name = ""
    best_status = "clear"
    for box in A_BOX[arm]:
        box_world = to_world(box["vertices"], r_arm, p_arm)
        distance, contact, status = gjk(box_world, lidar_world)
        if contact:
            return 0.0, True, f"{box['name']}|lidar_hull", status
        if np.isfinite(distance) and distance < best:
            best, best_name, best_status = distance, f"{box['name']}|lidar_hull", status
    if np.isfinite(best):
        return best, False, best_name, best_status
    return float("nan"), False, best_name, "unknown"


def d_distance(arm: str):
    """D: atomic arm convex hulls against the lidar convex hull; no merged group hull."""
    r_arm, p_arm = body_transform(arm)
    r_lidar, p_lidar = body_transform(LIDAR)
    lidar_world = to_world(LIDAR_HULL, r_lidar, p_lidar)
    best = float("inf")
    best_name = ""
    best_status = "clear"
    for name, hull in ARM_HULL[arm]:
        arm_world = to_world(hull, r_arm, p_arm)
        distance, contact, status = gjk(arm_world, lidar_world)
        if contact:
            return 0.0, True, f"{name}|lidar_hull", status
        if np.isfinite(distance) and distance < best:
            best, best_name, best_status = distance, f"{name}|lidar_hull", status
    if np.isfinite(best):
        return best, False, best_name, best_status
    return float("nan"), False, best_name, "unknown"


def reference(arm: str):
    """Independent mesh reference for one arm group against the lidar mesh."""
    r_arm, p_arm = body_transform(arm)
    r_lidar, p_lidar = body_transform(LIDAR)
    rotation = r_arm.T @ r_lidar
    translation = r_arm.T @ (p_lidar - p_arm)
    details = []
    exacts = []
    lowers = []
    failures = []
    contact = False
    contact_name = ""
    for name, ref in ARM_REF[arm]:
        result = ref.closest_to_in_frame(LIDAR_REF, rotation, translation, max_clearance=CAP)
        label = f"{name}|lidar"
        if result.intersects:
            contact = True
            contact_name = label
            details.append(f"{name}:intersecting")
        elif result.status == "exact":
            exacts.append((float(result.clearance), label))
            details.append(f"{name}:exact:{float(result.clearance):.9f}")
        elif result.status == "certified_culled":
            lower = float(result.extra.get("lower_bound", result.clearance))
            lowers.append((lower, label))
            details.append(f"{name}:certified_culled:{lower:.9f}")
        else:
            failures.append((result.status, label))
            details.append(f"{name}:{result.status}")
    detail = ";".join(details)
    if contact:
        return dict(status="exact", contact=True, distance=0.0, lower_bound=None, nearest=contact_name,
                    exact_count=0, culled_count=len(lowers), failed_count=len(failures), detail=detail)
    if exacts and not failures:
        distance, name = min(exacts, key=lambda item: item[0])
        return dict(status="exact", contact=False, distance=float(distance), lower_bound=None, nearest=name,
                    exact_count=len(exacts), culled_count=len(lowers), failed_count=0, detail=detail)
    if exacts and failures:
        # The exact value is only an upper bound: a failed constituent could be closer.
        distance, name = min(exacts, key=lambda item: item[0])
        return dict(status="failed", contact=False, distance=None, lower_bound=None, upper_bound=float(distance),
                    nearest=name, exact_count=len(exacts), culled_count=len(lowers), failed_count=len(failures),
                    detail=detail)
    if not exacts and not failures and lowers:
        lower, name = min(lowers, key=lambda item: item[0])
        return dict(status="certified_culled", contact=False, distance=None, lower_bound=float(lower),
                    nearest=name, exact_count=0, culled_count=len(lowers), failed_count=0, detail=detail)
    return dict(status="failed", contact=False, distance=None, lower_bound=None, nearest="",
                exact_count=len(exacts), culled_count=len(lowers), failed_count=max(1, len(failures)), detail=detail)


def reference_label(ref: dict, threshold: float) -> str:
    if ref["contact"]:
        return "unsafe"
    limit = max(threshold, CONTACT_TOL)
    if ref["status"] == "exact":
        return "unsafe" if ref["distance"] <= limit else "safe"
    if ref["status"] == "certified_culled":
        return "safe" if ref["lower_bound"] > limit else "unknown"
    return "unknown"


def candidate_label(distance: float, contact: bool, threshold: float) -> str:
    if contact or (np.isfinite(distance) and distance <= max(threshold, CONTACT_TOL)):
        return "unsafe"
    if np.isfinite(distance):
        return "safe"
    return "unknown"


def process_chunk(chunk_id: int):
    """Worker: evaluate a contiguous pose chunk and write it atomically."""
    out_dir = ROOT / args.out
    path = out_dir / "chunks" / f"chunk_{chunk_id:04d}.json"
    start = chunk_id * args.chunk
    end = min(start + args.chunk, len(POSE_ARRAY))
    rows = []
    pose_times = []
    for pose in range(start, end):
        pose_transforms(POSE_ARRAY[pose])
        a_started = time.perf_counter()
        a_values = a_distances()
        a_elapsed = time.perf_counter() - a_started
        pose_times.append(dict(pose=pose, A_seconds=a_elapsed))
        for arm in ARMS:
            started = time.perf_counter()
            ref = reference(arm)
            ref_elapsed = time.perf_counter() - started
            started = time.perf_counter()
            c_value, c_contact, c_nearest, c_status = c_distance(arm)
            c_elapsed = time.perf_counter() - started
            started = time.perf_counter()
            d_value, d_contact, d_nearest, d_status = d_distance(arm)
            d_elapsed = time.perf_counter() - started
            a_value, a_capped, a_nearest, a_status = a_values[arm]
            row = dict(
                pose=pose, arm=arm, lidar=LIDAR, pair=f"{arm} {LIDAR}",
                reference_status=ref["status"], reference_contact=bool(ref["contact"]),
                reference_distance_m=ref["distance"], reference_lower_bound_m=ref["lower_bound"],
                reference_threshold_m=CAP, reference_nearest=ref["nearest"],
                reference_exact_count=ref["exact_count"], reference_culled_count=ref["culled_count"],
                reference_failed_count=ref["failed_count"], reference_detail=ref["detail"],
                reference_contact_label=reference_label(ref, 0.0),
                reference_margin_label=reference_label(ref, MARGIN),
                A_distance_m=a_value, A_capped=bool(a_capped), A_nearest=a_nearest, A_status=a_status,
                A_contact=bool(a_value <= CONTACT_TOL), A_contact_label=candidate_label(a_value, a_value <= CONTACT_TOL, 0.0),
                A_margin_label=candidate_label(a_value, a_value <= CONTACT_TOL, MARGIN),
                C_distance_m=c_value, C_contact=bool(c_contact), C_nearest=c_nearest, C_status=c_status,
                C_contact_label=candidate_label(c_value, c_contact, 0.0),
                C_margin_label=candidate_label(c_value, c_contact, MARGIN),
                D_distance_m=d_value, D_contact=bool(d_contact), D_nearest=d_nearest, D_status=d_status,
                D_contact_label=candidate_label(d_value, d_contact, 0.0),
                D_margin_label=candidate_label(d_value, d_contact, MARGIN),
                reference_seconds=ref_elapsed, C_seconds=c_elapsed, D_seconds=d_elapsed,
                seconds=ref_elapsed + c_elapsed + d_elapsed,
            )
            rows.append(row)
    record = dict(chunk=chunk_id, pose_start=start, pose_end=end, rows=rows, pose_times=pose_times,
                  complete=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record))
    tmp.replace(path)
    return chunk_id, len(rows), sum(item["A_seconds"] for item in pose_times)


def build_summary(chunk_records, out_dir: Path) -> dict:
    rows = [row for record in chunk_records for row in record["rows"]]
    pose_times = [item for record in chunk_records for item in record["pose_times"]]
    rows.sort(key=lambda row: (row["pose"], ARMS.index(row["arm"])))
    pose_times.sort(key=lambda item: item["pose"])

    def table(subset, candidate):
        out = {}
        for name, threshold, ref_key, cand_key in (
            ("contact", 0.0, "reference_contact_label", f"{candidate}_contact_label"),
            ("margin_20mm", MARGIN, "reference_margin_label", f"{candidate}_margin_label"),
        ):
            valid = [row for row in subset if row[ref_key] in ("safe", "unsafe") and row[cand_key] in ("safe", "unsafe")]
            unknown = [row for row in subset if row[ref_key] == "unknown" or row[cand_key] == "unknown"]
            tp = sum(1 for row in valid if row[ref_key] == "unsafe" and row[cand_key] == "unsafe")
            fp = sum(1 for row in valid if row[ref_key] == "safe" and row[cand_key] == "unsafe")
            fn = sum(1 for row in valid if row[ref_key] == "unsafe" and row[cand_key] == "safe")
            tn = sum(1 for row in valid if row[ref_key] == "safe" and row[cand_key] == "safe")
            out[name] = dict(TP=tp, FP=fp, FN=fn, TN=tn, valid=len(valid), unknown=len(unknown),
                             miss_rate=(fn / (tp + fn)) if (tp + fn) else None,
                             exact=sum(1 for row in valid if row["reference_status"] == "exact"),
                             certified_culled=sum(1 for row in valid if row["reference_status"] == "certified_culled"),
                             failed=sum(1 for row in valid if row["reference_status"] == "failed"))
        return out

    def transitions(subset, candidate):
        out = {}
        for name, threshold, ref_key, cand_key in (
            ("contact", 0.0, "reference_contact_label", f"{candidate}_contact_label"),
            ("margin_20mm", MARGIN, "reference_margin_label", f"{candidate}_margin_label"),
        ):
            valid = [row for row in subset if row[ref_key] in ("safe", "unsafe") and row[cand_key] in ("safe", "unsafe")]
            a_key = "A_contact_label" if name == "contact" else "A_margin_label"
            eliminated_fp = [row for row in valid if row[a_key] == "unsafe"
                             and row[cand_key] == "safe" and row[ref_key] == "safe"]
            new_fp = [row for row in valid if row[a_key] == "safe"
                      and row[cand_key] == "unsafe" and row[ref_key] == "safe"]
            eliminated_tp = [row for row in valid if row[a_key] == "unsafe"
                             and row[cand_key] == "safe" and row[ref_key] == "unsafe"]
            new_tp = [row for row in valid if row[a_key] == "safe"
                      and row[cand_key] == "unsafe" and row[ref_key] == "unsafe"]
            out[name] = dict(eliminated_false_alarm=len(eliminated_fp), new_false_alarm=len(new_fp),
                             eliminated_true_alarm=len(eliminated_tp), new_true_alarm=len(new_tp))
        return out

    def pose_level(candidate):
        by_pose = {}
        for row in rows:
            entry = by_pose.setdefault(row["pose"], {"ref_contact": [], "ref_margin": [],
                                                     "cand_contact": [], "cand_margin": []})
            entry["ref_contact"].append(row["reference_contact_label"])
            entry["ref_margin"].append(row["reference_margin_label"])
            entry["cand_contact"].append(row[f"{candidate}_contact_label"])
            entry["cand_margin"].append(row[f"{candidate}_margin_label"])

        def any_label(labels):
            if "unsafe" in labels:
                return "unsafe"
            if "unknown" in labels:
                return "unknown"
            return "safe"

        pairs = {"contact": ([], []), "margin_20mm": ([], [])}
        for pose, entry in by_pose.items():
            for name, refs, cands in (("contact", entry["ref_contact"], entry["cand_contact"]),
                                      ("margin_20mm", entry["ref_margin"], entry["cand_margin"])):
                ref = any_label(refs)
                cand = any_label(cands)
                if ref == "unknown" or cand == "unknown":
                    continue
                pairs[name][0].append(ref)
                pairs[name][1].append(cand)
        out = {}
        for name, (refs, cands) in pairs.items():
            tp = sum(1 for r, c in zip(refs, cands) if r == "unsafe" and c == "unsafe")
            fp = sum(1 for r, c in zip(refs, cands) if r == "safe" and c == "unsafe")
            fn = sum(1 for r, c in zip(refs, cands) if r == "unsafe" and c == "safe")
            tn = sum(1 for r, c in zip(refs, cands) if r == "safe" and c == "safe")
            out[name] = dict(TP=tp, FP=fp, FN=fn, TN=tn, valid=len(refs),
                             unknown=len(by_pose) - len(refs),
                             miss_rate=(fn / (tp + fn)) if (tp + fn) else None,
                             poses_with_alarm=sum(1 for c in cands if c == "unsafe"))
        return out

    summary = {}
    summary["coverage"] = dict(
        poses_total=len(POSE_ARRAY), poses_complete=len({row["pose"] for row in rows}),
        pose_steps_expected=len(POSE_ARRAY) * len(ARMS), pair_steps=len(rows),
        missing=len(POSE_ARRAY) * len(ARMS) - len(rows), pairs=len(ARMS),
    )
    summary["rows"] = len(rows)
    summary["pairs"] = [f"{arm} {LIDAR}" for arm in ARMS]
    summary["thresholds"] = dict(contact_m=0.0, contact_tolerance_m=CONTACT_TOL, margin_m=MARGIN, reference_cap_m=CAP)
    summary["reference_status"] = {
        "exact": sum(1 for row in rows if row["reference_status"] == "exact"),
        "certified_culled": sum(1 for row in rows if row["reference_status"] == "certified_culled"),
        "failed": sum(1 for row in rows if row["reference_status"] == "failed"),
        "unknown": sum(1 for row in rows if row["reference_status"] not in ("exact", "certified_culled", "failed")),
    }
    for candidate in ("A", "C", "D"):
        summary[f"{candidate}_pair_step"] = table(rows, candidate)
        summary[f"{candidate}_per_pose_any"] = pose_level(candidate)
    summary["transitions_pair_step"] = {candidate: transitions(rows, candidate) for candidate in ("C", "D")}

    by_pair = {}
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        by_pair[arm] = dict(
            pair_steps=len(subset),
            reference=dict(
                exact=sum(1 for row in subset if row["reference_status"] == "exact"),
                certified_culled=sum(1 for row in subset if row["reference_status"] == "certified_culled"),
                failed=sum(1 for row in subset if row["reference_status"] == "failed"),
                contact=sum(1 for row in subset if row["reference_contact"]),
                margin_unsafe=sum(1 for row in subset if row["reference_margin_label"] == "unsafe"),
                exact_min_m=min((row["reference_distance_m"] for row in subset if row["reference_status"] == "exact"
                                 and row["reference_distance_m"] is not None), default=None),
            ),
        )
        for candidate in ("A", "C", "D"):
            by_pair[arm][candidate] = {
                "contact_alarm": sum(1 for row in subset if row[f"{candidate}_contact_label"] == "unsafe"),
                "margin_alarm": sum(1 for row in subset if row[f"{candidate}_margin_label"] == "unsafe"),
                "unknown": sum(1 for row in subset if row[f"{candidate}_contact_label"] == "unknown"
                               or row[f"{candidate}_margin_label"] == "unknown"),
                "nearest_examples": [
                    dict(pose=row["pose"], nearest=row[f"{candidate}_nearest"],
                         distance_m=row[f"{candidate}_distance_m"])
                    for row in sorted((row for row in subset if row[f"{candidate}_distance_m"] is not None),
                                      key=lambda row: row[f"{candidate}_distance_m"])[:3]
                ],
            }
    summary["per_pair"] = by_pair
    summary["cost"] = {
        "reference_seconds": dict(
            n=len(rows),
            median=float(np.median([row["reference_seconds"] for row in rows])),
            p95=float(np.percentile([row["reference_seconds"] for row in rows], 95)),
            max=float(max(row["reference_seconds"] for row in rows)),
        ),
        "A_seconds_per_pose": dict(
            n=len(pose_times),
            median=float(np.median([item["A_seconds"] for item in pose_times])),
            p95=float(np.percentile([item["A_seconds"] for item in pose_times], 95)),
            max=float(max(item["A_seconds"] for item in pose_times)),
        ),
        "C_seconds": dict(
            n=len(rows), median=float(np.median([row["C_seconds"] for row in rows])),
            p95=float(np.percentile([row["C_seconds"] for row in rows], 95)),
            max=float(max(row["C_seconds"] for row in rows)),
        ),
        "D_seconds": dict(
            n=len(rows), median=float(np.median([row["D_seconds"] for row in rows])),
            p95=float(np.percentile([row["D_seconds"] for row in rows], 95)),
            max=float(max(row["D_seconds"] for row in rows)),
        ),
    }
    summary["semantics"] = dict(
        A="deployed safety.py box proxy; d<=0 is a box overlap; value at 0.15 is a cap, not an exact distance",
        C="atomic arm boxes vs convex hull of the raw lidar STL; GJK tol 1e-8; hull contains the lidar mesh",
        D="atomic arm convex hulls vs convex hull of the raw lidar STL; Link6 keeps five atomic constituents",
        reference="original triangle meshes; exact/certified_culled at 0.05 m; failed is unknown; a cap is a lower bound",
        contact="unsafe if reference intersects or exact distance <= 1e-4 m; candidate contact is d<=1e-4 or intersecting",
        margin="unsafe if the same reference is inside 0.02 m",
        transforms="MuJoCo full FK; raw STL body frames; lidar hull in lidar_link0 frame; box table in the safety body frame",
        no_column_order_labels="A column names come from ARM_BODY_LIST/PROT_GROUPS entries, not from enumerate(PENALTY_COLUMNS)",
    )
    return summary


def main() -> None:
    out_dir = ROOT / args.out
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(POSE_ARRAY) + args.chunk - 1) // args.chunk
    if args.restart:
        for path in chunks_dir.glob("chunk_*.json"):
            path.unlink()
    expected = []
    for chunk_id in range(n_chunks):
        path = chunks_dir / f"chunk_{chunk_id:04d}.json"
        if not path.exists():
            expected.append(chunk_id)
    print(f"poses {len(POSE_ARRAY)}, chunks {n_chunks}, missing {len(expected)}, workers {args.workers}", flush=True)
    with mp.get_context("fork").Pool(processes=args.workers) as pool:
        for chunk_id, count, a_seconds in pool.imap_unordered(process_chunk, expected):
            print(f"chunk {chunk_id}: {count} rows, A {a_seconds:.2f}s", flush=True)
    records = []
    for chunk_id in range(n_chunks):
        path = chunks_dir / f"chunk_{chunk_id:04d}.json"
        records.append(json.loads(path.read_text()))
    records.sort(key=lambda record: record["chunk"])
    summary = build_summary(records, out_dir)
    rows = [row for record in records for row in record["rows"]]
    rows.sort(key=lambda row: (row["pose"], ARMS.index(row["arm"])))
    with (out_dir / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "poses.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pose", "A_seconds"])
        writer.writeheader()
        writer.writerows(item for record in records for item in record["pose_times"])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    manifest = dict(
        poses=str(Path(args.poses)), poses_sha256=sha256(Path(args.poses)),
        scene=str(SCENE), scene_sha256=sha256(SCENE),
        safety=str(SAFETY), safety_sha256=sha256(SAFETY),
        scripts={path.name: sha256(path) for path in (Path(__file__), HERE / "reference_geometry.py",
                                                       HERE / "convex_distance.py", HERE / "collision_parts.py",
                                                       HERE / "triangle_distance.py", HERE / "robust_intersect.py")},
        meshes={part.name: sha256(MESH_DIR / part.stl) for part in CONSTITUENTS if part.kind == "mesh"},
        pairs=[f"{arm} {LIDAR}" for arm in ARMS],
        constituents={arm: [part.name for part in constituents_of(arm)] for arm in ARMS + (LIDAR,)},
        a_columns={arm: [dict(column=column, name=name) for column, name in A_COLUMNS[arm]] for arm in ARMS},
        contact_tolerance_m=CONTACT_TOL, margin_m=MARGIN, reference_cap_m=CAP, box_cap_m=BOX_CAP,
        transform="MuJoCo mj_forward at each recorded pose; body-frame raw STL; relative transform R_arm^T R_lidar",
        complete=summary["coverage"]["missing"] == 0,
        coverage=summary["coverage"],
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
