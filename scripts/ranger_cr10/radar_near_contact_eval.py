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

"""Stage 4.13b: evaluate A/C/D on the frozen independent near-contact set.

The near set is read, never changed. The reference label is the one frozen by
``radar_near_contact_search.py``; this script only adds the three candidate representations:

    A  arm box + lidar box (deployed safety.py)
    C  arm box + lidar hull
    D  atomic arm hulls + lidar hull

Precision is computed only on rows where the reference is an exact separated distance.
Certified-culled rows are used for the 0/20 mm classification but never for an error P95.
False contact, margin false alarm/miss, reference-overlap detection and the per-bin error table
are reported separately. The synthetic box/hull and hull/hull cases verify the intersection path
for the exact atomic query types; the real FK overlaps are reported next to them.

Run:
    python scripts/ranger_cr10/radar_near_contact_eval.py
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from collision_parts import box_mesh, constituents_of  # noqa: E402
from convex_distance import gjk_distance  # noqa: E402

ROOT = HERE.parents[1]
SCENE = ROOT / "mujoco/robots/ranger_cr10/scene.xml"
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes"
SAFETY = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/safety.py"
NEAR = ROOT / "outputs/stage413/b/radar_near_contact/near_set.json"
ARMS = ("cr10_Link1", "cr10_Link2", "cr10_Link3", "cr10_Link4", "cr10_Link5", "cr10_Link6")
LIDAR = "lidar_link0"
CONTACT_TOL = 1e-4
MARGIN = 0.02
BINS = ("0-1mm", "1-3mm", "3-5mm", "5-10mm", "10-20mm", "20-30mm",
        ">=30mm_exact", ">=30mm_certified_culled", "overlap", "failed")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--near", default=str(NEAR))
ap.add_argument("--out", default="outputs/stage413/b/radar_near_contact")
args = ap.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_safety():
    spec = importlib.util.spec_from_file_location("safety_near_eval", SAFETY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


safety = load_safety()
MODEL = mujoco.MjModel.from_xml_path(str(SCENE))
DATA = mujoco.MjData(MODEL)
BID = {mujoco.mj_id2name(MODEL, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(MODEL.nbody)}
ARM_ADR = [MODEL.jnt_qposadr[mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)]


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


ARM_HULL: dict[str, list[tuple[str, np.ndarray]]] = {}
for arm in ARMS:
    ARM_HULL[arm] = []
    for part in constituents_of(arm):
        vertices, faces = _load_mesh(part)
        if part.kind == "mesh":
            hull = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).convex_hull.vertices
        else:
            hull = vertices
        ARM_HULL[arm].append((part.name, np.asarray(hull, dtype=np.float64)))
_vertices, _faces = _load_mesh(constituents_of(LIDAR)[0])
LIDAR_HULL = np.asarray(trimesh.Trimesh(vertices=_vertices, faces=_faces, process=False).convex_hull.vertices, dtype=np.float64)
LIDAR_HULL_OBJECT = ConvexHull(LIDAR_HULL, qhull_options="QJ")

A_BOX = {arm: [dict(name=entry[0], vertices=box_mesh(entry[1], entry[2], entry[3])[0]) for entry in safety.ARM_PARTS[arm]]
         for arm in ARMS}
LIDAR_BOX_BODY = safety.PROTECTED_PARTS[LIDAR][0]
LIDAR_BOX = [dict(name=entry[0], vertices=box_mesh(entry[1], entry[2], entry[3])[0])
             for entry in safety.PROTECTED_PARTS[LIDAR][1]]
ARM_PART_NAME = [entry[0] for entries in safety.ARM_PARTS.values() for entry in entries]
PROT_PART_NAME = [entry[0] for _, entries in safety.PROTECTED_PARTS.values() for entry in entries]
A_COLUMNS: dict[str, list[tuple[int, str]]] = {arm: [] for arm in ARMS}
for k, (i, p) in enumerate(safety.SENSOR_PAIRS):
    if safety.PROT_GROUPS[p] == LIDAR and safety.ARM_BODY_LIST[i] in A_COLUMNS:
        A_COLUMNS[safety.ARM_BODY_LIST[i]].append((len(safety.SELF_PAIRS) + k,
                                                   f"{ARM_PART_NAME[i]}|{PROT_PART_NAME[p]}"))


def set_q(q):
    mujoco.mj_resetDataKeyframe(MODEL, DATA, MODEL.key("home").id)
    for address, value in zip(ARM_ADR, q):
        DATA.qpos[address] = float(value)
    mujoco.mj_forward(MODEL, DATA)


def body_transform(name):
    body = BID[name]
    return np.array(DATA.xmat[body], dtype=np.float64).reshape(3, 3), np.array(DATA.xpos[body], dtype=np.float64)


def to_world(points, rotation, translation):
    return np.asarray(points, dtype=np.float64) @ rotation.T + translation


def gjk(points_a, points_b):
    result = gjk_distance(np.asarray(points_a, dtype=np.float64), np.asarray(points_b, dtype=np.float64), tol=1e-8)
    if result.status == "intersecting":
        return 0.0, True, "intersecting"
    if result.status == "separated" and np.isfinite(result.distance):
        return float(result.distance), False, "separated"
    if result.status == "degenerate" and np.isfinite(result.distance):
        if result.distance > CONTACT_TOL:
            return float(result.distance), False, "degenerate_separated"
        return 0.0, True, "degenerate_contact"
    return float("nan"), False, result.status


def a_distances():
    names = list(dict.fromkeys(safety.ARM_BODY_LIST + safety.PROT_BODY_LIST))
    position = torch.tensor(np.stack([DATA.xpos[BID[name]] for name in names])[None], dtype=torch.float32)
    quat = torch.tensor(np.stack([DATA.xquat[BID[name]] for name in names])[None], dtype=torch.float32)

    class Asset:
        body_names = names

        class data:  # noqa: N801
            body_pos_w = position
            body_quat_w = quat

    distances = safety.clearance(Asset, cap=0.15)[0].detach().cpu().numpy()
    out = {}
    for arm in ARMS:
        columns = A_COLUMNS[arm]
        if not columns:
            out[arm] = (float("nan"), False, "", "no_columns")
            continue
        index, name = min(columns, key=lambda item: float(distances[item[0]]))
        value = float(distances[index])
        out[arm] = (value, bool(value <= CONTACT_TOL), name, "capped" if value >= 0.15 - 1e-6 else "box")
    return out


def c_distance(arm):
    r_arm, p_arm = body_transform(arm)
    r_lidar, p_lidar = body_transform(LIDAR)
    lidar_world = to_world(LIDAR_HULL, r_lidar, p_lidar)
    best, best_name, best_status = float("inf"), "", "clear"
    for box in A_BOX[arm]:
        distance, contact, status = gjk(to_world(box["vertices"], r_arm, p_arm), lidar_world)
        if contact:
            return 0.0, True, f"{box['name']}|lidar_hull", status
        if np.isfinite(distance) and distance < best:
            best, best_name, best_status = distance, f"{box['name']}|lidar_hull", status
    return (best, False, best_name, best_status) if np.isfinite(best) else (float("nan"), False, best_name, "unknown")


def d_distance(arm):
    r_arm, p_arm = body_transform(arm)
    r_lidar, p_lidar = body_transform(LIDAR)
    lidar_world = to_world(LIDAR_HULL, r_lidar, p_lidar)
    best, best_name, best_status = float("inf"), "", "clear"
    for name, hull in ARM_HULL[arm]:
        distance, contact, status = gjk(to_world(hull, r_arm, p_arm), lidar_world)
        if contact:
            return 0.0, True, f"{name}|lidar_hull", status
        if np.isfinite(distance) and distance < best:
            best, best_name, best_status = distance, f"{name}|lidar_hull", status
    return (best, False, best_name, best_status) if np.isfinite(best) else (float("nan"), False, best_name, "unknown")


def reference_label(row, threshold):
    if row["reference_contact"]:
        return "unsafe"
    limit = max(threshold, CONTACT_TOL)
    if row["reference_status"] == "exact":
        if row["reference_distance_m"] is None:
            return "unknown"
        return "unsafe" if row["reference_distance_m"] <= limit else "safe"
    if row["reference_status"] == "certified_culled" and row["reference_lower_bound_m"] is not None:
        return "safe" if row["reference_lower_bound_m"] > limit else "unknown"
    return "unknown"


def candidate_label(distance, contact, threshold):
    if contact or (np.isfinite(distance) and distance <= max(threshold, CONTACT_TOL)):
        return "unsafe"
    if np.isfinite(distance):
        return "safe"
    return "unknown"


def confusion(rows, threshold, candidate_key):
    ref_key = "reference_contact_label" if threshold == 0.0 else "reference_margin_label"
    cand_key = f"{candidate_key}_{'contact' if threshold == 0.0 else 'margin'}_label"
    valid = [row for row in rows if row[ref_key] in ("safe", "unsafe") and row[cand_key] in ("safe", "unsafe")]
    unknown = [row for row in rows if row[ref_key] == "unknown" or row[cand_key] == "unknown"]
    tp = sum(1 for row in valid if row[ref_key] == "unsafe" and row[cand_key] == "unsafe")
    fp = sum(1 for row in valid if row[ref_key] == "safe" and row[cand_key] == "unsafe")
    fn = sum(1 for row in valid if row[ref_key] == "unsafe" and row[cand_key] == "safe")
    tn = sum(1 for row in valid if row[ref_key] == "safe" and row[cand_key] == "safe")
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, valid=len(valid), unknown=len(unknown),
                miss_rate=(fn / (tp + fn)) if (tp + fn) else None,
                exact=sum(1 for row in valid if row["reference_status"] == "exact"),
                certified_culled=sum(1 for row in valid if row["reference_status"] == "certified_culled"),
                failed=sum(1 for row in valid if row["reference_status"] == "failed"))


def error_stats(rows, candidate_key):
    subset = [row for row in rows if row["reference_status"] == "exact" and not row["reference_contact"]
              and row["reference_distance_m"] is not None and np.isfinite(row[f"{candidate_key}_distance_m"])]
    if not subset:
        return dict(n=0, abs_p95_m=None, abs_max_m=None, signed_mean_m=None, optimistic_max_m=None)
    reference = np.array([row["reference_distance_m"] for row in subset])
    candidate = np.array([row[f"{candidate_key}_distance_m"] for row in subset])
    error = candidate - reference
    return dict(n=len(subset), abs_p95_m=float(np.percentile(np.abs(error), 95)),
                abs_max_m=float(np.max(np.abs(error))), signed_mean_m=float(np.mean(error)),
                optimistic_max_m=float(np.max(np.maximum(error, 0.0))),
                pessimistic_max_m=float(np.max(np.maximum(-error, 0.0))))


def synthetic_intersection():
    """Known box/hull and hull/hull queries for the exact atomic query types."""
    hull = LIDAR_HULL
    vertex = hull[int(np.argmax(hull[:, 2]))]
    normal = None
    for equation in LIDAR_HULL_OBJECT.equations:
        if abs(float(equation[:3] @ vertex + equation[3])) < 1e-6:
            normal = equation[:3]
            break
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0])
    normal = normal / np.linalg.norm(normal)
    half = 0.001
    cube = np.array([[sx, sy, sz] for sx in (-half, half) for sy in (-half, half) for sz in (-half, half)])
    box_overlap = cube + vertex
    box_separated = cube + vertex + normal * (half + 0.005)
    overlap_distance, overlap_contact, overlap_status = gjk(box_overlap, hull)
    separated_distance, separated_contact, separated_status = gjk(box_separated, hull)
    tetra = np.array([[0.0, 0.0, 0.0], [half, 0.0, 0.0], [0.0, half, 0.0], [0.0, 0.0, half]])
    tetra_tangent = tetra + vertex
    tetra_deep = tetra + vertex - normal * (half + 0.0005)
    tetra_separated = tetra + vertex + normal * (half + 0.005)
    t_tangent = gjk(tetra_tangent, hull)
    t_deep = gjk(tetra_deep, hull)
    t_separated = gjk(tetra_separated, hull)
    return dict(
        lidar_hull_vertex=vertex.tolist(), outward_normal=normal.tolist(), cube_half_m=half,
        box_vs_hull={
            "overlap": dict(distance_m=overlap_distance, contact=bool(overlap_contact), status=overlap_status),
            "separated_5mm": dict(distance_m=separated_distance, contact=bool(separated_contact), status=separated_status),
        },
        hull_vs_hull={
            "tangent": dict(distance_m=t_tangent[0], contact=bool(t_tangent[1]), status=t_tangent[2]),
            "deep_overlap": dict(distance_m=t_deep[0], contact=bool(t_deep[1]), status=t_deep[2]),
            "separated_5mm": dict(distance_m=t_separated[0], contact=bool(t_separated[1]), status=t_separated[2]),
        },
        note="query types only; the real FK overlap rows below are the atomic-geometry check",
    )


def main() -> None:
    near = json.loads(Path(args.near).read_text())
    rows = near["rows"]
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        set_q(row["q"])
        a_started = time.perf_counter()
        a = a_distances()[row["arm"]]
        a_seconds = time.perf_counter() - a_started
        c_started = time.perf_counter()
        c = c_distance(row["arm"])
        c_seconds = time.perf_counter() - c_started
        d_started = time.perf_counter()
        d = d_distance(row["arm"])
        d_seconds = time.perf_counter() - d_started
        row.update(
            A_distance_m=a[0], A_contact=bool(a[1]), A_nearest=a[2], A_status=a[3],
            C_distance_m=c[0], C_contact=bool(c[1]), C_nearest=c[2], C_status=c[3],
            D_distance_m=d[0], D_contact=bool(d[1]), D_nearest=d[2], D_status=d[3],
            A_seconds=a_seconds, C_seconds=c_seconds, D_seconds=d_seconds,
            reference_contact_label=reference_label(row, 0.0),
            reference_margin_label=reference_label(row, MARGIN),
        )
        for key in ("A", "C", "D"):
            distance = row[f"{key}_distance_m"]
            contact = row[f"{key}_contact"]
            row[f"{key}_contact_label"] = candidate_label(distance, contact, 0.0)
            row[f"{key}_margin_label"] = candidate_label(distance, contact, MARGIN)

    summary = dict(
        near_set=args.near,
        near_set_sha256=sha256(Path(args.near)),
        rule_sha256=near["rule_sha256"],
        frozen_before_evaluation=near.get("frozen_before_evaluation"),
        rows=len(rows),
        thresholds=dict(contact_tolerance_m=CONTACT_TOL, margin_m=MARGIN,
                        reference_cap_m=near["reference_cap_m"]),
        reference_status={s: sum(1 for row in rows if row["reference_status"] == s)
                          for s in ("exact", "certified_culled", "failed")},
        by_pair={},
        synthetic_intersection=synthetic_intersection(),
        notes=[
            "Precision uses exact separated reference rows only; certified-culled rows are classification labels.",
            "A/C/D distances are candidate values; the reference cap is never used as an exact distance.",
            "C/D are convex hull/box queries; a false contact is reported separately from a missed overlap.",
            "Path s is a joint displacement, not a penetration depth.",
        ],
    )
    key_bins = ["0-1mm", "1-3mm", "3-5mm", "5-10mm", "10-20mm"]
    key_arms = ("cr10_Link2", "cr10_Link3")

    def bin_entry(subset, candidate, label):
        sub = [row for row in subset if row["bin"] == label]
        overlap = [row for row in sub if row["reference_contact"]]
        overlap_valid = [row for row in overlap if row[f"{candidate}_contact_label"] in ("safe", "unsafe")]
        return dict(
            n=len(sub),
            contact=confusion(sub, 0.0, candidate),
            margin=confusion(sub, MARGIN, candidate),
            error=error_stats(sub, candidate),
            false_contact=sum(1 for row in sub if row[f"{candidate}_contact"]
                              and row["reference_contact_label"] == "safe"),
            false_margin=sum(1 for row in sub if row[f"{candidate}_margin_label"] == "unsafe"
                             and row["reference_margin_label"] == "safe"),
            missed_margin=sum(1 for row in sub if row[f"{candidate}_margin_label"] == "safe"
                              and row["reference_margin_label"] == "unsafe"),
            overlap_denominator=len(overlap_valid),
            overlap_miss=sum(1 for row in overlap_valid if not row[f"{candidate}_contact"]),
            overlap_unknown=sum(1 for row in overlap if row[f"{candidate}_contact_label"] == "unknown"),
        )

    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        entry = dict(n=len(subset),
                     bins={label: sum(1 for row in subset if row["bin"] == label) for label in BINS},
                     overlap_rows=sum(1 for row in subset if row["reference_contact"]),
                     sample_types={t: sum(1 for row in subset if row["sample_type"] == t)
                                   for t in sorted({row["sample_type"] for row in subset})},
                     )
        for candidate in ("A", "C", "D"):
            overlap = [row for row in subset if row["reference_contact"]]
            overlap_valid = [row for row in overlap if row[f"{candidate}_contact_label"] in ("safe", "unsafe")]
            entry[candidate] = dict(
                contact=confusion(subset, 0.0, candidate),
                margin=confusion(subset, MARGIN, candidate),
                error=error_stats(subset, candidate),
                by_bin={label: bin_entry(subset, candidate, label) for label in BINS},
                false_contact=sum(1 for row in subset if row[f"{candidate}_contact"]
                                  and row["reference_contact_label"] == "safe"),
                overlap_denominator=len(overlap_valid),
                overlap_miss=sum(1 for row in overlap_valid if not row[f"{candidate}_contact"]),
                overlap_unknown=sum(1 for row in overlap if row[f"{candidate}_contact_label"] == "unknown"),
                unknown=sum(1 for row in subset if row[f"{candidate}_contact_label"] == "unknown"
                            or row[f"{candidate}_margin_label"] == "unknown"),
            )
        cluster_keys = sorted({row.get("path_id") or f"cluster_{row.get('cluster')}" for row in subset})
        entry["cluster_detail"] = {}
        for cluster in cluster_keys:
            cluster_rows = [row for row in subset
                            if (row.get("path_id") or f"cluster_{row.get('cluster')}") == cluster]
            entry["cluster_detail"][cluster] = dict(
                n=len(cluster_rows), sample_types=sorted({row["sample_type"] for row in cluster_rows}),
                reference_status={s: sum(1 for row in cluster_rows if row["reference_status"] == s)
                                  for s in ("exact", "certified_culled", "failed")},
                overlap_rows=sum(1 for row in cluster_rows if row["reference_contact"]),
                A=dict(error=error_stats(cluster_rows, "A"),
                       false_contact=sum(1 for row in cluster_rows if row["A_contact"]
                                         and row["reference_contact_label"] == "safe")),
                C=dict(error=error_stats(cluster_rows, "C"),
                       false_contact=sum(1 for row in cluster_rows if row["C_contact"]
                                         and row["reference_contact_label"] == "safe"),
                       overlap_miss=sum(1 for row in cluster_rows if row["reference_contact"]
                                        and row["C_contact_label"] in ("safe", "unsafe") and not row["C_contact"])),
                D=dict(error=error_stats(cluster_rows, "D"),
                       false_contact=sum(1 for row in cluster_rows if row["D_contact"]
                                         and row["reference_contact_label"] == "safe"),
                       overlap_miss=sum(1 for row in cluster_rows if row["reference_contact"]
                                        and row["D_contact_label"] in ("safe", "unsafe") and not row["D_contact"])),
            )
        summary["by_pair"][arm] = entry

    summary["cost"] = {
        key: dict(n=len(rows), median=float(np.median([row[f"{key}_seconds"] for row in rows])),
                  p95=float(np.percentile([row[f"{key}_seconds"] for row in rows], 95)),
                  max=float(max(row[f"{key}_seconds"] for row in rows)))
        for key in ("A", "C", "D")
    }
    summary["numerical_uncertainty"] = {
        "reference_coordinates": "float64 triangle vertices and witnesses; exact predicate only for the "
                                 "triangle-triangle intersection Boolean",
        "reference_cap": "0.03 m is a certified lower bound, not an exact distance",
        "gjk_tolerance_m": 1e-8,
        "box_mesh_coverage": "raw-STL vertices exceed the built-in arm/lidar boxes by at most 0.0079 mm "
                             "(7.9 micrometres); this is a small numerical under-coverage, reported separately",
        "lidar_hull_coverage": "convex hull of the raw lidar STL vertices contains the mesh",
        "path_s": "joint-space path displacement, not penetration depth",
        "caveat": "C/D are convex over-approximations; a contact is a candidate alarm, not a reference contact",
    }

    # The selection is made from the near set, not from the operating replay.
    choice = {}
    for candidate in ("C", "D"):
        checks = {}
        for arm in ARMS:
            entry = summary["by_pair"][arm]
            bins = entry[candidate]["by_bin"]
            p95_ok = all(bins[label]["n"] > 0 and bins[label]["error"]["abs_p95_m"] is not None
                         and bins[label]["error"]["abs_p95_m"] <= 0.003 for label in key_bins)
            overlap_ok = entry[candidate]["overlap_denominator"] == 0 or entry[candidate]["overlap_miss"] == 0
            checks[arm] = dict(
                p95_ok=p95_ok, overlap_ok=overlap_ok,
                bins={label: dict(n=bins[label]["n"], p95_m=bins[label]["error"]["abs_p95_m"],
                                  max_m=bins[label]["error"]["abs_max_m"]) for label in key_bins},
                overlap_rows=entry["overlap_rows"],
                overlap_denominator=entry[candidate]["overlap_denominator"],
                overlap_miss=entry[candidate]["overlap_miss"],
                false_contact=entry[candidate]["false_contact"],
            )
        p95_key = all(checks[arm]["p95_ok"] for arm in key_arms)
        overlap_key_coverage = all(checks[arm]["overlap_rows"] > 0 for arm in key_arms)
        overlap_key_ok = all(checks[arm]["overlap_ok"] for arm in key_arms)
        overlap_all_ok = all(checks[arm]["overlap_ok"] for arm in ARMS)
        coverage_complete = all(checks[arm]["bins"][label]["n"] > 0 for arm in key_arms for label in key_bins)
        choice[candidate] = dict(
            p95_key_pairs=p95_key,
            p95_all_pairs=all(item["p95_ok"] for item in checks.values()),
            overlap_key_coverage=overlap_key_coverage,
            overlap_key_ok=overlap_key_ok,
            overlap_all_ok=overlap_all_ok,
            coverage_complete=coverage_complete,
            by_pair=checks,
        )
    both = {}
    for candidate in ("C", "D"):
        both[candidate] = (choice[candidate]["coverage_complete"] and choice[candidate]["p95_key_pairs"]
                          and choice[candidate]["overlap_key_coverage"] and choice[candidate]["overlap_key_ok"]
                          and choice[candidate]["overlap_all_ok"])
    if both["C"]:
        decision = "C"
    elif both["D"]:
        decision = "D"
    elif not (choice["C"]["coverage_complete"] and choice["D"]["coverage_complete"]):
        decision = "neither_coverage_incomplete"
    else:
        decision = "neither"
    choice["decision"] = decision
    choice["note"] = ("The decision is only for the current validation range; it is not a full-body "
                      "or continuous-motion safety pass. Runtime-distribution improvement is reported "
                      "separately and is not a substitute for near-contact precision or overlap detection.")
    summary["choice"] = choice

    (out_dir / "eval.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "rows.csv").write_text("")  # placeholder replaced below
    fieldnames = list(rows[0]) if rows else []
    with (out_dir / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "choice.json").write_text(json.dumps(choice, indent=1))
    (out_dir / "synthetic_intersection.json").write_text(json.dumps(summary["synthetic_intersection"], indent=1))
    print(json.dumps(choice, indent=1))
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
