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

"""Full MM/HM/MH/HH attribution of the D errors on the frozen near set.

MM = arm mesh vs lidar mesh
HM = arm hull vs lidar mesh
MH = arm mesh vs lidar hull
HH = arm hull vs lidar hull (the D representation)

All four use the same original meshes, the same rigid transform, the same atomic constituents,
the same union-min aggregation and the same reference query (MeshRef) so their distances are
comparable. The stored D GJK value is kept next to HH as a query-semantics check.

For every selected row (reference exact separated, D underestimates by >3 mm) the script stores
the four distances, status, containment-unknown flag, winning atomic pair, witness points in arm
and lidar local frames, the single-side gains and the interaction term. The single-side gains are
reported, not forced to sum to the total.

Run:
    python scripts/ranger_cr10/radar_mm_hm_mh_hh_attribution.py
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import csv
import json
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from pathlib import Path

import mujoco
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from collision_parts import box_mesh, constituents_of  # noqa: E402
from reference_geometry import MeshRef  # noqa: E402

ROOT = HERE.parents[1]
SCENE = ROOT / "mujoco/robots/ranger_cr10/scene.xml"
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes"
NEAR = ROOT / "outputs/stage413/b/radar_near_contact/v3/near_set_containment_v3.json"
EVAL_ROWS = ROOT / "outputs/stage413/b/radar_near_contact/v2/rows.csv"
OUT_DIR = ROOT / "outputs/stage413/b/radar_round2_2026_11_07"
ARMS = ("cr10_Link1", "cr10_Link2", "cr10_Link3", "cr10_Link4", "cr10_Link5", "cr10_Link6")
LIDAR = "lidar_link0"
CAP = 0.05
CONTACT_TOL = 1e-4
ERROR_THRESHOLD = 3e-3

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--near", default=str(NEAR))
ap.add_argument("--rows", default=str(EVAL_ROWS))
ap.add_argument("--out", default=str(OUT_DIR))
ap.add_argument("--workers", type=int, default=10)
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


ARM_MESH = {}
ARM_HULL = {}
for arm in ARMS:
    ARM_MESH[arm] = []
    ARM_HULL[arm] = []
    for part in constituents_of(arm):
        vertices, faces = _load(part)
        ARM_MESH[arm].append((part.name, MeshRef(vertices, faces, name=f"{arm}/{part.name}")))
        hull = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).convex_hull
        ARM_HULL[arm].append((part.name, MeshRef(np.asarray(hull.vertices), np.asarray(hull.faces),
                                                 name=f"{arm}/{part.name}/hull")))
_lidar_vertices, _lidar_faces = _load(constituents_of(LIDAR)[0])
LIDAR_MESH = MeshRef(_lidar_vertices, _lidar_faces, name=LIDAR)
_lidar_hull = trimesh.Trimesh(vertices=_lidar_vertices, faces=_lidar_faces, process=False).convex_hull
LIDAR_HULL = MeshRef(np.asarray(_lidar_hull.vertices), np.asarray(_lidar_hull.faces), name=f"{LIDAR}/hull")


def set_q(q):
    mujoco.mj_resetDataKeyframe(MODEL, DATA, MODEL.key("home").id)
    for address, value in zip(ARM_ADR, q):
        DATA.qpos[address] = float(value)
    mujoco.mj_forward(MODEL, DATA)


def body_transform(name):
    body = BID[name]
    return np.array(DATA.xmat[body], dtype=np.float64).reshape(3, 3), np.array(DATA.xpos[body], dtype=np.float64)


def query_representation(arm, arm_refs, lidar_ref):
    r_arm, p_arm = body_transform(arm)
    r_lidar, p_lidar = body_transform(LIDAR)
    rotation = r_arm.T @ r_lidar
    translation = r_arm.T @ (p_lidar - p_arm)
    parts = []
    for name, ref in arm_refs:
        result = ref.closest_to_in_frame(lidar_ref, rotation, translation, max_clearance=CAP)
        entry = dict(atomic=name, status=result.status, intersects=bool(result.intersects),
                     reason=result.reason, clearance=float(result.clearance) if np.isfinite(result.clearance) else None,
                     containment_unknown=bool(result.extra.get("containment_unknown", False)))
        if result.witness_a is not None and result.witness_b is not None:
            witness_arm = np.asarray(result.witness_a, dtype=np.float64)
            witness_b_arm = np.asarray(result.witness_b, dtype=np.float64)
            witness_b_world = r_arm @ witness_b_arm + p_arm
            witness_lidar = r_lidar.T @ (witness_b_world - p_lidar)
            entry["witness_arm"] = witness_arm.tolist()
            entry["witness_lidar"] = witness_lidar.tolist()
        else:
            entry["witness_arm"] = None
            entry["witness_lidar"] = None
        parts.append(entry)
    contacts = [part for part in parts if part["intersects"]]
    if contacts:
        best = contacts[0]
        return dict(distance=0.0, status="intersects", winner=f"{best['atomic']}|{LIDAR}",
                    reason=best["reason"], containment_unknown=any(p["containment_unknown"] for p in parts),
                    winner_detail=best, parts=parts)
    exact = [part for part in parts if part["status"] == "exact" and part["clearance"] is not None]
    if exact:
        best = min(exact, key=lambda part: part["clearance"])
        return dict(distance=float(best["clearance"]), status="exact", winner=f"{best['atomic']}|{LIDAR}",
                    reason="", containment_unknown=any(p["containment_unknown"] for p in parts),
                    winner_detail=best, parts=parts)
    culled = [part for part in parts if part["status"] == "certified_culled"]
    if culled and all(part["status"] in ("exact", "certified_culled") for part in parts):
        # All exact values are <= cap; a culled constituent is >= cap, so the exact min is the
        # group min when one exists (handled above). With no exact, the group lower bound is cap.
        lower = max(part["clearance"] for part in culled)
        return dict(distance=float(lower), status="certified_culled", winner=f"{culled[0]['atomic']}|{LIDAR}",
                    reason="", containment_unknown=any(p["containment_unknown"] for p in parts),
                    winner_detail=culled[0], parts=parts)
    return dict(distance=None, status="failed", winner="", reason="",
                containment_unknown=any(p["containment_unknown"] for p in parts), winner_detail=None, parts=parts)


def attribute(task):
    row, near_row = task
    set_q(near_row["q"])
    arm = row["arm"]
    mm = query_representation(arm, ARM_MESH[arm], LIDAR_MESH)
    hm = query_representation(arm, ARM_HULL[arm], LIDAR_MESH)
    mh = query_representation(arm, ARM_MESH[arm], LIDAR_HULL)
    hh = query_representation(arm, ARM_HULL[arm], LIDAR_HULL)
    d_gjk = None if row["D_distance_m"] in ("", "None") else float(row["D_distance_m"])
    mm_stored = None if row["reference_distance_m"] in ("", "None") else float(row["reference_distance_m"])
    distances = dict(MM=mm["distance"], HM=hm["distance"], MH=mh["distance"], HH=hh["distance"])
    if mm["distance"] is not None and mm_stored is not None:
        mm_delta = mm["distance"] - mm_stored
    else:
        mm_delta = None
    hh_delta = None
    if hh["distance"] is not None and d_gjk is not None:
        hh_delta = hh["distance"] - d_gjk
    gains = {}
    if all(distances[key] is not None for key in ("MM", "HM", "MH", "HH")):
        mm_v, hm_v, mh_v, hh_v = distances["MM"], distances["HM"], distances["MH"], distances["HH"]
        total = mm_v - hh_v
        arm_gain = mh_v - hh_v
        radar_gain = hm_v - hh_v
        gains = dict(total_loss=total, arm_only_gain=arm_gain, radar_only_gain=radar_gain,
                     both_gain=total, interaction=total - arm_gain - radar_gain)
        violations = {}
        if hh_v > hm_v + 1e-6:
            violations["HH_gt_HM"] = hh_v - hm_v
        if hh_v > mh_v + 1e-6:
            violations["HH_gt_MH"] = hh_v - mh_v
        if hm_v > mm_v + 1e-6:
            violations["HM_gt_MM"] = hm_v - mm_v
        if mh_v > mm_v + 1e-6:
            violations["MH_gt_MM"] = mh_v - mm_v
    else:
        total = arm_gain = radar_gain = None
        violations = {}
    return dict(
        row_id=row["row_id"], pose_q=near_row["q"], arm=arm, bin=row["bin"], sample_type=row["sample_type"],
        reference_mm=mm_stored, d_gjk_m=d_gjk, distances_m=distances,
        statuses={key: value["status"] for key, value in (("MM", mm), ("HM", hm), ("MH", mh), ("HH", hh))},
        intersects={key: bool(value["status"] == "intersects") for key, value in (("MM", mm), ("HM", hm), ("MH", mh), ("HH", hh))},
        containment_unknown={key: bool(value["containment_unknown"]) for key, value in (("MM", mm), ("HM", hm), ("MH", mh), ("HH", hh))},
        winners={key: value["winner"] for key, value in (("MM", mm), ("HM", hm), ("MH", mh), ("HH", hh))},
        winner_details={key: value["winner_detail"] for key, value in (("MM", mm), ("HM", hm), ("MH", mh), ("HH", hh))},
        per_part={key: value["parts"] for key, value in (("MM", mm), ("HM", hm), ("MH", mh), ("HH", hh))},
        mm_delta_m=mm_delta, hh_vs_d_gjk_m=hh_delta,
        gains=gains, monotonic_violations_m=violations,
    )


def main() -> None:
    near = json.loads(Path(args.near).read_text())
    near_by_id = {row["row_id"]: row for row in near["rows"]}
    rows = list(csv.DictReader(open(args.rows)))
    selected = []
    for row in rows:
        if row["reference_status"] != "exact" or row["reference_contact"] == "True":
            continue
        if row["reference_distance_m"] in ("", "None") or row["D_distance_m"] in ("", "None"):
            continue
        if float(row["reference_distance_m"]) > CAP:
            continue
        if float(row["reference_distance_m"]) - float(row["D_distance_m"]) <= ERROR_THRESHOLD:
            continue
        if row["row_id"] not in near_by_id:
            continue
        selected.append((row, near_by_id[row["row_id"]]))
    tasks = selected
    results = []
    with mp.get_context("fork").Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(attribute, tasks):
            results.append(result)
    results.sort(key=lambda item: item["row_id"])

    for result in results:
        gains = result["gains"]
        if not gains or gains["total_loss"] is None:
            result["side"] = "unexplained"
            continue
        total = gains["total_loss"]
        arm_gain = gains["arm_only_gain"]
        radar_gain = gains["radar_only_gain"]
        if total > ERROR_THRESHOLD and arm_gain >= 0.5 and arm_gain >= radar_gain + 0.5:
            result["side"] = "arm_side"
        elif total > ERROR_THRESHOLD and radar_gain >= 0.5 and radar_gain >= arm_gain + 0.5:
            result["side"] = "radar_side"
        elif total > ERROR_THRESHOLD and arm_gain >= 0.5 and radar_gain >= 0.5:
            result["side"] = "both"
        elif total > ERROR_THRESHOLD:
            result["side"] = "unexplained"
        else:
            result["side"] = "small"

    by_pair = {}
    for arm in ARMS:
        subset = [result for result in results if result["arm"] == arm]
        if not subset:
            by_pair[arm] = dict(n=0)
            continue
        def stats(key):
            values = [result["gains"][key] for result in subset
                      if result["gains"] and result["gains"].get(key) is not None]
            if not values:
                return dict(n=0, mean_m=None, median_m=None, p95_m=None, max_m=None)
            return dict(n=len(values), mean_m=float(np.mean(values)), median_m=float(np.median(values)),
                        p95_m=float(np.percentile(values, 95)), max_m=float(np.max(values)))
        by_pair[arm] = dict(
            n=len(subset),
            bins=dict(Counter(result["bin"] for result in subset)),
            side_counts=dict(Counter(result["side"] for result in subset)),
            total_loss=stats("total_loss"),
            arm_only_gain=stats("arm_only_gain"),
            radar_only_gain=stats("radar_only_gain"),
            interaction=stats("interaction"),
            hh_vs_d_gjk_max_abs_m=max((abs(result["hh_vs_d_gjk_m"]) for result in subset
                                       if result["hh_vs_d_gjk_m"] is not None), default=None),
            monotonic_violations=sum(1 for result in subset if result["monotonic_violations_m"]),
            containment_unknown_rows=sum(1 for result in subset
                                         if any(result["containment_unknown"].values())),
        )
    # Rows with D error <= threshold are not failures; list coverage explicitly.
    excluded = [dict(row_id=row["row_id"], arm=row["arm"], bin=row["bin"],
                     reference_m=float(row["reference_distance_m"]), d_m=float(row["D_distance_m"]),
                     error_mm=(float(row["reference_distance_m"]) - float(row["D_distance_m"])) * 1000)
                for row in rows if row["reference_status"] == "exact" and row["reference_contact"] == "False"
                and row["reference_distance_m"] not in ("", "None") and row["D_distance_m"] not in ("", "None")
                and float(row["reference_distance_m"]) <= CAP
                and float(row["reference_distance_m"]) - float(row["D_distance_m"]) <= ERROR_THRESHOLD]
    output = dict(
        selection=dict(rule="reference exact, reference_contact false, MM <= cap, MM-D > 3 mm",
                       selected=len(results), excluded=len(excluded), excluded_rows=excluded,
                       near_set=args.near, eval_rows=args.rows,
                       note="All selected rows are listed; no cost-limited subsampling."),
        by_pair=by_pair,
        global_side_counts=dict(Counter(result["side"] for result in results)),
        results=results,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mm_hm_mh_hh_attribution.json").write_text(json.dumps(output, indent=1))
    fieldnames = ["row_id", "arm", "bin", "reference_mm", "d_gjk_m", "MM_m", "HM_m", "MH_m", "HH_m",
                  "HH_minus_D_m", "MM_minus_MMstored_m", "arm_only_gain_m", "radar_only_gain_m",
                  "total_loss_m", "interaction_m", "side", "winner_MM", "winner_HM", "winner_MH",
                  "winner_HH", "monotonic_violations"]
    with (out_dir / "mm_hm_mh_hh_attribution.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            gains = result["gains"]
            writer.writerow(dict(
                row_id=result["row_id"], arm=result["arm"], bin=result["bin"],
                reference_mm=result["reference_mm"], d_gjk_m=result["d_gjk_m"],
                MM_m=result["distances_m"]["MM"], HM_m=result["distances_m"]["HM"],
                MH_m=result["distances_m"]["MH"], HH_m=result["distances_m"]["HH"],
                HH_minus_D_m=result["hh_vs_d_gjk_m"], MM_minus_MMstored_m=result["mm_delta_m"],
                arm_only_gain_m=gains.get("arm_only_gain"), radar_only_gain_m=gains.get("radar_only_gain"),
                total_loss_m=gains.get("total_loss"), interaction_m=gains.get("interaction"),
                side=result["side"], winner_MM=result["winners"]["MM"], winner_HM=result["winners"]["HM"],
                winner_MH=result["winners"]["MH"], winner_HH=result["winners"]["HH"],
                monotonic_violations=json.dumps(result["monotonic_violations_m"]),
            ))
    (out_dir / "mm_hm_mh_hh_summary.json").write_text(json.dumps(
        dict(selected=len(results), global_side_counts=dict(Counter(result["side"] for result in results)),
             by_pair=by_pair), indent=1))
    print(json.dumps(dict(selected=len(results), global_side_counts=dict(Counter(result["side"] for result in results)),
                          by_pair={arm: dict(n=entry["n"], side_counts=entry.get("side_counts"),
                                             total_loss=entry.get("total_loss", {}).get("median_m"),
                                             arm_gain=entry.get("arm_only_gain", {}).get("median_m"),
                                             radar_gain=entry.get("radar_only_gain", {}).get("median_m"))
                                   for arm, entry in by_pair.items()}), indent=1))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
