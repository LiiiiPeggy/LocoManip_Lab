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

"""Build a small, independent new validation set before adding proxy patch boxes.

Rule (fixed here, before any patch geometry):
  * 1 Gaussian perturbation (sigma 0.03 rad, clipped to limits) per fixed-set row that is a
    per-pair tolerance miss: the 20 Link3 and 11 Link6 rows. Perturbations of different source
    rows are treated as separate evidence; the same path is never densely interpolated.
  * 30 random operating poses not equal to a fixed-set pose (unmodified regions).
  * 10 random uniform joint poses.
The exact reference is recomputed for every row from the original arm/lidar meshes. The fixed
342 rows remain the development regression set; this file is only for transfer/new-FP checks.

Run:
    python scripts/ranger_cr10/build_proxy_new_set.py \
        --out outputs/stage413/b/simple_collision_v2/proxy_new_set.json --workers 10
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from reference_geometry import MeshRef  # noqa: E402

ROOT = HERE.parents[1]
FIXED_ROWS = ROOT / "outputs/stage413/b/simple_collision_v2/near_rows_v2.csv"
FIXED_REF = ROOT / "outputs/stage413/b/radar_near_contact/v3/near_set_containment_v3.json"
POSES = ROOT / "outputs/stage411/operating_poses.npz"
ARM_MESH = ROOT / "rangerboxcr10lidar_description/meshes/cr10_meshes"
LIDAR_MESH = ROOT / "rangerboxcr10lidar_description/meshes/lidar_link0.STL"
OUT = ROOT / "outputs/stage413/b/simple_collision_v2/proxy_new_set.json"
LIMITS = np.array([[-3.92, 0.94], [-1.57, 1.57], [-2.86, 2.86],
                   [-3.14, 3.14], [-3.14, 3.14], [-3.14, 3.14]])
SEED = 20261107
ARMS = [f"cr10_Link{i}" for i in range(1, 7)]
_CACHE = {}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default=str(OUT))
ap.add_argument("--workers", type=int, default=10)
args = ap.parse_args()
rng = np.random.default_rng(SEED)


def welded(path):
    mesh = trimesh.load(str(path), force="mesh", process=False)
    mesh.merge_vertices(merge_tex=False, merge_norm=False)
    mesh.remove_unreferenced_vertices()
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def init_worker():
    if _CACHE:
        return
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(ROOT / "mujoco/robots/ranger_cr10/scene_simple_v2.xml"))
    data = mujoco.MjData(model)
    bodies = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)}
    addresses = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")]
                 for i in range(1, 7)]
    refs = {arm: MeshRef(*welded(ARM_MESH / f"{arm.replace('cr10_', '')}.STL"), arm) for arm in ARMS}
    refs["lidar"] = MeshRef(*welded(LIDAR_MESH), "lidar")
    _CACHE.update(model=model, data=data, bodies=bodies, addresses=addresses, refs=refs)


def evaluate(payload):
    init_worker()
    import mujoco

    model, data = _CACHE["model"], _CACHE["data"]
    q = payload["q"]
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for address, value in zip(_CACHE["addresses"], q):
        data.qpos[address] = float(value)
    mujoco.mj_forward(model, data)
    arm = payload["arm"]
    arm_body = _CACHE["bodies"][arm]
    lidar_body = _CACHE["bodies"]["lidar_link0"]
    ra = data.xmat[arm_body].reshape(3, 3)
    pa = data.xpos[arm_body]
    rl = data.xmat[lidar_body].reshape(3, 3)
    pl = data.xpos[lidar_body]
    rotation = ra.T @ rl
    translation = ra.T @ (pl - pa)
    result = _CACHE["refs"][arm].closest_to_in_frame(
        _CACHE["refs"]["lidar"], rotation, translation, max_clearance=0.06, intersection_filter="aabb"
    )
    if result.status in ("culled", "certified_culled"):
        distance = float(result.clearance)
        lower = distance
        distance_out = None
    elif np.isfinite(result.clearance):
        distance = float(result.clearance)
        lower = None
        distance_out = distance
    else:
        distance = None
        lower = None
        distance_out = None
    if result.intersects:
        bin_name = "overlap"
    elif distance_out is None:
        bin_name = ">=30mm_certified_culled"
    elif distance_out <= 1e-3:
        bin_name = "0-1mm"
    elif distance_out <= 3e-3:
        bin_name = "1-3mm"
    elif distance_out <= 5e-3:
        bin_name = "3-5mm"
    elif distance_out <= 10e-3:
        bin_name = "5-10mm"
    elif distance_out <= 20e-3:
        bin_name = "10-20mm"
    elif distance_out <= 30e-3:
        bin_name = "20-30mm"
    else:
        bin_name = ">=30mm"
    detail = f"{arm}:{result.reason}" if result.reason else f"{arm}:{result.status}"
    return dict(
        payload,
        reference_status=result.status,
        reference_contact=bool(result.intersects),
        reference_distance_m=distance_out,
        reference_lower_bound_m=lower,
        reference_detail=detail,
        reference_triangle_a=int(result.triangle_a),
        reference_seconds=float(result.extra.get("seconds", 0.0)) if result.extra else None,
        bin=bin_name,
    )


def fixed_qs():
    ref = json.load(open(FIXED_REF))["rows"]
    return {tuple(np.round(row["q"], 6)) for row in ref}


def main():
    fixed = list(csv.DictReader(open(FIXED_ROWS)))
    fixed_ref = {row["row_id"]: row for row in json.load(open(FIXED_REF))["rows"]}
    misses = [
        row for row in fixed
        if row["ref_tol"] == "unsafe" and row["new_row_tol"] == "safe" and row["arm"] in ("cr10_Link3", "cr10_Link6")
    ]
    rows = []
    for index, row in enumerate(misses, start=1):
        q = np.asarray(fixed_ref[row["row_id"]]["q"], dtype=float)
        noise = rng.normal(0.0, 0.03, size=6)
        perturbed = np.clip(q + noise, LIMITS[:, 0], LIMITS[:, 1])
        rows.append(dict(row_id=f"N{index:04d}", arm=row["arm"], pair=f"{row['arm']} lidar_link0",
                         q=perturbed.tolist(), source=f"perturb:{row['row_id']}",
                         sample_type="new_perturb", seed=SEED))
    used = fixed_qs()
    poses = np.load(POSES)["poses"]
    candidates = [i for i in range(len(poses)) if tuple(np.round(poses[i], 6)) not in used]
    chosen = rng.choice(candidates, size=min(30, len(candidates)), replace=False)
    for index, pose_index in enumerate(chosen, start=len(rows) + 1):
        rows.append(dict(row_id=f"N{index:04d}", arm=None, pair=None, q=poses[pose_index].tolist(),
                         source=f"operating:{int(pose_index)}", sample_type="new_unmodified", seed=SEED))
    for index in range(10):
        q = rng.uniform(LIMITS[:, 0], LIMITS[:, 1])
        rows.append(dict(row_id=f"N{len(rows)+1:04d}", arm=None, pair=None, q=q.tolist(),
                         source="random_uniform", sample_type="new_unmodified", seed=SEED))

    # For unlabeled rows choose the arm with the minimum v2 simple-box distance to the lidar
    # (a selection heuristic only; the reference below uses the original arm mesh).
    init_worker()
    import mujoco

    model, data = _CACHE["model"], _CACHE["data"]
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i): i for i in range(model.ngeom)}
    for row in rows:
        if row["arm"] is not None:
            continue
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        for address, value in zip(_CACHE["addresses"], row["q"]):
            data.qpos[address] = float(value)
        mujoco.mj_forward(model, data)
        distances = {}
        for arm in ARMS:
            geom = f"{arm}_col_simple"
            distances[arm] = float(mujoco.mj_geomDistance(model, data, names[geom], names["lidar_link0_col"], 0.3, None))
        row["arm"] = min(distances, key=distances.get)
        row["pair"] = f"{row['arm']} lidar_link0"
        row["selection_mujoco_m"] = distances[row["arm"]]

    payloads = [dict(row_id=row["row_id"], arm=row["arm"], pair=row["pair"], q=row["q"],
                     source=row["source"], sample_type=row["sample_type"], seed=row["seed"])
                for row in rows]
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("fork"),
        initializer=init_worker,
    ) as executor:
        evaluated = list(executor.map(evaluate, payloads, chunksize=1))
    result = {
        "rule": {
            "fixed_set": str(FIXED_ROWS.relative_to(ROOT)),
            "seed": SEED,
            "perturb_sigma_rad": 0.03,
            "perturb_one_per_source": True,
            "source_misses": {"Link3": sum(1 for row in misses if row["arm"] == "cr10_Link3"),
                              "Link6": sum(1 for row in misses if row["arm"] == "cr10_Link6")},
            "operating_random": 30,
            "uniform_random": 10,
            "dense_path_interpolation": False,
            "note": "fixed 342 rows are the development regression set; this set is for transfer/new FP",
        },
        "rows": evaluated,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    from collections import Counter
    print(json.dumps({
        "rows": len(evaluated),
        "by_source": dict(Counter(row["source"].split(":")[0] for row in evaluated)),
        "bins": dict(Counter(row["bin"] for row in evaluated)),
        "contact": sum(1 for row in evaluated if row["reference_contact"]),
        "unknown": sum(1 for row in evaluated
                       if row["reference_status"] not in ("exact", "culled", "certified_culled")),
        "out": str(args.out),
    }, indent=1))


if __name__ == "__main__":
    main()
