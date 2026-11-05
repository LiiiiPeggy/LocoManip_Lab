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

"""Stage 4.13c: put the three distances side by side and report e = d_proxy - d_reference.

For the priority pairs, on the recorded operating poses:

* ``d_mesh``   -- triangle-mesh to triangle-mesh, the geometry the real robot has;
* ``d_mujoco`` -- what the simulator's convex hulls report, which is what stages 4.11 and 4.12
  called "truth";
* ``d_box``    -- the box proxy the reward currently uses.

The differences between them answer the questions this stage was asked to settle: how much of the
error is the collision model (`d_mujoco - d_mesh`), how much is the proxy's geometry (`d_box -
d_mujoco`), and how much would be the algorithm (GJK against the same hulls; reported as exactly
zero at a tight tolerance by ``convex_distance.py --verify``).

The mesh distance is the min over each side's vertices of the distance to the other's surface: exact
for vertex-face closest features and an upper bound otherwise, which is stated rather than hidden.
Vertices are subsampled for runtime and the sample count is reported.

Usage:
    python scripts/ranger_cr10/validate_distance_reference.py --poses 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh_distance import MeshDistance

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "mujoco/robots/ranger_cr10/scene.xml"
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes"
POSES = ROOT / "outputs/stage411/operating_poses.npz"
SAFETY = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/safety.py"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--poses", type=int, default=300)
ap.add_argument("--vertices", type=int, default=1200, help="vertices per part used for the mesh distance")
ap.add_argument("--out", default="outputs/stage413/c")
args = ap.parse_args()

import mujoco  # noqa: E402

PAIRS = [
    ("cr10_Link3", "cr10_Link5"),
    ("cr10_Link6", "lidar_link0"),
    ("cr10_Link5", "lidar_link0"),
    ("cr10_Link4", "cr10_Link6"),
    ("cr10_Link4", "lidar_link0"),
    ("cr10_Link3", "lidar_link0"),
]
MESHES = {
    "cr10_Link3": ("cr10_meshes/Link3.STL", 1.0),
    "cr10_Link4": ("cr10_meshes/Link4.STL", 1.0),
    "cr10_Link5": ("cr10_meshes/Link5.STL", 1.0),
    "cr10_Link6": ("cr10_meshes/Link6.STL", 1.0),
    "lidar_link0": ("lidar_link0.STL", 1.0),
}
BANDS = [(0.0, 0.005), (0.005, 0.010), (0.010, 0.020)]


def load_safety():
    import importlib.util

    spec = importlib.util.spec_from_file_location("safety_validate", SAFETY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    import torch

    safety = load_safety()
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    bid = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)}
    arm_adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)]
    geom_of = {
        n: next(g for g in range(model.ngeom) if model.geom_bodyid[g] == bid[n] and model.geom_contype[g] != 0)
        for n in MESHES
    }
    meshes = {}
    for name, (rel, scale) in MESHES.items():
        mesh = trimesh.load(str(MESH_DIR / rel), force="mesh", process=False)
        if scale != 1.0:
            mesh.apply_scale(scale)
        meshes[name] = mesh

    queries = {name: MeshDistance(mesh.vertices, mesh.faces) for name, mesh in meshes.items()}
    poses = np.load(POSES)["poses"]
    step = max(1, len(poses) // args.poses)
    poses = poses[::step]
    rng = np.random.default_rng(0)
    subsets = {}
    for name, mesh in meshes.items():
        idx = rng.choice(len(mesh.vertices), size=min(args.vertices, len(mesh.vertices)), replace=False)
        subsets[name] = idx

    rows = []
    for q in poses:
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        for address, value in zip(arm_adr, q):
            data.qpos[address] = value
        mujoco.mj_forward(model, data)
        names = list(dict.fromkeys(safety.ARM_BODY_LIST + safety.PROT_BODY_LIST))
        pos = torch.tensor(np.stack([data.xpos[bid[n]] for n in names])[None].astype(np.float32))
        quat = torch.tensor(np.stack([data.xquat[bid[n]] for n in names])[None].astype(np.float32))

        class Asset:
            body_names = names

            class data:  # noqa: N801
                body_pos_w = pos
                body_quat_w = quat

        box = safety.clearance(Asset, cap=0.10)[0]
        for a, b in PAIRS:
            ga, gb = geom_of[a], geom_of[b]
            d_mujoco = mujoco.mj_geomDistance(model, data, ga, gb, 0.10, None)
            # The STL is in the *link* frame; MuJoCo's geom frame may be rotated relative to it
            # (the mesh is re-expressed in its own frame), so the link pose is what moves it.
            for name, geom in ((a, ga), (b, gb)):
                if not np.allclose(data.geom_xmat[geom].reshape(3, 3), data.xmat[bid[name]].reshape(3, 3), atol=1e-6):
                    pass  # the geom frame differs; the mesh file still follows the link, handled below
            va = meshes[a].vertices[subsets[a]] @ data.xmat[bid[a]].reshape(3, 3).T + data.xpos[bid[a]]
            vb = meshes[b].vertices[subsets[b]] @ data.xmat[bid[b]].reshape(3, 3).T + data.xpos[bid[b]]
            # query the other mesh in its own link frame to keep the index local
            da = queries[b].distance((va - data.xpos[bid[b]]) @ data.xmat[bid[b]].reshape(3, 3))
            db = queries[a].distance((vb - data.xpos[bid[a]]) @ data.xmat[bid[a]].reshape(3, 3))
            d_mesh = float(min(da.min(), db.min()))
            cols = [
                k
                for k, (i, j) in enumerate(safety.SELF_PAIRS)
                if {safety.ARM_BODY_LIST[i], safety.ARM_BODY_LIST[j]} == {a, b}
            ]
            if not cols:
                cols = [
                    len(safety.SELF_PAIRS) + k
                    for k, (i, p) in enumerate(safety.SENSOR_PAIRS)
                    if {safety.ARM_BODY_LIST[i], safety.PROT_GROUPS[p]} == {a, b}
                ]
            if not cols:
                raise SystemExit(f"pair {a} {b} has no column; the pair list and the safety module disagree")
            d_box = float(box[cols].min())
            rows.append(dict(pair=f"{a} {b}", d_mesh=d_mesh, d_mujoco=float(d_mujoco), d_box=d_box))

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.csv").write_text("pair,d_mesh,d_mujoco,d_box\n" + "\n".join(
        f"{r['pair']},{r['d_mesh']:.6f},{r['d_mujoco']:.6f},{r['d_box']:.6f}" for r in rows
    ))

    def summarize(subset, label):
        # beyond ~50 mm the KD-tree approximation in mesh_distance.py dominates and the number
        # stops being a measurement of the collision model, so it is excluded rather than reported
        subset = [r for r in subset if r["d_mesh"] <= 0.05]
        if not subset:
            return None
        mesh = np.array([r["d_mesh"] for r in subset])
        e_hull = np.array([r["d_mujoco"] for r in subset]) - mesh
        e_box = np.array([r["d_box"] for r in subset]) - mesh
        in_band = (mesh >= 0.0) & (mesh <= 0.020)
        result = dict(
            label=label,
            n=len(subset),
            e_hull_median=float(np.median(e_hull)) if len(e_hull) else None,
            e_hull_abs_p95=float(np.percentile(np.abs(e_hull), 95)),
            e_box_median=float(np.median(e_box)),
            e_box_abs_p95=float(np.percentile(np.abs(e_box), 95)),
            e_box_optimistic_max=float(max(0.0, e_box.max())),
            band_n=int(in_band.sum()),
            fn_hull=int(((mesh <= 0.0) & (np.array([r["d_mujoco"] for r in subset]) > 0.0)).sum()),
            fn_box=int(((mesh <= 0.0) & (np.array([r["d_box"] for r in subset]) > 0.0)).sum()),
        )
        print(
            f"[{label}] n={result['n']}\n"
            f"  hull  - mesh : median {result['e_hull_median'] * 1000:+.1f} mm, |e| p95 {result['e_hull_abs_p95'] * 1000:.1f} mm\n"
            f"  box   - mesh : median {result['e_box_median'] * 1000:+.1f} mm, |e| p95 {result['e_box_abs_p95'] * 1000:.1f} mm, "
            f"optimistic max {result['e_box_optimistic_max'] * 1000:+.1f} mm\n"
            f"  contact steps missed: hull {result['fn_hull']}, box {result['fn_box']}"
        )
        return result

    print(f"poses {len(poses)} (every {step}th), vertices per part <= {args.vertices}\n")
    summary = {"all": summarize(rows, "all pairs")}
    for a, b in PAIRS:
        summary[f"{a} {b}"] = summarize([r for r in rows if r["pair"] == f"{a} {b}"], f"{a} <-> {b}")
        for lo, hi in BANDS:
            band = [r for r in rows if r["pair"] == f"{a} {b}" and lo <= r["d_mesh"] < hi]
            if band:
                print(f"    band {lo * 1000:.0f}-{hi * 1000:.0f} mm: n={len(band)}")

    (out / "validation.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out / 'validation.json'} and validation.csv")


if __name__ == "__main__":
    main()
