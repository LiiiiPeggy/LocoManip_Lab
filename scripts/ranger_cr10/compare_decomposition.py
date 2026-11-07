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

"""Stage 4.13b: does a convex decomposition of Link3/Link5 beat the single hull?

Local trial, not a full acceptance set. Three representations are compared against the triangle-mesh
reference on the six known anomalous poses **and** on separately generated near-contact and overlap
configurations:

    mesh      the original triangles (reference)
    hull      one convex hull per link -- what MuJoCo collides
    pieces    the CoACD decomposition saved by `decompose_coacd.py`, each piece loaded on its own

Metrics per case: unsigned separation, intersection flag, whether the representation agrees with the
reference, piece and vertex counts, query time, and any failed query. The six anomalies can only show
that a known false alarm is removed; the added near/overlap cases are what can catch a decomposition
that introduced a miss, so the two are reported separately.

Usage:
    python scripts/ranger_cr10/compare_decomposition.py --configs 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/ranger_cr10"))

from convex_distance import gjk_distance  # noqa: E402
from reference_geometry import MeshRef  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out", default="outputs/stage413/b")
ap.add_argument("--near", type=int, default=30, help="near-contact and overlap cases to generate")
ap.add_argument("--pieces", default="outputs/stage413/b/decomposition", help="directory of CoACD pieces")
args = ap.parse_args()

SCENE = ROOT / "mujoco/robots/ranger_cr10/scene.xml"
POSES = ROOT / "outputs/stage411/operating_poses.npz"
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes/cr10_meshes"

import mujoco  # noqa: E402


def load_pieces(part: str, directory: Path) -> list[MeshRef]:
    out = []
    for path in sorted(directory.glob(f"{part}_p*.npz")):
        data = np.load(path)
        out.append(MeshRef(data["vertices"], data["faces"], name=path.stem))
    return out


def union_query(pieces_a, pieces_b, rotation, translation, cap):
    """Min separation over the piece pairs, or an intersection, with the offending pair named.

    A CoACD piece is convex, so GJK on its vertices is exact for separation and answers intersection
    directly. The first version reused the triangle-mesh BVH for every piece pair -- 65 pairs of
    mesh-mesh queries -- which is what made a single case take minutes.
    """
    best, best_pair, pairs, failed = np.inf, "", 0, 0
    t0 = time.time()
    for name_a, vertices_a in pieces_a:
        for name_b, vertices_b in pieces_b:
            result = gjk_distance(vertices_a, vertices_b @ rotation.T + translation, tol=1e-8)
            pairs += 1
            if result.status in ("degenerate", "max_iterations"):
                failed += 1
                continue
            if result.status == "intersecting":
                return 0.0, True, failed, f"{name_a}|{name_b}", time.time() - t0, pairs
            if result.distance < best:
                best, best_pair = float(result.distance), f"{name_a}|{name_b}"
    return min(best, cap), False, failed, best_pair, time.time() - t0, pairs


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    bid = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)}
    adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)]
    poses = np.load(POSES)["poses"]

    mesh3 = trimesh.load(str(MESH_DIR / "Link3.STL"), force="mesh", process=False)
    mesh5 = trimesh.load(str(MESH_DIR / "Link5.STL"), force="mesh", process=False)
    ref3 = MeshRef(np.asarray(mesh3.vertices), np.asarray(mesh3.faces), "Link3")
    ref5 = MeshRef(np.asarray(mesh5.vertices), np.asarray(mesh5.faces), "Link5")
    hull3 = MeshRef(np.asarray(mesh3.convex_hull.vertices), np.asarray(mesh3.convex_hull.faces), "Link3-hull")
    hull5 = MeshRef(np.asarray(mesh5.convex_hull.vertices), np.asarray(mesh5.convex_hull.faces), "Link5-hull")
    pieces_dir = ROOT / args.pieces
    pieces3 = [(p.name, p.vertices) for p in load_pieces("cr10_Link3", pieces_dir)]
    pieces5 = [(p.name, p.vertices) for p in load_pieces("cr10_Link5", pieces_dir)]
    if not pieces3 or not pieces5:
        raise SystemExit(f"no decomposition pieces under {pieces_dir}; run decompose_coacd.py first")
    print(f"pieces: Link3 {len(pieces3)} ({sum(len(v) for _, v in pieces3)} vertices), "
          f"Link5 {len(pieces5)} ({sum(len(v) for _, v in pieces5)} vertices)")

    # the six anomalous poses
    g3 = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == bid["cr10_Link3"] and model.geom_contype[g] != 0)
    g5 = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == bid["cr10_Link5"] and model.geom_contype[g] != 0)
    anomalies = []
    for index, q in enumerate(poses):
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        for address, value in zip(adr, q):
            data.qpos[address] = value
        mujoco.mj_forward(model, data)
        if mujoco.mj_geomDistance(model, data, g3, g5, 0.02, None) <= 0:
            anomalies.append((f"anomaly_{index}", q.copy()))
            if len(anomalies) >= 6:
                break

    # independent near-contact and overlap cases: shift Link3's pose relative to Link5 along the
    # axis between them, so the pair moves from a few millimetres to a real overlap
    near = []
    base = poses[0]
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for address, value in zip(adr, base):
        data.qpos[address] = value
    mujoco.mj_forward(model, data)
    axis = data.xpos[bid["cr10_Link5"]] - data.xpos[bid["cr10_Link3"]]
    axis /= np.linalg.norm(axis)
    for step, offset in enumerate(np.linspace(-0.04, 0.02, args.near)):
        near.append((f"near_{step}_{offset * 1000:+.0f}mm", base.copy(), offset * axis))

    rows = []
    for label, q, shift in [(name, q, np.zeros(3)) for name, q in anomalies] + near:
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        for address, value in zip(adr, q):
            data.qpos[address] = value
        mujoco.mj_forward(model, data)
        r3, p3 = data.xmat[bid["cr10_Link3"]].reshape(3, 3), data.xpos[bid["cr10_Link3"]]
        r5, p5 = data.xmat[bid["cr10_Link5"]].reshape(3, 3), data.xpos[bid["cr10_Link5"]]
        # the shift is applied to Link5 in the world, then expressed in Link3's frame
        p5 = p5 + shift
        rotation, translation = r3.T @ r5, r3.T @ (p5 - p3)

        t0 = time.time()
        truth = ref3.closest_to_in_frame(ref5, rotation, translation, max_clearance=0.05)
        mesh_time = time.time() - t0
        hull = hull3.closest_to_in_frame(hull5, rotation, translation, max_clearance=0.05)
        pieces_distance, pieces_intersect, failed, best_pair, pieces_time, pairs = union_query(
            pieces3, pieces5, rotation, translation, 0.05
        )
        pieces_status = "failed" if failed and not np.isfinite(pieces_distance) else "exact"
        rows.append(dict(
            case=label,
            reference=float(truth.clearance) if truth.status == "exact" else float("nan"),
            reference_intersects=bool(truth.intersects),
            reference_status=truth.status,
            hull_distance=float(hull.clearance) if hull.status == "exact" else float("nan"),
            hull_intersects=bool(hull.intersects),
            hull_error=float(hull.clearance - truth.clearance) if hull.status == "exact" and truth.status == "exact" else float("nan"),
            pieces_distance=pieces_distance,
            pieces_intersects=pieces_intersect,
            pieces_error=pieces_distance - float(truth.clearance) if truth.status == "exact" else float("nan"),
            pieces_status=pieces_status,
            mesh_time=mesh_time,
            pieces_time=pieces_time,
        ))
        print(f"  {label:22s} ref {rows[-1]['reference'] * 1000:7.1f} mm (hit {truth.intersects}) | "
              f"hull {rows[-1]['hull_distance'] * 1000:7.1f} | pieces {pieces_distance * 1000:7.1f} "
              f"({pieces_status})")

    def summary(name, subset):
        values = np.array([r[name] for r in subset])
        finite = values[np.isfinite(values)]
        if not len(finite):
            return None
        return dict(n=int(len(finite)), median=float(np.median(finite)), p95=float(np.percentile(np.abs(finite), 95)),
                    max=float(np.max(np.abs(finite))))

    report = {}
    for group, subset in (("anomalies", [r for r in rows if r["case"].startswith("anomaly")]),
                          ("near_overlap", [r for r in rows if r["case"].startswith("near")])):
        miss_hull = sum(1 for r in subset if r["reference_intersects"] and not r["hull_intersects"])
        miss_pieces = sum(1 for r in subset if r["reference_intersects"] and not r["pieces_intersects"])
        false_hull = sum(1 for r in subset if not r["reference_intersects"] and r["hull_intersects"])
        false_pieces = sum(1 for r in subset if not r["reference_intersects"] and r["pieces_intersects"])
        report[group] = dict(
            n=len(subset),
            hull_error=summary("hull_error", subset),
            pieces_error=summary("pieces_error", subset),
            hull_miss=miss_hull,
            pieces_miss=miss_pieces,
            hull_false=sum(1 for r in subset if not r["reference_intersects"] and r["hull_intersects"]),
            pieces_false=sum(1 for r in subset if not r["reference_intersects"] and r["pieces_intersects"]),
            reference_contacts=sum(1 for r in subset if r["reference_intersects"]),
        )
        print(f"\n[{group}] n={len(subset)}, reference contacts {report[group]['reference_contacts']}")
        for name in ("hull_error", "pieces_error"):
            s = report[group][name]
            if s:
                print(f"  {name:12s} median {s['median'] * 1000:+.1f} mm  |e| p95 {s['p95'] * 1000:.1f} mm  max {s['max'] * 1000:.1f}")
        print(f"  missed contacts: hull {miss_hull}, pieces {miss_pieces}")
        print(f"  false contacts:  hull {false_hull}, pieces {false_pieces}")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "decomposition_comparison.json").write_text(json.dumps(dict(rows=rows, summary=report), indent=1))
    print(f"\nwrote {out / 'decomposition_comparison.json'}")


if __name__ == "__main__":
    main()
