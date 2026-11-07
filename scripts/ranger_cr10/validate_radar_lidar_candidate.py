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

"""Sampled coverage/overfill audit for the lidar per-component candidate.

Coverage: points sampled on the original lidar surface, distance to the union of component
hulls. Overfill: points sampled on the piece surfaces, unsigned distance to the original mesh;
inside/outside is reported only where the welded source mesh is closed, otherwise unknown. A
point outside the original global hull would violate the replacement boundary; that is checked.

This is a sampled observation, not a global proof.

Run:
    python scripts/ranger_cr10/validate_radar_lidar_candidate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from convex_distance import gjk_distance  # noqa: E402
from mesh_distance import MeshDistance  # noqa: E402
from reference_geometry import MeshRef  # noqa: E402

ROOT = HERE.parents[1]
MESH = ROOT / "rangerboxcr10lidar_description/meshes/lidar_link0.STL"
PIECES = ROOT / "outputs/stage413/b/radar_round2_2026_11_07/candidate_lidar_v1"
OUT = ROOT / "outputs/stage413/b/radar_round2_2026_11_07/candidate_lidar_coverage.json"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--mesh", default=str(MESH))
ap.add_argument("--pieces", default=str(PIECES))
ap.add_argument("--samples", type=int, default=600)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=str(OUT))
args = ap.parse_args()


def sample_triangles(triangles, count, rng):
    if len(triangles) == 0:
        return np.zeros((0, 3))
    index = rng.choice(len(triangles), size=min(count, len(triangles)), replace=True)
    weights = rng.dirichlet([1.0, 1.0, 1.0], size=len(index))
    return np.einsum("ij,ijk->ik", weights, triangles[index])


def load_pieces(directory):
    out = []
    for path in sorted(Path(directory).glob("lidar_link0_p*.npz")):
        data = np.load(path)
        out.append((path.stem, np.asarray(data["vertices"], dtype=np.float64),
                    np.asarray(data["faces"], dtype=np.int64)))
    return out


def union_distance(pieces, points):
    out = np.full(len(points), np.inf)
    for _, vertices, _ in pieces:
        for index, point in enumerate(points):
            result = gjk_distance(point[None, :], vertices, tol=1e-8)
            if result.status == "intersecting":
                out[index] = 0.0
            elif result.status in ("separated", "degenerate") and np.isfinite(result.distance):
                out[index] = min(out[index], result.distance)
    return out


def main() -> None:
    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    mesh_distance = MeshDistance(vertices, faces)
    welded = mesh.copy()
    welded.merge_vertices(merge_tex=False, merge_norm=False)
    welded.remove_unreferenced_vertices()
    mesh_ref = MeshRef(np.asarray(welded.vertices), np.asarray(welded.faces), "lidar")
    pieces = load_pieces(args.pieces)
    if not pieces:
        raise SystemExit("no pieces")
    rng = np.random.default_rng(args.seed)
    mesh_samples = sample_triangles(vertices[faces], args.samples, rng)
    coverage = union_distance(pieces, mesh_samples)
    piece_triangles = [v[f] for _, v, f in pieces if len(f)]
    piece_triangles = np.concatenate(piece_triangles, axis=0) if piece_triangles else np.zeros((0, 3, 3))
    piece_samples = sample_triangles(piece_triangles, args.samples, rng)
    unsigned = mesh_distance.distance(piece_samples)
    outside = []
    for point in piece_samples:
        outcome = mesh_ref.contains(point)
        outside.append(None if outcome is None else bool(outcome))
    outside_flags = np.array([value is False for value in outside])
    unknown_flags = np.array([value is None for value in outside])
    # replacement boundary: every piece point must lie inside the old global hull
    global_hull = ConvexHull(vertices, qhull_options="QJ")
    hull_signed = piece_samples @ global_hull.equations[:, :3].T + global_hull.equations[:, 3]
    record = dict(
        mesh=args.mesh, pieces=args.pieces, n_pieces=len(pieces), samples=args.samples,
        coverage=dict(n=int(len(coverage)), median_m=float(np.median(coverage)),
                      p95_m=float(np.percentile(coverage, 95)), max_m=float(np.max(coverage)),
                      over_0_1mm=int((coverage > 1e-4).sum()), over_0_5mm=int((coverage > 5e-4).sum()),
                      over_1mm=int((coverage > 1e-3).sum())),
        overfill=dict(n=int(len(unsigned)), unsigned_p95_m=float(np.percentile(unsigned, 95)),
                      unsigned_max_m=float(np.max(unsigned)), outside=int(outside_flags.sum()),
                      inside=int(np.array([value is True for value in outside]).sum()),
                      unknown=int(unknown_flags.sum()),
                      outside_distance_p95_m=float(np.percentile(unsigned[outside_flags], 95))
                      if outside_flags.any() else 0.0,
                      outside_distance_max_m=float(np.max(unsigned[outside_flags]))
                      if outside_flags.any() else 0.0,
                      note="inside/outside unknown for a non-watertight source; unsigned distance is reported"),
        replacement_boundary=dict(max_signed_global_hull_m=float(hull_signed.max()),
                                  outside_global_hull=int((hull_signed > 1e-9).sum())),
        note="sampled observation; not a global coverage/overfill proof; the old global lidar hull is not retained",
    )
    Path(args.out).write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
