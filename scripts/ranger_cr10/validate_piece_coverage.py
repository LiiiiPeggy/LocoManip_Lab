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

"""Observed coverage and overfill of a convex-piece cover, on the original mesh.

Two questions, kept separate:

* coverage: for points sampled on the original mesh surface, how far is the nearest point of the
  piece union? A large distance means the cover leaves that surface region uncovered;
* overfill: for points sampled on the piece surfaces, how far outside the original mesh are they?
  An outside point at 5 mm means the cover protrudes.

Both are sampled observations, not a proof of global coverage. The point-to-convex-hull distance
uses GJK with a single query point against each piece; the containment classification welds the STL
first (the raw vertex soup fails the closed-mesh test).

Usage:
    python scripts/ranger_cr10/validate_piece_coverage.py \
        --pieces outputs/stage413/b/correction_v1/link5_t1mm \
        --mesh rangerboxcr10lidar_description/meshes/cr10_meshes/Link5.STL \
        --old outputs/stage413/b/metric20 --out outputs/stage413/b/correction_v1/coverage_link5.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from convex_distance import gjk_distance  # noqa: E402
from mesh_distance import MeshDistance  # noqa: E402
from reference_geometry import MeshRef  # noqa: E402

ROOT = HERE.parents[1]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--pieces", required=True, help="new piece directory")
ap.add_argument("--old", default="outputs/stage413/b/metric20", help="old piece directory for comparison")
ap.add_argument("--mesh", required=True)
ap.add_argument("--samples", type=int, default=600)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()


def sample_triangles(triangles: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(triangles) == 0:
        return np.zeros((0, 3))
    index = rng.choice(len(triangles), size=min(count, len(triangles)), replace=True)
    weights = rng.dirichlet([1.0, 1.0, 1.0], size=len(index))
    return np.einsum("ij,ijk->ik", weights, triangles[index])


def load_pieces_faces(part: str, directory: Path):
    out = []
    for path in sorted(directory.glob(f"{part}_p*.npz")):
        data = np.load(path)
        out.append((path.stem, np.asarray(data["vertices"]), np.asarray(data["faces"], dtype=np.int64)))
    return out


def union_distance(pieces, points: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest convex piece (0 when inside one)."""
    out = np.full(len(points), np.inf)
    for _, vertices, _ in pieces:
        for index, point in enumerate(points):
            result = gjk_distance(point[None, :], vertices, tol=1e-8)
            if result.status == "intersecting":
                out[index] = 0.0
            elif result.status in ("separated", "degenerate") and np.isfinite(result.distance):
                out[index] = min(out[index], result.distance)
    return out


def measure(pieces, mesh, mesh_distance, mesh_ref, mesh_samples, piece_count, rng):
    coverage = union_distance(pieces, mesh_samples)
    triangles = []
    for _, vertices, faces in pieces:
        if len(faces):
            triangles.append(vertices[faces])
    triangles = np.concatenate(triangles, axis=0) if triangles else np.zeros((0, 3, 3))
    if len(triangles):
        piece_points = sample_triangles(triangles, piece_count, rng)
        unsigned = mesh_distance.distance(piece_points)
        outside = np.array([mesh_ref.contains(point) is False for point in piece_points])
        outside_distance = unsigned[outside & np.isfinite(unsigned)]
    else:
        unsigned = np.zeros(0)
        outside_distance = np.zeros(0)
    return dict(
        n_pieces=len(pieces),
        n_vertices=int(sum(len(v) for _, v, _ in pieces)),
        coverage=dict(
            n=int(len(coverage)),
            median_m=float(np.median(coverage)),
            p95_m=float(np.percentile(coverage, 95)),
            max_m=float(np.max(coverage)),
            over_0_5mm=int((coverage > 0.5e-3).sum()),
            over_1mm=int((coverage > 1e-3).sum()),
            over_3mm=int((coverage > 3e-3).sum()),
        ),
        overfill_vertices=dict(
            n=int(len(unsigned)),
            outside=int(outside.sum()) if len(unsigned) else 0,
            outside_distance_p95_m=float(np.percentile(outside_distance, 95)) if len(outside_distance) else 0.0,
            outside_distance_max_m=float(np.max(outside_distance)) if len(outside_distance) else 0.0,
            outside_over_3mm=int((outside_distance > 3e-3).sum()) if len(outside_distance) else 0,
        ),
    )


def main() -> None:
    mesh = trimesh.load(str(ROOT / args.mesh), force="mesh", process=False)
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_faces = np.asarray(mesh.faces)
    mesh_triangles = mesh_vertices[mesh_faces]
    mesh_distance = MeshDistance(mesh_vertices, mesh_faces)
    welded = mesh.copy()
    welded.merge_vertices(merge_tex=False, merge_norm=False)
    welded.remove_unreferenced_vertices()
    mesh_ref = MeshRef(np.asarray(welded.vertices), np.asarray(welded.faces), Path(args.mesh).stem)

    rng = np.random.default_rng(args.seed)
    mesh_samples = sample_triangles(mesh_triangles, args.samples, rng)
    part = f"cr10_{Path(args.mesh).stem}"
    new_pieces = load_pieces_faces(part, ROOT / args.pieces)
    old_pieces = load_pieces_faces(part, ROOT / args.old)
    if not new_pieces:
        raise SystemExit("missing new pieces")

    record = dict(
        mesh=args.mesh,
        pieces=args.pieces,
        old_dir=args.old,
        samples=args.samples,
        new_measure=measure(new_pieces, mesh, mesh_distance, mesh_ref, mesh_samples, args.samples, rng),
        old_measure=(measure(old_pieces, mesh, mesh_distance, mesh_ref, mesh_samples, args.samples, rng)
                     if old_pieces else None),
        note="coverage is sampled on the original mesh surface; overfill is sampled on piece faces",
    )
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
