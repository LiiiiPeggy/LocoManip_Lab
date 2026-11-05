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

"""Stage 4.13a: split every link into convex pieces whose union tracks the original mesh.

`measure_hull_overfill.py` showed the current collision model -- one convex hull per link -- is
13-20 mm proud of the mesh at the median and up to 60 mm at the tail, because a curved link and a
tower with fins do not fit their hulls. That error is larger than anything the box proxy added, so
the geometry is what needs fixing first.

There is no CoACD or VHACD in this environment (`trimesh` has the bindings but neither backend is
installed), so this is a simple recursive plane split: cut the mesh with the plane orthogonal to its
longest axis at the point that balances surface area, and repeat on both halves until the piece's
hull is within `--tol` of the piece or the depth limit is reached. Each piece is then a convex hull
that is saved on its own -- importing the pieces back must not re-merge them into one hull, which is
exactly the mistake that made the current model so loose.

Fidelity is measured against the mesh, not against the decomposition: the acceptance number is
"how far is the hull surface from the original mesh surface".

Usage:
    python scripts/ranger_cr10/decompose_convex_parts.py --tol 0.003 --max_pieces 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--tol", type=float, default=0.003, help="target hull-vs-mesh distance, metres")
ap.add_argument("--max_pieces", type=int, default=12)
ap.add_argument("--samples", type=int, default=4000)
ap.add_argument("--out", default="outputs/stage413/a/decomposition")
args = ap.parse_args()

PARTS = {
    "cr10_Link1": ("cr10_meshes/Link1.STL", 1.0),
    "cr10_Link2": ("cr10_meshes/Link2.STL", 1.0),
    "cr10_Link3": ("cr10_meshes/Link3.STL", 1.0),
    "cr10_Link4": ("cr10_meshes/Link4.STL", 1.0),
    "cr10_Link5": ("cr10_meshes/Link5.STL", 1.0),
    "cr10_Link6": ("cr10_meshes/Link6.STL", 1.0),
    "gripper_base_link": ("ag95_meshes/ag95_base_link.STL", 0.001),
    "lidar_link0": ("lidar_link0.STL", 1.0),
}


def overfill(pieces, mesh: trimesh.Trimesh, samples: int) -> float:
    """95th percentile of how far the union's surface bulges *outside* the mesh.

    The cut faces of a decomposition lie inside the solid, and their distance to the mesh surface is
    tens of millimetres -- counting them would say a decomposition is worse than the single hull it
    replaced (measured: Link1 12 mm -> 38 mm by that mistake). Only points outside the mesh can make
    the robot collide earlier than it should, so only those are counted.

    Coverage needs no measurement: clipping assigns every triangle to exactly one piece and each
    piece's hull contains its triangles, so the union contains the mesh by construction.
    """
    hull_points = []
    for piece in pieces:
        points, _ = trimesh.sample.sample_surface(piece.convex_hull, samples, seed=0)
        hull_points.append(points)
    points = np.concatenate(hull_points)
    outside = ~mesh.contains(points)
    if not outside.any():
        return 0.0
    _, distance, _ = trimesh.proximity.ProximityQuery(mesh).on_surface(points[outside])
    return float(np.percentile(distance, 95))


def clip_halfspace(vertices: np.ndarray, faces: np.ndarray, normal: np.ndarray, origin: np.ndarray):
    """Split a triangle soup by a plane, returning the two halves.

    Written here rather than calling `trimesh.intersections.slice_mesh_plane`, which imports
    shapely even when capping is off, and shapely is not installed in this environment -- adding a
    dependency to the training environment for a measurement is not worth it.
    """
    dist = (vertices - origin) @ normal
    keep = {True: ([], []), False: ([], [])}
    vertex_map = {True: {}, False: {}}

    def add(side: bool, point: np.ndarray) -> int:
        store = keep[side][0]
        key = tuple(np.round(point, 9))
        index = vertex_map[side].get(key)
        if index is None:
            index = len(store)
            store.append(point)
            vertex_map[side][key] = index
        return index

    for face in faces:
        pts = vertices[face]
        d = dist[face]
        on_positive = d >= -1e-12
        on_negative = d <= 1e-12
        if on_positive.all():
            keep[True][1].append([add(True, pts[0]), add(True, pts[1]), add(True, pts[2])])
        elif on_negative.all():
            keep[False][1].append([add(False, pts[0]), add(False, pts[1]), add(False, pts[2])])
        else:
            # Sutherland-Hodgman: walk the edges, emitting the part on each side
            for side, sign in ((True, 1.0), (False, -1.0)):
                polygon = []
                for i in range(3):
                    a, b = pts[i], pts[(i + 1) % 3]
                    da, db = d[i] * sign, d[(i + 1) % 3] * sign
                    if da >= -1e-12:
                        polygon.append(a)
                    if (da > 1e-12 and db < -1e-12) or (da < -1e-12 and db > 1e-12):
                        t = da / (da - db)
                        polygon.append(a + t * (b - a))
                for i in range(1, len(polygon) - 1):
                    keep[side][1].append([add(side, polygon[0]), add(side, polygon[i]), add(side, polygon[i + 1])])
    out = []
    for side in (True, False):
        verts = np.array(keep[side][0]) if keep[side][0] else np.zeros((0, 3))
        faces_side = np.array(keep[side][1], dtype=np.int64) if keep[side][1] else np.zeros((0, 3), dtype=np.int64)
        out.append(trimesh.Trimesh(vertices=verts, faces=faces_side, process=False) if len(faces_side) else None)
    return tuple(out)


def split(piece: trimesh.Trimesh) -> tuple[trimesh.Trimesh, trimesh.Trimesh] | None:
    """Cut at the median of the longest axis; None when the cut produces nothing usable."""
    extent = piece.bounds[1] - piece.bounds[0]
    axis = int(np.argmax(extent))
    normal = np.zeros(3)
    normal[axis] = 1.0
    # split at the surface-area-weighted median rather than the midpoint: an L-shaped part is not
    # balanced geometrically, and an unbalanced cut just recurses on the same side
    centroids = piece.triangles_center[:, axis]
    areas = piece.area_faces
    order = np.argsort(centroids)
    cumulative = np.cumsum(areas[order]) / areas.sum()
    point = float(np.interp(0.5, cumulative, centroids[order]))
    origin = normal * point
    a, b = clip_halfspace(np.asarray(piece.vertices), np.asarray(piece.faces), normal, origin)
    if a is None or b is None:
        return None
    if len(a.faces) < 8 or len(b.faces) < 8:
        return None
    return a, b


def main() -> None:
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, (rel, scale) in PARTS.items():
        mesh = trimesh.load(str(MESH_DIR / rel), force="mesh", process=False)
        if scale != 1.0:
            mesh.apply_scale(scale)
        whole = overfill([mesh], mesh, args.samples)
        pieces = [mesh]
        depth = 0
        while len(pieces) < args.max_pieces:
            worst = max(range(len(pieces)), key=lambda i: overfill([pieces[i]], mesh, args.samples // 4))
            if overfill(pieces, mesh, args.samples) <= args.tol:
                break
            halves = split(pieces[worst])
            if halves is None:
                break
            pieces[worst : worst + 1] = list(halves)
            depth += 1
        records = []
        for index, piece in enumerate(pieces):
            hull = piece.convex_hull
            records.append(
                dict(
                    index=index,
                    vertices=len(hull.vertices),
                    faces=len(hull.faces),
                    volume=float(hull.volume),
                    overfill_p95=overfill([piece], mesh, args.samples // 2),
                    points=hull.vertices.tolist(),
                )
            )
            # one file per piece: loading them back must not re-merge the pieces into one hull
            np.savez_compressed(
                out / f"{name}_p{index}.npz",
                vertices=hull.vertices.astype(np.float64),
                faces=hull.faces.astype(np.int32),
            )
        entry = dict(
            part=name,
            pieces=len(records),
            splits=depth,
            whole_hull_overfill_p95=whole,
            decomposed_overfill_p95=overfill(pieces, mesh, args.samples),
            vertices_total=sum(r["vertices"] for r in records),
            per_piece=records,
        )
        summary.append(entry)
        print(
            f"{name:20s} pieces {len(records):2d}  hull p95 {whole * 1000:6.1f} mm -> "
            f"{entry['decomposed_overfill_p95'] * 1000:5.1f} mm   verts {entry['vertices_total']:5d}"
        )

    (out / "decomposition.json").write_text(
        json.dumps(dict(tol=args.tol, max_pieces=args.max_pieces, units="metres", parts=summary), indent=1)
    )
    print(f"\nwrote {out / 'decomposition.json'}")
    print(f"acceptance: tolerance {args.tol * 1000:.1f} mm")
    for entry in summary:
        ok = entry["decomposed_overfill_p95"] <= args.tol
        print(f"  {entry['part']:20s} {'PASS' if ok else 'FAIL'}  {entry['decomposed_overfill_p95'] * 1000:.1f} mm")


if __name__ == "__main__":
    main()
