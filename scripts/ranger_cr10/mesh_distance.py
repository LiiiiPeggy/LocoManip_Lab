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

"""Stage 4.13: point-to-triangle-mesh distance fast enough to serve as the reference.

`trimesh.proximity.ProximityQuery.on_surface` has no spatial index here (`rtree` is not installed),
so it tests every point against every triangle -- minutes per call on a 25k-triangle link, which is
why the first validation runs never finished. This indexes triangle centroids in a KD-tree and
evaluates the exact point-triangle distance for the nearest ``k`` triangles, vectorised over points.

``k`` is a heuristic: a large triangle whose centroid is far can still be the closest, so the
default is generous and `--check` measures the disagreement against trimesh's exhaustive answer on a
sample rather than assuming it away.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def closest_point_on_triangles(points: np.ndarray, tri_a: np.ndarray, tri_b: np.ndarray, tri_c: np.ndarray) -> np.ndarray:
    """Closest point on each triangle to its paired query point (Ericson, *Real-Time Collision*)."""
    ab = tri_b - tri_a
    ac = tri_c - tri_a
    ap = points - tri_a
    d1 = np.einsum("...i,...i->...", ab, ap)
    d2 = np.einsum("...i,...i->...", ac, ap)
    bp = points - tri_b
    d3 = np.einsum("...i,...i->...", ab, bp)
    d4 = np.einsum("...i,...i->...", ac, bp)
    cp = points - tri_c
    d5 = np.einsum("...i,...i->...", ab, cp)
    d6 = np.einsum("...i,...i->...", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    safe = np.where(np.abs(denom) < 1e-20, 1e-20, denom)
    v = vb / safe
    w = vc / safe
    closest = (1.0 - v - w)[..., None] * tri_a + v[..., None] * tri_b + w[..., None] * tri_c

    def set_region(mask, value):
        nonlocal closest
        closest = np.where(mask[..., None], value, closest)

    set_region((d1 <= 0) & (d2 <= 0), tri_a)
    set_region((d3 >= 0) & (d4 <= d3), tri_b)
    set_region((d6 >= 0) & (d5 <= d6), tri_c)
    denom_ab = np.where(np.abs(d1 - d3) < 1e-20, 1e-20, d1 - d3)
    set_region((vc <= 0) & (d1 >= 0) & (d3 <= 0), tri_a + (d1 / denom_ab)[..., None] * ab)
    denom_ac = np.where(np.abs(d2 - d6) < 1e-20, 1e-20, d2 - d6)
    set_region((vb <= 0) & (d2 >= 0) & (d6 <= 0), tri_a + (d2 / denom_ac)[..., None] * ac)
    denom_bc = np.where(np.abs((d4 - d3) + (d5 - d6)) < 1e-20, 1e-20, (d4 - d3) + (d5 - d6))
    set_region((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), tri_b + ((d4 - d3) / denom_bc)[..., None] * (tri_c - tri_b))
    return closest


class MeshDistance:
    """Nearest distance from query points to a triangle mesh, via a KD-tree over triangle centroids."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, k: int = 128):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.k = min(k, len(self.faces))
        self.tree = cKDTree(self.vertices[self.faces].mean(axis=1))

    def distance(self, points: np.ndarray, chunk: int = 20000) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        out = np.empty(len(points))
        for start in range(0, len(points), chunk):
            block = points[start : start + chunk]
            _, index = self.tree.query(block, k=self.k)
            triangles = self.faces[index]  # (P, k, 3)
            a = self.vertices[triangles[..., 0]]
            b = self.vertices[triangles[..., 1]]
            c = self.vertices[triangles[..., 2]]
            query = np.repeat(block[:, None, :], self.k, axis=1)
            closest = closest_point_on_triangles(query, a, b, c)
            out[start : start + chunk] = np.linalg.norm(closest - query, axis=-1).min(axis=1)
        return out

    def signed_outside(self, points: np.ndarray) -> np.ndarray:
        """Distance from the mesh surface, positive only for points outside (approximate sign).

        The sign comes from the dot product with the centroid-to-surface direction, which is enough
        to decide "does this bulge stick out of the robot" without a watertightness assumption.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        _, index = self.tree.query(points, k=self.k)
        triangles = self.faces[index]
        a = self.vertices[triangles[..., 0]]
        b = self.vertices[triangles[..., 1]]
        c = self.vertices[triangles[..., 2]]
        query = np.repeat(points[:, None, :], self.k, axis=1)
        closest = closest_point_on_triangles(query, a, b, c)
        distance = np.linalg.norm(closest - query, axis=-1)
        best = distance.argmin(axis=1)
        normals = np.cross(b - a, c - a)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-18
        chosen = np.take_along_axis(normals, best[:, None, None], axis=1)[:, 0]
        direction = points - closest[np.arange(len(points)), best]
        return np.where(np.einsum("ij,ij->i", direction, chosen) >= 0, distance.min(axis=1), -distance.min(axis=1))


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import trimesh

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", default="rangerboxcr10lidar_description/meshes/cr10_meshes/Link3.STL")
    ap.add_argument("--points", type=int, default=300)
    ap.add_argument("--k", type=int, default=128)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    mesh = trimesh.load(str(root / args.mesh), force="mesh", process=False)
    rng = np.random.default_rng(0)
    samples, _ = trimesh.sample.sample_surface(mesh, args.points, seed=1)
    samples = samples + rng.normal(0, 0.008, samples.shape)
    fast = MeshDistance(mesh.vertices, mesh.faces, k=args.k).distance(samples)
    reference = trimesh.proximity.ProximityQuery(mesh).on_surface(samples)[1]
    difference = np.abs(fast - reference)
    print(
        f"{args.points} points, k={args.k}: max |fast - trimesh| = {difference.max() * 1000:.4f} mm, "
        f"median {np.median(difference) * 1000:.4f} mm"
    )
