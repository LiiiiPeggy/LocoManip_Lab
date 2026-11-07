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

"""Stage 4.13b0: a reference mesh-to-mesh query with a status, not just a number.

What the previous reference could not do, and this does:

* **edge-edge** closest features (the old one only looked at vertex-to-surface, which reads a
  perpendicular crossing of two boxes as ~90 mm of clearance);
* **intersection**, tested separately from distance, because a zero distance is also what tangency
  produces;
* **containment**, tested by the winding number: a body fully inside another has no surface
  intersection and a positive surface distance, and is still a collision;
* **a status**, so a failed or inconclusive query cannot masquerade as a large safe distance.

Pruning is a provable radius bound rather than "the k nearest triangle centroids": every triangle
closer than the current best upper bound ``U`` has its centroid within ``U + max_circumradius`` of
the query triangle's centroid, so a ball query at that radius cannot miss it. A centroid KNN with
k = 128 can, and does -- it is what produced the earlier, unverified numbers.

FCL is not required; when the optional ``fcl`` module is importable it is used to cross-check the
result (`--cross_check`), never to replace the answer.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.spatial import cKDTree
import trimesh

from triangle_distance import (
    batch_triangle_distance,
    triangle_triangle_distance,
    triangle_triangle_intersect,
)

STATUS_OK = "exact"              # every leaf that could matter was evaluated
STATUS_CULLED = "certified_culled"  # pruned by a lower bound that exceeds the query range
STATUS_FAILED = "failed"
STATUS_NO_CANDIDATES = "no_candidates"
STATUS_DEGENERATE = "degenerate_mesh"
STATUS_UNKNOWN_CONTAINMENT = "containment_unknown"


@dataclass
class QueryResult:
    status: str
    clearance: float  # non-negative separation; 0 when intersecting
    intersects: bool
    witness_a: np.ndarray | None = None
    witness_b: np.ndarray | None = None
    triangle_a: int = -1
    triangle_b: int = -1
    reason: str = ""
    extra: dict = field(default_factory=dict)



class _Node:
    __slots__ = ("centre", "radius", "left", "right", "triangles")

    def __init__(self, centre, radius, left=None, right=None, triangles=None):
        self.centre = centre
        self.radius = radius
        self.left = left
        self.right = right
        self.triangles = triangles


def _build(centroids: np.ndarray, circumradius: np.ndarray, indices: np.ndarray, leaf: int) -> _Node:
    centre = centroids[indices].mean(axis=0)
    radius = float(np.linalg.norm(centroids[indices] - centre, axis=1).max() + circumradius[indices].max())
    if len(indices) <= leaf:
        return _Node(centre, radius, triangles=indices)
    extent = centroids[indices].max(axis=0) - centroids[indices].min(axis=0)
    axis = int(np.argmax(extent))
    order = indices[np.argsort(centroids[indices, axis])]
    half = len(order) // 2
    return _Node(centre, radius, _build(centroids, circumradius, order[:half], leaf), _build(centroids, circumradius, order[half:], leaf))


class MeshRef:
    """A triangle mesh with the indexes and warnings the reference query needs."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, name: str = "", unit_scale: float = 1.0):
        self.name = name
        self.unit_scale = unit_scale
        self.vertices = np.asarray(vertices, dtype=np.float64) * unit_scale
        self.faces = np.asarray(faces, dtype=np.int64)
        self.warnings: list[str] = []
        if len(self.faces) == 0:
            self.warnings.append("no_faces")
        areas = self.triangle_areas
        if len(areas) and (areas <= 1e-14).any():
            self.warnings.append(f"degenerate_faces={int((areas <= 1e-14).sum())}")
        self.triangles = self.vertices[self.faces] if len(self.faces) else np.zeros((0, 3, 3))
        if len(self.triangles):
            self.centroids = self.triangles.mean(axis=1)
            self.circumradius = np.linalg.norm(self.triangles - self.centroids[:, None, :], axis=2).max(axis=1)
        else:
            self.centroids = np.zeros((0, 3))
            self.circumradius = np.zeros(0)
        self.max_radius = float(self.circumradius.max()) if len(self.circumradius) else 0.0
        # Per-triangle AABBs for a conservative intersection candidate screen. A pair that
        # intersects has overlapping AABBs; the screen therefore cannot skip an intersection.
        self.tri_lo = self.triangles.min(axis=1) if len(self.triangles) else np.zeros((0, 3))
        self.tri_hi = self.triangles.max(axis=1) if len(self.triangles) else np.zeros((0, 3))
        self.aabb_lo = self.vertices.min(axis=0) if len(self.vertices) else np.zeros(3)
        self.aabb_hi = self.vertices.max(axis=0) if len(self.vertices) else np.zeros(3)
        self.tree = cKDTree(self.centroids) if len(self.centroids) else None
        # one static tree per mesh: a rigid transform keeps every node radius and only moves its
        # centre, so the same tree answers for every pose
        self.bvh = _build(self.centroids, self.circumradius, np.arange(len(self.centroids)), 8) if len(self.centroids) else None
        self.watertight = self._watertight()
        self._component_points: np.ndarray | None = None

    @property
    def component_points(self) -> np.ndarray:
        """One representative point per connected component, in this mesh's frame.

        A single first vertex is not enough for a multi-component mesh: the contained
        component may not contain the first vertex. The component split is done on a welded
        copy; the distance query still uses the original triangle coordinates.
        """
        if self._component_points is None:
            if len(self.faces) == 0:
                self._component_points = np.zeros((0, 3), dtype=np.float64)
            else:
                mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)
                mesh.merge_vertices(merge_tex=False, merge_norm=False)
                mesh.remove_unreferenced_vertices()
                parts = mesh.split(only_watertight=False)
                self._component_points = np.asarray([part.vertices[0] for part in parts if len(part.vertices)],
                                                   dtype=np.float64)
        return self._component_points

    @property
    def triangle_areas(self) -> np.ndarray:
        if len(self.faces) == 0:
            return np.zeros(0)
        tri = self.vertices[self.faces]
        return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)

    def _watertight(self) -> bool:
        """Every undirected edge used exactly twice -- the usual closed-manifold check."""
        if len(self.faces) == 0:
            return False
        edges = np.sort(
            np.concatenate([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]]), axis=1
        )
        _, counts = np.unique(edges, axis=0, return_counts=True)
        return bool((counts == 2).all())

    # -- containment -------------------------------------------------------------
    def winding_number(self, point: np.ndarray) -> float:
        """Solid angle / 4 pi: |w| > 0.5 means the point is inside a closed mesh."""
        if len(self.triangles) == 0:
            return 0.0
        a = self.triangles[:, 0] - point
        b = self.triangles[:, 1] - point
        c = self.triangles[:, 2] - point
        la = np.linalg.norm(a, axis=1)
        lb = np.linalg.norm(b, axis=1)
        lc = np.linalg.norm(c, axis=1)
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = (
            la * lb * lc
            + np.einsum("ij,ij->i", a, b) * lc
            + np.einsum("ij,ij->i", a, c) * lb
            + np.einsum("ij,ij->i", b, c) * la
        )
        angles = 2.0 * np.arctan2(numerator, denominator)
        return float(angles.sum() / (4.0 * np.pi))

    def contains(self, point: np.ndarray) -> bool | None:
        """Inside test, or None when the mesh is not closed and the sign cannot be trusted."""
        if not self.watertight:
            return None
        return abs(self.winding_number(point)) > 0.5

    # -- distance ----------------------------------------------------------------

    # -- BVH dual traversal -------------------------------------------------------
    def closest_to_in_frame(self, other: "MeshRef", rotation: np.ndarray, translation: np.ndarray,
                            max_clearance: float = 0.05, leaf_pairs_limit: int = 2000000,
                            intersection_filter: str = "aabb") -> QueryResult:
        """Exact distance with the other mesh moved into this mesh's frame by (rotation, translation).

        The trees are static: a rigid transform leaves every bounding-sphere radius untouched and only
        moves centres, so both meshes keep the index they were built with and only the visited leaves
        have their triangles transformed. Node pairs are pruned on `|cA - cB| - rA - rB > best`, which
        is a lower bound on any triangle distance inside them -- that is what makes a cull *certified*
        rather than a guess.
        """
        if self.bvh is None or other.bvh is None:
            return QueryResult(STATUS_DEGENERATE, float("nan"), False, reason="empty_mesh")
        rotation = np.asarray(rotation, dtype=np.float64)
        translation = np.asarray(translation, dtype=np.float64)
        best = QueryResult(STATUS_OK, max_clearance, False,
                           extra={"leaf_pairs": 0, "aabb_candidates": 0, "near_zero_candidates": 0,
                                  "exact_tests": 0})
        evaluated = 0
        stack = [(self.bvh, other.bvh)]
        while stack:
            node_a, node_b = stack.pop()
            centre_b = rotation @ node_b.centre + translation
            gap = float(np.linalg.norm(node_a.centre - centre_b)) - node_a.radius - node_b.radius
            if gap > best.clearance:
                continue  # certified: nothing in this pair can beat the current best
            if node_a.triangles is not None and node_b.triangles is not None:
                moved = other.triangles[node_b.triangles] @ rotation.T + translation
                triangles_a = self.triangles[node_a.triangles]
                if intersection_filter == "aabb":
                    # Conservative intersection candidates: triangle AABBs. A true intersection has
                    # overlapping AABBs, so this screen cannot skip it; the 1e-6 m allowance only
                    # absorbs float rounding at a tangency. The exact predicate then decides.
                    lo_a = self.tri_lo[node_a.triangles]
                    hi_a = self.tri_hi[node_a.triangles]
                    lo_b = moved.min(axis=1)
                    hi_b = moved.max(axis=1)
                    dx = np.maximum(np.maximum(lo_a[:, None, 0] - hi_b[None, :, 0],
                                               lo_b[None, :, 0] - hi_a[:, None, 0]), 0.0)
                    dy = np.maximum(np.maximum(lo_a[:, None, 1] - hi_b[None, :, 1],
                                               lo_b[None, :, 1] - hi_a[:, None, 1]), 0.0)
                    dz = np.maximum(np.maximum(lo_a[:, None, 2] - hi_b[None, :, 2],
                                               lo_b[None, :, 2] - hi_a[:, None, 2]), 0.0)
                    candidates = np.argwhere(dx * dx + dy * dy + dz * dz <= 1e-12)
                    best.extra["aabb_candidates"] += int(len(candidates))
                    for candidate in candidates:
                        i = int(node_a.triangles[int(candidate[0])])
                        j = int(node_b.triangles[int(candidate[1])])
                        best.extra["exact_tests"] += 1
                        if triangle_triangle_intersect(self.triangles[i], other.triangles[j] @ rotation.T + translation):
                            return QueryResult(STATUS_OK, 0.0, True, reason="surface_intersection",
                                               triangle_a=i, triangle_b=j, extra=best.extra)
                a_rep = np.repeat(triangles_a, len(moved), axis=0)
                b_rep = np.tile(moved, (len(triangles_a), 1, 1))
                distances = batch_triangle_distance(a_rep, b_rep)
                if intersection_filter != "aabb":
                    near_zero = np.flatnonzero(distances < 0.5e-3)
                    best.extra["near_zero_candidates"] += int(len(near_zero))
                    for flat in near_zero:
                        i = int(node_a.triangles[flat // len(moved)])
                        j = int(node_b.triangles[flat % len(moved)])
                        best.extra["exact_tests"] += 1
                        if triangle_triangle_intersect(self.triangles[i], other.triangles[j] @ rotation.T + translation):
                            return QueryResult(STATUS_OK, 0.0, True, reason="surface_intersection",
                                               triangle_a=i, triangle_b=j, extra=best.extra)
                evaluated += len(distances)
                if evaluated > leaf_pairs_limit:
                    return QueryResult(STATUS_FAILED, float("nan"), False, reason="leaf_pair_budget")
                smallest = float(distances.min())
                if smallest < best.clearance:
                    flat = int(distances.argmin())
                    i = int(node_a.triangles[flat // len(moved)])
                    j = int(node_b.triangles[flat % len(moved)])
                    result = triangle_triangle_distance(self.triangles[i], other.triangles[j] @ rotation.T + translation)
                    if result.intersects:
                        return QueryResult(STATUS_OK, 0.0, True, result.point_a, result.point_b, i, j,
                                           "surface_intersection", best.extra)
                    best = QueryResult(STATUS_OK, float(result.distance), False, result.point_a, result.point_b,
                                       i, j, "", best.extra)
                continue
            # split the node with the larger radius, or the one that is not a leaf
            if node_b.triangles is None and (node_a.triangles is not None or node_b.radius >= node_a.radius):
                stack.append((node_a, node_b.left))
                stack.append((node_a, node_b.right))
            else:
                stack.append((node_a.left, node_b))
                stack.append((node_a.right, node_b))
        best.extra["leaf_pairs"] = evaluated
        # Containment is a separate positive answer: a body fully inside another can have no
        # surface intersection and a positive surface distance. It must be checked before a
        # certified "farther than the cap" answer is returned. The guest vertex has to be moved
        # into the host frame: `closest_to_in_frame` stores the two meshes in different frames,
        # and testing raw coordinates here produced a false `containment` contact (found by the
        # near-contact audit, 2026-11-07).
        contained = self._containment(other, rotation, translation)
        if contained is not None:
            return contained
        if not (self.watertight and other.watertight):
            # One direction of the containment test cannot be decided on an open mesh. The
            # surface-distance result remains usable, but it is not a full collision label.
            best.extra["containment_unknown"] = True
        if best.clearance >= max_clearance and not best.intersects:
            # The traversal's lower bounds prove the distance is at least max_clearance. This is a
            # certified lower bound, not an exact distance, even when some leaves were evaluated.
            best.extra["lower_bound"] = max_clearance
            best.extra["threshold"] = max_clearance
            best.extra["evaluated_leaf_pairs"] = evaluated
            best.extra["pruned_by"] = "max_clearance / certified node lower bounds"
            return QueryResult(STATUS_CULLED, max_clearance, False, extra=best.extra)
        return best

    def closest_to(self, other: "MeshRef", max_clearance: float = 0.25, max_candidates: int = 200000) -> QueryResult:
        """Exact separation between the two triangle meshes, with the intersection flag.

        Two prunes keep this usable at mesh scale, and both are bounds rather than guesses:

        * if the coarse pass already exceeds ``max_clearance``, the answer is "at least that far"
          and no refinement is attempted -- the caller asked about contact, not about a number;
        * a candidate triangle pair is skipped when ``|c_i - c_j| - r_i - r_j`` already exceeds the
          current best, since no point on either triangle can then be closer.
        """
        if self.tree is None or other.tree is None:
            return QueryResult(STATUS_DEGENERATE, float("nan"), False, reason="empty_mesh")
        k = int(min(8, len(other.centroids)))
        _, idx = self.tree.query(other.centroids, k=k)
        idx = np.atleast_2d(idx)
        if idx.shape[0] != len(other.centroids):
            idx = idx.T
        coarse = set()
        for column in range(idx.shape[1]):
            for j, i in enumerate(idx[:, column]):
                coarse.add((int(i), int(j)))
        best = self._refine(coarse, other, QueryResult(STATUS_OK, max_clearance, False))
        if best.intersects:
            best.extra.setdefault("refined", False)
            return best
        if best.clearance > max_clearance:
            # far surfaces still collide when one body is entirely inside the other, so the
            # containment test has to run before the "too far to matter" shortcut
            contained = self._containment(other)
            best.extra.setdefault("refined", False)
            if contained is not None:
                return contained
            if not (self.watertight and other.watertight):
                best.extra["containment_unknown"] = True
            best.clearance = min(best.clearance, max_clearance)
            return best

        radius = best.clearance + self.max_radius + other.max_radius
        lower = float("inf")
        candidates = []
        # one ball-tree call yields every centroid pair within the radius; the per-centroid loop
        # it replaces was ~4000 Python calls per constituent pair and dominated the replay's time
        for i, neighbours in enumerate(self.tree.query_ball_tree(other.tree, radius)):
            for j in neighbours:
                bound = float(np.linalg.norm(self.centroids[i] - other.centroids[j])) - self.circumradius[i] - other.circumradius[j]
                lower = min(lower, max(bound, 0.0))
                if bound <= best.clearance:
                    candidates.append((i, j))
            if len(candidates) > max_candidates:
                return QueryResult(STATUS_DEGENERATE, float("nan"), False, reason=f"too_many_candidates>{max_candidates}")
        if candidates:
            best = self._refine(candidates, other, best)
        if best.intersects:
            return best
        contained = self._containment(other)
        if contained is not None:
            return contained
        if not (self.watertight and other.watertight):
            best.extra["containment_unknown"] = True
        if self.contains(other.vertices[0]) is None:
            best.extra["containment"] = STATUS_UNKNOWN_CONTAINMENT
        best.extra.setdefault("refined", True)
        best.extra["lower_bound"] = lower
        return best

    @staticmethod
    def _aabb_after(lo, hi, rotation, translation):
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        moved = corners @ rotation.T + translation
        return moved.min(axis=0), moved.max(axis=0)

    @staticmethod
    def _aabb_overlap(lo_a, hi_a, lo_b, hi_b) -> bool:
        return bool(np.all(lo_a <= hi_b) and np.all(lo_b <= hi_a))

    def _containment(self, other: "MeshRef", rotation: np.ndarray | None = None,
                     translation: np.ndarray | None = None) -> QueryResult | None:
        """One body entirely inside the other: no surface intersection, still a collision.

        ``rotation``/``translation`` map ``other``'s frame into ``self``'s frame. In
        :meth:`closest_to_in_frame` the two meshes live in different frames, so the guest
        vertex must be transformed before the host's winding number is evaluated. The raw
        coordinates were used here until 2026-11-07; that produced false containment contacts
        on the radar near set.
        """
        rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
        translation = np.zeros(3) if translation is None else np.asarray(translation, dtype=np.float64)
        if len(self.vertices) == 0 or len(other.vertices) == 0:
            return None
        if self.watertight:
            guest_lo, guest_hi = self._aabb_after(other.aabb_lo, other.aabb_hi, rotation, translation)
            if self._aabb_overlap(self.aabb_lo, self.aabb_hi, guest_lo, guest_hi):
                for point in other.component_points:
                    guest_vertex = point @ rotation.T + translation
                    if self.contains(guest_vertex) is True:
                        return QueryResult(STATUS_OK, 0.0, True, reason="containment:b_inside_a")
        if other.watertight:
            inverse_rotation = rotation.T
            inverse_translation = -rotation.T @ translation
            guest_lo, guest_hi = self._aabb_after(self.aabb_lo, self.aabb_hi, inverse_rotation, inverse_translation)
            if self._aabb_overlap(other.aabb_lo, other.aabb_hi, guest_lo, guest_hi):
                for point in self.component_points:
                    guest_vertex = (point - translation) @ rotation
                    if other.contains(guest_vertex) is True:
                        return QueryResult(STATUS_OK, 0.0, True, reason="containment:a_inside_b")
        return None

    def _refine(self, pairs, other: "MeshRef", best: QueryResult) -> QueryResult:
        for i, j in pairs:
            if not (0 <= i < len(self.triangles)) or not (0 <= j < len(other.triangles)):
                # impossible with a consistent index; refuse rather than index the wrong triangle
                return QueryResult(STATUS_DEGENERATE, float("nan"), False, reason=f"index_out_of_range {i},{j}")
            result = triangle_triangle_distance(self.triangles[i], other.triangles[j])
            if result.intersects:
                return QueryResult(STATUS_OK, 0.0, True, result.point_a, result.point_b, i, j, "surface_intersection")
            if result.distance < best.clearance:
                best = QueryResult(
                    STATUS_OK,
                    float(result.distance),
                    False,
                    result.point_a,
                    result.point_b,
                    i,
                    j,
                    "",
                    dict(best.extra) if best.extra else {},
                )
        return best


def cross_check_with_fcl(mesh_a: MeshRef, mesh_b: MeshRef, tol: float = 1e-4) -> str:
    """Compare against FCL when it is importable; returns a one-line report."""
    try:
        import fcl  # type: ignore
    except Exception:  # noqa: BLE001
        return "fcl not installed; cross-check skipped"
    model_a = fcl.BVHModel()
    model_a.beginModel(len(mesh_a.vertices), len(mesh_a.faces))
    model_a.addSubModel(mesh_a.vertices, mesh_a.faces)
    model_a.endModel()
    model_b = fcl.BVHModel()
    model_b.beginModel(len(mesh_b.vertices), len(mesh_b.faces))
    model_b.addSubModel(mesh_b.vertices, mesh_b.faces)
    model_b.endModel()
    request = fcl.DistanceRequest(enable_nearest_points=True)
    result = fcl.DistanceResult()
    fcl.distance(fcl.CollisionObject(model_a), fcl.CollisionObject(model_b), request, result)
    return f"fcl {fcl.__version__ if hasattr(fcl, '__version__') else 'unknown'} distance {result.min_distance * 1000:.3f} mm"
