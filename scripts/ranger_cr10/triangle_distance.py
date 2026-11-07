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

"""Stage 4.13b0: exact triangle-triangle distance and intersection, as the reference primitive.

The previous reference took the minimum over vertex-to-surface distances in both directions. That
misses the edge-edge case and cannot tell a crossing pair from a separated one: two boxes whose long
axes are perpendicular and whose centres coincide have a vertex-to-surface distance of ~90 mm and a
true distance of zero. Everything here exists so the reference has an exact primitive to stand on.

Two triangles are either intersecting, or separated with the closest features being vertex-face or
edge-edge. Both cases are enumerated exactly, and the intersection test is separate from the
distance because a zero distance is not by itself proof of intersection (tangency is also zero).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robust_intersect import triangles_intersect_exact

EPS = 1e-12


@dataclass
class PairDistance:
    distance: float
    intersects: bool
    point_a: np.ndarray | None = None
    point_b: np.ndarray | None = None


def _closest_point_on_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Closest point on triangle (a, b, c) to ``point`` (Ericson, *Real-Time Collision Detection* §5.1.5)."""
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(ab @ ap)
    d2 = float(ac @ ap)
    if d1 <= 0 and d2 <= 0:
        return a
    bp = point - b
    d3 = float(ab @ bp)
    d4 = float(ac @ bp)
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        return a + v * ab
    cp = point - c
    d5 = float(ab @ cp)
    d6 = float(ac @ cp)
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return a + w * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)
    denom = va + vb + vc
    if abs(denom) < EPS:
        return a
    v = vb / denom
    w = vc / denom
    return a + v * ab + w * ac


def segment_segment_distance(p1, q1, p2, q2) -> tuple[float, np.ndarray, np.ndarray]:
    """Closest points between two segments (Ericson §5.1.9), parallel case handled by projection."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)
    if a <= EPS and e <= EPS:
        return float(np.linalg.norm(p1 - p2)), p1, p2
    if a <= EPS:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = float(d1 @ r)
        if e <= EPS:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > EPS else np.clip(-c / a, 0.0, 1.0)
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)
    point_a = p1 + d1 * s
    point_b = p2 + d2 * t
    return float(np.linalg.norm(point_a - point_b)), point_a, point_b


def _signed_tetra_volume(a, b, c, d) -> float:
    return float(np.dot(np.cross(b - a, c - a), d - a))


def _triangle_triangle_intersect_double(t1: np.ndarray, t2: np.ndarray) -> bool:
    """Moller's triangle-triangle overlap test (coplanar cases handled by a 2-D projection).

    Kept as the fast diagnostic path. It is **not** used to decide a reference label: the interval
    comparison uses an absolute tolerance on ill-conditioned projected coordinates, and on this
    mesh it called 47 validation pairs intersecting that exact rational arithmetic separates.
    """
    a, b, c = t1
    d, e, f = t2
    n2 = np.cross(e - d, f - d)
    d0 = float(np.dot(n2, a - d))
    d1 = float(np.dot(n2, b - d))
    d2 = float(np.dot(n2, c - d))
    if (d0 > EPS and d1 > EPS and d2 > EPS) or (d0 < -EPS and d1 < -EPS and d2 < -EPS):
        return False
    n1 = np.cross(b - a, c - a)
    e0 = float(np.dot(n1, d - a))
    e1 = float(np.dot(n1, e - a))
    e2 = float(np.dot(n1, f - a))
    if (e0 > EPS and e1 > EPS and e2 > EPS) or (e0 < -EPS and e1 < -EPS and e2 < -EPS):
        return False
    direction = np.cross(n1, n2)
    if float(direction @ direction) > EPS:
        intervals = []
        for tri, dists in ((t1, (d0, d1, d2)), (t2, (e0, e1, e2))):
            values = _line_interval(tri, dists, direction)
            if values is None:
                return True
            intervals.append(values)
        (lo1, hi1), (lo2, hi2) = intervals
        return max(lo1, lo2) <= min(hi1, hi2) + EPS
    # parallel normals: the triangles are coplanar (or one is degenerate); decide in 2-D
    axis = int(np.argmax(np.abs(n1)))
    other = [i for i in range(3) if i != axis]
    return _coplanar_overlap(t1[:, other], t2[:, other])


def triangle_triangle_intersect(t1: np.ndarray, t2: np.ndarray) -> bool:
    """Exact triangle-triangle intersection for the stored double coordinates.

    Closest-feature distance and intersection are separate questions: two crossing triangles can
    have a non-zero closest *feature* distance, and two separated triangles can have overlapping
    plane-line interval projections when their normals are nearly parallel. The exact rational
    decision removes the tolerance from the second question, which is the one that was producing
    false reference contacts on this mesh.
    """
    return triangles_intersect_exact(t1, t2)


def _line_interval(tri, dists, direction):
    """Interval of a triangle along the intersection line of two planes (None = degenerate)."""
    points = []
    for i in range(3):
        j = (i + 1) % 3
        di, dj = dists[i], dists[j]
        if (di > 0 and dj < 0) or (di < 0 and dj > 0):
            t = di / (di - dj)
            points.append(tri[i] + t * (tri[j] - tri[i]))
        elif abs(di) <= EPS:
            points.append(tri[i])
    if len(points) < 2:
        return None
    projected = [float(p @ direction) for p in points]
    return min(projected), max(projected)


def _coplanar_overlap(p1: np.ndarray, p2: np.ndarray) -> bool:
    """2-D separating-axis test for two coplanar triangles."""
    for tri, other in ((p1, p2), (p2, p1)):
        for i in range(3):
            edge = tri[(i + 1) % 3] - tri[i]
            axis = np.array([-edge[1], edge[0]])
            if float(axis @ axis) <= EPS:
                continue
            a = [float(axis @ p) for p in tri]
            b = [float(axis @ p) for p in other]
            if max(a) < min(b) - EPS or max(b) < min(a) - EPS:
                return False
    return True


def triangle_triangle_distance(t1: np.ndarray, t2: np.ndarray) -> PairDistance:
    """Exact distance between two triangles, and whether they intersect.

    Features: the 9 edge-edge pairs and the 6 vertex-triangle pairs cover every separated case; the
    intersection test is run separately because a zero distance also happens for tangency.
    """
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    best = PairDistance(np.inf, False, None, None)
    for i in range(3):
        closest = _closest_point_on_triangle(t1[i], t2[0], t2[1], t2[2])
        distance = float(np.linalg.norm(t1[i] - closest))
        if distance < best.distance:
            best = PairDistance(distance, False, t1[i].copy(), closest)
    for i in range(3):
        closest = _closest_point_on_triangle(t2[i], t1[0], t1[1], t1[2])
        distance = float(np.linalg.norm(t2[i] - closest))
        if distance < best.distance:
            best = PairDistance(distance, False, closest, t2[i].copy())
    for i in range(3):
        for j in range(3):
            distance, pa, pb = segment_segment_distance(t1[i], t1[(i + 1) % 3], t2[j], t2[(j + 1) % 3])
            if distance < best.distance:
                best = PairDistance(distance, False, pa, pb)
    if triangle_triangle_intersect(t1, t2):
        best.intersects = True
        best.distance = 0.0
    elif best.distance <= EPS:
        best.distance = 0.0
    return best


if __name__ == "__main__":
    # the three analytic counterexamples from the 4.13b0 brief
    A = np.array(
        [
            [-0.1, -0.01, -0.01],
            [0.1, -0.01, -0.01],
            [0.0, 0.01, 0.01],
        ]
    )
    B = np.array(
        [
            [-0.01, -0.1, -0.01],
            [0.01, 0.1, -0.01],
            [0.0, 0.0, 0.01],
        ]
    )
    print("tri-tri on the crossing example (synthetic faces, not the boxes themselves):")
    result = triangle_triangle_distance(A, B)
    print(f"  distance {result.distance * 1000:.3f} mm, intersects {result.intersects}")


# -- vectorised batch path ---------------------------------------------------------------
def batch_triangle_distance(tri_a: np.ndarray, tri_b: np.ndarray) -> np.ndarray:
    """Distance for P triangle pairs at once, over the same 15 features.

    The scalar path above is the reference; this is the same enumeration evaluated on (P, 3, 3)
    arrays. Intersection is *not* decided here -- a zero distance is ambiguous (tangency is also
    zero) -- so callers confirm near-zero pairs with `triangle_triangle_intersect`.
    """
    tri_a = np.asarray(tri_a, dtype=np.float64)
    tri_b = np.asarray(tri_b, dtype=np.float64)
    best = np.full(len(tri_a), np.inf)

    # point of A against B, and point of B against A
    for i in range(3):
        closest = _batch_closest_on_triangle(tri_a[:, i], tri_b[:, 0], tri_b[:, 1], tri_b[:, 2])
        best = np.minimum(best, np.linalg.norm(tri_a[:, i] - closest, axis=1))
        closest = _batch_closest_on_triangle(tri_b[:, i], tri_a[:, 0], tri_a[:, 1], tri_a[:, 2])
        best = np.minimum(best, np.linalg.norm(tri_b[:, i] - closest, axis=1))

    # the nine edge pairs
    for i in range(3):
        p1 = tri_a[:, i]
        q1 = tri_a[:, (i + 1) % 3]
        for j in range(3):
            p2 = tri_b[:, j]
            q2 = tri_b[:, (j + 1) % 3]
            best = np.minimum(best, _batch_segment_distance(p1, q1, p2, q2))
    return best


def _batch_closest_on_triangle(points, a, b, c):
    ab = b - a
    ac = c - a
    ap = points - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = points - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = points - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = np.where(np.abs(va + vb + vc) < 1e-20, 1e-20, va + vb + vc)
    v = vb / denom
    w = vc / denom
    closest = (1.0 - v - w)[:, None] * a + v[:, None] * b + w[:, None] * c

    def region(mask, value):
        nonlocal closest
        closest = np.where(mask[:, None], value, closest)

    region((d1 <= 0) & (d2 <= 0), a)
    region((d3 >= 0) & (d4 <= d3), b)
    region((d6 >= 0) & (d5 <= d6), c)
    safe = np.where(np.abs(d1 - d3) < 1e-20, 1e-20, d1 - d3)
    region((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + (d1 / safe)[:, None] * ab)
    safe = np.where(np.abs(d2 - d6) < 1e-20, 1e-20, d2 - d6)
    region((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + (d2 / safe)[:, None] * ac)
    safe = np.where(np.abs((d4 - d3) + (d5 - d6)) < 1e-20, 1e-20, (d4 - d3) + (d5 - d6))
    region((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + ((d4 - d3) / safe)[:, None] * (c - b))
    return closest


def _batch_segment_distance(p1, q1, p2, q2):
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    denom = a * e - b * b
    s = np.where(denom > 1e-12, (b * f - c * e) / np.where(denom > 1e-12, denom, 1.0), 0.0)
    s = np.clip(s, 0.0, 1.0)
    t = (b * s + f) / np.where(e > 1e-12, e, 1.0)
    below = t < 0
    above = t > 1
    s = np.where(below, np.clip(-c / np.where(a > 1e-12, a, 1.0), 0, 1), s)
    t = np.where(below, 0.0, t)
    s = np.where(above, np.clip((b - c) / np.where(a > 1e-12, a, 1.0), 0, 1), s)
    t = np.where(above, 1.0, t)
    pa = p1 + d1 * s[:, None]
    pb = p2 + d2 * t[:, None]
    return np.linalg.norm(pa - pb, axis=1)
