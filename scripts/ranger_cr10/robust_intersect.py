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

"""Stage 4.13b: exact triangle-triangle intersection, used to audit the reference label.

The double-precision interval test in `triangle_distance.triangle_triangle_intersect` compares
projected intervals with an absolute ``EPS = 1e-12``. For nearly parallel triangles the intersection
line direction is tiny, the projected coordinates are ill-conditioned, and the comparison can call a
pair intersecting when it is separated (measured on one validation pair: exact rational arithmetic
says disjoint, with a 1.81 mm closest-feature distance).

This module re-decides the same question with exact rational arithmetic. Inputs are doubles, so
``Fraction(float)`` is exact: no tolerance is needed and the answer is the exact answer for the
stored mesh coordinates. It is used as an audit oracle, not in the fast training path.

Usage:
    python scripts/ranger_cr10/robust_intersect.py --triangles outputs/.../pair.npz
"""

from __future__ import annotations

import argparse
from fractions import Fraction

import numpy as np


def _vec(point) -> list[Fraction]:
    return [Fraction(float(value)) for value in point]


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _dot(a, b) -> Fraction:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _interval(triangle, distances, direction):
    """Projection interval of the triangle's intersection with the other plane, exactly."""
    points = []
    for i in range(3):
        j = (i + 1) % 3
        di, dj = distances[i], distances[j]
        if di == 0:
            points.append(triangle[i])
        if (di > 0 and dj < 0) or (di < 0 and dj > 0):
            t = di / (di - dj)
            points.append([triangle[i][k] + t * (triangle[j][k] - triangle[i][k]) for k in range(3)])
    if not points:
        return None
    projected = [_dot(point, direction) for point in points]
    return min(projected), max(projected)


def _sat_2d(points_a, points_b, axis: int) -> bool:
    """Exact 2-D separating-axis test after dropping one coordinate (coplanar case)."""
    keep = [i for i in range(3) if i != axis]
    a = [[point[i] for i in keep] for point in points_a]
    b = [[point[i] for i in keep] for point in points_b]
    for polygon in (a, b):
        for i in range(3):
            edge = [polygon[(i + 1) % 3][0] - polygon[i][0], polygon[(i + 1) % 3][1] - polygon[i][1]]
            axis_2d = [-edge[1], edge[0]]
            if axis_2d[0] == 0 and axis_2d[1] == 0:
                continue
            values_a = [axis_2d[0] * point[0] + axis_2d[1] * point[1] for point in a]
            values_b = [axis_2d[0] * point[0] + axis_2d[1] * point[1] for point in b]
            if max(values_a) < min(values_b) or max(values_b) < min(values_a):
                return False
    return True


def triangles_intersect_exact_sat(triangle_a, triangle_b) -> bool:
    """Independent exact oracle: a 3-D separating-axis test on rational coordinates.

    Axes: both face normals, the in-plane edge normals (needed for the coplanar case), and the
    nine edge-edge cross products. If any axis separates the two closed triangles, they are
    disjoint; otherwise they intersect. This is a different algorithm from the interval method in
    `triangles_intersect_exact`, so agreement between the two is evidence about the predicate
    rather than about one code path re-reading its own label.
    """
    a = [_vec(point) for point in np.asarray(triangle_a, dtype=np.float64)]
    b = [_vec(point) for point in np.asarray(triangle_b, dtype=np.float64)]
    normal_a = _cross(_sub(a[1], a[0]), _sub(a[2], a[0]))
    normal_b = _cross(_sub(b[1], b[0]), _sub(b[2], b[0]))
    if _dot(normal_a, normal_a) == 0 or _dot(normal_b, normal_b) == 0:
        return False
    edges_a = [_sub(a[1], a[0]), _sub(a[2], a[1]), _sub(a[0], a[2])]
    edges_b = [_sub(b[1], b[0]), _sub(b[2], b[1]), _sub(b[0], b[2])]
    axes = [normal_a, normal_b]
    axes += [_cross(edge, normal_a) for edge in edges_a]
    axes += [_cross(edge, normal_b) for edge in edges_b]
    axes += [_cross(edge_a, edge_b) for edge_a in edges_a for edge_b in edges_b]
    for axis in axes:
        if _dot(axis, axis) == 0:
            continue
        projected_a = [_dot(point, axis) for point in a]
        projected_b = [_dot(point, axis) for point in b]
        if max(projected_a) < min(projected_b) or max(projected_b) < min(projected_a):
            return False
    return True


def triangles_intersect_exact(triangle_a, triangle_b) -> bool:
    """Exact answer for the stored double coordinates (no tolerance)."""
    a = [_vec(point) for point in np.asarray(triangle_a, dtype=np.float64)]
    b = [_vec(point) for point in np.asarray(triangle_b, dtype=np.float64)]
    normal_a = _cross(_sub(a[1], a[0]), _sub(a[2], a[0]))
    normal_b = _cross(_sub(b[1], b[0]), _sub(b[2], b[0]))
    if _dot(normal_a, normal_a) == 0 or _dot(normal_b, normal_b) == 0:
        return False  # a degenerate triangle has no interior to cross
    direction = _cross(normal_a, normal_b)
    if _dot(direction, direction) == 0:
        # Parallel planes. Distinct parallel planes cannot intersect, regardless of the projection;
        # only the exactly coplanar case goes to the 2-D test.
        if _dot(normal_a, _sub(b[0], a[0])) != 0:
            return False
        axis = int(np.argmax([abs(float(value)) for value in normal_a]))
        return _sat_2d(a, b, axis)
    distances_a = [_dot(normal_b, _sub(point, b[0])) for point in a]
    distances_b = [_dot(normal_a, _sub(point, a[0])) for point in b]
    if all(value > 0 for value in distances_a) or all(value < 0 for value in distances_a):
        return False
    if all(value > 0 for value in distances_b) or all(value < 0 for value in distances_b):
        return False
    interval_a = _interval(a, distances_a, direction)
    interval_b = _interval(b, distances_b, direction)
    if interval_a is None or interval_b is None:
        return False
    return max(interval_a[0], interval_b[0]) <= min(interval_a[1], interval_b[1])


def _main() -> None:
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="outputs/stage413/b/validation/validation_results.json")
    ap.add_argument("--out", default="outputs/stage413/b/intersection_audit.json")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import trimesh

    mesh3 = trimesh.load(
        str(root / "rangerboxcr10lidar_description/meshes/cr10_meshes/Link3.STL"),
        force="mesh", process=False,
    )
    mesh5 = trimesh.load(
        str(root / "rangerboxcr10lidar_description/meshes/cr10_meshes/Link5.STL"),
        force="mesh", process=False,
    )
    rows = json.loads((root / args.results).read_text())["rows"]
    disagreements = []
    checked = 0
    for row in rows:
        if row.get("reference_triangle_a", -1) < 0 or row.get("reference_triangle_b", -1) < 0:
            continue
        rotation = np.asarray(row["rotation"])
        translation = np.asarray(row["translation"])
        triangle_a = np.asarray(mesh3.triangles[row["reference_triangle_a"]], dtype=np.float64)
        triangle_b = np.asarray(mesh5.triangles[row["reference_triangle_b"]], dtype=np.float64) @ rotation.T + translation
        exact = triangles_intersect_exact(triangle_a, triangle_b)
        checked += 1
        if exact != bool(row["reference_intersects"]):
            disagreements.append(dict(
                id=row["id"], type=row["type"], cluster=row["cluster"],
                reference_intersects=bool(row["reference_intersects"]),
                exact_intersects=bool(exact),
                reference_distance=row["reference_distance"],
                triangles=[row["reference_triangle_a"], row["reference_triangle_b"]],
            ))
    record = dict(checked=checked, disagreements=len(disagreements), rows=disagreements)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(checked=checked, disagreements=len(disagreements)), indent=1))
    for item in disagreements[:20]:
        print(item)
    print(f"wrote {out}")


if __name__ == "__main__":
    _main()
