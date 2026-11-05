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

"""Stage 4.13b: distance between two convex point sets, with the status made explicit.

A reference implementation first, on the CPU, so that any GPU version later has something to be
wrong against. GJK returns the separation distance when the sets are disjoint; when they are not,
the separation is zero and the caller is told so through ``status`` rather than being handed a
number that looks like clearance. Penetration depth is *not* reported here -- EPA is a separate
problem and calling a hull-pair depth "the non-convex penetration" would be the kind of quiet
overstatement this stage exists to avoid.

Everything that can go wrong is counted: iteration exhaustion, degeneracy and non-convergence are
returned in ``status`` / ``iterations`` so a caller can fail loudly instead of treating a bad query
as a safe distance.

Usage (self-check against MuJoCo's own hull distance):
    python scripts/ranger_cr10/convex_distance.py --verify
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass
class GjkResult:
    distance: float
    status: str  # "separated", "intersecting", "max_iterations", "degenerate"
    iterations: int


def _support(points: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return points[int(np.argmax(points @ direction))]


def _closest_point_on_simplex(simplex: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Closest point to the origin on a simplex of 1-3 points, with the reduced simplex.

    Johnson's algorithm specialised to the low-dimensional cases that actually occur in GJK.
    """
    if len(simplex) == 1:
        return simplex[0], simplex
    if len(simplex) == 2:
        a, b = simplex
        ab = b - a
        denom = float(ab @ ab)
        if denom <= 1e-18:
            return a, [a]
        t = np.clip(-float(a @ ab) / denom, 0.0, 1.0)
        point = a + t * ab
        if t <= 0.0:
            return point, [a]
        if t >= 1.0:
            return point, [b]
        return point, [a, b]
    if len(simplex) == 4:
        # a tetrahedron: either the origin is inside (intersecting) or the closest point is on one
        # of its faces
        a, b, c, d = simplex
        matrix = np.column_stack([b - a, c - a, d - a])
        try:
            bary = np.linalg.solve(matrix, -a)
        except np.linalg.LinAlgError:
            bary = None
        if bary is not None and (bary >= -1e-12).all() and (1.0 - bary.sum()) >= -1e-12:
            return np.zeros(3), [b, c, d]
        best = None
        for face in ([a, b, c], [a, b, d], [a, c, d], [b, c, d]):
            point, reduced = _closest_point_on_simplex(face)
            score = float(point @ point)
            if best is None or score < best[0]:
                best = (score, point, reduced)
        return best[1], best[2]
    a, b, c = simplex
    ab, ac = b - a, c - a
    # barycentric coordinates of the projection of the origin onto the triangle's plane
    n = np.cross(ab, ac)
    nn = float(n @ n)
    if nn <= 1e-18:
        # degenerate triangle: fall back to the best edge
        best = None
        for edge in ([a, b], [a, c], [b, c]):
            point, reduced = _closest_point_on_simplex(edge)
            score = float(point @ point)
            if best is None or score < best[0]:
                best = (score, point, reduced)
        return best[1], best[2]
    d00, d01, d11 = float(ab @ ab), float(ab @ ac), float(ac @ ac)
    d20, d21 = float(-a @ ab), float(-a @ ac)
    denom = d00 * d11 - d01 * d01
    if abs(denom) <= 1e-18:
        v, w = 1.0, 0.0
    else:
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    if u >= 0.0 and v >= 0.0 and w >= 0.0:
        return u * a + v * b + w * c, [a, b, c]
    # outside the triangle: test the three edges, keep the closest
    best = None
    for edge in ([a, b], [b, c], [c, a]):
        point, reduced = _closest_point_on_simplex(edge)
        score = float(point @ point)
        if best is None or score < best[0]:
            best = (score, point, reduced)
    return best[1], best[2]


def gjk_distance(points_a: np.ndarray, points_b: np.ndarray, tol: float = 1e-4, max_iterations: int = 64) -> GjkResult:
    """Distance between two convex hulls of point sets (both already in a common frame).

    The stopping rule is the distance variant, not the intersection one: convergence when the new
    support does not improve on the current closest point, `v.v - v.w <= tol`. Using the
    intersection form (`w.d <= 0`) here terminates early and reports the distance of an intermediate
    simplex -- measured on this robot at 110 mm against a true 32 mm.
    """
    if len(points_a) == 0 or len(points_b) == 0:
        return GjkResult(float("nan"), "degenerate", 0)
    direction = points_b.mean(axis=0) - points_a.mean(axis=0)
    if float(direction @ direction) <= 1e-18:
        direction = np.array([1.0, 0.0, 0.0])
    v = _support(points_a, direction) - _support(points_b, -direction)
    simplex = [v.copy()]
    for iteration in range(1, max_iterations + 1):
        if float(v @ v) <= tol * tol:
            return GjkResult(0.0, "intersecting", iteration)
        direction = -v
        w = _support(points_a, direction) - _support(points_b, -direction)
        # converged when the support in the search direction does not move us closer
        if float(v @ v) - float(v @ w) <= tol:
            return GjkResult(float(np.linalg.norm(v)), "separated", iteration)
        if float((v - w) @ (v - w)) <= 1e-18:
            return GjkResult(float(np.linalg.norm(v)), "degenerate", iteration)
        simplex = simplex + [w]
        v, simplex = _closest_point_on_simplex(simplex)
    return GjkResult(float(np.linalg.norm(v)), "max_iterations", max_iterations)


def verify(n_checks: int = 200, args_tol: float = 1e-4) -> int:
    """Check GJK against MuJoCo's hull distance on the robot's own geometry."""
    import json
    from pathlib import Path

    import mujoco

    root = Path(__file__).resolve().parents[2]
    model = mujoco.MjModel.from_xml_path(str(root / "mujoco/robots/ranger_cr10/scene.xml"))
    data = mujoco.MjData(model)
    bid = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)}
    arm_adr = [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)
    ]
    poses = np.load(root / "outputs/stage411/operating_poses.npz")["poses"]
    pairs = [("cr10_Link3", "cr10_Link5"), ("cr10_Link6", "lidar_link0"), ("cr10_Link5", "lidar_link0")]
    rng = np.random.default_rng(0)
    failures = 0
    worst = 0.0
    checked = 0
    differences = []
    for q in poses[rng.permutation(len(poses))][:n_checks]:
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        for address, value in zip(arm_adr, q):
            data.qpos[address] = value
        mujoco.mj_forward(model, data)
        for a, b in pairs:
            ga = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == bid[a] and model.geom_contype[g] != 0)
            gb = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == bid[b] and model.geom_contype[g] != 0)
            cap = 0.5
            reference = mujoco.mj_geomDistance(model, data, ga, gb, cap, None)
            if reference <= 0.0 or reference >= cap - 1e-9:
                # contact (no separation to compare) or beyond the query cap
                continue
            va = _geom_points(model, data, ga)
            vb = _geom_points(model, data, gb)
            result = gjk_distance(va, vb, tol=args_tol)
            checked += 1
            if result.status != "separated":
                failures += 1
                continue
            worst = max(worst, abs(result.distance - reference))
            differences.append(result.distance - reference)
    differences = np.array(differences)
    print(
        f"GJK vs mj_geomDistance on the same hulls: {checked} separated pairs\n"
        f"  difference (gjk - mujoco): median {np.median(differences) * 1000:+.3f} mm, "
        f"p95 {np.percentile(np.abs(differences), 95) * 1000:.3f} mm, max {worst * 1000:.3f} mm, "
        f"status failures {failures}"
    )
    print("  a difference here is either the algorithm or MuJoCo's own hull query; both are convex")
    print("  hull distances, so the arbitration uses vertex-to-hull distances in the report")
    return 0 if failures == 0 and worst < 1e-3 else 1


def _geom_points(model, data, geom: int) -> np.ndarray:
    import mujoco

    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        local = model.mesh_vert[start : start + count]
    else:
        h = model.geom_size[geom]
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        local = signs * h
    return local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()
    if args.verify:
        raise SystemExit(verify(args.n, args.tol))
    print("nothing to do; pass --verify")
