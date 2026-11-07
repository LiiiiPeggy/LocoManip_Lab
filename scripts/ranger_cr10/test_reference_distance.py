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

"""Stage 4.13b0: the reference distance has to pass analytic cases before it judges anything else.

The previous reference answered "the min over vertex-to-surface distances", which reported ~90 mm
for two boxes crossing at their centres and could not tell contact from separation. These are the
cases that expose exactly that, each with a number that can be derived by hand, plus an exhaustive
cross-check so the pruned query is not merely consistent with itself.

Run:
    python scripts/ranger_cr10/test_reference_distance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mesh_distance import MeshDistance  # noqa: E402
from reference_geometry import MeshRef  # noqa: E402
from triangle_distance import triangle_triangle_distance  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} {detail}")


def box(centre, half):
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    vertices = signs * np.asarray(half, float) + np.asarray(centre, float)
    faces = np.array(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]],
        dtype=np.int64,
    )
    return MeshRef(vertices, faces)


def old_vertex_only_distance(mesh_a: MeshRef, mesh_b: MeshRef) -> float:
    """The 4.13a reference: min over both directions of vertex-to-surface distance."""
    a = MeshDistance(mesh_a.vertices, mesh_a.faces).distance(mesh_b.vertices)
    b = MeshDistance(mesh_b.vertices, mesh_b.faces).distance(mesh_a.vertices)
    return float(min(a.min(), b.min()))


def main() -> None:
    print("analytic cases")
    a = box((0, 0, 0), (0.1, 0.01, 0.01))
    b = box((0, 0, 0), (0.01, 0.1, 0.01))
    result = a.closest_to(b, max_clearance=0.2)
    check(
        "crossing boxes, long axes perpendicular",
        result.intersects and result.clearance == 0.0,
        f"intersects={result.intersects} clearance={result.clearance * 1000:.3f} mm (old: {old_vertex_only_distance(a, b) * 1000:.1f} mm)",
    )

    b2 = box((0, 0, 0.025), (0.01, 0.1, 0.01))
    result = a.closest_to(b2, max_clearance=0.2)
    check(
        "same boxes, second moved 25 mm along z",
        (not result.intersects) and abs(result.clearance - 0.005) < 1e-9,
        f"clearance={result.clearance * 1000:.3f} mm (expected 5.000; old: {old_vertex_only_distance(a, b2) * 1000:.1f} mm)",
    )

    # a large triangle whose centroid is far, surrounded by small ones whose centroids are near
    vertices = [[-1.0, -1.0, 0.0], [100.0, -1.0, 0.0], [-1.0, 100.0, 0.0]]
    faces = [[0, 1, 2]]
    rng = np.random.default_rng(0)
    for _ in range(128):
        cx, cy = rng.normal(0.0, 0.006, 2) + 0.1
        i = len(vertices)
        vertices += [[cx - 0.004, cy - 0.004, 0.0], [cx + 0.004, cy - 0.004, 0.0], [cx, cy + 0.004, 0.0]]
        faces.append([i, i + 1, i + 2])
    query = MeshDistance(np.array(vertices), np.array(faces))
    distance = query.distance(np.array([[0.0, 0.0, 0.001]]))[0]
    check(
        "large triangle beats 128 nearer centroids",
        abs(distance - 0.001) < 1e-9,
        f"distance={distance * 1000:.3f} mm (expected 1.000)",
    )

    # tangency: two boxes exactly touching face to face
    c = box((0, 0, 0.02), (0.01, 0.01, 0.01))
    d = box((0, 0, 0.0), (0.01, 0.01, 0.01))
    result = c.closest_to(d, max_clearance=0.2)
    check(
        "faces exactly touching",
        result.clearance == 0.0,
        f"clearance={result.clearance * 1000:.3f} mm intersects={result.intersects}",
    )

    # shallow penetration: 0.5 mm overlap
    e = box((0, 0, 0.0195), (0.01, 0.01, 0.01))
    result = e.closest_to(d, max_clearance=0.2)
    check(
        "0.5 mm overlap is reported as intersection",
        result.intersects and result.clearance == 0.0,
        f"intersects={result.intersects} clearance={result.clearance * 1000:.3f} mm",
    )

    # containment: a small closed box entirely inside a large one
    big = box((0, 0, 0), (0.5, 0.5, 0.5))
    small = box((0.1, 0.1, 0.1), (0.01, 0.01, 0.01))
    result = big.closest_to(small, max_clearance=0.5)
    check(
        "fully contained body is a collision",
        result.intersects,
        f"intersects={result.intersects} reason={result.reason}",
    )

    # degenerate triangle: zero area, must not crash and must report a status
    degenerate = MeshRef(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), np.array([[0, 1, 2]]))
    result = triangle_triangle_distance(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]]),
    )
    check(
        "degenerate triangle handled without error",
        np.isfinite(result.distance),
        f"distance={result.distance * 1000:.3f} mm; mesh warnings={degenerate.warnings}",
    )

    print("\nexhaustive cross-check (pruned query vs every triangle pair)")
    rng = np.random.default_rng(1)
    worst = 0.0
    for trial in range(20):
        def random_mesh(centre):
            vertices = rng.normal(0, 0.03, (12, 3)) + np.asarray(centre)
            faces = rng.integers(0, len(vertices), (10, 3))
            faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
            return MeshRef(vertices, faces)

        mesh_a = random_mesh((0, 0, 0))
        mesh_b = random_mesh((rng.uniform(0.01, 0.12), 0, 0))
        pruned = mesh_a.closest_to(mesh_b, max_clearance=0.5)
        brute = np.inf
        brute_intersects = False
        for ta in mesh_a.triangles:
            for tb in mesh_b.triangles:
                outcome = triangle_triangle_distance(ta, tb)
                if outcome.intersects:
                    brute_intersects = True
                brute = min(brute, outcome.distance)
        if brute_intersects:
            brute = 0.0
        worst = max(worst, abs(pruned.clearance - brute))
        if pruned.intersects != brute_intersects:
            worst = max(worst, 1.0)
    check(
        "20 random mesh pairs agree with brute force",
        worst < 1e-9,
        f"max disagreement {worst * 1000:.6f} mm",
    )

    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    if failed:
        print("failed: " + ", ".join(failed))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
