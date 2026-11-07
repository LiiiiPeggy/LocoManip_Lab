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

"""Cross-check the exact intersection predicate with an independent exact SAT oracle.

`robust_intersect.triangles_intersect_exact` is the interval method used by the reference;
`triangles_intersect_exact_sat` is a different algorithm. They must agree on the reported pairs of
the old and corrected validation sets, on the old 47 false positives, on synthetic cases, and on
random triangles. Agreement between two exact algorithms is evidence about the predicate; it is not
an independent physical measurement of the mesh.

Run:
    python scripts/ranger_cr10/audit_intersection_methods.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from robust_intersect import (  # noqa: E402
    triangles_intersect_exact,
    triangles_intersect_exact_sat,
)

ROOT = HERE.parents[1]
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes/cr10_meshes"


def pair_for(row, mesh3, mesh5):
    rotation = np.asarray(row["rotation"], dtype=np.float64)
    translation = np.asarray(row["translation"], dtype=np.float64)
    triangle_a = np.asarray(mesh3.triangles[row["reference_triangle_a"]], dtype=np.float64)
    triangle_b = np.asarray(mesh5.triangles[row["reference_triangle_b"]], dtype=np.float64) @ rotation.T + translation
    return triangle_a, triangle_b


def main() -> None:
    mesh3 = trimesh.load(str(MESH_DIR / "Link3.STL"), force="mesh", process=False)
    mesh5 = trimesh.load(str(MESH_DIR / "Link5.STL"), force="mesh", process=False)
    old = json.loads((ROOT / "outputs/stage413/b/validation/validation_results.json").read_text())
    new = json.loads((ROOT / "outputs/stage413/b/validation_exact/validation_results.json").read_text())
    audit = json.loads((ROOT / "outputs/stage413/b/intersection_audit.json").read_text())
    false_positive_ids = {row["id"] for row in audit["rows"]}

    def scan(rows, label):
        disagreements = []
        checked = 0
        for row in rows:
            if row.get("reference_triangle_a", -1) < 0:
                continue
            triangle_a, triangle_b = pair_for(row, mesh3, mesh5)
            interval = triangles_intersect_exact(triangle_a, triangle_b)
            sat = triangles_intersect_exact_sat(triangle_a, triangle_b)
            checked += 1
            if interval != sat:
                disagreements.append(dict(id=row["id"], interval=bool(interval), sat=bool(sat)))
        return dict(label=label, checked=checked, disagreements=disagreements)

    result = dict(
        old_validation=scan(old["rows"], "old"),
        corrected_validation=scan(new["rows"], "corrected"),
    )
    # Regression cases: the old 47 false positives must be rejected by both exact methods; the
    # corrected 51 true overlaps must be accepted by both.
    old_by_id = {row["id"]: row for row in old["rows"]}
    new_by_id = {row["id"]: row for row in new["rows"]}
    fp_rows = [old_by_id[i] for i in false_positive_ids if i in old_by_id]
    fp_checks = []
    for row in fp_rows:
        triangle_a, triangle_b = pair_for(row, mesh3, mesh5)
        fp_checks.append(dict(id=row["id"],
                              interval=bool(triangles_intersect_exact(triangle_a, triangle_b)),
                              sat=bool(triangles_intersect_exact_sat(triangle_a, triangle_b))))
    overlap_rows = [row for row in new["rows"] if row["reference_intersects"]]
    overlap_checks = []
    for row in overlap_rows:
        triangle_a, triangle_b = pair_for(row, mesh3, mesh5)
        overlap_checks.append(dict(id=row["id"],
                                   interval=bool(triangles_intersect_exact(triangle_a, triangle_b)),
                                   sat=bool(triangles_intersect_exact_sat(triangle_a, triangle_b))))
    result["old_false_positive_regression"] = dict(
        n=len(fp_checks),
        interval_all_false=all(not item["interval"] for item in fp_checks),
        sat_all_false=all(not item["sat"] for item in fp_checks),
    )
    result["corrected_true_overlap_regression"] = dict(
        n=len(overlap_checks),
        interval_all_true=all(item["interval"] for item in overlap_checks),
        sat_all_true=all(item["sat"] for item in overlap_checks),
    )
    # Synthetic cases
    crossing_a = np.array([[-0.1, -0.01, -0.01], [0.1, -0.01, -0.01], [0.0, 0.01, 0.01]])
    crossing_b = np.array([[-0.01, -0.1, -0.01], [0.01, 0.1, -0.01], [0.0, 0.0, 0.01]])
    parallel_a = np.array([[-0.1, -0.01, 0.0], [0.1, -0.01, 0.0], [0.0, 0.01, 0.0]])
    parallel_b = parallel_a + np.array([0.0, 0.0, 0.01])
    coplanar_c = np.array([[-0.05, -0.005, 0.0], [0.05, -0.005, 0.0], [0.0, 0.005, 0.0]])
    synthetic = dict(
        crossing=[bool(triangles_intersect_exact(crossing_a, crossing_b)),
                  bool(triangles_intersect_exact_sat(crossing_a, crossing_b))],
        parallel_separated=[bool(triangles_intersect_exact(parallel_a, parallel_b)),
                            bool(triangles_intersect_exact_sat(parallel_a, parallel_b))],
        coplanar_overlap=[bool(triangles_intersect_exact(parallel_a, coplanar_c)),
                          bool(triangles_intersect_exact_sat(parallel_a, coplanar_c))],
    )
    result["synthetic"] = synthetic
    # Random triangles, including near-parallel copies
    rng = np.random.default_rng(0)
    disagreements = []
    for index in range(200):
        a = rng.normal(size=(3, 3)) * 0.1
        b = rng.normal(size=(3, 3)) * 0.1
        interval = triangles_intersect_exact(a, b)
        sat = triangles_intersect_exact_sat(a, b)
        if interval != sat:
            disagreements.append(dict(index=index, interval=bool(interval), sat=bool(sat)))
    near_parallel = []
    for index in range(100):
        a = rng.normal(size=(3, 3)) * 0.1
        normal = np.cross(a[1] - a[0], a[2] - a[0])
        normal /= np.linalg.norm(normal)
        offset = normal * (10.0 ** rng.uniform(-7, -2))
        b = a + offset + rng.normal(size=(3, 3)) * 1e-5
        interval = triangles_intersect_exact(a, b)
        sat = triangles_intersect_exact_sat(a, b)
        if interval != sat:
            near_parallel.append(dict(index=index, interval=bool(interval), sat=bool(sat)))
    result["random_triangles"] = dict(n=200, disagreements=disagreements)
    result["near_parallel_triangles"] = dict(n=100, disagreements=near_parallel)
    out = ROOT / "outputs/stage413/b/intersection_methods_audit.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({key: result[key] for key in (
        "old_validation", "corrected_validation", "old_false_positive_regression",
        "corrected_true_overlap_regression", "synthetic", "random_triangles",
        "near_parallel_triangles",
    )}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
