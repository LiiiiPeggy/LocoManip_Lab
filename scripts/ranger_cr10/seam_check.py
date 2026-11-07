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

"""Targeted observed-gap check at the old-p3 / new-piece seam.

The centroid selection is a region proposal and the whole-link coverage check is a sample. This
script samples only the original Link3 mesh triangles that straddle the old p3 hull boundary (the
new/old interface) and reports the distance to the new piece union. It is still an observation, not
a proof: the output records the sample size, the contact tolerance and the witness of the largest
gap, and a gap below 0.10 mm is not interpreted as a strict no-gap guarantee.

Run:
    python scripts/ranger_cr10/seam_check.py --out outputs/stage413/b/correction_v1/seam_check.json
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

from candidate_eval import load_pieces  # noqa: E402
from convex_distance import gjk_distance  # noqa: E402

ROOT = HERE.parents[1]
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes/cr10_meshes"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--old", default="outputs/stage413/b/metric20")
ap.add_argument("--new", default="outputs/stage413/b/correction_v1/link3_local_p3")
ap.add_argument("--mesh", default="rangerboxcr10lidar_description/meshes/cr10_meshes/Link3.STL")
ap.add_argument("--band", type=float, default=0.005, help="straddling-band half width, metres")
ap.add_argument("--samples", type=int, default=500)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="outputs/stage413/b/correction_v1/seam_check.json")
args = ap.parse_args()


def main() -> None:
    old = load_pieces("cr10_Link3", ROOT / args.old)
    new = load_pieces("cr10_Link3", ROOT / args.new)
    p3 = next(vertices for name, vertices in old if name == "cr10_Link3_p3")
    hull = ConvexHull(p3, qhull_options="QJ")
    mesh = trimesh.load(str(ROOT / args.mesh), force="mesh", process=False)
    mesh.merge_vertices(merge_tex=False, merge_norm=False)
    mesh.remove_unreferenced_vertices()
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    signed = vertices @ hull.equations[:, :3].T + hull.equations[:, 3]
    max_signed = signed.max(axis=1)
    # a triangle straddles the seam if its vertices are not all on one side and at least one is
    # within the band of the boundary
    near = np.abs(max_signed) <= args.band
    straddle = (max_signed[np.asarray(faces)].min(axis=1) <= 0.0) & (max_signed[np.asarray(faces)].max(axis=1) >= 0.0)
    selected_faces = straddle | near[np.asarray(faces)].any(axis=1)
    triangles = vertices[faces[selected_faces]]
    rng = np.random.default_rng(args.seed)
    if len(triangles) == 0:
        raise SystemExit("no seam triangles selected")
    index = rng.choice(len(triangles), size=min(args.samples, len(triangles)), replace=True)
    weights = rng.dirichlet([1.0, 1.0, 1.0], size=len(index))
    points = np.einsum("ij,ijk->ik", weights, triangles[index])
    distances = np.full(len(points), np.inf)
    for _, piece_vertices in new:
        for i, point in enumerate(points):
            result = gjk_distance(point[None, :], piece_vertices, tol=1e-8)
            value = 0.0 if result.status == "intersecting" else result.distance
            if np.isfinite(value):
                distances[i] = min(distances[i], value)
    worst = int(np.argmax(distances))
    record = dict(
        old=args.old, new=args.new, mesh=args.mesh,
        band_m=args.band, seam_faces=int(selected_faces.sum()),
        samples=int(len(points)),
        contact_tolerance_m=1e-4,
        gap_median_m=float(np.median(distances)),
        gap_p95_m=float(np.percentile(distances, 95)),
        gap_max_m=float(distances[worst]),
        over_0_05mm=int((distances > 0.5e-4).sum()),
        over_0_10mm=int((distances > 1e-4).sum()),
        over_0_5mm=int((distances > 5e-4).sum()),
        worst_witness=points[worst].tolist(),
        note="observed seam gap on sampled original triangles; not a strict no-gap proof",
    )
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
