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

"""Stage 4.13b: does a convex piece protrude outside the mesh it was cut from?

The "outside" flag is `MeshRef.contains(point) is False`. An earlier version of this script used
`contains(point) is True`, which counted points *inside* the mesh as outside; all numbers produced
by that version (notably p8 "42/300 outside at 5.876 mm") are invalid and were superseded.

The validation reports a conservative distance error (candidate closer than the reference) in one
rigid-transform family, localised to a specific piece pair. A convex cover can be closer for two
reasons: the cover protrudes outside the mesh (overfill), or the cover leaves a gap and the *other*
piece reaches into it. This is the direct check for the first: sample the piece surface, measure the
unsigned distance to the source mesh, and use the winding number to say whether each sample is
outside. An outside sample at 3 mm is overfill; an outside sample at 0 is a surface point.

Usage:
    python scripts/ranger_cr10/analyze_piece_overfill.py \
        --piece outputs/stage413/b/metric20/cr10_Link5_p8.npz \
        --mesh rangerboxcr10lidar_description/meshes/cr10_meshes/Link5.STL
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

from mesh_distance import MeshDistance  # noqa: E402
from reference_geometry import MeshRef  # noqa: E402

ROOT = HERE.parents[1]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--piece", required=True)
ap.add_argument("--mesh", required=True)
ap.add_argument("--samples", type=int, default=400)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="outputs/stage413/b/piece_overfill.json")
args = ap.parse_args()


def main() -> None:
    data = np.load(args.piece)
    vertices = np.asarray(data["vertices"], dtype=np.float64)
    faces = np.asarray(data["faces"], dtype=np.int64)
    mesh = trimesh.load(str(ROOT / args.mesh), force="mesh", process=False)
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_faces = np.asarray(mesh.faces)
    distance = MeshDistance(mesh_vertices, mesh_faces)
    # The STL repeats vertices per triangle; the raw mesh therefore fails the edge-pair
    # watertight test even though its welded geometry is closed. Weld for the containment test
    # only; the unsigned distance still uses the original triangles.
    welded = mesh.copy()
    welded.merge_vertices(merge_tex=False, merge_norm=False)
    welded.remove_unreferenced_vertices()
    mesh_ref = MeshRef(np.asarray(welded.vertices), np.asarray(welded.faces), Path(args.mesh).stem)

    rng = np.random.default_rng(args.seed)
    picked = rng.choice(len(faces), size=min(args.samples, len(faces)), replace=False)
    triangles = vertices[faces[picked]]
    weights = rng.dirichlet([1.0, 1.0, 1.0], size=len(picked))
    points = np.einsum("ij,ijk->ik", weights, triangles)
    unsigned = distance.distance(points)

    # contains() True means inside. The earlier version of this script used True as the outside
    # flag, so every reported "outside" number was actually an inside number.
    outside = []
    containment_known = True
    for point in points:
        outcome = mesh_ref.contains(point)
        if outcome is None:
            containment_known = False
            outside.append(None)
        else:
            outside.append(bool(outcome))
    outside_flags = np.array([value is False for value in outside], dtype=bool)
    inside_flags = np.array([value is True for value in outside], dtype=bool)
    finite = unsigned[np.isfinite(unsigned)]
    outside_distance = unsigned[outside_flags & np.isfinite(unsigned)]
    record = dict(
        piece=str(Path(args.piece)),
        piece_vertices=int(len(vertices)),
        piece_faces=int(len(faces)),
        mesh=args.mesh,
        mesh_vertices=int(len(mesh_vertices)),
        mesh_faces=int(len(mesh_faces)),
        samples=int(len(points)),
        containment_known=containment_known,
        raw_mesh_watertight=bool(mesh.is_watertight),
        welded_mesh_watertight=bool(mesh_ref.watertight),
        welded_vertices=int(len(welded.vertices)),
        welded_faces=int(len(welded.faces)),
        distance_median_m=float(np.median(finite)),
        distance_p95_m=float(np.percentile(finite, 95)),
        distance_max_m=float(np.max(finite)),
        inside_samples=int(inside_flags.sum()),
        outside_samples=int(outside_flags.sum()),
        outside_fraction=float(outside_flags.mean()),
        outside_distance_median_m=float(np.median(outside_distance)) if len(outside_distance) else 0.0,
        outside_distance_p95_m=float(np.percentile(outside_distance, 95)) if len(outside_distance) else 0.0,
        outside_distance_max_m=float(outside_distance.max()) if len(outside_distance) else 0.0,
        outside_over_1mm=int((outside_distance > 1e-3).sum()),
        outside_over_2mm=int((outside_distance > 2e-3).sum()),
        outside_over_3mm=int((outside_distance > 3e-3).sum()),
        furthest_point=points[int(np.argmax(unsigned))].tolist() if len(points) else None,
        furthest_outside_point=(
            points[outside_flags][int(np.argmax(outside_distance))].tolist()
            if len(outside_distance) else None
        ),
    )
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
