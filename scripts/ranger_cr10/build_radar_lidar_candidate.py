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

"""Build the minimal radar-side candidate: per-component convex hulls of the lidar mesh.

The attribution selected radar-side loss for Link3/4/5/6. The candidate removes the single global
lidar convex hull and replaces it with one convex hull per connected component of the original
lidar STL. Every component is retained, including degenerate point/triangle components; none is
deleted. The union is not the old global hull: the empty space between components is removed.
The candidate distance against an arm hull is the minimum over all lidar components.

This is a diagnostic offline candidate, not a deployment asset. The frozen Link3<->Link5
candidate is untouched.

Run:
    python scripts/ranger_cr10/build_radar_lidar_candidate.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trimesh  # noqa: E402

ROOT = HERE.parents[1]
MESH = ROOT / "rangerboxcr10lidar_description/meshes/lidar_link0.STL"
OUT = ROOT / "outputs/stage413/b/radar_round2_2026_11_07/candidate_lidar_v1"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default=str(OUT))
args = ap.parse_args()


def main() -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source_sha = hashlib.sha256(MESH.read_bytes()).hexdigest()
    mesh = trimesh.load(str(MESH), force="mesh", process=False)
    raw_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    raw_faces = np.asarray(mesh.faces, dtype=np.int64)
    welded = mesh.copy()
    welded.merge_vertices(merge_tex=False, merge_norm=False)
    welded.remove_unreferenced_vertices()
    parts = welded.split(only_watertight=False)
    records = []
    for index, part in enumerate(parts):
        vertices = np.asarray(part.vertices, dtype=np.float64)
        faces = np.asarray(part.faces, dtype=np.int64)
        method = "convex_hull"
        if len(vertices) >= 4:
            try:
                hull = part.convex_hull
                vertices = np.asarray(hull.vertices, dtype=np.float64)
                faces = np.asarray(hull.faces, dtype=np.int64)
            except Exception:  # noqa: BLE001 - degenerate/coplanar component, keep the source points
                method = "degenerate_source_component"
        else:
            method = "degenerate_source_component"
        path = out / f"lidar_link0_p{index}.npz"
        np.savez_compressed(path, vertices=vertices, faces=faces)
        records.append(dict(component=index, method=method, vertices=int(len(vertices)),
                            faces=int(len(faces)), file=path.name,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest = dict(
        candidate="radar_lidar_component_hulls_v1",
        source_mesh=str(MESH.relative_to(ROOT)), source_sha256=source_sha,
        raw_vertices=int(len(raw_vertices)), raw_faces=int(len(raw_faces)),
        welded_vertices=int(len(welded.vertices)), welded_faces=int(len(welded.faces)),
        components=len(parts), pieces=len(records),
        method="weld; one connected component per piece; convex hull per component; degenerate "
               "components keep their source vertices/faces; no component deleted",
        replacement="the single global lidar convex hull used by D is removed; the candidate distance "
                    "is the union-min over these component hulls",
        piece_records=records,
        old_coverage="the single lidar global hull is the previous coverage; it is not retained",
        note="one file per component; the candidate contains the original lidar surface but removes "
             "inter-component empty space",
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(dict(source_sha256=source_sha, components=len(parts), pieces=len(records),
                          degenerate=sum(1 for row in records if row["method"] != "convex_hull"),
                          out=str(out)), indent=1))


if __name__ == "__main__":
    main()
