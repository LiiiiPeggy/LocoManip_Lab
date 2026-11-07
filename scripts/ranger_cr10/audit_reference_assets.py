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

"""Stage 4.13b0: the reference meshes, checked rather than assumed.

For every constituent in `collision_parts.py`: file, hash, scale, vertex and face counts, bounding
box, degenerate faces, closed-manifold status, connected components. A mesh that is not closed cannot
support an inside/outside test, and the reference query says so (`containment_unknown`) instead of
guessing -- that is what this audit is for.

Usage:
    python scripts/ranger_cr10/audit_reference_assets.py --out outputs/stage413/b0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/ranger_cr10"))

from collision_parts import CONSTITUENTS, box_mesh  # noqa: E402
from reference_geometry import MeshRef  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out", default="outputs/stage413/b0")
args = ap.parse_args()
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def components(faces: np.ndarray, n_vertices: int) -> int:
    parent = list(range(n_vertices))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for face in faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[ra] = rb
    used = {find(int(v)) for v in faces.ravel()} if len(faces) else set()
    return len(used)


def main() -> None:
    rows = []
    print(f"{'constituent':24s} {'kind':5s} {'verts':>6s} {'faces':>6s} {'closed':>6s} {'degenerate':>10s} {'parts':>5s}")
    for part in CONSTITUENTS:
        if part.kind == "mesh":
            path = MESH_DIR / part.stl
            mesh = trimesh.load(str(path), force="mesh", process=False)
            digest = sha256(path)
            source = part.stl
            raw_vertices, raw_faces = len(mesh.vertices), len(mesh.faces)
            # STL is a vertex soup: every triangle carries its own three vertices, so an
            # index-sharing closedness test fails on a perfectly good mesh. Welding is an
            # in-memory repair; the file's hash stays the hash of the file, and the welded counts
            # are reported next to the raw ones rather than replacing them.
            mesh.merge_vertices()
            vertices, faces = np.asarray(mesh.vertices) * part.scale, np.asarray(mesh.faces)
            welded = (len(mesh.vertices), len(mesh.faces), bool(mesh.is_watertight))
        else:
            vertices, faces = box_mesh(part.centre, part.half, part.quat)
            digest = ""
            source = "analytic box"
            raw_vertices, raw_faces, welded = 8, 12, (8, 12, True)
        ref = MeshRef(vertices, faces, name=f"{part.group}/{part.name}")
        entry = dict(
            group=part.group,
            name=part.name,
            kind=part.kind,
            source=source,
            sha256=digest,
            scale=part.scale,
            vertices=int(len(vertices)),
            faces=int(len(faces)),
            raw_vertices=int(raw_vertices),
            raw_faces=int(raw_faces),
            welded_vertices=int(welded[0]),
            welded_closed=bool(welded[2]),
            closed=bool(welded[2]),
            degenerate_faces=int((ref.triangle_areas <= 1e-14).sum()),
            components=components(faces, len(vertices)),
            bounds=[[float(x) for x in vertices.min(axis=0)], [float(x) for x in vertices.max(axis=0)]],
            warnings=ref.warnings,
        )
        rows.append(entry)
        print(
            f"{part.group + '/' + part.name:24s} {part.kind:5s} {len(vertices):6d} {len(faces):6d} "
            f"{str(ref.watertight):>6s} {entry['degenerate_faces']:10d} {entry['components']:5d}"
        )

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "reference_assets.json").write_text(json.dumps(dict(units="metres", constituents=rows), indent=1))
    open_parts = [r["name"] for r in rows if r["kind"] == "mesh" and not r["closed"]]
    print(f"\nnot closed: {open_parts if open_parts else 'none'}")
    print("a non-closed mesh cannot answer an inside/outside test; the query reports containment as unknown")
    print(f"wrote {out / 'reference_assets.json'}")


if __name__ == "__main__":
    main()
