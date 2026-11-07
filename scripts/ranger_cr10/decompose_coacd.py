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

"""Stage 4.13b: one documented CoACD decomposition, Link3 and Link5 only.

CoACD is not installed in the training environment, so it is installed into an isolated directory
(`pip install --target /tmp/pip413 coacd`) and this script is run by an interpreter that puts that
directory first on `sys.path`. The training environment is untouched, and the version, parameters and
seed are written next to the pieces so a result can be reproduced or rejected.

Every convex piece is written to its own file. Loading them back must not re-merge them into a single
hull -- that is exactly the mistake that made the MuJoCo collision model 13-20 mm proud of the mesh.

Run (note the isolated path):
    PYTHONPATH=/tmp/pip413 python -I scripts/ranger_cr10/decompose_coacd.py --parts cr10_Link3 cr10_Link5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp/pip413")
import coacd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes/cr10_meshes"

PARTS = {"cr10_Link3": "Link3.STL", "cr10_Link5": "Link5.STL"}

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--parts", nargs="+", default=["cr10_Link3", "cr10_Link5"])
ap.add_argument("--threshold", type=float, default=0.01, help="CoACD concavity threshold (relative unless --real_metric)")
ap.add_argument("--resolution", type=int, default=4000)
ap.add_argument("--mcts_nodes", type=int, default=40)
ap.add_argument("--mcts_iterations", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--merge", action=argparse.BooleanOptionalAction, default=True,
                help="merge neighbouring convex hulls (CoACD default True); --no-merge keeps the split pieces")
# CoACD's own default is False (threshold in its relative mode). The previous version of this
# script recorded real_metric but never forwarded it, so the run silently used False; this flag
# now matches the library and is passed through explicitly.
ap.add_argument("--real_metric", action=argparse.BooleanOptionalAction, default=False,
                help="treat threshold as metres (CoACD default: False, a relative threshold)")
ap.add_argument("--out", default="outputs/stage413/b/decomposition")
args = ap.parse_args()


def read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Binary STL -> (vertices, faces), welded, without trimesh (this interpreter has numpy only)."""
    with path.open("rb") as handle:
        header = handle.read(84)
        count = struct.unpack("<I", header[80:84])[0]
        data = np.frombuffer(handle.read(count * 50), dtype=np.uint8).reshape(count, 50)
        triangles = data[:, 12:48].copy().view("<f4").reshape(count, 3, 3).astype(np.float64)
    flat = triangles.reshape(-1, 3)
    keys = np.round(flat, 7)
    _, index, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    vertices = flat[index]
    faces = inverse.reshape(-1, 3).astype(np.int32)
    return vertices, faces


def main() -> None:
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    try:
        version = metadata.version("coacd")
    except metadata.PackageNotFoundError:
        version = getattr(coacd, "__version__", "unknown")
    manifest = dict(
        tool="coacd",
        version=version,
        threshold=args.threshold,
        real_metric=args.real_metric,
        real_metric_forwarded=True,
        preprocess_mode="auto",
        preprocess_resolution=50,
        merge=args.merge,
        decimate=False,
        max_ch_vertex=256,
        max_convex_hull=-1,
        resolution=args.resolution,
        mcts_nodes=args.mcts_nodes,
        mcts_iterations=args.mcts_iterations,
        seed=args.seed,
        units="metres",
        parts={},
    )
    mode = "metric threshold" if args.real_metric else "relative threshold"
    print(f"coacd {manifest['version']}, {mode} {args.threshold:g}, seed {args.seed}")
    for part in args.parts:
        path = MESH_DIR / PARTS[part]
        vertices, faces = read_stl(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{part}: {len(vertices)} vertices, {len(faces)} faces, sha256 {digest[:16]}...")
        pieces = coacd.run_coacd(
            coacd.Mesh(vertices, faces),
            threshold=args.threshold,
            max_convex_hull=manifest["max_convex_hull"],
            preprocess_mode=manifest["preprocess_mode"],
            preprocess_resolution=manifest["preprocess_resolution"],
            resolution=args.resolution,
            mcts_nodes=args.mcts_nodes,
            mcts_iterations=args.mcts_iterations,
            merge=manifest["merge"],
            decimate=manifest["decimate"],
            max_ch_vertex=manifest["max_ch_vertex"],
            seed=args.seed,
            real_metric=args.real_metric,
        )
        records = []
        for index, (piece_vertices, piece_faces) in enumerate(pieces):
            # one file per piece: nothing on load can merge them back into one hull
            np.savez_compressed(
                out / f"{part}_p{index}.npz",
                vertices=np.asarray(piece_vertices, dtype=np.float64),
                faces=np.asarray(piece_faces, dtype=np.int32),
            )
            records.append(
                dict(piece=index, vertices=int(len(piece_vertices)), faces=int(len(piece_faces)),
                     file=f"{part}_p{index}.npz")
            )
        manifest["parts"][part] = dict(
            source=str(path.relative_to(ROOT)),
            sha256=digest,
            source_vertices=int(len(vertices)),
            source_faces=int(len(faces)),
            pieces=len(records),
            vertices_total=sum(r["vertices"] for r in records),
            detail=records,
        )
        print(f"  -> {len(records)} convex pieces, {sum(r['vertices'] for r in records)} vertices total")
    (out / "decomposition_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out / 'decomposition_manifest.json'}")


if __name__ == "__main__":
    main()
