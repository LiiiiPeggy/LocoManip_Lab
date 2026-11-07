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

"""Build one minimal radar candidate: a convex decomposition of the Link2 collision mesh.

This replaces the single Link2 convex hull used by D for the Link2-lidar pair. No old hull is
retained: the candidate distance is the union-min over these pieces. The pieces are generated
from the original ``Link2.STL`` with the documented CoACD settings, one file per piece; the
manifest records the source hash, CoACD version/parameters, seed and per-file hashes.

This script is only run after the MM/HM/MH/HH attribution selected Link2. It does not alter the
frozen Link3<->Link5 candidate or any deployment asset.

Run (CoACD is isolated in /tmp/pip413; the training env is untouched):
    python -I scripts/ranger_cr10/build_radar_link2_candidate.py
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
MESH = ROOT / "rangerboxcr10lidar_description/meshes/cr10_meshes/Link2.STL"
OUT = ROOT / "outputs/stage413/b/radar_round2_2026_11_07/candidate_link2_v1"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--threshold", type=float, default=0.002)
ap.add_argument("--real-metric", action=argparse.BooleanOptionalAction, default=True)
ap.add_argument("--resolution", type=int, default=4000)
ap.add_argument("--mcts-nodes", type=int, default=40)
ap.add_argument("--mcts-iterations", type=int, default=200)
ap.add_argument("--merge", action=argparse.BooleanOptionalAction, default=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=str(OUT))
args = ap.parse_args()


def read_stl(path: Path):
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
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vertices, faces = read_stl(MESH)
    source_sha = hashlib.sha256(MESH.read_bytes()).hexdigest()
    try:
        version = metadata.version("coacd")
    except metadata.PackageNotFoundError:
        version = getattr(coacd, "__version__", "unknown")
    params = dict(
        threshold=args.threshold, real_metric=args.real_metric, real_metric_forwarded=True,
        preprocess_mode="auto", preprocess_resolution=50, merge=args.merge, decimate=False,
        max_ch_vertex=256, max_convex_hull=-1, resolution=args.resolution,
        mcts_nodes=args.mcts_nodes, mcts_iterations=args.mcts_iterations, seed=args.seed,
        units="metres",
    )
    pieces = coacd.run_coacd(
        coacd.Mesh(vertices, faces), threshold=params["threshold"],
        max_convex_hull=params["max_convex_hull"], preprocess_mode=params["preprocess_mode"],
        preprocess_resolution=params["preprocess_resolution"], resolution=params["resolution"],
        mcts_nodes=params["mcts_nodes"], mcts_iterations=params["mcts_iterations"],
        merge=params["merge"], decimate=params["decimate"], max_ch_vertex=params["max_ch_vertex"],
        seed=params["seed"], real_metric=params["real_metric"],
    )
    records = []
    for index, (piece_vertices, piece_faces) in enumerate(pieces):
        path = out / f"cr10_Link2_p{index}.npz"
        np.savez_compressed(path, vertices=np.asarray(piece_vertices, dtype=np.float64),
                            faces=np.asarray(piece_faces, dtype=np.int32))
        records.append(dict(piece=index, file=path.name, vertices=int(len(piece_vertices)),
                            faces=int(len(piece_faces)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest = dict(
        candidate="radar_link2_link2_decomposition_v1",
        source_mesh=str(MESH.relative_to(ROOT)), source_sha256=source_sha,
        tool="coacd", version=version, parameters=params, seed=args.seed,
        pieces=len(records), piece_records=records,
        replacement="the single Link2 convex hull in D is replaced by the union of these pieces; "
                    "no old hull is retained",
        old_coverage="the D arm hull for Link2 is the previous coverage; it is not part of the candidate",
        note="one file per convex piece; the candidate distance is the minimum over atomic pieces",
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(dict(version=version, pieces=len(records), source_sha256=source_sha,
                          files=[row["file"] for row in records]), indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
