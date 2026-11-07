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

"""Replace one merged convex piece with original-geometry pieces from its region.

The merged candidate overfills in one localised region. Splitting the merged hull itself cannot
help: the union of the halves is the same hull. This script keeps the other old pieces and replaces
the offending one with pieces from a finer CoACD run **of the original mesh**, selected by their
centroid lying inside the old piece's hull. That is a region proposal, not a coverage proof; the
script therefore samples the original mesh triangles in the region and reports the observed
coverage gap, and samples the selected piece faces to report their overfill against the welded mesh.

The output is a new standalone piece directory plus a manifest mapping every new index to its
source file. Old IDs are not reused.

Run:
    python scripts/ranger_cr10/build_local_replacement.py \
        --link Link3 --old outputs/stage413/b/metric20 --fine outputs/stage413/b/correction_v1/link3_t2mm_nomerge \
        --target cr10_Link3_p3 --out outputs/stage413/b/correction_v1/link3_local_p3 \
        --manifest outputs/stage413/b/correction_v1/link3_local_p3_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from convex_distance import gjk_distance  # noqa: E402
from mesh_distance import MeshDistance  # noqa: E402
from reference_geometry import MeshRef  # noqa: E402

ROOT = HERE.parents[1]
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes/cr10_meshes"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--link", required=True, choices=["Link3", "Link5"])
ap.add_argument("--old", required=True)
ap.add_argument("--fine", required=True)
ap.add_argument("--target", required=True, help="old piece stem, e.g. cr10_Link3_p3")
ap.add_argument("--out", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--coverage-samples", type=int, default=400)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(part: str, directory: Path):
    out = []
    for path in sorted(directory.glob(f"{part}_p*.npz")):
        data = np.load(path)
        out.append(dict(stem=path.stem, path=path,
                        vertices=np.asarray(data["vertices"], dtype=np.float64),
                        faces=np.asarray(data["faces"], dtype=np.int64)))
    return out


def sample_triangles(triangles: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(triangles) == 0:
        return np.zeros((0, 3))
    index = rng.choice(len(triangles), size=min(count, len(triangles)), replace=True)
    weights = rng.dirichlet([1.0, 1.0, 1.0], size=len(index))
    return np.einsum("ij,ijk->ik", weights, triangles[index])


def point_union_distance(points: np.ndarray, pieces) -> np.ndarray:
    out = np.full(len(points), np.inf)
    for piece in pieces:
        for index, point in enumerate(points):
            result = gjk_distance(point[None, :], piece["vertices"], tol=1e-8)
            distance = 0.0 if result.status == "intersecting" else result.distance
            if np.isfinite(distance):
                out[index] = min(out[index], distance)
    return out


def outside_statistics(pieces, mesh_distance, mesh_ref, samples: int, rng: np.random.Generator):
    triangles = [piece["vertices"][piece["faces"]] for piece in pieces if len(piece["faces"])]
    triangles = np.concatenate(triangles, axis=0) if triangles else np.zeros((0, 3, 3))
    points = sample_triangles(triangles, samples, rng)
    unsigned = mesh_distance.distance(points)
    outside = np.array([mesh_ref.contains(point) is False for point in points])
    outside_distance = unsigned[outside & np.isfinite(unsigned)]
    return dict(
        n=int(len(points)),
        outside=int(outside.sum()) if len(points) else 0,
        outside_fraction=float(outside.mean()) if len(points) else 0.0,
        outside_p95_m=float(np.percentile(outside_distance, 95)) if len(outside_distance) else 0.0,
        outside_max_m=float(np.max(outside_distance)) if len(outside_distance) else 0.0,
        outside_over_3mm=int((outside_distance > 3e-3).sum()) if len(outside_distance) else 0,
    )


def main() -> None:
    part = f"cr10_{args.link}"
    old = load(part, ROOT / args.old)
    fine = load(part, ROOT / args.fine)
    target = next((piece for piece in old if piece["stem"] == args.target), None)
    if target is None:
        raise SystemExit(f"target {args.target} not found in {args.old}")
    hull = ConvexHull(target["vertices"], qhull_options="QJ")

    selected = []
    for piece in fine:
        centroid = piece["vertices"].mean(axis=0)
        violation = float((hull.equations[:, :3] @ centroid + hull.equations[:, 3]).max())
        if violation <= 1e-9:
            selected.append(piece)

    mesh = trimesh.load(str(MESH_DIR / f"{args.link}.STL"), force="mesh", process=False)
    mesh.merge_vertices(merge_tex=False, merge_norm=False)
    mesh.remove_unreferenced_vertices()
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    values = vertices @ hull.equations[:, :3].T + hull.equations[:, 3]
    inside_vertices = values.max(axis=1) <= 1e-9
    region_faces = inside_vertices[faces].all(axis=1)
    region_triangles = vertices[faces[region_faces]]
    rng = np.random.default_rng(args.seed)
    region_points = sample_triangles(region_triangles, args.coverage_samples, rng)
    coverage = point_union_distance(region_points, selected) if len(selected) else np.full(len(region_points), np.inf)

    mesh_distance = MeshDistance(vertices, faces)
    welded = mesh.copy()
    welded.merge_vertices(merge_tex=False, merge_norm=False)
    welded.remove_unreferenced_vertices()
    mesh_ref = MeshRef(np.asarray(welded.vertices), np.asarray(welded.faces), args.link)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob(f"{part}_p*.npz"):
        existing.unlink()

    kept = [piece for piece in old if piece["stem"] != args.target]
    order = kept + selected
    mapping = []
    for index, piece in enumerate(order):
        destination = out_dir / f"{part}_p{index}.npz"
        shutil.copy2(piece["path"], destination)
        mapping.append(dict(new_index=index, new_stem=destination.stem, source_stem=piece["stem"],
                            source_path=str(piece["path"].relative_to(ROOT)), sha256=sha(destination)))

    record = dict(
        link=args.link,
        old_dir=args.old,
        fine_dir=args.fine,
        target=args.target,
        selection_rule="fine piece centroid inside the old target hull (region proposal; coverage below)",
        target_vertices=int(len(target["vertices"])),
        target_faces=int(len(target["faces"])),
        old_pieces=len(old),
        fine_pieces=len(fine),
        selected_fine_pieces=len(selected),
        kept_old_pieces=len(kept),
        new_piece_count=len(order),
        region_faces=int(region_faces.sum()),
        region_area_m2=float(mesh.area_faces[region_faces].sum()),
        coverage=dict(
            n=int(len(coverage)),
            median_m=float(np.median(coverage)) if len(coverage) else float("nan"),
            p95_m=float(np.percentile(coverage, 95)) if len(coverage) else float("nan"),
            max_m=float(np.max(coverage)) if len(coverage) else float("nan"),
            over_0_5mm=int((coverage > 0.5e-3).sum()) if len(coverage) else 0,
            over_1mm=int((coverage > 1e-3).sum()) if len(coverage) else 0,
        ),
        overfill=dict(
            old_target=outside_statistics([target], mesh_distance, mesh_ref, args.coverage_samples, rng),
            selected_union=outside_statistics(selected, mesh_distance, mesh_ref, args.coverage_samples, rng),
        ),
        mapping=mapping,
        note="old IDs are not reused; the mapping above is the only link from new indices to sources",
    )
    manifest = ROOT / args.manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(record, indent=1))
    print(json.dumps({key: record[key] for key in (
        "target", "selected_fine_pieces", "kept_old_pieces", "new_piece_count",
        "region_faces", "coverage", "overfill",
    )}, indent=1))
    print(f"wrote {out_dir} and {manifest}")


if __name__ == "__main__":
    main()
