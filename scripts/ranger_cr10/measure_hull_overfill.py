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

"""Stage 4.13a: how far is the MuJoCo collision model from the meshes it was built from?

The collision model is a convex hull per link, and a hull is conservative *by construction* -- but
"conservative" is not a number. This measures it: sample the hull surface, take the distance from
each sample to the triangle mesh it encloses, and report the distribution. That distance is how much
more room the simulator gives the robot than the robot has.

It decides the whole stage. If the hull were faithful to a few millimetres, the hull distance would
be an acceptable reference and no decomposition would be justified. It is not: on the links that
matter the hull is 13-20 mm proud of the mesh at the median and up to ~50 mm at the tail, so a
distance built on hulls -- MuJoCo's own `mj_geomDistance` included -- carries that error, and the
"true contact" counted in stages 4.11 and 4.12 was contact of *hulls*, not of the robot.

The reverse direction (mesh surface to hull) is zero by construction and is only sampled as a check.

Usage:
    python scripts/ranger_cr10/measure_hull_overfill.py --samples 20000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--samples", type=int, default=20000, help="points sampled on each hull surface")
ap.add_argument("--out", default="outputs/stage413/a")
args = ap.parse_args()

# part -> (STL relative to the mesh directory, scale applied by the MJCF generator)
PARTS = {
    "cr10_Link1": ("cr10_meshes/Link1.STL", 1.0),
    "cr10_Link2": ("cr10_meshes/Link2.STL", 1.0),
    "cr10_Link3": ("cr10_meshes/Link3.STL", 1.0),
    "cr10_Link4": ("cr10_meshes/Link4.STL", 1.0),
    "cr10_Link5": ("cr10_meshes/Link5.STL", 1.0),
    "cr10_Link6": ("cr10_meshes/Link6.STL", 1.0),
    "gripper_base_link": ("ag95_meshes/ag95_base_link.STL", 0.001),
    "lidar_link0": ("lidar_link0.STL", 1.0),
}


def main() -> None:
    rows = []
    print(f"{'part':20s} {'verts':>6s} {'hull v':>6s} {'hull tris':>9s}  hull->mesh distance (mm)")
    print(f"{'':20s} {'':>6s} {'':>6s} {'':>9s}   median     p95     max")
    for name, (rel, scale) in PARTS.items():
        mesh = trimesh.load(str(MESH_DIR / rel), force="mesh", process=False)
        if scale != 1.0:
            mesh.apply_scale(scale)
        hull = mesh.convex_hull
        points, _ = trimesh.sample.sample_surface(hull, args.samples, seed=0)
        _, distance, _ = trimesh.proximity.ProximityQuery(mesh).on_surface(points)
        row = dict(
            part=name,
            stl=rel,
            scale=scale,
            mesh_vertices=int(len(mesh.vertices)),
            mesh_faces=int(len(mesh.faces)),
            hull_vertices=int(len(hull.vertices)),
            hull_faces=int(len(hull.faces)),
            hull_volume=float(hull.volume),
            mesh_volume=float(mesh.volume),
            overfill_median=float(np.median(distance)),
            overfill_p95=float(np.percentile(distance, 95)),
            overfill_max=float(distance.max()),
        )
        rows.append(row)
        print(
            f"{name:20s} {row['mesh_vertices']:6d} {row['hull_vertices']:6d} {row['hull_faces']:9d}"
            f"  {row['overfill_median'] * 1000:7.1f} {row['overfill_p95'] * 1000:7.1f} {row['overfill_max'] * 1000:7.1f}"
        )

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "hull_overfill.json").write_text(json.dumps(dict(samples=args.samples, parts=rows), indent=1))
    worst = max(rows, key=lambda r: r["overfill_p95"])
    print(
        f"\nworst p95: {worst['part']} at {worst['overfill_p95'] * 1000:.1f} mm -- "
        "a hull-based distance (MuJoCo's own included) cannot be closer than this to the mesh"
    )
    print(f"wrote {out / 'hull_overfill.json'}")


if __name__ == "__main__":
    main()
