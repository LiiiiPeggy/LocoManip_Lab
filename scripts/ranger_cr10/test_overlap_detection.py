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

"""Stage 4.13b: does the decomposition *miss* a real overlap?

Translating the whole link along the centre line barely closes the gap (measured 0.63 mm of
distance per millimetre of shift), so the first synthetic sweep never reached contact. This walks
Link5 along the **closest-approach direction** -- the vector between the reference's witness points
-- until the meshes actually intersect, and at each step asks the single hull and the convex
decomposition whether they see the same thing.

That is the evidence the candidate was missing: a convex cover can remove a false alarm by being
tighter and still miss a real overlap by being smaller than the mesh. Only the second column of this
table can tell the two apart.

Usage:
    python scripts/ranger_cr10/test_overlap_detection.py --pieces outputs/stage413/b/metric20 --steps 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/ranger_cr10"))

from convex_distance import gjk_distance  # noqa: E402
from reference_geometry import MeshRef  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--pieces", default="outputs/stage413/b/metric20")
ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--out", default="outputs/stage413/b")
args = ap.parse_args()

SCENE = ROOT / "mujoco/robots/ranger_cr10/scene.xml"
POSES = ROOT / "outputs/stage411/operating_poses.npz"
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes/cr10_meshes"

import mujoco  # noqa: E402


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    bid = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)}
    adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)]
    poses = np.load(POSES)["poses"]

    mesh3 = trimesh.load(str(MESH_DIR / "Link3.STL"), force="mesh", process=False)
    mesh5 = trimesh.load(str(MESH_DIR / "Link5.STL"), force="mesh", process=False)
    ref3 = MeshRef(np.asarray(mesh3.vertices), np.asarray(mesh3.faces), "Link3")
    ref5 = MeshRef(np.asarray(mesh5.vertices), np.asarray(mesh5.faces), "Link5")
    hull3 = MeshRef(np.asarray(mesh3.convex_hull.vertices), np.asarray(mesh3.convex_hull.faces), "hull3")
    hull5 = MeshRef(np.asarray(mesh5.convex_hull.vertices), np.asarray(mesh5.convex_hull.faces), "hull5")
    pieces3 = [np.asarray(np.load(p)["vertices"]) for p in sorted((ROOT / args.pieces).glob("cr10_Link3_p*.npz"))]
    pieces5 = [np.asarray(np.load(p)["vertices"]) for p in sorted((ROOT / args.pieces).glob("cr10_Link5_p*.npz"))]
    print(f"pieces: {len(pieces3)} x {len(pieces5)}; walking along the closest-approach direction")

    q = poses[93]
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for address, value in zip(adr, q):
        data.qpos[address] = value
    mujoco.mj_forward(model, data)
    r3, p3 = data.xmat[bid["cr10_Link3"]].reshape(3, 3), data.xpos[bid["cr10_Link3"]]
    r5, p5 = data.xmat[bid["cr10_Link5"]].reshape(3, 3), data.xpos[bid["cr10_Link5"]]

    first = ref3.closest_to_in_frame(ref5, r3.T @ r5, r3.T @ (p5 - p3), max_clearance=0.05)
    if first.witness_a is None or first.witness_b is None:
        raise SystemExit("reference gave no witness points; cannot define a direction")
    # world-frame direction from Link5's closest point toward Link3's
    direction_world = r3 @ first.witness_a - (r3 @ first.witness_b)
    distance0 = float(np.linalg.norm(direction_world))
    direction_world /= max(distance0, 1e-9)
    print(f"pose 93: separation {first.clearance * 1000:.1f} mm, direction length {distance0 * 1000:.1f} mm")

    rows = []
    for step in range(args.steps):
        advance = min(distance0 * (step / max(args.steps - 1, 1)) * 1.4, distance0 * 1.4)
        shifted = p5 + direction_world * advance
        rotation, translation = r3.T @ r5, r3.T @ (shifted - p3)
        reference = ref3.closest_to_in_frame(ref5, rotation, translation, max_clearance=0.05)
        hull = hull3.closest_to_in_frame(hull5, rotation, translation, max_clearance=0.05)
        best_piece = np.inf
        piece_hit = False
        for a in pieces3:
            for b in pieces5:
                result = gjk_distance(a, b @ rotation.T + translation, tol=1e-8)
                if result.status == "intersecting":
                    piece_hit = True
                    best_piece = 0.0
                    break
                if result.status == "separated":
                    best_piece = min(best_piece, result.distance)
            if piece_hit:
                break
        rows.append(dict(
            advance=float(advance),
            reference=float(reference.clearance), reference_hit=bool(reference.intersects),
            hull=max(0.0, float(hull.clearance)), hull_hit=bool(hull.intersects),
            pieces=float(min(best_piece, 0.05)), pieces_hit=bool(piece_hit),
        ))
        print(f"  advance {advance * 1000:6.1f} mm | ref {rows[-1]['reference'] * 1000:7.2f} hit {int(reference.intersects)} "
              f"| hull {rows[-1]['hull'] * 1000:6.2f} hit {int(hull.intersects)} "
              f"| pieces {rows[-1]['pieces'] * 1000:6.2f} hit {int(piece_hit)}")

    overlaps = [r for r in rows if r["reference_hit"]]
    missed = [r for r in overlaps if not r["pieces_hit"]]
    false_pieces = [r for r in rows if not r["reference_hit"] and r["pieces_hit"]]
    print(f"\nreference overlaps {len(overlaps)}/{len(rows)}; pieces missed {len(missed)}; pieces false {len(false_pieces)}")
    if overlaps:
        print(f"missing rate on this sample: {len(missed) / len(overlaps):.1%}")
    else:
        print("missing rate: N/A (no reference overlap reached within the walked distance)")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "overlap_detection.json").write_text(json.dumps(dict(rows=rows, missed=len(missed), overlaps=len(overlaps)), indent=1))
    print(f"wrote {out / 'overlap_detection.json'}")


if __name__ == "__main__":
    main()
