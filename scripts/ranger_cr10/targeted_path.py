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

"""Take a screened seed near the corrected Link3 p3 region and walk one local joint path.

Search correction (2026-11-07): a seed whose independent reference is separated is a valid start
even if the candidate is already in conservative contact at s=0. Future boundary searches must not
gate on candidate separation; confirm the start with the reference instead.


The seed screen (`screen_targeted.py`) proposes configurations whose closest pair involves a new
Link3 local piece. This script evaluates the reference at the seed and along a small number of
path points, so the expensive exact query is used only where the candidate already reports a
contact. The output is a targeted verification cluster, not an unbiased holdout.

Run:
    python scripts/ranger_cr10/targeted_path.py \
        --seeds outputs/stage413/b/targeted_cluster/screened_seeds.json \
        --out outputs/stage413/b/targeted_cluster
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from candidate_eval import CONTACT_TOL, load_pieces, union_query  # noqa: E402
from convex_distance import gjk_distance  # noqa: E402
from reference_geometry import MeshRef  # noqa: E402

ROOT = HERE.parents[1]
SCENE = ROOT / "mujoco/robots/ranger_cr10/scene.xml"
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes/cr10_meshes"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--seeds", default="outputs/stage413/b/targeted_cluster/screened_seeds.json")
ap.add_argument("--out", default="outputs/stage413/b/targeted_cluster")
ap.add_argument("--max-seeds", type=int, default=8)
ap.add_argument("--max-directions", type=int, default=3)
ap.add_argument("--seed", type=int, default=20261107)
args = ap.parse_args()

MODEL = mujoco.MjModel.from_xml_path(str(SCENE))
DATA = mujoco.MjData(MODEL)
HOME = MODEL.key("home").id
BID = {mujoco.mj_id2name(MODEL, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(MODEL.nbody)}
ARM_ADR = [MODEL.jnt_qposadr[mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)]
JOINT_IDS = [mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}") for i in range(1, 7)]
LO = np.array([MODEL.jnt_range[j][0] for j in JOINT_IDS])
HI = np.array([MODEL.jnt_range[j][1] for j in JOINT_IDS])


def fk(q):
    mujoco.mj_resetDataKeyframe(MODEL, DATA, HOME)
    for address, value in zip(ARM_ADR, q):
        DATA.qpos[address] = float(value)
    mujoco.mj_kinematics(MODEL, DATA)
    r3 = np.array(DATA.xmat[BID["cr10_Link3"]], dtype=np.float64).reshape(3, 3)
    p3 = np.array(DATA.xpos[BID["cr10_Link3"]], dtype=np.float64)
    r5 = np.array(DATA.xmat[BID["cr10_Link5"]], dtype=np.float64).reshape(3, 3)
    p5 = np.array(DATA.xpos[BID["cr10_Link5"]], dtype=np.float64)
    return r3.T @ r5, r3.T @ (p5 - p3)


def piece_index(name: str) -> int:
    return int(name.rsplit("_p", 1)[1])


def new_region_query(pieces3_new, pieces5, rotation, translation):
    best = float("inf")
    pair = ""
    for name_a, vertices_a in pieces3_new:
        if piece_index(name_a) < 4:
            continue
        for name_b, vertices_b in pieces5:
            result = gjk_distance(vertices_a, vertices_b @ rotation.T + translation, tol=1e-8)
            if result.status == "intersecting":
                return 0.0, True, f"{name_a}|{name_b}"
            if result.status == "separated" and result.distance < best:
                best = result.distance
                pair = f"{name_a}|{name_b}"
    return best, False, pair


def contact(result):
    return result["intersects"] or (result["status"] == "separated" and result["distance"] <= CONTACT_TOL)


def main() -> None:
    pieces3 = load_pieces("cr10_Link3", ROOT / "outputs/stage413/b/correction_v1/link3_local_p3")
    pieces5 = load_pieces("cr10_Link5", ROOT / "outputs/stage413/b/metric20")
    old3 = load_pieces("cr10_Link3", ROOT / "outputs/stage413/b/metric20")
    old_p3_vertices = next(vertices for name, vertices in old3 if name == "cr10_Link3_p3")
    p3_hull = ConvexHull(old_p3_vertices, qhull_options="QJ")
    mesh3 = trimesh.load(str(MESH_DIR / "Link3.STL"), force="mesh", process=False)
    mesh5 = trimesh.load(str(MESH_DIR / "Link5.STL"), force="mesh", process=False)
    ref3 = MeshRef(np.asarray(mesh3.vertices), np.asarray(mesh3.faces), "Link3")
    ref5 = MeshRef(np.asarray(mesh5.vertices), np.asarray(mesh5.faces), "Link5")
    seeds = json.loads((ROOT / args.seeds).read_text())
    rng = np.random.default_rng(args.seed)

    result = None
    for seed_index, seed in enumerate(seeds[: args.max_seeds], start=1):
        q0 = np.asarray(seed["q"], dtype=float)
        rotation0, translation0 = fk(q0)
        reference0 = ref3.closest_to_in_frame(ref5, rotation0, translation0, max_clearance=0.06)
        if reference0.status != "exact" or reference0.intersects or reference0.clearance > 0.03:
            print(f"seed {seed_index}: skipped (ref {reference0.status} {reference0.clearance*1000:.2f} mm)", flush=True)
            continue
        witness = np.asarray(reference0.witness_a) if reference0.witness_a is not None else None
        margin = None if witness is None else float(
            (p3_hull.equations[:, :3] @ witness + p3_hull.equations[:, 3]).max()
        )
        print(f"seed {seed_index}: ref {reference0.clearance*1000:.2f} mm, seed pair {seed['new_region_pair']}, "
              f"p3 margin {None if margin is None else round(margin*1000, 3)} mm", flush=True)
        if margin is not None and margin > 0.050:
            print(f"  skipped: witness is {margin*1000:.2f} mm outside the old p3 hull (>50 mm budget)", flush=True)
            continue
        for attempt in range(args.max_directions):
            direction = rng.uniform(-1.0, 1.0, size=6)
            direction /= max(float(np.abs(direction).max()), 1e-9)

            def build(s):
                q = np.clip(q0 + s * direction, LO, HI)
                rotation, translation = fk(q)
                return rotation, translation, q

            scan = []
            for s in np.linspace(-0.8, 1.6, 49):
                rotation, translation, _ = build(float(s))
                scan.append((float(s), union_query(pieces3, pieces5, rotation, translation)))
            first = next((i for i, (_, res) in enumerate(scan) if contact(res)), None)
            # The 2026-11-07 correction: do not require a candidate-separated start. A reference-separated
            # seed is valid even if the candidate is already conservatively in contact at s=0; the
            # reference decides the start, and the path can move backward to a separated state.
            if first is None or first == 0:
                continue
            s_lo, s_hi = scan[first - 1][0], scan[first][0]
            for _ in range(16):
                mid = 0.5 * (s_lo + s_hi)
                rotation, translation, _ = build(mid)
                if contact(union_query(pieces3, pieces5, rotation, translation)):
                    s_hi = mid
                else:
                    s_lo = mid
            s_b = s_hi
            rows = []
            s_start = s_lo
            for fraction in (0.0, 0.25, 0.5, 0.7, 0.85, 0.95, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0):
                s = s_start + fraction * (s_b - s_start)
                rotation, translation, q = build(s)
                reference = ref3.closest_to_in_frame(ref5, rotation, translation, max_clearance=0.06)
                new_distance, new_hit, new_pair = new_region_query(pieces3, pieces5, rotation, translation)
                candidate = union_query(pieces3, pieces5, rotation, translation)
                rows.append(dict(
                    s=float(s), q=q.tolist(),
                    reference_distance=float(reference.clearance) if reference.status in ("exact", "certified_culled") else None,
                    reference_status=reference.status, reference_intersects=bool(reference.intersects),
                    reference_triangle_a=int(reference.triangle_a), reference_triangle_b=int(reference.triangle_b),
                    reference_witness_a=None if reference.witness_a is None else np.asarray(reference.witness_a).tolist(),
                    reference_witness_b=None if reference.witness_b is None else np.asarray(reference.witness_b).tolist(),
                    candidate_distance=candidate["distance"], candidate_intersects=candidate["intersects"],
                    candidate_pair=candidate["pair"],
                    new_region_distance=float(new_distance), new_region_intersects=bool(new_hit),
                    new_region_pair=new_pair,
                    p3_hull_margin=None if reference.witness_a is None else float(
                        (p3_hull.equations[:, :3] @ np.asarray(reference.witness_a) + p3_hull.equations[:, 3]).max()
                    ),
                ))
            overlaps = [row for row in rows if row["reference_intersects"]]
            near = [row for row in rows if not row["reference_intersects"] and row["reference_distance"] is not None
                    and row["reference_distance"] <= 0.005]
            print(f"  direction {attempt}: s_b {s_b:.5f}, overlaps {len(overlaps)}, near {len(near)}", flush=True)
            if overlaps and near:
                result = dict(found=True, cluster="N01", type="C_targeted",
                              note="targeted near the corrected Link3 p3 region; not an unbiased global holdout",
                              seed_index=seed_index, q0=q0.tolist(), direction=direction.tolist(),
                              seed_reference=float(reference0.clearance), seed_p3_margin=margin,
                              seed_pair=seed["new_region_pair"], s_start=float(s_start),
                              s_boundary=float(s_b), rows=rows)
                break
        if result:
            break

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    if result is None:
        result = dict(found=False, cluster="N01", type="C_targeted",
                      budget=f"{args.max_seeds} seeds x {args.max_directions} directions; no overlap+near path",
                      source=args.seeds)
    else:
        result["summary"] = dict(
            n=len(result["rows"]),
            overlaps=sum(1 for row in result["rows"] if row["reference_intersects"]),
            near_contact=sum(1 for row in result["rows"] if not row["reference_intersects"]
                             and row["reference_distance"] is not None and row["reference_distance"] <= 0.005),
            uncertain=sum(1 for row in result["rows"] if not row["reference_intersects"]
                          and row["reference_distance"] is not None and 0.0 < row["reference_distance"] <= CONTACT_TOL),
            bins={label: sum(1 for row in result["rows"] if not row["reference_intersects"]
                             and row["reference_distance"] is not None and lo <= row["reference_distance"] < hi)
                  for (lo, hi), label in zip(
                      [(0, 1e-3), (1e-3, 3e-3), (3e-3, 5e-3), (5e-3, 10e-3), (10e-3, 20e-3)],
                      ["0-1 mm", "1-3 mm", "3-5 mm", "5-10 mm", "10-20 mm"])},
        )
    path = out_dir / "targeted_cluster_path.json"
    path.write_text(json.dumps(result, indent=1))
    print(json.dumps({key: result[key] for key in ("found", "summary", "budget", "seed_pair") if key in result}, indent=1), flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
