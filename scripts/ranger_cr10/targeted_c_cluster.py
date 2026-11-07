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

"""One targeted C cluster near the corrected Link3 p3 region.

The cluster is a **targeted verification set**, not an unbiased holdout: seeds are searched with a
new RNG seed, must be inside the joint limits and use full FK, and the path is selected so that the
closest pair involves one of the new Link3 local pieces (index >= 4, i.e. not one of the four kept
old pieces). The original surface witness is recorded next to the candidate sub-pair ID; the path
parameter is called `s`, never a penetration depth. The numerical uncertain band stays separate.

Run:
    python scripts/ranger_cr10/targeted_c_cluster.py \
        --pieces3 outputs/stage413/b/correction_v1/link3_local_p3 \
        --pieces5 outputs/stage413/b/metric20 \
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
ap.add_argument("--pieces3", default="outputs/stage413/b/correction_v1/link3_local_p3")
ap.add_argument("--pieces5", default="outputs/stage413/b/metric20")
ap.add_argument("--old", default="outputs/stage413/b/metric20")
ap.add_argument("--out", default="outputs/stage413/b/targeted_cluster")
ap.add_argument("--seed", type=int, default=20261107)
ap.add_argument("--candidate-seeds", type=int, default=2500)
ap.add_argument("--screen-keep", type=int, default=200)
ap.add_argument("--max-directions", type=int, default=6)
ap.add_argument("--max-seeds", type=int, default=4)
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
    return r3.T @ r5, r3.T @ (p5 - p3), r3, p3


def piece_index(name: str) -> int:
    return int(name.rsplit("_p", 1)[1])


def new_region_query(pieces3_new, pieces5, rotation, translation):
    """Closest pair where the Link3 side is one of the new local pieces (index >= 4)."""
    best = float("inf")
    pair = ""
    hit = False
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
    return best, hit, pair


def is_contact(result):
    return result["intersects"] or (result["status"] == "separated" and result["distance"] <= CONTACT_TOL)


def main() -> None:
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    pieces3 = load_pieces("cr10_Link3", ROOT / args.pieces3)
    pieces5 = load_pieces("cr10_Link5", ROOT / args.pieces5)
    old3 = load_pieces("cr10_Link3", ROOT / args.old)
    old5 = load_pieces("cr10_Link5", ROOT / args.old)
    if not pieces3 or not pieces5:
        raise SystemExit("missing pieces")

    mesh3 = trimesh.load(str(MESH_DIR / "Link3.STL"), force="mesh", process=False)
    mesh5 = trimesh.load(str(MESH_DIR / "Link5.STL"), force="mesh", process=False)
    ref3 = MeshRef(np.asarray(mesh3.vertices), np.asarray(mesh3.faces), "Link3")
    ref5 = MeshRef(np.asarray(mesh5.vertices), np.asarray(mesh5.faces), "Link5")
    old_p3_vertices = next(vertices for name, vertices in old3 if name == "cr10_Link3_p3")
    p3_hull = ConvexHull(old_p3_vertices, qhull_options="QJ")

    # Known development + holdout seeds are excluded from the new seed search; this is a targeted
    # set, not an unbiased holdout, but the seed itself is new.
    known = [np.load(ROOT / "outputs/stage413/b/dev_seed_q.npy")]
    for path in ["outputs/stage413/b/holdout_generation/validation_results.json"]:
        data = json.loads((ROOT / path).read_text())
        q = [row["q"] for row in data["rows"] if row.get("type") == "C" and row.get("s") == 0.0 and row.get("q")]
        if q:
            known.append(np.array(q))
    known = np.concatenate(known, axis=0)
    np.save(out_dir / "known_seed_q.npy", known)

    rng = np.random.default_rng(args.seed)
    q_all = rng.uniform(LO, HI, size=(args.candidate_seeds, 6))
    keep = np.array([float(np.min(np.max(np.abs(known - q), axis=1))) >= 0.15 for q in q_all])
    q_all = q_all[keep]
    candidates = []
    for q in q_all:
        rotation, translation, *_ = fk(q)
        coarse = union_query(old3, old5, rotation, translation)
        if coarse["status"] == "failed":
            continue
        distance = 0.0 if coarse["intersects"] else coarse["distance"]
        if distance <= 0.040:
            candidates.append((float(distance), q))
    candidates.sort(key=lambda item: item[0])
    candidates = candidates[: args.screen_keep]
    print(f"seed search: {len(q_all)} allowed of {args.candidate_seeds}; "
          f"{len(candidates)} coarse near-contact survivors (<=40 mm)")

    screened = []
    for _, q in candidates:
        rotation, translation, *_ = fk(q)
        new_distance, new_hit, new_pair = new_region_query(pieces3, pieces5, rotation, translation)
        if new_hit or new_distance <= 0.020:
            screened.append((new_distance, q, new_pair))
    screened.sort(key=lambda item: item[0])
    print(f"new-region near-contact seeds: {len(screened)}")

    rows = []
    chosen = None
    for seed_index, (_, q0, seed_pair) in enumerate(screened[: args.max_seeds], start=1):
        rotation0, translation0, *_ = fk(q0)
        reference0 = ref3.closest_to_in_frame(ref5, rotation0, translation0, max_clearance=0.06)
        if reference0.status != "exact" or reference0.intersects or reference0.clearance > 0.03:
            continue
        witness = np.asarray(reference0.witness_a) if reference0.witness_a is not None else None
        p3_margin = None
        if witness is not None:
            p3_margin = float((p3_hull.equations[:, :3] @ witness + p3_hull.equations[:, 3]).max())
        print(f"seed {seed_index}: ref {reference0.clearance*1000:.2f} mm, new-region pair {seed_pair}, "
              f"p3 hull margin {None if p3_margin is None else round(p3_margin*1000,3)} mm")
        for attempt in range(args.max_directions):
            direction = rng.uniform(-1.0, 1.0, size=6)
            direction /= max(float(np.abs(direction).max()), 1e-9)

            def build(s):
                q = np.clip(q0 + s * direction, LO, HI)
                rotation, translation, *_ = fk(q)
                return rotation, translation, q

            grid = np.linspace(0.0, 1.6, 33)
            scan = []
            for s in grid:
                rotation, translation, q = build(float(s))
                result = union_query(pieces3, pieces5, rotation, translation)
                scan.append((float(s), result))
            first = next((i for i, (_, result) in enumerate(scan) if is_contact(result)), None)
            if first is None or first == 0:
                continue
            s_lo, s_hi = scan[first - 1][0], scan[first][0]
            for _ in range(16):
                mid = 0.5 * (s_lo + s_hi)
                rotation, translation, _ = build(mid)
                if is_contact(union_query(pieces3, pieces5, rotation, translation)):
                    s_hi = mid
                else:
                    s_lo = mid
            s_b = s_hi
            fractions = [0.0, 0.2, 0.4, 0.6, 0.75, 0.9, 0.97, 1.0, 1.02, 1.05, 1.10, 1.20, 1.35]
            s_values = sorted({round(s_b * f, 7) for f in fractions if s_b * f <= 1.6})
            evaluated = []
            for s in s_values:
                rotation, translation, q = build(s)
                result = ref3.closest_to_in_frame(ref5, rotation, translation, max_clearance=0.06)
                new_distance, new_hit, new_pair = new_region_query(pieces3, pieces5, rotation, translation)
                evaluated.append(dict(
                    s=float(s), q=q.tolist(),
                    reference_distance=float(result.clearance) if result.status in ("exact", "certified_culled") else None,
                    reference_status=result.status, reference_intersects=bool(result.intersects),
                    reference_triangle_a=int(result.triangle_a), reference_triangle_b=int(result.triangle_b),
                    reference_witness_a=None if result.witness_a is None else np.asarray(result.witness_a).tolist(),
                    reference_witness_b=None if result.witness_b is None else np.asarray(result.witness_b).tolist(),
                    new_region_distance=float(new_distance), new_region_intersects=bool(new_hit),
                    new_region_pair=new_pair,
                    candidate=union_query(pieces3, pieces5, rotation, translation),
                    p3_hull_margin=None if result.witness_a is None else float(
                        (p3_hull.equations[:, :3] @ np.asarray(result.witness_a) + p3_hull.equations[:, 3]).max()
                    ),
                ))
            overlaps = [row for row in evaluated if row["reference_intersects"]]
            near = [row for row in evaluated if not row["reference_intersects"] and row["reference_distance"] is not None
                    and row["reference_distance"] <= 0.005]
            if overlaps and near:
                chosen = dict(seed_index=seed_index, q0=q0.tolist(), direction=direction.tolist(),
                              seed_pair=seed_pair, seed_reference=float(reference0.clearance),
                              seed_p3_margin=p3_margin, attempt=attempt,
                              rows=evaluated, s_boundary=float(s_b))
                break
        if chosen:
            break
    if chosen is None:
        record = dict(found=False, seed=args.seed, candidate_seeds=args.candidate_seeds,
                      allowed=len(q_all), screened=len(screened),
                      budget=f"{args.candidate_seeds} joint samples, {args.max_seeds} seeds, "
                             f"{args.max_directions} directions; no near-contact + shallow overlap path")
    else:
        rows = chosen.pop("rows")
        for row in rows:
            row.update(cluster="N01", type="C_targeted")
        record = dict(found=True, cluster="N01", type="C_targeted",
                      note="targeted near the corrected Link3 p3 region; not an unbiased global holdout",
                      seed=args.seed, candidate_seeds=args.candidate_seeds,
                      allowed=len(q_all), screened=len(screened),
                      chosen=chosen,
                      rows=rows,
                      summary=dict(
                          n=len(rows),
                          overlaps=sum(1 for row in rows if row["reference_intersects"]),
                          near_contact=sum(1 for row in rows if not row["reference_intersects"]
                                           and row["reference_distance"] is not None and row["reference_distance"] <= 0.005),
                          uncertain=sum(1 for row in rows if not row["reference_intersects"]
                                        and row["reference_distance"] is not None
                                        and 0.0 < row["reference_distance"] <= CONTACT_TOL),
                          bins={label: sum(1 for row in rows if not row["reference_intersects"]
                                           and row["reference_distance"] is not None
                                           and lo <= row["reference_distance"] < hi)
                                for (lo, hi), label in zip(
                                    [(0, 1e-3), (1e-3, 3e-3), (3e-3, 5e-3), (5e-3, 10e-3), (10e-3, 20e-3)],
                                    ["0-1 mm", "1-3 mm", "3-5 mm", "5-10 mm", "10-20 mm"])},
                      ),
                      )
    out = out_dir / "targeted_cluster.json"
    out.write_text(json.dumps(record, indent=1))
    print(json.dumps({key: record[key] for key in ("found", "cluster", "summary", "budget") if key in record}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
