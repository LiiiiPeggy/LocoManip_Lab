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

"""Stage 4.13b0: replay the recorded poses and attribute every alarm to a real one or a phantom.

For each pose and each priority pair this writes down the three distances on the *same* objects
(`collision_parts.py`), with the constituent that produced each one:

    d_ref     exact triangle-mesh distance, with an intersection flag (reference_geometry.py)
    d_mujoco  MuJoCo's convex hull / primitive distance
    d_box     the box proxy the reward uses

and classifies the step: a real contact (reference says intersecting), a hull-only phantom (the
hulls touch, the meshes do not), a box-proxy alarm (the box cover touches, the hulls do not), or
clear. Nothing is averaged across pairs: pair-steps and "any pair" steps are reported separately.

Usage:
    python scripts/ranger_cr10/replay_collision_attribution.py --poses 200
    python scripts/ranger_cr10/replay_collision_attribution.py --poses 1800 --out outputs/stage413/b0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/ranger_cr10"))

from collision_parts import PRIORITY, box_mesh, constituents_of  # noqa: E402
from reference_geometry import MeshRef  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--poses", type=int, default=200)
ap.add_argument("--band", type=float, default=0.05, help="only pairs this close are worth a full query")
ap.add_argument("--out", default="outputs/stage413/b0")
ap.add_argument("--anomalies", type=int, default=6)
ap.add_argument("--only_anomalies", action="store_true", help="evaluate just the hull-contact poses")
ap.add_argument("--pairs", default="", help="semicolon separated subset, e.g. 'cr10_Link3 cr10_Link5'")
ap.add_argument("--pose-indices", default="", help="comma separated pose indices; overrides the --poses step")
ap.add_argument("--box-margin", type=float, default=0.0, help="box alarm threshold; the reward margin is 0.02")
args = ap.parse_args()

SCENE = ROOT / "mujoco/robots/ranger_cr10/scene.xml"
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes"
POSES = ROOT / "outputs/stage411/operating_poses.npz"
SAFETY = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/safety.py"

import mujoco  # noqa: E402


def load_safety():
    import importlib.util

    spec = importlib.util.spec_from_file_location("safety_replay", SAFETY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RAW: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def raw_geometry(part) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and faces of one constituent in its link frame, loaded once."""
    key = f"{part.group}/{part.name}"
    if key not in _RAW:
        if part.kind == "mesh":
            mesh = trimesh.load(str(MESH_DIR / part.stl), force="mesh", process=False)
            _RAW[key] = (np.asarray(mesh.vertices) * part.scale, np.asarray(mesh.faces))
        else:
            _RAW[key] = box_mesh(part.centre, part.half, part.quat)
    return _RAW[key]


def reference_frame(group: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    return [(part.name, *raw_geometry(part)) for part in constituents_of(group)]


def main() -> None:
    import torch

    safety = load_safety()
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    bid = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)}
    geom_id = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g): g for g in range(model.ngeom)}
    arm_adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)]

    poses = np.load(POSES)["poses"]
    print(f"poses available {len(poses)}; scanning {min(args.poses, len(poses))}")

    # --- the six anomalous configurations: hull contact on Link3 <-> Link5 -------------------
    anomalies = []
    for index, q in enumerate(poses):
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        for address, value in zip(arm_adr, q):
            data.qpos[address] = value
        mujoco.mj_forward(model, data)
        ga, gb = geom_id["cr10_Link3_col"], geom_id["cr10_Link5_col"]
        distance = mujoco.mj_geomDistance(model, data, ga, gb, 0.02, None)
        if distance <= 0.0:
            anomalies.append((index, q.copy(), float(distance)))
            if len(anomalies) >= args.anomalies:
                break
    print(f"hull-contact anomalies found: {len(anomalies)}")

    active_pairs = [p for p in PRIORITY if not args.pairs or " ".join(p) in args.pairs.split(";")]
    # static reference geometry: one index per constituent, built once
    REF_A, REF_B = {}, {}
    for group in {g for pair in PRIORITY for g in pair}:
        entries = []
        for name, vertices, faces in reference_frame(group):
            entries.append((name, MeshRef(vertices, faces, name=f"{group}/{name}")))
        REF_A[group] = entries
        REF_B[group] = entries

    rows = []
    step = max(1, len(poses) // args.poses)
    if args.pose_indices:
        selected = [int(value) for value in args.pose_indices.split(",") if value.strip()]
    else:
        selected = [index for index, _, _ in anomalies] if args.only_anomalies else list(range(0, len(poses), step))

    for counter, index in enumerate(selected):
        q = poses[index]
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        for address, value in zip(arm_adr, q):
            data.qpos[address] = value
        mujoco.mj_forward(model, data)

        names = list(dict.fromkeys(safety.ARM_BODY_LIST + safety.PROT_BODY_LIST))
        position = torch.tensor(np.stack([data.xpos[bid[n]] for n in names])[None].astype(np.float32))
        quat = torch.tensor(np.stack([data.xquat[bid[n]] for n in names])[None].astype(np.float32))

        class Asset:
            body_names = names

            class data:  # noqa: N801
                body_pos_w = position
                body_quat_w = quat

        box = safety.clearance(Asset, cap=0.10)[0]

        for a, b in active_pairs:
            d_hull = np.inf
            for ca in constituents_of(a):
                for cb in constituents_of(b):
                    for geom_a in ca.mujoco_geoms:
                        for geom_b in cb.mujoco_geoms:
                            d_hull = min(d_hull, mujoco.mj_geomDistance(model, data, geom_id[geom_a], geom_id[geom_b], 0.10, None))
            columns = [
                k
                for k, (i, j) in enumerate(safety.SELF_PAIRS)
                if {safety.ARM_BODY_LIST[i], safety.ARM_BODY_LIST[j]} == {a, b}
            ]
            if not columns:
                columns = [
                    len(safety.SELF_PAIRS) + k
                    for k, (i, p) in enumerate(safety.SENSOR_PAIRS)
                    if {safety.ARM_BODY_LIST[i], safety.PROT_GROUPS[p]} == {a, b}
                ]
            d_box = float(box[columns].min()) if columns else np.nan

            # reference: only worth the exact query near contact
            best = dict(d_ref=np.nan, intersects=False, constituent="", status="not_examined")
            if min(d_hull, d_box if np.isfinite(d_box) else 1.0) <= args.band:
                ra, pa = data.xmat[bid[a]].reshape(3, 3), data.xpos[bid[a]]
                rb, pb = data.xmat[bid[b]].reshape(3, 3), data.xpos[bid[b]]
                best_distance = np.inf
                # both meshes keep the index built once in their link frames; only the relative
                # transform moves between poses
                for name_a, ref_a in REF_A[a]:
                    for name_b, ref_b in REF_B[b]:
                        result = ref_a.closest_to_in_frame(
                            ref_b, ra.T @ rb, ra.T @ (pb - pa), max_clearance=args.band
                        )
                        if result.intersects:
                            best = dict(d_ref=0.0, intersects=True, constituent=f"{name_a}|{name_b}", status=result.status)
                            best_distance = -1.0
                            break
                        if result.clearance < best_distance:
                            best_distance = result.clearance
                            best = dict(
                                d_ref=float(result.clearance),
                                intersects=False,
                                constituent=f"{name_a}|{name_b}",
                                status=result.status,
                            )
                    if best_distance < 0:
                        break

            reference_contact = bool(best["intersects"])
            hull_contact = d_hull <= 0.0
            box_contact = bool(np.isfinite(d_box) and d_box <= args.box_margin)
            if best["status"] == "not_examined":
                # no reference verdict for this pair-step; it is *not* evidence of "clear"
                verdict = "not_examined"
            elif best["status"] == "certified_culled":
                # a bound proves the meshes are beyond the threshold, but it is not a measurement
                verdict = "culled_clear"
            elif reference_contact:
                verdict = "real_contact"
            elif hull_contact and box_contact:
                verdict = "both_phantom"
            elif hull_contact:
                verdict = "hull_only_phantom"
            elif box_contact:
                verdict = "box_only_phantom"
            else:
                verdict = "clear"
            rows.append(
                dict(
                    pose=index,
                    pair=f"{a} {b}",
                    d_ref=best["d_ref"],
                    ref_intersects=reference_contact,
                    d_hull_signed=d_hull,
                    hull_clearance=max(0.0, float(d_hull)),
                    d_box=d_box,
                    box_alarm=box_contact,
                    box_margin=args.box_margin,
                    constituent=best["constituent"],
                    status=best["status"],
                    verdict=verdict,
                )
            )
        if counter % 20 == 0:
            print(f"  pose {counter}/{len(selected)}", flush=True)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / "attribution.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # --- summary ---------------------------------------------------------------------------
    print("\nverdicts over pair-steps (mutually exclusive):")
    categories = ("real_contact", "both_phantom", "hull_only_phantom", "box_only_phantom", "culled_clear", "not_examined", "clear")
    counts = {name: sum(1 for r in rows if r["verdict"] == name) for name in categories}
    for name in categories:
        print(f"  {name:18s} {counts[name]:6d}")
    total = sum(counts.values())
    print(f"  {'sum':18s} {total:6d}   (rows {len(rows)})")
    assert total == len(rows), f"verdict categories do not add up: {total} != {len(rows)}"

    labelled = [r for r in rows if r["status"] == "exact"]
    unlabelled = [r for r in rows if r["status"] != "exact"]
    print(f"\nreference labels: {len(labelled)} valid, {len(unlabelled)} undetermined "
          f"({', '.join(sorted({r['status'] for r in unlabelled})) if unlabelled else 'none'})")
    if labelled:
        print("confusion matrices against the mesh reference (valid labels only):")
        for name, contact_of in (
            ("hull", lambda r: r["d_hull_signed"] <= 0.0),
            ("box", lambda r: np.isfinite(r["d_box"]) and r["d_box"] <= 0.0),
        ):
            tp = sum(1 for r in labelled if contact_of(r) and r["ref_intersects"])
            fp = sum(1 for r in labelled if contact_of(r) and not r["ref_intersects"])
            fn = sum(1 for r in labelled if not contact_of(r) and r["ref_intersects"])
            tn = sum(1 for r in labelled if not contact_of(r) and not r["ref_intersects"])
            assert tp + fp + fn + tn == len(labelled), "confusion matrix does not cover the labelled samples"
            rate = f"{fn / (tp + fn):.1%}" if (tp + fn) else "N/A (no real-contact label)"
            print(f"  {name:5s} TP {tp:4d}  FP {fp:4d}  FN {fn:4d}  TN {tn:4d}   of {len(labelled)}   miss rate {rate}")
    else:
        print("confusion matrices: N/A (no valid reference labels)")

    print("\nper pair:")
    for a, b in active_pairs:
        subset = [r for r in rows if r["pair"] == f"{a} {b}"]
        exact = [r for r in subset if r["status"] == "exact"]
        culled = [r for r in subset if r["status"] == "certified_culled"]
        missing = [r for r in subset if r["status"] == "not_examined"]
        parts = [f"n={len(subset)}", f"exact={len(exact)}", f"culled={len(culled)}", f"undetermined={len(missing)}"]
        if exact:
            reference = np.array([r["d_ref"] for r in exact])
            signed = np.array([r["d_hull_signed"] for r in exact])
            parts.append(f"d_ref median {np.median(reference) * 1000:.1f} mm")
            parts.append(f"d_hull_signed median {np.median(signed) * 1000:+.1f} mm")
        print("  " + "<->".join((a, b)) + ": " + ", ".join(parts))

    print(f"\nwrote {path} and attribution_summary.json")


if __name__ == "__main__":
    main()
