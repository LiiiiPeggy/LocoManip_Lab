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

"""Radar 2x2 attribution: arm box/hull x lidar box/hull, with the original mesh reference.

The four offline representations are:

    A  arm box  + lidar box     (the reward proxy)
    B  arm hull + lidar box
    C  arm box  + lidar hull
    D  arm hull + lidar hull

The arm group keeps its atomic constituents; a composite group is never re-hulled as one object.
The reference is the original mesh/box query. A result at the query cap is a **certified lower
bound**, not an exact distance: it is stored as `lower_bound`/`threshold`, and never contributes to
an exact-distance median or a nearest-point claim.

Contact (0 mm) and reward-margin (20 mm) classifications are separate tables; each lists exact,
certified-culled and unknown counts, and TP/FP/FN/TN over the valid labels only. A sample without
a positive reference label has miss rate N/A.

Run:
    python scripts/ranger_cr10/radar_2x2.py --out outputs/stage413/b/radar_2x2
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from collision_parts import PRIORITY, box_mesh, constituents_of  # noqa: E402
from convex_distance import gjk_distance  # noqa: E402
from reference_geometry import MeshRef  # noqa: E402

ROOT = HERE.parents[1]
SCENE = ROOT / "mujoco/robots/ranger_cr10/scene.xml"
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes"
SAFETY = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/safety.py"
POSES = ROOT / "outputs/stage411/operating_poses.npz"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--poses", default="1534,1539,1583,1562,1533")
ap.add_argument("--extra", default="outputs/stage413/b/radar_extra_poses.json")
ap.add_argument("--pairs", default="cr10_Link1 lidar_link0;cr10_Link2 lidar_link0;cr10_Link3 lidar_link0;cr10_Link5 lidar_link0;cr10_Link6 lidar_link0")
ap.add_argument("--cap", type=float, default=0.05)
ap.add_argument("--box-margin", type=float, default=0.02)
ap.add_argument("--out", default="outputs/stage413/b/radar_2x2")
args = ap.parse_args()


def load_safety():
    spec = importlib.util.spec_from_file_location("safety_2x2", SAFETY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


safety = load_safety()
MODEL = mujoco.MjModel.from_xml_path(str(SCENE))
DATA = mujoco.MjData(MODEL)
BID = {mujoco.mj_id2name(MODEL, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(MODEL.nbody)}
ARM_ADR = [MODEL.jnt_qposadr[mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)]
POSES = np.load(POSES)["poses"]
LIMIT = 0.15

_RAW: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def raw_geometry(part):
    key = f"{part.group}/{part.name}"
    if key not in _RAW:
        if part.kind == "mesh":
            mesh = trimesh.load(str(MESH_DIR / part.stl), force="mesh", process=False)
            vertices = np.asarray(mesh.vertices) * part.scale
            faces = np.asarray(mesh.faces)
        else:
            vertices, faces = box_mesh(part.centre, part.half, part.quat)
        _RAW[key] = (vertices, faces)
    return _RAW[key]


def hull_points(part) -> np.ndarray:
    vertices, faces = raw_geometry(part)
    if part.kind == "box":
        return vertices
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return np.asarray(mesh.convex_hull.vertices)


def reference_ref(part) -> MeshRef:
    vertices, faces = raw_geometry(part)
    return MeshRef(vertices, faces, name=f"{part.group}/{part.name}")


LIDAR_PART = next(part for part in constituents_of("lidar_link0"))
LIDAR_HULL = hull_points(LIDAR_PART)
LIDAR_REF = reference_ref(LIDAR_PART)
ARM_HULL = {group: [(part.name, hull_points(part)) for part in constituents_of(group)] for group in
            {a for a, _ in PRIORITY}}
ARM_REF = {group: [(part.name, reference_ref(part)) for part in constituents_of(group)] for group in
           {a for a, _ in PRIORITY}}
# The arm box is the reward proxy: safety.ARM_PARTS entries, expressed in the group body frame.
ARM_BOX = {group: [dict(name=e[0], vertices=box_mesh(e[1], e[2], e[3])[0]) for e in safety.ARM_PARTS[group]]
           for group in {a for a, _ in PRIORITY}}
# The lidar box is the reward proxy for the sensor, expressed in the base_link frame.
LIDAR_BOX_BODY = safety.PROTECTED_PARTS["lidar_link0"][0]
LIDAR_BOX = [dict(name=e[0], vertices=box_mesh(e[1], e[2], e[3])[0])
             for e in safety.PROTECTED_PARTS["lidar_link0"][1]]
# Real flattened names, used only for reporting; no enumerate-order labels.
ARM_NAME = []
for body, entries in safety.ARM_PARTS.items():
    for entry in entries:
        ARM_NAME.append((body, entry[0]))
PROT_NAME = []
for group, (body, entries) in safety.PROTECTED_PARTS.items():
    for entry in entries:
        PROT_NAME.append((group, entry[0]))


def pose_transforms(q):
    mujoco.mj_resetDataKeyframe(MODEL, DATA, MODEL.key("home").id)
    for address, value in zip(ARM_ADR, q):
        DATA.qpos[address] = float(value)
    mujoco.mj_forward(MODEL, DATA)


def body_transform(name):
    body = BID[name]
    return np.array(DATA.xmat[body]).reshape(3, 3), np.array(DATA.xpos[body])


def transform(points, rotation, translation):
    return np.asarray(points) @ rotation.T + translation


def gjk(points_a, points_b):
    result = gjk_distance(np.asarray(points_a), np.asarray(points_b), tol=1e-8)
    if result.status == "intersecting":
        return 0.0, True, "intersecting"
    if result.status == "separated" and np.isfinite(result.distance):
        return float(result.distance), False, "separated"
    return float("nan"), False, result.status


def safety_asset():
    names = list(dict.fromkeys(safety.ARM_BODY_LIST + safety.PROT_BODY_LIST))
    position = torch.tensor(np.stack([DATA.xpos[BID[name]] for name in names])[None], dtype=torch.float32)
    quat = torch.tensor(np.stack([DATA.xquat[BID[name]] for name in names])[None], dtype=torch.float32)

    class Asset:
        body_names = names

        class data:  # noqa: N801
            body_pos_w = position
            body_quat_w = quat

    return Asset, names


def pair_box_distance(arm, lidar="lidar_link0"):
    """A: the reward box proxy, min over the arm group's atomic boxes and the lidar box."""
    asset, names = safety_asset()
    distances = safety.clearance(asset, cap=LIMIT)[0]
    columns = []
    for k, (i, p) in enumerate(safety.SENSOR_PAIRS):
        if safety.ARM_BODY_LIST[i] == arm and safety.PROT_GROUPS[p] == lidar:
            columns.append((len(safety.SELF_PAIRS) + k, ARM_NAME[i][1], PROT_NAME[p][1]))
    if not columns:
        return float("nan"), False, ""
    best = min(columns, key=lambda item: float(distances[item[0]]))
    value = float(distances[best[0]])
    return value, value <= 0.0, f"{best[1]}|{best[2]}"


def mixed_arm_hull_radar_box(arm):
    """B: atomic arm hull vs the lidar box, in world coordinates."""
    r_arm, p_arm = body_transform(arm)
    r_base, p_base = body_transform(LIDAR_BOX_BODY)
    best = float("inf")
    best_name = ""
    hit = False
    status = "clear"
    for name, points in ARM_HULL[arm]:
        arm_world = transform(points, r_arm, p_arm)
        for box in LIDAR_BOX:
            box_world = transform(box["vertices"], r_base, p_base)
            distance, intersects, state = gjk(arm_world, box_world)
            if intersects:
                return 0.0, True, f"{name}|{box['name']}", "intersecting"
            if np.isfinite(distance) and distance < best:
                best, best_name, status = distance, f"{name}|{box['name']}", state
    return best, hit, best_name, status


def mixed_arm_box_radar_hull(arm):
    """C: atomic arm boxes vs the lidar hull, in world coordinates."""
    r_arm, p_arm = body_transform(arm)
    r_lidar, p_lidar = body_transform("lidar_link0")
    lidar_world = transform(LIDAR_HULL, r_lidar, p_lidar)
    best = float("inf")
    best_name = ""
    status = "clear"
    for box in ARM_BOX[arm]:
        box_world = transform(box["vertices"], r_arm, p_arm)
        distance, intersects, state = gjk(box_world, lidar_world)
        if intersects:
            return 0.0, True, f"{box['name']}|lidar_hull", "intersecting"
        if np.isfinite(distance) and distance < best:
            best, best_name, status = distance, f"{box['name']}|lidar_hull", state
    return best, False, best_name, status


def arm_hull_radar_hull(arm):
    """D: atomic arm hulls vs the lidar hull, in world coordinates."""
    r_arm, p_arm = body_transform(arm)
    r_lidar, p_lidar = body_transform("lidar_link0")
    lidar_world = transform(LIDAR_HULL, r_lidar, p_lidar)
    best = float("inf")
    best_name = ""
    status = "clear"
    for name, points in ARM_HULL[arm]:
        arm_world = transform(points, r_arm, p_arm)
        distance, intersects, state = gjk(arm_world, lidar_world)
        if intersects:
            return 0.0, True, f"{name}|lidar_hull", "intersecting"
        if np.isfinite(distance) and distance < best:
            best, best_name, status = distance, f"{name}|lidar_hull", state
    return best, False, best_name, status


def reference_pair(arm):
    """Original mesh/box reference; a cap result is a certified lower bound, never exact."""
    r_arm, p_arm = body_transform(arm)
    r_lidar, p_lidar = body_transform("lidar_link0")
    rotation = r_arm.T @ r_lidar
    translation = r_arm.T @ (p_lidar - p_arm)
    exacts = []
    lowers = []
    failures = []
    contact = False
    contact_name = ""
    for name, ref in ARM_REF[arm]:
        result = ref.closest_to_in_frame(LIDAR_REF, rotation, translation, max_clearance=args.cap)
        if result.intersects:
            contact = True
            contact_name = f"{name}|lidar"
            break
        if result.status == "exact":
            exacts.append((float(result.clearance), f"{name}|lidar"))
        elif result.status == "certified_culled":
            lowers.append((float(result.clearance), f"{name}|lidar"))
        else:
            failures.append((result.status, f"{name}|lidar"))
    if contact:
        return dict(status="exact", contact=True, distance=0.0, lower_bound=None, threshold=args.cap,
                    nearest=contact_name, exact_count=0, culled_count=0, failed_count=0)
    if exacts:
        distance, name = min(exacts, key=lambda item: item[0])
        return dict(status="exact", contact=False, distance=float(distance), lower_bound=None,
                    threshold=args.cap, nearest=name, exact_count=len(exacts), culled_count=len(lowers),
                    failed_count=len(failures))
    if lowers:
        lower, name = min(lowers, key=lambda item: item[0])
        return dict(status="certified_culled", contact=False, distance=None, lower_bound=float(lower),
                    threshold=args.cap, nearest=name, exact_count=0, culled_count=len(lowers),
                    failed_count=len(failures))
    return dict(status="failed", contact=False, distance=None, lower_bound=None, threshold=args.cap,
                nearest="", exact_count=0, culled_count=0, failed_count=len(failures))


def classify(value, margin):
    if not np.isfinite(value):
        return "unknown"
    # The reward penalty is zero at the margin and positive inside it; the box alarm is d <= margin.
    return "unsafe" if value <= margin else "safe"


def reference_class(row, margin):
    if row["status"] == "exact":
        if row["contact"]:
            return "unsafe"
        return "unsafe" if row["distance"] <= margin else "safe"
    if row["status"] == "certified_culled":
        # lower_bound >= cap (0.05) > both the 0 and 0.02 m thresholds
        return "safe"
    return "unknown"


def main() -> None:
    selected = []
    for value in args.poses.split(","):
        if value.strip():
            selected.append((int(value), None))
    extra = json.loads((ROOT / args.extra).read_text())
    used = {pose for pose, _ in selected}
    for item in extra["extra"]:
        if item["pose"] not in used:
            selected.append((int(item["pose"]), item["pair"]))
            used.add(item["pose"])
    active = [(a, b) for a, b in PRIORITY if f"{a} {b}" in args.pairs.split(";")]
    rows = []
    for pose, only_arm in selected:
        pose_transforms(POSES[pose])
        for arm, lidar in active:
            if only_arm and arm != only_arm:
                continue
            started = time.perf_counter()
            ref = reference_pair(arm)
            a_distance, a_contact, a_name = pair_box_distance(arm, lidar)
            b_distance, b_contact, b_name, b_status = mixed_arm_hull_radar_box(arm)
            c_distance, c_contact, c_name, c_status = mixed_arm_box_radar_hull(arm)
            d_distance, d_contact, d_name, d_status = arm_hull_radar_hull(arm)
            elapsed = time.perf_counter() - started
            row = dict(
                pose=pose, pair=f"{arm} {lidar}", reference_status=ref["status"],
                reference_contact=bool(ref["contact"]),
                reference_distance=ref["distance"], reference_lower_bound=ref["lower_bound"],
                reference_threshold=ref["threshold"], reference_nearest=ref["nearest"],
                reference_exact_count=ref["exact_count"], reference_culled_count=ref["culled_count"],
                reference_failed_count=ref["failed_count"],
                A_distance=a_distance, A_contact=bool(a_contact), A_nearest=a_name,
                B_distance=b_distance, B_contact=bool(b_contact), B_nearest=b_name, B_status=b_status,
                C_distance=c_distance, C_contact=bool(c_contact), C_nearest=c_name, C_status=c_status,
                D_distance=d_distance, D_contact=bool(d_contact), D_nearest=d_name, D_status=d_status,
                seconds=elapsed,
            )
            for label, distance in (("A", a_distance), ("B", b_distance), ("C", c_distance), ("D", d_distance)):
                row[f"{label}_contact_label"] = classify(distance, 0.0)
                row[f"{label}_margin_label"] = classify(distance, args.box_margin)
            row["reference_contact_label"] = reference_class(ref, 0.0)
            row["reference_margin_label"] = reference_class(ref, args.box_margin)
            rows.append(row)
            print(f"pose {pose} {arm}: ref {ref['status']} "
                  f"{ref['distance'] if ref['distance'] is not None else ref['lower_bound']} "
                  f"A {a_distance*1000:.2f} B {b_distance*1000:.2f} C {c_distance*1000:.2f} D {d_distance*1000:.2f} mm",
                  flush=True)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tables = {}
    for margin_name, margin, ref_key in (("contact", 0.0, "reference_contact_label"),
                                         ("margin_20mm", args.box_margin, "reference_margin_label")):
        tables[margin_name] = {}
        for label in ("A", "B", "C", "D"):
            cand_key = f"{label}_{'contact' if margin_name == 'contact' else 'margin'}_label"
            valid = [row for row in rows if row[ref_key] in ("safe", "unsafe") and row[cand_key] in ("safe", "unsafe")]
            unknown = [row for row in rows if row[ref_key] == "unknown" or row[cand_key] == "unknown"]
            tp = sum(1 for row in valid if row[ref_key] == "unsafe" and row[cand_key] == "unsafe")
            fp = sum(1 for row in valid if row[ref_key] == "safe" and row[cand_key] == "unsafe")
            fn = sum(1 for row in valid if row[ref_key] == "unsafe" and row[cand_key] == "safe")
            tn = sum(1 for row in valid if row[ref_key] == "safe" and row[cand_key] == "safe")
            tables[margin_name][label] = dict(TP=tp, FP=fp, FN=fn, TN=tn,
                                              valid=len(valid), unknown=len(unknown),
                                              miss_rate=(fn / (tp + fn)) if (tp + fn) else None,
                                              exact=sum(1 for row in valid if row["reference_status"] == "exact"),
                                              certified_culled=sum(1 for row in valid if row["reference_status"] == "certified_culled"))
    summary = dict(rows=len(rows), poses=selected, pairs=[f"{a} {b}" for a, b in active],
                   cap_m=args.cap, box_margin_m=args.box_margin, tables=tables,
                   note="A=arm box+radar box; B=arm hull+radar box; C=arm box+radar hull; D=arm hull+radar hull; "
                        "cap results are certified lower bounds, not exact distances")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
