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

"""Recheck the 10,800 radar reference labels after the containment-transform fix.

The replay itself ran before the fix; it had zero reference contacts, so the containment path
never returned a contact. The bug also had a false-negative direction (a true containment could be
missed), so the reference labels are re-evaluated for every pose/pair and compared with the stored
rows. No A/C/D candidate is recomputed.

Run:
    python scripts/ranger_cr10/radar_replay_reference_check.py
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from collision_parts import box_mesh, constituents_of  # noqa: E402
from reference_geometry import MeshRef  # noqa: E402

ROOT = HERE.parents[1]
SCENE = ROOT / "mujoco/robots/ranger_cr10/scene.xml"
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes"
POSES = ROOT / "outputs/stage411/operating_poses.npz"
OLD_ROWS = ROOT / "outputs/stage413/b/radar_replay/rows.csv"
OUT = ROOT / "outputs/stage413/b/radar_replay/reference_check_after_containment_fix.json"
ARMS = ("cr10_Link1", "cr10_Link2", "cr10_Link3", "cr10_Link4", "cr10_Link5", "cr10_Link6")
LIDAR = "lidar_link0"
CAP = 0.05

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--workers", type=int, default=10)
ap.add_argument("--out", default=str(OUT))
args = ap.parse_args()

MODEL = mujoco.MjModel.from_xml_path(str(SCENE))
DATA = mujoco.MjData(MODEL)
BID = {mujoco.mj_id2name(MODEL, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(MODEL.nbody)}
ARM_ADR = [MODEL.jnt_qposadr[mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)]
POSE_ARRAY = np.load(POSES)["poses"]


def _load(part):
    if part.kind == "mesh":
        mesh = trimesh.load(str(MESH_DIR / part.stl), force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64) * part.scale
        faces = np.asarray(mesh.faces, dtype=np.int64)
    else:
        vertices, faces = box_mesh(part.centre, part.half, part.quat)
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
    return vertices, faces


ARM_REF = {}
for arm in ARMS:
    ARM_REF[arm] = []
    for part in constituents_of(arm):
        vertices, faces = _load(part)
        ARM_REF[arm].append((part.name, MeshRef(vertices, faces, name=f"{arm}/{part.name}")))
_vertices, _faces = _load(constituents_of(LIDAR)[0])
LIDAR_REF = MeshRef(_vertices, _faces, name=LIDAR)


def set_q(q):
    mujoco.mj_resetDataKeyframe(MODEL, DATA, MODEL.key("home").id)
    for address, value in zip(ARM_ADR, q):
        DATA.qpos[address] = float(value)
    mujoco.mj_forward(MODEL, DATA)


def body_transform(name):
    body = BID[name]
    return np.array(DATA.xmat[body], dtype=np.float64).reshape(3, 3), np.array(DATA.xpos[body], dtype=np.float64)


def reference_group(arm):
    r_arm, p_arm = body_transform(arm)
    r_lidar, p_lidar = body_transform(LIDAR)
    rotation = r_arm.T @ r_lidar
    translation = r_arm.T @ (p_lidar - p_arm)
    exacts, lowers, failures = [], [], []
    contact, contact_name = False, ""
    for name, ref in ARM_REF[arm]:
        result = ref.closest_to_in_frame(LIDAR_REF, rotation, translation, max_clearance=CAP)
        if result.intersects:
            contact = True
            contact_name = f"{name}|lidar"
        elif result.status == "exact":
            exacts.append((float(result.clearance), f"{name}|lidar"))
        elif result.status == "certified_culled":
            lowers.append((float(result.extra.get("lower_bound", result.clearance)), f"{name}|lidar"))
        else:
            failures.append((result.status, f"{name}|lidar"))
    if contact:
        return dict(status="exact", contact=True, distance=0.0, lower_bound=None, nearest=contact_name)
    if exacts and not failures:
        distance, name = min(exacts, key=lambda item: item[0])
        return dict(status="exact", contact=False, distance=float(distance), lower_bound=None, nearest=name)
    if not exacts and not failures and lowers:
        lower, name = min(lowers, key=lambda item: item[0])
        return dict(status="certified_culled", contact=False, distance=None, lower_bound=float(lower), nearest=name)
    return dict(status="failed", contact=False, distance=None, lower_bound=None, nearest="")


def check_pose(pose):
    set_q(POSE_ARRAY[pose])
    out = []
    for arm in ARMS:
        result = reference_group(arm)
        out.append(dict(pose=pose, arm=arm, **result))
    return out


def main() -> None:
    old = {}
    for row in csv.DictReader(open(OLD_ROWS)):
        old[(int(row["pose"]), row["arm"])] = row
    with mp.get_context("fork").Pool(processes=args.workers) as pool:
        grouped = pool.map(check_pose, range(len(POSE_ARRAY)))
    new_rows = [row for group in grouped for row in group]
    changes = []
    for row in new_rows:
        prior = old[(row["pose"], row["arm"])]
        prior_status = prior["reference_status"]
        prior_contact = prior["reference_contact"] == "True"
        prior_distance = None if prior["reference_distance_m"] in ("", "None") else float(prior["reference_distance_m"])
        prior_lower = None if prior["reference_lower_bound_m"] in ("", "None") else float(prior["reference_lower_bound_m"])
        changed = (row["status"] != prior_status or bool(row["contact"]) != prior_contact
                   or (row["distance"] is None) != (prior_distance is None)
                   or (row["distance"] is not None and abs(row["distance"] - prior_distance) > 1e-12)
                   or (row["lower_bound"] is None) != (prior_lower is None)
                   or (row["lower_bound"] is not None and abs(row["lower_bound"] - prior_lower) > 1e-12))
        if changed:
            changes.append(dict(pose=row["pose"], arm=row["arm"],
                                old=dict(status=prior_status, contact=prior_contact,
                                         distance_m=prior_distance, lower_bound_m=prior_lower),
                                new=dict(status=row["status"], contact=bool(row["contact"]),
                                         distance_m=row["distance"], lower_bound_m=row["lower_bound"])))
    status_counts = {}
    for row in new_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    result = dict(
        rows=len(new_rows), changed=len(changes), changes=changes,
        status_counts=status_counts,
        contacts=sum(1 for row in new_rows if row["contact"]),
        reference_geometry_sha256=hashlib.sha256(Path(HERE / "reference_geometry.py").read_bytes()).hexdigest(),
        note="reference labels only; no A/C/D candidate was recomputed. The replay's stored labels are "
             "unchanged if `changed` is 0.",
    )
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({key: result[key] for key in ("rows", "changed", "status_counts", "contacts")}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
