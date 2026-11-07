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

"""Stage 4.13b: locate where C/D lose a separated near-contact gap.

This is a diagnostic, not a fourth baseline. For representative false-contact rows from the
frozen near set it asks two hybrid questions:

* original arm mesh vs lidar convex hull: if this is contact while the mesh-mesh reference is
  separated, the lidar hull alone closes the gap;
* arm convex hull vs original lidar mesh: if this is contact, the arm hull alone closes it.

The original reference witness points are stored in the arm and lidar frames. The path parameter
is never called penetration depth. A small number of rows is selected by pair and reference bin;
this is not a global scan.

Run:
    python scripts/ranger_cr10/radar_near_failure_attribution.py
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import csv
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
import reference_geometry  # noqa: E402
from reference_geometry import MeshRef  # noqa: E402

ROOT = HERE.parents[1]
SCENE = ROOT / "mujoco/robots/ranger_cr10/scene.xml"
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes"
NEAR = ROOT / "outputs/stage413/b/radar_near_contact/near_set.json"
ROWS = ROOT / "outputs/stage413/b/radar_near_contact/rows.csv"
OUT = ROOT / "outputs/stage413/b/radar_near_contact/failure_attribution.json"
ARMS = ("cr10_Link2", "cr10_Link3")
LIDAR = "lidar_link0"
CONTACT_TOL = 1e-4
CAP = 0.03

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--near", default=str(NEAR))
ap.add_argument("--rows", default=str(ROWS))
ap.add_argument("--out", default=str(OUT))
ap.add_argument("--workers", type=int, default=10)
ap.add_argument("--max-rows", type=int, default=40)
ap.add_argument("--buggy-containment", action="store_true",
                help="reproduce the v1 attribution with the untransformed containment; diagnostic history only")
args = ap.parse_args()

MODEL = mujoco.MjModel.from_xml_path(str(SCENE))
DATA = mujoco.MjData(MODEL)
BID = {mujoco.mj_id2name(MODEL, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(MODEL.nbody)}
ARM_ADR = [MODEL.jnt_qposadr[mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)]


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
ARM_HULL_REF = {}
for arm in ARMS:
    ARM_REF[arm] = []
    ARM_HULL_REF[arm] = []
    for part in constituents_of(arm):
        vertices, faces = _load(part)
        ARM_REF[arm].append((part.name, MeshRef(vertices, faces, name=f"{arm}/{part.name}")))
        hull = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).convex_hull
        ARM_HULL_REF[arm].append((part.name, MeshRef(np.asarray(hull.vertices), np.asarray(hull.faces),
                                                     name=f"{arm}/{part.name}/hull")))
_vertices, _faces = _load(constituents_of(LIDAR)[0])
LIDAR_REF = MeshRef(_vertices, _faces, name=LIDAR)
_lidar_hull = trimesh.Trimesh(vertices=_vertices, faces=_faces, process=False).convex_hull
LIDAR_HULL_REF = MeshRef(np.asarray(_lidar_hull.vertices), np.asarray(_lidar_hull.faces), name=f"{LIDAR}/hull")


def set_q(q):
    mujoco.mj_resetDataKeyframe(MODEL, DATA, MODEL.key("home").id)
    for address, value in zip(ARM_ADR, q):
        DATA.qpos[address] = float(value)
    mujoco.mj_forward(MODEL, DATA)


def body_transform(name):
    body = BID[name]
    return np.array(DATA.xmat[body], dtype=np.float64).reshape(3, 3), np.array(DATA.xpos[body], dtype=np.float64)


def query(ref_a, ref_b, arm):
    r_arm, p_arm = body_transform(arm)
    r_lidar, p_lidar = body_transform(LIDAR)
    rotation = r_arm.T @ r_lidar
    translation = r_arm.T @ (p_lidar - p_arm)
    result = ref_a.closest_to_in_frame(ref_b, rotation, translation, max_clearance=CAP)
    return dict(status=result.status, intersects=bool(result.intersects), clearance=float(result.clearance),
                witness_a=None if result.witness_a is None else np.asarray(result.witness_a).tolist(),
                witness_b=None if result.witness_b is None else np.asarray(result.witness_b).tolist(),
                triangle_a=int(result.triangle_a), triangle_b=int(result.triangle_b), reason=result.reason)


def group_query(refs_a, ref_b, arm):
    best = None
    for name, ref in refs_a:
        result = query(ref, ref_b, arm)
        result["name"] = name
        if result["intersects"]:
            return result
        if best is None or (result["status"] == "exact" and
                            (best["status"] != "exact" or result["clearance"] < best["clearance"])):
            best = result
    return best


def attribute(task):
    row, frozen = task
    set_q(frozen["q"])
    original = group_query(ARM_REF[row["arm"]], LIDAR_REF, row["arm"])
    mesh_vs_lidar_hull = group_query(ARM_REF[row["arm"]], LIDAR_HULL_REF, row["arm"])
    hull_vs_lidar_mesh = group_query(ARM_HULL_REF[row["arm"]], LIDAR_REF, row["arm"])
    if original["witness_a"] is not None:
        arm_witness = np.asarray(original["witness_a"])
        r_lidar, p_lidar = body_transform(LIDAR)
        lidar_witness = np.asarray(original["witness_b"])
        # witness_b is in the arm frame; move it to the lidar body frame
        r_arm, p_arm = body_transform(row["arm"])
        lidar_in_world = r_arm @ lidar_witness + p_arm
        lidar_frame = r_lidar.T @ (lidar_in_world - p_lidar)
    else:
        arm_witness = None
        lidar_frame = None
    if arm_witness is not None:
        arm_witness = np.asarray(arm_witness).tolist()
    if lidar_frame is not None:
        lidar_frame = np.asarray(lidar_frame).tolist()
    references = float(row["reference_distance_m"]) if row["reference_distance_m"] not in ("", "None") else None
    d_value = float(row["D_distance_m"]) if row["D_distance_m"] not in ("", "None") else None
    c_value = float(row["C_distance_m"]) if row["C_distance_m"] not in ("", "None") else None
    if references is None:
        side = "reference_not_exact"
    elif mesh_vs_lidar_hull["intersects"] and hull_vs_lidar_mesh["intersects"]:
        side = "both_hulls"
    elif mesh_vs_lidar_hull["intersects"]:
        side = "lidar_hull_only"
    elif hull_vs_lidar_mesh["intersects"]:
        side = "arm_hull_only"
    elif row["D_contact"] == "True":
        side = "joint_hull_overfill"
    elif row["C_contact"] == "True":
        side = "arm_box_only"
    else:
        side = "not_reproduced"
    return dict(
        row_id=row["row_id"], q=frozen["q"], pair=row["pair"], arm=row["arm"], bin=row["bin"],
        sample_type=row["sample_type"], path_id=row["path_id"], s=row["s"], delta_s=row["delta_s"],
        reference=dict(distance_m=references, status=row["reference_status"],
                       intersects=row["reference_contact"] == "True", nearest=row["reference_nearest"],
                       witness_arm=arm_witness, witness_lidar=lidar_frame,
                       triangle_a=original["triangle_a"], triangle_b=original["triangle_b"]),
        C=dict(distance_m=c_value, contact=row["C_contact"] == "True", nearest=row["C_nearest"],
               distance_lost_m=None if c_value is None or references is None else references - c_value),
        D=dict(distance_m=d_value, contact=row["D_contact"] == "True", nearest=row["D_nearest"],
               distance_lost_m=None if d_value is None or references is None else references - d_value),
        diagnostic=dict(arm_mesh_vs_lidar_hull=mesh_vs_lidar_hull,
                        arm_hull_vs_lidar_mesh=hull_vs_lidar_mesh),
        side=side,
    )


def main() -> None:
    near = json.loads(Path(args.near).read_text())
    by_id = {row["row_id"]: row for row in near["rows"]}
    rows = list(csv.DictReader(open(args.rows)))
    selected = []
    for arm in ARMS:
        for bin_name in ("0-1mm", "1-3mm", "3-5mm", "5-10mm", "10-20mm", "overlap"):
            group = [row for row in rows if row["arm"] == arm and row["bin"] == bin_name
                     and row["reference_status"] == "exact"
                     and (row["C_contact"] == "True" or row["D_contact"] == "True")]
            group.sort(key=lambda row: float(row["reference_distance_m"] or 0.0))
            selected.extend(group[:2])
    selected = selected[: args.max_rows]
    if args.buggy_containment:
        # Historical v1 reproduction only: the guest vertex is tested in its own frame.
        def old_containment(self, other, rotation=None, translation=None):
            for host, guest, label in ((self, other, "b_inside_a"), (other, self, "a_inside_b")):
                outcome = host.contains(guest.vertices[0])
                if outcome is True:
                    return reference_geometry.QueryResult(reference_geometry.STATUS_OK, 0.0, True,
                                                          reason=f"containment:{label}")
            return None

        reference_geometry.MeshRef._containment = old_containment
    tasks = [(row, by_id[row["row_id"]]) for row in selected]
    records = []
    with mp.get_context("fork").Pool(processes=args.workers) as pool:
        for record in pool.imap_unordered(attribute, tasks):
            records.append(record)
    records.sort(key=lambda record: record["row_id"])
    summary = {}
    for record in records:
        summary[record["side"]] = summary.get(record["side"], 0) + 1
    output = dict(
        note="diagnostic side attribution for representative near-contact false contacts; not a global scan, "
             "not a fourth baseline",
        selected=len(records), side_summary=summary,
        rows=records,
    )
    Path(args.out).write_text(json.dumps(output, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
