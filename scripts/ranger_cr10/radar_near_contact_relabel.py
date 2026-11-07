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

"""Relabel the frozen near set after the containment-transform fix.

The near set's joint configurations and selection rule are unchanged. The reference labels are
recomputed because ``MeshRef.closest_to_in_frame`` tested the guest's first vertex in its own
frame, not the host's frame. That produced false ``containment:a_inside_b`` contacts (for
example the Link3 rows 0.7-12.5 mm from the lidar mesh). The old file is kept and marked
invalid; this script writes a corrected reference-label file.

The q set was frozen before any C/D evaluation. This relabel is a reference-query correction,
not a candidate selection change; the C/D evaluation is rerun against the corrected labels.

Run:
    python scripts/ranger_cr10/radar_near_contact_relabel.py
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
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
OLD = ROOT / "outputs/stage413/b/radar_near_contact/near_set.json"
OUT = ROOT / "outputs/stage413/b/radar_near_contact/v2/near_set_corrected.json"
ARMS = ("cr10_Link1", "cr10_Link2", "cr10_Link3", "cr10_Link4", "cr10_Link5", "cr10_Link6")
LIDAR = "lidar_link0"
CONTACT_TOL = 1e-4

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--old", default=str(OLD))
ap.add_argument("--out", default=str(OUT))
ap.add_argument("--workers", type=int, default=10)
args = ap.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def reference_group(arm, cap):
    r_arm, p_arm = body_transform(arm)
    r_lidar, p_lidar = body_transform(LIDAR)
    rotation = r_arm.T @ r_lidar
    translation = r_arm.T @ (p_lidar - p_arm)
    exacts, lowers, failures = [], [], []
    contact, contact_name, details = False, "", []
    for name, ref in ARM_REF[arm]:
        result = ref.closest_to_in_frame(LIDAR_REF, rotation, translation, max_clearance=cap)
        if result.intersects:
            contact = True
            contact_name = f"{name}|lidar"
            details.append(f"{name}:intersecting:{result.reason}")
        elif result.status == "exact":
            exacts.append((float(result.clearance), f"{name}|lidar"))
            details.append(f"{name}:exact:{float(result.clearance):.9f}")
        elif result.status == "certified_culled":
            lower = float(result.extra.get("lower_bound", result.clearance))
            lowers.append((lower, f"{name}|lidar"))
            details.append(f"{name}:certified_culled:{lower:.9f}")
        else:
            failures.append((result.status, f"{name}|lidar"))
            details.append(f"{name}:{result.status}")
    detail = ";".join(details)
    if contact:
        return dict(status="exact", contact=True, distance=0.0, lower_bound=None, nearest=contact_name,
                    exact_count=0, culled_count=len(lowers), failed_count=len(failures), detail=detail)
    if exacts and not failures:
        distance, name = min(exacts, key=lambda item: item[0])
        return dict(status="exact", contact=False, distance=float(distance), lower_bound=None, nearest=name,
                    exact_count=len(exacts), culled_count=len(lowers), failed_count=0, detail=detail)
    if exacts and failures:
        distance, name = min(exacts, key=lambda item: item[0])
        return dict(status="failed", contact=False, distance=None, lower_bound=None, upper_bound=float(distance),
                    nearest=name, exact_count=len(exacts), culled_count=len(lowers), failed_count=len(failures),
                    detail=detail)
    if lowers and not failures:
        lower, name = min(lowers, key=lambda item: item[0])
        return dict(status="certified_culled", contact=False, distance=None, lower_bound=float(lower),
                    nearest=name, exact_count=0, culled_count=len(lowers), failed_count=0, detail=detail)
    return dict(status="failed", contact=False, distance=None, lower_bound=None, nearest="",
                exact_count=len(exacts), culled_count=len(lowers), failed_count=max(1, len(failures)), detail=detail)


def bin_of(result, cap):
    if result["contact"]:
        return "overlap"
    if result["status"] == "exact":
        d = result["distance"]
        for label, (lo, hi) in (("0-1mm", (0.0, 0.001)), ("1-3mm", (0.001, 0.003)),
                                ("3-5mm", (0.003, 0.005)), ("5-10mm", (0.005, 0.010)),
                                ("10-20mm", (0.010, 0.020)), ("20-30mm", (0.020, 0.030))):
            if lo <= d < hi:
                return label
        return ">=30mm_exact"
    if result["status"] == "certified_culled":
        return ">=30mm_certified_culled"
    return "failed"


def relabel(row):
    set_q(row["q"])
    cap = float(row["reference_threshold_m"])
    result = reference_group(row["arm"], cap)
    out = dict(row)
    out.update(reference_status=result["status"], reference_contact=bool(result["contact"]),
               reference_distance_m=result["distance"], reference_lower_bound_m=result["lower_bound"],
               reference_threshold_m=cap, reference_nearest=result["nearest"],
               reference_detail=result["detail"], bin=bin_of(result, cap))
    if result["contact"]:
        out["overlap_class"] = out.get("overlap_class") or "reference_contact"
    else:
        out["overlap_class"] = None
    return out


def main() -> None:
    old = json.loads(Path(args.old).read_text())
    rows = old["rows"]
    with mp.get_context("fork").Pool(processes=args.workers) as pool:
        corrected = pool.map(relabel, rows)
    coverage = {}
    for arm in ARMS:
        subset = [row for row in corrected if row["arm"] == arm]
        bins = {}
        for label in ("0-1mm", "1-3mm", "3-5mm", "5-10mm", "10-20mm", "20-30mm",
                      ">=30mm_exact", ">=30mm_certified_culled", "overlap", "failed"):
            bins[label] = sum(1 for row in subset if row["bin"] == label)
        coverage[arm] = dict(n=len(subset), bins=bins,
                             overlap=sum(1 for row in subset if row["reference_contact"]),
                             exact=sum(1 for row in subset if row["reference_status"] == "exact"),
                             certified_culled=sum(1 for row in subset if row["reference_status"] == "certified_culled"),
                             failed=sum(1 for row in subset if row["reference_status"] == "failed"),
                             sample_types={t: sum(1 for row in subset if row["sample_type"] == t)
                                           for t in sorted({row["sample_type"] for row in subset})})
    record = dict(old,
                  reference_version=2,
                  supersedes=args.old,
                  reference_bug="closest_to_in_frame._containment used untransformed guest vertices",
                  reference_fix="transform guest vertex into the host frame before the winding-number test",
                  q_set_frozen_before_cd=True,
                  reference_relabeled_after_cd=True,
                  reference_cap_m=old["reference_cap_m"],
                  contact_tolerance_m=CONTACT_TOL,
                  rule_sha256=old["rule_sha256"],
                  relabel_script_sha256=sha256(Path(__file__)),
                  rows=corrected,
                  coverage=coverage,
                  notes=old.get("notes", []) + [
                      "v1 reference labels are invalid where a false containment contact was produced.",
                      "The q set and freeze rule are unchanged; only the reference label query was corrected.",
                  ])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=1))
    print(json.dumps(coverage, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
