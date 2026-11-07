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

"""Repair empty AG95 PhysX mimic referenceJoint relationships in the flat USD.

The URDF declares every AG95 mimic joint as following ``gripper_finger1_joint``,
but after conversion ``gripper_finger1_inner_knuckle_joint`` and
``gripper_finger1_finger_tip_joint`` have an empty
``physxMimicJoint:rotY:referenceJoint`` target. PhysX then has no driven joint for
those two constraints. This script sets the missing targets to the same
``gripper_finger1_joint`` used by the other five mimic joints and verifies every
target resolves to a joint prim. It does not change gearing, offset, mass or drive.

Run:
    python scripts/ranger_cr10/fix_mimic_references.py \
        --usd <flat.usd> --out-json <mimic.json>
"""

from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--usd", required=True)
ap.add_argument("--out-json", required=True)
ap.add_argument("--reference", default="gripper_finger1_joint")
args = ap.parse_args()

app = AppLauncher(headless=True).app

from pxr import Usd, UsdPhysics  # noqa: E402

stage = Usd.Stage.Open(args.usd, load=Usd.Stage.LoadAll)
if stage is None:
    raise SystemExit(f"could not open {args.usd}")
default_prim = stage.GetDefaultPrim()
reference_path = f"{default_prim.GetPath()}/joints/{args.reference}" if default_prim else f"/joints/{args.reference}"
records = []
for prim in stage.Traverse():
    schemas = [schema for schema in prim.GetAppliedSchemas() if "Mimic" in schema]
    if not schemas:
        continue
    for schema in schemas:
        axis = schema.split(":")[-1]
        rel_name = f"physxMimicJoint:{axis}:referenceJoint"
        rel = prim.GetRelationship(rel_name)
        if rel is None:
            continue
        before = [str(target) for target in rel.GetTargets()]
        if not before:
            rel.SetTargets([reference_path])
        after = [str(target) for target in rel.GetTargets()]
        gearing = prim.GetAttribute(f"physxMimicJoint:{axis}:gearing")
        offset = prim.GetAttribute(f"physxMimicJoint:{axis}:offset")
        records.append(
            {
                "joint": prim.GetName(),
                "axis": axis,
                "before": before,
                "after": after,
                "gearing": float(gearing.Get()) if gearing and gearing.HasValue() else None,
                "offset": float(offset.Get()) if offset and offset.HasValue() else None,
            }
        )

stage.GetRootLayer().Save()

check = Usd.Stage.Open(args.usd, load=Usd.Stage.LoadAll)
broken = []
for record in records:
    prim = check.GetPrimAtPath(f"{default_prim.GetPath()}/joints/{record['joint']}")
    for target in record["after"]:
        target_prim = check.GetPrimAtPath(target)
        if not target_prim or not target_prim.IsA(UsdPhysics.Joint):
            broken.append({"joint": record["joint"], "target": target})
result = {
    "usd": os.path.abspath(args.usd),
    "reference": reference_path,
    "mimic_joints": records,
    "repaired": [record["joint"] for record in records if not record["before"]],
    "broken_targets": broken,
    "note": "only empty referenceJoint relationships are filled; gearing/offset/mass/drive are unchanged",
}
os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
with open(args.out_json, "w") as handle:
    json.dump(result, handle, indent=1)
print(
    json.dumps(
        {
            "usd": result["usd"],
            "reference": reference_path,
            "repaired": result["repaired"],
            "n_mimic": len(records),
            "broken_targets": broken,
        },
        indent=1,
    ),
    flush=True,
)
print(f"wrote {args.out_json}", flush=True)
app.close()
