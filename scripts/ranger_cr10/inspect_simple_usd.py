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

"""Inspect the simple-collision candidate USD: bodies, colliders, filters, self-collision.

This reads the same flat USD that ``RANGER-CR10-TELEOP-SIMPLECOLL`` loads. It prints the
actual collider prim paths (after fixed-joint merging), the ``physics:filteredPairs``
relationships and whether every target resolves, and the articulation self-collision
attribute. The output is a JSON file; the terminal print is a short summary.

Run:
    python scripts/ranger_cr10/inspect_simple_usd.py \
        --usd source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10_simple/ranger_cr10_simple_flat.usd \
        --out outputs/stage413/b/simple_collision_v2/usd_inspection.json
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument(
    "--usd",
    default="source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10_simple/ranger_cr10_simple_flat.usd",
)
ap.add_argument("--out", default="outputs/stage413/b/simple_collision_v2/usd_inspection.json")
args = ap.parse_args()

app = AppLauncher(headless=True).app

import json  # noqa: E402
import os  # noqa: E402

from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: E402


def body_of(path: str) -> str:
    """Return the rigid-body name for a collider path.

    The flat candidate keeps the path shape ``/<robot>/<body>/collisions/...``. After
    fixed-joint merging a collider can also be under ``/base_link/...``.
    """
    parts = [p for p in path.split("/") if p]
    for index, part in enumerate(parts):
        prim = stage.GetPrimAtPath("/" + "/".join(parts[: index + 1]))
        if prim and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return part
    return parts[1] if len(parts) > 1 else ""


stage = Usd.Stage.Open(args.usd, load=Usd.Stage.LoadAll)
if stage is None:
    raise SystemExit(f"could not open {args.usd}")

rigid_bodies = {}
rigid_body_filters = []
colliders = []
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid_bodies[prim.GetName()] = prim.GetPath().pathString
        body_record = {
            "path": prim.GetPath().pathString,
            "name": prim.GetName(),
            "filtered_pairs_api": bool(prim.HasAPI(UsdPhysics.FilteredPairsAPI)),
            "targets": [],
            "broken_targets": [],
        }
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            rel = prim.GetRelationship("physics:filteredPairs")
            if rel:
                for target in rel.GetTargets():
                    target_path = str(target)
                    target_prim = stage.GetPrimAtPath(target_path)
                    body_record["targets"].append(target_path)
                    if (not target_prim) or (not target_prim.HasAPI(UsdPhysics.RigidBodyAPI)):
                        body_record["broken_targets"].append(target_path)
        if body_record["targets"]:
            rigid_body_filters.append(body_record)
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    path = prim.GetPath().pathString
    record = {
        "path": path,
        "body": body_of(path),
        "type": prim.GetTypeName(),
        "filtered_pairs_api": bool(prim.HasAPI(UsdPhysics.FilteredPairsAPI)),
        "targets": [],
        "broken_targets": [],
    }
    if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
        rel = prim.GetRelationship("physics:filteredPairs")
        if rel:
            for target in rel.GetTargets():
                target_path = str(target)
                target_prim = stage.GetPrimAtPath(target_path)
                record["targets"].append(target_path)
                if (not target_prim) or (not target_prim.HasAPI(UsdPhysics.CollisionAPI)):
                    record["broken_targets"].append(target_path)
    colliders.append(record)

# Articulation root: find the prim that carries PhysxArticulationAPI.
articulation = None
for prim in stage.Traverse():
    if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
        api = PhysxSchema.PhysxArticulationAPI(prim)
        attr = api.GetEnabledSelfCollisionsAttr()
        articulation = {
            "path": prim.GetPath().pathString,
            "enabled_self_collisions": bool(attr.Get()) if attr and attr.HasValue() else None,
        }
        break

by_body = {}
for record in colliders:
    by_body.setdefault(record["body"], []).append(record)

# Filter pair coverage: map body-pair -> number of directed relationships and broken targets.
pair_counts = {}
broken = []
for record in colliders:
    for target in record["targets"]:
        target_body = body_of(target)
        key = tuple(sorted((record["body"], target_body)))
        pair_counts[key] = pair_counts.get(key, 0) + 1
    for target in record["broken_targets"]:
        broken.append({"source": record["path"], "target": target, "kind": "collider"})
for record in rigid_body_filters:
    for target in record["broken_targets"]:
        broken.append({"source": record["path"], "target": target, "kind": "body"})

body_pair_counts = {}
for record in rigid_body_filters:
    for target in record["targets"]:
        target_name = target.rsplit("/", 1)[-1]
        key = tuple(sorted((record["name"], target_name)))
        body_pair_counts[key] = body_pair_counts.get(key, 0) + 1

result = {
    "usd": os.path.abspath(args.usd),
    "articulation": articulation,
    "rigid_bodies": rigid_bodies,
    "n_rigid_bodies": len(rigid_bodies),
    "n_colliders": len(colliders),
    "colliders": colliders,
    "colliders_by_body": {body: len(records) for body, records in sorted(by_body.items())},
    "collider_filter_pair_counts": {f"{a}|{b}": count for (a, b), count in sorted(pair_counts.items())},
    "rigid_body_filters": rigid_body_filters,
    "n_rigid_bodies_with_filters": len(rigid_body_filters),
    "body_filter_pair_counts": {f"{a}|{b}": count for (a, b), count in sorted(body_pair_counts.items())},
    "broken_targets": broken,
    "note": "collider paths are the actual paths in the flat candidate USD; body-level filters are the "
            "ones PhysX honored here; collider-level targets were measured not to take effect",
}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as handle:
    json.dump(result, handle, indent=1)
print(
    json.dumps(
        {
            "usd": result["usd"],
            "articulation": articulation,
            "n_rigid_bodies": len(rigid_bodies),
            "n_colliders": len(colliders),
            "colliders_by_body": result["colliders_by_body"],
            "n_collider_filter_pairs": len(pair_counts),
            "n_rigid_bodies_with_filters": len(rigid_body_filters),
            "n_body_filter_pairs": len(body_pair_counts),
            "n_broken_targets": len(broken),
        },
        indent=1,
    ),
    flush=True,
)
print(f"wrote {args.out}", flush=True)
app.close()
