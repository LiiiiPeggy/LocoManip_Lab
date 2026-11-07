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

"""Apply the simple-collision filtered pairs to the actual flat USD colliders.

The earlier script applied each body pair in a loop and called ``ClearTargets`` for
every pair. A body that participates in several pairs therefore kept only the targets
of the last pair. The chassis/wheel, steering/wheel and most gripper pairs were lost;
the default hold then resolved those deep overlaps with huge contact forces.

This script accumulates all targets per collider first and writes the relationship
once. It also adds the arm-mount pair at collider level: ``cr10_Link1`` against the
merged ``cr10_base_link`` collider only, while arm vs chassis/box_link/lidar stays
enabled. Filters are only for closed adjacent assemblies; they never cover
arm<->box_link or non-adjacent arm pairs.

Run:
    python scripts/ranger_cr10/apply_collision_filters.py \
        --usd <flat.usd> --out-json <filters.json>
"""

from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--usd", required=True)
ap.add_argument("--out-json", required=True)
ap.add_argument(
    "--mode",
    choices=("collider", "body"),
    default="collider",
    help="apply the API to collision prims or to rigid-body prims (PhysX may only honor one form)",
)
args = ap.parse_args()

app = AppLauncher(headless=True).app

from pxr import Usd, UsdPhysics  # noqa: E402

CHASSIS = [
    "base_link",
    "fr_steering_link",
    "fl_steering_wheel_link",
    "rl_steering_wheel_link",
    "rr_steering_wheel_link",
    "fr_wheel_link",
    "fl_wheel_link",
    "rl_wheel_link",
    "rr_wheel_link",
]
ARM = [f"cr10_Link{i}" for i in range(1, 7)]
GRIPPER = [
    "gripper_finger1_inner_knuckle_link",
    "gripper_finger1_finger_tip_link",
    "gripper_finger1_knuckle_link",
    "gripper_finger1_finger_link",
    "gripper_finger2_inner_knuckle_link",
    "gripper_finger2_finger_tip_link",
    "gripper_finger2_knuckle_link",
    "gripper_finger2_finger_link",
]


def body_pairs():
    pairs = set()
    for group in (CHASSIS, GRIPPER):
        for index, a in enumerate(group):
            for b in group[index + 1 :]:
                pairs.add(tuple(sorted((a, b))))
    for index in range(len(ARM) - 1):
        pairs.add(tuple(sorted((ARM[index], ARM[index + 1]))))
    for finger in GRIPPER:
        pairs.add(tuple(sorted((finger, "cr10_Link6"))))
    return pairs


stage = Usd.Stage.Open(args.usd, load=Usd.Stage.LoadAll)
if stage is None:
    raise SystemExit(f"could not open {args.usd}")

colliders_by_body: dict[str, list] = {}
bodies_by_name: dict[str, object] = {}
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        bodies_by_name[prim.GetName()] = prim
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    path = prim.GetPath().pathString.split("/")
    body = path[2]
    colliders_by_body.setdefault(body, []).append(prim)

missing = sorted({body for pair in body_pairs() for body in pair if body not in colliders_by_body})
print("collider bodies", {body: len(prims) for body, prims in sorted(colliders_by_body.items())}, flush=True)
print("missing bodies", missing, flush=True)

# target_map[source_path] = set(target paths)
target_map: dict[str, set[str]] = {}
records = []
if args.mode == "body":
    for a, b in sorted(body_pairs()):
        if a not in bodies_by_name or b not in bodies_by_name:
            continue
        target_map.setdefault(bodies_by_name[a].GetPath().pathString, set()).add(
            bodies_by_name[b].GetPath().pathString
        )
        target_map.setdefault(bodies_by_name[b].GetPath().pathString, set()).add(
            bodies_by_name[a].GetPath().pathString
        )
        records.append({"a": a, "b": b, "kind": "body", "reason": "closed adjacent assembly"})
else:
    for a, b in sorted(body_pairs()):
        if a not in colliders_by_body or b not in colliders_by_body:
            continue
        for source in colliders_by_body[a]:
            targets = target_map.setdefault(source.GetPath().pathString, set())
            targets.update(prim.GetPath().pathString for prim in colliders_by_body[b])
        for source in colliders_by_body[b]:
            targets = target_map.setdefault(source.GetPath().pathString, set())
            targets.update(prim.GetPath().pathString for prim in colliders_by_body[a])
        records.append({"a": a, "b": b, "kind": "body", "reason": "closed adjacent assembly"})

    # Arm mount: link1 rides on cr10_base_link, which is merged into base_link. Filter only
    # that collider; Link1 vs the chassis/box_link/lidar colliders remains enabled.
    mount_targets = [
        prim.GetPath().pathString
        for prim in colliders_by_body.get("base_link", [])
        if "/collisions/cr10_base_link/" in prim.GetPath().pathString
    ]
    if mount_targets and "cr10_Link1" in colliders_by_body:
        for source in colliders_by_body["cr10_Link1"]:
            target_map.setdefault(source.GetPath().pathString, set()).update(mount_targets)
        for target in mount_targets:
            target_map.setdefault(target, set()).update(
                prim.GetPath().pathString for prim in colliders_by_body["cr10_Link1"]
            )
        records.append(
            {
                "a": "cr10_Link1",
                "b": "base_link",
                "kind": "collider",
                "targets": mount_targets,
                "reason": "arm mount joint; only the cr10_base_link collider is filtered",
            }
        )

for source_path, targets in sorted(target_map.items()):
    prim = stage.GetPrimAtPath(source_path)
    UsdPhysics.FilteredPairsAPI.Apply(prim)
    rel = prim.GetRelationship("physics:filteredPairs")
    rel.SetTargets(sorted(targets))

stage.GetRootLayer().Save()

# Verify by reopening: every target must exist; in collider mode it must also be a collider.
check = Usd.Stage.Open(args.usd)
broken = []
filtered_prim_count = 0
target_count = 0
for prim in check.Traverse():
    if args.mode == "collider" and not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    if args.mode == "body" and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        continue
    rel = prim.GetRelationship("physics:filteredPairs")
    if not rel:
        continue
    targets = rel.GetTargets()
    if targets:
        filtered_prim_count += 1
        target_count += len(targets)
    for target in targets:
        target_prim = check.GetPrimAtPath(target)
        required = UsdPhysics.CollisionAPI if args.mode == "collider" else UsdPhysics.RigidBodyAPI
        if not target_prim or not target_prim.HasAPI(required):
            broken.append({"source": prim.GetPath().pathString, "target": str(target)})

result = {
    "usd": os.path.abspath(args.usd),
    "mode": args.mode,
    "body_pairs": len(records),
    "filtered_prims": filtered_prim_count,
    "directed_targets": target_count,
    "broken_targets": broken,
    "records": records,
    "keep_enabled": [
        "arm vs box_link colliders",
        "arm vs chassis/lidar/d435 colliders",
        "non-adjacent arm pairs",
    ],
    "note": "all targets are accumulated per prim before writing; the old last-pair-wins bug is removed",
}
os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
with open(args.out_json, "w") as handle:
    json.dump(result, handle, indent=1)
print(
    json.dumps(
        {
            "usd": result["usd"],
            "body_pairs": result["body_pairs"],
            "mode": result["mode"],
            "filtered_prims": filtered_prim_count,
            "directed_targets": target_count,
            "broken_targets": len(broken),
        },
        indent=1,
    ),
    flush=True,
)
print(f"wrote {args.out_json}", flush=True)
app.close()
