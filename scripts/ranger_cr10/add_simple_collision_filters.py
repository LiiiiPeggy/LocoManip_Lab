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

"""Add PhysX filtered-pair relationships to the independent simple-collision USD.

Only the candidate physics layer is edited. Filters have explicit reasons:
* chassis/steering/wheel bodies overlap at their joints by design;
* adjacent arm links overlap at joints by design;
* the AG95 finger linkage is a closed assembly.

Arm-vs-base (including box_link), arm-vs-lidar/d435 and non-adjacent arm pairs stay enabled.
No pair involving box_link and a moving arm link is filtered.
"""

from __future__ import annotations

import json

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from pxr import Usd, UsdPhysics, Sdf  # noqa: E402

USD = "source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10_simple/configuration/ranger_cr10_simple_physics.usd"
OUT = "outputs/stage413/b/simple_collision_validation/filtered_pairs.json"

CHASSIS = ["base_link", "fr_steering_link", "fl_steering_wheel_link", "rl_steering_wheel_link", "rr_steering_wheel_link",
           "fr_wheel_link", "fl_wheel_link", "rl_wheel_link", "rr_wheel_link"]
ARM = [f"cr10_Link{i}" for i in range(1, 7)]
GRIPPER = ["gripper_finger1_inner_knuckle_link", "gripper_finger1_finger_tip_link", "gripper_finger1_knuckle_link",
           "gripper_finger1_finger_link", "gripper_finger2_inner_knuckle_link", "gripper_finger2_finger_tip_link",
           "gripper_finger2_knuckle_link", "gripper_finger2_finger_link"]
pairs = set()
for i, a in enumerate(CHASSIS):
    for b in CHASSIS[i + 1:]:
        pairs.add(tuple(sorted((a, b))))
for i in range(len(ARM) - 1):
    pairs.add(tuple(sorted((ARM[i], ARM[i + 1]))))
for i, a in enumerate(GRIPPER):
    for b in GRIPPER[i + 1:]:
        pairs.add(tuple(sorted((a, b))))
for a in GRIPPER:
    pairs.add(tuple(sorted((a, "cr10_Link6"))))


def colliders_of(stage, body):
    root = stage.GetPrimAtPath(f"/colliders/{body}")
    out = []
    if root:
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                out.append(prim)
    return out


stage = Usd.Stage.Open(USD, load=Usd.Stage.LoadAll)
mapping = {body: colliders_of(stage, body) for body in CHASSIS + ARM + GRIPPER}
records = []
for a, b in sorted(pairs):
    if not mapping.get(a) or not mapping.get(b):
        continue
    def relative(source, target):
        sp, tp = source.GetPath().pathString.split('/')[1:], target.GetPath().pathString.split('/')[1:]
        common = 0
        while common < min(len(sp), len(tp)) and sp[common] == tp[common]:
            common += 1
        return Sdf.Path("../" * (len(sp) - common) + "/".join(tp[common:]))

    for prim in mapping[a]:
        UsdPhysics.FilteredPairsAPI.Apply(prim)
        rel = prim.GetRelationship("physics:filteredPairs")
        rel.ClearTargets(True)
        for other in mapping[b]:
            rel.AddTarget(relative(prim, other))
    for prim in mapping[b]:
        UsdPhysics.FilteredPairsAPI.Apply(prim)
        rel = prim.GetRelationship("physics:filteredPairs")
        rel.ClearTargets(True)
        for other in mapping[a]:
            rel.AddTarget(relative(prim, other))
    records.append(dict(a=a, b=b, reason="adjacent/closed assembly overlap at joint"))
stage.GetRootLayer().Save()
with open(OUT, "w") as f:
    json.dump(dict(usd=USD, filters=records, count=len(records),
                   keep_enabled=["arm link vs base_link/box_link", "arm link vs lidar/d435",
                                 "non-adjacent arm links"]), f, indent=1)
print(json.dumps(dict(count=len(records), pairs=[f"{a} <-> {b}" for a, b in sorted(pairs)]), indent=1))
app.close()
