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

"""Apply filtered pairs to the non-instanceable flattened simple-collision USD."""

from __future__ import annotations

import json

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from pxr import Usd, UsdPhysics  # noqa: E402

USD = "source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10_simple/ranger_cr10_simple_flat.usd"
OUT = "outputs/stage413/b/simple_collision_validation/flat_filtered_pairs.json"
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

stage = Usd.Stage.Open(USD, load=Usd.Stage.LoadAll)
colliders = {}
for prim in stage.Traverse():
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    path = prim.GetPath()
    parts = path.pathString.split('/')
    # /rangercr10lidar/<body>/collisions/...
    try:
        body = parts[2]
    except IndexError:
        continue
    colliders.setdefault(body, []).append(prim)
print('collider bodies', {k: len(v) for k, v in colliders.items()}, flush=True)
records = []
for a, b in sorted(pairs):
    if not colliders.get(a) or not colliders.get(b):
        continue
    for prim in colliders[a]:
        UsdPhysics.FilteredPairsAPI.Apply(prim)
        rel = prim.GetRelationship("physics:filteredPairs")
        rel.ClearTargets(True)
        for other in colliders[b]:
            rel.AddTarget(other.GetPath())
    for prim in colliders[b]:
        UsdPhysics.FilteredPairsAPI.Apply(prim)
        rel = prim.GetRelationship("physics:filteredPairs")
        rel.ClearTargets(True)
        for other in colliders[a]:
            rel.AddTarget(other.GetPath())
    records.append(dict(a=a, b=b, reason="adjacent/closed assembly overlap at joint"))
stage.GetRootLayer().Save()
with open(OUT, "w") as f:
    json.dump(dict(usd=USD, filters=records, count=len(records),
                   keep_enabled=["arm vs base_link/box_link", "arm vs lidar/d435", "non-adjacent arm"]), f, indent=1)
print('filters', len(records), flush=True)
app.close()
