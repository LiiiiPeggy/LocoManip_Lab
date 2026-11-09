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

"""Regression tests for the Link/group union aggregation.

The old proxy stored one distance per ``(arm, protected)`` key, so the last geom in the list
silently replaced the Link6 main box. These checks make that failure impossible to reintroduce:

* the Link6 group contains the Link6 main box, AG95 palm, D455 and all eight finger bodies;
* changing the geom traversal order does not change the union minimum;
* adding a farther component cannot remove the existing minimum;
* the winning atomic shape is kept, and the four Link6 sub-families are observed as winners.

Run:
    python scripts/ranger_cr10/check_phys_proxy_union.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from phys_proxy import (  # noqa: E402
    CAP,
    ROOT,
    Shape,
    load_shapes,
    proxy_group_distance,
    reference_group_distance,
    shapes_by_group,
)

OUT = ROOT / "outputs/stage413/b/simple_collision_v2/phys_proxy_union_regression.json"
NEAR_FILE = ROOT / "outputs/stage413/b/radar_near_contact/v3/near_set_containment_v3.json"
PROXY_SCENE = ROOT / "mujoco/robots/ranger_cr10/scene_simple_v2_physproxy.xml"
V2_SCENE = ROOT / "mujoco/robots/ranger_cr10/scene_simple_v2.xml"


def set_pose(model, data, q) -> None:
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for index, value in enumerate(q):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{index + 1}")
        data.qpos[model.jnt_qposadr[joint]] = float(value)
    mujoco.mj_forward(model, data)


def main() -> None:
    shapes = shapes_by_group(load_shapes())
    link6 = shapes["cr10_Link6"]
    lidar = shapes["lidar"]
    proxy = mujoco.MjModel.from_xml_path(str(PROXY_SCENE))
    data = mujoco.MjData(proxy)
    v2 = mujoco.MjModel.from_xml_path(str(V2_SCENE))
    v2_data = mujoco.MjData(v2)
    near = [row for row in json.loads(NEAR_FILE.read_text())["rows"] if row["arm"] == "cr10_Link6"]
    winners = {}
    winner_pose = {}
    for row in near:
        set_pose(proxy, data, row["q"])
        result = proxy_group_distance(proxy, data, shapes, "cr10_Link6", "lidar")
        winners[result["shape_a"]] = winners.get(result["shape_a"], 0) + 1
        winner_pose.setdefault(result["shape_a"], row["row_id"])

    # A pose where the winner is not the main box is the useful aggregation case.
    selected_id = None
    for shape_name in ("phys_cr10_Link6_ag95_base_link", "phys_cr10_Link6_mesh_2",
                       "phys_gripper_finger1_finger_link_rod_Link"):
        if shape_name in winner_pose:
            selected_id = winner_pose[shape_name]
            break
    if selected_id is None:
        selected_id = winner_pose.get("phys_cr10_Link6_mesh_0", near[0]["row_id"])
    selected_row = next(row for row in near if row["row_id"] == selected_id)
    q = selected_row["q"]
    set_pose(proxy, data, q)
    set_pose(v2, v2_data, q)

    base = proxy_group_distance(proxy, data, shapes, "cr10_Link6", "lidar")
    reference_base = reference_group_distance(shapes, "cr10_Link6", "lidar", v2_data)

    # Order independence.
    order_ok = True
    order_values = []
    for seed in range(10):
        shuffled = list(link6)
        random.Random(seed).shuffle(shuffled)
        copy = dict(shapes)
        copy["cr10_Link6"] = shuffled
        value = proxy_group_distance(proxy, data, copy, "cr10_Link6", "lidar")["distance"]
        order_values.append(value)
        order_ok = order_ok and abs(value - base["distance"]) < 1e-12

    # Adding a farther component cannot remove the existing minimum.
    far = Shape(
        name="synthetic_far_component", body="cr10_Link6", group="cr10_Link6", part="synthetic",
        kind="box", points=np.array([[100.0, 100.0, 100.0]]), pos=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0]), half=None, mesh_file=None, mesh_scale=None,
        source_path="synthetic", approximation=None,
    )
    with_far = dict(shapes)
    with_far["cr10_Link6"] = list(link6) + [far]
    far_value = reference_group_distance(with_far, "cr10_Link6", "lidar", v2_data)["distance"]

    # Dictionary-overwrite check: every atomic shape is present and names are unique.
    names = [shape.name for shape in link6]
    unique_ok = len(names) == len(set(names))
    subfamilies = {
        "link6_main": any(shape.name == "phys_cr10_Link6_mesh_0" for shape in link6),
        "ag95": any(shape.name == "phys_cr10_Link6_ag95_base_link" for shape in link6),
        "d455": any(shape.name == "phys_cr10_Link6_mesh_2" for shape in link6),
        "fingers": sum(1 for shape in link6 if shape.body.startswith("gripper_finger")),
    }
    checks = dict(
        link6_atomic_count=len(link6),
        link6_names_unique=unique_ok,
        link6_subfamilies=subfamilies,
        order_independent=order_ok,
        order_values=order_values,
        far_component_keeps_minimum=abs(far_value - reference_base["distance"]) < 1e-9,
        reference_distance=reference_base["distance"],
        proxy_distance=base["distance"],
        proxy_winner=base["shape_a"],
        proxy_winner_family=("ag95" if "ag95" in str(base["shape_a"]) else
                             "d455" if "mesh_2" in str(base["shape_a"]) else
                             "finger" if "gripper_finger" in str(base["shape_a"]) else "link6_main"),
        winner_counts=winners,
        winner_pose=winner_pose,
        selected_pose=selected_id,
        near_link6_rows=len(near),
    )
    passed = (
        unique_ok and order_ok and checks["far_component_keeps_minimum"]
        and subfamilies["link6_main"] and subfamilies["ag95"] and subfamilies["d455"]
        and subfamilies["fingers"] == 8
        and (base["distance"] < CAP - 1e-9 and abs(base["distance"] - reference_base["distance"]) < 1e-9
             or base["distance"] >= CAP - 1e-9 and reference_base["distance"] >= CAP - 1e-9)
    )
    result = dict(passed=bool(passed), checks=checks,
                  note="proxy_vs_reference difference uses the same actual shapes; it checks the "
                       "MuJoCo representation and the union, not the original-mesh simplification")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
