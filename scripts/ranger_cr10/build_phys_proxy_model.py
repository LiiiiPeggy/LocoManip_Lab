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

"""Build the MuJoCo distance-proxy model from the actual frozen PhysX colliders.

The generated files are proxy-only:

* ``mujoco/robots/ranger_cr10/ranger_cr10_simple_v2_physproxy.xml``
* ``mujoco/robots/ranger_cr10/scene_simple_v2_physproxy.xml``

The physical URDF/USD, the default ``safety.py`` and the existing v2 MuJoCo model are not modified.
The builder keeps the v2 kinematics/inertials/actuators, removes the simplified arm/base collision
geoms, and places the actual cube/convex-hull colliders in the same bodies.

Run:
    python scripts/ranger_cr10/build_phys_proxy_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lxml.etree as ET
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from phys_proxy import (  # noqa: E402
    ARM_GROUPS,
    PROTECTED_PARTS,
    ROOT,
    active_pairs,
    excluded_pairs,
    load_shapes,
)

MODEL_IN = ROOT / "mujoco/robots/ranger_cr10/ranger_cr10_simple_v2.xml"
MODEL_OUT = ROOT / "mujoco/robots/ranger_cr10/ranger_cr10_simple_v2_physproxy.xml"
SCENE_OUT = ROOT / "mujoco/robots/ranger_cr10/scene_simple_v2_physproxy.xml"
MAP_OUT = ROOT / "outputs/stage413/b/simple_collision_v2/collider_proxy_map.json"
REMOVE_BODIES = {
    "base_link", "cr10_base_link", "lidar_link0", "d435_link", "box_link",
    "cr10_Link1", "cr10_Link2", "cr10_Link3", "cr10_Link4", "cr10_Link5", "cr10_Link6",
    "gripper_base_link", "d455_link",
    "gripper_finger1_knuckle_link", "gripper_finger1_finger_link",
    "gripper_finger1_inner_knuckle_link", "gripper_finger1_finger_tip_link",
    "gripper_finger2_knuckle_link", "gripper_finger2_finger_link",
    "gripper_finger2_inner_knuckle_link", "gripper_finger2_finger_tip_link",
}


def body_element(root: ET._Element, name: str) -> ET._Element:
    found = root.xpath(f".//body[@name='{name}']")
    if len(found) != 1:
        raise SystemExit(f"body {name!r} found {len(found)} times")
    return found[0]


def fmt(values) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def main() -> None:
    shapes = [shape for shape in load_shapes() if shape.group in ARM_GROUPS + PROTECTED_PARTS]
    tree = ET.parse(MODEL_IN)
    root = tree.getroot()

    # Remove the simplified collision geoms from the bodies replaced by actual shapes.
    removed = []
    for body_name in REMOVE_BODIES:
        body = body_element(root, body_name)
        for geom in list(body.findall("geom")):
            if geom.get("class") == "robot":
                removed.append((body_name, geom.get("name")))
                body.remove(geom)

    # Add one mesh asset per actual convex-hull mesh collider. Names are unique and local to this
    # proxy model; the source STL and the USD scale are recorded in the map.
    asset = root.find("asset")
    existing_assets = {mesh.get("name") for mesh in asset.findall("mesh")}
    mesh_assets = {}
    for shape in shapes:
        if shape.kind != "mesh":
            continue
        asset_name = f"phys_{shape.name}"
        if asset_name in existing_assets:
            raise SystemExit(f"duplicate mesh asset {asset_name}")
        element = ET.SubElement(asset, "mesh")
        element.set("name", asset_name)
        element.set("file", shape.mesh_file)
        element.set("scale", fmt([shape.mesh_scale] * 3))
        mesh_assets[shape.name] = asset_name

    for shape in shapes:
        body = body_element(root, shape.body)
        geom = ET.SubElement(body, "geom")
        geom.set("name", shape.name)
        geom.set("class", "robot")
        geom.set("group", "3")
        geom.set("rgba", "1 0 1 0.35")
        geom.set("pos", fmt(shape.pos))
        geom.set("quat", fmt(shape.quat))
        if shape.kind == "box":
            geom.set("type", "box")
            geom.set("size", fmt(shape.half))
        elif shape.kind == "mesh":
            geom.set("type", "mesh")
            geom.set("mesh", mesh_assets[shape.name])
        else:
            raise SystemExit(f"unsupported shape {shape.name}")

    tree.write(str(MODEL_OUT), pretty_print=True, xml_declaration=True, encoding="utf-8")
    SCENE_OUT.write_text(
        "<?xml version='1.0' encoding='utf-8'?>\n"
        "<mujoco model=\"ranger_cr10 physproxy scene\">\n"
        "  <include file=\"ranger_cr10_simple_v2_physproxy.xml\" />\n"
        "\n"
        "  <statistic center=\"0 0 0.4\" extent=\"1.5\" />\n"
        "\n"
        "  <visual>\n"
        "    <headlight diffuse=\"0.6 0.6 0.6\" ambient=\"0.3 0.3 0.3\" specular=\"0 0 0\" />\n"
        "    <rgba haze=\"0.15 0.25 0.35 1\" />\n"
        "    <global azimuth=\"-130\" elevation=\"-20\" />\n"
        "  </visual>\n"
        "\n"
        "  <asset>\n"
        "    <texture type=\"skybox\" builtin=\"gradient\" rgb1=\"0.3 0.5 0.7\" rgb2=\"0 0 0\" width=\"512\" height=\"3072\" />\n"
        "    <texture type=\"2d\" name=\"groundplane\" builtin=\"checker\" mark=\"edge\" rgb1=\"0.2 0.3 0.4\" rgb2=\"0.1 0.2 0.3\" markrgb=\"0.8 0.8 0.8\" width=\"300\" height=\"300\" />\n"
        "    <material name=\"groundplane\" texture=\"groundplane\" texuniform=\"true\" texrepeat=\"5 5\" reflectance=\"0.2\" />\n"
        "  </asset>\n"
        "\n"
        "  <worldbody>\n"
        "    <light pos=\"0 0 3\" dir=\"0 0 -1\" directional=\"true\" />\n"
        "    <geom name=\"floor\" type=\"plane\" size=\"0 0 0.05\" material=\"groundplane\" contype=\"1\" conaffinity=\"7\" friction=\"0.35 0.005 0.0001\" />\n"
        "  </worldbody>\n"
        "</mujoco>\n"
    )

    # Compile and verify that the kinematics are unchanged.
    v2 = mujoco.MjModel.from_xml_path(str(MODEL_IN))
    proxy = mujoco.MjModel.from_xml_path(str(MODEL_OUT))
    v2_data = mujoco.MjData(v2)
    proxy_data = mujoco.MjData(proxy)
    mujoco.mj_resetDataKeyframe(v2, v2_data, v2.key("home").id)
    mujoco.mj_resetDataKeyframe(proxy, proxy_data, proxy.key("home").id)
    mujoco.mj_forward(v2, v2_data)
    mujoco.mj_forward(proxy, proxy_data)
    body_errors = {}
    for body_name in ("base_link", "cr10_Link1", "cr10_Link2", "cr10_Link3", "cr10_Link4", "cr10_Link5",
                      "cr10_Link6", "gripper_finger1_finger_link", "gripper_finger2_finger_tip_link"):
        first = mujoco.mj_name2id(v2, mujoco.mjtObj.mjOBJ_BODY, body_name)
        second = mujoco.mj_name2id(proxy, mujoco.mjtObj.mjOBJ_BODY, body_name)
        body_errors[body_name] = float(
            max(np.max(np.abs(v2_data.xpos[first] - proxy_data.xpos[second])),
                np.max(np.abs(v2_data.xquat[first] - proxy_data.xquat[second])))
        )
    if max(body_errors.values()) > 1e-12:
        raise SystemExit(f"proxy kinematics changed: {body_errors}")

    mapping = {
        "source": "outputs/stage413/b/simple_collision_v2/collider_geometry_v2.json",
        "model": str(MODEL_OUT.relative_to(ROOT)),
        "scene": str(SCENE_OUT.relative_to(ROOT)),
        "note": "actual flat-USD collider prims placed in the MuJoCo link frames; proxy-only",
        "removed_simplified_geoms": removed,
        "mesh_assets": mesh_assets,
        "shapes": [
            {
                "name": shape.name,
                "body": shape.body,
                "group": shape.group,
                "part": shape.part,
                "kind": shape.kind,
                "source_path": shape.source_path,
                "approximation": shape.approximation,
                "pos": shape.pos.tolist(),
                "quat": shape.quat.tolist(),
                "half_extents": shape.half.tolist() if shape.half is not None else None,
                "mesh_file": shape.mesh_file,
                "mesh_scale": shape.mesh_scale,
                "hull_points": int(len(shape.points)),
                "supported": bool(shape.supported),
                "note": shape.note,
            }
            for shape in shapes
        ],
        "actual_collider_count": 30,
        "proxy_relevant_collider_count": len(shapes),
        "out_of_scope_colliders": [
            "fl_steering_wheel_link", "fl_wheel_link", "fr_steering_link", "fr_wheel_link",
            "rl_steering_wheel_link", "rl_wheel_link", "rr_steering_wheel_link", "rr_wheel_link",
        ],
        "fixed_joint_merge": {
            "box_link": "base_link",
            "cr10_base_link": "base_link",
            "lidar_link0": "base_link",
            "d435_link": "base_link",
            "gripper_base_link": "cr10_Link6",
            "d455_link": "cr10_Link6",
        },
        "link_frame_note": "the collisions scope is authored in the URDF link frame, which is also "
                           "the MuJoCo body frame; the USD rigid-body frame is a different but "
                           "equivalent convention and is not used to rotate the shape",
        "unsupported_shapes": [shape.name for shape in shapes if not shape.supported],
        "unknown_reference_labels": [],
        "active_pairs": [list(pair) for pair in active_pairs()],
        "excluded_pairs": excluded_pairs(),
        "scope_note": "wheels/steering are represented by the existing simplified geoms and are out of "
                      "the internal-arm-obstacle scope; they are listed, not silently removed",
    }
    MAP_OUT.parent.mkdir(parents=True, exist_ok=True)
    MAP_OUT.write_text(json.dumps(mapping, indent=1))
    print(json.dumps({
        "model": str(MODEL_OUT),
        "scene": str(SCENE_OUT),
        "shapes": len(shapes),
        "mesh_assets": len(mesh_assets),
        "removed_geoms": len(removed),
        "proxy_nbody": proxy.nbody,
        "proxy_ngeom": proxy.ngeom,
        "v2_nbody": v2.nbody,
        "v2_ngeom": v2.ngeom,
        "max_body_error": max(body_errors.values()),
        "map": str(MAP_OUT),
    }, indent=1))


if __name__ == "__main__":
    main()
