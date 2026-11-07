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

"""Geometry acceptance data for the v2 simple boxes.

Reports, in one JSON:
  * default and 1800-pose adjacent box distances for v1 and v2;
  * non-adjacent arm and arm-protected minimum distances over the same poses;
  * sampled mesh surface coverage of each v2 box;
  * box_link v1/v2 volume, lid height, freed upper space and accessory points
    that remain outside the v2 box.

Run:
    python scripts/ranger_cr10/validate_simple_geometry_v2.py \
        --out outputs/stage413/b/simple_collision_v2/geometry_validation.json
"""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
URDF_V1 = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10_simple/ranger_cr10_simple.urdf"
URDF_V2 = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10_simple_v2/ranger_cr10_simple_v2.urdf"
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes"
POSES = ROOT / "outputs/stage411/operating_poses.npz"
V1_SCENE = ROOT / "mujoco/robots/ranger_cr10/scene_simple.xml"
V2_SCENE = ROOT / "mujoco/robots/ranger_cr10/scene_simple_v2.xml"
OUT = ROOT / "outputs/stage413/b/simple_collision_v2/geometry_validation.json"
ARMS = [f"cr10_Link{i}" for i in range(1, 7)]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default=str(OUT))
ap.add_argument("--samples", type=int, default=30000)
args = ap.parse_args()
rng = np.random.default_rng(0)


def boxes_from_urdf(path):
    root = ET.parse(path).getroot()
    out = {}
    for link in ARMS + ["box_link"]:
        element = root.find(f"link[@name='{link}']")
        if element is None:
            continue
        records = []
        for collision in element.findall("collision"):
            box = collision.find("./geometry/box")
            if box is None:
                continue
            origin = collision.find("origin")
            xyz = [float(v) for v in (origin.get("xyz", "0 0 0").split() if origin is not None else [0, 0, 0])]
            records.append({"center": np.asarray(xyz), "size": np.asarray([float(v) for v in box.get("size").split()])})
        out[link] = records
    return out


V1_BOXES = boxes_from_urdf(URDF_V1)
V2_BOXES = boxes_from_urdf(URDF_V2)


def sample_mesh(path, count):
    mesh = trimesh.load(str(path), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if len(faces) == 0:
        return vertices
    chosen = rng.choice(len(faces), size=min(count, len(faces)), replace=True)
    weights = rng.dirichlet([1, 1, 1], size=len(chosen))
    return np.einsum("ij,ijk->ik", weights, vertices[faces[chosen]])


def inside_box(points, box, margin=0.0):
    center, size = box["center"], box["size"] / 2 + margin
    return np.all(np.abs(points - center) <= size, axis=1)


def coverage(points, boxes):
    inside = np.zeros(len(points), dtype=bool)
    for box in boxes:
        inside |= inside_box(points, box)
    outside = points[~inside]
    distance = 0.0
    if len(outside):
        distances = []
        for point in outside:
            best = np.inf
            for box in boxes:
                delta = np.maximum(np.abs(point - box["center"]) - box["size"] / 2, 0.0)
                best = min(best, float(np.linalg.norm(delta)))
            distances.append(best)
        distance = float(np.median(distances))
    return {
        "n": int(len(points)),
        "inside": int(inside.sum()),
        "inside_fraction": float(inside.mean()),
        "outside_median_distance_m": distance,
    }


def load_model(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i): i for i in range(model.ngeom)}
    addresses = [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")]
        for i in range(1, 7)
    ]
    return model, data, names, addresses


MODELS = {"v1": load_model(V1_SCENE), "v2": load_model(V2_SCENE)}


def set_q(model, data, addresses, q):
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for address, value in zip(addresses, q):
        data.qpos[address] = float(value)
    mujoco.mj_forward(model, data)


def pair_distance(tag, a, b):
    model, data, names, _ = MODELS[tag]
    return float(mujoco.mj_geomDistance(model, data, names[a], names[b], 1.0, None))


def pose_stats(tag):
    model, data, names, addresses = MODELS[tag]
    poses = np.load(POSES)["poses"]
    adjacent = [(f"cr10_Link{i}_col_simple", f"cr10_Link{i + 1}_col_simple") for i in range(1, 6)]
    non_adjacent = []
    for index, a in enumerate(ARMS):
        for b in ARMS[index + 1 :]:
            if int(b.replace("cr10_Link", "")) != int(a.replace("cr10_Link", "")) + 1:
                non_adjacent.append((f"{a}_col_simple", f"{b}_col_simple"))
    protected = ["base_link_col", "cr10_base_link_col", "lidar_link0_col", "d435_link_col",
                 "box_link_col_left",
                 "box_link_col_plate" if tag == "v1" else "box_link_col_top"]
    tracks = {f"{a}|{b}": {"min": np.inf, "n_neg": 0, "n_lt2": 0, "worst_pose": None}
              for a, b in adjacent + non_adjacent
              + [(f"{a}_col_simple", protected_geom) for a in ARMS for protected_geom in protected]}
    for pose_index, q in enumerate(poses):
        set_q(model, data, addresses, q)
        for key in tracks:
            a, b = key.split("|")
            distance = pair_distance(tag, a, b)
            track = tracks[key]
            if distance < track["min"]:
                track["min"] = distance
                track["worst_pose"] = pose_index
            if distance < 0:
                track["n_neg"] += 1
            if distance < 0.002:
                track["n_lt2"] += 1
    return {
        key: {
            "min_m": float(track["min"]),
            "n_negative": track["n_neg"],
            "n_below_2mm": track["n_lt2"],
            "worst_pose": track["worst_pose"],
        }
        for key, track in tracks.items()
    }


geometry = {
    "poses": str(POSES.relative_to(ROOT)),
    "adjacent_default": {},
    "pose_stats": {},
}
for tag in ("v1", "v2"):
    model, data, names, addresses = MODELS[tag]
    set_q(model, data, addresses, np.array([0.236, 0.086, -2.398, -0.83, -1.335, 0.0]))
    geometry["adjacent_default"][tag] = {
        f"cr10_Link{i}|cr10_Link{i + 1}": pair_distance(
            tag, f"cr10_Link{i}_col_simple", f"cr10_Link{i + 1}_col_simple"
        )
        for i in range(1, 6)
    }
    geometry["pose_stats"][tag] = pose_stats(tag)

# Mesh surface coverage of each v2 box.
geometry["arm_coverage"] = {}
for link in ARMS:
    mesh_name = f"{link.replace('cr10_', '')}.STL"
    points = sample_mesh(MESH_DIR / "cr10_meshes" / mesh_name, args.samples)
    geometry["arm_coverage"][link] = {
        "v1": coverage(points, V1_BOXES.get(link, [])),
        "v2": coverage(points, V2_BOXES.get(link, [])),
        "v1_boxes": [{"center": box["center"].tolist(), "size": box["size"].tolist()} for box in V1_BOXES.get(link, [])],
        "v2_boxes": [{"center": box["center"].tolist(), "size": box["size"].tolist()} for box in V2_BOXES.get(link, [])],
    }

# box_link: surface coverage, lid height and freed volume.
box_points = sample_mesh(MESH_DIR / "box_link.STL", args.samples * 4)
v1_left = V1_BOXES["box_link"][0]
v2_left = V2_BOXES["box_link"][0]
v2_top = V2_BOXES["box_link"][1]
inside_v1 = inside_box(box_points, v1_left)
inside_v2 = inside_box(box_points, v2_left) | inside_box(box_points, v2_top)
lid = float(v2_left["center"][2] + v2_left["size"][2] / 2)
old_top = float(v1_left["center"][2] + v1_left["size"][2] / 2)
above_lid = box_points[:, 2] > lid
gross_freed = float(np.prod(v2_left["size"][:2]) * (old_top - lid))
top_volume = float(np.prod(v2_top["size"]))
geometry["box_link"] = {
    "v1_left": {"center": v1_left["center"].tolist(), "size": v1_left["size"].tolist(),
                "volume_m3": float(np.prod(v1_left["size"])), "top_z": old_top},
    "v2_left": {"center": v2_left["center"].tolist(), "size": v2_left["size"].tolist(),
                "volume_m3": float(np.prod(v2_left["size"])), "top_z": lid},
    "v2_top": {"center": v2_top["center"].tolist(), "size": v2_top["size"].tolist(),
               "volume_m3": top_volume, "z_range": [float(v2_top["center"][2] - v2_top["size"][2] / 2),
                                                    float(v2_top["center"][2] + v2_top["size"][2] / 2)]},
    "right_plate_collision_removed": True,
    "right_plate_visual_only": True,
    "freed_height_m": old_top - lid,
    "freed_volume_gross_m3": gross_freed,
    "freed_volume_net_m3": gross_freed - top_volume,
    "top_box_footprint_fraction_of_lid": float(
        (v2_top["size"][0] * v2_top["size"][1]) / (v2_left["size"][0] * v2_left["size"][1])
    ),
    "surface_points": int(len(box_points)),
    "v1_inside_fraction": float(inside_v1.mean()),
    "v2_inside_fraction": float(inside_v2.mean()),
    "points_above_lid": int(above_lid.sum()),
    "points_above_lid_outside_v2": int((above_lid & ~inside_v2).sum()),
    "note": "coverage is sampled surface points; the v2 left box stops at the lid, the small top "
            "accessory has its own box, and the right mounting plate collision was removed per user "
            "request (visual only)",
}

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as handle:
    json.dump(geometry, handle, indent=1)
print(
    json.dumps(
        {
            "adjacent_default": geometry["adjacent_default"],
            "v2_adjacent_pose_min": {
                key: round(value["min_m"] * 1000, 3) + 0
                for key, value in geometry["pose_stats"]["v2"].items()
                if key.startswith("cr10_Link") and "|cr10_Link" in key
            },
            "arm_coverage_v2": {
                link: round(data["v2"]["inside_fraction"], 3) for link, data in geometry["arm_coverage"].items()
            },
            "box_link": {key: geometry["box_link"][key] for key in
                         ("freed_height_m", "freed_volume_gross_m3", "freed_volume_net_m3",
                          "top_box_footprint_fraction_of_lid", "v1_inside_fraction", "v2_inside_fraction",
                          "points_above_lid_outside_v2")},
        },
        indent=1,
    )
)
print(f"wrote {args.out}")
