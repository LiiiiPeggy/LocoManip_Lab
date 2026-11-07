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

"""Measure each arm link's shaft and joint housing in its own link frame.

For every Link1--Link6 mesh this reports the AABB, the proximal (parent) and distal
(child) joint origins in the link frame, and cross-section extents in slices along
the proximal->distal axis. The slices show where the link is a thin shaft and where
the joint housings are thick, which is what the simplified box has to represent.
It also compares the current v1 box with the mesh and with the other v1 boxes.

Run:
    python scripts/ranger_cr10/analyze_simple_geometry.py \
        --out outputs/stage413/b/simple_collision_v2/arm_geometry.json
"""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10_simple_v2/ranger_cr10_simple_v2.urdf"
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes/cr10_meshes"
OUT = ROOT / "outputs/stage413/b/simple_collision_v2/arm_geometry.json"
LINKS = [f"cr10_Link{i}" for i in range(1, 7)]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--urdf", default=str(URDF))
ap.add_argument("--out", default=str(OUT))
ap.add_argument("--bins", type=int, default=24)
args = ap.parse_args()

tree = ET.parse(args.urdf)
robot = tree.getroot()
joints = {}
for joint in robot.findall("joint"):
    origin = joint.find("origin")
    xyz = [float(v) for v in (origin.get("xyz", "0 0 0").split() if origin is not None else [0, 0, 0])]
    rpy = [float(v) for v in (origin.get("rpy", "0 0 0").split() if origin is not None else [0, 0, 0])]
    joints[joint.get("name")] = {
        "parent": joint.find("parent").get("link"),
        "child": joint.find("child").get("link"),
        "xyz": xyz,
        "rpy": rpy,
        "type": joint.get("type"),
    }
parent_of = {record["child"]: (name, record) for name, record in joints.items() if record["type"] != "fixed"}
child_joint_of = {}
for name, record in joints.items():
    if record["type"] != "fixed":
        child_joint_of.setdefault(record["parent"], []).append((name, record))


def collision_box(link_name):
    link = robot.find(f"link[@name='{link_name}']")
    if link is None:
        return None
    collisions = link.findall("collision")
    boxes = []
    for collision in collisions:
        box = collision.find("./geometry/box")
        if box is None:
            continue
        origin = collision.find("origin")
        xyz = [float(v) for v in (origin.get("xyz", "0 0 0").split() if origin is not None else [0, 0, 0])]
        rpy = [float(v) for v in (origin.get("rpy", "0 0 0").split() if origin is not None else [0, 0, 0])]
        size = [float(v) for v in box.get("size").split()]
        boxes.append({"xyz": xyz, "rpy": rpy, "size": size})
    return boxes


def rotate_rpy(rpy):
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def slice_table(points, axis, bins):
    projection = points @ axis
    low, high = float(projection.min()), float(projection.max())
    edges = np.linspace(low, high, bins + 1)
    # choose the two axes least aligned with the main axis
    order = np.argsort(np.abs(axis))
    u = np.eye(3)[order[0]]
    v = np.eye(3)[order[1]]
    rows = []
    for index in range(bins):
        mask = (projection >= edges[index]) & (projection <= edges[index + 1])
        if not mask.any():
            rows.append({"s": [float(edges[index]), float(edges[index + 1])], "n": 0})
            continue
        subset = points[mask]
        pu, pv = subset @ u, subset @ v
        rows.append(
            {
                "s": [float(edges[index]), float(edges[index + 1])],
                "n": int(mask.sum()),
                "u": [float(np.percentile(pu, 2)), float(np.percentile(pu, 98))],
                "v": [float(np.percentile(pv, 2)), float(np.percentile(pv, 98))],
                "u_absmax": float(np.max(np.abs(pu))),
                "v_absmax": float(np.max(np.abs(pv))),
            }
        )
    return {"axis": axis.tolist(), "u_axis": u.tolist(), "v_axis": v.tolist(), "range": [low, high], "rows": rows}


result = {"urdf": os.path.abspath(args.urdf), "links": {}}
for link_name in LINKS:
    mesh_path = MESH_DIR / f"{link_name.replace('cr10_', '')}.STL"
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    points = np.asarray(mesh.vertices, dtype=np.float64)
    boxes = collision_box(link_name)
    parent_joint = parent_of.get(link_name)
    child = child_joint_of.get(link_name, [])
    proximal = np.zeros(3)
    distal = None
    if child:
        distal = np.asarray(child[0][1]["xyz"], dtype=np.float64)
    elif parent_joint:
        # Link6 has no child joint; use the proximal joint axis as the link direction.
        distal = np.asarray([0.0, 0.1, 0.0])
    axis = (distal - proximal) if distal is not None else np.asarray([1.0, 0.0, 0.0])
    if np.linalg.norm(axis) < 1e-9:
        axis = np.asarray([1.0, 0.0, 0.0])
    axis = axis / np.linalg.norm(axis)
    slices = slice_table(points, axis, args.bins)
    # Box projection range and cross-section envelope on the same axes.
    box_rows = []
    for box in boxes or []:
        center = np.asarray(box["xyz"], dtype=np.float64)
        half = np.asarray(box["size"], dtype=np.float64) / 2.0
        rotation = rotate_rpy(box["rpy"])
        corners = np.array(
            [[sx * half[0], sy * half[1], sz * half[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        )
        corners = corners @ rotation.T + center
        box_rows.append(
            {
                "xyz": box["xyz"],
                "rpy": box["rpy"],
                "size": box["size"],
                "aabb_min": corners.min(axis=0).tolist(),
                "aabb_max": corners.max(axis=0).tolist(),
                "axis_range": [float((corners @ axis).min()), float((corners @ axis).max())],
            }
        )
    result["links"][link_name] = {
        "mesh": str(mesh_path.relative_to(ROOT)),
        "aabb_min": points.min(axis=0).tolist(),
        "aabb_max": points.max(axis=0).tolist(),
        "extent": (points.max(axis=0) - points.min(axis=0)).tolist(),
        "proximal_joint": proximal.tolist(),
        "distal_joint": distal.tolist() if distal is not None else None,
        "parent_joint": parent_joint[0] if parent_joint else None,
        "child_joint": child[0][0] if child else None,
        "axis": axis.tolist(),
        "slices": slices,
        "boxes": box_rows,
    }

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as handle:
    json.dump(result, handle, indent=1)

# Compact terminal table.
for link_name, record in result["links"].items():
    print(f"\n{link_name}: axis {np.round(record['axis'], 3)}  AABB min {np.round(record['aabb_min'], 4)}")
    print(f"  distal joint {np.round(record['distal_joint'], 4) if record['distal_joint'] else None}")
    for box in record["boxes"]:
        print(f"  v1 box size {box['size']} at {box['xyz']}  axis range {np.round(box['axis_range'], 4)}")
    print("  s-range (main axis)      u-range 2-98%           v-range 2-98%          n")
    for row in record["slices"]["rows"]:
        if row["n"] == 0:
            continue
        print(
            f"  [{row['s'][0]: .3f},{row['s'][1]: .3f}]  "
            f"[{row['u'][0]: .3f},{row['u'][1]: .3f}]  "
            f"[{row['v'][0]: .3f},{row['v'][1]: .3f}]  {row['n']}"
        )
print(f"\nwrote {args.out}")
