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

"""Orthographic geometry overlays for the v2 simple boxes.

The MuJoCo camera renders were blank because the camera framing missed the robot.
These figures project the actual mesh surface points and the v1/v2 box rectangles
onto the three principal planes, so coverage and the freed box_link space are
visible without a renderer.

Outputs (PNG + one JSON summary):
  box_link_top/front/side.png
  arm_Link1..6.png
  arm_all.png

Run:
    python scripts/ranger_cr10/render_simple_geometry_v2.py \
        --out outputs/stage413/b/simple_collision_v2/geometry_figures
"""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
URDF_V1 = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10_simple/ranger_cr10_simple.urdf"
URDF_V2 = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10_simple_v2/ranger_cr10_simple_v2.urdf"
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes"
OUT = ROOT / "outputs/stage413/b/simple_collision_v2/geometry_figures"
LINKS = [f"cr10_Link{i}" for i in range(1, 7)]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default=str(OUT))
args = ap.parse_args()


def parse_boxes(path):
    root = ET.parse(path).getroot()
    boxes = {}
    for link in [f"cr10_Link{i}" for i in range(1, 7)] + ["box_link"]:
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
            size = [float(v) for v in box.get("size").split()]
            records.append({"center": xyz, "size": size})
        boxes[link] = records
    return boxes


V1 = parse_boxes(URDF_V1)
V2 = parse_boxes(URDF_V2)
os.makedirs(args.out, exist_ok=True)


def rectangle(center, size, axis_a, axis_b):
    return [
        (center[axis_a] - size[axis_a] / 2, center[axis_b] - size[axis_b] / 2),
        (center[axis_a] + size[axis_a] / 2, center[axis_b] - size[axis_b] / 2),
        (center[axis_a] + size[axis_a] / 2, center[axis_b] + size[axis_b] / 2),
        (center[axis_a] - size[axis_a] / 2, center[axis_b] + size[axis_b] / 2),
    ]


def draw_boxes(ax, boxes, axis_a, axis_b, colors, labels):
    for box, color, label in zip(boxes, colors, labels):
        xy = rectangle(box["center"], box["size"], axis_a, axis_b)
        patch = plt.Polygon(xy, closed=True, fill=False, edgecolor=color, linewidth=1.6, linestyle="--", label=label)
        ax.add_patch(patch)


def mesh_points(path, count=40000):
    mesh = trimesh.load(str(path), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if len(faces) == 0:
        return vertices
    rng = np.random.default_rng(0)
    chosen = rng.choice(len(faces), size=min(count, len(faces)), replace=True)
    weights = rng.dirichlet([1, 1, 1], size=len(chosen))
    return np.einsum("ij,ijk->ik", weights, vertices[faces[chosen]])


def box_link_figures():
    points = mesh_points(MESH_DIR / "box_link.STL")
    views = [("top", 0, 1, "x [m]", "y [m]"), ("front", 0, 2, "x [m]", "z [m]"), ("side", 1, 2, "y [m]", "z [m]")]
    for name, axis_a, axis_b, label_a, label_b in views:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(points[:, axis_a], points[:, axis_b], s=0.15, color="0.55", alpha=0.35, label="box_link mesh")
        draw_boxes(ax, V1["box_link"], axis_a, axis_b, ["#d62728", "#9467bd"], ["v1 boxes", "_"])
        draw_boxes(ax, V2["box_link"], axis_a, axis_b, ["#e69f00", "#f0e442"], ["v2 boxes", "_"])
        if name in ("front", "side"):
            ax.axhspan(0.2555, 0.3816, color="#2ca02c", alpha=0.10)
            ax.axhline(0.2555, color="#2ca02c", linewidth=1.0, linestyle=":")
            ax.axhline(0.3816, color="#d62728", linewidth=1.0, linestyle=":")
            ax.text(0.02, 0.262, "v2 lid top 0.2555 m", color="#2ca02c", transform=ax.get_yaxis_transform())
            ax.text(0.02, 0.372, "v1 top 0.3816 m", color="#d62728", transform=ax.get_yaxis_transform())
        ax.set_aspect("equal")
        ax.set_xlabel(label_a)
        ax.set_ylabel(label_b)
        ax.set_title(f"box_link v1 vs v2 - {name} projection")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", markerscale=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"box_link_{name}.png"), dpi=160)
        plt.close(fig)


def arm_figures():
    fig_all, axes_all = plt.subplots(6, 1, figsize=(10, 18))
    for index, link in enumerate(LINKS):
        mesh_name = f"{link.replace('cr10_', '')}.STL"
        points = mesh_points(MESH_DIR / "cr10_meshes" / mesh_name, count=12000)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, (axis_a, axis_b, label_a, label_b) in zip(
            axes,
            [(0, 1, "x [m]", "y [m]"), (0, 2, "x [m]", "z [m]"), (1, 2, "y [m]", "z [m]")],
        ):
            ax.scatter(points[:, axis_a], points[:, axis_b], s=0.2, color="0.55", alpha=0.35)
            draw_boxes(ax, V1.get(link, []), axis_a, axis_b, ["#d62728"], ["v1 box"])
            draw_boxes(ax, V2.get(link, []), axis_a, axis_b, ["#2ca02c"], ["v2 box"])
            ax.set_aspect("equal")
            ax.set_xlabel(label_a)
            ax.set_ylabel(label_b)
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right", markerscale=6, fontsize=7)
        fig.suptitle(f"{link}: mesh vs v1/v2 collision box (link frame)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"arm_{link}.png"), dpi=150)
        plt.close(fig)
        ax = axes_all[index]
        ax.scatter(points[:, 0], points[:, 1], s=0.15, color="0.6", alpha=0.3)
        draw_boxes(ax, V1.get(link, []), 0, 1, ["#d62728"], ["v1"])
        draw_boxes(ax, V2.get(link, []), 0, 1, ["#2ca02c"], ["v2"])
        ax.set_aspect("equal")
        ax.set_title(f"{link} xy", fontsize=9)
    fig_all.tight_layout()
    fig_all.savefig(os.path.join(args.out, "arm_all.png"), dpi=140)
    plt.close(fig_all)


def box3d_figure():
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    points = mesh_points(MESH_DIR / "box_link.STL", count=15000)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.2, color="0.55", alpha=0.3, label="box_link mesh")
    for name, boxes, colors in (
        ("v1", V1["box_link"], ["#d62728", "#9467bd"]),
        ("v2", V2["box_link"], ["#e69f00", "#f0e442"]),
    ):
        for box, color in zip(boxes, colors):
            c = np.asarray(box["center"])
            s = np.asarray(box["size"]) / 2
            corners = np.array([[sx * s[0], sy * s[1], sz * s[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) + c
            for i in range(8):
                for j in range(i + 1, 8):
                    if np.abs(corners[i] - corners[j]).sum() > 1e-9 and np.count_nonzero(np.abs(corners[i] - corners[j]) > 1e-9) == 1:
                        ax.plot(*zip(corners[i], corners[j]), color=color, linewidth=1.0, linestyle="--" if name == "v1" else "-")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_box_aspect((1.2, 0.9, 0.6))
    ax.set_title("box_link mesh with v1 (red dashed) and v2 (green) boxes")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "box_link_3d.png"), dpi=160)
    plt.close(fig)


def wrist3d_figure():
    """Default-pose wrist: Link4/5/6 meshes with the v1/v2 boxes in world."""
    import mujoco
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    model = mujoco.MjModel.from_xml_path(str(ROOT / "mujoco/robots/ranger_cr10/scene_simple_v2.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    bodies = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)}
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    for link in ("cr10_Link4", "cr10_Link5", "cr10_Link6"):
        mesh = trimesh.load(str(MESH_DIR / "cr10_meshes" / f"{link.replace('cr10_', '')}.STL"), force="mesh", process=False)
        vertices = np.asarray(mesh.vertices)
        body = bodies[link]
        rotation = data.xmat[body].reshape(3, 3)
        translation = data.xpos[body]
        world = vertices @ rotation.T + translation
        ax.scatter(world[:, 0], world[:, 1], world[:, 2], s=0.2, color="0.55", alpha=0.3)
        for name, boxes, color in (("v1", V1.get(link, []), "#d62728"), ("v2", V2.get(link, []), "#2ca02c")):
            for box in boxes:
                local = np.asarray(box["center"])
                half = np.asarray(box["size"]) / 2
                corners = np.array(
                    [[sx * half[0], sy * half[1], sz * half[2]]
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
                )
                corners = corners @ rotation.T + (rotation @ local + translation)
                for i in range(8):
                    for j in range(i + 1, 8):
                        if np.count_nonzero(np.abs(corners[i] - corners[j]) > 1e-9) == 1:
                            ax.plot(*zip(corners[i], corners[j]), color=color, linewidth=1.0,
                                    linestyle="--" if name == "v1" else "-")
    for joint in ("cr10_joint4", "cr10_joint5", "cr10_joint6"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        ax.scatter(*data.xanchor[jid], color="black", s=25)
        ax.text(*data.xanchor[jid], joint.split("joint")[-1], fontsize=8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_box_aspect((0.6, 0.7, 0.6))
    ax.set_title("default wrist: Link4/5/6 meshes, v1 (red) vs v2 (green) boxes")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "wrist_default_3d.png"), dpi=160)
    plt.close(fig)


box_link_figures()
box3d_figure()
arm_figures()
wrist3d_figure()
summary = {
    "out": os.path.abspath(args.out),
    "box_link_v1": V1["box_link"],
    "box_link_v2": V2["box_link"],
    "arm_v1": {link: V1[link] for link in LINKS},
    "arm_v2": {link: V2[link] for link in LINKS},
    "note": "orthographic projections of the original mesh surface points; rectangles are the URDF collision boxes",
}
with open(os.path.join(args.out, "boxes.json"), "w") as handle:
    json.dump(summary, handle, indent=1)
print(json.dumps(summary, indent=1)[:3000])
print(f"wrote {args.out}")
