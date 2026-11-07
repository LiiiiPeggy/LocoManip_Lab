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

"""Headless screenshots of the v1/v2 simple collision bodies (MuJoCo EGL).

The earlier renders were blank because the camera framing missed the robot; the
free camera auto-frames the whole model, and the close-ups below were checked
against the actual robot. Collision geoms are recolored at runtime:
  * arm simple boxes: green
  * box_link left enclosure: orange
  * box_link top accessory: yellow (the right mounting plate collision is removed;
    it remains visual only)
  * chassis/wheels/lidar: gray
v1 comparison boxes are red; the v1 right plate is purple for comparison.

Run:
    python scripts/ranger_cr10/render_simple_collision_v2.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCENES = {
    "v1": ROOT / "mujoco/robots/ranger_cr10/scene_simple.xml",
    "v2": ROOT / "mujoco/robots/ranger_cr10/scene_simple_v2.xml",
    "mesh": ROOT / "mujoco/robots/ranger_cr10/scene.xml",
}
OUT = ROOT / "outputs/stage413/b/simple_collision_v2/screenshots"
WIDTH, HEIGHT = 640, 480
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "arm": np.array([0.10, 0.85, 0.20, 0.55], dtype=np.float32),
    "left": np.array([1.00, 0.45, 0.00, 0.60], dtype=np.float32),
    "top": np.array([1.00, 0.85, 0.10, 0.65], dtype=np.float32),
    "chassis": np.array([0.55, 0.55, 0.60, 0.30], dtype=np.float32),
    "v1_arm": np.array([0.90, 0.10, 0.10, 0.55], dtype=np.float32),
    "v1_left": np.array([0.60, 0.10, 0.10, 0.55], dtype=np.float32),
    "v1_plate": np.array([0.60, 0.10, 0.60, 0.55], dtype=np.float32),
}


def load(scene):
    model = mujoco.MjModel.from_xml_path(str(SCENES[scene]))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return model, data


def recolor(model, version):
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        if name.startswith("cr10_Link") and name.endswith("_col_simple"):
            model.geom_rgba[index] = COLORS["arm"] if version == "v2" else COLORS["v1_arm"]
        elif name in ("gripper_base_link_col0", "gripper_base_link_col1", "gripper_base_link_col2",
                      "d455_link_col"):
            model.geom_rgba[index] = COLORS["arm"] if version == "v2" else COLORS["v1_arm"]
        elif name == "box_link_col_left":
            model.geom_rgba[index] = COLORS["left"] if version == "v2" else COLORS["v1_left"]
        elif name == "box_link_col_top":
            model.geom_rgba[index] = COLORS["top"]
        elif name == "box_link_col_plate":
            model.geom_rgba[index] = COLORS["v1_plate"]
        elif model.geom_group[index] == 3:
            model.geom_rgba[index] = COLORS["chassis"]


def camera(name):
    cam = mujoco.MjvCamera()
    if name == "box":
        cam.distance, cam.elevation, cam.azimuth = 1.5, -8, -125
        cam.lookat[:] = [-0.2, 0.0, 0.62]
    elif name == "wrist":
        cam.distance, cam.elevation, cam.azimuth = 0.65, -5, -110
        cam.lookat[:] = [0.7, -0.1, 0.8]
    elif name == "front":
        cam.distance, cam.elevation, cam.azimuth = 1.4, -3, -90
        cam.lookat[:] = [-0.2, 0.0, 0.6]
    else:
        return None
    return cam


def render(scene, version, camera_name, visuals=True, collisions=True):
    model, data = load(scene)
    recolor(model, version)
    option = mujoco.MjvOption()
    option.geomgroup[:] = [0, 0, 1 if visuals else 0, 1 if collisions else 0, 0, 0]
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = camera(camera_name)
    if cam is None:
        renderer.update_scene(data, scene_option=option)
    else:
        renderer.update_scene(data, camera=cam, scene_option=option)
    pixels = renderer.render()
    renderer.close()
    return Image.fromarray(pixels)


def label(image, text):
    canvas = Image.new("RGB", (image.width, image.height + 32), (22, 26, 34))
    canvas.paste(image, (0, 32))
    ImageDraw.Draw(canvas).text((10, 9), text, fill=(238, 238, 238))
    return canvas


def save_single(name, scene, version, camera_name, visuals, collisions, title):
    image = label(render(scene, version, camera_name, visuals, collisions), title)
    path = OUT / name
    image.save(path)
    print("wrote", path, flush=True)


def save_pair(name, camera_name, visuals, collisions, title):
    images = [
        label(render("v1", "v1", camera_name, visuals, collisions), f"{title} | v1"),
        label(render("v2", "v2", camera_name, visuals, collisions), f"{title} | v2"),
    ]
    combined = Image.new("RGB", (images[0].width * 2 + 8, images[0].height), (10, 12, 16))
    combined.paste(images[0], (0, 0))
    combined.paste(images[1], (images[0].width + 8, 0))
    path = OUT / name
    combined.save(path)
    print("wrote", path, flush=True)


# Whole robot: visual, visual+boxes, boxes only.
save_single("default_visual.png", "v2", "v2", None, True, False, "v2 default | visual meshes")
save_single("default_boxes.png", "v2", "v2", None, True, True, "v2 default | visual + collision boxes")
save_single("default_boxes_only.png", "v2", "v2", None, False, True, "v2 default | collision boxes only")

# box_link close-up and freed space.
save_single("box_boxes.png", "v2", "v2", "box", True, True, "v2 box_link | visual + boxes")
save_single("box_boxes_only.png", "v2", "v2", "box", False, True, "v2 box_link | boxes only")
save_pair("box_v1_vs_v2_boxes_only.png", "box", False, True, "box_link boxes only")
save_pair("box_front_v1_vs_v2_boxes_only.png", "front", False, True, "box_link front boxes only")

# wrist close-up.
save_single("wrist_boxes.png", "v2", "v2", "wrist", True, True, "v2 wrist | visual + boxes")
save_single("wrist_boxes_only.png", "v2", "v2", "wrist", False, True, "v2 wrist | boxes only")
save_pair("wrist_v1_vs_v2_boxes_only.png", "wrist", False, True, "wrist boxes only")
