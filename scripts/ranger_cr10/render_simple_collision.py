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

"""Render side-by-side old/hull versus simple-box collision visualizations (headless, EGL)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "mujoco/robots/ranger_cr10/scene.xml"
SIMPLE = ROOT / "mujoco/robots/ranger_cr10/scene_simple.xml"
POSES = ROOT / "outputs/stage411/operating_poses.npz"
OUT = ROOT / "outputs/stage413/b/simple_collision_validation/renders"
WIDTH, HEIGHT = 640, 480

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out", default=str(OUT))
args = ap.parse_args()
Path(args.out).mkdir(parents=True, exist_ok=True)


def load(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    return model, data


def set_pose(model, data, q):
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for i, v in enumerate(q, start=1):
        adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")]
        data.qpos[adr] = v
    mujoco.mj_forward(model, data)


def render(model, data, camera):
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()
    renderer.close()
    return Image.fromarray(pixels)


def make_camera(model, data, distance, elevation, azimuth, lookat):
    cam = mujoco.MjvCamera()
    cam.distance = distance
    cam.elevation = elevation
    cam.azimuth = azimuth
    cam.lookat[:] = lookat
    return cam


def label(image, text):
    canvas = Image.new("RGB", (image.width, image.height + 34), (20, 24, 32))
    canvas.paste(image, (0, 34))
    ImageDraw.Draw(canvas).text((12, 10), text, fill=(235, 235, 235))
    return canvas


def main():
    poses = np.load(POSES)["poses"]
    cases = [
        ("default", None, "default pose", 2.6, -18, -135, [0.35, 0.0, 0.55]),
        ("pose1534", poses[1534], "pose 1534", 2.4, -18, -135, [0.35, 0.0, 0.55]),
        ("pose1534_side", poses[1534], "pose 1534 side", 2.4, -12, -90, [0.35, 0.0, 0.55]),
        ("pose1534_front", poses[1534], "pose 1534 front", 2.4, -12, -180, [0.35, 0.0, 0.55]),
        ("box_closeup", poses[1534], "box_link close-up", 1.4, -10, -135, [0.18, 0.0, 0.62]),
        ("arm_closeup", poses[1534], "Link2/3 close-up", 1.3, -12, -120, [0.5, 0.0, 0.9]),
        ("folded", np.array([0.4, -0.3, -1.4, 1.8, -0.9, 0.2]), "folded pose", 2.4, -18, -135, [0.35, 0.0, 0.55]),
    ]
    for name, q, title, distance, elevation, azimuth, lookat in cases:
        images = []
        for label_text, scene_path in (("old collision", DEFAULT), ("simple candidate", SIMPLE)):
            model, data = load(scene_path)
            if q is not None:
                set_pose(model, data, q)
            # show visuals (group 2) and collisions (group 3); hide decorations
            option = mujoco.MjvOption()
            option.geomgroup[:] = [0, 0, 1, 1, 0, 0]
            with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
                renderer.update_scene(data, camera=make_camera(model, data, distance, elevation, azimuth, lookat),
                                      scene_option=option)
                pixels = renderer.render()
            images.append(label(Image.fromarray(pixels), f"{title} | {label_text}"))
        combined = Image.new("RGB", (images[0].width * 2 + 8, images[0].height), (10, 12, 16))
        combined.paste(images[0], (0, 0))
        combined.paste(images[1], (images[0].width + 8, 0))
        path = Path(args.out) / f"{name}.png"
        combined.save(path)
        print("wrote", path, flush=True)
        # collision-only overlay: boxes are inside the visual meshes and can be hidden in the
        # overlay above
        collision_images = []
        for label_text, scene_path in (("old collision only", DEFAULT), ("simple collision only", SIMPLE)):
            model, data = load(scene_path)
            if q is not None:
                set_pose(model, data, q)
            option = mujoco.MjvOption()
            option.geomgroup[:] = [0, 0, 0, 1, 0, 0]
            with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
                renderer.update_scene(data, camera=make_camera(model, data, distance, elevation, azimuth, lookat),
                                      scene_option=option)
                pixels = renderer.render()
            collision_images.append(label(Image.fromarray(pixels), f"{title} | {label_text}"))
        combined = Image.new("RGB", (collision_images[0].width * 2 + 8, collision_images[0].height), (10, 12, 16))
        combined.paste(collision_images[0], (0, 0))
        combined.paste(collision_images[1], (collision_images[0].width + 8, 0))
        path = Path(args.out) / f"{name}_collision.png"
        combined.save(path)
        print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
