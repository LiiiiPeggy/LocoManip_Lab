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

"""Stage 4.13: look at the three geometries the distance work keeps comparing.

* **collision** -- the geoms MuJoCo actually collides: convex hulls for the arm links, primitives for
  the chassis, the cameras and the gripper jaws (group 3);
* **visual** -- the original triangle meshes (group 2), which is what the reference distance measures;
* **boxes** -- the old box proxy `mdp/safety.py` uses, drawn as wireframe cartons only in
  ``--mode boxes``; ``--mode all`` is visual + collision and does not draw it.

They are not the same shape, and the whole stage exists because the difference was invisible. Toggling
between them in one window is the fastest way to see why a hull can report contact while the meshes
are 30 mm apart.

Usage:
    python scripts/ranger_cr10/view_collision_geometry.py                    # default scene, collision hulls only
    python scripts/ranger_cr10/view_collision_geometry.py --scene simple_v2  # v2 candidate collision boxes
    python scripts/ranger_cr10/view_collision_geometry.py --scene simple_v2 --mode all  # v2 visual + collision
    python scripts/ranger_cr10/view_collision_geometry.py --mode visual      # original meshes
    python scripts/ranger_cr10/view_collision_geometry.py --mode boxes       # old safety.py proxy boxes (explicit only)
    python scripts/ranger_cr10/view_collision_geometry.py --scene simple_v2 --mode all \
        --pose-file outputs/stage411/operating_poses.npz --index 93

Keys (the viewer's own): `0`-`8` toggle geom groups, so the collision/visual layers can be switched
off by hand as well.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENES = {
    "default": ROOT / "mujoco/robots/ranger_cr10/scene.xml",
    "simple": ROOT / "mujoco/robots/ranger_cr10/scene_simple.xml",
    "simple_v2": ROOT / "mujoco/robots/ranger_cr10/scene_simple_v2.xml",
    "simple_v2_proxy": ROOT / "mujoco/robots/ranger_cr10/scene_simple_v2_proxy.xml",
}
SAFETY = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/safety.py"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--scene", choices=tuple(SCENES), default="default",
                help="default mesh-collision scene, v1 simple boxes, or v2 candidate")
ap.add_argument("--mode", choices=("collision", "visual", "boxes", "all"), default="collision")
ap.add_argument("--pose-file", default="", help="npz with an 'poses' array of six arm joints")
ap.add_argument("--index", type=int, default=0)
ap.add_argument("--height", type=float, default=0.0, help="camera height above the robot origin")
ap.add_argument("--check_only", action="store_true", help="print what would be drawn and exit (no window)")
args = ap.parse_args()

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402


def load_safety():
    spec = importlib.util.spec_from_file_location("safety_view", SAFETY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quat_matrix(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def box_world_poses(model, data, safety) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(position, rotation, half extents) of every box-proxy part, in the world."""
    out = []
    tables = [(safety.ARM_PARTS, "arm"), (safety.PROTECTED_PARTS, "protected")]
    for table, kind in tables:
        for group, entries in table.items():
            if kind == "arm":
                parts = entries
                body = group
            else:
                body = entries[0]
                parts = entries[1]
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            if bid < 0:
                continue
            body_rotation = data.xmat[bid].reshape(3, 3)
            body_position = data.xpos[bid]
            for _, centre, half, quat in parts:
                local_rotation = quat_matrix(quat) if quat else np.eye(3)
                world_rotation = body_rotation @ local_rotation
                world_position = body_position + body_rotation @ np.asarray(centre)
                out.append((world_position, world_rotation, np.asarray(half)))
    return out


def color_v2_collisions(model) -> None:
    """Make the v2 candidate collision bodies readable in the interactive viewer."""
    green = np.array([0.10, 0.85, 0.20, 0.55], dtype=np.float32)
    orange = np.array([1.00, 0.45, 0.00, 0.60], dtype=np.float32)
    yellow = np.array([1.00, 0.85, 0.10, 0.65], dtype=np.float32)
    gray = np.array([0.55, 0.55, 0.60, 0.30], dtype=np.float32)
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        if name.startswith("cr10_Link") and name.endswith("_col_simple"):
            model.geom_rgba[index] = green
        elif name in ("gripper_base_link_col0", "gripper_base_link_col1", "gripper_base_link_col2",
                      "d455_link_col"):
            model.geom_rgba[index] = green
        elif name == "box_link_col_left":
            model.geom_rgba[index] = orange
        elif name == "box_link_col_top":
            model.geom_rgba[index] = yellow
        elif "col_patch" in name:
            model.geom_rgba[index] = np.array([1.0, 0.0, 1.0, 0.55], dtype=np.float32)
        elif model.geom_group[index] == 3:
            model.geom_rgba[index] = gray


def main() -> None:
    scene_path = SCENES[args.scene]
    print(f"scene: {args.scene} ({scene_path.relative_to(ROOT)})")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    if args.scene in ("simple_v2", "simple_v2_proxy"):
        color_v2_collisions(model)
        print("colors: green = arm/gripper, orange = box_link left, yellow = box_link top, "
              "magenta = proxy patch, gray = chassis/wheels/lidar")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    if args.pose_file:
        pose = np.load(ROOT / args.pose_file)["poses"][args.index]
        for i, value in enumerate(pose, start=1):
            address = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")]
            data.qpos[address] = value
        print(f"pose {args.index} from {args.pose_file}: {np.round(pose, 3)}")
    mujoco.mj_forward(model, data)

    safety = load_safety()
    # ``all`` is visual + collision only. The old safety.py proxy boxes are now an
    # explicit opt-in via --mode boxes; they are not drawn in the normal all view.
    boxes = box_world_poses(model, data, safety) if args.mode == "boxes" else []
    if boxes:
        print(f"box proxy: {len(boxes)} parts (orange wireframe)")

    group_mask = {
        "collision": [0, 0, 0, 1, 0, 0],  # group 3 = collision geoms
        "visual": [0, 0, 1, 0, 0, 0],  # group 2 = the STL meshes
        "boxes": [0, 0, 0, 0, 0, 0],
        "all": [0, 0, 1, 1, 0, 0],
    }[args.mode]

    if args.check_only:
        for position, rotation, half in boxes:
            print(f"  box at {np.round(position, 3)} half {np.round(half, 4)}")
        print(f"mode {args.mode}: {len(boxes)} boxes, collision geoms "
              f"{sum(1 for g in range(model.ngeom) if model.geom_contype[g] != 0)}, meshes "
              f"{sum(1 for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)}")
        return
    print(f"mode {args.mode}; close the window to exit (viewer keys 0-8 still toggle geom groups)")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # MuJoCo >= 3.x exposes the user scenegraph on the handle rather than as a constructor
        # argument; the boxes are drawn into it each frame
        scenegraph = viewer.user_scn
        viewer.opt.geomgroup[:] = group_mask
        viewer.cam.distance = 2.2
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = [0.3, 0.0, args.height + 0.4]
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            if boxes:
                with viewer.lock():
                    scenegraph.ngeom = 0
                    for position, rotation, half in boxes:
                        if scenegraph.ngeom >= scenegraph.maxgeom:
                            break
                        geom = scenegraph.geoms[scenegraph.ngeom]
                        mujoco.mjv_initGeom(
                            geom,
                            mujoco.mjtGeom.mjGEOM_BOX,
                            size=half,
                            pos=position,
                            mat=rotation.reshape(9),
                            rgba=np.array([1.0, 0.55, 0.1, 0.35], dtype=np.float32),
                        )
                        scenegraph.ngeom += 1
            viewer.sync()
    # the viewer teardown segfaults on this machine (stage 4.6); leave before it runs
    os._exit(0)


if __name__ == "__main__":
    main()
