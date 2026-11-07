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
* **boxes** -- the box proxy `mdp/safety.py` uses, drawn as wireframe cartons (user scenegraph).

They are not the same shape, and the whole stage exists because the difference was invisible. Toggling
between them in one window is the fastest way to see why a hull can report contact while the meshes
are 30 mm apart.

Usage:
    python scripts/ranger_cr10/view_collision_geometry.py                    # collision hulls only
    python scripts/ranger_cr10/view_collision_geometry.py --mode visual      # original meshes
    python scripts/ranger_cr10/view_collision_geometry.py --mode boxes       # proxy boxes
    python scripts/ranger_cr10/view_collision_geometry.py --mode all         # all three at once
    python scripts/ranger_cr10/view_collision_geometry.py --pose-file outputs/stage411/operating_poses.npz --index 93

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
SCENE = ROOT / "mujoco/robots/ranger_cr10/scene.xml"
SAFETY = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/safety.py"

ap = argparse.ArgumentParser(description=__doc__)
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


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
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
    boxes = box_world_poses(model, data, safety) if args.mode in ("boxes", "all") else []
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
