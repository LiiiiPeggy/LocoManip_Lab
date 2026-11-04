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

"""Stage 4.11: derive the analytic collision boxes from the MuJoCo model.

Before this script the box table in ``mdp/safety.py`` was read by hand out of
``m.mesh_vert``, which is **not the link frame**. MuJoCo re-expresses every mesh in its own
geom frame (``geom_quat``), so for a link whose mesh is stored rotated -- Link3 is 90 degrees
about y, Link5 87 degrees about x -- the hand-copied box was the *right shape in the wrong
frame*. Measured on 1200 random arm configurations, the resulting proxy was 62% blind to the
real ``cr10_Link3 <-> cr10_Link5`` contact: the box for a 675 mm link pointed along the wrong
axis, so its ends sat ~0.3 m away from the geometry they were meant to cover.

This script asks the model for the geometry instead of copying numbers: for each body it
takes every collision geom (mesh or primitive), moves it into the frame of the **Isaac body**
that will own it (the URDF->USD conversion merges fixed joints, so ``gripper_base_link`` and
``d455_link`` are part of ``cr10_Link6``), and reports the axis-aligned bounds there.
Re-run it whenever ``build_mjcf.py`` changes the collision geometry -- the D455 mount moved in
4.10 and the table in ``safety.py`` silently kept the old position until this stage.

Usage:
    python scripts/ranger_cr10/build_collision_boxes.py            # print the table
    python scripts/ranger_cr10/build_collision_boxes.py --json out # also write a JSON dump
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "mujoco/robots/ranger_cr10/scene.xml"

# Isaac body -> (frame, the MuJoCo bodies whose collision geometry it carries). The fixed
# joints are merged by `merge_fixed_joints=True`, so `gripper_base_link` and `d455_link` are
# part of `cr10_Link6`, and `lidar_link0` / `d435_link` are part of `base_link`. A frame that
# is not the owning body matters: it is the only body Isaac actually has, so a box expressed
# anywhere else would be applied to the wrong transform at runtime.
GROUPS = {
    "cr10_Link1": ("cr10_Link1", ["cr10_Link1"]),
    "cr10_Link2": ("cr10_Link2", ["cr10_Link2"]),
    "cr10_Link3": ("cr10_Link3", ["cr10_Link3"]),
    "cr10_Link4": ("cr10_Link4", ["cr10_Link4"]),
    "cr10_Link5": ("cr10_Link5", ["cr10_Link5"]),
    "cr10_Link6": ("cr10_Link6", ["cr10_Link6", "gripper_base_link", "d455_link"]),
    "base_link": ("base_link", ["base_link"]),
    # protected parts, expressed in the frame Isaac has -- all three are rigid to base_link
    "lidar_link0": ("base_link", ["lidar_link0"]),
    "d435_link": ("base_link", ["d435_link"]),
}


def rotation_from_quat(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Rotation matrix of a (w, x, y, z) quaternion."""
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def geom_corners_in(model, data, geom: int, target: int) -> np.ndarray:
    """Every vertex of one collision geom, in the frame of body ``target``."""
    rt = data.xmat[target].reshape(3, 3)
    pt = data.xpos[target]
    rg = data.geom_xmat[geom].reshape(3, 3)
    rel_rot = rt.T @ rg
    rel_pos = rt.T @ (data.geom_xpos[geom] - pt)
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        verts = model.mesh_vert[start : start + count]
        return verts @ rel_rot.T + rel_pos
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
        return (signs * model.geom_size[geom]) @ rel_rot.T + rel_pos
    raise ValueError(f"unsupported geom type {model.geom_type[geom]}")


def parts_for(model, data, target_name: str, bodies: list[str], slabs: int) -> list[dict]:
    """One entry per collision geom (or per slab): centre and half extent in the target frame.

    ``slabs > 1`` cuts a mesh into that many slabs along its longest axis and bounds each slab
    separately. One axis-aligned box around a bent 750 mm link covers a lot of empty space, and
    that emptiness is what the reward pays for: the bound is deliberately conservative, but a
    bound that is 100 mm loose turns a 25 mm margin into whitespace and the policy learns to
    avoid poses that are fine.
    """
    target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_name)
    by_body = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in bodies}
    out = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] not in by_body or model.geom_contype[geom] == 0:
            continue
        owner = model.geom_bodyid[geom]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        rg = data.geom_xmat[geom].reshape(3, 3)
        rt = data.xmat[target].reshape(3, 3)
        rel_rot = rt.T @ rg
        rel_pos = rt.T @ (data.geom_xpos[geom] - data.xpos[target])
        corners = geom_corners_in(model, data, geom, target)
        # a rotated primitive keeps its own box (an axis-aligned bound of a thin box that is
        # mounted at 45 degrees would be most of a cube); a mesh has no frame worth keeping
        rotated = model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX and not np.allclose(
            rel_rot, np.eye(3), atol=1e-6
        )
        if rotated:
            out.append(
                dict(
                    name=name,
                    body=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, owner),
                    centre=rel_pos.tolist(),
                    half=model.geom_size[geom].tolist(),
                    quat=quat_from_matrix(rel_rot).tolist(),
                )
            )
            continue
        if slabs <= 1 or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            centre = (corners.max(0) + corners.min(0)) / 2
            half = (corners.max(0) - corners.min(0)) / 2
            out.append(
                dict(
                    name=name,
                    body=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, owner),
                    centre=centre.tolist(),
                    half=half.tolist(),
                    quat=None,
                )
            )
            continue
        axis = int(np.argmax(corners.max(0) - corners.min(0)))
        order = np.argsort(corners[:, axis])
        for index, chunk in enumerate(np.array_split(corners[order], slabs)):
            centre = (chunk.max(0) + chunk.min(0)) / 2
            half = (chunk.max(0) - chunk.min(0)) / 2
            out.append(
                dict(
                    name=f"{name}_s{index}",
                    body=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, owner),
                    centre=centre.tolist(),
                    half=half.tolist(),
                    quat=None,
                )
            )
    return out


def quat_from_matrix(r: np.ndarray) -> np.ndarray:
    """(w, x, y, z) quaternion of a rotation matrix."""
    tr = np.trace(r)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    return np.array([w, x, y, z])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default="", help="write the table as JSON to this path")
    ap.add_argument("--decimals", type=int, default=5)
    ap.add_argument("--slabs", type=int, default=2, help="slabs per mesh along its longest axis")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    table = {name: parts_for(model, data, frame, bodies, args.slabs) for name, (frame, bodies) in GROUPS.items()}
    d = args.decimals
    for group, parts in table.items():
        print(f'    "{group}": [')
        for p in parts:
            centre = ", ".join(f"{v:.{d}f}" for v in p["centre"])
            half = ", ".join(f"{v:.{d}f}" for v in p["half"])
            quat = "None" if p["quat"] is None else "(" + ", ".join(f"{v:.6f}" for v in p["quat"]) + ")"
            print(f'        ("{p["name"]}", ({centre}), ({half}), {quat}),')
        print("    ],")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(table, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
