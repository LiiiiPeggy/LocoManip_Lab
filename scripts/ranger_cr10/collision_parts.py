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

"""Stage 4.13b0: one list of parts, so the three distances describe the same objects.

The earlier comparison was not like for like. `cr10_Link6` in Isaac is one rigid body made of the
link mesh, the AG95 palm, two jaw proxies and the D455; the reference script loaded only Link6.STL
and took ``next(geom)`` -- one arbitrary geom -- to stand for the whole group. A distance from the
wrist mesh says nothing about a jaw that is 5 mm from the lidar, and using it to declare an alarm
"false" is the mistake this file removes.

Each group is therefore a list of constituents, each with its own source (STL or analytic box),
its frame, and the MuJoCo geoms that represent it. The reference, the MuJoCo query and the box
proxy all iterate the same constituents and the same pair list.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MESH_DIR_REL = "rangerboxcr10lidar_description/meshes"


@dataclass(frozen=True)
class Constituent:
    group: str
    name: str
    kind: str  # "mesh" or "box"
    stl: str = ""
    scale: float = 1.0
    centre: tuple = (0.0, 0.0, 0.0)  # link frame, for boxes
    half: tuple = (0.0, 0.0, 0.0)
    quat: tuple | None = None  # local rotation, (w, x, y, z)
    mujoco_geoms: tuple = ()  # geoms that represent it in the MuJoCo model
    box_proxy_body: str = ""  # Isaac body the box proxy attaches this part to


CONSTITUENTS: tuple[Constituent, ...] = (
    Constituent("cr10_Link1", "Link1", "mesh", "cr10_meshes/Link1.STL", 1.0, mujoco_geoms=("cr10_Link1_col",), box_proxy_body="cr10_Link1"),
    Constituent("cr10_Link2", "Link2", "mesh", "cr10_meshes/Link2.STL", 1.0, mujoco_geoms=("cr10_Link2_col",), box_proxy_body="cr10_Link2"),
    Constituent("cr10_Link3", "Link3", "mesh", "cr10_meshes/Link3.STL", 1.0, mujoco_geoms=("cr10_Link3_col",), box_proxy_body="cr10_Link3"),
    Constituent("cr10_Link4", "Link4", "mesh", "cr10_meshes/Link4.STL", 1.0, mujoco_geoms=("cr10_Link4_col",), box_proxy_body="cr10_Link4"),
    Constituent("cr10_Link5", "Link5", "mesh", "cr10_meshes/Link5.STL", 1.0, mujoco_geoms=("cr10_Link5_col",), box_proxy_body="cr10_Link5"),
    Constituent("cr10_Link6", "Link6", "mesh", "cr10_meshes/Link6.STL", 1.0, mujoco_geoms=("cr10_Link6_col",), box_proxy_body="cr10_Link6"),
    Constituent(
        "cr10_Link6",
        "gripper_base",
        "mesh",
        "ag95_meshes/ag95_base_link.STL",
        0.001,
        mujoco_geoms=("gripper_base_link_col0",),
        box_proxy_body="cr10_Link6",
    ),
    Constituent(
        "cr10_Link6",
        "jaw_l",
        "box",
        centre=(-0.0495, 0.0, 0.129),
        half=(0.0235, 0.026, 0.056),
        mujoco_geoms=("gripper_base_link_col1",),
        box_proxy_body="cr10_Link6",
    ),
    Constituent(
        "cr10_Link6",
        "jaw_r",
        "box",
        centre=(0.0495, 0.0, 0.129),
        half=(0.0235, 0.026, 0.056),
        mujoco_geoms=("gripper_base_link_col2",),
        box_proxy_body="cr10_Link6",
    ),
    Constituent(
        "cr10_Link6",
        "d455",
        "box",
        centre=(-0.0475, 0.1, 0.13155),
        half=(0.013, 0.062, 0.0145),
        quat=(0.5, -0.5, -0.5, -0.5),
        mujoco_geoms=("d455_link_col",),
        box_proxy_body="cr10_Link6",
    ),
    Constituent("lidar_link0", "lidar", "mesh", "lidar_link0.STL", 1.0, mujoco_geoms=("lidar_link0_col",), box_proxy_body="lidar_link0"),
    Constituent(
        "d435_link",
        "d435",
        "box",
        centre=(0.0, -0.0175, 0.0),
        half=(0.0125, 0.045, 0.0125),
        mujoco_geoms=("d435_link_col",),
        box_proxy_body="d435_link",
    ),
    Constituent(
        "base_link",
        "chassis",
        "box",
        centre=(0.0, -0.0033, -0.0571),
        half=(0.651, 0.4347, 0.21755),
        mujoco_geoms=("base_link_col",),
        box_proxy_body="base_link",
    ),
)

GROUPS = tuple(dict.fromkeys(c.group for c in CONSTITUENTS))

# Parent/child pairs MuJoCo excludes; kept identical for all three methods.
ADJACENT = (
    ("cr10_Link1", "cr10_Link2"),
    ("cr10_Link2", "cr10_Link3"),
    ("cr10_Link3", "cr10_Link4"),
    ("cr10_Link4", "cr10_Link5"),
    ("cr10_Link5", "cr10_Link6"),
)
# Checked and reported, but no action changes it, so it is not rewarded.
STRUCTURAL = (("cr10_Link1", "base_link"),)

PRIORITY = (
    ("cr10_Link3", "cr10_Link5"),
    # The old box-proxy alarm scan found the dominant lidar alarms on Link2 and Link3, not on the
    # Link6 group; both pairs must be in the same constituent list for the radar attribution.
    ("cr10_Link1", "lidar_link0"),
    ("cr10_Link2", "lidar_link0"),
    ("cr10_Link6", "lidar_link0"),
    ("cr10_Link5", "lidar_link0"),
    ("cr10_Link4", "cr10_Link6"),
    ("cr10_Link4", "lidar_link0"),
    ("cr10_Link3", "lidar_link0"),
)


def constituents_of(group: str) -> list[Constituent]:
    return [c for c in CONSTITUENTS if c.group == group]


def box_mesh(centre, half, quat=None) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and faces of an oriented box, in the frame the constituent is declared in."""
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    vertices = signs * np.asarray(half, dtype=float)
    if quat is not None:
        w, x, y, z = quat
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        vertices = vertices @ rotation.T
    vertices = vertices + np.asarray(centre, dtype=float)
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def pairs() -> list[tuple[str, str]]:
    """Every group pair that is checked, with the exclusion rules applied once, here."""
    out = []
    for i, a in enumerate(GROUPS):
        for b in GROUPS[i + 1 :]:
            key = tuple(sorted((a, b)))
            if key in {tuple(sorted(p)) for p in ADJACENT}:
                continue
            out.append((a, b))
    return out
