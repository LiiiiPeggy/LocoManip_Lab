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

"""A distance proxy built from the frozen PhysX collider shapes.

This module deliberately separates three things that were previously mixed:

* the **actual** collider shapes in the frozen flat USD (cube with scale, or a mesh whose
  ``physics:approximation`` is ``convexHull``),
* an exact convex-hull **reference** query on those shapes (``convex_distance.gjk_distance``),
* a MuJoCo **proxy model** with the same shapes (``mj_geomDistance``), used for checking the
  representation and for the union-min aggregation.

The physical USD, URDF and default ``safety.py`` are never written. The proxy is offline only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
COLLIDER_JSON = ROOT / "outputs/stage413/b/simple_collision_v2/collider_geometry_v2.json"
V2_SCENE = ROOT / "mujoco/robots/ranger_cr10/scene_simple_v2.xml"
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes"
TOL = 1e-4
MARGIN = 0.02
CAP = 0.05

ARM_GROUPS = ("cr10_Link1", "cr10_Link2", "cr10_Link3", "cr10_Link4", "cr10_Link5", "cr10_Link6")
ADJACENT = (
    ("cr10_Link1", "cr10_Link2"),
    ("cr10_Link2", "cr10_Link3"),
    ("cr10_Link3", "cr10_Link4"),
    ("cr10_Link4", "cr10_Link5"),
    ("cr10_Link5", "cr10_Link6"),
)
PROTECTED_PARTS = ("chassis", "box_left", "box_top", "cr10_base", "lidar", "d435")
# Collider leaf name -> logical protected part. The names are the ones in the flat USD's
# body-hierarchy collision copy (mesh_1/mesh_2/mesh_5 are the two box_link colliders and the
# D435 box after fixed-joint merging).
BASE_PART = {
    "ranger_base_link": "chassis",
    "mesh_1": "box_left",
    "mesh_2": "box_top",
    "cr10_base_link": "cr10_base",
    "lidar_link0": "lidar",
    "mesh_5": "d435",
}
MESH_FILES = {
    "ranger_base_link": "ranger_meshes/ranger_base_link.STL",
    "cr10_base_link": "cr10_meshes/cr10_base_link.STL",
    "lidar_link0": "lidar_link0.STL",
    "ag95_base_link": "ag95_meshes/ag95_base_link.STL",
    "crank_Link": "ag95_meshes/crank_Link.STL",
    "rod_Link": "ag95_meshes/rod_Link.STL",
    "proximal_phalanx_Link": "ag95_meshes/proximal_phalanx_Link.STL",
    "distal_phalanx_Link": "ag95_meshes/distal_phalanx_Link.STL",
    "fl_steering_wheel_link": "ranger_meshes/fl_steering_wheel_link.STL",
    "fl_wheel_link": "ranger_meshes/fl_wheel_link.STL",
    "fr_steering_link": "ranger_meshes/fr_steering_link.STL",
    "fr_wheel_link": "ranger_meshes/fr_wheel_link.STL",
    "rl_steering_wheel_link": "ranger_meshes/rl_steering_wheel_link.STL",
    "rl_wheel_link": "ranger_meshes/rl_wheel_link.STL",
    "rr_steering_wheel_link": "ranger_meshes/rr_steering_wheel_link.STL",
    "rr_wheel_link": "ranger_meshes/rr_wheel_link.STL",
}


@dataclass
class Shape:
    name: str
    body: str
    group: str
    part: str
    kind: str
    points: np.ndarray
    pos: np.ndarray
    quat: np.ndarray
    half: np.ndarray | None
    mesh_file: str | None
    mesh_scale: float | None
    source_path: str
    approximation: str | None
    supported: bool = True
    note: str = ""


def _quat_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """(w, x, y, z) from a proper rotation matrix."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _decompose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (scale_x, scale_y, scale_z), rotation, translation of a scale*rotation matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)
    linear = matrix[:3, :3]
    scales = np.linalg.norm(linear, axis=0)
    rotation = linear / np.maximum(scales, 1e-15)
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return scales, rotation, matrix[:3, 3]


def _mesh_name(path: str) -> str:
    parts = [part for part in path.split("/") if part]
    for part in reversed(parts):
        if part in MESH_FILES:
            return part
    raise KeyError(f"no mesh file mapping for {path}")


def _welded_mesh(path: Path):
    mesh = trimesh.load(str(path), force="mesh", process=False)
    mesh.merge_vertices(merge_tex=False, merge_norm=False)
    mesh.remove_unreferenced_vertices()
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def _body_ids(model):
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)}


def _zero_data(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for joint in range(model.njnt):
        if (
            model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE
            or model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_SLIDE
        ):
            data.qpos[model.jnt_qposadr[joint]] = 0.0
    mujoco.mj_forward(model, data)
    return data


def _affine(matrix) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    out = np.eye(4)
    out[:3, :3] = value[:3, :3]
    out[:3, 3] = value[3, :3]
    return out


def load_shapes(path: Path = COLLIDER_JSON) -> list[Shape]:
    """Load the actual collider shapes and place them in the MuJoCo link frames."""
    import json

    info = json.loads(Path(path).read_text())
    model = mujoco.MjModel.from_xml_path(str(V2_SCENE))
    bodies = _body_ids(model)

    shapes: list[Shape] = []
    for collider in info["colliders"]:
        body = collider["body"]
        if body not in bodies:
            continue
        target = body
        if body in ("box_link", "cr10_base_link", "lidar_link0", "d435_link"):
            target = "base_link"
        # The `collisions` scope is authored in the URDF link frame, which is exactly the
        # MuJoCo body frame used by the v2 model. Do not compose the USD rigid-body frame:
        # that frame is a different (but equivalent) convention and would rotate the shape.
        local = _affine(collider["link_local"])
        leaf = collider["path"].split("/")[-2]
        if collider["kind"] == "cube":
            scales, rotation, translation = _decompose(local)
            half = scales / 2.0
            signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
            points = (signs * half) @ rotation.T + translation
            part = BASE_PART.get(leaf, leaf) if body == "base_link" else leaf
            if body == "cr10_Link6":
                part = {"mesh_0": "link6_main", "mesh_2": "d455"}.get(leaf, leaf)
                group = "cr10_Link6"
            elif body.startswith("gripper_finger"):
                group = "cr10_Link6"
            elif body == "base_link":
                group = part
            else:
                group = body
            shapes.append(Shape(
                name=f"phys_{body}_{leaf}", body=target, group=group, part=part, kind="box",
                points=points, pos=translation, quat=_quat_from_matrix(rotation), half=half,
                mesh_file=None, mesh_scale=None, source_path=collider["path"],
                approximation=None,
            ))
        elif collider["kind"] == "mesh":
            if body == "cr10_Link6" and leaf == "ag95_base_link":
                part = "gripper_base"
            elif body == "base_link":
                part = BASE_PART[leaf]
            else:
                part = leaf
            if body == "base_link":
                group = part
            elif body == "cr10_Link6" or body.startswith("gripper_finger"):
                group = "cr10_Link6"
            else:
                group = body
            mesh_name = _mesh_name(collider["path"])
            local_mesh = _affine(collider["mesh_link_local"])
            scales, rotation, translation = _decompose(local_mesh)
            scale = float(np.mean(scales))
            uniform = float(np.max(np.abs(scales - scale)))
            vertices, faces = _welded_mesh(MESH_DIR / MESH_FILES[mesh_name])
            transformed = (vertices * scale) @ rotation.T + translation
            hull = trimesh.Trimesh(vertices=transformed, faces=faces, process=False).convex_hull.vertices
            shapes.append(Shape(
                name=f"phys_{body}_{leaf}", body=target, group=group, part=part, kind="mesh",
                points=np.asarray(hull, dtype=np.float64), pos=translation,
                quat=_quat_from_matrix(rotation), half=None,
                mesh_file=MESH_FILES[mesh_name], mesh_scale=scale, source_path=collider["path"],
                approximation=collider.get("approximation"),
                supported=uniform < 1e-9,
                note="" if uniform < 1e-9 else f"non_uniform_scale={scales.tolist()}",
            ))
        else:
            shapes.append(Shape(
                name=f"phys_{body}_{leaf}", body=target, group=body, part=leaf, kind="unsupported",
                points=np.zeros((0, 3)), pos=np.zeros(3), quat=np.array([1.0, 0.0, 0.0, 0.0]),
                half=None, mesh_file=None, mesh_scale=None, source_path=collider["path"],
                approximation=None, supported=False, note="unsupported_collider",
            ))
    return shapes


def active_pairs() -> list[tuple[str, str]]:
    """Logical pairs that PhysX actually checks in the relevant internal scope.

    Adjacent arm pairs and the Link6-to-finger assembly are filtered by the frozen body-level
    filters. Wheel/steering pairs are outside the internal-arm-obstacle scope and are listed
    separately, not silently dropped.
    """
    pairs: list[tuple[str, str]] = []
    for index, first in enumerate(ARM_GROUPS):
        for second in ARM_GROUPS[index + 1 :]:
            if (first, second) in ADJACENT or (second, first) in ADJACENT:
                continue
            pairs.append((first, second))
    for arm in ARM_GROUPS:
        for part in PROTECTED_PARTS:
            # Link1-to-chassis/cr10_base is the joint1 mount. The old collision_parts.STRUCTURAL
            # already classified Link1-vs-base_link as checked-but-not-rewarded; the split
            # protected parts must not turn that structural pair back into a reward alarm.
            if arm == "cr10_Link1" and part in ("chassis", "cr10_base"):
                continue
            pairs.append((arm, part))
    return pairs


def excluded_pairs() -> list[dict]:
    """Pairs excluded for a documented reason, kept visible for the report."""
    out = []
    for first, second in ADJACENT:
        out.append({"pair": [first, second], "reason": "adjacent arm joint (body filter)"})
    out.append({"pair": ["cr10_Link6", "gripper_fingers"], "reason": "Link6-to-finger assembly (body filter)"})
    out.append({"pair": ["cr10_Link1", "chassis"], "reason": "joint1 mount structural pair; old "
                "collision_parts.STRUCTURAL, checked but not rewarded"})
    out.append({"pair": ["cr10_Link1", "cr10_base"], "reason": "joint1 mount structural pair; base_link "
                "merge leaves it active in PhysX, but it is the structural connection"})
    for wheel in ("fl_steering_wheel_link", "fl_wheel_link", "fr_steering_link", "fr_wheel_link",
                  "rl_steering_wheel_link", "rl_wheel_link", "rr_steering_wheel_link", "rr_wheel_link"):
        out.append({"pair": ["base_link", wheel], "reason": "base-to-wheel/steering assembly (body filter)"})
        out.append({"pair": ["arm", wheel], "reason": "wheel/ground scope, not an internal arm obstacle"})
    return out


def shapes_by_group(shapes: list[Shape]) -> dict[str, list[Shape]]:
    out: dict[str, list[Shape]] = {}
    for shape in shapes:
        if shape.supported:
            out.setdefault(shape.group, []).append(shape)
    return out


def world_points(shape: Shape, data) -> np.ndarray:
    body_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, shape.body)
    rotation = np.asarray(data.xmat[body_id]).reshape(3, 3)
    position = np.asarray(data.xpos[body_id])
    return shape.points @ rotation.T + position


def reference_group_distance(shapes: dict[str, list[Shape]], first: str, second: str, data,
                             tol: float = 1e-8) -> dict:
    """Exact GJK union-min for a logical pair, with the winning atomic pair kept."""
    from convex_distance import gjk_distance

    best = {"distance": float("inf"), "status": "unknown", "shape_a": None, "shape_b": None}
    for shape_a in shapes.get(first, []):
        points_a = world_points(shape_a, data)
        for shape_b in shapes.get(second, []):
            points_b = world_points(shape_b, data)
            result = gjk_distance(points_a, points_b, tol=tol)
            if result.status == "intersecting":
                distance = 0.0
                status = "contact"
            elif result.status == "separated" and np.isfinite(result.distance):
                distance = float(result.distance)
                status = "separated"
            else:
                distance = float("nan")
                status = result.status
            if status == "contact":
                return {"distance": 0.0, "status": "contact", "shape_a": shape_a.name,
                        "shape_b": shape_b.name}
            if np.isfinite(distance) and distance < best["distance"]:
                best = {"distance": distance, "status": status, "shape_a": shape_a.name,
                        "shape_b": shape_b.name}
    if not np.isfinite(best["distance"]):
        best["status"] = "unknown"
    return best


def proxy_group_distance(model, data, shapes: dict[str, list[Shape]], first: str, second: str,
                         cap: float = CAP) -> dict:
    """MuJoCo union-min for the same logical pair, with the winning geom pair kept."""
    best = {"distance": float("inf"), "signed": float("inf"), "shape_a": None, "shape_b": None}
    for shape_a in shapes.get(first, []):
        geom_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, shape_a.name)
        if geom_a < 0:
            return {"distance": float("nan"), "signed": float("nan"), "shape_a": shape_a.name,
                    "shape_b": None, "status": "missing_geom"}
        for shape_b in shapes.get(second, []):
            geom_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, shape_b.name)
            if geom_b < 0:
                return {"distance": float("nan"), "signed": float("nan"), "shape_a": shape_a.name,
                        "shape_b": shape_b.name, "status": "missing_geom"}
            signed = float(mujoco.mj_geomDistance(model, data, geom_a, geom_b, cap, None))
            distance = max(signed, 0.0)
            if distance < best["distance"]:
                best = {"distance": distance, "signed": signed, "shape_a": shape_a.name,
                        "shape_b": shape_b.name}
    if not np.isfinite(best["distance"]):
        best["status"] = "unknown"
    else:
        best["status"] = "contact" if best["signed"] <= 0.0 else "separated"
    return best


def label_from_distance(distance: float, threshold: float, status: str = "separated") -> str:
    if status in ("contact", "intersecting"):
        return "unsafe"
    if not np.isfinite(distance):
        return "unknown"
    return "unsafe" if distance <= max(threshold, TOL) else "safe"


STABLE_SHAPES = ROOT / "mujoco/robots/ranger_cr10/phys_proxy_shapes.json"


def export_shapes(path: Path = STABLE_SHAPES) -> list[Shape]:
    """Write the computed shapes to a small self-contained file for the runtime reward."""
    shapes = load_shapes()
    payload = {
        "source": "scripts/ranger_cr10/phys_proxy.py:load_shapes; actual flat-USD colliders",
        "shapes": [
            {
                "name": shape.name, "body": shape.body, "group": shape.group, "part": shape.part,
                "kind": shape.kind, "points": shape.points.tolist(), "pos": shape.pos.tolist(),
                "quat": shape.quat.tolist(),
                "half": None if shape.half is None else shape.half.tolist(),
                "mesh_file": shape.mesh_file, "mesh_scale": shape.mesh_scale,
                "source_path": shape.source_path, "approximation": shape.approximation,
                "supported": bool(shape.supported), "note": shape.note,
            }
            for shape in shapes
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(__import__("json").dumps(payload, indent=1))
    return shapes


def load_shapes_stable(path: Path = STABLE_SHAPES) -> list[Shape]:
    """Load the exported shapes without MuJoCo/trimesh/STL work (used by the reward)."""
    import json

    payload = json.loads(Path(path).read_text())
    shapes = []
    for item in payload["shapes"]:
        shapes.append(Shape(
            name=item["name"], body=item["body"], group=item["group"], part=item["part"],
            kind=item["kind"], points=np.asarray(item["points"], dtype=np.float64),
            pos=np.asarray(item["pos"], dtype=np.float64), quat=np.asarray(item["quat"], dtype=np.float64),
            half=None if item["half"] is None else np.asarray(item["half"], dtype=np.float64),
            mesh_file=item["mesh_file"], mesh_scale=item["mesh_scale"],
            source_path=item["source_path"], approximation=item["approximation"],
            supported=bool(item["supported"]), note=item.get("note", ""),
        ))
    return shapes
