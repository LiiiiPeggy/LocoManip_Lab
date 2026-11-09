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

"""Dump the actual imported USD collider shapes in each rigid body's frame.

This is the traceable source for the PhysX-consistent distance proxy. It records, for every
``UsdPhysics.CollisionAPI`` prim in the flat candidate:

* the owning rigid body (after fixed-joint merging),
* the collider type (Cube or a Mesh below an Xform),
* the local transform relative to the body,
* the cube size, or the mesh point bounds and PhysX approximation,
* contact/rest offset and collision-enabled attributes,
* the body-level filtered pairs and whether the target resolves.

The output is JSON only. It does not modify the USD.

Run:
    python scripts/ranger_cr10/inspect_collider_geometry.py \
        --usd source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10_simple_v2/ranger_cr10_simple_v2_flat.usd \
        --out outputs/stage413/b/simple_collision_v2/collider_geometry_v2.json
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument(
    "--usd",
    default="source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10_simple_v2/ranger_cr10_simple_v2_flat.usd",
)
ap.add_argument("--out", default="outputs/stage413/b/simple_collision_v2/collider_geometry_v2.json")
args = ap.parse_args()

app = AppLauncher(headless=True).app

import json  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402


def body_prim_of(prim):
    """Return the owning rigid body for a collision prim.

    The flat candidate keeps two copies of the collision hierarchy. The one PhysX uses is a
    referenced, instanceable ``collisions`` child under each articulation body; the separate
    ``/colliders`` scope is only a source copy and its local transforms are not the body-local
    transforms. Walking the body's instance proxies is therefore the traceable route.
    """
    current = prim
    while current and current.IsValid():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return current
        current = current.GetParent()
    return None


def gf_matrix(value):
    return [[float(value[r][c]) for c in range(4)] for r in range(4)]


def relative_matrix(collider, body):
    collider_world = UsdGeom.Xformable(collider).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    body_world = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return body_world.GetInverse() * collider_world


def link_local_matrix(prim):
    """Compose xformOps from the body's ``collisions`` scope down to ``prim``.

    The instance proxy's ``ComputeLocalToWorldTransform`` does not compose the owning body's
    transform, so the world relation cannot be used here. The ``collisions`` scope is authored
    in the URDF link frame, and every transform down to the shape is a plain local xform; the
    product is the shape pose in the link frame that PhysX consumes.
    """
    chain = []
    current = prim
    while current and current.IsValid() and current.GetName() != "collisions":
        chain.append(current)
        current = current.GetParent()
    if current is None or not current.IsValid():
        return None
    matrix = Gf.Matrix4d(1.0)
    for node in reversed(chain):
        matrix = matrix * UsdGeom.Xformable(node).GetLocalTransformation(Usd.TimeCode.Default())
    return matrix


def attr_value(prim, name):
    attr = prim.GetAttribute(name)
    if not attr or not attr.HasValue():
        return None
    value = attr.Get()
    if isinstance(value, Gf.Vec3f):
        return [float(x) for x in value]
    if isinstance(value, Gf.Vec3d):
        return [float(x) for x in value]
    if isinstance(value, Gf.Quatf):
        return [float(value.GetReal()), *[float(x) for x in value.GetImaginary()]]
    if isinstance(value, Gf.Quatd):
        return [float(value.GetReal()), *[float(x) for x in value.GetImaginary()]]
    if isinstance(value, (bool, int, float, str)):
        return value
    try:
        return [float(x) for x in value]
    except Exception:
        return str(value)


def mesh_points_bounds(mesh_prim):
    points = UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
    if not points:
        return None, 0
    array = np.asarray([[float(p[0]), float(p[1]), float(p[2])] for p in points], dtype=np.float64)
    return [[float(x) for x in array.min(axis=0)], [float(x) for x in array.max(axis=0)]], int(len(array))


stage = Usd.Stage.Open(args.usd, load=Usd.Stage.LoadAll)
if stage is None:
    raise SystemExit(f"could not open {args.usd}")

bodies = {}
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        body = {
            "name": prim.GetName(),
            "path": prim.GetPath().pathString,
            "type": prim.GetTypeName(),
            "local_to_world": gf_matrix(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())),
            "filtered_pairs_api": bool(prim.HasAPI(UsdPhysics.FilteredPairsAPI)),
            "targets": [],
            "broken_targets": [],
        }
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            rel = prim.GetRelationship("physics:filteredPairs")
            if rel:
                for target in rel.GetTargets():
                    target_path = str(target)
                    body["targets"].append(target_path)
                    target_prim = stage.GetPrimAtPath(target_path)
                    if not target_prim or not target_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        body["broken_targets"].append(target_path)
        bodies[body["path"]] = body

colliders = []
for body in [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]:
    # Instance proxies are where the referenced collision prims actually live; a plain
    # stage.Traverse() only sees the empty instanceable ``collisions`` Xform.
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        record = {
            "path": prim.GetPath().pathString,
            "body": body.GetName(),
            "body_path": body.GetPath().pathString,
            "type": prim.GetTypeName(),
            "collision_enabled": attr_value(prim, "physics:collisionEnabled"),
            "contact_offset": attr_value(prim, "physxCollision:contactOffset"),
            "rest_offset": attr_value(prim, "physxCollision:restOffset"),
            "link_local": gf_matrix(link_local_matrix(prim)),
            "local_to_body": gf_matrix(relative_matrix(prim, body)),
            "world_at_rest": gf_matrix(
                UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            ),
        }
        if prim.GetTypeName() == "Cube":
            size = attr_value(prim, "size")
            record.update(kind="cube", size=float(size) if size is not None else None,
                          half_extents=[float(size) / 2.0] * 3 if size is not None else None)
        else:
            mesh_prim = None
            for candidate in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                if candidate.GetTypeName() == "Mesh":
                    mesh_prim = candidate
                    break
            if mesh_prim is not None:
                bounds, count = mesh_points_bounds(mesh_prim)
                record.update(
                    kind="mesh",
                    mesh_path=mesh_prim.GetPath().pathString,
                    mesh_points_count=count,
                    mesh_local_bounds=bounds,
                    approximation=attr_value(mesh_prim, "physics:approximation")
                    or attr_value(prim, "physics:approximation"),
                    mesh_link_local=gf_matrix(link_local_matrix(mesh_prim)),
                    mesh_local_to_body=gf_matrix(relative_matrix(mesh_prim, body)),
                    mesh_world_at_rest=gf_matrix(
                        UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    ),
                )
            else:
                record.update(kind="unsupported", reason="no_mesh_descendant")
        colliders.append(record)

result = {
    "usd": os.path.abspath(args.usd),
    "n_bodies": len(bodies),
    "n_colliders": len(colliders),
    "bodies": bodies,
    "colliders": colliders,
    "note": "actual flat-candidate collider prims; Cube size is the full edge length; mesh bounds are in "
            "the mesh's local frame and the local_to_body matrix places that frame in the merged body frame. "
            "The mesh collision shape is PhysX's convex hull only when physics:approximation says convexHull.",
}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as handle:
    json.dump(result, handle, indent=1)
print(json.dumps({
    "usd": result["usd"],
    "n_bodies": result["n_bodies"],
    "n_colliders": result["n_colliders"],
    "by_body": {body["name"]: sum(1 for c in colliders if c["body"] == body["name"]) for body in bodies.values()},
    "kinds": {kind: sum(1 for c in colliders if c["kind"] == kind) for kind in ("cube", "mesh", "unsupported")},
    "out": args.out,
}, indent=1), flush=True)
with open(args.out) as handle:
    pass
app.close()
