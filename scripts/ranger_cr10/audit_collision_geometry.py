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

"""Stage 4.13a0: what geometry is the reference, and are the three sources the same thing?

There are three descriptions of the robot's collision geometry in this repository and they are
*not* interchangeable:

* the **MuJoCo collision model** in ``mujoco/robots/ranger_cr10/ranger_cr10.xml`` -- convex
  hulls for the arm links, analytic primitives for the chassis, the cameras and the gripper jaws;
* the **URDF trimeshes** the hulls were built from, which the meshes themselves are not convex
  approximations of;
* any **new convex set** a later stage generates.

A distance prototype can only be judged against one of them at a time. This script writes down
which body owns what, in which frame, with which units and scale, how many vertices each hull
has, and what the model's query settings are -- so the next scripts can say "reference = MuJoCo
hull distance" and mean it. It also hashes the artefacts the later measurements replay.

Usage:
    python scripts/ranger_cr10/audit_collision_geometry.py --out outputs/stage413/a0
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "mujoco/robots/ranger_cr10/scene.xml"
XML = ROOT / "mujoco/robots/ranger_cr10/ranger_cr10.xml"
URDF = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10/ranger_cr10_converted.urdf"
POSES = ROOT / "outputs/stage411/operating_poses.npz"
CHECKPOINT = ROOT / "logs/rsl_rl/ranger_cr10_teleop/2026-11-04_06-57-27_stage410f_ppo/model_9999.pt"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out", default="outputs/stage413/a0")
args = ap.parse_args()

import mujoco  # noqa: E402

# Isaac bodies after `merge_fixed_joints=True`; the groups are the ones `mdp/safety.py` uses.
ISAAC_GROUPS = {
    "cr10_Link1": ["cr10_Link1"],
    "cr10_Link2": ["cr10_Link2"],
    "cr10_Link3": ["cr10_Link3"],
    "cr10_Link4": ["cr10_Link4"],
    "cr10_Link5": ["cr10_Link5"],
    "cr10_Link6": ["cr10_Link6", "gripper_base_link", "d455_link"],
    "base_link": ["base_link"],
    "lidar_link0": ["lidar_link0"],
    "d435_link": ["d435_link"],
}
# MuJoCo excludes every parent/child pair; these are the ones a distance query must also skip.
ADJACENT = [
    ("cr10_Link1", "cr10_Link2"),
    ("cr10_Link2", "cr10_Link3"),
    ("cr10_Link3", "cr10_Link4"),
    ("cr10_Link4", "cr10_Link5"),
    ("cr10_Link5", "cr10_Link6"),
]
# The structural pair: real, reported, but no action changes it (see mdp/safety.py).
STRUCTURAL = [("cr10_Link1", "base_link")]
PRIORITY = [
    ("cr10_Link3", "cr10_Link5"),
    ("cr10_Link6", "lidar_link0"),
    ("cr10_Link5", "lidar_link0"),
    ("cr10_Link4", "cr10_Link6"),
    ("cr10_Link4", "lidar_link0"),
    ("cr10_Link3", "lidar_link0"),
]


def sha256(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    print(f"MuJoCo {mujoco.__version__}")
    print(
        f"  option: margin {model.opt.o_margin:.4f}  tolerance {model.opt.tolerance:.2e}  "
        f"cone {int(model.opt.cone)}  jacobian {int(model.opt.jacobian)}  iterations {model.opt.iterations}  "
        f"ccd_iterations {model.opt.ccd_iterations}  ccd_tolerance {model.opt.ccd_tolerance:.1e}"
    )
    enable = int(model.opt.enableflags)
    active = [n for n in dir(mujoco.mjtEnableBit) if n.startswith("mjENBL") and enable & int(getattr(mujoco.mjtEnableBit, n))]
    print(f"  enableflags {enable} ({', '.join(active) if active else 'none set'}), disableflags {int(model.opt.disableflags)}")
    print("  note: mj_geomDistance is a direct query; it does not depend on the integrator's CCD path")
    print(f"  model: nbody {model.nbody}  ngeom {model.ngeom}  nexclude {model.nexclude}")
    margins = sorted({round(float(model.geom_margin[g]), 6) for g in range(model.ngeom)})
    print(f"  geom margins in use: {margins}")
    print("  distance query: mj_geomDistance(distmax=cap) - the cap bounds the query, not the geometry")

    groups = {}
    for group, bodies in ISAAC_GROUPS.items():
        entries = []
        for body in bodies:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            if bid < 0:
                entries.append(dict(body=body, missing=True))
                continue
            for geom in range(model.ngeom):
                if model.geom_bodyid[geom] != bid or model.geom_contype[geom] == 0:
                    continue
                kind = model.geom_type[geom]
                entry = dict(
                    isaac_group=group,
                    mujoco_body=body,
                    geom=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom),
                    type=mujoco.mjtGeom(kind).name,
                    size=np.round(model.geom_size[geom], 6).tolist(),
                    pos=np.round(model.geom_pos[geom], 6).tolist(),
                    quat=np.round(model.geom_quat[geom], 6).tolist(),
                    margin=float(model.geom_margin[geom]),
                )
                if kind == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = model.geom_dataid[geom]
                    entry["mesh"] = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
                    entry["mesh_file"] = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
                    entry["n_vertices"] = int(model.mesh_vertnum[mid])
                    entry["n_faces"] = int(model.mesh_facenum[mid])
                    entry["mesh_scale"] = np.round(model.mesh_scale[mid], 6).tolist()
                entries.append(entry)
        groups[group] = entries

    print("\nIsaac groups (after merge_fixed_joints) and their MuJoCo collision geoms:")
    for group, entries in groups.items():
        print(f"  {group}")
        for e in entries:
            if e.get("missing"):
                print(f"    body {e['body']} NOT FOUND in the MuJoCo model")
                continue
            extra = f"  {e.get('n_vertices', '')} verts" if "n_vertices" in e else ""
            print(
                f"    {e['mujoco_body']:22s} {e['geom']:24s} {e['type']:8s} "
                f"size {np.round(e['size'], 4)}{extra}"
            )

    pairs = []
    seen = set()
    for i, a in enumerate(ISAAC_GROUPS):
        for b in list(ISAAC_GROUPS)[i + 1 :]:
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            structural = key in {tuple(sorted(p)) for p in STRUCTURAL}
            adjacent = key in {tuple(sorted(p)) for p in ADJACENT}
            pairs.append(
                dict(
                    pair=" ".join(key),
                    id=f"{key[0]}::{key[1]}",
                    checked=not adjacent,
                    in_reward=not adjacent and not structural,
                    reason="parent/child (MuJoCo excludes it)" if adjacent else ("structural" if structural else ""),
                )
            )
    print(f"\ncollision pairs: {len(pairs)} total, {sum(p['checked'] for p in pairs)} checked, "
          f"{sum(p['in_reward'] for p in pairs)} in the reward")
    for p in pairs:
        if not p["checked"]:
            print(f"  not checked: {p['pair']}  ({p['reason']})")

    artefacts = {
        "poses_operating": sha256(POSES),
        "scene_xml": sha256(SCENE),
        "model_xml": sha256(XML),
        "urdf": sha256(URDF),
        "policy_model_9999": sha256(CHECKPOINT),
    }
    print("\nartefacts (sha256):")
    for name, digest in artefacts.items():
        print(f"  {name:22s} {digest[:32]}{'...' if digest != 'MISSING' else ''}")

    payload = dict(
        mujoco_version=mujoco.__version__,
        options=dict(
            margin=float(model.opt.o_margin),
            tolerance=float(model.opt.tolerance),
            iterations=int(model.opt.iterations),
            enableflags=int(model.opt.enableflags),
            disableflags=int(model.opt.disableflags),
            geom_margins=margins,
        ),
        groups=groups,
        pairs=pairs,
        priority_pairs=[" ".join(p) for p in PRIORITY],
        artefacts=artefacts,
    )
    (out / "geometry_audit.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out / 'geometry_audit.json'}")


if __name__ == "__main__":
    main()
