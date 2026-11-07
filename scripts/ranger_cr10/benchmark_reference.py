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

"""Stage 4.13b0: what the exact reference costs, before and after the index work.

Two implementations are compared on the same poses:

* ``candidate``  -- the 4.13b0 initial path: a centroid KD-tree, a ball query at
  ``best + max_circumradius``, then scalar triangle-triangle distances (it is kept as the
  correctness baseline, not deleted);
* ``bvh``        -- the current path: one static bounding-sphere tree per mesh, a dual traversal that
  prunes on ``|cA - cB| - rA - rB > best``, and vectorised triangle pairs at the leaves.

The benchmark set is stratified on purpose -- far, near, contact and the six anomalous poses -- because
a mean over the operating set would be dominated by far pairs that both paths cull instantly.

Usage:
    python scripts/ranger_cr10/benchmark_reference.py --queries 60
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/ranger_cr10"))

from reference_geometry import MeshRef  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--queries", type=int, default=60)
ap.add_argument("--out", default="outputs/stage413/b0")
args = ap.parse_args()

SCENE = ROOT / "mujoco/robots/ranger_cr10/scene.xml"
POSES = ROOT / "outputs/stage411/operating_poses.npz"
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes/cr10_meshes"

import mujoco  # noqa: E402


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    bid = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)}
    adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)]
    poses = np.load(POSES)["poses"]

    t0 = time.time()
    mesh3 = trimesh.load(str(MESH_DIR / "Link3.STL"), force="mesh", process=False)
    mesh5 = trimesh.load(str(MESH_DIR / "Link5.STL"), force="mesh", process=False)
    load_time = time.time() - t0
    t0 = time.time()
    ref3 = MeshRef(np.asarray(mesh3.vertices), np.asarray(mesh3.faces))
    ref5 = MeshRef(np.asarray(mesh5.vertices), np.asarray(mesh5.faces))
    build_time = time.time() - t0
    print(f"init: load {load_time * 1000:.0f} ms, index build {build_time * 1000:.0f} ms "
          f"({len(ref3.triangles)}+{len(ref5.triangles)} triangles)")

    # stratify by MuJoCo's hull distance: far, near, contact, and the anomalous poses
    g3 = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == bid["cr10_Link3"] and model.geom_contype[g] != 0)
    g5 = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == bid["cr10_Link5"] and model.geom_contype[g] != 0)
    strata = {"far": [], "near": [], "contact": []}
    for index in range(0, len(poses), 10):
        q = poses[index]
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        for address, value in zip(adr, q):
            data.qpos[address] = value
        mujoco.mj_forward(model, data)
        distance = mujoco.mj_geomDistance(model, data, g3, g5, 0.2, None)
        if distance <= 0:
            strata["contact"].append(index)
        elif distance <= 0.02:
            strata["near"].append(index)
        else:
            strata["far"].append(index)

    per_stratum = max(1, args.queries // 3)
    sample = []
    for name, indices in strata.items():
        chosen = indices[:per_stratum]
        sample += [(name, index) for index in chosen]
    print(f"benchmark: {len(sample)} queries " + ", ".join(f"{k}={min(len(v), per_stratum)}" for k, v in strata.items()))

    timings = {"candidate": [], "bvh": []}
    statuses = {"candidate": {}, "bvh": {}}
    disagreements = 0
    for name, index in sample:
        q = poses[index]
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        for address, value in zip(adr, q):
            data.qpos[address] = value
        mujoco.mj_forward(model, data)
        r3, p3 = data.xmat[bid["cr10_Link3"]].reshape(3, 3), data.xpos[bid["cr10_Link3"]]
        r5, p5 = data.xmat[bid["cr10_Link5"]].reshape(3, 3), data.xpos[bid["cr10_Link5"]]
        rotation, translation = r3.T @ r5, r3.T @ (p5 - p3)

        t0 = time.time()
        moved5 = MeshRef(np.asarray(mesh5.vertices) @ rotation.T + translation, np.asarray(mesh5.faces))
        old = ref3.closest_to(moved5, max_clearance=0.05)
        timings["candidate"].append(time.time() - t0)
        statuses["candidate"][old.status] = statuses["candidate"].get(old.status, 0) + 1

        t0 = time.time()
        new = ref3.closest_to_in_frame(ref5, rotation, translation, max_clearance=0.05)
        timings["bvh"].append(time.time() - t0)
        statuses["bvh"][new.status] = statuses["bvh"].get(new.status, 0) + 1

        if (new.intersects != old.intersects) or (
            old.status == "exact" and new.status == "exact" and abs(new.clearance - old.clearance) > 1e-9
        ):
            disagreements += 1
            print(f"  DISAGREEMENT pose {index}: old {old.status} {old.clearance:.6f} {old.intersects} | "
                  f"new {new.status} {new.clearance:.6f} {new.intersects}")

    def report(name, values):
        values = np.array(values)
        print(f"  {name:10s} median {np.median(values) * 1000:8.1f} ms  p95 {np.percentile(values, 95) * 1000:8.1f} ms  "
              f"max {values.max() * 1000:8.1f} ms")
        return dict(median=float(np.median(values)), p95=float(np.percentile(values, 95)), max=float(values.max()))

    print("\nlatency per pair query")
    summary = {name: report(name, timings[name]) for name in timings}
    print(f"\nstatuses: {json.dumps(statuses)}")
    print(f"disagreements: {disagreements}")
    total = sum(timings["bvh"])
    print(f"\nbudget: {len(sample)} queries in {total:.1f} s -> "
          f"{len(poses) * 6 * (total / len(sample)) / 60:.0f} min for the full 1800 x 6 sweep at this cost "
          f"(every pair-step queried; only proxy alarms need it in practice)")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(dict(
        init=dict(load_ms=load_time * 1000, build_ms=build_time * 1000),
        queries=len(sample), latency=summary, statuses=statuses, disagreements=disagreements,
    ), indent=1))
    print(f"wrote {out / 'benchmark.json'}")


if __name__ == "__main__":
    main()
