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

"""Find v2 blocking and open-space targets offline in MuJoCo.

Targets are selected so that exactly one unfiltered pair is driven into contact:
  * box_left: an arm box vs the v2 left enclosure;
  * plate: an arm box vs the mounting plate;
  * arm: a non-adjacent arm pair (Link3 vs Link6);
  * over_box: a pose inside the *old* left box's upper slab but outside the new
    (lid-height) box, used to show the freed space is now traversable.

Every candidate is checked against all other unfiltered pairs, and a straight
joint-space path from home is sampled so the target pair is the first contact.

Run:
    python scripts/ranger_cr10/find_simple_blocking_targets_v2.py \
        --out outputs/stage413/b/simple_collision_v2/targets_v2.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "mujoco/robots/ranger_cr10/scene_simple.xml"
V2 = ROOT / "mujoco/robots/ranger_cr10/scene_simple_v2.xml"
POSES = ROOT / "outputs/stage411/operating_poses.npz"
OUT = ROOT / "outputs/stage413/b/simple_collision_v2/targets_v2.json"
HOME = np.array([0.236, 0.086, -2.398, -0.83, -1.335, 0.0])
LOWER = np.array([-3.92, -1.57, -2.86, -3.14, -3.14, -3.14])
UPPER = np.array([0.94, 1.57, 2.86, 3.14, 3.14, 3.14])

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--samples", type=int, default=20000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=str(OUT))
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

ARMS = [f"cr10_Link{i}" for i in range(1, 7)]
ADJACENT = {tuple(sorted((f"cr10_Link{i}", f"cr10_Link{i + 1}"))) for i in range(1, 6)}


def load(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i): i for i in range(model.ngeom)}
    addresses = [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")]
        for i in range(1, 7)
    ]
    return model, data, names, addresses


V1_MODEL, V1_DATA, V1_NAMES, V1_ADDR = load(V1)
V2_MODEL, V2_DATA, V2_NAMES, V2_ADDR = load(V2)
PROTECTED = ["base_link_col", "cr10_base_link_col", "lidar_link0_col", "d435_link_col"]
PAIRS = []
for index, a in enumerate(ARMS):
    for b in ARMS[index + 1 :]:
        if tuple(sorted((a, b))) not in ADJACENT:
            PAIRS.append((f"{a}_col_simple", f"{b}_col_simple"))
for a in ARMS:
    for protected in PROTECTED + ["box_link_col_left", "box_link_col_top"]:
        PAIRS.append((f"{a}_col_simple", protected))


def set_q(model, data, addresses, q):
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for address, value in zip(addresses, q):
        data.qpos[address] = float(value)
    mujoco.mj_forward(model, data)


def distances(model, data, names):
    out = {}
    for a, b in PAIRS:
        out[f"{a}|{b}"] = float(mujoco.mj_geomDistance(model, data, names[a], names[b], 1.0, None))
    return out


def path_check(q, target_pair):
    """Sample the joint-space path and return the first unfiltered contact per pair."""
    first = {}
    for alpha in np.linspace(0.0, 1.0, 40):
        qs = HOME * (1 - alpha) + q * alpha
        set_q(V2_MODEL, V2_DATA, V2_ADDR, qs)
        ds = distances(V2_MODEL, V2_DATA, V2_NAMES)
        for pair, distance in ds.items():
            if distance < 0.0 and pair not in first:
                first[pair] = {"alpha": float(alpha), "distance_m": distance}
    return first


def candidate_ok(ds, target):
    if ds[target] > -0.005:
        return False
    for pair, distance in ds.items():
        if pair == target:
            continue
        if distance < 0.015:
            return False
    return True


records = {"box_left": [], "top": [], "arm": [], "over_box": []}
poses = np.load(POSES)["poses"]
samples = np.vstack([poses, rng.uniform(LOWER, UPPER, size=(args.samples, 6))])
for q in samples:
    set_q(V2_MODEL, V2_DATA, V2_ADDR, q)
    v2 = distances(V2_MODEL, V2_DATA, V2_NAMES)
    set_q(V1_MODEL, V1_DATA, V1_ADDR, q)
    v1 = distances(V1_MODEL, V1_DATA, V1_NAMES)

    for arm in ARMS:
        target = f"{arm}_col_simple|box_link_col_left"
        if candidate_ok(v2, target) and len(records["box_left"]) < 40:
            records["box_left"].append({"q": q.tolist(), "pair": target, "distance_m": v2[target]})
        target = f"{arm}_col_simple|box_link_col_top"
        if candidate_ok(v2, target) and len(records["top"]) < 40:
            records["top"].append({"q": q.tolist(), "pair": target, "distance_m": v2[target]})
    for a, b in PAIRS:
        if a.startswith("cr10_Link") and b.startswith("cr10_Link"):
            target = f"{a}|{b}"
            if candidate_ok(v2, target) and len(records["arm"]) < 40:
                records["arm"].append({"q": q.tolist(), "pair": target, "distance_m": v2[target]})

    # freed-space candidate: inside the old left box, outside the new one.
    for arm in ARMS:
        target = f"{arm}_col_simple|box_link_col_left"
        if v1[target] < 0.0 and v2[target] > 0.03 and len(records["over_box"]) < 40:
            other = {k: v for k, v in v2.items() if k != target}
            if min(other.values()) > 0.02:
                records["over_box"].append(
                    {"q": q.tolist(), "pair": target, "v1_distance_m": v1[target], "v2_distance_m": v2[target]}
                )

# Keep the cleanest path target per case.
selected = {}
for case, rows in records.items():
    if not rows:
        continue
    rows = sorted(rows, key=lambda row: row["distance_m"] if case != "over_box" else -row["v2_distance_m"])
    chosen = None
    for row in rows[:20]:
        first = path_check(np.asarray(row["q"]), row["pair"])
        # The target pair must be the only pair that contacts, and no other pair may
        # contact first.
        contacts = {pair: value for pair, value in first.items() if pair == row["pair"] or value["alpha"] < first.get(row["pair"], {"alpha": 2.0})["alpha"]}
        if any(pair != row["pair"] for pair in first):
            if min(value["alpha"] for pair, value in first.items() if pair != row["pair"]) <= first.get(row["pair"], {"alpha": 2.0})["alpha"]:
                continue
        row = dict(row, path_first=first)
        chosen = row
        break
    selected[case] = chosen

result = {
    "home": HOME.tolist(),
    "v1_model": str(V1.relative_to(ROOT)),
    "v2_model": str(V2.relative_to(ROOT)),
    "candidates": {case: len(rows) for case, rows in records.items()},
    "selected": selected,
    "note": "offline MuJoCo target search; selected targets are checked on a 40-step joint-space path; "
            "adjacent pairs are filtered and excluded from the other-pair clearance check",
}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as handle:
    json.dump(result, handle, indent=1)
print(json.dumps({case: (None if row is None else {k: row[k] for k in row if k != "q"}) for case, row in selected.items()}, indent=1))
print(f"wrote {args.out}")
