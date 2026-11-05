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

"""Stage 4.12: per-pair error of the collision proxy on the poses the policy visits.

The 4.11 report could only say "46% of operating steps read as contact". That number is a
minimum over pairs, so it cannot say *which* pair is wrong or by how much. This compares each
pair on its own -- proxy distance against MuJoCo's hull distance -- over the 1800 configurations
recorded by `check_proxy_operating.py`, and reports the three numbers a geometry change has to
move: false alarms, misses, and the distance error inside the last 20 mm before contact.

Usage:
    python scripts/ranger_cr10/analyze_proxy_pairs.py
    python scripts/ranger_cr10/analyze_proxy_pairs.py --table outputs/stage411/collision_boxes_k6.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SAFETY = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/safety.py"
SCENE = ROOT / "mujoco/robots/ranger_cr10/scene.xml"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--table", default="outputs/stage411/collision_boxes.json")
ap.add_argument("--poses", default="outputs/stage411/operating_poses.npz")
ap.add_argument("--band", type=float, default=0.020)
ap.add_argument("--out", default="outputs/stage412")
ap.add_argument("--json", default="")
ap.add_argument("--all_pairs", action="store_true", help="report every pair, not just the watched three")
args = ap.parse_args()

import mujoco  # noqa: E402

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
bid = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)}
arm_adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)]

WATCH = [
    ("cr10_Link3", "cr10_Link5"),
    ("cr10_Link4", "cr10_Link6"),
    ("cr10_Link2", "lidar_link0"),
]

# MuJoCo bodies that make up each Isaac-side group
GROUPS = {
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
geoms = {
    n: [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid[n] and model.geom_contype[g] != 0]
    for group in GROUPS.values()
    for n in group
}


def load_safety():
    spec = importlib.util.spec_from_file_location("safety_pairs", SAFETY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install(module, table_path: Path) -> None:
    table = json.loads(table_path.read_text())
    module.ARM_PARTS = {
        k: [(e["name"], tuple(e["centre"]), tuple(e["half"]), tuple(e["quat"]) if e["quat"] else None) for e in v]
        for k, v in table.items()
        if k in module.ARM_PARTS
    }
    module.PROTECTED_PARTS = {
        k: (
            v[0]["body"],
            [(e["name"], tuple(e["centre"]), tuple(e["half"]), tuple(e["quat"]) if e["quat"] else None) for e in v],
        )
        for k, v in table.items()
        if k in module.PROTECTED_PARTS
    }
    module.ARM_BODY_LIST, module.ARM_CENTRES, module.ARM_HALVES, module.ARM_QUATS = module._flatten_arm()
    (
        module.PROT_GROUPS,
        module.PROT_BODY_LIST,
        module.PROT_CENTRES,
        module.PROT_HALVES,
        module.PROT_QUATS,
    ) = module._flatten_protected()
    module.SELF_PAIRS, module.SENSOR_PAIRS = module._pairs()
    module.PENALTY_COLUMNS = module._penalty_columns()


def mj_distance(group_a: str, group_b: str, cap: float = 0.25) -> float:
    best = cap
    for a in GROUPS[group_a]:
        for b in GROUPS[group_b]:
            if a == b:
                continue
            for g1 in geoms[a]:
                for g2 in geoms[b]:
                    best = min(best, mujoco.mj_geomDistance(model, data, g1, g2, best, None))
    return best


def pair_columns(module) -> dict[tuple[str, str], list[int]]:
    out: dict[tuple[str, str], list[int]] = {}
    for k, (i, j) in enumerate(module.SELF_PAIRS):
        out.setdefault(tuple(sorted((module.ARM_BODY_LIST[i], module.ARM_BODY_LIST[j]))), []).append(k)
    for k, (i, p) in enumerate(module.SENSOR_PAIRS):
        out.setdefault(tuple(sorted((module.ARM_BODY_LIST[i], module.PROT_GROUPS[p]))), []).append(
            len(module.SELF_PAIRS) + k
        )
    return out


def proxy_series(module, poses: np.ndarray, columns: dict, watch: list) -> dict:
    names = list(dict.fromkeys(module.ARM_BODY_LIST + module.PROT_BODY_LIST))
    out = {key: [] for key in watch}
    batch = 256
    for start in range(0, len(poses), batch):
        chunk = poses[start : start + batch]
        pos, quat = [], []
        for q in chunk:
            mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
            for address, value in zip(arm_adr, q):
                data.qpos[address] = value
            mujoco.mj_forward(model, data)
            pos.append([data.xpos[bid[n]].copy() for n in names])
            quat.append([data.xquat[bid[n]].copy() for n in names])
        pos_t = torch.tensor(np.array(pos, dtype=np.float32))
        quat_t = torch.tensor(np.array(quat, dtype=np.float32))

        class Asset:
            body_names = names

            class data:  # noqa: N801
                body_pos_w = pos_t
                body_quat_w = quat_t

        d = module.clearance(Asset, cap=0.25)
        for key in watch:
            cols = [c for c in columns.get(tuple(sorted(key)), []) if c < d.shape[1]]
            out[key].append(d[:, cols].amin(-1).numpy() if cols else np.full(len(chunk), 0.25))
    return {k: np.concatenate(v) for k, v in out.items()}


def main() -> None:
    poses = np.load(ROOT / args.poses)["poses"]
    module = load_safety()
    install(module, ROOT / args.table)
    columns = pair_columns(module)
    global WATCH
    if args.all_pairs:
        # every pair that ever reads as contact on these poses, most frequent first
        watch = list(columns)
        WATCH = [w for w in watch]

    truth = {key: [] for key in WATCH}
    for q in poses:
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        for address, value in zip(arm_adr, q):
            data.qpos[address] = value
        mujoco.mj_forward(model, data)
        for key in WATCH:
            truth[key].append(mj_distance(*key))
    truth = {k: np.array(v) for k, v in truth.items()}
    proxy = proxy_series(module, poses, columns, WATCH)

    print(f"table {args.table}")
    print(f"poses {len(poses)}\n")
    rows = {}
    for key in WATCH:
        t, p = truth[key], proxy[key]
        contact_t = t <= 0.0
        false = (p <= 0.0) & ~contact_t
        miss = (t <= 0.0) & (p > 0.0)
        band = (t > 0.0) & (t <= args.band)
        err = (p - t)[band] if band.any() else np.array([0.0])
        rows[" ".join(key)] = dict(
            truth_contact=float(contact_t.mean()),
            proxy_contact=float((p <= 0).mean()),
            false_alarm=float(false.mean()),
            miss=float(miss.mean()),
            band_n=int(band.sum()),
            band_median=float(np.median(err)),
            band_p10=float(np.percentile(err, 10)),
            band_p90=float(np.percentile(err, 90)),
        )
        print(
            f"{key[0]} <-> {key[1]}\n"
            f"  真实接触 {contact_t.mean() * 100:5.1f}%   代理判定接触 {(p <= 0).mean() * 100:5.1f}%   "
            f"误报 {false.mean() * 100:5.1f}%   漏报 {miss.mean() * 100:4.2f}%\n"
            f"  近接触带({args.band * 1000:.0f} mm 内, {int(band.sum())} 步)误差 中位 {np.median(err) * 1000:+.1f} mm, "
            f"p10 {np.percentile(err, 10) * 1000:+.1f}, p90 {np.percentile(err, 90) * 1000:+.1f}"
        )

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    path = Path(args.json) if args.json else out / f"proxy_pairs_{Path(args.table).stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(table=args.table, poses=int(len(poses)), pairs=rows), indent=1))
    print(f"\n已写出 {path}")


if __name__ == "__main__":
    main()
