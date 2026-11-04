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

"""Stage 4.11: is the distance the reward uses the distance physics has?

``mdp/safety.py`` builds a safety distance from forward kinematics and box geometry;
MuJoCo has the same geometry as convex hulls and answers the same question exactly with
``mj_geom_distance``. This script runs **the module itself** -- not a re-implementation -- over
thousands of arm configurations and reports where the two disagree.

The two numbers that decide whether the proxy is usable:

* **miss** -- physics says the pair is touching and the proxy says it is clear. This is the
  failure that matters; it is the one stage 4.10 had (62% of the ``Link3 <-> Link5`` contacts
  were invisible because the boxes were expressed in the mesh frame).
* **false alarm** -- the proxy reports contact where physics finds none. It costs workspace
  and reward, not safety, but a large number means the penalty is learning the wrong thing.

Because a single axis-aligned box is a *bound* on a mesh and not the mesh, a small positive
bias (proxy closer than physics) is expected and reported rather than tuned away.

Usage:
    python scripts/ranger_cr10/check_clearance_proxy.py
    python scripts/ranger_cr10/check_clearance_proxy.py --samples 2000 --timing_envs 2048
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SAFETY = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/safety.py"
SCENE = ROOT / "mujoco/robots/ranger_cr10/scene.xml"

ap = argparse.ArgumentParser(description="Compare the safety-distance proxy with MuJoCo's geometry.")
ap.add_argument("--samples", type=int, default=1200)
ap.add_argument("--timing_envs", type=int, default=2048, help="batch size for the cost measurement")
ap.add_argument("--cap", type=float, default=0.15, help="distance cap the reward uses")
ap.add_argument("--margin", type=float, default=0.03, help="penalty margin the verdict is judged at")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--operating", type=float, default=0.6, help="fraction of samples near the default pose")
ap.add_argument("--out", default="outputs/stage411")
args = ap.parse_args()


def load_safety():
    spec = importlib.util.spec_from_file_location("ranger_safety", SAFETY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sf = load_safety()
import mujoco  # noqa: E402

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
bid = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)
}

# Isaac groups: the bodies the URDF->USD conversion merges into one articulation body.
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
ARM_GROUPS = list(sf.ARM_PARTS)
PROT_GROUPS = list(sf.PROTECTED_PARTS)

excluded = set()
for i in range(model.nexclude):
    a, b = model.exclude_signature[i] >> 16, model.exclude_signature[i] & 0xFFFF
    excluded.add(
        tuple(
            sorted(
                (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, a),
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b),
                )
            )
        )
    )

collision_geoms = {
    name: [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid[name] and model.geom_contype[g] != 0]
    for group in GROUPS.values()
    for name in group
}
arm_joint = [
    model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)
]
joint_limits = np.array(
    [
        model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")]
        for i in range(1, 7)
    ]
)
DEFAULT_Q = np.array([0.236, 0.086, -2.398, -0.830, -1.335, 0.0])


def exact_between(group_a: str, group_b: str, cap: float) -> float:
    """Smallest MuJoCo hull distance between two Isaac groups."""
    best = cap
    for a in GROUPS[group_a]:
        for b in GROUPS[group_b]:
            if tuple(sorted((a, b))) in excluded:
                continue
            for g1 in collision_geoms[a]:
                for g2 in collision_geoms[b]:
                    dist = mujoco.mj_geomDistance(model, data, g1, g2, best, None)
                    if dist < best:
                        best = dist
    return best


def exact_pairs(cap: float) -> tuple[float, float]:
    """(arm self, arm protected) exact minima for the current configuration."""
    self_best = cap
    for i, a in enumerate(ARM_GROUPS):
        for b in ARM_GROUPS[i + 1 :]:
            if (a, b) in sf.ARM_ADJACENT or (b, a) in sf.ARM_ADJACENT or a == b:
                continue
            self_best = min(self_best, exact_between(a, b, self_best))
    prot_best = cap
    for a in ARM_GROUPS:
        for p in PROT_GROUPS:
            prot_best = min(prot_best, exact_between(a, p, prot_best))
    return self_best, prot_best


class _Data:
    def __init__(self, pos, quat):
        self.body_pos_w = pos
        self.body_quat_w = quat


class _Asset:
    def __init__(self, names, pos, quat):
        self.body_names = names
        self.data = _Data(pos, quat)


def proxy_batch(poses) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the production `clearance` on a batch of (body pos, quat) snapshots."""
    names = list(poses[0][0].keys())
    pos = torch.tensor(np.stack([[p[n] for n in names] for p, _ in poses]).astype(np.float32))
    quat = torch.tensor(np.stack([[q[n] for n in names] for _, q in poses]).astype(np.float32))
    asset = _Asset(names, pos, quat)
    d = sf.clearance(asset, cap=args.cap)
    n_self = len(sf.SELF_PAIRS)
    return d[:, :n_self].amin(-1), d[:, n_self:].amin(-1)


def snapshot() -> tuple[dict, dict]:
    bodies = list(dict.fromkeys([n for group in GROUPS.values() for n in group]))
    return (
        {n: data.xpos[bid[n]].copy() for n in bodies},
        {n: data.xquat[bid[n]].copy() for n in bodies},
    )


def main() -> None:
    rng = np.random.default_rng(args.seed)
    poses, exact = [], []
    for k in range(args.samples):
        if rng.random() < args.operating:
            q = np.clip(DEFAULT_Q + rng.normal(0.0, 0.5, size=6), joint_limits[:, 0], joint_limits[:, 1])
        else:
            q = rng.uniform(joint_limits[:, 0], joint_limits[:, 1])
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        for address, value in zip(arm_joint, q):
            data.qpos[address] = value
        mujoco.mj_forward(model, data)
        poses.append(snapshot())
        exact.append(exact_pairs(args.cap))

    proxy_self, proxy_prot = proxy_batch(poses)
    exact = np.array(exact)

    print(f"样本 {args.samples}; 判据 {args.margin * 1000:.0f} mm; 上限 {args.cap * 1000:.0f} mm")
    rows = []
    for name, proxy, exact_col in (
        ("臂内自碰撞", proxy_self.numpy(), exact[:, 0]),
        ("臂-底盘/传感器", proxy_prot.numpy(), exact[:, 1]),
    ):
        diff = proxy - exact_col
        contact = exact_col <= 0.0
        miss = int((contact & (proxy > 0.0)).sum())
        false = int(((proxy <= 0.0) & (exact_col > 0.0)).sum())
        near = exact_col <= args.margin
        near_miss = int((near & (proxy > args.margin)).sum())
        near_false = int(((proxy <= args.margin) & ~near).sum())
        # a contact that MuJoCo reports as penetration shows up as a positive difference by
        # construction -- the proxy clamps at zero, so only the clear side is comparable
        clear = ~contact
        rows.append(
            dict(
                name=name,
                contact_rate=float(contact.mean()),
                proxy_contact=float((proxy <= 0.0).mean()),
                miss=miss,
                miss_rate=miss / max(int(contact.sum()), 1),
                false_alarms=false,
                near_miss=near_miss,
                near_false=near_false,
                diff_median=float(np.median(diff)),
                diff_p90=float(np.percentile(diff, 90)),
                diff_min=float(diff.min()),
                clear_diff_median=float(np.median(diff[clear])) if clear.any() else 0.0,
                clear_diff_p90=float(np.percentile(diff[clear], 90)) if clear.any() else 0.0,
                clear_diff_p10=float(np.percentile(diff[clear], 10)) if clear.any() else 0.0,
                miss_worst=float(proxy[contact & (proxy > 0.0)].max() * 1000) if miss else 0.0,
            )
        )
        print(
            f"\n{name}: 真实接触 {contact.mean() * 100:.1f}%, 代理判定接触 {(proxy <= 0.0).mean() * 100:.1f}%"
            f"\n  漏报 {miss} ({miss / max(int(contact.sum()), 1) * 100:.1f}% of contacts), 误报 {false}"
            f"\n  {args.margin * 1000:.0f} mm 内漏报 {near_miss}, 误报 {near_false}"
            f"\n  代理-真实 (未接触样本) 中位 {np.median(diff[clear]) * 1000:+.1f} mm, "
            f"p10 {np.percentile(diff[clear], 10) * 1000:+.1f}, p90 {np.percentile(diff[clear], 90) * 1000:+.1f}"
            f"\n  漏报时代理读数最大 {rows[-1]['miss_worst']:.1f} mm"
        )

    # cost at the batch size training uses
    batch = []
    for _ in range(args.timing_envs):
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        for address, value in zip(arm_joint, rng.uniform(joint_limits[:, 0], joint_limits[:, 1])):
            data.qpos[address] = value
        mujoco.mj_forward(model, data)
        batch.append(snapshot())
    names = list(batch[0][0].keys())
    asset = _Asset(
        names,
        torch.tensor(np.stack([[p[n] for n in names] for p, _ in batch]).astype(np.float32)),
        torch.tensor(np.stack([[q[n] for n in names] for _, q in batch]).astype(np.float32)),
    )
    torch.set_num_threads(1)
    sf.clearance(asset, cap=args.cap)  # warm up
    t0 = time.time()
    runs = 5
    for _ in range(runs):
        sf.clearance(asset, cap=args.cap)
    per_call = (time.time() - t0) / runs
    print(f"\n开销: {args.timing_envs} 环境 {per_call * 1000:.2f} ms/次 (CPU, 单线程)")
    print(f"      {len(sf.SELF_PAIRS) + len(sf.SENSOR_PAIRS)} 个碰撞对")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "clearance_proxy.json").write_text(json.dumps(dict(samples=args.samples, rows=rows,
                                                            cost_ms=per_call * 1000), indent=1))
    print(f"已写出 {out / 'clearance_proxy.json'}")


if __name__ == "__main__":
    main()
