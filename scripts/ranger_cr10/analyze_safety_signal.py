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

"""Stage 4.11: what the safety term actually says near ``Link3 <-> Link5`` contact.

Three questions, answered from the geometry rather than from the training curve:

1. **Where does the contact happen?** Joint configurations of real ``Link3 <-> Link5``
   contacts, which joints sit at a limit, and how deep the penetration goes.
2. **How good is the proxy in the last 20 mm?** A safety term is only useful if its distance
   is faithful exactly where it fires; the median bias over the whole configuration space says
   nothing about the 20 mm band that the reward cares about.
3. **What gradient does the reward deliver there?** The penalty is quadratic, so its slope
   grows as the clearance shrinks -- what matters is whether it is steep *before* contact, and
   whether the authored pose pays anything.

It also checks the reference-gap accounting directly: the 4.9-4.10 term measured clearance
relative to a per-episode reference captured on the first call (`env._sensor_clearance_ref`).
That attribute must no longer exist -- a stale reference captured at reset would silently make
the penalty relative again, which is the bug stage 4.11 set out to remove.

Usage:
    python scripts/ranger_cr10/analyze_safety_signal.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SAFETY = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/safety.py"
SCENE = ROOT / "mujoco/robots/ranger_cr10/scene.xml"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--samples", type=int, default=4000)
ap.add_argument("--band", type=float, default=0.020, help="the last-N-metres band before contact")
ap.add_argument("--margin", type=float, default=0.015, help="reward margin, metres")
ap.add_argument("--out", default="outputs/stage411")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sf = load(SAFETY, "safety")
import mujoco  # noqa: E402

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
bid = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)}
geom_of = {
    n: [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid[n] and model.geom_contype[g] != 0]
    for n in ("cr10_Link3", "cr10_Link5")
}
arm = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)]
limits = np.array(
    [model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)]
)
DEFAULT_Q = np.array([0.236, 0.086, -2.398, -0.830, -1.335, 0.0])
names = list(dict.fromkeys(sf.ARM_BODY_LIST + sf.PROT_BODY_LIST))


def hull_distance() -> float:
    best = 0.5
    for g1 in geom_of["cr10_Link3"]:
        for g2 in geom_of["cr10_Link5"]:
            best = min(best, mujoco.mj_geomDistance(model, data, g1, g2, best, None))
    return best


def proxy_distance() -> float:
    pos = torch.tensor(np.stack([data.xpos[bid[n]] for n in names])[None].astype(np.float32))
    quat = torch.tensor(np.stack([data.xquat[bid[n]] for n in names])[None].astype(np.float32))

    class Asset:
        body_names = names

        class data:  # noqa: N801 - a stand-in for the Isaac asset, not a class to instantiate
            body_pos_w = pos
            body_quat_w = quat

    d = sf.clearance(Asset, cap=0.5)[0]
    cols = [
        k for k, (i, j) in enumerate(sf.SELF_PAIRS) if {sf.ARM_BODY_LIST[i], sf.ARM_BODY_LIST[j]} == {"cr10_Link3", "cr10_Link5"}
    ]
    return float(d[cols].min())


def set_pose(q) -> None:
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for address, value in zip(arm, q):
        data.qpos[address] = value
    mujoco.mj_forward(model, data)


def main() -> None:
    rng = np.random.default_rng(args.seed)
    rows = []
    for k in range(args.samples):
        if k % 3 == 0:
            q = rng.uniform(limits[:, 0], limits[:, 1])
        else:
            q = np.clip(DEFAULT_Q + rng.normal(0.0, 0.6, 6), limits[:, 0], limits[:, 1])
        set_pose(q)
        truth = hull_distance()
        if truth > args.band + 0.01:
            continue
        rows.append((q, truth, proxy_distance()))

    truth = np.array([r[1] for r in rows])
    proxy = np.array([r[2] for r in rows])
    near = truth <= args.band
    print(f"样本 {len(rows)} 个落在 {args.band * 1000:.0f} mm 带内; 其中真实接触 {int((truth <= 0).sum())}")

    # MuJoCo reports penetration depth as a *negative* distance while the proxy clamps at
    # zero, so the two are only comparable on the clear side -- comparing them across contact
    # turns every 40 mm penetration into a phantom "+40 mm overestimate".
    clear = (truth > 0.0) & near
    err = proxy - truth
    print(f"\n[带内代理误差, 未接触样本 {int(clear.sum())} 个] 中位 {np.median(err[clear]) * 1000:+.1f} mm, "
          f"p10 {np.percentile(err[clear], 10) * 1000:+.1f}, p90 {np.percentile(err[clear], 90) * 1000:+.1f}, "
          f"最大 {err[clear].max() * 1000:+.1f}")
    print(f"[带内漏检] 真实 ≤0 而代理 >0: {int(((truth <= 0) & (proxy > 0)).sum())} / {int((truth <= 0).sum())}")

    contact = truth <= 0.0
    if contact.any():
        depth = -truth[contact] * 1000
        print(f"\n[真实接触] {int(contact.sum())} 个构型; 穿透深度 中位 {np.median(depth):.1f} mm, "
              f"p90 {np.percentile(depth, 90):.1f} mm, 最大 {depth.max():.1f} mm")
        qc = np.array([r[0] for r in rows])[contact]
        for i in range(6):
            at_lo = (qc[:, i] - limits[i, 0] < 0.05).mean() * 100
            at_hi = (limits[i, 1] - qc[:, i] < 0.05).mean() * 100
            print(f"  q{i + 1}: 中位 {np.median(qc[:, i]):+.2f} rad; 贴近下限 {at_lo:4.0f}%, 贴近上限 {at_hi:4.0f}%")
        # is the proxy saying "contact" there?
        print(f"  这些构型里代理判定接触的比例 {(proxy[contact] <= 0).mean() * 100:.0f}%")
        # and what does the reward actually say there? (the term, before the weight)
        penalty = -np.clip((args.margin - np.clip(proxy[contact], 0.0, None)) / args.margin, 0.0, None) ** 2
        print(f"  这些构型上的惩罚值 中位 {np.median(penalty):+.3f}, 最弱 {penalty.max():+.3f} "
              f"(满惩罚 -1.000)")

    # the reward itself: value and slope at several clearances
    print("\n[奖励形状, margin = %.0f mm]" % (args.margin * 1000))
    for d_mm in (0, 2, 5, 10, 15, 20, 30):
        d = d_mm / 1000.0
        value = -max(0.0, (args.margin - d) / args.margin) ** 2
        slope = 2 * (args.margin - d) / args.margin**2 if d < args.margin else 0.0
        print(f"  d={d_mm:2d} mm  term={value:+.4f}  d(term)/dd={slope:6.1f} /m  ({slope / 1000:.3f} /mm)")
    track_slope = 4.5 * 2 * 0.020 / 0.1 * np.exp(-0.020**2 / 0.1)
    print(f"  对比: 位置跟踪项在 20 mm 处的斜率 {track_slope:.1f} /m ({track_slope / 1000:.4f} /mm)")

    # reference-gap accounting must be gone
    legacy = [n for n in dir(sf) if "clearance_ref" in n]
    has_attr_ref = any("_sensor_clearance_ref" in line for line in SAFETY.read_text().splitlines())
    print(f"\n[参考间隙] 模块内 _sensor_clearance_ref 引用: {'仍在' if (legacy or has_attr_ref) else '已移除'}")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "safety_signal.json").write_text(
        json.dumps(
            dict(
                samples=int(near.sum()),
                contacts=int((truth <= 0).sum()),
                band_error_median=float(np.median(err[clear])),
                band_error_p90=float(np.percentile(err[clear], 90)),
                band_miss=int(((truth <= 0) & (proxy > 0)).sum()),
                median_penetration_mm=float(np.median(-truth[contact] * 1000)) if contact.any() else 0.0,
                reference_gap_present=bool(legacy or has_attr_ref),
            ),
            indent=1,
        )
    )
    print(f"\n已写出 {out / 'safety_signal.json'}")


if __name__ == "__main__":
    main()
