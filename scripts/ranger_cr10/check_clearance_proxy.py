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

"""Stage 4.9: is the sphere proxy the training reward uses a faithful stand-in?

The reward in ``mdp/safety.py`` measures clearance with spheres around the arm links and
axis-aligned boxes for the lidar, the chassis camera and the chassis body. MuJoCo has the
real geometry -- convex hulls of the same meshes -- and can answer the same question exactly
with ``mj_geom_distance``. This script puts the two side by side over hundreds of arm
configurations and reports the disagreement, because a proxy that is *optimistic* would let
the policy train against a safety model that is not the one it is judged by.

A conservative proxy reports a smaller distance than the truth. That direction is safe (the
penalty starts early); the interesting failure is the other one.

Usage:
    python scripts/ranger_cr10/check_clearance_proxy.py
    python scripts/ranger_cr10/check_clearance_proxy.py --samples 2000 --out outputs/stage49
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SAFETY = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/safety.py"

ap = argparse.ArgumentParser(description="Compare the sphere clearance proxy with MuJoCo's exact geometry.")
ap.add_argument("--samples", type=int, default=600)
ap.add_argument("--out", default="outputs/stage49")
ap.add_argument("--margin", type=float, default=0.02, help="penalty margin the proxy is judged around")
args = ap.parse_args()


def load_safety():
    spec = importlib.util.spec_from_file_location("ranger_safety", SAFETY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


import mujoco  # noqa: E402

sf = load_safety()
m = mujoco.MjModel.from_xml_path(str(ROOT / "mujoco/robots/ranger_cr10/scene.xml"))
d = mujoco.MjData(m)

ARM_BODIES = list(dict.fromkeys(v[0] for v in sf.ARM_BOXES.values()))
PROT_BODIES = list(sf.PROTECTED_BOXES)

bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ARM_BODIES + PROT_BODIES}
excluded = set()
for i in range(m.nexclude):
    b1, b2 = m.exclude_signature[i] >> 16, m.exclude_signature[i] & 0xFFFF
    excluded.add(
        tuple(
            sorted(
                (
                    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1),
                    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2),
                )
            )
        )
    )

arm_geoms = {
    n: [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid[n] and m.geom_contype[g] != 0] for n in ARM_BODIES
}
prot_geoms = {
    n: [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid[n] and m.geom_contype[g] != 0] for n in PROT_BODIES
}
base = bid["base_link"]
adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)]
lim = np.array(
    [
        [
            m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")][0],
            m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")][1],
        ]
        for i in range(1, 7)
    ]
)
HOME = np.array([0.511, 0.494, -2.449, -1.186, -1.060, 0.0])


def exact_clearance(cap=0.5):
    best = cap
    for an, a_geoms in arm_geoms.items():
        for pn, pgs in prot_geoms.items():
            if tuple(sorted((an, pn))) in excluded:
                continue
            for g1 in a_geoms:
                for g2 in pgs:
                    dist = mujoco.mj_geomDistance(m, d, g1, g2, best, None)
                    if dist < best:
                        best = dist
    return best


def proxy_clearance(cap=0.5):
    """Same definition as `arm_clearance`, evaluated with numpy on MuJoCo's kinematics."""
    Rb = d.xmat[base].reshape(3, 3)
    pb = d.xpos[base]
    best = cap
    spheres = []
    for entry, (body, centre, half) in sf.ARM_BOXES.items():
        for c in sf._spheres(centre, half):
            b = bid[body]
            R = d.xmat[b].reshape(3, 3)
            world = R @ np.array(c[:3]) + d.xpos[b]
            rel = Rb.T @ (world - pb)
            spheres.append((body, rel, c[3]))
    for name, (centre, half) in sf.PROTECTED_BOXES.items():
        b = bid[name]
        R = d.xmat[b].reshape(3, 3)
        rel_rot = Rb.T @ R
        if not np.allclose(np.abs(rel_rot), np.eye(3), atol=1e-3):
            raise SystemExit(f"{name} is rotated relative to base_link")
        c_base = Rb.T @ (R @ np.array(centre) + d.xpos[b] - pb)
        skip = {b for b, prots in sf.PROTECTED_SKIP.items() if name in prots}
        for body, c, r in spheres:
            if body in skip:
                continue
            outside = np.maximum(np.abs(c - c_base) - np.array(half), 0.0)
            best = min(best, float(np.linalg.norm(outside)) - r)
    return best


def main() -> None:
    rng = np.random.default_rng(3)
    rows = []
    n = args.samples
    for i in range(n):
        # sample around the operating region: the home posture plus a perturbation that
        # stays inside the joint limits, so the comparison covers poses the policy uses
        if i < n // 4:
            q = HOME.copy()
        else:
            # most samples stay in the region the policy actually operates in; a quarter go
            # further, so the comparison also covers the deep-contact end
            sigma = 0.35 if i % 4 else 0.9
            q = np.clip(HOME + rng.normal(0.0, sigma, size=6), lim[:, 0], lim[:, 1])
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        for a, v in zip(adr, q):
            d.qpos[a] = v
        mujoco.mj_forward(m, d)
        rows.append((q.copy(), exact_clearance(), proxy_clearance()))

    exact = np.array([r[1] for r in rows])
    proxy = np.array([r[2] for r in rows])
    diff = proxy - exact
    print(f"样本 {n}; 判据间距 {args.margin * 1000:.0f} mm")
    print(f"  精确 (MuJoCo 凸包)   中位 {np.median(exact) * 1000:+.1f} mm, 最小 {exact.min() * 1000:+.1f} mm")
    print(f"  代理 (球+盒)         中位 {np.median(proxy) * 1000:+.1f} mm, 最小 {proxy.min() * 1000:+.1f} mm")
    print(
        f"  差 (代理-精确)       中位 {np.median(diff) * 1000:+.1f} mm, p10 {np.percentile(diff, 10) * 1000:+.1f}, "
        f"p90 {np.percentile(diff, 90) * 1000:+.1f}, 最小 {diff.min() * 1000:+.1f}"
    )
    cons = float((diff <= 0).mean())
    # the decision that matters is "inside the margin or not" -- compare the verdicts
    both = ((exact <= args.margin) & (proxy <= args.margin)).mean()
    only_exact = ((exact <= args.margin) & (proxy > args.margin)).mean()
    only_proxy = ((exact > args.margin) & (proxy <= args.margin)).mean()
    print(f"  代理更保守的比例 {cons * 100:.1f}%")
    print(
        f"  {args.margin * 1000:.0f} mm 判据一致: 都警戒 {both * 100:.1f}%, "
        f"漏报(真实接近但代理没报) {only_exact * 100:.1f}%, 误报 {only_proxy * 100:.1f}%"
    )
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "clearance_proxy.json").write_text(
        json.dumps(
            dict(
                n=n,
                exact=exact.tolist(),
                proxy=proxy.tolist(),
                diff_median=float(np.median(diff)),
                diff_p10=float(np.percentile(diff, 10)),
                diff_min=float(diff.min()),
                conservative_fraction=cons,
                missed=float(only_exact),
                false_alarm=float(only_proxy),
            ),
            indent=1,
        )
    )
    print(f"已写出 {out / 'clearance_proxy.json'}")


if __name__ == "__main__":
    main()
