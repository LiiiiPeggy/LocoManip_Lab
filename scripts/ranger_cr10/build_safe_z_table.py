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

"""Stage 4.10: how far down can the tool go at each (x, y), with the real geometry?

Stage 4.9 shipped a single ``Z >= -0.05`` clamp because the lidar sits under the old default
pose and any lower target went through it. That is true *above the lidar* and false everywhere
else: with the tool out in front of the chassis, most columns are clear well below -0.05.

This walks each column of the keyboard box downwards -- 2 cm at a time, continuous IK from the
previous solution, real collision geometry -- and records the lowest offset that touches
nothing. The result is written next to ``workspace_limits.py`` and the keyboard clamps against
it, so the limit becomes a function of where the tool is rather than one number for the whole
box.

The walk is *continuous*, matching how the operator reaches a target, so the table is what a
keyboard user can actually get to rather than what some far-away IK branch could hold.

Usage:
    python scripts/ranger_cr10/build_safe_z_table.py
    python scripts/ranger_cr10/build_safe_z_table.py --step 0.02 --margin 0.015 \
        --out scripts/ranger_cr10/safe_z_table.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ranger_cr10"))

import teleop_kinematics as tk  # noqa: E402
import workspace_limits as ws  # noqa: E402

import mujoco  # noqa: E402

ap = argparse.ArgumentParser(description="Measure the collision-free lower Z bound over (x, y).")
ap.add_argument("--out", default=str(ROOT / "scripts/ranger_cr10/safe_z_table.json"))
ap.add_argument("--nx", type=int, default=7)
ap.add_argument("--ny", type=int, default=7)
ap.add_argument("--step", type=float, default=0.02)
ap.add_argument("--floor", type=float, default=-0.40, help="deepest offset the walk tries")
ap.add_argument("--margin", type=float, default=0.015, help="required clearance at the recorded point")
args = ap.parse_args()

m = mujoco.MjModel.from_xml_path(str(ROOT / "mujoco/robots/ranger_cr10/scene.xml"))
d = mujoco.MjData(m)
adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)]
PROTECTED = {"lidar_link0", "d435_link", "base_link"}
prot_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in PROTECTED}
prot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in prot_ids and m.geom_contype[g]]
arm_ids = {
    mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    for n in (
        "cr10_Link1",
        "cr10_Link2",
        "cr10_Link3",
        "cr10_Link4",
        "cr10_Link5",
        "cr10_Link6",
        "gripper_base_link",
        "d455_link",
    )
}
arm_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in arm_ids and m.geom_contype[g]]


def apply(q):
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    for a, v in zip(adr, q):
        d.qpos[a] = v
    mujoco.mj_forward(m, d)


def internal_contact():
    for c in range(d.ncon):
        b1, b2 = m.geom_bodyid[d.contact[c].geom1], m.geom_bodyid[d.contact[c].geom2]
        if b1 != 0 and b2 != 0:
            return True
    return False


def clearance(cap=0.30):
    best = cap
    for g1 in arm_geoms:
        for g2 in prot_geoms:
            dist = mujoco.mj_geomDistance(m, d, g1, g2, best, None)
            if dist < best:
                best = dist
    return best


def walk(dx, dy):
    """Lowest dz for this column that is collision-free and clear of the protected parts."""
    q = np.array(tk.DEFAULT_Q, dtype=float)
    R = tk.R_NEUTRAL
    best = 0.0
    gap = clearance()
    dz = -args.step
    while dz >= args.floor - 1e-9:
        target = tk.P_NEUTRAL + np.array([dx, dy, dz])
        q_sol, ep, er, _, _ = tk.local_ik(q, target, R)
        if ep > 2e-3 or er > 2e-3:
            break
        q = q_sol
        apply(q)
        if internal_contact():
            break
        c = clearance()
        if c < args.margin:
            break
        best, gap = dz, c
        dz -= args.step
    return best, gap


def main() -> None:
    box = ws.TARGET_BOX_410
    xs = np.linspace(box["pos_x"][0], box["pos_x"][1], args.nx)
    ys = np.linspace(box["pos_y"][0], box["pos_y"][1], args.ny)
    table = np.zeros((args.ny, args.nx))
    gaps = np.zeros((args.ny, args.nx))
    print(f"walking {args.nx}x{args.ny} columns, step {args.step * 1000:.0f} mm, margin {args.margin * 1000:.0f} mm")
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            table[j, i], gaps[j, i] = walk(float(x), float(y))
        print("  y = %+.2f: " % y + " ".join(f"{v:+.2f}" for v in table[j]))
    payload = {
        "x": xs.tolist(),
        "y": ys.tolist(),
        "z_min": table.tolist(),
        "clearance": gaps.tolist(),
        "step": args.step,
        "margin": args.margin,
        "default_tcp_b": tk.P_NEUTRAL.tolist(),
        "note": "z_min is the lowest dz reachable *continuously* from the default pose with the "
        "tool orientation held; it already includes the margin above.",
    }
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {args.out}")
    print(f"  range of z_min: {table.min():+.2f} .. {table.max():+.2f} m")
    below = (table < -0.05).mean()
    print(f"  columns where z can go below the old -0.05 clamp: {below * 100:.0f}%")


if __name__ == "__main__":
    main()
