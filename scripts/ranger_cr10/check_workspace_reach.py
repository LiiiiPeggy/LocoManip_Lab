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

"""Stage 4.7, task A: what the arm can physically reach, before asking any policy.

The old training box was deliberately small because an unreachable target turns the
orientation reward into a constant penalty. Expanding it needs the physical answer first, so
that a *policy* failure is not recorded as an unreachable target (and vice versa).

Four measurements, no policy and no IK supervision -- IK here is a measuring instrument,
never a training signal:

1. ``joint sweep`` -- uniform joint samples, FK, distribution of the TCP in ``base_link``
   relative to the default pose. The outer envelope, attitude unconstrained.
2. ``attitude`` -- hold the position at the default and push each rpy axis until IK stops
   converging. The candidate box asks for +-45 deg / +-60 deg; some of that may not exist.
3. ``position`` -- hold the attitude and push each position axis. A coupled corner is not
   implied by the two single-axis results.
4. ``box`` -- draw six-dimensional targets uniformly from the candidate box and count how
   many a multi-restart IK solves, how many a from-the-default-posture IK solves (the policy
   moves continuously, so that is the harder and more relevant number), and how many of those
   put the arm into the ground or the payload box.

Usage:
    python scripts/ranger_cr10/check_workspace_reach.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ranger_cr10"))
import teleop_kinematics as tk  # noqa: E402
import workspace_limits as ws  # noqa: E402

ap = argparse.ArgumentParser(description="Kinematic + collision workspace analysis.")
ap.add_argument("--samples", type=int, default=60000)
ap.add_argument("--box_samples", type=int, default=300)
ap.add_argument("--out", default="outputs/stage47")
ap.add_argument("--box", choices=("4.7", "4.8"), default="4.7", help="candidate box to test")
ap.add_argument("--limit", type=float, default=0.55, help="furthest offset the position sweep tries (m)")
ap.add_argument("--reach-table", default=None, help="write the reachability oracle JSON here")
ap.add_argument("--reach-bins", type=int, default=5, help="attitude cells per axis in that table")
ap.add_argument("--reach-validate", type=int, default=200, help="random targets used to check the table")
ap.add_argument("--validate-only", action="store_true", help="check an existing --reach-table and exit")
args = ap.parse_args()

OUT = ROOT / args.out
OUT.mkdir(parents=True, exist_ok=True)


def _as_arrays(box):
    return dict(
        pos=tuple(box[k] for k in ws.POS_AXES),
        rpy=tuple(box[k] for k in ws.RPY_AXES),
    )


OLD = dict(pos=((-0.12, 0.12), (-0.15, 0.15), (-0.12, 0.10)), rpy=((-0.40, 0.40), (-0.40, 0.40), (-0.50, 0.50)))
NEW = _as_arrays(ws.TARGET_BOX_48 if args.box == "4.8" else ws.TARGET_BOX)

P0 = tk.P_NEUTRAL
R0 = tk.R_NEUTRAL
TOL_P, TOL_R = 1e-3, 1e-3


def solve(pos, rpy):
    """Reachability probe: a few restarts, a short run.

    ``tk.solve_ik`` costs 1.5 s per call (12 restarts x 200 iterations) which is right for
    producing a *solution* but far too slow for a sweep of hundreds of probes. Four guesses
    and 80 iterations answer "is there a configuration that reaches this" at about 0.2 s, and
    the sweep stops at the first failure, so a false negative is reported rather than hidden.
    """
    R = tk.euler_R(rpy) @ R0
    target = P0 + np.asarray(pos)
    rng = np.random.default_rng(0)
    guesses = [
        tk.DEFAULT_Q,
        np.zeros(6),
        rng.uniform(tk.LIMITS[:, 0], tk.LIMITS[:, 1]),
        rng.uniform(tk.LIMITS[:, 0], tk.LIMITS[:, 1]),
    ]
    best = None
    for g in guesses:
        ep, er, q = tk.ik(target, R, g, iters=80)
        if best is None or ep + er < best[0] + best[1]:
            best = (ep, er, q)
        if best[0] < TOL_P and best[1] < TOL_R:
            break
    return best[0], best[1], best[2], 0.0


def ok(pos, rpy):
    ep, er, _q, _s = solve(pos, rpy)
    return ep < TOL_P and er < TOL_R


def continuation(axis, sign, att_rpy, step=0.01, limit=None):
    """How far the TCP can go along one axis *while moving there continuously*.

    Random-restart IK answers "does any configuration reach this", and it does so unreliably:
    measured here, ``Y+0.20`` at the default attitude comes back in a branch with joint1 at
    -3.4 rad, which a policy that starts at the default posture will not discover, while the
    same probe reported only 0.05 as reachable. So the sweep is done the way a controller
    actually moves -- 1 cm at a time, each solve seeded from the previous one -- which yields
    the reachable set *connected to the default pose*. Both numbers are reported: this one is
    the one that matters for a walking policy, and the multi-restart rate in part 4 is the
    kinematic bound.
    """
    limit = args.limit if limit is None else limit
    R = tk.euler_R(att_rpy) @ R0
    q = np.asarray(tk.DEFAULT_Q, dtype=float)
    best = 0.0
    v = step
    while v <= limit + 1e-9:
        target = P0.copy()
        target[axis] += sign * v
        q_sol, ep, er, _, _ = tk.local_ik(q, target, R)
        if ep > 2e-3 or er > 2e-3:
            break
        q = q_sol
        best = v
        v += step
    return best


def sweep(fn, values):
    """Largest magnitude in ``values`` for which ``fn`` holds, and the first that fails."""
    last, first_bad = 0.0, None
    for v in values:
        if fn(v):
            last = v
        else:
            first_bad = v
            break
    return last, first_bad


def rpy_with(axis, value):
    out = [0.0, 0.0, 0.0]
    out[axis] = value
    return out


def main() -> None:
    if args.validate_only:
        validate_reach_table(Path(args.reach_table), args.reach_validate)
        return

    rng = np.random.default_rng(7)

    # -- 1. joint sweep -------------------------------------------------------------
    print(f"1. 关节空间扫掠 ({args.samples} 个构型, 关节限位内均匀)")
    q = rng.uniform(tk.LIMITS[:, 0], tk.LIMITS[:, 1], size=(args.samples, 6))
    pos = np.empty((args.samples, 3))
    for i in range(args.samples):
        pos[i] = tk.fk(q[i])[:3, 3] - P0
    lo, hi = np.percentile(pos, 1, axis=0), np.percentile(pos, 99, axis=0)
    full_lo, full_hi = pos.min(axis=0), pos.max(axis=0)
    print(
        f"   位置偏移 (1%-99%): x {lo[0]:+.3f}..{hi[0]:+.3f}  y {lo[1]:+.3f}..{hi[1]:+.3f}  "
        f"z {lo[2]:+.3f}..{hi[2]:+.3f}"
    )
    print(
        f"   位置偏移 (极值)  : x {full_lo[0]:+.3f}..{full_hi[0]:+.3f}  "
        f"y {full_lo[1]:+.3f}..{full_hi[1]:+.3f}  z {full_lo[2]:+.3f}..{full_hi[2]:+.3f}"
    )
    in_old = np.all(
        (pos >= np.array([OLD["pos"][k][0] for k in range(3)]))
        & (pos <= np.array([OLD["pos"][k][1] for k in range(3)])),
        axis=1,
    )
    in_new = np.all(
        (pos >= np.array([NEW["pos"][k][0] for k in range(3)]))
        & (pos <= np.array([NEW["pos"][k][1] for k in range(3)])),
        axis=1,
    )
    print(f"   关节空间里 TCP 落在旧盒内的比例 {in_old.mean() * 100:.1f}%, 新盒内 {in_new.mean() * 100:.1f}%")

    # -- 2. attitude sweep at the default position ----------------------------------
    print("\n2. 姿态可达范围(位置固定在默认点, 角度从 0 向外推)")
    grid = [math.radians(v) for v in (5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 60, 70, 80)]
    att = {}
    print("   (多起点 IK 的判定, 偏宽松; 连续可达见第 3 节)")
    for k, name in enumerate(("roll", "pitch", "yaw")):
        row = []
        for sign in (-1, 1):
            last, _bad = sweep(lambda v: ok((0, 0, 0), rpy_with(k, sign * v)), grid)
            row.append(abs(last))
        att[name] = row
        # the sweep above is the kinematic bound; below, the same question asked continuously
        print(f"   {name:<6}{-math.degrees(row[0]):>10.1f}°{math.degrees(row[1]):>10.1f}°")

    # -- 3. position sweep at a few attitudes ---------------------------------------
    print("\n3. 位置可达范围(从默认位姿连续移动, 每步 1 cm; 单位 m)")
    cases = [
        ("姿态=默认", (0.0, 0.0, 0.0)),
        ("roll=+45°", (math.radians(45), 0.0, 0.0)),
        ("roll=-45°", (math.radians(-45), 0.0, 0.0)),
        ("pitch=+45°", (0.0, math.radians(45), 0.0)),
        ("pitch=-30°", (0.0, math.radians(-30), 0.0)),
        ("yaw=+80°", (0.0, 0.0, math.radians(80))),
        # the two corners the 4.8 brief singles out: forward+down and back+up
        ("pitch+45,yaw+60", (0.0, math.radians(45), math.radians(60))),
        ("pitch-30,roll-45", (math.radians(-45), math.radians(-30), 0.0)),
    ]
    pos_reach = {}
    print(f"   {'姿态条件':<16}{'X-':>8}{'X+':>8}{'Y-':>8}{'Y+':>8}{'Z-':>8}{'Z+':>8}")
    for label, att_rpy in cases:
        vals = []
        for k in range(3):
            for sign in (-1, 1):
                vals.append(continuation(k, sign, att_rpy))
        pos_reach[label] = vals
        print(f"   {label:<16}" + "".join(f"{-vals[2 * k]:>8.2f}{vals[2 * k + 1]:>8.2f}" for k in range(3)))

    # -- 4. six-dimensional box sampling --------------------------------------------
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(ROOT / "mujoco" / "robots" / "ranger_cr10" / "scene.xml"))
    d = mujoco.MjData(m)
    arm_adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)]
    arm_geoms = {
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        for g in (
            "link1_collision",
            "link2_collision",
            "link3_collision",
            "link4_collision",
            "link5_collision",
            "link6_collision",
            "cr10_base_link_col",
        )
    }

    def collides(qq):
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        for a, v in zip(arm_adr, qq):
            d.qpos[a] = v
        mujoco.mj_forward(m, d)
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if (g1 in arm_geoms) != (g2 in arm_geoms):
                return True
        return False

    print(f"\n4. 六维目标抽样 ({args.box_samples} 个, 从候选盒均匀)")
    rng2 = np.random.default_rng(11)
    n_local = n_global = n_coll = 0
    unsolved = []
    for _ in range(args.box_samples):
        off = np.array([rng2.uniform(*NEW["pos"][k]) for k in range(3)])
        rpy = np.array([rng2.uniform(*NEW["rpy"][k]) for k in range(3)])
        R = tk.euler_R(rpy) @ R0
        q_loc, lp, lr, _, _ = tk.local_ik(np.asarray(tk.DEFAULT_Q), P0 + off, R)
        local_ok = lp < 2e-3 and lr < 2e-3
        gp, gr, q_g, _ = solve(off, rpy)
        global_ok = gp < TOL_P and gr < TOL_R
        n_local += local_ok
        n_global += global_ok
        if local_ok or global_ok:
            n_coll += collides(q_loc if local_ok else q_g)
        else:
            unsolved.append(off)
    n = args.box_samples
    print(
        f"   多起点 IK 可达 {n_global / n * 100:.1f}%   就近(不换构型)可达 {n_local / n * 100:.1f}%   "
        f"可达目标中与地面/货箱接触 {n_coll / max(n_global, 1) * 100:.1f}%"
    )
    if unsolved:
        u = np.array(unsolved)
        print(
            "   不可达目标的位置均值±标准差: "
            + ", ".join(f"{u[:, k].mean():+.3f}±{u[:, k].std():.3f}" for k in range(3))
        )

    payload = dict(
        joint_sweep=dict(
            lo_1pct=lo.tolist(),
            hi_99pct=hi.tolist(),
            lo=full_lo.tolist(),
            hi=full_hi.tolist(),
            frac_in_old=float(in_old.mean()),
            frac_in_new=float(in_new.mean()),
        ),
        attitude_limit_deg={k: [-math.degrees(v[0]), math.degrees(v[1])] for k, v in att.items()},
        position_limit_m=pos_reach,
        box=dict(n=n, global_rate=n_global / n, local_rate=n_local / n, collision_rate=n_coll / max(n_global, 1)),
        candidate=NEW,
        old=OLD,
    )
    (OUT / "workspace_reach.json").write_text(json.dumps(payload, indent=1))
    print(f"\n已写出 {OUT / 'workspace_reach.json'}")

    if args.reach_table:
        write_reach_table(Path(args.reach_table))
        validate_reach_table(Path(args.reach_table), args.reach_validate)


def _grid(rng, bins):
    lo, hi = rng
    if bins == 1:
        return [(lo + hi) / 2]
    step = (hi - lo) / (bins - 1)
    return [lo + i * step for i in range(bins)]


def _seed_for_attitude(rpy, half_extents, restarts=3, seed=0):
    """A joint configuration that reaches this attitude *somewhere* in the box.

    Section 3 shows why the default position is the wrong place to ask: ``roll=-45`` is
    unreachable at the default position but perfectly reachable elsewhere in the box, so an
    oracle that only probes the default would mark a whole attitude slice unreachable. The
    probe therefore tries the default first and then the six axis extremes, and takes the
    first pose it can actually solve.
    """
    R = tk.euler_R(rpy) @ R0
    rng = np.random.default_rng(seed)
    guesses = [tk.DEFAULT_Q, np.zeros(6)] + [rng.uniform(tk.LIMITS[:, 0], tk.LIMITS[:, 1]) for _ in range(restarts)]
    probes = [np.zeros(3)]
    for k in range(3):
        for sign in (-1, 1):
            off = np.zeros(3)
            off[k] = sign * half_extents[k]
            probes.append(off)
    for off in probes:
        for g in guesses:
            ep, er, q = tk.ik(P0 + off, R, g)
            if ep < 1e-3 and er < 1e-3:
                return q, off
    return None, None


def _extent_from(q_start, off_start, axis, sign, R, step=0.02, limit=0.55):
    """How far one axis continues from an already-solved pose, without changing branch."""
    q = np.asarray(q_start, float).copy()
    off = np.asarray(off_start, float).copy()
    while True:
        nxt = off.copy()
        nxt[axis] += sign * step
        if abs(nxt[axis]) > limit + 1e-9:
            break
        q_sol, ep, er, _, _ = tk.local_ik(q, P0 + nxt, R)
        if ep > 2e-3 or er > 2e-3:
            break
        q, off = q_sol, nxt
    return off[axis]


def write_reach_table(path: Path) -> None:
    """Build the reachability oracle both sides of training use.

    One entry per cell of a regular ``bins x bins x bins`` grid over the candidate box's
    attitude ranges. For a cell, the per-axis position range is measured as: find *one*
    solvable pose at that attitude (default position first, then the axis extremes), then
    walk each position axis from there at 2 cm per step, keeping the joint branch. That is
    the reachable set *connected to a solved configuration*, which is the right notion for a
    policy that is free to move anywhere in the box -- unlike section 3's sweep, which is
    anchored at the default posture and therefore reports ``roll=-45`` as 0 everywhere.

    The grid is coarse on purpose: reachability is a slowly varying function of attitude,
    and ``validate_reach_table`` measures how much the binning costs.
    """
    bins = args.reach_bins
    roll = _grid(NEW["rpy"][0], bins)
    pitch = _grid(NEW["rpy"][1], bins)
    yaw = _grid(NEW["rpy"][2], bins)
    half = [max(abs(NEW["pos"][k][0]), abs(NEW["pos"][k][1])) for k in range(3)]
    limits = []
    total = len(roll) * len(pitch) * len(yaw)
    print(f"\n5. 可达性查表: {len(roll)}x{len(pitch)}x{len(yaw)} = {total} 个姿态格, 每格 6 个方向连续推进")
    done = 0
    empty = 0
    for r in roll:
        rows_p = []
        for p in pitch:
            rows_y = []
            for y in yaw:
                R = tk.euler_R((r, p, y)) @ R0
                q0, off0 = _seed_for_attitude((r, p, y), half)
                if q0 is None:
                    rows_y.append([0.0] * 6)
                    empty += 1
                else:
                    row = [0.0] * 6
                    for k in range(3):
                        row[2 * k] = _extent_from(q0, off0, k, -1, R)
                        row[2 * k + 1] = _extent_from(q0, off0, k, +1, R)
                    rows_y.append(row)
                done += 1
            rows_p.append(rows_y)
        limits.append(rows_p)
        print(f"   已扫 {done}/{total} 个姿态格 (roll {math.degrees(r):+.0f}°), 无解 {empty}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"roll": roll, "pitch": pitch, "yaw": yaw, "limits": limits}, indent=1))
    print(f"   已写出 {path}")


def validate_reach_table(path: Path, n: int) -> None:
    """Does the table's verdict match an actual IK solve? Reported, not assumed.

    False positives (table says reachable, the arm cannot get there) are the dangerous
    direction: they make the curriculum blame the policy for a physical limit. False
    negatives simply make the gate more lenient than it needs to be.
    """
    table = json.loads(path.read_text())
    axes = ("roll", "pitch", "yaw")
    grid = [np.array(table[k]) for k in axes]
    lim = np.array(table["limits"])
    rng = np.random.default_rng(23)
    fp = fn = ok = bad = 0
    for _ in range(n):
        off = np.array([rng.uniform(*NEW["pos"][k]) for k in range(3)])
        rpy = np.array([rng.uniform(*NEW["rpy"][k]) for k in range(3)])
        idx = [int(np.argmin(np.abs(grid[k] - rpy[k]))) for k in range(3)]
        lo = lim[idx[0], idx[1], idx[2], 0::2]
        hi = lim[idx[0], idx[1], idx[2], 1::2]
        table_ok = bool(np.all(off >= lo) and np.all(off <= hi))
        # independent multi-restart solve, not the continuous walk the table was built from:
        # the question is whether the target is solvable at all, from any branch
        ep, er, _q, _s = solve(off, rpy)
        ik_ok = ep < TOL_P and er < TOL_R
        if table_ok and ik_ok:
            ok += 1
        elif table_ok and not ik_ok:
            fp += 1
        elif not table_ok and ik_ok:
            fn += 1
        else:
            bad += 1
    print(
        f"   查表校验 ({n} 个随机目标): 一致可达 {ok}, 误判可达 {fp}, 误判不可达 {fn}, 一致不可达 {bad}\n"
        f"   总一致率 {(ok + bad) / n * 100:.1f}%; 危险方向(误判可达) {fp / n * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
