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

"""Stage 4.9: does the arm actually go through the lidar, the cameras or the chassis?

Three questions, kept apart on purpose, because "no contacts were reported" is only
evidence of safety if the detector is known to fire:

1. **Is the model right?** Every protected part must carry a collision geom, the default
   pose must produce no *internal* contact, and a set of hand-checked penetrating joint
   configurations must light up. If those configurations do not report a contact, the rest
   of the report means nothing.
2. **Does the policy collide?** Low targets (Z down to the keyboard edge), low combined
   targets, low attitudes, a continuous keyboard sweep and random six-dimensional targets
   from the extended box are each driven with the real policy, and the contacts, the minimum
   clearance and the joint state are recorded.
3. **Are the three notions of "touching" the same?** Contacts only exist where the *collision*
   geoms overlap; the meshes the operator sees are a different set. The script therefore
   reports visual-mesh overlap, collision-geom overlap and physics contacts separately, so a
   mismatch (invisible collision geometry, or a visual that is not modelled) cannot hide.

Clearance is measured with ``mj_geom_distance`` between the arm chain and the protected set
-- convex-hull distance for the meshes, exact for the boxes -- not from the TCP, because the
elbow is what hits the lidar.

Usage:
    python scripts/ranger_cr10/check_self_collision.py                       # model checks only
    python scripts/ranger_cr10/check_self_collision.py --policy <ckpt.pt>    # + policy scenarios
    python scripts/ranger_cr10/check_self_collision.py --viewer --policy <ckpt.pt>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mujoco" / "deploy"))
sys.path.insert(0, str(ROOT / "scripts" / "ranger_cr10"))

from deploy_mujoco.ranger_cr10 import teleop_eval as teleop  # noqa: E402
from deploy_mujoco.ranger_cr10.ranger_policy import ARM_JOINT_NAMES  # noqa: E402

import mujoco  # noqa: E402

SCENE = ROOT / "mujoco" / "robots" / "ranger_cr10" / "scene.xml"
DEFAULT_POLICY = ROOT / "logs/rsl_rl/ranger_cr10_teleop/2026-11-02_00-30-47_stage48a_ppo/model_5999.pt"

# What the arm must not touch, and what may do the touching. Kept as names so the report
# reads like the requirement rather than like an index list.
PROTECTED = ("lidar_link0", "d435_link", "base_link")
ARM_CHAIN = (
    "cr10_base_link",
    "cr10_Link1",
    "cr10_Link2",
    "cr10_Link3",
    "cr10_Link4",
    "cr10_Link5",
    "cr10_Link6",
    "gripper_base_link",
    "d455_link",
    # The fingers are part of the tool and stick out beyond the gripper base, so they can
    # reach the lidar while the base is still clear. They carry no *collision* geometry yet
    # (see the 4.9 notes), which is exactly why the visual query has to include them: a
    # visual-only overlap is what an operator sees as "the arm went through the lidar".
    "gripper_finger1_finger_tip_link",
    "gripper_finger2_finger_tip_link",
    "gripper_finger1_finger_link",
    "gripper_finger2_finger_link",
    "gripper_finger1_knuckle_link",
    "gripper_finger2_knuckle_link",
    "gripper_finger1_inner_knuckle_link",
    "gripper_finger2_inner_knuckle_link",
)

# Joint configurations that put a link *inside* a protected part. Found by random sampling and
# then checked by eye in the viewer; they exist to prove the detector fires. A pose that does
# not report a contact here is a detector failure, not a safe pose.
MUST_FIRE = {
    "arm -> lidar": (-3.554, 1.453, 0.229, 1.720, 0.184, 0.701),
    "arm -> chassis": (-3.387, -0.932, -1.237, -1.167, -1.174, 0.482),
}

ap = argparse.ArgumentParser(description="Self-collision diagnostics for the RangerBox-CR10.")
ap.add_argument("--policy", default=None, help="rsl_rl checkpoint or TorchScript; omit for model-only checks")
ap.add_argument("--out", default="outputs/stage49")
ap.add_argument("--random_targets", type=int, default=60)
ap.add_argument("--hold", type=int, default=30, help="policy steps held per target")
ap.add_argument("--seed", type=int, default=5)
ap.add_argument("--box", choices=("4.8", "4.10"), default="4.10", help="which command box the random targets come from")
ap.add_argument("--viewer", action="store_true", help="show the collision geometry live")
args = ap.parse_args()

OUT = ROOT / args.out
OUT.mkdir(parents=True, exist_ok=True)


class Model:
    """The MuJoCo model plus the queries the report needs."""

    def __init__(self) -> None:
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        self.name = lambda g: mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        self.bname = lambda b: mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
        self.arm_adr = [
            self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")]
            for i in range(1, 7)
        ]
        self.arm_lim = np.array(
            [
                [
                    self.m.jnt_range[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")][0],
                    self.m.jnt_range[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")][1],
                ]
                for i in range(1, 7)
            ]
        )
        body_id = {n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in PROTECTED + ARM_CHAIN}
        self.protected = {}
        self.arm = {}
        for n in PROTECTED:
            self.protected[n] = [
                g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == body_id[n] and self.m.geom_contype[g] != 0
            ]
        for n in ARM_CHAIN:
            self.arm[n] = [
                g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == body_id[n] and self.m.geom_contype[g] != 0
            ]
        self.excluded_pairs = self._excluded_pairs()

    def _excluded_pairs(self):
        """Body-name pairs MuJoCo will never contact."""
        out = set()
        for i in range(self.m.nexclude):
            b1, b2 = self.m.exclude_signature[i] >> 16, self.m.exclude_signature[i] & 0xFFFF
            out.add(tuple(sorted((self.bname(b1), self.bname(b2)))))
        return out

    def reset_home(self, q=None) -> None:
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("home").id)
        if q is not None:
            for a, v in zip(self.arm_adr, q):
                self.d.qpos[a] = v
        mujoco.mj_forward(self.m, self.d)

    def internal_contacts(self, only_protected: bool = False):
        """Contacts that are not with the ground, as (bodyA, bodyB) pairs.

        ``only_protected`` keeps just the pairs that involve the lidar, the chassis camera or
        the chassis body -- the question the operator asked. The rest of the arm's
        self-contacts (the wrist camera against Link3 at extreme attitudes) are real but are
        a different trade and would otherwise dominate a workspace sweep.
        """
        pairs = {}
        for c in range(self.d.ncon):
            g1, g2 = self.d.contact[c].geom1, self.d.contact[c].geom2
            b1, b2 = self.m.geom_bodyid[g1], self.m.geom_bodyid[g2]
            if b1 == 0 or b2 == 0:
                continue
            key = tuple(sorted((self.bname(b1), self.bname(b2))))
            if only_protected and not (set(key) & set(PROTECTED)):
                continue
            pairs.setdefault(key, []).append(float(self.d.contact[c].dist))
        return pairs

    def clearance(self, cap=0.5) -> float:
        """Smallest distance between the arm chain and the protected parts, metres.

        Negative means the *collision* geoms overlap. Excluded pairs are skipped: they are
        structural and would otherwise report a permanent hard zero.
        """
        best = cap
        for an, a_geoms in self.arm.items():
            for pn, pgs in self.protected.items():
                if tuple(sorted((an, pn))) in self.excluded_pairs:
                    continue
                for g1 in a_geoms:
                    for g2 in pgs:
                        dist = mujoco.mj_geomDistance(self.m, self.d, g1, g2, best, None)
                        if dist < best:
                            best = dist
        return best

    def joint_slack(self) -> float:
        q = np.array([self.d.qpos[a] for a in self.arm_adr])
        return float(np.min(np.minimum(q - self.arm_lim[:, 0], self.arm_lim[:, 1] - q)))


def model_checks(model: Model) -> dict:
    print("=" * 100)
    print("1. 模型检查: 受保护部件的碰撞几何 / 排除对 / 默认位姿")
    print("=" * 100)
    m = model.m
    rows = []
    for n, geoms in model.protected.items():
        for g in geoms:
            kind = mujoco.mjtGeom(m.geom_type[g]).name
            vis = [gg for gg in range(m.ngeom) if m.geom_bodyid[gg] == m.geom_bodyid[g] and not m.geom_contype[gg]]
            rows.append(
                dict(
                    body=n,
                    geom=model.name(g),
                    kind=kind,
                    visual_meshes=len(vis),
                    size=np.round(m.geom_size[g], 4).tolist(),
                )
            )
    for r in rows:
        print(f"  {r['body']:<14}{r['geom']:<22}{r['kind']:<12}size {r['size']}   visual geoms: {r['visual_meshes']}")

    model.reset_home()
    home = model.internal_contacts()
    print(f"\n  默认位姿内部接触: {len(home)} 对  {sorted(home) if home else '(无)'}")
    print(f"  排除对 (父/子 + 结构): {len(model.excluded_pairs)}")

    print("\n2. 强制碰撞验证 (构造的穿透位姿, 检测器必须报警)")
    fired = {}
    for label, q in MUST_FIRE.items():
        model.reset_home(np.array(q))
        pairs = model.internal_contacts()
        dist = model.clearance()
        ok = bool(pairs)
        fired[label] = dict(ok=ok, pairs=sorted(pairs), clearance=dist)
        found = sorted(pairs) if pairs else ""
        print(f"  {label:<16}{'检测到' if ok else '**未检测到**'}  {found}  最小间距 {dist * 1000:+.0f} mm")
    model.reset_home()
    return dict(protected=rows, home_contacts=sorted(home), excludes=len(model.excluded_pairs), must_fire=fired)


def run_scenario(sim, name: str, commands, steps_each: int = 1) -> dict:
    """Drive one sequence of EE targets and summarise contacts, clearance and errors."""
    sim.reset()
    worst = dict(
        name=name,
        pairs={},
        min_clearance=1.0,
        min_visual_clearance=1.0,
        max_contacts=0,
        first_contact_step=None,
        samples=0,
    )
    ee_err, ori_err, slack = [], [], []
    for k, cmd in enumerate(commands):
        sim.set_commands(0.0, 0.0, cmd)
        for _ in range(steps_each):
            row = sim.step()
            worst["samples"] += 1
            pairs = sim_internal_contacts(sim)
            if pairs and worst["first_contact_step"] is None:
                worst["first_contact_step"] = k
            for p, ds in pairs.items():
                worst["pairs"][p] = worst["pairs"].get(p, 0) + 1
            worst["max_contacts"] = max(worst["max_contacts"], sum(len(v) for v in pairs.values()))
            worst["min_clearance"] = min(worst["min_clearance"], clearance_of(sim))
            worst["min_visual_clearance"] = min(worst["min_visual_clearance"], clearance_of(sim, kind="vis"))
            ee_err.append(row["ee_pos"])
            ori_err.append(math.degrees(row["ee_ori"]))
            slack.append(joint_slack_of(sim))
    worst["pairs"] = {f"{a} <-> {b}": c for (a, b), c in sorted(worst["pairs"].items(), key=lambda kv: -kv[1])}
    worst["contact_rate"] = sum(worst["pairs"].values()) / max(worst["samples"], 1)
    worst["ee_err_median"] = float(np.median(ee_err)) if ee_err else 0.0
    worst["ori_err_median"] = float(np.median(ori_err)) if ori_err else 0.0
    worst["joint_slack_min"] = float(min(slack)) if slack else 0.0
    return worst


def sim_internal_contacts(sim) -> dict:
    m, d = sim.m, sim.d
    out = {}
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
        if b1 == 0 or b2 == 0:
            continue
        key = tuple(
            sorted(
                (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2))
            )
        )
        out.setdefault(key, []).append(float(d.contact[c].dist))
    return out


def _prepare_clearance_cache(sim, kind: str):
    """Geom lists for one kind of query: ``coll`` (physics geoms) or ``vis`` (drawn meshes)."""
    m = sim.m
    key = f"_clearance_{kind}"
    if hasattr(sim, key):
        return getattr(sim, key)
    body = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in PROTECTED + ARM_CHAIN}

    def geoms(names):
        if kind == "coll":
            return {
                n: [g for g in range(m.ngeom) if m.geom_bodyid[g] == body[n] and m.geom_contype[g] != 0] for n in names
            }
        return {
            n: [
                g
                for g in range(m.ngeom)
                if m.geom_bodyid[g] == body[n] and m.geom_contype[g] == 0 and m.geom_type[g] == 7
            ]
            for n in names
        }

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
    cache = (geoms(ARM_CHAIN), geoms(PROTECTED), excluded)
    setattr(sim, key, cache)
    return cache


def clearance_of(sim, cap=0.5, kind: str = "coll") -> float:
    """Smallest arm-chain-to-protected distance, skipping parent/child pairs.

    ``kind="vis"`` asks the same question about the *drawn* meshes. They differ wherever a
    part has no collision geometry, and the operator sees the meshes -- so both are reported
    and a disagreement is the interesting case, not a numerical detail.
    """
    m, d = sim.m, sim.d
    arm, prot, excluded = _prepare_clearance_cache(sim, kind)
    best = cap
    for an, a_geoms in arm.items():
        for pn, pgs in prot.items():
            if tuple(sorted((an, pn))) in excluded:
                continue
            for g1 in a_geoms:
                for g2 in pgs:
                    dist = mujoco.mj_geomDistance(m, d, g1, g2, best, None)
                    if dist < best:
                        best = dist
    return best


def joint_slack_of(sim) -> float:
    q = np.array(
        [sim.d.qpos[sim.m.jnt_qposadr[mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_JOINT, n)]] for n in ARM_JOINT_NAMES]
    )
    lim = []
    for n in ARM_JOINT_NAMES:
        j = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_JOINT, n)
        lim.append(sim.m.jnt_range[j])
    lim = np.array(lim)
    return float(np.min(np.minimum(q - lim[:, 0], lim[:, 1] - q)))


def policy_scenarios(sim) -> dict:
    ws = teleop.WS
    box = ws.TARGET_BOX_410 if args.box == "4.10" else ws.TARGET_BOX_48
    print()
    print("=" * 100)
    print("3. 策略场景 (低位目标 / 低位组合 / 低位姿态 / 连续键盘 / 随机六维)")
    print("=" * 100)
    print(f"  {'场景':<26}{'接触率':>9}{'碰撞间距':>10}{'视觉间距':>10}{'位置中位':>10}{'姿态中位':>10}   接触对")

    zero_rpy = (0.0, 0.0, 0.0)
    scen = {}

    def one(name, ees, steps=1):
        r = run_scenario(sim, name, ees, steps_each=steps)
        scen[name] = r
        pairs = ", ".join(f"{k}x{v}" for k, v in list(r["pairs"].items())[:2]) or "-"
        print(
            f"  {name:<26}{r['contact_rate'] * 100:>8.0f}%{r['min_clearance'] * 1000:>9.0f}mm"
            f"{r['min_visual_clearance'] * 1000:>9.0f}mm{r['ee_err_median'] * 1000:>8.0f}mm"
            f"{r['ori_err_median']:>9.1f}°   {pairs}"
        )
        return r

    # (a) straight down to the keyboard floor, then held
    def ramp(down_to, axis=2):
        seq = []
        n = max(2, int(abs(down_to) / 0.01))
        for i in range(1, n + 1):
            d = [0.0, 0.0, 0.0]
            d[axis] = down_to * i / n
            seq.append(teleop.ee_command_from_offset(d, zero_rpy))
        return seq + [seq[-1]] * 20

    one("Z- 连续下降到 -0.25", ramp(-0.25))
    one("X+0.30 / Z-0.20", [teleop.ee_command_from_offset((0.30, 0.0, -0.20), zero_rpy)] * 30)
    one("Y-0.40 / Z-0.15", [teleop.ee_command_from_offset((0.0, -0.40, -0.15), zero_rpy)] * 30)
    one("X+0.40 / Z-0.10", [teleop.ee_command_from_offset((0.40, 0.0, -0.10), zero_rpy)] * 30)
    one("X-0.05 / Z-0.25", [teleop.ee_command_from_offset((-0.05, 0.0, -0.25), zero_rpy)] * 30)

    # (b) low attitude sweeps: the wrist pitches/rolls down where the lidar sits
    for label, rpy in (
        ("低位 pitch -20°", (0.0, math.radians(-20), 0.0)),
        ("低位 pitch +60°", (0.0, math.radians(60), 0.0)),
        ("低位 roll +60°", (math.radians(60), 0.0, 0.0)),
        ("低位 yaw +80°", (0.0, 0.0, math.radians(80))),
    ):
        one(label, [teleop.ee_command_from_offset((0.05, 0.0, -0.20), rpy)] * 30)

    # (c) continuous keyboard motion through the low corner
    seq = []
    for i in range(1, 26):
        seq.append(teleop.ee_command_from_offset((0.25 * i / 25, 0.0, -0.25 * i / 25), zero_rpy))
    for i in range(1, 26):
        seq.append(teleop.ee_command_from_offset((0.25, -0.40 * i / 25, -0.25 + 0.10 * i / 25), zero_rpy))
    one("连续键盘: 前下 -> 右下", seq)

    # (d) the eight corners of the keyboard box: the operator can reach these with three keys
    # held at once, and they are where a low reach and a long reach combine
    kb = ws.KEYBOARD_BOX_48
    corners = []
    for sx in (kb["pos_x"][1], kb["pos_x"][0]):
        for sy in (kb["pos_y"][1], kb["pos_y"][0]):
            for sz in (kb["pos_z"][1], kb["pos_z"][0]):
                corners.append(teleop.ee_command_from_offset((sx, sy, sz), zero_rpy))
    one("键盘盒 8 个角落", [c for c in corners for _ in range(25)])

    # (e) random targets over the whole extended box
    rng = np.random.default_rng(args.seed)
    targets = []
    for _ in range(args.random_targets):
        targets.append(
            (
                [rng.uniform(*box[k]) for k in ("pos_x", "pos_y", "pos_z")],
                [rng.uniform(*box[k]) for k in ("roll", "pitch", "yaw")],
            )
        )
    seq = [teleop.ee_command_from_offset(d, r) for d, r in targets]
    policy_scenarios.targets = targets
    r = run_scenario(sim, f"随机六维 ({args.box} 盒)", seq, steps_each=args.hold)
    scen[f"随机六维 ({args.box} 盒)"] = r
    print(
        f"  {f'随机六维 ({args.box} 盒)':<26}{r['contact_rate'] * 100:>8.0f}%{r['min_clearance'] * 1000:>9.0f}mm"
        f"{r['min_visual_clearance'] * 1000:>9.0f}mm{r['ee_err_median'] * 1000:>8.0f}mm"
        f"{r['ori_err_median']:>9.1f}°   {r['pairs']}"
    )
    return scen


def collision_free_axis_limits(
    model: Model, tries: int = 12, step_pos: float = 0.02, step_ang: float = math.radians(5)
):
    """How far along each axis can the tool go on a path that never touches anything?

    This is the geometric answer to "what is left of the workspace once the lidar, the
    cameras and the chassis are real". It is resolved with *any* configuration (multi-restart
    IK), so it is the optimistic bound -- a policy that only moves continuously will do
    worse, and only the E1 scan can say how much worse.
    """
    import teleop_kinematics as tk

    P0, R0 = tk.P_NEUTRAL, tk.R_NEUTRAL
    rng = np.random.default_rng(1)
    axes = {"X": 0, "Y": 1, "Z": 2, "roll": 3, "pitch": 4, "yaw": 5}
    print()
    print("=" * 100)
    print("5. 无碰撞单轴范围 (任意构型, 每个格点 6 次重启 IK + 接触检查)")
    print("=" * 100)
    print(f"  {'轴':<6}{'负向':>10}{'正向':>10}   首个失败点")
    out = {}
    for name, k in axes.items():
        is_pos = k < 3
        step = step_pos if is_pos else step_ang
        limits = []
        for sign in (-1, 1):
            best, first_bad = 0.0, None
            q_prev = None
            mag = step
            while mag <= 0.55:
                dpos = [0.0, 0.0, 0.0]
                drpy = [0.0, 0.0, 0.0]
                (dpos if is_pos else drpy)[k % 3] = sign * mag
                R = tk.euler_R(drpy) @ R0
                ok = False
                for i in range(tries):
                    # previous solution first (continuation, which is how the 4.8 reach study
                    # asked the same question), then near-default guesses, then uniform ones:
                    # a large attitude usually only exists in a far branch, and sampling only
                    # near the default reported roll +30 where the policy holds +60.
                    if i == 0 and q_prev is not None:
                        guess = q_prev
                    elif i < 3:
                        guess = np.clip(tk.DEFAULT_Q + rng.normal(0, 0.5, 6), *tk.LIMITS.T)
                    else:
                        guess = rng.uniform(tk.LIMITS[:, 0], tk.LIMITS[:, 1])
                    ep, er, q = tk.ik(P0 + np.asarray(dpos), R, guess)
                    if ep < 1e-3 and er < 1e-3:
                        model.reset_home(q)
                        if not model.internal_contacts(only_protected=True):
                            ok = True
                            q_prev = q
                            break
                if ok:
                    best = mag
                elif first_bad is None:
                    first_bad = mag
                mag += step
            limits.append(sign * best)
        scale = 1.0 if is_pos else 57.2957795
        out[name] = limits
        unit = "m" if is_pos else "°"
        print(
            f"  {name:<6}{limits[0] * scale:>9.2f}{unit}{limits[1] * scale:>9.2f}{unit}   "
            f"{'无' if first_bad is None else f'{first_bad * scale:+.2f}{unit}'}"
        )
    model.reset_home()
    return out


def collision_aware_reachability(model: Model, targets, tries: int = 3) -> dict:
    """Task C3: of the reachable targets, how many can be held *without* touching anything?

    The 4.8 reachability table only answers "is there a joint configuration", so a target can
    be counted reachable and still be impossible to execute without putting the tool inside
    the lidar. For every target this solves IK, checks the solution for internal contacts, and
    reports the three statistics separately: kinematic, collision-free, and what the policy
    actually did (from the scenario above).
    """
    import teleop_kinematics as tk

    P0, R0 = tk.P_NEUTRAL, tk.R_NEUTRAL
    rng = np.random.default_rng(0)
    kin = free = 0
    rows = []
    for dpos, drpy in targets:
        R = tk.euler_R(drpy) @ R0
        best = None
        for _ in range(tries):
            ep, er, q = tk.ik(P0 + np.asarray(dpos), R, np.clip(tk.DEFAULT_Q + rng.normal(0, 0.6, 6), *tk.LIMITS.T))
            if best is None or ep + er < best[0] + best[1]:
                best = (ep, er, q)
        solved = best[0] < 1e-3 and best[1] < 1e-3
        ok_free = False
        if solved:
            model.reset_home(best[2])
            ok_free = not model.internal_contacts()
        kin += solved
        free += ok_free
        rows.append(dict(dpos=dpos, drpy=drpy, kinematic=bool(solved), collision_free=bool(ok_free)))
    model.reset_home()
    n = len(targets)
    print()
    print("=" * 100)
    print("4. 可达性口径 (C3): 运动学可达 vs 无碰撞可达 vs 策略实际")
    print("=" * 100)
    print(f"  运动学可达 {kin}/{n} ({kin / n * 100:.0f}%)")
    print(f"  无碰撞可达 {free}/{n} ({free / n * 100:.0f}%), 占运动学可达的 {free / max(kin, 1) * 100:.0f}%")
    return dict(n=n, kinematic=kin, collision_free=free, rows=rows)


def viewer(model: Model) -> None:
    """Show the collision geometry: group 3 on, visual meshes off."""
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model.m, model.d) as v:
        v.opt.geomgroup[:] = [0, 0, 0, 1, 0, 0]
        v.sync()
        print("collision geometry only (group 3). Close the window to exit.")
        while v.is_running():
            mujoco.mj_forward(model.m, model.d)
            v.sync()
    # the viewer's teardown segfaults on this machine (stage 4.6); leave the process before it
    os._exit(0)


def main() -> None:
    model = Model()
    payload = {"model": model_checks(model)}
    payload["collision_free_axes"] = collision_free_axis_limits(model)

    if args.viewer:
        viewer(model)

    if args.policy:
        sim = teleop.Sim(
            str(ROOT / args.policy if not Path(args.policy).is_absolute() else args.policy), teleop.BaseCommandFilter()
        )
        payload["scenarios"] = policy_scenarios(sim)
        payload["reachability"] = collision_aware_reachability(model, policy_scenarios.targets)

    out = OUT / ("self_collision.json" if not args.policy else f"self_collision_{Path(args.policy).stem}.json")
    out.write_text(json.dumps(payload, indent=1, default=float))
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
