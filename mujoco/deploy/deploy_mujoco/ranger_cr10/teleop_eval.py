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

"""Stage 4.6: the headless Sim-to-Sim protocol, three modes, filter on and off.

Three suites, each a re-implementation of a protocol that already exists on the Isaac Lab
side, so the two simulators can be compared without anyone redefining a metric:

``drift``
    Exactly ``scripts/ranger_cr10/check_base_drift.py``: the five scenarios (static,
    static_near, low, normal, switch), the three filter configurations and the 30 s static
    hold with the arm moving. Comparable with ``outputs/base_drift/db00*/base_drift.json``.
``modes``
    The three modes of the acceptance protocol -- chassis only, arm only, both -- with the
    end effector either held or commanded along one axis at a time, which is what the
    keyboard produces.
``convergence``
    The C2 protocol of ``check_teleop_stage45i.py``: the same 14 fixed commands (10
    single-axis, 4 held-out from ``train_mix21.json``), the actor driving from wherever it
    is for 2.5 s, plus the C2b case where the chassis moves at the same time.

Nothing here trains, filters by hand or rescales anything: the policy runs exactly as it
does in ``ranger_cr10_deploy.py``.

Usage:
    python mujoco/deploy/deploy_mujoco/ranger_cr10/teleop_eval.py --suite all
    python .../teleop_eval.py --suite drift --out outputs/stage46_mujoco
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEPLOY = HERE.parent.parent
ROOT = DEPLOY.parent.parent
for p in (str(HERE), str(DEPLOY), str(ROOT / "scripts" / "ranger_cr10")):
    if p not in sys.path:
        sys.path.append(p)

import teleop_kinematics as tk  # noqa: E402
from base_command_filter import BaseCommandFilter  # noqa: E402
from deploy_mujoco.ranger_cr10.ranger_policy import (  # noqa: E402
    ACTION_SCALE,
    ARM_DEFAULT,
    ARM_JOINT_NAMES,
    STEERING_JOINT_NAMES,
    TCP_DEFAULT_POS_B,
    TCP_DEFAULT_QUAT_B,
    WHEEL_JOINT_NAMES,
    ChassisAction,
    EndEffector,
    RangerPolicyObs,
    act,
    ee_command_from_offset,
    load_actor,
)

import mujoco  # noqa: E402

DEFAULT_POLICY = "logs/rsl_rl/ranger_cr10_teleop/2026-10-29_19-12-08_stage45j_ppo/model_5999.pt"
RUN_TAG = ""
SCENE = ROOT / "mujoco" / "robots" / "ranger_cr10" / "scene.xml"
TRAIN_RECORD = ROOT / "logs/rsl_rl/ranger_cr10_teleop/_stage45i/train_mix21.json"


class Sim:
    """The MuJoCo robot plus the policy, driven one 10 Hz step at a time."""

    def __init__(self, policy_path: str, filt: BaseCommandFilter, wheel_frictionloss: float | None = None):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        if wheel_frictionloss is not None:
            for name in WHEEL_JOINT_NAMES:
                jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.m.dof_frictionloss[self.m.jnt_dofadr[jid]] = wheel_frictionloss
        self.d = mujoco.MjData(self.m)
        self.decimation = round(0.1 / self.m.opt.timestep)
        self.actor, self.kind = load_actor(policy_path)
        self.obs = RangerPolicyObs(self.m, self.d)
        self.ee = EndEffector(self.m, self.d)
        self.chassis = ChassisAction()
        self.filt = filt

        def cid(name):
            return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

        self.steer_ids = [cid(f"{n}_pos") for n in STEERING_JOINT_NAMES]
        self.wheel_ids = [cid(f"{n}_vel") for n in WHEEL_JOINT_NAMES]
        self.arm_ids = [cid(f"{n}_pos") for n in ARM_JOINT_NAMES]
        self.base_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.home = self.m.key("home").id

        self.ee_cmd = ee_command_from_offset((0, 0, 0), (0, 0, 0))
        self.vel_cmd = (0.0, 0.0)
        self.last_action = np.zeros(8)
        self.path = 0.0
        self.t = 0.0

    def set_slope_deg(self, slope_deg: float) -> None:
        """Tilt gravity instead of the ground: the same dynamics, no geometry surgery."""
        a = np.radians(slope_deg)
        self.m.opt.gravity[:] = [9.81 * np.sin(a), 0.0, -9.81 * np.cos(a)]

    def set_wheel_friction(self, mu: float) -> None:
        """Set sliding friction on the wheel geoms.

        The floor's value does nothing here on purpose: the wheel geoms carry `priority=1`
        (so wheel-ground contacts use the wheels' own coefficient, which is how the task pins
        the tyre to 0.35). A slippery floor is therefore modelled by lowering the *wheels'*
        coefficient -- measured: writing it on the floor geom changed nothing at all.
        """
        for name in WHEEL_JOINT_NAMES:
            gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, name.replace("_joint", "_link_col"))
            self.m.geom_friction[gid] = [mu, 0.005, 0.0001]

    def reset(self) -> None:
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.home)
        mujoco.mj_forward(self.m, self.d)
        self.ee_cmd = ee_command_from_offset((0, 0, 0), (0, 0, 0))
        self.vel_cmd = (0.0, 0.0)
        self.last_action = np.zeros(8)
        self.filt = BaseCommandFilter(
            self.filt.vx_deadband, self.filt.wz_deadband, self.filt.lock_output, self.filt.hysteresis
        )
        vx_t, wz_t = self.filt.filter_command(*[np_to_t(x) for x in self.vel_cmd])
        self.obs.reset(self.last_action, (float(vx_t), float(wz_t)), self.ee_cmd)
        self.path = 0.0
        self.t = 0.0

    def set_commands(self, vx: float, wz: float, ee_cmd) -> None:
        self.vel_cmd = (vx, wz)
        self.ee_cmd = np.asarray(ee_cmd, dtype=np.float64)

    def step(self) -> dict:
        vx_t, wz_t = self.filt.filter_command(np_to_t(self.vel_cmd[0]), np_to_t(self.vel_cmd[1]))
        vx_eff, wz_eff = float(vx_t[0]), float(wz_t[0])

        obs = self.obs.push(self.last_action, (vx_eff, wz_eff), self.ee_cmd)
        raw = act(self.actor, obs)
        a_t = np_to_t(raw, batch=True)
        raw = self.filt.filter_action(a_t, vx_t, wz_t)[0].numpy().astype(np.float64)

        base = self.chassis.apply(raw[:2])
        self.d.ctrl[self.steer_ids] = base["steering_target"]
        self.d.ctrl[self.wheel_ids] = base["wheel_speed_target"]
        self.d.ctrl[self.arm_ids] = ARM_DEFAULT + ACTION_SCALE * raw[2:]

        x0 = self.d.qpos[0:2].copy()
        for _ in range(self.decimation):
            mujoco.mj_step(self.m, self.d)
        self.path += float(np.linalg.norm(self.d.qpos[0:2] - x0))
        self.t += self.decimation * self.m.opt.timestep
        self.last_action = raw

        mujoco.mj_forward(self.m, self.d)
        lin, ang = self.obs.base_velocity_local()
        pos_err, ori_err = self.ee.errors(self.ee_cmd)
        return dict(
            t=self.t,
            vx_cmd=vx_eff,
            wz_cmd=wz_eff,
            raw_vx=float(raw[0]),
            raw_wz=float(raw[1]),
            proc_vx=base["vx"],
            proc_wz=base["wz"],
            act_vx=float(lin[0]),
            act_vy=float(lin[1]),
            act_wz=float(ang[2]),
            base_x=float(self.d.qpos[0]),
            base_y=float(self.d.qpos[1]),
            yaw=float(
                math.atan2(
                    2 * (self.d.qpos[3] * self.d.qpos[6] + self.d.qpos[4] * self.d.qpos[5]),
                    1 - 2 * (self.d.qpos[5] ** 2 + self.d.qpos[6] ** 2),
                )
            ),
            ee_pos=pos_err,
            ee_ori=ori_err,
            path=self.path,
        )


def np_to_t(x, batch: bool = False):
    import torch

    v = np.asarray(x, dtype=np.float32)
    t = torch.from_numpy(v)
    return t.unsqueeze(0) if batch else t.reshape(-1)


def summarise(rows) -> dict:
    a = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    n = len(rows)

    dx = a["base_x"][-1] - a["base_x"][n // 2]
    dy = a["base_y"][-1] - a["base_y"][n // 2]
    return dict(
        steps=n,
        cmd_vx=float(np.mean(a["vx_cmd"])),
        cmd_wz=float(np.mean(a["wz_cmd"])),
        proc_vx_abs=float(np.mean(np.abs(a["proc_vx"]))),
        proc_wz_abs=float(np.mean(np.abs(a["proc_wz"]))),
        act_speed=float(np.mean(np.hypot(a["act_vx"], a["act_vy"]))),
        act_vx=float(np.mean(a["act_vx"])),
        act_wz_abs=float(np.mean(np.abs(a["act_wz"]))),
        drift_m=float(math.hypot(dx, dy)),
        path_m=float(a["path"][-1]),
        drift_yaw_deg=float(math.degrees(a["yaw"][-1] - a["yaw"][n // 2])),
        ee_pos=float(np.mean(a["ee_pos"])),
        ee_ori=float(np.mean(a["ee_ori"])),
        ee_pos_tail=float(np.mean(a["ee_pos"][-5:])),
        ee_ori_tail=float(np.mean(a["ee_ori"][-5:])),
        vx_err=float(np.mean(np.abs(a["act_vx"] - a["vx_cmd"]))),
        wz_err=float(np.mean(np.abs(a["act_wz"] - a["wz_cmd"]))),
        vel_rms=float(math.sqrt(np.mean(a["act_vx"] ** 2 + a["act_vy"] ** 2))),
        yaw_total_deg=float(math.degrees(a["yaw"][-1] - a["yaw"][0])),
    )


EE_TARGETS = [
    (0.0, 0.0, 0.0),
    (0.08, 0.0, 0.0),
    (0.0, 0.10, 0.0),
    (0.0, 0.0, 0.08),
    (-0.08, -0.10, 0.0),
    (0.0, 0.0, 0.0),
]
EE_ATT = [(0.0, 0.0, 0.0), (0.15, 0.0, 0.0), (0.0, 0.15, 0.0), (0.0, 0.0, 0.2), (0.1, 0.1, -0.1), (0.0, 0.0, 0.0)]


def drift_suite(sim: Sim, out: Path, static_steps: int = 300, move_steps: int = 150):
    """The five scenarios of check_base_drift.py, same step counts and targets."""
    S, M = static_steps, move_steps
    schedules = {
        "static": [(0.0, 0.0, i, (S // 6) + (1 if i < S % 6 else 0)) for i in range(6)],
        "static_near": [(0.01, 0.012, 0, S)],
        "low": [(0.05, 0.03, 0, M)],
        "normal": [(0.25, 0.15, 0, M)],
        "switch": [(0.0, 0.0, 0, M // 3), (0.20, 0.10, 0, M // 3), (0.0, 0.0, 0, M // 3), (0.20, 0.10, 0, M // 3)],
    }
    configs = [
        ("none", dict(vx_deadband=0.0, wz_deadband=0.0, lock_output=False)),
        ("input", dict(vx_deadband=0.02, wz_deadband=0.02, lock_output=False)),
        ("input+lock", dict(vx_deadband=0.02, wz_deadband=0.02, lock_output=True)),
    ]

    results = []
    print("=" * 108)
    print("drift suite (与 Isaac check_base_drift.py 同一组场景/步数/目标)")
    print("=" * 108)
    for cfg_name, kw in configs:
        sim.filt = BaseCommandFilter(**kw)
        for scen, sched in schedules.items():
            sim.reset()
            rows = []
            for vx, wz, ee_idx, steps in sched:
                sim.set_commands(vx, wz, ee_command_from_offset(EE_TARGETS[ee_idx], EE_ATT[ee_idx]))
                for _ in range(steps):
                    rows.append(sim.step())
            s = summarise(rows)
            s.update(config=cfg_name, scenario=scen)
            results.append(s)
            with open(out / f"trace_{RUN_TAG}{cfg_name}_{scen}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(
                f"  {cfg_name:<11}{scen:<12} proc|vx| {s['proc_vx_abs']:.4f} act_vx {s['act_vx']:+.4f} "
                f"speed {s['act_speed']:.4f} |wz| {s['act_wz_abs']:.4f}  drift {s['drift_m'] * 1000:6.1f} mm "
                f"path {s['path_m'] * 1000:6.1f} mm  yaw {s['drift_yaw_deg']:+.2f}°  "
                f"EE {s['ee_pos'] * 1000:5.1f} mm / {math.degrees(s['ee_ori']):4.2f}°"
            )
    return results


def modes_suite(sim: Sim, out: Path):
    """Mode A (chassis only), B (arm only), C (both), 6 s or 4 s per command."""
    A = 60  # 6 s
    B = 40  # 4 s
    zero_ee = ee_command_from_offset((0, 0, 0), (0, 0, 0))
    cases = [
        ("A1 前进 0.25", (0.25, 0.0), zero_ee, A),
        ("A2 后退 -0.25", (-0.25, 0.0), zero_ee, A),
        ("A3 原地转 0.15", (0.0, 0.15), zero_ee, A),
        ("A4 低速 0.05/0.03", (0.05, 0.03), zero_ee, A),
        ("B1 X +0.10", (0.0, 0.0), ee_command_from_offset((0.10, 0, 0), (0, 0, 0)), B),
        ("B2 Y -0.10", (0.0, 0.0), ee_command_from_offset((0, -0.10, 0), (0, 0, 0)), B),
        ("B3 Z +0.08", (0.0, 0.0), ee_command_from_offset((0, 0, 0.08), (0, 0, 0)), B),
        ("B4 roll +0.20", (0.0, 0.0), ee_command_from_offset((0, 0, 0), (0.20, 0, 0)), B),
        ("B5 pitch -0.20", (0.0, 0.0), ee_command_from_offset((0, 0, 0), (0, -0.20, 0)), B),
        ("B6 yaw +0.25", (0.0, 0.0), ee_command_from_offset((0, 0, 0), (0, 0, 0.25)), B),
        ("B7 组合位姿", (0.0, 0.0), ee_command_from_offset((0.06, 0.06, 0.03), (0.12, -0.10, 0.15)), B),
        ("C1 vx0.25 wz0.15 roll+0.2", (0.25, 0.15), ee_command_from_offset((0, 0, 0), (0.20, 0, 0)), A),
        ("C2 vx0.25 wz-0.12 六维", (0.25, -0.12), ee_command_from_offset((0.05, -0.05, 0.04), (0.10, 0.10, -0.15)), A),
        ("C3 vx-0.20 wz0.15 六维", (-0.20, 0.15), ee_command_from_offset((-0.05, 0.04, -0.03), (-0.10, 0.12, 0.20)), A),
    ]
    results = []
    print()
    print("=" * 108)
    print("modes suite (A 底盘 / B 末端 / C 联合; 每段命令保持 6 s 或 4 s)")
    print("=" * 108)
    print(
        f"  {'case':<28}{'filter':<12}{'vx err':>8}{'wz err':>8}{'base |v|':>10}"
        f"{'EE pos':>9}{'EE att':>9}{'drift':>9}{'path':>9}{'yaw':>8}"
    )
    for use in (False, True):
        sim.filt = BaseCommandFilter(0.02 if use else 0.0, 0.02 if use else 0.0, use)
        for name, (vx, wz), ee_cmd, steps in cases:
            sim.reset()
            sim.set_commands(vx, wz, ee_cmd)
            rows = [sim.step() for _ in range(steps)]
            s = summarise(rows)
            s.update(case=name, filter=("on" if use else "off"))
            results.append(s)
            print(
                f"  {name:<28}{('on' if use else 'off'):<12}{s['vx_err']:>8.4f}{s['wz_err']:>8.4f}"
                f"{s['act_speed']:>10.4f}{s['ee_pos_tail'] * 1000:>8.1f}mm"
                f"{math.degrees(s['ee_ori_tail']):>8.2f}°{s['drift_m'] * 1000:>8.1f}mm"
                f"{s['path_m'] * 1000:>8.1f}mm{s['yaw_total_deg']:>8.2f}°"
            )
    return results


def terrain_suite(sim: Sim, out: Path, static_steps: int = 300):
    """Does the zero-lock survive a slope or a slippery floor? (Task C's follow-up.)

    The lock hands "stay where you are" entirely to the tyres, so the case to test is the one
    where the tyres have something to be pushed against. Gravity is tilted rather than the
    ground, and the wheel coefficient is the knob that actually matters (the floor's is
    overridden by the wheels' `priority=1` contact parameter).
    """
    print()
    print("=" * 108)
    print("terrain suite (static 场景 30 s, 手臂持续运动; 坡度用倾斜重力实现)")
    print("=" * 108)
    print(
        f"  {'坡度':<8}{'轮-地 mu':<10}{'配置':<8}{'净漂移':>10}"
        f"{'路径长度':>10}{'|v|':>9}{'偏航':>9}{'EE 位置/姿态':>16}"
    )
    results = []
    for slope in (0.0, 3.0, 5.0):
        for mu in (0.35, 0.20, 0.10):
            for cfg_name, kw in (
                ("lock", dict(vx_deadband=0.02, wz_deadband=0.02, lock_output=True)),
                ("none", dict(vx_deadband=0.0, wz_deadband=0.0, lock_output=False)),
            ):
                sim.set_slope_deg(slope)
                sim.set_wheel_friction(mu)
                sim.filt = BaseCommandFilter(**kw)
                sim.reset()
                rows = []
                for i in range(6):
                    sim.set_commands(0.0, 0.0, ee_command_from_offset(EE_TARGETS[i], EE_ATT[i]))
                    rows += [sim.step() for _ in range(static_steps // 6)]
                s = summarise(rows)
                s.update(slope_deg=slope, wheel_mu=mu, config=cfg_name)
                results.append(s)
                print(
                    f"  {slope:<8.0f}{mu:<10}{cfg_name:<8}{s['drift_m'] * 1000:>9.1f}mm"
                    f"{s['path_m'] * 1000:>9.1f}mm{s['act_speed']:>9.4f}{s['drift_yaw_deg']:>8.2f}°"
                    f"{s['ee_pos'] * 1000:>10.1f}mm / {math.degrees(s['ee_ori']):.2f}°"
                )
    sim.set_slope_deg(0.0)
    sim.set_wheel_friction(0.35)
    return results


def _ws_module():
    import importlib.util

    f = ROOT / "scripts" / "ranger_cr10" / "workspace_limits.py"
    spec = importlib.util.spec_from_file_location("ranger_workspace_limits", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


WS = _ws_module()


def _reach_oracle():
    """The same reachability table the training curriculum uses, as a Python predicate.

    Loaded from ``scripts/ranger_cr10/reach_table_48.json`` rather than recomputed: the point
    of the evaluation is to separate "the arm cannot hold this" from "the policy does not
    hold this", and both sides have to agree on the first question. Returns a callable that
    answers ``True`` when the target is inside the cell's reachable position range, or
    ``None`` when the table is missing (then the report falls back to the unfiltered numbers).
    """
    path = Path(WS.__file__).resolve().parent / "reach_table_48.json"
    if not path.is_file():
        return lambda dpos, drpy: True
    table = json.loads(path.read_text())
    grid = {k: np.array(table[k]) for k in ("roll", "pitch", "yaw")}
    limits = np.array(table["limits"])

    def reach(dpos, drpy):
        idx = [int(np.argmin(np.abs(grid[k] - drpy[i]))) for i, k in enumerate(("roll", "pitch", "yaw"))]
        lo = limits[idx[0], idx[1], idx[2], 0::2]
        hi = limits[idx[0], idx[1], idx[2], 1::2]
        return bool(np.all(np.asarray(dpos) >= lo) and np.all(np.asarray(dpos) <= hi))

    return reach


def _collision_free_oracle(sim, targets, tries: int = 12):
    """Stage 4.9: is each target executable *without* the arm touching the chassis equipment?

    The kinematic table above only says a joint configuration exists. After the collision
    model was completed, "reachable" and "safely reachable" stopped being the same set, so the
    coverage report has to carry both: success on every target, and success on the targets
    that some collision-free configuration can hold. Anything else would either overstate the
    policy (counting impossible targets as its fault) or excuse it (dropping them silently).
    """
    import mujoco

    protected = ("lidar_link0", "d435_link", "base_link")
    prot_ids = {mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in protected}
    m = sim.m
    d = mujoco.MjData(m)
    arm_adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)]
    rng = np.random.default_rng(0)
    P0, R0 = tk.P_NEUTRAL, tk.R_NEUTRAL

    def free(dpos, drpy):
        R = tk.euler_R(drpy) @ R0
        for i in range(tries):
            guess = (
                np.clip(tk.DEFAULT_Q + rng.normal(0, 0.5, 6), *tk.LIMITS.T)
                if i < 3
                else rng.uniform(tk.LIMITS[:, 0], tk.LIMITS[:, 1])
            )
            ep, er, q = tk.ik(P0 + np.asarray(dpos), R, guess)
            if ep > 1e-3 or er > 1e-3:
                continue
            mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
            for a, v in zip(arm_adr, q):
                d.qpos[a] = v
            mujoco.mj_forward(m, d)
            hit = False
            for c in range(d.ncon):
                b1, b2 = m.geom_bodyid[d.contact[c].geom1], m.geom_bodyid[d.contact[c].geom2]
                if b1 == 0 or b2 == 0:
                    continue
                if b1 in prot_ids or b2 in prot_ids:
                    hit = True
                    break
            if not hit:
                return True
        return False

    return [free(dpos, drpy) for dpos, drpy in targets]


def workspace_suite(
    sim: Sim,
    out: Path,
    hold: int = 30,
    pos_tol: float = 0.030,
    ori_tol_deg: float = 2.0,
    scan_box: str = "profile",
):
    """E1: how far along each axis does the policy hold the reference accuracy?

    Single axis at a time, held for ``hold`` policy steps (3 s), error taken from the last
    five steps. The largest offset that stays under the thresholds is the *usable* range --
    which is the number the keyboard clamp should reflect, not the box the sampler drew from.

    ``scan_box="4.8"`` sweeps past the keyboard clamps to the candidate box, which is how the
    4.8 clamps were measured in the first place; with ``"profile"`` the sweep stops where the
    keyboard stops.
    """
    scan = {
        "profile": WS.keyboard_box(),
        "4.8": WS.TARGET_BOX_48,
        "4.10": WS.TARGET_BOX_410,
    }[scan_box]
    print()
    print("=" * 108)
    print(f"workspace suite (单轴, 每档保持 {hold} 步; 判据 位置 {pos_tol * 1000:.0f} mm / 姿态 {ori_tol_deg:.1f}°)")
    print("=" * 108)
    axes = {
        "X": (0, "pos_x"),
        "Y": (1, "pos_y"),
        "Z": (2, "pos_z"),
        "Roll": (0, "roll"),
        "Pitch": (1, "pitch"),
        "Yaw": (2, "yaw"),
    }
    results = []
    print(f"  {'轴':<6}{'方向':<6}{'扫掠范围':>16}{'稳定跟踪到':>12}{'该点误差':>14}   首个失败点")
    for name, (idx, key) in axes.items():
        lo, hi = scan[key]
        step = 0.05 if name in ("X", "Y", "Z") else math.radians(10)
        unit = "m" if name in ("X", "Y", "Z") else "°"
        for sign, edge in ((1, hi), (-1, lo)):
            stable, last_err = 0.0, (0.0, 0.0)
            bad_at, bad_err = None, (0.0, 0.0)
            curve = []
            mag = step
            while mag <= abs(edge) + 1e-9:
                sim.reset()
                dpos = [0.0, 0.0, 0.0]
                drpy = [0.0, 0.0, 0.0]
                if name in ("X", "Y", "Z"):
                    dpos[idx] = sign * mag
                else:
                    drpy[idx] = sign * mag
                sim.set_commands(0.0, 0.0, ee_command_from_offset(dpos, drpy))
                rows = [sim.step() for _ in range(hold)]
                pe = float(np.mean([r["ee_pos"] for r in rows[-5:]]))
                oe = math.degrees(float(np.mean([r["ee_ori"] for r in rows[-5:]])))
                curve.append((sign * mag, pe, oe))
                if pe <= pos_tol and oe <= ori_tol_deg:
                    stable = sign * mag
                    last_err = (pe, oe)
                elif bad_at is None:
                    bad_at, bad_err = sign * mag, (pe, oe)
                # the whole grid is evaluated, not stopped at the first failure: the curves are
                # what the report compares, and a truncated curve cannot be re-thresholded
                mag += step
            shown = stable if name in ("X", "Y", "Z") else math.degrees(stable)
            rng = f"{lo:.2f}..{hi:.2f}" if unit == "m" else f"{math.degrees(lo):.0f}..{math.degrees(hi):.0f}"
            bad_txt = (
                f"{bad_at:+.3f}{unit} -> {bad_err[0] * 1000:.0f}mm/{bad_err[1]:.2f}°"
                if bad_at is not None
                else "整段都在界内"
            )
            # 4.7's roll+ curve failed around 10-20 deg and then came back under the threshold
            # by 40 deg. Reporting only the furthest passing point would have hidden the bad
            # band, so a curve that recovers after failing is flagged rather than averaged.
            first_bad_i = next((i for i, c in enumerate(curve) if c[1] > pos_tol or c[2] > ori_tol_deg), None)
            dip = first_bad_i is not None and any(c[1] <= pos_tol and c[2] <= ori_tol_deg for c in curve[first_bad_i:])
            if dip:
                n_bad = sum(1 for c in curve if c[1] > pos_tol or c[2] > ori_tol_deg)
                bad_txt += f"  [非单调: {n_bad}/{len(curve)} 档越界后回升]"
            print(
                f"  {name:<6}{'+' if sign > 0 else '-':<6}{rng:>14}{unit}{shown:>10.3f}{unit}"
                f"{last_err[0] * 1000:>7.0f}mm{last_err[1]:>6.2f}°   {bad_txt}"
            )
            results.append(
                dict(
                    axis=name,
                    sign=sign,
                    box_lo=lo,
                    box_hi=hi,
                    stable=stable,
                    err_pos=last_err[0],
                    err_ori=last_err[1],
                    first_bad=bad_at,
                    first_bad_err=bad_err,
                    nonmonotonic=dip,
                    curve=curve,
                )
            )
    return results


def coverage_suite(sim: Sim, out: Path, n: int = 60, hold: int = 30, seed: int = 5, box_name: str = "candidate"):
    """E2: six-dimensional coverage -- what fraction of the new box is actually usable.

    Single-axis reach does not imply the corners work: the reachability study measured 65% of
    the candidate box solvable with IK, and the coupling between attitude and position is what
    drops the rest. This draws whole six-dimensional targets and reports the error
    distribution plus the joint-limit and arm-ground-contact rates.
    """
    import mujoco

    m, d = sim.m, sim.d
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
    limits = np.array(
        [[-3.92, 0.94], [-1.570, 1.570], [-2.860, 2.860], [-3.140, 3.140], [-3.140, 3.140], [-3.140, 3.140]]
    )
    rng = np.random.default_rng(seed)
    # ``candidate`` is the box the 4.7 brief asked for (comparable across policies and with the
    # stage-4.6 baseline); ``keyboard`` is the smaller box the deployed keyboard allows. The
    # ``4.8*`` names are the stage-4.8 boxes, so a policy can be compared on old and new
    # targets with the same protocol.
    boxes = {
        "candidate": WS.TARGET_BOX,
        "keyboard": WS.KEYBOARD_BOX,
        "4.8": WS.TARGET_BOX_48,
        "4.8_keyboard": WS.KEYBOARD_BOX_48,
        "4.9_keyboard": WS.KEYBOARD_BOX_49,
        "4.10": WS.TARGET_BOX_410,
        "4.10_keyboard": WS.KEYBOARD_BOX_410,
    }
    box = boxes[box_name]
    reach = _reach_oracle()
    print()
    print("=" * 108)
    print(f"coverage suite ({n} 个六维目标, 从 {box_name} 盒均匀抽取, 每个保持 {hold} 步)")
    print("=" * 108)
    rows_out, pos_e, ori_e = [], [], []
    reachable_flag = []
    targets = []
    at_limit = 0
    n_contact = 0
    for _ in range(n):
        dpos = [rng.uniform(*box[k]) for k in ("pos_x", "pos_y", "pos_z")]
        drpy = [rng.uniform(*box[k]) for k in ("roll", "pitch", "yaw")]
        targets.append((dpos, drpy))
        sim.reset()
        sim.set_commands(0.0, 0.0, ee_command_from_offset(dpos, drpy))
        rows = [sim.step() for _ in range(hold)]
        pe = float(np.mean([r["ee_pos"] for r in rows[-5:]]))
        oe = math.degrees(float(np.mean([r["ee_ori"] for r in rows[-5:]])))
        pos_e.append(pe)
        ori_e.append(oe)
        reachable_flag.append(reach(dpos, drpy))
        q = np.array([d.qpos[a] for a in arm_adr])
        if np.any(np.abs(q - limits[:, 0]) < 0.02) or np.any(np.abs(q - limits[:, 1]) < 0.02):
            at_limit += 1
        if any((d.contact[c].geom1 in arm_geoms) != (d.contact[c].geom2 in arm_geoms) for c in range(d.ncon)):
            n_contact += 1
        rows_out.append(dict(dpos=dpos, drpy=drpy, pos_err=pe, ori_err_deg=oe))
    pos_e = np.array(pos_e)
    ori_e = np.array(ori_e)
    reach_arr = np.array(reachable_flag, dtype=bool)
    print(
        f"  可达性: {reach_arr.mean() * 100:.1f}% 的目标在可达集内 ({int(reach_arr.sum())}/{n}), "
        "下面同时报全部目标与可达目标两个口径"
    )
    cfree = np.array(_collision_free_oracle(sim, targets), dtype=bool)
    print(
        f"  无碰撞可达: {cfree.mean() * 100:.1f}% ({int(cfree.sum())}/{n})  "
        f"—— 与运动学可达各报一列, 两者不同说明有目标只能穿过雷达完成"
    )
    tiers = (("30 mm / 2°", 0.030, 2.0), ("50 mm / 5°", 0.050, 5.0), ("80 mm / 10°", 0.080, 10.0))
    for tag, pos_lim, ori_lim in tiers:
        ok = (pos_e <= pos_lim) & (ori_e <= ori_lim)
        ok_r = ok & reach_arr
        ok_c = ok & cfree
        cond = ok_r.sum() / max(int(reach_arr.sum()), 1)
        cond_c = ok_c.sum() / max(int(cfree.sum()), 1)
        print(
            f"  精度档 {tag:<12}: 全部目标 {ok.mean() * 100:5.1f}%   运动学可达 {cond * 100:5.1f}%   "
            f"无碰撞可达 {cond_c * 100:5.1f}%"
        )
    print(f"  位置误差 中位 {np.median(pos_e) * 1000:.0f} mm, p90 {np.percentile(pos_e, 90) * 1000:.0f} mm")
    print(f"  姿态误差 中位 {np.median(ori_e):.1f}°, p90 {np.percentile(ori_e, 90):.1f}°")
    if reach_arr.any() and (~reach_arr).any():
        print(
            f"  可达目标误差 中位 {np.median(pos_e[reach_arr]) * 1000:.0f} mm / "
            f"{np.median(ori_e[reach_arr]):.1f}°; 不可达目标 "
            f"{np.median(pos_e[~reach_arr]) * 1000:.0f} mm / {np.median(ori_e[~reach_arr]):.1f}°"
        )
    print(f"  关节接近限位(<0.02 rad)的样本 {at_limit}/{n}; 手臂与地面/货箱接触 {n_contact}/{n}")
    return dict(
        rows=rows_out,
        reachable=reachable_flag,
        success={
            t: float(((pos_e <= pos_lim) & (ori_e <= ori_lim)).mean())
            for t, pos_lim, ori_lim in (("30mm2deg", 0.030, 2.0), ("50mm5deg", 0.050, 5.0), ("80mm10deg", 0.080, 10.0))
        },
        success_reachable={
            t: float(((pos_e <= pos_lim) & (ori_e <= ori_lim) & reach_arr).sum() / max(int(reach_arr.sum()), 1))
            for t, pos_lim, ori_lim in (("30mm2deg", 0.030, 2.0), ("50mm5deg", 0.050, 5.0), ("80mm10deg", 0.080, 10.0))
        },
        reachable_fraction=float(reach_arr.mean()),
        collision_free=cfree.tolist(),
        collision_free_fraction=float(cfree.mean()),
        success_collision_free={
            t: float(((pos_e <= pos_lim) & (ori_e <= ori_lim) & cfree).sum() / max(int(cfree.sum()), 1))
            for t, pos_lim, ori_lim in (("30mm2deg", 0.030, 2.0), ("50mm5deg", 0.050, 5.0), ("80mm10deg", 0.080, 10.0))
        },
        pos_median=float(np.median(pos_e)),
        ori_median=float(np.median(ori_e)),
        at_limit=at_limit,
        contact=n_contact,
    )


def lowspace_suite(
    sim: Sim, out: Path, hold: int = 20, step: float = 0.05, floor: float = -0.40, pos_tol=0.05, ori_tol=5.0
):
    """Stage 4.10 D2: how far down can the tool actually be driven, per (x, y) column?

    Stage 4.9 shipped one number for the whole box (Z >= -0.05) because the lidar sits under
    the default pose. With the tool out in front the answer depends on where it is, so this
    walks each column downwards and records the lowest offset the *policy* still holds within
    50 mm / 5 deg without touching anything -- the same criterion the keyboard uses.
    """
    ws = WS
    box = ws.TARGET_BOX_410
    xs = np.linspace(box["pos_x"][0] * 0.5, box["pos_x"][1] * 0.6, 3)
    ys = np.linspace(box["pos_y"][0] * 0.6, box["pos_y"][1] * 0.6, 3)
    zero = (0.0, 0.0, 0.0)
    print()
    print("=" * 100)
    print(
        f"lowspace suite (每列从默认高度向下走 {step * 100:.0f} cm, 每档保持 {hold} 步; "
        f"判据 {pos_tol * 1000:.0f} mm / {ori_tol:.0f}°)"
    )
    print("=" * 100)
    print(f"  {'x':>6}{'y':>7}{'最低可用 z':>12}{'该点误差':>16}{'最小间距':>10}{'碰撞':>6}")
    rows = []
    for y in ys:
        for x in xs:
            sim.reset()
            best, last, gap, hit = 0.0, (0.0, 0.0), 1.0, 0
            z = 0.0
            while z >= floor - 1e-9:
                sim.set_commands(0.0, 0.0, ee_command_from_offset((x, y, z), zero))
                rows_step = [sim.step() for _ in range(hold)]
                pe = float(np.mean([r["ee_pos"] for r in rows_step[-5:]]))
                oe = math.degrees(float(np.mean([r["ee_ori"] for r in rows_step[-5:]])))
                gap = min(gap, _clearance_coll(sim))
                hit += _has_internal_contact(sim)
                if pe <= pos_tol and oe <= ori_tol:
                    best, last = z, (pe, oe)
                z -= step
            print(
                f"  {x:+.2f}{y:+.2f}{best:+.2f} m{last[0] * 1000:>8.0f} mm{last[1]:>6.1f}°{gap * 1000:>8.0f} mm{hit:>6}"
            )
            rows.append(
                dict(x=float(x), y=float(y), z_min=best, err_pos=last[0], err_ori=last[1], clearance=gap, contacts=hit)
            )
    return rows


def _collision_geoms(sim):
    m = sim.m
    key = "_coll_geoms_sets"
    if not hasattr(sim, key):
        prot = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("lidar_link0", "d435_link", "base_link")}
        arm = {
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
        a = [g for g in range(m.ngeom) if m.geom_bodyid[g] in arm and m.geom_contype[g]]
        p = [g for g in range(m.ngeom) if m.geom_bodyid[g] in prot and m.geom_contype[g]]
        setattr(sim, key, (a, p))
    return getattr(sim, key)


def _clearance_coll(sim, cap=0.30):
    a, p = _collision_geoms(sim)
    best = cap
    for g1 in a:
        for g2 in p:
            dv = mujoco.mj_geomDistance(sim.m, sim.d, g1, g2, best, None)
            if dv < best:
                best = dv
    return best


def _has_internal_contact(sim) -> int:
    m, d = sim.m, sim.d
    for c in range(d.ncon):
        if m.geom_bodyid[d.contact[c].geom1] and m.geom_bodyid[d.contact[c].geom2]:
            return 1
    return 0


def path_suite(sim: Sim, out: Path, pos_rate: float = 0.10, ori_every: int = 3, hold: int = 20, scan_box: str = "4.8"):
    """E3 (stage 4.8): move the target *continuously*, the way a keyboard does.

    The workspace suite holds each endpoint still and reads the steady-state error; an
    operator never does that. Here the target walks from neutral to the edge at the keyboard's
    own rate -- 0.10 m/s for position, one 5 deg press every ``ori_every`` policy steps for
    attitude -- and the report separates the error *while moving* (tracking lag) from the
    error after the target stops (convergence). A policy can be excellent at the second and
    useless at the first, and only the second is visible in a static test.
    """
    axes = {
        "X": (0, "pos_x"),
        "Y": (1, "pos_y"),
        "Z": (2, "pos_z"),
        "Roll": (0, "roll"),
        "Pitch": (1, "pitch"),
        "Yaw": (2, "yaw"),
    }
    dt = 0.1  # one policy step
    print()
    print("=" * 108)
    print(f"path suite (目标连续移动: 位置 {pos_rate:.2f} m/s, 姿态每 {ori_every} 步 5°, 到位后保持 {hold} 步)")
    print("=" * 108)
    print(f"  {'轴':<6}{'方向':<6}{'终点':>10}{'运动中最大':>12}{'运动中平均':>12}{'到位后':>12}{'超调':>10}")
    rows_out = []
    box = WS.keyboard_box() if scan_box == "profile" else WS.KEYBOARD_BOX_48
    for name, (idx, key) in axes.items():
        lo, hi = box[key]
        is_pos = name in ("X", "Y", "Z")
        for sign, edge in ((1, hi), (-1, lo)):
            sim.reset()
            pos = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]
            tgt = pos if is_pos else rpy
            moving = []
            k = 0
            while abs(tgt[idx]) < abs(edge) - 1e-9:
                if is_pos:
                    tgt[idx] += sign * pos_rate * dt
                else:
                    tgt[idx] += sign * math.radians(5.0)
                tgt[idx] = sign * min(abs(tgt[idx]), abs(edge))
                sim.set_commands(0.0, 0.0, ee_command_from_offset(pos, rpy))
                for _ in range(1 if is_pos else ori_every):
                    r = sim.step()
                    moving.append(r["ee_pos"] if is_pos else math.degrees(r["ee_ori"]))
                k += 1
            settled = [sim.step() for _ in range(hold)]
            if is_pos:
                moving_max, moving_mean = max(moving), float(np.mean(moving))
                final = float(np.mean([r["ee_pos"] for r in settled[-5:]]))
                over = max(0.0, max([r["ee_pos"] for r in settled]) - final)
                unit = "m"
            else:
                moving_max = max(moving)
                moving_mean = float(np.mean(moving))
                final = float(np.mean([math.degrees(r["ee_ori"]) for r in settled[-5:]]))
                over = max(0.0, max([math.degrees(r["ee_ori"]) for r in settled]) - final)
                unit = "deg"
            scale = 1000 if is_pos else 1
            print(
                f"  {name:<6}{'+' if sign > 0 else '-':<6}{sign * abs(edge) * (1 if is_pos else 57.2957795):>9.2f}"
                f"{'m' if is_pos else '°'}{moving_max * scale:>11.1f}{moving_mean * scale:>12.1f}"
                f"{final * scale:>12.1f}{over * scale:>10.1f}   ({'mm' if is_pos else 'deg'})"
            )
            rows_out.append(
                dict(
                    axis=name,
                    sign=sign,
                    edge=sign * abs(edge),
                    moving_max=moving_max,
                    moving_mean=moving_mean,
                    settled=final,
                    overshoot=over,
                    unit=unit,
                )
            )
    return rows_out


def convergence_commands():
    """The 14 commands of check_teleop_stage45i.py, converted the same way."""
    cmds = [
        (0.2, 0.0, 0.0),
        (-0.2, 0.0, 0.0),
        (0.0, 0.2, 0.0),
        (0.0, -0.2, 0.0),
        (0.0, 0.0, 0.25),
        (0.0, 0.0, -0.25),
        (0.15, 0.10, -0.10),
        ((0.0, 0.0, 0.0), (0.09, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (-0.09, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.09)),
    ]
    if TRAIN_RECORD.exists():
        js = json.loads(TRAIN_RECORD.read_text())
        test_cmds = js.get("test_cmds", [])
        if test_cmds:
            rng = np.random.default_rng(2026)
            pick = rng.choice(len(test_cmds), min(4, len(test_cmds)), replace=False)
            for k in pick:
                c = np.asarray(test_cmds[k])
                off = tk.quat_to_R(c[3:]) @ tk.quat_to_R(np.array(TCP_DEFAULT_QUAT_B)).T
                cmds.append((tuple(tk.R_to_rpy(off)), tuple(c[:3] - np.asarray(TCP_DEFAULT_POS_B))))
    else:
        print(f"警告: {TRAIN_RECORD} 不存在, 只用单轴命令")
    return cmds


def convergence_suite(sim: Sim, out: Path, watch: int = 25):
    cmds = convergence_commands()

    def target(entry):
        if len(entry) == 2 and isinstance(entry[0], tuple):
            rpy, dpos = entry
        else:
            rpy, dpos = entry, (0.0, 0.0, 0.0)
        return ee_command_from_offset(dpos, rpy)

    print()
    print("=" * 108)
    print(f"convergence suite (C2: 命令设定后 Actor 自己走 {watch} 步 = {watch * 0.1:.1f} s)")
    print("=" * 108)
    rows_out = []
    print(f"  {'命令':<34}{'起始':>8}{'0.5s':>8}{'2.5s':>8}{'位置':>9}{'抖动':>8}")
    for k, entry in enumerate(cmds):
        sim.reset()
        sim.set_commands(0.0, 0.0, target(entry))
        trace, pos = [], []
        for t in range(watch):
            r = sim.step()
            trace.append(math.degrees(r["ee_ori"]))
            pos.append(r["ee_pos"])
        rec = dict(
            cmd=str(entry),
            e0=trace[0],
            e05=trace[4],
            e25=trace[-1],
            pos=float(np.mean(pos[-5:])),
            jitter=float(np.std(trace[-10:])),
        )
        rows_out.append(rec)
        print(
            f"  {str(entry)[:33]:<34}{trace[0]:>7.1f}°{trace[4]:>7.1f}°{trace[-1]:>7.1f}°"
            f"{rec['pos'] * 1000:>8.0f}mm{rec['jitter']:>7.2f}°"
        )
    e25 = [r["e25"] for r in rows_out]
    p25 = [r["pos"] for r in rows_out]
    summary = dict(
        c2_25=float(np.median(e25)),
        c2_pos=float(np.median(p25)),
        c2_05=float(np.median([r["e05"] for r in rows_out])),
        jitter=float(np.median([r["jitter"] for r in rows_out])),
    )
    print(
        f"  中位数: 2.5 s 姿态 {summary['c2_25']:.2f}°  位置 {summary['c2_pos'] * 1000:.1f} mm  "
        f"抖动 {summary['jitter']:.3f}°"
    )

    # C2b: the chassis moves while the arm is commanded -- an arm result must not be paid for
    # by the chassis, and it is the case the acceptance protocol calls "both"
    print("\n  C2b 底盘 + 机械臂同时 (vx 0.25 / wz 0.15, 姿态命令 roll +0.2 -- 与 Isaac 同一条命令)")
    sim.reset()
    sim.set_commands(0.25, 0.15, ee_command_from_offset((0, 0, 0), (0.20, 0, 0)))
    rows = [sim.step() for _ in range(60)]
    tail = rows[-10:]
    c2b = dict(
        vx=float(np.mean([r["act_vx"] for r in tail])),
        wz=float(np.mean([r["act_wz"] for r in tail])),
        ee_att=float(math.degrees(np.mean([r["ee_ori"] for r in tail]))),
        ee_pos=float(np.mean([r["ee_pos"] for r in tail])),
    )
    print(
        f"    vx 0.250 -> {c2b['vx']:+.3f}   wz 0.150 -> {c2b['wz']:+.3f}   "
        f"EE {c2b['ee_pos'] * 1000:.0f} mm / {c2b['ee_att']:.2f}°"
    )
    summary["c2b"] = c2b
    return dict(summary=summary, rows=rows_out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default=DEFAULT_POLICY)
    ap.add_argument(
        "--suite",
        default="all",
        choices=["drift", "modes", "convergence", "terrain", "workspace", "coverage", "path", "lowspace", "all"],
    )
    ap.add_argument("--out", default="outputs/stage46_mujoco")
    ap.add_argument(
        "--wheel_frictionloss",
        type=float,
        default=None,
        help="override the model's wheel rolling resistance (Nm); 0 gives the rigid-cylinder model",
    )
    ap.add_argument("--tag", default=None, help="output file suffix, e.g. 'no_rolling'")
    ap.add_argument(
        "--coverage_box",
        default="candidate",
        choices=["candidate", "keyboard", "4.8", "4.8_keyboard", "4.9_keyboard", "4.10", "4.10_keyboard"],
        help="which workspace the coverage suite draws its targets from",
    )
    ap.add_argument(
        "--scan_box",
        default="profile",
        choices=["profile", "4.8", "4.10"],
        help="how far the workspace suite pushes each axis (profile = the keyboard's own clamps)",
    )
    args = ap.parse_args()

    global RUN_TAG
    RUN_TAG = f"{args.tag}_" if args.tag else ""
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    sim = Sim(str(ROOT / args.policy), BaseCommandFilter(), args.wheel_frictionloss)
    print(f"policy {args.policy} ({sim.kind}); scene {SCENE}")
    print(f"model joints {sim.m.nq} qpos, {sim.m.nu} actuators, mass {sim.m.body_subtreemass[1]:.2f} kg")

    payload = {}
    if args.suite in ("drift", "all"):
        payload["drift"] = drift_suite(sim, out)
    if args.suite in ("modes", "all"):
        payload["modes"] = modes_suite(sim, out)
    if args.suite in ("convergence", "all"):
        payload["convergence"] = convergence_suite(sim, out)
    if args.suite in ("terrain", "all"):
        payload["terrain"] = terrain_suite(sim, out)
    if args.suite in ("workspace", "all"):
        payload["workspace"] = workspace_suite(sim, out, scan_box=args.scan_box)
    if args.suite in ("coverage", "all"):
        payload["coverage"] = coverage_suite(sim, out, box_name=args.coverage_box)
    if args.suite in ("path", "all"):
        payload["path"] = path_suite(sim, out, scan_box=args.scan_box)
    if args.suite in ("lowspace", "all"):
        payload["lowspace"] = lowspace_suite(sim, out)

    suffix = f"_{args.tag}" if args.tag else ""
    (out / f"mujoco_{args.suite}{suffix}.json").write_text(json.dumps(payload, indent=1))
    print(f"\n已写出 {out / f'mujoco_{args.suite}{suffix}.json'}")


if __name__ == "__main__":
    main()
