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

"""Stage 4.5b: can the arm hold the tool attitudes the teleoperation task commands?

The teleoperation policy tracks commanded tool *positions* well and commanded tool
*attitudes* barely at all. The acceptance sweep measured an achieved rotation of 1.1x the
commanded magnitude but a residual equal to the command, which is what you see when the
achieved axis is roughly 60 degrees from the commanded one. This script decides whether that
is a workspace limit or a policy limit, because the two need opposite fixes.

Kinematics, and a correction
----------------------------
``get_jacobians()`` is unusable directly: its last axis is **28 = 22 joints + 6 floating-base
DOFs**, so ``get_jacobians()[:, body_idx, :, joint_ids]`` -- the recipe in Isaac Lab's own
``DifferentialIKController`` example, written for a fixed-base robot -- does not pick out the
arm's six columns.

Forward kinematics taken from the URDF the USD was converted from works, but the first
attempt concluded it did **not**, reporting a 4-5 cm mismatch at ``cr10_Link6`` that varied
with configuration. That conclusion was wrong, and the reason matters: the comparison ran
with **gravity on**, so it was comparing the URDF pose at the *written* joint angles against
the simulator's pose after a step of sagging. Run with gravity off, the same comparison
agrees to 1.3e-4 m and the chain-end-to-TCP transform is constant to 2.3e-4 m across
configurations. Section 2 reproduces that check, so the claim is verified in the script that
depends on it rather than asserted from a log.

So: the URDF chain is the model, cross-validated against the simulator, and the IK is damped
least squares on it with the simulator's own joint limits. Hundreds of targets and restarts
cost seconds, which is what makes the study thorough enough to be believed.

Sections:
  1  frames -- the neutral pose, and how far the tool's axes are from the chassis'
  2  the analytic chain against the simulator (gravity off)
  3  the Jacobian, including its rank at the neutral configuration
  4  IK reachability, chassis-axis definition (the one the code implements)
  5  the same targets under the tool-axis definition
  6  what the acceptance residuals imply about the achieved axis
  7  sensitivity: where the chassis-axis box starts to fail

Usage: python scripts/ranger_cr10/check_ee_attitude_reachability.py [--restarts N]
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description="EE attitude reachability check.")
parser.add_argument("--restarts", type=int, default=12, help="IK initial guesses per target")
parser.add_argument("--iters", type=int, default=200)
parser.add_argument("--seed", type=int, default=0)
args_cli = parser.parse_args()

REPO = Path(__file__).resolve().parents[2]
URDF = REPO / "source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10/ranger_cr10_converted.urdf"

JOINT_NAMES = [f"cr10_joint{i}" for i in range(1, 7)]
DEFAULT_Q = np.array([0.6, 1.0, -2.4, 0.0, 1.4, 0.0])
# the virtual TCP offset from mdp/tcp.py; section 2 re-measures it from the simulator
TCP_OFFSET_POS = np.array([0.0, 0.00003, 0.14389])
# the simulator's limits, which is what the policy is trained against
LIMITS = np.array([[-3.920, 0.940], [-1.570, 1.570], [-2.860, 2.860],
                   [-3.140, 3.140], [-3.140, 3.140], [-3.140, 3.140]])
# the trained EE box: `ranges` in teleop_env_cfg.py, and the keyboard's limits by design
TRAIN_BOX_RPY = ((-0.40, 0.40), (-0.40, 0.40), (-0.50, 0.50))
TRAIN_BOX_POS = ((-0.12, 0.12), (-0.15, 0.15), (-0.12, 0.10))


# ------------------------------------------------------------------ maths (no Isaac) ----
def rpy_matrix(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_matrix(axis, ang):
    a = np.asarray(axis, float)
    a = a / max(np.linalg.norm(a), 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def R_to_rpy(R):
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(math.cos(pitch)) < 1e-9:
        return np.array([math.atan2(R[0, 1], R[1, 1]), pitch, 0.0])
    return np.array([math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0])])


def log_R(R):
    """Rotation vector: direction is the axis, norm is the angle."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    if abs(ang - math.pi) < 1e-6:
        d = np.clip((np.diag(R) + 1.0) / 2.0, 0.0, None)
        ax = np.sqrt(d)
        if R[2, 1] - R[1, 2] < 0:
            ax[0] = -ax[0]
        if R[0, 2] - R[2, 0] < 0:
            ax[1] = -ax[1]
        if R[1, 0] - R[0, 1] < 0:
            ax[2] = -ax[2]
        n = np.linalg.norm(ax)
        return ax / n * ang if n > 1e-9 else np.zeros(3)
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(ang)) * ang


def euler_R(rpy):
    """Rz(y) Ry(p) Rx(r) -- the convention `quat_from_euler_xyz` uses."""
    return rpy_matrix(*rpy)


# -- the chain, straight out of the URDF the USD was converted from -------------------
_root = ET.parse(URDF).getroot()
_child_of = {j.find("child").get("link"): j for j in _root.findall("joint")}
_chain, _cur = [], "cr10_Link6"
while _cur in _child_of:
    _j = _child_of[_cur]
    _o = _j.find("origin")
    _T = np.eye(4)
    _T[:3, :3] = rpy_matrix(*[float(v) for v in (_o.get("rpy") or "0 0 0").split()])
    _T[:3, 3] = [float(v) for v in (_o.get("xyz") or "0 0 0").split()]
    _chain.append((_j.get("name"), _T, np.array([float(v) for v in _j.find("axis").get("xyz").split()])))
    _cur = _j.find("parent").get("link")
_chain.reverse()


def fk(q):
    """TCP pose in base_link as a 4x4. Fixed joints contribute their transform only."""
    angles = dict(zip(JOINT_NAMES, q))
    T = np.eye(4)
    for name, T_fixed, axis in _chain:
        T = T @ T_fixed
        if name in angles:
            A = np.eye(4)
            A[:3, :3] = axis_matrix(axis, angles[name])
            T = T @ A
    T = T.copy()
    T[:3, 3] += T[:3, :3] @ TCP_OFFSET_POS
    return T


def jacobian(q, eps=1e-6):
    """6x6 body Jacobian in the base frame, by central differences on the analytic FK."""
    J = np.zeros((6, 6))
    for k in range(6):
        dq = np.zeros(6)
        dq[k] = eps
        Tp, Tm = fk(q + dq), fk(q - dq)
        J[:3, k] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * eps)
        J[3:, k] = log_R(Tp[:3, :3] @ Tm[:3, :3].T) / (2 * eps)
    return J


R_NEUTRAL = fk(DEFAULT_Q)[:3, :3]
P_NEUTRAL = fk(DEFAULT_Q)[:3, 3]

print("=" * 86)
print("1. 坐标系")
print("=" * 86)
print(f"  机械臂关节顺序 {JOINT_NAMES}")
print(f"  默认关节姿态   {DEFAULT_Q.tolist()}")
print(f"  中性 TCP 位置  {np.round(P_NEUTRAL, 5)}  |p| = {np.linalg.norm(P_NEUTRAL):.4f} m")
print(f"  中性 TCP 姿态  rpy(deg) = {np.round(np.degrees(R_to_rpy(R_NEUTRAL)), 2)}")
print("  工具轴与机身轴的夹角 (0° = 对齐, 180° = 反向):")
for i, nm in enumerate("xyz"):
    print(f"    {nm}: {math.degrees(math.acos(max(-1.0, min(1.0, R_NEUTRAL[i, i])))):6.1f}°"
          f"   工具 {nm} 轴指向 {np.round(R_NEUTRAL[:, i], 4)}")
print("  -> 三个都不接近 0°, 所以'绕机身轴 roll/pitch/yaw'对腕部不是小扰动")

# ------------------------------------------------------------------ 2. cross-check ------
print("\n" + "=" * 86)
print("2. 解析链 vs 仿真 (重力关闭, 否则重力下垂会污染比较)")
print("=" * 86)

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import LeggedManip_Lab.tasks  # noqa: E402,F401
from LeggedManip_Lab.assets.ranger_cr10.ranger_cr10_articulation_cfg import ARM_JOINT_NAMES  # noqa: E402
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.teleop_env_cfg import (  # noqa: E402
    RangerCr10TeleopEnvCfg,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.tcp import tcp_pose_b  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

cfg = RangerCr10TeleopEnvCfg()
cfg.scene.num_envs = 1
cfg.sim.gravity = (0.0, 0.0, 0.0)
cfg.commands.ee_pose.debug_vis = False
cfg.events.reset_robot_joints = None
cfg.events.randomize_actuator_gains = None
env = gym.make("RANGER-CR10-TELEOP-WBC", cfg=cfg)
u = env.unwrapped
robot, sim = u.scene["robot"], u.sim
arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
base_id = robot.find_bodies("base_link")[0][0]
link6_id = robot.find_bodies("cr10_Link6")[0][0]

rng = np.random.default_rng(args_cli.seed)
worst_p = worst_r = 0.0
consts = []
print(f"  {'case':<12}{'TCP 位置差':>14}{'TCP 姿态差':>14}")
for label, q in [("default", DEFAULT_Q)] + [
        (f"perturb {i}", DEFAULT_Q + rng.uniform(-0.8, 0.8, size=6)) for i in range(5)] + [
        ("random", rng.uniform(LIMITS[:, 0], LIMITS[:, 1]))]:
    q = np.clip(q, LIMITS[:, 0], LIMITS[:, 1])
    robot.write_joint_state_to_sim(
        torch.tensor(q, dtype=torch.float32, device=robot.device).unsqueeze(0).expand(robot.num_instances, -1),
        torch.zeros(robot.num_instances, 6, device=robot.device), joint_ids=arm_ids)
    sim.step()
    q_read = robot.data.joint_pos[0, arm_ids].cpu().numpy()
    tp, tq = tcp_pose_b(robot, link6_id, base_id)
    T_sim = np.eye(4)
    T_sim[:3, 3] = tp[0].cpu().numpy()
    T_sim[:3, :3] = Rotation.from_quat(tq[0].cpu().numpy()[[1, 2, 3, 0]]).as_matrix()
    T_an = fk(q_read)
    dp = float(np.linalg.norm(T_sim[:3, 3] - T_an[:3, 3]))
    dr = float(np.linalg.norm(log_R(T_sim[:3, :3] @ T_an[:3, :3].T)))
    consts.append(np.linalg.inv(T_an) @ T_sim)
    worst_p, worst_r = max(worst_p, dp), max(worst_r, dr)
    print(f"  {label:<12}{dp:>14.2e}{dr:>14.2e}")
spread = max(float(np.linalg.norm(c[:3, 3] - consts[0][:3, 3])) for c in consts)
print(f"  最大残差 {worst_p:.2e} m / {worst_r:.2e} rad; "
      f"链末端->TCP 常量在位形间散布 {spread:.2e} m")
print(f"  -> {'解析链可用' if worst_p < 1e-3 and worst_r < 1e-3 else '**不可用**'}")
# NOTE: env.close()/app.close() are deliberately NOT called here. app.close() tears the
# process down from native code, so everything below -- which needs no simulator -- would be
# silently dropped, with the process still exiting 0. They run at the end of the file.

# ------------------------------------------------------------------ 3. Jacobian ---------
print("\n" + "=" * 86)
print("3. 雅可比与奇异位形")
print("=" * 86)
J0 = jacobian(DEFAULT_Q)
sv = np.linalg.svd(J0, compute_uv=False)
print(f"  默认姿态奇异值 {np.round(sv, 4)}")
print(f"  秩 (sv > 1e-3) = {int((sv > 1e-3).sum())} / 6")
for eps in (0.02, 0.1, 0.3):
    qs = [DEFAULT_Q + eps * np.eye(6)[k] for k in range(6)]
    ranks = [int((np.linalg.svd(jacobian(q), compute_uv=False) > 1e-3).sum()) for q in qs]
    print(f"  从默认姿态单关节偏移 {eps} rad: 秩 {ranks}")
print("  (j4 与 j6 都在 0 位 -> 两轴重合, 这是腕部奇异位形)")


def ik(target_p, target_R, q_init, iters, lam=0.03, step_cap=0.2, rot_weight=1.0, pos_weight=1.0):
    q = q_init.copy()
    best = (1e9, 1e9, q.copy())
    for _ in range(iters):
        T = fk(q)
        e = np.concatenate([pos_weight * (target_p - T[:3, 3]),
                            rot_weight * log_R(target_R @ T[:3, :3].T)])
        err = (float(np.linalg.norm(target_p - T[:3, 3])), float(np.linalg.norm(log_R(target_R @ T[:3, :3].T))))
        if err[0] + err[1] < best[0] + best[1]:
            best = (err[0], err[1], q.copy())
        if err[0] < 1e-5 and err[1] < 1e-5:
            break
        J = jacobian(q)
        dq = J.T @ np.linalg.solve(J @ J.T + (lam**2) * np.eye(6), e)
        n = np.linalg.norm(dq)
        if n > step_cap:
            dq *= step_cap / n
        q = np.clip(q + dq, LIMITS[:, 0], LIMITS[:, 1])
    return best


GUESSES = [DEFAULT_Q, np.clip(DEFAULT_Q + 0.4, *LIMITS.T), np.clip(DEFAULT_Q - 0.4, *LIMITS.T),
           np.zeros(6)] + [rng.uniform(LIMITS[:, 0], LIMITS[:, 1]) for _ in range(args_cli.restarts)]


def solve(tp, tR, **kw):
    best = None
    for g in GUESSES:
        ep, er, q = ik(tp, tR, g, args_cli.iters, **kw)
        if best is None or ep + er < best[0] + best[1]:
            slack = float(np.min(np.minimum(q - LIMITS[:, 0], LIMITS[:, 1] - q)))
            best = (ep, er, q, slack)
    return best


def command_R(rpy, mode):
    dR = euler_R(rpy)
    return dR @ R_NEUTRAL if mode == "chassis" else R_NEUTRAL @ dR


TARGETS = [("中性 (offset=0)", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))]
for mag in (0.025, 0.05, 0.1, 0.15, 0.2):
    for axis, nm in ((0, "roll"), (1, "pitch"), (2, "yaw")):
        for sgn in (+1, -1):
            rpy = [0.0, 0.0, 0.0]
            rpy[axis] = sgn * mag
            TARGETS.append((f"{nm}{'+' if sgn > 0 else '-'} {mag}", tuple(rpy), (0.0, 0.0, 0.0)))
for off in [(0.12, 0.0, 0.0), (0.0, -0.15, 0.0), (0.0, 0.0, 0.10)]:
    TARGETS.append((f"纯位置 {off}, 姿态不变", (0.0, 0.0, 0.0), off))

for mode, mode_name in (("chassis", "机身轴 ΔR_B·R_NE  —— 代码当前实现"),
                        ("tool", "工具轴 R_NE·ΔR_E")):
    print("\n" + "=" * 86)
    print(f"4. IK 可达性 — {mode_name}")
    print("=" * 86)
    print(f"  {'目标':<26}{'位置残差':>10}{'姿态残差':>10}{'关节余量':>9}"
          f"{'|Δq| 离默认':>12}{'max|Δq|':>9}{'最小奇异值':>11}")
    n_ok = 0
    for label, rpy, off in TARGETS:
        ep, er, q, slack = solve(P_NEUTRAL + np.asarray(off), command_R(rpy, mode))
        ok = ep < 2e-3 and er < 0.02
        n_ok += ok
        dq = q - DEFAULT_Q
        # the smallest singular value at the solution: an arm resting against a singularity
        # is a different failure from an arm that simply has not been asked properly
        sv_min = float(np.linalg.svd(jacobian(q), compute_uv=False)[-1])
        print(f"  {label:<26}{ep:>10.2e}{er:>10.4f}{math.degrees(slack):>8.1f}°"
              f"{np.linalg.norm(dq):>12.3f}{np.abs(dq).max():>9.3f}{sv_min:>11.4f}")
    print(f"  -> {n_ok}/{len(TARGETS)} 可达 (位置 <2 mm 且姿态 <0.02 rad)")
    print("  |Δq| 是 IK 解相对默认关节姿态的关节空间距离 —— 若很大, '保持默认姿态不动'")
    print("  就是一个代价很低、而 PPO 很难自己走出来的局部最优")

# --------------------------------------------------------- 5. box corners + sensitivity --
print("\n" + "=" * 86)
print("5. 训练采样盒的角点与边界 (机身轴定义)")
print("=" * 86)
corners = []
for sx in (+1, -1):
    for sy in (+1, -1):
        for sz in (+1, -1):
            corners.append((sx * TRAIN_BOX_RPY[0][1], sy * TRAIN_BOX_RPY[1][1], sz * TRAIN_BOX_RPY[2][1]))
print(f"  {'角点 rpy':<30}{'位置残差':>11}{'姿态残差':>11}{'判定':>8}")
n_ok = 0
for rpy in corners:
    ep, er, q, slack = solve(P_NEUTRAL, command_R(rpy, "chassis"))
    ok = ep < 2e-3 and er < 0.02
    n_ok += ok
    print(f"  {str(tuple(round(v, 2) for v in rpy)):<30}{ep:>11.2e}{er:>11.4f}"
          f"{'可达' if ok else ('勉强' if er < 0.1 else '不可达'):>8}")
print(f"  -> {n_ok}/8 个角点可达")

print("\n" + "=" * 86)
print("6. 验收数据的几何解释")
print("=" * 86)
for c in (0.025, 0.1, 0.2):
    a, e = 1.1 * c, c
    ct = (c * c + a * a - e * e) / (2 * c * a)
    print(f"  指令 {c:.3f} rad, 实测 {a:.3f} rad, 残余 {e:.3f} rad"
          f"  -> 实测轴与指令轴夹角 {math.degrees(math.acos(max(-1, min(1, ct)))):.1f}°")

env.close()
app.close()
