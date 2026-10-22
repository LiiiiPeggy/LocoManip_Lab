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

"""CR10 arm kinematics for the teleoperation diagnostics: FK, Jacobian, damped-least-squares IK.

Pure numpy, no Isaac Sim, so the IK studies run in seconds and can be imported by any of the
`scripts/ranger_cr10/check_*` tools. The chain is read from the same URDF the USD was
converted from.

It is only trustworthy because it is checked: `check_ee_attitude_reachability.py` section 2
compares `fk()` against the simulator at seven joint configurations and requires agreement to
1e-3. That check has to run with **gravity off** -- with it on, the simulator's pose is where
the arm sagged to, not where the joint angles put it, and the comparison fails by centimetres
in a way that looks like a modelling error and is not.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

JOINT_NAMES = [f"cr10_joint{i}" for i in range(1, 7)]
DEFAULT_Q = np.array([0.6, 1.0, -2.4, 0.0, 1.4, 0.0])
# the virtual TCP offset from mdp/tcp.py, in the frame of cr10_Link6
TCP_OFFSET_POS = np.array([0.0, 0.00003, 0.14389])
# the simulator's joint limits, which is what the policy is trained against
LIMITS = np.array([[-3.920, 0.940], [-1.570, 1.570], [-2.860, 2.860],
                   [-3.140, 3.140], [-3.140, 3.140], [-3.140, 3.140]])
# the trained EE box: `ranges` in teleop_env_cfg.py, and the keyboard's limits by design
TRAIN_BOX_RPY = ((-0.40, 0.40), (-0.40, 0.40), (-0.50, 0.50))
TRAIN_BOX_POS = ((-0.12, 0.12), (-0.15, 0.15), (-0.12, 0.10))

_URDF = (Path(__file__).resolve().parents[2] / "source/LeggedManip_Lab/LeggedManip_Lab"
         / "assets/ranger_cr10/ranger_cr10_converted.urdf")


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


def R_to_quat(R):
    """(w, x, y, z), the convention Isaac Lab uses."""
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return np.array([s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2
    out = np.zeros(4)
    out[0] = (R[k, j] - R[j, k]) / s
    out[i + 1] = s / 4
    out[j + 1] = (R[j, i] + R[i, j]) / s
    out[k + 1] = (R[k, i] + R[i, k]) / s
    return out


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _build_chain():
    root = ET.parse(_URDF).getroot()
    child_of = {j.find("child").get("link"): j for j in root.findall("joint")}
    chain, cur = [], "cr10_Link6"
    while cur in child_of:
        j = child_of[cur]
        o = j.find("origin")
        T = np.eye(4)
        T[:3, :3] = rpy_matrix(*[float(v) for v in (o.get("rpy") or "0 0 0").split()])
        T[:3, 3] = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
        chain.append((j.get("name"), T, np.array([float(v) for v in j.find("axis").get("xyz").split()])))
        cur = j.find("parent").get("link")
    chain.reverse()
    return chain


CHAIN = _build_chain()


def fk(q):
    """TCP pose in base_link as a 4x4. Fixed joints contribute their transform only."""
    angles = dict(zip(JOINT_NAMES, q))
    T = np.eye(4)
    for name, T_fixed, axis in CHAIN:
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


def exp_so3(v):
    """Rodrigues: a rotation vector to a rotation matrix."""
    v = np.asarray(v, float)
    ang = float(np.linalg.norm(v))
    if ang < 1e-12:
        return np.eye(3)
    return axis_matrix(v / ang, ang)


def local_ik(q_start, target_p, target_R, steps=40, dls_iters=6, lam=0.05,
             step_cap=0.08, tol=1e-5, norm="l2", irls_eps=0.02):
    """Differential IK that walks to the target instead of jumping to a branch.

    This exists because every global solver in this module has been unreliable in the same
    way: damped least squares from a handful of restarts lands on whichever arm branch it
    finds first, and for this robot that meant the '+' attitudes came out needing 9.5 rad of
    joint travel against 0.2-0.9 for the '-'. Any measurement that uses such a solution as
    "the" reference measures the branch lottery, not the policy.

    So: interpolate the *goal pose* from the current one in `steps` increments -- linearly in
    position, `exp(t*log(dR))*R0` in rotation, so the rotation error is also interpolated in
    the base frame, matching the Jacobian -- and run a few DLS iterations at each increment
    with a small step cap. Continuity is structural: every increment starts where the last
    one ended, so the arm cannot hop to another branch without walking there.

    `norm="l1"` swaps the per-step inverse for an iteratively-reweighted one, which
    approximates the solution that minimises the *sum of absolute* joint motion instead of
    its Euclidean norm. That matters here because the reward penalises `sum |q - q_default|`
    -- an L1 cost -- and an L1-optimal joint motion generally points in a different direction
    from an L2-optimal one. Comparing the two against the actor's response is how section E
    tests whether the deviation penalty is what rotates the policy's answer.

    Returns (q, pos_err, rot_err, max_dq, cond) where the residuals are FK'd back against the
    *final* target and max_dq is the largest single DLS step taken (a branch jump would show
    up here as a step far larger than step_cap).
    """
    q = np.asarray(q_start, float).copy()
    T0 = fk(q)
    p0, R0 = T0[:3, 3], T0[:3, :3]
    dR = log_R(target_R @ R0.T)          # rotation vector, base frame, R0 -> target
    dp = np.asarray(target_p, float) - p0
    max_dq = 0.0
    cond = float("nan")
    dq_prev = np.full(len(q), 0.1)   # IRLS seed: a uniform, non-zero motion
    for k in range(1, steps + 1):
        t = k / steps
        p_t = p0 + t * dp
        R_t = exp_so3(t * dR) @ R0
        for _ in range(dls_iters):
            T = fk(q)
            e = np.concatenate([p_t - T[:3, 3], log_R(R_t @ T[:3, :3].T)])
            if np.linalg.norm(e) < tol:
                break
            J = jacobian(q)
            if norm == "l1":
                # IRLS: weight joint i by 1/(|dq_i| + eps) so the quadratic surrogate
                # approximates sum|dq_i|. A Newton-like weighting, one update per DLS step.
                w = 1.0 / (np.abs(dq_prev) + irls_eps)
                W = np.diag(w)
                Winv = np.diag(1.0 / w)
                dq = Winv @ J.T @ np.linalg.solve(J @ Winv @ J.T + (lam**2) * np.eye(6), e)
                dq_prev = dq.copy()
            else:
                dq = J.T @ np.linalg.solve(J @ J.T + (lam**2) * np.eye(6), e)
            n = np.linalg.norm(dq)
            if n > step_cap:
                dq *= step_cap / n
            q = np.clip(q + dq, LIMITS[:, 0], LIMITS[:, 1])
            max_dq = max(max_dq, float(np.linalg.norm(dq)))
    cond = float(np.linalg.cond(jacobian(q)))
    T = fk(q)
    return (q,
            float(np.linalg.norm(np.asarray(target_p, float) - T[:3, 3])),
            float(np.linalg.norm(log_R(target_R @ T[:3, :3].T))),
            max_dq, cond)


def ik(target_p, target_R, q_init, iters=200, lam=0.03, step_cap=0.2):
    """Damped least squares with joint clipping. Returns (pos_err, rot_err, q)."""
    q = np.asarray(q_init, float).copy()
    best = (1e9, 1e9, q.copy())
    for _ in range(iters):
        T = fk(q)
        e = np.concatenate([target_p - T[:3, 3], log_R(target_R @ T[:3, :3].T)])
        err = (float(np.linalg.norm(target_p - T[:3, 3])), float(np.linalg.norm(log_R(target_R @ T[:3, :3].T))))
        if err[0] + err[1] < best[0] + best[1]:
            best = (err[0], err[1], q.copy())
        if err[0] < 1e-6 and err[1] < 1e-6:
            break
        J = jacobian(q)
        dq = J.T @ np.linalg.solve(J @ J.T + (lam**2) * np.eye(6), e)
        n = np.linalg.norm(dq)
        if n > step_cap:
            dq *= step_cap / n
        q = np.clip(q + dq, LIMITS[:, 0], LIMITS[:, 1])
    return best


def solve_ik(target_p, target_R, restarts=12, seed=0):
    """Best of several restarts, because one failed solve is not evidence of unreachability."""
    rng = np.random.default_rng(seed)
    guesses = [DEFAULT_Q, np.clip(DEFAULT_Q + 0.4, *LIMITS.T), np.clip(DEFAULT_Q - 0.4, *LIMITS.T), np.zeros(6)]
    guesses += [rng.uniform(LIMITS[:, 0], LIMITS[:, 1]) for _ in range(restarts)]
    best = None
    for g in guesses:
        ep, er, q = ik(target_p, target_R, g)
        if best is None or ep + er < best[0] + best[1]:
            slack = float(np.min(np.minimum(q - LIMITS[:, 0], LIMITS[:, 1] - q)))
            best = (ep, er, q, slack)
    return best


def solve_ik_near_state(target_p, target_R, q_ref, restarts=8, seed=0, tol=1e-3):
    """The closest solution *to q_ref*, for questions asked at a configuration that is not
    the default.

    `solve_ik_nearest` minimises distance from DEFAULT_Q, which is the right objective only
    when the state under test is the default. Asking it about a perturbed state returns a
    branch chosen for the wrong reason -- measured: it produced |dq| up to 0.81 rad and
    cosines that swung between +0.88 and -0.79 for the same command, which is the objective
    mismatch showing up as apparent policy incoherence.
    """
    rng = np.random.default_rng(seed)
    guesses = [q_ref, np.clip(q_ref + 0.2, *LIMITS.T), np.clip(q_ref - 0.2, *LIMITS.T)]
    guesses += [np.clip(q_ref + rng.uniform(-0.3, 0.3, size=6), *LIMITS.T) for _ in range(restarts)]
    best = None
    for g in guesses:
        ep, er, q = ik(target_p, target_R, g)
        if ep > tol or er > tol:
            continue
        cost = float(np.abs(q - q_ref).sum())
        if best is None or cost < best[0]:
            best = (cost, q)
    if best is None:
        ep, er, q = ik(target_p, target_R, q_ref)
        return ep, er, q
    return 0.0, 0.0, best[1]


def solve_ik_nearest(target_p, target_R, restarts=16, seed=0, tol=1e-3):
    """The *closest* joint configuration that reaches the target, not just any of them.

    This matters more than it looks. A 6R arm usually has several branches that reach the same
    pose, and damped least squares from random restarts lands on whichever it happens to find:
    measured here, `yaw +0.2` comes out at 9.4 rad of total joint travel from the default on one
    branch and 0.27 rad on another. Comparing a reward at the near solution against one at the
    far solution compares two different things, and the far one is dominated by the joint
    deviation penalty rather than by whether the task was achieved.
    """
    rng = np.random.default_rng(seed)
    guesses = [DEFAULT_Q, np.clip(DEFAULT_Q + 0.3, *LIMITS.T), np.clip(DEFAULT_Q - 0.3, *LIMITS.T)]
    guesses += [rng.uniform(LIMITS[:, 0], LIMITS[:, 1]) for _ in range(restarts)]
    best = None
    for g in guesses:
        ep, er, q = ik(target_p, target_R, g)
        if ep > tol or er > tol:
            continue
        cost = float(np.abs(q - DEFAULT_Q).sum())
        if best is None or cost < best[0]:
            best = (cost, q)
    if best is None:  # nothing converged; fall back to whatever is closest overall
        ep, er, q = ik(target_p, target_R, DEFAULT_Q)
        return ep, er, q, float(np.min(np.minimum(q - LIMITS[:, 0], LIMITS[:, 1] - q)))
    cost, q = best
    slack = float(np.min(np.minimum(q - LIMITS[:, 0], LIMITS[:, 1] - q)))
    return 0.0, 0.0, q, slack


R_NEUTRAL = fk(DEFAULT_Q)[:3, :3]
P_NEUTRAL = fk(DEFAULT_Q)[:3, 3]


def command_R(rpy, mode="chassis"):
    """The tool attitude a command of `rpy` names, under either rotation-axis convention.

    `chassis` is what the code implements: the offset is pre-multiplied, so it is a rotation
    about the chassis' own axes. `tool` post-multiplies, which rotates about the tool's axes.
    """
    dR = euler_R(rpy)
    return dR @ R_NEUTRAL if mode == "chassis" else R_NEUTRAL @ dR
