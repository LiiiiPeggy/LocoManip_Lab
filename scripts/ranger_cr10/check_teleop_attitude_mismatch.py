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

"""Stage 4.5d: at many configurations, is the policy's attitude response badly *scaled* or
badly *aimed*, and what does that say about where to fix it?

4.5c measured one configuration and found the actor's joint-space response ~40 degrees off
the IK direction and 27-84% of its magnitude. It could not say whether that is a property of
the policy or of that one operating point, because the reference IK it used kept landing on
other arm branches -- the '+' targets came out needing 9.5 rad of travel where the '-' needed
0.9. This script fixes the reference first (continuous local IK, section A), then repeats the
comparison across configurations the policy actually visits (B), separates gain from aim with
controlled interventions (C), and reads the training distribution the actor was fitted to (D).

The reference
-------------
`teleop_kinematics.local_ik` walks the goal pose in small increments and runs a few damped
least-squares steps at each, so every step starts where the last ended and the arm cannot hop
branches. Solutions are only used if FK reproduces the target (position and attitude
residuals below tolerance) and no single step exceeded the cap; section A prints all of that
per command so the reference can be judged rather than trusted.

What is measured, and in which space
------------------------------------
Joint space answers "does the actor command the same joint motion IK would". Task space
answers "does that motion produce the commanded tool rotation". The brief is right that the
first does not imply the second, so both are reported: each response is evaluated by FK, and
the attitude change it produces is compared against the commanded offset.

Usage: python scripts/ranger_cr10/check_teleop_attitude_mismatch.py --policy <policy.pt>
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description="Teleoperation attitude mismatch across configurations.")
parser.add_argument("--policy", required=True)
parser.add_argument("--settle", type=int, default=80, help="policy steps used to reach each state")
parser.add_argument("--skip-f", action="store_true", help="skip the offline supervised sections")
args_cli = parser.parse_args()

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teleop_kinematics as tk  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as iu  # noqa: E402
import LeggedManip_Lab.tasks  # noqa: E402,F401
from LeggedManip_Lab.assets.ranger_cr10.ranger_cr10_articulation_cfg import ARM_JOINT_NAMES  # noqa: E402
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.teleop_env_cfg import (  # noqa: E402
    RangerCr10TeleopEnvCfg,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.pose_command_teleop import (  # noqa: E402
    TCP_DEFAULT_POS_B,
    TCP_DEFAULT_QUAT_B,
)

cfg = RangerCr10TeleopEnvCfg()
cfg.scene.num_envs = 4
cfg.commands.ee_pose.debug_vis = False
cfg.commands.ee_pose.resampling_time_range = (1e6, 1e6)
cfg.commands.base_velocity.resampling_time_range = (1e6, 1e6)
cfg.observations.policy.enable_corruption = False
env = gym.make("RANGER-CR10-TELEOP-WBC", cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
ee_cmd = u.command_manager.get_term("ee_pose")
vel_cmd = u.command_manager.get_term("base_velocity")
arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
net = torch.jit.load(args_cli.policy, map_location=u.device)
DEV, N = u.device, u.num_envs
JOINT_SCALE = u.action_manager.get_term("joint_pos").cfg.scale
HISTORY = u.observation_manager._group_obs_term_history_buffer["policy"][
    u.observation_manager._group_obs_term_names["policy"][0]].max_length

R_NEUTRAL = tk.R_NEUTRAL
P_NEUTRAL = np.array(TCP_DEFAULT_POS_B)


def set_command(rpy, pos_offset=(0.0, 0.0, 0.0)):
    dR = iu.quat_from_euler_xyz(*[torch.tensor([v], device=DEV) for v in rpy])
    tq = iu.quat_mul(dR, torch.tensor(TCP_DEFAULT_QUAT_B, device=DEV).unsqueeze(0))[0]
    ee_cmd.pose_command_b[:, :3] = torch.tensor(P_NEUTRAL + np.asarray(pos_offset), device=DEV)
    ee_cmd.pose_command_b[:, 3:] = tq
    vel_cmd.is_standing_env[:] = False
    vel_cmd.vel_command_b[:] = 0.0


def observe():
    """The observation the manager hands the policy, with its 3-frame history refreshed.

    `compute()` defaults to `update_history=False`, and with history_length > 0 it then
    returns the buffer `env.step` last wrote -- a stale vector. Three appends fill the whole
    history with the current command so two calls differ only in the command.
    """
    for _ in range(HISTORY):
        obs = u.observation_manager.compute(update_history=True)["policy"]
    return obs


def actor_arm_action(rpy, pos_offset=(0.0, 0.0, 0.0)):
    set_command(rpy, pos_offset)
    # observe() must run OUTSIDE inference_mode: it appends to the observation history
    # buffers, and a tensor created under inference_mode makes the buffer an inference
    # tensor, so the next env.reset() cannot zero it in place
    # ("Inplace update to inference tensor outside InferenceMode is not allowed").
    o = observe()
    with torch.inference_mode():
        return net(o)[0, 2:].cpu().numpy()


def q_current():
    return robot.data.joint_pos[0, arm_ids].cpu().numpy()


def attitude_change(q_from, q_to):
    """Rotation vector of the tool attitude change produced by moving q_from -> q_to."""
    return tk.log_R(tk.fk(q_to)[:3, :3] @ tk.fk(q_from)[:3, :3].T)


CMDS = []
for mag in (0.025, 0.05, 0.1, 0.2):
    for axis, nm in ((0, "roll"), (1, "pitch"), (2, "yaw")):
        for sgn in (+1, -1):
            rpy = [0.0, 0.0, 0.0]
            rpy[axis] = sgn * mag
            CMDS.append((f"{nm}{'+' if sgn > 0 else '-'}{mag}", tuple(rpy)))
NEUTRAL = ("neutral", (0.0, 0.0, 0.0))

# ------------------------------------------------------------------ A. the reference ------
print("=" * 104)
print("A. 连续局部 IK 参考 (多步小增量 + DLS), 中性位形")
print("=" * 104)
print("  每个命令: 从当前关节角出发把目标位姿分 40 步内插, 每步 6 次 DLS。判定标准:")
print("  FK 回代残差 <1e-3, 单步 |dq| <= step_cap(0.08) —— 超过就说明跳了分支。")
print(f"  {'命令':<14}{'位置残差':>11}{'姿态残差':>11}{'Σ|Δq|':>9}{'max|Δq|':>9}"
      f"{'最大单步':>10}{'条件数':>9}{'判定':>6}")
LOCAL = {}
for label, rpy in [NEUTRAL] + CMDS:
    q_ik, ep, er, max_step, cond = tk.local_ik(tk.DEFAULT_Q, P_NEUTRAL, tk.command_R(rpy))
    ok = ep < 1e-3 and er < 1e-3 and max_step <= 0.08 + 1e-9
    LOCAL[label] = (q_ik, ok, cond)
    print(f"  {label:<14}{ep:>11.2e}{er:>11.2e}{np.abs(q_ik - tk.DEFAULT_Q).sum():>9.3f}"
          f"{np.abs(q_ik - tk.DEFAULT_Q).max():>9.3f}{max_step:>10.4f}{cond:>9.1f}"
          f"{'可用' if ok else '弃用':>6}")
n_ok = sum(1 for _ in LOCAL.values() if _[1])
print(f"  -> {n_ok}/{len(LOCAL)} 个参考解可用")
print(f"  对照 4.5c 用的全局解: 同一批 '+' 目标曾解出 Σ|Δq| 高达 9.5 rad (远分支), 这里最大 "
      f"{max(np.abs(v[0] - tk.DEFAULT_Q).sum() for v in LOCAL.values()):.3f} rad")

# ------------------------------------------------------- B. many configurations ----------
print("\n" + "=" * 104)
print("B. 多位形: actor 响应 vs 局部 IK 参考")
print("=" * 104)

STATES = [("neutral(default_q)", None, None)]
SETTLE_CMDS = [("settled roll+0.2", (0.2, 0.0, 0.0), (0.0, 0.0, 0.0)),
               ("settled pitch-0.2", (0.0, -0.2, 0.0), (0.0, 0.0, 0.0)),
               ("settled yaw+0.2", (0.0, 0.0, 0.2), (0.0, 0.0, 0.0)),
               ("settled pos+att", (0.0, 0.0, 0.15), (0.10, -0.12, 0.08))]
for name, rpy, off in SETTLE_CMDS:
    STATES.append((name, rpy, off))
# a controlled wrist perturbation, so the sample is not all near-solutions
STATES.append(("wrist perturbed", "perturb", None))

records = []
for name, rpy, off in STATES:
    if rpy is None:
        q_state = tk.DEFAULT_Q.copy()
    elif isinstance(rpy, str):
        q_state = tk.DEFAULT_Q + np.array([0.05, -0.08, 0.10, 0.0, 0.22, 0.18])
    else:
        # let the policy itself drive the arm there: a state it actually visits
        set_command(rpy, off)
        obs, _ = env.reset()
        a = torch.zeros(N, u.action_manager.total_action_dim, device=DEV)
        for _ in range(args_cli.settle):
            with torch.inference_mode():
                a = net(obs["policy"])
            obs, _, _, _, _ = env.step(a)
            set_command(rpy, off)
        q_state = q_current()
    # Write the state into the simulator and re-read it, so the state the *actor* is
    # evaluated at is exactly the state the IK reference is computed from. The first version
    # only assigned a numpy array for the perturbed case, so the actor was measured at the
    # previous state while IK used the perturbed one -- the two rows came out numerically
    # identical, which is what exposed it.
    q_t = torch.tensor(q_state, dtype=torch.float32, device=DEV).unsqueeze(0).expand(N, -1)
    robot.write_joint_state_to_sim(q_t, torch.zeros_like(q_t), joint_ids=arm_ids)
    u.sim.step()
    q_state = q_current()
    cond_state = float(np.linalg.cond(tk.jacobian(q_state)))
    a_neutral = actor_arm_action((0.0, 0.0, 0.0))
    q_ik_neutral = tk.local_ik(q_state, P_NEUTRAL, tk.command_R((0.0, 0.0, 0.0)))[0]
    for label, rpy_c in CMDS:
        target_R = tk.command_R(rpy_c)
        q_ik, ep, er, max_step, cond_ik = tk.local_ik(q_state, P_NEUTRAL, target_R)
        ref_ok = ep < 1e-3 and er < 1e-3 and max_step <= 0.08 + 1e-9
        a_c = actor_arm_action(rpy_c)
        # command-induced changes, both measured from the same state's neutral command
        dq_ik = (q_ik - q_state) - (q_ik_neutral - q_state)
        dq_actor = JOINT_SCALE * (a_c - a_neutral)
        n_ik, n_actor = np.linalg.norm(dq_ik), np.linalg.norm(dq_actor)
        if n_ik < 1e-6 or n_actor < 1e-6:
            records.append(dict(state=name, cmd=label, ok=False, reason="near-zero", cond=cond_state))
            continue
        cos = float(np.clip(np.dot(dq_ik, dq_actor) / (n_ik * n_actor), -1.0, 1.0))
        records.append(dict(
            state=name, cmd=label, rpy=rpy_c, ok=ref_ok, cond=cond_state, cond_ik=cond_ik,
            n_ik=n_ik, n_actor=n_actor, ratio=n_actor / n_ik,
            ang=math.degrees(math.acos(cos)), cos=cos,
            dq_ik=dq_ik, dq_actor=dq_actor, q_state=q_state.copy(),
        ))

valid = [r for r in records if r.get("ok")]
print(f"  {len(STATES)} 个位形 x {len(CMDS)} 个命令 = {len(records)} 个样本, 参考解可用 {len(valid)} 个")
print(f"  {'位形':<22}{'条件数':>9}{'样本':>6}{'幅值比 中位':>12}{'幅值比 范围':>16}"
      f"{'夹角 中位':>11}{'夹角 范围':>15}")
for name, _, _ in STATES:
    rs = [r for r in valid if r["state"] == name]
    if not rs:
        continue
    ratios = np.array([r["ratio"] for r in rs])
    angs = np.array([r["ang"] for r in rs])
    print(f"  {name:<22}{rs[0]['cond']:>9.1f}{len(rs):>6}{np.median(ratios):>12.3f}"
          f"{f'{ratios.min():.2f}-{ratios.max():.2f}':>16}"
          f"{np.median(angs):>10.1f}°{f'{angs.min():.1f}-{angs.max():.1f}°':>15}")
allr = np.array([r["ratio"] for r in valid])
alla = np.array([r["ang"] for r in valid])
print(f"  {'全部':<22}{'':>9}{len(valid):>6}{np.median(allr):>12.3f}"
      f"{f'{allr.min():.2f}-{allr.max():.2f}':>16}{np.median(alla):>10.1f}°"
      f"{f'{alla.min():.1f}-{alla.max():.1f}°':>15}")

print("\n  按命令幅值 (跨位形):")
print(f"  {'幅值':<8}{'样本':>6}{'幅值比 中位':>12}{'夹角 中位':>11}{'夹角 标准差':>12}")
for mag in (0.025, 0.05, 0.1, 0.2):
    rs = [r for r in valid if str(mag) in r["cmd"]]
    if rs:
        print(f"  {mag:<8}{len(rs):>6}{np.median([r['ratio'] for r in rs]):>12.3f}"
              f"{np.median([r['ang'] for r in rs]):>10.1f}°"
              f"{np.std([r['ang'] for r in rs]):>12.1f}°")

print("\n  按旋转轴 (跨位形):")
print(f"  {'轴':<8}{'样本':>6}{'幅值比 中位':>12}{'夹角 中位':>11}{'夹角 标准差':>12}")
for ax in ("roll", "pitch", "yaw"):
    rs = [r for r in valid if r["cmd"].startswith(ax)]
    if rs:
        print(f"  {ax:<8}{len(rs):>6}{np.median([r['ratio'] for r in rs]):>12.3f}"
              f"{np.median([r['ang'] for r in rs]):>10.1f}°"
              f"{np.std([r['ang'] for r in rs]):>12.1f}°")

# ------------------------------------------------- C. gain vs aim, task space -------------
print("\n" + "=" * 104)
print("C. 增益 vs 方向: 在同一状态下对比四种响应, 用 FK 在任务空间评分")
print("=" * 104)
print("  A 原始 actor | B 保持 actor 方向、幅值放大到 IK 的 |Δq| | C 局部 IK |")
print("  D 从 actor 的结果再跑一次局部 IK (任务空间残差修正)")
print("  任务空间评分: 按响应的 Δq 移动后, 工具姿态相对'中性姿态'的转角, 与命令要求的转角之比,"
      "\n  以及 FK 回代后相对目标位姿的姿态残差。")
print(f"  {'命令':<14}{'要求转角':>9}{'A 实际':>8}{'B 实际':>8}{'C 实际':>8}{'D 实际':>8}"
      f"{'| 残差 A':>10}{'B':>8}{'C':>8}{'D':>8}")
picks = [("roll+0.2", (0.2, 0.0, 0.0)), ("pitch+0.2", (0.0, 0.2, 0.0)),
         ("yaw+0.2", (0.0, 0.0, 0.2)), ("yaw-0.2", (0.0, 0.0, -0.2))]
q_state = tk.DEFAULT_Q.copy()
a_neutral = actor_arm_action((0.0, 0.0, 0.0))
for label, rpy in picks:
    target_R = tk.command_R(rpy)
    q_ik, ep, er, max_step, _ = tk.local_ik(q_state, P_NEUTRAL, target_R)
    a_c = actor_arm_action(rpy)
    dq_actor = JOINT_SCALE * (a_c - a_neutral)          # command-induced, same as section B
    q_ik_neutral = tk.local_ik(q_state, P_NEUTRAL, tk.command_R((0.0, 0.0, 0.0)))[0]
    dq_ik = q_ik - q_ik_neutral                          # command-induced
    n_actor, n_ik = np.linalg.norm(dq_actor), np.linalg.norm(dq_ik)

    variants = {"A": dq_actor.copy()}
    variants["B"] = dq_actor * (n_ik / n_actor) if n_actor > 1e-9 else dq_actor
    variants["C"] = dq_ik
    q_actor_end = q_state + dq_actor
    q_d, _, _, _, _ = tk.local_ik(q_actor_end, P_NEUTRAL, target_R)
    variants["D"] = q_d - q_state

    required = np.linalg.norm(tk.log_R(target_R @ R_NEUTRAL.T))
    line = f"  {label:<14}{math.degrees(required):>8.1f}°"
    res = {}
    for key in ("A", "B", "C", "D"):
        q_end = q_state + variants[key]
        got = np.linalg.norm(attitude_change(q_state, q_end))
        line += f"{math.degrees(got):>7.1f}°"
        res[key] = np.linalg.norm(tk.log_R(target_R @ tk.fk(q_end)[:3, :3].T))
    line += "  |"
    for key in ("A", "B", "C", "D"):
        line += f"{math.degrees(res[key]):>7.1f}°"
    print(line)
print("  (A/B/C/D 都是离线关节目标 + FK 评分, 不是闭环动力学结果; 闭环在 4.5c 的 A' 已验 13/13)")

# ------------------------------------------------------- D. training distribution --------
print("\n" + "=" * 104)
print("D. 训练命令分布与奖励权衡")
print("=" * 104)
rng = np.random.default_rng(0)
lo = np.array([r[0] for r in tk.TRAIN_BOX_RPY])
hi = np.array([r[1] for r in tk.TRAIN_BOX_RPY])
samples = rng.uniform(lo, hi, size=(200000, 3))
offs = np.array([np.linalg.norm(tk.log_R(tk.euler_R(s))) for s in samples])
print(f"  采样盒 roll {tk.TRAIN_BOX_RPY[0]} pitch {tk.TRAIN_BOX_RPY[1]} yaw {tk.TRAIN_BOX_RPY[2]} rad")
print(f"  合成转角 |offset|: 均值 {offs.mean():.3f}, 中位 {np.median(offs):.3f} rad")
for thr in (0.025, 0.05, 0.1, 0.2):
    print(f"    |offset| < {thr:.3f} rad 的比例: {(offs < thr).mean():.2%}")
print(f"  位置偏移盒 {tk.TRAIN_BOX_POS}")
# the *training* config, not this script's: the env above had resampling disabled for the
# measurements, and reading its cfg would report (1e6, 1e6) as if that were what training used
train_cfg = RangerCr10TeleopEnvCfg()
train_resample = train_cfg.commands.ee_pose.resampling_time_range
print(f"  命令保持 (训练配置) resampling_time_range = {train_resample} s, "
      f"episode = {u.max_episode_length_s} s -> 每个 episode 约 "
      f"{u.max_episode_length_s / np.mean(train_resample):.1f} 条命令")
print(f"  训练时姿态目标的采样盒是 ranges = {train_cfg.commands.ee_pose.ranges}")
print(f"  速度命令独立重采样 {train_cfg.commands.base_velocity.resampling_time_range} s,"
      f" rel_standing_envs = {train_cfg.commands.base_velocity.rel_standing_envs}")
print("  位置目标与姿态目标同属一条命令 (同一个 pose_command_b), 底盘速度命令独立重采样 ——")
print("  即训练时三者同时存在, 这一点由 4.5c 的重采样计数 (1.67%) 与配置确认。")

print("\n  奖励项在当前工作点的量级与局部斜率 (取自 teleop_env_cfg.py 与实际 reward 代码):")
pos_w, ori_w, vlin_w, vang_w = 4.5, -4.0, 3.5, 2.5
e_pos, e_ori = 0.08, 0.10   # 收敛时的典型值 (4.5c 实测 0.069-0.081 / 0.096)
print(f"    位置  {pos_w:+.1f}*exp(-d^2/0.1): d={e_pos:.2f} -> {pos_w * math.exp(-e_pos**2 / 0.1):.3f}"
      f"   斜率 {pos_w * math.exp(-e_pos**2 / 0.1) * 2 * e_pos / 0.1:.2f}/m")
print(f"    姿态  {ori_w:+.1f}*e:            e={e_ori:.2f} -> {ori_w * e_ori:.3f}"
      f"   斜率 {ori_w:.2f}/rad (恒定)")
print(f"    arm_deviation -0.1*L1: |Δq|~0.3 -> {-0.1 * 0.3 * 6 / 6:.3f} (按单关节平均算)")
print("  -> 姿态项的局部斜率在收敛点远大于位置项 (位置核已饱和), 奖励本身没有任何'停在默认")
print("     姿态更好'的倾向; 唯一常驻的对抗项是 arm_deviation, 量级 5-15%。")

print("\n  训练分布 vs 本次评估样本:")
print(f"    训练 |offset| 中位 {np.median(offs):.3f} rad; 本脚本用 0.025/0.05/0.1/0.2 单轴目标")
print(f"    单轴 0.2 rad 在训练分布中的分位: {(offs < 0.2).mean():.1%}")
print("    -> 0.2 rad 单轴目标的合成转角只有 0.2 rad, 落在训练分布的 **低尾**",
      f"({(offs < 0.2).mean():.1%} 的训练目标比它小), 不是中上段。")
print("       也就是说本脚本测的是比训练典型值更**容易**的目标, 而策略在这些目标上")
print("       仍然只做到 38% 的幅值、方向偏 42° —— 这使结论更强, 不是更弱。")

# ---------------------------------------------------- E. is arm_deviation the culprit? ---
print("\n" + "=" * 104)
print("E. arm_deviation 的 L1 代价是否把方向拉偏? (L1-最优 vs L2-最优 vs actor)")
print("=" * 104)
print("  `arm_deviation` 罚的是 sum|q - q_default|, 即关节运动的 L1 范数, 而 DLS 解最小化的是")
print("  L2 范数。两者的最优方向通常不同 —— 如果 actor 的方向更接近 L1 解, 那就是它。")
print(f"  {'命令':<14}{'L1 vs L2 夹角':>15}{'actor vs L2':>14}{'actor vs L1':>14}{'更接近':>9}")
l1l2, av_l2, av_l1 = [], [], []
a_neutral0 = actor_arm_action((0.0, 0.0, 0.0))
for label, rpy in CMDS:
    d_l2 = tk.local_ik(tk.DEFAULT_Q, P_NEUTRAL, tk.command_R(rpy))[0] - tk.DEFAULT_Q
    d_l1 = tk.local_ik(tk.DEFAULT_Q, P_NEUTRAL, tk.command_R(rpy), norm="l1")[0] - tk.DEFAULT_Q
    a_c = actor_arm_action(rpy)
    d_a = JOINT_SCALE * (a_c - a_neutral0)

    def _ang(u, v):
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-9 or nv < 1e-9:
            return float("nan")
        return math.degrees(math.acos(float(np.clip(u @ v / (nu * nv), -1, 1))))

    e12, e2, e1 = _ang(d_l2, d_l1), _ang(d_a, d_l2), _ang(d_a, d_l1)
    l1l2.append(e12); av_l2.append(e2); av_l1.append(e1)
    print(f"  {label:<14}{e12:>14.1f}°{e2:>13.1f}°{e1:>13.1f}°{'L1' if e1 < e2 else 'L2':>9}")
print(f"  中位: L1 vs L2 {np.median(l1l2):.1f}°, actor vs L2 {np.median(av_l2):.1f}°, "
      f"actor vs L1 {np.median(av_l1):.1f}°")
print("  -> 若 L1 vs L2 远小于 actor vs L2, 则 arm_deviation 解释不了 actor 的方向偏差。")

print("\n  E2. actor 的响应是否更接近**工具轴**语义的 IK? (坐标系约定错配的判定)")
print("  命令是按机身轴构造的 (左乘 offset)。若 actor 学到的是工具轴语义 (右乘), 它的关节")
print("  方向应对齐 tool-mode 的 IK 解, 而不是当前命令对应的 chassis-mode 解。")
print(f"  {'命令':<14}{'actor vs chassis':>18}{'actor vs tool':>15}{'更接近':>9}")
a_l2c, a_l2t = [], []
for label, rpy in CMDS:
    d_c = tk.local_ik(tk.DEFAULT_Q, P_NEUTRAL, tk.command_R(rpy, "chassis"))[0] - tk.DEFAULT_Q
    d_t = tk.local_ik(tk.DEFAULT_Q, P_NEUTRAL, tk.command_R(rpy, "tool"))[0] - tk.DEFAULT_Q
    d_a = JOINT_SCALE * (actor_arm_action(rpy) - a_neutral0)

    def _ang2(u, v):
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-9 or nv < 1e-9:
            return float("nan")
        return math.degrees(math.acos(float(np.clip(u @ v / (nu * nv), -1, 1))))

    ec, et = _ang2(d_a, d_c), _ang2(d_a, d_t)
    a_l2c.append(ec); a_l2t.append(et)
    print(f"  {label:<14}{ec:>17.1f}°{et:>14.1f}°{'tool' if et < ec else 'chassis':>9}")
print(f"  中位: vs chassis {np.median(a_l2c):.1f}°, vs tool {np.median(a_l2t):.1f}°")
print("  -> 若 vs tool 明显更小, 说明 actor 实现的是工具轴语义, 与命令的机身轴语义不一致。")

if args_cli.skip_f:
    env.close()
    app.close()
    sys.exit(0)

# --------------------------------- F. offline supervised capacity test (not a policy) ----
print("\n" + "=" * 104)
print("F. 离线监督表达力测试: 用同样的网络结构直接学习局部 IK 映射")
print("=" * 104)
print("  **这不是策略性能**, 只用来判断'同样的输入 + 同样的网络形状'能否表达这个映射。")
print("  输入 = 真实 policy observation (117 维, 含 joint_pos/命令/历史), "
      "标签 = (q_ik_local - default)/scale。")
print("  说明: 输入里的 last_action 在采样时统一置零, 否则它就是与标签无关的陈旧噪声。")

ROUNDS = max(1, 2400 // N)   # samples scale with envs; ~2400 is enough to separate
                            # 'cannot represent' from 'not fitted'
rng2 = np.random.default_rng(3)
obs_list, tgt_list, dropped = [], [], 0
for _ in range(ROUNDS):
    # States are built as the local-IK solution for a *small* attitude command, so they lie
    # on the set of poses the task can actually hold -- the same kind of state the policy
    # visits. Sampling joint angles independently (the first attempt) puts the arm at poses
    # whose distance back to the commanded pose is often unreachable, and 1516 of 1536
    # samples were dropped for that reason.
    rpy0 = rng2.uniform(-0.2, 0.2, size=(N, 3))
    rpy1 = rng2.uniform(np.array([r[0] for r in tk.TRAIN_BOX_RPY]),
                        np.array([r[1] for r in tk.TRAIN_BOX_RPY]), size=(N, 3))
    q_batch, tgt_batch = np.zeros((N, 6)), np.zeros((N, 6))
    good = np.ones(N, dtype=bool)
    for i in range(N):
        q0, ep0, er0, ms0, _ = tk.local_ik(tk.DEFAULT_Q, P_NEUTRAL, tk.command_R(tuple(rpy0[i])))
        q1, ep1, er1, ms1, _ = tk.local_ik(q0, P_NEUTRAL, tk.command_R(tuple(rpy1[i])))
        if min(ep0, ep1) > 1e-3 or min(er0, er1) > 1e-3 or max(ms0, ms1) > 0.08:
            good[i] = False
            continue
        q_batch[i] = q0
        tgt_batch[i] = (q1 - tk.DEFAULT_Q) / JOINT_SCALE
    dropped += int((~good).sum())
    robot.write_joint_state_to_sim(
        torch.tensor(q_batch, dtype=torch.float32, device=DEV),
        torch.zeros(N, 6, device=DEV), joint_ids=arm_ids)
    for i in range(N):
        set_command(tuple(rpy1[i]))
    u.action_manager.action[:] = 0.0
    u.scene.write_data_to_sim()
    u.sim.step()
    o = observe()
    obs_list.append(o.clone())
    tgt_list.append(tgt_batch)
X = torch.cat(obs_list).float().to(DEV)
Y = torch.tensor(np.concatenate(tgt_list), dtype=torch.float32, device=DEV)
keep = ~torch.isnan(Y).any(dim=1)
X, Y = X[keep], Y[keep]
print(f"  样本 {len(X)} 条 (生成时丢弃 {dropped} 条参考解不可用的); "
      f"输入 {tuple(X.shape)}, 标签 {tuple(Y.shape)} (只监督 6 个臂通道)")

torch.manual_seed(0)
net_sup = torch.nn.Sequential(
    torch.nn.Linear(X.shape[1], 512), torch.nn.ELU(),
    torch.nn.Linear(512, 256), torch.nn.ELU(),
    torch.nn.Linear(256, 128), torch.nn.ELU(),
    torch.nn.Linear(128, 6),   # arm channels only: the 2 base channels answer a different question
).to(DEV)
opt = torch.optim.Adam(net_sup.parameters(), lr=1e-3)
split = int(0.85 * len(X))
idx = torch.randperm(len(X))
tr, te = idx[:split], idx[split:]
err = lambda p, t: float(torch.mean(torch.abs(p - t)).item()) * JOINT_SCALE  # noqa: E731, per-joint rad
best_val, best_state, patience = float("inf"), None, 0
for epoch in range(1500):
    net_sup.train()
    perm = tr[torch.randperm(len(tr))]
    for b in range(0, len(perm), 256):
        mb = perm[b:b + 256]
        opt.zero_grad()
        loss = torch.mean((net_sup(X[mb]) - Y[mb]) ** 2)
        loss.backward()
        opt.step()
    net_sup.eval()
    with torch.inference_mode():
        v = float(torch.mean((net_sup(X[te]) - Y[te]) ** 2))
    if v < best_val - 1e-9:
        best_val, best_state, patience = v, {k: t.clone() for k, t in net_sup.state_dict().items()}, 0
    else:
        patience += 1
        if patience > 60:      # early stop: the point is generalisation, not fitting the train split
            break
net_sup.load_state_dict(best_state)
with torch.inference_mode():
    pred_te = net_sup(X[te])
    # the *actor's* error on the same held-out samples, same input, same metric: without this
    # the supervised number cannot be compared to anything
    a_actor = net(X[te])[:, 2:]
print(f"  早停于 epoch {epoch}, 验证集最优 MSE {best_val:.5f}")
print(f"  监督网络 验证集: 每关节平均绝对误差 {err(pred_te, Y[te]):.4f} rad")
print(f"  actor 在**同一批**样本上: 每关节平均绝对误差 {err(a_actor, Y[te]):.4f} rad")
print(f"  比值 actor/监督 = {err(a_actor, Y[te]) / max(err(pred_te, Y[te]), 1e-9):.2f}x")
print("  -> 若监督误差远小于 actor 的, 则同样的输入与网络形状足以表达这个映射,")
print("     问题在优化/训练信号, 不在表达能力; 若两者同量级, 则表达能力就是瓶颈。")

# --------------------- F2: same net, narrow distribution -- capacity or complexity? ------
print("\n  F2. 同样的网络, 只学近中性小范围 (判定: 容量 vs 分布复杂度)")
print("  若窄范围拟合得好而宽范围拟合不好, 说明容量够、难在'把随位形变化的映射学出来',")
print("  而不是'根本表达不了'。")


def _fit_and_report(rpy0_hi, rpy1_hi, rounds, tag):
    rng3 = np.random.default_rng(11)
    obs_l, tgt_l = [], []
    for _ in range(rounds):
        r0 = rng3.uniform(-rpy0_hi, rpy0_hi, size=(N, 3))
        r1 = rng3.uniform(-rpy1_hi, rpy1_hi, size=(N, 3))
        qb, tb = np.zeros((N, 6)), np.zeros((N, 6))
        for i in range(N):
            q0, ep0, er0, ms0, _ = tk.local_ik(tk.DEFAULT_Q, P_NEUTRAL, tk.command_R(tuple(r0[i])))
            q1, ep1, er1, ms1, _ = tk.local_ik(q0, P_NEUTRAL, tk.command_R(tuple(r1[i])))
            if min(ep0, ep1) > 1e-3 or min(er0, er1) > 1e-3 or max(ms0, ms1) > 0.08:
                continue
            qb[i] = q0
            tb[i] = (q1 - tk.DEFAULT_Q) / JOINT_SCALE
        robot.write_joint_state_to_sim(
            torch.tensor(qb, dtype=torch.float32, device=DEV),
            torch.zeros(N, 6, device=DEV), joint_ids=arm_ids)
        for i in range(N):
            set_command(tuple(r1[i]))
        u.action_manager.action[:] = 0.0
        u.scene.write_data_to_sim()
        u.sim.step()
        obs_l.append(observe().clone())
        tgt_l.append(tb)
    Xn = torch.cat(obs_l).float().to(DEV)
    Yn = torch.tensor(np.concatenate(tgt_l), dtype=torch.float32, device=DEV)
    torch.manual_seed(0)
    netn = torch.nn.Sequential(
        torch.nn.Linear(Xn.shape[1], 512), torch.nn.ELU(),
        torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 6),
    ).to(DEV)
    on = torch.optim.Adam(netn.parameters(), lr=1e-3)
    sp = int(0.85 * len(Xn))
    ix = torch.randperm(len(Xn))
    trn, ten = ix[:sp], ix[sp:]
    best_v, best_s, pat = float("inf"), None, 0
    for ep in range(1500):
        netn.train()
        pm = trn[torch.randperm(len(trn))]
        for b in range(0, len(pm), 256):
            mb = pm[b:b + 256]
            on.zero_grad()
            torch.mean((netn(Xn[mb]) - Yn[mb]) ** 2).backward()
            on.step()
        netn.eval()
        with torch.inference_mode():
            v = float(torch.mean((netn(Xn[ten]) - Yn[ten]) ** 2))
        if v < best_v - 1e-9:
            best_v, best_s, pat = v, {k: t.clone() for k, t in netn.state_dict().items()}, 0
        else:
            pat += 1
            if pat > 60:
                break
    netn.load_state_dict(best_s)
    with torch.inference_mode():
        pn, an = netn(Xn[ten]), net(Xn[ten])[:, 2:]
    print(f"    {tag:<34} 样本 {len(Xn):>4}  epoch {ep:>4}  "
          f"监督 {err(pn, Yn[ten]):.4f} rad  actor {err(an, Yn[ten]):.4f} rad")
    return err(pn, Yn[ten]), err(an, Yn[ten])


narrow = _fit_and_report(0.08, 0.10, max(1, 1200 // N), "近中性 (|cmd0|,|cmd1| <= 0.08/0.10)")
wide = (err(pred_te, Y[te]), err(a_actor, Y[te]))
print(f"    对照 (F 的宽分布)                   样本 {len(X[te]):>4}  "
      f"监督 {wide[0]:.4f} rad  actor {wide[1]:.4f} rad")
print(f"  -> 窄分布监督误差 {narrow[0]:.4f} vs 宽分布 {wide[0]:.4f}")

env.close()
app.close()
