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

"""Stage 4.5c: is the fault in the reward, in the policy, or in the control chain?

4.5b established that the commanded tool attitudes are reachable and that the trained policy
does not go to them -- a 0.2 rad yaw command produces 0.013 rad of motion. This script asks
where between those two facts the failure lives, by measurement rather than argument.

Task A  IK oracle: put the arm at the exact IK solution for a commanded attitude and read the
        *real* reward manager term by term. If arriving at the target lowers the total reward,
        the reward is the fault and the offending term is named. If it raises the total, the
        reward is doing its job and the policy is not.
Task A' the same targets driven through the ordinary action interface and joint controller,
        because a state-injection result is not evidence that the actuators can get there.
Task B  policy sensitivity: vary only the attitude command in a fixed scene, through the real
        observation manager, and measure the mean action's response. This separates "the
        policy ignores the command" from "the policy responds but too little".
Task C  reward shape, exploration in joint units, and the end-to-end chain for yaw +0.2.
Task D  the two attitude-error numbers that disagreed in training (0.096 rad logged against
        0.477 rad implied by the reward), measured side by side on the same rollout.

What is deliberately *not* assumed: that a linear attitude penalty beside an exponential
position reward is a defect (it is a choice), and that std 0.13 is too little exploration (a
number without units says nothing). Both are measured before anything is changed.

Usage: python scripts/ranger_cr10/check_teleop_attitude_rootcause.py --policy <policy.pt>
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description="Teleoperation attitude-tracking root cause.")
parser.add_argument("--policy", required=True)
parser.add_argument("--steps", type=int, default=120, help="steps per dynamics phase")
parser.add_argument("--only-a", action="store_true", help="stop after the IK oracle")
parser.add_argument("--only-b", action="store_true", help="jump straight to task B")
parser.add_argument("--only-dir", action="store_true", help="only the direction test")
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
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.tcp import tcp_pose_b  # noqa: E402

cfg = RangerCr10TeleopEnvCfg()
cfg.scene.num_envs = 4
cfg.commands.ee_pose.debug_vis = False
cfg.commands.ee_pose.resampling_time_range = (1e6, 1e6)
cfg.commands.base_velocity.resampling_time_range = (1e6, 1e6)
cfg.observations.policy.enable_corruption = False
env = gym.make("RANGER-CR10-TELEOP-WBC", cfg=cfg)
u = env.unwrapped
robot, sim = u.scene["robot"], u.sim
ee_cmd = u.command_manager.get_term("ee_pose")
vel_cmd = u.command_manager.get_term("base_velocity")
arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
base_id = robot.find_bodies("base_link")[0][0]
anchor_id = robot.find_bodies("cr10_Link6")[0][0]
net = torch.jit.load(args_cli.policy, map_location=u.device)
DEV, N = u.device, u.num_envs
DT = u.step_dt

default_q_t = robot.data.default_joint_pos[0, arm_ids].cpu().numpy()
print(f"仿真里的默认关节姿态 {np.round(default_q_t, 6)}")
print(f"(脚本常数 {tk.DEFAULT_Q} —— 两者必须一致, 否则 IK 解不能直接注入)")
assert np.allclose(default_q_t, tk.DEFAULT_Q, atol=1e-5), "默认关节姿态与运动学模块不一致"

# the arm action -> joint target path, read from the code rather than assumed:
# JointPositionAction.process_actions is `raw * scale + default`, and RangerActionsCfg sets
# scale 0.25 with use_default_offset; no clip is configured for joint_pos (the +/-0.5 clip in
# that file belongs to base_vel).
JOINT_SCALE = u.action_manager.get_term("joint_pos").cfg.scale
print(f"机械臂动作 scale = {JOINT_SCALE}  (关节目标 = 默认 + scale * 原始动作, joint_pos 未配 clip)")
HISTORY_LEN = u.observation_manager._group_obs_term_history_buffer["policy"][
    u.observation_manager._group_obs_term_names["policy"][0]].max_length
print(f"policy 观测历史长度 = {HISTORY_LEN}")


def observe():
    """The policy observation the manager would hand the policy, for the current state.

    `ObservationManager.compute()` defaults to `update_history=False`, and for a group with
    `history_length > 0` it then returns the *history buffer as `env.step` last left it*,
    ignoring the terms it just computed. Calling it that way after changing the command
    returns a stale vector and makes the policy look insensitive to a command it never saw.
    Three appends fill the whole 3-frame history with the current command, so two calls under
    different commands differ only in the command and not in how much history is new.
    """
    for _ in range(HISTORY_LEN):
        obs = u.observation_manager.compute(update_history=True)["policy"]
    return obs


def set_command(rpy, pos_offset=(0.0, 0.0, 0.0)):
    dR = iu.quat_from_euler_xyz(*[torch.tensor([v], device=DEV) for v in rpy])
    tq = iu.quat_mul(dR, torch.tensor(TCP_DEFAULT_QUAT_B, device=DEV).unsqueeze(0))[0]
    ee_cmd.pose_command_b[:, :3] = torch.tensor(np.array(TCP_DEFAULT_POS_B) + np.asarray(pos_offset),
                                                device=DEV)
    ee_cmd.pose_command_b[:, 3:] = tq
    vel_cmd.is_standing_env[:] = False
    vel_cmd.vel_command_b[:] = 0.0
    return tq


def inject(q, hold=True):
    """Put the arm at q and (optionally) hold it there with the position controller."""
    q_t = torch.tensor(q, dtype=torch.float32, device=DEV).unsqueeze(0).expand(N, -1)
    robot.write_joint_state_to_sim(q_t, torch.zeros_like(q_t), joint_ids=arm_ids)
    if hold:
        robot.set_joint_position_target(q_t, joint_ids=arm_ids)
    sim.step()


def step_rewards():
    """Per-term weighted step reward from the real reward manager, weight * term."""
    u.reward_manager.compute(DT)
    return {n: u.reward_manager._step_reward[0, i].item() for i, n in enumerate(u.reward_manager._term_names)}


def robot_joint_deviation(q):
    """|q - default| for the six arm joints, in radians -- what joint_deviation_l1 sums."""
    return np.asarray(q) - tk.DEFAULT_Q


def measured_errors():
    """TCP pose error against the commanded target, computed two independent ways.

    The command manager is refreshed first: `ee_cmd.metrics` is only written inside
    `CommandManager.compute`, which runs in `env.step`, so after a state injection the metric
    still holds the value from the last real step -- a frozen number that looks like a
    measurement and is not. (Reading it stale is what made the first version of this table
    report the same error for six different states.)
    """
    u.command_manager.compute(DT)
    _, tcp_q_b = tcp_pose_b(robot, anchor_id, base_id)
    cmd_q = ee_cmd.pose_command_b[:, 3:]
    # (a) the reward's own function
    err_reward = iu.quat_error_magnitude(tcp_q_b, cmd_q).mean().item()
    # (b) the command term's metric, which goes through the world frame
    err_metric = ee_cmd.metrics["orientation_error"].mean().item()
    return err_reward, err_metric


TARGETS = [("中性", (0.0, 0.0, 0.0))]
for mag in (0.1, 0.2):
    for axis, nm in ((0, "roll"), (1, "pitch"), (2, "yaw")):
        for sgn in (+1, -1):
            rpy = [0.0, 0.0, 0.0]
            rpy[axis] = sgn * mag
            TARGETS.append((f"{nm}{'+' if sgn > 0 else '-'} {mag}", tuple(rpy)))

# ----------------------------------------------------------------- Task A: IK oracle -----
print("\n" + "=" * 100)
print("A. IK oracle: 把机械臂放到 IK 解上, 当前奖励是否更鼓励它?")
print("=" * 100)
print("  公平性说明: 两种状态用相同的动作(全零)、相同命令、相同的关节速度(零), 保持驱动器")
print("  都指向各自的姿态。因此 action_rate / last_action 相关的项在两状态下完全相同, 不参与")
print("  差值 —— 它们在表里会显示为 0.000。单步注入不产生持续运动, 所以与速度/接触有关的项")
print("  两边同样接近 0, 这一点在表后单独说明。")

IK_SOLUTIONS = {}
rows = []
for label, rpy in TARGETS:
    tq = set_command(rpy)
    # the *nearest* branch: a far-branch solution would be dominated by arm_deviation and
    # would compare the wrong thing (see solve_ik_nearest)
    ep, er, q_ik, slack = tk.solve_ik_nearest(tk.P_NEUTRAL, tk.command_R(rpy))
    IK_SOLUTIONS[label] = (q_ik, rpy, ep, er)

    inject(tk.DEFAULT_Q, hold=True)
    set_command(rpy)
    base = step_rewards()
    base_err = measured_errors()

    inject(q_ik, hold=True)
    set_command(rpy)
    ik = step_rewards()
    ik_err = measured_errors()

    dev_base = np.abs(robot_joint_deviation(tk.DEFAULT_Q)).sum()
    dev_ik = np.abs(robot_joint_deviation(q_ik)).sum()
    rows.append(dict(label=label, rpy=rpy, ep=ep, er=er, slack=slack,
                     base=base, ik=ik, base_err=base_err, ik_err=ik_err,
                     dq=float(np.abs(q_ik - tk.DEFAULT_Q).max()),
                     dev=(dev_base, dev_ik)))

terms = [n for n in rows[0]["base"]]
print(f"\n  {'目标':<14}{'IK位置残差':>11}{'IK姿态残差':>11}{'基线姿态误差':>13}"
      f"{'IK姿态误差':>11}{'总奖励 基线':>12}{'总奖励 IK':>11}{'差值':>10}")
for r in rows:
    tb, ti = sum(r["base"].values()), sum(r["ik"].values())
    print(f"  {r['label']:<14}{r['ep']:>11.2e}{r['er']:>11.4f}{r['base_err'][0]:>13.4f}"
          f"{r['ik_err'][0]:>11.4f}{tb:>12.4f}{ti:>11.4f}{ti - tb:>+10.4f}"
          f"   L1|Δq| 基线 {r['dev'][0]:.3f} / IK {r['dev'][1]:.3f}")

print(f"\n  逐项奖励差值 (IK - 基线), 单位 = weight × term (每步再乘 dt={DT} 才是实际回报):")
hdr = f"  {'term':<36}" + "".join(f"{r['label'][:11]:>13}" for r in rows)
print(hdr)
for name in terms:
    line = f"  {name:<36}"
    for r in rows:
        line += f"{r['ik'][name] - r['base'][name]:>13.4f}"
    print(line)
print(f"  {'总计':<36}" + "".join(
    f"{sum(r['ik'].values()) - sum(r['base'].values()):>+13.4f}" for r in rows))

if args_cli.only_a:
    env.close()
    app.close()
    sys.exit(0)

if args_cli.only_b:
    print("\n(仅任务 B; 跳过 A 与 A' 的表格)")

# ------------------------------------------------------------- A' dynamics execution -----
print("\n" + "=" * 100)
print("A'. 动力学执行: 通过原有动作接口把关节驱动到 IK 解, 而不是注入状态")
print("=" * 100)
print("  动作 = (q_ik - 默认) / scale, 每一步都下发同一个动作, 让关节位置控制器自己去收敛。")
print("  这检验的是驱动器、关节限位与动作接口能不能真的到那里。")
print(f"  {'目标':<14}{'|Δq| 需要':>11}{'实际到位误差':>14}{'关节误差(rad)':>15}"
      f"{'EE 位置误差':>12}{'EE 姿态残差':>13}{'达标':>7}")
dyn_rows = []
for label, rpy in TARGETS:
    q_ik, _, _, _ = IK_SOLUTIONS[label]
    set_command(rpy)
    obs, _ = env.reset()
    a = torch.zeros(N, u.action_manager.total_action_dim, device=DEV)
    a[:, 2:] = torch.tensor((q_ik - tk.DEFAULT_Q) / JOINT_SCALE, dtype=torch.float32, device=DEV)
    for _ in range(args_cli.steps):
        obs, _, _, _, _ = env.step(a)
        set_command(rpy)
    q_now = robot.data.joint_pos[0, arm_ids].cpu().numpy()
    pos_b, quat_b = tcp_pose_b(robot, anchor_id, base_id)
    R_now = tk.quat_to_R(quat_b[0].cpu().numpy())
    pos_err = float(np.linalg.norm(pos_b[0].cpu().numpy() - np.array(TCP_DEFAULT_POS_B)))
    rot_err = float(np.linalg.norm(tk.log_R(tk.command_R(rpy) @ R_now.T)))
    dq_err = float(np.abs(q_now - q_ik).max())
    ok = dq_err < 0.05 and rot_err < 0.05
    dyn_rows.append((label, dq_err, pos_err, rot_err, ok))
    print(f"  {label:<14}{np.abs(q_ik - tk.DEFAULT_Q).max():>11.3f}{dq_err:>14.4f}"
          f"{dq_err:>15.4f}{pos_err:>12.4f}{rot_err:>13.4f}{'是' if ok else '否':>7}")
n_ok = sum(r[4] for r in dyn_rows)
print(f"  -> {n_ok}/{len(dyn_rows)} 个目标能通过真实控制器到位")

# ------------------------------------------------- Task B: policy command sensitivity ----
print("\n" + "=" * 100)
print("B. 策略对姿态命令的敏感度 (同一状态, 只改姿态命令, 走真实 observation 路径)")
print("=" * 100)
# hold one fixed, mid-episode state: the arm driven to neutral by the policy itself, so the
# state the sensitivity is measured at is one the policy actually visits
set_command((0.0, 0.0, 0.0))
obs, _ = env.reset()
a0 = torch.zeros(N, u.action_manager.total_action_dim, device=DEV)
for _ in range(60):
    with torch.inference_mode():
        a0 = net(obs["policy"])
    obs, _, _, _, _ = env.step(a0)
    set_command((0.0, 0.0, 0.0))
print(f"  固定状态: 关节 {np.round(robot.data.joint_pos[0, arm_ids].cpu().numpy(), 4)}")

B_TARGETS = [("中性", (0.0, 0.0, 0.0))]
for mag in (0.025, 0.1, 0.2):
    for axis, nm in ((0, "roll"), (1, "pitch"), (2, "yaw")):
        for sgn in (+1, -1):
            rpy = [0.0, 0.0, 0.0]
            rpy[axis] = sgn * mag
            B_TARGETS.append((f"{nm}{'+' if sgn > 0 else '-'} {mag}", tuple(rpy)))

print(f"  {'命令':<14}{'actor |Δa_arm|':>16}{'最大变化关节':>14}{'Δa[0..1]':>12}"
      f"{'关节目标变化(rad)':>18}")
# The action must not be trusted until the observation is shown to have changed: a stale
# observation would make the policy look insensitive when the test is at fault.
print("\n  观测自检 (两次不同命令下 pos_commands 那片是否真的不同):")
set_command((0.0, 0.0, 0.0))
o_neutral = observe()[0].clone()
set_command((0.2, 0.0, 0.0))
o_roll = observe()[0].clone()
print(f"    整个 policy 观测的最大差 {torch.abs(o_neutral - o_roll).max().item():.3e}")
cmd_delta = torch.abs(ee_cmd.pose_command_b[0, 3:] - torch.tensor(TCP_DEFAULT_QUAT_B, device=DEV)).max()
print(f"    命令项本身与中性姿态的差 {cmd_delta.item():.3e}")

sensitivity = {}
for label, rpy in B_TARGETS:
    set_command(rpy)
    o = observe()
    with torch.inference_mode():
        a = net(o)
    sensitivity[label] = a[0, 2:].cpu().numpy()
    da = a[0, 2:].cpu().numpy() - sensitivity["中性"]
    j = int(np.argmax(np.abs(da)))
    print(f"  {label:<14}{np.linalg.norm(da):>16.5f}{ARM_JOINT_NAMES[j]:>14}{a[0, 0].item():>12.3f}"
          f"{np.abs(da).max() * JOINT_SCALE:>18.4f}")

print("\n  绝对幅值对照 (关键: 策略下发的关节目标偏移 是否够 IK 所需):")
print(f"  {'命令':<14}{'max|a_arm|x scale':>18}{'|a_arm| 范数 x scale':>20}"
      f"{'IK 所需 max|Δq|':>18}{'比值':>8}")
for label, rpy in B_TARGETS:
    set_command(rpy)
    with torch.inference_mode():
        a = net(observe())[0, 2:].cpu().numpy()
    need = None
    if label in IK_SOLUTIONS:
        q_ik = IK_SOLUTIONS[label][0]
        need = float(np.abs(q_ik - tk.DEFAULT_Q).max())
    cmd_offset = float(np.abs(a).max() * JOINT_SCALE)
    ratio = f"{cmd_offset / need:.2f}" if need and need > 1e-6 else "-"
    print(f"  {label:<14}{cmd_offset:>18.4f}{float(np.linalg.norm(a)) * JOINT_SCALE:>20.4f}"
          f"{(f'{need:.3f}' if need is not None else '-'):>18}{ratio:>8}")

print("\n  方向对比 (决定性): 策略动作的变化方向 vs IK 解的方向")
print("  幅度对而方向错的表现是残余误差 ≈ 指令本身, 与 4.5b 的观测吻合。")
print(f"  {'命令':<14}{'|Δa_arm|':>10}{'|Δq_ik|':>10}{'夹角':>9}{'cos':>8}{'幅度比':>9}")
set_command((0.0, 0.0, 0.0))
with torch.inference_mode():
    a_neutral = net(observe())[0, 2:].cpu().numpy()
for label, rpy in B_TARGETS:
    if label == "中性" or label not in IK_SOLUTIONS:
        continue
    set_command(rpy)
    with torch.inference_mode():
        a = net(observe())[0, 2:].cpu().numpy()
    da = (a - a_neutral) * JOINT_SCALE          # what the policy commands, in radians
    dq = IK_SOLUTIONS[label][0] - tk.DEFAULT_Q  # what IK requires, in radians
    na, nq = np.linalg.norm(da), np.linalg.norm(dq)
    if na < 1e-9 or nq < 1e-9:
        continue
    cos = float(np.dot(da, dq) / (na * nq))
    ang = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    print(f"  {label:<14}{na:>10.4f}{nq:>10.4f}{ang:>8.1f}°{cos:>8.3f}{na / nq:>9.2f}")

print("\n  有限差分灵敏度 J = ∂a_arm/∂[roll, pitch, yaw]  (每列 = 1 rad 命令引起的 6 关节动作变化)")
J = np.zeros((6, 3))
for col, axis in enumerate((0, 1, 2)):
    rpy_p = [0.0, 0.0, 0.0]
    rpy_p[axis] = +0.1
    rpy_m = [0.0, 0.0, 0.0]
    rpy_m[axis] = -0.1
    set_command(tuple(rpy_p))
    with torch.inference_mode():
        ap = net(observe())[0, 2:].cpu().numpy()
    set_command(tuple(rpy_m))
    with torch.inference_mode():
        am = net(observe())[0, 2:].cpu().numpy()
    J[:, col] = (ap - am) / 0.2
print(f"  {'关节':<14}{'∂/∂roll':>12}{'∂/∂pitch':>12}{'∂/∂yaw':>12}   (动作单位 / rad)")
for i, nm in enumerate(ARM_JOINT_NAMES):
    print(f"  {nm:<14}{J[i, 0]:>12.4f}{J[i, 1]:>12.4f}{J[i, 2]:>12.4f}")
print(f"  {'列范数':<14}{np.linalg.norm(J[:, 0]):>12.4f}{np.linalg.norm(J[:, 1]):>12.4f}"
      f"{np.linalg.norm(J[:, 2]):>12.4f}")
print(f"  换算成关节目标: 列范数 × scale({JOINT_SCALE}) = "
      f"{np.round(np.linalg.norm(J, axis=0) * JOINT_SCALE, 5)} rad/rad")
print(f"  对照: IK 达到这些目标所需的 |Δq| = "
      f"{[round(np.abs(IK_SOLUTIONS[l][0] - tk.DEFAULT_Q).max(), 3) for l, _ in TARGETS if l != '中性'][:6]} ...")

# ------------------------------------------------- Task C3: end-to-end control chain ------
print("\n" + "=" * 100)
print("C3. 端到端控制链路: 命令 -> 观测 -> actor -> 动作缩放 -> 关节目标 -> 实际关节 -> FK 姿态")
print("=" * 100)
for label, rpy in (("yaw+ 0.2", (0.0, 0.0, 0.2)), ("roll+ 0.2", (0.2, 0.0, 0.0))):
    _, _, q_ik, _ = tk.solve_ik_nearest(tk.P_NEUTRAL, tk.command_R(rpy))  # (ep, er, q, slack)
    set_command(rpy)
    obs, _ = env.reset()
    print(f"\n  --- 命令 {label} (IK 需要 max|Δq| = {np.abs(q_ik - tk.DEFAULT_Q).max():.3f} rad) ---")
    print(f"  {'step':>5}{'|a_arm|max':>12}{'关节目标偏移':>14}{'实际关节偏移':>14}"
          f"{'EE 位置误差':>12}{'EE 姿态误差':>12}")
    for i in range(args_cli.steps):
        with torch.inference_mode():
            a = net(obs["policy"])
        obs, _, _, _, _ = env.step(a)
        set_command(rpy)
        if i % 20 == 0 or i == args_cli.steps - 1:
            q_now = robot.data.joint_pos[0, arm_ids].cpu().numpy()
            a_arm = a[0, 2:].cpu().numpy()
            q_tgt = robot.data.joint_pos_target[0, arm_ids].cpu().numpy()
            p_b, quat_b = tcp_pose_b(robot, anchor_id, base_id)
            pos_err = float(np.linalg.norm(p_b[0].cpu().numpy() - np.array(TCP_DEFAULT_POS_B)))
            rot_err = float(np.linalg.norm(tk.log_R(tk.command_R(rpy) @ tk.quat_to_R(quat_b[0].cpu().numpy()).T)))
            print(f"  {i:>5}{np.abs(a_arm).max() * JOINT_SCALE:>12.4f}"
                  f"{np.abs(q_tgt - tk.DEFAULT_Q).max():>14.4f}"
                  f"{np.abs(q_now - tk.DEFAULT_Q).max():>14.4f}{pos_err:>12.4f}{rot_err:>12.4f}")
    print(f"    参考: 把 IK 解直接当动作下发(A' 已证)能把 max|Δq| 推到 "
          f"{np.abs(q_ik - tk.DEFAULT_Q).max():.3f} rad, EE 姿态残差 <0.012 rad")

# -------------------------------------------- D: the two orientation-error numbers ---
print("\n" + "=" * 100)
print("D. 训练记录里两个姿态误差口径 (0.096 rad vs 奖励反推 0.477 rad)")
print("=" * 100)
print("  在同一状态上同时算: (a) 奖励用的 quat_error_magnitude, (b) 命令项 metric 用的")
print("  compute_pose_error -> axis_angle 的模。")
print(f"  {'状态':<22}{'(a) 奖励口径':>14}{'(b) metric 口径':>17}{'比值':>9}")
for label, rpy in [("中性", (0.0, 0.0, 0.0)), ("yaw+ 0.2", (0.0, 0.0, 0.2)),
                   ("roll+ 0.2", (0.2, 0.0, 0.0))]:
    set_command(rpy)
    q_ik = tk.solve_ik_nearest(tk.P_NEUTRAL, tk.command_R(rpy))[2]
    for tag, q in (("在默认姿态", tk.DEFAULT_Q), ("在 IK 解", q_ik)):
        inject(q, hold=True)
        set_command(rpy)
        er_reward, er_metric = measured_errors()
        ratio = er_metric / er_reward if er_reward > 1e-9 else float("nan")
        print(f"  {label + ' ' + tag:<22}{er_reward:>14.4f}{er_metric:>17.4f}{ratio:>9.3f}")

# ---------------------------------------------------------------- C2: exploration ------
print("\n" + "=" * 100)
print("C2. 探索是否够 (把 std 换算成关节角, 而不是只看数字)")
print("=" * 100)
import glob  # noqa: E402
from tensorboard.backend.event_processing import event_accumulator  # noqa: E402

runs = sorted(glob.glob("logs/rsl_rl/ranger_cr10_teleop/*/events.out.tfevents.*"))
if runs:
    ea = event_accumulator.EventAccumulator(runs[-1])
    ea.Reload()
    std = {e.step: e.value for e in ea.Scalars("Policy/mean_std")}
    ent = {e.step: e.value for e in ea.Scalars("Loss/entropy")} if "Loss/entropy" in ea.Tags()["scalars"] else {}
    print(f"  训练日志: {runs[-1].split('/')[-3]}")
    for st in (0, 200, 500, 800, 999):
        if st in std:
            noise_joint = std[st] * JOINT_SCALE
            print(f"    iter {st:>4}: mean_std {std[st]:.4f} -> 关节目标噪声 {noise_joint:.4f} rad/步"
                  f"   (命令存活 40-80 步, 随机游走累计 sigma ~ "
                  f"{noise_joint * (60 ** 0.5):.3f} rad)"
                  + (f"   熵 {ent[st]:.2f}" if st in ent else ""))
print(f"  对照: IK 达到单轴姿态目标需要的 max|Δq| = 0.12-0.43 rad")
print(f"  也就是说: 探索噪声在一条命令的存活期内累计到 0.20-0.25 rad, 与所需量同量级 ——")
print(f"  仅凭 std=0.13 不能判定探索不足, 需要看受控实验。")

# ------------------------------- E: is the direction error a local linearisation artefact?
print("\n" + "=" * 100)
print("E. 方向偏差随位形是否变化 —— **本节结论不可用, 仅作记录**")
print("=" * 100)
print("  做法: 在每个位形上把姿态命令从 0 变到 yaw+0.2, 比较策略动作变化方向与 IK 所需方向。")
print("  ⚠️ 为什么不能用: 被扰动状态上解出的 |Δq_ik| 高达 0.81 rad, 而状态本身只偏离默认")
print("     0.25 rad —— 这说明 IK 仍然落在远分支上, 没有收敛到局部解。用错误的参考方向去")
print("     衡量策略方向, 得到的 cos 忽正忽负 (cos 在 +0.88 与 -0.79 之间跳) 是参考的毛病,")
print("     不是策略的。**该表不用于任何结论。** 要做对需要: 以当前状态为初值做局部 IK 并")
print("     验证残差, 只保留收敛的样本。")
rng = np.random.default_rng(7)
_, _, q_ik_yaw, _ = tk.solve_ik_nearest(tk.P_NEUTRAL, tk.command_R((0.0, 0.0, 0.2)))
dq_yaw = q_ik_yaw - tk.DEFAULT_Q
print(f"  {'状态':<26}{'|Δa_arm|':>11}{'|Δq_ik|':>10}{'夹角':>9}{'cos':>8}{'幅度比':>9}")
cosines = []
for trial in range(5):
    if trial == 0:
        q_state = tk.DEFAULT_Q
        tag = "默认姿态"
    else:
        q_state = np.clip(tk.DEFAULT_Q + rng.uniform(-0.25, 0.25, size=6), *tk.LIMITS.T)
        tag = f"训练分布内扰动 {trial}"
    inject(q_state, hold=True)
    set_command((0.0, 0.0, 0.0))
    with torch.inference_mode():
        a0 = net(observe())[0, 2:].cpu().numpy()
    set_command((0.0, 0.0, 0.2))
    with torch.inference_mode():
        a1 = net(observe())[0, 2:].cpu().numpy()
    da = (a1 - a0) * JOINT_SCALE
    # re-solved from *this* state, minimising distance from it: choosing a branch that is
    # near the default would be answering a different question (see solve_ik_near_state)
    T_state = tk.fk(q_state)
    _, _, q_ik_s = tk.solve_ik_near_state(T_state[:3, 3], tk.command_R((0.0, 0.0, 0.2)), q_state)
    dq = q_ik_s - q_state
    na, nq = np.linalg.norm(da), np.linalg.norm(dq)
    cos = float(np.dot(da, dq) / (na * nq)) if na > 1e-9 and nq > 1e-9 else float("nan")
    cosines.append(cos)
    ang = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    print(f"  {tag:<26}{na:>11.4f}{nq:>10.4f}{ang:>8.1f}°{cos:>8.3f}{na / nq:>9.2f}")
print(f"  cos 均值 {np.nanmean(cosines):.3f}, 标准差 {np.nanstd(cosines):.3f}"
      f"  -> 参考方向不可信, 不下结论 (见上)")

env.close()
app.close()
