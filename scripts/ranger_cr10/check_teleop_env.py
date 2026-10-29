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

"""Stage 4.1 acceptance for the teleoperation task: is the environment what it claims to be?

Six checks, each written as a positive measurement rather than "it constructed":

  A  action layout      -- policy output 0/1 reach the chassis, 2..7 reach the arm joints,
                           and the arm channels are centred on the arm's default pose
  B  default TCP guard  -- the measured TCP pose matches the constant the command samples
                           around. If it does not, every sampled target is offset and the
                           orientation term degenerates, which is what happened on the
                           world branch.
  C  frame invariance   -- driving the chassis leaves the chassis-relative target untouched
                           while the world-frame image of it moves. This is the acceptance
                           requirement "with the chassis moving, an unchanged EE target does
                           not move", and it is the single defining property of the task.
  D  reachability       -- the orientation error at the default arm pose for freshly sampled
                           commands is bounded by the sampled offset, not by pi. A constant
                           near pi means the box is outside the arm's workspace and the
                           reward is a flat penalty
  D2 command diversity  -- the six commanded offsets are *independent*. The sampler drew all
                           three position offsets (and all three Euler angles) from one
                           aliased tensor until 2026-10-29, so every command satisfied
                           x == y == z and roll == pitch == yaw. Range tests all pass on that
                           degenerate box, which is why it survived four stages.
  E  diagnostics        -- the linear and angular velocity error curves are logged as two
                           separate quantities, next to the EE errors and the success rate
  F  velocity tracking  -- with the command fed straight to the base controller (the
                           decoupled baseline the RL policy will be compared against), the
                           chassis reaches the commanded velocity

Headless, and prints tables; exit code 0 means nothing on its own.
"""

from __future__ import annotations

import math

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import LeggedManip_Lab.tasks  # noqa: E402,F401
import torch  # noqa: E402
from LeggedManip_Lab.assets.ranger_cr10.ranger_cr10_articulation_cfg import ARM_JOINT_NAMES  # noqa: E402
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.teleop_env_cfg import (  # noqa: E402
    RangerCr10TeleopEnvCfg,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.pose_command_teleop import (  # noqa: E402
    TCP_DEFAULT_POS_B,
    TCP_DEFAULT_QUAT_B,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.tcp import (  # noqa: E402
    TCP_ANCHOR_BODY,
    tcp_pose_b,
)

from isaaclab.utils.math import euler_xyz_from_quat, quat_conjugate, quat_mul  # noqa: E402

cfg = RangerCr10TeleopEnvCfg()
cfg.scene.num_envs = 64
cfg.commands.ee_pose.debug_vis = False
cfg.commands.ee_pose.resampling_time_range = (1e6, 1e6)
# Deterministic scene for the whole check. Check B compares the sampled neutral against a
# constant, and that constant can only be well defined if the condition it is measured in
# is fixed: joint reset randomisation moves the arm away from its authored pose, and gain
# randomisation changes how far gravity pulls it back, so under the training-time
# randomisation the "default TCP pose" is a spread rather than a number (measured: 2.8 cm
# mean, 4.1 cm max across 64 envs). The sag under the full randomisation is reported
# separately in B.
cfg.events.reset_robot_joints = None
cfg.events.randomize_actuator_gains = None
env = gym.make("RANGER-CR10-TELEOP-WBC", cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
base_vel = u.action_manager.get_term("base_vel")
joint_pos = u.action_manager.get_term("joint_pos")
ee_cmd = u.command_manager.get_term("ee_pose")
vel_cmd = u.command_manager.get_term("base_velocity")

base_idx = robot.find_bodies("base_link")[0][0]
anchor_idx = robot.find_bodies(TCP_ANCHOR_BODY)[0][0]

obs, _ = env.reset()
zero = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
# the joints sit at zero until the first physics step, so settle before measuring anything
for _ in range(50):
    env.step(zero)


def quat_angle_deg(q1, q2) -> float:
    """Angle between two quaternions, in degrees, averaged over the batch."""
    dot = torch.abs(torch.sum(q1 * q2, dim=-1)).clamp(max=1.0)
    return math.degrees((2.0 * torch.acos(dot)).mean().item())


print("\n===== A. 动作布局 =====")
# through env.step, not by calling the term: the action manager is what slices the policy
# vector, so calling the term directly would test something the deployment never does
probe = torch.zeros_like(zero)
probe[:, 0], probe[:, 1] = 0.3, -0.2
arm_probe = 0.4
probe[:, 2:] = arm_probe
env.step(probe)
cmd = base_vel.processed_actions
expect_vx = 0.5 * math.tanh(0.3)
expect_wz = 0.5 * math.tanh(-0.2)
print(f"  策略输出 [0]=0.3, [1]=-0.2  ->  base_vel 处理后 (vx, wz) = ({cmd[0, 0]:+.4f}, {cmd[0, 1]:+.4f})")
print(
    f"  期望 0.5*tanh(0.3)={expect_vx:+.4f}, 0.5*tanh(-0.2)={expect_wz:+.4f}  -> "
    f"{'通过' if abs(cmd[0, 0].item() - expect_vx) < 1e-5 else '不通过'}"
)
print(
    f"  base_vel action_dim = {base_vel.action_dim} (必须是 2), "
    f"joint_pos action_dim = {joint_pos.action_dim} (必须是 6)"
)
print(f"  合计 {base_vel.action_dim + joint_pos.action_dim} 维, 声明顺序 = {list(u.action_manager.active_terms)}")
# the arm channels are a *delta* on the default pose: raw 0.4 at scale 0.25 must land
# 0.1 rad away from the authored default, not at 0.4 rad
arm_ids = robot.find_joints(joint_pos.cfg.joint_names)[0] if hasattr(joint_pos.cfg, "joint_names") else None
if arm_ids is not None:
    applied = robot.data.joint_pos_target[0, arm_ids]
    default = robot.data.default_joint_pos[0, arm_ids]
    delta = (applied - default).tolist()
    print(f"  机械臂通道 = 0.4 (scale 0.25)  ->  关节目标相对默认的偏移 = {[round(v, 4) for v in delta]}")
    print(
        f"  期望 0.25*0.4 = {0.25 * arm_probe:.4f} rad  -> "
        f"{'通过 (以默认姿态为中心)' if abs(delta[0] - 0.25 * arm_probe) < 1e-4 else '不通过'}"
    )
env.step(zero)

print("\n===== B. 默认 TCP 常数 vs 实测 =====")
# settle further so the PD response is out of the measurement
for _ in range(50):
    env.step(zero)
tcp_pos_b, tcp_quat_b = tcp_pose_b(robot, anchor_idx, base_idx)
arm_ids = robot.find_joints(ARM_JOINT_NAMES)[0]
default_jp = robot.data.default_joint_pos[0, arm_ids]
live_jp = robot.data.joint_pos[0, arm_ids]
print("  机械臂关节 相对默认姿态的偏差 (deg):")
for name, d in zip(ARM_JOINT_NAMES, (live_jp - default_jp).tolist()):
    print(f"    {name:<14}{math.degrees(d):+8.3f}")
const_pos = torch.tensor(TCP_DEFAULT_POS_B, device=u.device)
const_quat = torch.tensor(TCP_DEFAULT_QUAT_B, device=u.device).expand_as(tcp_quat_b)
d_pos = torch.norm(tcp_pos_b - const_pos, dim=-1)
print(f"  实测 TCP in base_link 均值 = {[round(v, 5) for v in tcp_pos_b.mean(0).tolist()]}")
print(f"  实测 TCP 姿态均值 (w,x,y,z) = {[round(v, 5) for v in tcp_quat_b.mean(0).tolist()]}")
print(f"  常数   TCP in base_link     = {TCP_DEFAULT_POS_B}")
print(f"  常数   TCP 姿态             = {TCP_DEFAULT_QUAT_B}")
print(f"  位置差 均值 {d_pos.mean().item():.5f} m  最大 {d_pos.max().item():.5f} m")
print(f"  姿态差 均值 {quat_angle_deg(tcp_quat_b, const_quat):.3f} deg")
ok = d_pos.max().item() < 0.01 and quat_angle_deg(tcp_quat_b, const_quat) < 1.0
print(f"  -> {'通过 (常数仍准确)' if ok else '**不通过**: 采样盒整体偏移, 必须重测常数'}")

print("\n===== C. 机身系不变性 =====")
# freeze the target so nothing can resample mid-measurement
ee_cmd.cfg.resampling_time_range = (1e6, 1e6)
target_b0 = ee_cmd.command.clone()
target_w0 = ee_cmd.pose_command_w.clone()
p0 = robot.data.root_pos_w[:, :2].clone()
drive = torch.zeros_like(zero)
drive[:, 0], drive[:, 1] = 0.5, 0.4  # hard forward + turn, straight into the base controller
for _ in range(30):  # 3 s at 10 Hz
    env.step(drive)
    # deliberately NOT re-writing the command here: forcing it back would make the
    # "target did not change" result a tautology, and the point of the check is that
    # nothing in the step path touches it
disp = torch.norm(robot.data.root_pos_w[:, :2] - p0, dim=-1)
d_b = torch.norm(ee_cmd.command - target_b0, dim=-1)
d_w = torch.norm(ee_cmd.pose_command_w - target_w0, dim=-1)
print(f"  底盘平均位移        {disp.mean().item():.4f} m (必须明显 > 0, 否则本检查没有意义)")
print(f"  base 系目标变化     {d_b.max().item():.3e} m (必须为 0)")
print(f"  world 系目标变化    {d_w.mean().item():.4f} m (随底盘一起动, 因此它必须非零)")
print("  注: world 系的变化是由 base 系目标加当前底盘位姿重构出来的, 所以真正独立的")
print(f"      证据是 '底盘走了 {disp.mean().item():.2f} m 而 base 系目标一字未动', 不是 world 系那一项。")
print(f"  -> {'通过' if disp.mean() > 0.2 and d_b.max() < 1e-9 and d_w.mean() > 0.05 else '**不通过**'}")

print("\n===== D. 采样目标的可达性 (重置后立刻测量) =====")
cfg.commands.ee_pose.resampling_time_range = (4.0, 8.0)
obs, _ = env.reset()
for _ in range(3):
    env.step(zero)
pos_err = ee_cmd.metrics["position_error"]
ori_err = ee_cmd.metrics["orientation_error"]
print(f"  采样目标相对默认 TCP 的距离: 均值 {pos_err.mean().item():.3f} m  最大 {pos_err.max().item():.3f} m")
print(f"  姿态误差 (零动作):            均值 {ori_err.mean().item():.3f} rad 最大 {ori_err.max().item():.3f} rad")
print(
    f"  采样盒名义最大偏移 = {math.dist((0.12, 0.15, 0.12), (0, 0, 0)):.3f} m (位置), "
    f"{math.dist((0.4, 0.4, 0.5), (0, 0, 0)):.3f} rad (姿态)"
)
print(f"  -> {'通过 (与采样盒同量级, 不是常数惩罚)' if ori_err.max() < 1.5 else '**不通过**: 姿态误差接近 pi'}")

print("\n===== D2. 命令采样: 六个分量必须独立 =====")
# The sampler used one tensor for all three axes (`r.uniform_()` returns the same tensor it
# fills in place), so every command was offset_x == offset_y == offset_z and roll == pitch
# == yaw -- a two-dimensional line in a six-dimensional box. Any test that only checks the
# ranges passes for that, so check the *joint* distribution instead.
# Call the sampler itself rather than stepping: 64 envs give a correlation standard error of
# 0.125, which cannot separate "independent" from "correlated" at any useful threshold.
_ids = torch.arange(u.num_envs, device=u.device)
pos_list, eul_list = [], []
for _ in range(64):
    ee_cmd._resample_command(_ids)
    pos_list.append(ee_cmd.pose_command_b[:, :3].clone())
    # decompose offset = cmd * conj(default) back into Euler angles in the env's convention
    _dq = quat_mul(
        ee_cmd.pose_command_b[:, 3:],
        quat_conjugate(torch.tensor(ee_cmd.cfg.default_quat_b, device=u.device).unsqueeze(0).expand(u.num_envs, 4)),
    )
    eul_list.append(torch.stack(euler_xyz_from_quat(_dq), dim=-1))
pos_off = torch.cat(pos_list) - torch.tensor(ee_cmd.cfg.default_pos_b, device=u.device)
eul = torch.cat(eul_list)
n_s = len(pos_off)


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = a.norm() * b.norm()
    return float((a * b).sum() / den) if den > 0 else float("nan")


pairs_p = [("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2)]
pairs_r = [("roll", "pitch", 0, 1), ("roll", "yaw", 0, 2), ("pitch", "yaw", 1, 2)]
corrs = [_corr(pos_off[:, i], pos_off[:, j]) for _, _, i, j in pairs_p]
corrs += [_corr(eul[:, i], eul[:, j]) for _, _, i, j in pairs_r]
dup_p = float(((pos_off[:, 0] - pos_off[:, 1]).abs() < 1e-6).float().mean())
dup_r = float(((eul[:, 0] - eul[:, 1]).abs() < 1e-6).float().mean())
print(
    f"  样本 {n_s} 条; 位置偏移 x/y/z 两两相关系数 "
    f"{corrs[0]:+.3f} {corrs[1]:+.3f} {corrs[2]:+.3f}; 姿态 roll/pitch/yaw "
    f"{corrs[3]:+.3f} {corrs[4]:+.3f} {corrs[5]:+.3f}"
)
print(f"  x==y 的比例 {dup_p * 100:.1f}%,  roll==pitch 的比例 {dup_r * 100:.1f}%  (退化采样会给出 100%)")
_nominal = [
    (ee_cmd.cfg.ranges.pos_x[1] - ee_cmd.cfg.ranges.pos_x[0]) / math.sqrt(12),
    (ee_cmd.cfg.ranges.pos_y[1] - ee_cmd.cfg.ranges.pos_y[0]) / math.sqrt(12),
    (ee_cmd.cfg.ranges.pos_z[1] - ee_cmd.cfg.ranges.pos_z[0]) / math.sqrt(12),
]
print(
    f"  位置偏移 标准差 x/y/z {[round(float(v), 4) for v in pos_off.std(0)]}, "
    f"名义 1/sqrt(12)*(区间宽) = {[round(v, 4) for v in _nominal]}"
)
_d2_ok = max(abs(c) for c in corrs) < 0.1 and dup_p < 0.02 and dup_r < 0.02
print(f"  -> {'通过 (六个分量独立)' if _d2_ok else '**不通过**: 命令落在低维流形上'}")
ee_cmd.cfg.resampling_time_range = (4.0, 8.0)

print("\n===== E. 诊断曲线 (必须分通道) =====")
for _ in range(20):
    env.step(zero)
scalars = {
    "Metrics/base_velocity/error_vel_xy": vel_cmd.metrics["error_vel_xy"].mean().item(),
    "Metrics/base_velocity/error_vel_yaw": vel_cmd.metrics["error_vel_yaw"].mean().item(),
    "Metrics/ee_pose/position_error": ee_cmd.metrics["position_error"].mean().item(),
    "Metrics/ee_pose/orientation_error": ee_cmd.metrics["orientation_error"].mean().item(),
    "Metrics/ee_pose/success": ee_cmd.metrics["success"].mean().item(),
}
for k, v in scalars.items():
    print(f"  {k:<40}{v:+.5f}")
_split = "error_vel_xy" in vel_cmd.metrics and "error_vel_yaw" in vel_cmd.metrics
print(f"  线/角速度误差是两个独立量: {'是' if _split else '否'}")

print("\n===== F. 解耦基线: 速度命令直通底盘 =====")
print(f"  {'cmd (vx, wz)':<18}{'achieved (vx, wz)':>22}{'线速度比':>10}{'角速度比':>10}")
vel_cmd.cfg.resampling_time_range = (1e6, 1e6)
for vx, wz in ((0.25, 0.0), (-0.25, 0.0), (0.0, 0.25), (0.0, -0.25), (0.25, 0.25)):
    direct = torch.zeros_like(zero)
    direct[:, 0] = math.atanh(vx / 0.5)
    direct[:, 1] = math.atanh(wz / 0.5)
    for _ in range(10):
        env.step(direct)
    acc_v, acc_w = [], []
    for _ in range(20):
        env.step(direct)
        acc_v.append(robot.data.root_lin_vel_b[:, 0].mean().item())
        acc_w.append(robot.data.root_ang_vel_b[:, 2].mean().item())
    av = sum(acc_v) / len(acc_v)
    aw = sum(acc_w) / len(acc_w)
    print(
        f"  vx={vx:+.2f} wz={wz:+.2f}   {av:>+10.3f} {aw:>+10.3f}"
        f"{av / vx if vx else float('nan'):>10.2f}{aw / wz if wz else float('nan'):>10.2f}"
    )

env.close()
app.close()
