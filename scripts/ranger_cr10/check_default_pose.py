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

"""Stage 4.10: is the shipped default pose the one the command centre assumes?

The task now starts with the tool pointing along ``base_link`` +X at 0.88 m above the ground.
Three constants describe that pose and they can drift apart: the articulation's ``init_state``,
the command's ``TCP_DEFAULT_POS_B`` / ``TCP_DEFAULT_QUAT_B``, and (on the MuJoCo side) the
keyframe and ``ARM_DEFAULT``. If they disagree the policy is asked to hold a target it is not
standing at, and the first thing it does is a large joint move -- which looks like a policy
failure and is not one.

This measures, in the simulator, what the shipped pose actually is:

  * TCP height above the ground plane;
  * the angle between the tool's approach axis (TCP +z) and the chassis' +X;
  * the joint slack to the limits;
  * the distance between the settled TCP and the command centre;
  * stability over 5 s with a zero command, and whether the chassis creeps.

Usage:
    python scripts/ranger_cr10/check_default_pose.py
    python scripts/ranger_cr10/check_default_pose.py --seconds 5
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=5.0)
args = ap.parse_args()

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
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.tcp import TCP_ANCHOR_BODY, tcp_pose_w  # noqa: E402

from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul  # noqa: E402


def main() -> None:
    cfg = RangerCr10TeleopEnvCfg()
    cfg.scene.num_envs = 8
    cfg.commands.ee_pose.debug_vis = False
    cfg.commands.ee_pose.resampling_time_range = (1e6, 1e6)
    # deterministic: no joint or gain randomisation, so "the default pose" is one number
    cfg.events.reset_robot_joints = None
    cfg.events.randomize_actuator_gains = None
    env = gym.make("RANGER-CR10-TELEOP-WBC", cfg=cfg)
    u = env.unwrapped
    robot = u.scene["robot"]
    env.reset()
    zero = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
    for _ in range(50):
        env.step(zero)

    base_idx = robot.find_bodies("base_link")[0][0]
    anchor_idx = robot.find_bodies(TCP_ANCHOR_BODY)[0][0]
    tcp_pos, tcp_quat = tcp_pose_w(robot, anchor_idx)

    base_pos = robot.data.body_pos_w[:, base_idx]
    base_quat = robot.data.body_quat_w[:, base_idx]
    ground_z = 0.0
    height = (tcp_pos[:, 2] - ground_z).mean().item()

    # tool approach axis = TCP +z, expressed in the base frame
    z_axis_w = quat_apply(tcp_quat, torch.tensor([0.0, 0.0, 1.0], device=u.device).expand(u.num_envs, 3))
    z_axis_b = quat_apply(quat_conjugate(base_quat), z_axis_w)
    angle = torch.acos(z_axis_b[:, 0].clamp(-1.0, 1.0)).mean().item()

    default_pos = torch.tensor(TCP_DEFAULT_POS_B, device=u.device).expand(u.num_envs, 3)
    default_quat = torch.tensor(TCP_DEFAULT_QUAT_B, device=u.device).expand(u.num_envs, 4)
    tcp_vec_b = quat_apply(quat_conjugate(base_quat), tcp_pos - base_pos)
    misalign_pos = torch.linalg.norm(tcp_vec_b - default_pos, dim=-1).mean().item()
    dot = torch.abs(torch.sum(quat_mul(quat_conjugate(base_quat), tcp_quat) * default_quat, dim=-1)).clamp(max=1.0)
    misalign_rot = math.degrees(2.0 * torch.acos(dot).mean().item())

    joint_idx = robot.joint_names.index(ARM_JOINT_NAMES[0])
    q = robot.data.joint_pos[:, joint_idx : joint_idx + 6]
    lim = robot.data.joint_pos_limits[:, joint_idx : joint_idx + 6]
    slack = torch.minimum(q - lim[..., 0], lim[..., 1] - q).min().item()

    start_base, start_tcp = base_pos.clone(), tcp_pos.clone()
    steps = int(args.seconds / (u.step_dt))
    drift = 0.0
    for _ in range(steps):
        env.step(zero)
    drift = torch.linalg.norm(robot.data.body_pos_w[:, base_idx] - start_base, dim=-1).max().item()
    tcp_moved = torch.linalg.norm(tcp_pose_w(robot, anchor_idx)[0] - start_tcp, dim=-1).max().item()

    # The D455 is a fixed child of Link6 and Isaac's USD merges fixed joints, so its pose is
    # reconstructed from Link6 with the URDF's own mount transform (stage 4.10).
    link6_idx = robot.find_bodies("cr10_Link6")[0][0]
    mount_p = torch.tensor([0.0, 0.10, 0.02], device=u.device)
    mount_q = torch.tensor([0.5, -0.5, -0.5, -0.5], device=u.device)  # rpy(45,-90,-135), (w,x,y,z)
    cam_pos = robot.data.body_pos_w[:, link6_idx] + quat_apply(
        robot.data.body_quat_w[:, link6_idx], mount_p.expand(u.num_envs, 3)
    )
    cam_quat = quat_mul(robot.data.body_quat_w[:, link6_idx], mount_q.expand(u.num_envs, 4))
    # depth-optical frame: rpy(-90, 0, -90) from the d455 link
    opt_q = torch.tensor([0.5, -0.5, 0.5, -0.5], device=u.device)
    opt_quat = quat_mul(cam_quat, opt_q.expand(u.num_envs, 4))
    opt_z_w = quat_apply(opt_quat, torch.tensor([0.0, 0.0, 1.0], device=u.device).expand(u.num_envs, 3))
    opt_z_b = quat_apply(quat_conjugate(base_quat), opt_z_w)
    opt_y_w = quat_apply(opt_quat, torch.tensor([0.0, 1.0, 0.0], device=u.device).expand(u.num_envs, 3))
    cam_height = cam_pos[:, 2].mean().item()
    axis_err = torch.acos(opt_z_b[:, 0].clamp(-1.0, 1.0)).mean().item()
    up_err = torch.acos((-opt_y_w[:, 2]).clamp(-1.0, 1.0)).mean().item()

    print("\n===== 默认姿态检查 (Isaac Lab) =====")
    print(f"  D455 离地高度          {cam_height:.4f} m   (目标 0.88)")
    print(f"  D455 光轴 与 base+X     {math.degrees(axis_err):.2f}°   (目标 0)")
    print(f"  图像上方向 与 世界+Z    {math.degrees(up_err):.2f}°   (目标 0 = 不歪)")
    print(f"  TCP 离地高度           {height:.4f} m")
    print(f"  工具 +z 与 base+X 夹角  {math.degrees(angle):.2f}°   (目标 0)")
    print(f"  关节余量 (最小)         {slack:.3f} rad")
    print(f"  TCP 与命令中心偏差      {misalign_pos * 1000:.1f} mm / {misalign_rot:.2f}°")
    print(f"  {args.seconds:.0f}s 零命令: TCP 移动 {tcp_moved * 1000:.1f} mm, 底盘漂移 {drift * 1000:.1f} mm")
    env.close()


if __name__ == "__main__":
    main()
