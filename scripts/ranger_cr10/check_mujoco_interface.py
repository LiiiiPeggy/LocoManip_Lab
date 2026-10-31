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

"""Stage 4.6: does the MuJoCo deployment see and act on the same thing Isaac Lab does?

The robot moving in MuJoCo proves nothing. Four things are checked here, each against the
*live* Isaac Lab env rather than against a re-derivation:

1. **Observation layout.** The env's own 117-D policy vector is compared against the same
   eight terms assembled independently from `robot.data`, three frames, in declaration
   order. If the block order, the per-term scales or the history direction were wrong, this
   is what shows it.
2. **Observation sampling in MuJoCo.** The Isaac state (root pose/velocity, all 14 joint
   positions and velocities, both commands, last action) is written into the MuJoCo model
   and the same 117-D vector is assembled from *its* sensors. Same numbers out means the
   conventions -- base-frame velocity, projected gravity, joint offsets -- agree.
3. **Forward kinematics.** The TCP pose relative to `base_link`, from Isaac's USD and from
   the generated MJCF, at the same joint configuration. This is the check that the MuJoCo
   joint axes, signs and link offsets are the ones the policy was trained against.
4. **Action mapping.** 256 random raw 8-D actions through the env's own `BaseVelocityAction`
   and through the deployment's `ChassisAction`: steering targets, wheel-speed targets and
   the post-`tanh` `(vx, wz)`.

Usage:
    python scripts/ranger_cr10/check_mujoco_interface.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mujoco" / "deploy"))

from deploy_mujoco.ranger_cr10.ranger_policy import (  # noqa: E402
    TCP_DEFAULT_POS_B,
    TCP_DEFAULT_QUAT_B,
    ChassisAction,
    EndEffector,
    RangerPolicyObs,
    act,
    load_actor,
    orientation_error,
)

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import LeggedManip_Lab.tasks  # noqa: E402,F401
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.teleop_env_cfg import (  # noqa: E402
    RangerCr10TeleopEnvCfg,
)

from isaaclab.utils.math import quat_from_euler_xyz, quat_mul  # noqa: E402

OUT = ROOT / "outputs" / "mujoco_interface"
OUT.mkdir(parents=True, exist_ok=True)

cfg = RangerCr10TeleopEnvCfg()
cfg.scene.num_envs = 1
cfg.episode_length_s = 1e6
cfg.commands.ee_pose.debug_vis = False
cfg.commands.ee_pose.resampling_time_range = (1e6, 1e6)
cfg.commands.base_velocity.resampling_time_range = (1e6, 1e6)
cfg.observations.policy.enable_corruption = False
cfg.curriculum.lin_vel_cmd_levels = None
cfg.curriculum.ang_vel_cmd_levels = None
env = gym.make("RANGER-CR10-TELEOP-WBC", cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
ee_cmd_term = u.command_manager.get_term("ee_pose")
vel_cmd_term = u.command_manager.get_term("base_velocity")

# -- fixed commands, composed exactly as the env composes them -------------------------
VEL_CMD = (0.17, -0.11)
EE_OFFSET = ((0.03, -0.05, 0.02), (0.10, -0.15, 0.20))


def set_commands() -> None:
    vx, wz = VEL_CMD
    vel_cmd_term.is_standing_env[:] = False
    vel_cmd_term.vel_command_b[:, 0] = vx
    vel_cmd_term.vel_command_b[:, 1] = 0.0
    vel_cmd_term.vel_command_b[:, 2] = wz
    dpos, drpy = EE_OFFSET
    # pre-multiplied, exactly as UniformTCPBodyPoseCommand._resample_command does it
    dR = quat_from_euler_xyz(
        torch.tensor([drpy[0]], device=u.device),
        torch.tensor([drpy[1]], device=u.device),
        torch.tensor([drpy[2]], device=u.device),
    )
    qd = torch.tensor(TCP_DEFAULT_QUAT_B, device=u.device).unsqueeze(0)
    ee_cmd_term.pose_command_b[:, :3] = torch.tensor(np.asarray(TCP_DEFAULT_POS_B) + np.asarray(dpos), device=u.device)
    ee_cmd_term.pose_command_b[:, 3:] = quat_mul(dR, qd)


# -- drive the env ---------------------------------------------------------------------
env.reset(seed=7)
set_commands()
RAW_ACTION = np.array([0.20, -0.15, 0.10, -0.08, 0.12, 0.05, -0.10, 0.07])

FRAME_ORDER = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "velocity_commands",
    "pos_commands",
    "base_lin_vel",
)
SCALES = dict(
    base_ang_vel=0.2,
    projected_gravity=1.0,
    joint_pos=1.0,
    joint_vel=0.05,
    actions=1.0,
    velocity_commands=1.0,
    pos_commands=1.0,
    base_lin_vel=2.0,
)


# arm-only indices inside joint_pos / joint_vel (the obs group names cr10_joint1..6)
arm_ids, arm_names = robot.find_joints([f"cr10_joint{i}" for i in range(1, 7)], preserve_order=True)
all_names = robot.joint_names
arm_idx = [all_names.index(n) for n in arm_names]


def isaac_frame() -> dict:
    """The eight terms, computed from the live env state with the documented formulas."""
    d = robot.data
    return dict(
        base_ang_vel=d.root_ang_vel_b[0].cpu().numpy().astype(np.float64),
        projected_gravity=d.projected_gravity_b[0].cpu().numpy().astype(np.float64),
        joint_pos=(d.joint_pos[0] - d.default_joint_pos[0]).cpu().numpy()[arm_idx].astype(np.float64),
        joint_vel=(d.joint_vel[0] - d.default_joint_vel[0]).cpu().numpy()[arm_idx].astype(np.float64),
        actions=u.action_manager.action[0].cpu().numpy().astype(np.float64),
        velocity_commands=u.command_manager.get_command("base_velocity")[0].cpu().numpy().astype(np.float64),
        pos_commands=u.command_manager.get_command("ee_pose")[0].cpu().numpy().astype(np.float64),
        base_lin_vel=d.root_lin_vel_b[0].cpu().numpy().astype(np.float64),
    )


frames = []
obs = None
for _ in range(3):
    obs, _, _, _, _ = env.step(torch.tensor(RAW_ACTION, dtype=torch.float32, device=u.device).unsqueeze(0))
    set_commands()
    frames.append(isaac_frame())

isaac_obs = obs["policy"][0].cpu().numpy().astype(np.float64)
assembled = np.concatenate([np.concatenate([frames[k][name] for k in range(3)]) * SCALES[name] for name in FRAME_ORDER])

print("=" * 100)
print("1. 观测拼接 (Isaac 实际 117D vs 从 robot.data 按文档公式重组)")
print("=" * 100)
print(f"  Isaac joint_names 顺序: {all_names}")
print(f"  机械臂取用顺序        : {arm_names}")
blocks = [
    "base_ang_vel 9",
    "proj_gravity 9",
    "joint_pos 18",
    "joint_vel 18",
    "actions 24",
    "vel_cmd 9",
    "pos_cmd 21",
    "base_lin_vel 9",
]
start = 0
layout_ok = True
for label, width in zip(blocks, (9, 9, 18, 18, 24, 9, 21, 9)):
    a, b = isaac_obs[start : start + width], assembled[start : start + width]
    err = float(np.max(np.abs(a - b)))
    layout_ok &= err < 1e-5
    print(f"  [{start:3d}:{start + width:3d}] {label:<16} max|diff| {err:.2e}")
    start += width
print(f"  结论: {'一致' if layout_ok else '**不一致**'}")

# -- part 2: the same state through the MuJoCo model -----------------------------------
import mujoco

SCENE = ROOT / "mujoco" / "robots" / "ranger_cr10" / "scene.xml"
m = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(m)

d = robot.data
root_pos = d.root_pos_w[0].cpu().numpy()
root_quat = d.root_quat_w[0].cpu().numpy()
root_lin_w = d.root_lin_vel_w[0].cpu().numpy()
root_ang_w = d.root_ang_vel_w[0].cpu().numpy()
joint_pos = d.joint_pos[0].cpu().numpy()
joint_vel = d.joint_vel[0].cpu().numpy()

data.qpos[0:3] = root_pos
data.qpos[3:7] = root_quat
for i, name in enumerate(all_names):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:  # the gripper linkage is welded in the MuJoCo model
        continue
    data.qpos[m.jnt_qposadr[jid]] = joint_pos[i]
    data.qvel[m.jnt_dofadr[jid]] = joint_vel[i]
data.qvel[0:3] = root_lin_w
data.qvel[3:6] = root_ang_w
mujoco.mj_forward(m, data)

obs_builder = RangerPolicyObs(m, data)
mujoco_frame = obs_builder.frame(frames[-1]["actions"], (VEL_CMD[0], VEL_CMD[1]), frames[-1]["pos_commands"])

print()
print("=" * 100)
print("2. 同一状态在 MuJoCo 中的观测采样 (逐项 vs Isaac)")
print("=" * 100)
sampling_ok = True
for name in FRAME_ORDER:
    a = frames[-1][name]
    b = np.asarray(mujoco_frame[name])
    if name in ("joint_pos", "joint_vel"):
        n = "机械臂 " + ("位置(rel)" if name == "joint_pos" else "速度")
    else:
        n = name
    err = float(np.max(np.abs(a - b)))
    ok = err < 2e-4
    sampling_ok &= ok
    print(f"  {n:<28} dim {len(b):2d}  max|diff| {err:.2e}  Isaac {np.round(a, 4)}")
    if err >= 2e-4:
        print(f"    MuJoCo {np.round(b, 4)}")
print(f"  结论: {'一致' if sampling_ok else '**不一致**'}")

# -- part 3: forward kinematics ---------------------------------------------------------
ee = EndEffector(m, data)
tcp_pos, tcp_quat = ee.tcp_pose_b()
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.tcp import tcp_pose_b as isaac_tcp  # noqa: E402

from isaaclab.utils.math import quat_unique  # noqa: E402

anchor_ids, _ = robot.find_bodies("cr10_Link6")
base_ids, _ = robot.find_bodies("base_link")
ip, iq = isaac_tcp(robot, anchor_ids[0], base_ids[0])
ip = ip[0].cpu().numpy()
iq = quat_unique(iq)[0].cpu().numpy()
if np.dot(iq, tcp_quat) < 0:
    tcp_quat = -tcp_quat
print()
print("=" * 100)
print("3. 末端 TCP 的正运动学 (同一个关节构型)")
print("=" * 100)
print(f"  Isaac  pos_b {np.round(ip, 6)}  quat_b {np.round(iq, 6)}")
print(f"  MuJoCo pos_b {np.round(tcp_pos, 6)}  quat_b {np.round(tcp_quat, 6)}")
print(
    f"  位置差 {np.linalg.norm(ip - tcp_pos) * 1000:.3f} mm, 姿态差 "
    f"{np.degrees(orientation_error(tcp_quat, iq)):.4f}°  (手臂关节补角/网格差异应小于 1 mm)"
)

# -- part 4: action mapping -------------------------------------------------------------
term = u.action_manager.get_term("base_vel")
rng = np.random.default_rng(3)
raws = rng.uniform(-4.0, 4.0, size=(256, 2))
chassis = ChassisAction()
mismatch = []
spinning_n = 0
for i in range(len(raws)):
    x = torch.tensor(raws[i], dtype=torch.float32, device=u.device).unsqueeze(0)
    term.process_actions(x)
    term.apply_actions()
    ref_s = term._steering_target[0].cpu().numpy()
    ref_w = term._wheel_target[0].cpu().numpy()
    ref_c = term._command[0].cpu().numpy()
    mine = chassis.apply(raws[i])
    # `_command` is written before the spinning branch rewrites (vx, wz), so the comparable
    # quantity is the post-tanh value, not what `apply` returns as the executed command
    post_tanh = ChassisAction.command_from_raw(raws[i])
    mismatch.append(
        (
            float(np.max(np.abs(ref_s - mine["steering_target"]))),
            float(np.max(np.abs(ref_w - mine["wheel_speed_target"]))),
            float(np.max(np.abs(ref_c - post_tanh))),
        )
    )
    if mine["spinning"]:
        spinning_n += 1
        assert mine["vx"] == 0.0
mismatch = np.array(mismatch)
print()
print("=" * 100)
print("4. 8D 动作 -> 底盘执行量 (256 组随机 raw action, Isaac BaseVelocityAction vs ChassisAction)")
print("=" * 100)
for k, label in enumerate(("转向角目标", "轮速目标", "(vx, wz)")):
    print(f"  {label:<12} max|diff| {mismatch[:, k].max():.3e}  均值 {mismatch[:, k].mean():.3e}")
print(f"  结论: {'一致' if mismatch.max() < 1e-5 else '**不一致**'}")

# -- part 5: deterministic actor parity -------------------------------------------------
policy = ROOT / "logs/rsl_rl/ranger_cr10_teleop/2026-10-29_19-12-08_stage45j_ppo/model_5999.pt"
actor, kind = load_actor(policy)
a_muj = act(actor, isaac_obs)

ck = torch.load(policy, map_location="cpu", weights_only=False)
sd = {
    (k[4:] if k.startswith("mlp.") else k): v
    for k, v in ck["actor_state_dict"].items()
    if k.split(".")[-1] in ("weight", "bias")
}
net = torch.nn.Sequential(
    torch.nn.Linear(117, 512),
    torch.nn.ELU(),
    torch.nn.Linear(512, 256),
    torch.nn.ELU(),
    torch.nn.Linear(256, 128),
    torch.nn.ELU(),
    torch.nn.Linear(128, 8),
)
net.load_state_dict(sd)
net.eval()
with torch.inference_mode():
    a_isaac = net(torch.tensor(isaac_obs, dtype=torch.float32).unsqueeze(0))[0].numpy()
print()
print("=" * 100)
print("5. 确定性 Actor (同一 117D 观测)")
print("=" * 100)
print(f"  Isaac 侧 {np.round(a_isaac, 6)}")
print(f"  MuJoCo 侧({kind}) {np.round(a_muj, 6)}   max|diff| {np.abs(a_isaac - a_muj).max():.2e}")

summary = dict(
    layout_max_diff=float(np.max(np.abs(isaac_obs - assembled))),
    layout_ok=bool(layout_ok),
    sampling_max_diff=float(max(np.max(np.abs(np.asarray(mujoco_frame[n]) - frames[-1][n])) for n in FRAME_ORDER)),
    sampling_ok=bool(sampling_ok),
    tcp_pos_diff_mm=float(np.linalg.norm(ip - tcp_pos) * 1000),
    tcp_ori_diff_deg=float(np.degrees(orientation_error(tcp_quat, iq))),
    action_steer_max=float(mismatch[:, 0].max()),
    action_wheel_max=float(mismatch[:, 1].max()),
    action_cmd_max=float(mismatch[:, 2].max()),
    actor_max_diff=float(np.abs(a_isaac - a_muj).max()),
    isaac_obs=isaac_obs.tolist(),
    isaac_action=a_isaac.tolist(),
)
(OUT / "interface_check.json").write_text(json.dumps(summary, indent=1))
print(f"\n已写出 {OUT / 'interface_check.json'}")

env.close()
app.close()
