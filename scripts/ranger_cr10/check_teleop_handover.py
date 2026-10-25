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

"""Stage 4.5g, task A: what does the actor command at the moment it is handed a correct pose?

4.5f established *that* the actor walks away from a converged state (0.4 deg -> 10-12 deg
in half a second). This script asks *what it commands*: the online IK's joint solution, the
actor's action, the absolute joint target each implies, and the task-space pose each of those
targets would produce. The action is an absolute joint target, so the judgement is FK(q_des)
against the commanded pose -- not "how sensitive is the output to the observed joints", which
is a statement about one particular parameterisation. Several joint configurations reach the
same pose, and any of them is a correct action.

Usage:
    python scripts/ranger_cr10/check_teleop_handover.py --policy <checkpoint or exported .pt>
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teleop_kinematics as tk  # noqa: E402

parser = argparse.ArgumentParser(description="Stage 4.5g handover diagnosis.")
parser.add_argument("--policy", default="logs/rsl_rl/ranger_cr10_teleop/_stage45e/pretrained_B.pt")
parser.add_argument("--warm", type=int, default=25, help="steps of online IK before the handover")
parser.add_argument("--watch", type=int, default=25, help="steps after the handover")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--out", default="outputs/stage45g")
parser.add_argument("--random_commands", type=int, default=12)
parser.add_argument("--random_handover", type=int, default=10)
parser.add_argument("--residual", action="store_true",
                    help="the policy predicts a correction: a = a_now + net(obs)[:, 2:]")
args_cli = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
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

DEFAULT = np.asarray(tk.DEFAULT_Q)
SCALE = 0.25
OUT = Path(args_cli.out)
OUT.mkdir(parents=True, exist_ok=True)


def load_net(path: str, device):
    """Either an rsl_rl checkpoint or a plain state dict; returns a deterministic actor."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj["actor_state_dict"] if isinstance(obj, dict) and "actor_state_dict" in obj else obj
    sd = {(k[4:] if k.startswith("mlp.") else k): v for k, v in sd.items() if k.split(".")[-1] in ("weight", "bias")}
    net = torch.nn.Sequential(
        torch.nn.Linear(sd["0.weight"].shape[1], 512), torch.nn.ELU(),
        torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 8),
    ).to(device)
    net.load_state_dict(sd)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def poses(cmd_pos, cmd_quat):
    return np.asarray(cmd_pos, dtype=float), tk.quat_to_R(np.asarray(cmd_quat, dtype=float))


def task_error(q, cmd_pos, cmd_R):
    T = tk.fk(np.asarray(q, dtype=float))
    dp = float(np.linalg.norm(T[:3, 3] - cmd_pos))
    da = math.degrees(np.linalg.norm(tk.log_R(cmd_R @ T[:3, :3].T)))
    return dp, da


def main():
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
    ee_cmd = u.command_manager.get_term("ee_pose")
    vel_cmd = u.command_manager.get_term("base_velocity")
    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    dev = u.device
    base_net = load_net(args_cli.policy, dev)

    class _Pol:
        """Uniform callable: absolute actor, or residual actor that adds the current pose."""

        def __call__(self, obs_t):
            with torch.inference_mode():
                out = base_net(obs_t)
            if not args_cli.residual:
                return out
            a = out.clone()
            a[:, 2:] = a[:, 2:] + obs_t[:, 18:24] / 0.25   # radians -> action units
            return a

    net = _Pol()
    print(f"policy {args_cli.policy}")

    def set_command(rpy):
        dR = iu.quat_from_euler_xyz(*[torch.tensor([v], device=dev) for v in rpy])
        tq = iu.quat_mul(dR, torch.tensor(TCP_DEFAULT_QUAT_B, device=dev).unsqueeze(0))[0]
        ee_cmd.pose_command_b[:, :3] = torch.tensor(np.asarray(TCP_DEFAULT_POS_B), device=dev)
        ee_cmd.pose_command_b[:, 3:] = tq
        vel_cmd.is_standing_env[:] = False
        vel_cmd.vel_command_b[:] = 0.0

    def ik_action():
        q_now = robot.data.joint_pos[0, arm_ids].cpu().numpy()
        q_sol, ep, er, _, _ = tk.local_ik(q_now, ee_cmd.pose_command_b[0, :3].cpu().numpy(),
                                          tk.quat_to_R(ee_cmd.pose_command_b[0, 3:].cpu().numpy()))
        a = torch.zeros(1, u.action_manager.total_action_dim, device=dev)
        a[:, 2:] = torch.tensor((q_sol - DEFAULT) / SCALE, dtype=torch.float32, device=dev)
        return a, q_sol, (ep, er)

    rows = []
    print(f"\n{'命令':<18}{'交接误差':>9}{'actor q_des 姿态':>16}{'IK q_des 姿态':>14}"
          f"{'|q_des-default|':>16}{'|q_ik-default|':>15}{'稳定误差':>10}")
    for rpy in ((0.2, 0.0, 0.0), (-0.2, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, -0.2, 0.0),
                (0.0, 0.0, 0.2), (0.0, 0.0, -0.2), (0.15, 0.1, -0.1)):
        obs, _ = env.reset(seed=args_cli.seed)
        set_command(rpy)
        for _ in range(args_cli.warm):
            a, q_ik, _ = ik_action()
            obs, _, _, _, _ = env.step(a)
        cmd_pos, cmd_R = poses(ee_cmd.pose_command_b[0, :3].cpu(), ee_cmd.pose_command_b[0, 3:].cpu())
        q_hand = robot.data.joint_pos[0, arm_ids].cpu().numpy().copy()
        _, err_hand = task_error(q_hand, cmd_pos, cmd_R)
        # the reference: what the online IK would command from exactly this state
        q_ik, ep, er, _, _ = tk.local_ik(q_hand, cmd_pos, cmd_R)
        _, ik_des_err = task_error(q_ik, cmd_pos, cmd_R)

        # first actor step: the absolute joint target it asks for, judged in task space
        obs_hand = obs["policy"][0].cpu().numpy().astype(float).copy()   # the handover observation
        with torch.inference_mode():
            a_actor = net(obs["policy"])
        raw = a_actor[0, 2:].cpu().numpy()
        q_des = DEFAULT + SCALE * raw
        _, actor_des_err = task_error(q_des, cmd_pos, cmd_R)

        # the same observation, but with the three history frames all equal to the present:
        # if this changes the action, the failure is a history/timing artefact rather than
        # the state itself
        obs_fresh, _ = None, None
        for _ in range(3):
            obs_fresh = u.observation_manager.compute(update_history=True)
        with torch.inference_mode():
            a_fresh = net(obs_fresh["policy"])
        q_fresh = DEFAULT + SCALE * a_fresh[0, 2:].cpu().numpy()
        _, fresh_des_err = task_error(q_fresh, cmd_pos, cmd_R)

        # observation ablations on the same state: which term, if any, carries the wrong row
        o = obs["policy"][0].cpu().numpy().copy()
        abl = {}
        for tag, sl in (("joint_vel", slice(36, 54)), ("actions", slice(54, 78)),
                        ("base_vel", slice(0, 9)), ("base_lin_vel", slice(108, 117))):
            o2 = o.copy()
            o2[sl] = 0.0
            with torch.inference_mode():
                a2 = net(torch.tensor(o2, dtype=torch.float32, device=dev).unsqueeze(0))
            abl[tag] = task_error(DEFAULT + SCALE * a2[0, 2:].cpu().numpy(), cmd_pos, cmd_R)[1]

        trace = []
        for _ in range(args_cli.watch):
            with torch.inference_mode():
                a_actor = net(obs["policy"])
            obs, _, _, _, _ = env.step(a_actor)
            q = robot.data.joint_pos[0, arm_ids].cpu().numpy()
            trace.append(task_error(q, cmd_pos, cmd_R)[1])
        print(f"{str(rpy):<18}{err_hand:>8.1f}°{actor_des_err:>15.1f}°{ik_des_err:>13.1f}°"
              f"{np.abs(q_des - DEFAULT).max():>16.3f}{np.abs(q_ik - DEFAULT).max():>15.3f}"
              f"{np.mean(trace[-5:]):>9.1f}°")
        # what command would the actor's target pose correspond to? A constant offset across
        # +-roll/pitch/yaw would point at a frame or joint offset rather than at a bad map
        T_des = tk.fk(q_des)
        dR = cmd_R @ T_des[:3, :3].T
        delta_rpy = np.degrees(tk.R_to_rpy(dR))
        dp = T_des[:3, 3] - cmd_pos
        # and the command the actor was actually shown, checked against the one we set
        o_cmd = obs["policy"][0].cpu().numpy()[87:108][-7:]
        cmd_mismatch = float(np.abs(o_cmd - np.concatenate([cmd_pos, ee_cmd.pose_command_b[0, 3:].cpu().numpy()])).max())
        rows.append(dict(rpy=rpy, hand_err=err_hand, actor_des=actor_des_err, ik_des=ik_des_err,
                         delta_rpy=delta_rpy.tolist(), delta_pos=dp.tolist(), cmd_mismatch=cmd_mismatch,
                         fresh_des=fresh_des_err, abl=abl,
                         obs=obs_hand,
                         q_des_dev=float(np.abs(q_des - DEFAULT).max()),
                         q_ik_dev=float(np.abs(q_ik - DEFAULT).max()),
                         settled=float(np.mean(trace[-5:])), trace=trace,
                         q_des=q_des, q_ik=q_ik, q_hand=q_hand))

    print("\n第一步之后每 0.1 s 的实际姿态误差 (roll +0.2 / combo):")
    for r in rows:
        if r["rpy"] in ((0.2, 0.0, 0.0), (0.15, 0.1, -0.1)):
            print(f"  {str(r['rpy']):<16}" + " ".join(f"{v:5.1f}" for v in r["trace"][:10]))

    print("\nActor 的 q_des 对应到'哪个命令' (相对真实命令的差, 度/毫米):")
    for r in rows:
        print(f"  {str(r['rpy']):<18}dRPY ({r['delta_rpy'][0]:+6.1f}, {r['delta_rpy'][1]:+6.1f}, "
              f"{r['delta_rpy'][2]:+6.1f})°   dP ({r['delta_pos'][0] * 1000:+5.0f}, "
              f"{r['delta_pos'][1] * 1000:+5.0f}, {r['delta_pos'][2] * 1000:+5.0f})mm   "
              f"obs 命令与设定命令最大差 {r['cmd_mismatch']:.1e}")

    print("\n历史刷新后 (三帧都=当前) 与逐项置零后的 q_des 任务空间误差:")
    print(f"  {'命令':<18}{'原样':>7}{'刷新历史':>9}{'joint_vel=0':>12}{'actions=0':>11}"
          f"{'base_vel=0':>11}{'base_lin=0':>11}")
    for r in rows:
        print(f"  {str(r['rpy']):<18}{r['actor_des']:>6.1f}°{r['fresh_des']:>8.1f}°"
              f"{r['abl']['joint_vel']:>11.1f}°{r['abl']['actions']:>10.1f}°"
              f"{r['abl']['base_vel']:>10.1f}°{r['abl']['base_lin_vel']:>10.1f}°")

    # ---- the same handover, but with commands drawn from the training distribution ----
    # The seven above are pure-axis with a zero position offset; the original supervision set
    # contains 126 samples with |offset| < 10 mm out of 3142, so "zero offset" is a corner.
    print(f"\n随机命令的交接测试 (n={args_cli.random_handover}, 训练分布采样):")
    rng2 = np.random.default_rng(1)
    hand, settled = [], []
    for _ in range(args_cli.random_handover):
        rpy = tuple(rng2.uniform(-0.4, 0.4, 3))
        pos = (rng2.uniform(-0.12, 0.12), rng2.uniform(-0.15, 0.15), rng2.uniform(-0.12, 0.10))
        obs, _ = env.reset(seed=args_cli.seed)
        dR = iu.quat_from_euler_xyz(*[torch.tensor([v], device=dev) for v in rpy])
        tq = iu.quat_mul(dR, torch.tensor(TCP_DEFAULT_QUAT_B, device=dev).unsqueeze(0))[0]
        ee_cmd.pose_command_b[:, :3] = torch.tensor(np.asarray(TCP_DEFAULT_POS_B) + np.asarray(pos), device=dev)
        ee_cmd.pose_command_b[:, 3:] = tq
        for _ in range(args_cli.warm):
            a, q_ik, _ = ik_action()
            obs, _, _, _, _ = env.step(a)
        cmd_pos, cmd_R = poses(ee_cmd.pose_command_b[0, :3].cpu(), ee_cmd.pose_command_b[0, 3:].cpu())
        hand.append(task_error(robot.data.joint_pos[0, arm_ids].cpu().numpy(), cmd_pos, cmd_R)[1])
        for _ in range(args_cli.watch):
            with torch.inference_mode():
                a_actor = net(obs["policy"])
            obs, _, _, _, _ = env.step(a_actor)
        settled.append(task_error(robot.data.joint_pos[0, arm_ids].cpu().numpy(), cmd_pos, cmd_R)[1])
    print(f"  交接时 {np.median(hand):.1f}°, 2.5 s 后 中位 {np.median(settled):.1f}° "
          f"(范围 {np.min(settled):.1f}–{np.max(settled):.1f}°)")

    print("\n关节层面 (roll +0.2 那一组):")
    r = rows[0]
    np.set_printoptions(precision=3, suppress=True)
    print(f"  交接时 q        {np.round(r['q_hand'], 3)}")
    print(f"  在线 IK q_ik    {np.round(r['q_ik'], 3)}   与 q 的差 {np.abs(r['q_ik'] - r['q_hand']).max():.3f} rad")
    print(f"  Actor q_des     {np.round(r['q_des'], 3)}   与 q_ik 的差 {np.abs(r['q_des'] - r['q_ik']).max():.3f} rad")
    print(f"  Actor q_des    与 default 的差 {np.abs(r['q_des'] - DEFAULT).max():.3f} rad")
    print(f"  在线 IK q_ik 与 default 的差 {np.abs(r['q_ik'] - DEFAULT).max():.3f} rad")

    # ---- generic commands, sampled the way the command term samples them ---------------
    # The seven commands above are pure-axis, zero position offset: a measure-zero corner of
    # the training box. If the actor tracks *these* worse than generic ones, the finding is
    # about coverage of my test set, not about the map.
    print("\n随机命令 (按训练分布采样, 每个 25 步):")
    rng = np.random.default_rng(0)
    errs = {0: [], 1: []}
    for _ in range(args_cli.random_commands):
        rpy = tuple(rng.uniform(-0.4, 0.4, 3))
        pos = (rng.uniform(-0.12, 0.12), rng.uniform(-0.15, 0.15), rng.uniform(-0.12, 0.10))
        obs, _ = env.reset(seed=args_cli.seed)
        dR = iu.quat_from_euler_xyz(*[torch.tensor([v], device=dev) for v in rpy])
        tq = iu.quat_mul(dR, torch.tensor(TCP_DEFAULT_QUAT_B, device=dev).unsqueeze(0))[0]
        ee_cmd.pose_command_b[:, :3] = torch.tensor(np.asarray(TCP_DEFAULT_POS_B) + np.asarray(pos), device=dev)
        ee_cmd.pose_command_b[:, 3:] = tq
        cmd_pos, cmd_R = poses(ee_cmd.pose_command_b[0, :3].cpu(), ee_cmd.pose_command_b[0, 3:].cpu())
        for _ in range(25):
            with torch.inference_mode():
                a_actor = net(obs["policy"])
            obs, _, _, _, _ = env.step(a_actor)
        q = robot.data.joint_pos[0, arm_ids].cpu().numpy()
        dp, da = task_error(q, cmd_pos, cmd_R)
        errs[0].append(da)
        errs[1].append(dp)
    print(f"  Actor 稳态姿态误差 中位 {np.median(errs[0]):.1f}°, "
          f"位置误差 中位 {np.median(errs[1]):.3f} m  (n={args_cli.random_commands})")

    import json
    with open(OUT / "handover.json", "w") as f:
        json.dump([{k: (list(v) if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in rows], f,
                  indent=1, default=float)
    print(f"\n已写出 {OUT / 'handover.json'}")


if __name__ == "__main__":
    main()
