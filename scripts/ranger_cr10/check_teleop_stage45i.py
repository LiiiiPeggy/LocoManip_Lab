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

"""Stage 4.5i, task C: closed-loop acceptance for the mixed-supervision actors.

Two tests, both on commands the supervision never contained (`train_*.json` -> ``test_cmds``,
plus the single-axis sweeps that the keyboard produces):

C1 handover  -- online local IK drives the arm to the commanded pose to about 0.5 deg, then
                the actor takes over for 2.5 s. The quantity that matters is the *first step's*
                joint target (`FK(q_des)` against the command), which 4.5f showed predicts the
                settled error to within 1.4 deg, and then whether the error grows.

C2 active    -- the actor drives from wherever the arm is when the command is set, for 2.5 s.
                Single-axis roll/pitch/yaw, combined attitude, and a chassis-moving case so a
                fix for the arm cannot silently cost the chassis.

Every policy is evaluated on the same episodes and the same seeds. The online IK is run as a
reference on the same commands, because a command that no arm configuration reaches would
otherwise look like a policy failure.

Usage:
    python scripts/ranger_cr10/check_teleop_stage45i.py \
        --policies "B0=logs/.../pretrained_B.pt" "mix21=logs/.../actor_mix21_best.pt" ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teleop_kinematics as tk  # noqa: E402

parser = argparse.ArgumentParser(description="Stage 4.5i closed-loop acceptance.")
parser.add_argument("--policies", nargs="+", required=True, help="name=path pairs")
parser.add_argument(
    "--cmds_json",
    default="logs/rsl_rl/ranger_cr10_teleop/_stage45i/train_mix21.json",
    help="training record whose test_cmds are held out from supervision",
)
parser.add_argument("--n_test_cmds", type=int, default=8)
parser.add_argument("--warm", type=int, default=40, help="IK steps before the handover")
parser.add_argument("--watch", type=int, default=25, help="actor steps after the handover (10 Hz)")
parser.add_argument("--out", default="outputs/stage45i")
parser.add_argument("--seed", type=int, default=2026)
args_cli = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import LeggedManip_Lab.tasks  # noqa: E402,F401
from LeggedManip_Lab.assets.ranger_cr10.ranger_cr10_articulation_cfg import ARM_JOINT_NAMES  # noqa: E402
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.teleop_env_cfg import (  # noqa: E402
    RangerCr10TeleopEnvCfg,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.pose_command_teleop import (  # noqa: E402
    TCP_DEFAULT_POS_B,
    TCP_DEFAULT_QUAT_B,
)

import isaaclab.utils.math as iu  # noqa: E402

OUT = Path(args_cli.out)
OUT.mkdir(parents=True, exist_ok=True)
DEFAULT = np.asarray(tk.DEFAULT_Q, dtype=np.float64)
SCALE = 0.25
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_net(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj["actor_state_dict"] if isinstance(obj, dict) and "actor_state_dict" in obj else obj
    sd = {(k[4:] if k.startswith("mlp.") else k): v for k, v in sd.items() if k.split(".")[-1] in ("weight", "bias")}
    net = torch.nn.Sequential(
        torch.nn.Linear(sd["0.weight"].shape[1], 512),
        torch.nn.ELU(),
        torch.nn.Linear(512, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 8),
    ).to(DEV)
    net.load_state_dict(sd)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def quat_from_euler_xyz_np(rpy):
    r, p, y = rpy
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def task_error(q, cmd_pos, cmd_R):
    T = tk.fk(np.asarray(q, dtype=np.float64))
    dp = float(np.linalg.norm(T[:3, 3] - cmd_pos))
    da = math.degrees(np.linalg.norm(tk.log_R(cmd_R @ T[:3, :3].T)))
    return dp, da


def build_env():
    cfg = RangerCr10TeleopEnvCfg()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 1e6
    cfg.commands.ee_pose.debug_vis = False
    cfg.commands.ee_pose.resampling_time_range = (1e6, 1e6)
    cfg.commands.base_velocity.resampling_time_range = (1e6, 1e6)
    cfg.observations.policy.enable_corruption = False
    cfg.curriculum.lin_vel_cmd_levels = None
    cfg.curriculum.ang_vel_cmd_levels = None
    return gym.make("RANGER-CR10-TELEOP-WBC", cfg=cfg)


CMDS = [
    (0.2, 0.0, 0.0),
    (-0.2, 0.0, 0.0),
    (0.0, 0.2, 0.0),
    (0.0, -0.2, 0.0),
    (0.0, 0.0, 0.25),
    (0.0, 0.0, -0.25),
    (0.15, 0.10, -0.10),
    # what the keyboard actually sends: one axis at a time, here the "I" key for 0.9 s
    ((0.0, 0.0, 0.0), (0.09, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (-0.09, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.09)),
]
if Path(args_cli.cmds_json).exists():
    with open(args_cli.cmds_json) as _f:
        js = json.load(_f)
    test_cmds = js.get("test_cmds", [])
    if test_cmds:
        rng = np.random.default_rng(args_cli.seed)
        pick = rng.choice(len(test_cmds), min(args_cli.n_test_cmds, len(test_cmds)), replace=False)
        for k in pick:
            c = np.asarray(test_cmds[k])
            # the stored command is (pos, quat); convert back to the chassis-frame offset
            off = tk.quat_to_R(c[3:]) @ tk.quat_to_R(np.array(TCP_DEFAULT_QUAT_B)).T
            rpy = tk.R_to_rpy(off)
            dpos = c[:3] - np.asarray(TCP_DEFAULT_POS_B)
            CMDS.append((tuple(rpy), tuple(dpos)))
else:
    print(f"警告: {args_cli.cmds_json} 不存在, 只用单轴命令 (这些也不是训练分布里的)")

env = build_env()
u = env.unwrapped
robot = u.scene["robot"]
ee_cmd = u.command_manager.get_term("ee_pose")
vel_cmd = u.command_manager.get_term("base_velocity")
arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
NACT = u.action_manager.total_action_dim


def set_command(entry, vel=(0.0, 0.0)):
    if len(entry) == 2 and isinstance(entry[0], tuple):
        rpy, dpos = entry
    else:
        rpy, dpos = entry, (0.0, 0.0, 0.0)
    dR = iu.quat_from_euler_xyz(*[torch.tensor([float(v)], device=u.device) for v in rpy])
    tq = iu.quat_mul(dR, torch.tensor(TCP_DEFAULT_QUAT_B, device=u.device).unsqueeze(0))[0]
    ee_cmd.pose_command_b[:, :3] = torch.tensor(np.asarray(TCP_DEFAULT_POS_B) + np.asarray(dpos), device=u.device)
    ee_cmd.pose_command_b[:, 3:] = tq
    vel_cmd.is_standing_env[:] = False
    vel_cmd.vel_command_b[:, 0] = vel[0]
    vel_cmd.vel_command_b[:, 1] = 0.0
    vel_cmd.vel_command_b[:, 2] = vel[1]
    return (np.asarray(ee_cmd.pose_command_b[0, :3].cpu()), tk.quat_to_R(ee_cmd.pose_command_b[0, 3:].cpu().numpy()))


def arm_q():
    return robot.data.joint_pos[0, arm_ids].cpu().numpy().astype(np.float64)


def actor_step(net, obs):
    with torch.inference_mode():
        return net(obs["policy"])


def ik_action(cmd_p, cmd_R):
    a = torch.zeros(1, NACT, device=u.device)
    q_sol, ep, er, _, _ = tk.local_ik(arm_q(), cmd_p, cmd_R)
    a[0, 2:] = torch.tensor((q_sol - DEFAULT) / SCALE, dtype=torch.float32, device=u.device)
    return a, q_sol


results = {"C1": {}, "C2": {}, "IK": {}}
for name_path in args_cli.policies:
    name, path = name_path.split("=", 1)
    net = load_net(path)
    print(f"\n{'=' * 100}\n策略 {name}: {path}\n{'=' * 100}")
    c1, c2 = [], []
    print(f"C1 IK 交接 (IK 预热 {args_cli.warm} 步 -> Actor {args_cli.watch} 步 = {args_cli.watch * 0.1:.1f} s)")
    print(f"  {'命令':<30}{'交接误差':>9}{'首步 q_des':>11}{'0.5s':>7}{'1.5s':>7}{'2.5s':>7}{'位置':>8}{'抖动':>7}")
    for k, entry in enumerate(CMDS):
        obs, _ = env.reset(seed=args_cli.seed + k)
        cmd_p, cmd_R = set_command(entry)
        for _ in range(args_cli.warm):
            a, _ = ik_action(cmd_p, cmd_R)
            obs, _, _, _, _ = env.step(a)
        _, err_hand = task_error(arm_q(), cmd_p, cmd_R)
        # first actor step: what joint target does it ask for?
        a = actor_step(net, obs)
        q_des = DEFAULT + SCALE * a[0, 2:].cpu().numpy()
        _, err_des = task_error(q_des, cmd_p, cmd_R)
        trace, pos_trace = [], []
        for t in range(args_cli.watch):
            obs, _, _, _, _ = env.step(a)
            dp, da = task_error(arm_q(), cmd_p, cmd_R)
            trace.append(da)
            pos_trace.append(dp)
            a = actor_step(net, obs)
        jit = float(np.std(trace[-10:]))
        rec = dict(
            cmd=str(entry),
            hand=err_hand,
            des=err_des,
            e05=trace[4],
            e15=trace[14],
            e25=trace[-1],
            pos=float(np.mean(pos_trace[-5:])),
            jitter=jit,
            trace=trace,
        )
        c1.append(rec)
        print(
            f"  {str(entry)[:29]:<30}{err_hand:>7.1f}°{err_des:>10.1f}°{trace[4]:>7.1f}°"
            f"{trace[14]:>7.1f}°{trace[-1]:>7.1f}°{rec['pos'] * 1000:>7.0f}mm{jit:>7.2f}°"
        )

    print(f"\nC2 主动收敛 (命令设定后 Actor 自己走 {args_cli.watch} 步)")
    print(f"  {'命令':<30}{'起始误差':>9}{'0.5s':>7}{'2.5s':>7}{'位置':>8}{'抖动':>7}{'终止':>6}")
    for k, entry in enumerate(CMDS):
        obs, _ = env.reset(seed=args_cli.seed + 100 + k)
        cmd_p, cmd_R = set_command(entry)
        obs, _, _, _, _ = env.step(torch.zeros(1, NACT, device=u.device))
        _, err0 = task_error(arm_q(), cmd_p, cmd_R)
        a = actor_step(net, obs)
        trace, pos_trace, term = [], [], False
        for t in range(args_cli.watch):
            obs, _, done, _, _ = env.step(a)
            term = term or bool(done.any())
            dp, da = task_error(arm_q(), cmd_p, cmd_R)
            trace.append(da)
            pos_trace.append(dp)
            a = actor_step(net, obs)
        rec = dict(
            cmd=str(entry),
            e0=err0,
            e05=trace[4],
            e25=trace[-1],
            pos=float(np.mean(pos_trace[-5:])),
            jitter=float(np.std(trace[-10:])),
            term=term,
        )
        c2.append(rec)
        print(
            f"  {str(entry)[:29]:<30}{err0:>7.1f}°{trace[4]:>7.1f}°{trace[-1]:>7.1f}°"
            f"{rec['pos'] * 1000:>7.0f}mm{rec['jitter']:>7.2f}°{'是' if term else '否':>6}"
        )

    # chassis + arm together: a fix for the arm must not be paid for by the chassis
    print("\nC2b 底盘 + 机械臂同时动 (vx 0.25 / wz 0.15, 姿态命令 roll+0.2)")
    obs, _ = env.reset(seed=args_cli.seed + 300)
    cmd_p, cmd_R = set_command(CMDS[0], vel=(0.25, 0.15))
    for _ in range(60):
        a = actor_step(net, obs)
        obs, _, _, _, _ = env.step(a)
    ee_err = task_error(arm_q(), cmd_p, cmd_R)
    vx = robot.data.root_lin_vel_b[0, 0].item()
    wz = robot.data.root_ang_vel_b[0, 2].item()
    print(
        f"  底盘 vx 指令/实际 {0.25:+.3f}/{vx:+.3f}, wz {0.15:+.3f}/{wz:+.3f};  "
        f"EE 姿态 {ee_err[1]:.1f}° 位置 {ee_err[0] * 1000:.0f} mm"
    )
    results["C1"][name] = c1
    results["C2"][name] = c2
    results["C2b"] = results.get("C2b", {})
    results["C2b"][name] = dict(vx=vx, wz=wz, ee_att=ee_err[1], ee_pos=ee_err[0])

# online IK as the reachability reference
print(f"\n{'=' * 100}\n参考: 同一批命令用在线 IK 直接驱动\n{'=' * 100}")
ik_rows = []
for k, entry in enumerate(CMDS):
    obs, _ = env.reset(seed=args_cli.seed + k)
    cmd_p, cmd_R = set_command(entry)
    for _ in range(args_cli.warm + args_cli.watch):
        a, _ = ik_action(cmd_p, cmd_R)
        obs, _, _, _, _ = env.step(a)
    dp, da = task_error(arm_q(), cmd_p, cmd_R)
    ik_rows.append(dict(cmd=str(entry), pos=dp, att=da))
    print(f"  {str(entry)[:40]:<42}{da:5.2f}°  {dp * 1000:4.0f} mm")
results["IK"] = ik_rows

# ------------------------------------------------------------------ summary ----
print(f"\n{'=' * 100}\n汇总 (中位数)\n{'=' * 100}")
print(
    f"  {'策略':<14}{'C1 首步':>9}{'C1 0.5s':>9}{'C1 2.5s':>9}{'C1 位置':>9}"
    f"{'C2 起始':>9}{'C2 0.5s':>9}{'C2 2.5s':>9}{'C2 位置':>9}{'抖动':>7}"
)
summary = {}
for name in results["C1"]:
    c1, c2 = results["C1"][name], results["C2"][name]
    row = dict(
        c1_des=float(np.median([r["des"] for r in c1])),
        c1_05=float(np.median([r["e05"] for r in c1])),
        c1_25=float(np.median([r["e25"] for r in c1])),
        c1_pos=float(np.median([r["pos"] for r in c1])),
        c2_e0=float(np.median([r["e0"] for r in c2])),
        c2_05=float(np.median([r["e05"] for r in c2])),
        c2_25=float(np.median([r["e25"] for r in c2])),
        c2_pos=float(np.median([r["pos"] for r in c2])),
        jitter=float(np.median([r["jitter"] for r in c2])),
        c2b=results["C2b"][name],
    )
    summary[name] = row
    print(
        f"  {name:<14}{row['c1_des']:>8.1f}°{row['c1_05']:>8.1f}°{row['c1_25']:>8.1f}°"
        f"{row['c1_pos'] * 1000:>8.0f}mm{row['c2_e0']:>8.1f}°{row['c2_05']:>8.1f}°"
        f"{row['c2_25']:>8.1f}°{row['c2_pos'] * 1000:>8.0f}mm{row['jitter']:>6.2f}°"
    )
print(
    f"  {'在线 IK':<14}{'':>9}{'':>9}{np.median([r['att'] for r in ik_rows]):>8.1f}°"
    f"{np.median([r['pos'] for r in ik_rows]) * 1000:>8.0f}mm"
)
print("\n  底盘 + 机械臂:")
for name, row in summary.items():
    b = row["c2b"]
    print(
        f"  {name:<14}vx {b['vx']:+.3f} (指令 +0.250)  wz {b['wz']:+.3f} (指令 +0.150)  "
        f"EE 姿态 {b['ee_att']:5.1f}°  位置 {b['ee_pos'] * 1000:4.0f} mm"
    )

with open(OUT / "stage45i.json", "w") as _f:
    json.dump(dict(summary=summary, raw=results, cmds=[str(c) for c in CMDS]), _f, indent=1, default=float)
print(f"\n已写出 {OUT / 'stage45i.json'}")
env.close()
app.close()
