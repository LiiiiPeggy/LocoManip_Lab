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

"""Stage 4.5k: where does the chassis drift come from, and does a deadband remove it?

Mode B of the acceptance protocol ("the arm moves, the chassis command is zero") is the case
that matters: the chassis still travels a few millimetres per second while the operator has
asked for nothing. Three candidate sources are separated by logging them side by side:

    external command   the `(vx, wz)` the operator asked for
    actor output       `0.5 * tanh(raw[0:2])`, i.e. what the policy tells the base controller
    resulting motion   the measured base linear / angular velocity and the accumulated pose

If the actor's output is already ~0 while the base still moves, the deadband is the wrong
tool and the report says so instead of filtering.

The second half applies `BaseCommandFilter` (`base_command_filter.py`) in two stages and
repeats every scenario, so the effect on the arm is measured, not assumed.

Usage:
    python scripts/ranger_cr10/check_base_drift.py \
        --policy logs/rsl_rl/ranger_cr10_teleop/2026-10-29_19-12-08_stage45j_ppo/model_5999.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_command_filter import BaseCommandFilter  # noqa: E402

parser = argparse.ArgumentParser(description="Chassis drift diagnosis and deadband comparison.")
parser.add_argument("--policy", default="logs/rsl_rl/ranger_cr10_teleop/2026-10-29_19-12-08_stage45j_ppo/model_5999.pt")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--vx_deadband", type=float, default=0.02)
parser.add_argument("--wz_deadband", type=float, default=0.02)
parser.add_argument("--hysteresis", type=float, default=0.0)
parser.add_argument("--out", default="outputs/base_drift")
parser.add_argument("--static_steps", type=int, default=300, help="30 s at the 10 Hz policy rate")
parser.add_argument("--move_steps", type=int, default=150)
args_cli = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import LeggedManip_Lab.tasks  # noqa: E402,F401
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
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_net(path: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj["actor_state_dict"] if isinstance(obj, dict) and "actor_state_dict" in obj else obj
    sd = {(k[4:] if k.startswith("mlp.") else k): v for k, v in sd.items() if k.split(".")[-1] in ("weight", "bias")}
    net = torch.nn.Sequential(
        torch.nn.Linear(117, 512),
        torch.nn.ELU(),
        torch.nn.Linear(512, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 8),
    ).to(DEV)
    net.load_state_dict(sd)
    net.eval()
    return net


def build_env(num_envs: int):
    cfg = RangerCr10TeleopEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = 1e6
    cfg.commands.ee_pose.debug_vis = False
    cfg.commands.ee_pose.resampling_time_range = (1e6, 1e6)
    cfg.commands.base_velocity.resampling_time_range = (1e6, 1e6)
    cfg.observations.policy.enable_corruption = False
    cfg.curriculum.lin_vel_cmd_levels = None
    cfg.curriculum.ang_vel_cmd_levels = None
    return gym.make("RANGER-CR10-TELEOP-WBC", cfg=cfg)


env = build_env(args_cli.num_envs)
u = env.unwrapped
robot = u.scene["robot"]
ee_cmd = u.command_manager.get_term("ee_pose")
vel_cmd = u.command_manager.get_term("base_velocity")
base_term = u.action_manager.get_term("base_vel")
net = load_net(args_cli.policy)
N = u.num_envs
DT = u.step_dt

EE_TARGETS = [
    (0.0, 0.0, 0.0),
    (0.08, 0.0, 0.0),
    (0.0, 0.10, 0.0),
    (0.0, 0.0, 0.08),
    (-0.08, -0.10, 0.0),
    (0.0, 0.0, 0.0),
]
EE_ATT = [(0.0, 0.0, 0.0), (0.15, 0.0, 0.0), (0.0, 0.15, 0.0), (0.0, 0.0, 0.2), (0.1, 0.1, -0.1), (0.0, 0.0, 0.0)]


def set_ee(i: int):
    dpos, drpy = EE_TARGETS[i % len(EE_TARGETS)], EE_ATT[i % len(EE_ATT)]
    dR = iu.quat_from_euler_xyz(*[torch.tensor([float(v)], device=u.device) for v in drpy])
    tq = iu.quat_mul(dR, torch.tensor(TCP_DEFAULT_QUAT_B, device=u.device).unsqueeze(0))[0]
    ee_cmd.pose_command_b[:, :3] = torch.tensor(np.asarray(TCP_DEFAULT_POS_B) + np.asarray(dpos), device=u.device)
    ee_cmd.pose_command_b[:, 3:] = tq


def set_vel(vx: float, wz: float):
    vel_cmd.is_standing_env[:] = False
    vel_cmd.vel_command_b[:, 0] = vx
    vel_cmd.vel_command_b[:, 1] = 0.0
    vel_cmd.vel_command_b[:, 2] = wz


def pose_error():
    return (ee_cmd.metrics["position_error"].mean().item(), ee_cmd.metrics["orientation_error"].mean().item())


def run(name: str, filt: BaseCommandFilter, schedule):
    """`schedule` is a list of (vx, wz, ee_index) held for `phase_steps` steps."""
    obs, _ = env.reset(seed=2026)
    rows = []
    for vx_cmd, wz_cmd, ee_idx, steps in schedule:
        set_vel(vx_cmd, wz_cmd)
        set_ee(ee_idx)
        for _ in range(steps):
            a = net(obs["policy"])
            raw = a[:, :2].clone()
            vx_eff, wz_eff = filt.filter_command(raw.new_tensor([vx_cmd]).expand(N), raw.new_tensor([wz_cmd]).expand(N))
            if filt.enabled:
                set_vel(float(vx_eff[0]), float(wz_eff[0]))
            a = filt.filter_action(a, vx_eff, wz_eff)
            obs, _, _, _, _ = env.step(a)
            # log what was actually *applied*, not the pre-filter raw output
            proc = 0.5 * torch.tanh(a[:, :2])
            rows.append(
                dict(
                    vx_cmd=float(vx_eff[0]),
                    wz_cmd=float(wz_eff[0]),
                    raw_vx=raw[:, 0].mean().item(),
                    raw_wz=raw[:, 1].mean().item(),
                    proc_vx=proc[:, 0].mean().item(),
                    proc_wz=proc[:, 1].mean().item(),
                    act_vx=robot.data.root_lin_vel_b[:, 0].mean().item(),
                    act_vy=robot.data.root_lin_vel_b[:, 1].mean().item(),
                    act_wz=robot.data.root_ang_vel_b[:, 2].mean().item(),
                    base_x=robot.data.root_pos_w[:, 0].mean().item(),
                    base_y=robot.data.root_pos_w[:, 1].mean().item(),
                    yaw=robot.data.heading_w.mean().item(),
                    ee_pos=ee_cmd.metrics["position_error"].mean().item(),
                    ee_ori=ee_cmd.metrics["orientation_error"].mean().item(),
                )
            )
    with open(OUT / f"trace_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def summarise(name: str, rows) -> dict:
    arr = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    n = len(rows)
    # drift is measured on the second half, after the initial settle
    half = slice(n // 2, None)
    dx = arr["base_x"][-1] - arr["base_x"][n // 2]
    dy = arr["base_y"][-1] - arr["base_y"][n // 2]
    return dict(
        scenario=name,
        steps=n,
        cmd_vx=float(np.mean(arr["vx_cmd"])),
        cmd_wz=float(np.mean(arr["wz_cmd"])),
        proc_vx_abs=float(np.mean(np.abs(arr["proc_vx"]))),
        proc_wz_abs=float(np.mean(np.abs(arr["proc_wz"]))),
        raw_vx_abs=float(np.mean(np.abs(arr["raw_vx"]))),
        act_speed=float(np.mean(np.hypot(arr["act_vx"], arr["act_vy"]))),
        act_vx=float(np.mean(arr["act_vx"])),
        act_wz_abs=float(np.mean(np.abs(arr["act_wz"]))),
        drift_m=float(math.hypot(dx, dy)),
        drift_yaw_deg=float(math.degrees(arr["yaw"][-1] - arr["yaw"][n // 2])),
        ee_pos=float(np.mean(arr["ee_pos"])),
        ee_ori=float(np.mean(arr["ee_ori"])),
        vx_err=float(np.mean(np.abs(arr["act_vx"] - arr["vx_cmd"]))),
        wz_err=float(np.mean(np.abs(arr["act_wz"] - arr["wz_cmd"]))),
        noise_act_vx=float(np.std(arr["act_vx"][half])),
    )


CONFIGS = [
    ("none", BaseCommandFilter(0.0, 0.0, False)),
    ("input", BaseCommandFilter(args_cli.vx_deadband, args_cli.wz_deadband, False, args_cli.hysteresis)),
    ("input+lock", BaseCommandFilter(args_cli.vx_deadband, args_cli.wz_deadband, True, args_cli.hysteresis)),
]

S, M = args_cli.static_steps, args_cli.move_steps
SCHEDULES = {
    # arm keeps moving (six targets), chassis command is zero the whole time
    "static": [(0.0, 0.0, i, (S // 6) + (1 if i < S % 6 else 0)) for i in range(6)],
    # a command that is nonzero but below the deadband: this is the case the input
    # deadband exists for (an operator that does not quite return the stick to zero)
    "static_near": [(0.01, 0.012, 0, S)],
    "low": [(0.05, 0.03, 0, M)],
    "normal": [(0.25, 0.15, 0, M)],
    # zero -> move -> zero -> move, to expose lag introduced by the filter
    "switch": [(0.0, 0.0, 0, M // 3), (0.20, 0.10, 0, M // 3), (0.0, 0.0, 0, M // 3), (0.20, 0.10, 0, M // 3)],
}

results = []
print("=" * 104)
print(
    f"policy {args_cli.policy}; deadband vx {args_cli.vx_deadband} m/s, wz {args_cli.wz_deadband} rad/s, "
    f"hysteresis {args_cli.hysteresis}"
)
print("=" * 104)
for cfg_name, filt in CONFIGS:
    for scen, sched in SCHEDULES.items():
        rows = run(f"{cfg_name}_{scen}", filt, sched)
        summ = summarise(f"{cfg_name}/{scen}", rows)
        summ["config"] = cfg_name
        summ["scenario_name"] = scen
        results.append(summ)
        print(
            f"  {cfg_name:<11}{scen:<8} proc|vx| {summ['proc_vx_abs']:.4f} act_vx {summ['act_vx']:+.4f} "
            f"speed {summ['act_speed']:.4f} |wz| {summ['act_wz_abs']:.4f}  drift {summ['drift_m'] * 1000:5.1f} mm "
            f"yaw {summ['drift_yaw_deg']:+.2f}°  EE {summ['ee_pos'] * 1000:5.1f} mm / "
            f"{math.degrees(summ['ee_ori']):4.2f}°"
        )

# ---- summary table required by the brief ---------------------------------------------
print("\n" + "=" * 104)
print("对照表(static = 30 s 内手臂持续运动、底盘命令为 0)")
print("=" * 104)
print(f"  {'指标':<28}{'无滤波':>14}{'输入死区':>14}{'输入死区+零速锁定':>20}")
keys = [
    ("act_speed", "静止线速度 |v| (m/s)"),
    ("act_wz_abs", "静止角速度 |wz| (rad/s)"),
    ("drift_m", "累计漂移距离 (mm)"),
    ("drift_yaw_deg", "累计偏航变化 (°)"),
    ("ee_pos", "EE 位置误差 (mm)"),
    ("ee_ori", "EE 姿态误差 (°)"),
]
by = {(r["config"], r["scenario_name"]): r for r in results}
for key, label in keys:
    vals = []
    for cfg_name in ("none", "input", "input+lock"):
        v = by[(cfg_name, "static")][key]
        vals.append(float(v) * (1000 if key in ("drift_m", "ee_pos") else 1))
        if key == "ee_ori":
            vals[-1] = math.degrees(v)
    print(f"  {label:<28}{vals[0]:>14.4f}{vals[1]:>14.4f}{vals[2]:>20.4f}")

print("\n  底盘运动跟踪(命令 |误差|):")
for scen in ("low", "normal", "switch"):
    for cfg_name in ("none", "input", "input+lock"):
        r = by[(cfg_name, scen)]
        print(
            f"    {cfg_name:<11}{scen:<8} vx {r['vx_err']:.4f}  wz {r['wz_err']:.4f}  "
            f"EE {r['ee_pos'] * 1000:5.1f} mm / {math.degrees(r['ee_ori']):4.2f}°"
        )

(OUT / "base_drift.json").write_text(json.dumps(results, indent=1))
print(f"\n已写出 {OUT / 'base_drift.json'} 与 trace_*.csv")
env.close()
app.close()
