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

"""Acceptance tests for the world-frame WBC policy (docs/plans/plan.md section 7, stage 4).

Three stages, because they need incompatible scene setups and Isaac Sim's
``SimulationContext`` is a singleton -- a second ``gym.make`` in one process hangs rather
than failing, so each stage must be its own process:

    # does the policy's (vx, wz) action actually move the chassis as commanded?
    python scripts/ranger_cr10/eval_world_wbc.py --stage base

    # with the target frozen: does tracking hold, does the chassis settle on arrival,
    # and how does that break down by how far the target is?
    python scripts/ranger_cr10/eval_world_wbc.py --stage track --policy <path/to/policy.pt>

    # control group: freeze (vx, wz) to zero and see what the base is worth
    python scripts/ranger_cr10/eval_world_wbc.py --stage frozen --policy <path/to/policy.pt>

All three are headless and print a table; none of them is meaningful from an exit code.
"""

from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="World-frame WBC acceptance tests.")
parser.add_argument("--stage", choices=("base", "track", "frozen"), required=True)
parser.add_argument(
    "--policy",
    default="",
    help="TorchScript policy exported by scripts/rsl_rl/play.py. Required by track and frozen.",
)
parser.add_argument("--num_envs", type=int, default=0, help="0 = the stage's own default.")
parser.add_argument(
    "--task",
    default="RANGER-CR10-WORLD-WBC",
    help="Registered gym id. Override with RANGER-CR10-WBC to evaluate a pre-stage-4 "
    "checkpoint, which is the only way to compare against the baseline policy: its "
    "99-dim observation does not fit the current config.",
)
parser.add_argument(
    "--curriculum_box",
    action="store_true",
    help="Track/frozen only: sample from where the curriculum starts instead of its end "
    "state. Useful for a like-for-like check against a near-target run, but every target "
    "then lands in the near stratum.",
)
args_cli = parser.parse_args()

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import LeggedManip_Lab.tasks  # noqa: E402,F401
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.wbc_env_cfg import (  # noqa: E402
    RangerCr10WBCEnvCfg,
)

if args_cli.stage in ("track", "frozen") and not args_cli.policy:
    sys.exit(f"--stage {args_cli.stage} needs --policy")


def make_env(num_envs: int, freeze_target: bool, deterministic_scene: bool, full_range: bool):
    """Build the WBC env for one stage.

    ``freeze_target`` pushes the command's resampling interval past the episode length;
    ``deterministic_scene`` also drops the reset randomisation so a measured displacement
    is the robot's motion and not the spawn.
    """
    cfg = RangerCr10WBCEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = 600.0
    cfg.commands.ee_pose.debug_vis = False
    cfg.observations.policy.enable_corruption = False
    if freeze_target:
        cfg.commands.ee_pose.resampling_time_range = (1e6, 1e6)
        # hold the sampled box still for the whole run
        cfg.curriculum.pos_cmd_levels = None
    if full_range:
        # Sample from the curriculum's end state, not from where it starts. The starting
        # box is small enough that every target lands in the "near" stratum, which is
        # exactly the stratum the policy is already good at -- so a run at the starting
        # box would report a good number and answer none of the questions. This is also
        # what pos_cmd_levels itself does when the curriculum is off, and what the -Play
        # config does.
        cfg.commands.ee_pose.ranges = cfg.commands.ee_pose.limit_ranges
    if deterministic_scene:
        cfg.events.reset_base = None
        cfg.events.reset_robot_joints = None
    return gym.make(args_cli.task, cfg=cfg)


def raw_action_for(term, vx: float, wz: float) -> list[float]:
    """Invert the action term's ``tanh`` squash so a sweep can be written in physical units.

    ``process_actions`` maps the policy's raw output through ``mid + half * tanh(raw)``, so
    a raw action is not a velocity: feeding 0.3 in asks for ``0.5 * tanh(0.3)`` = 0.146 m/s.
    Every number below is a physical command, so the inversion belongs here and not in the
    reader's head.
    """
    low = torch.tensor(term.cfg.clip[0], device=term.device)
    high = torch.tensor(term.cfg.clip[1], device=term.device)
    mid, half = (low + high) / 2, (high - low) / 2
    if bool((half == 0).any()):
        raise ValueError("base_vel is braked in this config; there is nothing to sweep.")
    cmd = torch.tensor([vx, wz], device=term.device)
    return torch.atanh(torch.clamp((cmd - mid) / half, -0.999, 0.999)).tolist()


def yaw_of(q) -> float:
    return math.degrees(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


# ---------------------------------------------------------------- stage: base ------
def stage_base() -> None:
    """Sweep (vx, wz) into the action term directly and report commanded vs achieved.

    The policy never sees the chassis joints, so this is the layer where a systematic
    sim-to-real error would hide: an open-loop action that the real driver executes
    faithfully but the simulation does not would make a trained policy over-command.
    """
    env = make_env(args_cli.num_envs or 1, freeze_target=True, deterministic_scene=True, full_range=False)
    u = env.unwrapped
    robot = u.scene["robot"]
    term = u.action_manager.get_term("base_vel")
    wheel_ids = term._wheel_ids
    steer_ids = term._steering_ids

    print("\n== 车轮位置: 模型实际 vs 控制器假定 (base_link 坐标系) ==")
    from isaaclab.utils.math import quat_apply_inverse

    base_idx = robot.find_bodies("base_link")[0][0]
    base_p, base_q = robot.data.body_pos_w[0, base_idx], robot.data.body_quat_w[0, base_idx]
    for link_name, cfg_xy in zip(
        ["fr_wheel_link", "fl_wheel_link", "rl_wheel_link", "rr_wheel_link"], term.cfg.wheel_xy
    ):
        idx = robot.find_bodies(link_name)[0]
        if len(idx) == 0:
            continue
        p_b = quat_apply_inverse(base_q, robot.data.body_pos_w[0, idx[0]] - base_p)
        err = math.hypot(p_b[0].item() - cfg_xy[0], p_b[1].item() - cfg_xy[1])
        print(
            f"  {link_name:<16} 实际 ({p_b[0]:+.3f},{p_b[1]:+.3f})"
            f"  假定 ({cfg_xy[0]:+.3f},{cfg_xy[1]:+.3f})  差 {err:.4f} m"
        )

    print("\n== 指令 -> 实际底盘速度 (4 s, 稳态取后 3 s) ==")
    print(f"{'cmd':<18}{'achieved':>10}{'cmd':>8}{'ratio':>8}   {'wheel |w| act/tgt':>22}")
    for vx, wz in [(0.3, 0.0), (0.5, 0.0), (-0.3, 0.0), (0.3, 0.3), (0.5, 0.5), (0.0, 0.3), (0.0, 0.5)]:
        env.reset()
        a = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
        a[:, :2] = torch.tensor(raw_action_for(term, vx, wz), device=u.device)
        vb, wb, wh_t, wh_a = [], [], [], []
        for i in range(40):
            env.step(a)
            if i >= 10:
                vb.append(robot.data.root_lin_vel_b[0].cpu().numpy().copy())
                wb.append(robot.data.root_ang_vel_b[0].cpu().numpy().copy())
                wh_t.append(term._wheel_target[0].cpu().numpy().copy())
                wh_a.append(robot.data.joint_vel[0, wheel_ids].cpu().numpy().copy())
        vb_m, wb_m = sum(vb) / len(vb), sum(wb) / len(wb)
        t_m, a_m = sum(wh_t) / len(wh_t), sum(wh_a) / len(wh_a)
        if wz == 0.0:
            label, cmd_s, act_s = f"vx {vx:+.1f}", abs(vx), abs(vb_m[0])
        else:
            label, cmd_s, act_s = f"wz {wz:+.1f} (vx {vx:+.1f})", abs(wz), abs(wb_m[2])
        print(
            f"{label:<18}{act_s:>10.3f}{cmd_s:>8.2f}{act_s / cmd_s:>8.2f}   "
            f"{abs(a_m).mean():>8.2f} / {abs(t_m).mean():.2f}"
        )

    print("\n== 关节级: 目标 vs 实际 (后 3 s 均值) ==")
    for vx, wz in [(0.3, 0.0), (0.3, 0.3), (0.0, 0.3)]:
        env.reset()
        a = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
        a[:, :2] = torch.tensor(raw_action_for(term, vx, wz), device=u.device)
        acc = []
        for i in range(40):
            env.step(a)
            if i >= 10:
                acc.append(
                    (
                        term._steering_target[0].cpu().numpy().copy(),
                        robot.data.joint_pos[0, steer_ids].cpu().numpy().copy(),
                        term._wheel_target[0].cpu().numpy().copy(),
                        robot.data.joint_vel[0, wheel_ids].cpu().numpy().copy(),
                    )
                )
        st, sa, wt, wa = (sum(x[i] for x in acc) / len(acc) for i in range(4))
        print(f"\n  cmd vx={vx:+.1f} wz={wz:+.1f}")
        print(f"    {'joint':<28}{'steer tgt':>11}{'steer act':>11}{'wheel tgt':>11}{'wheel act':>11}")
        for i, name in enumerate(term._steering_names):
            print(
                f"    {name:<28}{math.degrees(st[i]):>10.1f}°{math.degrees(sa[i]):>10.1f}°"
                f"{wt[i]:>11.3f}{wa[i]:>11.3f}"
            )

    env.close()


# --------------------------------------------------- stages: track and frozen ------
def rollout(env, net, freeze_base: bool, steps: int):
    """One frozen-target episode; returns the step traces and the final per-env errors."""
    u = env.unwrapped
    robot = u.scene["robot"]
    cmd = u.command_manager.get_term("ee_pose")

    obs, _ = env.reset()
    fixed_pos, fixed_quat = cmd.target_pos_w.clone(), cmd.target_quat_w.clone()
    # Measured from the freshly frozen target rather than from
    # ``cmd.metrics["distance_to_target"]``, which reads exactly zero here:
    # ``CommandTerm.reset`` zeroes the metrics and only ``compute`` -- i.e. the first
    # ``env.step`` -- recomputes them, so a read straight after ``env.reset()`` is stale.
    base_idx = robot.find_bodies("base_link")[0][0]
    d0 = torch.norm(fixed_pos - robot.data.body_pos_w[:, base_idx], dim=-1).clone()

    trace = {k: [] for k in ("vx_cmd", "wz_cmd", "vx_act", "wz_act", "pos", "ori", "dist", "speed", "disp")}
    p0 = robot.data.root_pos_w[:, :2].clone()
    last_action = None
    for _ in range(steps):
        with torch.inference_mode():
            a = net(obs["policy"])
        if freeze_base:
            a = a.clone()
            a[:, :2] = 0.0
        obs, _, _, _, _ = env.step(a)
        cmd.target_pos_w[:] = fixed_pos
        cmd.target_quat_w[:] = fixed_quat
        last_action = a
        # the action is squashed by the term, so read the command it landed on, not the raw
        # network output: those differ by exactly the factor the tanh fix is about
        processed = u.action_manager.get_term("base_vel").processed_actions
        trace["vx_cmd"].append(processed[:, 0].abs().mean().item())
        trace["wz_cmd"].append(processed[:, 1].abs().mean().item())
        trace["vx_act"].append(robot.data.root_lin_vel_b[:, 0].abs().mean().item())
        trace["wz_act"].append(robot.data.root_ang_vel_b[:, 2].abs().mean().item())
        trace["pos"].append(cmd.metrics["position_error"].mean().item())
        trace["ori"].append(cmd.metrics["orientation_error"].mean().item())
        trace["dist"].append(cmd.metrics["distance_to_target"].mean().item())
        trace["speed"].append(robot.data.root_lin_vel_b[:, :2].norm(dim=-1).mean().item())
        trace["disp"].append((robot.data.root_pos_w[:, :2] - p0).norm(dim=-1).mean().item())

    return trace, d0, last_action, cmd, robot


def stage_track() -> None:
    env = make_env(
        args_cli.num_envs or 512,
        freeze_target=True,
        deterministic_scene=False,
        full_range=not args_cli.curriculum_box,
    )
    u = env.unwrapped
    net = torch.jit.load(args_cli.policy, map_location=u.device)
    print(f"policy: {args_cli.policy}")

    trace, d0, last_action, cmd, robot = rollout(env, net, False, 120)

    print(f"\n{'step':>5}" + "".join(f"{k:>10}" for k in trace))
    for i in range(0, 120, 10):
        print(f"{i:>5}" + "".join(f"{trace[k][i]:>10.4f}" for k in trace))

    pos_err = cmd.metrics["position_error"]
    ori_err = cmd.metrics["orientation_error"]
    speed = robot.data.root_lin_vel_b[:, :2].norm(dim=-1)

    print("\n== 按初始目标距离分层 ==")
    print(
        f"  初始距离分布: 最小 {d0.min().item():.3f} / 均值 {d0.mean().item():.3f} "
        f"/ 最大 {d0.max().item():.3f} m"
    )
    print(f"{'stratum':<10}{'n':>6}{'pos err':>10}{'ori err':>10}{'success':>10}{'base spd':>10}")
    for name, lo, hi in (("near", 0.0, 0.75), ("edge", 0.75, 1.15), ("far", 1.15, 99.0)):
        m = (d0 >= lo) & (d0 < hi)
        if m.sum() == 0:
            continue
        print(
            f"{name:<10}{int(m.sum()):>6}{pos_err[m].mean().item():>10.3f}"
            f"{ori_err[m].mean().item():>10.3f}{(pos_err[m] < 0.15).float().mean().item():>10.2%}"
            f"{speed[m].mean().item():>10.3f}"
        )

    print("\n== 到位后底盘是否停住 (位置误差 < 0.05 m) ==")
    settled = pos_err < 0.05
    yaw_rate = robot.data.root_ang_vel_b[:, 2].abs()
    yaw_cmd = u.action_manager.get_term("base_vel").processed_actions[:, 1].abs()
    print(f"到位 env 数: {int(settled.sum())} / {len(pos_err)}")
    if settled.sum() > 0:
        base_cmd = u.action_manager.get_term("base_vel").processed_actions
        print(f"  线速度   均值 {speed[settled].mean().item():.4f} m/s   "
              f">0.05 m/s 的比例 {(speed[settled] > 0.05).float().mean().item():.2%}")
        # Angular rate separately: a chassis pirouetting on the spot holds its position
        # to within the tolerance while being anything but settled, and a linear-speed
        # check would call that a pass.
        print(f"  角速度   均值 {yaw_rate[settled].mean().item():.4f} rad/s   "
              f">0.05 rad/s 的比例 {(yaw_rate[settled] > 0.05).float().mean().item():.2%}")
        cmd_vx = base_cmd[settled, 0].abs().mean().item()
        raw_vx = last_action[settled, 0].abs().mean().item()
        raw_wz = last_action[settled, 1].abs().mean().item()
        ratio = yaw_rate[settled].mean().item() / max(yaw_cmd[settled].mean().item(), 1e-6)
        print(f"  策略下发 |vx| / |wz|   {cmd_vx:.4f} / {yaw_cmd[settled].mean().item():.4f}")
        print(f"  网络原始 |vx| / |wz|   {raw_vx:.4f} / {raw_wz:.4f}")
        print(f"  角速度指令 -> 实际     {ratio:.2%}")
        print("  (动作上限 ±0.5 经 tanh 压缩;原始输出接近或超过它即说明底盘通道顶在饱和处)")

    print("\n== 指令 vs 实际 (全程均值) ==")
    for comp, kc, ka in (("vx", "vx_cmd", "vx_act"), ("wz", "wz_cmd", "wz_act")):
        print(f"  |{comp}| 指令 {sum(trace[kc]) / len(trace[kc]):.4f}  实际 {sum(trace[ka]) / len(trace[ka]):.4f}")

    env.close()


def stage_frozen() -> None:
    env = make_env(
        args_cli.num_envs or 256,
        freeze_target=True,
        deterministic_scene=False,
        full_range=not args_cli.curriculum_box,
    )
    u = env.unwrapped
    net = torch.jit.load(args_cli.policy, map_location=u.device)
    print(f"policy: {args_cli.policy}")

    print(f"\n{'case':<12}{'pos err':>10}{'ori err':>10}{'base disp':>12}{'base spd':>10}")
    for tag, freeze in (("normal", False), ("base frozen", True)):
        trace, _, _, cmd, robot = rollout(env, net, freeze, 200)
        n = 20  # drop the first second of settling
        p = sum(trace["pos"][n:]) / len(trace["pos"][n:])
        o = sum(trace["ori"][n:]) / len(trace["ori"][n:])
        d = sum(trace["disp"][n:]) / len(trace["disp"][n:])
        s = sum(trace["speed"][n:]) / len(trace["speed"][n:])
        print(f"{tag:<12}{p:>10.4f}{o:>10.4f}{d:>12.4f}{s:>10.4f}")
    print("\n冻结底盘若明显变差, 说明策略确实在用底盘扩大工作空间.")

    env.close()


{
    "base": stage_base,
    "track": stage_track,
    "frozen": stage_frozen,
}[args_cli.stage]()

app.close()
