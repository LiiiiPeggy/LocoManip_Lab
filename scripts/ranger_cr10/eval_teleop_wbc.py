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

"""Acceptance tests for the teleoperation WBC policy (docs/plans/teleop_plan.md stage 4.3).

Four measurements, in one process because Isaac Sim's SimulationContext is a singleton:

    mode A  chassis only  -- the velocity command moves, the EE target is held
    mode B  arm only      -- the velocity command is zero, the EE target moves
    mode C  both          -- both move together
    decoupled             -- mode C again, but the policy's two chassis channels are
                             replaced by the command fed straight to the base controller,
                             which is the "hand the velocity to the chassis and control the
                             arm separately" baseline the 8-D action has to beat

Commands are written directly and resampling is disabled, so what is measured is the
policy's response to a known input rather than a sample of the training distribution.

One environment trap worth naming: `UniformVelocityCommand._update_command` zeroes the
command for every env in `is_standing_env`, so writing `vel_command_b` alone silently
does nothing for the 25% of envs the training config marks as standing. `is_standing_env`
is cleared here before the commands are written.

Usage:
    python scripts/ranger_cr10/eval_teleop_wbc.py --policy <path/to/policy.pt>
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teleoperation WBC acceptance tests.")
parser.add_argument("--policy", required=True, help="TorchScript policy from scripts/rsl_rl/play.py")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--phase_steps", type=int, default=25, help="steps per command phase (10 Hz)")
args_cli = parser.parse_args()

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import LeggedManip_Lab.tasks  # noqa: E402,F401
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.teleop_env_cfg import (  # noqa: E402
    RangerCr10TeleopEnvCfg,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.pose_command_teleop import (  # noqa: E402
    TCP_DEFAULT_POS_B,
    TCP_DEFAULT_QUAT_B,
)

cfg = RangerCr10TeleopEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
cfg.commands.ee_pose.debug_vis = False
# nothing may resample: the test writes every command itself
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
net = torch.jit.load(args_cli.policy, map_location=u.device)
print(f"policy: {args_cli.policy}")

N = u.num_envs
DEV = u.device
default_pos = torch.tensor(TCP_DEFAULT_POS_B, device=DEV)
default_quat = torch.tensor(TCP_DEFAULT_QUAT_B, device=DEV)


def set_commands(vx: float, wz: float, ee_offset=(0.0, 0.0, 0.0)):
    """Write both commands directly, bypassing resampling and the standing-env override."""
    vel_cmd.is_standing_env[:] = False
    vel_cmd.vel_command_b[:, 0] = vx
    vel_cmd.vel_command_b[:, 1] = 0.0
    vel_cmd.vel_command_b[:, 2] = wz
    ee_cmd.pose_command_b[:, :3] = default_pos + torch.tensor(ee_offset, device=DEV)
    ee_cmd.pose_command_b[:, 3:] = default_quat


SETTLE = 10  # steps dropped at the start of every phase, so a transition is not measured


def run(phases, decoupled: bool, label: str):
    """Roll the policy through a list of command phases and report the per-mode metrics.

    Each phase is a step change in the command, so the first second of each phase is the
    response transient. Averaging over it would report a mixture of "how fast does it
    react" and "how well does it hold", and the reaction is not what these columns are
    for -- drop it, per phase, not once for the whole rollout.
    """
    obs, _ = env.reset()
    prev_action = None
    acc = {
        "lin_err": [], "ang_err": [], "pos_err": [], "ori_err": [],
        "success": [], "base_spd": [], "tgt_drift": [], "smooth": [],
    }
    for vx, wz, ee_offset in phases:
        set_commands(vx, wz, ee_offset)
        vel_cmd.compute(0.0)  # let the term apply the standing/heading post-processing
        for i in range(args_cli.phase_steps):
            with torch.inference_mode():
                a = net(obs["policy"])
            if decoupled:
                # hand the velocity command to the base controller and keep only the arm
                # channels from the policy -- the architecture this one has to beat
                a = a.clone()
                a[:, 0] = math.atanh(vx / 0.5)
                a[:, 1] = math.atanh(wz / 0.5)
            obs, _, _, _, _ = env.step(a)
            # the step may have re-applied the command term; hold the test's values
            set_commands(vx, wz, ee_offset)
            if prev_action is not None:
                acc["smooth"].append((a - prev_action).abs().mean().item())
            prev_action = a.clone()
            if i < SETTLE:
                continue

            acc["lin_err"].append((robot.data.root_lin_vel_b[:, 0] - vx).abs().mean().item())
            acc["ang_err"].append((robot.data.root_ang_vel_b[:, 2] - wz).abs().mean().item())
            acc["pos_err"].append(ee_cmd.metrics["position_error"].mean().item())
            acc["ori_err"].append(ee_cmd.metrics["orientation_error"].mean().item())
            acc["success"].append((ee_cmd.metrics["position_error"] < cfg.commands.ee_pose.success_threshold)
                                  .float().mean().item())
            acc["base_spd"].append(robot.data.root_lin_vel_b[:, :2].norm(dim=-1).mean().item())
            acc["tgt_drift"].append(
                (ee_cmd.pose_command_b[:, :3] - (default_pos + torch.tensor(ee_offset, device=DEV))).abs().max().item()
            )
    return {k: (sum(v) / len(v) if v else float("nan")) for k, v in acc.items()}


# -- mode A: chassis moves, EE target held -------------------------------------------
HOLD = (0.06, -0.05, 0.04)
mode_a = run(
    [(0.0, 0.0, HOLD), (0.25, 0.0, HOLD), (0.25, 0.25, HOLD), (0.0, 0.25, HOLD),
     (-0.25, 0.0, HOLD), (0.0, -0.25, HOLD), (0.0, 0.0, HOLD)],
    decoupled=False,
    label="A",
)

# -- mode B: chassis still, EE target moves -------------------------------------------
mode_b = run(
    [(0.0, 0.0, o) for o in [(0.0, 0.0, 0.0), (0.10, 0.0, 0.0), (0.0, 0.12, 0.0),
                             (0.0, 0.0, 0.09), (-0.10, -0.12, 0.0), (0.0, 0.0, 0.0)]],
    decoupled=False,
    label="B",
)

# -- mode C: both move ----------------------------------------------------------------
BOTH = [(0.25, 0.0, (0.10, 0.0, 0.0)), (0.25, 0.25, (0.0, 0.12, 0.0)),
        (0.0, 0.25, (0.0, 0.0, 0.09)), (-0.25, 0.0, (-0.10, 0.0, 0.0)),
        (0.0, -0.25, (0.0, -0.12, 0.0)), (0.0, 0.0, (0.0, 0.0, 0.0))]
mode_c = run(BOTH, decoupled=False, label="C")
decoup = run(BOTH, decoupled=True, label="D")

# -- mode A again, decoupled: does the policy spend any velocity authority on the arm? --
# The joint scheme's whole justification is that it can trade a little chassis tracking
# for arm performance. If it does, its EE error under chassis motion should be lower
# than the decoupled controller's at a slightly higher velocity error.
mode_a_dec = run(
    [(0.0, 0.0, HOLD), (0.25, 0.0, HOLD), (0.25, 0.25, HOLD), (0.0, 0.25, HOLD),
     (-0.25, 0.0, HOLD), (0.0, -0.25, HOLD), (0.0, 0.0, HOLD)],
    decoupled=True,
    label="A-decoupled",
)

hdr = f"{'mode':<26}{'线速度误差':>12}{'角速度误差':>12}{'EE 位置':>10}{'EE 姿态':>10}" \
      f"{'成功率':>9}{'底盘速度':>10}{'目标漂移':>10}{'动作平滑':>10}"
print(hdr)
print("-" * len(hdr))
for name, m in (("A 只控制底盘", mode_a), ("B 只控制末端", mode_b),
                ("C 同时控制", mode_c), ("C 的解耦基线", decoup),
                ("A 的解耦基线", mode_a_dec)):
    print(f"{name:<26}{m['lin_err']:>12.4f}{m['ang_err']:>12.4f}{m['pos_err']:>10.4f}"
          f"{m['ori_err']:>10.4f}{m['success']:>9.2%}{m['base_spd']:>10.4f}"
          f"{m['tgt_drift']:>10.2e}{m['smooth']:>10.4f}")

print("\n判读:")
print(f"  A 的目标漂移必须为 0 (底盘动而机身系目标不变): {mode_a['tgt_drift']:.2e}")
print(f"  B 的底盘速度必须接近 0 (只动末端时底盘该静止): {mode_b['base_spd']:.4f} m/s")
print(f"  解耦基线 vs C: 线速度误差 {decoup['lin_err']:.4f} vs {mode_c['lin_err']:.4f}, "
      f"EE 位置 {decoup['pos_err']:.4f} vs {mode_c['pos_err']:.4f}")
print(f"  解耦基线 vs A: 线速度误差 {mode_a_dec['lin_err']:.4f} vs {mode_a['lin_err']:.4f}, "
      f"EE 位置 {mode_a_dec['pos_err']:.4f} vs {mode_a['pos_err']:.4f}, "
      f"成功率 {mode_a_dec['success']:.2%} vs {mode_a['success']:.2%}")

env.close()
app.close()
