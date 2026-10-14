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

"""Drive the trained teleoperation policy from the keyboard, in Isaac Lab.

The keyboard controller is shared with the MuJoCo deployment (`keyboard_teleop.py` lives in
the deploy directory and imports nothing framework-specific), so "the MuJoCo keyboard
behaves like the Isaac Lab one" is a property of there being one implementation rather than
two that are kept in step by hand. What this script adds is the Isaac Lab half of the
evidence:

* it asserts numerically that the controller's quaternion helpers agree with
  ``isaaclab.utils.math`` -- they are reimplemented there because that module must stay
  importable without Isaac Sim, and a reimplementation that drifts would rotate every
  commanded attitude by a constant nobody would notice
* it asserts that the pose the controller builds is the pose the trained command term would
  have sampled: an attitude offset pre-multiplied onto the measured default
* it runs the policy against a scripted key sequence (``--selftest``) so the whole path is
  verifiable without a human at the keyboard, and can then be re-run interactively

Usage:
    python scripts/ranger_cr10/play_teleop_keyboard.py --policy <policy.pt> --selftest --headless
    python scripts/ranger_cr10/play_teleop_keyboard.py --policy <policy.pt>
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description="Keyboard teleoperation of the trained teleop policy.")
parser.add_argument("--policy", required=True, help="TorchScript policy from scripts/rsl_rl/play.py")
parser.add_argument("--num_envs", type=int, default=16, help="more than a handful is unreadable")
parser.add_argument("--selftest", action="store_true", help="run a scripted key sequence instead of a listener")
parser.add_argument("--headless", action="store_true")
args_cli, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=args_cli.headless).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as iu  # noqa: E402
import LeggedManip_Lab.tasks  # noqa: E402,F401
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.teleop_env_cfg import (  # noqa: E402
    RangerCr10TeleopEnvCfg,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.pose_command_teleop import (  # noqa: E402
    TCP_DEFAULT_POS_B,
    TCP_DEFAULT_QUAT_B,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mujoco/deploy/deploy_mujoco/ranger_cr10"))
from keyboard_teleop import (  # noqa: E402
    HELP_TEXT,
    KeyboardTeleopController,
    quat_from_euler_xyz,
    quat_mul,
)

DT = 0.1  # the policy rate

# ---------------------------------------------------------------- math agreement ------
print("\n== 四元数工具函数 vs Isaac Lab ==")
worst_euler, worst_mul = 0.0, 0.0
for r, p, y in [(0.3, -0.7, 1.1), (-1.9, 0.4, -2.6), (0.0, 0.0, 0.0), (3.0, -1.5, 0.2)]:
    mine = torch.tensor([quat_from_euler_xyz(r, p, y)], dtype=torch.float32)
    theirs = iu.quat_from_euler_xyz(torch.tensor([r]), torch.tensor([p]), torch.tensor([y]))
    worst_euler = max(worst_euler, (mine - theirs).abs().max().item())
a = torch.tensor([quat_from_euler_xyz(0.3, -0.7, 1.1)], dtype=torch.float32)
b = torch.tensor([quat_from_euler_xyz(-1.9, 0.4, -2.6)], dtype=torch.float32)
worst_mul = (torch.tensor([quat_mul(tuple(a[0]), tuple(b[0]))]) - iu.quat_mul(a, b)).abs().max().item()
print(f"  quat_from_euler_xyz 最大逐分量差 {worst_euler:.3e}")
print(f"  quat_mul            最大逐分量差 {worst_mul:.3e}")
assert worst_euler < 1e-6 and worst_mul < 1e-6, "the reimplemented quaternion helpers have drifted"

cfg = RangerCr10TeleopEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
cfg.commands.ee_pose.debug_vis = not args_cli.headless
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
print(f"\npolicy: {args_cli.policy}")

ctrl = KeyboardTeleopController(TCP_DEFAULT_POS_B, TCP_DEFAULT_QUAT_B, start_listener=not args_cli.selftest)

# ------------------------------------------------- command-path agreement --------------
# The controller composes the target itself, so the path it feeds the policy has to be the
# path the policy was trained on: position is default + offset in the chassis frame, and
# attitude is the offset *pre-multiplied* onto the default. Rebuild both with Isaac Lab's
# own helpers and compare -- with a non-zero offset, since an identity offset would pass
# even if the composition order were wrong.
ctrl.press("i")
ctrl.update(DT)
ctrl.release("i")
ctrl.press("2")
ctrl.update(DT)
ctrl.release("2")
pos_off, rpy_off = ctrl.pose_offset
composed = torch.tensor(ctrl.ee_pose_command_b, device=u.device)
expect_pos = torch.tensor(TCP_DEFAULT_POS_B, device=u.device) + torch.tensor(pos_off, device=u.device)
# 1-element tensors, not scalars: with 0-dim inputs Isaac Lab returns shape (4,) and
# quat_mul wants (N, 4)
offset_quat = iu.quat_from_euler_xyz(*[torch.tensor([v], device=u.device) for v in rpy_off])
expect_quat = iu.quat_mul(
    offset_quat, torch.tensor(TCP_DEFAULT_QUAT_B, device=u.device).unsqueeze(0)
)[0]
print("\n== 控制器组出的目标 vs 训练时的采样语义 ==")
print(f"  位置偏移 {tuple(round(v, 4) for v in pos_off)} m, 姿态偏移 "
      f"{tuple(round(math.degrees(v), 2) for v in rpy_off)} deg")
print(f"  位置差 {(composed[:3] - expect_pos).abs().max().item():.3e}")
print(f"  姿态差 {(composed[3:] - expect_quat).abs().max().item():.3e}")
assert (composed[:3] - expect_pos).abs().max().item() < 1e-6
assert (composed[3:] - expect_quat).abs().max().item() < 1e-6

# The agreement block above drove the controller to a non-zero offset. Reset it, or the
# run below silently starts with a tool target nobody asked for -- which is exactly what
# the first version of this script did.
ctrl.reset()
assert ctrl.pose_offset == ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "the run must start from neutral"
assert ctrl.velocity_command == (0.0, 0.0)

# ---------------------------------------------------------------- run loop -------------
SCRIPT = [  # (seconds, keys held during that window), only used by --selftest
    (1.5, "w"), (1.0, "wq"), (1.0, "wi"), (1.0, "i"), (1.0, "i2"),
    (1.0, "s"), (1.0, "e"), (1.0, ""), (1.0, "r"), (1.0, ""),
]

if not args_cli.headless:
    print(HELP_TEXT)
    ctrl.start_display(fps=5.0)

obs, _ = env.reset()
prev_keys: set[str] = set()
step = 0
script_i, script_t = 0, 0.0
t0 = time.time()
print(f"\n{'step':>6}{'keys':>8}{'vx_cmd':>9}{'wz_cmd':>9}{'vx':>9}{'wz':>9}"
      f"{'ee_x':>9}{'ee_y':>9}{'ee_z':>9}{'pos_err':>9}{'ori_err':>9}")

while ctrl.running and step < (100000 if not args_cli.selftest else 100):
    if args_cli.selftest:
        keys = set(SCRIPT[script_i][1])
        script_t += DT
        if script_t >= SCRIPT[script_i][0]:
            script_t, script_i = 0.0, min(script_i + 1, len(SCRIPT) - 1)
    else:
        keys = set()
        with ctrl._lock:  # noqa: SLF001 - the display and listener share this set
            keys = set(ctrl._pressed)

    for ch in prev_keys - keys:
        ctrl.release(ch)
    for ch in keys - prev_keys:
        ctrl.press(ch)
    prev_keys = keys
    ctrl.update(DT)

    vx_cmd, wz_cmd = ctrl.velocity_command
    vel_cmd.is_standing_env[:] = False
    vel_cmd.vel_command_b[:, 0] = vx_cmd
    vel_cmd.vel_command_b[:, 1] = 0.0
    vel_cmd.vel_command_b[:, 2] = wz_cmd
    ee_cmd.pose_command_b[:] = torch.tensor(ctrl.ee_pose_command_b, device=u.device)

    with torch.inference_mode():
        action = net(obs["policy"])
    obs, _, _, _, _ = env.step(action)

    if step % 5 == 0:
        pos, _ = ctrl.pose_offset
        print(f"{step:>6}{''.join(sorted(keys)) or '-':>8}{vx_cmd:>+9.3f}{wz_cmd:>+9.3f}"
              f"{robot.data.root_lin_vel_b[:, 0].mean().item():>+9.3f}"
              f"{robot.data.root_ang_vel_b[:, 2].mean().item():>+9.3f}"
              f"{pos[0]:>+9.3f}{pos[1]:>+9.3f}{pos[2]:>+9.3f}"
              f"{ee_cmd.metrics['position_error'].mean().item():>9.3f}"
              f"{ee_cmd.metrics['orientation_error'].mean().item():>9.3f}")
    step += 1
    if not args_cli.headless:
        time.sleep(max(0.0, DT - (time.time() - t0 - step * DT)))

print(f"\n{step} steps, {time.time() - t0:.1f} s 实际耗时;键盘控制器退出: {not ctrl.running}")
ctrl.close()
env.close()
app.close()
