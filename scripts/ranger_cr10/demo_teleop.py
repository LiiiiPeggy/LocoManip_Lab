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

"""Record a demo video of a trained teleoperation policy.

The commands go through the *keyboard controller's* input path -- the same
``KeyboardTeleopController`` that ``play_teleop_keyboard.py`` runs interactively and that the
MuJoCo deployment uses -- driven by a scripted, labelled operator sequence instead of a human.
What is on screen is therefore what an operator would get: chassis velocity that spring-returns
to zero when the key is released, attitude keys that step in 5 deg increments, position keys
that ramp at 0.10 m/s.

The command term's own resampling is disabled, so the only source of commands is the script.
The camera follows the chassis, and every phase prints the command next to what the robot did.

Usage:
    python scripts/ranger_cr10/demo_teleop.py --policy <run>/exported/policy.pt \
        --out outputs/teleop_demo/teleop_demo.mp4 --headless
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description="Record a scripted teleoperation demo video.")
parser.add_argument("--policy", required=True, help="TorchScript policy from scripts/rsl_rl/play.py")
parser.add_argument("--out", default="outputs/teleop_demo/teleop_demo.mp4")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--fps", type=int, default=10, help="matches the 10 Hz policy rate")
parser.add_argument("--headless", action="store_true")
parser.add_argument(
    "--camera_eye",
    type=float,
    nargs=3,
    default=(2.60, -2.40, 1.70),
    help="camera offset from the chassis origin, metres (world-aligned, so the view does not spin); "
    "~3.7 m back, which is what the vertical field of view needs to hold a 1.6 m tall arm",
)
parser.add_argument("--camera_target", type=float, nargs=3, default=(0.10, 0.0, 0.45))
args_cli = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=True)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import LeggedManip_Lab.tasks  # noqa: E402,F401
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.teleop_env_cfg import (  # noqa: E402
    RangerCr10TeleopEnvCfg,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.pose_command_teleop import (  # noqa: E402
    TCP_DEFAULT_POS_B,
    TCP_DEFAULT_QUAT_B,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mujoco/deploy/deploy_mujoco/ranger_cr10"))
from keyboard_teleop import KeyboardTeleopController  # noqa: E402

# the labels are Chinese; a Latin-only font would draw boxes
_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)
FONT = next(
    (ImageFont.truetype(p, 24) for p in _FONT_CANDIDATES if Path(p).exists()),
    ImageFont.load_default(),
)


def taps(key: str, n: int, gap: float = 0.3) -> list[tuple[str, float]]:
    """``n`` discrete key presses, each held for exactly one control step."""
    out: list[tuple[str, float]] = []
    for i in range(n):
        out.append((key, 0.1))
        if i < n - 1:
            out.append(("", gap))
    return out


def reset() -> list[tuple[str, float]]:
    """Press R, then wait for the walk-back to finish and settle."""
    return taps("r", 1) + [("", 2.5)]


# Each group asks one question, and the chassis is still for the arm groups so the two channels
# do not confound. Offsets stay inside the trained box -- keyboard_teleop.py explains why going
# past it is worse than useless: +-22.9 deg roll/pitch, +-28.6 deg yaw, +-0.12/0.15/0.10 m.
GROUPS: list[tuple[str, list[tuple[str, float]]]] = [
    ("初始保持", [("", 1.5)]),
    ("前进 W", [("w", 3.0)]),
    ("左转 Q", [("q", 2.5)]),
    ("前进+左转 W+Q", [("wq", 3.0)]),
    ("停车", [("", 1.5)]),
    ("pitch +15° (4 按三次)", taps("4", 3) + [("", 2.5)]),
    ("复位 R → 回中性", reset()),
    ("roll +15° (2 按三次)", taps("2", 3) + [("", 2.5)]),
    ("复位 R → 回中性", reset()),
    ("yaw +20° (6 按四次)", taps("6", 4) + [("", 2.5)]),
    ("复位 R → 回中性", reset()),
    ("工具前伸 x +0.09 m (I 按住 0.9 s)", [("i", 0.9), ("", 2.5)]),
    ("复位 R → 回中性", reset()),
    ("前进 + 保持 pitch +15°", taps("4", 3) + [("w", 3.0)]),
    ("停车", [("", 1.5)]),
]


def ori_error_deg(env) -> float:
    """Angle of the command->TCP rotation, in degrees.

    ``metrics['orientation_error']`` is the norm of the rotation vector, which is the axis-angle
    angle only to first order; ``2*asin(|v|/2)`` is exact.
    """
    n = env.unwrapped.command_manager.get_term("ee_pose").metrics["orientation_error"].mean().item()
    return math.degrees(2.0 * math.asin(min(1.0, n / 2.0)))


def main() -> None:
    cfg = RangerCr10TeleopEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.episode_length_s = 1e6  # the script owns when things change; no episode reset mid-demo
    cfg.commands.ee_pose.debug_vis = True
    cfg.commands.ee_pose.resampling_time_range = (1e6, 1e6)
    cfg.commands.base_velocity.resampling_time_range = (1e6, 1e6)
    cfg.observations.policy.enable_corruption = False
    cfg.curriculum.lin_vel_cmd_levels = None
    cfg.curriculum.ang_vel_cmd_levels = None
    env = gym.make("RANGER-CR10-TELEOP-WBC", cfg=cfg, render_mode="rgb_array")
    u = env.unwrapped
    robot = u.scene["robot"]
    ee_cmd = u.command_manager.get_term("ee_pose")
    vel_cmd = u.command_manager.get_term("base_velocity")
    net = torch.jit.load(args_cli.policy, map_location=u.device)
    dt = u.step_dt
    print(f"policy {args_cli.policy}\n控制频率 {1.0 / dt:.1f} Hz, 相机跟随, 输出 {args_cli.out}")

    ctrl = KeyboardTeleopController(TCP_DEFAULT_POS_B, TCP_DEFAULT_QUAT_B, start_listener=False)
    obs, _ = env.reset()

    out_path = Path(args_cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=args_cli.fps, codec="libx264", quality=8, macro_block_size=None)

    prev_keys: set[str] = set()
    t = 0.0
    # running statistics for the group that is currently on screen
    acc = {"vx_cmd": [], "vx": [], "wz_cmd": [], "wz": [], "pos": [], "ori": []}
    print(
        f"\n{'t':>6}  {'命令':<24}{'vx 指令/实际':>14}{'wz 指令/实际':>14}"
        f"{'位置误差':>10}{'姿态误差':>10}"
    )
    base0 = robot.data.root_pos_w[0].clone()

    group_i = 0
    while simulation_app.is_running() and group_i < len(GROUPS):
        label, phases = GROUPS[group_i]
        for keys, secs in phases:
            for _ in range(max(1, round(secs / dt))):
                key_set = set(keys)
                for ch in prev_keys - key_set:
                    ctrl.release(ch)
                for ch in key_set - prev_keys:
                    ctrl.press(ch)
                prev_keys = key_set
                ctrl.update(dt)

                vx_cmd, wz_cmd = ctrl.velocity_command
                vel_cmd.is_standing_env[:] = False
                vel_cmd.vel_command_b[:, 0] = vx_cmd
                vel_cmd.vel_command_b[:, 1] = 0.0
                vel_cmd.vel_command_b[:, 2] = wz_cmd
                ee_cmd.pose_command_b[:] = torch.tensor(ctrl.ee_pose_command_b, device=u.device)

                with torch.inference_mode():
                    action = net(obs["policy"])
                obs, _, _, _, _ = env.step(action)

                # move the camera before rendering, otherwise every frame is one step behind
                # the robot -- visible as it drifting off-centre while it drives
                base = robot.data.root_pos_w[0]
                eye = base + torch.tensor(args_cli.camera_eye, device=u.device)
                tgt = base + torch.tensor(args_cli.camera_target, device=u.device)
                u.sim.set_camera_view(eye=eye.tolist(), target=tgt.tolist())

                img = Image.fromarray(np.asarray(env.render())[..., :3])
                draw = ImageDraw.Draw(img, "RGBA")
                pos_off, rpy_off = ctrl.pose_offset
                lines = [
                    f"{label}",
                    f"vx {vx_cmd:+.2f} -> {robot.data.root_lin_vel_b[0, 0].item():+.2f} m/s"
                    f"    wz {wz_cmd:+.2f} -> {robot.data.root_ang_vel_b[0, 2].item():+.2f} rad/s",
                    f"tool offset  xyz ({pos_off[0]:+.3f}, {pos_off[1]:+.3f}, {pos_off[2]:+.3f}) m"
                    f"  rpy ({math.degrees(rpy_off[0]):+.1f}, {math.degrees(rpy_off[1]):+.1f},"
                    f" {math.degrees(rpy_off[2]):+.1f}) deg",
                    f"error  pos {ee_cmd.metrics['position_error'].mean().item():.3f} m"
                    f"    att {ori_error_deg(env):.1f} deg",
                ]
                draw.rectangle([14, 14, 700, 24 + 32 * len(lines)], fill=(0, 0, 0, 150))
                for i, ln in enumerate(lines):
                    draw.text((26, 18 + 32 * i), ln, font=FONT, fill=(255, 255, 255, 255))
                writer.append_data(np.asarray(img))

                acc["vx_cmd"].append(vx_cmd)
                acc["wz_cmd"].append(wz_cmd)
                acc["vx"].append(robot.data.root_lin_vel_b[0, 0].item())
                acc["wz"].append(robot.data.root_ang_vel_b[0, 2].item())
                acc["pos"].append(ee_cmd.metrics["position_error"].mean().item())
                acc["ori"].append(ori_error_deg(env))
                t += dt

        # one row per group, over the whole group: a 15 deg attitude command is three 0.1 s
        # presses 0.3 s apart, and a row per press would report single-step transients
        mean = {k: float(np.mean(v)) if v else 0.0 for k, v in acc.items()}
        print(
            f"{t:>6.1f}  {label:<24}{mean['vx_cmd']:>+7.2f}/{mean['vx']:>+6.2f}"
            f"{mean['wz_cmd']:>+7.2f}/{mean['wz']:>+6.2f}{mean['pos']:>9.3f}m{mean['ori']:>9.1f}°"
        )
        acc = {k: [] for k in acc}
        group_i += 1

    travelled = robot.data.root_pos_w[0] - base0
    print(f"\n底盘位移 ({travelled[0].item():+.2f}, {travelled[1].item():+.2f}) m, 共 {t:.1f} s")
    writer.close()
    env.close()
    print(f"已写出 {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
