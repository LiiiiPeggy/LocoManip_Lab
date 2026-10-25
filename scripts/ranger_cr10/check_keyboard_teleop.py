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

"""Stage 4.5 acceptance: does the keyboard produce the commands the brief specifies?

Pure logic, no Isaac Sim and no MuJoCo, so it runs in seconds and can be re-run after any
edit to the controller. Keys are injected through ``press``/``release`` instead of a real
listener, which is why the controller separates the two.

Each requirement in the task brief becomes one check with a printed number:

  1  chassis keys alone leave the tool target untouched
  2  tool keys alone leave the velocity command at zero
  3  both key groups work at the same time, independently
  4  releasing the chassis keys returns the velocity to zero
  5  releasing the tool keys holds the target where it was
  6  reset does not produce a jump: the velocity stops at once, the pose target walks back
  7  A and D are reserved and do nothing
  8  the ramps stop at the limits rather than running away

Usage: python scripts/ranger_cr10/check_keyboard_teleop.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mujoco/deploy/deploy_mujoco/ranger_cr10"))

from keyboard_teleop import (  # noqa: E402
    KeyboardTeleopConfig,
    KeyboardTeleopController,
    quat_from_euler_xyz,
    quat_mul,
)

# the tool's pose at the arm's default configuration, from pose_command_teleop.py
DEFAULT_POS = (0.5519, -0.0742, 0.4696)
DEFAULT_QUAT = (0.032072, -0.875128, 0.469424, -0.112978)

DT = 0.1  # the policy runs at 10 Hz
results: list[tuple[str, bool, str]] = []


def make() -> KeyboardTeleopController:
    return KeyboardTeleopController(DEFAULT_POS, DEFAULT_QUAT, start_listener=False)


def hold(ctrl: KeyboardTeleopController, chars: str, seconds: float) -> None:
    for ch in chars:
        ctrl.press(ch)
    for _ in range(int(round(seconds / DT))):
        ctrl.update(DT)


def check(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    print(f"  [{'通过' if ok else '**失败**'}] {name}: {detail}")


print(f"\n控制器: {KeyboardTeleopConfig()}\ndt = {DT} s\n")

# -- 1 ---------------------------------------------------------------------
print("1. 只按底盘键不改动末端目标")
c = make()
hold(c, "wq", 1.0)
vx, wz = c.velocity_command
pos, rpy = c.pose_offset
check("底盘键只改速度", abs(vx) > 0.2 and abs(wz) > 0.2, f"vx={vx:+.3f} wz={wz:+.3f}")
check("末端目标零变化", pos == (0.0, 0.0, 0.0) and rpy == (0.0, 0.0, 0.0), f"pos={pos} rpy={rpy}")

# -- 2 ---------------------------------------------------------------------
print("\n2. 只按末端键不改动速度命令")
c = make()
hold(c, "i", 0.5)
hold(c, "2", 0.1)
c.release("2")
hold(c, "", 0.5)
vx, wz = c.velocity_command
pos, rpy = c.pose_offset
check("速度保持为 0", vx == 0.0 and wz == 0.0, f"vx={vx:+.3f} wz={wz:+.3f}")
check(
    "末端目标确实动了",
    pos[0] > 0.01 and abs(rpy[0]) > 1e-3,
    f"x={pos[0]:+.4f} roll={math.degrees(rpy[0]):+.1f}deg",
)

# -- 3 ---------------------------------------------------------------------
print("\n3. 两组键同时操作, 互不影响")
c = make()
hold(c, "w", 0.6)
vx_before, _ = c.velocity_command
pos_before, _ = c.pose_offset
hold(c, "wi", 0.4)          # arm key joins while the chassis key stays held
vx_after, _ = c.velocity_command
pos_after, _ = c.pose_offset
check("速度继续按自己的斜坡走", vx_after > vx_before, f"{vx_before:+.3f} -> {vx_after:+.3f}")
check("末端目标同时被推进", pos_after[0] > pos_before[0], f"x {pos_before[0]:+.4f} -> {pos_after[0]:+.4f}")

# -- 4 ---------------------------------------------------------------------
print("\n4. 松开底盘键后速度归零")
c = make()
hold(c, "w", 2.0)
vx_held, _ = c.velocity_command
c.release("w")
hold(c, "", 0.6)
vx_after, wz_after = c.velocity_command
check("按住时确实有速度", abs(vx_held) > 0.3, f"vx={vx_held:+.3f}")
check("松开后归零", vx_after == 0.0 and wz_after == 0.0, f"vx={vx_after:+.4f} wz={wz_after:+.4f}")

# -- 5 ---------------------------------------------------------------------
print("\n5. 松开末端键后目标保持")
c = make()
hold(c, "i", 0.5)
pos_held, rpy_held = c.pose_offset
c.release("i")
hold(c, "", 1.0)
pos_after, rpy_after = c.pose_offset
check("目标一字未动", pos_after == pos_held and rpy_after == rpy_held,
      f"x {pos_held[0]:+.4f} -> {pos_after[0]:+.4f}")

# -- 6 ---------------------------------------------------------------------
print("\n6. 重置: 速度立即归零, 目标按斜坡走回, 不跳变")
c = make()
hold(c, "wi", 1.5)
hold(c, "2", 0.1)
c.release("2")
# realistically the operator lets go before reaching for R; a *held* tool key cancels the
# walk-back by design, and that case is checked separately below
c.release("w")
c.release("i")
hold(c, "", 0.2)
pos_before, rpy_before = c.pose_offset
vx_before, _ = c.velocity_command
c.press("r")
c.update(DT)
vx_now, wz_now = c.velocity_command
pos_1, rpy_1 = c.pose_offset
d_pos = math.dist(pos_1, pos_before)
d_ori = max(abs(rpy_1[k] - rpy_before[k]) for k in range(3))
cfg = c.cfg
check("速度在一步内归零", vx_now == 0.0 and wz_now == 0.0, f"{vx_before:+.3f} -> {vx_now:+.4f}")
check("位置每步位移 <= pos_rate*dt", d_pos <= cfg.pos_rate * DT + 1e-9,
      f"{d_pos:.5f} m <= {cfg.pos_rate * DT:.5f} m")
check("姿态每步位移 <= ori_home_rate*dt", d_ori <= math.radians(cfg.ori_home_rate_deg) * DT + 1e-9,
      f"{math.degrees(d_ori):.2f}deg <= {cfg.ori_home_rate_deg * DT:.2f}deg")
c.release("r")
steps = 0
while steps < 400 and any(abs(v) > 1e-6 for v in c.pose_offset[0] + c.pose_offset[1]):
    c.update(DT)
    steps += 1
pos_end, rpy_end = c.pose_offset
check("最终走回中性", all(abs(v) < 1e-6 for v in pos_end + rpy_end),
      f"{steps * DT:.1f} s 后 pos={tuple(round(v, 6) for v in pos_end)} rpy={tuple(round(v, 6) for v in rpy_end)}")

print("\n6b. 归位途中按住末端键 = 取消归位(有意的), 但要能解释清楚")
c = make()
hold(c, "i", 1.0)
c.release("i")
hold(c, "", 0.2)
pos_held, _ = c.pose_offset
c.press("r")
hold(c, "r", 0.5)
pos_homing, _ = c.pose_offset
c.press("i")                      # grab the arm while it is walking home
hold(c, "ri", 0.5)
pos_cancel, _ = c.pose_offset
c.release("r")
c.release("i")
check("按 R 后目标开始回落", pos_homing[0] < pos_held[0], f"x {pos_held[0]:+.4f} -> {pos_homing[0]:+.4f}")
check("按住末端键后归位停止并重新接手", pos_cancel[0] > pos_homing[0],
      f"x {pos_homing[0]:+.4f} -> {pos_cancel[0]:+.4f} (不再回落)")

# -- 7 ---------------------------------------------------------------------
print("\n7. A / D 是保留键, 不应有任何效果")
c = make()
hold(c, "ad", 1.0)
vx, wz = c.velocity_command
pos, rpy = c.pose_offset
check("速度不受影响", vx == 0.0 and wz == 0.0, f"vx={vx:+.4f} wz={wz:+.4f}")
check("末端目标不受影响", pos == (0.0, 0.0, 0.0) and rpy == (0.0, 0.0, 0.0), f"pos={pos}")

# -- 8 ---------------------------------------------------------------------
print("\n8. 斜坡在限位处停住")
c = make()
hold(c, "w", 10.0)
hold(c, "q", 10.0)
vx, wz = c.velocity_command
hold(c, "i", 20.0)
pos, _ = c.pose_offset
check("速度上限 = 配置值", abs(vx - c.cfg.vx_max) < 1e-9 and abs(wz - c.cfg.wz_max) < 1e-9,
      f"vx={vx:+.3f}/{c.cfg.vx_max} wz={wz:+.3f}/{c.cfg.wz_max}")
check("位置上限 = 配置值", abs(pos[0] - c.cfg.pos_limits[0][1]) < 1e-9,
      f"x={pos[0]:+.4f}/{c.cfg.pos_limits[0][1]}")


def tap(ctrl: KeyboardTeleopController, ch: str) -> None:
    """One discrete key press: down for exactly one control step, then up."""
    ctrl.press(ch)
    ctrl.update(DT)
    ctrl.release(ch)


# -- 8b --------------------------------------------------------------------
# Added after the demo exposed it: the limit was written as ((22.9, 22.9), ...) and read as
# (low, high), so low == high == +22.9 deg and the *first* press of any attitude key snapped
# to the positive limit -- including the minus keys. The position limits were always signed,
# which is why the checks above passed. Test the step size and both signs, not just "clamps".
print("\n8b. 姿态键: 单次按键一个 ori_step, 限位正负对称")
# the trained sampling box, from teleop_env_cfg.EE_OFFSET_RANGES (roll/pitch +-0.40 rad, yaw
# +-0.50 rad); the controller states them in degrees, rounded to 0.1 deg
BOX_RP, BOX_YAW = math.degrees(0.40), math.degrees(0.50)
MAX_RP = make().cfg.ori_limits_deg[0][1]
MAX_YAW = make().cfg.ori_limits_deg[2][1]
for axis, name, box in ((0, "roll", BOX_RP), (1, "pitch", BOX_RP), (2, "yaw", BOX_YAW)):
    lo, hi = make().cfg.ori_limits_deg[axis]
    check(f"{name} 限位 ≈ 训练采样盒 ±{box:.1f}deg 且正负对称",
          abs(hi - box) < 0.05 and abs(lo + box) < 0.05, f"[{lo:+.1f}, {hi:+.1f}]deg")
c = make()
tap(c, "2")
_, rpy = c.pose_offset
check("roll 单次 + = ori_step", abs(math.degrees(rpy[0]) - c.cfg.ori_step_deg) < 1e-9,
      f"{math.degrees(rpy[0]):+.2f}deg vs {c.cfg.ori_step_deg:.2f}deg")
c = make()
tap(c, "1")
_, rpy = c.pose_offset
check("roll 单次 - = -ori_step", abs(math.degrees(rpy[0]) + c.cfg.ori_step_deg) < 1e-9,
      f"{math.degrees(rpy[0]):+.2f}deg")

c = make()
for _ in range(20):
    tap(c, "2")
_, rpy = c.pose_offset
check("roll 正限位 = +22.9deg", abs(math.degrees(rpy[0]) - MAX_RP) < 1e-6, f"{math.degrees(rpy[0]):+.2f}deg")
c = make()
for _ in range(20):
    tap(c, "1")
_, rpy = c.pose_offset
check("roll 负限位 = -22.9deg", abs(math.degrees(rpy[0]) + MAX_RP) < 1e-6, f"{math.degrees(rpy[0]):+.2f}deg")

c = make()
for _ in range(20):
    tap(c, "4")
_, rpy = c.pose_offset
check("pitch 正限位 = +22.9deg", abs(math.degrees(rpy[1]) - MAX_RP) < 1e-6, f"{math.degrees(rpy[1]):+.2f}deg")
c = make()
for _ in range(20):
    tap(c, "6")
_, rpy = c.pose_offset
check("yaw 正限位 = +28.6deg", abs(math.degrees(rpy[2]) - MAX_YAW) < 1e-6, f"{math.degrees(rpy[2]):+.2f}deg")
c = make()
for _ in range(20):
    tap(c, "5")
_, rpy = c.pose_offset
check("yaw 负限位 = -28.6deg", abs(math.degrees(rpy[2]) + MAX_YAW) < 1e-6, f"{math.degrees(rpy[2]):+.2f}deg")

c = make()
for _ in range(20):
    tap(c, "2")
    tap(c, "4")
    tap(c, "6")
_, rpy = c.pose_offset
check("三轴独立 (roll/pitch/yaw 同时到限位)",
      abs(math.degrees(rpy[0]) - MAX_RP) < 1e-6 and abs(math.degrees(rpy[1]) - MAX_RP) < 1e-6
      and abs(math.degrees(rpy[2]) - MAX_YAW) < 1e-6,
      f"({math.degrees(rpy[0]):+.1f}, {math.degrees(rpy[1]):+.1f}, {math.degrees(rpy[2]):+.1f})deg")

# -- 9 ---------------------------------------------------------------------
print("\n9. 四元数工具函数自检 (与 Isaac Lab 版本在集成脚本里逐值对比)")
q = quat_from_euler_xyz(math.pi / 2, 0.0, 0.0)  # 90 deg roll
expect = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
check("roll 90deg", all(abs(q[i] - expect[i]) < 1e-12 for i in range(4)),
      f"{tuple(round(v, 6) for v in q)}")
identity = (1.0, 0.0, 0.0, 0.0)
check("单位元乘法", quat_mul(q, identity) == q, f"{tuple(round(v, 6) for v in quat_mul(q, identity))}")
check("逆序抵消", all(abs(v - identity[i]) < 1e-12 for i, v in enumerate(
    quat_mul(q, (q[0], -q[1], -q[2], -q[3])))), "q * q^-1 = identity")

failed = [n for n, ok, _ in results if not ok]
print(f"\n===== {len(results) - len(failed)}/{len(results)} 通过 =====")
if failed:
    for n in failed:
        print(f"  失败: {n}")
sys.exit(1 if failed else 0)
