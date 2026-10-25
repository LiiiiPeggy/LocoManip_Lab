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

"""Keyboard teleoperation for the RangerBox-CR10: keys in, the two commands out.

Deliberately free of Isaac Sim, MuJoCo and numpy so the *same* object drives the policy in
both places -- the deployment acceptance criterion is "the MuJoCo keyboard behaves like the
Isaac Lab one", and one implementation is the only way to be sure of that. The only
optional dependency is ``pynput`` for the real listener; the state machine itself is driven
by ``press``/``release`` and can be exercised without a keyboard at all, which is what
``scripts/ranger_cr10/check_keyboard_teleop.py`` does.

What it produces, and why not the same thing GO2-PIPER's controller produces
--------------------------------------------------------------------------
``KeyboardController`` in the parent directory holds both ramps at their last value when
the keys are released. For an end-effector target that is right -- the task in
``docs/plans/teleop_plan.md`` says "releasing the arm keys leaves the target where it was".
For the chassis it is not: a velocity that stays at 0.4 m/s after the operator lets go is a
robot that keeps driving. Here the velocity **spring-returns to zero** and the pose target
holds, which is what the task brief asks for and is the one behavioural difference from the
reference implementation worth naming.

The other difference is the end-effector attitude. GO2-PIPER's controller computes pitch
and yaw compensation from the mount point so that the tool points at the target. That is a
property of its arm and its task; the trained policy here expects an attitude *offset about
the chassis axes from the tool's own default attitude*, which is what
``UniformTCPBodyPoseCommand`` samples. Reusing the mount-point heuristic would command
poses the policy never saw.

Reset semantics
---------------
``R`` zeroes the velocity **immediately** and walks the end-effector target back to neutral
at the ordinary key rate instead of snapping it. Snapping would be a step change in the
commanded pose, and "the reset must not cause a dangerous jump" is a requirement; ramping
costs a second or two and moves the arm no faster than the operator could have.

Key map
-------
    W / S      chassis forward / backward          I / K   tool X + / -
    Q / E      turn left / right                   J / L   tool Y + / -
    A / D      reserved (no vy on this chassis)    U / O   tool Z + / -
    1 / 2      roll  - / +                         3 / 4   pitch - / +
    5 / 6      yaw   - / +                         R       reset    ESC  quit
"""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# key map
# ---------------------------------------------------------------------------

# (positive key, negative key, limit field) per chassis axis, in (vx, wz) order
VELOCITY_KEYS = (("w", "s", "vx_max"), ("q", "e", "wz_max"))
RESERVED_KEYS = {"a", "d"}

POSITION_KEYS = {
    "i": (+1, 0), "k": (-1, 0),   # x
    "j": (+1, 1), "l": (-1, 1),   # y
    "u": (+1, 2), "o": (-1, 2),   # z
}

# discrete attitude steps, applied once per press: char -> (axis, sign)
ORIENTATION_KEYS = {
    "1": (0, -1), "2": (0, +1),   # roll
    "3": (1, -1), "4": (1, +1),   # pitch
    "5": (2, -1), "6": (2, +1),   # yaw
}

HOME_KEY = "r"

HELP_TEXT = """
========================================
  RangerBox-CR10 keyboard teleoperation
========================================
  [Chassis]                      [Tool position]
    W / S   forward / backward     I / K   X + / -
    Q / E   turn left / right      J / L   Y + / -
    A / D   reserved (no vy)       U / O   Z + / -

  [Tool orientation]             [Other]
    1 / 2   roll  - / +            R       reset (velocity to zero,
    3 / 4   pitch - / +                    target walks back to neutral)
    5 / 6   yaw   - / +            ESC     quit

  Releasing the chassis keys returns the velocity to zero.
  Releasing the tool keys holds the target where it is.
========================================
"""


def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """(w, x, y, z), matching ``isaaclab.utils.math.quat_from_euler_xyz``.

    Reimplemented rather than imported because this module must stay importable without
    Isaac Sim. ``check_keyboard_teleop.py`` compares it against the Isaac Lab version
    numerically, so the duplication cannot drift unnoticed.
    """
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def quat_mul(a, b):
    """Hamilton product, ``(w, x, y, z)``, matching ``isaaclab.utils.math.quat_mul``."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


@dataclass
class KeyboardTeleopConfig:
    """Rates and limits. Defaults match the trained policy's command ranges."""

    vx_max: float = 0.5
    wz_max: float = 0.5
    vel_rate: float = 0.5
    """units/s while a chassis key is held"""
    vel_return_rate: float = 1.0
    """units/s decay toward zero once no chassis key is held"""

    pos_rate: float = 0.10
    """m/s while a position key is held"""

    pos_limits: tuple = ((-0.12, 0.12), (-0.15, 0.15), (-0.12, 0.10))
    """per-axis offset limits, metres.

    These are the trained sampling box, not the model's limit ranges. Driving the tool past
    what the policy was trained on makes the arm lag visibly rather than fail loudly --
    measured: at a 0.30 m offset the end-effector error climbed to 0.28 m. Widen these only
    after widening `EE_OFFSET_RANGES` in teleop_env_cfg.py and retraining, which is the open
    item in docs/plans/teleop_plan.md. The limits are enforced by clamping, so the operator
    simply stops at the edge of what works.
    """

    ori_step_deg: float = 5.0
    """degrees per discrete attitude press"""

    ori_home_rate_deg: float = 30.0
    """degrees/s for the walk-back to neutral, which has no held key to pace it"""

    ori_limits_deg: tuple = ((-22.9, 22.9), (-22.9, 22.9), (-28.6, 28.6))
    """per-axis offset limits, degrees (low, high) -- the trained box, same reasoning as pos_limits.

    Signed pairs, not magnitudes. Written as ``((22.9, 22.9), ...)`` first, which read as a
    ``(low, high)`` box collapses to the single point +22.9 deg: every attitude key then jumped
    to the positive limit on the first press, and the minus keys did the same. The training box
    (``EE_OFFSET_RANGES``: roll/pitch +-0.40 rad = +-22.9 deg, yaw +-0.50 rad = +-28.6 deg) is
    symmetric, which is what these numbers are.
    """

    home_rate_scale: float = 1.0
    """the reset walk-back uses the ordinary rates times this"""


class KeyboardTeleopController:
    """Turns key presses into ``(vx, wz)`` and a chassis-relative tool target.

    ``default_pos`` and ``default_quat`` are the tool's pose at the arm's default
    configuration, relative to ``base_link`` -- the neutral the offsets are measured from.
    They are arguments rather than constants here because the numbers themselves belong to
    the task definition (``pose_command_teleop.py``); a check script asserts that the ones
    the deployment passes in are the ones the policy was trained against.
    """

    def __init__(
        self,
        default_pos,
        default_quat,
        cfg: KeyboardTeleopConfig | None = None,
        start_listener: bool = True,
    ):
        self.cfg = cfg or KeyboardTeleopConfig()
        self.default_pos = tuple(float(v) for v in default_pos)
        self.default_quat = tuple(float(v) for v in default_quat)

        self._velocity = [0.0, 0.0]           # (vx, wz)
        self._pos_offset = [0.0, 0.0, 0.0]    # metres, chassis frame
        self._rpy_offset = [0.0, 0.0, 0.0]    # radians, chassis axes

        self._pressed: set[str] = set()
        self._consumed: set[str] = set()      # edge-trigger state for the discrete keys
        self._homing = False
        self._lock = threading.Lock()
        self.running = True
        self._warned_reserved = False

        self._listener = None
        self._display_thread = None
        self._last_time = time.time()
        if start_listener:
            self._start_listener()

    # -- public interface --------------------------------------------------

    def press(self, char: str) -> None:
        """Register a key as held. The real listener calls this; tests call it directly."""
        with self._lock:
            self._pressed.add(char)
            if char in RESERVED_KEYS and not self._warned_reserved:
                self._warned_reserved = True
                print(f"[keyboard] '{char}' is reserved on this chassis (no lateral motion); ignored")

    def release(self, char: str) -> None:
        with self._lock:
            self._pressed.discard(char)
            self._consumed.discard(char)

    def update(self, dt: float) -> None:
        """Advance the ramps by ``dt`` seconds. Call once per control step."""
        with self._lock:
            keys = frozenset(self._pressed)

        # the home trigger is handled first so that a tool key pressed on the same step
        # cancels the walk-back *before* it is applied -- otherwise that first press is
        # consumed with no effect and the operator has to press twice
        self._update_home_trigger(keys)
        self._update_velocity(keys, dt)
        self._update_position(keys, dt)
        self._update_orientation(keys)
        self._update_home_walk(dt)

    @property
    def velocity_command(self) -> tuple[float, float]:
        with self._lock:
            return (self._velocity[0], self._velocity[1])

    @property
    def pose_offset(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """The commanded offset from neutral: (position in m, rpy offset in rad)."""
        with self._lock:
            return (tuple(self._pos_offset), tuple(self._rpy_offset))

    @property
    def ee_pose_command_b(self) -> tuple[float, ...]:
        """The 7-vector the policy observes: ``(x, y, z, qw, qx, qy, qz)`` relative to base_link."""
        with self._lock:
            pos = tuple(self.default_pos[i] + self._pos_offset[i] for i in range(3))
            offset_quat = quat_from_euler_xyz(*self._rpy_offset)
            quat = quat_mul(offset_quat, self.default_quat)
        return pos + quat

    def start_display(self, fps: float = 10.0) -> None:
        if self._display_thread is not None:
            return
        self._display_thread = threading.Thread(target=self._display_loop, args=(fps,), daemon=True)
        self._display_thread.start()

    def reset(self) -> None:
        """Return to neutral *immediately*, for programmatic use.

        The keyboard's ``R`` deliberately ramps the target back instead (no dangerous jump);
        this is the blunt version, for a caller that is setting up rather than operating.
        """
        with self._lock:
            self._velocity = [0.0, 0.0]
            self._pos_offset = [0.0, 0.0, 0.0]
            self._rpy_offset = [0.0, 0.0, 0.0]
            self._homing = False
            self._consumed.clear()

    def close(self) -> None:
        self.running = False
        if self._listener is not None and self._listener.is_alive():
            self._listener.stop()

    # -- implementation ----------------------------------------------------

    def _start_listener(self) -> None:
        try:
            from pynput import keyboard
        except ImportError:
            print(
                "[keyboard] pynput is not installed, so no real keys will be read. "
                "The controller still works through press()/release()."
            )
            return

        def on_press(key):
            ch = getattr(key, "char", None)
            if ch:
                self.press(ch.lower())
            return self.running

        def on_release(key):
            ch = getattr(key, "char", None)
            if ch:
                self.release(ch.lower())
            elif key == keyboard.Key.esc:
                self.running = False
                return False
            return self.running

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        try:
            self._listener.start()
        except Exception as exc:  # a headless box has no display to grab
            print(f"[keyboard] listener could not start ({exc}); use press()/release()")
            self._listener = None

    def _update_velocity(self, keys: frozenset, dt: float) -> None:
        cfg = self.cfg
        for axis, (pos_ch, neg_ch, limit_name) in enumerate(VELOCITY_KEYS):
            limit = getattr(cfg, limit_name)
            pos_held = pos_ch in keys
            neg_held = neg_ch in keys
            with self._lock:
                val = self._velocity[axis]
            if self._homing:
                val = self._decay(val, cfg.vel_return_rate * dt)
            elif pos_held and not neg_held:
                val = min(limit, val + cfg.vel_rate * dt)
            elif neg_held and not pos_held:
                val = max(-limit, val - cfg.vel_rate * dt)
            else:
                # the spring return: this is the behavioural difference from the reference
                # controller, which would leave the velocity where the operator left it
                val = self._decay(val, cfg.vel_return_rate * dt)
            with self._lock:
                self._velocity[axis] = val

    def _update_position(self, keys: frozenset, dt: float) -> None:
        cfg = self.cfg
        for ch, (sign, axis) in POSITION_KEYS.items():
            if ch not in keys or self._homing:
                continue
            lo, hi = cfg.pos_limits[axis]
            with self._lock:
                val = self._pos_offset[axis]
                self._pos_offset[axis] = max(lo, min(hi, val + sign * cfg.pos_rate * dt))

    def _update_orientation(self, keys: frozenset) -> None:
        cfg = self.cfg
        step = math.radians(cfg.ori_step_deg)
        for ch, (axis, sign) in ORIENTATION_KEYS.items():
            with self._lock:
                if ch not in keys or ch in self._consumed:
                    continue
                self._consumed.add(ch)
                if self._homing:
                    continue
                lo, hi = (math.radians(v) for v in cfg.ori_limits_deg[axis])
                self._rpy_offset[axis] = max(lo, min(hi, self._rpy_offset[axis] + sign * step))

    def _update_home_trigger(self, keys: frozenset) -> None:
        """R starts a walk back to neutral and zeroes the velocity on the spot.

        The velocity goes immediately because a moving chassis is the part that has to stop
        now. The pose target is only *marked* for ramping: snapping it would be a step
        change in the commanded pose, and the requirement is that a reset causes no
        dangerous jump. Any tool key cancels the walk-back.
        """
        with self._lock:
            if HOME_KEY in keys and HOME_KEY not in self._consumed:
                self._consumed.add(HOME_KEY)
                self._homing = True
                self._velocity = [0.0, 0.0]
            if self._homing and (keys & (set(POSITION_KEYS) | set(ORIENTATION_KEYS))):
                self._homing = False

    def _update_home_walk(self, dt: float) -> None:
        """Advance the walk-back at the same rates the keys use, so it is never faster.

        The orientation gets its own rate because its keys are discrete, so there is no
        held-key rate to inherit.
        """
        if not self._homing:
            return
        cfg = self.cfg
        pos_step = cfg.pos_rate * cfg.home_rate_scale * dt
        ori_step = math.radians(cfg.ori_home_rate_deg) * cfg.home_rate_scale * dt
        with self._lock:
            for axis in range(3):
                self._pos_offset[axis] = self._decay(self._pos_offset[axis], pos_step)
                self._rpy_offset[axis] = self._decay(self._rpy_offset[axis], ori_step)

    @staticmethod
    def _decay(val: float, amount: float) -> float:
        """Move ``val`` toward zero by ``amount`` without overshooting."""
        if val > 0.0:
            return max(0.0, val - amount)
        if val < 0.0:
            return min(0.0, val + amount)
        return 0.0

    def _display_loop(self, fps: float) -> None:
        interval = 1.0 / max(fps, 1.0)
        while self.running:
            vx, wz = self.velocity_command
            pos, rpy = self.pose_offset
            lines = [
                "+----------- command -----------+",
                f"| vx  {vx:+.3f} m/s   wz {wz:+.3f} rad/s |",
                f"| pos {pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f} m    |",
                f"| rpy {math.degrees(rpy[0]):+6.1f} {math.degrees(rpy[1]):+6.1f} "
                f"{math.degrees(rpy[2]):+6.1f} deg |",
                "+------------------------------+",
            ]
            sys.stdout.write("\033[2J\033[H" + "\n".join(lines) + "\n")
            sys.stdout.flush()
            time.sleep(interval)
