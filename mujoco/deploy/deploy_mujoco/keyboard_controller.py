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

"""
KeyboardController — Keyboard teleoperation controller.
Simultaneously controls a legged robot (velocity commands) and an arm (end-effector pose commands).

Dependencies:
    pip install pynput scipy
"""

import math
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional

from pynput import keyboard
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LegConfig:
    """Leg velocity limits."""
    vel_x_max: float = 1.0    # forward/backward  (m/s)
    vel_y_max: float = 0.8    # left/right       (m/s)
    vel_yaw_max: float = 0.8  # rotation         (rad/s)
    accel: float = 0.5        # velocity increment slope (units/s)


@dataclass
class ArmConfig:
    """Arm end-effector position limits."""
    x_range: List[float] = field(default_factory=lambda: [0.4, 0.8])
    y_range: List[float] = field(default_factory=lambda: [-0.35, 0.35])
    z_range: List[float] = field(default_factory=lambda: [0.05, 1.0])
    mount_z_offset: float = 0.45  # end-effector mount point Z offset from base (for pitch compensation)
    accel: float = 0.5            # position increment slope (m/s)


@dataclass
class RPYConfig:
    """End-effector orientation (Euler angle) limits, in degrees."""
    roll_range: List[float] = field(default_factory=lambda: [-30.0, 30.0])
    pitch_range: List[float] = field(default_factory=lambda: [-45.0, 45.0])
    yaw_range: List[float] = field(default_factory=lambda: [0.0, 0.0])
    step: float = 5.0  # increment/decrement per key press (°)


# ---------------------------------------------------------------------------
# Key mappings (char → action)
# ---------------------------------------------------------------------------

# Leg control (continuous)
LEG_KEY_MAP = {
    "forward": "w", "backward": "s",
    "left":    "a", "right":    "d",
    "turn_l":  "q", "turn_r":   "e",
}

# Arm position control (continuous)
ARM_KEY_MAP = {
    "forward": "i", "backward": "k",
    "left":    "j", "right":    "l",
    "up":      "u", "down":     "o",
}

# Discrete step control: char → tag (one-to-one mapping, consumed set tracks chars)
DISCRETE_KEY_TAG: dict = {
    "1": "roll_neg",  "2": "roll_pos",
    "3": "pitch_neg", "4": "pitch_pos",
    "5": "yaw_neg",   "6": "yaw_pos",
    "r": "stop",
}
# Reverse mapping tag → char
_TAG_TO_CHAR = {v: k for k, v in DISCRETE_KEY_TAG.items()}

HELP_TEXT = """
══════════════════════════════════
  Keyboard Teleoperation Controller
══════════════════════════════════
  [Leg]
    W / S     Forward / Backward
    A / D     Left / Right
    Q / E     Turn Left / Turn Right

  [Arm End-Effector Position]
    I / K     X+ / X-
    J / L     Y+ / Y-
    U / O     Z+ / Z-

  [End-Effector Orientation]
    1 / 2     Roll-  / Roll+
    3 / 4     Pitch- / Pitch+
    5 / 6     Yaw-   / Yaw+

  [Other]
    R         Stop all motion
    ESC       Exit
══════════════════════════════════
"""


# ---------------------------------------------------------------------------
# Terminal display (full-screen clear)
# ---------------------------------------------------------------------------

def print_state(velocity: List[float], pos: List[float],
                rpy_deg: List[float]) -> None:
    """Clear screen and print status panel."""
    vx, vy, vyaw = velocity
    x, y, z = pos[0], pos[1], pos[2]
    roll, pitch, yaw = rpy_deg

    lines = [
        "┌─────────── Command ───────────┐",
        f"│ vel : {vx:+.3f}  {vy:+.3f}  {vyaw:+.3f}     │",
        f"│ pos : {x:.3f}  {y:+.3f}  {z:.3f}     │",
        f"│ rpy : {roll:+.1f}°  {pitch:+.1f}°  {yaw:+.1f}°     │",
        "└───────────────────────────────┘",
    ]

    sys.stdout.write("\033[2J\033[H")  # clear screen + cursor home
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

class KeyboardController:
    """
    Keyboard teleoperation controller.

    Call get_command() each frame to retrieve the latest command dict:
        {
            "velocity": [vx, vy, vyaw],              # leg velocities
            "pos":      [x, y, z, qw, qx, qy, qz],  # arm end-effector pose
        }

    Optional: call start_display(fps) to launch a background thread that
    continuously refreshes the terminal status panel.
    """

    def __init__(
        self,
        leg_cfg: LegConfig = LegConfig(),
        arm_cfg: ArmConfig = ArmConfig(),
        rpy_cfg: RPYConfig = RPYConfig(),
    ):
        self.leg_cfg    = leg_cfg
        self.arm_cfg    = arm_cfg
        self.rpy_cfg    = rpy_cfg

        # State variables
        self._velocity:  List[float] = [0.0, 0.0, 0.0]
        self._pos:       List[float] = [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
        self._rpy_deg:   List[float] = [0.0, 0.0, 0.0]

        # BUG FIX: consumed set uses chars (consistent with discard in _on_release)
        self._discrete_consumed: set = set()   # stores chars, e.g. "1" "2" ...

        self._pressed_keys: set = set()
        self._lock = threading.Lock()
        self._last_time: float = time.time()
        self.running: bool = True

        self._display_thread: Optional[threading.Thread] = None

        self._start_listener()
        print(HELP_TEXT)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_command(self) -> dict:
        """Called each frame in the main loop; returns the latest command."""
        self._update()
        with self._lock:
            return {
                "velocity": self._velocity.copy(),
                "pos":      self._pos.copy(),
            }

    def start_display(self, fps: float = 10.0) -> None:
        """
        Start a background thread that refreshes the terminal status panel at the given fps.
        Should only be called once; call close() to stop.
        """
        if self._display_thread is not None:
            return
        self._display_thread = threading.Thread(
            target=self._display_loop, args=(fps,), daemon=True
        )
        self._display_thread.start()

    def stop_motion(self) -> None:
        """Reset all motion state to initial values."""
        with self._lock:
            self._velocity = [0.0, 0.0, 0.0]
            self._pos      = [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
            self._rpy_deg  = [0.0, 0.0, 0.0]

    def close(self) -> None:
        """Release keyboard listener resources."""
        self.running = False
        if hasattr(self, "_listener") and self._listener.is_alive():
            self._listener.stop()

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _start_listener(self) -> None:
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        try:
            self._listener.start()
        except Exception as exc:
            print(f"[KeyboardController] Listener start failed: {exc}")

    def _on_press(self, key) -> None:
        try:
            char = key.char and key.char.lower()
            if char:
                with self._lock:
                    self._pressed_keys.add(char)
        except AttributeError:
            pass

    def _on_release(self, key) -> None:
        try:
            char = key.char and key.char.lower()
            if char:
                with self._lock:
                    self._pressed_keys.discard(char)
                    # BUG FIX: consumed uses chars, just discard the char directly
                    self._discrete_consumed.discard(char)
            elif key == keyboard.Key.esc:
                self.running = False
                return False  # stop listener
        except AttributeError:
            pass

    # ---- Main update ---------------------------------------------------

    def _update(self) -> None:
        now = time.time()
        dt  = now - self._last_time
        self._last_time = now

        with self._lock:
            keys = frozenset(self._pressed_keys)

        self._update_leg(keys, dt)
        self._update_arm_pos(keys, dt)
        self._update_discrete(keys)
        self._update_arm_quat()

    # ---- Continuous updates --------------------------------------------

    def _update_leg(self, keys: frozenset, dt: float) -> None:
        inc = self.leg_cfg.accel * dt
        m   = self.leg_cfg

        with self._lock:
            vx, vy, vyaw = self._velocity

        vx   = self._ramp(vx,   "w" in keys, "s" in keys, inc, m.vel_x_max)
        vy   = self._ramp(vy,   "a" in keys, "d" in keys, inc, m.vel_y_max)
        vyaw = self._ramp(vyaw, "q" in keys, "e" in keys, inc, m.vel_yaw_max)

        with self._lock:
            self._velocity = [vx, vy, vyaw]

    def _update_arm_pos(self, keys: frozenset, dt: float) -> None:
        inc = self.arm_cfg.accel * dt
        a   = self.arm_cfg

        with self._lock:
            x, y, z = self._pos[0], self._pos[1], self._pos[2]

        x = self._ramp(x, "i" in keys, "k" in keys, inc, a.x_range[1], a.x_range[0])
        y = self._ramp(y, "j" in keys, "l" in keys, inc, a.y_range[1], a.y_range[0])
        z = self._ramp(z, "u" in keys, "o" in keys, inc, a.z_range[1], a.z_range[0])

        with self._lock:
            self._pos[0], self._pos[1], self._pos[2] = x, y, z

    # ---- Discrete step updates -----------------------------------------

    def _update_discrete(self, keys: frozenset) -> None:
        """
        Each press triggers only one step (must release before re-triggering).
        BUG FIX: consumed set uses chars, consistent with discard in _on_release.
        """
        with self._lock:
            r_deg, p_deg, y_deg = self._rpy_deg[:]
            consumed = set(self._discrete_consumed)  # copy to avoid holding lock

        rc, pc, yc = self.rpy_cfg, self.rpy_cfg, self.rpy_cfg
        step = self.rpy_cfg.step

        # (char, compute_fn, apply_fn)
        pairs = [
            ("1", lambda: max(rc.roll_range[0],  r_deg - step), lambda v: self._set_rpy(roll=v)),
            ("2", lambda: min(rc.roll_range[1],  r_deg + step), lambda v: self._set_rpy(roll=v)),
            ("3", lambda: max(pc.pitch_range[0], p_deg - step), lambda v: self._set_rpy(pitch=v)),
            ("4", lambda: min(pc.pitch_range[1], p_deg + step), lambda v: self._set_rpy(pitch=v)),
            ("5", lambda: max(yc.yaw_range[0],   y_deg - step), lambda v: self._set_rpy(yaw=v)),
            ("6", lambda: min(yc.yaw_range[1],   y_deg + step), lambda v: self._set_rpy(yaw=v)),
        ]

        for char, compute_fn, apply_fn in pairs:
            # BUG FIX: query consumed by char, matching _on_release discard(char)
            if char in keys and char not in consumed:
                apply_fn(compute_fn())
                with self._lock:
                    self._discrete_consumed.add(char)  # store char

        if "r" in keys and "r" not in consumed:
            self.stop_motion()
            with self._lock:
                self._discrete_consumed.add("r")

    # ---- Quaternion computation ---------------------------------------

    def _update_arm_quat(self) -> None:
        """
        Compute end-effector quaternion from current XYZ position and target RPY angles.
        Pitch is automatically compensated by the elevation angle from the mount point.
        Yaw is automatically compensated by the azimuth angle from the base.
        """
        with self._lock:
            r_deg, p_deg, y_deg = self._rpy_deg[:]
            x, y_pos, z = self._pos[0], self._pos[1], self._pos[2]

        r = math.radians(r_deg)
        p = math.radians(p_deg)
        y = math.radians(y_deg)

        dz      = z - self.arm_cfg.mount_z_offset
        dist_xy = math.hypot(x, y_pos)
        p -= math.atan2(dz, dist_xy)   # pitch compensation
        y += math.atan2(y_pos, x)      # yaw compensation

        xyzw = Rotation.from_euler("xyz", [r, p, y]).as_quat()  # [x, y, z, w]
        qw, qx, qy, qz = xyzw[3], xyzw[0], xyzw[1], xyzw[2]

        with self._lock:
            self._pos[3:] = [qw, qx, qy, qz]

    # ---- Display thread ------------------------------------------------

    def _display_loop(self, fps: float) -> None:
        interval = 1.0 / max(fps, 1.0)
        while self.running:
            with self._lock:
                vel     = self._velocity[:]
                pos     = self._pos[:]
                rpy_deg = self._rpy_deg[:]
            print_state(vel, pos, rpy_deg)
            time.sleep(interval)

    # ---- Utility methods ----------------------------------------------

    @staticmethod
    def _ramp(val: float,
              pos_key: bool, neg_key: bool,
              inc: float,
              upper: float, lower: float = None) -> float:
        """Bidirectional ramp with upper/lower bounds. Holds current value when keys released."""
        if lower is None:
            lower = -upper
        if pos_key and not neg_key:
            return min(upper, val + inc)
        if neg_key and not pos_key:
            return max(lower, val - inc)
        return val

    def _set_rpy(self, roll: float = None, pitch: float = None, yaw: float = None) -> None:
        with self._lock:
            if roll  is not None: self._rpy_deg[0] = roll
            if pitch is not None: self._rpy_deg[1] = pitch
            if yaw   is not None: self._rpy_deg[2] = yaw

