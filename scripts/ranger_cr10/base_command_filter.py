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

"""Optional deadband on the *chassis* channels of the teleoperation policy.

The policy emits eight numbers: `[vx, wz, six arm joints]`. The two chassis numbers go through
`0.5 * tanh(raw)` in `BaseVelocityAction` and are then resolved by the same base controller the
real AgileX driver uses; the six arm numbers are `q_default + 0.25 * raw` and are never touched
here.

Two independent, separately switchable stages:

* **input deadband** -- an operator command below the threshold is treated as exactly zero.
  The deadbanded value is what the policy *observes*, so the policy is not asked to track a
  command the chassis is not going to receive.
* **conditional output lock** -- when a channel's effective command is zero, the policy's own
  output for that channel is forced to zero. This is the part that suppresses a residual
  output bias; it is deliberately conditional, so a low *commanded* speed (which the policy
  should track) is untouched.

Hysteresis is optional and off by default: the state flips to "active" below `threshold` and
back above `threshold * (1 + hysteresis)`. With hysteresis 0 the two boundaries coincide and the
filter is a plain deadband.

Everything is per-batch, so the same code runs on 1 environment or on 4096.
"""

from __future__ import annotations

import torch


class BaseCommandFilter:
    """Deadband + conditional zero-lock on the `(vx, wz)` channels only."""

    def __init__(
        self,
        vx_deadband: float = 0.0,
        wz_deadband: float = 0.0,
        lock_output: bool = False,
        hysteresis: float = 0.0,
    ) -> None:
        self.vx_deadband = float(vx_deadband)
        self.wz_deadband = float(wz_deadband)
        self.lock_output = bool(lock_output)
        self.hysteresis = float(hysteresis)
        self._active = None  # (N, 2) bool, created lazily on the first call

    @property
    def enabled(self) -> bool:
        return self.vx_deadband > 0.0 or self.wz_deadband > 0.0

    def filter_command(self, vx: torch.Tensor, wz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the deadbanded `(vx, wz)`; zero means "this channel is off"."""
        cmd = torch.stack([vx, wz], dim=-1)
        if not self.enabled:
            return vx, wz
        thr = torch.tensor([self.vx_deadband, self.wz_deadband], device=cmd.device, dtype=cmd.dtype)
        mag = cmd.abs()
        if self._active is None or self._active.shape != cmd.shape or self._active.device != cmd.device:
            self._active = mag < thr
        else:
            # switch on strictly below `thr`, release at or above `thr * (1 + hysteresis)`.
            # With hysteresis 0 the two boundaries coincide, so a command sitting exactly on
            # the threshold is *not* filtered and the state does not stick to a constant input.
            on = mag < thr
            off = mag >= thr * (1.0 + self.hysteresis)
            self._active = torch.where(self._active, ~off, on)
        out = torch.where(self._active, torch.zeros_like(cmd), cmd)
        return out[..., 0], out[..., 1]

    def filter_action(self, action: torch.Tensor, vx_eff: torch.Tensor, wz_eff: torch.Tensor) -> torch.Tensor:
        """Zero the chassis channels of the raw action where the effective command is zero.

        The arm channels (`action[:, 2:]`) are returned untouched -- deliberately, and the
        callers rely on it.
        """
        if not (self.lock_output and self.enabled):
            return action
        out = action.clone()
        out[:, 0] = torch.where(vx_eff.abs() < 1e-9, torch.zeros_like(out[:, 0]), out[:, 0])
        out[:, 1] = torch.where(wz_eff.abs() < 1e-9, torch.zeros_like(out[:, 1]), out[:, 1])
        return out
