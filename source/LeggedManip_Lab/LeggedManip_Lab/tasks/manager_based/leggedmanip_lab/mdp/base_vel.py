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

"""Base velocity action for the 4-wheel-steering Ranger chassis.

The policy emits ``[vx, wz]`` and this term turns that into the steering angles and
wheel speeds the articulation actually needs, reproducing the branches the AgileX
driver uses on the real robot (``ranger_messenger.cpp``). The policy therefore never
sees individual chassis joints, and the same ``(vx, wz)`` means the same thing in
simulation and on the hardware, where it goes out over ``/cmd_vel``.

Kinematics
----------
With the instantaneous centre of rotation at ``(0, R)`` in the base frame, where
``R = vx / wz``, a wheel at ``(x_i, y_i)`` must move along

    v_vec_i = (vx - wz * y_i, wz * x_i)

which is finite even when ``wz -> 0`` (straight line) -- unlike writing the radius
directly. Its direction gives the steering angle and its magnitude the wheel speed.
Note the four wheel speeds are generally *not* equal: when spinning on the spot the
left and right sides differ. Do not assume otherwise.

The steering and wheel joint sign conventions (which way is "forward" for a wheel,
which sign of steering angle turns left) depend on how the URDF authored the joint
axes. Both are exposed as config fields and are to be pinned down empirically in
stage 2 of docs/plans/plan.md, not assumed.
"""

from __future__ import annotations

import math
from dataclasses import field
from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class BaseVelocityAction(ActionTerm):
    """Maps a 2-D ``(vx, wz)`` action onto the chassis' eight actuated joints."""

    cfg: BaseVelocityActionCfg

    def __init__(self, cfg: BaseVelocityActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        self._steering_ids, self._steering_names = self.robot.find_joints(
            cfg.steering_joint_names, preserve_order=True
        )
        self._wheel_ids, self._wheel_names = self.robot.find_joints(cfg.wheel_joint_names, preserve_order=True)

        if len(self._steering_ids) != len(cfg.wheel_xy) or len(self._wheel_ids) != len(cfg.wheel_xy):
            raise ValueError(
                f"BaseVelocityAction expects {len(cfg.wheel_xy)} steering and wheel joints, got "
                f"{len(self._steering_ids)} and {len(self._wheel_ids)}."
            )

        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._command = torch.zeros(self.num_envs, 2, device=self.device)  # (vx, wz) after scaling and clipping
        self._prev_command = torch.zeros_like(self._command)

        # wheel positions in the base frame, ordered to match cfg.wheel_xy
        self._wheel_xy = torch.tensor(cfg.wheel_xy, device=self.device, dtype=torch.float32)

        # targets written to the articulation every control step
        self._steering_target = torch.zeros(self.num_envs, len(cfg.wheel_xy), device=self.device)
        self._wheel_target = torch.zeros(self.num_envs, len(cfg.wheel_xy), device=self.device)

        # metrics so a run can be checked without guessing
        self._steer_angles = self._steering_target
        self._wheel_speeds = self._wheel_target

    """
    Properties
    """

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._command

    @property
    def velocity_rate(self) -> torch.Tensor:
        """Squared change in the commanded ``(vx, wz)`` since the previous step."""
        return torch.sum(torch.square(self._command - self._prev_command), dim=1)

    """
    Operations
    """

    def process_actions(self, actions: torch.Tensor):
        self._prev_command[:] = self._command
        self._raw_actions[:] = actions
        low = torch.tensor(self.cfg.clip[0], device=self.device, dtype=torch.float32)
        high = torch.tensor(self.cfg.clip[1], device=self.device, dtype=torch.float32)
        self._command[:] = torch.clamp(actions, min=low, max=high)

    def apply_actions(self):
        vx = self._command[:, 0]
        wz = self._command[:, 1]

        # -- pick the branch the real driver would pick ------------------------
        # DUAL_ACKERMAN by default; SPINNING when the turn radius falls below the
        # driver's minimum, or when there is no forward speed at all.
        abs_wz = torch.abs(wz)
        with torch.no_grad():
            radius = torch.where(abs_wz > 1e-4, torch.abs(vx) / abs_wz.clamp(min=1e-4), torch.full_like(vx, 1e6))
            spinning = (abs_wz > 1e-4) & ((radius < self.cfg.min_turn_radius) | (torch.abs(vx) < 1e-3))

        if spinning.any():
            wz = torch.where(spinning, torch.clamp(wz, -self.cfg.max_spin_rate, self.cfg.max_spin_rate), wz)
            vx = torch.where(spinning, torch.zeros_like(vx), vx)

        # -- per-wheel velocity vectors ---------------------------------------
        # v_vec_i = (vx - wz * y_i, wz * x_i); finite for wz = 0 as well
        wx = self._wheel_xy[:, 0].unsqueeze(0)  # (1, W)
        wy = self._wheel_xy[:, 1].unsqueeze(0)
        vx_col = vx.unsqueeze(1)
        wz_col = wz.unsqueeze(1)
        v_vec_x = vx_col - wz_col * wy
        v_vec_y = wz_col * wx

        steer = torch.atan2(v_vec_y, v_vec_x)
        speed = torch.sqrt(v_vec_x**2 + v_vec_y**2)

        # Fold the angle into the nearest half turn and carry the difference in the wheel
        # speed. Without this, driving backwards commands steer = atan2(0, -v) = pi, which
        # then gets clamped to the steering limit -- so "reverse" turns into "steer hard
        # while driving forwards", measured as +0.57 m of travel where -1.2 m was asked
        # for, with 53 degrees of unwanted yaw.
        flip = torch.abs(steer) > (math.pi / 2)
        steer = torch.where(flip, steer - torch.sign(steer) * math.pi, steer)
        speed = torch.where(flip, -speed, speed)

        # The steering clamp depends on the branch: the driver only limits the angle in
        # Ackermann mode, while spinning commands each wheel to point tangentially, which
        # needs up to 90 degrees. Clamping the spin case to the Ackermann limit pushes the
        # wheels part-way radial, so they scrub instead of rolling -- measured as a spin of
        # 0.05 rad/s where 0.3 was asked for.
        steer_limit = torch.where(
            spinning.unsqueeze(1),  # (N,) -> (N, 1) so it broadcasts against the (N, W) steer
            torch.full_like(steer, self.cfg.max_spin_steer_angle),
            torch.full_like(steer, self.cfg.max_steer_angle),
        )
        steer = torch.clamp(steer, -steer_limit, steer_limit)

        self._steering_target[:] = self.cfg.steering_sign * steer
        sign = torch.as_tensor(self.cfg.wheel_sign, device=self.device, dtype=torch.float32).reshape(1, -1)
        self._wheel_target[:] = sign * speed / self.cfg.wheel_radius

        # -- write to the articulation ----------------------------------------
        # The articulation's own PD/velocity actuators hold these targets across the
        # physics substeps between policy steps, i.e. zero-order hold on (vx, wz).
        self.robot.set_joint_position_target(self._steering_target, joint_ids=self._steering_ids)
        self.robot.set_joint_velocity_target(self._wheel_target, joint_ids=self._wheel_ids)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            self._raw_actions.zero_()
            self._command.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._command[env_ids] = 0.0


def base_speed_above_threshold_l2(
    env: ManagerBasedRLEnv,
    threshold: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise chassis ground speed, but only above ``threshold``.

    A plain L2 on base speed would fight the whole point of the joint scheme -- the
    policy is supposed to drive the base. Thresholding keeps the penalty aimed at the
    failure mode it is for (charging around, or twitching) without taxing the moderate
    motion the task needs.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    speed = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=-1)
    return torch.square(torch.clamp(speed - threshold, min=0.0))


def base_velocity_rate_l2(env: ManagerBasedRLEnv, action_name: str = "base_vel") -> torch.Tensor:
    """Penalise fast changes in the commanded ``(vx, wz)``.

    The global ``action_rate_l2`` covers every action term; this one isolates the
    chassis so its smoothness can be weighted on its own.
    """
    return env.action_manager.get_term(action_name).velocity_rate


@configclass
class BaseVelocityActionCfg(ActionTermCfg):
    """Configuration for :class:`BaseVelocityAction`.

    Inherits :class:`ActionTermCfg`: the action manager validates terms with
    ``isinstance(term_cfg, ActionTermCfg)`` and rejects a plain configclass.
    """

    class_type: type[ManagerTermBase] = BaseVelocityAction

    asset_name: str = "robot"
    """Name of the articulation in the scene."""

    steering_joint_names: list[str] = field(default_factory=list)
    """The four steering joints, in the order matching ``wheel_xy``."""

    wheel_joint_names: list[str] = field(default_factory=list)
    """The four wheel spin joints, in the order matching ``wheel_xy``."""

    wheel_xy: list[tuple[float, float]] = field(default_factory=list)
    """Wheel positions ``(x, y)`` in the base frame, paired with the joint lists above."""

    wheel_radius: float = 0.1531
    """Wheel radius in metres, measured from the model."""

    min_turn_radius: float = 0.810330349
    """Below this radius the driver switches to spinning on the spot. From ``ranger_params.hpp``."""

    max_steer_angle: float = 0.6981
    """Steering clamp in DUAL_ACKERMAN mode, from ``ranger_params.hpp`` (about 40 degrees)."""

    max_spin_rate: float = 0.7853
    """Yaw-rate clamp when spinning on the spot, from ``ranger_params.hpp``."""

    max_spin_steer_angle: float = 1.5708
    """Steering clamp while spinning. The driver only limits the angle in Ackermann mode;
    spinning points each wheel tangentially, which needs up to a quarter turn."""

    steering_sign: float = 1.0
    """Sign relating the URDF steering joint axis to a left-positive steering angle."""

    wheel_sign: list[float] | float = 1.0
    """Per-wheel sign relating the URDF joint axis to forward rolling.

    The URDF is not consistent between sides: ``fr`` and ``rr`` spin about ``0 0 1``
    while ``fl`` and ``rl`` spin about ``0 0 -1``, so one scalar cannot express it.
    With a single sign the left and right wheels push in opposite directions and the
    robot pirouettes instead of driving -- measured: -144 degrees of yaw while
    commanded straight at 0.3 m/s.
    """

    clip: tuple[tuple[float, float], tuple[float, float]] = ((-0.5, -0.5), (0.5, 0.5))
    """Per-component ``(vx, wz)`` clamp, applied after scaling. Conservative by design."""
