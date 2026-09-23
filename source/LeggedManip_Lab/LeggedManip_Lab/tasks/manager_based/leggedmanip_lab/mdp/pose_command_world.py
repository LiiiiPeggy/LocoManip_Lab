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

"""End-effector pose command whose target is fixed in the world.

Why this exists next to ``UniformPoseWBCCommand``: that one defines its XY target in
the robot's body frame. Under the joint base+arm scheme that is unusable, because a
body-frame target does not move when the chassis drives -- so a target that "requires
driving the base to reach" could never exist and the base would have nothing to learn.
See docs/plans/plan.md section 4.2.

The three frames, kept deliberately distinct:

* ``target_pos_w`` / ``target_quat_w`` -- the task target, frozen in the world when the
  command is resampled and never moved afterwards.
* ``command`` (what the policy observes) -- the same target expressed relative to the
  chassis, recomputed every step. Giving the policy a relative quantity keeps it from
  having to learn absolute world coordinates that depend on ``env_origin``.
* the reward -- compares the end-effector in the *world* frame against the frozen
  target. See ``tcp.position_command_world_error_exp``.

The base-relative command and the world-frame reward agree by construction: the reward
reconstructs the world target from the command and the current base pose, which is
exactly how the command was derived in the first place.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_mul,
    quat_unique,
)

from .tcp import tcp_pose_b, tcp_pose_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .cfg.command_cfg import UniformPoseWorldCommandCfg


class UniformPoseWorldCommand(CommandTerm):
    """Samples a target pose, freezes it in the world, and reports it relative to the chassis."""

    cfg: UniformPoseWorldCommandCfg

    def __init__(self, cfg: UniformPoseWorldCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        self.base_idx = self.robot.find_bodies(cfg.link_name)[0][0]

        # the frozen world target
        self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.target_quat_w[:, 0] = 1.0

        # what the policy sees: the same target expressed relative to the chassis
        self.pose_command = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command[:, 3] = 1.0

        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        # distance from the chassis to the target: the quantity the base can actually reduce
        self.metrics["distance_to_target"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformPoseWorldCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Target pose relative to the chassis, ``(x, y, z, qw, qx, qy, qz)``."""
        return self.pose_command

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        num = len(env_ids)
        base_pos_w = self.robot.data.body_pos_w[env_ids, self.base_idx]
        base_quat_w = self.robot.data.body_quat_w[env_ids, self.base_idx]

        # -- sample the offset in the chassis frame -----------------------------
        x_lim = torch.tensor(self.cfg.ranges.pos_x, device=self.device)
        y_lim = torch.tensor(self.cfg.ranges.pos_y, device=self.device)
        z_lim = torch.tensor(self.cfg.ranges.pos_z, device=self.device)

        rand_pos = torch.rand(num, 3, device=self.device)
        offset_b = torch.stack(
            [
                x_lim[0] + rand_pos[:, 0] * (x_lim[1] - x_lim[0]),
                y_lim[0] + rand_pos[:, 1] * (y_lim[1] - y_lim[0]),
                z_lim[0] + rand_pos[:, 2] * (z_lim[1] - z_lim[0]),
            ],
            dim=-1,
        )

        # -- freeze it in the world --------------------------------------------
        # Done relative to the chassis pose *at this instant*; from here on the target
        # stays put while the robot moves, which is what lets the base make progress.
        self.target_pos_w[env_ids] = base_pos_w + quat_apply(base_quat_w, offset_b)

        # Orientation: keep the attitude the tool already has, plus a bounded random tilt,
        # then freeze that in the world.
        #
        # The earlier version derived pitch and yaw to "point the tool at the target",
        # which quietly assumes the tool's forward axis is the chassis' +x. That holds for
        # go2_piper's Piper but not for the CR10 with an AG95 on the flange: the commanded
        # attitude ended up about 2.1 rad away from anything the arm could reach. Measured
        # on a real run, the orientation error sat at 2.11 rad for 56 iterations -- its
        # initial value -- while the policy paid -8.6 reward per step for it, dominating
        # every other term. Anchoring on the tool's own attitude needs no hand-measured
        # tool axis and is reachable by construction.
        _, tcp_quat_w = tcp_pose_w(self.robot, self.body_idx)
        tcp_quat_b = quat_mul(quat_conjugate(base_quat_w), tcp_quat_w[env_ids])

        roll_lim = torch.tensor(self.cfg.limit_ranges.roll, device=self.device)
        pitch_lim = torch.tensor(self.cfg.limit_ranges.pitch, device=self.device)
        yaw_lim = torch.tensor(self.cfg.limit_ranges.yaw, device=self.device)
        rand_euler = torch.rand(num, 3, device=self.device)

        euler = torch.stack(
            [
                roll_lim[0] + rand_euler[:, 0] * (roll_lim[1] - roll_lim[0]),
                pitch_lim[0] + rand_euler[:, 1] * (pitch_lim[1] - pitch_lim[0]),
                yaw_lim[0] + rand_euler[:, 2] * (yaw_lim[1] - yaw_lim[0]),
            ],
            dim=-1,
        )
        euler = torch.clamp(euler, -3.14 / 4, 3.14 / 3)
        offset_quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        self.target_quat_w[env_ids] = quat_mul(base_quat_w, quat_mul(tcp_quat_b, offset_quat))

        # the command tensor must be valid immediately, not only after the next update
        self._update_command()

    def _update_command(self):
        """Refresh the chassis-relative view of the frozen world target."""
        base_pos_w = self.robot.data.body_pos_w[:, self.base_idx]
        base_quat_w = self.robot.data.body_quat_w[:, self.base_idx]

        self.pose_command[:, :3] = quat_apply(quat_conjugate(base_quat_w), self.target_pos_w - base_pos_w)
        quat_b = quat_mul(quat_conjugate(base_quat_w), self.target_quat_w)
        self.pose_command[:, 3:] = quat_unique(quat_b) if self.cfg.make_quat_unique else quat_b

    def _update_metrics(self):
        """Errors are measured in the world frame, matching the target definition."""
        tcp_pos_w, tcp_quat_w = tcp_pose_w(self.robot, self.body_idx)

        self.metrics["position_error"] = torch.norm(tcp_pos_w - self.target_pos_w, dim=-1)
        self.metrics["orientation_error"] = quat_error_magnitude(tcp_quat_w, self.target_quat_w)
        self.metrics["distance_to_target"] = torch.norm(self.pose_command[:, :3], dim=-1)

    """
    Debug visualization
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # the target is already in the world frame, so it can be drawn directly
        self.goal_pose_visualizer.visualize(self.target_pos_w, self.target_quat_w)
        tcp_pos_w, tcp_quat_w = tcp_pose_w(self.robot, self.body_idx)
        self.current_pose_visualizer.visualize(tcp_pos_w, tcp_quat_w)
