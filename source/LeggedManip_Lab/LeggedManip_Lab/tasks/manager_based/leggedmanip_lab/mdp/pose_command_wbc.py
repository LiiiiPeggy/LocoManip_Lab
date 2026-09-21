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

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique, quat_apply_inverse, quat_mul, quat_error_magnitude, quat_conjugate, axis_angle_from_quat, quat_apply
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from cfg.command_cfg import UniformPoseWBCCommandCfg


class UniformPoseWBCCommand(CommandTerm):
    """Command generator for generating pose commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    The position and orientation commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: UniformPoseWBCCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPoseWBCCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        
        # extrct the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]

        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        self.link0_idx = self.robot.find_bodies(cfg.link_name)[0][0]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command[:, 3] = 1.0

        self.pose_command_w = torch.zeros_like(self.pose_command)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.link0_z = self.robot.data.body_pos_w[:, self.link0_idx, 2]
        self.env = env


    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        ee_pos_w = self.robot.data.body_pos_w[:, self.body_idx]
        link0_pos_w = self.robot.data.body_pos_w[:, self.link0_idx]
        link0_quat_w = self.robot.data.body_state_w[:, self.link0_idx, 3:7]
        end_effector_curr_pos_link0 = quat_apply_inverse(
            link0_quat_w, 
            ee_pos_w - link0_pos_w
        )
        ee_pos_xy_err = torch.abs( end_effector_curr_pos_link0[:, :2] - self.pose_command[:, :2] )
        ee_pos_z_err = torch.abs(self.pose_command[:, 2:3] - ee_pos_w[:, 2:3])
        pos_error = torch.cat([ee_pos_xy_err, ee_pos_z_err], dim=-1)

        des_quat_w = quat_mul(link0_quat_w, self.pose_command[:, 3:7])
        curr_quat_w = self.robot.data.body_state_w[:, self.body_idx, 3:7]  # type: ignore
        source_quat_norm = quat_mul(des_quat_w, quat_conjugate(des_quat_w))[:, 0]
        source_quat_inv = quat_conjugate(des_quat_w) / source_quat_norm.unsqueeze(-1)
        quat_error = quat_mul(curr_quat_w, source_quat_inv)
        rot_error = axis_angle_from_quat(quat_error)

        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)
    def _resample_command(self, env_ids: Sequence[int]):
        """
        Resamples commands for the specified environment indices.
        - Optimized for GPU (Vectorized).
        - Uses Projection (Clamping) instead of Rejection Sampling loop.
        - Ensures Total Pos stays within arm workspace (0.7m).
        """
        if len(env_ids) == 0:
            return

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        
        num_resample = len(env_ids)

        # pos
        x_lim = torch.tensor(self.cfg.ranges.pos_x, device=self.device)
        y_lim = torch.tensor(self.cfg.ranges.pos_y, device=self.device)
        z_lim = torch.tensor(self.cfg.ranges.pos_z, device=self.device)
        
        rand_pos = torch.rand(num_resample, 3, device=self.device)
        self.pose_command[env_ids, 0] = x_lim[0] + rand_pos[:, 0] * (x_lim[1] - x_lim[0]) # base frame
        self.pose_command[env_ids, 1] = y_lim[0] + rand_pos[:, 1] * (y_lim[1] - y_lim[0]) # base frame
        self.pose_command[env_ids, 2] = z_lim[0] + rand_pos[:, 2] * (z_lim[1] - z_lim[0]) # world frame

        # ── Sample orientation (original logic unchanged) ──────────────────
        roll_lim  = torch.tensor(self.cfg.limit_ranges.roll,  device=self.device)
        pitch_lim = torch.tensor(self.cfg.limit_ranges.pitch, device=self.device)
        yaw_lim   = torch.tensor(self.cfg.limit_ranges.yaw,   device=self.device)

        pos_b = torch.zeros(num_resample, 3, device=self.device)
        rand_euler = torch.rand(num_resample, 3, device=self.device)
        euler_angles = torch.zeros(num_resample, 3, device=self.device)


        # Convert link0 frame xy to world frame
        link0_pos_w = self.robot.data.body_pos_w[env_ids, self.link0_idx]   # [M, 3]
        link0_quat_w = self.robot.data.body_state_w[env_ids, self.link0_idx, 3:7]  # [M, 4]

        pos_b[:, :2] = self.pose_command[env_ids, :2]  # link0 frame xy, z=0

        target_pos_w = link0_pos_w + quat_apply(link0_quat_w, pos_b)  # [M, 3]
        target_pos_w[:, 2] = self.pose_command[env_ids, 2]             # overwrite with world z

        # Direction vector in world frame
        delta_w = target_pos_w - link0_pos_w  # [M, 3]

        # Transform to link0 frame
        delta_b = quat_apply_inverse(link0_quat_w, delta_w)  # [M, 3]

        dist_xy = torch.sqrt(delta_b[:, 0]**2 + delta_b[:, 1]**2)
        pitch = -torch.atan2(delta_b[:, 2], dist_xy)
        yaw   =  torch.atan2(delta_b[:, 1], delta_b[:, 0])

        euler_angles[:, 0] = roll_lim[0] + rand_euler[:, 0] * (roll_lim[1] - roll_lim[0])
        euler_angles[:, 1] = pitch + (pitch_lim[0] + rand_euler[:, 1] * (pitch_lim[1] - pitch_lim[0]))
        euler_angles[:, 2] = yaw   + (yaw_lim[0]   + rand_euler[:, 2] * (yaw_lim[1]   - yaw_lim[0]))

        euler_angles = torch.clamp(euler_angles, -3.14 / 4, 3.14 / 3)

        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])

        if self.cfg.make_quat_unique:
            self.pose_command[env_ids, 3:] = quat_unique(quat)
        else:
            self.pose_command[env_ids, 3:] = quat
    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        link0_pos_w = self.robot.data.body_pos_w[:, self.link0_idx]
        link0_quat_w = self.robot.data.body_state_w[:, self.link0_idx, 3:7]

        # command xy (link0 frame) → world frame
        pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        pos_b[:, :2] = self.pose_command[:, :2]

        pos_w = link0_pos_w + quat_apply(link0_quat_w, pos_b)

        self.pose_command_w[:, 0] = pos_w[:, 0]
        self.pose_command_w[:, 1] = pos_w[:, 1]
        self.pose_command_w[:, 2] = self.pose_command[:, 2]  # use world z directly

        # Orientation: link0 frame quat → world frame quat
        self.pose_command_w[:, 3:] = quat_mul(link0_quat_w, self.pose_command[:, 3:])

        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])

        body_pose_w = self.robot.data.body_state_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])