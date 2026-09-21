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

"""Observation term functions for the legged manipulation environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_dtype,
    record_joint_names,
    record_joint_pos_offsets,
    record_joint_vel_offsets,
    record_shape,
)
from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul


# ---------------------------------------------------------------------------
# Root state observations
# ---------------------------------------------------------------------------


@generic_io_descriptor(
    units="rad/s", axes=["X", "Y", "Z"],
    observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the robot's body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


@generic_io_descriptor(
    units="m/s^2", axes=["X", "Y", "Z"],
    observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity direction projected onto the robot's body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


@generic_io_descriptor(
    units="m/s", axes=["X", "Y", "Z"],
    observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the robot's body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b

def end_effector_link0_relative_pose(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    ee_ids, _ = asset.find_bodies("end_effector")
    link0_ids, _ = asset.find_bodies("link0")

    if len(ee_ids) == 0 or len(link0_ids) == 0:
        return torch.zeros((env.num_envs, 7), device=env.device)

    ee_pos_w    = asset.data.body_pos_w[:, ee_ids[0], :]
    link0_pos_w = asset.data.body_pos_w[:, link0_ids[0], :]
    link0_quat_w = asset.data.body_quat_w[:, link0_ids[0], :]
    ee_quat_w    = asset.data.body_quat_w[:, ee_ids[0], :]

    # Relative position: transform to link0 frame (not root frame)
    rel_pos_w = ee_pos_w - link0_pos_w
    rel_pos   = quat_apply_inverse(link0_quat_w, rel_pos_w)  # fixed

    # Relative orientation: q_rel = q_link0^{-1} * q_ee
    link0_quat_conj = quat_conjugate(link0_quat_w)           # fixed
    rel_quat = quat_mul(link0_quat_conj, ee_quat_w)

    return torch.cat([rel_pos, rel_quat], dim=-1)


# ---------------------------------------------------------------------------
# Joint state observations
# ---------------------------------------------------------------------------


@generic_io_descriptor(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_dtype, record_shape, record_joint_pos_offsets],
    units="rad",
)
def joint_pos_rel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: list = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "joint.*",
    ],
) -> torch.Tensor:
    """Joint positions relative to default positions.

    Includes all leg joints plus arm joints (matched by ``joint.*``).
    Joint order is preserved as given.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(joint_names, preserve_order=True)
    return asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]


@generic_io_descriptor(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_dtype, record_shape, record_joint_vel_offsets],
    units="rad/s",
)
def joint_vel_rel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: list = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "joint.*",
    ],
) -> torch.Tensor:
    """Joint velocities relative to default velocities.

    Includes all leg joints plus arm joints (matched by ``joint.*``).
    Joint order is preserved as given.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(joint_names, preserve_order=True)
    return asset.data.joint_vel[:, joint_ids] - asset.data.default_joint_vel[:, joint_ids]


def contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    # first_contact = contact_sensor.data.net_forces_w(env.step_dt)[:, sensor_cfg.body_ids]

    contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(
        dim=-1
    )
    return contact_force

# ---------------------------------------------------------------------------
# Action / command passthrough
# ---------------------------------------------------------------------------


@generic_io_descriptor(dtype=torch.float32, observation_type="Action", on_inspect=[record_shape])
def last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The previous action applied to the environment."""
    if action_name is None:
        return env.action_manager.action
    return env.action_manager.get_term(action_name).raw_actions


@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def generated_commands(env: "ManagerBasedRLEnv", command_name: str | None = None) -> torch.Tensor:
    """The current active command from the command manager."""
    return env.command_manager.get_command(command_name)
