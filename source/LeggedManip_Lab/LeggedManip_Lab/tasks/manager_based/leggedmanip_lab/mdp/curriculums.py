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

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _safe_update_range(
    target_obj, limit_obj, attr_name: str, delta: torch.Tensor, device: torch.device
):
    """Helper: safely update a config range while clamping to limit_obj thresholds."""
    if hasattr(target_obj, attr_name) and hasattr(limit_obj, attr_name):
        curr_val = torch.tensor(getattr(target_obj, attr_name), device=device)
        limits = getattr(limit_obj, attr_name)
        new_val = torch.clamp(curr_val + delta, limits[0], limits[1])
        setattr(target_obj, attr_name, new_val.tolist())


def ang_vel_cmd_levels(
    env,  # : ManagerBasedRLEnv
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z_exp",
) -> torch.Tensor:
    cfg = env.command_manager.get_term("base_velocity").cfg

    # 1. Safely get attribute (compatible with typos)
    is_curriculum_enabled = getattr(
        cfg, "curriculum_enabled", getattr(cfg, "currirulum_enabled", False)
    )
    limit_ranges = getattr(cfg, "limit_ranges", None)
    ranges = getattr(cfg, "ranges", None)

    # 2. Handle disabled curriculum
    if not is_curriculum_enabled:
        # When disabled, force ranges to limit_ranges (only needs to happen once, or rely on init)
        if ranges is not None and limit_ranges is not None:
            cfg.ranges = limit_ranges
        # Don't return -1.0; return actual limit value for log consistency
        val = (
            limit_ranges.ang_vel_z[1]
            if limit_ranges and hasattr(limit_ranges, "ang_vel_z")
            else 0.0
        )
        return torch.tensor(-1.0, device=env.device)

    # Basic safety check
    if ranges is None or limit_ranges is None or not hasattr(ranges, "ang_vel_z"):
        return torch.tensor(0.0, device=env.device)

    # 3. Performance: only compute reward and update difficulty at episode end
    if env.common_step_counter % env.max_episode_length == 0:
        reward_sums = env.reward_manager._episode_sums.get(reward_term_name)
        if reward_sums is not None:
            reward = torch.mean(reward_sums[env_ids]) / env.max_episode_length_s
            reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

            if reward > reward_term_cfg.weight * 0.8:
                delta_command = torch.tensor([-0.05, 0.05], device=env.device)
                _safe_update_range(
                    ranges, limit_ranges, "ang_vel_z", delta_command, env.device
                )

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def lin_vel_cmd_levels(
    env,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
) -> torch.Tensor:
    cfg = env.command_manager.get_term("base_velocity").cfg

    is_curriculum_enabled = getattr(
        cfg, "curriculum_enabled", getattr(cfg, "currirulum_enabled", False)
    )
    ranges = getattr(cfg, "ranges", None)
    limit_ranges = getattr(cfg, "limit_ranges", None)
    force_range = getattr(cfg, "force_range", None)
    limit_force_range = getattr(cfg, "limit_force_range", None)

    # --- Handle disabled curriculum ---
    if not is_curriculum_enabled:
        # Fix bug: original directly assigned ranges = limit_ranges (only assigned local variable)
        if ranges is not None and limit_ranges is not None:
            cfg.ranges = limit_ranges
        val = (
            limit_ranges.lin_vel_x[1]
            if limit_ranges and hasattr(limit_ranges, "lin_vel_x")
            else 0.0
        )
        return torch.tensor(-1.0, device=env.device)

    if ranges is None or not hasattr(ranges, "lin_vel_x"):
        return torch.tensor(0.0, device=env.device)

    # --- Performance: only compute at episode end ---
    if (
        env.common_step_counter > 0
        and env.common_step_counter % env.max_episode_length == 0
    ):
        reward_sums = env.reward_manager._episode_sums.get(reward_term_name)
        if reward_sums is not None:
            reward = torch.mean(reward_sums[env_ids]) / env.max_episode_length_s
            reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

            if reward > reward_term_cfg.weight * 0.8:
                # print(f"Episode ended. Reward: {reward:.4f}, Threshold: {reward_term_cfg.weight * 0.8:.4f}")

                # -- Update velocity range --
                if limit_ranges is not None:
                    delta_vel = torch.tensor([-0.05, 0.05], device=env.device)
                    _safe_update_range(
                        ranges, limit_ranges, "lin_vel_x", delta_vel, env.device
                    )
                    _safe_update_range(
                        ranges, limit_ranges, "lin_vel_y", delta_vel, env.device
                    )

                # -- Update force range --
                if force_range is not None and limit_force_range is not None:
                    delta_force = torch.tensor([-5.0, 5.0], device=env.device)
                    _safe_update_range(
                        force_range,
                        limit_force_range,
                        "lin_force_x",
                        delta_force,
                        env.device,
                    )
                    _safe_update_range(
                        force_range,
                        limit_force_range,
                        "lin_force_y",
                        delta_force,
                        env.device,
                    )

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def pos_cmd_levels(
    env,
    env_ids: Sequence[int],
    reward_term_name: str = "end_effector_position_tracking_exp",
) -> torch.Tensor:
    cfg = env.command_manager.get_term("ee_pose").cfg

    is_curriculum_enabled = getattr(
        cfg, "curriculum_enabled", getattr(cfg, "currirulum_enabled", False)
    )
    ranges = getattr(cfg, "ranges", None)
    limit_ranges = getattr(cfg, "limit_ranges", None)
    force_range = getattr(cfg, "force_range", None)
    limit_force_range = getattr(cfg, "limit_force_range", None)

    # --- Handle disabled curriculum ---
    if not is_curriculum_enabled:
        if ranges is not None and limit_ranges is not None:
            cfg.ranges = limit_ranges
        val = (
            limit_ranges.pos_x[1]
            if limit_ranges and hasattr(limit_ranges, "pos_x")
            else 0.0
        )
        return torch.tensor(-1.0, device=env.device)

    if ranges is None or not hasattr(ranges, "pos_x"):
        return torch.tensor(0.0, device=env.device)

    # --- Performance: only compute at episode end ---
    if (
        env.common_step_counter > 0
        and env.common_step_counter % env.max_episode_length == 0
    ):
        reward_sums = env.reward_manager._episode_sums.get(reward_term_name)
        if reward_sums is not None:
            reward = torch.mean(reward_sums[env_ids]) / env.max_episode_length_s
            reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

            if reward > reward_term_cfg.weight * 0.8:
                pos_delta = torch.tensor([-0.05, 0.05], device=env.device)
                ori_delta = torch.tensor([-3.14 / 18, 3.14 / 18], device=env.device)
                force_delta = torch.tensor([-5.0, 5.0], device=env.device)

                # Update position ranges
                if limit_ranges is not None:
                    for axis in ["pos_x", "pos_y", "pos_z"]:
                        _safe_update_range(
                            ranges, limit_ranges, axis, pos_delta, env.device
                        )
                    for axis in ["roll", "pitch", "yaw"]:
                        _safe_update_range(
                            ranges, limit_ranges, axis, ori_delta, env.device
                        )

                # Update force ranges
                if force_range is not None and limit_force_range is not None:
                    for axis in ["lin_force_x", "lin_force_y", "lin_force_z"]:
                        _safe_update_range(
                            force_range,
                            limit_force_range,
                            axis,
                            force_delta,
                            env.device,
                        )

    return torch.tensor(ranges.pos_x[1], device=env.device)
