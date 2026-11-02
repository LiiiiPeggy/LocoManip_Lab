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

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


def _safe_update_range(target_obj, limit_obj, attr_name: str, delta: torch.Tensor, device: torch.device):
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
    is_curriculum_enabled = getattr(cfg, "curriculum_enabled", getattr(cfg, "currirulum_enabled", False))
    limit_ranges = getattr(cfg, "limit_ranges", None)
    ranges = getattr(cfg, "ranges", None)

    # 2. Handle disabled curriculum
    if not is_curriculum_enabled:
        # When disabled, force ranges to limit_ranges (only needs to happen once, or rely on init)
        if ranges is not None and limit_ranges is not None:
            cfg.ranges = limit_ranges
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
                _safe_update_range(ranges, limit_ranges, "ang_vel_z", delta_command, env.device)

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def lin_vel_cmd_levels(
    env,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
) -> torch.Tensor:
    cfg = env.command_manager.get_term("base_velocity").cfg

    is_curriculum_enabled = getattr(cfg, "curriculum_enabled", getattr(cfg, "currirulum_enabled", False))
    ranges = getattr(cfg, "ranges", None)
    limit_ranges = getattr(cfg, "limit_ranges", None)
    force_range = getattr(cfg, "force_range", None)
    limit_force_range = getattr(cfg, "limit_force_range", None)

    # --- Handle disabled curriculum ---
    if not is_curriculum_enabled:
        # Fix bug: original directly assigned ranges = limit_ranges (only assigned local variable)
        if ranges is not None and limit_ranges is not None:
            cfg.ranges = limit_ranges
        return torch.tensor(-1.0, device=env.device)

    if ranges is None or not hasattr(ranges, "lin_vel_x"):
        return torch.tensor(0.0, device=env.device)

    # --- Performance: only compute at episode end ---
    if env.common_step_counter > 0 and env.common_step_counter % env.max_episode_length == 0:
        reward_sums = env.reward_manager._episode_sums.get(reward_term_name)
        if reward_sums is not None:
            reward = torch.mean(reward_sums[env_ids]) / env.max_episode_length_s
            reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

            if reward > reward_term_cfg.weight * 0.8:
                # print(f"Episode ended. Reward: {reward:.4f}, Threshold: {reward_term_cfg.weight * 0.8:.4f}")

                # -- Update velocity range --
                if limit_ranges is not None:
                    delta_vel = torch.tensor([-0.05, 0.05], device=env.device)
                    _safe_update_range(ranges, limit_ranges, "lin_vel_x", delta_vel, env.device)
                    _safe_update_range(ranges, limit_ranges, "lin_vel_y", delta_vel, env.device)

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

    is_curriculum_enabled = getattr(cfg, "curriculum_enabled", getattr(cfg, "currirulum_enabled", False))
    ranges = getattr(cfg, "ranges", None)
    limit_ranges = getattr(cfg, "limit_ranges", None)
    force_range = getattr(cfg, "force_range", None)
    limit_force_range = getattr(cfg, "limit_force_range", None)

    # --- Handle disabled curriculum ---
    if not is_curriculum_enabled:
        if ranges is not None and limit_ranges is not None:
            cfg.ranges = limit_ranges
        return torch.tensor(-1.0, device=env.device)

    if ranges is None or not hasattr(ranges, "pos_x"):
        return torch.tensor(0.0, device=env.device)

    # --- Performance: only compute at episode end ---
    if env.common_step_counter > 0 and env.common_step_counter % env.max_episode_length == 0:
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
                        _safe_update_range(ranges, limit_ranges, axis, pos_delta, env.device)
                    for axis in ["roll", "pitch", "yaw"]:
                        _safe_update_range(ranges, limit_ranges, axis, ori_delta, env.device)

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


def world_pos_cmd_levels(
    env,
    env_ids: Sequence[int],
    success_rate: float = 0.6,
    command_name: str = "ee_pose",
) -> torch.Tensor:
    """Promote the world-target sampling box on measured task success.

    A ranger_cr10-specific replacement for :func:`pos_cmd_levels`, which promotes when the
    mean of the position-tracking *reward* exceeds 0.8 of its weight. That condition has
    two problems for a target that the base has to drive to:

    * the reward is an ``exp(-d^2 / 0.1)`` kernel, dead past about 0.75 m, so the mean
      saturates as soon as the easy targets are solved and stops responding to the far
      ones at all. The 2026-09-23 run sat at 3.52 against a threshold of 3.60 for 400
      iterations -- pinned to the criterion, not improving towards it.
    * a mean is not a success rate. It can be held up by the near half of the box while
      the far half is untouched, which is precisely the failure being fixed.

    Success rate answers the question the curriculum is actually asking -- has the policy
    solved the box it has -- and it is scale-free, so it stays meaningful if the reward is
    ever re-weighted. The per-stratum version of the same measurement is what the stage-4
    acceptance test reports.

    The default 0.6 is deliberately not 1.0: the far end of the box is sampled uniformly,
    so some draws sit at the edge of what the arm can reach even after driving, and
    demanding all of them would stall the curriculum on outliers.
    """
    cfg = env.command_manager.get_term(command_name).cfg

    is_curriculum_enabled = getattr(cfg, "curriculum_enabled", False)
    ranges = getattr(cfg, "ranges", None)
    limit_ranges = getattr(cfg, "limit_ranges", None)

    if not is_curriculum_enabled:
        if ranges is not None and limit_ranges is not None:
            cfg.ranges = limit_ranges
        return torch.tensor(-1.0, device=env.device)

    if ranges is None or limit_ranges is None or not hasattr(ranges, "pos_x"):
        return torch.tensor(0.0, device=env.device)

    # the promotion test is expensive to evaluate and the box only needs to move once per
    # episode, so keep the original's episode-end gate
    if env.common_step_counter > 0 and env.common_step_counter % env.max_episode_length == 0:
        rate = env.command_manager.get_term(command_name).success_rate.mean().item()
        if rate > success_rate:
            pos_delta = torch.tensor([-0.05, 0.05], device=env.device)
            ori_delta = torch.tensor([-3.14 / 18, 3.14 / 18], device=env.device)
            for axis in ["pos_x", "pos_y", "pos_z"]:
                _safe_update_range(ranges, limit_ranges, axis, pos_delta, env.device)
            for axis in ["roll", "pitch", "yaw"]:
                _safe_update_range(ranges, limit_ranges, axis, ori_delta, env.device)

    return torch.tensor(ranges.pos_x[1], device=env.device)


def workspace_cmd_levels(
    env,
    env_ids: Sequence[int],
    command_name: str = "ee_pose",
    promote_at: float = 0.75,
    min_stage_steps: int = 0,
    window_steps: int = 2000,
    promote_windows: int = 3,
    force_after_steps: int = 0,
    force_floor: float = 0.70,
) -> torch.Tensor:
    """Stage the end-effector box up on measured success (stage 4.7, revised in 4.8).

    The stages are explicit boxes (``cfg.stage_boxes``), not a per-episode nudge of every
    limit: the reachability study found the useful space is not a scaled copy of the old box
    (the arm reaches +0.45 m forward but 0.05 m backward from a continuous start), so scaling
    all six limits together would spend most of the new volume on targets the arm cannot hold.

    What the 4.8 revision changes, and why:

    * **the statistic is filtered.** A target the arm cannot physically hold can never be
      reached, so counting it against the policy caps the success rate at the reachable share
      of the box. On the 4.8 box that share is well under the promotion threshold, which is
      how a curriculum stops at stage 1 forever. The gate now reads success over *reachable*
      targets only; the unfiltered rate and the reachable share are printed on every window
      so the exclusion is visible rather than silently helpful.
    * **the threshold is 0.80, not 0.85.** 4.7's run sat at an EMA of 0.838 for 3000
      iterations against a 0.85 gate. 0.80 is still a demand that the policy solve four fifths
      of the box at once.
    * **promotion needs consecutive windows**, not one sample: ``promote_windows`` successive
      evaluations (``window_steps`` policy steps apart) all above the threshold. One lucky
      window is not a measurement.
    * **a stage cannot sit forever.** If a stage has run ``force_after_steps`` and is above
      ``force_floor``, it advances anyway and prints that it was forced. This is a schedule
      escape hatch, not a success claim, and it is logged as such.

    ``min_stage_steps`` additionally forbids promoting within that many steps of the last
    promotion, so a box that is solved trivially cannot run through all stages in a few
    episodes before the policy has actually seen it.
    """
    term = env.command_manager.get_term(command_name)
    cfg = term.cfg

    if not getattr(cfg, "curriculum_enabled", False):
        if cfg.ranges is not None and cfg.limit_ranges is not None:
            cfg.ranges = cfg.limit_ranges
        return torch.tensor(-1.0, device=env.device)

    stages = getattr(cfg, "stage_boxes", None)
    if not stages:
        return torch.tensor(0.0, device=env.device)

    stage = term.stage_index
    if stage + 1 >= len(stages):
        # Nothing left to promote to, but the final stage is the longest one and the most
        # interesting to watch: keep printing the same window line so the run reports what it
        # is actually achieving there. (4.8's own run lost this after iteration 500 -- the
        # numbers are still recoverable from `Metrics/ee_pose/success`, but only unfiltered.)
        if env.common_step_counter % window_steps == 0:
            succ = term.instant_success
            reach = getattr(term, "reachable", None)
            if reach is not None and not bool(reach.all()):
                mask = reach.to(succ.dtype)
                inst_reach = ((succ * mask).sum() / mask.sum().clamp(min=1.0)).item()
            else:
                inst_reach = succ.mean().item()
            print(
                f"[curriculum] window step {env.common_step_counter} stage {stage} (final), "
                f"instant success {inst_reach:.3f} over reachable, raw {succ.mean().item():.3f}, "
                f"reachable share {getattr(term, 'reachable_fraction', 1.0):.3f}"
            )
        return torch.tensor(float(stage), device=env.device)

    # Two things are measured every step and one only every window:
    #
    # * ``instant_success`` -- the fraction of environments on target *now*, which is what
    #   ``Metrics/ee_pose/success`` prints. ``success_rate`` (a running average since each
    #   environment's last resample) is dragged down by the transient after every command
    #   change; the first 4.7 attempt read that one and never promoted.
    # * ``inst_reach`` -- the same fraction over the reachable subset, which is what the gate
    #   uses. ``raw`` is the unfiltered number, printed alongside so a high gate value can
    #   never be mistaken for the box being solved.
    succ = term.instant_success
    reach = getattr(term, "reachable", None)
    if reach is not None and not bool(reach.all()):
        mask = reach.to(succ.dtype)
        inst_reach = ((succ * mask).sum() / mask.sum().clamp(min=1.0)).item()
    else:
        inst_reach = succ.mean().item()
    raw = succ.mean().item()
    term.success_ema = inst_reach if term.success_ema is None else 0.95 * term.success_ema + 0.05 * inst_reach

    # one line at startup, so a short smoke run still shows which box is active and how much
    # of it the reachability oracle considers solvable
    if not getattr(term, "startup_logged", False):
        term.startup_logged = True
        print(
            f"[curriculum] ee_pose start at stage {stage} "
            f"({cfg.stage_names[stage] if cfg.stage_names else ''}): "
            f"reachable share {getattr(term, 'reachable_fraction', 1.0):.3f}, "
            f"ranges {cfg.ranges}"
        )

    # one evaluation window per ``window_steps`` policy steps
    if env.common_step_counter % window_steps != 0:
        return torch.tensor(float(stage), device=env.device)

    elapsed = env.common_step_counter - term.stage_entered_step
    hits = getattr(term, "gate_hits", 0)
    hits = hits + 1 if term.success_ema > promote_at else 0
    term.gate_hits = hits
    print(
        f"[curriculum] window step {env.common_step_counter} stage {stage} "
        f"({cfg.stage_names[stage] if cfg.stage_names else ''}), "
        f"success_ema {term.success_ema:.3f} over reachable, raw {raw:.3f}, "
        f"reachable share {getattr(term, 'reachable_fraction', 1.0):.3f}, "
        f"hits {hits}/{promote_windows}, {elapsed} steps in stage"
    )

    if min_stage_steps and elapsed < min_stage_steps:
        return torch.tensor(float(stage), device=env.device)

    reason = None
    if hits >= promote_windows:
        reason = "success"
    elif force_after_steps and elapsed >= force_after_steps and term.success_ema > force_floor:
        reason = "forced"

    if reason is not None:
        new_stage = stage + 1
        box = stages[new_stage]
        ema_at_promotion = term.success_ema
        term.set_stage(new_stage, box)
        term.gate_hits = 0
        # restart the moving average: the new box is being judged, not the old one
        term.success_ema = None
        print(
            f"[curriculum] ee_pose stage {stage} -> {new_stage} "
            f"({cfg.stage_names[new_stage] if cfg.stage_names else ''}) by {reason}: "
            f"success_ema {ema_at_promotion:.3f} (raw {raw:.3f}) after {elapsed} steps"
        )
        return torch.tensor(float(new_stage), device=env.device)

    return torch.tensor(float(stage), device=env.device)
