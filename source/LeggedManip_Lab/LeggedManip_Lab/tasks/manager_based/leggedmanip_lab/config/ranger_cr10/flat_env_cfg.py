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

"""Fixed-base baseline for the RangerBox-CR10.

The chassis is braked ((vx, wz) clamped to zero) so the policy can only move the arm.
It faces the same target distribution as the joint WBC env, which makes it the control
group for the question the joint scheme exists to answer: does driving the base actually
extend the arm's reach, or would the arm alone have sufficed?

See docs/plans/plan.md section 7, stage 3.
"""

from isaaclab.utils import configclass

from .wbc_env_cfg import RangerCr10WBCEnvCfg


@configclass
class RangerCr10FlatEnvCfg(RangerCr10WBCEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # brake the chassis: every wheel velocity target and steering angle becomes zero,
        # so the base cannot roll or turn.
        self.actions.base_vel.clip = ((0.0, 0.0), (0.0, 0.0))

        # No point ramping the difficulty up over training here -- the base cannot help,
        # so the env faces the full target range from the start. ``pos_cmd_levels`` pins
        # ``ranges`` to ``limit_ranges`` when the curriculum is disabled.
        self.commands.ee_pose.curriculum_enabled = False


class RangerCr10FlatEnvCfg_PLAY(RangerCr10FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 6.0
        self.observations.policy.enable_corruption = False
