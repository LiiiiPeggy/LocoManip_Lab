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

from isaaclab.utils import configclass
from ...leggedmanip_lab_env_cfg import LeggedManipLabEnvCfg
from LeggedManip_Lab.assets.go1_wx250s.go1_wx250s_articulation_cfg import GO1_WX250S_CFG
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.leggedmanip_lab_env_cfg import *


@configclass
class Go1WX250SFlatEnvCfg(LeggedManipLabEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # scene
        self.scene.robot: ArticulationCfg = GO1_WX250S_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )


        # observations


        # curriculum

        # commands
        self.commands.ee_pose.limit_ranges.pos_x = (0.4, 0.65)
        self.commands.ee_pose.limit_ranges.pos_y = (-0.35, 0.35)
        self.commands.ee_pose.limit_ranges.pos_z = (-0.2, 0.5)

        # events
        self.events.push_robot = None

        # actions
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": (-10.0, 10.0)}
        # rewards
        self.rewards.track_base_height_exp.params["target_height"] = 0.28

        # terminals

        self.disable_zero_weight_rewards()


class Go1WX250SFlatEnvCfg_PLAY(Go1WX250SFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event

        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.ee_pose.ranges = self.commands.ee_pose.limit_ranges
