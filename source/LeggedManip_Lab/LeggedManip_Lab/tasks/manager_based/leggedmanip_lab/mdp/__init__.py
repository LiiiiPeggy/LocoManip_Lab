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

"""This sub-module contains the functions that are specific to the environment."""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from .rewards import *  # noqa: F401, F403
from .cfg import command_cfg  # noqa: F401
from .events import randomize_rigid_body_inertia  # noqa: F401
from .curriculums import *  # noqa: F401, F403
from .observations import *
from .pose_command_wbc import *
from .pose_command_b import *

# -- additions for the wheeled ranger_cr10 platform (docs/plans/plan.md)
# imported by name rather than with ``*`` so their module-level helpers (dataclasses.field,
# configclass, ActionTerm, ...) do not leak into the mdp namespace
from .base_vel import (  # noqa: F401
    BaseVelocityAction,
    BaseVelocityActionCfg,
    base_speed_above_threshold_l2,
    base_velocity_rate_l2,
)
from .pose_command_world import UniformPoseWorldCommand  # noqa: F401
from .tcp import (  # noqa: F401
    TCP_ANCHOR_BODY,
    TCP_OFFSET_POS,
    TCP_OFFSET_QUAT,
    end_effector_tcp_pose,
    orientation_command_world_error,
    position_command_world_error_exp,
)