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

from isaaclab.envs.mdp.commands import (
    UniformPoseCommandCfg as UniformPoseCommandBaseCfg,
)
from isaaclab.envs.mdp.commands import (
    UniformVelocityCommandCfg as UniformVelocityCommandBaseCfg,
)
from ..pose_command_wbc import UniformPoseWBCCommand
from ..pose_command_b import UniformBodyPoseCommand
from ..pose_command_world import UniformPoseWorldCommand
from ..pose_command_teleop import (
    TCP_DEFAULT_POS_B,
    TCP_DEFAULT_QUAT_B,
    UniformTCPBodyPoseCommand,
)

# -- pose
@configclass
class UniformPoseCommandCfg(UniformPoseCommandBaseCfg):

    class_type: type  = UniformBodyPoseCommand

    ranges: UniformPoseCommandBaseCfg.Ranges = None  # type: ignore
    """The initial range"""
    limit_ranges: UniformPoseCommandBaseCfg.Ranges = None  # type: ignore
    """The range limit """
    curriculum_enabled: bool = False
    root_name: str = "link0"

@configclass
class UniformTCPBodyPoseCommandCfg(UniformPoseCommandCfg):
    """Chassis-relative end-effector pose command, tracked at the virtual TCP.

    ``root_name`` is what makes it chassis-relative; set it to ``base_link`` here rather
    than the default ``link0``, which is the GO2-PIPER trunk and does not exist on this
    robot. ``ranges`` are **offsets from the default TCP pose**, in the chassis frame, not
    absolute poses -- see ``pose_command_teleop.py`` for why that is not a stylistic
    choice. The reported error and the success metric are measured at the fingertip centre
    instead of the flange.
    """

    class_type: type = UniformTCPBodyPoseCommand

    default_pos_b: tuple[float, float, float] = TCP_DEFAULT_POS_B
    """TCP position relative to ``root_name`` at the arm's authored default pose, metres."""

    default_quat_b: tuple[float, float, float, float] = TCP_DEFAULT_QUAT_B
    """TCP attitude relative to ``root_name`` at the arm's authored default pose, ``(w, x, y, z)``."""

    success_threshold: float = 0.10
    """Position error below which a command counts as reached. Tighter than the world
    task's 0.15 m because a chassis-relative target is always inside the arm's own
    workspace, so nothing here depends on how well the chassis drove."""


@configclass
class UniformPoseWBCCommandCfg(UniformPoseCommandBaseCfg):

    class_type: type  = UniformPoseWBCCommand

    ranges: UniformPoseCommandBaseCfg.Ranges = None  # type: ignore
    """The initial range"""
    limit_ranges: UniformPoseCommandBaseCfg.Ranges = None  # type: ignore
    """The range limit """
    link_name: str = "link0"
    """The name of the root body, which serves as the reference frame for the command."""
    curriculum_enabled: bool = False


@configclass
class UniformPoseWorldCommandCfg(UniformPoseCommandBaseCfg):
    """Pose command whose target is frozen in the world; the policy sees it chassis-relative.

    Used by ranger_cr10, where the policy drives the base as well as the arm -- see
    docs/plans/plan.md section 4.2. The existing WBC command keeps its body-frame XY.
    """

    class_type: type = UniformPoseWorldCommand

    ranges: UniformPoseCommandBaseCfg.Ranges = None  # type: ignore
    """Sampling range of the target offset, expressed in the chassis frame at resample time."""
    limit_ranges: UniformPoseCommandBaseCfg.Ranges = None  # type: ignore
    """Range limit for the curriculum."""
    link_name: str = "base_link"
    """Body the command is expressed relative to, and against which the target is frozen."""
    curriculum_enabled: bool = False
    success_threshold: float = 0.15
    """Position error below which a command counts as reached, for the success metric and
    for the curriculum's promotion test."""


# -- velocity
@configclass
class UniformVelocityCommandCfg(UniformVelocityCommandBaseCfg):
    """Configuration for the uniform velocity command generator."""

    ranges: UniformVelocityCommandBaseCfg.Ranges = None  # type: ignore
    """The initial range"""

    limit_ranges: UniformVelocityCommandBaseCfg.Ranges = None  # type: ignore
    """The range limit """
    curriculum_enabled: bool = False
