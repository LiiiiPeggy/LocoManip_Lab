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

"""Body-relative teleoperation WBC for the RangerBox-CR10 (branch `rangercr10-teleop`).

ruff is told to skip F403/F405 here: the module deliberately re-exports the shared env config
with a star import (that is the repository's config pattern), so `mdp` and the shared names are
undefined from ruff's point of view.

Two commands, one policy, eight actions:

    base_velocity_cmd = [vx_cmd, wz_cmd]        -> what the operator asks the chassis for
    ee_pose_cmd_base  = [x, y, z, r, p, yaw]    -> where the tool should be, relative to base_link
    policy action     = [vx, wz, cr10_joint1..6]

The chassis velocity is an *action*, not a passthrough: the policy reads the operator's
velocity command and produces the (vx, wz) the base controller executes. That matters for
the comparison the acceptance test runs -- feeding the command straight to the base
controller would not be the same controller at all, and the point of the 8-D action is
that the policy can spend chassis motion on the arm's behalf.

Three operating modes come out of the same policy, differing only in what the operator
changes. They are the things the acceptance test measures separately:

* chassis only   -- the EE target is held, the velocity command changes
* arm only       -- the velocity command is zero, the EE target changes
* both           -- both change together

Why the base velocity command and the EE target resample independently
---------------------------------------------------------------------
`rel_standing_envs` forces a fraction of envs to a *zero* velocity command while their
pose target keeps changing, which is the "arm only" mode; independent resampling of the
two commands produces the "both" mode. Without the standing fraction the policy sees
"arm only" rarely enough to learn to stand still, and the acceptance test would find out
after a full training run instead of before it.

Nothing here is shared with `rangercr10-world`, deliberately: that branch's task is a
world-frozen target with no velocity command, and the two must be able to move apart
without either one's numbers changing. The action space, chassis controller, CR10 model
and TCP definition *are* shared -- they are the parts already verified.
"""

import importlib.util

# ruff: noqa: F403, F405
import math
from pathlib import Path

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from LeggedManip_Lab.assets.ranger_cr10.ranger_cr10_articulation_cfg import (
    ARM_JOINT_NAMES,
    RANGER_CR10_CFG,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.leggedmanip_lab_env_cfg import *
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.leggedmanip_lab_env_cfg import LeggedManipLabEnvCfg

from ...mdp.tcp import TCP_ANCHOR_BODY
from .wbc_env_cfg import RangerActionsCfg


# -- the workspace: one definition, shared with the MuJoCo keyboard -------------------
# `scripts/ranger_cr10/workspace_limits.py` holds the stage boxes, the replay ratio and the
# keyboard clamps. It is pure Python and deliberately lives outside the extension so the
# deployment can import it too; loading it by path here is the price of that single source.
def _find_workspace_file() -> Path:
    """Walk up until the repository root (the directory holding scripts/) is found."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "scripts" / "ranger_cr10" / "workspace_limits.py"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("scripts/ranger_cr10/workspace_limits.py not found above this file")


_WORKSPACE_FILE = _find_workspace_file()
_spec = importlib.util.spec_from_file_location("ranger_workspace_limits", _WORKSPACE_FILE)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load the workspace definition from {_WORKSPACE_FILE}")
ws = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ws)

# -- end-effector target box, as offsets from the default TCP pose ------------------
# Conservative on purpose for the first training run. The default TCP is 0.73 m from
# base_link and the CR10 reaches about 1.3 m, but reach is not the binding constraint --
# joint 2 is limited to +-90 degrees, so the useful box is much smaller than the sphere,
# and a target the arm cannot hold would turn the orientation term into a constant
# penalty. `check_teleop_env.py` measures the achieved error over this box so it can be
# widened on evidence rather than optimism.
# Stage 4.7: the box the policy is trained on is now *staged*. `ranges` starts at the box
# every earlier stage used and grows on measured success (`workspace_cmd_levels`); the limit is
# the box the reachability study supports. Both come from workspace_limits.py, as do the
# keyboard clamps, so the two sides cannot drift apart.
EE_OFFSET_RANGES = ws.as_ranges_kwargs(ws.STAGE_BOXES[0])
EE_OFFSET_LIMIT_RANGES = ws.as_ranges_kwargs(ws.TARGET_BOX)


@configclass
class RangerTeleopCommandsCfg:
    """One velocity command and one chassis-relative pose command, resampled independently."""

    base_velocity = mdp.command_cfg.UniformVelocityCommandCfg(
        asset_name="robot",
        # Shorter than go2_piper's 10 s: teleoperation changes the command far more often
        # than a nominal locomotion episode does, and a policy trained only to hold one
        # velocity for ten seconds has little practice at transitions. The two commands
        # use independent draws, which is what makes the "both change at once" mode occur.
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=0.25,
        debug_vis=False,
        heading_command=False,
        ranges=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.25, 0.25),
            # no crab steering: the Ranger has no vy, and a command channel that is
            # always zero would only add an input the policy has to learn to ignore
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.25, 0.25),
            heading=(0.0, 0.0),
        ),
        limit_ranges=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
            # the base controller's own limits, so the command can never ask for a
            # velocity the action space cannot represent
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.5, 0.5),
            heading=(0.0, 0.0),
        ),
        curriculum_enabled=True,
    )

    ee_pose = mdp.command_cfg.UniformTCPBodyPoseCommandCfg(
        asset_name="robot",
        body_name=TCP_ANCHOR_BODY,
        # the chassis is the reference frame -- this is what makes driving leave the
        # target untouched, and the single largest behavioural difference from
        # rangercr10-world's UniformPoseWorldCommand
        root_name="base_link",
        resampling_time_range=(4.0, 8.0),
        debug_vis=True,
        ranges=mdp.command_cfg.UniformTCPBodyPoseCommandCfg.Ranges(**EE_OFFSET_RANGES),
        limit_ranges=mdp.command_cfg.UniformTCPBodyPoseCommandCfg.Ranges(**EE_OFFSET_LIMIT_RANGES),
        # stage 4.7: grow the box stage by stage, gated on measured success (see
        # `workspace_cmd_levels`), and keep a share of the original box in the distribution
        replay_ranges=mdp.command_cfg.UniformTCPBodyPoseCommandCfg.Ranges(**ws.as_ranges_kwargs(ws.OLD_BOX)),
        replay_ratio=ws.REPLAY_RATIO,
        stage_boxes=[ws.as_ranges_kwargs(b) for b in ws.STAGE_BOXES],
        stage_names=list(ws.STAGE_NAMES),
        start_stage=ws.START_STAGE,
        curriculum_enabled=True,
    )


@configclass
class RangerTeleopRewardsCfg(RewardsCfg):
    """The shared reward set, unchanged.

    Deliberately no new terms. `track_lin_vel_xy_exp` and `track_ang_vel_z_exp` are already
    two separate terms, and `UniformVelocityCommand` already logs `error_vel_xy` and
    `error_vel_yaw` as two separate metrics -- so the linear and angular channels stay
    independently diagnosable, which is the thing the world-frame task got wrong by
    fusing them into one penalty and could not then explain.
    """


@configclass
class RangerTeleopCurriculumCfg:
    """Two difficulties: how fast the operator may ask the chassis to go, and how far the arm
    has to reach. The arm box is promoted on the *command term's own success rate*, not on the
    mean tracking reward -- see `workspace_cmd_levels` for why a mean is the wrong statistic."""

    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)  # type: ignore
    ang_vel_cmd_levels = CurrTerm(func=mdp.ang_vel_cmd_levels)  # type: ignore
    ee_pose_levels = CurrTerm(  # type: ignore
        func=mdp.workspace_cmd_levels,
        params={"promote_at": ws.PROMOTE_AT, "min_stage_steps": 2000},
    )


@configclass
class RangerCr10TeleopEnvCfg(LeggedManipLabEnvCfg):
    """Body-relative teleoperation. Target: `RANGER-CR10-TELEOP-WBC`."""

    actions: RangerActionsCfg = RangerActionsCfg()
    commands: RangerTeleopCommandsCfg = RangerTeleopCommandsCfg()
    rewards: RangerTeleopRewardsCfg = RangerTeleopRewardsCfg()
    curriculum: RangerTeleopCurriculumCfg = RangerTeleopCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # Must run before any reward is set to None: the helper reads reward.weight and
        # does not tolerate None entries.
        self.disable_zero_weight_rewards()

        # scene
        self.scene.robot = RANGER_CR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 10 Hz policy on 200 Hz physics, matching the world task and the stage-2 finding
        self.decimation = 20
        self.sim.render_interval = self.decimation

        # -- events -----------------------------------------------------------------
        self.events.push_robot = None
        # Wheels keep the low fixed friction measured in the world task; the shared
        # whole-body randomisation makes the 176 kg chassis sink far enough for rolling
        # resistance to exceed the wheel drive. See wbc_env_cfg.py for the measurement.
        self.events.wheel_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link"),
                "static_friction_range": (0.35, 0.35),
                "dynamic_friction_range": (0.35, 0.35),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1,
                "make_consistent": True,
            },
        )
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")
        self.events.base_external_force_torque.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")

        # -- observations -----------------------------------------------------------
        # The joint terms must name their joints explicitly: the shared defaults list
        # twelve quadruped leg joints, which match nothing here and would silently produce
        # an empty observation.
        ee_cfg = SceneEntityCfg("robot", body_names=TCP_ANCHOR_BODY)
        for group in (self.observations.policy, self.observations.critic):
            group.joint_pos.params = {"joint_names": ARM_JOINT_NAMES}
            group.joint_vel.params = {"joint_names": ARM_JOINT_NAMES}
            # kept, unlike the world task: here the velocity command exists and is the
            # policy's instruction for its own chassis channels

        # The chassis is an action, so the policy controls a velocity it cannot otherwise
        # observe. The world task measured what that costs -- a policy that held a 0.13 m/s
        # creep while commanding 0.5, with nothing in the observation saying it was moving.
        # The real chassis reports this over /odom.
        self.observations.policy.base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            scale=2.0,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        self.observations.critic.feet_contact = None
        self.observations.critic.ee_link0_rel_pose.params = {
            "asset_cfg": ee_cfg,
            "base_body_name": "base_link",
        }
        self.observations.critic.ee_link0_rel_pose.func = mdp.end_effector_tcp_pose

        # -- rewards ----------------------------------------------------------------
        # Both tracking terms are measured at the virtual TCP against the chassis-relative
        # command, so the posture is judged where the operator's tool actually is.
        self.rewards.end_effector_position_tracking_exp.func = mdp.tcp_position_command_b_error_exp
        self.rewards.end_effector_position_tracking_exp.weight = 4.5
        self.rewards.end_effector_position_tracking_exp.params = {
            "asset_cfg": ee_cfg,
            "command_name": "ee_pose",
            "std": math.sqrt(0.1),
        }
        self.rewards.end_effector_orientation_tracking.func = mdp.tcp_orientation_command_b_error
        self.rewards.end_effector_orientation_tracking.weight = -4.0
        self.rewards.end_effector_orientation_tracking.params = {
            "asset_cfg": ee_cfg,
            "command_name": "ee_pose",
        }

        # go2_piper's weights; the two channels stay separate terms on purpose
        self.rewards.track_lin_vel_xy_exp.weight = 3.5
        self.rewards.track_ang_vel_z_exp.weight = 2.5
        # the wheels fix the chassis height, so there is nothing to track
        self.rewards.track_base_height_exp = None

        # arm_deviation ships pointed at "joint.*", which matches the go2_piper arm and
        # nothing at all here -- a silent no-op. Aim it at the CR10 joints.
        self.rewards.arm_deviation.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)

        # -- quadruped-only terms ----------------------------------------------------
        for name in (
            "feet_slide",
            "feet_air_time",
            "feet_long_air",
            "air_time_variance",
            "hip_torques_max",
            "thigh_torques_max",
            "calf_torques_max",
            "hip_deviation",
            "joint_deviation",
            "joint_mirror",
        ):
            setattr(self.rewards, name, None)

        # -- terminations ------------------------------------------------------------
        # "base" does not exist; use the chassis body, and drop the four-legged tip-over test
        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names="base_link")


class RangerCr10TeleopEnvCfg_PLAY(RangerCr10TeleopEnvCfg):
    """Playback with the difficulty dialled to its limit and the scene made small."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 6.0
        self.observations.policy.enable_corruption = False

        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.ee_pose.ranges = self.commands.ee_pose.limit_ranges
