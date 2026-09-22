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

"""Joint base + arm WBC environment for the RangerBox-CR10.

The policy emits ``[vx, wz, cr10_joint1..6]`` and drives the chassis and the arm
together. See docs/plans/plan.md section 4 for why the base is an action rather than
an external velocity command, and section 4.2 for the world-target/base-relative
observation split.
"""

import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from LeggedManip_Lab.assets.ranger_cr10.ranger_cr10_articulation_cfg import (
    ARM_JOINT_NAMES,
    RANGER_CR10_CFG,
    STEERING_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    WHEEL_RADIUS,
    WHEEL_X,
    WHEEL_Y,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.leggedmanip_lab_env_cfg import *
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.leggedmanip_lab_env_cfg import LeggedManipLabEnvCfg

from ...mdp.tcp import TCP_ANCHOR_BODY


@configclass
class RangerActionsCfg:
    """Action terms for this platform, declared base-first.

    The order matters: the action manager slices the policy output by term declaration
    order, and the documented action vector is ``[vx, wz, cr10_joint1..6]``. This is why
    the class does not inherit from the shared ``ActionsCfg`` -- a subclass would put
    the inherited ``joint_pos`` first and yield ``[6 arm joints, vx, wz]`` instead.
    """

    base_vel = mdp.BaseVelocityActionCfg(
        asset_name="robot",
        steering_joint_names=STEERING_JOINT_NAMES,
        wheel_joint_names=WHEEL_JOINT_NAMES,
        wheel_xy=[
            (WHEEL_X, -WHEEL_Y),  # fr
            (WHEEL_X, WHEEL_Y),  # fl
            (-WHEEL_X, WHEEL_Y),  # rl
            (-WHEEL_X, -WHEEL_Y),  # rr
        ],
        wheel_radius=WHEEL_RADIUS,
        # fr, fl, rl, rr -- the URDF gives fr/rr axis 0 0 1 and fl/rl axis 0 0 -1,
        # so the two sides need opposite signs to roll the same way.
        wheel_sign=(-1.0, 1.0, 1.0, -1.0),
        # sign of the steering joint axis relative to a left-positive steer angle,
        # fixed by measurement: at +1 a commanded +wz produced -23.5 deg of yaw
        # instead of +68.8 deg
        steering_sign=-1.0,
        clip=((-0.5, -0.5), (0.5, 0.5)),
    )

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class RangerWBCCommandsCfg:
    """Only the end-effector pose. There is no base velocity command any more."""

    ee_pose = mdp.command_cfg.UniformPoseWorldCommandCfg(
        asset_name="robot",
        body_name=TCP_ANCHOR_BODY,
        link_name="base_link",
        resampling_time_range=(8.0, 10.0),
        debug_vis=True,
        # Sampled in the chassis frame at resample time, then frozen in the world.
        # Starts inside what the arm can reach without moving the base (curriculum 1).
        ranges=mdp.command_cfg.UniformPoseWorldCommandCfg.Ranges(
            # centred on where the arm actually is: measured TCP is about
            # (0.52, -0.07, 0.47) m in the chassis frame at the initial pose
            pos_x=(0.40, 0.70),
            pos_y=(-0.35, 0.05),
            pos_z=(0.30, 0.65),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        # Curriculum end state: includes points the arm can only reach after driving
        # (curriculum 3). Far-forward and far-lateral targets are the point of the
        # joint scheme -- see docs/plans/plan.md section 7, stage 3.
        limit_ranges=mdp.command_cfg.UniformPoseWorldCommandCfg.Ranges(
            pos_x=(-0.80, 3.00),
            pos_y=(-2.00, 2.00),
            pos_z=(0.15, 1.30),
            roll=(-3.14 / 3, 3.14 / 3),
            pitch=(-3.14 / 4, 3.14 / 4),
            yaw=(-3.14 / 6, 3.14 / 6),
        ),
        curriculum_enabled=True,
    )


@configclass
class RangerRewardsCfg(RewardsCfg):
    """Shared rewards, with the legs/feet terms dropped and two chassis terms added."""

    base_speed_penalty = RewTerm(
        func=mdp.base_speed_above_threshold_l2,
        weight=-1.0,
        params={"threshold": 0.3},
    )
    base_action_rate = RewTerm(
        func=mdp.base_velocity_rate_l2,
        weight=-0.1,
        params={"action_name": "base_vel"},
    )


@configclass
class RangerCurriculumCfg:
    """The velocity curricula are gone with the velocity command; the pose one remains."""

    pos_cmd_levels = CurrTerm(func=mdp.pos_cmd_levels)  # type: ignore


@configclass
class RangerCr10WBCEnvCfg(LeggedManipLabEnvCfg):

    actions: RangerActionsCfg = RangerActionsCfg()
    commands: RangerWBCCommandsCfg = RangerWBCCommandsCfg()
    rewards: RangerRewardsCfg = RangerRewardsCfg()
    curriculum: RangerCurriculumCfg = RangerCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # Must run before any reward is set to None: the helper reads reward.weight and
        # does not tolerate None entries.
        self.disable_zero_weight_rewards()

        # scene
        self.scene.robot = RANGER_CR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 10 Hz policy on 200 Hz physics; the shared default is decimation 4 (50 Hz).
        # The base controller holds (vx, wz) across the intervening physics steps.
        self.decimation = 20
        # keep rendering at the policy rate instead of the shared default of 4
        self.sim.render_interval = self.decimation

        # -- events: rename the bodies that only existed on the quadruped ----------
        self.events.push_robot = None

        # Wheels get their own low, fixed friction, defined after the whole-body
        # randomisation so it wins.
        #
        # Measured: with the shared randomisation (friction up to 1.2) the 176 kg chassis
        # sank 7 mm into the ground and the resulting contact patch produced a rolling
        # resistance larger than the wheel drive could supply -- the wheels turned 0.84 rad
        # and then jammed solid, with 10 Nm applied and no motion. Rolling needs only
        # enough friction to transmit the drive force: 392 N of thrust over 1727 N of
        # weight is mu >= 0.23, so 0.35 keeps traction at a fraction of the resistance.
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

        # -- observations ----------------------------------------------------------
        # The observation terms must name their joints explicitly: the shared defaults
        # list twelve quadruped leg joints, which match nothing here and would silently
        # produce an empty observation.
        ee_cfg = SceneEntityCfg("robot", body_names=TCP_ANCHOR_BODY)
        for group in (self.observations.policy, self.observations.critic):
            group.joint_pos.params = {"joint_names": ARM_JOINT_NAMES}
            group.joint_vel.params = {"joint_names": ARM_JOINT_NAMES}
            # the base velocity command no longer exists; this term would raise at startup
            group.velocity_commands = None

        self.observations.critic.feet_contact = None
        self.observations.critic.ee_link0_rel_pose.params = {
            "asset_cfg": ee_cfg,
            "base_body_name": "base_link",
        }
        self.observations.critic.ee_link0_rel_pose.func = mdp.end_effector_tcp_pose

        # -- actions ---------------------------------------------------------------
        # declared in RangerActionsCfg; nothing to override here

        # -- rewards ---------------------------------------------------------------
        self.rewards.end_effector_position_tracking_exp.func = mdp.position_command_world_error_exp
        self.rewards.end_effector_position_tracking_exp.weight = 4.5
        self.rewards.end_effector_position_tracking_exp.params = {
            "asset_cfg": ee_cfg,
            "command_name": "ee_pose",
            "std": math.sqrt(0.1),
        }
        self.rewards.end_effector_orientation_tracking.func = mdp.orientation_command_world_error
        self.rewards.end_effector_orientation_tracking.weight = -4.0
        self.rewards.end_effector_orientation_tracking.params = {
            "asset_cfg": ee_cfg,
            "command_name": "ee_pose",
        }

        # no speed command left to track; the chassis penalties replace these
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        # the wheels fix the chassis height, so there is nothing to track
        self.rewards.track_base_height_exp = None

        # arm_deviation ships pointed at "joint.*", which matches the go2_piper arm and
        # nothing at all here -- a silent no-op. Aim it at the CR10 joints.
        self.rewards.arm_deviation.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)

        # -- quadruped-only terms --------------------------------------------------
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

        # -- terminations ----------------------------------------------------------
        # "base" does not exist; use the chassis body, and drop the four-legged tip-over test
        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names="base_link"
        )



class RangerCr10WBCEnvCfg_PLAY(RangerCr10WBCEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 6.0
        self.observations.policy.enable_corruption = False

        # hold the arm targets still for long enough to watch the base do the work
        self.commands.ee_pose.ranges = self.commands.ee_pose.limit_ranges
