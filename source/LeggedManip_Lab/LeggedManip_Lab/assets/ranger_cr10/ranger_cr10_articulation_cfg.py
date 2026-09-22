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

"""Articulation configuration for the RangerBox-CR10-Lidar wheeled mobile manipulator.

Joint limits come from the converted USD, which the URDF importer filled in from
``rangercr10lidar.urdf`` (see docs/plans/plan.md section 2.2 for why that file wins
over ``cr10_robot.urdf``). The URDF leaves the arm joints at ``effort=0``, so the
effort limits below are estimates -- see ARM_EFFORT_LIMITS_NM.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

current_dir = os.path.dirname(os.path.abspath(__file__))

RANGER_CR10_USD = os.path.join(current_dir, "ranger_cr10.usd")

##
# Geometry, measured from the model rather than assumed
##

# wheel disc diameter is 0.3062 m in the mesh bounding box of {fr,fl}_wheel_link.STL
WHEEL_RADIUS = 0.1531
# steering joint origins sit at (+-0.445, +-0.28) relative to base_link
WHEEL_X = 0.445
WHEEL_Y = 0.28
# wheel centres sit 0.2583 m below base_link, so geometrically base_link rides this high
BASE_HEIGHT_GEOMETRIC = 0.2583 + WHEEL_RADIUS  # 0.4114
# measured: once the tyres take up contact the chassis rests at 0.4071 m, so spawn there
# and skip the few millimetres of initial settling
BASE_HEIGHT = 0.4071

STEERING_JOINT_NAMES = [
    "fr_steering_joint",
    # note the naming inconsistency: only the front-right one lacks the "wheel" suffix
    "fl_steering_wheel_joint",
    "rl_steering_wheel_joint",
    "rr_steering_wheel_joint",
]
WHEEL_JOINT_NAMES = ["fr_wheel_joint", "fl_wheel_joint", "rl_wheel_joint", "rr_wheel_joint"]
ARM_JOINT_NAMES = [f"cr10_joint{i}" for i in range(1, 7)]
GRIPPER_JOINT_NAMES = ["gripper_finger1_joint"]
# The seven joints the AG95 linkage drives through PhysX mimic constraints. They keep
# their mimic relationship, and additionally get a holding drive at 0 -- which is exactly
# where the mimic relation puts them when the driven joint is 0, so the two never fight.
# The constraint alone is too soft: measured at naturalFrequency 25 / dampingRatio 0.005,
# the linkage sagged until two of these joints were off by 0.72 and 2.79 rad.
GRIPPER_MIMIC_JOINT_NAMES = [
    "gripper_finger1_finger_joint",
    "gripper_finger1_inner_knuckle_joint",
    "gripper_finger1_finger_tip_joint",
    "gripper_finger2_joint",
    "gripper_finger2_finger_joint",
    "gripper_finger2_inner_knuckle_joint",
    "gripper_finger2_finger_tip_joint",
]

##
# Actuator gains
##
# The arm joints have no limits in the URDF. These are estimated from the arm's own
# mass (24.8 kg) plus the CR10's rated 10 kg payload: holding ~0.35 m of arm at the
# shoulder is ~68 Nm, and a 10 kg payload at 1 m adds ~98 Nm more, so the shoulder
# needs order 150 Nm. To be replaced with the controller's real values once known.
ARM_EFFORT_LIMITS_NM = {
    "cr10_joint1": 150.0,
    "cr10_joint2": 150.0,
    "cr10_joint3": 100.0,
    "cr10_joint4": 50.0,
    "cr10_joint5": 50.0,
    "cr10_joint6": 30.0,
}
# Wheel torque budget: 176 kg / 4 wheels over a 0.1531 m radius. 1 m/s^2 needs
# ~6.7 Nm per wheel, so 60 Nm leaves ample margin for turning and slopes.
WHEEL_EFFORT_LIMIT_NM = 60.0

##
# Configuration
##

RANGER_CR10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=RANGER_CR10_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            # a 176 kg chassis on four wheels needs more solver work than a quadruped
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, BASE_HEIGHT),
        joint_pos={
            # chassis straight and at rest
            "fr_steering_joint": 0.0,
            "fl_steering_wheel_joint": 0.0,
            "rl_steering_wheel_joint": 0.0,
            "rr_steering_wheel_joint": 0.0,
            "fr_wheel_joint": 0.0,
            "fl_wheel_joint": 0.0,
            "rl_wheel_joint": 0.0,
            "rr_wheel_joint": 0.0,
            # arm: measured, not guessed. At j1 = 0.94 (its upper limit) the TCP sits dead
            # ahead at (0.549, -0.001, 0.470) m, so j1 = 0.6 points the tool forward while
            # leaving headroom for the reset scaling of (0.5, 1.5) to stay inside the limit.
            "cr10_joint1": 0.6,
            "cr10_joint2": 1.0,
            "cr10_joint3": -2.4,
            "cr10_joint4": 0.0,
            "cr10_joint5": 1.4,
            "cr10_joint6": 0.0,
            # gripper closed; its seven mimic joints are driven by PhysX mimic constraints
            "gripper_finger1_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # position driven: the base controller writes steering angle targets
        "steering": DelayedPDActuatorCfg(
            joint_names_expr=STEERING_JOINT_NAMES,
            effort_limit=100.0,
            velocity_limit=10.0,
            stiffness=300.0,
            damping=30.0,
            armature=0.5,
            friction=0.02,
            min_delay=0,
            max_delay=2,
        ),
        # velocity driven: stiffness is zero so only the velocity target acts
        "wheels": DelayedPDActuatorCfg(
            joint_names_expr=WHEEL_JOINT_NAMES,
            effort_limit=WHEEL_EFFORT_LIMIT_NM,
            velocity_limit=20.0,
            stiffness=0.0,
            damping=5.0,
            armature=0.05,
            friction=0.02,
            min_delay=0,
            max_delay=2,
        ),
        # one group per arm joint, mirroring the go2_piper layout
        "joint1": DelayedPDActuatorCfg(
            joint_names_expr=["cr10_joint1"],
            effort_limit=ARM_EFFORT_LIMITS_NM["cr10_joint1"],
            velocity_limit=3.0,
            stiffness=1500.0,
            damping=100.0,
            armature=0.5,
            friction=0.01,
            min_delay=0,
            max_delay=4,
        ),
        "joint2": DelayedPDActuatorCfg(
            joint_names_expr=["cr10_joint2"],
            effort_limit=ARM_EFFORT_LIMITS_NM["cr10_joint2"],
            velocity_limit=3.0,
            stiffness=4000.0,
            damping=200.0,
            armature=1.0,
            friction=0.01,
            min_delay=0,
            max_delay=4,
        ),
        "joint3": DelayedPDActuatorCfg(
            joint_names_expr=["cr10_joint3"],
            effort_limit=ARM_EFFORT_LIMITS_NM["cr10_joint3"],
            velocity_limit=3.0,
            stiffness=3000.0,
            damping=150.0,
            armature=0.5,
            friction=0.01,
            min_delay=0,
            max_delay=4,
        ),
        "joint4": DelayedPDActuatorCfg(
            joint_names_expr=["cr10_joint4"],
            effort_limit=ARM_EFFORT_LIMITS_NM["cr10_joint4"],
            velocity_limit=3.2,
            stiffness=800.0,
            damping=40.0,
            armature=0.2,
            friction=0.01,
            min_delay=0,
            max_delay=4,
        ),
        "joint5": DelayedPDActuatorCfg(
            joint_names_expr=["cr10_joint5"],
            effort_limit=ARM_EFFORT_LIMITS_NM["cr10_joint5"],
            velocity_limit=3.2,
            stiffness=500.0,
            damping=25.0,
            armature=0.1,
            friction=0.01,
            min_delay=0,
            max_delay=4,
        ),
        "joint6": DelayedPDActuatorCfg(
            joint_names_expr=["cr10_joint6"],
            effort_limit=ARM_EFFORT_LIMITS_NM["cr10_joint6"],
            velocity_limit=3.2,
            stiffness=300.0,
            damping=15.0,
            armature=0.1,
            friction=0.01,
            min_delay=0,
            max_delay=4,
        ),
        # the seven mimic-driven linkage joints, held at the closed pose
        "gripper_mimic": DelayedPDActuatorCfg(
            joint_names_expr=GRIPPER_MIMIC_JOINT_NAMES,
            effort_limit=50.0,
            velocity_limit=2.0,
            stiffness=500.0,
            damping=20.0,
            armature=0.05,
            friction=0.01,
            min_delay=0,
            max_delay=2,
        ),
        # the one actively driven finger joint
        "gripper": DelayedPDActuatorCfg(
            joint_names_expr=GRIPPER_JOINT_NAMES,
            effort_limit=10.0,
            velocity_limit=2.0,
            stiffness=200.0,
            damping=10.0,
            armature=0.1,
            friction=0.01,
            min_delay=0,
            max_delay=2,
        ),
    },
)
