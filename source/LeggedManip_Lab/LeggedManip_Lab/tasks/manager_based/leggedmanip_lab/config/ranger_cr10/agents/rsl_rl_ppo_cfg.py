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

import os

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


# World WBC -- policy drives the Ranger chassis and the CR10 arm together.
# The action is 8-D (docs/plans/plan.md section 4.3). The observation grew from 99-D to
# 108-D when the policy group gained base_lin_vel -- 3 numbers, and the group keeps a
# 3-step history (see wbc_env_cfg.py); the hidden sizes are unchanged from go2_piper's.
@configclass
class RangerCr10WBCPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    check_for_nan = True
    max_iterations = 10000
    save_interval = 1000

    experiment_name = "ranger_cr10_world"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="log",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="log",
        ),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# Teleop -- body-relative end-effector target plus a human velocity command.
# Branch `rangercr10-teleop`; see teleop_env_cfg.py. Same 8-D action and the same
# 512/256/128 network as the world task, so a difference in the results is a difference
# in the task, not in the optimiser.
@configclass
class RangerCr10TeleopPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    check_for_nan = True
    max_iterations = 10000
    # stage 4.5j asks for a checkpoint and a fixed-protocol validation every 500 iterations;
    # nothing else depends on the interval, and existing runs keep their own saved models
    save_interval = 500

    experiment_name = "ranger_cr10_teleop"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="log",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="log",
        ),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# Flat -- fixed-base baseline (the chassis is braked).
@configclass
class RangerCr10FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    check_for_nan = True
    max_iterations = 10000
    save_interval = 1000

    experiment_name = "ranger_cr10_world_flat"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="log",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="log",
        ),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# ---------------------------------------------------------------------------------------
# Stage 4.5e variant C: the same PPO plus a supervised term toward the local-IK teacher.
# Everything else is copied from RangerCr10TeleopPPORunnerCfg so that A, B and C differ only
# in the supervision mechanism. Registered under its own agent entry point on the SAME task
# id, so the default `rsl_rl_cfg_entry_point` keeps behaving exactly as before.
# ---------------------------------------------------------------------------------------
@configclass
class RangerCr10TeleopIKAuxAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "LeggedManip_Lab.ik_aux_ppo:PPOWithIKAux"
    ik_aux_coef: float = 0.5
    """Weight on MSE(actor mean, teacher) over the six arm channels. Chosen so the term is
    comparable to the surrogate at the start of training rather than dominating it."""
    ik_teacher_path: str = os.environ.get("IK_TEACHER_PATH", "logs/ik_teacher/teacher.pt")
    """Produced by scripts/ranger_cr10/teleop_ik_teacher.py. Repo-relative, which is what
    train.py's working directory is; override with IK_TEACHER_PATH if it lives elsewhere."""


@configclass
class RangerCr10TeleopIKAuxPPORunnerCfg(RangerCr10TeleopPPORunnerCfg):
    algorithm = RangerCr10TeleopIKAuxAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
