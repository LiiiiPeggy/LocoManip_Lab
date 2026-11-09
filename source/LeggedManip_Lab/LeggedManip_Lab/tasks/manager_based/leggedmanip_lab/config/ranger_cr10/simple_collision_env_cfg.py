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

"""Independent SIMPLECOLL training candidate: B-distance reward on the frozen v2 asset.

The independent task uses the PhysX-consistent B distance over the 44 declared active pairs;
the default `RangerCr10TeleopEnvCfg` and its `ranger_cr10.usd` are untouched. The candidate USD
is built from
`assets/ranger_cr10_simple_v2/ranger_cr10_simple_v2.urdf`; Link1--Link6 collision is a single
oriented box per link with joint-end retreats, and box_link has the enclosure body up to its lid
plus a small top-accessory box; the right mounting plate collision was removed per user request
(visual only). Self-collisions are enabled for this candidate so the
physical blocking can be measured. Filtered pairs are applied at rigid-body level by
`scripts/ranger_cr10/apply_collision_filters.py`; collider-level filters were measured not to
take effect in PhysX here.
"""

from __future__ import annotations

import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from LeggedManip_Lab.assets.ranger_cr10.ranger_cr10_articulation_cfg import RANGER_CR10_CFG
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.teleop_env_cfg import (
    RangerCr10TeleopEnvCfg,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.phys_proxy_reward import (
    phys_proxy_clearance_penalty,
)

_SIMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../assets/ranger_cr10_simple_v2")
)
SIMPLE_USD = os.path.join(_SIMPLE_DIR, "ranger_cr10_simple_v2_flat.usd")

SIMPLE_RANGER_CR10_CFG = RANGER_CR10_CFG.replace(
    spawn=RANGER_CR10_CFG.spawn.replace(
        usd_path=SIMPLE_USD,
        articulation_props=RANGER_CR10_CFG.spawn.articulation_props.replace(
            enabled_self_collisions=True
        ),
    )
)


@configclass
class RangerCr10TeleopSimpleCollisionEnvCfg(RangerCr10TeleopEnvCfg):
    """Same task/rewards/actions as teleop, but with the independent simple-collision asset."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot = SIMPLE_RANGER_CR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Independent reward distance source: the PhysX-consistent B proxy over the 44
        # declared active pairs. The formula and weight are the existing quadratic margin
        # penalty; the independent task sets the 20 mm margin explicitly. The default
        # teleop `safety.py` term remains untouched.
        self.rewards.arm_sensor_clearance = RewTerm(
            func=phys_proxy_clearance_penalty,
            weight=2.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "margin": 0.02, "cap": 0.15},
        )
