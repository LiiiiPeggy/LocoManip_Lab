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

"""Independent test config: simple arm collision boxes + solid box_link collider.

This is a candidate/test configuration only. The default `RangerCr10TeleopEnvCfg` and its
`ranger_cr10.usd` are untouched. The candidate USD is built from
`assets/ranger_cr10_simple/ranger_cr10_simple.urdf`; Link1--Link6 collision is a single oriented
box per link, and the box_link platform has a solid box collider. Self-collisions are enabled for
this candidate so the physical blocking can be measured; no PPO is run.
"""

from __future__ import annotations

import os

from isaaclab.utils import configclass

from LeggedManip_Lab.assets.ranger_cr10.ranger_cr10_articulation_cfg import RANGER_CR10_CFG
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.teleop_env_cfg import (
    RangerCr10TeleopEnvCfg,
)

_SIMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../assets/ranger_cr10_simple")
)
SIMPLE_USD = os.path.join(_SIMPLE_DIR, "ranger_cr10_simple_flat.usd")

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
