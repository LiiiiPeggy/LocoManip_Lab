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

"""One-env check that the AG95 stays at its fixed, non-actuated pose.

The policy action contains only the six CR10 arm joints and the two base channels.  The gripper
is a fixed observation device: the driven joint target is 0, the mimic linkage is held at the
closed pose, and the B proxy uses fixed finger shapes.  This check reports the actual reset and
steady-state joint positions; it does not add gripper actions or dynamics.

Run:
    python scripts/ranger_cr10/check_gripper_fixed.py
"""

from __future__ import annotations

import json
import os
import sys

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import LeggedManip_Lab.tasks  # noqa: E402
from LeggedManip_Lab.assets.ranger_cr10.ranger_cr10_articulation_cfg import (  # noqa: E402
    GRIPPER_JOINT_NAMES,
    GRIPPER_MIMIC_JOINT_NAMES,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.simple_collision_env_cfg import (  # noqa: E402
    RangerCr10TeleopSimpleCollisionEnvCfg,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "outputs/stage413/b/long_ppo/gripper_fixed_check.json")
NAMES = list(GRIPPER_JOINT_NAMES) + list(GRIPPER_MIMIC_JOINT_NAMES)

cfg = RangerCr10TeleopSimpleCollisionEnvCfg()
cfg.seed = 42
cfg.scene.num_envs = 1
cfg.observations.policy.enable_corruption = False
torch.manual_seed(42)
np.random.seed(42)
env = gym.make("RANGER-CR10-TELEOP-SIMPLECOLL", cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
indices = [robot.joint_names.index(name) for name in NAMES]


def read() -> dict:
    actual = robot.data.joint_pos[:, indices].detach().cpu().numpy()[0]
    target = robot.data.joint_pos_target[:, indices].detach().cpu().numpy()[0]
    return {name: {"actual": float(actual[i]), "target": float(target[i])} for i, name in enumerate(NAMES)}


env.reset()
reset = read()
reset_step = int(u.episode_length_buf[0].item())
zero = torch.zeros(1, u.action_manager.total_action_dim, device=u.device)
steady = []
with torch.no_grad():
    for _ in range(100):
        _, _, terminated, truncated, _ = u.step(zero)
        steady.append(read())
        if bool((terminated | truncated)[0].item()):
            break
actual = np.array([[row[name]["actual"] for name in NAMES] for row in steady])
target = np.array([[row[name]["target"] for name in NAMES] for row in steady])
summary = {
    "reset_step": reset_step,
    "steps": len(steady),
    "action_dim": int(u.action_manager.total_action_dim),
    "gripper_in_action": False,
    "names": NAMES,
    "reset": reset,
    "steady_min": {name: float(actual[:, i].min()) for i, name in enumerate(NAMES)},
    "steady_max": {name: float(actual[:, i].max()) for i, name in enumerate(NAMES)},
    "target_min": {name: float(target[:, i].min()) for i, name in enumerate(NAMES)},
    "target_max": {name: float(target[:, i].max()) for i, name in enumerate(NAMES)},
    "max_abs_actual": float(np.abs(actual).max()),
    "note": "B proxy uses fixed AG95 shapes at the authored closed pose; this measures whether "
            "the Isaac gripper departs from that fixed pose in reset/steady state.",
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as handle:
    json.dump(summary, handle, indent=1)
print(json.dumps(summary, indent=1), flush=True)
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
