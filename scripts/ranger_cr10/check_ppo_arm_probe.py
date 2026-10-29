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

"""Stage 4.5j: did PPO keep the arm map? A 30-second, no-simulator probe.

The TensorBoard task metrics are episode averages over the training distribution, so they
cannot say whether the *deterministic* policy still reaches the commanded pose -- the thing
the fixed protocol measures but at 6 minutes per checkpoint. This runs the cheap half of that
question: on real observations from `logs/ik_teacher_box`, at states where the arm is already
on target and on the fresh-command probe set from 4.5i, what is `FK(q_des)` against the
command?

Reference values (stage 4.5i, same construction):
    B0                  21.5 deg      mix11      0.66 deg
A checkpoint that reads ~2-5 deg has lost precision but kept the channel; ~20 deg means the
command map is gone and the PPO run is not going to win it back by itself.

Usage:
    python scripts/ranger_cr10/check_ppo_arm_probe.py --ckpts <ckpt1> <ckpt2> ...
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teleop_kinematics as tk  # noqa: E402

parser = argparse.ArgumentParser(description="Offline arm-precision probe for PPO checkpoints.")
parser.add_argument("--ckpts", nargs="+", required=True)
parser.add_argument("--data", default="logs/ik_teacher_box/dataset.pt")
parser.add_argument("--n", type=int, default=200)
args_cli = parser.parse_args()

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT = np.asarray(tk.DEFAULT_Q, dtype=np.float64)
SCALE = 0.25
PC, JP, JV, ACT = slice(87, 108), slice(18, 36), slice(36, 54), slice(54, 78)
QD = np.array([0.032072, -0.875128, 0.469424, -0.112978])


def load_actor(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["actor_state_dict"] if isinstance(ck, dict) and "actor_state_dict" in ck else ck
    sd = {(k[4:] if k.startswith("mlp.") else k): v for k, v in sd.items() if k.split(".")[-1] in ("weight", "bias")}
    net = torch.nn.Sequential(
        torch.nn.Linear(117, 512),
        torch.nn.ELU(),
        torch.nn.Linear(512, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 8),
    ).to(DEV)
    net.load_state_dict(sd)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


d = torch.load(args_cli.data, map_location="cpu", weights_only=False)
X = d["X"].numpy().astype(np.float64)
cmd_all = d["cmd"].numpy().astype(np.float64) if "cmd" in d else X[:, PC][:, -7:]
rng = np.random.default_rng(0)
idx = rng.choice(len(X), min(args_cli.n, len(X)), replace=False)

# (a) real states, command as observed -- the label's own pose is the reference
rows = X[idx]
cmds = cmd_all[idx]
# (b) probe: fresh commands (independent axes) at an exactly-on-target posture
probe_X, probe_cmd = [], []
tries = 0
while len(probe_X) < 128 and tries < 4000:
    tries += 1
    rpy = rng.uniform(np.array(tk.TRAIN_BOX_RPY)[:, 0], np.array(tk.TRAIN_BOX_RPY)[:, 1])
    pos = np.asarray(tk.P_NEUTRAL) + rng.uniform(np.array(tk.TRAIN_BOX_POS)[:, 0], np.array(tk.TRAIN_BOX_POS)[:, 1])
    a = tk.R_to_quat(tk.euler_R(rpy))
    b = QD
    qc = np.array(
        [
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ]
    )
    q0, ep, er, _, _ = tk.local_ik(np.asarray(DEFAULT), pos, tk.quat_to_R(qc))
    if ep > 1e-3 or er > 1e-3:
        continue
    o = X[rng.integers(len(X))].copy()
    o[PC] = np.tile(np.concatenate([pos, qc]), 3)
    o[JP] = np.tile(q0 - DEFAULT, 3)
    o[JV] = 0.0
    o[ACT] = np.tile(np.concatenate([[0.0, 0.0], (q0 - DEFAULT) / SCALE]), 3)
    probe_X.append(o)
    probe_cmd.append(np.concatenate([pos, qc]))
probe_X = torch.tensor(np.stack(probe_X), dtype=torch.float32, device=DEV)
probe_cmd = np.stack(probe_cmd)


def att_deg(q_action, cmd):
    T = tk.fk(DEFAULT + SCALE * q_action)
    Rc = tk.quat_to_R(cmd[3:])
    return math.degrees(np.linalg.norm(tk.log_R(Rc @ T[:3, :3].T)))


Xr = torch.tensor(rows, dtype=torch.float32, device=DEV)
print(f"{'checkpoint':<52}{'训练状态':>12}{'新命令+到位':>14}{'位置(新命令)':>14}")
print("-" * 94)
for path in args_cli.ckpts:
    net = load_actor(path)
    with torch.inference_mode():
        a = net(Xr)[:, 2:].cpu().numpy()
        b = net(probe_X)[:, 2:].cpu().numpy()
    e1 = [att_deg(a[k], cmds[k]) for k in range(0, len(a), 2)]
    e2, dp = [], []
    for k in range(len(b)):
        e2.append(att_deg(b[k], probe_cmd[k]))
        dp.append(np.linalg.norm(tk.fk(DEFAULT + SCALE * b[k])[:3, 3] - probe_cmd[k][:3]))
    name = Path(path).parent.name + "/" + Path(path).name
    print(f"{name[-52:]:<52}{np.median(e1):>10.2f}°{np.mean(e2):>12.2f}°{np.mean(dp) * 1000:>12.1f} mm")
