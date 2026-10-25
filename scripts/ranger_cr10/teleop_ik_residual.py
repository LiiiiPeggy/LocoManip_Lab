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

"""Stage 4.5g: predict a *correction* to the current arm configuration, not an absolute one.

Every other variant here regresses the absolute action `(q_ik - q_default) / scale`. 4.5g
measured what that formulation does in closed loop: at the state the loop settles in -- the
arm already at the commanded pose -- the output is indifferent to the command (changing the
commanded roll by d leaves the target attitude wrong by ~d), so the actor drifts to a posture
that is not the solution and stops there.

The correction target cannot degenerate that way: where the arm is already correct the target
is zero, and a network that outputs zero stays put. The correction is also small and smooth
where the absolute configuration is not, and it is what a feedback controller needs.

    a = a_now + corr,   a_now = (q_now - q_default) / scale, taken from the observation

The action interface is unchanged: the environment still receives an 8-D absolute action.

Usage:
    python scripts/ranger_cr10/teleop_ik_residual.py --out logs/rsl_rl/ranger_cr10_teleop/_stage45g
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teleop_kinematics as tk  # noqa: E402

parser = argparse.ArgumentParser(description="Residual (correction) parameterisation.")
parser.add_argument("--sets", nargs="+", default=["logs/ik_teacher/dataset.pt",
                                                  "logs/rsl_rl/ranger_cr10_teleop/_stage45g/aug_dataset.pt"])
parser.add_argument("--out", default="logs/rsl_rl/ranger_cr10_teleop/_stage45g")
parser.add_argument("--epochs", type=int, default=800)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--tag", default="resid")
args_cli = parser.parse_args()

OUT = Path(args_cli.out)
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT = np.asarray(tk.DEFAULT_Q, dtype=np.float64)
SCALE = 0.25
JP = slice(18, 36)

Xs, Ys = [], []
for k, path in enumerate(args_cli.sets):
    d = torch.load(path, map_location="cpu", weights_only=False)
    X, Y = d["X"], d["Y"]
    # the second set is only used for training (its own split has no held-out commands)
    Xs.append(X)
    Ys.append(Y)
X = torch.cat(Xs, dim=0)
Y = torch.cat(Ys, dim=0)
print(f"合并 {len(X)} 条样本: {[len(x) for x in Xs]}")

a_now = X[:, 18:24] / SCALE               # joint_pos frame 0 is (q - default) in rad; the
                                          # action unit is rad / SCALE, so divide -- mixing the
                                          # two is a factor-4 error in every correction
corr = Y[:, 2:] - a_now                  # the correction the actor must learn
print(f"修正量 |corr| 中位 {float(corr.abs().max(dim=1).values.median()):.3f} "
      f"(绝对值 |a| 中位 {float(Y[:, 2:].abs().max(dim=1).values.median()):.3f})")

n_va = len(X) // 10
perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(args_cli.seed))
va, tr = perm[:n_va], perm[n_va:]
Xtr, Ytr, Ctr = X[tr].to(DEV), Y[tr].to(DEV), corr[tr].to(DEV)
Xva, Yva, Cva = X[va].to(DEV), Y[va].to(DEV), corr[va].to(DEV)
AN_va = a_now[va].to(DEV)


def make():
    torch.manual_seed(args_cli.seed)
    return torch.nn.Sequential(
        torch.nn.Linear(117, 512), torch.nn.ELU(), torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 8)).to(DEV)


net = make()
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
best_v, best_s, patience = float("inf"), None, 0
for epoch in range(args_cli.epochs):
    net.train()
    order = torch.randperm(len(Xtr), device=DEV)
    for b in range(0, len(order), 256):
        mb = order[b:b + 256]
        opt.zero_grad()
        torch.mean((net(Xtr[mb])[:, 2:] - Ctr[mb]) ** 2).backward()
        opt.step()
    net.eval()
    with torch.inference_mode():
        v = float(torch.mean((net(Xva)[:, 2:] - Cva) ** 2))
    if v < best_v - 1e-9:
        best_v, best_s, patience = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
    else:
        patience += 1
        if patience > 80:
            break
net.load_state_dict(best_s)
with torch.inference_mode():
    pred = net(Xva)[:, 2:] + AN_va
    jm = float(torch.mean(torch.abs(pred - Yva[:, 2:])) * SCALE)
    dp, da = [], []
    for k in range(0, len(Xva), 4):
        T = tk.fk(DEFAULT + SCALE * pred[k].cpu().numpy())
        Tt = tk.fk(DEFAULT + SCALE * Yva[k, 2:].cpu().numpy())
        dp.append(np.linalg.norm(T[:3, 3] - Tt[:3, 3]))
        da.append(math.degrees(np.linalg.norm(tk.log_R(Tt[:3, :3] @ T[:3, :3].T))))
print(f"早停于 epoch {epoch}, val MSE {best_v:.5f}")
print(f"验证集: 关节 MAE {jm:.4f} rad, FK 姿态 {np.mean(da):.1f}°, FK 位置 {np.mean(dp) * 1000:.1f} mm")

torch.save(net.state_dict(), OUT / f"actor_{args_cli.tag}.pt")
with open(OUT / f"{args_cli.tag}_stats.json", "w") as f:
    json.dump(dict(n=len(X), val_mse=best_v, joint_mae=jm, fk_att=float(np.mean(da)),
                   fk_pos=float(np.mean(dp))), f, indent=1)
print(f"已写出 {OUT / f'actor_{args_cli.tag}.pt'}  (推理时 a = a_now + net(obs)[:, 2:])")
