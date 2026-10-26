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

"""Stage 4.5g, variant B3: break the shortcut that let the actor ignore the command.

Measured on real supervision states, changing only the commanded roll by d moves the actor's
FK attitude by nothing, so its error grows as 3.1 / 6.1 / 12.2 deg for d = 0.05 / 0.10 / 0.20
rad -- exactly d. The command channel is dead. It can be dead and still fit the data to
0.005 rad because in every rollout the arm is already where the command put it: the joint
configuration alone predicts the label, so the command adds nothing and nothing in the loss
asks for it.

The fix is to break that correlation rather than to add more of the same data. Take a real
state (its base motion, gravity, velocity command and *commanded pose* -- everything except the
arm), replace the arm configuration with a random one, and relabel with local IK. Now the joint
configuration carries no information about the answer and the only way to fit is to read the
command.

The observations are still valid: every policy term except `joint_pos`, `joint_vel` and
`actions` describes the base, not the arm. Only those three are rewritten, and the joint
velocity of a settled pose is zero.

Usage:
    python scripts/ranger_cr10/teleop_ik_decorrelate.py --out logs/rsl_rl/ranger_cr10_teleop/_stage45g
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

parser = argparse.ArgumentParser(description="Decorrelate arm configuration from command, then refit.")
parser.add_argument("--base", default="logs/ik_teacher/dataset.pt")
parser.add_argument("--states", default="logs/rsl_rl/ranger_cr10_teleop/_stage45g/aug_dataset.pt",
                    help="real (obs, command) pairs whose base state and command are reused")
parser.add_argument("--init", default="logs/rsl_rl/ranger_cr10_teleop/_stage45g/actor_aug.pt")
parser.add_argument("--out", default="logs/rsl_rl/ranger_cr10_teleop/_stage45g")
parser.add_argument("--n_states", type=int, default=200)
parser.add_argument("--per_state", type=int, default=12)
parser.add_argument("--spread", type=float, default=0.35, help="rad, uniform around the solution")
parser.add_argument("--epochs", type=int, default=600)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--tag", default="decor")
parser.add_argument("--random_cmds", type=int, default=0,
                    help="instead of reusing the states' commands, draw this many *new* commands "
                         "uniformly from the training box -- the 213 commands the env's RNG actually "
                         "produces are all the old datasets contain, which is why their held-out sets "
                         "looked accurate")
parser.add_argument("--postures_per_cmd", type=int, default=1)
parser.add_argument("--fresh", action="store_true", help="train from scratch instead of from --init")
parser.add_argument("--old_weight", type=int, default=1,
                    help="0 trains on the decorrelated set only; the old rollouts correlate q with "
                         "the command and let the network fit without reading it")
parser.add_argument("--steps", type=int, default=0, help="optimiser steps (0 = use --epochs)")
args_cli = parser.parse_args()

OUT = Path(args_cli.out)
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT = np.asarray(tk.DEFAULT_Q, dtype=np.float64)
SCALE = 0.25
JP, JV, ACT = slice(18, 36), slice(36, 54), slice(54, 78)

print("=" * 92)
print("1. 在同一真实命令下换掉手臂位形, 用 local_ik 重标注 (打断 q 与命令的相关性)")
print("=" * 92)
states = torch.load(args_cli.states, map_location="cpu", weights_only=False)
Xs = states["X"].numpy()
rng = np.random.default_rng(args_cli.seed)
pick = rng.choice(len(Xs), min(args_cli.n_states, len(Xs)), replace=False)

Q_DEFAULT = np.array([0.032072, -0.875128, 0.469424, -0.112978])
P_DEFAULT = np.asarray(tk.P_NEUTRAL)


def random_command(rng_):
    rpy = rng_.uniform([-0.40, -0.40, -0.50], [0.40, 0.40, 0.50])
    off = rng_.uniform([-0.12, -0.15, -0.12], [0.12, 0.15, 0.10])
    dR = tk.euler_R(rpy)
    q = tk.R_to_quat(dR @ tk.R_NEUTRAL)
    return P_DEFAULT + off, q

obs_out, lab_out, dropped = [], [], 0
near = []
jobs = []
if args_cli.random_cmds:
    for k in range(args_cli.random_cmds):
        pos, quat = random_command(rng)
        jobs.append((Xs[pick[k % len(pick)]], np.concatenate([pos, quat])))
else:
    jobs = [(Xs[i], Xs[i][87:108][-7:]) for i in pick]

for o0, cmd in jobs:
    cmd_pos, cmd_R = cmd[:3], tk.quat_to_R(cmd[3:])
    q_sol0, ep, er, _, _ = tk.local_ik(np.asarray(DEFAULT), cmd_pos, cmd_R)
    if ep > 1e-3 or er > 1e-3:
        continue
    for _ in range(args_cli.postures_per_cmd if args_cli.random_cmds else args_cli.per_state):
        q = np.clip(q_sol0 + rng.uniform(-args_cli.spread, args_cli.spread, 6), tk.LIMITS[:, 0], tk.LIMITS[:, 1])
        q_ik, ep, er, _, _ = tk.local_ik(q, cmd_pos, cmd_R)
        if ep > 1e-3 or er > 1e-3:
            dropped += 1
            continue
        o = o0.copy()
        o[87:108] = np.tile(cmd, 3)                           # the command this sample is labelled for
        o[JP] = np.tile(q - DEFAULT, 3)                       # the arm as it is now
        o[JV] = 0.0                                           # a settled configuration
        act = np.concatenate([[0.0, 0.0], (q_ik - DEFAULT) / SCALE])
        o[ACT] = np.tile(act, 3)                              # a consistent previous action
        obs_out.append(o.astype(np.float32))
        lab_out.append(((q_ik - DEFAULT) / SCALE).astype(np.float32))
        T0, T1 = tk.fk(q), tk.fk(q_ik)
        near.append(math.degrees(np.linalg.norm(tk.log_R(T1[:3, :3] @ T0[:3, :3].T))))

lab = np.stack(lab_out)
X_new = torch.tensor(np.stack(obs_out), device=DEV)
Y_new = torch.cat([torch.zeros(len(lab), 2, device=DEV), torch.tensor(lab, device=DEV)], dim=1)
near = np.array(near)
print(f"  {'随机新命令 ' + str(args_cli.random_cmds) + ' 个' if args_cli.random_cmds else str(len(pick)) + ' 个真实命令'}"
      f" -> {len(obs_out)} 条 (丢弃 {dropped})")
print(f"  这些样本'当前姿态距 IK 解': 中位 {np.median(near):.1f}°, 范围 {near.min():.1f}-{near.max():.1f}°")
print("  -> 关节位形与标签不再相关, 唯一能解释标签的是命令")

# ---------------------------------------------------------------- merge + fit -----------
data = torch.load(args_cli.base, map_location="cpu", weights_only=False)
X_old, Y_old = data["X"], data["Y"]
tr, va, te = data["tr"], data["va"], data["te"]
if args_cli.old_weight:
    Xtr = torch.cat([X_old[tr].to(DEV)] * args_cli.old_weight + [X_new, X_new], dim=0)
    Ytr = torch.cat([Y_old[tr].to(DEV)] * args_cli.old_weight + [Y_new, Y_new], dim=0)
    Xva = torch.cat([X_old[va].to(DEV), X_new[: len(X_new) // 5]], dim=0)
    Yva = torch.cat([Y_old[va].to(DEV), Y_new[: len(Y_new) // 5]], dim=0)
else:
    Xtr, Ytr = X_new, Y_new
    Xva, Yva = X_new[: len(X_new) // 5], Y_new[: len(Y_new) // 5]
Xte, Yte = X_old[te].to(DEV), Y_old[te].to(DEV)


def make_net():
    torch.manual_seed(args_cli.seed)
    n = torch.nn.Sequential(
        torch.nn.Linear(117, 512), torch.nn.ELU(), torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 8)).to(DEV)
    if args_cli.fresh:
        return n
    ck = torch.load(args_cli.init, map_location="cpu", weights_only=False)
    sd = ck["actor_state_dict"] if isinstance(ck, dict) and "actor_state_dict" in ck else ck
    n.load_state_dict({(k[4:] if k.startswith("mlp.") else k): v for k, v in sd.items()
                       if k.split(".")[-1] in ("weight", "bias")})
    return n


def cmd_sensitivity(net, n=60):
    """The measurement that matters: sweep the command at a fixed state, report the error."""
    outs = []
    for dr in (0.1, 0.2):
        errs = []
        for i in range(0, len(X_new), max(1, len(X_new) // n)):
            o = X_new[i].cpu().numpy().copy()
            cmd = o[87:108][-7:]
            dq = np.array([math.cos(dr / 2), math.sin(dr / 2), 0, 0])
            a, b = dq, cmd[3:]
            qc = np.array([a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3], a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
                           a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1], a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]])
            o[87:108] = np.tile(np.concatenate([cmd[:3], qc]), 3)
            with torch.inference_mode():
                act = net(torch.tensor(o, dtype=torch.float32, device=DEV).unsqueeze(0))[0, 2:].cpu().numpy()
            Rc = tk.quat_to_R(qc)
            Rp = tk.fk(DEFAULT + SCALE * act)[:3, :3]
            errs.append(math.degrees(np.linalg.norm(tk.log_R(Rc @ Rp.T))))
        outs.append(float(np.median(errs)))
    return outs


def offline(net):
    with torch.inference_mode():
        pred = net(Xte)[:, 2:]
    return float(torch.mean(torch.abs(pred - Yte[:, 2:])) * SCALE)


print("\n" + "=" * 92)
print("2. 重新拟合")
print("=" * 92)
net = make_net()
print(f"  训练前: 原测试集关节 MAE {offline(net):.4f} rad, 命令灵敏度(Δroll=0.1/0.2) "
      f"{cmd_sensitivity(net)[0]:.1f}° / {cmd_sensitivity(net)[1]:.1f}°")
opt = torch.optim.Adam(net.parameters(), lr=(1e-3 if args_cli.fresh else 2e-4))
best_v, best_s, patience = float("inf"), None, 0
for epoch in range(args_cli.steps or args_cli.epochs):
    net.train()
    perm = torch.randperm(len(Xtr), device=DEV)
    for b in range(0, len(perm), 256):
        mb = perm[b:b + 256]
        opt.zero_grad()
        torch.mean((net(Xtr[mb])[:, 2:] - Ytr[mb][:, 2:]) ** 2).backward()
        opt.step()
    net.eval()
    with torch.inference_mode():
        v = float(torch.mean((net(Xva)[:, 2:] - Yva[:, 2:]) ** 2))
    if v < best_v - 1e-9:
        best_v, best_s, patience = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
    else:
        patience += 1
        if patience > 60:
            break
net.load_state_dict(best_s)
print(f"  早停于 epoch {epoch}, val MSE {best_v:.5f}")
sens = cmd_sensitivity(net)
print(f"  训练后: 原测试集关节 MAE {offline(net):.4f} rad, 命令灵敏度 {sens[0]:.1f}° / {sens[1]:.1f}°")
print("  (灵敏度 = 只改指令、状态不变时, actor 目标姿态相对新指令的误差; 越小说明命令被真正使用)")

torch.save(net.state_dict(), OUT / f"actor_{args_cli.tag}.pt")
with open(OUT / f"{args_cli.tag}_stats.json", "w") as f:
    json.dump(dict(n=len(obs_out), dropped=dropped, val_mse=best_v,
                   offline=offline(net), sens=sens), f, indent=1)
print(f"\n已写出 {OUT / f'actor_{args_cli.tag}.pt'}")
