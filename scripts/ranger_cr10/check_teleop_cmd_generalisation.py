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

"""Stage 4.5h: can this network learn the command -> joint-solution map at all?

Two parts, both offline (no simulator):

A. Split the existing supervision by *command* instead of by episode, and measure the actor
   on commands it has never seen. 4.5g showed the episode split shares commands with training
   (median distance 0.00 deg), so every offline number so far was a training number.

B. The minimal experiment. Fix one real base state, sample a handful of commands and, for each,
   a handful of arm postures, label with local IK, and fit the same architecture. Then test
   interpolation in the *posture* dimension (same commands) and in the *command* dimension
   (same postures). If the first works and the second does not, the command is not being
   learned, and no amount of extra data of the old kind can fix that. The variants then ask
   whether the *encoding* is the reason: the commanded quaternion has two valid signs for one
   rotation, and the data contains both (3% negative), so the map the network sees is
   discontinuous where it should be smooth.

Usage:
    python scripts/ranger_cr10/check_teleop_cmd_generalisation.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teleop_kinematics as tk  # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT = np.asarray(tk.DEFAULT_Q, dtype=np.float64)
SCALE = 0.25
PC, JP = slice(87, 108), slice(18, 36)
Q_DEFAULT = np.array([0.032072, -0.875128, 0.469424, -0.112978])
R_N = tk.R_NEUTRAL


def load_actor(path: str):
    n = torch.nn.Sequential(
        torch.nn.Linear(117, 512), torch.nn.ELU(), torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 8)).to(DEV)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["actor_state_dict"] if isinstance(ck, dict) and "actor_state_dict" in ck else ck
    n.load_state_dict({(k[4:] if k.startswith("mlp.") else k): v for k, v in sd.items()
                       if k.split(".")[-1] in ("weight", "bias")})
    n.eval()
    return n


def new_net(init=None):
    torch.manual_seed(0)
    n = torch.nn.Sequential(
        torch.nn.Linear(117, 512), torch.nn.ELU(), torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 8)).to(DEV)
    if init is not None:
        n.load_state_dict(init)
    return n


def cmd_of(obs_row):
    return obs_row[PC][-7:]


def rot_angle(c1, c2):
    return math.degrees(np.linalg.norm(tk.log_R(tk.quat_to_R(c2[3:]) @ tk.quat_to_R(c1[3:]).T)))


def encode(obs, mode):
    """Rewrite the commanded quaternion (all three history frames) under one encoding."""
    o = obs.copy()
    cmd = o[PC][-7:]
    pos, q = cmd[:3], cmd[3:]
    if mode == "canonical":
        q = q if q[0] >= 0 else -q
        new = np.concatenate([pos, q])
    elif mode == "rotvec":
        rv = tk.log_R(tk.quat_to_R(q))
        new = np.concatenate([pos, rv, [0.0]])
    else:
        new = cmd
    o[PC] = np.tile(new, 3)
    return o


def fk_att_error(a6, cmd):
    T = tk.fk(DEFAULT + SCALE * np.asarray(a6))
    Rc = tk.quat_to_R(cmd[3:])
    return math.degrees(np.linalg.norm(tk.log_R(Rc @ T[:3, :3].T)))


# ======================================================================================
print("=" * 96)
print("A. 按命令划分评价: Actor 在'没见过的命令'上是什么水平")
print("=" * 96)
d = torch.load("logs/ik_teacher/dataset.pt", map_location="cpu", weights_only=False)
X, Y = d["X"].numpy(), d["Y"].numpy()
tr, te = d["tr"].numpy(), d["te"].numpy()
cmds_all = np.stack([cmd_of(X[i]) for i in range(len(X))])
# greedy grouping: two samples share a command group when the commanded pose is within
# 2 deg / 5 mm -- tighter than the box spacing, so groups are genuinely distinct commands
grp = np.full(len(X), -1)
reps = []
for i in range(len(X)):
    for g, j in enumerate(reps):
        if rot_angle(cmds_all[i], cmds_all[j]) < 2.0 and np.linalg.norm(cmds_all[i][:3] - cmds_all[j][:3]) < 0.005:
            grp[i] = g
            break
    else:
        grp[i] = len(reps)
        reps.append(i)
n_g = len(reps)
train_g = set(grp[tr].tolist())
test_g = set(grp[te].tolist())
print(f"  命令组 {n_g} 个; 训练 {len(train_g)}, 测试 {len(test_g)}, "
      f"两者重叠 {len(train_g & test_g)} 个 (旧划分)")
rng = np.random.default_rng(0)
order = rng.permutation(n_g)
hold_g = set(order[: max(1, n_g // 5)].tolist())          # 20% of commands, never trained on
seen_idx = [i for i in range(len(X)) if grp[i] not in hold_g]
unseen_idx = [i for i in range(len(X)) if grp[i] in hold_g]
print(f"  命令不相交划分: 见 {len(seen_idx)} 条 / 未见 {len(unseen_idx)} 条 "
      f"({len(hold_g)} 个命令被整个扣掉)")

for name, path in (("B0 (4.5e)", "logs/rsl_rl/ranger_cr10_teleop/_stage45e/pretrained_B.pt"),
                   ("T (教师)", "logs/ik_teacher/teacher.pt"),
                   ("resid (4.5g)", "logs/rsl_rl/ranger_cr10_teleop/_stage45g/actor_resid.pt")):
    try:
        net = load_actor(path)
    except FileNotFoundError:
        print(f"  {name}: 缺失"); continue
    if "resid" in name:
        continue          # 残差版需要额外的 a_now，单独处理没有意义，下面只比较绝对参数化
    with torch.inference_mode():
        out = {}
        for tag, idx in (("见过命令", seen_idx), ("未见命令", unseen_idx)):
            pred = net(torch.tensor(X[idx], dtype=torch.float32, device=DEV))[:, 2:].cpu().numpy()
            jm = float(np.mean(np.abs(pred - Y[idx][:, 2:])) * SCALE)
            errs = [fk_att_error(pred[k], cmds_all[idx[k]]) for k in range(0, len(idx), 3)]
            out[tag] = (jm, float(np.mean(errs)))
    print(f"  {name:<12} 见过: 关节 {out['见过命令'][0]:.4f} rad, FK 姿态 {out['见过命令'][1]:5.1f}°   |   "
          f"未见: 关节 {out['未见命令'][0]:.4f} rad, FK 姿态 {out['未见命令'][1]:5.1f}°")

# ======================================================================================
print("\n" + "=" * 96)
print("B. 最小实验: 少量命令 x 少量位形, 能否学会'命令 -> 关节解', 以及命令维度的泛化")
print("=" * 96)
base = X[np.random.default_rng(1).choice(len(X))].copy()
base[PC] = np.tile(np.concatenate([np.asarray(tk.P_NEUTRAL), Q_DEFAULT]), 3)


def sample_samples(cmds_rpy, postures, rng_):
    Xs, Ys, kept_cmd = [], [], []
    for i, rpy in enumerate(cmds_rpy):
        dq = tk.R_to_quat(tk.euler_R(rpy))
        # the environment composes the offset *pre-multiplied* onto the default quaternion
        a = dq
        b = Q_DEFAULT
        qc = np.array([a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3], a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
                       a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1], a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]])
        q_ik0, ep, er, _, _ = tk.local_ik(DEFAULT, np.asarray(tk.P_NEUTRAL), tk.quat_to_R(qc))
        if ep > 1e-3 or er > 1e-3:
            continue
        for _ in range(postures):
            q = np.clip(q_ik0 + rng_.uniform(-0.30, 0.30, 6), tk.LIMITS[:, 0], tk.LIMITS[:, 1])
            q_ik, ep, er, _, _ = tk.local_ik(q, np.asarray(tk.P_NEUTRAL), tk.quat_to_R(qc))
            if ep > 1e-3 or er > 1e-3:
                continue
            o = base.copy()
            o[PC] = np.tile(np.concatenate([np.asarray(tk.P_NEUTRAL), qc]), 3)
            o[JP] = np.tile(q - DEFAULT, 3)
            o[36:54] = 0.0
            o[54:78] = np.tile(np.concatenate([[0, 0], (q_ik - DEFAULT) / SCALE]), 3)
            Xs.append(o.astype(np.float32))
            # the actor emits 8 numbers; the two chassis channels are not supervised here
            Ys.append(np.concatenate([np.zeros(2), (q_ik - DEFAULT) / SCALE]).astype(np.float32))
            kept_cmd.append(np.concatenate([np.asarray(tk.P_NEUTRAL), qc]))
    return np.stack(Xs), np.stack(Ys), np.stack(kept_cmd)


rng = np.random.default_rng(7)
n_cmd_train, n_cmd_test, n_post = 16, 16, 8
rpy_tr = rng.uniform([-0.35, -0.35, -0.45], [0.35, 0.35, 0.45], (n_cmd_train, 3))
rpy_te = rng.uniform([-0.35, -0.35, -0.45], [0.35, 0.35, 0.45], (n_cmd_test, 3))
Xtr, Ytr, Ctr = sample_samples(rpy_tr, n_post, np.random.default_rng(11))
Xte, Yte, Cte = sample_samples(rpy_te, n_post, np.random.default_rng(12))
print(f"  训练 {len(Xtr)} 条 ({n_cmd_train} 命令 x {n_post} 位形); 测试 {len(Xte)} 条 "
      f"({n_cmd_test} 个**新命令** x 新位形)")
signs = np.mean([c[3] < 0 for c in Ctr])
print(f"  训练命令里 w<0 的比例 {signs * 100:.0f}%  (同一个旋转有两种合法四元数, 数据里两种都有)")


def fit(Xtr_i, Ytr_i, steps=3000, lr=1e-3, init=None):
    net = new_net(init)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.tensor(Xtr_i, device=DEV)
    Yt = torch.tensor(Ytr_i, device=DEV)
    for s in range(steps):
        opt.zero_grad()
        loss = torch.mean((net(Xt)[:, 2:] - Yt[:, 2:]) ** 2)
        loss.backward()
        opt.step()
    net.eval()
    return net


def evaluate(net, Xs, Ys, Cs, tag, mode):
    Xe = torch.tensor(np.stack([encode(x, mode) for x in Xs]), device=DEV)
    with torch.inference_mode():
        pred = net(Xe)[:, 2:].cpu().numpy()
    jm = float(np.mean(np.abs(pred - Ys[:, 2:])) * SCALE)
    fk = float(np.mean([fk_att_error(pred[k], Cs[k]) for k in range(len(pred))]))
    print(f"    {tag:<22}关节 MAE {jm:.4f} rad   FK 姿态误差 {fk:5.1f}°")
    return jm, fk


for mode in ("raw", "canonical", "rotvec"):
    label = {"raw": "四元数原样", "canonical": "四元数规范化 (w>=0)", "rotvec": "旋转向量 (3 维, 仅诊断)"}[mode]
    print(f"\n  编码: {label}")
    net = fit(np.stack([encode(x, mode) for x in Xtr]), Ytr)
    evaluate(net, Xtr, Ytr, Ctr, "训练命令 + 新位形", mode)
    evaluate(net, Xte, Yte, Cte, "**新命令** + 新位形", mode)
    # command response at one fixed posture
    gains = []
    for k in (0, 1, 2):
        c = Ctr[k]
        q = DEFAULT + Xtr[k][18:24]
        for dr in (0.10, 0.20):
            o = encode(Xtr[k], mode).copy()
            dq = tk.R_to_quat(tk.euler_R([dr, 0, 0]))
            b = c[3:]
            qc = np.array([dq[0]*b[0]-dq[1]*b[1]-dq[2]*b[2]-dq[3]*b[3], dq[0]*b[1]+dq[1]*b[0]+dq[2]*b[3]-dq[3]*b[2],
                           dq[0]*b[2]-dq[1]*b[3]+dq[2]*b[0]+dq[3]*b[1], dq[0]*b[3]+dq[1]*b[2]-dq[2]*b[1]+dq[3]*b[0]])
            o[PC] = np.tile(np.concatenate([c[:3], qc]), 3)
            o = encode(o, mode)
            with torch.inference_mode():
                a = net(torch.tensor(o, dtype=torch.float32, device=DEV).unsqueeze(0))[0, 2:].cpu().numpy()
            gains.append(fk_att_error(a, np.concatenate([c[:3], qc])))
    print(f"    命令灵敏度 (只改 roll +0.1/+0.2): 中位 {np.median(gains):.1f}° "
          f"(理想 ~0°, 完全不响应 ≈ 5.7°/11.5°)")
