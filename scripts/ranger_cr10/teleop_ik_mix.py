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

"""Stage 4.5i: mixed real-rollout + decorrelated supervision, trained to convergence.

4.5h's two extremes both failed for different reasons: the all-real set lets the network read
the label off the joint configuration and never learn the command map, and the all-synthetic
set teaches the map but leaves the real states uncovered, so the closed loop walks out of
distribution and diverges. This trains the mixture at a controlled real:decorrelated sampling
ratio, with the budget the task asks for, and reports the loss curve instead of assuming
that a low training loss means the task is solved.

Data
----
real          `teleop_ik_collect.py` output: real observations, real three-frame history, base
              moving, label = local_ik(q_now, target) verified by FK.
decorrelated  the same real observations (base terms untouched) with the arm configuration
              replaced, and the command drawn independently of it. Three posture sources:
              the IK solution itself (on target), a random perturbation of it, and a point
              along the path from a *different* sample's real posture to the solution. Labels
              are always `local_ik(posture, command)`, so the pair is executable by
              construction and the posture carries no information about the command.

Splits are by *command group*, never by sample or episode: 4.5h measured that an episode
split leaves the test set 0.00 deg (median) from the training commands, which is why every
offline number before it was a training number. The test groups are not used for early
stopping or for any hyper-parameter.

Loss, stated exactly
--------------------
    L = mean_{arm}( (a_hat - a*) ** 2 ) + anchor * mean_{chassis}( (a_hat - a_ref) ** 2 )

The first term is the normalised-action MSE on the six arm channels (action units: the
executed joint target is `default + 0.25 * a`). The second is a chassis anchor on the *real*
part of the batch only: supervised data says nothing about `[vx, wz]`, so without it the
shared trunk drifts and the chassis output silently changes. `a_ref` is the frozen initial
actor (B0). Its weight is fixed for every group, and both terms are logged separately.

Usage:
    python scripts/ranger_cr10/teleop_ik_mix.py --ratio 2:1 --updates 20000 --tag mix21
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teleop_kinematics as tk  # noqa: E402

parser = argparse.ArgumentParser(description="Mixed real + decorrelated IK supervision.")
parser.add_argument("--real", default="logs/ik_teacher_box/dataset.pt")
parser.add_argument("--out", default="logs/rsl_rl/ranger_cr10_teleop/_stage45i")
parser.add_argument("--tag", default="mix21")
parser.add_argument("--ratio", default="2:1", help="real:decorrelated sampling ratio, e.g. 2:1")
parser.add_argument(
    "--init",
    default="logs/rsl_rl/ranger_cr10_teleop/_stage45e/pretrained_B.pt",
    help="initial weights; 'none' trains from scratch",
)
parser.add_argument("--updates", type=int, default=20000)
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--lr_final", type=float, default=1e-4)
parser.add_argument("--val_every", type=int, default=500)
parser.add_argument("--patience", type=int, default=4000, help="stop after this many updates without val improvement")
parser.add_argument("--anchor", type=float, default=0.02, help="weight of the chassis anchor loss")
parser.add_argument("--split", default="0.6,0.2,0.2")
parser.add_argument("--postures_per_cmd", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--smoke", type=int, default=0, help="run only this many updates (smoke test)")
args_cli = parser.parse_args()

OUT = Path(args_cli.out)
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT = np.asarray(tk.DEFAULT_Q, dtype=np.float64)
SCALE = 0.25
PC, JP, JV, ACT = slice(87, 108), slice(18, 36), slice(36, 54), slice(54, 78)
QD_DEFAULT = np.array([0.032072, -0.875128, 0.469424, -0.112978])
P_DEFAULT = np.asarray(tk.P_NEUTRAL)
TRAIN_RPY = np.array(tk.TRAIN_BOX_RPY)
TRAIN_POS = np.array(tk.TRAIN_BOX_POS)
REAL_POS_DEFAULT = np.array([0.5519, -0.0742, 0.4696])

real_w, decor_w = (float(v) for v in args_cli.ratio.split(":"))
W_REAL = real_w / (real_w + decor_w)
SPLIT = [float(v) for v in args_cli.split.split(",")]
N_UPDATES = args_cli.smoke or args_cli.updates

rng_data = np.random.default_rng(args_cli.seed)
gen = torch.Generator(device=DEV)
gen.manual_seed(args_cli.seed + 1000)


def quat_mul_np(a, b):
    return np.array(
        [
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ]
    )


def quat_from_euler_np(rpy):
    """Isaac Lab's ``quat_from_euler_xyz``: Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def command_quat(rpy):
    """Exactly what the env writes: offset pre-multiplied onto the default quaternion.

    The quaternion *components* are part of the observation, so the sign convention matters:
    a rotation matrix round-tripped through ``R_to_quat`` can come back antipodal, which is the
    same rotation and a completely different input row. Always compose the way the env does.
    """
    return quat_mul_np(quat_from_euler_np(rpy), QD_DEFAULT)


def random_command(rng):
    rpy = rng.uniform(TRAIN_RPY[:, 0], TRAIN_RPY[:, 1])
    off = rng.uniform(TRAIN_POS[:, 0], TRAIN_POS[:, 1])
    return P_DEFAULT + off, command_quat(rpy)


def fk_att_deg(q, cmd_quat):
    T = tk.fk(np.asarray(q, dtype=np.float64))
    Rc = tk.quat_to_R(np.asarray(cmd_quat, dtype=np.float64))
    return math.degrees(np.linalg.norm(tk.log_R(Rc @ T[:3, :3].T)))


def fk_pos_m(q, cmd_pos):
    return float(np.linalg.norm(tk.fk(np.asarray(q, dtype=np.float64))[:3, 3] - cmd_pos))


# =====================================================================================
print("=" * 100)
print(f"1. 数据: {args_cli.real}, 比例 真实:解相关 = {args_cli.ratio}")
print("=" * 100)
data = torch.load(args_cli.real, map_location="cpu", weights_only=False)
X_real = data["X"].numpy().astype(np.float64)
Y_real = data["Y"].numpy().astype(np.float64)
if Y_real.shape[1] == 6:  # sets written before 2026-10-29 stored the six arm dims only
    Y_real = np.concatenate([np.zeros((len(Y_real), 2)), Y_real], axis=1)
# older sets did not store the command separately; it is the last frame of `pos_commands`
CMD = (data["cmd"].numpy() if "cmd" in data else X_real[:, PC][:, -7:]).astype(np.float64)
print(f"  真实样本 {len(X_real)} 条")


# Group commands. Exact equality first (the env copies the same tensor into the observation,
# so equal commands are bit-identical), then merge near-duplicates: two draws on the same
# 2-D line can sit 0.1 deg apart, and if they land in different splits the "held-out command"
# is a training command with a rounding difference. 2 deg / 5 mm is 4.5h's grouping.
def group_rpy(c):
    off = tk.quat_to_R(c[3:]) @ tk.quat_to_R(QD_DEFAULT).T
    return np.degrees(tk.R_to_rpy(off))


keys = {}
gid = np.empty(len(CMD), dtype=np.int64)
uniq = []
for i, c in enumerate(CMD):
    k = tuple(np.round(c, 9).tolist())
    if k not in keys:
        keys[k] = len(keys)
        uniq.append(c)
    gid[i] = keys[k]
uniq = np.stack(uniq)
uniq_rpy = np.stack([group_rpy(c) for c in uniq])
n_u = len(uniq)
uc = np.full(n_u, -1, dtype=np.int64)  # unique command -> cluster
centres = []
for i in range(n_u):
    for g, j in enumerate(centres):
        if np.linalg.norm(uniq_rpy[i] - uniq_rpy[j]) < 2.0 and np.linalg.norm(uniq[i, :3] - uniq[j, :3]) < 0.005:
            uc[i] = g
            break
    else:
        uc[i] = len(centres)
        centres.append(i)
gid = uc[gid]
key_of = uniq[np.array(centres)]
n_g = len(key_of)
perm = rng_data.permutation(n_g)
n_tr = int(SPLIT[0] * n_g)
n_va = int(SPLIT[1] * n_g)
g_tr, g_va, g_te = set(perm[:n_tr].tolist()), set(perm[n_tr : n_tr + n_va].tolist()), set(perm[n_tr + n_va :].tolist())
m_tr = np.array([g in g_tr for g in gid])
m_va = np.array([g in g_va for g in gid])
m_te = np.array([g in g_te for g in gid])
print(f"  唯一命令 {n_u} 条 -> 合并为 {n_g} 组 (2°/5 mm)")
print(f"  命令组 -> 训练 {len(g_tr)} / 验证 {len(g_va)} / 测试 {len(g_te)}  (按命令整组划分)")
print(f"  样本     训练 {m_tr.sum()} / 验证 {m_va.sum()} / 测试 {m_te.sum()}")
# the property the split exists for: how far is a validation command from every training one
rpy_all = np.stack([group_rpy(c) for c in key_of])
d_va = [min(np.linalg.norm(rpy_all[g] - rpy_all[t]) for t in g_tr) for g in g_va]
d_te = [min(np.linalg.norm(rpy_all[g] - rpy_all[t]) for t in g_tr) for g in g_te]
print(
    f"  验证/测试命令到最近训练命令的姿态距离: 中位 {np.median(d_va):.1f}° / {np.median(d_te):.1f}°, "
    f"最小 {min(d_va):.1f}° / {min(d_te):.1f}°  (4.5h 的按 episode 划分是 0.00°)"
)


def build_decorrelated(cmds, n_out, base_rows, rng, postures_per_cmd):
    """Independent postures for fixed commands, labelled by local IK from that posture."""
    Xs, Ys, kept_cmd, near = [], [], [], []
    attempts = 0
    while len(Xs) < n_out and attempts < n_out * 20:
        attempts += 1
        ci = rng.integers(len(cmds))
        cmd = cmds[ci]
        base = base_rows[rng.integers(len(base_rows))].copy()
        # starting configuration: a real posture from another sample, then IK to this command
        q_start = base[JP][-6:] + DEFAULT
        mode = int(rng.integers(postures_per_cmd + 1))
        if mode == 0:
            q0, ep, er, _, _ = tk.local_ik(np.asarray(DEFAULT), cmd[:3], tk.quat_to_R(cmd[3:]))
            if ep > 1e-3 or er > 1e-3:
                continue
            q = q0.copy()  # exactly on target
        else:
            q_ref, ep, er, _, _ = tk.local_ik(q_start, cmd[:3], tk.quat_to_R(cmd[3:]))
            if ep > 1e-3 or er > 1e-3:
                continue
            if mode == 1:
                q = np.clip(q_ref + rng.uniform(-0.35, 0.35, 6), tk.LIMITS[:, 0], tk.LIMITS[:, 1])
            else:
                # along the path from another real posture to the solution: arrival states
                q_other = base_rows[rng.integers(len(base_rows))][JP][-6:] + DEFAULT
                a = rng.uniform(0.25, 0.85)
                q = np.clip(q_ref + a * (q_other - q_ref), tk.LIMITS[:, 0], tk.LIMITS[:, 1])
        q_ik, ep, er, _, _ = tk.local_ik(q, cmd[:3], tk.quat_to_R(cmd[3:]))
        if ep > 1e-3 or er > 1e-3:
            continue
        o = base.copy()
        o[PC] = np.tile(cmd, 3)
        o[JP] = np.tile(q - DEFAULT, 3)
        o[JV] = 0.0
        # the previous action is set from the *current* posture, not from the label: using the
        # label there would put the answer inside the observation and teach the trunk to copy
        # it, which is the same "hold where you are" shortcut in another disguise
        o[ACT] = np.tile(np.concatenate([[0.0, 0.0], (q - DEFAULT) / SCALE]), 3)
        Xs.append(o.astype(np.float32))
        Ys.append(np.concatenate([[0.0, 0.0], (q_ik - DEFAULT) / SCALE]).astype(np.float32))
        kept_cmd.append(cmd.copy())
        near.append(fk_att_deg(q, cmd[3:]))
    return np.stack(Xs), np.stack(Ys), np.stack(kept_cmd), np.array(near), attempts


cmds_of = {name: key_of[sorted(gs)] for name, gs in (("train", g_tr), ("val", g_va), ("test", g_te))}
n_decor_target = int(len(X_real[m_tr]) * (decor_w / real_w))
print(f"\n  解相关样本目标数 = 训练真实样本 x {decor_w / real_w:.2f} = {n_decor_target}")
decor = {}
for name in ("train", "val", "test"):
    rows = X_real[m_tr if name == "train" else (m_va if name == "val" else m_te)]
    n_target = n_decor_target if name == "train" else max(48, int(len(rows) * decor_w / real_w))
    if name != "train":
        n_target = min(n_target, 3 * len(cmds_of[name]))
    Xd, Yd, Cd, near, att = build_decorrelated(cmds_of[name], n_target, rows, rng_data, args_cli.postures_per_cmd)
    decor[name] = (Xd, Yd, Cd)
    print(
        f"  解相关 {name:<5}: {len(Xd)} 条 ({args_cli.postures_per_cmd} 种位形/命令轮回), "
        f"试了 {att} 次; 这些位形距 IK 解的中位 {np.median(near):.1f}°"
    )

# probe: fresh commands never used by any set, at an exactly-on-target posture. This is the
# measurement that separates "learned the command map" from "memorised the training rows":
# 4.5h measured 1.0 deg for a 112-sample decorrelated net and 18 deg for B0 here.
print("\n  探针集 (新采命令 + 恰好到位, 不参与训练/验证):")
probe_cmd, probe_X, probe_Y = [], [], []
while len(probe_X) < 128:
    pos, quat = random_command(rng_data)
    q0, ep, er, _, _ = tk.local_ik(np.asarray(DEFAULT), pos, tk.quat_to_R(quat))
    if ep > 1e-3 or er > 1e-3:
        continue
    base = X_real[rng_data.integers(len(X_real))].copy()
    o = base.copy()
    o[PC] = np.tile(np.concatenate([pos, quat]), 3)
    o[JP] = np.tile(q0 - DEFAULT, 3)
    o[JV] = 0.0
    o[ACT] = np.tile(np.concatenate([[0.0, 0.0], (q0 - DEFAULT) / SCALE]), 3)
    probe_X.append(o.astype(np.float32))
    probe_Y.append(np.concatenate([[0.0, 0.0], (q0 - DEFAULT) / SCALE]).astype(np.float32))
    probe_cmd.append(np.concatenate([pos, quat]))
probe_X = np.stack(probe_X)
probe_cmd = np.stack(probe_cmd)
# never used for training or model selection: reported only
d_probe = [min(np.linalg.norm(group_rpy(c) - rpy_all[t]) for t in g_tr) for c in probe_cmd]
print(f"    {len(probe_X)} 条; 到最近训练命令的姿态距离 中位 {np.median(d_probe):.1f}° (最小 {min(d_probe):.1f}°)")

# tensors
Rtr = torch.tensor(X_real[m_tr], dtype=torch.float32, device=DEV)
Rtr_Y = torch.tensor(Y_real[m_tr], dtype=torch.float32, device=DEV)
Rva = torch.tensor(X_real[m_va], dtype=torch.float32, device=DEV)
Rva_Y = torch.tensor(Y_real[m_va], dtype=torch.float32, device=DEV)
Rte = torch.tensor(X_real[m_te], dtype=torch.float32, device=DEV)
Rte_Y = torch.tensor(Y_real[m_te], dtype=torch.float32, device=DEV)
Dtr = torch.tensor(decor["train"][0], device=DEV)
Dtr_Y = torch.tensor(decor["train"][1], device=DEV)
Dva = torch.tensor(decor["val"][0], device=DEV)
Dva_Y = torch.tensor(decor["val"][1], device=DEV)
P_X = torch.tensor(probe_X, device=DEV)
print(
    f"\n  训练池: 真实 {len(Rtr)} + 解相关 {len(Dtr)}; 每批预计 真实 {int(args_cli.batch * W_REAL)} "
    f"/ 解相关 {args_cli.batch - int(args_cli.batch * W_REAL)}"
)


# =====================================================================================
def make_net(init_path, device):
    n = torch.nn.Sequential(
        torch.nn.Linear(117, 512),
        torch.nn.ELU(),
        torch.nn.Linear(512, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 8),
    ).to(device)
    if init_path != "none":
        ck = torch.load(init_path, map_location="cpu", weights_only=False)
        sd = ck["actor_state_dict"] if isinstance(ck, dict) and "actor_state_dict" in ck else ck
        n.load_state_dict(
            {(k[4:] if k.startswith("mlp.") else k): v for k, v in sd.items() if k.split(".")[-1] in ("weight", "bias")}
        )
    return n


net = make_net(args_cli.init, DEV)
ANCHOR = 0.0 if args_cli.init == "none" else args_cli.anchor
ref = make_net(args_cli.init, DEV).eval()
for p in ref.parameters():
    p.requires_grad_(False)


def fk_metrics(Xs, Ys, tag, max_n=192):
    """FK attitude error of the actor's own target against the command in the observation."""
    idx = np.linspace(0, len(Xs) - 1, min(max_n, len(Xs))).astype(int)
    with torch.inference_mode():
        pred = net(Xs[idx])[:, 2:].cpu().numpy()
    q_pred = DEFAULT + SCALE * pred
    errs_a, errs_p, labels_a = [], [], []
    for k, i in enumerate(idx):
        cmd = Xs[i].cpu().numpy()[PC][-7:] if torch.is_tensor(Xs) else Xs[i][PC][-7:]
        errs_a.append(fk_att_deg(q_pred[k], cmd[3:]))
        errs_p.append(fk_pos_m(q_pred[k], cmd[:3]))
        labels_a.append(
            fk_att_deg(DEFAULT + SCALE * Ys[i].cpu().numpy()[2:], cmd[3:])
            if torch.is_tensor(Ys)
            else fk_att_deg(DEFAULT + SCALE * Ys[i][2:], cmd[3:])
        )
    print(
        f"    {tag:<26}FK 姿态 {np.mean(errs_a):6.2f}° (中位 {np.median(errs_a):5.2f})  "
        f"位置 {np.mean(errs_p) * 1000:6.1f} mm   [标签自身 {np.mean(labels_a):.2f}°]"
    )
    return float(np.mean(errs_a)), float(np.mean(errs_p)), float(np.median(errs_a))


def probe_metrics(tag):
    with torch.inference_mode():
        pred = net(P_X)[:, 2:].cpu().numpy()
    q_pred = DEFAULT + SCALE * pred
    errs = [fk_att_deg(q_pred[k], probe_cmd[k][3:]) for k in range(len(q_pred))]
    poss = [fk_pos_m(q_pred[k], probe_cmd[k][:3]) for k in range(len(q_pred))]
    print(
        f"    {tag:<26}FK 姿态 {np.mean(errs):6.2f}° (中位 {np.median(errs):5.2f})  位置 {np.mean(poss) * 1000:6.1f} mm"
    )
    return float(np.mean(errs)), float(np.median(errs))


def sensitivity():
    """Change only the command at a fixed state; the ideal answer is 0 deg."""
    outs = []
    for dr in (0.1, 0.2):
        errs = []
        for i in range(0, len(Dtr), max(1, len(Dtr) // 64)):
            o = Dtr[i].cpu().numpy().copy()
            cmd = o[PC][-7:]
            dq = quat_from_euler_np([dr, 0, 0])
            qc = quat_mul_np(dq, cmd[3:])
            o[PC] = np.tile(np.concatenate([cmd[:3], qc]), 3)
            with torch.inference_mode():
                a = net(torch.tensor(o, dtype=torch.float32, device=DEV).unsqueeze(0))[0, 2:].cpu().numpy()
            errs.append(fk_att_deg(DEFAULT + SCALE * a, qc))
        outs.append(float(np.median(errs)))
    return outs


def chassis_drift():
    with torch.inference_mode():
        d = (net(Rva)[:, :2] - ref(Rva)[:, :2]).abs().mean().item()
    return float(d)


print("\n" + "=" * 100)
print("2. 训练前 (B0 初始化)")
print("=" * 100)
fk_metrics(Rva, Rva_Y, "验证(真实, 未见命令)")
probe_metrics("探针(新命令+到位)")
print(f"    命令灵敏度 Δroll=0.1/0.2: {sensitivity()[0]:.1f}° / {sensitivity()[1]:.1f}°")


# =====================================================================================
print("\n" + "=" * 100)
print(
    f"3. 训练: {N_UPDATES} 次 optimizer update, batch {args_cli.batch}, "
    f"lr {args_cli.lr}->{args_cli.lr_final}, anchor {ANCHOR}"
)
print("=" * 100)
opt = torch.optim.Adam(net.parameters(), lr=args_cli.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_UPDATES, eta_min=args_cli.lr_final)
log_path = OUT / f"loss_{args_cli.tag}.csv"
log_f = open(log_path, "w", newline="")  # noqa: SIM115 -- closed explicitly after the loop
writer = csv.writer(log_f)
writer.writerow(
    [
        "update",
        "lr",
        "train_arm_mse",
        "train_anchor",
        "train_arm_mae_rad",
        "val_real_mse",
        "val_decor_mse",
        "val_arm_mae_rad",
        "val_fk_att_deg",
        "val_fk_pos_mm",
        "probe_fk_att_deg",
        "chassis_drift",
        "n_real",
        "n_decor",
        "grad_norm",
    ]
)

best = {"score": float("inf")}
history = []
t0 = __import__("time").time()
for upd in range(1, N_UPDATES + 1):
    net.train()
    is_real = torch.rand(args_cli.batch, generator=gen, device=DEV) < W_REAL
    n_r = int(is_real.sum())
    n_d = args_cli.batch - n_r
    ir = torch.randint(len(Rtr), (n_r,), generator=gen, device=DEV)
    idc = torch.randint(len(Dtr), (n_d,), generator=gen, device=DEV)
    Xb = torch.cat([Rtr[ir], Dtr[idc]], dim=0)
    Yb = torch.cat([Rtr_Y[ir], Dtr_Y[idc]], dim=0)
    out = net(Xb)
    arm = torch.mean((out[:, 2:] - Yb[:, 2:]) ** 2)
    anch = torch.mean((out[:n_r, :2] - ref(Rtr[ir])[:, :2]) ** 2) if n_r > 0 else torch.zeros((), device=DEV)
    loss = arm + ANCHOR * anch
    opt.zero_grad()
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0))
    opt.step()
    sched.step()

    if upd % args_cli.val_every and upd != N_UPDATES:
        continue
    net.eval()
    with torch.inference_mode():
        vr = float(torch.mean((net(Rva)[:, 2:] - Rva_Y[:, 2:]) ** 2))
        vd = float(torch.mean((net(Dva)[:, 2:] - Dva_Y[:, 2:]) ** 2))
        mae = float(torch.mean(torch.abs(net(Rva)[:, 2:] - Rva_Y[:, 2:])) * SCALE)
        pa = net(P_X)[:, 2:].cpu().numpy()
    q_pred = DEFAULT + SCALE * pa
    probe_att = float(np.mean([fk_att_deg(q_pred[k], probe_cmd[k][3:]) for k in range(len(q_pred))]))
    # FK on a subsample of the validation set, for the table
    idx = np.linspace(0, len(Rva) - 1, min(128, len(Rva))).astype(int)
    with torch.inference_mode():
        pr = net(Rva[idx])[:, 2:].cpu().numpy()
    qp = DEFAULT + SCALE * pr
    _cmd_rows = np.stack([X_real[m_va][i][PC][-7:] for i in idx])
    va_att = float(np.mean([fk_att_deg(qp[k], _cmd_rows[k][3:]) for k in range(len(idx))]))
    va_pos = float(np.mean([fk_pos_m(qp[k], _cmd_rows[k][:3]) for k in range(len(idx))]))
    drift = chassis_drift()
    score = vr + vd
    train_mae = float(torch.mean(torch.abs(out[:, 2:] - Yb[:, 2:])) * SCALE)
    row = [
        upd,
        sched.get_last_lr()[0],
        float(arm),
        float(anch),
        train_mae,
        vr,
        vd,
        mae,
        va_att,
        va_pos,
        probe_att,
        drift,
        n_r,
        n_d,
        gn,
    ]
    writer.writerow(row)
    log_f.flush()
    history.append(row)
    flag = ""
    if score < best["score"] - 1e-9:
        best = {
            "score": score,
            "update": upd,
            "probe": probe_att,
            "state": {k: t.detach().clone() for k, t in net.state_dict().items()},
        }
        flag = " *best"
    if upd in (3000, 10000):
        torch.save(net.state_dict(), OUT / f"actor_{args_cli.tag}_u{upd}.pt")
    print(
        f"  u{upd:<6d} train {row[2]:.5f}+{row[3]:.5f}  val 真实 {vr:.5f} 解相关 {vd:.5f}  "
        f"MAE {mae:.4f} rad  FK {va_att:5.2f}° / {va_pos * 1000:5.1f} mm  "
        f"探针 {probe_att:5.2f}°  底盘漂移 {drift:.3f}  |g| {gn:.3f}{flag}"
    )
    if upd - best.get("update", upd) >= args_cli.patience:
        print(f"  连续 {args_cli.patience} 次 update 验证无改善, 提前停止")
        break

log_f.close()
torch.save(net.state_dict(), OUT / f"actor_{args_cli.tag}_last.pt")
torch.save(best["state"], OUT / f"actor_{args_cli.tag}_best.pt")
print(
    f"\n  用时 {(__import__('time').time() - t0) / 60:.1f} min; 最佳验证 update {best['update']} "
    f"(score {best['score']:.5f})"
)
_written = sorted(p.name for p in OUT.glob(f"actor_{args_cli.tag}_*.pt"))
print(f"  已写出 {', '.join(_written)}")

# =====================================================================================
print("\n" + "=" * 100)
print("4. 收敛摘要与判定")
print("=" * 100)
net.load_state_dict(best["state"])
print(f"  {'':<16}{'Train MSE':>11}{'Val 真实':>10}{'Val 解相关':>11}{'Train MAE':>11}{'Val MAE':>10}{'Val FK':>9}")
marks, _seen = [], set()
_marks = [history[min(len(history) - 1, int(len(history) * f))] for f in (0.25, 0.5, 0.75)]
for r in [history[0]] + _marks + [history[-1]]:
    if r[0] not in _seen:
        _seen.add(r[0])
        marks.append(r)
for r in marks:
    print(f"  {f'u{r[0]:,d}':<16}{r[2]:>11.5f}{r[5]:>10.5f}{r[6]:>11.5f}{r[4]:>11.4f}{r[7]:>10.4f}{r[8]:>9.2f}")
b = [r for r in history if r[0] == best["update"]][0]
print(f"  {f'best u{b[0]:,d}':<16}{b[2]:>11.5f}{b[5]:>10.5f}{b[6]:>11.5f}{b[4]:>11.4f}{b[7]:>10.4f}{b[8]:>9.2f}")
tr_l, va_l = history[-1][2], history[-1][5]
va_best = min(r[5] for r in history)
# judge by the *trend over the last 10k updates*, not by "last == best": with a monotonically
# improving validation curve those two are equal by construction, which would call any run
# converged on the last point alone
_prev = [r for r in history if r[0] <= history[-1][0] - 10000]
va_prev = _prev[-1][5] if _prev else history[0][5]
gain = (va_prev - va_l) / max(va_prev, 1e-9)
print(
    f"  训练/验证分叉: 最后 train_arm {tr_l:.5f}, val_real {va_l:.5f} (最佳 {va_best:.5f}, "
    f"train/val = {tr_l / max(va_l, 1e-12):.0f}x)"
)
print(f"  最近 10000 次 update 的验证改善: {gain * 100:+.1f}%  (u{_prev[-1][0] if _prev else 0}-u{history[-1][0]})")
if va_l > va_best * 1.3 and history[-1][0] > best["update"] + args_cli.val_every:
    verdict = "疑似过拟合 (验证在最佳之后明显回升, 按最佳 checkpoint 取用)"
elif history[-1][0] < N_UPDATES:
    verdict = "尚未收敛 (早停触发, 验证仍在改善)"
elif gain < 0.05:
    verdict = "正常收敛 (最近 10000 次改善 < 5%)"
else:
    verdict = f"验证仍在缓慢改善 ({gain * 100:.0f}%/10000), 未见平台但预算已用尽"
print(f"  判定: {verdict}")
print(f"  底盘通道: 相对 B0 的平均绝对漂移 {b[11]:.4f} (action 单位)")
print("\n  最终模型:")
fk_metrics(Rva, Rva_Y, "验证(真实, 未见命令)")
fk_metrics(Rte, Rte_Y, "测试(真实, 未见命令)")
probe_metrics("探针(新命令+到位)")
print(f"    命令灵敏度 Δroll=0.1/0.2: {sensitivity()[0]:.1f}° / {sensitivity()[1]:.1f}°")

with open(OUT / f"train_{args_cli.tag}.json", "w") as _f:
    json.dump(
        dict(
            tag=args_cli.tag,
            ratio=args_cli.ratio,
            updates=history[-1][0],
            batch=args_cli.batch,
            lr=args_cli.lr,
            lr_final=args_cli.lr_final,
            anchor=args_cli.anchor,
            n_real_train=len(Rtr),
            n_decor_train=len(Dtr),
            n_real_val=len(Rva),
            n_decor_val=len(Dva),
            n_real_test=len(Rte),
            n_groups=n_g,
            n_groups_train=len(g_tr),
            n_groups_val=len(g_va),
            n_groups_test=len(g_te),
            best_update=best["update"],
            best_score=best["score"],
            verdict=verdict,
            # the held-out command values, so the closed-loop test cannot accidentally use
            # a command the supervision contained
            test_cmds=key_of[sorted(g_te)].tolist(),
            val_cmds=key_of[sorted(g_va)].tolist(),
            history=history,
        ),
        _f,
        indent=1,
    )
print(f"\n已写出 {OUT / f'train_{args_cli.tag}.json'} 与 {log_path}")
