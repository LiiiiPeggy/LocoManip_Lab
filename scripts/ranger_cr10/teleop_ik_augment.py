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

"""Stage 4.5g, variant B1: supervision collected *while the IK is driving*.

4.5f's supervision set came from rollouts of a policy that never reaches the commanded
attitude, so the states where the arm is *at* the target are scarce (8.4% within 2 deg) and
the region the closed loop actually settles in is under-covered. The actor then has a fixed
point about 10 deg away from the command, which task A showed it commands on the very first
step (FK(q_des) predicts the settled error to within 1.4 deg).

This collects the missing region the cheapest way that keeps every sample real: drive the
arm with online local IK, let the command term resample every second, and record the genuine
observation (three-frame history included) together with the IK label solved *from that same
state*. The result is a set dominated by "arriving" and "holding" states, plus the transients.

The samples are merged with the 4.5e set rather than replacing it: the old data is what gives
the policy its ability to travel to a distant target at all.

Usage:
    python scripts/ranger_cr10/teleop_ik_augment.py --out /tmp/ik_aug
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

parser = argparse.ArgumentParser(description="Collect IK-driving supervision and refit the actor.")
parser.add_argument("--base", default="logs/ik_teacher/dataset.pt")
parser.add_argument("--init", default="logs/rsl_rl/ranger_cr10_teleop/_stage45e/pretrained_B.pt")
parser.add_argument("--out", default="logs/rsl_rl/ranger_cr10_teleop/_stage45g")
parser.add_argument("--episodes", type=int, default=2)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--every", type=int, default=2)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--epochs", type=int, default=600)
parser.add_argument("--new_weight", type=int, default=3, help="oversample factor for the new region")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--collect_only", action="store_true", help="save the collected set and stop")
parser.add_argument("--driver", default="ik", choices=["ik", "actor"],
                    help="'ik' collects the IK's own trajectory (B1); 'actor' is DAgger (B2): the "
                         "policy drives and every state it visits is labelled by online IK")
parser.add_argument("--actor", default="logs/rsl_rl/ranger_cr10_teleop/_stage45g/actor_aug.pt")
parser.add_argument("--tag", default="aug")
args_cli = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402

import LeggedManip_Lab.tasks  # noqa: E402,F401
from LeggedManip_Lab.assets.ranger_cr10.ranger_cr10_articulation_cfg import ARM_JOINT_NAMES  # noqa: E402
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.teleop_env_cfg import (  # noqa: E402
    RangerCr10TeleopEnvCfg,
)

OUT = Path(args_cli.out)
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT = np.asarray(tk.DEFAULT_Q, dtype=np.float64)

cfg = RangerCr10TeleopEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
cfg.commands.ee_pose.debug_vis = False
# IK driver: a command change every second gives ten steps of travel and ten of hold, both
# regions matter. DAgger: keep the deployment distribution (4-8 s, base moving) -- the point is
# to label the states the policy actually visits, which is only meaningful in that distribution.
cfg.commands.ee_pose.resampling_time_range = (1.0, 1.2) if args_cli.driver == "ik" else (4.0, 8.0)
cfg.commands.base_velocity.resampling_time_range = (1e6, 1e6) if args_cli.driver == "ik" else (4.0, 8.0)
cfg.observations.policy.enable_corruption = False
env = gym.make("RANGER-CR10-TELEOP-WBC", cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
ee_cmd = u.command_manager.get_term("ee_pose")
vel_cmd = u.command_manager.get_term("base_velocity")
arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
SCALE = u.action_manager.get_term("joint_pos").cfg.scale
N, DEV = u.num_envs, u.device

def load_actor(path, device):
    n = torch.nn.Sequential(
        torch.nn.Linear(117, 512), torch.nn.ELU(), torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 8)).to(device)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["actor_state_dict"] if isinstance(ck, dict) and "actor_state_dict" in ck else ck
    n.load_state_dict({(k[4:] if k.startswith("mlp.") else k): v for k, v in sd.items()
                       if k.split(".")[-1] in ("weight", "bias")})
    n.eval()
    for p in n.parameters():
        p.requires_grad_(False)
    return n


driver_net = load_actor(args_cli.actor, DEV) if args_cli.driver == "actor" else None
print("=" * 92)
print("1. " + ("用被测 Actor 自己驱动 (DAgger), 它访问到的每个状态都用在线 IK 重新标注"
              if driver_net is not None else "用在线 IK 驱动, 采集'正在到达'与'已经保持'的状态"))
print("=" * 92)
obs_list, lab_list, ep_list, near, drop, cand = [], [], [], [], 0, 0
for ep in range(args_cli.episodes):
    obs, _ = env.reset(seed=args_cli.seed + ep)
    if args_cli.driver == "ik":
        vel_cmd.is_standing_env[:] = False
        vel_cmd.vel_command_b[:] = 0.0
    kept = 0
    for step in range(args_cli.steps):
        q_all = robot.data.joint_pos[:, arm_ids].cpu().numpy()
        cmd_pos = ee_cmd.pose_command_b[:, :3].cpu().numpy()
        cmd_quat = ee_cmd.pose_command_b[:, 3:].cpu().numpy()
        a = torch.zeros(N, u.action_manager.total_action_dim, device=DEV)
        if driver_net is not None:
            with torch.inference_mode():
                a = driver_net(obs["policy"].to(DEV)).to(DEV)
            q_sol_all = np.empty_like(q_all)
            if step % args_cli.every == 0:      # labels are only needed at sampled steps
                for i in range(N):
                    q_sol_all[i], _, _, _, _ = tk.local_ik(q_all[i], cmd_pos[i], tk.quat_to_R(cmd_quat[i]))
        else:
            q_sol_all = np.empty_like(q_all)
            for i in range(N):
                q_sol, ep_p, ep_r, _, _ = tk.local_ik(q_all[i], cmd_pos[i], tk.quat_to_R(cmd_quat[i]))
                q_sol_all[i] = q_sol
                a[i, 2:] = torch.tensor((q_sol - DEFAULT) / SCALE, dtype=torch.float32, device=DEV)
        if step % args_cli.every == 0:
            o = obs["policy"].cpu().numpy()
            for i in range(N):
                cand += 1
                # the solution the controller is about to execute *is* the label for this
                # observation: both were solved from the same (q_all[i], cmd[i]). Re-solving
                # here would double the cost of the collection for the same number.
                q_sol = q_sol_all[i]
                T0, T1 = tk.fk(q_all[i]), tk.fk(q_sol)
                ep_p = float(np.linalg.norm(T1[:3, 3] - cmd_pos[i]))
                ep_r = float(np.linalg.norm(tk.log_R(tk.quat_to_R(cmd_quat[i]) @ T1[:3, :3].T)))
                if ep_p > 1e-3 or ep_r > 1e-3:
                    drop += 1
                    continue
                obs_list.append(o[i].astype(np.float32))
                lab_list.append(((q_sol - DEFAULT) / SCALE).astype(np.float32))
                ep_list.append(ep)
                near.append(math.degrees(np.linalg.norm(tk.log_R(T1[:3, :3] @ T0[:3, :3].T))))
                kept += 1
        obs, _, _, _, _ = env.step(a)
    print(f"  episode {ep}: 记录 {kept} 条")

near = np.array(near)
X_new = torch.tensor(np.stack(obs_list), device=DEV)
lab = np.stack(lab_list)
Y_new = torch.cat([torch.zeros(len(lab), 2, device=DEV),
                   torch.tensor(lab, device=DEV)], dim=1)
eps_new = np.array(ep_list)
print(f"  候选 {cand}, 有效 {len(obs_list)} ({len(obs_list) / cand * 100:.1f}%), 丢弃 {drop}")
print(f"  这些状态'当前姿态距 IK 解': 中位 {np.median(near):.1f}°, "
      f"<2° 占 {np.mean(near < 2) * 100:.1f}%  (原数据集 8.4%)")

torch.save(dict(X=X_new.cpu(), Y=Y_new.cpu(), episode=torch.tensor(eps_new),
                near=torch.tensor(near)), OUT / f"{args_cli.tag}_dataset.pt")
print(f"  已保存 {OUT / f'{args_cli.tag}_dataset.pt'}")
if args_cli.collect_only:
    raise SystemExit(0)

# ---------------------------------------------------------------- merge ----------------
data = torch.load(args_cli.base, map_location="cpu", weights_only=False)
X_old, Y_old = data["X"], data["Y"]
tr, va, te = data["tr"], data["va"], data["te"]
# hold out one whole new episode so the new region has its own honest split
new_val_eps = {args_cli.episodes - 1}
new_tr = torch.tensor([i for i in range(len(obs_list)) if eps_new[i] not in new_val_eps])
new_va = torch.tensor([i for i in range(len(obs_list)) if eps_new[i] in new_val_eps])
print(f"\n  合并: 原训练 {len(tr)} + 新训练 {len(new_tr)} (x{args_cli.new_weight}), "
      f"验证 {len(va)} + {len(new_va)}, 原测试 {len(te)}")


def make_net(init_from: Path | None):
    torch.manual_seed(args_cli.seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(117, 512), torch.nn.ELU(),
        torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 8),
    ).to(DEV)
    if init_from is not None:
        ck = torch.load(init_from, map_location="cpu", weights_only=False)
        sd = {(k[4:] if k.startswith("mlp.") else k): v for k, v in ck["actor_state_dict"].items()
              if k.split(".")[-1] in ("weight", "bias")}
        net.load_state_dict(sd)
        print(f"  从 {init_from} 初始化")
    return net


Xtr = torch.cat([X_old[tr].to(DEV)] + [X_new[new_tr]] * args_cli.new_weight, dim=0)
Ytr = torch.cat([Y_old[tr].to(DEV)] + [Y_new[new_tr]] * args_cli.new_weight, dim=0)
Xva = torch.cat([X_old[va].to(DEV), X_new[new_va]], dim=0)
Yva = torch.cat([Y_old[va].to(DEV), Y_new[new_va]], dim=0)
Xte = X_old[te].to(DEV)
Yte = Y_old[te].to(DEV)


def report(net, tag):
    with torch.inference_mode():
        out = {}
        for name, Xs, Ys in (("原测试集", Xte, Yte), ("新留出集", X_new[new_va], Y_new[new_va]),
                             ("新训练集", X_new[new_tr], Y_new[new_tr])):
            pred = net(Xs)[:, 2:]
            jm = float(torch.mean(torch.abs(pred - Ys[:, 2:])) * SCALE)
            dp, da = [], []
            q_pred = (DEFAULT + SCALE * pred.cpu().numpy())
            q_tgt = (DEFAULT + SCALE * Ys[:, 2:].cpu().numpy())
            for k in range(len(pred)):
                T, Tt = tk.fk(q_pred[k]), tk.fk(q_tgt[k])
                dp.append(np.linalg.norm(T[:3, 3] - Tt[:3, 3]))
                da.append(math.degrees(np.linalg.norm(tk.log_R(Tt[:3, :3] @ T[:3, :3].T))))
            out[name] = (jm, float(np.mean(da)), float(np.mean(dp)))
    print(f"  {tag:<10}" + "".join(f"{k}: {v[0]:.4f} rad / {v[1]:.1f}° / {v[2] * 100:.1f}cm   "
                                   for k, v in out.items()))
    return out


print("\n" + "=" * 92)
print("2. 从 B0 出发, 在合并集上短时监督")
print("=" * 92)
net = make_net(Path(args_cli.init))
report(net, "训练前")
opt = torch.optim.Adam(net.parameters(), lr=2e-4)
best_v, best_s, patience = float("inf"), None, 0
for epoch in range(args_cli.epochs):
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
after = report(net, "训练后")

torch.save(net.state_dict(), OUT / f"actor_{args_cli.tag}.pt")
with open(OUT / "augment_stats.json", "w") as f:
    json.dump(dict(n_new=len(obs_list), near2=float(np.mean(near < 2)), drop=drop, cand=cand,
                   val_mse=best_v, offline={k: list(v) for k, v in after.items()}), f, indent=1)
print(f"\n已写出 {OUT / f'actor_{args_cli.tag}.pt'}")
