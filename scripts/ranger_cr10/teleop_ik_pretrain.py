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

"""Stage 4.5e, variant B: supervised IK pre-training of the actor, written back as a checkpoint.

Starts from the *current* actor (`model_999.pt`), fits the teacher labels on the training
episodes, early-stops on the validation episodes, reports the held-out test episodes, and
writes a checkpoint that PPO can resume from. Everything except `actor_state_dict` is copied
unchanged, including the optimiser state -- rsl_rl's `PPO.load` expects that key to be
present, and Adam's momentum from an unrelated objective decays to nothing over the 200
iterations the comparison runs.

The offline numbers printed here are *not* policy performance: they say whether the actor's
weights moved toward the teacher. The closed-loop question is answered afterwards by running
the same evaluation as variant A.

Usage:
    python scripts/ranger_cr10/teleop_ik_pretrain.py --teacher /tmp/ik_teacher \
        --checkpoint <model_999.pt> --out /tmp/ik_teacher/pretrained.pt
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

parser = argparse.ArgumentParser(description="Supervised IK pre-training for variant B.")
parser.add_argument("--teacher", default="/tmp/ik_teacher")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--epochs", type=int, default=800)
parser.add_argument("--lr", type=float, default=3e-4)
args_cli = parser.parse_args()

data = torch.load(f"{args_cli.teacher}/dataset.pt", weights_only=False)
X, Y = data["X"], data["Y"]
tr, va, te = data["tr"], data["va"], data["te"]
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"数据集 {tuple(X.shape)}, train/val/test = {len(tr)}/{len(va)}/{len(te)} (按 episode 划分)")

ck = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
actor_sd = ck["actor_state_dict"]
SCALE = 0.25  # action -> joint scale, from RangerActionsCfg.joint_pos


def build_mlp():
    return torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 512), torch.nn.ELU(),
        torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 8),
    ).to(DEV)


def load_actor_into(mlp):
    # the checkpoint stores the MLP under `mlp.0.weight`; our Sequential uses `0.weight`
    sd = {k[len("mlp."):]: v for k, v in actor_sd.items() if k.startswith("mlp.")}
    mlp.load_state_dict(sd)


actor = build_mlp()
load_actor_into(actor)
X, Y = X.to(DEV), Y.to(DEV)


def report(tag: str, idx) -> None:
    """Per-joint error and direction error on the arm channels, over one split."""
    with torch.inference_mode():
        pred = actor(X[idx])[:, 2:]
        tgt = Y[idx][:, 2:]
    a_err = (pred - tgt).abs().mean().item()
    angs = []
    for i in range(len(idx)):
        u = (pred[i] * SCALE).cpu().numpy()
        v = (tgt[i] * SCALE).cpu().numpy()
        if np.linalg.norm(u) > 0.05 and np.linalg.norm(v) > 0.05:
            c = np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)
            angs.append(math.degrees(math.acos(c)))
    ang = f"{np.median(angs):.1f}°" if angs else "—"
    print(f"  {tag:<22} 动作 MAE {a_err:.4f} (关节 {a_err * SCALE:.4f} rad)  "
          f"方向 {ang}  n={len(idx)}")


print("\n预训练前:")
for tag, idx in (("train", tr), ("val", va), ("test", te)):
    report(f"actor/{tag}", idx)

opt = torch.optim.Adam(actor.parameters(), lr=args_cli.lr)
best_v, best_s, patience = float("inf"), None, 0
for epoch in range(args_cli.epochs):
    actor.train()
    perm = tr[torch.randperm(len(tr))]
    for b in range(0, len(perm), 128):
        mb = perm[b:b + 128]
        opt.zero_grad()
        # arm channels only: channels 0-1 are the chassis velocity, which the teacher
        # does not supervise
        loss = torch.mean((actor(X[mb])[:, 2:] - Y[mb][:, 2:]) ** 2)
        loss.backward()
        opt.step()
    actor.eval()
    with torch.inference_mode():
        v = torch.mean((actor(X[va])[:, 2:] - Y[va][:, 2:]) ** 2).item()
    if v < best_v - 1e-9:
        best_v, best_s, patience = v, {k: t.clone() for k, t in actor.state_dict().items()}, 0
    else:
        patience += 1
        if patience > 80:
            break
actor.load_state_dict(best_s)
print(f"\n早停于 epoch {epoch}, val MSE {best_v:.5f}")
print("预训练后 (离线拟合, 不是策略性能):")
for tag, idx in (("train", tr), ("val", va), ("test", te)):
    report(f"actor/{tag}", idx)

# write the checkpoint back
new_sd = dict(actor_sd)
for k, v in actor.state_dict().items():
    new_sd["mlp." + k] = v.cpu()
ck["actor_state_dict"] = new_sd
torch.save(ck, args_cli.out)
print(f"\n已写出 checkpoint: {args_cli.out} (仅 actor 权重改变, critic/optimizer 原样保留)")
