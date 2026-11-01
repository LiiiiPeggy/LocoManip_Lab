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

"""Start a new training budget from an existing checkpoint without losing the policy.

``rsl_rl``'s ``--resume`` restores the iteration counter, and ``learn(max_iterations)`` runs
*to* that number, so resuming from ``model_5999`` with ``--max_iterations 6000`` performs one
update rather than 6000. Stage 4.5j solved this by writing a fresh ``model_0.pt`` with a
re-initialised critic and optimiser; that was right for a cold start from a supervised actor,
but wrong here -- stage 4.7 continues a trained policy and wants its critic and Adam moments.

So this copies the checkpoint and rewrites only ``iter``. The actor, critic, optimiser state
and action standard deviation are untouched, and the script asserts that the actor output is
bit-identical before and after.

Usage:
    python scripts/ranger_cr10/rebase_checkpoint.py \
        --src logs/rsl_rl/ranger_cr10_teleop/2026-10-29_19-12-08_stage45j_ppo/model_5999.pt \
        --dst logs/rsl_rl/ranger_cr10_teleop/_stage47/model_0.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

ap = argparse.ArgumentParser(description="Copy a checkpoint and reset its iteration counter.")
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
args = ap.parse_args()

src = Path(args.src)
dst = Path(args.dst)
dst.parent.mkdir(parents=True, exist_ok=True)

ck = torch.load(src, map_location="cpu", weights_only=False)
print(f"源: {src}  iter={ck['iter']}")

raw = ck["actor_state_dict"]
sd = {(k[4:] if k.startswith("mlp.") else k): v for k, v in raw.items() if k.split(".")[-1] in ("weight", "bias")}
net = torch.nn.Sequential(
    torch.nn.Linear(117, 512),
    torch.nn.ELU(),
    torch.nn.Linear(512, 256),
    torch.nn.ELU(),
    torch.nn.Linear(256, 128),
    torch.nn.ELU(),
    torch.nn.Linear(128, 8),
)
net.load_state_dict(sd)
net.eval()
probe = torch.randn(64, 117) * 0.5
with torch.inference_mode():
    before = net(probe).clone()

ck["iter"] = 0
torch.save(ck, dst)

again = torch.load(dst, map_location="cpu", weights_only=False)
sd2 = {
    (k[4:] if k.startswith("mlp.") else k): v
    for k, v in again["actor_state_dict"].items()
    if k.split(".")[-1] in ("weight", "bias")
}
net.load_state_dict(sd2)
with torch.inference_mode():
    after = net(probe)

print(f"目标: {dst}  iter={again['iter']}")
same = torch.equal(before, after)
print(f"Actor 输出最大差 {float((before - after).abs().max()):.3e} -> {'一致' if same else '**不一致**'}")
print(f"std 保存在 checkpoint 中: {again['actor_state_dict']['distribution.log_std_param'].exp().tolist()}")
print(f"optimizer state 条数 {len(again['optimizer_state_dict']['state'])}(保留, 继续用原来的 Adam 动量)")
