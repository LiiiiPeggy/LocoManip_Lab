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

"""Stage 4.5j: write the checkpoint that standard PPO starts from.

Three things are replaced and four are deliberately *not* carried over:

* the actor weights come from the 4.5i supervised actor (`actor_mix11_best.pt`), remapped to
  the `mlp.*` naming the runner uses;
* the critic is freshly initialised -- carrying over a critic that was fitted to a *different*
  policy is how the 4.5e resume experiment ended up with a first `Loss/value` of 15.6 against
  2.85 for the cold-start control;
* the optimiser state is emptied (Adam moments belong to the old weights; with them kept the
  first updates take steps sized for a different objective);
* the iteration counter is reset to 0, so `learn(6000)` really runs 6000 iterations.

The one parameter that is *not* a default is the action standard deviation. The actor
distribution is diagonal Gaussian with a learned per-dimension `log_std_param`; the config
default is `init_std=1.0`, i.e. 0.25 rad of joint noise per step, which is 20x the arm map's
own precision target (0.012 rad) and would swamp it before the reward could act. The std is
initialised instead to what *standard PPO itself converged to* in the 1000-iteration run
(0.08-0.16 action units), so exploration starts where the same algorithm put it. The algorithm,
network, reward and `init_std` config entry are untouched.

Usage:
    python scripts/ranger_cr10/make_ppo_init.py \
        --actor logs/rsl_rl/ranger_cr10_teleop/_stage45i/actor_mix11_best.pt \
        --std_from logs/rsl_rl/ranger_cr10_teleop/2026-10-16_19-21-59_resume200/model_1198.pt \
        --out logs/rsl_rl/ranger_cr10_teleop/_stage45j/model_0.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description="Build the standard-PPO initial checkpoint.")
parser.add_argument("--actor", default="logs/rsl_rl/ranger_cr10_teleop/_stage45i/actor_mix11_best.pt")
parser.add_argument("--std_from", default="logs/rsl_rl/ranger_cr10_teleop/2026-10-16_19-21-59_resume200/model_1198.pt")
parser.add_argument("--out", default="logs/rsl_rl/ranger_cr10_teleop/_stage45j/model_0.pt")
parser.add_argument("--critic_obs_dim", type=int, default=68)
parser.add_argument("--actor_obs_dim", type=int, default=117)
parser.add_argument("--action_dim", type=int, default=8)
parser.add_argument("--lr", type=float, default=5e-4, help="PPO learning rate, for the fresh Adam group")
parser.add_argument(
    "--arm_std",
    type=float,
    default=None,
    help="override the per-dim std for the six arm channels; the two chassis "
    "channels keep the value taken from --std_from. Used by the stage-4.5j "
    "single-factor comparison (arm exploration 0.12 -> 0.03).",
)
parser.add_argument("--seed", type=int, default=42)
args_cli = parser.parse_args()

HIDDEN = (512, 256, 128)
OUT = Path(args_cli.out)
OUT.parent.mkdir(parents=True, exist_ok=True)
torch.manual_seed(args_cli.seed)


def mlp(d_in: int, d_out: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, HIDDEN[0]),
        torch.nn.ELU(),
        torch.nn.Linear(HIDDEN[0], HIDDEN[1]),
        torch.nn.ELU(),
        torch.nn.Linear(HIDDEN[1], HIDDEN[2]),
        torch.nn.ELU(),
        torch.nn.Linear(HIDDEN[2], d_out),
    )


def to_mlp_keys(sd: dict) -> dict:
    return {(f"mlp.{k}" if k[0].isdigit() else k): v for k, v in sd.items()}


# ------------------------------------------------------------------ actor -----
a = torch.load(args_cli.actor, map_location="cpu", weights_only=False)
mix_sd = a["actor_state_dict"] if isinstance(a, dict) and "actor_state_dict" in a else a
mix_sd = {k: v for k, v in mix_sd.items() if k.split(".")[-1] in ("weight", "bias")}
ref_net = mlp(args_cli.actor_obs_dim, args_cli.action_dim)
ref_net.load_state_dict(mix_sd)
ref_net.eval()

std_ck = torch.load(args_cli.std_from, map_location="cpu", weights_only=False)
log_std = std_ck["actor_state_dict"]["distribution.log_std_param"].clone()
if args_cli.arm_std is not None:
    import math

    log_std[2:] = math.log(args_cli.arm_std)
assert log_std.shape == (args_cli.action_dim,), log_std.shape

actor_sd = to_mlp_keys(mix_sd)
actor_sd["distribution.log_std_param"] = log_std

# ----------------------------------------------------------------- critic -----
critic = mlp(args_cli.critic_obs_dim, 1)
critic_sd = to_mlp_keys(critic.state_dict())
critic_sd["distribution.log_std_param"] = torch.zeros(1)

# -------------------------------------------------------------- optimiser -----
# empty state: Adam restarts with zero moments; one group, as `chain(actor, critic)` in PPO
# A fresh optimizer needs the *group* description but no moments. The group dict is copied from
# an existing checkpoint (Adam's step reads betas/eps out of it, so a hand-built group with only
# `lr` raises `KeyError: 'betas'`), the learning rate is reset, and `state` is emptied.
old_opt = std_ck["optimizer_state_dict"]
groups = [{k: v for k, v in g.items()} for g in old_opt["param_groups"]]
for g in groups:
    g["lr"] = args_cli.lr
    g.pop("initial_lr", None)
optimizer_sd = {"state": {}, "param_groups": groups}

ck = dict(actor_state_dict=actor_sd, critic_state_dict=critic_sd, optimizer_state_dict=optimizer_sd, iter=0, infos={})
torch.save(ck, OUT)

# ------------------------------------------------------------ verification ----
print("=" * 92)
print("1. Actor 权重是否完整且未改变")
print("=" * 92)
check = mlp(args_cli.actor_obs_dim, args_cli.action_dim)
check.load_state_dict({k.removeprefix("mlp."): v for k, v in actor_sd.items() if k.startswith("mlp.")})
check.eval()
x = torch.randn(64, args_cli.actor_obs_dim) * 0.5
with torch.inference_mode():
    d = (ref_net(x) - check(x)).abs().max().item()
print(
    f"  观测 {args_cli.actor_obs_dim} -> 动作 {args_cli.action_dim}; 与 mix11 输出最大差 {d:.3e} "
    f"-> {'一致' if d == 0 else '**不一致**'}"
)
print(f"  动作 std (每维) {torch.exp(log_std).tolist()}")
print(f"  这些 std 取自 {args_cli.std_from} 的收敛值, 不是 config 的 init_std=1.0")

print("\n" + "=" * 92)
print("2. Critic 是否为新初始化")
print("=" * 92)
with torch.inference_mode():
    v_new = critic(torch.randn(256, args_cli.critic_obs_dim))
print(f"  观测 {args_cli.critic_obs_dim} -> 1; 新 critic 输出 均值 {v_new.mean():+.4f} 标准差 {v_new.std():.4f}")
old_v = std_ck["critic_state_dict"]["mlp.6.weight"].norm().item()
new_v = critic_sd["mlp.6.weight"].norm().item()
print(
    f"  最后一层权重范数: 旧 {old_v:.4f} / 新 {new_v:.4f} -> "
    f"{'已重新初始化' if abs(old_v - new_v) > 1e-6 else '**与旧 critic 相同**'}"
)
print(f"  optimizer: state 条数 {len(optimizer_sd['state'])}, param_groups {len(optimizer_sd['param_groups'])}")
print(f"  iter = {ck['iter']}")

print("\n" + "=" * 92)
print("3. 是否还有任何 IK / 监督路径")
print("=" * 92)
keys = set(actor_sd) | set(critic_sd)
bad = [k for k in keys if any(t in k.lower() for t in ("ik", "teacher", "aux"))]
print(f"  checkpoint 中的可疑键: {bad if bad else '无'}")
print(f"  已写出 {OUT}")
