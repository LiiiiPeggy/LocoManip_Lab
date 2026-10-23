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

"""Stage 4.5e, teacher: build the local-IK supervision dataset and fit a teacher network.

The label, stated precisely because three different quantities are easy to confuse here:

    q_current                     the arm joints right now, from the simulator
    q_ik                          local_ik(q_current, target_pos, target_R): the nearest
                                  continuous joint configuration that reaches the commanded
                                  *6D pose*, verified by FK before use
    label = (q_ik - q_default) / scale    the actor's normalised action vector, because
                                  `JointPositionAction.process_actions` is
                                  `q_default + scale * raw` with scale 0.25 and no clip

Only the third one is what the network is trained on; mixing it with the first two is how a
supervision set ends up silently wrong by a factor of four and an offset.

Data comes from the real training distribution, not from a sampling scheme: each episode is
a live rollout of the current policy, with the env's own command generator resampling
attitude/position at the training rate and the base velocity command running alongside, so
samples include the base moving, the command mid-life, and the history that `env.step`
actually produced. Splits are by **episode**, never by sample -- consecutive states within a
rollout are nearly identical and splitting them would make the validation number optimistic.

The teacher is the actor's own architecture (117 -> 512 -> 256 -> 128 -> 8) fitted
supervised. It exists because running `local_ik` for every parallel environment inside a PPO
step is far too expensive; the teacher is the cheap, batched stand-in used by variant C.

Usage:
    python scripts/ranger_cr10/teleop_ik_teacher.py --policy <policy.pt> --out /tmp/ik_teacher
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description="Build the IK supervision set and fit a teacher.")
parser.add_argument("--policy", required=True, help="TorchScript policy (used to drive the rollouts)")
parser.add_argument("--out", default="/tmp/ik_teacher", help="output directory")
parser.add_argument("--episodes", type=int, default=8)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--every", type=int, default=4, help="record every N steps")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--epochs", type=int, default=400)
args_cli = parser.parse_args()

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teleop_kinematics as tk  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as iu  # noqa: E402
import LeggedManip_Lab.tasks  # noqa: E402,F401
from LeggedManip_Lab.assets.ranger_cr10.ranger_cr10_articulation_cfg import ARM_JOINT_NAMES  # noqa: E402
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.teleop_env_cfg import (  # noqa: E402
    RangerCr10TeleopEnvCfg,
)

OUT = Path(args_cli.out)
OUT.mkdir(parents=True, exist_ok=True)

cfg = RangerCr10TeleopEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
cfg.commands.ee_pose.debug_vis = False
cfg.observations.policy.enable_corruption = False   # labels must correspond to the clean obs
env = gym.make("RANGER-CR10-TELEOP-WBC", cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
ee_cmd = u.command_manager.get_term("ee_pose")
arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
net_actor = torch.jit.load(args_cli.policy, map_location=u.device)
DEV, N = u.device, u.num_envs
SCALE = u.action_manager.get_term("joint_pos").cfg.scale
DEFAULT_Q = robot.data.default_joint_pos[0, arm_ids].cpu().numpy()
print(f"默认关节 {np.round(DEFAULT_Q, 6)}  (运动学模块 {tk.DEFAULT_Q})")
assert np.allclose(DEFAULT_Q, tk.DEFAULT_Q, atol=1e-5)
print(f"动作语义: q_target = default + {SCALE} * raw_action, joint_pos 未配 clip")
_obs_dim = sum(sum(d) for d in u.observation_manager._group_obs_term_dim["policy"])
_hbuf = u.observation_manager._group_obs_term_history_buffer["policy"]
_hist = _hbuf[u.observation_manager._group_obs_term_names["policy"][0]].max_length
print(f"观测维度 {_obs_dim}, 历史长度 {_hist}")

# ---------------------------------------------------------------- collect ----------------
print("\n" + "=" * 92)
print("1. 采集 (真实 rollout, 真实 117D 观测, 按 episode 划分)")
print("=" * 92)
records = []          # dicts: obs, label, episode, cmd_pos, cmd_quat, q, q_ik
dropped = 0
for ep in range(args_cli.episodes):
    obs, _ = env.reset()
    n_rec = 0
    for step in range(args_cli.steps):
        with torch.inference_mode():
            action = net_actor(obs["policy"])
        # a little exploration so the set is not only the current policy's own trajectory
        action = action + 0.15 * torch.randn_like(action)
        obs, _, _, _, _ = env.step(action)
        if step % args_cli.every:
            continue
        q = robot.data.joint_pos[:, arm_ids].cpu().numpy()
        cmd_pos = ee_cmd.pose_command_b[:, :3].cpu().numpy()
        cmd_quat = ee_cmd.pose_command_b[:, 3:].cpu().numpy()
        o = obs["policy"].cpu().numpy()
        for i in range(N):
            p = cmd_pos[i]
            R = tk.quat_to_R(cmd_quat[i])
            q_ik, ep_p, ep_r, max_step, _ = tk.local_ik(q[i], p, R)
            if ep_p > 1e-3 or ep_r > 1e-3 or max_step > 0.08:
                dropped += 1
                continue
            records.append(dict(obs=o[i].astype(np.float32),
                                label=((q_ik - DEFAULT_Q) / SCALE).astype(np.float32),
                                episode=ep, q=q[i], q_ik=q_ik,
                                cmd_pos=p, cmd_rot=R))
            n_rec += 1
    print(f"  episode {ep}: 记录 {n_rec} 条")
print(f"  合计 {len(records)} 条, 丢弃 {dropped} 条 (参考解未收敛/跳支)")

X = torch.tensor(np.stack([r["obs"] for r in records]), device=DEV)
# The actor emits 8 numbers; the teacher says nothing about the two chassis channels, so the
# label is padded with zeros there and every loss below is taken on [:, 2:] only.
_lab6 = torch.tensor(np.stack([r["label"] for r in records]), device=DEV)
Y = torch.cat([torch.zeros(len(_lab6), 2, device=DEV), _lab6], dim=1)

# split by episode: consecutive states inside a rollout are nearly identical
eps = sorted({r["episode"] for r in records})
n_test = max(1, len(eps) // 4)
n_val = max(1, len(eps) // 4)
test_eps = set(eps[-n_test:])
val_eps = set(eps[-n_test - n_val:-n_test])
tr_idx = [i for i, r in enumerate(records) if r["episode"] not in test_eps | val_eps]
va_idx = [i for i, r in enumerate(records) if r["episode"] in val_eps]
te_idx = [i for i, r in enumerate(records) if r["episode"] in test_eps]
print(f"  划分 (按 episode): train {len(tr_idx)} / val {len(va_idx)} / test {len(te_idx)}"
      f"  episodes -> train {sorted(set(eps) - test_eps - val_eps)}, val {sorted(val_eps)}, "
      f"test {sorted(test_eps)}")

# ------------------------------------------------------- label quality checks -------------
print("\n" + "=" * 92)
print("2. 标签质量 (连续性 / 可执行性 / 同一状态的多解)")
print("=" * 92)
# continuity: perturb the state slightly and re-solve; the nearest solution should move little
rng = np.random.default_rng(0)
deltas = []
for k in rng.choice(len(records), size=40, replace=False):
    r = records[k]
    q2, ep2, er2, ms2, _ = tk.local_ik(r["q"] + rng.uniform(-0.01, 0.01, size=6), r["cmd_pos"], r["cmd_rot"])
    if ep2 < 1e-3 and er2 < 1e-3:
        deltas.append(float(np.abs(q2 - r["q_ik"]).max()))
print(f"  连续性: 状态扰动 ±0.01 rad 后标签最大变化 {np.median(deltas):.4f} rad (中位), "
      f"最大 {max(deltas):.4f} rad  (应远小于一个关节分辨率)")
# executability: push the label through the action interface and see whether the arm gets there
feasible = 0
trials = 0
FEAS_STEPS = 60
for k in rng.choice(len(records), size=6, replace=False):
    r = records[k]
    q_t = torch.tensor(r["q"], dtype=torch.float32, device=DEV).unsqueeze(0).expand(N, -1)
    robot.write_joint_state_to_sim(q_t, torch.zeros_like(q_t), joint_ids=arm_ids)
    a = torch.zeros(N, u.action_manager.total_action_dim, device=DEV)
    a[:, 2:] = torch.tensor(r["label"], device=DEV)
    # through env.step, not a hand-rolled process_action/write_data_to_sim pair: the first
    # attempt used the latter and the joints did not move at all, so it was measuring nothing
    obs, _ = env.reset()
    a[:, 2:] = torch.tensor(r["label"], device=DEV)
    for _ in range(FEAS_STEPS):
        env.step(a)
    err = float(np.abs(robot.data.joint_pos[0, arm_ids].cpu().numpy() - r["q_ik"]).max())
    trials += 1
    feasible += int(err < 0.05)
    print(f"    样本 {k}: 标签下发 {FEAS_STEPS} 步后关节误差 {err:.4f} rad "
          f"(|Δq| 需要 {np.abs(r['q_ik'] - r['q']).max():.3f})")
print(f"  可执行性: {feasible}/{trials} 个标签能通过原始动作接口到位")

torch.save(dict(X=X.cpu(), Y=Y.cpu(), episode=torch.tensor([r["episode"] for r in records]),
                tr=torch.tensor(tr_idx), va=torch.tensor(va_idx), te=torch.tensor(te_idx)),
           OUT / "dataset.pt")
print(f"  数据集已保存 {OUT}/dataset.pt")

# ------------------------------------------------------------------ teacher --------------
print("\n" + "=" * 92)
print("3. 监督教师 (actor 同构) 在独立测试集上的表现")
print("=" * 92)


def make_net():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 512), torch.nn.ELU(),
        torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 8),
    ).to(DEV)


teacher = make_net()
opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
tr = torch.tensor(tr_idx, device=DEV)
va = torch.tensor(va_idx, device=DEV)
te = torch.tensor(te_idx, device=DEV)
best_v, best_s, patience = float("inf"), None, 0
for epoch in range(args_cli.epochs):
    teacher.train()
    perm = tr[torch.randperm(len(tr))]
    for b in range(0, len(perm), 256):
        mb = perm[b:b + 256]
        opt.zero_grad()
        torch.mean((teacher(X[mb])[:, 2:] - Y[mb][:, 2:]) ** 2).backward()
        opt.step()
    teacher.eval()
    with torch.inference_mode():
        v = float(torch.mean((teacher(X[va])[:, 2:] - Y[va][:, 2:]) ** 2))
    if v < best_v - 1e-9:
        best_v, best_s, patience = v, {k: t.clone() for k, t in teacher.state_dict().items()}, 0
    else:
        patience += 1
        if patience > 60:
            break
teacher.load_state_dict(best_s)
print(f"  早停于 epoch {epoch}, val MSE {best_v:.5f}")


def metrics(pred_action, idx, tag):
    """Joint error, direction error and the FK task-space error the action would produce."""
    a = pred_action[:, 2:] * SCALE                       # rad, relative to default
    # the FK check needs the absolute target configuration, not the normalised label
    q_tgt = np.stack([records[i]["q_ik"] for i in idx])
    q_now = np.stack([records[i]["q"] for i in idx])
    q_pred = DEFAULT_Q + a.cpu().numpy()
    per_joint = float(np.mean(np.abs(q_pred - q_tgt)))
    # direction error only where both motions are large enough to have a direction
    angs = []
    for k in range(len(idx)):
        d_p, d_t = q_pred[k] - q_now[k], q_tgt[k] - q_now[k]
        if np.linalg.norm(d_p) > 0.02 and np.linalg.norm(d_t) > 0.02:
            c = np.clip(d_p @ d_t / (np.linalg.norm(d_p) * np.linalg.norm(d_t)), -1, 1)
            angs.append(math.degrees(math.acos(c)))
    pos_errs, rot_errs = [], []
    for k in range(len(idx)):
        T = tk.fk(q_pred[k])
        pos_errs.append(np.linalg.norm(T[:3, 3] - records[idx[k]]["cmd_pos"]))
        rot_errs.append(np.linalg.norm(tk.log_R(records[idx[k]]["cmd_rot"] @ T[:3, :3].T)))
    print(f"  {tag:<22} 关节 MAE {per_joint:.4f} rad   方向 {np.median(angs):.1f}° "
          f"({len(angs)} 个有效)   FK 位置 {np.mean(pos_errs):.4f} m   姿态 "
          f"{math.degrees(np.mean(rot_errs)):.2f}°")


with torch.inference_mode():
    metrics(teacher(X[te]), te_idx, "教师(测试集)")
    metrics(teacher(X[va]), va_idx, "教师(验证集)")
    # the current PPO actor on the same held-out samples, for reference
    metrics(net_actor(X[te]), te_idx, "现有 actor(测试集)")

torch.save(dict(X=X.cpu(), Y=Y.cpu(), episode=torch.tensor([r["episode"] for r in records]),
                tr=torch.tensor(tr_idx), va=torch.tensor(va_idx), te=torch.tensor(te_idx)),
           OUT / "dataset.pt")
torch.save(teacher.state_dict(), OUT / "teacher.pt")
(OUT / "stats.json").write_text(json.dumps(dict(
    n=len(records), dropped=dropped, n_train=len(tr_idx), n_val=len(va_idx), n_test=len(te_idx),
    scale=SCALE, default_q=DEFAULT_Q.tolist(), epochs=int(epoch), val_mse=best_v,
), indent=2))
print(f"\n  已保存 {OUT}/dataset.pt, teacher.pt, stats.json")
env.close()
app.close()
