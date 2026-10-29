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

"""Stage 4.5i: real-rollout supervision with the *fixed* command sampler.

Stage 4.5e's set was collected while `UniformTCPBodyPoseCommand._resample_command` drew all
three position offsets and all three Euler angles from one aliased tensor, so every one of
its 3142 commands satisfied `offset_x == offset_y == offset_z` and `roll == pitch == yaw` --
a two-dimensional line inside the six-dimensional box. That set can therefore never teach
the full map, and a policy fitted to it is out of distribution on any independent-axis
command, which is exactly the "unseen command" failure of 4.5e-4.5h.

This re-collects the same kind of data with the sampler fixed. Everything else is the 4.5e
recipe: the *deployment* policy drives, the env resamples its own commands, the base is
moving, the three-frame history is the one `env.step` produced, and the label is
`local_ik(q_now, target)` verified by FK before it is kept.

Compared with `teleop_ik_teacher.py` it stores the command and the current joint vector per
sample, so the mixed-training script can group samples by command and split by *command*
rather than by episode.

Usage:
    python scripts/ranger_cr10/teleop_ik_collect.py \
        --policy logs/rsl_rl/ranger_cr10_teleop/2026-10-16_19-21-59_resume200/exported/policy.pt \
        --out logs/ik_teacher_box
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

parser = argparse.ArgumentParser(description="Collect IK supervision from real rollouts.")
parser.add_argument("--policy", required=True, help="TorchScript policy driving the rollouts")
parser.add_argument("--out", default="logs/ik_teacher_box")
parser.add_argument("--episodes", type=int, default=6)
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--every", type=int, default=4)
parser.add_argument("--num_envs", type=int, default=96)
parser.add_argument("--explore", type=float, default=0.15, help="std of the action noise")
parser.add_argument("--seed", type=int, default=0)
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
DEFAULT_Q = np.asarray(tk.DEFAULT_Q, dtype=np.float64)

cfg = RangerCr10TeleopEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
cfg.commands.ee_pose.debug_vis = False
cfg.observations.policy.enable_corruption = False  # labels must correspond to the clean obs
env = gym.make("RANGER-CR10-TELEOP-WBC", cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
ee_cmd = u.command_manager.get_term("ee_pose")
arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
actor = torch.jit.load(args_cli.policy, map_location=u.device)
SCALE = u.action_manager.get_term("joint_pos").cfg.scale
N = u.num_envs
assert np.allclose(robot.data.default_joint_pos[0, arm_ids].cpu().numpy(), DEFAULT_Q, atol=1e-5)

print("=" * 96)
print(f"1. 采集: {args_cli.episodes} 回合 x {args_cli.steps} 步 x {N} env, 每 {args_cli.every} 步记录")
print("=" * 96)

torch.manual_seed(args_cli.seed)
np.random.seed(args_cli.seed)
X, Y, Q, CMD, EP, ENV = [], [], [], [], [], []
dropped = saturated = 0
for ep in range(args_cli.episodes):
    obs, _ = env.reset(seed=args_cli.seed + ep)
    kept = cand = 0
    for step in range(args_cli.steps):
        with torch.inference_mode():
            action = actor(obs["policy"])
        if args_cli.explore > 0:
            action = action + args_cli.explore * torch.randn_like(action)
        obs, _, _, _, _ = env.step(action)
        if step % args_cli.every:
            continue
        q_all = robot.data.joint_pos[:, arm_ids].cpu().numpy()
        cmd_pos = ee_cmd.pose_command_b[:, :3].cpu().numpy()
        cmd_quat = ee_cmd.pose_command_b[:, 3:].cpu().numpy().astype(np.float64)
        cmd_quat /= np.linalg.norm(cmd_quat, axis=1, keepdims=True)
        o_all = obs["policy"].cpu().numpy()
        for i in range(N):
            cand += 1
            q_ik, ep_p, ep_r, max_step, _ = tk.local_ik(q_all[i], cmd_pos[i], tk.quat_to_R(cmd_quat[i]))
            if max_step >= 0.08 - 1e-12:
                saturated += 1
            if ep_p > 1e-3 or ep_r > 1e-3:
                dropped += 1
                continue
            X.append(o_all[i].astype(np.float32))
            # padded to the actor's 8 action dims; the two chassis channels are not supervised
            Y.append(np.concatenate([[0.0, 0.0], (q_ik - DEFAULT_Q) / SCALE]).astype(np.float32))
            Q.append(q_all[i].astype(np.float32))
            CMD.append(np.concatenate([cmd_pos[i], cmd_quat[i]]).astype(np.float32))
            EP.append(ep)
            ENV.append(i)
            kept += 1
    print(f"  episode {ep}: 有效 {kept} / {cand}")

X = np.stack(X)
Y = np.stack(Y)
CMD = np.stack(CMD)
print(
    f"  合计 {len(X)} 条, 丢弃 {dropped} ({dropped / max(1, dropped + len(X)) * 100:.1f}%), "
    f"步长饱和 {saturated} 条 (记录, 不丢弃)"
)

# ---- command diversity: the reason this set exists --------------------------------
pos_off = CMD[:, :3] - np.asarray([0.5519, -0.0742, 0.4696])
rpy = np.stack(
    [
        np.degrees(
            tk.R_to_rpy(tk.quat_to_R(c[3:]) @ tk.quat_to_R(np.array([0.032072, -0.875128, 0.469424, -0.112978])).T)
        )
        for c in CMD
    ]
)
corr = np.corrcoef(np.column_stack([pos_off, rpy]).T)
offdiag = [abs(corr[a, b]) for a in range(6) for b in range(a + 1, 6)]
print(f"  命令: 位置偏移标准差 {np.round(pos_off.std(0), 3)}, 姿态偏移标准差 {np.round(rpy.std(0), 1)}°")
print(f"  六分量之间 |相关系数| 最大 {max(offdiag):.3f}, 中位 {np.median(offdiag):.3f}  (退化采样器是 1.000)")
print(
    f"  x==y 比例 {np.mean(np.abs(pos_off[:, 0] - pos_off[:, 1]) < 1e-6) * 100:.1f}%, "
    f"roll==pitch 比例 {np.mean(np.abs(rpy[:, 0] - rpy[:, 1]) < 1e-3) * 100:.1f}%"
)

torch.save(
    dict(
        X=torch.tensor(X),
        Y=torch.tensor(Y),
        q=torch.tensor(Q),
        cmd=torch.tensor(CMD),
        episode=torch.tensor(EP),
        env=torch.tensor(ENV),
    ),
    OUT / "dataset.pt",
)
(OUT / "stats.json").write_text(
    json.dumps(
        dict(
            n=len(X),
            dropped=dropped,
            saturated=saturated,
            num_envs=N,
            episodes=args_cli.episodes,
            steps=args_cli.steps,
            every=args_cli.every,
            explore=args_cli.explore,
            seed=args_cli.seed,
            policy=args_cli.policy,
            cmd_std_pos=pos_off.std(0).tolist(),
            cmd_std_rpy_deg=rpy.std(0).tolist(),
            cmd_max_abs_corr=float(max(offdiag)),
        ),
        indent=2,
    )
)
print(f"\n已写出 {OUT / 'dataset.pt'}")

# one executable-label spot check, so a silently broken label convention cannot pass
from isaaclab.app import AppLauncher  # noqa: F401,E402  (already launched)

print("\n标签自检: FK(q_ik) 与命令的残差")
errs_p, errs_r = [], []
for k in range(0, len(X), max(1, len(X) // 200)):
    T = tk.fk(DEFAULT_Q + SCALE * Y[k])
    errs_p.append(np.linalg.norm(T[:3, 3] - CMD[k, :3]))
    errs_r.append(math.degrees(np.linalg.norm(tk.log_R(tk.quat_to_R(CMD[k, 3:]) @ T[:3, :3].T))))
print(f"  位置最大 {max(errs_p) * 1000:.2f} mm, 姿态最大 {max(errs_r):.3f}°  (过滤阈值 1 mm / 0.057°)")

env.close()
app.close()
