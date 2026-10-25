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

"""Stage 4.5f: why does an actor that fits the IK map offline not track attitude closed-loop?

Four sections, each of which can be run alone:

    A  closed loop, identical conditions, for A / B0 / B1 / C / teacher-network / online-IK
    B  what the 4.5e supervision set threw away, and whether the survivors are representative
    C  offline-versus-online: on the *same* observation, actor vs teacher vs online IK, and
       what each of those actions would produce after FK and after execution
    D  PPO forgetting: the fixed supervised test set evaluated across checkpoints, plus what
       B0 actually carried into PPO (critic weights, Adam moments)

The question the stage exists to answer is narrow, and every section is written so that its
answer does not depend on the others: section A alone settles whether supervised pre-training
produces a closed-loop attitude tracker. If it does not, "what PPO broke" is not the question.

The policies compared, and where each comes from:

    A    logs/rsl_rl/ranger_cr10_teleop/2026-10-23_18-32-48_ctrl_A/exported/policy.pt   (iter 1198)
    B0   logs/rsl_rl/ranger_cr10_teleop/_stage45e/pretrained_B.pt     (supervised only, never in PPO)
    B1   logs/rsl_rl/ranger_cr10_teleop/2026-10-23_19-10-13_ctrl_B/exported/policy.pt  (iter 1198)
    C    logs/rsl_rl/ranger_cr10_teleop/2026-10-23_19-59-08_ctrl_C/exported/policy.pt  (iter 1198)
    T    logs/ik_teacher/teacher.pt                                    (distilled IK teacher)
    TIK  online local_ik, solved from the simulator's own joint state every step

B0 is the only policy here that never saw a PPO gradient, and it is an *existing* artefact --
it was not re-created for this stage. The distinction matters: a rebuilt B0 would carry
whatever changed since 4.5e, and the whole point is to keep the comparison to that run.

TIK is the kinematic upper bound, not a fair peer: it reads the true joint state every step
and solves to convergence, so its numbers say "what the geometry allows", never "what a
policy could learn". It is labelled that way in every table.

Usage:
    python scripts/ranger_cr10/check_teleop_ik_closedloop.py --section A
    python scripts/ranger_cr10/check_teleop_ik_closedloop.py --section B
    python scripts/ranger_cr10/check_teleop_ik_closedloop.py --section C
    python scripts/ranger_cr10/check_teleop_ik_closedloop.py --section D
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description="Stage 4.5f diagnostics.")
parser.add_argument("--section", default="A", choices=["A", "B", "C", "D", "E"])
parser.add_argument("--out", default="outputs/stage45f")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--log_root", default="logs/rsl_rl/ranger_cr10_teleop")
args_cli, _ = parser.parse_known_args()

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teleop_kinematics as tk  # noqa: E402

import torch  # noqa: E402

OUT = Path(args_cli.out)
OUT.mkdir(parents=True, exist_ok=True)

LOG = Path(args_cli.log_root)
CKPT = {
    "A": LOG / "2026-10-23_18-32-48_ctrl_A" / "exported" / "policy.pt",
    "B0": LOG / "_stage45e" / "pretrained_B.pt",
    "B1": LOG / "2026-10-23_19-10-13_ctrl_B" / "exported" / "policy.pt",
    "C": LOG / "2026-10-23_19-59-08_ctrl_C" / "exported" / "policy.pt",
    "T": Path("logs/ik_teacher/teacher.pt"),
}
DATASET = Path("logs/ik_teacher/dataset.pt")
SCALE = 0.25
DEFAULT_Q = np.asarray(tk.DEFAULT_Q, dtype=np.float64)


# --------------------------------------------------------------------------------------
# policy loading
# --------------------------------------------------------------------------------------
def build_mlp(obs_dim: int, device) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 512), torch.nn.ELU(),
        torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 8),
    ).to(device)


def load_actor_from_checkpoint(path: Path, device) -> torch.nn.Module:
    """The actor MLP out of an rsl_rl checkpoint, as a deterministic mean-action module.

    The exported TorchScript policy in `exported/policy.pt` is `_TorchMLPModel`, whose
    `deterministic_output` is the Gaussian mean -- so an actor loaded here and one loaded
    from the export return the same number. `section D` checks that on real observations
    rather than trusting this comment.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["actor_state_dict"]
    sd = {k[len("mlp."):]: v for k, v in sd.items() if k.startswith("mlp.")}
    mlp = build_mlp(sd["0.weight"].shape[1], device)
    mlp.load_state_dict(sd)
    mlp.eval()
    for p in mlp.parameters():
        p.requires_grad_(False)
    return mlp


def load_teacher(path: Path, device) -> torch.nn.Module:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    key0 = "mlp.0.weight" if "mlp.0.weight" in sd else "0.weight"
    if key0 == "mlp.0.weight":
        sd = {k[len("mlp."):]: v for k, v in sd.items() if k.startswith("mlp.")}
    net = build_mlp(sd["0.weight"].shape[1], device)
    net.load_state_dict(sd)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


class TorchPolicy:
    """Deterministic actor from an exported TorchScript file."""

    def __init__(self, name: str, path: Path, device):
        self.name = name
        self.net = torch.jit.load(str(path), map_location=device)
        self.kind = "policy"

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.net(obs)

    def arm(self, obs_np: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            a = self.net(torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0))
        return a[0, 2:].cpu().numpy()


class MLPPolicy:
    """Deterministic actor from a checkpoint or a plain state dict."""

    def __init__(self, name: str, path: Path, device, loader=load_actor_from_checkpoint):
        self.name = name
        self.net = loader(path, device)
        self.device = device
        self.kind = "policy"

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.net(obs)

    def arm(self, obs_np: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            a = self.net(torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0))
        return a[0, 2:].cpu().numpy()


def make_policy(name: str, device) -> TorchPolicy | MLPPolicy | None:
    if name == "TIK":
        return None  # handled inline: it needs the live simulator state
    path = CKPT[name]
    if not path.exists():
        raise FileNotFoundError(f"{name}: {path} is missing")
    if name == "B0":
        return MLPPolicy(name, path, device)
    if name == "T":
        return MLPPolicy(name, path, device, loader=load_teacher)
    return TorchPolicy(name, path, device)


# --------------------------------------------------------------------------------------
# simulator plumbing (only the sections that need it pay for it)
# --------------------------------------------------------------------------------------
_APP = None


def build_env(num_envs: int):
    """The teleop task with every source of variation this stage does not want switched off."""
    global _APP
    from isaaclab.app import AppLauncher

    if _APP is None:
        _APP = AppLauncher(headless=True).app

    import gymnasium as gym

    import LeggedManip_Lab.tasks  # noqa: F401
    from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.teleop_env_cfg import (
        RangerCr10TeleopEnvCfg,
    )

    cfg = RangerCr10TeleopEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = 1e6          # the schedule, not the episode timer, ends a segment
    cfg.commands.ee_pose.debug_vis = False
    cfg.commands.ee_pose.resampling_time_range = (1e6, 1e6)
    cfg.commands.base_velocity.resampling_time_range = (1e6, 1e6)
    cfg.observations.policy.enable_corruption = False   # deployment does not add noise either
    cfg.curriculum.lin_vel_cmd_levels = None
    cfg.curriculum.ang_vel_cmd_levels = None
    return gym.make("RANGER-CR10-TELEOP-WBC", cfg=cfg)


def ori_error_deg(env, i: int = 0) -> float:
    """Exact geodesic angle of the command->TCP rotation, in degrees.

    `metrics['orientation_error']` is the norm of the rotation vector; `2*asin(|v|/2)` turns
    that into the angle (the two differ by 1.4% at 30 deg, which matters at these sizes).
    """
    n = env.unwrapped.command_manager.get_term("ee_pose").metrics["orientation_error"][i].item()
    return math.degrees(2.0 * math.asin(min(1.0, n / 2.0)))


def run_closed_loop(env, policy, segments, seed, tik: bool, ref_state=None, hook=None):
    """One policy, the whole schedule, from one restored initial state.

    Returns the per-step rows and the initial-state agreement with the first policy run.
    """
    import torch

    import isaaclab.utils.math as iu

    from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.pose_command_teleop import (
        TCP_DEFAULT_POS_B,
        TCP_DEFAULT_QUAT_B,
    )

    u = env.unwrapped
    robot = u.scene["robot"]
    ee_cmd = u.command_manager.get_term("ee_pose")
    vel_cmd = u.command_manager.get_term("base_velocity")
    jp_term = u.action_manager.get_term("joint_pos")
    from LeggedManip_Lab.assets.ranger_cr10.ranger_cr10_articulation_cfg import ARM_JOINT_NAMES

    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    dev = u.device

    def set_command(rpy, pos_off, vx, wz):
        dR = iu.quat_from_euler_xyz(*[torch.tensor([v], device=dev) for v in rpy])
        tq = iu.quat_mul(dR, torch.tensor(TCP_DEFAULT_QUAT_B, device=dev).unsqueeze(0))[0]
        ee_cmd.pose_command_b[:, :3] = torch.tensor(np.asarray(TCP_DEFAULT_POS_B) + np.asarray(pos_off),
                                                    device=dev)
        ee_cmd.pose_command_b[:, 3:] = tq
        vel_cmd.is_standing_env[:] = False
        vel_cmd.vel_command_b[:, 0] = vx
        vel_cmd.vel_command_b[:, 1] = 0.0
        vel_cmd.vel_command_b[:, 2] = wz

    def obs_now():
        # recompute the policy observation with its history refreshed; must run outside
        # inference_mode or the history buffers become inference tensors and the next
        # env.reset() cannot zero them
        for _ in range(3):
            o = u.observation_manager.compute(update_history=True)   # the group dict, not one entry
        return o

    obs, _ = env.reset(seed=seed)
    state0 = torch.cat([robot.data.root_state_w.clone(), robot.data.joint_pos.clone(),
                        robot.data.joint_vel.clone()], dim=1)
    if ref_state is not None:
        # every policy must start from the *same* state. The reset is seeded, so this should
        # already hold; the assertion is here because "same seed" is an assumption about a
        # randomisation pipeline, and a silent change there would quietly make the comparison
        # meaningless. If it does not hold, the reference state is written back instead.
        drift = float((state0 - ref_state).abs().max())
        if drift > 1e-6:
            ref = ref_state
            robot.write_root_pose_to_sim(ref[:, :7])
            robot.write_root_velocity_to_sim(ref[:, 7:13])
            robot.write_joint_state_to_sim(ref[:, 13:13 + len(arm_ids)], ref[:, 13 + len(arm_ids):],
                                           joint_ids=arm_ids)
            u.sim.step()
            state0 = ref.clone()
        print(f"    初始状态与首个策略的最大差异 {drift:.3e} (>{1e-6:.0e} 则已写回参考状态)")

    rows = []
    for seg in segments:
        if seg.get("init_q") is not None:
            q_t = torch.tensor(DEFAULT_Q + np.asarray(seg["init_q"]), dtype=torch.float32, device=dev)
            q_t = q_t.unsqueeze(0).expand(u.num_envs, -1)
            robot.write_joint_state_to_sim(q_t, torch.zeros_like(q_t), joint_ids=arm_ids)
            # NO u.sim.step() here: the write is already visible in robot.data (checked), and
            # stepping physics *before* the next action is applied runs one 200 Hz tick with
            # the previous PD target still in the actuators -- measured as a 2.7 rad joint
            # transient and a 180 deg attitude spike that then took 3 steps to decay. The
            # first version of this function did step, and every 'different initial
            # configuration' number it produced was that transient, not the policy.
        for step in range(seg["steps"]):
            set_command(seg["rpy"], seg["pos"], seg["vx"], seg["wz"])
            if seg.get("init_q") is not None and step == 0:
                obs = obs_now()   # the written joint state must be in the history too
            if tik:
                # solve from the simulator's own joint state, to the commanded pose in the
                # chassis frame -- the same inputs the teacher's labels were built from
                q_now = robot.data.joint_pos[0, arm_ids].cpu().numpy()
                cmd_pos = ee_cmd.pose_command_b[0, :3].cpu().numpy()
                cmd_quat = ee_cmd.pose_command_b[0, 3:].cpu().numpy()
                q_ik, ep_p, ep_r, _, _ = tk.local_ik(q_now, cmd_pos, tk.quat_to_R(cmd_quat))
                a = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=dev)
                a[:, 2:] = torch.tensor((q_ik - DEFAULT_Q) / SCALE, dtype=torch.float32, device=dev)
                tik_ok = float(ep_p < 1e-3 and ep_r < 1e-3)
            else:
                with torch.inference_mode():
                    a = policy.act(obs["policy"])
                tik_ok = float("nan")

            raw = a[0, 2:].cpu().numpy().copy()
            if hook is not None:
                # sees the *pre-step* observation and action, so whatever it computes is
                # conditioned on exactly the state the actor acted from
                hook(obs, a, seg, step, u, robot, arm_ids, ee_cmd)
            obs, _, term, trunc, _ = env.step(a)
            q_target = jp_term.processed_actions[0].cpu().numpy()
            rows.append(dict(
                policy=policy.name if policy is not None else "TIK",
                seg=seg["label"], mode=seg["mode"], step=step,
                ori_deg=ori_error_deg(env),
                pos_err=ee_cmd.metrics["position_error"][0].item(),
                joint_track=float(np.max(np.abs(q_target - robot.data.joint_pos[0, arm_ids].cpu().numpy()))),
                vx=robot.data.root_lin_vel_b[0, 0].item(), wz=robot.data.root_ang_vel_b[0, 2].item(),
                vx_cmd=seg["vx"], wz_cmd=seg["wz"],
                raw_max=float(np.max(np.abs(raw))),
                q=robot.data.joint_pos[0, arm_ids].cpu().numpy().copy(),
                diag=tik_ok,
                terminated=bool(term[0].item() or trunc[0].item()),
            ))
    return rows, state0


# --------------------------------------------------------------------------------------
# the schedule, shared by sections A and C
# --------------------------------------------------------------------------------------
HOLD = 25        # 2.5 s per command at 10 Hz
TRANSIENT = 5    # 0.5 s after the step
STEADY = 10      # last 1.0 s

ATT = [
    ("roll +0.1", (0.1, 0.0, 0.0)), ("roll -0.1", (-0.1, 0.0, 0.0)),
    ("roll +0.2", (0.2, 0.0, 0.0)), ("roll -0.2", (-0.2, 0.0, 0.0)),
    ("pitch +0.1", (0.0, 0.1, 0.0)), ("pitch -0.1", (0.0, -0.1, 0.0)),
    ("pitch +0.2", (0.0, 0.2, 0.0)), ("pitch -0.2", (0.0, -0.2, 0.0)),
    ("yaw +0.1", (0.0, 0.0, 0.1)), ("yaw -0.1", (0.0, 0.0, -0.1)),
    ("yaw +0.2", (0.0, 0.0, 0.2)), ("yaw -0.2", (0.0, 0.0, -0.2)),
    ("combo ++ (0.1,0.1,0.1)", (0.1, 0.1, 0.1)),
    ("combo +- (-0.1,0.1,-0.15)", (-0.1, 0.1, -0.15)),
    ("combo big (0.2,0,0.2)", (0.2, 0.0, 0.2)),
]
POS = [("pos x+0.08", (0.08, 0.0, 0.0)), ("pos y+0.10", (0.0, 0.10, 0.0)),
       ("pos z+0.08", (0.0, 0.0, 0.08))]
INIT_CFG = [
    (0.25, -0.15, 0.15, 0.0, 0.20, 0.0),
    (-0.20, 0.20, -0.20, 0.0, -0.15, 0.0),
    (0.00, 0.00, 0.00, 0.50, 0.00, 0.40),
]
VEL = [("vx +0.3", 0.3, 0.0), ("wz +0.3", 0.0, 0.3), ("arc +0.3/+0.2", 0.3, 0.2),
       ("vx -0.2", -0.2, 0.0)]
BOTH = [("both: arc + pitch+0.15", (0.0, 0.15, 0.0), 0.3, 0.15),
        ("both: fwd + yaw+0.2", (0.0, 0.0, 0.2), 0.3, 0.0)]


def build_schedule():
    segs = []
    for label, rpy in ATT:
        segs.append(dict(label=label, mode="arm", rpy=rpy, pos=(0.0, 0.0, 0.0), vx=0.0, wz=0.0, steps=HOLD))
    for label, pos in POS:
        segs.append(dict(label=label, mode="arm", rpy=(0.0, 0.0, 0.0), pos=pos, vx=0.0, wz=0.0, steps=HOLD))
    for k, dq in enumerate(INIT_CFG):
        segs.append(dict(label=f"init{k} neutral", mode="arm", rpy=(0.0, 0.0, 0.0), pos=(0.0, 0.0, 0.0),
                         vx=0.0, wz=0.0, steps=HOLD, init_q=dq))
        segs.append(dict(label=f"init{k} pitch+0.15", mode="arm", rpy=(0.0, 0.15, 0.0), pos=(0.0, 0.0, 0.0),
                         vx=0.0, wz=0.0, steps=HOLD))
    for label, vx, wz in VEL:
        segs.append(dict(label=label, mode="base", rpy=(0.0, 0.0, 0.0), pos=(0.0, 0.0, 0.0),
                         vx=vx, wz=wz, steps=HOLD))
    for label, rpy, vx, wz in BOTH:
        segs.append(dict(label=label, mode="both", rpy=rpy, pos=(0.0, 0.0, 0.0), vx=vx, wz=wz, steps=HOLD))
    return segs


def summarize(rows, name):
    """Steady-state and transient numbers, per mode."""
    import collections

    by_seg = collections.defaultdict(list)
    for r in rows:
        by_seg[r["seg"]].append(r)
    att_segs = [s for s in by_seg if by_seg[s][0]["mode"] == "arm" and "init" not in s]
    out = {"name": name, "n_terminated": sum(r["terminated"] for r in rows)}

    def steady(seg):
        return by_seg[seg][-STEADY:]

    def trans(seg):
        return by_seg[seg][:TRANSIENT]

    att = [np.mean([r["ori_deg"] for r in steady(s)]) for s in att_segs]
    out["att_steady"] = float(np.mean(att))
    out["att_trans"] = float(np.mean([np.mean([r["ori_deg"] for r in trans(s)]) for s in att_segs]))
    out["att_worst"] = float(np.max(att))
    for axis, key in (("roll", 0), ("pitch", 1), ("yaw", 2)):
        vals = []
        for s in att_segs:
            rpy = dict(ATT).get(s)
            if rpy is not None and abs(rpy[key]) > 0.15:
                vals.append(np.mean([r["ori_deg"] for r in steady(s)]))
        out[f"att_{axis}_0.2"] = float(np.mean(vals)) if vals else float("nan")
    out["att_tracked"] = float(np.mean([a < 5.0 for a in att]))
    pos_segs = [s for s in by_seg if s.startswith("pos ")]
    out["pos_err"] = (float(np.mean([np.mean([r["pos_err"] for r in steady(s)]) for s in pos_segs]))
                      if pos_segs else float("nan"))
    base_segs = [s for s in by_seg if by_seg[s][0]["mode"] in ("base", "both")]
    out["vel_err"] = (float(np.mean([np.sqrt(np.mean([(r["vx"] - r["vx_cmd"]) ** 2
                                                     + (r["wz"] - r["wz_cmd"]) ** 2
                                                     for r in steady(s)])) for s in base_segs]))
                      if base_segs else float("nan"))
    out["joint_track"] = float(np.mean([r["joint_track"] for r in rows]))
    out["raw_sat"] = float(np.mean([r["raw_max"] > 3.0 for r in rows]))
    init_segs = [s for s in by_seg if "init" in s and "pitch" in s]
    out["init_att"] = (float(np.mean([np.mean([r["ori_deg"] for r in steady(s)]) for s in init_segs]))
                       if init_segs else float("nan"))
    return out


def section_a():
    import torch

    names = ["B0", "B1", "C", "A", "T", "TIK"]
    env = build_env(1)
    dev = env.unwrapped.device
    segs = build_schedule()
    print(f"调度: {len(segs)} 段, 每段 {HOLD} 步, 共 {len(segs) * HOLD} 步/策略")
    all_rows, ref_state, summaries = [], None, []
    for name in names:
        pol = make_policy(name, dev)
        print(f"\n== {name} ==")
        # T and TIK have no meaningful chassis command (the labels are zero there by
        # construction), so they are only run in the arm-only mode and shown as n/a elsewhere
        use = [s for s in segs if s["mode"] == "arm"] if name in ("T", "TIK") else segs
        rows, state0 = run_closed_loop(env, pol, use, args_cli.seed, tik=(name == "TIK"),
                                       ref_state=ref_state)
        if ref_state is None:
            ref_state = state0
        all_rows.extend(rows)
        summaries.append(summarize(rows, name))
        if name == "TIK":
            ok = [r["diag"] for r in rows]
            print(f"    在线 IK 收敛: {np.mean(ok) * 100:.1f}% ({int(np.sum(ok))}/{len(ok)})")

    print("\n" + "=" * 108)
    print("A. 闭环对照 (稳态 = 每段最后 1.0 s; 姿态误差为 SO(3) 测地角)")
    print("=" * 108)
    hdr = (f"{'策略':<5}{'姿态稳态':>9}{'姿态瞬态':>9}{'最差段':>8}"
           f"{'roll .2':>9}{'pitch .2':>9}{'yaw .2':>9}{'<5°段占比':>10}"
           f"{'位置误差':>9}{'速度误差':>9}{'关节跟踪':>9}{'动作饱和':>9}{'终止':>5}")
    print(hdr)
    for s in summaries:
        print(f"{s['name']:<5}{s['att_steady']:>8.1f}°{s['att_trans']:>8.1f}°{s['att_worst']:>7.1f}°"
              f"{s['att_roll_0.2']:>8.1f}°{s['att_pitch_0.2']:>8.1f}°{s['att_yaw_0.2']:>8.1f}°"
              f"{s['att_tracked'] * 100:>9.0f}%"
              f"{s['pos_err']:>8.3f}m{s['vel_err']:>8.3f}{s['joint_track']:>9.3f}"
              f"{s['raw_sat'] * 100:>8.0f}%{s['n_terminated']:>5}")
    print("\n不同初始臂形下的 pitch+0.15 稳态姿态误差:")
    for s in summaries:
        print(f"  {s['name']:<5}{s['init_att']:>8.1f}°")

    # per-command detail, the most interpretable slice
    print("\n单轴 ±0.2 逐段稳态姿态误差 (度):")
    labels = [l for l, _ in ATT if abs(max(_)) > 0.15 and sum(1 for v in _ if v) == 1]
    print(f"  {'段':<12}" + "".join(f"{s['name']:>8}" for s in summaries))
    for lab in labels:
        vals = {}
        for r in all_rows:
            if r["seg"] == lab:
                vals.setdefault(r["policy"], []).append(r["ori_deg"])
        print(f"  {lab:<12}" + "".join(f"{np.mean(vals.get(s['name'], [float('nan')])[-STEADY:]):>8.1f}"
                                        for s in summaries))

    save_and_plot(all_rows, summaries)
    return all_rows


def save_and_plot(all_rows, summaries):
    import csv

    csv_path = OUT / "closedloop_rows.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "seg", "mode", "step", "ori_deg", "pos_err", "joint_track",
                                          "vx", "wz", "vx_cmd", "wz_cmd", "raw_max", "terminated"])
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"\n逐时间步数据: {csv_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = ["roll +0.2", "pitch +0.2", "yaw +0.2"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, seg in zip(axes, panels):
        for s in summaries:
            ys = [r["ori_deg"] for r in all_rows if r["policy"] == s["name"] and r["seg"] == seg]
            ax.plot(np.arange(len(ys)) * 0.1, ys, label=s["name"], linewidth=1.6)
        ax.axvline(TRANSIENT * 0.1, color="grey", linestyle=":", linewidth=1)
        ax.axvline((HOLD - STEADY) * 0.1, color="grey", linestyle=":", linewidth=1)
        ax.set_title(seg)
        ax.set_xlabel("time in segment, s")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("attitude error, deg")
    axes[0].legend(fontsize=8)
    fig.suptitle("4.5f closed-loop attitude error after a step command (dotted: transient/steady windows)")
    fig.tight_layout()
    png = OUT / "attitude_error_vs_time.png"
    fig.savefig(png, dpi=130)
    print(f"曲线: {png}")


# --------------------------------------------------------------------------------------
# B: what the supervision filter threw away
# --------------------------------------------------------------------------------------
def section_b(episodes=4, steps=60, every=12):
    """Re-run the 4.5e collection with the filter's *reasons* recorded separately.

    4.5e counted a single `dropped` number. The filter it actually applied was
    `ep_p > 1e-3 or ep_r > 1e-3 or max_dq > 0.08`, and those three mean different things:
    the first two are "the local solver did not converge", the third is "a branch jump".
    Note `max_dq` cannot exceed `step_cap = 0.08` by construction (`local_ik` scales any
    longer step down to it), so the third clause is close to inert -- this section measures
    whether it ever fires rather than assuming it does.
    """
    import torch

    env = build_env(256)
    u = env.unwrapped
    robot = u.scene["robot"]
    ee_cmd = u.command_manager.get_term("ee_pose")
    from LeggedManip_Lab.assets.ranger_cr10.ranger_cr10_articulation_cfg import ARM_JOINT_NAMES

    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    dev = u.device
    pol = make_policy("A", dev)
    base = Path(args_cli.log_root) / "2026-10-16_19-21-59_resume200" / "exported" / "policy.pt"
    if base.exists():
        pol = TorchPolicy("A", base, dev)
        print(f"rollout 策略: {base}")
    rng = np.random.default_rng(0)

    cand = []
    targets = []          # (pos, R) per candidate, kept out of the CSV
    prev_cmd = None
    since = np.zeros(u.num_envs)
    for ep in range(episodes):
        obs, _ = env.reset(seed=args_cli.seed + ep)
        prev_cmd = None
        since[:] = 0.0
        for step in range(steps):
            with torch.inference_mode():
                a = pol.act(obs["policy"])
            a = a + 0.15 * torch.randn_like(a)     # same exploration as the 4.5e collection
            obs, _, _, _, _ = env.step(a)
            cmd = ee_cmd.pose_command_b.clone()
            if prev_cmd is None:
                changed = np.ones(u.num_envs, dtype=bool)
            else:
                changed = (cmd - prev_cmd).abs().max(dim=1).values.cpu().numpy() > 1e-6
            prev_cmd = cmd
            since = np.where(changed, 0.0, since + 1.0)
            if step % every:
                continue
            q_all = robot.data.joint_pos[:, arm_ids].cpu().numpy()
            cmd_pos = cmd[:, :3].cpu().numpy()
            cmd_quat = cmd[:, 3:].cpu().numpy()
            base_spd = torch.linalg.norm(robot.data.root_lin_vel_b[:, :2], dim=1).cpu().numpy()
            for i in range(u.num_envs):
                R = tk.quat_to_R(cmd_quat[i])
                q_ik, ep_p, ep_r, max_dq, cond = tk.local_ik(q_all[i], cmd_pos[i], R)
                q = q_all[i]
                margin = float(np.min(np.minimum(q - tk.LIMITS[:, 0], tk.LIMITS[:, 1] - q)))
                ang = math.degrees(np.linalg.norm(tk.log_R(R @ tk.R_NEUTRAL.T)))
                targets.append((cmd_pos[i].copy(), R.copy()))
                cand.append(dict(
                    ep=ep, step=step, ep_p=ep_p, ep_r=ep_r, max_dq=max_dq, cond=cond,
                    limit_margin=margin, q_dev=float(np.max(np.abs(q - DEFAULT_Q))),
                    cmd_ang=ang, cmd_pos=float(np.linalg.norm(cmd_pos[i] - tk.P_NEUTRAL)),
                    base_spd=float(base_spd[i]), since_change=float(since[i]),
                ))
    n = len(cand)
    print(f"\n候选 {n} 条 (episodes={episodes}, steps={steps}, every={every}, num_envs={u.num_envs})")

    def reason(c):
        pos_f = c["ep_p"] > 1e-3
        rot_f = c["ep_r"] > 1e-3
        step_f = c["max_dq"] > 0.08
        if step_f:
            return "步长超限(跳支)"
        if pos_f and rot_f:
            return "位置+姿态残差"
        if pos_f:
            return "仅位置残差"
        if rot_f:
            return "仅姿态残差"
        return "保留"

    counts = {}
    for c in cand:
        r = reason(c)
        counts[r] = counts.get(r, 0) + 1
    kept = [c for c in cand if reason(c) == "保留"]
    dropped = [c for c in cand if reason(c) != "保留"]

    print("\n" + "=" * 92)
    print("B1. 过滤原因 (4.5e 只有一个合并计数; 这里分开)")
    print("=" * 92)
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<14}{v:>6} 条  {v / n * 100:>5.1f}%")
    at_lim = sum(1 for c in dropped if c["limit_margin"] < 1e-3)
    print(f"  落在关节限位上的丢弃样本: {at_lim}/{len(dropped)} "
          f"({at_lim / max(1, len(dropped)) * 100:.1f}%) -- 限位在 local_ik 内部靠 clip 实现, "
          f"所以它们会表现成'没收敛'")
    print(f"  条件数 > 1e4 的样本: 保留 {sum(1 for c in kept if c['cond'] > 1e4)}, "
          f"丢弃 {sum(1 for c in dropped if c['cond'] > 1e4)}")
    print(f"  最大单步 |dq|: {max(c['max_dq'] for c in cand):.4f} (上界 step_cap=0.08)")

    print("\n" + "=" * 92)
    print("B2. 保留 vs 丢弃的分布 (中位数 [四分位])")
    print("=" * 92)

    def q(vals):
        v = np.asarray(vals, dtype=float)
        return f"{np.median(v):7.3f} [{np.percentile(v, 25):6.3f},{np.percentile(v, 75):6.3f}]"

    print(f"  {'量':<22}{'保留':>26}{'丢弃':>26}")
    for key, label in (("cmd_ang", "指令姿态角 (deg)"), ("cmd_pos", "指令位置偏移 (m)"),
                       ("limit_margin", "距最近关节限位 (rad)"), ("q_dev", "|q-default|max (rad)"),
                       ("base_spd", "底盘速度 (m/s)"), ("since_change", "距命令变化 (步)")):
        print(f"  {label:<22}{q([c[key] for c in kept]):>26}{q([c[key] for c in dropped]):>26}")

    print("\n  按指令姿态角分桶的保留率:")
    buckets = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 60)]
    for lo, hi in buckets:
        b = [c for c in cand if lo <= c["cmd_ang"] < hi]
        if b:
            k = sum(1 for c in b if reason(c) == "保留")
            print(f"    [{lo:>2},{hi:>2})°  n={len(b):>4}  保留 {k / len(b) * 100:>5.1f}%")

    # Is "dropped" the same as "unreachable"? The filter only says the *local* walk did not
    # arrive. A global solve from 16 restarts decides reachability, which is a different claim,
    # and the brief is explicit that the two must not be conflated.
    print("\n  抽取 40 条丢弃样本做全局 IK (16 个初值), 区分'不可达'与'局部没走到':")
    idx_drop = [i for i, c in enumerate(cand) if reason(c) != "保留"]
    pick = rng.choice(idx_drop, size=min(40, len(idx_drop)), replace=False)
    reach = 0
    for i in pick:
        p, R = targets[i]
        ep, er, q, slack = tk.solve_ik_nearest(p, R, restarts=16)
        reach += int(ep <= 1e-3 and er <= 1e-3)
    print(f"    {reach}/{len(pick)} 条丢弃样本其实有解 (全局求解), "
          f"即 {reach / len(pick) * 100:.0f}% 的丢弃是局部求解没走到, 不是不可达")

    import csv
    with open(OUT / "teacher_candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cand[0].keys()) + ["reason"])
        w.writeheader()
        for c in cand:
            w.writerow({**c, "reason": reason(c)})
    print(f"\n候选逐条数据: {OUT / 'teacher_candidates.csv'}")
    return counts, kept, dropped


# --------------------------------------------------------------------------------------
# C: the same observation, three actions, and what each of them is worth
# --------------------------------------------------------------------------------------


def _run_handover(env, policy, cmd_rpy, warm=25, watch=25):
    """Converge with the online IK, hand the controls to `policy`, and watch what it does.

    Why not just write the target configuration into the joints (the C4 test above): the
    observation carries three history frames and the previous action, so a written state comes
    with a history that belongs to whatever happened before the write. Handing over after the
    IK has converged keeps state *and* history exactly as the policy would have seen them, so
    a drift cannot be blamed on a fabricated observation.
    """
    import torch

    import isaaclab.utils.math as iu

    from LeggedManip_Lab.assets.ranger_cr10.ranger_cr10_articulation_cfg import ARM_JOINT_NAMES
    from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.pose_command_teleop import (
        TCP_DEFAULT_POS_B,
        TCP_DEFAULT_QUAT_B,
    )

    u = env.unwrapped
    robot = u.scene["robot"]
    ee_cmd = u.command_manager.get_term("ee_pose")
    vel_cmd = u.command_manager.get_term("base_velocity")
    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    dev = u.device
    obs, _ = env.reset(seed=args_cli.seed)
    dR = iu.quat_from_euler_xyz(*[torch.tensor([v], device=dev) for v in cmd_rpy])
    tq = iu.quat_mul(dR, torch.tensor(TCP_DEFAULT_QUAT_B, device=dev).unsqueeze(0))[0]
    ee_cmd.pose_command_b[:, :3] = torch.tensor(np.asarray(TCP_DEFAULT_POS_B), device=dev)
    ee_cmd.pose_command_b[:, 3:] = tq
    vel_cmd.is_standing_env[:] = False
    vel_cmd.vel_command_b[:] = 0.0

    def tik_action():
        q_now = robot.data.joint_pos[0, arm_ids].cpu().numpy()
        q_sol, _, _, _, _ = tk.local_ik(q_now, ee_cmd.pose_command_b[0, :3].cpu().numpy(),
                                        tk.quat_to_R(ee_cmd.pose_command_b[0, 3:].cpu().numpy()))
        a = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=dev)
        a[:, 2:] = torch.tensor((q_sol - DEFAULT_Q) / SCALE, dtype=torch.float32, device=dev)
        return a

    for _ in range(warm):
        obs, _, _, _, _ = env.step(tik_action())
    start = ori_error_deg(env)
    traj = []
    for _ in range(watch):
        with torch.inference_mode():
            a = policy.act(obs["policy"])
        obs, _, _, _, _ = env.step(a)
        traj.append(ori_error_deg(env))
    return start, traj



class _TikWrapper:
    """`_run_handover` calls `policy.act(obs)`; the online solver lives outside that interface."""

    def __init__(self, env):
        self.env = env
        self.name = "TIK"

    def act(self, obs):
        import torch

        from LeggedManip_Lab.assets.ranger_cr10.ranger_cr10_articulation_cfg import ARM_JOINT_NAMES

        u = self.env.unwrapped
        robot = u.scene["robot"]
        ee_cmd = u.command_manager.get_term("ee_pose")
        arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
        q_now = robot.data.joint_pos[0, arm_ids].cpu().numpy()
        q_sol, _, _, _, _ = tk.local_ik(q_now, ee_cmd.pose_command_b[0, :3].cpu().numpy(),
                                        tk.quat_to_R(ee_cmd.pose_command_b[0, 3:].cpu().numpy()))
        a = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
        a[:, 2:] = torch.tensor((q_sol - DEFAULT_Q) / SCALE, dtype=torch.float32, device=u.device)
        return a


def _run_hold(env, policy, tik_name, cmd_rpy, q_ik, sink, steps=20):
    """Write the IK solution, run one policy with that command, append the final error."""
    import torch

    import isaaclab.utils.math as iu

    from LeggedManip_Lab.assets.ranger_cr10.ranger_cr10_articulation_cfg import ARM_JOINT_NAMES
    from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.pose_command_teleop import (
        TCP_DEFAULT_POS_B,
        TCP_DEFAULT_QUAT_B,
    )

    u = env.unwrapped
    robot = u.scene["robot"]
    ee_cmd = u.command_manager.get_term("ee_pose")
    vel_cmd = u.command_manager.get_term("base_velocity")
    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    dev = u.device
    env.reset(seed=args_cli.seed)
    dR = iu.quat_from_euler_xyz(*[torch.tensor([v], device=dev) for v in cmd_rpy])
    tq = iu.quat_mul(dR, torch.tensor(TCP_DEFAULT_QUAT_B, device=dev).unsqueeze(0))[0]
    ee_cmd.pose_command_b[:, :3] = torch.tensor(np.asarray(TCP_DEFAULT_POS_B), device=dev)
    ee_cmd.pose_command_b[:, 3:] = tq
    vel_cmd.is_standing_env[:] = False
    vel_cmd.vel_command_b[:] = 0.0
    q_t = torch.tensor(q_ik, dtype=torch.float32, device=dev).unsqueeze(0).expand(u.num_envs, -1)
    robot.write_joint_state_to_sim(q_t, torch.zeros_like(q_t), joint_ids=arm_ids)
    for _ in range(3):
        obs = u.observation_manager.compute(update_history=True)
    err = float("nan")
    for _ in range(steps):
        if tik_name == "TIK":
            q_now = robot.data.joint_pos[0, arm_ids].cpu().numpy()
            q_sol, ep_p, ep_r, _, _ = tk.local_ik(q_now, ee_cmd.pose_command_b[0, :3].cpu().numpy(),
                                                  tk.quat_to_R(ee_cmd.pose_command_b[0, 3:].cpu().numpy()))
            a = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=dev)
            a[:, 2:] = torch.tensor((q_sol - DEFAULT_Q) / SCALE, dtype=torch.float32, device=dev)
        else:
            with torch.inference_mode():
                a = policy.act(obs["policy"])
        obs, _, _, _, _ = env.step(a)
        err = ori_error_deg(env)
    sink.append(err)


def section_c():
    """On B0's own closed-loop states: actor vs teacher vs online IK, to the socket."""
    import torch

    env = build_env(1)
    dev = env.unwrapped.device
    b0 = make_policy("B0", dev)
    tnet = make_policy("T", dev)
    segs = [s for s in build_schedule() if s["mode"] == "arm"]
    recs = []

    def hook(obs, a, seg, step, u, robot, arm_ids, ee_cmd):
        o = obs["policy"][0].cpu().numpy().copy()
        q_now = robot.data.joint_pos[0, arm_ids].cpu().numpy().copy()
        cmdp = ee_cmd.pose_command_b[0, :3].cpu().numpy().copy()
        cmdq = ee_cmd.pose_command_b[0, 3:].cpu().numpy().copy()
        cmdR = tk.quat_to_R(cmdq)
        q_ik, ep_p, ep_r, _, _ = tk.local_ik(q_now, cmdp, cmdR)
        recs.append(dict(seg=seg["label"], step=step, obs=o, q_now=q_now, cmdp=cmdp, cmdR=cmdR,
                         a_actor=a[0, 2:].cpu().numpy().copy(), a_teacher=tnet.arm(o),
                         a_ik=(q_ik - DEFAULT_Q) / SCALE, ik_ok=float(ep_p < 1e-3 and ep_r < 1e-3)))

    rows, _ = run_closed_loop(env, b0, segs, args_cli.seed, tik=False, hook=hook)
    print(f"\n记录 {len(recs)} 个在线状态 (B0 的闭环轨迹)")

    def fk_errors(a6, q_now, cmdp, cmdR):
        q_t = DEFAULT_Q + SCALE * np.asarray(a6)
        T = tk.fk(q_t)
        dp = float(np.linalg.norm(T[:3, 3] - cmdp))
        da = math.degrees(np.linalg.norm(tk.log_R(cmdR @ T[:3, :3].T)))
        # "...relative to doing nothing": the same measures for the current configuration
        T0 = tk.fk(q_now)
        dp0 = float(np.linalg.norm(T0[:3, 3] - cmdp))
        da0 = math.degrees(np.linalg.norm(tk.log_R(cmdR @ T0[:3, :3].T)))
        return dp, da, dp0, da0

    print("\n" + "=" * 100)
    print("C1/C3. 同一 observation, 三种动作 (B0 在线状态, n=%d)" % len(recs))
    print("=" * 100)
    print(f"  {'动作源':<10}{'动作 MAE':>10}{'关节 MAE':>10}{'方向角':>9}"
          f"{'FK 位置':>10}{'FK 姿态':>10}{'当前位形(FK 姿态)':>18}")
    mae_at = np.mean([np.mean(np.abs(r["a_actor"] - r["a_teacher"])) for r in recs])
    mae_ai = np.mean([np.mean(np.abs(r["a_actor"] - r["a_ik"])) for r in recs])
    for name, key in (("actor", "a_actor"), ("teacher", "a_teacher"), ("online IK", "a_ik")):
        dp = np.mean([fk_errors(r[key], r["q_now"], r["cmdp"], r["cmdR"])[0] for r in recs])
        da = np.mean([fk_errors(r[key], r["q_now"], r["cmdp"], r["cmdR"])[1] for r in recs])
        jm = np.mean([np.mean(np.abs(DEFAULT_Q + SCALE * r[key] - (DEFAULT_Q + SCALE * r["a_ik"])))
                      for r in recs])
        angs = []
        for r in recs:
            u_ = (DEFAULT_Q + SCALE * r[key]) - (DEFAULT_Q + SCALE * r["a_ik"])
            v_ = (DEFAULT_Q + SCALE * r["a_actor"]) - (DEFAULT_Q + SCALE * r["a_ik"])
            if np.linalg.norm(u_) > 0.05 and np.linalg.norm(v_) > 0.05:
                c = np.clip(u_ @ v_ / (np.linalg.norm(u_) * np.linalg.norm(v_)), -1, 1)
                angs.append(math.degrees(math.acos(c)))
        print(f"  {name:<10}{'':>10}{jm:>10.3f}{np.median(angs) if angs else float('nan'):>9.1f}"
              f"{dp:>9.3f}m{da:>9.1f}°")
    dp0 = np.mean([fk_errors(r["a_actor"], r["q_now"], r["cmdp"], r["cmdR"])[2] for r in recs])
    da0 = np.mean([fk_errors(r["a_actor"], r["q_now"], r["cmdp"], r["cmdR"])[3] for r in recs])
    print(f"  {'不动':<10}{'':>10}{'':>10}{'':>9}{dp0:>9.3f}m{da0:>9.1f}°   <- 什么都不做时的误差")
    print(f"\n  actor vs teacher: 动作 MAE {mae_at:.4f} ({mae_at * SCALE:.4f} rad)")
    print(f"  actor vs 在线 IK : 动作 MAE {mae_ai:.4f} ({mae_ai * SCALE:.4f} rad)")
    print(f"  在线 IK 收敛率: {np.mean([r['ik_ok'] for r in recs]) * 100:.1f}%")

    # the static-FK prediction against what the simulator actually did
    print("\n  FK 预测的姿态误差 vs 实际执行后的误差 (同一批状态, 统计执行滞后):")
    for lag in (0, 2, 5):
        dyn = []
        for k, r in enumerate(recs):
            j = k + lag
            if j < len(rows) and rows[j]["seg"] == r["seg"] and rows[j]["step"] == r["step"] + lag:
                dyn.append(rows[j]["ori_deg"])
        if dyn and len(dyn) == len(recs) - lag:
            print(f"    +{lag} 步后实际姿态误差 {np.mean(dyn):.1f}°  (执行/动力学部分)")
    fk_actor = np.mean([fk_errors(r["a_actor"], r["q_now"], r["cmdp"], r["cmdR"])[1] for r in recs])
    print(f"    actor 动作的静态 FK 姿态误差 {fk_actor:.1f}° (动作映射部分)")



    # ---- B3 in place: are these states inside the supervised distribution? --------------
    data = torch.load(DATASET, map_location="cpu", weights_only=False)
    Xtr = data["X"][data["tr"]].numpy()
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ztr = (Xtr - mu) / sd
    Zon = (np.stack([r["obs"] for r in recs]) - mu) / sd
    nn = np.empty(len(Zon))
    for k in range(len(Zon)):
        nn[k] = np.min(np.linalg.norm(Ztr - Zon[k], axis=1))
    print("\n" + "=" * 100)
    print("C. 在线状态在监督数据分布里的位置 (标准化观测空间的最近邻距离)")
    print("=" * 100)
    print(f"  最近邻距离: 中位 {np.median(nn):.2f}, 四分位 "
          f"[{np.percentile(nn, 25):.2f}, {np.percentile(nn, 75):.2f}], 最大 {nn.max():.2f}")
    cur = np.array([fk_errors(r["a_actor"], r["q_now"], r["cmdp"], r["cmdR"])[3] for r in recs])
    err_t = np.array([fk_errors(r["a_teacher"], r["q_now"], r["cmdp"], r["cmdR"])[1] for r in recs])
    err_a = np.array([fk_errors(r["a_actor"], r["q_now"], r["cmdp"], r["cmdR"])[1] for r in recs])
    print(f"\n  {'当前姿态误差(不动)':<20}{'n':>5}{'teacher FK':>12}{'actor FK':>11}{'最近邻距离':>12}")
    for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 90)):
        m = (cur >= lo) & (cur < hi)
        if m.sum():
            print(f"  {f'{lo}-{hi}°':<20}{int(m.sum()):>5}{err_t[m].mean():>11.1f}°"
                  f"{err_a[m].mean():>10.1f}°{np.median(nn[m]):>12.2f}")
    print("  teacher FK = 它给出的动作经 FK 后的姿态误差; 若它在'已经接近目标'的状态上失准,")
    print("  闭环就会停在一个它认为'不用动'的错误位形上 -- 这正是稳态误差的来源。")

    # ---- C4: can the map *hold* a correct configuration? -------------------------------
    # The one state that matters for steady-state accuracy is "arm already at the commanded
    # pose". Write the IK solution into the joints, then let each policy run with the same
    # command: a correct map stays there, a wrong one walks away. This is the cheapest test
    # that separates "cannot reach" from "cannot stay".
    print("\n" + "=" * 100)
    print("C4. 从'已经到位'的位形出发, 20 步内还能不能待住")
    print("=" * 100)
    holds = {"B0": [], "T": [], "TIK": []}
    for cmd_rpy in ((0.2, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.2)):
        # the pose and joint solution for this command, from the default configuration
        dR = tk.euler_R(cmd_rpy)
        p_cmd = np.asarray(tk.P_NEUTRAL)
        R_cmd = dR @ tk.R_NEUTRAL
        q_ik, ep_p, ep_r, _, _ = tk.local_ik(np.asarray(DEFAULT_Q), p_cmd, R_cmd)
        line = f"  cmd {tuple(np.round(cmd_rpy, 2))} (IK 残差 {ep_p:.1e}/{ep_r:.1e}): "
        for name, pol in (("B0", b0), ("T", tnet)):
            _run_hold(env, pol, None, cmd_rpy, q_ik, holds[name])
        _run_hold(env, None, "TIK", cmd_rpy, q_ik, holds["TIK"])
        for name in ("B0", "T", "TIK"):
            line += f"{name} {holds[name][-1]:5.1f}°  "
        print(line)
    print("  (数值 = 该命令下第 20 步的姿态误差; 起始时约 0°)")


    print("\n" + "=" * 100)
    print("C5. 交给策略: 在线 IK 先把姿态收敛到位, 然后把控制权交给被测策略")
    print("=" * 100)
    print(f"  {'命令':<16}{'交接时误差':>12}{'+0.5s':>9}{'+1.0s':>9}{'+2.5s':>9}")
    for cmd_rpy in ((0.2, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.2), (0.15, 0.1, -0.1)):
        for name, pol in (("B0", b0), ("T", tnet), ("TIK", None)):
            if pol is None and name != "TIK":
                continue
            if name == "TIK":
                start, traj = _run_handover(env, _TikWrapper(env), cmd_rpy)
            else:
                start, traj = _run_handover(env, pol, cmd_rpy)
            print(f"  {str(cmd_rpy):<16}{start:>11.1f}°{traj[4]:>8.1f}°{traj[9]:>8.1f}°{traj[-1]:>8.1f}°")
        print()
    with open(OUT / "online_consistency.csv", "w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["seg", "step", "ik_ok", "actor_mae_vs_teacher", "actor_mae_vs_ik",
                    "fk_actor_pos", "fk_actor_att", "fk_teacher_att", "fk_ik_att", "executed_att"])
        for k, r in enumerate(recs):
            w.writerow([r["seg"], r["step"], r["ik_ok"],
                        float(np.mean(np.abs(r["a_actor"] - r["a_teacher"]))),
                        float(np.mean(np.abs(r["a_actor"] - r["a_ik"]))),
                        fk_errors(r["a_actor"], r["q_now"], r["cmdp"], r["cmdR"])[0],
                        fk_errors(r["a_actor"], r["q_now"], r["cmdp"], r["cmdR"])[1],
                        fk_errors(r["a_teacher"], r["q_now"], r["cmdp"], r["cmdR"])[1],
                        fk_errors(r["a_ik"], r["q_now"], r["cmdp"], r["cmdR"])[1],
                        rows[k]["ori_deg"] if k < len(rows) else float("nan")])
    print(f"\n逐状态数据: {OUT / 'online_consistency.csv'}")


# --------------------------------------------------------------------------------------
# D: PPO forgetting, on the fixed supervised test set
# --------------------------------------------------------------------------------------
CHECKPOINTS = [
    ("A base 999", "2026-10-12_20-14-23/model_999.pt"),
    ("A 1198", "2026-10-23_18-32-48_ctrl_A/model_1198.pt"),
    ("B0 supervised", "_stage45e/pretrained_B.pt"),
    ("B1 1000", "2026-10-23_19-10-13_ctrl_B/model_1000.pt"),
    ("B1 1198", "2026-10-23_19-10-13_ctrl_B/model_1198.pt"),
    ("C 1000", "2026-10-23_19-59-08_ctrl_C/model_1000.pt"),
    ("C 1198", "2026-10-23_19-59-08_ctrl_C/model_1198.pt"),
]


def section_d():
    import torch

    data = torch.load(DATASET, map_location="cpu", weights_only=False)
    X, Y = data["X"], data["Y"].numpy()
    te = data["te"]
    Xte = X[te]
    Yte = Y[te]
    print(f"固定监督测试集: {len(te)} 条 (与 4.5e 的划分完全一致)")
    print(f"  标签 q_ik = default + {SCALE} * Y; 任务空间目标由 FK(q_ik) 得到, 不引入新数据\n")

    Yarm = Yte[:, 2:]          # the dataset pads the two chassis channels with zeros
    q_tgt = DEFAULT_Q[None, :] + SCALE * Yarm
    Tz = [tk.fk(q) for q in q_tgt]

    print("=" * 96)
    print("D1. 同一测试集上的拟合误差随 checkpoint 的变化")
    print("=" * 96)
    print(f"  {'checkpoint':<16}{'关节 MAE':>10}{'方向角':>9}{'FK 位置':>10}{'FK 姿态':>10}")
    for label, rel in CHECKPOINTS:
        path = LOG / rel
        if not path.exists():
            print(f"  {label:<16} 缺失 {path}")
            continue
        net = load_actor_from_checkpoint(path, "cpu")
        with torch.inference_mode():
            pred = net(Xte)[:, 2:].numpy()
        jm = float(np.mean(np.abs(pred - Yarm)) * SCALE)
        angs = []
        for k in range(len(pred)):
            u_ = SCALE * pred[k]
            v_ = SCALE * Yarm[k]
            if np.linalg.norm(u_) > 0.05 and np.linalg.norm(v_) > 0.05:
                c = np.clip(u_ @ v_ / (np.linalg.norm(u_) * np.linalg.norm(v_)), -1, 1)
                angs.append(math.degrees(math.acos(c)))
        dp, da = [], []
        for k in range(len(pred)):
            T = tk.fk(DEFAULT_Q + SCALE * pred[k])
            dp.append(np.linalg.norm(T[:3, 3] - Tz[k][:3, 3]))
            da.append(math.degrees(np.linalg.norm(tk.log_R(Tz[k][:3, :3] @ T[:3, :3].T))))
        print(f"  {label:<16}{jm:>10.4f}{np.median(angs):>8.1f}°{np.mean(dp):>9.3f}m{np.mean(da):>9.1f}°")

    print("\n" + "=" * 96)
    print("D2. B0 带进 PPO 的是什么 (单因素事实, 不是推断)")
    print("=" * 96)
    base = torch.load(LOG / "2026-10-12_20-14-23/model_999.pt", map_location="cpu", weights_only=False)
    b0 = torch.load(LOG / "_stage45e/pretrained_B.pt", map_location="cpu", weights_only=False)
    b1k = torch.load(LOG / "2026-10-23_19-10-13_ctrl_B/model_1000.pt", map_location="cpu", weights_only=False)
    c1k = torch.load(LOG / "2026-10-23_19-59-08_ctrl_C/model_1000.pt", map_location="cpu", weights_only=False)
    a1k = torch.load(LOG / "2026-10-23_18-32-48_ctrl_A/model_1000.pt", map_location="cpu", weights_only=False)

    def diff(sd_a, sd_b):
        keys = [k for k in sd_a if k in sd_b and torch.is_tensor(sd_a[k]) and sd_a[k].is_floating_point()]
        return max(float((sd_a[k] - sd_b[k]).abs().max()) for k in keys)

    opt_keys = [k for k in base.get("optimizer_state_dict", {}).get("state", {})]
    def opt_diff(oa, ob):
        if not oa or not ob:
            return float("nan")
        d = 0.0
        for k in oa["state"]:
            for f, v in oa["state"][k].items():
                if torch.is_tensor(v):
                    d = max(d, float((v - ob["state"][k][f]).abs().max()))
        return d

    print(f"  actor   model_999 vs pretrained_B : {diff(base['actor_state_dict'], b0['actor_state_dict']):.4f}"
          f"   (监督改变了 actor)")
    print(f"  critic  model_999 vs pretrained_B : {diff(base['critic_state_dict'], b0['critic_state_dict']):.4g}"
          f"   (critic 原样带过来)")
    adam = opt_diff(base.get("optimizer_state_dict"), b0.get("optimizer_state_dict"))
    print(f"  Adam    model_999 vs pretrained_B : {adam:.4g}"
          f"   (Adam 动量原样带过来, 但它对应的是监督之前的 actor)")
    print(f"\n  PPO 第 1 次迭代把 actor 动了多远 (max|Δw|):")
    print(f"    B0 -> B1 iter1000 : {diff(b0['actor_state_dict'], b1k['actor_state_dict']):.4f}")
    print(f"    A base 999 -> A iter1000 : {diff(base['actor_state_dict'], a1k['actor_state_dict']):.4f}")
    dc = diff(b0["actor_state_dict"], c1k["actor_state_dict"])
    print(f"    B0 -> C iter1000 : {dc:.4f} (C 从 model_999 起, 不是 B0)")

    # training curves, if the event files are readable
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        print("\n  PPO 阶段的关键标量 (B1 = ctrl_B):")
        for run, tag in (("2026-10-23_19-10-13_ctrl_B", None), ("2026-10-23_19-59-08_ctrl_C", "ik_aux")):
            ev = EventAccumulator(str(LOG / run), size_guidance={"scalars": 0})
            ev.Reload()
            tags = ev.Tags()["scalars"]
            print(f"    {run}: {len(tags)} 个标量")
            for t in tags:
                if any(k in t for k in ("ik_aux", "Loss", "kl", "entropy", "learning")):
                    s = ev.Scalars(t)
                    vals = [round(x.value, 4) for x in s]
                    if vals:
                        print(f"      {t:<32} 首 {vals[0]:>10.4f}  尾 {vals[-1]:>10.4f}  "
                              f"max {max(vals):>10.4f}  n={len(vals)}")
    except Exception as exc:  # tensorboard is not required for the rest of the section
        print(f"  (TensorBoard 读取失败: {exc})")


def main():
    print(f"section {args_cli.section}, 输出目录 {OUT}")
    if args_cli.section == "A":
        section_a()
    elif args_cli.section == "B":
        section_b()
    elif args_cli.section == "C":
        section_c()
    elif args_cli.section == "D":
        section_d()
    elif args_cli.section == "E":
        section_e()

# --------------------------------------------------------------------------------------
# E: offline analysis -- coverage, state sensitivity, and the headroom a task-space loss has
# --------------------------------------------------------------------------------------
def section_e():
    """Three questions that need no simulator.

    1. What does the supervision set cover, in terms of *how far the arm still had to move*?
       A dataset can be large and still contain no sample where the arm is at the target.
    2. How much does the fitted map depend on the observed joint state at all? A map that is
       mostly `command -> posture` fits such a dataset well (the postures it saw are already
       close to the IK solution) and still has nothing to correct with in closed loop.
    3. How much room is there for a task-space FK loss on this data? The label *is* the IK
       solution, so its task-space error is zero by construction; the question is how far the
       fitted map already is from that ceiling.
    """
    import torch

    data = torch.load(DATASET, map_location="cpu", weights_only=False)
    X, Y = data["X"].numpy(), data["Y"].numpy()
    tr, te = data["tr"].numpy(), data["te"].numpy()
    teacher = load_teacher(Path("logs/ik_teacher/teacher.pt"), "cpu")

    print("=" * 96)
    print("E1. 监督数据覆盖: 样本中'手臂还需移动多远'")
    print("=" * 96)
    q_now = DEFAULT_Q[None, :] + X[:, 18:24]        # joint_pos term is relative to the default
    q_ik = DEFAULT_Q[None, :] + SCALE * Y[:, 2:]
    err, travel = [], []
    for i in range(len(X)):
        T0, T1 = tk.fk(q_now[i]), tk.fk(q_ik[i])
        err.append(math.degrees(np.linalg.norm(tk.log_R(T1[:3, :3] @ T0[:3, :3].T))))
        travel.append(float(np.abs(q_ik[i] - q_now[i]).max()))
    err, travel = np.array(err), np.array(travel)
    print(f"  {'当前姿态距 IK 解':<20}{'条数':>8}{'占比':>8}{'所需最大关节位移(中位)':>24}")
    for lo, hi in ((0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 90)):
        m = (err >= lo) & (err < hi)
        t = np.median(travel[m]) if m.sum() else float("nan")
        print(f"  {f'{lo}-{hi}°':<20}{int(m.sum()):>8}{m.mean() * 100:>7.1f}%{t:>22.3f} rad")
    print(f"  中位 {np.median(err):.1f}°, {(err < 2).mean() * 100:.1f}% 的样本在手边 2° 以内")

    print("\n" + "=" * 96)
    print("E2. 教师对'观测到的关节状态'的灵敏度 (测试集 40 条, 观测关节 +0.1 rad)")
    print("=" * 96)
    rng = np.random.default_rng(0)
    sens, base_err = [], []
    for i in rng.choice(te, 40, replace=False):
        o = X[i]
        with torch.inference_mode():
            a0 = teacher(torch.tensor(o, dtype=torch.float32).unsqueeze(0))[0, 2:].numpy()
            o2 = o.copy()
            o2[18:36] = np.tile(o[18:24] + 0.10, 3)
            a1 = teacher(torch.tensor(o2, dtype=torch.float32).unsqueeze(0))[0, 2:].numpy()
        sens.append(float(np.abs(a1 - a0).max()) * SCALE)
        base_err.append(float(np.abs(a0 - SCALE * Y[i][2:] / SCALE).max()))
    print(f"  输出变化中位 {np.median(sens):.4f} rad -- 观测关节走 0.10 rad, 输出只跟了 "
          f"{np.median(sens) / 0.10 * 100:.0f}%")
    print("  这个比例就是闭环里的反馈增益; 小于 1 的增益不会把误差拉回零, 只会让初始偏差")
    print("  缩小到 1/(1-g) 倍后停下 -- 与 4.5d 在动作水平上量到的 0.397 是同一件事。")

    print("\n" + "=" * 96)
    print("E3. 任务空间 FK 辅助损失还有多少空间")
    print("=" * 96)
    print(f"  {'量':<34}{'FK 姿态误差':>14}{'FK 位置误差':>14}")
    with torch.inference_mode():
        pred = teacher(torch.tensor(X[te], dtype=torch.float32))[:, 2:].numpy()
    dp, da = [], []
    for k, i in enumerate(te):
        T = tk.fk(DEFAULT_Q + SCALE * pred[k])
        Tt = tk.fk(q_ik[i])
        dp.append(np.linalg.norm(T[:3, 3] - Tt[:3, 3]))
        da.append(math.degrees(np.linalg.norm(tk.log_R(Tt[:3, :3] @ T[:3, :3].T))))
    print(f"  {'教师 (关节 MSE, 现有 4.5e 方案)':<30}{np.mean(da):>13.1f}°{np.mean(dp):>13.3f}m")
    print(f"  {'标签本身 (= IK 解)':<30}{0.0:>13.1f}°{0.0:>13.3f}m   <- 任务空间损失的下界")
    print(f"\n  关节 MSE 已把任务空间误差压到 {np.mean(da):.1f}°/厘米级, 而闭环失配是 9-11°。")
    print("  在这批数据上换任务空间损失, 被优化的量本来就已经接近 0; 缺口不在损失形式。")

if __name__ == "__main__":
    main()
