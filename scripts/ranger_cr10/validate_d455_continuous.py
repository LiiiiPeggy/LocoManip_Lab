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

"""One continuous drive -> park -> observe sequence on the D455, frozen inference.

Unlike ``validate_d455_views.py`` (one reset per view), this entry point resets *once* and then
runs the whole sequence without any further reset:

    settle -> drive (vx) -> turn (wz) -> park (zero command, wait for the chassis to stop)
           -> observe front -> observe side -> observe top (6 s hold each)

The deployment zero-velocity filter is always on and follows the deployment order
(deadband the command -> the policy observes the deadbanded value -> ``filter_action`` -> step).
The three targets are A1-style world-fixed points derived from the **planned** parking pose; the
actual parking pose is recorded separately, so parking error shows up in the pixels instead of
being hidden.

Two modes, so the target poses can be fixed before the evaluated run:

    # 1. plan: settle/drive/turn/park only, writes plan.json (planned and actual park pose)
    python scripts/ranger_cr10/validate_d455_continuous.py --run-mode plan --out-dir .../plan
    # 2. full: the evaluated sequence, targets taken from plan.json
    python scripts/ranger_cr10/validate_d455_continuous.py --run-mode full \
        --plan .../plan/plan.json --out-dir .../run

Every step is recorded (upper command, effective command, raw and filtered chassis action, the
action term's post-tanh command, steering/wheel targets and speeds, base pose, base velocities,
EE tracking error). Terminations are monitored on every step; the episode timeout is suppressed
by keeping the episode counter at zero, which is recorded explicitly. Any real termination,
arrival timeout or parking timeout marks the flow failed and stops it where it happened.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", default="models/ranger_cr10/model_9999.pt")
ap.add_argument("--out-dir", default="outputs/stage413/b/d455_continuous")
ap.add_argument("--run-mode", choices=("plan", "full"), default="plan")
ap.add_argument("--plan", default="", help="plan.json written by --run-mode plan (required for full)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--filter-config", default="mujoco/deploy/deploy_mujoco/ranger_cr10/config.yaml")
# Frozen sequence parameters. Chosen once and kept; see the written config in the output dir.
ap.add_argument("--settle-steps", type=int, default=10)
ap.add_argument("--drive-vx", type=float, default=0.15)
ap.add_argument("--drive-steps", type=int, default=10)
ap.add_argument("--turn-wz", type=float, default=0.25)
ap.add_argument("--turn-steps", type=int, default=6)
ap.add_argument("--park-timeout", type=int, default=60)
ap.add_argument("--arrival-timeout", type=int, default=80)
ap.add_argument("--hold-steps", type=int, default=60)
args = ap.parse_args()

app = AppLauncher(headless=True, enable_cameras=True).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.sim import PinholeCameraCfg  # noqa: E402
from isaaclab.utils.math import convert_camera_frame_orientation_convention  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import LeggedManip_Lab.tasks  # noqa: E402
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.simple_collision_env_cfg import (  # noqa: E402
    RangerCr10TeleopSimpleCollisionEnvCfg,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.tcp import (  # noqa: E402
    TCP_ANCHOR_BODY,
    tcp_pose_b,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "ranger_cr10"))
from base_command_filter import BaseCommandFilter  # noqa: E402
from d455_eval_common import (  # noqa: E402
    ARRIVAL_TIMEOUT,
    DEFAULT_TCP_POS,
    DEFAULT_TCP_QUAT,
    F_PX,
    HOLD_STEPS,
    IMG_H,
    IMG_W,
    LINK6_CAM_POS,
    LINK6_CAM_QUAT,
    MARKER_DISTANCE_M,
    MARKER_SIZE_M,
    MIN_VISIBLE_PIXELS,
    ORI_TOL,
    PIXEL_JITTER_MAX,
    PIXEL_OFFSET_MAX,
    POS_TOL,
    VISIBLE_MIN_RATIO,
    compose,
    detect,
    invert,
    project_point,
    quat_rotate,
    view_geometry,
)

CKPT = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(ROOT, args.checkpoint)
OUT_DIR = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
os.makedirs(OUT_DIR, exist_ok=True)

VIEW_ORDER = ("front", "side", "top")
GEOMETRY = {name: view_geometry(name) for name in VIEW_ORDER}
FOCAL_LENGTH_MM = 18.6
HORIZONTAL_APERTURE_MM = FOCAL_LENGTH_MM * IMG_W / F_PX

# -- frozen configuration, written before any simulation step -------------------------
CONFIG = {
    "purpose": "one continuous drive -> park -> observe(front, side, top) sequence, frozen inference",
    "checkpoint": os.path.relpath(CKPT, ROOT),
    "env": "RANGER-CR10-TELEOP-SIMPLECOLL (SIMPLECOLL v2)",
    "seed": args.seed,
    "num_envs": 1,
    "resets": "exactly one, at the start; the episode counter is held at zero so no auto reset occurs",
    "fastening": "gripper fixed (no gripper action); the six arm joints follow the policy",
    "sequence": {
        "settle": {"steps": args.settle_steps, "base_command": [0.0, 0.0],
                   "arm_command": "default TCP pose (DEFAULT_TCP_POS/QUAT) in the base frame"},
        "drive": {"steps": args.drive_steps, "vx": args.drive_vx, "wz": 0.0},
        "turn": {"steps": args.turn_steps, "vx": 0.0, "wz": args.turn_wz},
        "park": {"command": [0.0, 0.0], "timeout_steps": args.park_timeout,
                 "criterion": "|v_xy| < 0.02 m/s and |yaw rate| < 0.05 rad/s for 5 consecutive steps"},
        "observe": {"views": list(VIEW_ORDER), "hold_steps": args.hold_steps,
                    "arrival_timeout_steps": args.arrival_timeout},
    },
    "why_these_numbers": {
        "drive_vx": "0.15 m/s is ~7x the 0.02 m/s deploy deadband and inside the low-speed band the "
                    "policy was trained on; 1.0 s gives a 0.15 m nominal displacement, short enough "
                    "to keep the planned park pose within reach of all three arm targets",
        "turn_wz": "0.25 rad/s is ~12x the 0.02 rad/s deadband; 0.6 s spins ~8.6 deg, which "
                   "exercises the wz channel while keeping the yaw error small",
        "park": "the timeouts and thresholds are reused from the existing protocol (0.02 m/s "
                "matches the deploy deadband; 5 steps = 0.5 s of quiet)",
        "hold": "6 s / 60 frames is the existing observation window",
    },
    "criteria": {"visible_ratio_min": VISIBLE_MIN_RATIO, "offset_mean_max_px": PIXEL_OFFSET_MAX,
                 "std_max_px": PIXEL_JITTER_MAX, "arrival_pos_tol_m": POS_TOL, "arrival_ori_tol_rad": ORI_TOL},
    "command_frame": "the ee_pose command term's pose_command_b is expressed in the robot base "
                     "(base_link) frame; tcp_pose_b is measured in the same frame. The three "
                     "targets are fixed world points; the arm command stays the constant base-frame "
                     "pose, so parking error appears as pixel offset.",
}

# -- deployment filter, read from the deployment YAML ---------------------------------
_cfg_path = args.filter_config if os.path.isabs(args.filter_config) else os.path.join(ROOT, args.filter_config)
_deploy_cfg = yaml.safe_load(open(_cfg_path))
_fcfg = _deploy_cfg["filter"]
FILTER = BaseCommandFilter(
    vx_deadband=float(_fcfg["vx_deadband"]) if _fcfg.get("enabled", True) else 0.0,
    wz_deadband=float(_fcfg["wz_deadband"]) if _fcfg.get("enabled", True) else 0.0,
    lock_output=bool(_fcfg.get("lock_output", False)) and bool(_fcfg.get("enabled", True)),
    hysteresis=float(_fcfg.get("hysteresis", 0.0)),
)
CONFIG["filter"] = {
    "config_file": os.path.relpath(_cfg_path, ROOT),
    "vx_deadband": FILTER.vx_deadband, "wz_deadband": FILTER.wz_deadband,
    "lock_output": FILTER.lock_output, "hysteresis": FILTER.hysteresis, "enabled": FILTER.enabled,
    "call_order": "filter_command -> policy(obs with deadbanded command) -> filter_action -> env.step",
    "note": "the effective command reaches the observation one step after the phase switch; with "
            "constant per-phase commands this is a single-step alignment effect only",
}
json.dump(CONFIG, open(os.path.join(OUT_DIR, "run_config.json"), "w"), indent=1)

# -- env --------------------------------------------------------------------------------
cfg = RangerCr10TeleopSimpleCollisionEnvCfg()
cfg.seed = args.seed
cfg.scene.num_envs = 1
cfg.commands.ee_pose.debug_vis = False
cfg.observations.policy.enable_corruption = False
cfg.scene.d455_camera = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/cr10_Link6/d455_color",
    update_period=0.0,
    update_latest_camera_pose=True,
    width=IMG_W,
    height=IMG_H,
    data_types=["rgb"],
    offset=CameraCfg.OffsetCfg(pos=tuple(float(v) for v in LINK6_CAM_POS),
                               rot=tuple(float(v) for v in LINK6_CAM_QUAT), convention="ros"),
    spawn=PinholeCameraCfg(focal_length=FOCAL_LENGTH_MM, focus_distance=MARKER_DISTANCE_M,
                           horizontal_aperture=HORIZONTAL_APERTURE_MM, clipping_range=(0.02, 100.0),
                           lock_camera=False),
)
for _name in VIEW_ORDER:
    setattr(cfg.scene, f"marker_{_name}",
            AssetBaseCfg(prim_path=f"{{ENV_REGEX_NS}}/marker_{_name}",
                         spawn=sim_utils.CuboidCfg(
                             size=(MARKER_SIZE_M,) * 3,
                             visual_material=sim_utils.PreviewSurfaceCfg(
                                 diffuse_color=tuple(GEOMETRY[_name]["marker_color"])),
                             rigid_props=None, collision_props=None),
                         init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0))))

torch.manual_seed(args.seed)
np.random.seed(args.seed)
env = gym.make("RANGER-CR10-TELEOP-SIMPLECOLL", cfg=cfg)
u = env.unwrapped
train_cfg = yaml.safe_load(open(os.path.join(os.path.dirname(CKPT), "params", "agent.yaml")))
train_cfg["device"] = str(u.device)
train_cfg["seed"] = args.seed
wrapper = RslRlVecEnvWrapper(env, clip_actions=train_cfg.get("clip_actions"))
runner = OnPolicyRunner(wrapper, train_cfg, log_dir=None, device=u.device)
runner.load(CKPT)
policy = runner.get_inference_policy(device=u.device)

robot = u.scene["robot"]
camera = u.scene["d455_camera"]
contact_sensor = u.scene["contact_forces"]
body_names = list(contact_sensor.body_names)
wheel_rows = {i for i, n in enumerate(body_names) if n.endswith("_wheel_link")}
non_wheel_rows = [i for i in range(len(body_names)) if i not in wheel_rows]
anchor_id = robot.find_bodies(TCP_ANCHOR_BODY)[0][0]
base_id = robot.find_bodies("base_link")[0][0]
dt = u.step_dt
term_names = list(u.termination_manager.active_terms)
markers = {name: u.scene[f"marker_{name}"] for name in VIEW_ORDER}
for _view in markers.values():
    if hasattr(_view, "_sync_usd_on_fabric_write"):
        _view._sync_usd_on_fabric_write = True

base_term = u.action_manager.get_term("base_vel")
base_cfg = base_term.cfg
_original_apply = base_term.apply_actions
upper_command = {"vx": 0.0, "wz": 0.0}
filter_state = {"vx": 0.0, "wz": 0.0}
arm_command = {"pos": torch.tensor(DEFAULT_TCP_POS, dtype=torch.float32),
               "quat": torch.tensor(DEFAULT_TCP_QUAT, dtype=torch.float32)}


def apply_current_command() -> None:
    ee = u.command_manager.get_term("ee_pose")
    ee.pose_command_b[:, :3] = arm_command["pos"].to(u.device)
    ee.pose_command_b[:, 3:] = arm_command["quat"].to(u.device)
    vel = u.command_manager.get_term("base_velocity")
    vel.vel_command_b[:, 0] = filter_state["vx"]
    vel.vel_command_b[:, 1] = 0.0
    vel.vel_command_b[:, 2] = filter_state["wz"]
    vel.is_standing_env[:] = abs(filter_state["vx"]) < 1e-9 and abs(filter_state["wz"]) < 1e-9


def _compute_commands(dt):        # the env calls this as compute(dt=step_dt)
    apply_current_command()


u.command_manager.compute = _compute_commands
def _reset_commands(env_ids=None):
    apply_current_command()
    return {}          # the manager merges this into env.extras["log"]; returning None breaks it


u.command_manager.reset = _reset_commands


def base_pose():
    return (robot.data.body_pos_w[0, base_id].detach().cpu().numpy(),
            robot.data.body_quat_w[0, base_id].detach().cpu().numpy())


def base_velocities():
    lin = robot.data.body_lin_vel_w[0, base_id].detach().cpu().numpy()
    ang = robot.data.body_ang_vel_w[0, base_id].detach().cpu().numpy()
    return lin, ang


def read_frame():
    if "rgb" not in camera.data.output:
        camera.update(dt)
    if "rgb" not in camera.data.output:
        raise RuntimeError("camera produced no rgb buffer; was the app started with cameras?")
    return camera.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)


def _set_marker_pose(marker, position_np, orientation_np):
    position = torch.tensor(position_np, dtype=torch.float32, device=u.device).unsqueeze(0)
    orientation = torch.tensor(orientation_np, dtype=torch.float32, device=u.device).unsqueeze(0)
    if hasattr(marker, "set_world_poses"):
        marker.set_world_poses(positions=position, orientations=orientation)
    else:
        marker.write_root_pose_to_sim(torch.cat([position, orientation], dim=-1))


def marker_world(name):
    return markers[name].get_world_poses()[0][0].detach().cpu().numpy()


def place_active_marker(active: str, target_world: np.ndarray, target_quat: np.ndarray):
    """Only the active view's marker sits at its target; the others are parked out of frame."""
    bp, _bq = base_pose()
    for name in VIEW_ORDER:
        if name == active:
            _set_marker_pose(markers[name], target_world, target_quat)
        else:
            _set_marker_pose(markers[name], bp + np.array([0.0, 0.0, -100.0]), target_quat)


records: list[dict] = []
step_index = 0


def policy_step(obs):
    """Deployment order: deadband -> policy -> filter_action -> step, then diagnostics."""
    global step_index
    vx_t = torch.tensor([upper_command["vx"]], dtype=torch.float32, device=u.device)
    wz_t = torch.tensor([upper_command["wz"]], dtype=torch.float32, device=u.device)
    vx_eff, wz_eff = FILTER.filter_command(vx_t, wz_t)
    filter_state["vx"] = float(vx_eff[0].item())
    filter_state["wz"] = float(wz_eff[0].item())
    # Keep the episode counter at zero: the sequence is longer than the 200-step timeout, and an
    # auto reset in the middle would make the run a stitched fragment rather than one flow.
    u.episode_length_buf[:] = 0
    with torch.no_grad():
        raw = policy(obs)
        used = FILTER.filter_action(raw, vx_eff, wz_eff)
        obs_next, _rew, _done, _extra = wrapper.step(used)
    step_index += 1

    terminated = int(u.termination_manager.terminated.sum().item())
    timeouts = int(u.termination_manager.time_outs.sum().item())
    lin, ang = base_velocities()
    bp, bq = base_pose()
    tcp_pos, tcp_quat = tcp_pose_b(robot, anchor_id, base_id)
    cmd = base_term._command[0].detach().cpu().numpy()
    frames = contact_sensor.data.net_forces_w
    contact_events = 0
    if frames is not None and non_wheel_rows:
        contact_events = int((torch.linalg.norm(frames[:, non_wheel_rows], dim=-1) > 100.0).sum().item())
    records.append({
        "step": step_index,
        "phase": current_phase[0],
        "sub_phase": current_sub_phase[0],
        "sim_time_s": float(u.sim.current_time),
        "upper_command_vx_wz": [float(upper_command["vx"]), float(upper_command["wz"])],
        "effective_command_vx_wz": [filter_state["vx"], filter_state["wz"]],
        "action_base_raw": [float(raw[0, 0]), float(raw[0, 1])],
        "action_base_filtered": [float(used[0, 0]), float(used[0, 1])],
        "arm_actions_identical": bool(np.array_equal(raw[0, 2:].detach().cpu().numpy(),
                                                     used[0, 2:].detach().cpu().numpy())),
        "term_command_vx_wz": [float(cmd[0]), float(cmd[1])],
        "base_lin_vel_w": [float(v) for v in lin],
        "base_ang_vel_w": [float(v) for v in ang],
        "base_speed_mps": float(np.linalg.norm(lin[:2])),
        "base_pos_w": [float(v) for v in bp],
        "base_quat_w": [float(v) for v in bq],
        "base_yaw_deg": float(np.degrees(np.arctan2(*(quat_rotate(bq, np.array([1.0, 0.0, 0.0]))[[1, 0]])))),
        "steering_target": [float(v) for v in base_term._steering_target[0].detach().cpu().numpy()],
        "wheel_target": [float(v) for v in base_term._wheel_target[0].detach().cpu().numpy()],
        "wheel_actual": [float(v) for v in robot.data.joint_vel[0, base_term._wheel_ids].detach().cpu().numpy()],
        "ee_pos_err_m": float(torch.linalg.norm(tcp_pos[0] - arm_command["pos"].to(u.device)).item()),
        "terminated": terminated,
        "time_outs": timeouts,
        "contact_body_events": contact_events,
    })
    return obs_next, {"terminated": terminated, "time_outs": timeouts, "contact_events": contact_events}


current_phase = [""]
current_sub_phase = [""]

# ---------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------
flow = {"failed": False, "failure_phase": None, "failure_reason": None, "steps": 0}
obs, _ = wrapper.reset()
reset_pose = base_pose()
flow["reset_base_pos_w"] = [float(v) for v in reset_pose[0]]
flow["reset_base_quat_w"] = [float(v) for v in reset_pose[1]]


def run_steps(obs_now, total_steps, phase, sub_phase=""):
    """Advance ``total_steps`` steps in one phase; returns (ok, obs, failure_reason)."""
    current_phase[0] = phase
    current_sub_phase[0] = sub_phase
    for _ in range(total_steps):
        obs_now, info = policy_step(obs_now)
        if info["terminated"]:
            return False, obs_now, f"termination fired at step {step_index}"
    return True, obs_now, None


settle_ok, obs, reason = run_steps(obs, args.settle_steps, "settle")
if not settle_ok:
    flow.update(failed=True, failure_phase="settle", failure_reason=reason)
settle_pose = base_pose() if settle_ok else None
if settle_ok:
    upper_command.update(vx=args.drive_vx, wz=0.0)
    settle_ok, obs, reason = run_steps(obs, args.drive_steps, "drive")
    if not settle_ok:
        flow.update(failed=True, failure_phase="drive", failure_reason=reason)

if settle_ok:
    upper_command.update(vx=0.0, wz=args.turn_wz)
    settle_ok, obs, reason = run_steps(obs, args.turn_steps, "turn")
    if not settle_ok:
        flow.update(failed=True, failure_phase="turn", failure_reason=reason)

park_info = {"zero_command_step": step_index if settle_ok else None, "park_met_step": None,
             "quiet_steps": 0}
if settle_ok:
    upper_command.update(vx=0.0, wz=0.0)
    current_phase[0] = "park"
    park_met = False
    for _ in range(args.park_timeout):
        obs_new, info = policy_step(obs)
        obs = obs_new
        if info["terminated"]:
            flow.update(failed=True, failure_phase="park",
                        failure_reason=f"termination fired at step {step_index}")
            break
        lin, ang = base_velocities()
        quiet = float(np.linalg.norm(lin[:2])) < 0.02 and abs(float(ang[2])) < 0.05
        park_info["quiet_steps"] = park_info["quiet_steps"] + 1 if quiet else 0
        if park_info["quiet_steps"] >= 5:
            park_info["park_met_step"] = step_index
            park_met = True
            break
    if not park_met and not flow["failed"]:
        flow.update(failed=True, failure_phase="park",
                    failure_reason=f"chassis did not settle within {args.park_timeout} steps")
    settle_ok = park_met and not flow["failed"]

actual_park = base_pose()

# -- planned park pose: dead reckoning from the settled pose and the commanded motion --------
planned = None
if settle_pose is not None:
    sp, sq = settle_pose
    yaw0 = float(np.arctan2(*(quat_rotate(sq, np.array([1.0, 0.0, 0.0]))[[1, 0]])))
    yaw_plan = yaw0 + args.turn_wz * args.turn_steps * dt
    heading0 = np.array([math.cos(yaw0), math.sin(yaw0), 0.0])
    p_plan = sp + heading0 * (args.drive_vx * args.drive_steps * dt)
    q_plan = np.array([math.cos(yaw_plan / 2), 0.0, 0.0, math.sin(yaw_plan / 2)])
    planned = (p_plan, q_plan)

plan_doc = {
    "mode": args.run_mode,
    "reset_base_pos_w": flow["reset_base_pos_w"],
    "settle_base_pos_w": None if settle_pose is None else settle_pose[0].tolist(),
    "planned_park_pos_w": None if planned is None else planned[0].tolist(),
    "planned_park_quat_w": None if planned is None else planned[1].tolist(),
    "actual_park_pos_w": actual_park[0].tolist(),
    "actual_park_quat_w": actual_park[1].tolist(),
    "park_error_m": None if planned is None
    else float(np.linalg.norm(actual_park[0][:2] - planned[0][:2])),
    "park_yaw_error_deg": None if planned is None else float(
        np.degrees(np.arctan2(*(quat_rotate(actual_park[1], np.array([1.0, 0.0, 0.0]))[[1, 0]])))
        - np.degrees(np.arctan2(*(quat_rotate(planned[1], np.array([1.0, 0.0, 0.0]))[[1, 0]])))
    ),
    "park_info": park_info,
    "flow": flow,
}
plan_doc["flow"]["steps_at_plan"] = len(records)
json.dump(plan_doc, open(os.path.join(OUT_DIR, "plan.json"), "w"), indent=1)
print(f"[plan] planned={None if planned is None else np.round(planned[0], 4).tolist()} "
      f"actual={np.round(actual_park[0], 4).tolist()} "
      f"error={plan_doc['park_error_m']:.4f} m yaw_err={plan_doc['park_yaw_error_deg']:.2f} deg", flush=True)

# -- observation phase (full mode only) ------------------------------------------------------
view_results = {}
if args.run_mode == "full" and not flow["failed"]:
    if not args.plan:
        raise SystemExit("--run-mode full needs --plan <plan.json>")
    plan = json.load(open(args.plan if os.path.isabs(args.plan) else os.path.join(ROOT, args.plan)))
    park_pos = np.array(plan["planned_park_pos_w"])
    park_quat = np.array(plan["planned_park_quat_w"])

    for view in VIEW_ORDER:
        current_phase[0] = "observe"
        current_sub_phase[0] = f"{view}:arrival"
        geom = GEOMETRY[view]
        target_world = park_pos + quat_rotate(park_quat, np.array(geom["marker_pos_b"]))
        target_quat = park_quat
        place_active_marker(view, target_world, target_quat)
        arm_command["pos"] = torch.tensor(geom["tcp_pos_b"], dtype=torch.float32)
        arm_command["quat"] = torch.tensor(geom["tcp_quat_b"], dtype=torch.float32)

        entry = {
            "target_world": target_world.tolist(),
            "target_quat_w": target_quat.tolist(),
            "arm_command_pos_b": list(geom["tcp_pos_b"]),
            "arm_command_quat_b": list(geom["tcp_quat_b"]),
            "expected_camera_pos_w": (park_pos + quat_rotate(park_quat, np.array(geom["expected_camera_pos_b"]))).tolist(),
            "arrived": False, "arrival_step": None, "arrival_steps_taken": None,
            "hold_records": [], "terminated": False,
        }
        arrival = False
        for _ in range(args.arrival_timeout):
            obs_new, info = policy_step(obs)
            obs = obs_new
            if info["terminated"]:
                entry["terminated"] = True
                flow.update(failed=True, failure_phase=f"observe:{view}",
                            failure_reason=f"termination fired during arrival at step {step_index}")
                break
            tcp_pos, tcp_quat = tcp_pose_b(robot, anchor_id, base_id)
            pos_err = float(torch.linalg.norm(tcp_pos[0] - arm_command["pos"].to(u.device)).item())
            q_cur = tcp_quat[0].detach().cpu().numpy()
            ori_err = 2.0 * math.acos(min(1.0, abs(float(np.dot(q_cur, np.array(geom["tcp_quat_b"]))))))
            if pos_err <= POS_TOL and ori_err <= ORI_TOL:
                arrival = True
                entry["arrived"] = True
                entry["arrival_step"] = step_index
                entry["arrival_steps_taken"] = step_index - (park_info["park_met_step"] or 0)
                break
        if not arrival:
            if not flow["failed"]:
                flow.update(failed=True, failure_phase=f"observe:{view}",
                            failure_reason=f"did not reach the target within {args.arrival_timeout} steps")
            view_results[view] = entry
            break

        current_sub_phase[0] = f"{view}:hold"
        intr = camera.data.intrinsic_matrices[0].detach().cpu().numpy()
        for hold_idx in range(args.hold_steps):
            obs_new, info = policy_step(obs)
            obs = obs_new
            if info["terminated"]:
                entry["terminated"] = True
                flow.update(failed=True, failure_phase=f"observe:{view}",
                            failure_reason=f"termination fired during the hold at step {step_index}")
                break
            frame = read_frame()
            det = detect(frame, geom["marker_color"])
            tcp_pos, tcp_quat = tcp_pose_b(robot, anchor_id, base_id)
            cam_pos = camera.data.pos_w[0].detach().cpu().numpy()
            cam_quat_world = camera.data.quat_w_world[0].detach().cpu().numpy()
            cam_quat_ros = convert_camera_frame_orientation_convention(
                torch.tensor(cam_quat_world, dtype=torch.float32).unsqueeze(0), origin="world", target="ros"
            )[0].numpy()
            proj = project_point(cam_pos, cam_quat_ros, intr, target_world)
            visible = det["count"] >= MIN_VISIBLE_PIXELS
            q_cur = tcp_quat[0].detach().cpu().numpy()
            ori_err = 2.0 * math.acos(min(1.0, abs(float(np.dot(q_cur, np.array(geom["tcp_quat_b"]))))))
            entry["hold_records"].append({
                "hold_index": hold_idx,
                "step": step_index,
                "sim_time_s": float(u.sim.current_time),
                "u": det["u"], "v": det["v"], "pixels": det["count"], "visible": bool(visible),
                "extent_px": det["extent_px"],
                "projected_center_uv": proj,
                "pos_err_m": float(torch.linalg.norm(tcp_pos[0] - arm_command["pos"].to(u.device)).item()),
                "ori_err_rad": ori_err,
                "base_pos_w": [float(v) for v in base_pose()[0]],
                "base_yaw_deg": records[-1]["base_yaw_deg"],
                "base_speed_mps": records[-1]["base_speed_mps"],
                "terminated": int(info["terminated"]),
                "contact_body_events": int(info["contact_events"]),
            })
            if hold_idx in (0, args.hold_steps // 2, args.hold_steps - 1):
                from PIL import Image
                Image.fromarray(frame).save(os.path.join(OUT_DIR, f"{view}_hold_{hold_idx:03d}.png"))
        view_results[view] = entry

# -- summary ---------------------------------------------------------------------------------
flow["steps"] = len(records)
summary = {
    "config": CONFIG,
    "flow": flow,
    "plan": plan_doc,
    "views": view_results,
    "step_records": len(records),
}
json.dump(summary, open(os.path.join(OUT_DIR, "continuous_results.json"), "w"), indent=1)
json.dump(records, open(os.path.join(OUT_DIR, "continuous_steps.json"), "w"), indent=1)

print(f"[flow] failed={flow['failed']} phase={flow['failure_phase']} reason={flow['failure_reason']}")
for view, entry in view_results.items():
    uu = [r["u"] for r in entry["hold_records"] if r["u"] is not None]
    vv = [r["v"] for r in entry["hold_records"] if r["v"] is not None]
    if uu:
        off = np.hypot(np.array(uu) - IMG_W / 2, np.array(vv) - IMG_H / 2)
        print(f"[{view}] arrived={entry['arrived']} step={entry['arrival_step']} "
              f"visible={sum(r['visible'] for r in entry['hold_records'])}/{len(entry['hold_records'])} "
              f"mean_offset={off.mean():.1f}px std={np.std(uu):.2f}/{np.std(vv):.2f}")
print(f"wrote {OUT_DIR}")
