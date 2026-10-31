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

"""Stage 4.6: drive the RangerBox-CR10 in MuJoCo with the trained 8-D teleoperation policy.

One policy, one actor, eight numbers out: ``[vx, wz, cr10_joint1..6]``. The chassis numbers
go through the same base-controller math Isaac Lab uses (``ChassisAction``), the arm numbers
through ``q_default + 0.25 * raw``. The observation is assembled by ``RangerPolicyObs`` from
the same eight terms, in the same order, with the same scales.

Everything that is an *interface* lives in ``ranger_policy.py`` and is checked against the
live Isaac Lab env by ``scripts/ranger_cr10/check_mujoco_interface.py``. This file only wires
it to the keyboard, the zero-velocity filter and the viewer.

Usage:
    python mujoco/deploy/deploy_mujoco/ranger_cr10/ranger_cr10_deploy.py
    python .../ranger_cr10_deploy.py --no-filter          # raw PPO, for comparison
    python .../ranger_cr10_deploy.py --headless --duration 20

The MuJoCo viewer (the C++ ``simulate`` UI) binds plain letters to rendering toggles and
``pynput`` only observes keys, so keep the **terminal** focused while teleoperating -- see
MEMORY.md, "The MuJoCo viewer steals the teleop keys".
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

import mujoco

HERE = Path(__file__).resolve().parent
DEPLOY_MUJOCO = HERE.parent.parent  # mujoco/deploy
ROOT = DEPLOY_MUJOCO.parent.parent  # repository root
for p in (str(HERE), str(DEPLOY_MUJOCO), str(ROOT / "scripts" / "ranger_cr10")):
    if p not in sys.path:
        sys.path.append(p)

# the one implementation of the 4.5k filter; deliberately not copied
from base_command_filter import BaseCommandFilter  # noqa: E402
from deploy_mujoco.ranger_cr10.keyboard_teleop import (  # noqa: E402
    HELP_TEXT,
    KeyboardTeleopConfig,
    KeyboardTeleopController,
)
from deploy_mujoco.ranger_cr10.ranger_policy import (  # noqa: E402
    ACTION_SCALE,
    ARM_DEFAULT,
    ARM_JOINT_NAMES,
    STEERING_JOINT_NAMES,
    TCP_DEFAULT_POS_B,
    TCP_DEFAULT_QUAT_B,
    WHEEL_JOINT_NAMES,
    ChassisAction,
    EndEffector,
    RangerPolicyObs,
    act,
    ee_command_from_offset,
    load_actor,
)


def build_script(name: str) -> list[tuple[float, str, str]]:
    """A canned key timeline, so the whole teleop path can be exercised headless.

    It is the sequence the acceptance test describes: drive forward, stop, move the tool
    along X, tilt it, turn the chassis, then reset. ``--headless --script demo`` runs it
    once and the status lines show each command taking effect.
    """
    if name != "demo":
        raise SystemExit(f"unknown script '{name}' (only 'demo')")
    return [
        (1.0, "press", "w"),
        (3.0, "release", "w"),
        (4.0, "press", "i"),
        (6.0, "release", "i"),
        (6.5, "press", "2"),  # roll +5 deg (discrete key)
        (6.6, "release", "2"),
        (7.5, "press", "q"),
        (9.5, "release", "q"),
        (10.0, "press", "r"),  # reset: velocity to zero, target walks home
        (10.5, "release", "r"),
    ]


def load_config(path: Path) -> dict:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    for key in ("policy_path", "scene_path"):
        cfg[key] = cfg[key].replace("{CURRENT_ROOT_DIR}", str(ROOT))
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml", help="YAML in the same directory as this script")
    ap.add_argument("--policy", default=None, help="override policy_path from the config")
    ap.add_argument("--filter", dest="filter", action="store_true", default=None, help="force the filter on")
    ap.add_argument("--no-filter", dest="filter", action="store_false", help="force the filter off")
    ap.add_argument("--headless", action="store_true", help="no viewer window (for timing / CI)")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds of sim time; 0 = run until closed")
    ap.add_argument("--script", default=None, help="inject a canned key timeline (only 'demo')")
    ap.add_argument("--device", default="cpu")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = load_config(HERE / args.config)
    policy_path = args.policy or cfg["policy_path"]

    model = mujoco.MjModel.from_xml_path(cfg["scene_path"])
    model.opt.timestep = float(cfg["simulation_dt"])
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    decimation = int(cfg["control_decimation"])
    actor, kind = load_actor(policy_path, args.device)

    obs_builder = RangerPolicyObs(model, data)
    ee = EndEffector(model, data)
    chassis = ChassisAction()

    def ctrl_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    steer_ids = [ctrl_id(f"{n}_pos") for n in STEERING_JOINT_NAMES]
    wheel_ids = [ctrl_id(f"{n}_vel") for n in WHEEL_JOINT_NAMES]
    arm_ids = [ctrl_id(f"{n}_pos") for n in ARM_JOINT_NAMES]

    fcfg = cfg["filter"]
    use_filter = fcfg["enabled"] if args.filter is None else args.filter
    filt = BaseCommandFilter(
        vx_deadband=fcfg["vx_deadband"] if use_filter else 0.0,
        wz_deadband=fcfg["wz_deadband"] if use_filter else 0.0,
        lock_output=bool(fcfg["lock_output"]) and use_filter,
        hysteresis=fcfg["hysteresis"],
    )

    kcfg = cfg["keyboard"]
    kb = KeyboardTeleopController(
        default_pos=TCP_DEFAULT_POS_B,
        default_quat=TCP_DEFAULT_QUAT_B,
        cfg=KeyboardTeleopConfig(
            vx_max=kcfg["vx_max"],
            wz_max=kcfg["wz_max"],
            vel_rate=kcfg["vel_rate"],
            vel_return_rate=kcfg["vel_return_rate"],
        ),
    )

    print(HELP_TEXT)
    print(f"policy  {policy_path}  ({kind})")
    print(
        f"filter  {'on' if filt.enabled else 'off'}  vx_deadband={filt.vx_deadband} wz_deadband={filt.wz_deadband} "
        f"lock={filt.lock_output}"
    )
    print("keep this terminal focused: the MuJoCo viewer also listens to letters\n")

    import torch

    last_action = np.zeros(8)
    vx_eff, wz_eff = 0.0, 0.0
    obs = obs_builder.reset(last_action, (vx_eff, wz_eff), ee_command_from_offset((0, 0, 0), (0, 0, 0)))

    physics_dt = float(model.opt.timestep)
    tick = 0
    sim_time = 0.0
    status_period = 1.0 / float(cfg.get("status_fps", 5.0))
    next_status = 0.0

    def step_once() -> None:
        nonlocal last_action, vx_eff, wz_eff, obs, tick, sim_time
        kb.update(decimation * physics_dt)
        raw_vx, raw_wz = kb.velocity_command

        vx_t = torch.tensor([raw_vx], dtype=torch.float32)
        wz_t = torch.tensor([raw_wz], dtype=torch.float32)
        vx_t, wz_t = filt.filter_command(vx_t, wz_t)
        vx_eff, wz_eff = float(vx_t[0]), float(wz_t[0])

        ee_cmd = np.asarray(kb.ee_pose_command_b, dtype=np.float64)
        obs = obs_builder.push(last_action, (vx_eff, wz_eff), ee_cmd)
        raw = act(actor, obs, args.device)

        a_t = torch.as_tensor(raw, dtype=torch.float32).unsqueeze(0)
        a_t = filt.filter_action(a_t, vx_t, wz_t)
        raw = a_t[0].numpy().astype(np.float64)

        base = chassis.apply(raw[:2])
        data.ctrl[steer_ids] = base["steering_target"]
        data.ctrl[wheel_ids] = base["wheel_speed_target"]
        data.ctrl[arm_ids] = ARM_DEFAULT + ACTION_SCALE * raw[2:]

        for _ in range(decimation):
            mujoco.mj_step(model, data)
        sim_time += decimation * physics_dt
        last_action = raw
        tick += 1

    def status() -> None:
        vx, wz = kb.velocity_command
        pos_off, rpy_off = kb.pose_offset
        ee_cmd = np.asarray(kb.ee_pose_command_b, dtype=np.float64)
        pos_err, ori_err = ee.errors(ee_cmd)
        lin, ang = obs_builder.base_velocity_local()
        line = (
            f"t={sim_time:6.1f}s | cmd vx {vx:+.3f} wz {wz:+.3f} -> {vx_eff:+.3f} {wz_eff:+.3f} | "
            f"base v {lin[0]:+.3f} w {ang[2]:+.3f} | "
            f"EE off {pos_off[0]:+.3f} {pos_off[1]:+.3f} {pos_off[2]:+.3f} m "
            f"rpy {np.degrees(rpy_off[0]):+5.1f} {np.degrees(rpy_off[1]):+5.1f} {np.degrees(rpy_off[2]):+5.1f} deg | "
            f"err {pos_err * 1000:5.1f} mm {np.degrees(ori_err):4.2f} deg | filter {'on' if filt.enabled else 'off'}"
        )
        if sys.stdout.isatty():
            sys.stdout.write("\r" + line)
        else:
            print(line)
        sys.stdout.flush()

    script = build_script(args.script) if args.script else []
    if script:
        print(f"scripted keyboard: {len(script)} events, ends at {script[-1][0]:.1f} s\n")

    def run_script_events() -> None:
        while script and sim_time >= script[0][0]:
            _, mode, key = script.pop(0)
            (kb.press if mode == "press" else kb.release)(key)
            print(f"  [t={sim_time:5.1f}s] {mode} '{key}'")

    start = time.time()
    if args.headless:
        while args.duration <= 0.0 or sim_time < args.duration:
            t0 = time.time()
            run_script_events()
            step_once()
            if sim_time >= next_status:
                status()
                next_status += status_period
            if args.duration <= 0.0 and kb.running is False:
                break
            throttle = t0 + decimation * physics_dt - time.time()
            if throttle > 0:
                time.sleep(throttle)
    else:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(model, data) as v:
            base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = base_id
            v.cam.distance = 3.5
            v.cam.elevation = -18
            v.cam.azimuth = 120
            while v.is_running() and kb.running:
                t0 = time.time()
                step_once()
                v.sync()
                if sim_time >= next_status:
                    status()
                    next_status += status_period
                throttle = t0 + decimation * physics_dt - time.time()
                if throttle > 0:
                    time.sleep(throttle)
                if args.duration > 0.0 and sim_time >= args.duration:
                    break

    print()
    print(f"stopped after {tick} policy steps, {sim_time:.1f} s of sim time (wall {time.time() - start:.1f} s)")
    kb.close()
    if not args.headless:
        # The passive viewer's GL teardown segfaults on this machine (reproduced with a bare
        # `launch_passive` loop and no policy at all, on both the default GL backend and EGL).
        # Nothing is left to do after the summary, so skip interpreter teardown rather than
        # exit 139 on every session.
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
