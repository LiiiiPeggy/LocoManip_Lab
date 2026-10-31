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

"""The 117-D observation and the 8-D action, ported from the Isaac Lab task, for MuJoCo.

Two classes and a loader, all free of Isaac Sim so they can run in the MuJoCo process:

* :class:`RangerPolicyObs` assembles the policy observation exactly as the env config does:
  eight terms in declaration order, three frames each, scales applied per term
  (``base_ang_vel * 0.2``, ``joint_vel * 0.05``, ``base_lin_vel * 2.0``, everything else 1).
  The 3-frame history is a per-term rolling buffer initialised by repeating the first frame,
  which is what Isaac Lab's history manager does at reset.
* :class:`ChassisAction` is a line-for-line port of ``BaseVelocityAction.apply_actions`` --
  the DUAL_ACKERMAN / SPINNING branch, the half-turn fold, the two clamps and the per-side
  wheel signs. ``scripts/ranger_cr10/check_mujoco_interface.py`` drives both implementations
  from the same raw actions and asserts they agree, so the duplication cannot rot silently.
* :func:`load_actor` reads an rsl_rl checkpoint (``actor_state_dict``) or a TorchScript
  export. The deterministic action is the network's mean: the Gaussian's ``log_std_param``
  is not used here, which is what "deterministic Actor inference" means.

Every constant below is copied from the task definition with its source named, because the
whole point of the sim-to-sim comparison is that these are *not* free parameters.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# -- from ranger_cr10_articulation_cfg.py ------------------------------------------------
ARM_JOINT_NAMES = [f"cr10_joint{i}" for i in range(1, 7)]
STEERING_JOINT_NAMES = [
    "fr_steering_joint",
    "fl_steering_wheel_joint",
    "rl_steering_wheel_joint",
    "rr_steering_wheel_joint",
]
WHEEL_JOINT_NAMES = ["fr_wheel_joint", "fl_wheel_joint", "rl_wheel_joint", "rr_wheel_joint"]
ARM_DEFAULT = np.array([0.6, 1.0, -2.4, 0.0, 1.4, 0.0], dtype=np.float64)

# -- from TCPBodyPoseCommand (pose_command_teleop.py / tcp.py) ---------------------------
TCP_BODY = "cr10_Link6"
TCP_OFFSET_POS = (0.0, 0.00003, 0.14389)
TCP_DEFAULT_POS_B = np.array([0.5519, -0.0742, 0.4696])
TCP_DEFAULT_QUAT_B = np.array([0.032072, -0.875128, 0.469424, -0.112978])

# -- from the env cfg ---------------------------------------------------------------------
ACTION_SCALE = 0.25  # JointPositionActionCfg(scale=0.25, use_default_offset=True)
BASE_COMMAND_LIMIT = 0.5  # BaseVelocityActionCfg.clip = ((-0.5, -0.5), (0.5, 0.5))
OBS_HISTORY = 3
NUM_ACTIONS = 8
NUM_OBS = 117

# -- from BaseVelocityActionCfg ----------------------------------------------------------
WHEEL_XY = np.array(
    [
        (0.445, -0.28),  # fr
        (0.445, 0.28),  # fl
        (-0.445, 0.28),  # rl
        (-0.445, -0.28),  # rr
    ]
)
WHEEL_RADIUS = 0.1531
MIN_TURN_RADIUS = 0.810330349
MAX_STEER_ANGLE = 0.6981
MAX_SPIN_RATE = 0.7853
MAX_SPIN_STEER_ANGLE = 1.5708
STEERING_SIGN = -1.0
WHEEL_SIGN = np.array([-1.0, 1.0, 1.0, -1.0])

HIDDEN = (512, 256, 128)


# --------------------------------------------------------------------------------------
# quaternion helpers, (w, x, y, z), matching isaaclab.utils.math
# --------------------------------------------------------------------------------------
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_euler_xyz(roll, pitch, yaw):
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(R):
    """Rotation matrix -> (w, x, y, z) with w >= 0, so it is a stable observation input."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def axis_angle_from_quat(q):
    """``isaaclab.utils.math.axis_angle_from_quat``: the rotation vector of a quaternion."""
    q = q if q[3] >= 0.0 else -q  # the shortest representation
    w = min(1.0, max(-1.0, abs(q[0])))
    if w > 1.0 - 1e-9:
        return 2.0 * q[1:]
    angle = 2.0 * math.acos(w)
    s = math.sqrt(1.0 - w * w)
    return angle * q[1:] / s


def orientation_error(q_actual, q_cmd):
    """Angle (rad) between two attitudes, matching ``compute_pose_error(..., 'axis_angle')``."""
    q_err = quat_mul(q_cmd, quat_conjugate(q_actual))
    return float(np.linalg.norm(axis_angle_from_quat(q_err)))


# --------------------------------------------------------------------------------------
# chassis action
# --------------------------------------------------------------------------------------
class ChassisAction:
    """``BaseVelocityAction`` for a single robot: ``(vx, wz)`` in, eight joint targets out."""

    def __init__(self) -> None:
        self.wheel_xy = WHEEL_XY.copy()
        self.wheel_sign = WHEEL_SIGN.copy()

    @staticmethod
    def command_from_raw(raw_2):
        """`(low + high)/2 + (high - low)/2 * tanh(raw)` with low/high = -0.5 / 0.5."""
        return BASE_COMMAND_LIMIT * np.tanh(np.asarray(raw_2, dtype=np.float64))

    def apply(self, raw_2):
        vx, wz = self.command_from_raw(raw_2)

        spinning = False
        abs_wz = abs(wz)
        if abs_wz > 1e-4:
            radius = abs(vx) / max(abs_wz, 1e-4)
            spinning = (radius < MIN_TURN_RADIUS) or (abs(vx) < 1e-3)
        if spinning:
            wz = float(np.clip(wz, -MAX_SPIN_RATE, MAX_SPIN_RATE))
            vx = 0.0

        wx = self.wheel_xy[:, 0]
        wy = self.wheel_xy[:, 1]
        v_vec_x = vx - wz * wy
        v_vec_y = wz * wx
        steer = np.arctan2(v_vec_y, v_vec_x)
        speed = np.hypot(v_vec_x, v_vec_y)

        flip = np.abs(steer) > (math.pi / 2)
        steer = np.where(flip, steer - np.sign(steer) * math.pi, steer)
        speed = np.where(flip, -speed, speed)

        steer_limit = MAX_SPIN_STEER_ANGLE if spinning else MAX_STEER_ANGLE
        steer = np.clip(steer, -steer_limit, steer_limit)

        return dict(
            vx=vx,
            wz=wz,
            spinning=spinning,
            steering_target=STEERING_SIGN * steer,
            wheel_speed_target=self.wheel_sign * speed / WHEEL_RADIUS,
        )


# --------------------------------------------------------------------------------------
# observation
# --------------------------------------------------------------------------------------
class RangerPolicyObs:
    """Assemble the 117-D policy observation from a MuJoCo ``mjData``."""

    # per-term scale, from ObservationsCfg.PolicyCfg
    SCALES = dict(
        base_ang_vel=0.2,
        projected_gravity=1.0,
        joint_pos=1.0,
        joint_vel=0.05,
        actions=1.0,
        velocity_commands=1.0,
        pos_commands=1.0,
        base_lin_vel=2.0,
    )
    # per-term width
    DIMS = dict(
        base_ang_vel=3,
        projected_gravity=3,
        joint_pos=6,
        joint_vel=6,
        actions=8,
        velocity_commands=3,
        pos_commands=7,
        base_lin_vel=3,
    )
    ORDER = (
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "velocity_commands",
        "pos_commands",
        "base_lin_vel",
    )

    def __init__(self, model, data) -> None:
        import mujoco

        self.m = model
        self.d = data
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self._arm_qadr = []
        for name in ARM_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._arm_qadr.append(model.jnt_qposadr[jid])
            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"{name} is not a hinge in the MuJoCo model")
        self._arm_dadr = []
        for name in ARM_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._arm_dadr.append(model.jnt_dofadr[jid])
        self.history: dict[str, list[np.ndarray]] = {}
        self._buf = np.zeros(6)
        self._res = np.zeros(6, dtype=np.float64)

    def base_velocity_local(self):
        """``(linear, angular)`` of base_link in the base frame, metres/s and rad/s.

        MuJoCo writes a free joint's six velocities in the **world** frame (verified by
        setting a yaw of 90 degrees and ``qvel[0] = 1``: the body then moves along world +x,
        not along its own +x). ``mj_objectVelocity(..., flg_local=1)`` returned the opposite
        sign of what the rotation implies, so the rotation is done here, explicitly.
        """
        R = self.d.xmat[self.base_id].reshape(3, 3)
        return R.T @ self.d.qvel[0:3], R.T @ self.d.qvel[3:6]

    def joint_state(self):
        q = np.array([self.d.qpos[a] for a in self._arm_qadr])
        dq = np.array([self.d.qvel[a] for a in self._arm_dadr])
        return q, dq

    def frame(self, last_action, vel_cmd, ee_cmd):
        """One observation frame, before history, as the eight named terms."""
        lin_vel, ang_vel = self.base_velocity_local()
        q, dq = self.joint_state()
        R = self.d.xmat[self.base_id].reshape(3, 3)
        gravity = R.T @ np.array([0.0, 0.0, -1.0])
        return dict(
            base_ang_vel=np.asarray(ang_vel),
            projected_gravity=gravity,
            joint_pos=q - ARM_DEFAULT,
            joint_vel=dq,
            actions=np.asarray(last_action, dtype=np.float64),
            velocity_commands=np.array([vel_cmd[0], 0.0, vel_cmd[1]]),
            pos_commands=np.asarray(ee_cmd, dtype=np.float64),
            base_lin_vel=np.asarray(lin_vel),
        )

    def reset(self, last_action, vel_cmd, ee_cmd):
        """Fill the whole history with the current frame -- what Isaac does at reset."""
        f = self.frame(last_action, vel_cmd, ee_cmd)
        self.history = {k: [f[k].copy() for _ in range(OBS_HISTORY)] for k in self.ORDER}
        return self._concat()

    def push(self, last_action, vel_cmd, ee_cmd):
        f = self.frame(last_action, vel_cmd, ee_cmd)
        for k in self.ORDER:
            self.history[k].append(f[k])
            del self.history[k][0]
        return self._concat()

    def _concat(self):
        parts = []
        for k in self.ORDER:
            stacked = np.concatenate(self.history[k]) * self.SCALES[k]
            assert stacked.size == OBS_HISTORY * self.DIMS[k], (k, stacked.size)
            parts.append(stacked)
        obs = np.concatenate(parts).astype(np.float32)
        assert obs.size == NUM_OBS, obs.size
        return obs


# --------------------------------------------------------------------------------------
# end-effector
# --------------------------------------------------------------------------------------
class EndEffector:
    """TCP pose relative to ``base_link``, from the MuJoCo forward kinematics."""

    def __init__(self, model, data) -> None:
        import mujoco

        self.m, self.d = model, data
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.anchor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TCP_BODY)
        self.offset = np.asarray(TCP_OFFSET_POS, dtype=np.float64)

    def tcp_pose_b(self):
        import mujoco

        mujoco.mj_forward(self.m, self.d)
        Rb = self.d.xmat[self.base_id].reshape(3, 3)
        pos_b = Rb.T @ (
            self.d.xpos[self.anchor_id]
            + self.d.xmat[self.anchor_id].reshape(3, 3) @ self.offset
            - self.d.xpos[self.base_id]
        )
        quat_b = matrix_to_quat(Rb.T @ self.d.xmat[self.anchor_id].reshape(3, 3))
        return pos_b, quat_b

    def errors(self, cmd7):
        pos, quat = self.tcp_pose_b()
        return float(np.linalg.norm(pos - cmd7[:3])), orientation_error(quat, cmd7[3:])


def ee_command_from_offset(offset_pos, offset_rpy):
    """Chassis-relative target from an offset, composed exactly as the env samples one."""
    pos = TCP_DEFAULT_POS_B + np.asarray(offset_pos, dtype=np.float64)
    offset_quat = quat_from_euler_xyz(*offset_rpy)
    quat = quat_mul(offset_quat, TCP_DEFAULT_QUAT_B)
    return np.concatenate([pos, quat])


# --------------------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------------------
def load_actor(path, device="cpu"):
    """Load the deterministic actor from an rsl_rl checkpoint or a TorchScript module."""
    import torch

    path = Path(path)
    try:
        module = torch.jit.load(str(path), map_location=device)
        module.eval()
        return module, "torchscript"
    except Exception:
        pass

    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    raw = ck["actor_state_dict"] if isinstance(ck, dict) and "actor_state_dict" in ck else ck
    sd = {(k[4:] if k.startswith("mlp.") else k): v for k, v in raw.items() if k.split(".")[-1] in ("weight", "bias")}
    net = torch.nn.Sequential(
        torch.nn.Linear(NUM_OBS, HIDDEN[0]),
        torch.nn.ELU(),
        torch.nn.Linear(HIDDEN[0], HIDDEN[1]),
        torch.nn.ELU(),
        torch.nn.Linear(HIDDEN[1], HIDDEN[2]),
        torch.nn.ELU(),
        torch.nn.Linear(HIDDEN[2], NUM_ACTIONS),
    )
    net.load_state_dict(sd)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, "rsl_rl"


def act(actor, obs, device="cpu"):
    """One deterministic forward pass: float32 in, float32 out, no grad."""
    import torch

    with torch.inference_mode():
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device).unsqueeze(0)
        y = actor(x)
    return y[0].detach().cpu().numpy().astype(np.float64)
