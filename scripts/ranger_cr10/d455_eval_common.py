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

"""Shared, run-independent pieces of the D455 fixed-view evaluation.

Both entry points use this module so the camera model, the three view definitions, the target
geometry and the pixel detection cannot drift apart:

* ``validate_d455_views.py`` - one reset per view, the A1 world-fixed target protocol;
* ``validate_d455_continuous.py`` - one reset for a drive/park/observe sequence.

Everything here is pure geometry or image processing; nothing touches an Isaac Lab object. Import
it only after ``AppLauncher`` has been constructed, because the TCP/camera chain constants come
from the task package.
"""

from __future__ import annotations

import math

import numpy as np

from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.pose_command_teleop import (
    TCP_DEFAULT_POS_B,
    TCP_DEFAULT_QUAT_B,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.tcp import TCP_OFFSET_POS, TCP_OFFSET_QUAT

# ---------------------------------------------------------------------------------------
# Pre-registered thresholds.  Nothing here depends on a run result.
# ---------------------------------------------------------------------------------------
WARMUP_STEPS = 5
ARRIVAL_TIMEOUT = 80
HOLD_STEPS = 60
POS_TOL = 0.03                 # m
ORI_TOL = 0.10                 # rad
VISIBLE_MIN_RATIO = 0.90
PIXEL_OFFSET_MAX = 20.0        # px, mean centroid distance from the principal point
PIXEL_JITTER_MAX = 5.0         # px, std of the centroid over the hold window
AXIS_MAX_DEG = 10.0
MOUNT_CONSISTENCY_MAX = 0.005  # m, Link6-derived camera position vs Camera.data.pos_w
MIN_VISIBLE_PIXELS = 30

# D455 colour camera, nominal and deliberately conservative: 640x480, square pixels, f = 372 px.
# VFOV then equals the real colour camera's 65.6 deg; HFOV is 81.3 deg versus the real ~86 deg,
# i.e. narrower, which can only make the visibility test stricter.
IMG_W, IMG_H, F_PX = 640, 480, 372.0
FOCAL_LENGTH_MM = 18.6
HORIZONTAL_APERTURE_MM = FOCAL_LENGTH_MM * IMG_W / F_PX   # 32.0 mm
MARKER_SIZE_M = 0.06
MARKER_DISTANCE_M = 0.35

# URDF chain: cr10_Link6 --(identity)--> gripper_base_link --(d455_joint)--> d455_link
#             --(d455_color_joint)--> d455_color_frame --(d455_color_optical_joint)-->
#             d455_color_optical_frame, from ranger_cr10_simple_v2.urdf.
D455_JOINT_POS = (0.0, 0.10, 0.14)
D455_JOINT_RPY = (0.785398, -1.570796, -2.356194)
D455_COLOR_POS = (0.0, -0.059, 0.0)
D455_COLOR_OPTICAL_RPY = (-1.5707963267948966, 0.0, -1.5707963267948966)

# TCP command poses as offsets from the default TCP pose (same parameterisation as
# UniformTCPBodyPoseCommand), all inside the stage-3 training box: pos_x [-0.15, 0.40],
# pos_y [-0.40, 0.10], pos_z [-0.30, 0.40], roll +-45 deg, pitch [-30, 45] deg, yaw +-60 deg.
VIEWS = {
    "front": {"offset": (0.05, 0.00, 0.00), "rpy": (0.0, 0.0, 0.0), "color": (1.0, 0.0, 0.0)},
    "side": {"offset": (0.05, -0.26, 0.00), "rpy": (0.0, 0.0, math.radians(50.0)), "color": (0.0, 1.0, 0.0)},
    "top": {"offset": (0.10, 0.00, 0.12), "rpy": (math.radians(-25.0), math.radians(40.0), 0.0),
            "color": (0.0, 0.0, 1.0)},
}


# ---------------------------------------------------------------------------------------
# Minimal wxyz quaternion helpers, so the geometry can be checked before the env exists.
# ---------------------------------------------------------------------------------------
def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw to (w, x, y, z), i.e. Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_mul(quat_mul(q, np.array([0.0, *v])), quat_conj(q))[1:]


def compose(p1, q1, p2, q2):
    return np.asarray(p1) + quat_rotate(np.asarray(q1), np.asarray(p2)), quat_mul(np.asarray(q1), np.asarray(q2))


def invert(p, q):
    qc = quat_conj(np.asarray(q))
    return -quat_rotate(qc, np.asarray(p)), qc


def axis_angle_deg(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Angle between the optical (+Z) axes of two orientations, degrees."""
    za = quat_rotate(np.asarray(q_a), np.array([0.0, 0.0, 1.0]))
    zb = quat_rotate(np.asarray(q_b), np.array([0.0, 0.0, 1.0]))
    return math.degrees(math.acos(float(np.clip(np.dot(za, zb), -1.0, 1.0))))


_mid_p, _mid_q = compose(np.array(D455_JOINT_POS), quat_from_rpy(*D455_JOINT_RPY),
                         np.array(D455_COLOR_POS), np.array([1.0, 0.0, 0.0, 0.0]))
LINK6_CAM_POS, LINK6_CAM_QUAT = compose(_mid_p, _mid_q, np.zeros(3), quat_from_rpy(*D455_COLOR_OPTICAL_RPY))
TCP_LINK6_POS, TCP_LINK6_QUAT = invert(np.array(TCP_OFFSET_POS), np.array(TCP_OFFSET_QUAT))
DEFAULT_TCP_POS = np.array(TCP_DEFAULT_POS_B)
DEFAULT_TCP_QUAT = np.array(TCP_DEFAULT_QUAT_B)


def tcp_to_camera(pos_b, quat_b):
    """Expected camera pose in the chassis frame for a commanded TCP pose."""
    link6_pos, link6_quat = compose(np.asarray(pos_b), np.asarray(quat_b), TCP_LINK6_POS, TCP_LINK6_QUAT)
    return compose(link6_pos, link6_quat, LINK6_CAM_POS, LINK6_CAM_QUAT)


def view_geometry(name: str) -> dict:
    spec = VIEWS[name]
    pos = DEFAULT_TCP_POS + np.array(spec["offset"])
    quat = quat_mul(quat_from_rpy(*spec["rpy"]), DEFAULT_TCP_QUAT)
    cam_pos, cam_quat = tcp_to_camera(pos, quat)
    axis = quat_rotate(cam_quat, np.array([0.0, 0.0, 1.0]))
    return {
        "name": name,
        "tcp_pos_b": pos.tolist(),
        "tcp_quat_b": quat.tolist(),
        "expected_camera_pos_b": cam_pos.tolist(),
        "expected_camera_quat_b": cam_quat.tolist(),
        "expected_optical_axis_b": axis.tolist(),
        "marker_pos_b": (cam_pos + MARKER_DISTANCE_M * axis).tolist(),
        "marker_size_m": MARKER_SIZE_M,
        "marker_distance_m": MARKER_DISTANCE_M,
        "marker_color": list(spec["color"]),
        "expected_pixel": [IMG_W / 2.0, IMG_H / 2.0],
        "expected_marker_edge_px": F_PX * MARKER_SIZE_M / MARKER_DISTANCE_M,
    }


# Self-check against plan section 20.5: at the authored default pose the D455 optical axis is
# measured within ~0.9 deg of base +X.  A wrong rpy/quaternion convention fails here loudly,
# instead of showing up later as a mysteriously empty frame.
_default_axis = quat_rotate(tcp_to_camera(DEFAULT_TCP_POS, DEFAULT_TCP_QUAT)[1], np.array([0.0, 0.0, 1.0]))
DEFAULT_AXIS_ERR_DEG = math.degrees(
    math.acos(float(np.clip(np.dot(_default_axis, [1.0, 0.0, 0.0]), -1.0, 1.0)))
)
if DEFAULT_AXIS_ERR_DEG > 5.0:
    raise RuntimeError(
        f"D455 mount self-check failed: optical axis is {DEFAULT_AXIS_ERR_DEG:.2f} deg from base +X "
        "(plan section 20.5 measured 0.87 deg). Check the URDF rpy convention."
    )


def project_point(cam_pos, cam_quat_ros, intrinsic, point_w):
    """Pixel of a world point in the D455 image, or None if it is behind the camera."""
    local = quat_rotate(quat_conj(np.asarray(cam_quat_ros)), np.asarray(point_w) - np.asarray(cam_pos))
    if local[2] <= 1e-6:
        return None
    u = intrinsic[0, 0] * local[0] / local[2] + intrinsic[0, 2]
    v = intrinsic[1, 1] * local[1] / local[2] + intrinsic[1, 2]
    return [float(u), float(v), float(local[2])]


def detect(frame: np.ndarray, color) -> dict:
    """Pixels dominated by the marker's channel, e.g. red = r > 1.6*max(g, b) and r > 50.

    Channel dominance rather than a distance to the nominal RGB: a Lambertian surface under the
    scene lights renders darker than its albedo, so an absolute-distance match would miss a
    correctly placed marker.
    """
    rgb = frame[..., :3].astype(np.int16)
    dominant = int(np.argmax(color))
    others = [i for i in range(3) if i != dominant]
    channel = rgb[..., dominant]
    mask = (channel > 50) & (channel > 1.6 * rgb[..., others[0]]) & (channel > 1.6 * rgb[..., others[1]])
    count = int(mask.sum())
    if count == 0:
        return {"count": 0, "u": None, "v": None, "extent_px": None}
    ys, xs = np.nonzero(mask)
    return {
        "count": count,
        "u": float(xs.mean()),
        "v": float(ys.mean()),
        "extent_px": float(max(xs.max() - xs.min(), ys.max() - ys.min()) + 1),
    }
