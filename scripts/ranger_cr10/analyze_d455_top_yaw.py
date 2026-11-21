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

"""Heading-only reprojection of the A1 ``top`` trajectory, plus piecewise yaw.

The four-way decomposition in ``analyze_d455_top_motion.py`` already showed that most of the
first-second slide comes from chassis rotation rather than translation or arm motion. This script
asks the narrower question: of that rotation, how much is *heading* (yaw about the world vertical
axis) alone?

Heading convention: the chassis yaw is the ZYX (yaw-pitch-roll) yaw of the ``base_link`` world
orientation, ``yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))`` for a ``(w, x, y, z)`` quaternion.
The yaw-only pose applies ``Rz(dyaw)`` about the world +Z **through the base origin**, with the
origin and the camera-relative-to-base pose frozen at the first hold frame.

Reads ``d455_detail_top.json``; no simulation.

Usage:
    python scripts/ranger_cr10/analyze_d455_top_yaw.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--detail", default=str(ROOT / "outputs/stage413/b/d455_views_detail/d455_detail_top.json"))
ap.add_argument("--out", default=str(ROOT / "outputs/stage413/b/d455_views_detail/top_yaw_only_projection.json"))
args = ap.parse_args()


def q_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_rot(q, v):
    return q_mul(q_mul(q, np.array([0.0, *v])), q_conj(q))[1:]


def yaw_of(q):
    """ZYX yaw of a (w, x, y, z) quaternion, radians."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def q_from_yaw(yaw):
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def compose(p1, q1, p2, q2):
    return np.asarray(p1) + q_rot(np.asarray(q1), np.asarray(p2)), q_mul(np.asarray(q1), np.asarray(q2))


def invert(p, q):
    """Inverse rigid transform: (-R^T p, q^-1). Note the rotation is applied to p, not to the
    difference of two positions."""
    qc = q_conj(np.asarray(q))
    return -q_rot(qc, np.asarray(p)), qc


def project(cam_pos, cam_quat, intr, point_w):
    local = q_rot(q_conj(np.asarray(cam_quat)), np.asarray(point_w) - np.asarray(cam_pos))
    if local[2] <= 1e-6:
        return None
    return np.array([
        intr[0][0] * local[0] / local[2] + intr[0][2],
        intr[1][1] * local[1] / local[2] + intr[1][2],
    ])


data = json.loads(Path(args.detail).read_text())
records, intr = data["records"], data["intrinsics"]
dt = data["control_dt_s"]
target = np.array(records[0]["target_pos_w"])

base = [(np.array(r["base_pos_w"]), np.array(r["base_quat_w"])) for r in records]
cam_w = [(np.array(r["cam_pos_w"]), np.array(r["cam_quat_w_ros"])) for r in records]
# camera pose in the chassis frame at the first hold frame
c0_pos, c0_quat = compose(*invert(base[0][0], base[0][1]), cam_w[0][0], cam_w[0][1])

heading = np.unwrap(np.array([yaw_of(q) for _, q in base]))
dyaw = np.degrees(heading - heading[0])

# yaw-only, rotation-only and measured poses, all with the camera frozen relative to the base.
yaw_only, rot_only = [], []
for i, ((bp, bq), _cam) in enumerate(zip(base, cam_w)):
    bq_yaw = q_mul(q_from_yaw(heading[i] - heading[0]), base[0][1])
    yaw_cam = compose(base[0][0], bq_yaw, c0_pos, c0_quat)
    rot_cam = compose(base[0][0], bq, c0_pos, c0_quat)
    yaw_only.append(project(yaw_cam[0], yaw_cam[1], intr, target))
    rot_only.append(project(rot_cam[0], rot_cam[1], intr, target))

detected = np.array([[r["u"] if r["u"] is not None else np.nan,
                      r["v"] if r["v"] is not None else np.nan] for r in records], dtype=float)


def delta(series, idx):
    a = series[0]
    b = series[idx]
    if a is None or b is None:
        return None
    return {"du": float(b[0] - a[0]), "dv": float(b[1] - a[1]),
            "magnitude": float(np.hypot(b[0] - a[0], b[1] - a[1]))}


out = {
    "source": str(Path(args.detail)),
    "heading_convention": "ZYX yaw of the base_link world orientation: "
                          "atan2(2(wz+xy), 1-2(y^2+z^2)); yaw-only pose = Rz(dyaw) about the "
                          "world +Z through the base origin, camera-relative pose frozen at "
                          "hold frame 0",
    "control_dt_s": dt,
    "entry_offset_px": {
        "du": float(detected[0][0] - 320.0), "dv": float(detected[0][1] - 240.0),
        "magnitude": float(np.hypot(detected[0][0] - 320.0, detected[0][1] - 240.0)),
        "note": "already present at the first hold frame; excluded from the drift figures",
    },
    "heading_deg": {
        "per_frame": [float(v) for v in dyaw],
        "first_second": float(dyaw[9] - dyaw[0]),
        "second_second": float(dyaw[19] - dyaw[9]),
        "last_second": float(dyaw[59] - dyaw[49]),
        "whole_window": float(dyaw[59] - dyaw[0]),
        "peak_abs": float(np.max(np.abs(dyaw))),
        "first_step_above_0p1_deg": next((int(i) for i, v in enumerate(np.abs(dyaw)) if v > 0.1), None),
        "first_step_above_0p5_deg": next((int(i) for i, v in enumerate(np.abs(dyaw)) if v > 0.5), None),
    },
    "projections": {},
}
for label, idx in (("1s", 9), ("2s", 19), ("6s", 59)):
    out["projections"][label] = {
        "yaw_only": delta(yaw_only, idx),
        "rotation_only": delta(rot_only, idx),
        "detected": {
            "du": float(detected[idx][0] - detected[0][0]),
            "dv": float(detected[idx][1] - detected[0][1]),
            "magnitude": float(np.hypot(detected[idx][0] - detected[0][0],
                                        detected[idx][1] - detected[0][1])),
        },
    }
    ro = out["projections"][label]["rotation_only"]
    yo = out["projections"][label]["yaw_only"]
    out["projections"][label]["yaw_share_of_rotation"] = (
        yo["magnitude"] / ro["magnitude"] if ro and ro["magnitude"] else None
    )

Path(args.out).write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "heading_deg"}, indent=1)[:4000])
print("heading:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                               for k, v in out["heading_deg"].items() if k != "per_frame"}))
print(f"wrote {args.out}")
