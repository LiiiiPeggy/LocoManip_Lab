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

"""Reproduce the A1 ``top`` pixel motion from the measured chassis and arm motion.

The detailed trajectory (``d455_detail_top.json``) records, for every hold frame, the chassis
world pose, the D455 optical-frame world pose and the fixed target's world pose. That is enough
to recompose the camera pose from two independent parts and ask four questions:

    fixed          B(0) with C(0)     -- the target would not move at all
    base only      B(t) with C(0)     -- only the chassis moved
    camera only    B(0) with C(t)     -- only the arm moved (camera relative to chassis)
    both           B(t) with C(t)     -- the measured camera pose

where ``C(t) = B(t)^-1 * camera(t)``. The interaction term is
``both - base - camera + fixed``; the image residual is ``detected - both``.

The target is a 6 cm cube, so the projection of its 3D centre and the centroid of its visible
silhouette differ by a constant offset. That offset is removed by working with displacements from
the first hold frame, and it is reported separately as the entry offset.

Usage:
    python scripts/ranger_cr10/analyze_d455_top_motion.py
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
ap.add_argument("--out", default=str(ROOT / "outputs/stage413/b/d455_views_detail/top_motion_decomposition.json"))
ap.add_argument("--windows", default="1,2,6")
args = ap.parse_args()

THRESHOLD_PX = 20.0


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


def compose(p1, q1, p2, q2):
    return np.asarray(p1) + q_rot(np.asarray(q1), np.asarray(p2)), q_mul(np.asarray(q1), np.asarray(q2))


def invert(p, q):
    qc = q_conj(np.asarray(q))
    return -q_rot(qc, np.asarray(p)), qc


def project(cam_pos, cam_quat, intr, point_w):
    local = q_rot(q_conj(np.asarray(cam_quat)), np.asarray(point_w) - np.asarray(cam_pos))
    if local[2] <= 1e-6:
        return None
    return np.array([
        intr[0][0] * local[0] / local[2] + intr[0][2],
        intr[1][1] * local[1] / local[2] + intr[1][2],
    ]), float(local[2])


data = json.loads(Path(args.detail).read_text())
if data.get("marker_mode") != "world":
    raise SystemExit(f"expected marker_mode='world', got {data.get('marker_mode')!r}")

records = data["records"]
intr = data["intrinsics"]
target = np.array(records[0]["target_pos_w"])
dt = data["control_dt_s"]

base = [(np.array(r["base_pos_w"]), np.array(r["base_quat_w"])) for r in records]
cam_w = [(np.array(r["cam_pos_w"]), np.array(r["cam_quat_w_ros"])) for r in records]
# Camera pose relative to the chassis, recomposed from the two world poses.
cam_b = []
for (bp, bq), (cp, cq) in zip(base, cam_w):
    inv_p, inv_q = invert(bp, bq)
    cam_b.append(compose(inv_p, inv_q, cp, cq))

b0, c0 = base[0], cam_b[0]
fixed_cam = compose(b0[0], b0[1], c0[0], c0[1])

cases = {"fixed": [], "base_only": [], "camera_only": [], "both": []}
cases["base_translation_only"] = []
cases["base_rotation_only"] = []
for i, ((bp, bq), (cpq, cqq)) in enumerate(zip(base, cam_b)):
    base_cam = compose(bp, bq, c0[0], c0[1])          # chassis moved, arm frozen at frame 0
    cam_only = compose(b0[0], b0[1], cpq, cqq)        # chassis frozen, arm moved
    # Split the chassis motion into its translation and rotation halves.
    trans_cam = compose(bp, b0[1], c0[0], c0[1])
    rot_cam = compose(b0[0], bq, c0[0], c0[1])
    cases["fixed"].append(project(fixed_cam[0], fixed_cam[1], intr, target))
    cases["base_only"].append(project(base_cam[0], base_cam[1], intr, target))
    cases["camera_only"].append(project(cam_only[0], cam_only[1], intr, target))
    cases["both"].append(project(cam_w[i][0], cam_w[i][1], intr, target))
    cases["base_translation_only"].append(project(trans_cam[0], trans_cam[1], intr, target))
    cases["base_rotation_only"].append(project(rot_cam[0], rot_cam[1], intr, target))

uv = {name: np.array([item[0] if item else [np.nan, np.nan] for item in items], dtype=float)
      for name, items in cases.items()}
detected = np.array([[r["u"] if r["u"] is not None else np.nan,
                      r["v"] if r["v"] is not None else np.nan] for r in records], dtype=float)

# Constant silhouette-vs-centre offset, measured at the first hold frame.
bias = detected[0] - uv["both"][0]

result = {
    "source": str(Path(args.detail)),
    "view": data["view"],
    "arrival_step": data.get("arrival_step"),
    "first_hold_frame_step": data.get("first_hold_frame_step"),
    "control_dt_s": dt,
    "frames": len(records),
    "target_world_fixed": target.tolist(),
    "camera_rel_base_frame0": {"pos": c0[0].tolist(), "quat_wxyz": c0[1].tolist()},
    "entry_offset_px": {
        "du": float(detected[0][0] - 320.0),
        "dv": float(detected[0][1] - 240.0),
        "magnitude": float(np.linalg.norm(detected[0] - np.array([320.0, 240.0]))),
        "note": "error already present at the first hold frame; not counted as hold drift",
    },
    "silhouette_vs_centre_bias_px": {"du": float(bias[0]), "dv": float(bias[1]),
                                     "magnitude": float(np.linalg.norm(bias)),
                                     "note": "detected silhouette centroid minus projected 3D cube "
                                             "centre at the first hold frame"},
    "windows": {},
    "per_frame": [],
}

for seconds in [float(x) for x in args.windows.split(",")]:
    n = min(len(records), int(round(seconds / dt)))
    idx = n - 1
    entry = {}
    for name in ("fixed", "base_only", "base_translation_only", "base_rotation_only",
                 "camera_only", "both"):
        d = uv[name][idx] - uv[name][0]
        entry[name] = {"du": float(d[0]), "dv": float(d[1]), "magnitude": float(np.linalg.norm(d))}
    # Which chassis degree of freedom moved: translation and rotation, in metres and degrees.
    dp = base[idx][0] - base[0][0]
    # Relative rotation B(0)^-1 * B(t), reported as yaw / pitch / roll.
    rel = q_mul(q_conj(base[0][1]), base[idx][1])
    w, x, y, z = rel
    yaw = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))
    roll = math.degrees(math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    entry["base_motion"] = {
        "dxyz_mm": [float(v * 1000.0) for v in dp],
        "dtranslation_norm_mm": float(np.linalg.norm(dp) * 1000.0),
        "dyaw_deg": float(yaw), "dpitch_deg": float(pitch), "droll_deg": float(roll),
    }
    interaction = uv["both"][idx] - uv["base_only"][idx] - uv["camera_only"][idx] + uv["fixed"][idx]
    entry["interaction"] = {"du": float(interaction[0]), "dv": float(interaction[1]),
                            "magnitude": float(np.linalg.norm(interaction))}
    det_d = detected[idx] - detected[0]
    entry["detected"] = {"du": float(det_d[0]), "dv": float(det_d[1]),
                         "magnitude": float(np.linalg.norm(det_d))}
    resid = det_d - (uv["both"][idx] - uv["both"][0])
    entry["residual_detected_minus_both"] = {"du": float(resid[0]), "dv": float(resid[1]),
                                             "magnitude": float(np.linalg.norm(resid))}
    entry["base_share_of_motion"] = {
        "of_both_px": float(np.linalg.norm(uv["base_only"][idx] - uv["base_only"][0])),
        "of_camera_px": float(np.linalg.norm(uv["camera_only"][idx] - uv["camera_only"][0])),
        "note": "norms of the two single-cause motions over the same window; they add vectorially "
                "only up to the interaction term",
    }
    entry["window_s"] = seconds
    entry["frames"] = n
    entry["above_threshold"] = bool(np.linalg.norm(det_d) > THRESHOLD_PX)
    result["windows"][f"{seconds:g}s"] = entry

for i, r in enumerate(records):
    row = {
        "hold_index": r["hold_index"],
        "steps_since_reset": r["steps_since_reset"],
        "sim_time_s": r["sim_time_s"],
        "detected_du": None if r["du"] is None else float(r["du"]),
        "detected_dv": None if r["dv"] is None else float(r["dv"]),
        "both_du": float(uv["both"][i][0] - 320.0),
        "both_dv": float(uv["both"][i][1] - 240.0),
        "base_only_du": float(uv["base_only"][i][0] - 320.0),
        "base_only_dv": float(uv["base_only"][i][1] - 240.0),
        "camera_only_du": float(uv["camera_only"][i][0] - 320.0),
        "camera_only_dv": float(uv["camera_only"][i][1] - 240.0),
        "base_pos_w": r["base_pos_w"],
        "cam_pos_w": r["cam_pos_w"],
    }
    if r["u"] is not None:
        row["residual_du"] = float((r["u"] - detected[0][0]) - (uv["both"][i][0] - uv["both"][0][0]))
        row["residual_dv"] = float((r["v"] - detected[0][1]) - (uv["both"][i][1] - uv["both"][0][1]))
    result["per_frame"].append(row)

residuals = np.array([[row.get("residual_du", np.nan), row.get("residual_dv", np.nan)]
                      for row in result["per_frame"]], dtype=float)
valid = ~np.isnan(residuals).any(axis=1)
result["frame_sync_check"] = {
    "frames_compared": int(valid.sum()),
    "residual_rms_px": float(np.sqrt(np.nanmean(np.sum(residuals[valid] ** 2, axis=1)))),
    "residual_max_px": float(np.nanmax(np.linalg.norm(residuals[valid], axis=1))),
    "note": "detected-minus-projected displacement, both measured from the first hold frame, so the "
            "constant silhouette bias cancels",
}

Path(args.out).write_text(json.dumps(result, indent=1))
print(json.dumps({k: v for k, v in result.items() if k not in ("per_frame",)}, indent=1)[:8000])
print(f"wrote {args.out}")
