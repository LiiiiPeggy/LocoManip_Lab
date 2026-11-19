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

"""Split the D455 centroid offset into a steady offset, a drift and a residual jitter.

The rollout records one centroid per hold frame. A single standard deviation cannot distinguish

* "steadily aimed a bit off"  -> constant offset, small residual
* "the view slowly slides"    -> linear trend, large total movement, small residual
* "the image keeps shaking"   -> small trend, large residual

so each centroid track is fitted with ``u(t) = a_u + b_u t`` (same for ``v``) and reported as
mean offset, total drift over the hold window, and detrended residual std.

Reads the existing ``d455_view_results.json``; no re-run, no training.

Usage:
    python scripts/ranger_cr10/analyze_d455_views.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--results", default=str(ROOT / "outputs/stage413/b/d455_views/d455_view_results.json"))
ap.add_argument("--out", default=str(ROOT / "outputs/stage413/b/d455_views/d455_offset_decomposition.json"))
args = ap.parse_args()

results = json.loads(Path(args.results).read_text())
centroid_u, centroid_v = results["camera"]["image_wh"]
centroid_u, centroid_v = centroid_u / 2.0, centroid_v / 2.0

summary = {"center_px": [centroid_u, centroid_v], "views": {}}
for name, view in results["views"].items():
    records = view["hold_records"]
    us = np.array([r["u"] for r in records if r["u"] is not None], dtype=float)
    vs = np.array([r["v"] for r in records if r["v"] is not None], dtype=float)
    if len(us) < 3:
        summary["views"][name] = {"frames": int(len(us)), "note": "fewer than 3 detected frames"}
        continue
    t = np.arange(len(us), dtype=float) * results["protocol"]["hold_steps"] / max(len(us), 1)
    du, dv = us - centroid_u, vs - centroid_v
    offset = np.hypot(du, dv)

    slope_u, icept_u = np.polyfit(t, du, 1)
    slope_v, icept_v = np.polyfit(t, dv, 1)
    resid_u = du - (icept_u + slope_u * t)
    resid_v = dv - (icept_v + slope_v * t)
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0

    drift_u = slope_u * duration
    drift_v = slope_v * duration
    base_drift = np.array([r["base_drift_xy_m"] for r in records], dtype=float)
    # Only A2-era records carry base_yaw_deg. Do not default a missing field to 0.0: a silent zero
    # reads as a measured "no yaw drift", which is exactly the opposite of the truth for A1.
    if records and all("base_yaw_deg" in r for r in records):
        yaw = np.unwrap(np.radians(np.array([r["base_yaw_deg"] for r in records], dtype=float)))
        yaw_drift_deg = float(np.degrees(yaw[-1] - yaw[0]))
    else:
        yaw_drift_deg = None

    summary["views"][name] = {
        "frames": int(len(us)),
        "offset_mean_px": float(offset.mean()),
        "offset_p95_px": float(np.percentile(offset, 95)),
        "offset_at_first_frame_px": float(offset[0]),
        "offset_at_last_frame_px": float(offset[-1]),
        "displacement_2d_px": float(math.hypot(du[-1] - du[0], dv[-1] - dv[0])),
        "steady_offset_px": float(math.hypot(icept_u, icept_v)),
        "drift_total_px": float(math.hypot(drift_u, drift_v)),
        "drift_px_per_s": float(math.hypot(slope_u, slope_v)),
        "residual_std_px": {"u": float(resid_u.std()), "v": float(resid_v.std()),
                            "radial": float(np.hypot(resid_u, resid_v).std())},
        "raw_std_px": {"u": float(du.std()), "v": float(dv.std())},
        "base_drift_xy_m": {"mean": float(base_drift.mean()), "max": float(base_drift.max())},
        "base_yaw_drift_deg": yaw_drift_deg,
        "predicted_px_from_base_motion": {
            "from_translation": float(base_drift.max()) * 372.0 / 0.35,
            "from_yaw": None if yaw_drift_deg is None else abs(yaw_drift_deg) * 372.0 * math.tan(math.radians(1.0)),
            "yaw_available": yaw_drift_deg is not None,
        },
        "dominant_term": None,
        "note": "steady = offset of the fitted line at t=0; drift = fitted line movement over the "
                "hold window; residual = deviation from that line (what visually reads as shaking)",
    }
    entry = summary["views"][name]
    terms = {"steady": entry["steady_offset_px"], "drift": entry["drift_total_px"],
             "jitter": entry["residual_std_px"]["radial"]}
    entry["dominant_term"] = max(terms, key=terms.get)
    entry["terms_px"] = terms

Path(args.out).write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
print(f"wrote {args.out}")
