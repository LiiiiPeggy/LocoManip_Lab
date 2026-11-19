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

"""Short-window (1 s / 2 s) re-reading of the existing D455 hold records.

No simulation and no new rollout: this reads the hold records already written by
``validate_d455_views.py`` and re-slices them into **fixed prefixes** of the hold window.

Two protocols are read:

* A1 (``--marker-mode world``, primary): the target is fixed in the world from the first warm
  step, so the pixel offset includes everything the chassis did between marker placement and the
  hold.
* A2 (``--marker-mode hold``, diagnostic): the target is re-registered against the chassis at the
  start of the hold, so the offset isolates aim plus drift during the hold.

Rules kept from the pre-registered protocol, so the short windows are not cherry-picked:

* the slice is always the **prefix** ``hold_index 0 .. N-1``, never the best N frames;
* frames where the target was not detected keep their slot in the denominator (the visible ratio
  is over all frames in the slice), and are excluded only from the centroid statistics, where the
  detected count is reported next to the value;
* the thresholds are the ones fixed before the run: visible ratio >= 0.90, mean centre offset
  <= 20 px, and both coordinate standard deviations <= 5 px.

The control period is 0.1 s (10 Hz policy on 200 Hz physics, ``decimation = 20``); the result JSON
does not record it, so it is passed in and echoed in the output.

Usage:
    python scripts/ranger_cr10/analyze_d455_windows.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--a1", default=str(ROOT / "outputs/stage413/b/d455_views/d455_view_results.json"))
ap.add_argument("--a2", default=str(ROOT / "outputs/stage413/b/d455_views_hold/d455_view_results.json"))
ap.add_argument("--control-dt", type=float, default=0.1)
ap.add_argument("--windows", default="1,2,6")
ap.add_argument("--dist-m", type=float, default=0.35)
ap.add_argument("--f-px", type=float, default=372.0)
ap.add_argument("--out", default=str(ROOT / "outputs/stage413/b/d455_views/d455_short_windows.json"))
args = ap.parse_args()

VISIBLE_MIN_RATIO = 0.90
OFFSET_MAX_PX = 20.0
STD_MAX_PX = 5.0
WINDOWS = [float(value) for value in args.windows.split(",")]
VIEWS = ("front", "side", "top")


def slice_prefix(records: list[dict], n: int) -> list[dict]:
    return records[:n]


def window_stats(records: list[dict], dt: float) -> dict:
    n_total = len(records)
    visible = [bool(r["visible"]) for r in records]
    detected = [r for r in records if r.get("u") is not None]
    us = np.array([r["u"] for r in detected], dtype=float)
    vs = np.array([r["v"] for r in detected], dtype=float)
    offsets = np.hypot(
        np.array([r["du"] for r in detected], dtype=float),
        np.array([r["dv"] for r in detected], dtype=float),
    )
    first, last = detected[0] if detected else None, detected[-1] if detected else None
    duration = (n_total - 1) * dt if n_total > 1 else 0.0
    displacement = (
        float(math.hypot(last["du"] - first["du"], last["dv"] - first["dv"]))
        if first is not None and last is not None else None
    )
    base = np.array([r["base_drift_xy_m"] for r in records], dtype=float) if records else np.zeros(0)
    yaw = None
    if records and "base_yaw_deg" in records[0]:
        yaw_values = np.unwrap(np.radians(np.array([r["base_yaw_deg"] for r in records], dtype=float)))
        yaw = float(np.degrees(yaw_values[-1] - yaw_values[0]))
    stats = {
        "frames": n_total,
        "visible_frames": int(sum(visible)),
        "visible_ratio": float(sum(visible)) / n_total if n_total else None,
        "detected_frames": len(detected),
        "offset_mean_px": float(offsets.mean()) if len(offsets) else None,
        "offset_p95_px": float(np.percentile(offsets, 95)) if len(offsets) else None,
        "std_u_px": float(us.std()) if len(us) > 1 else None,
        "std_v_px": float(vs.std()) if len(vs) > 1 else None,
        "first_du_px": float(first["du"]) if first is not None else None,
        "first_dv_px": float(first["dv"]) if first is not None else None,
        "last_du_px": float(last["du"]) if last is not None else None,
        "last_dv_px": float(last["dv"]) if last is not None else None,
        "displacement_2d_px": displacement,
        "duration_s": duration,
        "displacement_px_per_s": (displacement / duration) if displacement is not None and duration > 0 else None,
        "base_drift_xy_max_mm": float(base.max()) * 1000.0 if base.size else None,
        "base_yaw_drift_deg": yaw,
    }
    stats["checks"] = {
        "visible_ratio_ok": stats["visible_ratio"] is not None and stats["visible_ratio"] >= VISIBLE_MIN_RATIO,
        "offset_ok": stats["offset_mean_px"] is not None and stats["offset_mean_px"] <= OFFSET_MAX_PX,
        "std_u_ok": stats["std_u_px"] is not None and stats["std_u_px"] <= STD_MAX_PX,
        "std_v_ok": stats["std_v_px"] is not None and stats["std_v_px"] <= STD_MAX_PX,
    }
    stats["all_ok"] = all(stats["checks"].values())
    # Entry state vs sliding: an entry offset above the threshold is "already off centre"; an
    # entry inside it with a large displacement is "aligned, then slid away".
    entry = float(math.hypot(stats["first_du_px"], stats["first_dv_px"])) if first is not None else None
    stats["entry_offset_px"] = entry
    if entry is None:
        stats["pattern"] = "no detection"
    elif entry > OFFSET_MAX_PX:
        stats["pattern"] = "already off centre at entry"
    elif displacement is not None and displacement > OFFSET_MAX_PX:
        stats["pattern"] = "aligned at entry, then slid"
    else:
        stats["pattern"] = "aligned and stayed"
    return stats


def protocol_report(data: dict, label: str) -> dict:
    out = {"label": label, "protocol": data["protocol"], "views": {}}
    for view in VIEWS:
        if view not in data["views"]:
            continue
        records = data["views"][view]["hold_records"]
        entry = {"stored": {
            "frames": data["views"][view]["hold_frames"],
            "visible_frames": data["views"][view]["visible_frames"],
            "visible_ratio": data["views"][view]["visible_ratio"],
            "offset_mean_px": data["views"][view]["pixel_offset_mean_px"],
            "std_u_px": data["views"][view]["centroid_std_px"]["u"],
            "std_v_px": data["views"][view]["centroid_std_px"]["v"],
        }, "windows": {}}
        for seconds in WINDOWS:
            n = min(len(records), int(round(seconds / args.control_dt)))
            entry["windows"][f"{seconds:g}s"] = window_stats(slice_prefix(records, n), args.control_dt)
        # Consistency check of the full window against what the rollout wrote.
        full = entry["windows"][f"{max(WINDOWS):g}s"]
        stored = entry["stored"]
        entry["consistency_with_stored_full_window"] = {
            "visible_frames_delta": None if stored["visible_frames"] is None else
            full["visible_frames"] - stored["visible_frames"],
            "offset_mean_delta_px": None if stored["offset_mean_px"] is None else
            full["offset_mean_px"] - stored["offset_mean_px"],
            "std_u_delta_px": None if stored["std_u_px"] is None else full["std_u_px"] - stored["std_u_px"],
            "std_v_delta_px": None if stored["std_v_px"] is None else full["std_v_px"] - stored["std_v_px"],
        }
        out["views"][view] = entry
    return out


a1 = protocol_report(json.loads(Path(args.a1).read_text()), "A1 world-fixed target (primary)")
a2 = protocol_report(json.loads(Path(args.a2).read_text()), "A2 target re-registered at hold start (diagnostic)")

px_per_mm = args.f_px / (args.dist_m * 1000.0)
px_per_deg = args.f_px * math.tan(math.radians(1.0))
result = {
    "source": {"a1": str(args.a1), "a2": str(args.a2)},
    "control_dt_s": args.control_dt,
    "windows_s": WINDOWS,
    "thresholds": {"visible_ratio_min": VISIBLE_MIN_RATIO, "offset_mean_max_px": OFFSET_MAX_PX,
                   "std_max_px": STD_MAX_PX},
    "slicing": "fixed prefix of the hold records, hold_index 0..N-1; undetected frames keep their "
               "slot in the visible-ratio denominator and are excluded only from centroid stats",
    "estimate_conversions": {
        "px_per_mm_at_0.35m": px_per_mm,
        "px_per_deg": px_per_deg,
        "note": "order-of-magnitude only: base translation and yaw are converted with the pinhole "
                "scale at 0.35 m. The records do not carry a per-frame base pose or camera pose, "
                "so they cannot apportion the drift between chassis and arm.",
    },
    "A1": a1,
    "A2": a2,
}
Path(args.out).write_text(json.dumps(result, indent=1))

header = f"{'proto':4} {'view':6} {'win':>4} {'frames':>6} {'vis':>7} {'ratio':>6} {'mean':>7} {'std_u':>6} {'std_v':>6} {'ok':>5} {'pattern':>32}"
print(header)
for tag, report in (("A1", a1), ("A2", a2)):
    for view, entry in report["views"].items():
        for win, s in entry["windows"].items():
            print(
                f"{tag:4} {view:6} {win:>4} {s['frames']:6d} {s['visible_frames']:3d}/{s['frames']:<3d} "
                f"{s['visible_ratio']:6.2f} {s['offset_mean_px']:7.1f} {s['std_u_px']:6.2f} {s['std_v_px']:6.2f} "
                f"{str(s['all_ok']):>5} {s['pattern']:>32}"
            )
print()
for tag, report in (("A1", a1), ("A2", a2)):
    for view, entry in report["views"].items():
        for win, s in entry["windows"].items():
            yaw_text = "N/A" if s["base_yaw_drift_deg"] is None else f"{s['base_yaw_drift_deg']:+.2f} deg"
            rate = 0.0 if s["displacement_px_per_s"] is None else s["displacement_px_per_s"]
            print(
                f"{tag} {view:6} {win:>4} first=({s['first_du_px']:+7.1f},{s['first_dv_px']:+7.1f}) "
                f"last=({s['last_du_px']:+7.1f},{s['last_dv_px']:+7.1f}) "
                f"move={s['displacement_2d_px']:6.1f} {rate:5.1f} px/s  "
                f"base={s['base_drift_xy_max_mm']:5.1f} mm  yaw={yaw_text}"
            )
print("\nwrote", args.out)
