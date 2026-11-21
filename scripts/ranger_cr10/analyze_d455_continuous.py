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

"""Phase timeline, filter checks and per-view verdicts for the continuous D455 flow.

Reads ``continuous_results.json`` and ``continuous_steps.json`` written by
``validate_d455_continuous.py --run-mode full`` and reports:

* one phase timeline (settle / drive / turn / park / observe per view) with step, sim time and
  the events the protocol asks to be distinguished ("zero command issued" vs "park criterion met");
* the deployment-filter checks per phase: does an intentional non-zero command pass the
  deadband (drive, turn), is the chassis output locked while the effective command is zero
  (park, observe), are the six arm actions bit-identical before and after ``filter_action``;
* per view the 1 / 2 / 6 s fixed-prefix statistics against the original criteria, with the
  displacement and chassis-motion baselines labelled relative to the flow start, the parking
  moment and the view's first hold frame.

Usage:
    python scripts/ranger_cr10/analyze_d455_continuous.py --run-dir outputs/stage413/b/d455_cont_full
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

CENTER = (320.0, 240.0)
VISIBLE_MIN_RATIO = 0.90
OFFSET_MAX_PX = 20.0
STD_MAX_PX = 5.0

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--run-dir", default="outputs/stage413/b/d455_cont_full")
ap.add_argument("--out", default="")
args = ap.parse_args()

run_dir = Path(args.run_dir)
results = json.loads((run_dir / "continuous_results.json").read_text())
steps = json.loads((run_dir / "continuous_steps.json").read_text())
dt = 0.1  # 10 Hz control (policy) rate, fixed by the env decimation

# -- phase timeline -------------------------------------------------------------------
timeline = []
for row in steps:
    if not timeline or timeline[-1]["phase"] != row["phase"] or timeline[-1]["sub_phase"] != row["sub_phase"]:
        timeline.append({
            "phase": row["phase"], "sub_phase": row["sub_phase"],
            "start_step": row["step"], "start_sim_time_s": row["sim_time_s"],
            "upper_command_vx_wz": row["upper_command_vx_wz"],
            "effective_command_vx_wz": row["effective_command_vx_wz"],
        })
    timeline[-1]["end_step"] = row["step"]
    timeline[-1]["end_sim_time_s"] = row["sim_time_s"]

park_info = results["plan"]["park_info"]
timeline.append({
    "phase": "event", "sub_phase": "zero_velocity_command_issued",
    "start_step": park_info["zero_command_step"], "end_step": park_info["zero_command_step"],
})
timeline.append({
    "phase": "event", "sub_phase": "park_criterion_met",
    "start_step": park_info["park_met_step"], "end_step": park_info["park_met_step"],
})
timeline.sort(key=lambda entry: (entry.get("start_step") or 0))

# -- filter checks --------------------------------------------------------------------
def phase_rows(phase: str) -> list[dict]:
    return [r for r in steps if r["phase"] == phase]


drive = phase_rows("drive")
turn = phase_rows("turn")
park = phase_rows("park")
observe = phase_rows("observe")


def summarize_command(rows: list[dict]) -> dict:
    if not rows:
        return {}
    upper = np.array([r["upper_command_vx_wz"] for r in rows], dtype=float)
    effective = np.array([r["effective_command_vx_wz"] for r in rows], dtype=float)
    raw = np.array([r["action_base_raw"] for r in rows], dtype=float)
    filt = np.array([r["action_base_filtered"] for r in rows], dtype=float)
    return {
        "steps": len(rows),
        "upper_abs_max": float(np.abs(upper).max()),
        "effective_abs_max": float(np.abs(effective).max()),
        "effective_equals_upper": bool(np.allclose(upper, effective, atol=1e-9)),
        "passed_deadband": bool(np.abs(effective).max() > 0.02),
        "raw_action_abs_max": float(np.abs(raw).max()),
        "filtered_action_abs_max": float(np.abs(filt).max()),
        "output_locked": bool(np.abs(filt).max() < 1e-9),
        "arm_actions_identical_all_steps": all(r["arm_actions_identical"] for r in rows),
        "wheel_target_abs_max": float(np.max(np.abs([r["wheel_target"] for r in rows]))),
        "base_speed_max_mps": float(np.max([r["base_speed_mps"] for r in rows])),
        "yaw_rate_abs_max_dps": float(np.degrees(np.max(np.abs([r["base_ang_vel_w"][2] for r in rows])))),
    }


filter_checks = {
    "drive": summarize_command(drive),
    "turn": summarize_command(turn),
    "park": summarize_command(park),
    "observe": summarize_command(observe),
    "expectation": {
        "drive": "effective command equals the upper 0.15 m/s (above the 0.02 deadband), output NOT locked",
        "turn": "effective wz equals the upper 0.25 rad/s (above the deadband), vx channel locked to 0",
        "park": "effective command identically zero and the chassis output locked to zero",
        "observe": "effective command identically zero and the chassis output locked to zero across view switches",
    },
}

# -- per-view verdicts ----------------------------------------------------------------
flow_start_pos = np.array(results["plan"]["reset_base_pos_w"], dtype=float)
park_pos = np.array(results["plan"]["actual_park_pos_w"], dtype=float)

view_summary = {}
for view, entry in results["views"].items():
    hold = entry["hold_records"]
    out = {
        "arrived": entry["arrived"],
        "arrival_step": entry["arrival_step"],
        "hold_frames": len(hold),
        "target_world": entry["target_world"],
        "arm_command_pos_b": entry["arm_command_pos_b"],
        "entry_offset": None,
        "windows": {},
    }
    if hold:
        du0 = hold[0]["u"] - CENTER[0] if hold[0]["u"] is not None else None
        dv0 = hold[0]["v"] - CENTER[1] if hold[0]["v"] is not None else None
        out["entry_offset"] = {"du": du0, "dv": dv0,
                               "magnitude": None if du0 is None else float(math.hypot(du0, dv0))}
        for seconds in (1.0, 2.0, 6.0):
            n = min(len(hold), int(round(seconds / dt)))
            seg = hold[:n]
            visible = [bool(r["visible"]) for r in seg]
            u = np.array([r["u"] for r in seg if r["u"] is not None], dtype=float)
            v = np.array([r["v"] for r in seg if r["v"] is not None], dtype=float)
            offsets = np.hypot(u - CENTER[0], v - CENTER[1])
            first = np.array(seg[0]["base_pos_w"][:2], dtype=float)
            last = np.array(seg[-1]["base_pos_w"][:2], dtype=float)
            w = {
                "frames": n,
                "visible_frames": int(sum(visible)),
                "visible_ratio": sum(visible) / n,
                "offset_mean_px": float(offsets.mean()) if len(offsets) else None,
                "std_u_px": float(u.std()) if len(u) > 1 else None,
                "std_v_px": float(v.std()) if len(v) > 1 else None,
                "displacement_px": None if len(u) < 2 else {
                    "du": float(u[-1] - u[0]), "dv": float(v[-1] - v[0]),
                    "magnitude": float(math.hypot(u[-1] - u[0], v[-1] - v[0])),
                },
                "ee_pos_err_mean_mm": float(np.mean([r["pos_err_m"] for r in seg]) * 1000.0),
                "contact_body_events": int(sum(r["contact_body_events"] for r in seg)),
                "terminations": int(sum(r["terminated"] for r in seg)),
                "chassis_from_flow_start_mm": float(np.linalg.norm(last - flow_start_pos[:2]) * 1000.0),
                "chassis_from_park_mm": float(np.linalg.norm(last - park_pos[:2]) * 1000.0),
                "chassis_from_first_hold_mm": float(np.linalg.norm(last - first) * 1000.0),
            }
            w["criteria"] = {
                "visible_ok": w["visible_ratio"] >= VISIBLE_MIN_RATIO,
                "offset_ok": w["offset_mean_px"] is not None and w["offset_mean_px"] <= OFFSET_MAX_PX,
                "std_u_ok": w["std_u_px"] is not None and w["std_u_px"] <= STD_MAX_PX,
                "std_v_ok": w["std_v_px"] is not None and w["std_v_px"] <= STD_MAX_PX,
            }
            w["passes"] = all(w["criteria"].values())
            out["windows"][f"{seconds:g}s"] = w
    view_summary[view] = out

expected_frames = {f"{s:g}s": int(round(s / dt)) for s in (1.0, 2.0, 6.0)}
verdicts = {}
for view, s in view_summary.items():
    missing = [w for w in ("1s", "2s", "6s") if w not in s["windows"]]
    short = [w for w in ("1s", "2s", "6s")
             if w in s["windows"] and s["windows"][w]["frames"] < expected_frames[w]]
    verdicts[view] = {
        "present": True, "arrived": s["arrived"], "windows_complete": not missing and not short,
        "missing_windows": missing, "short_windows": short,
        "passes": bool(s["arrived"] and not missing and not short
                       and all(w["passes"] for w in s["windows"].values())),
    }

report = {
    "run_dir": str(run_dir),
    "flow": results["flow"],
    "timeline": timeline,
    "filter_checks": filter_checks,
    "park": {
        "zero_command_step": park_info["zero_command_step"],
        "park_criterion_step": park_info["park_met_step"],
        "settle_time_s": None if park_info["park_met_step"] is None or park_info["zero_command_step"] is None
        else (park_info["park_met_step"] - park_info["zero_command_step"]) * dt,
        "park_error_m": results["plan"]["park_error_m"],
        "park_yaw_error_deg": results["plan"]["park_yaw_error_deg"],
    },
    "views": view_summary,
    "verdicts": verdicts,
    "all_views_pass": all(v["passes"] for v in verdicts.values()) and not results["flow"]["failed"],
}
out_path = Path(args.out) if args.out else run_dir / "continuous_analysis.json"
out_path.write_text(json.dumps(report, indent=1))

print("== timeline ==")
for entry in timeline:
    print(f"  {entry['phase']:8} {entry['sub_phase']:22} steps {entry['start_step']:>4}-{entry['end_step']:<4}")
print("\n== filter checks ==")
for phase in ("drive", "turn", "park", "observe"):
    c = filter_checks[phase]
    if c:
        print(f"  {phase:8} upper_max={c['upper_abs_max']:.3f} eff_max={c['effective_abs_max']:.3f} "
              f"eff==upper={c['effective_equals_upper']} locked={c['output_locked']} "
              f"arms_identical={c['arm_actions_identical_all_steps']} "
              f"speed_max={c['base_speed_max_mps']:.3f} yawrate_max={c['yaw_rate_abs_max_dps']:.2f} deg/s")
print("\n== views ==")
for view, s in view_summary.items():
    if not s["arrived"]:
        print(f"  {view}: NOT ARRIVED")
        continue
    e = s["entry_offset"]["magnitude"]
    print(f"  {view}: arrived step {s['arrival_step']} entry {e:.2f} px")
    for w, x in s["windows"].items():
        print(f"    {w}: vis {x['visible_frames']}/{x['frames']} mean {x['offset_mean_px']:.1f} "
              f"std {x['std_u_px']:.2f}/{x['std_v_px']:.2f} move {x['displacement_px']['magnitude']:.1f} "
              f"ee {x['ee_pos_err_mean_mm']:.1f}mm pass={x['passes']}")
print("\nflow failed:", results["flow"]["failed"], "| all views pass:", report["all_views_pass"])
print(f"wrote {out_path}")
