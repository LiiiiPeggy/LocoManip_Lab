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

"""Deployment zero-velocity filter vs no filter, on the D455 ``top`` view.

Reads the two runs produced by

    validate_d455_views.py --detail-view top --yaw-log [--deploy-filter --marker-world <off.json>]

and answers two separate questions, which the table keeps apart:

1. does the deploy filter reduce the chassis yaw and the image slide?
2. once the slide is reduced, does the target meet the pre-registered centring and stability
   criteria (visible >= 90 %, mean centre offset <= 20 px, std(u) <= 5 px, std(v) <= 5 px)?

The first-frame offset is reported on its own: it is an aiming error already present when the
hold starts, not drift, and the criteria are evaluated both with and without it.

Usage:
    python scripts/ranger_cr10/compare_d455_filter.py \
        --off-dir outputs/stage413/b/d455_cmp_off --on-dir outputs/stage413/b/d455_cmp_on
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--off-dir", default=str(ROOT / "outputs/stage413/b/d455_cmp_off"))
ap.add_argument("--on-dir", default=str(ROOT / "outputs/stage413/b/d455_cmp_on"))
ap.add_argument("--out", default=str(ROOT / "outputs/stage413/b/d455_cmp_off/filter_comparison.json"))
ap.add_argument("--plot", default=str(ROOT / "outputs/stage413/b/d455_cmp_off/filter_comparison.png"))
args = ap.parse_args()

VISIBLE_MIN_RATIO = 0.90
OFFSET_MAX_PX = 20.0
STD_MAX_PX = 5.0
CENTER = (320.0, 240.0)


def load(directory: str) -> dict:
    d = Path(directory)
    results = json.loads((d / "d455_view_results.json").read_text())
    chain = json.loads((d / "d455_yaw_chain_top.json").read_text())
    return {"results": results, "chain": chain}


def summarize(run: dict) -> dict:
    view = run["results"]["views"]["top"]
    hold_records = view["hold_records"]
    chain = [r for r in run["chain"]["records"] if r["phase"] == "hold"]
    dt = run["chain"]["control_dt_s"]
    yaw = np.unwrap(np.radians(np.array([r["base_yaw_deg"] for r in chain], dtype=float)))
    dyaw = np.degrees(yaw - yaw[0])
    pos = np.array([r["base_pos_w"] for r in chain], dtype=float)

    out = {
        "arrived": bool(view["arrived"]),
        "arrival_step": view["arrival_step"],
        "first_hold_step": chain[0]["steps_since_reset"] if chain else None,
        "hold_frames": len(chain),
        "visible_frames": int(sum(1 for r in chain if r["u"] is not None)),
        "marker_registered_world": view.get("marker_registered_world"),
        "initial_state": view.get("initial_state"),
        "entry_offset_px": None,
        "windows": {},
        "lock_evidence": None,
    }
    if chain:
        du0 = chain[0]["u"] - CENTER[0] if chain[0]["u"] is not None else None
        dv0 = chain[0]["v"] - CENTER[1] if chain[0]["v"] is not None else None
        out["entry_offset_px"] = {
            "du": None if du0 is None else float(du0),
            "dv": None if dv0 is None else float(dv0),
            "magnitude": None if du0 is None else float(math.hypot(du0, dv0)),
        }

    for seconds in (1.0, 2.0, 6.0):
        n = min(len(chain), int(round(seconds / dt)))
        seg = chain[:n]
        if not seg:
            continue
        u = np.array([r["u"] for r in seg if r["u"] is not None], dtype=float)
        v = np.array([r["v"] for r in seg if r["v"] is not None], dtype=float)
        offsets = np.hypot(u - CENTER[0], v - CENTER[1])
        entry_offsets = np.hypot(
            u - (CENTER[0] + (du0 or 0.0)), v - (CENTER[1] + (dv0 or 0.0))
        )
        idx = n - 1
        out["windows"][f"{seconds:g}s"] = {
            "frames": n,
            "visible_ratio": len(u) / n,
            "yaw_change_deg": float(dyaw[idx]),
            "translation_mm": float(np.linalg.norm(pos[idx, :2] - pos[0, :2]) * 1000.0),
            "du_px": float(u[-1] - u[0]) if len(u) > 1 else None,
            "dv_px": float(v[-1] - v[0]) if len(v) > 1 else None,
            "offset_mean_px": float(offsets.mean()) if len(offsets) else None,
            "std_u_px": float(u.std()) if len(u) > 1 else None,
            "std_v_px": float(v.std()) if len(v) > 1 else None,
            "offset_mean_about_entry_px": float(entry_offsets.mean()) if len(entry_offsets) else None,
            "ee_pos_err_mean_mm": float(np.mean([r["ee_pos_err_m"] for r in seg]) * 1000.0),
            "wz_cmd_abs_max": float(np.max(np.abs([r["term_command_after_spin"][1] for r in seg]))),
            "yaw_rate_abs_max_dps": float(np.max(np.abs([r["base_yaw_rate_dps"] for r in seg]))),
        }
        w = out["windows"][f"{seconds:g}s"]
        w["criteria"] = {
            "visible_ok": w["visible_ratio"] >= VISIBLE_MIN_RATIO,
            "offset_ok": w["offset_mean_px"] is not None and w["offset_mean_px"] <= OFFSET_MAX_PX,
            "std_u_ok": w["std_u_px"] is not None and w["std_u_px"] <= STD_MAX_PX,
            "std_v_ok": w["std_v_px"] is not None and w["std_v_px"] <= STD_MAX_PX,
        }
        w["passes"] = all(w["criteria"].values())

    raw = np.array([r["action_base_raw"] for r in chain if r["action_base_raw"]], dtype=float)
    filt = np.array([r["action_base_filtered"] for r in chain if r["action_base_filtered"]], dtype=float)
    arms_same = [r["arm_actions_identical_before_after_filter"] for r in chain
                 if r["arm_actions_identical_before_after_filter"] is not None]
    out["lock_evidence"] = {
        "filter_enabled": bool(run["chain"].get("records", [{}])[0].get("filter_enabled", False)),
        "upper_cmd_max_abs": float(np.max(np.abs([r["upper_cmd_vx_wz"] for r in chain]))),
        "raw_base_action_max_abs": float(np.max(np.abs(raw))) if raw.size else None,
        "filtered_base_action_max_abs": float(np.max(np.abs(filt))) if filt.size else None,
        "steps_with_wz_request_before_filter": int(sum(1 for r in chain
                                                      if abs(r["action_base_raw"][1]) > 1e-6)),
        "steps_with_wz_request_after_filter": int(sum(1 for r in chain
                                                     if abs(r["action_base_filtered"][1]) > 1e-6)),
        "arm_actions_identical_all_steps": bool(arms_same and all(arms_same)),
        "wheel_target_abs_max": float(np.max(np.abs([r["wheel_target"] for r in chain]))),
        "wheel_actual_abs_max": float(np.max(np.abs([r["wheel_actual"] for r in chain]))),
    }
    return out


off, on = load(args.off_dir), load(args.on_dir)
result = {
    "off_dir": args.off_dir,
    "on_dir": args.on_dir,
    "protocol": {
        "env": "RANGER-CR10-TELEOP-SIMPLECOLL (v2)",
        "checkpoint": off["results"]["checkpoint"],
        "inference": off["results"]["inference"],
        "seed": off["results"]["seed"],
        "view_order": "front -> side -> top (detail only on top)",
        "target": "A1 world-fixed; the ON group reuses the OFF group's registered world pose",
        "control_dt_s": off["chain"]["control_dt_s"],
        "criteria": {"visible_ratio_min": VISIBLE_MIN_RATIO, "offset_mean_max_px": OFFSET_MAX_PX,
                     "std_max_px": STD_MAX_PX},
    },
    "same_target": {
        "off": off["results"]["views"]["top"].get("marker_registered_world"),
        "on": on["results"]["views"]["top"].get("marker_registered_world"),
        "off_initial_state": off["results"]["views"]["top"].get("initial_state"),
        "on_initial_state": on["results"]["views"]["top"].get("initial_state"),
    },
    "filter_config_off": off["results"]["protocol"].get("deploy_filter"),
    "filter_config_on": on["results"]["protocol"].get("deploy_filter"),
    "off": summarize(off),
    "on": summarize(on),
}
result["same_target"]["initial_state_identical"] = (
    result["same_target"]["off_initial_state"] is not None
    and result["same_target"]["on_initial_state"] is not None
    and result["same_target"]["off_initial_state"]["joint_pos_digest"]
    == result["same_target"]["on_initial_state"]["joint_pos_digest"]
    and np.allclose(result["same_target"]["off_initial_state"]["base_pos_w"],
                    result["same_target"]["on_initial_state"]["base_pos_w"], atol=1e-9)
)
result["same_target"]["identical"] = (
    result["same_target"]["off"] is not None
    and result["same_target"]["on"] is not None
    and np.allclose(result["same_target"]["off"], result["same_target"]["on"], atol=1e-9)
)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(result, indent=1))

# -- same-timeline curve: turn request, measured yaw rate, image slide --------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    for label, run, style in (("filter OFF", off, "-"), ("filter ON", on, "--")):
        chain = [r for r in run["chain"]["records"] if r["phase"] == "hold"]
        if not chain:
            continue
        t = (np.array([r["steps_since_reset"] for r in chain]) - chain[0]["steps_since_reset"]) * \
            run["chain"]["control_dt_s"]
        axes[0].plot(t, [r["term_command_after_spin"][1] for r in chain], style,
                     label=f"{label}: wz into base controller [rad/s]")
        axes[0].plot(t, [r["action_base_raw"][1] for r in chain], style, alpha=0.35,
                     label=f"{label}: raw policy wz action")
        axes[1].plot(t, [r["base_yaw_rate_dps"] for r in chain], style,
                     label=f"{label}: measured yaw rate [deg/s]")
        u0 = chain[0]["u"]
        axes[2].plot(t, [None if r["u"] is None else r["u"] - u0 for r in chain], style,
                     label=f"{label}: image du [px]")
    axes[0].set_ylabel("wz [rad/s]")
    axes[1].set_ylabel("yaw rate [deg/s]")
    axes[2].set_ylabel("du from first hold frame [px]")
    axes[2].set_xlabel("time from first hold frame [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("D455 top view: deployment zero-velocity filter vs unfiltered")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=130)
    result["plot"] = args.plot
    Path(args.out).write_text(json.dumps(result, indent=1))
except Exception as error:  # noqa: BLE001
    result["plot_error"] = f"{type(error).__name__}: {error}"
    Path(args.out).write_text(json.dumps(result, indent=1))

print(json.dumps({k: v for k, v in result.items() if k not in ("off", "on")}, indent=1)[:3000])
for tag in ("off", "on"):
    s = result[tag]
    print(f"\n== {tag}: arrived={s['arrived']} arrival_step={s['arrival_step']} "
          f"visible={s['visible_frames']}/{s['hold_frames']} entry={s['entry_offset_px']}")
    for w, v in s["windows"].items():
        print(f"  {w}: yaw={v['yaw_change_deg']:+.3f} deg  trans={v['translation_mm']:.1f} mm  "
              f"du/dv={v['du_px']:+.1f}/{v['dv_px']:+.1f}  mean={v['offset_mean_px']:.1f}  "
              f"std={v['std_u_px']:.2f}/{v['std_v_px']:.2f}  ee={v['ee_pos_err_mean_mm']:.1f} mm  "
              f"pass={v['passes']}")
    print(f"  lock: {s['lock_evidence']}")
print(f"\nwrote {args.out}")
