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

"""Where does the ``top``-view yaw come from: the policy's turn command, or the chassis itself?

Reads the command-to-motion chain recorded by
``validate_d455_views.py --detail-view top --yaw-log`` and answers, in order of causal depth:

1. upper velocity command (what the operator asked for),
2. raw policy action for the base channel,
3. the action term's post-tanh command ``(vx, wz)`` and the post-branch values actually sent,
4. steering and wheel targets, and the measured wheel speeds,
5. measured base yaw and yaw rate,
6. the target's image centroid.

It reports which link moves first, the yaw rate already present at the arrival step and at the
first hold frame, and - if an intervention run is supplied - the comparison under
"wz zeroed from the first hold step".

Usage:
    python scripts/ranger_cr10/analyze_d455_yaw_chain.py \
        --baseline outputs/stage413/b/d455_yaw_chain/d455_yaw_chain_top.json \
        [--intervention outputs/stage413/b/d455_yaw_zero/d455_yaw_chain_top.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--baseline", required=True)
ap.add_argument("--intervention", default="")
ap.add_argument("--out", default=str(ROOT / "outputs/stage413/b/d455_yaw_chain/yaw_chain_analysis.json"))
ap.add_argument("--plot", default=str(ROOT / "outputs/stage413/b/d455_yaw_chain/yaw_chain.png"))
args = ap.parse_args()

WZ_EPS = 0.005          # rad/s, "a turn request is present"
RATE_EPS = 0.5          # deg/s, "the chassis is actually turning"
WHEEL_EPS = 0.01        # rad/s of wheel spin difference from straight running


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def rows(data: dict) -> list[dict]:
    return data["records"]


def first_step(records: list[dict], predicate) -> int | None:
    for row in records:
        if predicate(row):
            return int(row["steps_since_reset"])
    return None


def yaw_at(records: list[dict], phase: str) -> dict | None:
    for row in records:
        if row["phase"] == phase:
            return row
    return None


def summarize(data: dict) -> dict:
    records = rows(data)
    hold = [r for r in records if r["phase"] == "hold"]
    arrival = [r for r in records if r["phase"] == "arrival"]
    yaw0 = hold[0]["base_yaw_deg"] if hold else None
    yaw = np.array([r["base_yaw_deg"] for r in hold], dtype=float)
    yaw = np.unwrap(np.radians(yaw))
    dyaw = np.degrees(yaw - yaw[0]) if len(yaw) else np.zeros(0)

    def window(seconds: float):
        n = min(len(hold), int(round(seconds / data["control_dt_s"])))
        idx = n - 1
        seg = hold[:n]
        u = np.array([r["u"] for r in seg if r["u"] is not None], dtype=float)
        v = np.array([r["v"] for r in seg if r["v"] is not None], dtype=float)
        offsets = np.hypot(u - 320.0, v - 240.0)
        return {
            "frames": n,
            "yaw_change_deg": float(dyaw[idx]),
            "du_px": float(u[-1] - u[0]) if len(u) > 1 else None,
            "dv_px": float(v[-1] - v[0]) if len(v) > 1 else None,
            "offset_mean_px": float(offsets.mean()) if len(offsets) else None,
            "std_u_px": float(u.std()) if len(u) > 1 else None,
            "std_v_px": float(v.std()) if len(v) > 1 else None,
            "ee_pos_err_mean_mm": float(np.mean([r["ee_pos_err_m"] for r in seg])) * 1000.0,
            "wz_cmd_abs_max": float(np.max(np.abs([r["term_command_after_spin"][1] for r in seg]))),
            "yaw_rate_abs_max_dps": float(np.max(np.abs([r["base_yaw_rate_dps"] for r in seg]))),
        }

    return {
        "arrival_step": hold[0]["steps_since_reset"] - 1 if hold else None,
        "first_hold_step": hold[0]["steps_since_reset"] if hold else None,
        "wz_cmd_at_arrival": arrival[-1]["term_command_after_spin"][1] if arrival else None,
        "yaw_rate_at_arrival_dps": arrival[-1]["base_yaw_rate_dps"] if arrival else None,
        "wz_cmd_at_first_hold": hold[0]["term_command_after_spin"][1] if hold else None,
        "yaw_rate_at_first_hold_dps": hold[0]["base_yaw_rate_dps"] if hold else None,
        "onset": {
            "first_wz_cmd_above_eps": first_step(records, lambda r: abs(r["term_command_after_spin"][1]) > WZ_EPS),
            "first_yaw_rate_above_eps": first_step(records, lambda r: abs(r["base_yaw_rate_dps"]) > RATE_EPS),
            "first_steer_target_above_eps": first_step(
                records, lambda r: max(abs(s) for s in r["steering_target"]) > 0.01
            ),
            "first_wheel_spread_above_eps": first_step(
                records,
                lambda r: (max(r["wheel_actual"]) - min(r["wheel_actual"])) > WHEEL_EPS,
            ),
            "eps": {"wz_rad_s": WZ_EPS, "yaw_rate_deg_s": RATE_EPS, "wheel_rad_s": WHEEL_EPS},
        },
        "windows": {f"{s:g}s": window(s) for s in (1.0, 2.0, 6.0)},
        "hold_wz_cmd_max_abs": float(np.max(np.abs([r["term_command_after_spin"][1] for r in hold]))) if hold else None,
        "hold_yaw_rate_max_abs_dps": float(np.max(np.abs([r["base_yaw_rate_dps"] for r in hold]))) if hold else None,
    }


baseline = load(args.baseline)
result = {
    "baseline_file": args.baseline,
    "control_dt_s": baseline["control_dt_s"],
    "wz_suppressed": baseline.get("wz_suppressed_after_arrival"),
    "base_term_cfg": baseline.get("base_term_cfg"),
    "baseline": summarize(baseline),
    "intervention": None,
}

if args.intervention:
    inter = load(args.intervention)
    result["intervention_file"] = args.intervention
    result["intervention"] = summarize(inter)
    table = {}
    for key in ("1s", "2s", "6s"):
        b, i = result["baseline"]["windows"][key], result["intervention"]["windows"][key]
        table[key] = {
            "yaw_change_deg": {"baseline": b["yaw_change_deg"], "suppressed": i["yaw_change_deg"]},
            "du_px": {"baseline": b["du_px"], "suppressed": i["du_px"]},
            "dv_px": {"baseline": b["dv_px"], "suppressed": i["dv_px"]},
            "offset_mean_px": {"baseline": b["offset_mean_px"], "suppressed": i["offset_mean_px"]},
            "std_u_px": {"baseline": b["std_u_px"], "suppressed": i["std_u_px"]},
            "std_v_px": {"baseline": b["std_v_px"], "suppressed": i["std_v_px"]},
            "ee_pos_err_mean_mm": {"baseline": b["ee_pos_err_mean_mm"], "suppressed": i["ee_pos_err_mean_mm"]},
        }
    result["comparison"] = table

Path(args.out).write_text(json.dumps(result, indent=1))

# -- time curve: turn request vs measured yaw rate, plus image displacement -----------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for label, data, style in (
        ("baseline", baseline, "-"),
        ("wz suppressed", load(args.intervention) if args.intervention else None, "--"),
    ):
        if data is None:
            continue
        recs = data["records"]
        t = np.array([r["sim_time_s"] for r in recs])
        t = t - t[0]
        wz = np.array([r["term_command_after_spin"][1] for r in recs])
        rate = np.array([r["base_yaw_rate_dps"] for r in recs])
        axes[0].plot(t, wz, style, label=f"{label}: wz command (rad/s)")
        axes[1].plot(t, rate, style, label=f"{label}: measured yaw rate (deg/s)")
        hold = [r for r in recs if r["phase"] == "hold"]
        th = np.array([r["sim_time_s"] for r in hold])
        th = th - th[0]
        du = np.array([(r["u"] - hold[0]["u"]) if r["u"] is not None else np.nan for r in hold])
        axes[2].plot(th, du, style, label=f"{label}: image du (px)")
    axes[0].set_ylabel("wz command [rad/s]")
    axes[1].set_ylabel("yaw rate [deg/s]")
    axes[2].set_ylabel("du from first hold frame [px]")
    axes[2].set_xlabel("time from first recorded step [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("D455 top view: turn command vs chassis yaw vs image slide")
    fig.tight_layout()
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=130)
    result["plot"] = args.plot
    Path(args.out).write_text(json.dumps(result, indent=1))
except Exception as error:  # noqa: BLE001
    result["plot_error"] = f"{type(error).__name__}: {error}"
    Path(args.out).write_text(json.dumps(result, indent=1))

print(json.dumps({k: v for k, v in result.items() if k != "comparison"}, indent=1)[:6000])
if result.get("comparison"):
    print(json.dumps(result["comparison"], indent=1))
print(f"wrote {args.out}")
