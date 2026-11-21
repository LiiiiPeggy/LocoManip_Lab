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

"""Three-view summary of the D455 fixed-view observation, deployment filter off vs on.

Reads the result JSONs already on disk; no simulation.

    view   off run                         on run
    top    d455_cmp_off                    d455_cmp_on
    front  d455_fs_off                     d455_fs_on
    side   d455_fs_off                     d455_fs_on

Per view and group it reports the arrival step, the first-frame offset, and for the fixed
prefixes 1 s / 2 s / 6 s the visible-frame count (from the recorded ``visible`` flag, with every
frame of the prefix in the denominator), the mean centre offset, std(u), std(v), and the
first-to-last displacement of the prefix. Displacement and spread are reported separately so
"steadily off" and "sliding" stay distinguishable.

Criteria are the ones fixed before the runs: visible ratio >= 0.90, mean centre offset <= 20 px,
std(u) <= 5 px, std(v) <= 5 px.

Usage:
    python scripts/ranger_cr10/summarize_d455_three_views.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/stage413/b"
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default=str(BASE / "d455_three_views_summary.json"))
args = ap.parse_args()

RUNS = {
    ("front", "off"): BASE / "d455_fs_off",
    ("front", "on"): BASE / "d455_fs_on",
    ("side", "off"): BASE / "d455_fs_off",
    ("side", "on"): BASE / "d455_fs_on",
    ("top", "off"): BASE / "d455_cmp_off",
    ("top", "on"): BASE / "d455_cmp_on",
}
VIEWS = ("front", "side", "top")
CENTER = (320.0, 240.0)
VISIBLE_MIN_RATIO = 0.90
OFFSET_MAX_PX = 20.0
STD_MAX_PX = 5.0


def summarize(directory: Path, view: str) -> dict | None:
    path = directory / "d455_view_results.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if view not in data["views"]:
        return None
    v = data["views"][view]
    records = v["hold_records"]
    dt = data["protocol"]["control_dt_s"] if "control_dt_s" in data["protocol"] else 0.1
    out = {
        "checkpoint": data["checkpoint"],
        "arrived": bool(v["arrived"]),
        "arrival_step": v["arrival_step"],
        "frames": len(records),
        "target_world": v.get("marker_registered_world"),
        "target_quat": v.get("marker_registered_quat"),
        "registration": v.get("marker_registration"),
        "initial_state": v.get("initial_state"),
        "entry_offset": None,
        "windows": {},
    }
    if records:
        du0, dv0 = records[0]["du"], records[0]["dv"]
        out["entry_offset"] = {"du": du0, "dv": dv0, "magnitude": float(math.hypot(du0, dv0)),
                               "note": "already present at the first hold frame; not drift"}
    for seconds in (1.0, 2.0, 6.0):
        n = min(len(records), int(round(seconds / dt)))
        seg = records[:n]
        if not seg:
            continue
        visible = [bool(r["visible"]) for r in seg]
        u = np.array([r["u"] for r in seg if r["u"] is not None], dtype=float)
        vv = np.array([r["v"] for r in seg if r["v"] is not None], dtype=float)
        offsets = np.hypot(u - CENTER[0], vv - CENTER[1])
        w = {
            "frames": n,
            "visible_frames": int(sum(visible)),
            "visible_ratio": float(sum(visible)) / n,
            "detected_frames": int(len(u)),
            "offset_mean_px": float(offsets.mean()) if len(offsets) else None,
            "std_u_px": float(u.std()) if len(u) > 1 else None,
            "std_v_px": float(vv.std()) if len(vv) > 1 else None,
            # Two different baselines, kept apart on purpose:
            #   * max horizontal offset of the chassis from its position at reset, within this
            #     window (the historical ``base_drift_xy_m``, a scalar distance);
            #   * displacement from the first hold frame of this window, only available when the
            #     run recorded per-frame base positions.
            "max_offset_from_reset_mm": max((r["base_drift_xy_m"] for r in seg), default=0.0) * 1000.0,
            "max_offset_from_first_hold_mm": (
                None if not all("base_pos_w" in r for r in seg)
                else float(max(math.hypot(r["base_pos_w"][0] - seg[0]["base_pos_w"][0],
                                          r["base_pos_w"][1] - seg[0]["base_pos_w"][1]) for r in seg) * 1000.0)
            ),
            "max_offset_from_first_hold_unavailable_reason": (
                None if all("base_pos_w" in r for r in seg)
                else "this run did not record per-frame base_pos_w"
            ),
            "yaw_change_deg": float(seg[-1]["base_yaw_deg"] - seg[0]["base_yaw_deg"]),
            "displacement_px": None if len(u) < 2 else {
                "du": float(u[-1] - u[0]),
                "dv": float(vv[-1] - vv[0]),
                "magnitude": float(math.hypot(u[-1] - u[0], vv[-1] - vv[0])),
            },
        }
        w["criteria"] = {
            "visible_ok": w["visible_ratio"] >= VISIBLE_MIN_RATIO,
            "offset_ok": w["offset_mean_px"] is not None and w["offset_mean_px"] <= OFFSET_MAX_PX,
            "std_u_ok": w["std_u_px"] is not None and w["std_u_px"] <= STD_MAX_PX,
            "std_v_ok": w["std_v_px"] is not None and w["std_v_px"] <= STD_MAX_PX,
        }
        w["passes"] = all(w["criteria"].values())
        out["windows"][f"{seconds:g}s"] = w
    return out


dt_seen = None
for _view in ("front", "side", "top"):
    for _group in ("off", "on"):
        _path = RUNS[(_view, _group)] / "d455_view_results.json"
        if _path.is_file():
            _doc = json.loads(_path.read_text())
            dt_seen = _doc.get("protocol", {}).get("control_dt_s")
            break
    if dt_seen:
        break

results = {"criteria": {"visible_ratio_min": VISIBLE_MIN_RATIO, "offset_mean_max_px": OFFSET_MAX_PX,
                        "std_max_px": STD_MAX_PX},
           "protocol": "model_9999 + v2, fixed gripper, frozen inference, A1 world-fixed target, "
                       "original seed and view order; front/side re-paired with a shared preset "
                       "identity orientation because the earlier runs did not record the marker "
                       "quaternion",
           "views": {}}
for view in VIEWS:
    entry = {}
    for group in ("off", "on"):
        entry[group] = summarize(RUNS[(view, group)], view)
    # Pairing checks: same target position, same marker orientation, same reset state.
    off, on = entry["off"], entry["on"]
    entry["pairing"] = None
    if off and on:
        entry["pairing"] = {
            "target_identical": off["target_world"] is not None and on["target_world"] is not None
            and np.allclose(off["target_world"], on["target_world"], atol=1e-9),
            # The top pair predates the quaternion recording, so its orientation is "not
            # recorded", not "different": both arms used the chassis attitude at the same
            # registration step, and the initial state is identical. Keep the two cases apart.
            "quat_recorded": off["target_quat"] is not None and on["target_quat"] is not None,
            "quat_identical": (None if (off["target_quat"] is None or on["target_quat"] is None)
                               else bool(np.allclose(off["target_quat"], on["target_quat"], atol=1e-9))),
            "initial_state_identical": (off["initial_state"] is not None and on["initial_state"] is not None
                                        and off["initial_state"]["joint_pos_digest"]
                                        == on["initial_state"]["joint_pos_digest"]
                                        and np.allclose(off["initial_state"]["base_pos_w"],
                                                        on["initial_state"]["base_pos_w"], atol=1e-9)),
        }
    results["views"][view] = entry

# -- total verdict ---------------------------------------------------------------
# A view only counts when it is present, arrived, and every one of its three windows has the
# full frame count. Missing data and short windows are marked explicitly rather than being
# dropped, so a partial run can never sum to "pass".
expected_frames = {f"{s:g}s": int(round(s / dt_seen)) for s in (1.0, 2.0, 6.0)} if dt_seen else {}


def view_verdict(view: str) -> dict:
    on = results["views"][view].get("on")
    if on is None:
        return {"present": False, "arrived": None, "windows_complete": None, "passes": False,
                "reason": "missing ON result"}
    windows = on["windows"]
    missing = [w for w in ("1s", "2s", "6s") if w not in windows]
    short = [w for w in ("1s", "2s", "6s")
             if w in windows and expected_frames and windows[w]["frames"] < expected_frames[w]]
    verdict = {
        "present": True,
        "arrived": bool(on["arrived"]),
        "windows_complete": not missing and not short,
        "missing_windows": missing,
        "short_windows": short,
        "expected_frames": expected_frames,
        "window_passes": {w: windows[w]["passes"] for w in ("1s", "2s", "6s") if w in windows},
    }
    verdict["passes"] = bool(verdict["arrived"] and verdict["windows_complete"]
                             and all(verdict["window_passes"].values())
                             and len(verdict["window_passes"]) == 3)
    reasons = []
    if not verdict["arrived"]:
        reasons.append("did not arrive")
    if missing:
        reasons.append(f"missing windows {missing}")
    if short:
        reasons.append(f"short windows {short}")
    if any(not p for p in verdict["window_passes"].values()):
        reasons.append("window criteria failed: "
                       + ",".join(w for w, p in verdict["window_passes"].items() if not p))
    verdict["reason"] = "; ".join(reasons) if reasons else "all windows present, complete and passing"
    return verdict


results["summary"] = {
    "per_view": {view: view_verdict(view) for view in VIEWS},
    "all_three_views_pass": all(view_verdict(view)["passes"] for view in VIEWS),
    "note": "passing requires all three views to be present, arrived, with complete 1/2/6 s "
            "window frames and all per-window criteria met",
}
Path(args.out).write_text(json.dumps(results, indent=1))

print(f"{'view':6} {'grp':4} {'arr':>4} {'entry':>7} | {'1s vis':>8} {'mean':>6} {'std_u':>6} {'std_v':>6} {'move':>6} "
      f"| {'2s mean':>7} {'std_v':>6} | {'6s mean':>7} {'std_u':>6} {'move':>6} {'pass':>5}")
for view in VIEWS:
    for group in ("off", "on"):
        s = results["views"][view][group]
        if not s:
            print(f"{view:6} {group:4}  MISSING")
            continue
        w1, w2, w6 = s["windows"]["1s"], s["windows"]["2s"], s["windows"]["6s"]
        print(f"{view:6} {group:4} {str(s['arrival_step']):>4} {s['entry_offset']['magnitude']:7.2f} | "
              f"{w1['visible_frames']:3d}/{w1['frames']:<4d} {w1['offset_mean_px']:6.1f} {w1['std_u_px']:6.2f} "
              f"{w1['std_v_px']:6.2f} {w1['displacement_px']['magnitude']:6.1f} | {w2['offset_mean_px']:7.1f} "
              f"{w2['std_v_px']:6.2f} | {w6['offset_mean_px']:7.1f} {w6['std_u_px']:6.2f} "
              f"{w6['displacement_px']['magnitude']:6.1f} {str(w6['passes']):>5}")
    p = results["views"][view]["pairing"]
    print(f"{'':6} pairing: {p}")
print()
for view in VIEWS:
    v = results["summary"]["per_view"][view]
    print(f"verdict {view:6}: present={v['present']} arrived={v['arrived']} "
          f"complete={v['windows_complete']} passes={v['passes']} -- {v['reason']}")
print("all three views pass:", results["summary"]["all_three_views_pass"])
print(f"wrote {args.out}")
