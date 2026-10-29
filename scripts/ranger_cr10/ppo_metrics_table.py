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

"""Stage 4.5j: the per-500-iteration PPO table, read from the run's own TensorBoard record.

Every number here is a logged scalar, not a re-derivation: `Loss/value`, `Loss/surrogate`,
`Loss/entropy`, `Loss/learning_rate`, `Policy/mean_std`, `Train/mean_reward`, and the three
task metrics the environment itself logs (`Metrics/ee_pose/position_error`,
`Metrics/ee_pose/orientation_error`, `Metrics/base_velocity/error_vel_xy` / `error_vel_yaw`).

A note on KL. rsl_rl 5.0.1 uses KL only inside its adaptive learning-rate controller and does
not write it to the logger, so there is no `KL` scalar to report. What *is* logged is the
learning rate that controller produces: with `desired_kl = 0.01` it multiplies the rate by 1.5
when the measured KL drops below 0.005 and divides it by 1.5 when KL exceeds 0.02. The LR
column therefore bounds the per-update KL from both sides, and a stationary LR is evidence that
KL sat inside the tolerated band; it is not the KL itself and is not labelled as such.

Usage:
    python scripts/ranger_cr10/ppo_metrics_table.py --run <run dir> --every 500 --out <csv>
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser(description="Per-iteration PPO metrics table from tfevents.")
parser.add_argument("--run", required=True, help="run directory containing events.out.tfevents*")
parser.add_argument("--every", type=int, default=500)
parser.add_argument("--out", default=None, help="CSV path; defaults to <run>/metrics.csv")
parser.add_argument("--last", type=int, default=None, help="stop at this iteration (default: last logged)")
args_cli = parser.parse_args()

RUN = Path(args_cli.run)
events = sorted(glob.glob(str(RUN / "events.out.tfevents*")))
if not events:
    raise SystemExit(f"no tfevents under {RUN}")
ea = EventAccumulator(events[0], size_guidance={"scalars": 0})
ea.Reload()
tags = set(ea.Tags()["scalars"])

WANT = {
    "Loss/surrogate": "surrogate",
    "Loss/value": "value",
    "Loss/entropy": "entropy",
    "Loss/learning_rate": "lr",
    "Policy/mean_std": "std",
    "Train/mean_reward": "reward",
    "Train/mean_episode_length": "ep_len",
    "Metrics/ee_pose/position_error": "ee_pos",
    "Metrics/ee_pose/orientation_error": "ee_ori",
    "Metrics/ee_pose/success": "success",
    "Metrics/base_velocity/error_vel_xy": "lin_err",
    "Metrics/base_velocity/error_vel_yaw": "yaw_err",
    "Episode_Termination/base_contact": "term_contact",
    "Episode_Termination/bad_orientation": "term_ori",
    "Episode_Termination/time_out": "term_timeout",
}
missing = [t for t in WANT if t not in tags]
if missing:
    print(f"警告: 以下标签不存在, 相应列留空: {missing}")

series = {}
for tag, key in WANT.items():
    if tag in tags:
        series[key] = {int(s.step): float(s.value) for s in ea.Scalars(tag)}

all_iters = sorted({s for d in series.values() for s in d})
if not all_iters:
    raise SystemExit("no scalars logged")
lo, hi = all_iters[0], args_cli.last or all_iters[-1]
marks = [it for it in range(0, hi + 1, args_cli.every) if it >= lo]
if hi not in marks:
    marks.append(hi)


def at(key, it):
    d = series.get(key, {})
    if not d:
        return None
    # the closest logged iteration at or below `it`
    cand = [s for s in d if s <= it]
    return d[max(cand)] if cand else None


ORDER = [
    "surrogate",
    "value",
    "entropy",
    "lr",
    "std",
    "reward",
    "ep_len",
    "ee_pos",
    "ee_ori",
    "success",
    "lin_err",
    "yaw_err",
    "term_contact",
    "term_ori",
    "term_timeout",
]

hdr = (
    f"{'iter':>6}{'surrogate':>11}{'value':>10}{'entropy':>10}{'lr':>10}{'std':>7}"
    f"{'reward':>9}{'ep_len':>8}{'EE pos':>9}{'EE ori':>8}{'succ':>7}{'lin err':>9}{'yaw err':>9}"
)
print(hdr)
print("-" * len(hdr))
FMT = {
    "surrogate": ".4f",
    "value": ".4f",
    "entropy": ".4f",
    "lr": ".2e",
    "std": ".4f",
    "reward": ".2f",
    "ep_len": ".1f",
    "ee_pos": ".4f",
    "ee_ori": ".4f",
    "success": ".3f",
    "lin_err": ".4f",
    "yaw_err": ".4f",
    "term_contact": ".3f",
    "term_ori": ".3f",
    "term_timeout": ".3f",
}
WIDTH = {
    "surrogate": 11,
    "value": 10,
    "entropy": 10,
    "lr": 10,
    "std": 7,
    "reward": 9,
    "ep_len": 8,
    "ee_pos": 9,
    "ee_ori": 8,
    "success": 7,
    "lin_err": 9,
    "yaw_err": 9,
}
rows = []
for it in marks:
    vals = {k: at(k, it) for k in ORDER}
    rows.append({"iter": it, **vals})
    line = f"{it:>6}"
    for k in (
        "surrogate",
        "value",
        "entropy",
        "lr",
        "std",
        "reward",
        "ep_len",
        "ee_pos",
        "ee_ori",
        "success",
        "lin_err",
        "yaw_err",
    ):
        v = vals[k]
        cell = "" if v is None else format(v, FMT[k])
        line += f"{cell:>{WIDTH[k]}}"
    print(line)

out = Path(args_cli.out) if args_cli.out else RUN / "metrics.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["iter"] + ORDER)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"\n已写出 {out}  ({len(rows)} 行, 日志到 iteration {all_iters[-1]})")
