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

"""Stage 4.11: one table from the per-checkpoint evaluation logs.

The suites are run by `eval_stage411.sh`, which writes `mujoco_<suite>_<tag>_<suite>.json`
for the three simulator suites and a text log for the collision suite (that script counts
contacts, it does not emit JSON). This reads them back and prints the rows the stage is judged
on, so the comparison is generated from the artifacts rather than retyped from a terminal.

Usage:
    python scripts/ranger_cr10/compare_stage411.py base9999 iter0500 iter1000 ...
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("tags", nargs="+", help="tags given to eval_stage411.sh, in report order")
ap.add_argument("--dir", default="outputs/stage411/eval")
ap.add_argument("--json", default="", help="also write the table as JSON here")
args = ap.parse_args()

OUT = ROOT / args.dir
CONTACT_ROW = re.compile(r"随机六维 \(4\.10 盒\)\s+(\d+)%\s+(-?\d+)mm\s+(\d+)mm\s+(\d+)mm\s+([\d.]+)°\s+(.*)")


def find(tag: str, suite: str) -> Path:
    hits = sorted(glob.glob(str(OUT / f"*_{tag}_{suite}.json")))
    if not hits:
        raise SystemExit(f"missing {suite} json for tag {tag} in {OUT}")
    return Path(hits[0])


def load(tag: str) -> dict:
    row = {}
    coverage = json.loads(find(tag, "cov").read_text())["coverage"]
    row["success30"] = coverage["success"]["30mm2deg"]
    row["success50"] = coverage["success"]["50mm5deg"]
    row["pos_mm"] = coverage["pos_median"] * 1000
    row["ori_deg"] = coverage["ori_median"]
    row["coverage_contact"] = coverage["contact"]
    row["at_limit"] = coverage["at_limit"]

    low = json.loads(find(tag, "low").read_text())["lowspace"]
    row["low_pos_mm"] = sum(t["err_pos"] for t in low) / len(low) * 1000
    row["low_ori_deg"] = sum(t["err_ori"] for t in low) / len(low)
    row["low_contact"] = sum(t["contacts"] for t in low)

    modes = json.loads(find(tag, "modes").read_text())["modes"]
    for case in modes:
        name = case["case"].split()[0]
        if name in ("C1", "C2", "C3"):
            row[f"{name}_ee_mm"] = case["ee_pos"] * 1000
            row[f"{name}_vx_err"] = case["vx_err"]

    drift = json.loads(find(tag, "drift").read_text())["drift"]
    for case in drift:
        if case["config"] == "input+lock" and case["scenario"] == "static":
            row["drift_mm"] = case["drift_m"] * 1000
            row["drift_ee_mm"] = case["ee_pos"] * 1000

    log = OUT / f"{tag}_col.log"
    row["contact"] = None
    row["contact_pairs"] = {}
    if log.exists():
        for line in log.read_text().splitlines():
            m = CONTACT_ROW.search(line)
            if m:
                row["contact"] = int(m.group(1)) / 100.0
                row["clearance_mm"] = int(m.group(2))
                row["contact_pairs"] = m.group(6).strip()
    return row


def main() -> None:
    rows = {tag: load(tag) for tag in args.tags}
    header = (
        f"{'checkpoint':10s} {'50mm/5°':>8s} {'30mm/2°':>8s} {'pos(mm)':>7s} {'ori(°)':>6s} "
        f"{'contact':>8s} {'gap(mm)':>7s} {'low(mm)':>7s} {'C1(mm)':>6s} {'C3(mm)':>6s} {'drift(mm)':>9s}"
    )
    print(header)
    print("-" * len(header))
    for tag, r in rows.items():
        print(
            f"{tag:10s} {r['success50'] * 100:7.1f}% {r['success30'] * 100:7.1f}% "
            f"{r['pos_mm']:7.0f} {r['ori_deg']:6.1f} "
            f"{(r['contact'] if r['contact'] is not None else -1) * 100:7.1f}% "
            f"{r.get('clearance_mm', -1):7d} {r['low_pos_mm']:7.0f} {r['C1_ee_mm']:6.0f} "
            f"{r['C3_ee_mm']:6.0f} {r['drift_mm']:9.1f}"
        )
    print()
    for tag, r in rows.items():
        print(f"{tag:10s} contact pairs: {r['contact_pairs'] or '-'}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
