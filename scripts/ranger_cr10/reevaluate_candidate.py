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

"""Re-evaluate one candidate on a fixed set of transforms, keeping the reference labels.

The reference is expensive and was fixed for the development and holdout sets; changing only the
convex pieces must not change the reference fields. This script loads a validation-style results
file, replaces the candidate distance/intersection fields for the given piece directories, and
recomputes the same summaries. Reference distance, status, intersection label and the fixed contact
tolerance are never recomputed here.

Run (minimal comparison):
    python scripts/ranger_cr10/reevaluate_candidate.py \
        --results outputs/stage413/b/validation_exact/validation_results.json \
        --pieces3 outputs/stage413/b/metric20 --pieces5 outputs/stage413/b/correction_v1/link5_t2mm \
        --out outputs/stage413/b/correction_v1/compare_old3_new5.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from candidate_eval import CONTACT_TOL, load_pieces, union_query  # noqa: E402

ROOT = HERE.parents[1]
BINS = [(0.0, 0.001), (0.001, 0.003), (0.003, 0.005), (0.005, 0.010), (0.010, 0.020)]
BIN_LABELS = ["0-1 mm", "1-3 mm", "3-5 mm", "5-10 mm", "10-20 mm"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--results", required=True)
ap.add_argument("--pieces3", required=True)
ap.add_argument("--pieces5", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--only-cluster", default=None)
api = ap.parse_args()


def percentile(values, q):
    values = [v for v in values if np.isfinite(v)]
    return float(np.percentile(values, q)) if values else float("nan")


def bucket(row):
    if row["reference_status"] == "certified_culled":
        return "beyond_cap"
    if row["reference_status"] != "exact":
        return "unknown"
    if row["reference_intersects"]:
        return "overlap"
    if row["reference_distance"] <= CONTACT_TOL:
        return "uncertain"
    if row["reference_distance"] <= 0.020:
        return "0-20mm"
    return "far"


def candidate_state(row):
    if row["pieces_status"] == "failed":
        return "unknown"
    if row["pieces_intersects"] or row["pieces_distance"] <= CONTACT_TOL:
        return "contact"
    return "separated"


def summarize(rows):
    errors = [row["signed_error"] for row in rows if np.isfinite(row["signed_error"])]
    return dict(
        n=len(rows),
        n_error=len(errors),
        abs_p95=percentile([abs(e) for e in errors], 95),
        abs_max=max((abs(e) for e in errors), default=float("nan")),
        signed_median=percentile(errors, 50),
        signed_min=min(errors, default=float("nan")),
        signed_max=max(errors, default=float("nan")),
        false_contact=sum(1 for row in rows if row["candidate_state"] == "contact"),
        candidate_unknown=sum(1 for row in rows if row["candidate_state"] == "unknown"),
    )


def main() -> None:
    started = time.time()
    data = json.loads((ROOT / api.results).read_text())
    rows = data["rows"]
    if api.only_cluster:
        rows = [row for row in rows if row.get("cluster") == api.only_cluster]
    pieces3 = load_pieces("cr10_Link3", ROOT / api.pieces3)
    pieces5 = load_pieces("cr10_Link5", ROOT / api.pieces5)
    if not pieces3 or not pieces5:
        raise SystemExit("missing pieces")

    for row in rows:
        rotation = np.asarray(row["rotation"], dtype=np.float64)
        translation = np.asarray(row["translation"], dtype=np.float64)
        result = union_query(pieces3, pieces5, rotation, translation)
        row["pieces_distance"] = result["distance"]
        row["pieces_status"] = result["status"]
        row["pieces_intersects"] = result["intersects"]
        row["pieces_pair"] = result["pair"]
        row["pieces_failed"] = result["failed"]
        row["pieces_pairs"] = result["pairs"]
        row["pieces_iterations"] = result["iterations"]
        row["pieces_seconds"] = result["seconds"]
        row["reference_bucket"] = bucket(row)
        row["candidate_state"] = candidate_state(row)
        if (
            row["reference_status"] == "exact"
            and not row["reference_intersects"]
            and row["pieces_status"] != "failed"
        ):
            row["signed_error"] = float(row["pieces_distance"] - row["reference_distance"])
            row["abs_error"] = abs(row["signed_error"])
        else:
            row["signed_error"] = float("nan")
            row["abs_error"] = float("nan")

    validation = [row for row in rows if row.get("type") in ("B", "C")]
    regression = [row for row in rows if row.get("type") == "A"]
    separated_20 = [
        row for row in validation if row["reference_bucket"] == "0-20mm" and np.isfinite(row["signed_error"])
    ]
    known = [row for row in validation if row["reference_bucket"] in ("0-20mm", "far", "overlap")]
    separated = [row for row in validation if row["reference_bucket"] in ("0-20mm", "far")]
    overlap_rows = [row for row in validation if row["reference_bucket"] == "overlap"]
    overlap_known = [row for row in overlap_rows if row["candidate_state"] != "unknown"]
    overlap_missed = [row for row in overlap_known if row["candidate_state"] == "separated"]
    uncertain = [row for row in validation if row["reference_bucket"] == "uncertain"]

    bin_summary = {}
    for (lo, hi), label in zip(BINS, BIN_LABELS):
        subset = [
            row for row in validation
            if row["reference_bucket"] == "0-20mm" and lo <= row["reference_distance"] < hi
        ]
        bin_summary[label] = summarize(subset)

    by_type = {}
    for kind in ("C", "B"):
        subset = [row for row in validation if row.get("type") == kind]
        entry = summarize([row for row in subset if row["reference_bucket"] == "0-20mm"])
        entry["bins"] = {
            label: summarize([
                row for row in subset
                if row["reference_bucket"] == "0-20mm" and lo <= row["reference_distance"] < hi
            ])
            for (lo, hi), label in zip(BINS, BIN_LABELS)
        }
        entry["overlap_samples"] = sum(1 for row in subset if row["reference_bucket"] == "overlap")
        entry["missed_overlaps"] = sum(
            1 for row in subset
            if row["reference_bucket"] == "overlap" and row["candidate_state"] == "separated"
        )
        entry["false_contact_all_bins"] = sum(
            1 for row in subset
            if row["candidate_state"] == "contact" and row["reference_bucket"] in ("0-20mm", "far")
        )
        by_type[kind] = entry

    cluster_summary = {}
    for cluster in sorted({row["cluster"] for row in validation if row.get("type") == "C"}):
        subset = [row for row in validation if row.get("type") == "C" and row["cluster"] == cluster]
        entry = summarize([row for row in subset if row["reference_bucket"] == "0-20mm"])
        positives = sum(1 for row in subset if row["reference_bucket"] == "overlap")
        misses = sum(
            1 for row in subset
            if row["reference_bucket"] == "overlap" and row["candidate_state"] == "separated"
        )
        entry.update(
            bins=sorted({
                label for (lo, hi), label in zip(BINS, BIN_LABELS)
                if any(row["reference_bucket"] == "0-20mm" and lo <= row["reference_distance"] < hi for row in subset)
            }),
            overlap_samples=positives,
            missed_overlaps=misses,
            miss_rate=None if positives == 0 else misses / positives,
            unknown=sum(1 for row in subset if row["candidate_state"] == "unknown"),
        )
        cluster_summary[cluster] = entry

    margins = {}
    for margin in (0.0, 0.005, 0.010):
        subset = [
            row for row in validation
            if row["reference_bucket"] in ("0-20mm", "far", "overlap")
            and row["candidate_state"] != "unknown"
        ]
        reference_unsafe = [
            row for row in subset
            if row["reference_bucket"] == "overlap"
            or (margin == 0.0 and row["reference_distance"] <= CONTACT_TOL)
            or (margin > 0.0 and row["reference_distance"] < margin)
        ]
        candidate_unsafe = [
            row for row in subset
            if row["candidate_state"] == "contact"
            or (margin > 0.0 and row["pieces_distance"] < margin)
        ]
        ref_ids = {row["id"] for row in reference_unsafe}
        cand_ids = {row["id"] for row in candidate_unsafe}
        fp = [row for row in subset if row["id"] in cand_ids and row["id"] not in ref_ids]
        fn = [row for row in subset if row["id"] in ref_ids and row["id"] not in cand_ids]
        margins[f"{margin * 1000:.0f} mm"] = dict(
            n_known=len(subset), reference_unsafe=len(ref_ids), candidate_unsafe=len(cand_ids),
            false_positive=len(fp), false_negative=len(fn),
            false_positive_rate=len(fp) / len(subset) if subset else float("nan"),
            false_negative_rate=len(fn) / len(ref_ids) if ref_ids else float("nan"),
        )

    false_contacts = [
        row for row in separated if row["candidate_state"] == "contact"
    ]
    worst = sorted(
        [row for row in validation if np.isfinite(row["signed_error"])],
        key=lambda row: row["signed_error"],
    )[:5]

    summary = dict(
        evaluated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        results_input=api.results,
        pieces3=api.pieces3,
        pieces5=api.pieces5,
        only_cluster=api.only_cluster,
        piece_stats=dict(
            link3_pieces=len(pieces3), link3_vertices=sum(len(v) for _, v in pieces3),
            link5_pieces=len(pieces5), link5_vertices=sum(len(v) for _, v in pieces5),
            piece_pairs=len(pieces3) * len(pieces5),
            candidate_seconds_median=percentile([row["pieces_seconds"] for row in rows], 50),
            candidate_seconds_max=max((row["pieces_seconds"] for row in rows), default=float("nan")),
            candidate_seconds_total=sum(row["pieces_seconds"] for row in rows),
        ),
        counts=dict(
            total_rows=len(rows), validation_B_C=len(validation), regression_A=len(regression),
            known=len(known), separated=len(separated),
            reference_overlap=len(overlap_rows), reference_uncertain=len(uncertain),
            candidate_unknown=sum(1 for row in validation if row["candidate_state"] == "unknown"),
            reference_unknown=sum(1 for row in validation if row["reference_bucket"] == "unknown"),
        ),
        bin_summary=bin_summary,
        overall_0_20=summarize(separated_20),
        by_type=by_type,
        cluster_summary=cluster_summary,
        overlap=dict(
            positives=len(overlap_rows), known=len(overlap_known), detected=len(overlap_known) - len(overlap_missed),
            missed=len(overlap_missed), unknown=len(overlap_rows) - len(overlap_known),
            miss_rate=None if not overlap_known else len(overlap_missed) / len(overlap_known),
            missed_ids=[row["id"] for row in overlap_missed],
        ),
        false_contact=dict(
            fp=len(false_contacts),
            n_total_known=len(known),
            n_reference_negative=len(separated),
            fp_over_n_total=len(false_contacts) / len(known) if known else float("nan"),
            fp_over_n_reference_negative=len(false_contacts) / len(separated) if separated else float("nan"),
            uncertain_rows=len(uncertain),
            uncertain_contacts=sum(1 for row in uncertain if row["candidate_state"] == "contact"),
        ),
        margins=margins,
        worst_errors=[
            dict(id=row["id"], type=row["type"], cluster=row["cluster"],
                 reference=row["reference_distance"], candidate=row["pieces_distance"],
                 signed_error=row["signed_error"], pair=row["pieces_pair"],
                 reference_triangle_a=row.get("reference_triangle_a"),
                 reference_triangle_b=row.get("reference_triangle_b"))
            for row in worst
        ],
        regression=dict(
            n=len(regression),
            hull_false_contacts=sum(1 for row in regression if row["hull_intersects"]),
            pieces_false_contacts=sum(1 for row in regression if row["candidate_state"] == "contact"),
            reference_overlaps=sum(1 for row in regression if row["reference_intersects"]),
            pieces_missed=sum(
                1 for row in regression
                if row["reference_intersects"] and row["candidate_state"] == "separated"
            ),
        ),
    )

    out = ROOT / api.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
    with out.with_suffix(".csv").open("w", newline="") as handle:
        fields = [
            "id", "type", "cluster", "seed", "s", "reference_distance", "reference_status",
            "reference_intersects", "reference_bucket", "pieces_distance", "pieces_status",
            "pieces_intersects", "pieces_pair", "pieces_failed", "pieces_iterations",
            "pieces_seconds", "candidate_state", "signed_error", "abs_error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(json.dumps(summary, indent=1))
    print(f"wrote {out} and {out.with_suffix('.csv')} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
