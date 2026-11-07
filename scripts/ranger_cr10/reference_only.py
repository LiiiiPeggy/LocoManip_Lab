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

"""Re-evaluate only the reference on a validation-style set, keeping the transforms.

This is the label audit for the conservative AABB intersection screen. The old set is the file
passed in `--results`; the new set is written next to it. Every row keeps its old reference fields
and gains the new ones, so a changed label is explicit rather than overwriting the old file.

Run:
    python scripts/ranger_cr10/reference_only.py \
        --results outputs/stage413/b/validation_exact/validation_results.json \
        --out outputs/stage413/b/reference_conservative_dev.json --workers 10
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from reference_geometry import MeshRef  # noqa: E402

ROOT = HERE.parents[1]
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes/cr10_meshes"
_CACHE = {}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--results", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=10)
ap.add_argument("--cap", type=float, default=0.06)
args = ap.parse_args()


def init_worker() -> None:
    if _CACHE:
        return
    m3 = trimesh.load(str(MESH_DIR / "Link3.STL"), force="mesh", process=False)
    m5 = trimesh.load(str(MESH_DIR / "Link5.STL"), force="mesh", process=False)
    _CACHE["ref3"] = MeshRef(np.asarray(m3.vertices), np.asarray(m3.faces), "Link3")
    _CACHE["ref5"] = MeshRef(np.asarray(m5.vertices), np.asarray(m5.faces), "Link5")


def evaluate(payload):
    rotation = np.asarray(payload["rotation"], dtype=np.float64)
    translation = np.asarray(payload["translation"], dtype=np.float64)
    started = time.perf_counter()
    result = _CACHE["ref3"].closest_to_in_frame(_CACHE["ref5"], rotation, translation, max_clearance=args.cap)
    return dict(
        id=payload["id"],
        new_status=result.status,
        new_distance=float(result.clearance),
        new_intersects=bool(result.intersects),
        new_triangle_a=int(result.triangle_a),
        new_triangle_b=int(result.triangle_b),
        new_reason=result.reason,
        new_extra=result.extra,
        new_seconds=time.perf_counter() - started,
    )


def main() -> None:
    data = json.loads((ROOT / args.results).read_text())
    rows = data["rows"]
    payloads = [dict(id=row["id"], rotation=row["rotation"], translation=row["translation"]) for row in rows]
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("fork"),
                             initializer=init_worker) as executor:
        new_rows = list(executor.map(evaluate, payloads, chunksize=1))
    by_id = {row["id"]: row for row in new_rows}
    label_changes = []
    distance_changes = []
    compared = []
    for row in rows:
        new = by_id[row["id"]]
        old_intersects = bool(row.get("reference_intersects", False))
        old_distance = float(row.get("reference_distance", float("nan")))
        changed_label = old_intersects != new["new_intersects"]
        delta = abs(old_distance - new["new_distance"]) if np.isfinite(old_distance) and np.isfinite(new["new_distance"]) else float("nan")
        if changed_label:
            label_changes.append(dict(id=row["id"], type=row.get("type"), cluster=row.get("cluster"),
                                      old_distance=old_distance, old_intersects=old_intersects,
                                      new_distance=new["new_distance"], new_intersects=new["new_intersects"],
                                      new_triangle_a=new["new_triangle_a"], new_triangle_b=new["new_triangle_b"]))
        if np.isfinite(delta) and delta > 1e-9:
            distance_changes.append(dict(id=row["id"], old=old_distance, new=new["new_distance"], delta=delta))
        compared.append(dict(
            id=row["id"],
            type=row.get("type"), cluster=row.get("cluster"),
            old_distance=old_distance, old_status=row.get("reference_status"),
            old_intersects=old_intersects,
            old_triangle_a=row.get("reference_triangle_a"), old_triangle_b=row.get("reference_triangle_b"),
            new_distance=new["new_distance"], new_status=new["new_status"],
            new_intersects=new["new_intersects"],
            new_triangle_a=new["new_triangle_a"], new_triangle_b=new["new_triangle_b"],
            new_reason=new["new_reason"], new_extra=new["new_extra"], new_seconds=new["new_seconds"],
            changed_label=changed_label, distance_delta=delta,
        ))
    extras = [row["new_extra"] for row in new_rows]
    def median(key):
        values = [float(item.get(key, 0)) for item in extras]
        return float(np.median(values)) if values else float("nan")
    summary = dict(
        results=args.results,
        rows=len(rows),
        label_changes=len(label_changes),
        distance_changes=len(distance_changes),
        max_distance_delta=max((item["delta"] for item in distance_changes), default=0.0),
        changed_ids=[item["id"] for item in label_changes],
        label_change_rows=label_changes,
        distance_change_rows=distance_changes[:50],
        aabb_candidates_median=median("aabb_candidates"),
        aabb_candidates_max=max((int(item.get("aabb_candidates", 0)) for item in extras), default=0),
        exact_tests_median=median("exact_tests"),
        exact_tests_max=max((int(item.get("exact_tests", 0)) for item in extras), default=0),
        leaf_pairs_median=median("leaf_pairs"),
        leaf_pairs_max=max((int(item.get("leaf_pairs", 0)) for item in extras), default=0),
        seconds_median=float(np.median([row["new_seconds"] for row in new_rows])) if new_rows else float("nan"),
        seconds_total=sum(row["new_seconds"] for row in new_rows),
        wall_seconds=time.time() - started,
    )
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(summary=summary, rows=compared), indent=1))
    print(json.dumps({key: summary[key] for key in (
        "rows", "label_changes", "distance_changes", "max_distance_delta",
        "aabb_candidates_median", "aabb_candidates_max", "exact_tests_median", "exact_tests_max",
        "seconds_median", "seconds_total", "wall_seconds",
    )}, indent=1))
    print("changed ids:", summary["changed_ids"][:20])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
