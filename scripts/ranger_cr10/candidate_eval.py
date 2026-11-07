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

"""Stage 4.13b: one evaluation of the frozen convex-decomposition candidate.

The candidate is a set of convex pieces per link. Its distance is the minimum over all piece
pairs, and a pair that intersects makes the whole set intersect. A pair whose GJK does not
converge is *not* a large distance: the set is marked ``failed`` so that the caller cannot count
it as safe.

The reference is the original triangle mesh, queried by `reference_geometry.MeshRef`. The single
convex hull is evaluated as well, because it is the representation that produced the false
contacts this candidate is meant to remove.

The module is imported by both the orchestrator and its worker processes; `init_worker` builds the
static meshes once per process.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trimesh  # noqa: E402

from convex_distance import gjk_distance  # noqa: E402
from reference_geometry import MeshRef  # noqa: E402

ROOT = HERE.parents[1]
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes/cr10_meshes"

# Fixed before looking at any validation result: a reference separation at or below this is
# numerically indistinguishable from contact and is reported in its own bucket.
CONTACT_TOL = 1e-4  # metres

_CACHE: dict[str, object] = {}


def load_pieces(part: str, directory: str | Path) -> list[tuple[str, np.ndarray]]:
    """Load each convex piece separately; nothing here can merge them back into one hull."""
    out = []
    for path in sorted(Path(directory).glob(f"{part}_p*.npz")):
        data = np.load(path)
        out.append((path.stem, np.asarray(data["vertices"], dtype=np.float64)))
    return out


def _classify_gjk(result, tol: float) -> tuple[str, float]:
    """Map a GJK status to (intersecting | separated | failed, distance)."""
    if result.status == "intersecting":
        return "intersecting", 0.0
    if result.status == "separated":
        return "separated", float(result.distance)
    if result.status == "degenerate":
        # v == w means the support did not move: the closest point is converged. A near-zero
        # closest vector is contact, not a failure.
        if np.isfinite(result.distance) and result.distance > CONTACT_TOL:
            return "separated", float(result.distance)
        return "intersecting", 0.0
    return "failed", float("nan")


def _gjk_pair(vertices_a: np.ndarray, vertices_b: np.ndarray, tol: float = 1e-8) -> tuple[str, float, int]:
    result = gjk_distance(vertices_a, vertices_b, tol=tol, max_iterations=128)
    status, distance = _classify_gjk(result, tol)
    iterations = int(result.iterations)
    if status == "failed":
        # One retry with a stricter tolerance and a larger iteration budget. If it still does not
        # converge the pair is reported as failed; it must never become a safe distance.
        result = gjk_distance(vertices_a, vertices_b, tol=1e-10, max_iterations=512)
        status, distance = _classify_gjk(result, 1e-10)
        iterations += int(result.iterations)
    return status, distance, iterations


def union_query(
    pieces_a: list[tuple[str, np.ndarray]],
    pieces_b: list[tuple[str, np.ndarray]],
    rotation: np.ndarray,
    translation: np.ndarray,
    tol: float = 1e-8,
) -> dict:
    """Minimum separation over all piece pairs, or an intersection, with the offending pair named.

    ``status == "failed"`` if any piece pair failed to converge. The minimum over the pairs that
    did converge is still returned for diagnosis, but the caller must treat it as unknown.
    """
    moved_b = [(name, vertices @ rotation.T + translation) for name, vertices in pieces_b]
    best = float("inf")
    best_pair = ""
    failed = 0
    pairs = 0
    iterations = 0
    started = time.perf_counter()
    for name_a, vertices_a in pieces_a:
        for name_b, vertices_b in moved_b:
            pairs += 1
            status, distance, used = _gjk_pair(vertices_a, vertices_b, tol)
            iterations += used
            if status == "failed":
                failed += 1
                continue
            if status == "intersecting":
                return dict(
                    status="intersecting",
                    distance=0.0,
                    intersects=True,
                    pair=f"{name_a}|{name_b}",
                    failed=failed,
                    pairs=pairs,
                    iterations=iterations,
                    seconds=time.perf_counter() - started,
                )
            if distance < best:
                best = float(distance)
                best_pair = f"{name_a}|{name_b}"
    status = "failed" if failed else "separated"
    if not np.isfinite(best):
        status = "failed"
    return dict(
        status=status,
        distance=best if np.isfinite(best) else float("nan"),
        intersects=False,
        pair=best_pair,
        failed=failed,
        pairs=pairs,
        iterations=iterations,
        seconds=time.perf_counter() - started,
    )


def init_worker(pieces_dir: str | Path) -> None:
    """Build the static meshes once per worker process."""
    if _CACHE:
        return
    mesh3 = trimesh.load(str(MESH_DIR / "Link3.STL"), force="mesh", process=False)
    mesh5 = trimesh.load(str(MESH_DIR / "Link5.STL"), force="mesh", process=False)
    _CACHE["reference"] = (
        MeshRef(np.asarray(mesh3.vertices), np.asarray(mesh3.faces), "Link3"),
        MeshRef(np.asarray(mesh5.vertices), np.asarray(mesh5.faces), "Link5"),
    )
    hull3 = mesh3.convex_hull
    hull5 = mesh5.convex_hull
    _CACHE["hull"] = (
        MeshRef(np.asarray(hull3.vertices), np.asarray(hull3.faces), "Link3-hull"),
        MeshRef(np.asarray(hull5.vertices), np.asarray(hull5.faces), "Link5-hull"),
    )
    _CACHE["pieces"] = (
        load_pieces("cr10_Link3", pieces_dir),
        load_pieces("cr10_Link5", pieces_dir),
    )


def _query_reference(pair, rotation: np.ndarray, translation: np.ndarray, cap: float) -> dict:
    started = time.perf_counter()
    result = pair[0].closest_to_in_frame(pair[1], rotation, translation, max_clearance=cap)
    elapsed = time.perf_counter() - started
    exact = result.status == "exact"
    return dict(
        status=result.status,
        distance=float(result.clearance) if (exact or result.status == "certified_culled") else float("nan"),
        intersects=bool(result.intersects),
        witness_a=None if result.witness_a is None else np.asarray(result.witness_a).tolist(),
        witness_b=None if result.witness_b is None else np.asarray(result.witness_b).tolist(),
        triangle_a=int(result.triangle_a),
        triangle_b=int(result.triangle_b),
        reason=result.reason,
        seconds=elapsed,
    )


def evaluate_sample(payload: dict, cap: float = 0.06) -> dict:
    """Reference, single hull and convex-piece union for one relative Link5 -> Link3 transform."""
    rotation = np.asarray(payload["rotation"], dtype=np.float64)
    translation = np.asarray(payload["translation"], dtype=np.float64)

    reference = _query_reference(_CACHE["reference"], rotation, translation, cap)
    hull = _query_reference(_CACHE["hull"], rotation, translation, cap)
    pieces = union_query(_CACHE["pieces"][0], _CACHE["pieces"][1], rotation, translation)

    row = {key: value for key, value in payload.items()}
    row.update(
        reference_distance=reference["distance"],
        reference_status=reference["status"],
        reference_intersects=reference["intersects"],
        reference_reason=reference["reason"],
        reference_triangle_a=reference["triangle_a"],
        reference_triangle_b=reference["triangle_b"],
        reference_witness_a=reference["witness_a"],
        reference_witness_b=reference["witness_b"],
        reference_seconds=reference["seconds"],
        hull_distance=hull["distance"],
        hull_status=hull["status"],
        hull_intersects=hull["intersects"],
        hull_seconds=hull["seconds"],
        pieces_distance=pieces["distance"],
        pieces_status=pieces["status"],
        pieces_intersects=pieces["intersects"],
        pieces_pair=pieces["pair"],
        pieces_failed=pieces["failed"],
        pieces_pairs=pieces["pairs"],
        pieces_iterations=pieces["iterations"],
        pieces_seconds=pieces["seconds"],
    )
    return row
