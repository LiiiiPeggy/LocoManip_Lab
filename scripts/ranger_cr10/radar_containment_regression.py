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

"""Deterministic containment regressions for MeshRef after the transform fix.

Cases: nonzero rotation/translation, swapped query, both meshes transformed by the same rigid
transform, a multi-component guest where only one component is inside, and a non-watertight host.
The multi-component case is the one a first-vertex-only test misses. A non-watertight host must
report containment as unknown, not as separation.

Run:
    python scripts/ranger_cr10/radar_containment_regression.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from collision_parts import box_mesh  # noqa: E402
from reference_geometry import MeshRef  # noqa: E402

OUT = HERE.parents[1] / "outputs/stage413/b/radar_round2_2026_11_07/containment_regression.json"


def make_box(centre, half):
    return box_mesh(centre, half)


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def query(host, guest, rotation, translation):
    result = host.closest_to_in_frame(guest, rotation, translation, max_clearance=0.05)
    return dict(status=result.status, intersects=bool(result.intersects), clearance=float(result.clearance),
                reason=result.reason, containment_unknown=bool(result.extra.get("containment_unknown", False)),
                component_points_host=len(host.component_points), component_points_guest=len(guest.component_points),
                watertight_host=host.watertight, watertight_guest=guest.watertight)


def main() -> None:
    outer_v, outer_f = make_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    inner_v, inner_f = make_box((5.0, 0.0, 0.0), (0.1, 0.1, 0.1))
    outer = MeshRef(outer_v, outer_f, "outer")
    inner = MeshRef(inner_v, inner_f, "inner")

    R = rot_z(0.37)
    t = -R @ np.array([5.0, 0.0, 0.0])
    transformed = query(outer, inner, R, t)
    swapped = query(inner, outer, R.T, -R.T @ t)

    # same rigid transform on both meshes: inner starts at the outer centre, then both move
    inner0_v, inner0_f = make_box((0.0, 0.0, 0.0), (0.1, 0.1, 0.1))
    shift = np.array([2.0, -1.0, 0.5])
    outer_t = MeshRef(outer_v @ R.T + shift, outer_f, "outer_t")
    inner_t = MeshRef(inner0_v @ R.T + shift, inner0_f, "inner_t")
    same_transform = query(outer_t, inner_t, np.eye(3), np.zeros(3))

    # multi-component guest: first component far, second component inside the host
    far_v, far_f = make_box((5.0, 0.0, 0.0), (0.1, 0.1, 0.1))
    inside_v, inside_f = make_box((0.0, 0.0, 0.0), (0.1, 0.1, 0.1))
    multi_v = np.vstack([far_v, inside_v])
    multi_f = np.vstack([far_f, inside_f + len(far_v)])
    multi = MeshRef(multi_v, multi_f, "multi")
    multi_query = query(outer, multi, np.eye(3), np.zeros(3))

    # non-watertight host: remove one face from the outer box
    open_v, open_f = make_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    open_host = MeshRef(open_v, open_f[:-1], "open_outer")
    open_query = query(open_host, inner, R, t)

    cases = {
        "nonzero_rotation_translation_containment": transformed,
        "swapped_query_containment": swapped,
        "same_rigid_transform_both": same_transform,
        "multi_component_one_inside": multi_query,
        "nonwatertight_host_unknown": open_query,
    }
    checks = {
        "nonzero_rotation_translation_containment": transformed["intersects"] and transformed["reason"].startswith("containment"),
        "swapped_query_containment": swapped["intersects"] and swapped["reason"].startswith("containment"),
        "same_rigid_transform_both": same_transform["intersects"] and same_transform["reason"].startswith("containment"),
        "multi_component_one_inside": multi_query["intersects"] and multi_query["reason"].startswith("containment"),
        "nonwatertight_host_unknown": (not open_query["intersects"]) and open_query["containment_unknown"],
    }
    output = dict(cases=cases, checks=checks, passed=all(checks.values()),
                  note="The multi-component case would fail a first-vertex-only containment test. "
                       "A non-watertight host is flagged containment_unknown, not silently separated.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=1))
    print(json.dumps(output, indent=1))
    print(f"wrote {OUT}")
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
