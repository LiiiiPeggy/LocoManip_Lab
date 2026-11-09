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

"""Unit checks for the B-distance reward term, without booting Isaac Sim.

Checks the units/sign, union-min aggregation, pair count, contact zero and the quadratic margin
formula on two frozen rows. The actual env integration is exercised by
``rollout_phys_proxy_policy.py``.

Run:
    python scripts/ranger_cr10/check_phys_proxy_reward.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp"))

from phys_proxy_reward import compute_pair_distances  # noqa: E402

NEAR = ROOT / "outputs/stage413/b/radar_near_contact/v3/near_set_containment_v3.json"


def main() -> None:
    rows = {row["row_id"]: row for row in json.loads(NEAR.read_text())["rows"]}
    q = np.asarray([rows["D0106"]["q"], rows["D0000"]["q"]], dtype=np.float64)
    values, pairs, records = compute_pair_distances(q)
    index = {tuple(pair): i for i, pair in enumerate(pairs)}
    link3_lidar = values[0, index[("cr10_Link3", "lidar")]]
    min0 = float(values[0].min())
    min0_pair = pairs[int(values[0].argmin())]
    contact = float(values[1].min())
    failures = sum(1 for env_records in records for record in env_records if "unknown" in record["status"])

    def reward(distance: float, margin: float = 0.02) -> float:
        excess = max(0.0, margin - min(distance, 0.15)) / margin
        return -(excess**2)

    result = {
        "pairs": len(pairs),
        "failures": failures,
        "D0106_link3_lidar_m": float(link3_lidar),
        "D0106_min_m": min0,
        "D0106_min_pair": list(min0_pair),
        "D0106_reward": reward(min0),
        "D0000_min_m": contact,
        "D0000_reward": reward(contact),
        "checks": {
            "pair_count_44": bool(len(pairs) == 44),
            "link3_lidar_matches_B": bool(abs(link3_lidar - 0.04852374414816252) < 1e-8),
            "min_is_link3_link5": bool(min0_pair == ("cr10_Link3", "cr10_Link5")),
            "contact_is_zero": bool(contact == 0.0),
            "contact_reward_is_minus_one": bool(abs(reward(contact) + 1.0) < 1e-12),
            "finite": bool(np.isfinite(values).all()),
        },
    }
    result["passed"] = all(bool(value) for value in result["checks"].values()) and failures == 0
    print(json.dumps(result, indent=1))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
