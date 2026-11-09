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

"""Minimal fault injection for atomic GJK failures in ``reference_group_distance``.

Four cases:
    1. normal separated + failed atomic query;
    2. failed atomic query followed by a confirmed contact;
    3. all atomic queries separated;
    4. all atomic queries failed.

The injected query results are synthetic; this checks the union aggregation, not geometry. The
existing union regression remains in ``check_phys_proxy_union.py``.

Run:
    python scripts/ranger_cr10/check_phys_proxy_failure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import convex_distance  # noqa: E402
import phys_proxy  # noqa: E402
from phys_proxy import ROOT, Shape, reference_group_distance  # noqa: E402

OUT = ROOT / "outputs/stage413/b/simple_collision_v2/phys_proxy_failure_regression.json"


def shape(name: str) -> Shape:
    return Shape(
        name=name, body="synthetic", group="synthetic", part="synthetic", kind="box",
        points=np.zeros((1, 3)), pos=np.zeros(3), quat=np.array([1.0, 0.0, 0.0, 0.0]),
        half=None, mesh_file=None, mesh_scale=None, source_path="synthetic", approximation=None,
    )


def run_case(name: str, results: list[convex_distance.GjkResult]) -> dict:
    queue = list(results)
    calls = {"count": 0}

    def fake_gjk(points_a, points_b, tol=1e-8, max_iterations=64):
        index = calls["count"]
        calls["count"] += 1
        return queue[min(index, len(queue) - 1)]

    original = convex_distance.gjk_distance
    original_world_points = phys_proxy.world_points
    convex_distance.gjk_distance = fake_gjk
    phys_proxy.world_points = lambda item, data: item.points
    try:
        result = reference_group_distance(
            {"A": [shape("A1"), shape("A2")], "B": [shape("B1")]}, "A", "B", None
        )
    finally:
        convex_distance.gjk_distance = original
        phys_proxy.world_points = original_world_points
    return {"name": name, "result": result, "calls": calls["count"]}


def main() -> None:
    separated = convex_distance.GjkResult(0.05, "separated", 1)
    separated_2 = convex_distance.GjkResult(0.03, "separated", 1)
    failed = convex_distance.GjkResult(float("nan"), "max_iterations", 64)
    contact = convex_distance.GjkResult(0.0, "intersecting", 1)

    cases = [
        run_case("separated_plus_failure", [separated, failed]),
        run_case("failure_then_contact", [failed, contact]),
        run_case("all_separated", [separated, separated_2]),
        run_case("all_failed", [failed, failed]),
    ]
    expected = {
        "separated_plus_failure": {"status": "unknown", "distance": 0.0, "failure_count": 1},
        "failure_then_contact": {"status": "contact", "distance": 0.0, "failure_count": 1},
        "all_separated": {"status": "separated", "distance": 0.03, "failure_count": 0},
        "all_failed": {"status": "unknown", "distance": 0.0, "failure_count": 2},
    }
    checks = []
    for case in cases:
        result = case["result"]
        want = expected[case["name"]]
        failures = result.get("failures", [])
        checks.append({
            "name": case["name"],
            "status": result["status"],
            "distance": result["distance"],
            "failure_count": len(failures),
            "failed_shapes": [item.get("shape_a") for item in failures],
            "atomic_queries": result.get("atomic_queries"),
            "expected": want,
            "passed": (
                result["status"] == want["status"]
                and abs(result["distance"] - want["distance"]) < 1e-12
                and len(failures) == want["failure_count"]
            ),
        })
    # Reward-layer propagation: the unknown result must become distance 0 and be counted.
    import importlib.util

    reward_path = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/phys_proxy_reward.py"
    spec = importlib.util.spec_from_file_location("phys_proxy_reward_check", reward_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {reward_path}")
    reward = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reward)
    target_pair = ("cr10_Link3", "cr10_Link5")

    def fake_reward_distance(shapes, first, second, data, tol=1e-8):
        if (first, second) == target_pair:
            return {
                "distance": 0.0, "status": "unknown", "shape_a": "A", "shape_b": "B",
                "atomic_queries": 2,
                "failures": [{"shape_a": "A", "shape_b": "B", "status": "max_iterations"}],
            }
        return {
            "distance": 0.05, "status": "separated", "shape_a": "A", "shape_b": "B",
            "atomic_queries": 1, "failures": [],
        }

    reward.reference_group_distance = fake_reward_distance
    asset = SimpleNamespace(
        joint_names=["cr10_joint1", "cr10_joint2", "cr10_joint3",
                     "cr10_joint4", "cr10_joint5", "cr10_joint6"],
        data=SimpleNamespace(joint_pos=torch.zeros(1, 6)),
    )
    env = SimpleNamespace(common_step_counter=0)
    tensor, pairs, records, failure_count = reward.cached_pair_distances(env, asset)
    pair_index = pairs.index(target_pair) if target_pair in pairs else pairs.index(list(target_pair))
    target_record = next(
        item for item in records[0] if tuple(item["pair"]) == target_pair
    )
    reward_check = {
        "target_pair": list(target_pair),
        "tensor_distance": float(tensor[0, pair_index].item()),
        "record_status": target_record["status"],
        "record_distance": target_record["distance"],
        "atomic_failure_count": len(target_record.get("atomic_failures", [])),
        "cached_failure_count": failure_count,
        "passed": (
            float(tensor[0, pair_index].item()) == 0.0
            and target_record["status"] == "unknown_as_contact"
            and target_record["distance"] == 0.0
            and len(target_record.get("atomic_failures", [])) == 1
            and failure_count == 1
        ),
    }
    payload = {
        "passed": all(item["passed"] for item in checks) and reward_check["passed"],
        "cases": checks,
        "reward_layer": reward_check,
        "note": "Synthetic GJK results; checks aggregation/unknown propagation only, not geometry.",
        "union_regression": "run scripts/ranger_cr10/check_phys_proxy_union.py separately",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
