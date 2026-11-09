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

"""Independent reward term using the PhysX-consistent B distance proxy.

This module is imported only by the independent ``RANGER-CR10-TELEOP-SIMPLECOLL`` config. It
does not modify the default ``safety.py``, default teleop rewards or deployment assets.

The distance source is the stable shape table exported from the frozen flat USD
(``mujoco/robots/ranger_cr10/phys_proxy_shapes.json``), evaluated with the existing
``scripts/ranger_cr10/phys_proxy`` convex-hull query. The reward shape is the existing
quadratic margin penalty: ``-(max(0, margin - d) / margin)^2``, with a 20 mm margin in the
independent config.

Failure policy: a non-finite/unknown pair is treated as distance 0 (contact) and counted; it is
never silently converted to a safe distance. The fail count and per-pair snapshot are exposed
through ``phys_proxy_snapshot`` for the rollout report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Locate the repository's existing proxy implementation without copying it into the extension.
_ROOT = None
for _parent in Path(__file__).resolve().parents:
    if (_parent / "scripts" / "ranger_cr10" / "phys_proxy.py").is_file():
        _ROOT = _parent
        break
if _ROOT is None:
    raise ImportError("scripts/ranger_cr10/phys_proxy.py not found above the extension")
_SCRIPTS = _ROOT / "scripts" / "ranger_cr10"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from phys_proxy import (  # noqa: E402
    MARGIN,
    active_pairs,
    load_shapes_stable,
    reference_group_distance,
    shapes_by_group,
)

ARM_JOINT_NAMES = tuple(f"cr10_joint{index}" for index in range(1, 7))
_CACHE: dict = {"model": None, "data": None, "shapes": None, "pairs": None}


def _ensure_loaded():
    if _CACHE["model"] is not None:
        return
    import mujoco

    from phys_proxy import V2_SCENE

    shapes = shapes_by_group(load_shapes_stable())
    model = mujoco.MjModel.from_xml_path(str(V2_SCENE))
    data = mujoco.MjData(model)
    _CACHE.update(model=model, data=data, shapes=shapes, pairs=active_pairs())


def _joint_columns(asset) -> list[int]:
    names = list(asset.joint_names)
    missing = [name for name in ARM_JOINT_NAMES if name not in names]
    if missing:
        raise KeyError(f"arm joints missing from asset: {missing}")
    return [names.index(name) for name in ARM_JOINT_NAMES]


def compute_pair_distances(q: np.ndarray, tol: float = 1e-6):
    """Return (N, P) distances, the pair list, and the per-entry query statuses.

    ``q`` is (N, 6) arm joint positions in radians. The base is held at the model home; the
    arm/protected distances are invariant to the base world pose and depend only on q.
    """
    import mujoco

    _ensure_loaded()
    model, data, shapes = _CACHE["model"], _CACHE["data"], _CACHE["shapes"]
    pairs = _CACHE["pairs"]
    values = np.full((len(q), len(pairs)), np.nan, dtype=np.float64)
    records = []
    for env_index, row in enumerate(q):
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        for index, name in enumerate(ARM_JOINT_NAMES):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[joint]] = float(row[index])
        mujoco.mj_forward(model, data)
        env_records = []
        for pair_index, (first, second) in enumerate(pairs):
            result = reference_group_distance(shapes, first, second, data, tol=tol)
            distance = float(result["distance"])
            status = result["status"]
            if status in ("max_iterations", "degenerate", "unknown") or not np.isfinite(distance):
                # Never turn a failed/approximate query into a safe distance. Contact is the
                # conservative finite fallback; the failure counter is reported separately.
                distance = 0.0
                status = "unknown_as_contact"
            elif status == "contact":
                distance = 0.0
            values[env_index, pair_index] = distance
            env_records.append({
                "pair": [first, second],
                "distance": distance,
                "status": status,
                "shape_a": result.get("shape_a"),
                "shape_b": result.get("shape_b"),
            })
        records.append(env_records)
    return values, pairs, records


def cached_pair_distances(env, asset):
    """Compute once per environment step; return (N, P) tensor and metadata."""
    step = int(env.common_step_counter)
    cache = getattr(env, "_phys_proxy_cache", None)
    if cache is not None and cache["step"] == step:
        return cache["tensor"], cache["pairs"], cache["records"], cache["failures"]
    columns = _joint_columns(asset)
    q = asset.data.joint_pos[:, columns].detach().cpu().numpy()
    values, pairs, records = compute_pair_distances(q)
    failures = sum(
        1 for env_records in records for record in env_records
        if record["status"] in ("unknown", "unknown_as_contact", "max_iterations", "degenerate")
    )
    tensor = torch.as_tensor(values, dtype=torch.float32, device=asset.data.joint_pos.device)
    env._phys_proxy_cache = {
        "step": step, "tensor": tensor, "pairs": pairs, "records": records, "failures": failures,
    }
    env._phys_proxy_last = {
        "step": step, "tensor": tensor, "pairs": pairs, "records": records, "failures": failures,
    }
    return tensor, pairs, records, failures


def phys_proxy_clearance_penalty(env, asset_cfg, margin: float = 0.02, cap: float = 0.15):
    """Reward term: zero while clear, quadratic inside ``margin``; contact gives -1."""
    asset = env.scene[asset_cfg.name]
    distance, _, _, _ = cached_pair_distances(env, asset)
    distance = torch.minimum(distance, torch.full_like(distance, cap)).amin(dim=-1)
    excess = (margin - distance).clamp(min=0.0) / margin
    return -(excess**2)


def phys_proxy_snapshot(env, asset_cfg):
    """Return per-pair aggregates for the rollout report (mean/min/counts)."""
    asset = env.scene[asset_cfg.name]
    last = getattr(env, "_phys_proxy_last", None)
    if last is not None and last.get("tensor") is not None:
        tensor, pairs, records, failures = last["tensor"], last["pairs"], last["records"], last["failures"]
    else:
        tensor, pairs, records, failures = cached_pair_distances(env, asset)
    distances = tensor.detach().cpu().numpy()
    out = []
    for index, pair in enumerate(pairs):
        column = distances[:, index]
        out.append({
            "pair": list(pair),
            "mean_m": float(np.mean(column)),
            "min_m": float(np.min(column)),
            "count_margin_20mm": int(np.sum(column <= MARGIN)),
            "count_contact": int(np.sum(column <= 1e-6)),
        })
    return out, failures, records


def reset_cache(env) -> None:
    for name in ("_phys_proxy_cache", "_phys_proxy_last"):
        if hasattr(env, name):
            delattr(env, name)
