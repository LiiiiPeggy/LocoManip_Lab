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

"""Flatten the converted simple-collision USD into a standalone, non-instanceable stage.

PhysX filtered-pair relationships must target paths that exist at the simulation
scope. The URDF converter writes a small wrapper that references configuration
layers; flattening resolves the references and instancing once, so the collider
paths are stable and can carry body-level filters.

Run:
    python scripts/ranger_cr10/flatten_simple_usd.py \
        --input <ranger_cr10_simple_v2.usd> --output <ranger_cr10_simple_v2_flat.usd>
"""

from __future__ import annotations

import argparse
import hashlib
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
args = ap.parse_args()

app = AppLauncher(headless=True).app

from pxr import Usd, UsdPhysics  # noqa: E402

stage = Usd.Stage.Open(args.input, load=Usd.Stage.LoadAll)
if stage is None:
    raise SystemExit(f"could not open {args.input}")
flat_layer = stage.Flatten()
os.makedirs(os.path.dirname(args.output), exist_ok=True)
flat_layer.Export(args.output)

check = Usd.Stage.Open(args.output, load=Usd.Stage.LoadAll)
bodies = sum(1 for prim in check.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI))
colliders = sum(1 for prim in check.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI))
sha = hashlib.sha256(open(args.output, "rb").read()).hexdigest()
print(
    {
        "input": os.path.abspath(args.input),
        "output": os.path.abspath(args.output),
        "bytes": os.path.getsize(args.output),
        "rigid_bodies": bodies,
        "colliders": colliders,
        "sha256": sha,
        "default_prim": check.GetDefaultPrim().GetPath().pathString if check.GetDefaultPrim() else None,
    },
    flush=True,
)
app.close()
