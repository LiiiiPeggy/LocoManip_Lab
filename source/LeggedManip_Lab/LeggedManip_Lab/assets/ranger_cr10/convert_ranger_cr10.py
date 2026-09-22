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

"""Convert the RangerBox-CR10 URDF to USD, then print the resulting joint table.

Isaac Lab's ``scripts/tools/convert_urdf.py`` only exposes a subset of the URDF
importer's settings, so this driver reads the sibling ``config.yaml`` (same field
names as ``UrdfConverterCfg``) and runs the converter directly.

The printed joint table is the acceptance check for stage 0 of
``docs/plans/plan.md``: the chassis (4 steering + 4 wheel) and arm (6) joints must
all be present, and the arm limits must equal those in
``rangerboxcr10lidar_description/urdf/rangercr10lidar.urdf``.

Usage (from the repository root):

    python source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10/convert_ranger_cr10.py
"""

from __future__ import annotations

import os

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import yaml  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

ASSET_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ASSET_DIR, "config.yaml")


def build_cfg() -> UrdfConverterCfg:
    """Map the YAML config onto a UrdfConverterCfg, resolving paths to this directory."""
    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f)

    asset_path = raw.pop("asset_path", None)
    usd_dir = raw.pop("usd_dir", None)
    usd_file_name = raw.pop("usd_file_name", None)
    joint_drive = raw.pop("joint_drive", None)

    if joint_drive is not None:
        gains = joint_drive.pop("gains", None)
        if gains is not None:
            joint_drive["gains"] = UrdfConverterCfg.JointDriveCfg.PDGainsCfg(**gains)
        joint_drive = UrdfConverterCfg.JointDriveCfg(**joint_drive)

    return UrdfConverterCfg(
        asset_path=os.path.join(ASSET_DIR, asset_path),
        usd_dir=os.path.join(ASSET_DIR, usd_dir) if usd_dir else ASSET_DIR,
        usd_file_name=usd_file_name,
        joint_drive=joint_drive,
        **raw,
    )


def print_joint_table(usd_path: str) -> None:
    """Print every joint in the converted USD with its type and limits."""
    stage = Usd.Stage.Open(usd_path)
    rows = []
    for prim in stage.Traverse():
        for usd_type, kind in ((UsdPhysics.RevoluteJoint, "revolute"), (UsdPhysics.PrismaticJoint, "prismatic")):
            if not prim.IsA(usd_type):
                continue
            lo = prim.GetAttribute("physics:lowerLimit")
            hi = prim.GetAttribute("physics:upperLimit")
            rows.append(
                (
                    prim.GetName(),
                    kind,
                    "-" if not lo or not lo.HasValue() else f"{lo.Get():.5f}",
                    "-" if not hi or not hi.HasValue() else f"{hi.Get():.5f}",
                )
            )
    rows.sort()
    print(f"\n{'joint':<34}{'type':<11}{'lower':>12}{'upper':>12}")
    print("-" * 69)
    for name, kind, lo, hi in rows:
        print(f"{name:<34}{kind:<11}{lo:>12}{hi:>12}")
    print(f"\n合计 {len(rows)} 个关节")


if __name__ == "__main__":
    cfg = build_cfg()
    converter = UrdfConverter(cfg)
    print(f"[convert] URDF : {cfg.asset_path}")
    print(f"[convert] USD  : {converter.usd_path}")
    print_joint_table(converter.usd_path)
    simulation_app.close()
