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

"""Export the v1/v2 collision-box table with the joint-end retreats.

Run:
    python scripts/ranger_cr10/export_box_table_v2.py
"""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10_simple/ranger_cr10_simple.urdf"
V2 = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10_simple_v2/ranger_cr10_simple_v2.urdf"
OUT = ROOT / "outputs/stage413/b/simple_collision_v2"
LINKS = [f"cr10_Link{i}" for i in range(1, 7)]

# Proximal / distal joint centres in each link frame, from the URDF joint origins.
JOINTS = {
    "cr10_Link1": {"proximal": [0.0, 0.0, 0.0], "distal": [0.0, 0.0, 0.0], "axis": "z"},
    "cr10_Link2": {"proximal": [0.0, 0.0, 0.0], "distal": [-0.607, 0.0, 0.0], "axis": "x"},
    "cr10_Link3": {"proximal": [0.0, 0.0, 0.0], "distal": [-0.568, 0.0, 0.191], "axis": "x (shaft)"},
    "cr10_Link4": {"proximal": [0.0, 0.0, 0.0], "distal": [0.0, -0.125, 0.0], "axis": "y"},
    "cr10_Link5": {"proximal": [0.0, 0.0, 0.0], "distal": [0.0, 0.1084, 0.0], "axis": "y"},
    "cr10_Link6": {"proximal": [0.0, 0.0, 0.0], "distal": None, "axis": "z (tool)"},
}


def parse(path):
    root = ET.parse(path).getroot()
    out = {}
    for link in LINKS + ["box_link"]:
        element = root.find(f"link[@name='{link}']")
        records = []
        if element is None:
            out[link] = records
            continue
        for collision in element.findall("collision"):
            box = collision.find("./geometry/box")
            if box is None:
                continue
            origin = collision.find("origin")
            xyz = [float(v) for v in (origin.get("xyz", "0 0 0").split() if origin is not None else [0, 0, 0])]
            records.append({"center": xyz, "size": [float(v) for v in box.get("size").split()]})
        out[link] = records
    return out


def ends(link, box):
    if link not in JOINTS:
        return {}
    joint = JOINTS[link]
    center = box["center"]
    size = box["size"]
    axis = joint["axis"][0]
    index = {"x": 0, "y": 1, "z": 2}[axis]
    low = center[index] - size[index] / 2
    high = center[index] + size[index] / 2
    result = {"box_axis_min": low, "box_axis_max": high}
    proximal = joint["proximal"][index]
    distal = joint["distal"][index] if joint["distal"] is not None else None
    result["proximal_joint_coordinate"] = proximal
    result["box_min_minus_proximal_m"] = low - proximal
    result["box_max_minus_proximal_m"] = high - proximal
    if distal is not None:
        result["distal_joint_coordinate"] = distal
        result["box_min_minus_distal_m"] = low - distal
        result["box_max_minus_distal_m"] = high - distal
    return result


v1 = parse(V1)
v2 = parse(V2)
rows = []
for link in LINKS + ["box_link"]:
    for index, (box1, box2) in enumerate(zip(v1.get(link, []), v2.get(link, []))):
        row = {
            "link": link,
            "index": index,
            "v1_center": box1["center"],
            "v1_size": box1["size"],
            "v2_center": box2["center"],
            "v2_size": box2["size"],
            "axis": JOINTS.get(link, {}).get("axis", "link frame"),
        }
        if link in JOINTS:
            row["v2_ends"] = ends(link, box2)
            row["v1_ends"] = ends(link, box1)
        rows.append(row)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "box_table_v2.json").write_text(json.dumps({"v1": str(V1.relative_to(ROOT)), "v2": str(V2.relative_to(ROOT)),
                                                  "rows": rows}, indent=1))
with (OUT / "box_table_v2.csv").open("w", newline="") as handle:
    fieldnames = [
        "link", "index", "axis",
        "v1_center_x", "v1_center_y", "v1_center_z", "v1_size_x", "v1_size_y", "v1_size_z",
        "v2_center_x", "v2_center_y", "v2_center_z", "v2_size_x", "v2_size_y", "v2_size_z",
        "v2_box_min_minus_proximal_m", "v2_box_max_minus_proximal_m",
        "v2_box_min_minus_distal_m", "v2_box_max_minus_distal_m",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        ends_v2 = row.get("v2_ends", {})
        entry = {
            "link": row["link"], "index": row["index"], "axis": row["axis"],
            "v2_box_min_minus_proximal_m": ends_v2.get("box_min_minus_proximal_m"),
            "v2_box_max_minus_proximal_m": ends_v2.get("box_max_minus_proximal_m"),
            "v2_box_min_minus_distal_m": ends_v2.get("box_min_minus_distal_m"),
            "v2_box_max_minus_distal_m": ends_v2.get("box_max_minus_distal_m"),
        }
        for prefix, values in (("v1_center", row["v1_center"]), ("v1_size", row["v1_size"]),
                               ("v2_center", row["v2_center"]), ("v2_size", row["v2_size"])):
            for axis, value in zip("xyz", values):
                entry[f"{prefix}_{axis}"] = value
        writer.writerow(entry)
print(json.dumps(rows, indent=1))
print(f"wrote {OUT / 'box_table_v2.json'} and {OUT / 'box_table_v2.csv'}")
