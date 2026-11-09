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

"""Export the PhysX-consistent shape table for the runtime reward.

Run after any collider change:
    python scripts/ranger_cr10/export_phys_proxy_shapes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from phys_proxy import STABLE_SHAPES, export_shapes  # noqa: E402

shapes = export_shapes()
print(json.dumps({"shapes": len(shapes), "out": str(STABLE_SHAPES)}, indent=1))
