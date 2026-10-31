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

"""Stage 4.6: prove that a teleop key no longer changes the MuJoCo picture.

The viewer's shortcuts are bound to the teleop letters (W = wireframe, S = shadow, 1-6 = geom
groups). `ViewerFocusGuard` keeps the focus off the viewer window so it never receives them,
and `ViewerKeyPin` restores the option flags it could still reach. This measures the
*picture*, not the API, because that is what an operator actually sees:

* **without** the guards -- send a real key and expect a large fraction of the window's
  pixels to change (the failure they exist to prevent);
* **with** the guards -- the same key must leave the picture pixel-identical.

Screenshots use ``ffmpeg -f x11grab`` because ImageMagick is not installed here.

Usage:
    python scripts/ranger_cr10/check_viewer_key_pinning.py
    python scripts/ranger_cr10/check_viewer_key_pinning.py --display :1 --size 1280x800
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
from PIL import Image

import mujoco

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mujoco" / "deploy"))
from deploy_mujoco.ranger_cr10.ranger_cr10_deploy import ViewerFocusGuard, ViewerKeyPin  # noqa: E402

ap = argparse.ArgumentParser(description="Does a teleop key change the viewer's picture?")
ap.add_argument("--display", default=":1")
ap.add_argument("--size", default="1280x800")
ap.add_argument("--key", default="w")
ap.add_argument("--threshold", type=float, default=1.0, help="percent of pixels that counts as a change")
args = ap.parse_args()


def screenshot(tag: str) -> np.ndarray:
    path = Path(tempfile.gettempdir()) / f"viewer_key_{tag}.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "x11grab",
            "-video_size",
            args.size,
            "-i",
            args.display,
            "-frames:v",
            "1",
            str(path),
        ],
        check=True,
    )
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def changed_percent(a: np.ndarray, b: np.ndarray) -> float:
    return float((np.abs(a - b) > 8).mean() * 100.0)


def run(guarded: bool) -> float:
    from pynput.keyboard import Controller

    m = mujoco.MjModel.from_xml_path(str(ROOT / "mujoco" / "robots" / "ranger_cr10" / "scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)

    pin = ViewerKeyPin(enabled=guarded)
    guard = ViewerFocusGuard(enabled=guarded)
    with mujoco.viewer.launch_passive(m, d, key_callback=pin.key_callback) as v:
        pin.attach(v)
        guard.on_viewer_open()
        time.sleep(1.5)
        kb = Controller()
        before = screenshot("before")
        kb.press(args.key)
        kb.release(args.key)
        time.sleep(0.8)
        after = screenshot("after")
    return changed_percent(before, after)


def main() -> None:
    raw = run(guarded=False)
    fixed = run(guarded=True)
    print(f"无保护: 按 '{args.key}' 后画面变化 {raw:.1f}% 的像素")
    print(f"加保护: 按 '{args.key}' 后画面变化 {fixed:.1f}% 的像素")
    print()
    reachable = raw > args.threshold
    fixed_ok = fixed <= args.threshold
    print(f"1. 这个键确实会改画面(无保护时): {'是' if reachable else '未观察到 -- 窗口没有拿到焦点?'}")
    print(f"2. 加保护后不再改画面: {'通过' if fixed_ok else '**失败**'}")
    if reachable and fixed_ok:
        print(f"\n结论: 保护生效, 画面变化 {raw:.1f}% -> {fixed:.1f}%")
    sys.exit(0 if (fixed_ok and reachable) else 1)


if __name__ == "__main__":
    main()
