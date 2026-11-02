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

"""Stage 4.8, task 3: how much does the position reward care about 20 mm versus 300 mm?

The task's position reward is ``w * exp(-d^2 / s^2)`` with ``s^2 = 0.1`` (``std =
sqrt(0.1) = 0.316 m``). Its derivative peaks at ``d = s/sqrt(2) = 0.22 m`` and decays to
zero at both ends, which is why a policy can sit at 30-50 mm error and feel almost no
pull to close the last few centimetres while caring strongly about 100 mm versus 200 mm.

This prints the numbers the change is judged on -- reward, gradient, and the *reward
gained by closing 10 mm* at the distances the 4.8 brief names -- for the current kernel
and for the alternatives. Pure arithmetic, no Isaac import, so it runs anywhere.

Usage:
    python scripts/ranger_cr10/check_position_reward.py
    python scripts/ranger_cr10/check_position_reward.py --dists 0.02 0.05 0.08 0.15 0.30
"""

from __future__ import annotations

import argparse
import math

ap = argparse.ArgumentParser(description="Position reward sensitivity table.")
ap.add_argument("--weight", type=float, default=4.5, help="current weight of the position term")
ap.add_argument("--std", type=float, default=math.sqrt(0.1), help="current std of the position kernel")
ap.add_argument("--dists", type=float, nargs="+", default=[0.02, 0.05, 0.08, 0.15, 0.30])
ap.add_argument("--close", type=float, default=0.01, help="error reduction whose reward is reported (m)")
ap.add_argument("--fine_weight", type=float, default=0.5, help="candidate precision term weight")
ap.add_argument("--fine_std", type=float, default=0.10, help="candidate precision term std (m)")
args = ap.parse_args()


def kernel(d, w, s):
    return w * math.exp(-(d**2) / (s**2))


def grad(d, w, s):
    """d(reward)/dd -- negative: reward grows as the error shrinks."""
    return -2.0 * w * d / (s**2) * math.exp(-(d**2) / (s**2))


def closing(d, w, s, step):
    """Reward gained by shrinking the error from d to d - step."""
    return kernel(d - step, w, s) - kernel(d, w, s)


def main() -> None:
    w, s = args.weight, args.std
    print(f"当前: reward = {w} * exp(-d^2 / {s**2:.3f})   (std = {s:.3f} m)")
    print(f"      梯度峰值在 d = {s / math.sqrt(2) * 1000:.0f} mm; 0 处和远处都趋于 0\n")

    print(f"{'误差 d':>8}{'reward':>10}{'dR/dd (1/m)':>15}{f'收拢 {args.close * 1000:.0f} mm 得到':>18}{'占满分':>10}")
    full = w
    for d in args.dists:
        g = grad(d, w, s)
        c = closing(d, w, s, args.close)
        print(f"{d * 1000:>7.0f}mm{kernel(d, w, s):>10.3f}{g:>15.2f}{c:>18.4f}{c / full * 100:>9.2f}%")

    print("\n候选: 保持粗项不动, 加一个窄核精度项 (不改动原有权重)")
    print(f"      + {args.fine_weight} * exp(-d^2 / {args.fine_std**2:.3f})   (std = {args.fine_std:.2f} m)\n")
    print(
        f"{'误差 d':>8}{'旧 reward':>11}{'新 reward':>11}{'旧 收拢':>10}{'新 收拢':>10}{'梯度倍数':>10}{'远处保留':>10}"
    )
    for d in args.dists:
        old_r, new_r = kernel(d, w, s), kernel(d, w, s) + kernel(d, args.fine_weight, args.fine_std)
        old_c = closing(d, w, s, args.close)
        new_c = old_c + closing(d, args.fine_weight, args.fine_std, args.close)
        old_g, new_g = grad(d, w, s), grad(d, w, s) + grad(d, args.fine_weight, args.fine_std)
        frac = math.exp(-(d**2) / (s**2))  # how much of the far-field reward survives unchanged
        print(
            f"{d * 1000:>7.0f}mm{old_r:>11.3f}{new_r:>11.3f}{old_c:>10.4f}{new_c:>10.4f}"
            f"{new_g / old_g:>10.2f}x{frac * 100:>9.0f}%"
        )
    del full


if __name__ == "__main__":
    main()
