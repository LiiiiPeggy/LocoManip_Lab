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

"""Stage 4.11: contact and accuracy, split by whether the target *can* be reached safely.

The 4.10e evaluation reports one contact rate over 60 targets. That number mixes two very
different situations: a target some collision-free configuration can hold (so a contact is the
policy's doing) and a target with no collision-free solution at all (so a contact is the
geometry's doing). `check_self_collision.py` measured 49/60 collision-free reachable on the
4.10 box, which means up to a fifth of the sweep cannot be executed without contact and a
single rate cannot say whether the policy became safer or just slower.

This runs the same target set, the same seed and the same hold as `coverage_suite`, records the
contact pairs and the duration of every contact for every target, and prints both categories
side by side. A policy that improves the first column without touching the second has learned
something; one that improves both by standing still has not.

Usage:
    python scripts/ranger_cr10/eval_contact_by_target.py --policy <ckpt> --tag iter1500
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "mujoco" / "deploy" / "deploy_mujoco"
for p in (
    str(ROOT / "mujoco/deploy/deploy_mujoco/ranger_cr10"),
    str(DEPLOY),
    str(DEPLOY.parent),
    str(ROOT / "scripts/ranger_cr10"),
):
    if p not in sys.path:
        sys.path.append(p)

import mujoco  # noqa: E402
import teleop_kinematics as tk  # noqa: E402
from base_command_filter import BaseCommandFilter  # noqa: E402
from deploy_mujoco.ranger_cr10.ranger_policy import ee_command_from_offset  # noqa: E402
from teleop_eval import Sim, _collision_free_oracle, _reach_oracle  # noqa: E402
import workspace_limits as WS  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--policy", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", default="outputs/stage411/eval")
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--hold", type=int, default=30)
ap.add_argument("--seed", type=int, default=5)
ap.add_argument("--box", default="4.10")
args = ap.parse_args()

BOXES = {
    "candidate": WS.TARGET_BOX,
    "4.8": WS.TARGET_BOX_48,
    "4.10": WS.TARGET_BOX_410,
}


def contacts_of(sim: Sim) -> dict[str, float]:
    """Non-ground contacts in the current MuJoCo state, as {(bodyA, bodyB): depth}."""
    m, d = sim.m, sim.d
    out = {}
    for c in range(d.ncon):
        b1, b2 = m.geom_bodyid[d.contact[c].geom1], m.geom_bodyid[d.contact[c].geom2]
        if b1 == 0 or b2 == 0:
            continue
        a = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1)
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2)
        if a is None or b is None:
            continue
        key = tuple(sorted((a, b)))
        depth = float(d.contact[c].dist)
        if key not in out or depth < out[key]:
            out[key] = depth
    return out


def main() -> None:
    sim = Sim(args.policy, BaseCommandFilter())
    box = BOXES[args.box]
    reach = _reach_oracle()

    rng = np.random.default_rng(args.seed)
    targets = []
    for _ in range(args.n):
        dpos = [rng.uniform(*box[k]) for k in ("pos_x", "pos_y", "pos_z")]
        drpy = [rng.uniform(*box[k]) for k in ("roll", "pitch", "yaw")]
        targets.append((dpos, drpy))

    free = _collision_free_oracle(sim, targets)
    kin = [reach(dpos, drpy) for dpos, drpy in targets]

    rows = []
    for (dpos, drpy) in targets:
        sim.reset()
        sim.set_commands(0.0, 0.0, ee_command_from_offset(dpos, drpy))
        poses, per_step = [], []
        for _ in range(args.hold):
            poses.append(sim.step())
            per_step.append(contacts_of(sim))
        ee_pos = float(np.mean([p["ee_pos"] for p in poses[-5:]]))
        ee_ori = math.degrees(float(np.mean([p["ee_ori"] for p in poses[-5:]])))
        busy = [bool(c) for c in per_step]
        longest = cur = 0
        for b in busy:
            cur = cur + 1 if b else 0
            longest = max(longest, cur)
        pairs = {}
        for c in per_step:
            for key, depth in c.items():
                entry = pairs.setdefault(" ".join(key), dict(steps=0, depth=0.0))
                entry["steps"] += 1
                entry["depth"] = min(entry["depth"], depth)
        rows.append(
            dict(
                dpos=dpos,
                drpy=drpy,
                ee_pos=ee_pos,
                ee_ori=ee_ori,
                contact_steps=sum(busy),
                longest_contact=int(longest),
                pairs=pairs,
                collision_free=bool(free[len(rows)]) if len(rows) < len(free) else None,
            )
        )

    for row, is_free, is_kin in zip(rows, free, kin):
        row["collision_free"] = bool(is_free)
        row["kinematic"] = bool(is_kin)

    print(f"\npolicy {args.policy}")
    print(f"targets {args.n} from the {args.box} box, seed {args.seed}, hold {args.hold} steps")
    print(f"运动学可达 {sum(kin)}/{args.n}; 无碰撞可达 {sum(free)}/{args.n}")

    def summarize(subset: list[dict], label: str) -> dict:
        if not subset:
            return {}
        steps = sum(r["contact_steps"] for r in subset)
        total = len(subset) * args.hold
        durations = [r["longest_contact"] for r in subset]
        pairs = {}
        for r in subset:
            for key, entry in r["pairs"].items():
                agg = pairs.setdefault(key, dict(steps=0, depth=0.0, targets=0))
                agg["steps"] += entry["steps"]
                agg["depth"] = min(agg["depth"], entry["depth"])
                agg["targets"] += 1
        result = dict(
            label=label,
            targets=len(subset),
            ee_pos_median=float(np.median([r["ee_pos"] for r in subset])),
            ee_ori_median=float(np.median([r["ee_ori"] for r in subset])),
            contact_rate=steps / total,
            targets_with_contact=sum(1 for r in subset if r["contact_steps"]),
            contact_steps_median=float(np.median([r["contact_steps"] for r in subset])),
            longest_contact_median=float(np.median(durations)),
            longest_contact_max=int(max(durations)),
            pairs={k: dict(steps=v["steps"], targets=v["targets"], depth_mm=v["depth"] * 1000)
                   for k, v in sorted(pairs.items(), key=lambda kv: -kv[1]["steps"])},
        )
        print(f"\n[{label}] {result['targets']} 个目标")
        print(f"  EE 位置中位 {result['ee_pos_median'] * 1000:.0f} mm, 姿态中位 {result['ee_ori_median']:.1f}°")
        print(f"  接触步占比 {result['contact_rate'] * 100:.1f}%  ({steps}/{total})")
        print(f"  有接触的目标 {result['targets_with_contact']}/{result['targets']}; "
              f"单目标最长接触 中位 {result['longest_contact_median']:.0f} 步, 最大 {result['longest_contact_max']} 步")
        for key, v in list(result["pairs"].items())[:4]:
            print(f"    {key:38s} {v['steps']:4d} 步 / {v['targets']} 个目标, 最深 {v['depth_mm']:.0f} mm")
        return result

    summary = dict(
        policy=args.policy,
        box=args.box,
        seed=args.seed,
        hold=args.hold,
        n=args.n,
        reachable=sum(kin),
        collision_free=sum(free),
        all_targets=summarize(rows, "全部"),
        collision_free_targets=summarize([r for r in rows if r["collision_free"]], "无碰撞可达"),
        not_collision_free_targets=summarize([r for r in rows if not r["collision_free"]], "疑似不可达"),
    )

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / f"contact_by_target_{args.tag}.json").write_text(json.dumps(summary, indent=1))
    print(f"\n已写出 {out / f'contact_by_target_{args.tag}.json'}")


if __name__ == "__main__":
    main()
