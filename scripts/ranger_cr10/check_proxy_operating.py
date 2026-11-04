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

"""Stage 4.11: how often does the safety proxy fire while the policy is working normally?

`check_clearance_proxy.py` measures the proxy on random configurations, which is a stress test
and not the distribution that matters. A penalty is only a workspace tax if it fires on the
*poses the policy actually visits*, so this records those poses from a real rollout (the
60-target coverage suite, same seed as the evaluation) and then asks, for the same poses:

  * how many does the proxy call contact (``d <= 0``)?
  * how many does MuJoCo actually find in contact?
  * how often does the proxy fire on a pose that is clear?

That last number is the tax. It is measured here for the shipped one-box-per-link table and for
a four-slab-per-link table, which is the change the next stage proposes -- the comparison is the
point, not the absolute number.

Usage:
    python scripts/ranger_cr10/check_proxy_operating.py --policy <ckpt> --tag base9999
    python scripts/ranger_cr10/check_proxy_operating.py --tag base9999 --replay   # reuse poses
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SAFETY = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/safety.py"
SCENE = ROOT / "mujoco/robots/ranger_cr10/scene.xml"
for p in (str(ROOT / "mujoco/deploy/deploy_mujoco/ranger_cr10"), str(ROOT / "mujoco/deploy"), str(ROOT / "scripts/ranger_cr10")):
    if p not in sys.path:
        sys.path.append(p)

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--policy", default="logs/rsl_rl/ranger_cr10_teleop/2026-11-04_06-57-27_stage410f_ppo/model_9999.pt")
ap.add_argument("--tag", default="base9999")
ap.add_argument("--tables", default="outputs/stage411/collision_boxes.json,outputs/stage411/collision_boxes_s4.json")
ap.add_argument("--poses", default="outputs/stage411/operating_poses.npz")
ap.add_argument("--replay", action="store_true", help="reuse the recorded poses instead of rolling out")
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--hold", type=int, default=30)
ap.add_argument("--seed", type=int, default=5)
ap.add_argument("--out", default="outputs/stage411")
args = ap.parse_args()

import mujoco  # noqa: E402

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
bid = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)}
arm_adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)]
TABLE_NAMES = list(dict.fromkeys(["cr10_Link%d" % i for i in range(1, 7)]))


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("safety_variant", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install(module, table_path: Path) -> None:
    """Point a freshly loaded safety module at one generated box table."""
    table = json.loads(table_path.read_text())
    arm = {
        k: [(e["name"], tuple(e["centre"]), tuple(e["half"]), tuple(e["quat"]) if e["quat"] else None) for e in v]
        for k, v in table.items()
        if k in module.ARM_PARTS
    }
    prot = {
        k: (
            v[0]["body"],
            [(e["name"], tuple(e["centre"]), tuple(e["half"]), tuple(e["quat"]) if e["quat"] else None) for e in v],
        )
        for k, v in table.items()
        if k in module.PROTECTED_PARTS
    }
    module.ARM_PARTS = arm
    module.PROTECTED_PARTS = prot
    module.ARM_BODY_LIST, module.ARM_CENTRES, module.ARM_HALVES, module.ARM_QUATS = module._flatten_arm()
    (
        module.PROT_GROUPS,
        module.PROT_BODY_LIST,
        module.PROT_CENTRES,
        module.PROT_HALVES,
        module.PROT_QUATS,
    ) = module._flatten_protected()
    module.SELF_PAIRS, module.SENSOR_PAIRS = module._pairs()
    module.PENALTY_COLUMNS = module._penalty_columns()


def record_poses() -> np.ndarray:
    """Run the coverage protocol and keep every arm configuration it visits."""
    from base_command_filter import BaseCommandFilter
    from deploy_mujoco.ranger_cr10.ranger_policy import ee_command_from_offset
    from teleop_eval import Sim
    import workspace_limits as WS

    sim = Sim(args.policy, BaseCommandFilter())
    box = WS.TARGET_BOX_410
    rng = np.random.default_rng(args.seed)
    poses = []
    with torch.no_grad():
        for _ in range(args.n):
            dpos = [rng.uniform(*box[k]) for k in ("pos_x", "pos_y", "pos_z")]
            drpy = [rng.uniform(*box[k]) for k in ("roll", "pitch", "yaw")]
            sim.reset()
            sim.set_commands(0.0, 0.0, ee_command_from_offset(dpos, drpy))
            for _ in range(args.hold):
                sim.step()
                poses.append(np.array([sim.d.qpos[a] for a in arm_adr]))
    return np.array(poses)


def truth_contact(poses: np.ndarray) -> np.ndarray:
    """MuJoCo contact (any non-ground pair) for each recorded configuration."""
    out = np.zeros(len(poses), dtype=bool)
    for i, q in enumerate(poses):
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        for address, value in zip(arm_adr, q):
            data.qpos[address] = value
        mujoco.mj_forward(model, data)
        for c in range(data.ncon):
            if model.geom_bodyid[data.contact[c].geom1] != 0 and model.geom_bodyid[data.contact[c].geom2] != 0:
                out[i] = True
                break
    return out


def proxy_contact(module, poses: np.ndarray, cap: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    names = list(dict.fromkeys(module.ARM_BODY_LIST + module.PROT_BODY_LIST))
    contacts, minima = [], []
    batch = 256
    for start in range(0, len(poses), batch):
        chunk = poses[start : start + batch]
        pos, quat = [], []
        for q in chunk:
            mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
            for address, value in zip(arm_adr, q):
                data.qpos[address] = value
            mujoco.mj_forward(model, data)
            pos.append([data.xpos[bid[n]].copy() for n in names])
            quat.append([data.xquat[bid[n]].copy() for n in names])
        pos_t = torch.tensor(np.array(pos, dtype=np.float32))
        quat_t = torch.tensor(np.array(quat, dtype=np.float32))

        class Asset:
            body_names = names

            class data:  # noqa: N801
                body_pos_w = pos_t
                body_quat_w = quat_t

        d = module.clearance(Asset, cap=cap)[:, module.PENALTY_COLUMNS].amin(-1).numpy()
        contacts.append(d <= 0.0)
        minima.append(d)
    return np.concatenate(contacts), np.concatenate(minima)


def main() -> None:
    poses_path = ROOT / args.poses
    if args.replay and poses_path.exists():
        poses = np.load(poses_path)["poses"]
    else:
        print(f"rolling out {args.policy} to record operating poses ...", flush=True)
        poses = record_poses()
        poses_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(poses_path, poses=poses)
    print(f"poses {len(poses)}")

    truth = truth_contact(poses)
    print(f"MuJoCo 真实接触步: {truth.mean() * 100:.1f}%")

    rows = {}
    for table in args.tables.split(","):
        module = load_module(SAFETY)
        install(module, ROOT / table)
        contact, minimum = proxy_contact(module, poses)
        false = contact & ~truth
        miss = ~contact & truth
        rows[table] = dict(
            proxy_contact=float(contact.mean()),
            false_alarm=float(false.mean()),
            miss=float(miss.mean()),
            median_min=float(np.median(minimum)),
        )
        print(
            f"\n{table}\n"
            f"  代理判定接触 {contact.mean() * 100:.1f}%  (真实 {truth.mean() * 100:.1f}%)\n"
            f"  误报 {false.mean() * 100:.1f}%  漏报 {miss.mean() * 100:.2f}%\n"
            f"  最小间距中位 {np.median(minimum) * 1000:.1f} mm"
        )

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "proxy_operating.json").write_text(
        json.dumps(dict(tag=args.tag, poses=int(len(poses)), truth_contact=float(truth.mean()), tables=rows), indent=1)
    )
    print(f"\n已写出 {out / 'proxy_operating.json'}")


if __name__ == "__main__":
    main()
