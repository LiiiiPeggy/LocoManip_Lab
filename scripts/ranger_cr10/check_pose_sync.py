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

"""Stage 4.10: is the default pose the same number in every place that stores it?

The pose now lives in six places, and a policy trained against one of them while the robot
stands at another is a failure that looks like a bad policy:

    teleop_kinematics.DEFAULT_Q         the offline IK tools
    ranger_cr10_articulation_cfg        Isaac's ``init_state`` (also the action's offset)
    build_mjcf.ARM_DEFAULT              the MuJoCo keyframe
    ranger_policy.ARM_DEFAULT           the deployment controller
    pose_command_teleop.TCP_DEFAULT_*   the command centre (Isaac)
    ranger_policy.TCP_DEFAULT_*         the command centre (deployment)

This compares all six against the **MuJoCo model itself** (the keyframe and the TCP site),
which is the one source that cannot be edited into agreement by mistake.

Usage:
    python scripts/ranger_cr10/check_pose_sync.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ranger_cr10"))
sys.path.insert(0, str(ROOT / "mujoco" / "deploy"))

import teleop_kinematics as tk  # noqa: E402
from deploy_mujoco.ranger_cr10 import ranger_policy as rp  # noqa: E402

import mujoco  # noqa: E402

ASSETS = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10"
MPL = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/pose_command_teleop.py"


def load_attr(path: Path, name: str):
    spec = importlib.util.spec_from_file_location("probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def main() -> None:
    fails = []

    def check(name, ok, detail):
        print(f"  [{'通过' if ok else '**失败**'}] {name}: {detail}")
        if not ok:
            fails.append(name)

    m = mujoco.MjModel.from_xml_path(str(ROOT / "mujoco/robots/ranger_cr10/scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)

    key_q = np.array(
        [d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")]] for i in range(1, 7)]
    )
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    Rb = d.xmat[base].reshape(3, 3)
    tcp_b = Rb.T @ (d.site_xpos[sid] - d.xpos[base])

    # -- the six default-joint vectors ---------------------------------------------------
    # Read out of the source rather than imported: that module imports isaaclab, which needs
    # `pxr`, which only exists inside a running Isaac app (MEMORY.md). Parsing the
    # `init_state` block is what makes this check runnable as a plain python process.
    import re

    src = (ASSETS / "ranger_cr10_articulation_cfg.py").read_text()
    block = src[src.index("init_state=") : src.index("joint_vel=")]
    isaac_q = np.array([float(re.search(rf'"cr10_joint{i}":\s*([-0-9.eE]+)', block).group(1)) for i in range(1, 7)])
    build = load_attr(ROOT / "mujoco/robots/ranger_cr10/build_mjcf.py", "ARM_DEFAULT")

    print("=" * 92)
    print("1. 默认关节角: MuJoCo keyframe / Isaac init_state / 动作偏移 / 部署控制器 / 离线 IK")
    print("=" * 92)
    for name, q in (
        ("Isaac init_state (也是 JointPositionAction 的 default offset)", isaac_q),
        ("build_mjcf.ARM_DEFAULT", np.array(build)),
        ("ranger_policy.ARM_DEFAULT (部署)", np.array(rp.ARM_DEFAULT)),
        ("teleop_kinematics.DEFAULT_Q", np.array(tk.DEFAULT_Q)),
    ):
        check(
            name,
            np.allclose(q, key_q, atol=1e-6),
            f"{np.round(q, 3).tolist()} vs keyframe {np.round(key_q, 3).tolist()}",
        )

    # -- the command centre -------------------------------------------------------------
    # same reason as above: the command term module imports isaaclab
    mdp_src = MPL.read_text()
    isaac_pos = np.array([float(v) for v in re.search(r"TCP_DEFAULT_POS_B = \(([^)]+)\)", mdp_src).group(1).split(",")])
    isaac_quat = np.array(
        [float(v) for v in re.search(r"TCP_DEFAULT_QUAT_B = \(([^)]+)\)", mdp_src).group(1).split(",")]
    )
    dep_pos = np.array(rp.TCP_DEFAULT_POS_B)
    dep_quat = np.array(rp.TCP_DEFAULT_QUAT_B)
    Rt = d.site_xmat[sid].reshape(3, 3)
    Rrel = Rb.T @ Rt
    t = np.trace(Rrel)
    w = np.sqrt(max(0.0, 1 + t)) / 2
    quat_ref = np.array(
        [
            w,
            (Rrel[2, 1] - Rrel[1, 2]) / (4 * w),
            (Rrel[0, 2] - Rrel[2, 0]) / (4 * w),
            (Rrel[1, 0] - Rrel[0, 1]) / (4 * w),
        ]
    )

    print()
    print("=" * 92)
    print("2. EE 命令中心 vs 模型里默认位姿下的 TCP")
    print("=" * 92)
    check("Isaac TCP_DEFAULT_POS_B", np.allclose(isaac_pos, tcp_b, atol=2e-4), f"{np.round(isaac_pos, 4).tolist()}")
    check("部署 TCP_DEFAULT_POS_B", np.allclose(dep_pos, tcp_b, atol=2e-4), f"{np.round(dep_pos, 4).tolist()}")
    check("两处位置一致", np.allclose(isaac_pos, dep_pos, atol=1e-9), "")
    check(
        "Isaac TCP_DEFAULT_QUAT_B",
        abs(abs(float(np.dot(isaac_quat, quat_ref))) - 1.0) < 5e-4,
        f"夹角 {np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(isaac_quat, quat_ref)))))):.3f}°",
    )
    check(
        "部署 TCP_DEFAULT_QUAT_B",
        abs(abs(float(np.dot(dep_quat, quat_ref))) - 1.0) < 5e-4,
        f"夹角 {np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(dep_quat, quat_ref)))))):.3f}°",
    )

    # -- the camera mount, in both URDFs -------------------------------------------------
    print()
    print("=" * 92)
    print("3. D455 安装位姿: 原始 URDF 与转换 URDF 必须一致")
    print("=" * 92)
    import re

    def mount(p):
        s = (ROOT / p).read_text()
        i = s.find('name="d455_joint"')
        return re.search(r'<origin xyz="([^"]+)" rpy="([^"]+)"', s[i : i + 800])

    a = mount("rangerboxcr10lidar_description/urdf/rangercr10lidar.urdf")
    b = mount("source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10/ranger_cr10_converted.urdf")
    check(
        "两份 URDF 的 origin",
        a is not None and b is not None and a.group(0) == b.group(0),
        a.group(0) if a else "missing",
    )

    print()
    print("=" * 92)
    print("汇总")
    print("=" * 92)
    if fails:
        print("失败项: " + ", ".join(fails))
        sys.exit(1)
    print("全部一致")


if __name__ == "__main__":
    main()
