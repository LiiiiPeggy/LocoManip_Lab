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

"""Generate `ranger_cr10.xml` from the URDF Isaac Lab already uses.

The point is *not* to be a general URDF importer. It is to make the sim-to-sim model
traceable: every body, inertia and joint transform below is read out of
`ranger_cr10_converted.urdf`, which is the file the Isaac Lab articulation is built from,
so a difference between the two simulators cannot come from the model being transcribed
wrong by hand. Actuator gains, armature and the contact parameters come from
`ranger_cr10_articulation_cfg.py`, the same numbers Isaac uses.

What is deliberately *not* a literal copy of the URDF:

* **Collision.** The URDF has one mesh per link, and Isaac builds PhysX meshes from them.
  Here the chassis, steering forks and wheels get analytic primitives (box / box /
  cylinder) sized from the mesh bounding boxes: MuJoCo's mesh collision is the convex
  hull, and a hull of the L-shaped chassis would collide with everything. Self-collision
  is off in Isaac and off here (see the contype/conaffinity scheme below).
* **The gripper is welded shut.** Isaac drives `gripper_finger1_joint` at 0 with stiff
  drives and the other seven linkage joints through PhysX mimic constraints, all of which
  sit at 0 for the closed pose. Welding them at 0 is the same configuration without a
  linkage that can be driven apart. The TCP is a virtual point on `cr10_Link6`, so the
  policy never sees the fingers.
* **Contact groups.** Robot links collide with the ground only (`contype=2, conaffinity=1`),
  wheels with the ground only (`contype=4, conaffinity=1`), the ground with everything
  (`contype=1, conaffinity=7`). A chassis box that overlaps the wheels in space -- as it
  does on the real robot, the wheels are inside the body's footprint -- therefore does not
  produce a contact, which is what `enabled_self_collisions=False` means in Isaac.

Usage:
    python mujoco/robots/ranger_cr10/build_mjcf.py
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
URDF = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10/ranger_cr10_converted.urdf"

# -- values from ranger_cr10_articulation_cfg.py ---------------------------------------
BASE_HEIGHT = 0.4071
ARM_DEFAULT = (0.6, 1.0, -2.4, 0.0, 1.4, 0.0)
STEERING = {"kp": 300.0, "kv": 30.0, "effort": 100.0, "armature": 0.5, "friction": 0.02}
WHEEL = {"kv": 60.0, "effort": 150.0, "armature": 0.05, "friction": 0.02, "vmax": 20.0}
ARM = {
    "cr10_joint1": (1500.0, 100.0, 150.0, 0.5, 3.0),
    "cr10_joint2": (4000.0, 200.0, 150.0, 1.0, 3.0),
    "cr10_joint3": (3000.0, 150.0, 100.0, 0.5, 3.0),
    "cr10_joint4": (800.0, 40.0, 50.0, 0.2, 3.2),
    "cr10_joint5": (500.0, 25.0, 50.0, 0.1, 3.2),
    "cr10_joint6": (300.0, 15.0, 30.0, 0.1, 3.2),
}
WHEEL_RADIUS = 0.1531
WHEEL_HALF_WIDTH = 0.1095
WHEEL_FRICTION = "0.35 0.005 0.0001"  # Isaac: wheel material 0.35, ground 1.0, combine=multiply

# Rolling resistance of the tyre, as a Coulomb torque on each wheel joint.
#
# A MuJoCo cylinder rolls on a mathematical line; Isaac's wheel is a mesh tyre that sinks
# into the ground and squashes, and that deformation is a real resisting torque. Measured
# in Isaac at two speeds (stage 4.6, `teleop_eval.py` drift suite, 1 env): a wheel-speed
# command of 0.39 rad/s reaches 0.39 - 0.095 and 2.0 rad/s reaches 2.0 - 0.212 -- i.e. a
# droop of kp * dw = 60 * 0.095 = 5.7 Nm and 12.7 Nm. Fitting a Coulomb + viscous pair to
# those two points gives ~4 Nm and ~4 Nm*s/rad; only the Coulomb half is added here, because
# the viscous half would just raise the velocity servo's gain.
#
# Measured effect (`--wheel_frictionloss 0` vs the default): at a 0.05 m/s command the
# achieved speed is 0.0711 without it and 0.0613 with it, against Isaac's 0.0622; the 30 s
# static drift with the arm moving falls from 81 mm to 48 mm, against Isaac's 13.6 mm. So it
# is a real improvement to the low-speed regime and *not* a complete explanation of the
# residual drift -- see docs/plans/teleop_plan.md section 16.
WHEEL_ROLLING_RESISTANCE_NM = 4.0

# -- geometry that is world-facing -----------------------------------------------------
MESHDIR = "../../../rangerboxcr10lidar_description/meshes"
MESH = {
    "base_link": "ranger_meshes/ranger_base_link.STL",
    "fr_steering_link": "ranger_meshes/fr_steering_link.STL",
    "fl_steering_wheel_link": "ranger_meshes/fl_steering_wheel_link.STL",
    "rl_steering_wheel_link": "ranger_meshes/rl_steering_wheel_link.STL",
    "rr_steering_wheel_link": "ranger_meshes/rr_steering_wheel_link.STL",
    "fr_wheel_link": "ranger_meshes/fr_wheel_link.STL",
    "fl_wheel_link": "ranger_meshes/fl_wheel_link.STL",
    "rl_wheel_link": "ranger_meshes/rl_wheel_link.STL",
    "rr_wheel_link": "ranger_meshes/rr_wheel_link.STL",
    "cr10_base_link": "cr10_meshes/cr10_base_link.STL",
    "cr10_Link1": "cr10_meshes/Link1.STL",
    "cr10_Link2": "cr10_meshes/Link2.STL",
    "cr10_Link3": "cr10_meshes/Link3.STL",
    "cr10_Link4": "cr10_meshes/Link4.STL",
    "cr10_Link5": "cr10_meshes/Link5.STL",
    "cr10_Link6": "cr10_meshes/Link6.STL",
    "gripper_base_link": "ag95_meshes/ag95_base_link.STL",
    # The AG95 linkage. In the URDF each finger link names the same four meshes.
    "gripper_finger1_knuckle_link": "ag95_meshes/crank_Link.STL",
    "gripper_finger2_knuckle_link": "ag95_meshes/crank_Link.STL",
    "gripper_finger1_finger_link": "ag95_meshes/rod_Link.STL",
    "gripper_finger2_finger_link": "ag95_meshes/rod_Link.STL",
    "gripper_finger1_inner_knuckle_link": "ag95_meshes/proximal_phalanx_Link.STL",
    "gripper_finger2_inner_knuckle_link": "ag95_meshes/proximal_phalanx_Link.STL",
    "gripper_finger1_finger_tip_link": "ag95_meshes/distal_phalanx_Link.STL",
    "gripper_finger2_finger_tip_link": "ag95_meshes/distal_phalanx_Link.STL",
    "lidar_link0": "lidar_link0.STL",
    "d455_link": "realsense_meshes/d455.stl",
    "d435_link": "realsense_meshes/d435.stl",
    "box_link": "box_link.STL",
}
# mesh scale, as written in the URDF (the URDF mixes metres and millimetres)
# the AG95 and RealSense meshes are authored in millimetres
MESH_SCALE = {"d455_link": 0.001, "d435_link": 0.001}
for _m in (
    "gripper_base_link",
    "gripper_finger1_knuckle_link",
    "gripper_finger2_knuckle_link",
    "gripper_finger1_finger_link",
    "gripper_finger2_finger_link",
    "gripper_finger1_inner_knuckle_link",
    "gripper_finger2_inner_knuckle_link",
    "gripper_finger1_finger_tip_link",
    "gripper_finger2_finger_tip_link",
):
    MESH_SCALE[_m] = 0.001

# Visual colours, following the conversion in ~/locomani/UniLab (assets/robots/ranger_box):
# the steering forks and the tyre sidewalls are black, the wheels and the arm are light.
VIS_RGBA = {
    "fr_steering_link": "0.05 0.05 0.05 1",
    "fl_steering_wheel_link": "0.05 0.05 0.05 1",
    "rl_steering_wheel_link": "0.05 0.05 0.05 1",
    "rr_steering_wheel_link": "0.05 0.05 0.05 1",
    "gripper_base_link": "0.15 0.15 0.18 1",
    "box_link": "0.55 0.42 0.25 1",
}
DEFAULT_RGBA = "0.82 0.82 0.84 1"

# -- bodies that carry collision, and how ----------------------------------------------
# name -> (kind, size/box params). Everything else is visual-only mass.
COLLISION = {
    "base_link": ("box", (1.302, 0.8694, 0.4351), (0.0, -0.0033, -0.0571)),
    "fr_steering_link": ("box", (0.088, 0.2333, 0.2663), (0.0041, -0.0082, -0.1730)),
    "fl_steering_wheel_link": ("box", (0.0881, 0.2400, 0.2703), (0.0041, 0.0104, -0.1750)),
    "rl_steering_wheel_link": ("box", (0.0881, 0.2344, 0.2636), (0.0041, 0.0076, -0.1717)),
    "rr_steering_wheel_link": ("box", (0.088, 0.2333, 0.2673), (0.0041, -0.0082, -0.1735)),
}
for _w in ("fr", "fl", "rl", "rr"):
    COLLISION[f"{_w}_wheel_link"] = ("cylinder", (WHEEL_RADIUS, WHEEL_HALF_WIDTH), (0.0, 0.0, 0.0))
for _l in ("cr10_base_link", "cr10_Link1", "cr10_Link2", "cr10_Link3", "cr10_Link4", "cr10_Link5", "cr10_Link6"):
    COLLISION[_l] = ("mesh", None, None)

# Frames with no inertial and no mesh. The camera *bodies* (``d455_link``, ``d435_link``)
# stay: they carry a mesh and a mass, and dropping their parent frame used to take the whole
# subtree with it, which is why the cameras were invisible before 2026-10-31.
DROP_LINKS = {
    "d455_depth_frame",
    "d455_depth_optical_frame",
    "d455_infra1_frame",
    "d455_infra1_optical_frame",
    "d455_infra2_frame",
    "d455_infra2_optical_frame",
    "d455_color_frame",
    "d455_color_optical_frame",
    "d455_accel_frame",
    "d455_accel_optical_frame",
    "d455_gyro_frame",
    "d455_gyro_optical_frame",
    "d455_imu_optical_frame",
    "d435_depth_frame",
    "d435_depth_optical_frame",
    "d435_infra1_frame",
    "d435_infra1_optical_frame",
    "d435_infra2_frame",
    "d435_infra2_optical_frame",
    "d435_color_frame",
    "d435_color_optical_frame",
}
# The AG95 linkage: welded at 0, see the module docstring.
WELD_JOINT_PREFIX = "gripper_"

VISUAL_ORIGIN = {}  # link name -> (xyz, quat) from the URDF's <visual><origin>

TCP_BODY = "cr10_Link6"
TCP_OFFSET = (0.0, 0.00003, 0.14389)


def rpy_to_quat(rpy):
    """URDF fixed-axis rpy -> MuJoCo (w, x, y, z), same convention as Isaac Lab."""
    r, p, y = rpy
    cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
    cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
    cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def quat_to_R(q):
    w_, x_, y_, z_ = q
    return [
        [1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - w_ * z_), 2 * (x_ * z_ + w_ * y_)],
        [2 * (x_ * y_ + w_ * z_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - w_ * x_)],
        [2 * (x_ * z_ - w_ * y_), 2 * (y_ * z_ + w_ * x_), 1 - 2 * (x_ * x_ + y_ * y_)],
    ]


def fmt(v):
    return " ".join(f"{x:.9g}" for x in v)


def parse_urdf():
    root = ET.parse(URDF).getroot()
    links = {ln.get("name"): ln for ln in root.findall("link")}
    # A link's mesh is not necessarily drawn in the link frame: the URDF gives every <visual>
    # its own <origin>. Ignoring it put the wrist RealSenses on their side -- both declare
    # rpy = (90 deg, 0, 90 deg) -- while every other link happens to be identity.
    for name, ln in links.items():
        vis = ln.find("visual")
        if vis is None:
            continue
        o = vis.find("origin")
        if o is None:
            continue
        xyz = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()]
        if any(abs(v) > 1e-9 for v in xyz) or any(abs(v) > 1e-9 for v in rpy):
            VISUAL_ORIGIN[name] = (xyz, rpy_to_quat(rpy))
    joints = root.findall("joint")
    children = {}
    for j in joints:
        children.setdefault(j.find("parent").get("link"), []).append(j)
    return links, joints, children


def is_welded(joint):
    return joint.get("type") == "fixed" or joint.get("name", "").startswith(WELD_JOINT_PREFIX)


def build(links, joints, children):
    out = []
    w = out.append

    w('<mujoco model="ranger_cr10">')
    w(f'  <compiler angle="radian" meshdir="{MESHDIR}" autolimits="true"/>')
    # 200 Hz physics, the same as Isaac Lab's sim.dt; implicitfast keeps the velocity-driven
    # wheels stable at that step where plain Euler needs a smaller one.
    w('  <option timestep="0.005" integrator="implicitfast" cone="elliptic" impratio="100"/>')
    w('  <size nconmax="200" njmax="400"/>')
    w("")

    # ---- defaults ---------------------------------------------------------------------
    w("  <default>")
    w('    <geom friction="0.8 0.005 0.0001" solref="0.005 1" solimp="0.95 0.99 0.001"/>')
    w('    <default class="robot">')
    # collide with the ground only; see the contype scheme in the docstring.
    # group 3 is not drawn by default (`geomgroup` 0-2 are), so the collision proxies stay
    # invisible unless the operator switches group 3 on.
    w('      <geom contype="2" conaffinity="1" group="3" rgba="0 1 0 0.25"/>')
    w("    </default>")
    w('    <default class="wheel">')
    w(
        f'      <geom contype="4" conaffinity="1" condim="4" priority="1" group="3" '
        f'rgba="0 0.6 1 0.35" friction="{WHEEL_FRICTION}"/>'
    )
    w("    </default>")
    w('    <default class="visual">')
    w('      <geom type="mesh" contype="0" conaffinity="0" group="2" material="gray"/>')
    w("    </default>")
    w('    <default class="steer_act">')
    eff = STEERING["effort"]
    w(f'      <position kp="{STEERING["kp"]}" kv="{STEERING["kv"]}" forcerange="-{eff} {eff}"/>')
    w("    </default>")
    w('    <default class="wheel_act">')
    w(f'      <velocity kv="{WHEEL["kv"]}" forcerange="-{WHEEL["effort"]} {WHEEL["effort"]}"/>')
    w("    </default>")
    w('    <default class="arm_act">')
    w("      <position/>")
    w("    </default>")
    w("  </default>")
    w("")

    # ---- assets -----------------------------------------------------------------------
    w("  <asset>")
    for name in ("gray", "black", "white"):
        rgba = {"gray": "0.67 0.69 0.77 1", "black": "0.05 0.05 0.05 1", "white": "0.95 0.95 0.95 1"}[name]
        w(f'    <material name="{name}" rgba="{rgba}"/>')
    for asset_name, (path, scale) in sorted(mesh_assets().items()):
        extra = f' scale="{scale} {scale} {scale}"' if scale else ""
        w(f'    <mesh name="{asset_name}" file="{path}"{extra}/>')
    w("  </asset>")
    w("")

    # ---- body tree --------------------------------------------------------------------
    w("  <worldbody>")
    w(f'    <body name="base_link" pos="0 0 {BASE_HEIGHT:.6f}">')
    w('      <freejoint name="root"/>')
    emit_link_geoms(w, links["base_link"], "base_link", indent=6)
    emit_link_inertial(w, links["base_link"], indent=6)

    def walk(link_name, indent):
        pad = " " * indent
        for j in children.get(link_name, []):
            child = j.find("child").get("link")
            if child in DROP_LINKS:
                continue
            xyz = [float(v) for v in (j.find("origin").get("xyz") or "0 0 0").split()]
            rpy = [float(v) for v in (j.find("origin").get("rpy") or "0 0 0").split()]
            q = rpy_to_quat(rpy)
            w(f'{pad}<body name="{child}" pos="{fmt(xyz)}" quat="{fmt(q)}">')
            if not is_welded(j):
                emit_joint(w, j, indent + 2)
            emit_link_geoms(w, links[child], child, indent + 2)
            emit_link_inertial(w, links[child], indent + 2)
            if child == TCP_BODY:
                w(f'{pad}  <site name="tcp" pos="{fmt(TCP_OFFSET)}" size="0.01"/>')
            walk(child, indent + 2)
            w(f"{pad}</body>")

    walk("base_link", 6)
    w("    </body>")
    w("  </worldbody>")
    w("")

    # ---- actuators --------------------------------------------------------------------
    w("  <actuator>")
    for j in joints:
        name = j.get("name")
        if is_welded(j):
            continue
        if "steering" in name:
            w(f'    <position name="{name}_pos" joint="{name}" class="steer_act" ctrlrange="-1.57 1.57"/>')
        elif name.endswith("_wheel_joint"):
            vmax = WHEEL["vmax"]
            w(f'    <velocity name="{name}_vel" joint="{name}" class="wheel_act" ctrlrange="-{vmax} {vmax}"/>')
        elif name in ARM:
            kp, kv, eff, _, vmax = ARM[name]
            rng = joint_range(j)
            w(
                f'    <position name="{name}_pos" joint="{name}" class="arm_act" kp="{kp}" kv="{kv}" '
                f'forcerange="-{eff} {eff}" ctrlrange="{rng[0]:.5f} {rng[1]:.5f}"/>'
            )
    w("  </actuator>")
    w("")

    # ---- keyframe ---------------------------------------------------------------------
    qpos = [0.0, 0.0, BASE_HEIGHT, 1.0, 0.0, 0.0, 0.0]
    for name in qpos_names(children):
        qpos.append(ARM_DEFAULT[list(ARM).index(name)] if name in ARM else 0.0)
    # actuator order: four steering, four wheels, six arm joints
    ctrl = [0.0] * 8 + list(ARM_DEFAULT)
    w(f'  <keyframe><key name="home" qpos="{fmt(qpos)}" ctrl="{fmt(ctrl)}"/></keyframe>')
    w("</mujoco>")
    return "\n".join(out) + "\n"


def joint_range(j):
    lim = j.find("limit")
    if lim is None or lim.get("lower") is None:
        return (-1.57, 1.57)
    return (float(lim.get("lower")), float(lim.get("upper")))


def qpos_names(children):
    """Joint names in the order they are emitted into the tree."""
    out = []

    def walk(link):
        for j in children.get(link, []):
            child = j.find("child").get("link")
            if child in DROP_LINKS:
                continue
            if not is_welded(j):
                out.append(j.get("name"))
            walk(child)

    walk("base_link")
    return out


def emit_joint(w, j, indent):
    pad = " " * indent
    name = j.get("name")
    axis = j.find("axis").get("xyz") if j.find("axis") is not None else "0 0 1"
    rng = joint_range(j)
    if "steering" in name:
        arm_val, fric = STEERING["armature"], STEERING["friction"]
        lim = f' range="{rng[0]} {rng[1]}"'
    elif name.endswith("_wheel_joint"):
        arm_val = WHEEL["armature"]
        # WHEEL["friction"] is the bearing drag Isaac's 0.02 Nm models; the tyre's rolling
        # resistance is added on top, see WHEEL_ROLLING_RESISTANCE_NM.
        fric = WHEEL["friction"] + WHEEL_ROLLING_RESISTANCE_NM
        lim = ""  # continuous
    else:
        _, _, _, arm_val, _ = ARM[name]
        fric = 0.01
        lim = f' range="{rng[0]} {rng[1]}"'
    w(f'{pad}<joint name="{name}" type="hinge" axis="{axis}"{lim} armature="{arm_val}" frictionloss="{fric}"/>')


def mesh_asset_for(link):
    """(asset name, scale) for a link's mesh: one asset per file, shared between links."""
    path = MESH[link]
    return Path(path).stem, MESH_SCALE.get(link)


def mesh_assets():
    """``{asset_name: (path, scale)}`` over every mesh the model references."""
    out = {}
    for link, path in MESH.items():
        name, scale = mesh_asset_for(link)
        if name in out and out[name] != (path, scale):
            raise SystemExit(f"mesh asset name clash: {name}")
        out[name] = (path, scale)
    return out


def emit_link_geoms(w, link, name, indent):
    pad = " " * indent
    if name in MESH:
        asset, _ = mesh_asset_for(name)
        rgba = VIS_RGBA.get(name, DEFAULT_RGBA)
        extra = ""
        if name in VISUAL_ORIGIN:
            pos, quat = VISUAL_ORIGIN[name]
            extra = f' pos="{fmt(pos)}" quat="{fmt(quat)}"'
        w(f'{pad}<geom name="{name}_vis" type="mesh" mesh="{asset}" class="visual" rgba="{rgba}"{extra}/>')
    spec = COLLISION.get(name)
    if spec is None or spec[0] == "mesh":
        if spec is not None:
            asset, _ = mesh_asset_for(name)
            w(f'{pad}<geom name="{name}_col" type="mesh" mesh="{asset}" class="robot"/>')
        return
    kind, size, pos = spec
    if kind == "box":
        w(
            f'{pad}<geom name="{name}_col" type="box" size="{fmt((size[0] / 2, size[1] / 2, size[2] / 2))}" '
            f'pos="{fmt(pos)}" class="robot"/>'
        )
    elif kind == "cylinder":
        w(f'{pad}<geom name="{name}_col" type="cylinder" size="{fmt(size)}" pos="{fmt(pos)}" class="wheel"/>')


def emit_link_inertial(w, link, indent):
    pad = " " * indent
    i = link.find("inertial")
    if i is None:
        w(f'{pad}<inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>')
        return
    mass = float(i.find("mass").get("value"))
    pos = [float(v) for v in (i.find("origin").get("xyz") or "0 0 0").split()]
    rpy = [float(v) for v in (i.find("origin").get("rpy") or "0 0 0").split()]
    it = i.find("inertia")
    vals = [float(it.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
    # MuJoCo will not take a quat and a full inertia at once, so rotate the tensor into the
    # body frame here. Every inertial rpy in this URDF is zero, so this is identity in
    # practice; doing it keeps the generator honest if the URDF ever changes.
    R = np.array(quat_to_R(rpy_to_quat(rpy)))
    T = np.array([[vals[0], vals[3], vals[4]], [vals[3], vals[1], vals[5]], [vals[4], vals[5], vals[2]]])
    T = R @ T @ R.T
    out = [T[0, 0], T[1, 1], T[2, 2], T[0, 1], T[0, 2], T[1, 2]]
    w(f'{pad}<inertial pos="{fmt(pos)}" mass="{mass:.6g}" fullinertia="{fmt(out)}"/>')


def main() -> None:
    global WHEEL_ROLLING_RESISTANCE_NM
    ap = argparse.ArgumentParser(description="Generate the MuJoCo model of the RangerBox-CR10.")
    ap.add_argument("--out", default=str(Path(__file__).with_name("ranger_cr10.xml")))
    ap.add_argument(
        "--wheel_frictionloss",
        type=float,
        default=WHEEL_ROLLING_RESISTANCE_NM,
        help="Coulomb rolling resistance per wheel joint, Nm; 0 gives the rigid-cylinder model",
    )
    args = ap.parse_args()
    WHEEL_ROLLING_RESISTANCE_NM = args.wheel_frictionloss

    links, joints, children = parse_urdf()
    xml = build(links, joints, children)
    Path(args.out).write_text(xml)
    print(f"wrote {args.out} ({len(xml.splitlines())} lines)")

    import mujoco

    m = mujoco.MjModel.from_xml_path(args.out)
    print(f"nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} njnt={m.njnt}")
    print(f"total mass {m.body_subtreemass[1]:.3f} kg")


if __name__ == "__main__":
    main()
