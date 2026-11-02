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

"""Stage 4.9: how close is the arm to the lidar, the chassis camera, and the chassis body?

The task learned end-effector tracking with ``enabled_self_collisions=False``, so nothing in
training ever saw the arm going through the chassis-mounted sensors. Turning PhysX self
collision on is not enough by itself -- a contact is a binary, late signal, and the reward
needs a gradient *before* the impact -- so this module computes a clearance from forward
kinematics:

* **the arm** is approximated by spheres spread along each link's collision box (the boxes
  come from the URDF meshes, the same values ``mujoco/robots/ranger_cr10/build_mjcf.py``
  uses, so the two simulators describe the same geometry);
* **the protected parts** (lidar tower, chassis camera, chassis body) are axis-aligned boxes
  in their own link frames -- all three are mounted with identity rotation relative to
  ``base_link``, which is asserted rather than assumed.

Sphere-to-box and sphere-to-sphere distances are exact and cheap, which is what makes this
usable as a per-step reward at 2048 environments. The approximation is on the arm side: a
sphere set that circumscribes a link is *conservative*, so the penalty starts slightly early.
`scripts/ranger_cr10/check_clearance_proxy.py` measures that bias against MuJoCo's exact
convex-hull distance and reports it rather than assuming it is small.

Nothing here is used by the actor: the observation and action spaces are untouched.
"""

from __future__ import annotations

import math

import torch

# Collision boxes in each link's own frame: (centre, half-extent), metres. Read out of the
# MuJoCo collision geoms generated from `ranger_cr10_converted.urdf` (stage 4.9) -- the arm
# links are convex hulls of their meshes, described here by their axis-aligned bounds.
# The bodies are the ones **Isaac Lab's USD actually has**. The URDF→USD conversion merges
# fixed joints, so `cr10_base_link`, `gripper_base_link`, `d455_link`, `lidar_link0` and
# `d435_link` do not exist as articulation bodies in Isaac at all -- verified by listing
# `robot.body_names`. Every box below is therefore expressed in a body that exists in both
# simulators, with the merged links' offsets folded in:
#   * the AG95 tool is coincident with `cr10_Link6` (identity, zero offset);
#   * the D455 is mounted on it with a rotation, so its box is its rotated AABB in Link6;
#   * the arm mount (`cr10_base_link`) is fused to `base_link` and is skipped -- it is the
#     parent/child pair MuJoCo excludes anyway.
ARM_BOXES = {
    "cr10_Link1": ("cr10_Link1", (0.0000, -0.0130, 0.0058), (0.0772, 0.0905, 0.1045)),
    "cr10_Link2": ("cr10_Link2", (-0.2824, 0.0034, 0.1823), (0.0772, 0.1006, 0.3728)),
    "cr10_Link3": ("cr10_Link3", (-0.2797, -0.0026, 0.0622), (0.0608, 0.0742, 0.3378)),
    "cr10_Link4": ("cr10_Link4", (0.0000, 0.0059, -0.0105), (0.0464, 0.0593, 0.0805)),
    "cr10_Link5": ("cr10_Link5", (0.0000, -0.0225, -0.0095), (0.0464, 0.0538, 0.0855)),
    "cr10_Link6": ("cr10_Link6", (0.0016, -0.0005, -0.0178), (0.0211, 0.0450, 0.0502)),
    # the AG95: palm plus one box per jaw. The jaw boxes stand in for the finger links, which
    # carry a visual mesh but no collision geometry -- which is exactly how the tool used to
    # pass 27 mm into the lidar without a contact being reported.
    "gripper_base_link_palm": ("cr10_Link6", (0.0000, 0.0004, 0.0656), (0.0335, 0.0466, 0.0691)),
    "gripper_base_link_jaw_l": ("cr10_Link6", (-0.0495, 0.0, 0.1290), (0.0235, 0.0260, 0.0560)),
    "gripper_base_link_jaw_r": ("cr10_Link6", (0.0495, 0.0, 0.1290), (0.0235, 0.0260, 0.0560)),
    "d455_link": ("cr10_Link6", (-0.0100, -0.0746, 0.0328), (0.0620, 0.0146, 0.0130)),
}

# The parts the arm may not touch, in the **base_link** frame. All three are rigid to the
# chassis, and in Isaac they are *part of* base_link (merged fixed joints), so expressing
# them here is both correct and the only thing that works on both sides.
PROTECTED_BOXES = {
    "lidar_link0": ((0.5086, 0.0027, 0.1738), (0.0858, 0.1269, 0.1427)),
    "d435_link": ((0.6215, 0.0, 0.2364), (0.0125, 0.045, 0.0125)),
    "base_link": ((0.0, -0.0033, -0.0571), (0.651, 0.4347, 0.2176)),
}

# Arm-to-arm pairs that are *not* worth penalising: parent/child pairs overlap at the joint
# by design, and the tool parts are one rigid body. Everything else is a real self-collision
# (measured in MuJoCo: Link3 <-> the wrist camera, Link2 <-> gripper).
ARM_ADJACENT = {
    ("cr10_Link1", "cr10_Link2"),
    ("cr10_Link2", "cr10_Link3"),
    ("cr10_Link3", "cr10_Link4"),
    ("cr10_Link4", "cr10_Link5"),
    ("cr10_Link5", "cr10_Link6"),
}

# Arm entries that must not be measured against a protected part. The mount is fused to the
# chassis in Isaac and excluded in MuJoCo, so nothing is skipped today; the table is kept
# because it is the place that decision belongs.
PROTECTED_SKIP = {}


def _spheres(centre, half):
    """Spheres that *cover* a box, along its longest axis.

    The radius is the half-diagonal of the cross-section, so the corners are covered rather
    than the inscribed circle; that makes the proxy conservative (it reports a smaller
    clearance than the true geometry), which is the safe direction for a safety term. The
    centres stop one radius short of each end so the set does not grow the link.
    `check_clearance_proxy.py` measures the resulting bias against MuJoCo's convex hulls.
    """
    axis = max(range(3), key=lambda i: half[i])
    long_h = half[axis]
    r = math.hypot(half[(axis + 1) % 3], half[(axis + 2) % 3])
    if long_h <= r:
        return [(*centre, max(long_h, r))]
    inner = long_h - r
    step = r * 0.5
    n = max(2, int(math.ceil(2.0 * inner / step)) + 1)
    out = []
    for i in range(n):
        t = -inner + (2.0 * inner) * i / (n - 1)
        c = list(centre)
        c[axis] = centre[axis] + t
        out.append((*c, r))
    return out


# Flattened sphere set: local centre, radius, and the index of the owning body.
_BODY_ORDER = list(dict.fromkeys(v[0] for v in ARM_BOXES.values()))


def _flatten():
    centres, radii, owner = [], [], []
    for entry, (body, centre, half) in ARM_BOXES.items():
        for c in _spheres(centre, half):
            centres.append(c[:3])
            radii.append(c[3])
            owner.append(_BODY_ORDER.index(body))
    return (
        torch.tensor(centres, dtype=torch.float32),
        torch.tensor(radii, dtype=torch.float32),
        torch.tensor(owner, dtype=torch.long),
    )


SPHERE_CENTRES, SPHERE_RADII, SPHERE_OWNER = _flatten()


def _pair_mask():
    """(S, S) boolean: arm-sphere pairs whose distance is worth penalising."""
    n = len(SPHERE_OWNER)
    body = [ARM_BOXES[k][0] for k in ARM_BOXES for _ in _spheres(ARM_BOXES[k][1], ARM_BOXES[k][2])]
    assert len(body) == n
    mask = torch.zeros((n, n), dtype=torch.bool)
    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = body[i], body[j]
            if bi == bj:
                continue
            if tuple(sorted((bi, bj))) in {tuple(sorted(p)) for p in ARM_ADJACENT}:
                continue
            mask[i, j] = True
    return mask


SELF_PAIR_MASK = _pair_mask()

# For each protected part, the arm spheres that must not be measured against it.
SPHERE_SKIP = {}
for _protected_name in PROTECTED_BOXES:
    _skip = torch.zeros(len(SPHERE_OWNER), dtype=torch.bool)
    for i, _owner in enumerate(SPHERE_OWNER.tolist()):
        if _protected_name in PROTECTED_SKIP.get(_BODY_ORDER[_owner], set()):
            _skip[i] = True
    SPHERE_SKIP[_protected_name] = _skip


def _base_relative(asset, body_ids, base_id, device):
    """(pos, quat) of ``body_ids`` expressed in the ``base_id`` frame."""
    from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

    base_pos = asset.data.body_pos_w[:, base_id]
    base_quat = asset.data.body_quat_w[:, base_id]
    conj = quat_conjugate(base_quat).unsqueeze(1).expand(-1, len(body_ids), -1)
    pos = asset.data.body_pos_w[:, body_ids] - base_pos.unsqueeze(1)
    pos = quat_apply(conj, pos)
    rel_quat = quat_mul(conj, asset.data.body_quat_w[:, body_ids])
    return pos, rel_quat


def _assemble_spheres(asset, base_body_name="base_link"):
    from isaaclab.utils.math import quat_apply

    device = asset.data.body_pos_w.device
    base_ids, _ = asset.find_bodies([base_body_name])
    ids, _ = asset.find_bodies(_BODY_ORDER, preserve_order=True)
    pos, quat = _base_relative(asset, ids, base_ids[0], device)
    owner = SPHERE_OWNER.to(device)
    centres = quat_apply(quat[:, owner], SPHERE_CENTRES.to(device).unsqueeze(0).expand(pos.shape[0], -1, -1))
    centres = centres + pos[:, owner]
    return centres, SPHERE_RADII.to(device), pos, quat


def arm_clearance(asset, base_body_name: str = "base_link", cap: float = 0.5, protected=None):
    """Per-environment minimum distance (m) between the arm chain and the protected parts.

    Everything is computed in the ``base_link`` frame. That is exact for the protected boxes:
    the lidar and the camera are fixed to the chassis with identity rotation, so both the box
    and the distances stay axis-aligned there however the chassis turns.
    """
    protected = PROTECTED_BOXES if protected is None else protected
    device = asset.data.body_pos_w.device
    centres, radii, _, _ = _assemble_spheres(asset, base_body_name)

    best = torch.full((centres.shape[0],), cap, device=device)
    for name, (centre, half) in protected.items():
        box_centre = torch.tensor(centre, device=device).unsqueeze(0)
        half_t = torch.tensor(half, device=device).unsqueeze(0)
        outside = (centres - box_centre.unsqueeze(1)).abs() - half_t.unsqueeze(1)
        dist = torch.linalg.norm(outside.clamp(min=0.0), dim=-1) - radii
        skip = SPHERE_SKIP.get(name)
        if skip is not None:
            dist = dist.masked_fill(skip.to(device).unsqueeze(0), cap)
        best = torch.minimum(best, dist.amin(dim=-1))
    return best.clamp(max=cap)


def arm_self_clearance(asset, base_body_name: str = "base_link", cap: float = 0.5):
    """Per-environment minimum distance between non-adjacent arm spheres."""
    device = asset.data.body_pos_w.device
    centres, radii, _, _ = _assemble_spheres(asset, base_body_name)
    diff = centres.unsqueeze(2) - centres.unsqueeze(1)
    dist = torch.linalg.norm(diff, dim=-1) - radii.unsqueeze(0) - radii.unsqueeze(1)
    mask = SELF_PAIR_MASK.to(device)
    dist = torch.where(mask.unsqueeze(0), dist, torch.full_like(dist, cap))
    return dist.amin(dim=(1, 2)).clamp(max=cap)


def total_clearance(asset, cap: float = 0.5, include_self: bool = True):
    """Minimum of the sensor clearance and (optionally) the arm's own self clearance."""
    d = arm_clearance(asset, cap=cap)
    if include_self:
        d = torch.minimum(d, arm_self_clearance(asset, cap=cap))
    return d


def sensor_clearance_penalty(env, asset_cfg, margin: float = 0.02, cap: float = 0.5, include_self: bool = True):
    """Reward term: zero while clear, rising quadratically as the arm closes in.

    The reference is the clearance the robot has at the *start of the episode*, not an
    absolute number. The proxy is deliberately conservative (spheres circumscribe the links,
    the chassis is a box around a rounded body), so in absolute terms the shoulder sits about
    16 mm "inside" a 20 mm margin at the authored pose and a plain margin would penalise
    standing still. Measuring the loss of clearance relative to the starting posture keeps the
    signal about what the policy does, and `check_clearance_proxy.py` reports the remaining
    bias against MuJoCo's exact geometry.

    ``margin`` is where the penalty starts, not a hard constraint -- the arm may still enter
    it, it just pays. A convex quadratic gives a direction to move in before anything touches;
    a step at contact gives none until it is too late.
    """
    asset = env.scene[asset_cfg.name]
    d = total_clearance(asset, cap=cap, include_self=include_self)
    ref = getattr(env, "_sensor_clearance_ref", None)
    if ref is None:
        ref = d.detach().clone()
        env._sensor_clearance_ref = ref
    excess = (margin - (d - ref)).clamp(min=0.0) / margin
    return -(excess**2)
