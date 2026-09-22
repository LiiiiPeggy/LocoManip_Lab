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

"""End-effector TCP for the RangerBox-CR10, expressed as a virtual frame.

The gripper's fingertip centre is the point that matters, but it cannot be referenced
directly as a body. Adding a frame to the USD does not work either: ``Articulation``
only knows the bodies PhysX reports, and ``find_bodies`` matches against
``root_physx_view.shared_metatype.link_names``. An extra Xform is not one of them, and
``merge_fixed_joints`` would not keep it anyway. Worse, a missing body makes
``end_effector_link0_relative_pose`` silently return zeros.

So the TCP is computed instead: take a real rigid body and apply a constant offset.

    cr10_Link6                          real articulation body, the CR10 flange
        + TCP_OFFSET_POS in its frame   midpoint of the two fingertips
        = virtual fingertip-centre TCP

``TCP_OFFSET_POS`` was measured rather than derived by hand: the USD was loaded as an
articulation, allowed to settle, and the midpoint of ``gripper_finger1_finger_tip_link``
and ``gripper_finger2_finger_tip_link`` was read back relative to ``cr10_Link6``.

``gripper_base_link`` does not exist as a body in this USD -- ``merge_fixed_joints``
folded it into ``cr10_Link6`` -- which is why the anchor is the flange.

The offset is constant only while the gripper does not move. That holds here because
the gripper is not in the action space; if it ever is, this becomes a function of the
finger joint angle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.utils.io_descriptors import generic_io_descriptor, record_dtype, record_shape
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# -- measured from the converted USD, see the module docstring ------------------
TCP_ANCHOR_BODY = "cr10_Link6"
TCP_OFFSET_POS = (0.0, 0.00003, 0.14389)
TCP_OFFSET_QUAT = (1.0, 0.0, 0.0, 0.0)


def _offset_tensors(asset: Articulation, offset_pos, offset_quat):
    pos = torch.tensor(offset_pos, device=asset.device, dtype=torch.float32).unsqueeze(0)
    quat = torch.tensor(offset_quat, device=asset.device, dtype=torch.float32).unsqueeze(0)
    return pos.expand(asset.num_instances, 3), quat.expand(asset.num_instances, 4)


def tcp_pose_w(
    asset: Articulation,
    anchor_body_id: int,
    offset_pos: tuple[float, float, float] = TCP_OFFSET_POS,
    offset_quat: tuple[float, float, float, float] = TCP_OFFSET_QUAT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """World position and orientation of the virtual TCP, given its anchor body."""
    anchor_pos_w = asset.data.body_pos_w[:, anchor_body_id]
    anchor_quat_w = asset.data.body_quat_w[:, anchor_body_id]
    off_pos, off_quat = _offset_tensors(asset, offset_pos, offset_quat)
    return anchor_pos_w + quat_apply(anchor_quat_w, off_pos), quat_mul(anchor_quat_w, off_quat)


def tcp_pose_b(
    asset: Articulation,
    anchor_body_id: int,
    base_body_id: int,
    offset_pos: tuple[float, float, float] = TCP_OFFSET_POS,
    offset_quat: tuple[float, float, float, float] = TCP_OFFSET_QUAT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TCP pose expressed in the frame of ``base_body_id``."""
    tcp_pos_w, tcp_quat_w = tcp_pose_w(asset, anchor_body_id, offset_pos, offset_quat)
    base_pos_w = asset.data.body_pos_w[:, base_body_id]
    base_quat_w = asset.data.body_quat_w[:, base_body_id]

    pos_b = quat_apply(quat_conjugate(base_quat_w), tcp_pos_w - base_pos_w)
    quat_b = quat_mul(quat_conjugate(base_quat_w), tcp_quat_w)
    return pos_b, quat_b


@generic_io_descriptor(dtype=torch.float32, observation_type="EndEffectorPose", on_inspect=[record_dtype, record_shape])
def end_effector_tcp_pose(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, base_body_name: str = "base_link") -> torch.Tensor:
    """Policy-facing observation: TCP pose relative to the chassis, 7-D.

    Stands in for the repo's ``end_effector_link0_relative_pose``, which looks up a body
    literally named ``end_effector`` and silently returns zeros when it is absent.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    base_ids, _ = asset.find_bodies(base_body_name)
    pos_b, quat_b = tcp_pose_b(asset, asset_cfg.body_ids[0], base_ids[0])
    return torch.cat([pos_b, quat_b], dim=-1)


def position_command_world_error_exp(
    env: ManagerBasedEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
    base_body_name: str = "base_link",
) -> torch.Tensor:
    """Reward TCP position tracking against a target that is fixed in the world.

    The command the policy observes is base-relative, so the world target is
    reconstructed from it using the current base pose. That reconstruction is exact:
    the command generator derives its base-relative value from the same frozen world
    target, so reward and task can never disagree about where the target is.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    base_ids, _ = asset.find_bodies(base_body_name)
    base_pos_w = asset.data.body_pos_w[:, base_ids[0]]
    base_quat_w = asset.data.body_quat_w[:, base_ids[0]]
    target_pos_w = base_pos_w + quat_apply(base_quat_w, command[:, :3])

    tcp_pos_w, _ = tcp_pose_w(asset, asset_cfg.body_ids[0])
    return torch.exp(-torch.sum(torch.square(tcp_pos_w - target_pos_w), dim=1) / (std**2))


def orientation_command_world_error(
    env: ManagerBasedEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    base_body_name: str = "base_link",
) -> torch.Tensor:
    """TCP orientation error against the world-fixed target, as a quaternion distance."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    base_ids, _ = asset.find_bodies(base_body_name)
    base_quat_w = asset.data.body_quat_w[:, base_ids[0]]
    target_quat_w = quat_mul(base_quat_w, command[:, 3:7])

    _, tcp_quat_w = tcp_pose_w(asset, asset_cfg.body_ids[0])
    return quat_error_magnitude(tcp_quat_w, target_quat_w)
