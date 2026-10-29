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

"""End-effector pose command for teleoperation: a target held in the chassis frame.

``UniformBodyPoseCommand`` already samples a pose in the robot's root frame and is the
right shape, so this subclasses it rather than duplicating the machinery. Four things
differ, and the first two would each be silently wrong if inherited:

* **The tracked point is the virtual TCP, not the flange.** ``UniformBodyPoseCommand``
  reports the error of ``cfg.body_name`` directly. Here that body is ``cr10_Link6`` and
  the point an operator steers is 143.9 mm further out along the tool (see ``tcp.py``),
  so the inherited metric would be off by the length of the gripper -- a number that
  looks plausible and is wrong.
* **The target is an offset from a measured default pose, not an absolute one.** See
  below; this is the difference that matters most.
* **The debug markers**, which have the same flange/TCP problem for the same reason.
* **A per-command success rate**, so the acceptance tests can report it per mode
  (chassis only / arm only / both) instead of inferring it from an error average.

Why offsets from a fixed default
--------------------------------
The obvious implementation -- sample ``roll/pitch/yaw`` around zero -- assumes the tool's
frame is aligned with the chassis at the default arm configuration. It is not: measured at
the default pose the TCP sits at ``(0.552, -0.074, 0.470)`` m relative to ``base_link``
with an attitude of ``rpy = (-170.5, -9.7, -57.2)`` degrees. Sampling around identity would
therefore command an attitude roughly pi radians away from anything the arm can hold, and
the orientation term would be a near-constant penalty that dominates the reward. The
world-frame task on branch ``rangercr10-world`` lost a full training run to exactly that
mistake, which is why the reference here is a measured constant rather than an assumption.

The default is measured at the *arm's authored configuration*, not read from the scene at
construction: ``gym.make`` leaves every joint at zero until the first physics step, so a
reading taken in ``__init__`` describes a pose the robot is never in (verified -- it put
the TCP 1.58 m above the chassis instead of 0.47 m). The constant is re-checked against the
live scene by ``scripts/ranger_cr10/check_teleop_env.py`` so it cannot rot silently.

Offsets are expressed in the **chassis** frame: positions add directly, and the rotation
is pre-multiplied (``offset @ default``) so a "roll" key tilts the tool about the chassis'
own forward axis regardless of where the wrist currently is. Post-multiplying would make
the same key mean different things at different attitudes.

The chassis frame is the whole design. Because the target is stored relative to
``base_link`` and only ever written here, driving the chassis leaves ``ee_pose_cmd_base``
untouched -- which is the acceptance requirement "with the chassis moving and the operator
not touching the arm keys, the body-relative target does not move". The contrasting
world-frame variant is ``pose_command_world.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_mul

from .pose_command_b import UniformBodyPoseCommand
from .tcp import tcp_pose_b, tcp_pose_w

if TYPE_CHECKING:
    from .cfg.command_cfg import UniformTCPBodyPoseCommandCfg

# -- measured from the converted USD at the authored arm pose, after the physics settles --
# ``cr10_joint1..6 = 0.6, 1.0, -2.4, 0.0, 1.4, 0.0`` rad, read back through ``tcp_pose_b``.
# Re-verify with ``scripts/ranger_cr10/check_teleop_env.py`` after any change to the arm's
# default joint positions or to ``TCP_OFFSET_POS``.
TCP_DEFAULT_POS_B = (0.5519, -0.0742, 0.4696)
TCP_DEFAULT_QUAT_B = (0.032072, -0.875128, 0.469424, -0.112978)


def _uniform_axes(n: int, ranges, device) -> torch.Tensor:
    """Independent uniform samples per axis, shape ``(n, 3)``.

    Written out because the obvious one-liner is silently wrong:

        r = torch.empty(n, device=device)
        torch.stack([r.uniform_(*lo_hi_x), r.uniform_(*lo_hi_y), r.uniform_(*lo_hi_z)], dim=-1)

    ``Tensor.uniform_`` fills *in place and returns the same tensor*, so all three entries of
    the stack are the same object and all three columns hold the last draw. Between 2026-09
    and 2026-10-29 that made every sampled command a point on a two-dimensional line --
    ``offset_x == offset_y == offset_z`` and ``roll == pitch == yaw`` -- so the policy was
    trained on 2 of the 6 commanded dimensions and treated every independent-axis command as
    out of distribution. Measured on the 3142-sample supervision set: all 3142 commands sit
    on that line to 1e-7. Section D2 of ``check_teleop_env.py`` now asserts independence.
    """
    lo = torch.tensor([float(r[0]) for r in ranges], device=device)
    hi = torch.tensor([float(r[1]) for r in ranges], device=device)
    return lo + (hi - lo) * torch.rand(n, 3, device=device)


class UniformTCPBodyPoseCommand(UniformBodyPoseCommand):
    """Samples a chassis-relative target pose around the tool's default pose, tracks the TCP."""

    cfg: UniformTCPBodyPoseCommandCfg

    def __init__(self, cfg: UniformTCPBodyPoseCommandCfg, env):
        super().__init__(cfg, env)
        self.metrics["success"] = torch.zeros(self.num_envs, device=self.device)
        self._success_sum = torch.zeros(self.num_envs, device=self.device)
        self._success_count = torch.zeros(self.num_envs, device=self.device)

    @property
    def success_rate(self) -> torch.Tensor:
        """Fraction of steps since each env's target was sampled that were on target.

        Per command rather than per episode: an episode spans several commands, and the
        question the acceptance test asks is whether *this* command was met.
        """
        return self._success_sum / self._success_count.clamp(min=1.0)

    def _resample_command(self, env_ids):
        n = len(env_ids)

        # expanded to (n, ...) because quat_mul requires matching shapes -- it does not broadcast
        default_pos = torch.tensor(self.cfg.default_pos_b, device=self.device).expand(n, 3)
        default_quat = torch.tensor(self.cfg.default_quat_b, device=self.device).expand(n, 4)

        # -- position: an offset from the default, in the chassis frame ---------------
        offset_pos = _uniform_axes(
            n,
            (self.cfg.ranges.pos_x, self.cfg.ranges.pos_y, self.cfg.ranges.pos_z),
            self.device,
        )
        self.pose_command_b[env_ids, :3] = default_pos + offset_pos

        # -- orientation: an offset about the chassis axes, pre-multiplied ------------
        euler = _uniform_axes(
            n,
            (self.cfg.ranges.roll, self.cfg.ranges.pitch, self.cfg.ranges.yaw),
            self.device,
        )
        offset_quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        self.pose_command_b[env_ids, 3:] = quat_mul(offset_quat, default_quat)

        # a new target starts a new success measurement
        self._success_sum[env_ids] = 0.0
        self._success_count[env_ids] = 0.0

    def _update_metrics(self):
        # the target in the world, for the error and for the debug markers
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.body_pos_w[:, self.root_idx],
            self.robot.data.body_quat_w[:, self.root_idx],
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        tcp_pos_w, tcp_quat_w = tcp_pose_w(self.robot, self.body_idx)
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            tcp_pos_w,
            tcp_quat_w,
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

        success = (self.metrics["position_error"] < self.cfg.success_threshold).float()
        self.metrics["success"] = success
        self._success_sum += success
        self._success_count += 1.0

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        tcp_pos_w, tcp_quat_w = tcp_pose_w(self.robot, self.body_idx)
        self.current_pose_visualizer.visualize(tcp_pos_w, tcp_quat_w)


def measured_tcp_default(asset, anchor_body_name: str, base_body_name: str = "base_link"):
    """Read the live TCP default pose, for the test that guards the constants above."""
    anchor_ids, _ = asset.find_bodies(anchor_body_name)
    base_ids, _ = asset.find_bodies(base_body_name)
    return tcp_pose_b(asset, anchor_ids[0], base_ids[0])
