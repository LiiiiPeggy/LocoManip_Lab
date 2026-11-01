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

"""Stage 4.7: one description of the end-effector workspace, shared by both sides.

The Isaac Lab task and the MuJoCo deployment used to carry the six limits twice -- the training
box in ``teleop_env_cfg.py`` and the keyboard clamps in ``keyboard_teleop.py`` -- and the only
thing keeping them equal was a check script. With the box now staged (it grows during training)
that duplication would drift immediately, so both sides import *this* module:

* the env config reads ``STAGE_BOXES`` for the curriculum and ``TARGET_BOX`` as its limit;
* the keyboard controller takes its clamps from ``KEYBOARD_BOX``, which is the verified stage.

Pure Python on purpose: no numpy, no Isaac imports, so the MuJoCo process can use it too. The
import from inside the extension package is done by explicit path -- see ``teleop_env_cfg.py``.

Ranges are offsets from the default TCP pose in ``base_link``, exactly as the command term and
the keyboard already define them. Angles are radians here (the training side's unit) and are
converted where a degree API exists.

Where the asymmetric limits come from
-------------------------------------
``check_workspace_reach.py`` measures, from the default posture and moving 1 cm / 2 deg at a
time, how far the TCP actually goes while *staying continuous* -- which is what a policy that
starts at the default pose can use. Measured (``outputs/stage47/workspace_reach.json``):

    position, default attitude   X -0.05..+0.45+   Y -0.45+..+0.07   Z -0.26..+0.45+   (m)
    position, pitch +45 deg      X -0.24..+0.45+   Y -0.45+..+0.34   Z -0.45+..+0.45+
    attitude, default position   roll -18..+70     pitch -14..+70    yaw -70..+70      (deg)

Two things follow and are *not* smoothed over:

* the negative roll/pitch side is the tight one (about -15 deg at the default position, more
  when the position moves too), so ``TARGET_BOX`` keeps the brief's +-45 deg on the positive
  side and caps the negative side at -30 deg, which the multi-restart IK does reach through
  the arm's other branch;
* everything else reaches the brief's numbers, so the box is the brief's box outside that cap.

The box stays rectangular on purpose -- the operator drives single axes with the keyboard, and
a rectangular box is what lets the same key mean the same thing everywhere. The unreachable
*corners* are documented in section 17 and are what the success-based curriculum handles: the
policy is never asked to hold a corner until it has solved the box it already has.
"""

from __future__ import annotations

import math

POS_AXES = ("pos_x", "pos_y", "pos_z")
RPY_AXES = ("roll", "pitch", "yaw")


def _box(px, py, pz, roll, pitch, yaw):
    return {"pos_x": px, "pos_y": py, "pos_z": pz, "roll": roll, "pitch": pitch, "yaw": yaw}


# The box the policy was trained on up to stage 4.5j. Kept as stage 0 and as the replay
# distribution; do not edit, it is the accuracy reference every expansion is judged against.
OLD_BOX = _box((-0.12, 0.12), (-0.15, 0.15), (-0.12, 0.10), (-0.40, 0.40), (-0.40, 0.40), (-0.50, 0.50))


def _scaled(f):
    return _box(
        (-0.12 * f, 0.12 * f),
        (-0.15 * f, 0.15 * f),
        (-0.12 * f, 0.10 * f),
        (-0.40 * f, 0.40 * f),
        (-0.40 * f, 0.40 * f),
        (-0.50 * f, 0.50 * f),
    )


# The candidate box from the 4.7 brief, applied except where the reach study says otherwise:
# pitch is capped at -30 deg on the negative side (see the module docstring).
TARGET_BOX = _box(
    (-0.30, 0.30),
    (-0.35, 0.35),
    (-0.25, 0.30),
    (-math.radians(45), math.radians(45)),
    (-math.radians(30), math.radians(45)),
    (-math.radians(60), math.radians(60)),
)

STAGE_BOXES = (_scaled(1.0), _scaled(1.5), _scaled(2.0), TARGET_BOX)
STAGE_NAMES = ("stage0_old", "stage1_1.5x", "stage2_2x", "stage3_target")

# Probability that a resample draws from the *old* box instead of the current stage, so the
# accuracy the earlier stages paid for does not decay while the box grows.
REPLAY_RATIO = 0.30

# Fraction of *environments currently on target* (position error under the command term's
# 0.10 m threshold) that lets the curriculum promote. 0.85 rather than 0.75: the statistic is
# the instantaneous one, and demanding 85% of 2048 environments simultaneously on target
# already leaves the box corners unsolved. Minimum time in a stage is enforced separately
# (2000 policy steps), so a stage cannot be skipped by a lucky sample.
PROMOTE_AT = 0.85

# Stage a *new* run starts at. Zero for a fresh curriculum. The 4.7 run was resumed at 1: its
# first 2000 iterations ran at stage 1 with success pinned at 0.775 +- 0.04 against a strict
# instantaneous gate, so the gate was made an exponential moving average and the run restarted
# from its own checkpoint at stage 1 rather than replaying stage 0.
START_STAGE = 1

# What the keyboard may command: the *measured* usable range of the deployed policy, not the
# box it was trained on. `teleop_eval.py --suite workspace` pushes each axis in 0.05 m / 10 deg
# steps and holds 3 s; the numbers below are the furthest offset whose steady-state error stays
# within 50 mm and 5 deg (rounded to the grid, and never beyond the stage-3 box).
#
# Measured on the stage-4.7 policy (`stage47c_ppo/model_2999`, outputs/stage47/e1_new):
#
#     X   -0.10 .. +0.25      Y   -0.35 .. +0.10      Z   -0.20 .. +0.25
#     roll -30 .. +40 deg     pitch -20 .. +40 deg    yaw -60 .. +60 deg
#
# Two of these are *asymmetric on purpose*: the arm reaches 0.25 m forward and only 0.10 m
# back, and it rolls/pitches further one way than the other. The old box (uniform +-0.12 /
# +-0.15 / -0.12..0.10, +-22.9 deg) was symmetric because it was never measured.
KEYBOARD_BOX = {
    "pos_x": (-0.10, 0.25),
    "pos_y": (-0.35, 0.10),
    "pos_z": (-0.20, 0.25),
    "roll": (-math.radians(30), math.radians(40)),
    "pitch": (-math.radians(20), math.radians(40)),
    "yaw": (-math.radians(60), math.radians(60)),
}


def as_ranges_kwargs(box):
    """``**kwargs`` for a ``Ranges`` configclass."""
    return {k: tuple(v) for k, v in box.items()}


def degrees(box):
    """The same box with the three angles in degrees, for the keyboard controller."""
    out = dict(box)
    for axis in RPY_AXES:
        out[axis] = (math.degrees(box[axis][0]), math.degrees(box[axis][1]))
    return out
