#!/usr/bin/env bash
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

# Stage 4.11: the fixed-protocol evaluation, one checkpoint at a time.
#
# Every checkpoint is measured with the same targets, the same seed and the same MuJoCo model,
# so a difference between rows is a difference between policies and nothing else:
#   coverage   60 six-dimensional targets drawn from the 4.10 box -- success at 50 mm / 5 deg
#              and 30 mm / 2 deg, position and orientation error
#   lowspace   the low workspace the 4.10 pose opened, and how much of it the policy covers
#   collisions the real contacts, counted by MuJoCo, over the same random targets --
#              `Link3 <-> Link5` and every other arm-internal pair
#   modes     chassis tracking with the arm moving
#   drift     what the robot does when the command is zero
#
# Usage: scripts/ranger_cr10/eval_stage411.sh <checkpoint> <tag>
set -euo pipefail

CKPT=${1:?checkpoint path}
TAG=${2:?tag}
OUT=outputs/stage411/eval
mkdir -p "$OUT"

python -u mujoco/deploy/deploy_mujoco/ranger_cr10/teleop_eval.py \
  --policy "$CKPT" --suite coverage --coverage_box 4.10 --out "$OUT" --tag "${TAG}_cov" \
  > "$OUT/${TAG}_cov.log" 2>&1
python -u mujoco/deploy/deploy_mujoco/ranger_cr10/teleop_eval.py \
  --policy "$CKPT" --suite lowspace --out "$OUT" --tag "${TAG}_low" \
  > "$OUT/${TAG}_low.log" 2>&1
python -u mujoco/deploy/deploy_mujoco/ranger_cr10/teleop_eval.py \
  --policy "$CKPT" --suite modes --out "$OUT" --tag "${TAG}_modes" \
  > "$OUT/${TAG}_modes.log" 2>&1
python -u mujoco/deploy/deploy_mujoco/ranger_cr10/teleop_eval.py \
  --policy "$CKPT" --suite drift --out "$OUT" --tag "${TAG}_drift" \
  > "$OUT/${TAG}_drift.log" 2>&1
python -u scripts/ranger_cr10/check_self_collision.py \
  --policy "$CKPT" --random_targets 60 --box 4.10 --seed 5 --out "$OUT" \
  > "$OUT/${TAG}_col.log" 2>&1

echo "done: $TAG"
