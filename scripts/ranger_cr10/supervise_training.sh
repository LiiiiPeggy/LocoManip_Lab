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

# Stage 4.11: run a long PPO job to a target iteration, surviving the value-function blow-ups.
#
# The failure this exists for: `rsl_rl` died inside a PPO update at iteration 1405 with
# `torch.normal` and a negative std. The environment was healthy one iteration earlier, the
# process exited without a usable status (Isaac Lab's `close()` swallows the traceback), and
# nothing restarted it -- the run was dead for four hours while a `pgrep -f` on the script name
# kept matching the shell that launched it.
#
# So this writes down what actually happened: the child PID while it runs, the child's exit
# code when it stops, and the iteration it stopped at. Before resuming it verifies the newest
# checkpoint loads and holds only finite tensors, because a corrupted file is not a place to
# restart from.
#
# Usage:
#   scripts/ranger_cr10/supervise_training.sh \
#       --target 3500 --run_name stage411_ppo \
#       --resume logs/rsl_rl/ranger_cr10_teleop/2026-11-04_12-11-51_stage411_ppo/model_1000.pt \
#       --train_args "--task RANGER-CR10-TELEOP-WBC --num_envs 2048 --headless --seed 42"
#
# Check on it with:  python scripts/ranger_cr10/check_training_health.py --pattern "*stage411_ppo*"
set -uo pipefail

TARGET=0
RUN_NAME=""
FIRST_CKPT=""
TRAIN_ARGS=""
STATUS=outputs/stage411/supervisor_status.txt
LOG=outputs/stage411/train_supervisor.log
MAX_ATTEMPTS=5

while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET=$2; shift 2 ;;
    --run_name) RUN_NAME=$2; shift 2 ;;
    --resume) FIRST_CKPT=$2; shift 2 ;;
    --train_args) TRAIN_ARGS=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$RUN_NAME" ] || { echo "--run_name is required" >&2; exit 2; }

mkdir -p "$(dirname "$STATUS")" "$(dirname "$LOG")"
say() { echo "$(date -Is)  $*" | tee -a "$STATUS"; }

# newest checkpoint across every run directory of this name
newest() {
  ls -t logs/rsl_rl/ranger_cr10_teleop/*"$RUN_NAME"/model_*.pt 2>/dev/null | head -1
}

# exit status of a checkpoint: 0 when every tensor is finite
verify() {
  python - "$1" <<'PY'
import sys, torch
try:
    payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
except Exception as error:  # noqa: BLE001
    print(f"unreadable: {error}")
    raise SystemExit(1)
bad = []
def scan(obj, trail=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            scan(value, f"{trail}/{key}")
    elif isinstance(obj, torch.Tensor) and not bool(torch.isfinite(obj).all()):
        bad.append(trail)
scan(payload)
if bad:
    print("non-finite: " + ", ".join(bad[:5]))
    raise SystemExit(1)
PY
}

CKPT=$FIRST_CKPT
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    CKPT=$(newest)
  fi
  [ -n "$CKPT" ] || { say "no checkpoint to start from"; exit 3; }
  ITER=$(basename "$CKPT" | sed 's/model_//;s/\.pt//')
  if [ "$ITER" -ge "$TARGET" ]; then
    say "done: newest checkpoint is $ITER >= $TARGET"
    exit 0
  fi

  # walk back to a checkpoint that loads, rather than resuming a broken file
  while ! verify "$CKPT" > /dev/null 2>&1; do
    say "checkpoint $CKPT failed verification, stepping back"
    PREV=$(ls -t logs/rsl_rl/ranger_cr10_teleop/*"$RUN_NAME"/model_*.pt 2>/dev/null |
      awk -v cur="$(basename "$CKPT")" 'BEGIN{seen=0} {if(seen)print; if($0 ~ cur)seen=1}' | head -1)
    [ -n "$PREV" ] || { say "no verified checkpoint left"; exit 4; }
    CKPT=$PREV
    ITER=$(basename "$CKPT" | sed 's/model_//;s/\.pt//')
  done

  REMAIN=$((TARGET - ITER))
  RUN_DIR=$(basename "$(dirname "$CKPT")")
  say "attempt $attempt: resume $CKPT (iter $ITER), $REMAIN to go"

  python -u scripts/rsl_rl/train.py $TRAIN_ARGS \
    --max_iterations "$REMAIN" --resume --load_run "$RUN_DIR" \
    --checkpoint "$(basename "$CKPT")" --run_name "$RUN_NAME" >> "$LOG" 2>&1 &
  CHILD=$!
  say "attempt $attempt: child pid $CHILD"
  wait "$CHILD"
  CODE=$?
  say "attempt $attempt: child exited with $CODE at $(newest || echo 'no checkpoint')"
  [ "$CODE" -eq 0 ] && exit 0
  CKPT=""
done

say "gave up after $MAX_ATTEMPTS attempts"
exit 5
