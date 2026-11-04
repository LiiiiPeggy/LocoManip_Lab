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

"""Is the training run actually alive, and are its checkpoints usable?

Written after getting this wrong once: `pgrep -f "rsl_rl/train.py"` matches the *shell* that
launched it (and the shell running the check), so a dead run looked alive for four hours. The
process is identified by reading `/proc/*/cmdline` and requiring an argv[0] that is a Python
interpreter and an argv that names `scripts/rsl_rl/train.py` -- shells that merely mention the
string do not match.

The checkpoint side is the other half: a run can die *inside* a PPO update and leave the last
saved file untouched (that is what happened at iteration 1405), so the newest file is checked
by loading it and rejecting any tensor that is not finite. A checkpoint that fails this is not
a resume point.

Usage:
    python scripts/ranger_cr10/check_training_health.py --pattern "*stage411_ppo*" --target 3500
    python scripts/ranger_cr10/check_training_health.py --pattern "*stage411_ppo*" --json out.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = ROOT / "logs" / "rsl_rl" / "ranger_cr10_teleop"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--pattern", default="*", help="glob on the run directory name")
ap.add_argument("--script", default="scripts/rsl_rl/train.py", help="train script the process must name")
ap.add_argument("--target", type=int, default=None, help="iteration count the run is aiming for")
ap.add_argument("--json", default="")
ap.add_argument("--supervisor-status", default="outputs/stage411/supervisor_status.txt")
args = ap.parse_args()


def training_processes(script: str) -> list[dict]:
    """Processes whose argv[0] is a Python interpreter and whose argv names ``script``."""
    out = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            argv = [a.decode(errors="replace") for a in argv if a]
        except (FileNotFoundError, PermissionError):
            continue
        if not argv:
            continue
        if "python" not in os.path.basename(argv[0]):
            continue
        if not any(a.endswith(script) for a in argv):
            continue
        try:
            stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            state = stat[0]
            start_ticks = int(stat[19])
        except (FileNotFoundError, IndexError):
            continue
        # a zombie has already exited; its exit status is only visible to its parent
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        elapsed = uptime - start_ticks / os.sysconf("SC_CLK_TCK")
        out.append(dict(pid=int(entry.name), state=state, elapsed_s=round(elapsed, 1), argv=argv[1:]))
    return out


def checkpoints(pattern: str) -> list[dict]:
    rows = []
    for path in glob.glob(str(LOG_ROOT / pattern / "model_*.pt")):
        match = re.search(r"model_(\d+)\.pt$", path)
        if not match:
            continue
        rows.append(dict(path=Path(path), iteration=int(match.group(1)), run=Path(path).parent.name))
    return sorted(rows, key=lambda r: (r["iteration"], r["run"]))


def verify(path: Path) -> dict:
    """Load one checkpoint and check every tensor is finite."""
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - a corrupt file is a result, not a crash
        return dict(ok=False, reason=f"unreadable: {error}")
    bad = []

    def scan(obj, trail=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                scan(value, f"{trail}/{key}")
        elif isinstance(obj, torch.Tensor) and not bool(torch.isfinite(obj).all()):
            bad.append(trail)

    scan(payload)
    recorded = payload.get("iter")
    return dict(ok=not bad, reason="non-finite tensors: " + ", ".join(bad[:5]) if bad else "", iter=recorded)


def main() -> None:
    procs = training_processes(args.script)
    ckpts = checkpoints(args.pattern)
    newest = ckpts[-1] if ckpts else None

    print(f"训练进程: {len(procs)} 个")
    for p in procs:
        print(f"  pid {p['pid']}  state {p['state']}  已运行 {p['elapsed_s'] / 60:.1f} 分钟")
        print(f"    {' '.join(p['argv'])[:160]}")
    status = Path(args.supervisor_status)
    if status.exists():
        print("\n监督状态:")
        for line in status.read_text().splitlines()[-6:]:
            print(f"  {line}")
    if not procs:
        print("  没有正在训练的进程 —— 如果目标轮数还没到, 这就是一次未被处理的退出")

    print(f"\ncheckpoint: {len(ckpts)} 个")
    for row in ckpts[-3:]:
        info = verify(row["path"])
        flag = "ok" if info["ok"] else f"BAD ({info['reason']})"
        print(f"  {row['run']}/model_{row['iteration']}.pt  iter={info.get('iter')}  {flag}")
    if newest and args.target is not None:
        print(f"\n最新 {newest['iteration']} / 目标 {args.target}"
              f" —— {'已达标' if newest['iteration'] >= args.target else '未达标'}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(dict(checked_at=time.strftime("%Y-%m-%dT%H:%M:%S"), processes=procs,
                            checkpoints=[dict(run=r["run"], iteration=r["iteration"]) for r in ckpts]),
                       indent=1)
        )
        print(f"wrote {args.json}")

    # non-zero exit when nothing is running and the target is not reached
    if not procs and args.target is not None and (newest is None or newest["iteration"] < args.target):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
