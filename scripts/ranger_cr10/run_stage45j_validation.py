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

"""Stage 4.5j: the fixed validation protocol, applied to every saved checkpoint.

Two measurements per checkpoint, both run through the *existing* scripts so the numbers are
directly comparable with 4.2/4.5i:

* `eval_teleop_wbc.py`  -- modes A/B/B'/C plus the roll sweep, 256 envs. Its input is a
  TorchScript policy, so each rsl_rl checkpoint is exported first (same MLP, same weights).
* `check_teleop_stage45i.py` -- IK handover (C1) and active convergence (C2) on commands the
  supervision never contained. Optional, because it costs ~8 min per checkpoint.

The tables are parsed out of stdout rather than recomputed: the point is to reuse the verified
protocols unchanged, not to re-implement them a third time.

Usage:
    python scripts/ranger_cr10/run_stage45j_validation.py \
        --run_dir logs/rsl_rl/ranger_cr10_teleop/<run> --iters 500 1000 ... \
        --closedloop_iters 500 3000 6000
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description="Validate every stage-4.5j checkpoint.")
parser.add_argument("--run_dir", required=True)
parser.add_argument("--iters", nargs="+", type=int, required=True)
parser.add_argument("--closedloop_iters", nargs="*", type=int, default=[])
parser.add_argument("--out", default="outputs/stage45j")
parser.add_argument("--mode3_envs", type=int, default=256)
parser.add_argument("--n_test_cmds", type=int, default=4)
parser.add_argument("--warm", type=int, default=25)
parser.add_argument("--skip_mode3", action="store_true")
parser.add_argument("--skip_closedloop", action="store_true")
args_cli = parser.parse_args()

OUT = Path(args_cli.out)
OUT.mkdir(parents=True, exist_ok=True)
RUN = Path(args_cli.run_dir)
PY = sys.executable
NUM = re.compile(r"([+-]?\d+(?:\.\d+)?)")


def export(ckpt: Path, dst: Path) -> Path:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    raw = ck["actor_state_dict"] if isinstance(ck, dict) and "actor_state_dict" in ck else ck
    sd = {(k[4:] if k.startswith("mlp.") else k): v for k, v in raw.items() if k.split(".")[-1] in ("weight", "bias")}
    net = torch.nn.Sequential(
        torch.nn.Linear(117, 512),
        torch.nn.ELU(),
        torch.nn.Linear(512, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 8),
    )
    net.load_state_dict(sd)
    net.eval()
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(net).save(str(dst))
    return dst


def run(cmd: list[str], log: Path) -> str:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    txt = log.read_text(errors="replace")
    if p.returncode != 0 or "Traceback" in txt:
        print(f"    ** 运行失败: {log}")
    return txt


MODE_KEYS = {
    "A 只控制底盘": "A",
    "B 只控制末端": "B",
    "B' 只改姿态": "Bp",
    "C 同时控制": "C",
    "C 的解耦基线": "C_dec",
    "A 的解耦基线": "A_dec",
}


def parse_mode3(txt: str) -> dict:
    out = {}
    for ln in txt.splitlines():
        s = ln.strip()
        for name, key in MODE_KEYS.items():
            if s.startswith(name):
                v = NUM.findall(s[len(name) :])
                if len(v) >= 9:
                    out[key] = dict(
                        lin_err=float(v[0]),
                        ang_err=float(v[1]),
                        pos_err=float(v[2]),
                        ori_err=float(v[3]),
                        ori_cmd=float(v[4]),
                        ori_act=float(v[5]),
                        success=float(v[6].rstrip("%")) if "%" in s else float(v[6]),
                        base_spd=float(v[7]),
                        smooth=float(v[9]) if len(v) > 9 else None,
                    )
    # attitude sweep: rows of four numbers after the header
    sweep = []
    in_sweep = False
    for ln in txt.splitlines():
        if "姿态偏移幅值扫描" in ln:
            in_sweep = True
            continue
        if in_sweep:
            v = NUM.findall(ln)
            if len(v) == 4:
                sweep.append([float(x) for x in v])
            elif ln.strip().startswith("("):
                break
    if sweep:
        out["att_sweep_resid_0p2"] = sweep[-1][3]
        out["att_sweep"] = sweep
    return out


SUM_HDR = re.compile(r"策略\s+C1 首步")


def parse_closedloop(txt: str) -> dict:
    out = {}
    lines = txt.splitlines()
    for i, ln in enumerate(lines):
        if SUM_HDR.search(ln):
            for row in lines[i + 2 :]:
                s = row.strip()
                if not s or s.startswith("在线 IK") or s.startswith("底盘"):
                    break
                # the first token is the policy *name*, and names like `mix11` contain digits
                parts = s.split(None, 1)
                if len(parts) < 2:
                    break
                v = [float(x) for x in NUM.findall(parts[1])]
                if len(v) >= 9:
                    out["cl_" + parts[0]] = dict(
                        c1_first=v[0],
                        c1_05=v[1],
                        c1_25=v[2],
                        c1_pos_mm=v[3],
                        c2_e0=v[4],
                        c2_05=v[5],
                        c2_25=v[6],
                        c2_pos_mm=v[7],
                        jitter=v[8],
                    )
    return out


rows = []
for it in args_cli.iters:
    ckpt = RUN / f"model_{it}.pt"
    if not ckpt.exists():
        print(f"[{it}] {ckpt} 不存在, 跳过")
        continue
    rec = {"iter": it}
    print(f"\n=== iteration {it} ===")
    if not args_cli.skip_mode3:
        pol = export(ckpt, OUT / "exported" / f"policy_{it}.pt")
        txt = run(
            [
                PY,
                "scripts/ranger_cr10/eval_teleop_wbc.py",
                "--policy",
                str(pol),
                "--num_envs",
                str(args_cli.mode3_envs),
            ],
            OUT / f"mode3_{it}.log",
        )
        m = parse_mode3(txt)
        rec["mode3"] = m
        if "A" in m and "C" in m:
            print(
                f"  A: lin {m['A']['lin_err']:.4f} pos {m['A']['pos_err']:.4f} ori {m['A']['ori_err']:.4f} | "
                f"B: pos {m['B']['pos_err']:.4f} succ {m['B']['success']:.1f} | "
                f"B': ori {m['Bp']['ori_err']:.4f} | C: pos {m['C']['pos_err']:.4f} ori {m['C']['ori_err']:.4f} | "
                f"sweep0.2 {m.get('att_sweep_resid_0p2', float('nan')):.4f}"
            )
        else:
            print("  ** mode3 解析失败")
    if not args_cli.skip_closedloop and it in set(args_cli.closedloop_iters):
        txt = run(
            [
                PY,
                "scripts/ranger_cr10/check_teleop_stage45i.py",
                "--policies",
                f"ppo{it}={ckpt}",
                "--cmds_json",
                "logs/rsl_rl/ranger_cr10_teleop/_stage45i/train_mix21.json",
                "--n_test_cmds",
                str(args_cli.n_test_cmds),
                "--warm",
                str(args_cli.warm),
                "--out",
                str(OUT),
            ],
            OUT / f"closedloop_{it}.log",
        )
        c = parse_closedloop(txt)
        rec["closedloop"] = c
        k = f"cl_ppo{it}"
        if k in c:
            d = c[k]
            print(
                f"  C1 首步 {d['c1_first']:.2f}° 2.5s {d['c1_25']:.2f}° 位置 {d['c1_pos_mm']:.0f}mm | "
                f"C2 2.5s {d['c2_25']:.2f}° 位置 {d['c2_pos_mm']:.0f}mm"
            )
        else:
            print("  ** closedloop 解析失败")
    rows.append(rec)
    (OUT / "validation.json").write_text(json.dumps(rows, indent=1, default=float))
    with open(OUT / "validation.csv", "w") as f:
        f.write("iter,A_lin,A_pos,A_ori,B_pos,B_succ,Bp_ori,C_pos,C_ori,att0p2,cl_first,cl_25,cl_pos_mm\n")
        for r in rows:
            m, c = r.get("mode3", {}), r.get("closedloop", {}).get(f"cl_ppo{r['iter']}", {})
            f.write(
                f"{r['iter']},{m.get('A', {}).get('lin_err', '')},{m.get('A', {}).get('pos_err', '')},"
                f"{m.get('A', {}).get('ori_err', '')},{m.get('B', {}).get('pos_err', '')},"
                f"{m.get('B', {}).get('success', '')},{m.get('Bp', {}).get('ori_err', '')},"
                f"{m.get('C', {}).get('pos_err', '')},{m.get('C', {}).get('ori_err', '')},"
                f"{m.get('att_sweep_resid_0p2', '')},{c.get('c1_first', '')},{c.get('c1_25', '')},"
                f"{c.get('c1_pos_mm', '')}\n"
            )
print(f"\n已写出 {OUT / 'validation.csv'} 与 validation.json")
