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

"""Stage 4.11: measure what actually explodes when `rsl_rl`'s value loss goes to 1e8.

Two runs died the same way -- 4.10e at iteration 9336 and 4.11 at 1405 -- with value loss
jumping 3 -> 1.6e8 -> 6e28 -> inf in three iterations while every environment metric stayed
healthy, and `torch.normal(std < 0)` ending the process. The logged loss alone cannot say
*which* quantity went first, and assuming it is "the new reward" would be as unfounded as
assuming it is the value target.

This instruments the update and separates the candidate quantities every iteration:

  value prediction   what the critic output for the collected rollout
  value target       the bootstrapped return it is asked to fit
  advantage          target minus prediction, raw and after normalisation
  gradient           the pre-clipping norm of the actor and critic gradients

`PPO.compute_returns`, `PPO.update` and `torch.nn.utils.clip_grad_norm_` are wrapped, not
edited: the installed `rsl_rl` is shared with every other project in this environment, and a
diagnostic must not change the thing it measures. The wrappers only read.

Usage:
    python scripts/ranger_cr10/diagnose_value_blowup.py \
        --resume logs/rsl_rl/ranger_cr10_teleop/2026-11-04_12-11-51_stage411_ppo/model_1000.pt \
        --iterations 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--resume", required=True, help="checkpoint to start from (weights only)")
ap.add_argument("--iterations", type=int, default=300)
ap.add_argument("--num_envs", type=int, default=2048)
ap.add_argument("--task", default="RANGER-CR10-TELEOP-WBC")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default="outputs/stage411/value_diagnostics.jsonl")
ap.add_argument(
    "--log_dir",
    default="",
    help="where rsl_rl writes its event files and checkpoints; a fresh directory avoids the "
    "TensorBoard writer sharing a directory with a previous attempt",
)
ap.add_argument(
    "--full_resume",
    action="store_true",
    help="restore the optimiser and iteration counter as well (continue a lineage) instead of "
    "starting a fresh optimiser at iteration 0",
)
ap.add_argument("--stop_on_nonfinite", action="store_true", default=True)
ap.add_argument(
    "--reward_terms",
    action="store_true",
    default=False,
    help="also record each reward term's peak. Off by default: the per-step device-to-host sync it "
    "needs serialises the rollout and made a 2048-env iteration take minutes instead of seconds.",
)
args = ap.parse_args()

if os.environ.get("STAGE411_FAULTHANDLER"):
    # a hung run that prints nothing is worse than a crash: dump where it is instead of guessing
    import faulthandler

    faulthandler.dump_traceback_later(int(os.environ["STAGE411_FAULTHANDLER"]), exit=True)

app_launcher = AppLauncher(headless=True, device="cuda:0")
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import LeggedManip_Lab.tasks  # noqa: E402,F401
from rsl_rl.algorithms.ppo import PPO  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    RangerCr10TeleopPPORunnerCfg,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.ranger_cr10.teleop_env_cfg import (  # noqa: E402
    RangerCr10TeleopEnvCfg,
)

OUT = Path(args.out)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("")
STATE = {"iteration": 0, "last_record": {}}


def stats(t: torch.Tensor | None) -> dict | None:
    if t is None:
        return None
    t = t.detach().float()
    return dict(
        mean=float(t.mean()),
        absmax=float(t.abs().max()),
        nonfinite=int((~torch.isfinite(t)).sum()),
        numel=int(t.numel()),
    )


_orig_compute_returns = PPO.compute_returns
_orig_update = PPO.update
_orig_clip = torch.nn.utils.clip_grad_norm_


def compute_returns(self, obs):  # noqa: ANN001 - wrapped method
    _orig_compute_returns(self, obs)
    st = self.storage
    raw = st.returns - st.values
    STATE["last_record"].update(
        values=stats(st.values),
        rewards=stats(st.rewards),
        returns=stats(st.returns),
        advantages_raw=stats(raw),
        advantages_norm=stats(st.advantages),
    )


def update(self):  # noqa: ANN001 - wrapped method
    STATE["iteration"] += 1
    # keep what `compute_returns` recorded for this iteration; only add the update half
    STATE.setdefault("last_record", {}).update(
        {"iteration": STATE["iteration"], "grad_norm": None, "grad_nonfinite": 0}
    )
    out = _orig_update(self)
    STATE["last_record"]["loss"] = {k: float(v) for k, v in (out or {}).items()} if isinstance(out, dict) else {}
    peaks = {k: float(v) for k, v in STATE.pop("term_peak", {}).items()} if STATE.get("term_peak") else {}
    STATE["last_record"]["reward_term_peak"] = peaks
    line = json.dumps(STATE["last_record"])
    with OUT.open("a") as handle:
        handle.write(line + "\n")
    rec = STATE["last_record"]
    print(
        f"it {rec['iteration']:4d}  V {rec['values']['absmax']:.3e}  R {rec['returns']['absmax']:.3e}  "
        f"A {rec['advantages_norm']['absmax']:.3e}  grad {rec['grad_norm'] or float('nan'):.3e}  "
        f"loss {rec['loss'].get('value', float('nan')):.3e}  "
        f"rpeak {max(peaks.values()) if peaks else 0:.3e}",
        flush=True,
    )
    if rec["values"]["nonfinite"] or rec["returns"]["nonfinite"] or rec["grad_nonfinite"]:
        print("non-finite detected, stopping", flush=True)
        raise SystemExit(9)
    return out


def clip_grad_norm(parameters, max_norm, *a, **kw):
    params = [p for p in parameters if p.grad is not None]
    if params:
        total = torch.sqrt(sum((p.grad.detach().float() ** 2).sum() for p in params))
        STATE["last_record"]["grad_norm"] = float(total)
        if not bool(torch.isfinite(total)):
            STATE["last_record"]["grad_nonfinite"] = 1
    return _orig_clip(parameters, max_norm, *a, **kw)


PPO.compute_returns = compute_returns
PPO.update = update
torch.nn.utils.clip_grad_norm_ = clip_grad_norm


def instrument_reward_terms(env) -> None:
    """Record each reward term's largest absolute value this iteration.

    A value target only explodes if a *reward* did, and the total reward cannot say which term
    did it. The per-term maxima are accumulated on the device and read once per iteration, so
    the instrumentation costs one small GPU op per term per step rather than a host sync.
    """
    manager = getattr(env.unwrapped, "reward_manager", None)
    if manager is None:
        print("no reward manager found, per-term diagnostics disabled", flush=True)
        return
    terms = getattr(manager, "_term_cfgs", None) or getattr(manager, "term_cfgs", None) or []
    for name, cfg in zip(manager._term_names, terms):
        original = cfg.func

        def wrapped(*a, __original=original, __name=name, **kw):
            value = __original(*a, **kw)
            peak = STATE.setdefault("term_peak", {})
            current = peak.get(__name)
            local = value.detach().abs().max()
            peak[__name] = local if current is None else torch.maximum(current, local)
            return value

        cfg.func = wrapped
    print(f"instrumented {len(terms)} reward terms", flush=True)


def main() -> None:
    agent_cfg = RangerCr10TeleopPPORunnerCfg()
    # train.py does this through hydra; without it the legacy `stochastic` key survives into
    # `to_dict()` and rsl-rl >= 5 rejects it when it builds the actor (TypeError, swallowed by
    # Isaac's close() -- which is why the first attempts looked like a hang with no message).
    import importlib.metadata as metadata

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.iterations
    agent_cfg.device = "cuda:0"

    env_cfg = RangerCr10TeleopEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    # the command markers cost more than the rollout at 2048 envs and change nothing being measured
    env_cfg.commands.ee_pose.debug_vis = False
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    log_dir = Path(args.log_dir or "logs/rsl_rl/ranger_cr10_teleop/_stage411_diag")
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.reward_terms:
        instrument_reward_terms(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    print("[diag] runner built, loading checkpoint", flush=True)
    if args.full_resume:
        runner.load(args.resume)
    else:
        runner.load(args.resume, load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False})
    print("[diag] checkpoint loaded, starting learn", flush=True)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    # Isaac Lab's `close()` terminates the process from native code, so an exception raised in
    # `main` is otherwise never printed and the process still looks like it merely stopped
    # (AGENTS.md). Print it here, before close has the chance to swallow it.
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
