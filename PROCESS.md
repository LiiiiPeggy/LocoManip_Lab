# PROCESS.md — Engineering History

Why the project ended up the way it is: background, what was tried, what failed, root cause, final
solution, verification.

Append-only in spirit — **keep old entries even when a later one supersedes them**, because the dead
ends are the value. Current state → [PROGRESS.md](PROGRESS.md) · reusable facts →
[MEMORY.md](MEMORY.md).

---

## 1. `scripts/list_envs.py` printed an empty table

**Date:** 2026-09-21 · **Status:** resolved, fix verified, not yet committed

### Background

`python scripts/list_envs.py` is the first command in the README — the quickest way to confirm the
extension is installed and its tasks are registered. It printed a table header and **zero rows**,
with no error. An empty result reads as "this repo registers nothing", which is exactly the wrong
conclusion for someone setting up the environment.

### Attempted solution

Keep the shape of the upstream filter but read the name from the right place: identify the extension
through `spec.kwargs["env_cfg_entry_point"]` instead of `spec.entry_point`, and wrap the printed
value in `str()`. The accompanying code comment blames `entry_point` being a callable under
gymnasium ≥ 1.0.

That comment is **true but incomplete** — it is not the dominant cause (see root cause).

### Investigation

Both predicates were evaluated over the same live `gym.registry`, in one
`AppLauncher(headless=True)` session, so the comparison cannot be distorted by environment
differences:

| filter | matches |
|---|---|
| `isinstance(ep, str) and "LeggedManip" in spec.entry_point` (old) | **0** |
| `"LeggedManip" in str(spec.kwargs.get("env_cfg_entry_point", ""))` (new) | **28** |

Registry composition: 263 specs — 244 with a `str` `entry_point`, 19 with a `function`.

### Root cause

Two independent defects in one line:

1. **The identity check looked at the wrong field.** This repo registers every task with the generic
   `entry_point="isaaclab.envs:ManagerBasedRLEnv"`. That string never contains `LeggedManip`, so the
   `continue` fired for *every* task in the registry — including this extension's own. The filter
   selected nothing, always, by construction.
2. **The comparison was not type-safe.** 19 of 263 specs carry a callable `entry_point`, so
   `"x" not in spec.entry_point` is a latent `TypeError` waiting on registry iteration order rather
   than a correct check.

The failure mode is what made this invisible: a predicate that fails *closed* looks like absent data,
not like broken code. Nothing crashed, so nothing was reported.

### Final solution

Keep the working-tree version of `scripts/list_envs.py`: match on
`spec.kwargs["env_cfg_entry_point"]` and `str()` the values that get printed.

### Verification

0 → 28 rows, matching the documented total of 7 platforms × Flat/WBC × normal/`-Play`.

### Lesson

Assert an expected count on any filter whose failure mode is an empty result.

---

## 2. A failing script exits 0 and prints no traceback

**Date:** 2026-09-21 · **Status:** resolved (behavior explained; documented as a rule)

### Background

Every script here ends with:

```python
try:
    main()
except Exception as e:
    raise e
finally:
    simulation_app.close()
```

This reads as if exceptions propagate. In practice a failing run can produce exit code 0 and no
output at all, which makes "the script did nothing" indistinguishable from "the script succeeded" —
and makes any verification based on exit status worthless.

### Experiment

A minimal script that starts `AppLauncher(headless=True)` and immediately raises
`RuntimeError("DELIBERATE_TEST_EXCEPTION_XYZ")` from `main()`, run with `python -u` and both streams
redirected to a file:

- exit code: **0**
- occurrences of the marker string: **0**
- occurrences of `Traceback`: **0**
- total output: 45 lines, all Isaac Sim startup noise

The marker was also absent from `~/.nvidia-omniverse/logs/omni.kit*.log`.

### Root cause

`SimulationApp.close()` — in
`site-packages/isaacsim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py` —
calls `self._app.shutdown_and_release_framework()`. That tears the process down from native code, so
the Python interpreter never unwinds the pending exception and never prints it; the C++ side exits 0.
The `raise e` in the `except` block is therefore decorative.

### Final solution / rule

Never use exit status or absence of error output as evidence. Every verification must assert on a
positive artifact — expected stdout, a written file, or a metric. Encoded in
[AGENTS.md](AGENTS.md) and [MEMORY.md](MEMORY.md).

### Correction of an earlier belief

An earlier session note claimed the traceback could be recovered from
`isaacsim/kit/logs/Kit/Isaac-Sim/5.1/kit_*.log`. That path **does not exist** on this machine: the
only Kit log directory is `~/.nvidia-omniverse/logs/Kit/Isaac-Sim/`, it contains only a `4.2` tree,
and its newest file predates today's runs. Isaac Sim 5.1 writes `~/.nvidia-omniverse/logs/omni.kit*.log`,
and the exception is not there either. Do not go looking for tracebacks in that path.

---

## 3. Repository trimmed to GO2-PIPER; upstream history dropped

**Date:** 2026-09-21 · **Status:** decision applied; intent not confirmed with the user

### Background

`.pre-commit-config.yaml` enforces `check-added-large-files --maxkb=5000`, while the seven robot USD
assets are 26–72 MB each (go2_piper at 32 MB is the smallest). They cannot pass the hook.

### What was done

`.gitignore` gained:

```gitignore
# 大型 USD 资产（只保留 go2_piper_base.usd 入库）
source/LeggedManip_Lab/LeggedManip_Lab/assets/*/configuration/*_base.usd
!source/LeggedManip_Lab/LeggedManip_Lab/assets/go2_piper/configuration/go2_piper_base.usd
```

The captured diff from the `logs/rsl_rl/go2_piper_wbc/2026-09-21_22-31-54/` run shows the six
`*_base.usd` files staged as deletions at the same time, so this and the training work happened in
one sitting.

### Trade-off accepted

The six ignored USDs remain **on disk**, so B1/B2/AGO/GO1 still run locally — they are only absent
from git. A fresh clone cannot run those six platforms.

### Consequence to be aware of

Local git history is now a single `first commit` (`7ca5f66`). The upstream commit that the
2026-09-21 training logs recorded (`8b817d7d23f0d23b7d70541a06862ce22b5fa9d7`) still exists as an
object but is reachable from no ref, so it will be garbage-collected and upstream
comparison/rebasing is no longer possible in this clone.

### Open

Whether the other six platforms should be dropped outright or restored (e.g. Git LFS) is undecided.
Tracked in [PROGRESS.md](PROGRESS.md) → Next steps.
