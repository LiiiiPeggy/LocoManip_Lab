# MEMORY.md — Durable Project Knowledge

Knowledge that stays true and should not have to be rediscovered. Verified only — no guesses, no
current tasks, no raw logs. If a fact here stops being true, fix or delete it.

Current state → [PROGRESS.md](PROGRESS.md) · background and experiments → [PROCESS.md](PROCESS.md)

## Runtime environment (verified 2026-09-21)

| what | value |
|---|---|
| conda env | `env_isaaclab5` (`/home/ubuntu/miniconda3/envs/env_isaaclab5`) |
| Isaac Lab | `0.54.4`, **editable** → `/home/ubuntu/locomani/IsaacLab5/source/isaaclab` |
| isaaclab_rl | `0.5.2`, editable → `/home/ubuntu/locomani/IsaacLab5/source/isaaclab_rl` |
| `rsl-rl-lib` | `5.0.1` (README requires ≥ 5.0.1) |
| Isaac Sim | `5.1.0.0` (pip-installed, provides `isaacsim.*`) |
| GPU | 1× RTX A6000, 48 GB |

- The editable install **pins the absolute path** `/home/ubuntu/locomani/IsaacLab5`. Move or delete
  that directory and `import isaaclab` breaks; the only fix is to clone Isaac Lab `main` back to that
  exact path and re-run `pip install -e`.
- The identically named conda env **`isaaclab` is a trap**: it ships `rsl-rl-lib 2.3.1`, which is
  incompatible with this repo. Using it surfaces as an RSL-RL config/attribute error, not as an
  import error, so it looks like a code bug rather than a wrong-env mistake.
- Isaac Sim prints a wall of pip dependency-conflict warnings (torchaudio / psutil / cryptography) on
  every launch. They are cosmetic — do not try to "fix" them.

## `pxr` is unavailable standalone; `import isaaclab` is fine

- `python -c "import pxr"` → `ModuleNotFoundError: No module named 'pxr'`. **Expected.** pxr is
  injected by the Isaac Sim runtime.
- `python -c "import isaaclab"` **succeeds** (prints `0.54.4`). A `pxr` error is not a broken install.
- Consequence: any module that imports USD/pxr at import time must be imported *after*
  `AppLauncher(...)` is constructed. This is exactly why every `scripts/*.py` here is split by the
  `# launch omniverse app` block. Keep that structure when adding scripts.

## Exit code 0 does not mean success

`SimulationApp.close()` calls `self._app.shutdown_and_release_framework()`, tearing the process down
from native code. A Python exception raised earlier in the script is never printed — no traceback, no
stderr, **and nothing in the Kit logs either** — and the process still exits 0. Reproduced
deliberately: a raised `RuntimeError` produced 45 lines of startup noise, zero lines of traceback,
exit 0. Experiment in [PROCESS.md](PROCESS.md) §2.

Always require a positive artifact (expected stdout, a written file, a metric). Never infer success
from the exit status, and never infer "no work to do" from empty output.

## Identify this extension by `env_cfg_entry_point`, never by `entry_point`

- In `gym.registry`, this repo registers every task with the generic
  `entry_point="isaaclab.envs:ManagerBasedRLEnv"` — a string that contains no extension name.
  Filtering on it silently matches **nothing**.
- `spec.entry_point` is not always a string: of 263 specs here, 244 are `str` and 19 are
  `function`. Comparing it without `str()` is a latent `TypeError`.
- The extension identity lives in `spec.kwargs["env_cfg_entry_point"]`, e.g.
  `…leggedmanip_lab.config.go2_piper.flat_env_cfg:Go2PiperFlatEnvCfg`.

**Expected row count for `python scripts/list_envs.py`: 28** (7 platforms × Flat/WBC ×
normal/`-Play`). Any other number means registration is broken. Assert the count — an empty table
looks like "no data", not "broken filter".

## Throughput scaling (RTX A6000 48 GB, GO2-PIPER-WBC)

`Perf/total_fps` covers collection + learning. Measured 2026-09-21:

| num_envs | fps | s / iteration |
|---------:|----:|--------------:|
|       64 |   2 136 |          0.72 |
|    1 024 |  23 188 |          1.06 |
|    4 096 |  71 372 |          1.38 |
|    8 192 | 119 888 |          1.64 |
|   16 384 | 162 057 |          2.43 |
|   32 768 | 194 147 |          4.05 |

One iteration = `num_envs × num_steps_per_env` = `num_envs × 24` env steps.

Scaling is strongly sub-linear past ~8 192 envs (4× the envs ≈ 1.7× the speed), so **4 096 is the
practical size for long runs**; more envs buy wall-clock, not throughput-to-convergence. 32 768 envs
does fit in 48 GB (≈17.6 GB peak) and completes — it is valid, just not proportionally faster.

## Large binary assets are deliberately untracked

`.pre-commit-config.yaml` runs `check-added-large-files --maxkb=5000`, while the robot USDs are
26–72 MB each. `.gitignore` therefore tracks only
`assets/go2_piper/configuration/go2_piper_base.usd` and ignores the other six robots' `*_base.usd`.

- Do **not** `git add` those files — the pre-commit hook rejects them.
- Do **not** delete them either: they are still on disk, so B1/B2/AGO/GO1 still run locally. A fresh
  clone cannot run those six platforms. Rationale and trade-off: [PROCESS.md](PROCESS.md) §3.

## The MuJoCo viewer steals the teleop keys (verified 2026-09-22)

MuJoCo's built-in viewer (the C++ `simulate` UI behind `mujoco.viewer.launch_passive`) binds plain
letters to visualization toggles. The deploy scripts read teleop keys with `pynput`, which is a
global **observer** — it does not consume the event — so a focused viewer window receives the same
keystrokes and flips rendering flags. Control is unaffected; the picture is not.

Read from the tables inside the installed `libmujoco.so.3.13.0` (`mjVISSTRING` / `mjRNDSTRING`), not
from docs. Overlaps that bite `go2_piper.py`: `W` wireframe (meshes go see-through and the collision
geometry inside them becomes visible), `S` shadow, `A` auto-connect, `D` static body, `Q` camera,
`E` equality, `R` reflection, `I` inertia, `J` joint, `L` additive, `K` skybox, `U` actuator,
`O` perturb object.

All 26 letters are bound, so letter keys cannot be remapped out of the conflict — only digits,
arrows and function keys are free. The cheap fix is focus, not code: keep the **terminal** focused
while teleoperating.

## Where the design is documented

- `PROJECT_INTRO.md` — the deep description: observation/action space, every reward term and weight,
  curriculum, domain randomization, PPO hyper-parameters. **Read before changing a reward.**
- `docs/ENV_DETAILS.md` — project structure and key-file map.
- `docs/WBC_MIXED_FRAME.md` — why the WBC end-effector command is XY in link0 but Z in world. Read
  before touching `UniformPoseWBCCommand`: the mixed frame is intentional, not a bug.
- `CONTRIBUTING.md` + `.pre-commit-config.yaml` — style rules and the enforced hooks.

## Common failure modes seen here

- **Silent empty output** from a script → a swallowed exception (§ exit code 0), or a filter
  predicate that legitimately matched nothing. Check for a missing artifact before debugging logic.
- **"It works when I run it by hand"** → the env is wrong (`isaaclab` instead of `env_isaaclab5`) or
  an import happens before `AppLauncher`.
- **A run that "did nothing"** → empty `hydra.log` is normal (Hydra's launcher log is unused here);
  the real record is the run's `events.out.tfevents.*` and `params/`.
