# AGENTS.md — Agent Working Rules

Rules for every AI agent working in this repository (Claude Code, Cursor, Codex, Codex Desktop, …).
They are deliberately shared: this is the only agent-facing rules file, and the project memory it
points at is shared too.

**Do not create tool-specific memory files** (`CLAUDE.md`, `.cursorrules`, `.github/copilot-instructions.md`,
per-agent notes, …). The four files below are the single source of truth:

| file | answers | read it |
|---|---|---|
| [AGENTS.md](AGENTS.md) | how to work here | always |
| [PROGRESS.md](PROGRESS.md) | where are we now | at the start of every task |
| [MEMORY.md](MEMORY.md) | what stays true | when the task touches env, configs, or tooling |
| [PROCESS.md](PROCESS.md) | why it is this way | when changing or questioning an existing solution |

## Project Memory

Before substantial work:
- Read PROGRESS.md.
- Read relevant MEMORY.md.
- Read PROCESS.md when historical reasoning is needed.
- Trust current code, tests, and runtime evidence over outdated documents.

After substantial work:
- Update PROGRESS.md when state changes.
- Update MEMORY.md when durable knowledge is discovered.
- Update PROCESS.md when important engineering history should be preserved.

修改已验证方案前，先检查 MEMORY.md、PROCESS.md 和验证结果。不要仅因为新方案更“优雅”而替换已有有效方案。

## How to tell whether a run actually succeeded

Isaac Lab scripts here end with `finally: simulation_app.close()`, and `close()` terminates the
process from native code. A Python exception raised earlier is therefore **never printed**, and the
process still exits **0**. Measured, not assumed — see PROCESS.md §2.

- Never treat exit code 0, or "no error output", as evidence that anything worked.
- Require a positive artifact: expected stdout, a written file (checkpoint, exported policy,
  run directory), or a metric.
- When a script is silent and exits 0, assume a swallowed exception until proven otherwise.

## Environment

- Use the conda env **`env_isaaclab5`**. The similarly named `isaaclab` env is incompatible — see
  MEMORY.md.
- Launch through `from isaaclab.app import AppLauncher` and import USD-dependent modules *after*
  the launcher is constructed. Standalone `import pxr` is expected to fail.
- Details, versions and gotchas: MEMORY.md → "Runtime environment".

## Development conventions

Inherited from [CONTRIBUTING.md](CONTRIBUTING.md) and `.pre-commit-config.yaml`; follow them.
- `ruff check` + `ruff format`, line length 120, Google-style docstrings.
- Comments and newly written text in **English** (repo convention).
- New `.py` / `.yaml` files need the Apache header from `.github/LICENSE_HEADER.txt`
  (the `insert-license` hook inserts it). Run `pre-commit run --all-files` before proposing a commit.
- Files above **5000 KB** are rejected by `check-added-large-files`. Never `git add` large USD assets
  or `logs/` checkpoints — see MEMORY.md → "Large binary assets are deliberately untracked".

## Modification principles

- Do not rewrite verified, working code for style. Reward terms, env configs and PPO configs are
  tuned artifacts; "cleaner" is not a reason to change them.
- Change one thing at a time, then re-run the identical command and compare numbers against the
  previous run.
- Keep `scripts/*.py` runnable standalone and headless.
- Touch `mujoco/`, `docs/`, assets and reward configs only when the task calls for it.
- Never delete or overwrite anything under `logs/` or `outputs/`: those runs are the evidence base.
- Comment *why*, not *what*. Do not add narration comments to untouched code.

## Testing requirements

- There is no unit-test suite (`tests/` is gitignored). Verification means running the thing.
- Smoke-test before any long run: `--num_envs 64 --max_iterations 3 --headless`.
- Registration/env changes must be verified by listing environments and reading the **row count**
  (expect **28**), not by "it imports".
- Report the exact command and its observed output. "Should work" is not a test result.
- If a run's output is empty, say so explicitly rather than reporting success.

## Documentation maintenance

- Paired files must be updated together: `README.md`/`README_CN.md`,
  `docs/ENV_DETAILS.md`/`docs/ENV_DETAILS_CN.md`,
  `docs/WBC_MIXED_FRAME.md`/`docs/WBC_MIXED_FRAME_CN.md`.
- `PROJECT_INTRO.md` is the deep design description (observations, reward terms, curriculum, PPO
  hyper-parameters). Update it when a reward term or hyper-parameter actually changes.
- Status, TODOs, blockers and logs go in PROGRESS.md — never in README or docs.
- Keep each memory file to its own job; cross-reference with a link instead of copying text.
