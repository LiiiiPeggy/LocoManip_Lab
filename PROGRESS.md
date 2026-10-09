# PROGRESS.md — Current Project State

A snapshot, not a log. Delete what is no longer true; do not append history here (that is
PROCESS.md). Do not record unverified conclusions.

**Last verified: 2026-10-09** — every claim below was produced by re-running the command on this
machine, unless marked otherwise.

## Current objective

Two WBC policies on the wheeled **RangerBox-CR10** (AgileX Ranger 4WS/4WD + Dobot CR10 + AG95),
kept on separate branches so they never share a task id, experiment name or log directory:

- `rangercr10-world` — an end-effector target frozen in the *world*; the policy decides `(vx, wz)`
  itself. Stages 0-4 done and accepted (below); stages 5-6 (MuJoCo, real-robot alignment) not started.
- `rangercr10-teleop` — a *body-relative* end-effector target plus a human velocity command, the
  GO2-PIPER teleoperation shape. **Not created yet**; to be cut from the world branch.

GO2-PIPER remains the reference implementation and its policy is still staged for MuJoCo (below).

## Verified working

### Runtime environment

Confirmed by direct inspection on 2026-09-21, conda env `env_isaaclab5`:

| component | version | notes |
|---|---|---|
| `rsl-rl-lib` | 5.0.1 | README requires ≥ 5.0.1 |
| `isaaclab` | 0.54.4 | editable → `/home/ubuntu/locomani/IsaacLab5/source/isaaclab` |
| `isaaclab_rl` | 0.5.2 | editable → `/home/ubuntu/locomani/IsaacLab5/source/isaaclab_rl` |
| `isaacsim` | 5.1.0.0 | pip-installed |

- Isaac Sim starts headless and `import LeggedManip_Lab.tasks` executes cleanly → the extension is
  installed and registered.
- The conda env `isaaclab` has `rsl-rl-lib 2.3.1` → **incompatible**, do not use.

### Training runs end to end

`python scripts/rsl_rl/train.py --task GO2-PIPER-WBC --num_envs <N> --headless --max_iterations <M>`
completed on all six configurations. Each run wrote its final checkpoint (`model_{M-1}.pt`) beside
`model_0.pt`, plus 46 TensorBoard scalars — i.e. no run was silently swallowed (see AGENTS.md).

| num_envs | max_iterations | fps (last iter) | s / iteration |
|---------:|---------------:|----------------:|--------------:|
|       64 |              3 |           2 136 |          0.72 |
|    1 024 |             30 |          23 188 |          1.06 |
|    4 096 |             20 |          71 372 |          1.38 |
|    8 192 |             15 |         119 888 |          1.64 |
|   16 384 |             15 |         162 057 |          2.43 |
|   32 768 |             10 |         194 147 |          4.05 |

`s / iteration` = `num_envs × 24 / fps` (one iteration is 24 steps per env, collection + learning).
Hardware: 1× RTX A6000 (48 GB); peak training footprint ≈ 17.6 GB at 32 768 envs.

Evidence: `logs/rsl_rl/go2_piper_wbc/2026-09-21_*/` (tfevents, `model_*.pt`, `params/`, `git/`);
launcher dirs in `outputs/2026-09-21/*/.hydra/`. All six runs record the same seed (`42`) and
experiment name (`go2_piper_wbc`).

### `scripts/list_envs.py` filter fix — verified

The working-tree change replaces the `spec.entry_point` check with
`spec.kwargs["env_cfg_entry_point"]`. Measured over one live registry: old filter → **0** rows,
new filter → **28** rows at the time (7 platforms × Flat/WBC × normal/`-Play`); **32** now that
rangercr10 is registered (8 platforms).

**Status: committed on branch `rangercr10-world`.**

### First WBC policy trained, exported and staged for MuJoCo (2026-09-22)

`python scripts/rsl_rl/train.py --task GO2-PIPER-WBC --num_envs 8192 --headless --max_iterations 2500`
completed in **3 825.74 s (63.8 min)** = 8192 × 24 × 2500 = **4.92e8 env steps**, run directory
`logs/rsl_rl/go2_piper_wbc/2026-09-21_22-40-10/` (`model_{0,1000,2000,2499}.pt`, tfevents, `params/`).

| iteration | mean reward | episode length | ee position error | ee orientation error | base_contact |
|---|---|---|---|---|---|
| 0 | −1.39 | 22 | 0.477 | 0.830 | 0.031 |
| 250 | 80.81 | 996 | 0.485 | 0.128 | 0.017 |
| 1000 | 150.19 | 1000 | 0.101 | 0.136 | 0.005 |
| 2499 | 157.52 | 1000 | **0.065** | **0.095** | **0.0017** |

End-effector position error 0.477 → 0.065 m and orientation error 0.830 → 0.095: WBC tracking works
under the mixed-frame command. Episode length saturates at the 1000-step cap by iteration ~250 and
`base_contact` falls to 0.0017, so the robot is not falling. Learning flattens but does not stop —
`Mean action std` is still 0.45 at iteration 2 499, and the config default is 10 000 iterations, so
this run covers roughly a quarter of the schedule.

Artifacts (each verified by loading/playing it back, not by exit code):

- `logs/…/exported/policy.pt` and `policy.onnx`.
- `logs/…/videos/play/rl-video-step-0.mp4` — 50 envs, 500 steps, 1280×720 @50fps, 10 s
  (`play.py --task GO2-PIPER-WBC-Play --headless --video --video_length 500`).
- **`mujoco/deploy/policy/go2_piper/wbc/policy.pt` now holds this trained policy.** The shipped
  pretrained weights are kept next to it as `policy_pretrained.pt` (md5
  `63a9ca42e1aaf4d30d392f38c14a42e3`); roll back with
  `cp policy_pretrained.pt policy.pt`. Both files load with the same `(1, 210) → (1, 18)` signature,
  which is the observation the MuJoCo deploy script assembles.

## Open problems

1. The MuJoCo deployment path is still **untested end to end**. The policy file is in place and its
   I/O signature matches, but `go2_piper.py` has not yet been run against it.
2. `agx` (the real-robot ROS interface, a submodule) has an **uncommitted pointer move**
   (`28beddb` → `505ae3c`) in the working tree. Left alone deliberately: it is the user's own
   change, not this branch's. The training logs record the parent repo's diff, not the
   submodule's, so a run cannot be reproduced from its recorded git state alone.
3. `pre-commit` and `ruff` are not installed in `env_isaaclab5`, so the hooks in
   `.pre-commit-config.yaml` could not be run here; commits were checked by hand (trailing
   whitespace, final newline, the 5000 KB `check-added-large-files` limit) instead.

## Next steps

1. Run `python mujoco/deploy/deploy_mujoco/go2_piper/go2_piper.py config_wbc.yaml` to close the loop
   on the trained policy.
2. Decide whether to continue to the config's 10 000 iterations
   (`--resume --load_run 2026-09-21_22-40-10`, ≈1.5 h more at 8192 envs): reward and tracking error
   were still improving slowly at iteration 2 500.
3. Compare the trained policy against `policy_pretrained.pt` under the same `config_wbc.yaml`.
4. Decide the fate of the six gitignored `*_base.usd` assets (see PROCESS.md §3).
5. **Create `rangercr10-teleop`** from the world branch's current tip and build the body-relative
   teleoperation WBC there (plan.md section 7, and the task brief in the branch history). This is
   the immediate next piece of work, not the world branch again.
6. **New platform**: the port plan for the wheeled `rangerboxcr10lidar` (AgileX Ranger 4WS/4WD +
   Dobot CR10 + AG95) is at `docs/plans/plan.md`. **Stages 0-4 are done; stages 5-6 (MuJoCo, real-robot alignment) have not started.**
   The work lives on branch `rangercr10-world`; branch `rangercr10-teleop` (body-relative
   teleoperation WBC) is to be cut from it and does not exist yet. The two are meant to be kept
   apart by task id, experiment name, log directory and exported model.
   The scheme was **revised on 2026-09-23**: the base is no longer an external velocity command.
   The policy now outputs **8 dims — `[vx, wz, cr10_joint1..6]`** (`vx`, `wz` limited to ±0.5, no
   `vy`), so RL decides the chassis motion and the arm together given an end-effector pose target;
   the 8 chassis joints stay out of the action space and are resolved by a base controller in sim /
   by the AgileX driver on the real robot (`/cmd_vel`), which keeps the two semantics identical.
   Still locked: joint limits follow `rangercr10lidar.urdf`, end-effector frame is the gripper
   fingertip centre, one unified 10 Hz policy (200 Hz physics, `decimation = 20`, base held
   zero-order between policy steps). The speed-tracking rewards are dropped in favour of base
   velocity and base action-rate penalties.
   **R5 is resolved (2026-09-23)**: the end-effector target is kept in the **world frame** internally
   (`target_pose_w`, resampled as the current base pose plus a local offset and then frozen for that
   command's lifetime), the policy observes it **relative to `base_link`** (`target_pose_b`), and the
   tracking reward is computed **in the world frame** — a new `UniformPoseWorldCommand` used only by
   `ranger_cr10`, leaving GO2-PIPER's `UniformPoseWBCCommand` untouched. This is what makes "targets
   that require driving the base" exist at all.
   Remaining open items are the plan's §9 R1, R3 and R4 (TCP offset, success criteria, robot access).

   **Stage 1-3 complete (2026-09-23).** The platform now trains end to end. Stage 1
   (articulation + env cfgs + registration) and stage 2 (base controller) are done, and a
   first WBC policy was trained: 2048 envs x 1000 iterations in 85 min, run
   `logs/rsl_rl/ranger_cr10_wbc/2026-09-23_14-04-03/`, exported to `exported/policy.pt`
   and `policy.onnx`, with a video at `videos/play/rl-video-step-0.mp4`.

   Result: orientation error 1.06 -> 0.25 rad and mean reward -5.8 -> +32.4 over the run.
   At the near-target regime the policy reaches **0.076 m position / 0.180 rad orientation**
   error, and **freezing the base costs 60% more position error** (0.076 -> 0.122 m), so the
   policy is genuinely using the chassis rather than the arm alone.

   Honest limits: the command curriculum **stalled at 1.4 of its 3.0 m limit** -- the far
   targets that most need base motion are not being reached yet, so more training or a
   slower curriculum is needed. Yaw tracking also under-delivers in sim (about 40% of the
   commanded rate, from tyre scrub at large steer angles), which matters for sim-to-real.

   Stages 1-3 turned up nine defects that had to be fixed before training was meaningful;
   they are documented in the plan's stage-1 and stage-2 notes and in the commit history.
   Seven were measured, not inferred (friction/rolling resistance, per-wheel sign
   convention, mimic constraints dropped by PhysX, reverse kinematics, steering sign,
   branch-dependent steering clamp, and an unreachable orientation command that made the
   reward mostly a constant penalty).

   **Stage 4 (stability and curriculum optimisation) complete (2026-10-09).** The task and
   experiment were renamed for the branch split -- `RANGER-CR10-WORLD-WBC` /
   `ranger_cr10_world` -- and the 2026-09-23 baseline stays under the old name, unreplayable
   by design (its exported policy is 99-dim; the observation is now 108-dim).

   Six defects, each pinned to a measurement, are in plan.md section 7 stage 4. The two that
   mattered most: the base action was **hard-clamped** while the policy's raw output sat at
   4x the limit, where `clamp` has zero derivative and the channel was therefore dead-locked
   with no way back; and the position reward's `exp(-d^2 / 0.1)` kernel is dead past ~0.75 m,
   while the mean distance to target was 1.06 m -- so far targets paid the same (nothing)
   whether the policy drove to them or not.

   Acceptance, three policies under identical frozen-target tests (`scripts/ranger_cr10/eval_world_wbc.py`):

   | | baseline | optimised |
   |---|---|---|
   | position error, targets mean 1.99 m away | 0.0999 m | **0.0801 m** |
   | far stratum (>1.15 m, 449/512) success | 94.95% | **95.99%** |
   | envs within 5 cm | 122/512 | **306/512** |
   | chassis linear rate once on target | 0.128 m/s | **0.0053 m/s** |
   | chassis yaw rate once on target | 0.270 rad/s | **0.0158 rad/s** |
   | position error with `(vx, wz)` forced to zero | 1.0493 m | 1.0578 m |

   Freezing the base costs **13x** the position error, so the chassis is doing real work, not
   decorating an arm that could reach anyway.

   Honest limits: orientation ended **12% worse** (0.283 vs 0.252 rad) in both optimised runs,
   most likely the linear distance term trading attitude for position -- recorded as a
   deliberate trade, not hidden. The curriculum reached 1.35 of its 3.00 limit; it is still
   advancing at the end (success rate 0.73 against a 0.60 promotion threshold) but the claim
   supported by evidence is "96% success on targets well beyond the training box", not "the
   curriculum is finished".


   **Stage 0 (URDF→USD) complete**: `assets/ranger_cr10/` holds the conversion-only URDF, config,
   driver script and the USD (22 MB, kept as a regular Git object like the source meshes). Verified
   by a printed joint table: 22 joints, correct parenting, arm limits equal to the URDF's.
   Two things differ from go2_piper and are easy to get wrong — see the plan's stage-0 notes:
   `convert_mimic_joints_to_normal_joints` must be **true** here (the AG95 has 7 mimic joints, and
   `false` leaves them as free hinges), and the USD's limits are in **degrees** on both platforms.
   The source meshes were decimated first: 83 MB / 1.73 M faces → 19 MB / 0.38 M faces.

## Known minor documentation drift (verify before editing docs)

Recorded because it misleads edits; neither item changes runtime behavior.

- `PROJECT_INTRO.md` §6 says "所有 8 个机器人"; `README.md` and `PROJECT_INTRO.md` §3 both say
  7 platforms.
- `PROJECT_INTRO.md` §5.5 lists a reward term `arm_torques_max` (−5e-5), but the completed runs log
  `Episode_Reward/arm_deviation` and no `arm_torques_max` tag exists in any run.
