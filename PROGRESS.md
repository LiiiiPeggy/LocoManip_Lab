# PROGRESS.md — Current Project State

A snapshot, not a log. Delete what is no longer true; do not append history here (that is
PROCESS.md). Do not record unverified conclusions.

**Last verified: 2026-10-23** — every claim below was produced by re-running the command on this
machine, unless marked otherwise.

## Current objective

Two WBC policies on the wheeled **RangerBox-CR10** (AgileX Ranger 4WS/4WD + Dobot CR10 + AG95),
kept on separate branches so they never share a task id, experiment name or log directory:

- `rangercr10-world` — an end-effector target frozen in the *world*; the policy decides `(vx, wz)`
  itself. Stages 0-4 done and accepted (below); stages 5-6 (MuJoCo, real-robot alignment) not started.
- `rangercr10-teleop` — a *body-relative* end-effector target plus a human velocity command, the
  GO2-PIPER teleoperation shape. **Trained and accepted** (stages 4.1-4.4 below); the keyboard
  controller and MuJoCo deployment are not written yet.

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
5. **Continue `rangercr10-teleop`**: the keyboard controller (task 4.5) and MuJoCo deployment
   (4.6) are the remaining pieces of the current phase; see the entry below.
6. **`rangercr10-teleop`**: the plan is `docs/plans/teleop_plan.md`. Stages 4.1 (environment +
   six-check acceptance), 4.2 (training, 2048 envs x 1000 iterations in 105 min), 4.3 (three-mode
   acceptance) and 4.4 (decoupled-baseline control group) are done. Still open: 4.5 keyboard
   teleoperation (`WASDQE` + `IJKLUO123456`) and 4.6 MuJoCo deployment.
   Run `logs/rsl_rl/ranger_cr10_teleop/2026-10-12_20-14-23/`; EE position error 0.189 -> 0.081 m,
   orientation 0.649 -> 0.096 rad, linear velocity error 0.855 -> 0.074 m/s, no falls.
   Acceptance: chassis-only success 100% with the body-relative target unchanged to 0.0 while the
   chassis drove 0.6 m; arm-only leaves the chassis at 0.0038 m/s; both-together succeeds 78.9%.
   **The decoupled control group is a tie**, reported as such: with a body-relative target the
   chassis cannot help the arm's task, so there is nothing for the joint policy to trade -- see
   teleop_plan.md section 4.3.
   **Stage 4.5b (2026-10-14)**: the policy tracks commanded tool attitude almost not at all
   (a 0.2 rad yaw command produces 0.013 rad of motion and essentially no joint movement).
   IK on the URDF chain, with the simulator's joint limits and twelve restarts, shows every
   commanded attitude **is** reachable -- 34/34 single-axis targets, 6/8 box corners, most
   needing only 0.03-0.43 rad of joint travel -- and so is the tool-axis alternative, so the
   definition is not the problem. The observation carries the command bit-for-bit, commands
   resample at the configured 4-8 s, and reward and command share their frame transform.
   **Verdict: case B (reachable, verification correct) -- a learning problem, not geometry.**
   **Stage 4.5c (2026-10-16)**: the fault is the actor, and three of the four candidates are
   ruled out by measurement. An IK-oracle test shows the reward strictly prefers the IK
   solution (+0.19 to +0.79 per step), driving the IK solution through the ordinary action
   interface reaches 13/13 targets, and the policy *does* respond to attitude commands with
   the right joints -- but ~40 degrees off in direction and only 27-84% of the required
   magnitude. A 200-iteration resume improved the distribution-average error by 8.6% while
   moving the local response by <=2%, so "train longer" is not supported either. **No task
   configuration was changed** (`git diff --name-only` touches only docs/ and new scripts).
   See teleop_plan.md section 6. The earlier claim that the reward implied a 0.477 rad
   attitude error was my own cross-run mix-up -- the two error measures agree to 0.998.
   **Stage 4.5d (2026-10-22)** rebuilt the IK reference as a continuous local solve (FK
   residuals 1e-6, no branch jumps) and re-measured over six configurations: the actor's
   joint-space response has a **median 40.9 deg direction error with std 3.9 deg**, constant
   across command magnitude, axis and configuration, and a gain of 0.397 (0.80 for roll,
   0.38 for pitch/yaw). Task-space interventions settle that **aim, not gain, is the
   defect** -- scaling the response up makes the residual worse (12.6 -> 15.9 deg). Three
   explanations are rejected by measurement: arm_deviation (L1 and L2 optima differ by
   1.2 deg), the axis convention (actor is 40 deg from chassis-axis IK, 126 deg from
   tool-axis), and command distribution (test targets sit in the *easy* 5% tail). An offline
   supervised fit of the same map with the same inputs and architecture is **1.94x closer**
   than the PPO actor (0.129 vs 0.249 rad/joint), so the next change is **auxiliary supervised
   IK guidance**, not reward reweighting, not longer training, and not a wider network.
   **Stage 4.5e (2026-10-23)** ran that as three controlled 200-iteration runs. The IK teacher
   is excellent on held-out episodes (0.0046 rad joint MAE, 1.9 deg, 3 mm/0.31 deg task space)
   and its labels are continuous (0.0000 rad) and executable (6/6 through the action
   interface); supervised pre-training of the actor reaches 0.0054 rad / 0.5 deg offline. But
   **the improvement does not transfer to task space**: the residual in the FK intervention test
   is 12.9/9.3/9.1 deg for A, 12.9/10.6/9.1 for B and 14.2/10.5/8.7 for C, and the closed-loop
   attitude error under a 0.15 rad command is unchanged (0.1449/0.1428/0.1400). Joint-space gain
   moved 0.428 -> 0.773 (B) and direction 39.2 -> 30.4 deg, which is not enough because the
   residual is set by the direction. **B also regresses the chassis and position tasks** (mode A
   success 99.2% -> 61.3%, velocity error 3x) -- most likely because only the actor was
   pre-trained and the critic was left stale; **C does not regress** and lifts mode B success
   from 83% to 100%. Answer to the stage's question: IK supervision did **not** make the policy
   an accurate attitude tracker. Next: re-fit the critic with B, and move C's auxiliary term
   into task space. Two honest regressions are recorded there too: position accuracy
   was traded for orientation after iteration 300, and the arm-only success rate is the lowest of
   the three modes at 83%.
7. **New platform**: the port plan for the wheeled `rangerboxcr10lidar` (AgileX Ranger 4WS/4WD +
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
