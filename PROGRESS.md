# PROGRESS.md — Current Project State

A snapshot, not a log. Delete what is no longer true; do not append history here (that is
PROCESS.md). Do not record unverified conclusions.

**Last verified: 2026-10-25** — every claim below was produced by re-running the command on this
machine, unless marked otherwise.

## Current objective

Two WBC policies on the wheeled **RangerBox-CR10** (AgileX Ranger 4WS/4WD + Dobot CR10 + AG95),
kept on separate branches so they never share a task id, experiment name or log directory:

- `rangercr10-world` — an end-effector target frozen in the *world*; the policy decides `(vx, wz)`
  itself. Stages 0-4 done and accepted (below); stages 5-6 (MuJoCo, real-robot alignment) not started.
- `rangercr10-teleop` — a *body-relative* end-effector target plus a human velocity command, the
  GO2-PIPER teleoperation shape. **Trained, accepted and now deployed in MuJoCo** (stages 4.1-4.6
  below); 4.6 (sim-to-sim) runs the 4.5j policy with the 4.5k filter, keyboard included.

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
5. **Continue `rangercr10-teleop`**: 4.6 MuJoCo deployment is done (below). The open items are
   the policy levers named in 4.5j -- reward-side (orientation weight / position kernel width),
   a wider `EE_OFFSET_RANGES` with retraining -- and the real-robot alignment stage after them.
6. **`rangercr10-teleop`**: the plan is `docs/plans/teleop_plan.md`. Stages 4.1 (environment +
   six-check acceptance), 4.2 (training, 2048 envs x 1000 iterations in 105 min), 4.3 (three-mode
   acceptance) and 4.4 (decoupled-baseline control group) are done; 4.5 keyboard teleoperation
   (`WASDQE` + `IJKLUO123456`) is done and verified (see the bug below). 4.6 MuJoCo deployment is
   done (entry below).
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
   **Keyboard + demo (2026-10-25)**: `KeyboardTeleopController` had a real bug -- its attitude
   limits were written as magnitudes `((22.9, 22.9), ...)` but read as `(low, high)`, so `lo == hi`
   and the *first* press of any attitude key snapped to +22.9 deg, minus keys included. The
   position limits were always signed, which is why the original 22/22 check passed: it tested the
   position clamp and never the attitude step size or the negative side. Fixed to signed pairs and
   covered by six new checks (33/33 now). Only the keyboard path was affected -- the evaluation
   and diagnostic scripts write command tensors directly, so stages 4.5b-4.5e are untouched, and
   MuJoCo (4.6) has never run. `scripts/ranger_cr10/demo_teleop.py` records a scripted demo
   through the real keyboard path with a following camera and a command/error HUD; on the main
   policy a 15 deg pitch command still leaves 11.2 deg of residual (roll 12.4, yaw 14.9 at 20 deg)
   while position and chassis tracking are fine, which is section 8's conclusion seen end to end.
   **Stage 4.5f (2026-10-25)** asked why an actor that fits the IK map to 0.0054 rad offline does
   not track attitude closed-loop, and answered it without any new training.
   **B0 -- the supervised actor that never saw a PPO gradient -- is not better than A, B1 or C**
   (6.3/6.4/6.3/6.5 deg steady state; 9-10 deg on +-0.2 rad commands), and the *distilled teacher
   itself* is just as bad (6.1 deg) while online local IK reaches 0.7 deg. Handing control over
   from a converged state (0.4 deg) makes B0 and the teacher walk to 10-12 deg within 0.5 s; IK
   stays. So the failure is neither PPO nor the executor: **the learned map's fixed point is not
   the commanded attitude**. Mechanism: the map's output moves only 16% as much as the observed
   joint state does (feedforward table, almost no feedback), so nothing pulls the loop back; the
   online states are *not* out of distribution (nearest-neighbour 5.60 vs 6.60 for held-out test
   samples), so "data drift" is ruled out. PPO's erosion is gradual (0.0054 -> 0.0057 after one
   iteration, 0.0282 after 200), and the old critic/Adam state carried into B is a measurable
   mismatch (first `Loss/value` 15.6 vs 2.85 for C) but not the cause. A task-space FK auxiliary
   loss has nothing to win: joint MSE already leaves 0.3 deg / 3 mm of task-space error. The
   supervision set's real gap is elsewhere: 35.8% of candidates fail the local IK walk, and they
   are systematically the large-attitude, near-limit targets (retention 98.4% at 0-5 deg vs 40.8%
   at 30-60 deg), half of which a global solver does find. Next stage (4.5g) is data, not PPO:
   recover that corner, add IK-driver (DAgger) rollouts that visit the states the loop actually
   settles in, and accept on the handover test rather than on offline MAE.
   **Stage 4.5g (2026-10-26)** went after the fix and found the evaluation was the reason the
   paradox had survived. The actor's *first* handed-over step is already wrong: at one state and
   one target, online IK commands a pose whose FK is 0.0 deg off while the actor's is 9.2-12.1,
   and `FK(q_des)` predicts the settled error to within 1.4 deg -- so it is neither execution nor
   smoothing, and there is no bug in the labels, the observation (bit-for-bit command match), the
   history (refreshing it changes nothing) or the IK branch.
   The offline numbers that contradicted this were measured on **the same commands as training**:
   every episode is started with `env.reset(seed=...)`, so the command stream repeats and the
   held-out commands sit 0.00 deg (median) from the training ones. Changing only the command at a
   real state leaves the actor's target attitude wrong by ~100% of the change (3.1/6.1/12.2 deg
   for 0.05/0.10/0.20 rad), i.e. the command channel has ~3% gain.
   Six interventions were run and **none moves the closed loop**: IK-driven near-target data
   (10.0 deg), DAgger on the actor's own states, decorrelated postures, 2766 samples at ~2700
   *new* commands (10.0), near-solution postures at new commands (10.1), and a residual
   parameterisation whose target is zero at the solution (9.3). All of them improved their own
   offline metric first. Root cause of *why* the command is ignored at the states that matter is
   **not established**; data-volume, state coverage, command coverage, parameterisation,
   initialisation, observation completeness, the executor and IK multi-solution are all excluded.
   Recommendation: fix the evaluation (split by command), and stop trying to learn the command
   map from data -- put an analytic IK layer behind the actor or change the formulation.
   **Stage 4.5h (2026-10-27)** found the mechanism. A command-disjoint split of the supervision
   (20% of the 312 command groups held out entirely) shows the actor is *not* merely memorising:
   on unseen commands it is as accurate as on seen ones at supervision states (0.2 deg). The
   failure is a shortcut: in every rollout the arm is already where its command put it, so the
   joint state alone predicts the label and the command channel is never used. Evidence: the same
   architecture trained from scratch on **112 samples** (16 commands x 8 *randomised* postures)
   reaches **1.0 deg** on new commands at exactly-on-target states, while the real actor is
   **18.0 deg** there -- 112 decorrelated samples beat 3142 correlated ones. Encodings are not
   the issue (canonical quaternion 1.3 deg, rotation vector 1.5 deg).
   Training on decorrelated samples does open the channel -- the first handover step improves from
   9.4-12.1 deg to 1.7-6.3 deg (from scratch) or 2.5-10.0 deg (from B0) -- but both variants lose
   real-state coverage and settle worse (28-47 deg, and 5.3-23.8 deg with the original test set
   degrading to 0.10 rad). 4.5g's blend diluted the decorrelated data too far to matter (6.4->5.8
   deg). The working ratio is between those two and has not been found yet; no closed-loop fix
   was delivered this stage, but the mechanism and the direction are now established.
   **Stage 4.5i (2026-10-29) fixed it, and the root cause was upstream of the policy.** The
   command sampler `UniformTCPBodyPoseCommand._resample_command` drew all three position offsets
   (and all three Euler angles) from one tensor through `Tensor.uniform_`, which fills in place
   and returns the same tensor: every command the task ever trained on satisfied
   `offset_x == offset_y == offset_z` and `roll == pitch == yaw` -- a 2-D line in the 6-D box
   (verified on all 3142 old commands, max deviation 1e-7). The keyboard emits exactly the
   off-line commands, which is why "I" moved the tool ~4 mm instead of 90 mm. Fixed with
   per-axis sampling; `check_teleop_env.py` section D2 now asserts independence (4096 commands,
   |corr| <= 0.019, 0% duplicates). Supervision was re-collected with the fixed sampler
   (`logs/ik_teacher_box`: 4105 samples, 414 commands, labels FK-verified to 0.81 mm / 0.011 deg),
   then trained as a mixture of real rollouts and decorrelated postures at 2:1 and 1:1
   (command-disjoint 248/82/84 split; 20000 optimiser updates each, identical batch/lr/val/early-stop).
   Result: unseen-command FK attitude 20.2 deg -> **0.69 / 0.61 deg** (validation),
   **0.73 / 0.67 deg** (test), 0.75 / 0.66 deg on fresh-command on-target probes, command
   sensitivity (droll 0.1/0.2) 21.3/23.4 -> **0.4/0.7 deg**. Closed loop (18 commands, 8 of them
   never supervised): C1 first step 14.3 (B0) / 9.8 (b0dec) -> **0.5 / 0.4 deg**, 2.5 s
   12.5 (B0) / 16.3 (b0dec) -> **0.5-0.7 deg**, position 76-115 -> **10 mm**, jitter 0.21 -> 0.02 deg --
   equal to the online-IK reference (0.6 deg / 9 mm). Three-mode regression: mode A EE position
   0.0919 -> **0.0091 m**, mode C 0.0814 -> **0.0238 m**, attitude residual at 0.2 rad
   0.204 -> **0.012 rad**. **Cost: the chassis got worse** (mode A linear-velocity error
   0.0114 -> 0.0550, mode B chassis creep 0.0019 -> 0.039 m/s), because supervised training only
   anchors `[vx, wz]` to B0 on real states. Neither run reached a plateau within the 20000-update
   budget (last-10000 gain 18-20%), so "converged" is not claimed. Next: the PPO fine-tune in
   plan 13.6 (small arm exploration std, IK supervision mixed into the PPO loss, closed-loop
   regression every 250 iterations); MuJoCo (4.6) stays blocked until that restores chassis
   tracking.
   **Stage 4.5j (2026-10-30) ran standard PPO on top of it for 6000 iterations.** The actor was
   initialised from `actor_mix11_best.pt`, the critic re-initialised, the optimizer state emptied,
   and the action std set to the per-dimension values standard PPO itself had converged to
   (0.079-0.157) rather than the config default 1.0 (0.25 rad of joint noise, 20x the arm map's
   precision). No IK teacher, no auxiliary loss, no new module: the default agent config only.
   Training was healthy -- reward 17 -> 187, value loss 59 -> 0.36, success 0.86, 9.26 h at
   2048 envs -- and by iteration 4000 it had matched the original 1000-iteration run's final
   training metrics. Closed loop (10 commands, same protocol as 4.5i): chassis yaw tracking
   **54% -> 97%** (wz 0.081 -> 0.146 of a 0.15 command) and steady-state chassis drift
   **0.0392 -> 0.0015 m/s**, while EE attitude went 0.55 -> **1.20 deg** and position 10 -> **24 mm**.
   Three modes at 64 envs put `model_5999` between mix11 and `model_1198` on every metric and
   clearly on the right side of each: mode B' attitude 0.0268 rad (mix11 0.0157, 1198 0.1443),
   mode C position 0.0476 m (0.0236 / 0.0814), mode A EE position 0.0365 m (0.0091 / 0.0919),
   success 100% in A/B/C.
   The key measurement is *when* the precision was lost: **one PPO update** moved the
   deterministic EE error from 0.59 deg to 8.02 deg (action means shifted 0.52-0.94 units,
   KL(init||new) median 123 against a target of 0.01), and 6000 iterations only clawed it back
   to 3.44 deg -- the adaptive controller dropped the learning rate to its 1e-5 floor inside
   that first iteration, after the damage. Offline one-step error and closed-loop steady-state
   error are not the same quantity: the 3.44 deg probe is a single-step target error, the
   closed loop rejects it down to 1.2 deg.
   Verdict: **situation B of the 4.5j brief** -- chassis recovered, EE accuracy partly lost.
   The single-factor follow-up was then **executed** (plan 14.9): arm exploration std
   0.12 -> 0.03, initial lr 5e-4 -> 1e-4, 1500 iterations, everything else -- algorithm,
   network, reward, observations, action interface -- unchanged. It held the **one-step** map
   (2.26 deg vs the main run's 3.44 deg, position 25 vs 51 mm) and the best training success
   (0.866), but its **closed loop was worse** (2.1 deg / 36 mm against 1.2 deg / 24 mm) and its
   chassis yaw recovered to only 55% (main run 97%). So neither direction reaches the target
   set at once, and `model_5999` stays the best 8-D policy. MuJoCo (4.6) then ran on that
   checkpoint (entry below); the next lever remains the reward (orientation weight, position
   kernel width), not more training or a new module.
   **Stage 4.5k (2026-10-30) measured the static drift and added an optional deadband.** On the
   4.5j policy, with the external chassis command exactly zero and the arm moving for 30 s, the
   actor still commands `|vx| = 0.0117 m/s` and `|wz| = 0.0112 rad/s`; the base controller obeys
   it and the robot accumulates 1.4-2.1 mm of drift (two independent runs). So the source is the
   actor's output, not the operator command and not the low-level controller. A conditional
   zero-lock (`base_command_filter.py`: when a channel's effective command is zero, that
   channel's action is zeroed; the six arm channels are untouched) cuts the 30 s drift to
   0.43-0.87 mm (four repeats; unfiltered 1.41-2.50 mm, so about 1/3 and lower in every paired
   run) and the static yaw rate from 1.7-2.1 to **0.6-0.7 mrad/s**, with no measurable change to
   EE accuracy (40.2 mm / 1.71 deg either way) or to low/normal/zero-switch tracking. The input
   deadband alone is a no-op when the command is exactly zero -- it only matters for a stick
   that does not return fully to zero (0.01 m/s residual). Threshold 0.02 m/s / 0.02 rad/s:
   0.05 demonstrably eats a legitimate 0.05 m/s command (actual speed 0.0018 instead of 0.05),
   0.01 misses the 0.01 residual. Caveat for MuJoCo: the lock hands position holding to tyre
   friction, so it needs re-checking on slopes and low friction.
   **Stage 4.6 (2026-10-31) deployed the 8-D policy in MuJoCo and the two simulators agree.**
   There was no MJCF for this robot, so `mujoco/robots/ranger_cr10/build_mjcf.py` generates one
   from the same converted URDF Isaac uses (every mass, inertia and joint transform parsed, not
   retyped); actuator gains, armature and the contact groups come from the articulation config.
   `scripts/ranger_cr10/check_mujoco_interface.py` then compares the deployment against the *live*
   Isaac env: 117-D observation assembly `<=3.7e-10`, MuJoCo's own sampling of the same state
   `<=2.5e-8`, TCP forward kinematics **0.002 mm / 0.0001 deg**, 256 random actions through
   `BaseVelocityAction` vs the ported `ChassisAction` `<=6.1e-7`, deterministic actor identical.
   Closed loop, MuJoCo vs Isaac on the same fixed commands: C2 median 2.5 s attitude
   **1.40 deg vs 1.37 deg**, position **25.5 vs 25.9 mm**; C2b (vx 0.25 / wz 0.15 / roll +0.2)
   vx 0.284 vs 0.280, EE 30 mm / 0.79 deg vs 31 mm / 0.72 deg; 15 s driving at 0.05 / 0.25 m/s
   matches in speed (0.0619 vs 0.0622, 0.2873 vs 0.2891) and in distance travelled. So EE accuracy
   does **not** degrade in MuJoCo, and the three modes work (arm-only leaves the chassis at
   `<=0.0002 m/s`). The 4.5k filter is even more valuable here: the 30 s arm-moving static drift
   falls 54.6 -> **0.5 mm** and the driven path 448 -> 3.9 mm, with identical low/normal/switch
   tracking and identical EE error. Two real defects were found and fixed on the way: MuJoCo
   writes a free joint's velocity in the **world** frame (the observation needs the body frame),
   and the analytic cylinder wheels had **no rolling resistance** (added 4.0 Nm, derived from
   Isaac's measured droop: low-speed speed becomes 0.0619 vs 0.0622, static drift 100 -> 55 mm;
   `--wheel_frictionloss 0` reproduces the rigid-wheel model). **One correction to 4.5k**: its
   drift numbers averaged `base_x/base_y` over **16 envs** whose `base_com` randomisation differs,
   which cancelled the drift -- single-env the same script gives 13.6 mm, not 2.5 mm. The
   conclusion (the lock removes ~2/3 or more) stands; the absolute figures were low by ~5x, and
   drift must be reported per-environment from now on. The 4.5k slope/low-friction caveat was then
   settled in MuJoCo (`teleop_eval.py --suite terrain`): on flat ground the lock holds 0.4-0.5 mm
   at wheel-ground mu 0.35, 0.20 **and** 0.10 (friction is not involved when nothing pushes
   sideways); on a 3 deg / 5 deg slope it creeps 37 / 66 mm per 30 s against 248-1683 / 303-5196
   unfiltered, and it is never worse than no filter in any cell -- so no "disable the lock on a
   slope" fallback is warranted, only the statement that its guarantee is flat-ground. Known residual differences: arm-driven
   chassis creep 54.6 vs 13.6 mm, ride height 0.4125 vs 0.4071 m (a cylinder does not squash), and
   no actuator delay (Isaac has 0-10 ms). Full detail: `docs/plans/teleop_plan.md` section 16.
   **Model completeness (2026-10-31):** the MJCF now matches the URDF->MuJoCo conversion already
   in `~/locomani/UniLab` link for link (identical to 0.000 mm), the AG95 linkage and both
   RealSense bodies are back (they were dropped with their parent frame), and the collision
   proxies moved to geom group 3 -- they had been *drawn* over the meshes, which is why the robot
   looked like a grey box. The wrist d455 and the front d435 were drawn on their side because the
   generator ignored each link's `<visual><origin>` (rpy = (90, 0, 90) for both cameras; every other
   link happens to be identity) -- now honoured, and checked against the URDF chain rather than by
   eye: the d455 optical +z is 0.0 deg from the tool axis. **Viewer key conflict solved:** pressing W toggled the viewer's
   wireframe because `pynput` only observes keys; `ViewerFocusGuard` hands the X input focus back
   to the launching window (and re-checks 4x/s), with `ViewerKeyPin` as a backstop. Measured on
   the picture itself: 38.0% of pixels changed without the guard, 0.0% with it.

   **Stage 4.7 (2026-11-01) expanded the end-effector workspace and rebuilt the keyboard limits
   around measurement.** Task A first: `check_workspace_reach.py` sweeps the six axes by moving
   1 cm / 2 deg at a time from the default pose (a multi-restart IK answers "does any
   configuration reach this" and is reported alongside, because the two disagree badly -- `Y+0.20`
   only exists in a branch with joint1 at -3.4 rad). Measured: X -0.05..+0.45, Y -0.45..+0.07,
   Z -0.26..+0.45 at the default attitude; roll -18..+70, pitch **-14**..+70, yaw +-70 deg. So
   the old +X weakness was never a reach problem (the arm goes 0.45 m forward while the old
   policy was 39 mm off at 0.05), while part of the old box (X-0.12 at the default attitude) was
   already outside the continuously reachable set; pitch -45 deg is genuinely unavailable, so the
   target box ships with pitch capped at -30 deg. 66-68% of the candidate box is reachable by
   multi-restart IK, 0% of reachable targets put the arm into the ground.
   The training box now lives in `scripts/ranger_cr10/workspace_limits.py` -- one file read by
   both the Isaac task and the MuJoCo keyboard -- and grows in four stages promoted by the
   command's own success rate (threshold 0.85, at least 2000 steps per stage, 30% of samples
   replayed from the old box). 6000 PPO iterations ran in three segments (2000 + 1000 + 3000);
   the gate had to be fixed twice on evidence (a single-sample comparison on a statistic that
   fluctuates +-0.04, then reading `success_rate`, which is a running average since the last
   resample and sits 0.10 below the instantaneous fraction). **The curriculum still stopped at
   stage 1**: success EMA plateaued at 0.84 against the 0.85 needed, so stages 2-3 never ran.
   Result on a fixed 60-target set from the *new* box: median position error 142 -> **61 mm**,
   orientation 10.1 -> **4.0 deg**, 80 mm/10 deg coverage 8.3 -> **66.7%** (30 mm/2 deg 0 -> 8.3%).
   On the *old* commands nothing degraded: 2.5 s convergence 0.79 deg / 23.4 mm (was 1.37 / 25.9),
   three-mode EE attitude improved across the board, chassis linear/yaw error 0.159/0.145 against
   4.5j's 0.154/0.138, static drift with the lock still 0.5 mm. The keyboard box is now the
   *measured* usable range rather than the training box (X -0.10..+0.25, Y -0.35..+0.10,
   Z -0.20..+0.25, roll -30..+40 deg, pitch -20..+40 deg, yaw +-60 deg), it is asymmetric because
   the arm is, and the status line names the axis that hit its clamp.

   **Stage 4.8 (2026-11-02) expanded the workspace again and fixed what stopped 4.7.** Three
   changes, each pinned to a measurement. (a) *The curriculum gate.* 4.7 stalled because success
   was averaged over targets the arm can never hold, so its ceiling was the reachable share;
   the gate now counts only targets inside a kinematic reachability table
   (`scripts/ranger_cr10/reach_table_48.json`, 5x5x5 attitude cells, checked in), promotes at
   0.80 on three consecutive 2000-step windows, prints the *unfiltered* rate next to the filtered
   one on every window, and has a logged forced-promotion escape hatch. The table's error was
   measured in both directions: 3.0% "says reachable, IK cannot solve" (dangerous) against 25.5%
   "says unreachable, IK solves" (conservative). (b) *The reward.* `4.5*exp(-d^2/0.1)` has its
   steepest slope at 224 mm and only 1.8 reward/m at 20 mm; a second narrow term
   `0.5*exp(-d^2/0.01)` was added and nothing else touched (gradient x2.07 at 20 mm, x1.00 at
   300 mm). (c) *Reachability was redefined.* The 4.7 sweep walked from the default posture and
   therefore reported `roll=-45` / `pitch=-30` as unreachable; the policy demonstrably holds roll
   -30, so the table seeds each attitude cell with any solvable configuration and only then walks
   position. Training: 6000 PPO iterations from 4.7's `model_2999` (iter rebased, actor
   bit-identical), 8.9 h; **the curriculum reached the final stage at iteration 500, both times by
   success (EMA 0.950 then 0.939, never forced)**. Results on a fixed 60-target set from the 4.8
   box: median position error 95 -> **24 mm**, orientation 6.7 -> **2.7 deg**, 50 mm/5 deg
   coverage 18 -> **77%** (reachable targets 85%); on the new keyboard box 16 mm / 1.6 deg and
   95% (100% of reachable). The keyboard box is now X -0.05..+0.40, Y -0.50..+0.10,
   Z -0.25..+0.40, roll -30..+60 deg, pitch -20..+60 deg, yaw +-80 deg, with three switchable
   profiles (4.7 / 4.8 / 4.8_explore) and startup printing the active limits. **Honest regression:**
   the old box's six-dimensional targets improved (36 -> 14 mm, 2.5 -> 1.6 deg), but the 14 fixed
   convergence commands' median orientation went 0.81 -> 1.28 deg, concentrated in negative pitch
   (B5: 0.37 -> 3.33 deg); the mechanism was not isolated by ablation and is the next thing to test.

   **Stage 4.10 (2026-11-03) points the tool forward and puts the camera on top of the gripper.**
   The default pose now has the tool (TCP +z) along base +X, the D455 mounted **on top of the
   AG95** rather than on its side, and the whole arm extended forward so the receiver sits at
   the front of the chassis. Three things had to be fixed on the way: (a) the tool's default
   orientation was pointing *down*, not forward, so "point forward" is a 77 deg change and not
   `rpy = 0`; (b) with the tool horizontal the arm can only hold the pose extended (x = 0.80 m,
   no solution at the old 0.55), which moves the command box forward -- trimmed to
   X -0.15..+0.40, Y -0.40..+0.10; (c) rolling `cr10_joint6` by -142 deg put the camera above
   but tilted the image and left 0.66 rad of joint travel, so the mount itself was changed in
   the URDF (`d455_joint` origin (0, 0.10, 0.14) rpy (45, -90, -135) deg), the USD regenerated
   from it, and **q6 returned to 0**. Final pose `[0.236, 0.086, -2.398, -0.830, -1.335, 0]`:
   D455 centre **0.880 m** above ground (Isaac 0.872 settled), optical axis **0.03 deg** from
   base +X (Isaac 0.87), image-up **0.02 deg** from world +Z (Isaac 0.91), camera 100 mm
   straight above the TCP, TCP 0.780 m, joint slack 0.46 rad, zero internal contacts, and 5 s
   of zero command moves the tool 1.4 mm. The position-dependent Z floor was re-measured for
   the new geometry: **96% of the (x, y) columns can now go below the old -0.05**, the three
   far columns are unrestricted to -0.40, and only the columns next to the lidar keep
   -0.04..-0.08. The 4.9 policy is **not** compatible with the new pose (650 mm / 136 deg error
   at it), and a fine-tune from it does not recover (stalled at 0.34 m / 2.25 rad, KL -> 0);
   zeroing the actor's 8-D output layer breaks that basin (0.29 m / 0.74 rad within 100
   iterations) and is the starting point for the retrain.

   **Stage 4.10 training (2026-11-04) converged in 6000 iterations (8.97 h).** Started from
   4.9's `model_5000` hidden layers with the 8-D output layer **zeroed** -- fine-tuning from
   4.9 directly stalls in the "pull the arm back to the old pose" basin (0.34 m / 2.25 rad
   after 525 iterations, KL -> 0), while the zeroed readout reaches 0.29 m / 0.74 rad in 100.
   Final: EE position **0.056 m**, orientation **0.141 rad**, success **90.4%**, collision
   proxy **0.24%**; value 8.26 -> 1.65, entropy +0.55 -> -3.77, std 0.28 -> 0.168. Deployed
   checkpoint `stage410d_ppo/model_5999` (chosen over 4500/5000/5500 on 50 mm/5 deg coverage:
   65.0% against 43.3/48.3/53.3). On the 4.10 command box: 65.0% at 50 mm/5 deg (69.6% over
   collision-free-reachable targets), position median 32 mm, orientation 3.7 deg. **All eleven
   low-target collision scenarios are at 0% contact** with 10-19 mm clearance -- the gripper
   through the lidar is gone. New residual: arm self-collision `Link3 <-> Link5` on **11% of
   the steps** of random six-dimensional targets (4.9 had 3%, `Link3 <-> d455`), because the
   new default folds the wrist more; the relative 20 mm margin in the safety term is too weak
   for that fold and is the next thing to fix. The position-dependent Z floor is now actually
   used: per column the tool reaches -0.10/-0.15/-0.40 m on the far side and stops at 0.00
   directly above the lidar, against 4.9's single -0.05 everywhere.

   **Stage 4.10e (2026-11-04) extended the run to cumulative 10000.** `rsl_rl`'s
   `learn(num_learning_iterations)` is *relative*, so `--max_iterations 10000` from `model_5999`
   targets 15999 (the log says `9335/15999`), and stopping at 10000 needs `4001` from 5999 or `1000`
   from 9000. The extended run **diverged at iteration 9336** -- value loss `inf` then `nan`, then
   `normal(std < 0)` -- with the environment metrics still healthy one iteration earlier (EE 0.0515,
   success 93.7%, collision 0%); same critic-target blow-up as 4.5j. Log kept as
   `outputs/stage410/train_e_diverged_at_9336.log`; the run resumed from the last healthy checkpoint
   (`model_9000`) for the remaining 1000 iterations with no further NaN. Final measurements on the
   4.10 box (5999 -> 9999): 50 mm/5 deg 65.0 -> **68.3%**, 30 mm/2 deg 16.7 -> **28.3%**, position
   median 32 -> **22 mm**, orientation 3.7 -> **2.5 deg**, real self-collision 11.0 -> **9.5%**. The
   50 mm/5 deg figure is saturated (65-68% across 7000/8000/9000/9999), so the honest summary is
   "the last 4000 iterations bought 10 mm of position and 12 points of the strict tier, not a new
   operating regime". **Deployed `stage410f_ppo/model_9999`**, with `5999` kept as the margin-first
   alternative: the chassis clearance fell from 15 mm (5999) to 0.4 mm at 9000 and 7 mm at 9999 --
   never a contact, but the reward barely distinguishes anything inside 2 cm, so "trains into the
   margin" is structural. `Link3 <-> Link5` self-collision stays at 9.5% and is listed as the next
   stage's job, not folded into this result.

   **Stage 4.11 (2026-11-04) fixed the self-collision distance and did not fix the collision.**
   The distance proxy in `mdp/safety.py` had been copied by hand out of `m.mesh_vert`, which is the
   **geom frame**, not the link frame -- MuJoCo rotates these meshes 60-90 degrees, so a 675 mm
   link's box pointed along the wrong axis. Measured against MuJoCo, the old proxy was blind to
   **63.4%** of real `cr10_Link3 <-> cr10_Link5` contacts and optimistic by +107 mm; the new box
   distance is exact against a brute-force check (worst error 0.17 mm), misses **0** contacts, and
   is 8-15 mm conservative. The table is now generated by
   `scripts/ranger_cr10/build_collision_boxes.py` from the model, which also picked up the D455
   re-mount that 4.10 had left stale. Reward: absolute 15 mm margin (replacing the per-episode
   reference), self pairs included, structural pair excluded, self/sensor and per-pair clearance
   logged separately.

   Fine-tuned from `model_9999` with a fresh optimiser (`--load_weights_only`). The run stopped at
   **2800 of the planned 3500**: every restart from `model_2800` died within three minutes inside a
   PPO update (`value loss 3 -> 1e14 -> inf -> normal(std < 0)`), so the supervisor spent its five
   attempts without making progress. Checkpoints 500/1000/1500/2000/2200/2400/2600/2800 all verify
   finite. Accuracy plateaus after 1500 (50 mm/5 deg 17-23%, 30 mm/2 deg 0% from 2000 on) while
   real contact hovers at 7-9%. **It did not pass acceptance and `model_9999` stays deployed**: real
   contact went 10.0% -> 7.0% (target was <1%) while 50 mm/5 deg fell 68.3% -> 21.7% and 30 mm/2 deg
   28.3% -> 0.0%, with the minimum clearance getting *worse* (7 mm -> 1 mm). Per-target analysis
   shows the contact comes from 8 targets that are collision-free reachable (the 4 unreachable ones
   are never touched, 105 mm error), and the fine-tune rescued 3 of those 8 while sacrificing
   accuracy. The reward is not the problem -- at every real contact configuration the penalty is at
   its maximum and its slope is 150x the tracking term's -- the *proxy* is: one box per link is
   8-17 mm conservative, so 28.1% of configurations read as contact. Offline, a 4-slab-per-link table
   drops that to 9.5% and the bias to -5.3 mm; that is the next version's starting point, to be
   validated offline before any further training.

   **Stage 4.12 pre-flight (2026-11-05): the geometry did not pass, so no training was started.**
   Four representations were measured on the same 1800 operating poses: one box per link (45.9%
   false-alarm steps, band error -8.5 mm, 0 misses), four slabs (38.3%), the lidar as 16 radial
   sectors (43.2%), 6 oriented-box clusters (33.0%, band error -3.5 mm, **6.3% dangerous misses**),
   6 clusters with overlap/inflation (49-51%, 0 misses) and 16 clusters (18.9%, +18.7 mm tail).
   A union of boxes cannot be tight and conservative at once -- the mesh surface between two
   clusters lies inside neither box -- so nothing met the "<=3 mm with no misses" gate. The next
   step is an exact distance (convex decomposition + GJK), not a tighter box. The per-pair
   attribution from the previous entry was also wrong (an ad-hoc script labelled columns by their
   enumerate index); measured properly the lidar dominates: Link6-lidar 24.4%, Link3-Link5 19.1%,
   Link5-lidar 12.5%, Link4-Link6 11.2%.
   The critic blow-up was attacked from both arms: full restore and weights-only, 400 iterations
   each from `model_2800`, **neither reproduced it** (targets and predictions 8.6-9.7e1, gradients
   <=1.5e2 before clipping, value loss <=9.7). The training crash therefore does not follow from
   the checkpoint -- it depends on the run's data stream -- and the only established amplifier is
   the timeout bootstrap writing `gamma*V ~ 93` into a single-step reward. Recommended (not
   implemented): a guard that skips the update and rolls back instead of letting
   `torch.normal(std < 0)` end the process.

   **Stage 4.13 (2026-11-05) audited the reference geometry and built a distance baseline.** No
   training was run. The audit pins the three sources apart: MuJoCo 3.13.0 with margin 0 and no
   enable flags, 52 collision geoms over 9 Isaac groups, **36 pairs of which 31 are checked and 30
   enter the reward** (five parent/child pairs excluded by MuJoCo, the structural pair checked but
   not rewarded), and hashes for the poses, models and checkpoint. The convex hull is a loose *shape*
   -- 13-19 mm from the mesh at the median, up to 60 mm on the lidar -- but a faithful *interface*:
   where two links actually come closest the hull and mesh differ by **-0.2 mm median**. The box
   proxy on the same poses is **-17 to -22 mm median, -32 mm at p95**, so the 4.11/4.12 "false
   alarms" were the cover's fault, not the simulator's. A GJK prototype on the same hulls matches
   `mj_geomDistance` to **0.000 mm** at `tol=1e-8`, so algorithm error and geometry error are now
   separated and measurable. Convex decomposition is justified by the shape measurement but was not
   completed: `trimesh` here has neither shapely nor rtree, and the fidelity metric that works
   around it had not finished by the end of the stage -- reported as not run, not as passed. The GPU
   path and the 1/64/512/2048 performance table are also not done; CPU GJK measures 0.18 ms per pair.

   Two runs' worth of instability to record: `rsl_rl` died inside a PPO update at iterations 1402,
   1209 and 2037 with value loss 3 -> 1e14 -> inf -> `normal(std < 0)`, environment metrics healthy
   throughout -- the same failure as 4.10e at 9336, but now every 200-1000 iterations. Checkpoints
   500/1000/1500/2000 were all verified finite and the run was auto-restarted from the newest good
   one each time (`scripts/ranger_cr10/supervise_training.sh`, `check_training_health.py` -- the
   latter exists because `pgrep -f` on the script name matched the launching shell and made a dead
   run look alive for four hours).

   **Stage 4.9 (2026-11-02) gave the robot collision geometry and taught the policy to respect it.**
   The user saw the arm pass through the lidar and the chassis camera in MuJoCo; the reason was
   structural -- only the chassis, the steering forks, the wheels and the six arm links carried
   collision geometry, so the lidar, both RealSenses and the whole AG95 were invisible to
   physics and `d.ncon` stayed at 0 while the tool swept through them. Adding geometry exposed
   the real culprit: the *palms* kept 19 mm of clearance while the *fingertips* went 27 mm into
   the lidar at Z-0.25 (-44 mm at the keyboard-box corners), because the AG95's eight finger
   links had never had collision bodies. The generator now emits them (two jaw boxes, one per
   finger, leaving the jaw gap open), takes the lidar and both cameras from the URDF, fixes a
   1000x mesh-scale error that had made the chassis camera invisible since stage 4.6, and turns
   on robot-robot contacts with parent/child pairs excluded (30 pairs, all structural). Measured
   with the 4.8 policy on the corrected model: **51% of the steps of the Z-0.25 sweep are in
   contact** with the lidar, 93% at X-0.05/Z-0.25 and at low negative pitch, 5% of random
   six-dimensional targets. Isaac Lab cannot see any of it: its URDF conversion merges fixed
   joints, so `lidar_link0`, `d435_link`, `d455_link`, `gripper_base_link` and `cr10_base_link`
   are not articulation bodies there at all, and `enabled_self_collisions` is False. The
   safety signal is therefore an analytic clearance (`mdp/safety.py`: link covering spheres
   against base-frame boxes, reference-relative 20 mm margin, validated against MuJoCo's convex
   hulls as 94% conservative with 0% missed at the 20 mm criterion), not a contact force. A
   6000-iteration fine-tune from 4.8's `model_5999` ran with that penalty (8.8 h, curriculum
   starting at the final stage); **deployed checkpoint `stage49a_ppo/model_5000`**. Measured on
   the corrected model: contact with the lidar falls from 51%/93%/93%/12% (Z-0.25 sweep,
   X-0.05/Z-0.25, low pitch, keyboard corners) to **0% in every scenario**, and the only
   remaining contacts (3% of random six-dimensional targets) are the wrist camera grazing its
   own elbow. On targets that are collision-free reachable the 50 mm/5 deg success rate is
   **identical to 4.8 (77.2% vs 77.2%)** -- the safety term costs nothing where the target is
   executable, and the 1.7-point overall difference is exactly the 5% of targets that can only
   be "reached" by passing through the lidar. The keyboard box is re-measured and one axis had
   to shrink: **Z- goes from -0.25 to -0.05** because at Z-0.20 the tool is inside the lidar's
   volume -- the old range was never physically reachable, and 4.8 only appeared to hold it by
   penetrating. X-/Y+ grew slightly (X- -0.10, Y+ +0.15); roll/pitch/yaw unchanged.

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
