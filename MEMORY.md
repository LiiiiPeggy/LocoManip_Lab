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

**Expected row count for `python scripts/list_envs.py`: 32** (8 platforms × Flat/WBC ×
normal/`-Play`). Any other number means registration is broken. Assert the count — an empty table
looks like "no data", not "broken filter".

## Reading `Episode_Reward/*` without guessing the scale

Isaac Lab's `RewardManager` accumulates `term_cfg.weight * term_cfg.func(...) * dt`, and the RSL-RL
wrapper divides by `max_episode_length_s`. For an episode of `max_episode_length_s / dt` steps those
cancel, so the logged value is exactly

    Episode_Reward/<name> = weight * mean(term)

Measured, not derived: `end_effector_orientation_tracking` at weight -4.0 with a mean quaternion
error of 0.252 rad logs -1.048. The `* 10` and `/ 10` variants are both wrong, which is easy to get
wrong because `Train/mean_reward` is a different normalisation again.

The practical use is the reverse direction: a term whose mean is not logged (a speed, a distance)
can be read back off its own curve by dividing by the weight — that is how the chassis creep was
tracked mid-run before the acceptance test existed.

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

## The MuJoCo viewer steals the teleop keys — solved by handing the focus back (2026-10-31)

MuJoCo's built-in viewer (the C++ `simulate` UI behind `mujoco.viewer.launch_passive`) binds plain
letters to visualization toggles and 0-8 to geom-group visibility. `pynput` is a global **observer**
— it does not consume the event — so a focused viewer window receives the same keystrokes. Control
is unaffected; the picture is not.

Read from the tables in the installed `libmujoco.so.3.13.0` (`mjVISSTRING` / `mjRNDSTRING`), not
from docs. Overlaps: `W` wireframe, `S` shadow, `R` reflection, `L` additive, `K` skybox, `G` fog,
`A` auto-connect, `D` static body, `Q` camera, `E` equality, `I` inertia, `J` joint, `U` actuator,
`O` perturb object; `0`-`8` toggle geom groups. All 26 letters are bound, so letters cannot be
remapped out of the conflict.

`launch_passive(key_callback=...)` receives the key (int ASCII) but **cannot consume it** — measured
with the callback installed, pressing `w` still changed 37% of the window's pixels. Pinning
`handle.opt.flags` is not enough either: wireframe/shadow/reflection live in `mjRND_*` flags inside
the viewer's private scene, which no Python handle exposes.

What works: keep the **X input focus off the viewer window**. `ViewerFocusGuard` in
`mujoco/deploy/deploy_mujoco/ranger_cr10/ranger_cr10_deploy.py` records the focus owner before
`launch_passive`, hands it back if the viewer took it, and re-checks 4x/s (mouse control does not
need focus). `ViewerKeyPin` restores `opt.flags`/`opt.geomgroup` as a backstop.
`scripts/ranger_cr10/check_viewer_key_pinning.py` verifies on the picture itself: 38.0% of pixels
change without the guard, 0.0% with it.

## `m.mesh_vert` is the geom frame, not the link frame (2026-11-04)

MuJoCo re-expresses every mesh in its own geom frame: `geom_quat` is whatever the compiler
computed, and for the CR10 meshes it is a 60–90° rotation relative to the URDF link (Link3 turns
90° about y, Link5 87° about x, Link6 ~90° about z). `m.mesh_vert` therefore comes back **rotated**,
and a bounding box read off it is the right *shape* in the wrong *frame*.

This was silent for two stages: the numbers look plausible (same extents), the picture in the
viewer is unchanged (geom_xmat puts it back), and only the *distance* is wrong. Stage 4.9's
`mdp/safety.py` table was copied that way, so a 675 mm link's box pointed along the wrong axis and
its ends sat ~0.3 m from the geometry they were meant to cover. Measured over 1200 random arm
configurations: 62% of the real `cr10_Link3 <-> cr10_Link5` contacts were invisible to that proxy,
while MuJoCo found them on 9.5% of the steps of the 4.10 evaluation sweep.

Build the table from the model, not by hand — `scripts/ranger_cr10/build_collision_boxes.py` walks
each geom into the frame of the Isaac body that owns it (the URDF→USD conversion merges
`gripper_base_link`/`d455_link` into `cr10_Link6`, and `lidar_link0`/`d435_link` into `base_link`).
Re-run it whenever `build_mjcf.py` changes collision geometry: the 4.10 D455 re-mount left the old
mount position in the table for a whole stage.

Two things that look like the same bug and are not: `mj_geomDistance` returns **penetration depth**
(negative) for real contacts, so comparing it to a clearance clamps at zero and shows a large
positive "error"; and a box bound is 8–17 mm *conservative* against the mesh, so a proxy reading of
0 is not a contact. `scripts/ranger_cr10/check_clearance_proxy.py` reports both separately.

## `pgrep -f <script>` cannot tell you whether a training run is alive (2026-11-04)

`pgrep -f "rsl_rl/train.py"` matches **the shell that launched it** and the shell running the
check — their command lines contain the string. A run that had died at 14:22 looked alive at
18:50 because of this, and four hours of wall clock went into waiting for a process that was not
there.

`scripts/ranger_cr10/check_training_health.py` reads `/proc/*/cmdline` and requires **argv[0] to be
a Python interpreter** *and* an argv entry ending in `scripts/rsl_rl/train.py`; shells that merely
mention the path do not match. It also loads the newest checkpoint and rejects one holding any
non-finite tensor, because a run can die inside a PPO update and leave its last saved file fine
(that is what happens here: `value loss 3 -> 1e14 -> inf -> normal(std < 0)`, environment metrics
healthy throughout, at iterations 1402 / 1209 / 2037 of stage 4.11 and 9336 of 4.10e).

Related trap: this machine **suspends intermittently**. `ps etime` and the run's own
`Time elapsed` do not include the suspended time, so "2 h 10 m elapsed" and six hours of calendar
time can both be true. Trust checkpoints and log lines, not clock arithmetic.

## A union of bounding boxes cannot be both tight and conservative (2026-11-05)

Measured on the same 1800 operating poses each time (`scripts/ranger_cr10/check_proxy_operating.py`,
`analyze_proxy_pairs.py`), with the false-alarm share, the distance error inside the last 20 mm, and
the share of real contacts the proxy misses:

| representation | false-alarm steps | band error median | misses |
|---|---|---|---|
| one box per link | 45.9% | -8.5 mm | 0% |
| four slabs per link | 38.3% | - | 0% |
| lidar as 16 radial sectors | 43.2% | - | 0% |
| 6 oriented-box clusters | 33.0% | -3.5 mm | **6.3%** |
| 6 clusters, 2-overlap | 50.8% | - | 0% |
| 6 clusters, +3 mm inflate | 49.1% | - | 1.4% |
| 16 clusters | 18.9% | +2.7 mm | 7.8% |

The reason is geometric, not a tuning miss: the mesh surface between two clusters lies inside
neither box, so a box cover is *either* tight (and optimistic at contact) *or* conservative (and
loose everywhere). An exact distance needs a real convex decomposition plus GJK, not a better box.

Two things this cost, worth not repeating: the false alarms dominated by the **lidar** (Link6-lidar
24.4%, Link5-lidar 12.5%, Link4-lidar 7.7%, Link3-lidar 7.6% of operating steps), not by Link2 as an
earlier ad-hoc script claimed; and PCA-oriented boxes can be *larger* than the axis-aligned ones on
thin curved shells, so a per-cluster orientation has to be chosen by volume, not assumed.

## A runner built outside hydra must call `handle_deprecated_rsl_rl_cfg` (2026-11-05)

`train.py` goes through hydra, which migrates the legacy `stochastic` key into `distribution_cfg`.
A standalone script that builds `OnPolicyRunner` from `RangerCr10TeleopPPORunnerCfg().to_dict()`
without that call dies in `MLPModel.__init__() got an unexpected keyword argument 'stochastic'` --
and Isaac's `close()` swallows it, so the run looks like a hang with no message at all (see the
exit-code section above). Call it explicitly:

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

and print a traceback before `simulation_app.close()` in any script whose failure would otherwise be
invisible.

## A run stuck in `simulation_app.close()` keeps its GPU memory (2026-11-05)

Three diagnostic runs finished writing their output and then sat in `close()` for 500+ minutes of CPU
each, holding ~18 GB of the GPU between them and slowing every later run. `ps` showed them in state
`R` with a large `TIME`. Kill by PID when a finished run's log already says what it was going to say;
`pkill -f <pattern>` is not safe here because the pattern matches the shell issuing it.

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
- **A reward term that never fires.** Two independent ways, both seen on ranger_cr10: a threshold
  above anything the robot reaches (`clamp(speed - 0.3, 0) ** 2` against a chassis that moves at
  0.13 m/s), and an `exp(-d^2 / std^2)` kernel whose std is smaller than the distances the task
  actually samples (dead past ~3 std, i.e. 0.75 m for std 0.316). Check the term's *magnitude* in
  TensorBoard against its weight before believing it is doing anything.
- **An action term that rails and cannot come back.** `clamp()` has zero derivative outside its
  bounds, so a policy whose raw output drifts past the limit receives no gradient that could pull it
  back and stays pinned forever. Measured here at 4x the limit. Squash with `tanh` instead of
  clamping; check for it by logging the *raw* action next to the processed one.
- **A limit written as a magnitude but read as a range (2026-10-25).**
  `ori_limits_deg = ((22.9, 22.9), ...)` says "±22.9 deg" to a human and "low = high = +22.9" to
  `max(lo, min(hi, x))`, so the *first* press of any attitude key in `KeyboardTeleopController`
  snapped to +22.9 deg, minus keys included. Its check suite passed 22/22 because it tested the
  *position* clamp (which was correctly signed) and never the attitude step size or the negative
  side. A clamp test has to assert the step size and both signs at both ends -- "the value stopped
  changing" passes for a degenerate box.

- **An analysis script that labels columns by their enumerate index (found 2026-11-05).**
  A throwaway script iterated `enumerate(PENALTY_COLUMNS)` and then used the *enumerate index* to look
  up which body pair a column belonged to. The counts were right and every label was shifted, so the
  4.11 report blamed `Link2 <-> lidar_link0` for 9.2% of false alarms that actually belonged to other
  pairs. If a number is going to be attributed to a name, look the name up by the column id, and
  re-derive it a second way before it reaches a report.

- **A command sampler that aliases one tensor across axes (found 2026-10-29).**
  `torch.stack([r.uniform_(lo, hi) for ...])` looks like three draws and is one: `Tensor.uniform_`
  fills in place and returns the same tensor, so every stacked column holds the last value.
  `UniformTCPBodyPoseCommand._resample_command` did this for all three position offsets and all
  three Euler angles, so **every command the ranger_cr10 teleop task ever trained on** satisfied
  `offset_x == offset_y == offset_z` and `roll == pitch == yaw` -- a 2-D line inside the 6-D box
  (measured on 3142 real commands: all on the line to 1e-7). The policy matched that line to
  0.8 deg and was 15 deg off on independent-axis commands, which is what the keyboard produces.
  Check the *joint* distribution (pairwise correlation, duplicate fraction), never just the ranges
  -- a degenerate sampler passes every range test. Fixed in `pose_command_teleop.py`
  (`_uniform_axes`), asserted by `check_teleop_env.py` section D2 (4096 samples, |corr| < 0.1).
- **A quaternion is not interchangeable with its negation as a network input (2026-10-29).**
  `R_to_quat(quat_to_R(q))` can return `-q`; it is the same rotation and a completely different
  input row. Building a command with a matrix round-trip therefore produced a state the actor had
  only seen in the 3% of training rows with a negative `w`, and its output looked like a broken
  command channel (measured: 55 deg error for a command whose reconstruction differed only in
  sign). Always compose the commanded quaternion exactly the way the env does --
  `quat_mul(quat_from_euler_xyz(r, p, y), TCP_DEFAULT_QUAT_B)` -- and never through a matrix.
- **A supervision "test split" that shares commands with training is not a generalisation test
  (2026-10-26).** Every episode here is started with `env.reset(seed=...)`, so the command term
  replays the same RNG stream: the 4.5e IK dataset's held-out split sits **0.00 deg (median)**
  from the nearest training command, and its successor sets added 1089 and 1338 samples without
  adding a single new command. The fitted actor therefore measured 0.4 deg offline and produced
  10-12 deg on any command it had not memorised -- the "offline/closed-loop paradox" of stages
  4.5e-4.5f. Split by *command* (not only by episode) before believing an offline number, and
  check the command distance of the test set to the training set the same way.
- **MuJoCo free-joint velocities are in the world frame; the policy observes the body frame
  (2026-10-31).** Passing `qvel[0:6]` straight into `base_lin_vel` / `base_ang_vel` silently
  rotates the observation by the robot's heading, which looks like "the policy drifts when it
  turns" rather than like an interface bug. Verified by setting `yaw = 90 deg, qvel[0] = 1` and
  watching the body move along world +x. Do it explicitly: `R.T @ qvel[0:3]`.
  `mj_objectVelocity(..., flg_local=1)` returned the opposite sign of that rotation on this
  model, so it is not used. Checked end to end by `scripts/ranger_cr10/check_mujoco_interface.py`
  (same state into both simulators: `max|diff| = 2.5e-8` on the 117-D observation).
- **A drift measured as the *mean* of per-env positions is not a drift (2026-10-31).**
  `check_base_drift.py` averaged `root_pos_w` and `heading_w` over its 16 envs, and those envs
  differ in `base_com` randomisation, so they creep in different directions and cancel: the same
  script reports 2.5 mm per 30 s at 16 envs and **13.6 mm at `--num_envs 1`**. The stage-4.5k
  conclusion (the zero-lock removes most of it) survived; the magnitudes were low by ~5x. Report
  per-environment displacement (or both numbers), and prefer path length alongside net drift so
  an oscillation cannot masquerade as "standing still".
- **A MuJoCo cylinder wheel has no rolling resistance; an Isaac mesh tyre squashes and does
  (2026-10-31).** In Isaac the 176 kg chassis sinks ~7 mm into the wheel contact, so a small
  wheel-speed command produces no motion; with analytic cylinders the robot follows it, and
  low-speed tracking reads 0.071 m/s for a 0.05 command (Isaac: 0.060). Isaac's steady-state
  droop (5.7 Nm at 0.39 rad/s, 12.7 at 2.0) fits ~4 Nm Coulomb + ~4 Nm*s/rad viscous; adding only
  `frictionloss=4.02` to the wheel joints brings 0.0619 vs 0.0622 and halves the arm-driven
  drift. Keep the parameter switchable (`build_mjcf.py --wheel_frictionloss 0`).
- **"Success rate" is two different statistics and a curriculum must say which (2026-11-01).**
  `UniformTCPBodyPoseCommand` keeps `metrics["success"]` (the instantaneous fraction of
  environments on target, which `Metrics/ee_pose/success` logs) *and* `success_rate` (the running
  average since each environment's last resample). The second is dragged down ~0.10 by the 1-2 s
  transient after every command change: measured on the same training run, 0.824 instantaneous
  against 0.722 running. A stage gate reading the running average never promoted; the same gate
  reading the instantaneous value (with a 0.95 EMA, because that fraction fluctuates +-0.04 per
  sample) moves. Name the statistic *and* the smoothing before trusting a curriculum threshold.
- **A curriculum gated on *unfiltered* success has a ceiling equal to the reachable share
  (2026-11-02).** Stage 4.7's workspace curriculum stalled at success EMA 0.838 against a 0.85
  gate and never reached its last two boxes; the deeper reason is that a target the arm cannot
  physically hold fails forever, so no policy can exceed the fraction of the sampled box that is
  kinematically solvable (measured 84.2% for the 4.8 candidate box, 68.3% for 4.7's). Stage 4.8
  therefore counts success over a reachability-filtered subset and logs the *unfiltered* rate and
  the reachable share on the same line, so the filter can never quietly manufacture a promotion.
  Corollary: measure the filter's error in *both* directions -- the 4.8 table's dangerous
  direction (says reachable, IK cannot solve) is 3.0%, its conservative direction 25.5%, and only
  the first one corrupts the curriculum.
- **"Continuous reachability from the default posture" is the wrong test for *attitudes*
  (2026-11-02).** The stage-4.7 sweep walked each axis from `DEFAULT_Q` and reported
  `roll=-45` / `pitch=-30` as 0 everywhere, yet the policy demonstrably holds roll -30 deg in
  teleoperation -- because a policy may move the tool away first and rotate second. The reach
  table that trains the 4.8 curriculum therefore seeds each attitude cell with *any* solvable
  configuration (default position first, then the six axis extremes) and only then walks
  position. Use the from-default sweep for "what a single keystroke can reach"; use the seeded
  walk for "what the arm can hold at all".
- **A visual mesh is not a collision body, and nothing reports the difference (2026-11-02).**
  Until stage 4.9 the MuJoCo model had collision geometry only for the chassis, the steering
  forks, the wheels and the six arm links. The lidar, both RealSense cameras and the whole
  AG95 had *visual meshes only*, so the arm passed through them with `d.ncon` staying at 0 --
  and every earlier "no collisions" report was therefore vacuous. Two related traps: (a) a
  URDF `<mesh>` used for collision is *not* the real shape -- MuJoCo and PhysX both collide
  its convex hull, so an L-shaped part needs an explicit proxy; (b) check the mesh's authored
  units before applying a scale -- `d435.stl` is in metres while `d455.stl` is in millimetres,
  and a blanket 0.001 made the chassis camera 1000x too small and invisible for four stages.
- **Isaac Lab's URDF conversion merges fixed joints, so the sensors have no bodies there
  (2026-11-02).** `robot.body_names` on the converted articulation lists the arm links and the
  AG95 fingers but *not* `cr10_base_link`, `gripper_base_link`, `d455_link`, `lidar_link0` or
  `d435_link`: their meshes are folded into `base_link` / `cr10_Link6`. Any safety term must
  therefore be written in terms of bodies that exist on both sides, with the merged links'
  offsets folded in -- looking up `lidar_link0` by name works in MuJoCo and raises in Isaac.
  With `enabled_self_collisions=False` (every stage before 4.9) the folded-in geometry still
  never generates a contact, so the policy could not be taught to avoid it.
- **The "converted" URDF is generated from the vendor description URDF, and only one of them is
  tracked by the build (2026-11-04).** Stage 4.10 moves the D455 to a top mount and the change had
  to be made in *both* `rangerboxcr10lidar_description/urdf/rangercr10lidar.urdf` (the source) and
  `assets/ranger_cr10/ranger_cr10_converted.urdf` (what `build_mjcf.py` and `convert_ranger_cr10.py`
  actually read). Editing only the converted one works until the next regeneration, which silently
  restores the old geometry. `scripts/ranger_cr10/check_pose_sync.py` now asserts the two agree.
- **MuJoCo cameras look along their local -z (2026-11-04).** A `<camera>` placed at the D455's
  optical frame rendered the robot's own gripper until the quaternion was composed with a 180 deg
  roll about x: the optical frame's +z is the viewing direction, which is the *opposite* of what
  MuJoCo wants. Two related traps: a camera at the centre of a 12 cm camera mesh renders the inside
  of its own housing (offset past the front face), and `d.cam_xmat` is the camera frame, so its
  third column points *backwards*.
- **Changing the arm's default joint pose makes the trained policy unusable, and the obvious
  fine-tune does not recover (2026-11-04).** The action is `q_target = default + 0.25 * a`, and the
  observation contains the raw joint positions, so moving the default from
  `[0.6, 1.0, -2.4, 0, 1.4, 0]` to the forward-pointing pose sent the 4.9 policy to 885 mm / 134 deg
  error. Fine-tuning it from the trained actor stalls at 0.34 m / 2.25 rad for 500+ iterations with
  KL collapsing to 0 -- the policy keeps driving the arm back to the posture it knows. Zeroing only
  the 8-D *output* layer (keeping the hidden layers) breaks that basin: 0.29 m / 0.74 rad within 100
  iterations, and 0.056 m / 0.141 rad after 6000. Any future change to the default pose should
  expect to retrain this way, not to "warm start".
- **The sphere clearance proxy does not see shallow face-to-face grazing (2026-11-04).** It was
  validated against MuJoCo's convex hulls as *conservative* (94% of samples, 0% missed at the 20 mm
  criterion), yet at the end of the 4.10 run it reported 0.24% collision while MuJoCo's exact
  geometry reported 11% of the steps of the same random-target set in contact -- all of them
  `cr10_Link3 <-> cr10_Link5`, two long flat links sliding past each other. Spheres placed along a
  link's axis cover its edges but not the flat faces that graze across the gap between two sphere
  centres, so anything inside that residual should be measured with MuJoCo, not with the proxy the
  reward uses.
- **`rsl_rl`'s `learn(num_learning_iterations)` counts *forward* from the resumed iteration, and a
  long PPO run can die on a single critic-target blow-up (2026-11-04).** Resuming `model_5999` with
  `--max_iterations 10000` targets 15999, not 10000 (the log prints `iteration 9335/15999`); to stop
  at a cumulative number, pass the difference. The 4.10e continuation then diverged at iteration
  9336 -- value loss `inf`, then `nan`, then `torch.normal(std < 0)` -- while the environment metrics
  one iteration earlier were still healthy (EE 0.0515 m, success 93.7%, collision 0%). Nothing in the
  env was wrong: a single return of the wrong magnitude fed the critic, and every later parameter is
  downstream of it. Resume from the last saved checkpoint (9000 here), keep the crashed log, and
  treat "value loss rising for a few hundred iterations" as the warning that precedes it.
