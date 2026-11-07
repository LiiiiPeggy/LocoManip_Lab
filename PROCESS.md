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

## 4. Stage 4.10: why the default pose moved forward, and why the retrain had to start from a zeroed output layer

Recorded because the numbers that decided it are not visible from the code.

### The requirement arrived as a pose, and the pose did not exist where it was asked for

The task was "tool along `base_link` +X, D455 at 0.88 m, camera on top". Two measurements
reordered the work:

* the tool's default approach axis is the **TCP +z**, and at the old default it pointed *down*
  (`(0.234, -0.046, -0.971)` in the base frame). "Point forward" is therefore a ~77 deg change, not
  `rpy = 0`;
* with the tool horizontal at a fixed 0.473 m, the arm can only hold the pose **extended**. IK has
  no solution at the old TCP x = 0.55 m (10 cm of position error with the attitude already met);
  the smallest workable position is x ~= 0.74, and 0.80 was chosen because it had the largest
  motion envelope of the candidates (X -0.10/+0.44, Y -0.44/+0.08, Z +-0.44).

That position change is not cosmetic: the command box is expressed *relative* to the default, so
its X and Y ranges were re-trimmed (`-0.15..+0.40`, `-0.40..+0.10`) to what the new posture can
cover. It also moved the TCP in front of the lidar, which is why the position-dependent Z floor
suddenly had room to open up (`safe_z_table.json`: 96% of columns below the old flat -0.05).

### Rolling `cr10_joint6` was tried and rejected on measurement

First implementation put the camera on top by rotating the wrist -142.3 deg. It works, but: the
camera ends up *above* while its housing is rolled 142 deg (the image is tilted), the joint is left
with 0.66 rad of travel, and 180 deg is not even reachable (the joint limit is +-3.14 rad, so
+-179.9 deg). The camera is also 94 mm off the tool axis, so "camera up" is a 142 deg roll, not 180.
The mount was changed in the URDF instead, `cr10_joint6` went back to 0, and the USD was
regenerated from the edited URDF -- the one place where a "fixed" joint is still a number worth
getting right.

### The retrain, and the failure that made it work

Changing the default joint pose invalidates any policy trained on the old one: the action is
`q_target = default + 0.25 * a` and the observation contains raw joint positions, so the 4.9 policy
went to 885 mm / 134 deg. The first attempt -- a plain fine-tune from the 4.9 checkpoint -- looked
like a learning-rate problem for an hour: position error fell 0.55 -> 0.35 and then flattened,
surrogate loss went to zero, measured KL to zero. It was not a learning-rate problem. The policy was
holding the *old* posture and treating the new one as an error to correct.

Zeroing the last (8-D) layer of the actor -- keeping every hidden layer, changing nothing else --
removed that bias: the policy initialises holding the new default, and the advantage signal is then
about tracking rather than about undoing the pose. 100 iterations later it was at 0.29 m / 0.74 rad,
and 6000 iterations later at 0.056 m / 0.141 rad. This is recorded as an *initialisation* trick, not
supervision: no IK target, no auxiliary loss, the 8-D action and 117-D observation are untouched.

### What the run did not fix, and why more iterations would not

The end-of-run MuJoCo evaluation found 11% of the steps of random six-dimensional targets in
`cr10_Link3 <-> cr10_Link5` contact, while the reward's own collision metric reported 0.24% on the
same box. The proxy is not wrong in the direction it was validated for (it never *misses* a chassis
contact), but spheres along a link's axis do not cover the flat faces that graze past each other, so
this class of self-contact is invisible to the training signal. Extending the run to 10 000
iterations would improve tracking; it would not remove that contact. Fixing it needs a box-to-box
distance (or a finer sphere set for those pairs) and therefore a separate training run.

## 5. Stage 4.12: why no training was started, and what the geometry can and cannot do

Stage 4.11 failed for a reason that took a while to see properly: the safety penalty was firing on
nearly half of nominal operating steps, so the policy bought a two-point contact reduction with a
48-point accuracy loss. The user's 4.12 brief set a gate -- the proxy had to be validated before any
further safety training -- and the honest outcome is that **it did not pass, so nothing was trained**.

### What was measured, in the order it mattered

1. **The attribution was wrong first.** An ad-hoc script from 4.11 blamed `Link2 <-> lidar_link0` for
   the false alarms; it had used an `enumerate` index where a column index was meant, so every label
   was shifted. Written properly (`analyze_proxy_pairs.py`), the lidar still dominates, but through
   `Link6`, `Link5`, `Link4` rather than `Link2`. A number that will be attributed to a name has to be
   looked up by the id, and derived twice before it goes into a report.
2. **Four representations, one measurement harness.** Every candidate was scored on the same 1800
   configurations a real policy visits (`check_proxy_operating.py`), not on random samples: random
   configurations made slab-splitting look like a 28% -> 9.5% improvement that turned out to be
   55.4% -> 47.8% on the poses that matter.
3. **The best-looking candidate was unsound.** Six oriented-box clusters cut the false alarms to 33%
   and the near-contact error to -3.5 mm, and missed 6.3% of real contacts. The union of boxes covers
   the vertices but not the surface between clusters, so tightness and coverage trade directly against
   each other. Every variant that restored coverage (overlap, inflation) went back to 49-51% false
   alarms, i.e. no better than the single box it was meant to replace.

### The decision

No safety PPO run was started, `model_9999` stays deployed, and the next attempt should attack the
*distance* rather than the *bounding volume*: convex decomposition into 4-8 pieces per link plus an
exact convex-convex distance. That is a bigger piece of engineering than a better fit, and the
measurement harness is already in place to judge it.

### The critic blow-up, which is a separate matter

Both diagnostic arms (full restore and weights-only, 400 iterations each, from the checkpoint the
training side died on five times) ran **without reproducing it**: targets and predictions stayed at
8.6-9.7e1, pre-clipping gradients at or below 1.5e2, value loss at or below 9.7. Every condition that
can be restored matches, so the blow-up does not follow from the checkpoint -- it depends on the run's
data stream. What is established is one amplifier: the timeout bootstrap writes `gamma * V ~ 93` into a
single-step reward, so a divergent value estimate is immediately amplified. The practical fix is a
guard that skips the update and rolls back on a non-finite loss, instead of letting
`torch.normal(std < 0)` end the process; that was recommended and deliberately not implemented during
a measurement stage.

## 6. Stage 4.13b0: the reference was wrong in a way that flattered it

The 4.13 report said the box proxy was 17 mm from the truth and the hull 0.2 mm. Both numbers came
from a reference that could not see two of the three cases that matter: it took a minimum of
vertex-to-surface distances, so two boxes crossing at their centres read as 90 mm of clearance, and
it never asked whether the meshes intersect at all. A reference that cannot represent contact cannot
validate a contact proxy, however carefully the comparison is tabulated afterwards.

The rebuild is deliberately small and checkable rather than a library integration: python-fcl could
not be installed (the conda-forge channel is unreachable from this machine), so the exact primitive is
a 15-feature triangle-triangle distance with a separate Moller intersection test and a winding-number
containment test, and the independent check is brute-force enumeration over every triangle pair
(0.000000 mm over 20 random mesh pairs). The analytic cases from the brief are the regression test:
crossing boxes must be 0, the 25 mm offset must be exactly 5 mm, and the large-triangle case must be
1 mm where the old centroid KNN returned 118.

Two things this cost, worth recording. The STL assets are vertex soups, so a closedness test that
shares vertex indices calls a perfectly good mesh open; welding is an in-memory repair whose result
must be reported next to the original, not instead of it. And even welded, the gripper base and the
lidar are not closed -- their containment is genuinely unknown, and the query says so instead of
guessing, which is the behaviour the stage asked for.

The replay at the new reference is ~10 s per pair in pure Python, so the six anomaly poses were run
and the full 1800-step sweep was not finished in this stage; that is reported as incomplete rather
than summarised from a partial sample.

## 7. Stage 4.13b: the judge had to be fixed before the candidate could be judged

The first independent-boundary run said the convex candidate missed two real overlaps. The
temptation was to tune the decomposition until those two samples passed. That would have been
tuning to a wrong label: the reference's triangle-triangle intersection test was the defect.

The test compared intervals on the intersection line using an absolute `EPS = 1e-12`. For nearly
parallel triangles the line direction `cross(n1, n2)` is tiny, the projected coordinates are
ill-conditioned, and the comparison calls separated triangles intersecting. Re-deciding the
reported triangle pairs with exact rational arithmetic (the inputs are doubles, so `Fraction(float)`
is exact) found **47 of 360** rows were reference false positives and none the other way. The two
"misses" were in that set: their triangles are 1.81-2.25 mm apart by the exact feature distance,
while the candidate pieces are 0.27-0.35 mm apart, so the candidate was being conservative, not
missing. The old "pass" in section 27.10 and the 77-overlap denominator were both built on the same
false positives and were retracted.

The fix is deliberately narrow: `triangle_triangle_intersect` now calls the exact rational test;
parallel-but-distinct planes are rejected by their plane offset, and only exactly coplanar pairs go
to a 2-D SAT. The reference rerun has 0 disagreements over 346 rows, and the corrected set has 51
true overlaps, all 51 detected, no misses. The exact test costs little because it is only used on
ambiguous pairs.

With a trustworthy judge, the candidate's real limitation was visible and different: it is
*conservative* everywhere (no optimistic error, no missed contact), but its convex cover overfills.
The 3-5 mm bin's P95 error was 3.73 mm, driven by four B2_2 samples on `Link3_p3 | Link5_p8`.
The first attribution of that overfill was wrong. The sampling script used `contains(point) is True`
(inside) as the outside flag, so the reported "p8 42/300 outside at 5.876 mm" and "p3 174/300 up to
38.5 mm" were **inside** points. With `contains is False`, p8 has 353/400 outside points but only
to P95/max 1.13/1.32 mm, while p3 has 157/400 outside at 4.02/5.31 mm (24 >3 mm): the dominant
side is Link3 p3, not Link5 p8.

The local correction therefore replaced only Link3 p3, using 50 original-mesh pieces selected from
a 2 mm `merge=False` decomposition (4 old + 50 = 54 pieces; p3-region coverage max 0.078 mm;
whole-Link3 overfill P95 1.56 mm, max 2.40 mm). Link5 stayed at 13 pieces. The fixed set gives
3-5 mm P95 **1.64 mm**, B **1.54**, C 1.58, overall 1.54, 51/51 overlaps, 0 misses, 0 unknown,
A 0/0. The 2x2 isolation is decisive: Link5-only leaves B2_2 at -3.89 mm; Link3-only drops it to
-0.92 mm; both sides add 810 pairs and no benefit. Query cost 702 pairs, median 0.071 s.

The pre-fixed holdout (seed 20261106, 317 rows, 12 new C clusters, 8 B families) gives 3-5 mm P95
1.58, C/B 1.60/1.42, 26/26 overlaps, 0 misses, 0 unknown. Coverage is short: only two new C
clusters reach overlap, so the cumulative count is 9, not 10; the holdout also did not reproduce
the dev p3 overfill, so it confirms no regression rather than independently reproducing the gain.
The branch is "passes the measured accuracy/safety acceptance; add one independent shallow-overlap
C cluster before discussing lidar; no PPO".

Three process lessons: a measurement chain used as an oracle needs its own oracle, and exact
arithmetic is cheap enough to use on the ambiguous cases; a candidate that never reports a larger
gap than the reference can still fail an accuracy criterion, because "conservative" is not the
same as "within tolerance"; and a sign is a measurement: "outside" must be `contains == False`, and
every overfill number must say which side of the surface it is on.

## 8. Stage 4.13b close: the near-zero trigger and the lidar alarm were both measurement questions

Two loose ends remained after the local correction. The first was the 0.5 mm near-zero condition in
`closest_to_in_frame`: it calls the exact triangle predicate only on leaf pairs whose 15-feature
distance is below 0.5 mm. That is an acceleration heuristic, not a conservative trigger; a true
intersection can have a larger closest-feature distance, and the BVH leaf pair can contain it while
the feature distance misses it. The final reference replaces it with a conservative triangle-AABB
screen inside each visited leaf pair, followed by the exact predicate on every AABB-overlapping
pair. Re-running the development and holdout sets gives **no label and no distance changes**
(346 and 317 rows), with AABB/exact-test medians 9.5/7 and 56/56 and maxima 1572/1251; the reference
median rises from ~6.6/6.8 s to 7.26/8.49 s. The candidate results therefore did not need a
recompute, and the heuristic is no longer used for certification.

The second was the radar question. The old box proxy's lidar alarms over the 1800 recorded operating
poses are dominated by Link2 (298 steps at the 0.02 m reward margin) and Link3 (50) with the
deployed built-in safety table; the external 1-box/4-slab tables flag Link5/Link6 instead. These
are different proxy versions on the same poses and model, so the two are reported separately and
their percentages are not compared. The earlier five-pose table mixed cap values with exact
distances and mixed the 0 mm and 20 mm thresholds; it is marked superseded. The corrected cap
semantics: 1 exact (48.876 mm), 28 `certified_culled` at a 50 mm lower bound, 0 unknown; the lower
bound is valid for 0/20 mm classification but not for an exact median/P95 or a nearest point.

The 2x2 on 29 alarm rows (five poses plus four by a pre-fixed rule) compares arm box/hull x lidar
box/hull. Contact/margin false positives: A (box+box) 11/16, B (arm hull+radar box) 7/10,
C (arm box+radar hull) **0/0**, D (hull+hull) **0/0**. The lidar box is the dominant false-alarm
source; the arm box contributes less; replacing only the radar box with the radar hull removes the
original proxy's **11 contact and 16 margin false alarms in these 29 valid samples**. All 29
reference distances are at least ~48.9 mm, so this is far-distance attribution only, not
near-contact precision or miss detection. The earlier "arm-side box" conclusion is retracted.
Alarm-selected proportions are not run-distribution rates. A future C-cluster
search may start from any seed whose independent reference is separated even when the candidate is
already conservatively in contact at s=0; the previous candidate-separation gate was wrong.

The local seam check (857 boundary faces, 500 samples) found an observed maximum gap of 0.0745 mm
with none above the 0.10 mm contact tolerance; that is an observation, not a strict no-gap proof.
The targeted tenth C cluster was not found within the recorded budget: its top seeds had reference
distances 1.5-3.4 mm but the candidate was already in contact at s=0 and their Link3 witness was
~39 mm outside the old p3 hull (a new-piece boundary, not the old p3 region). The cumulative
true-overlap count stays at 9, so the stage-close coverage condition is explicitly not met; no dense
interpolation was used to manufacture a tenth cluster.

## 9. Stage 4.13b radar replay: the far-distance baseline did not survive near contact

The next step after the 2x2 was to test the radar hull on the full operating distribution and on a
pre-fixed near-contact set. Both were run before the choice; the near set's selection rule (seed
20261108, 50k joint-limit samples, MuJoCo proposal screen, 4 joint clusters, 6 screen bands,
0.03 m reference cap, 900-query budget) was written to `radar_near_contact_rule.json` before C/D
were evaluated.

The 1800-pose x 6-pair replay is complete (10800/10800, 0 missing). The independent reference has
37 exact rows (minimum 40.538 mm), 10763 certified_culled at 50 mm, 0 failed/unknown and 0
reference contacts, so the run-distribution miss rate is N/A. A has 192 contact / 379 margin false
alarms; C and D have 0/0. A->C and A->D remove all 192/379 with no new alarms. That is a real
far-distance benefit, but it cannot choose the baseline: no reference row is below 20 mm.

The pre-fixed near set has 342 reference rows (248 exact, 94 culled, 0 failed, 80 surface
overlaps). C/D are not sufficient. Contact false positives are C 114, D 20 (A 226); margin false
positives C 105, D 43 (A 139); exact-separated error P95 is C 25.11 mm, D 19.60 mm (A 26.71 mm).
D is accurate on most 0-5 mm rows and at 40-50 mm (0.71 mm P95 on the 37 far exact rows), but it
fails 5-20 mm: Link2 5-10/10-20 P95 9.25/12.44 mm, Link3 6.40/15.28 mm, and Link4/5/6 also exceed
3 mm. C fails even Link3 3-5 mm at 3.43 mm because its arm box is looser. All three detect 80/80
reference overlaps, so the intersection path is exercised; the failure is precision and false
contact, not missed overlap. Coverage gaps are recorded, not padded: Link2 has no 0-5 mm sample,
Link6 no 0-3 mm, Link1 no near sample.

The 16-row diagnostic side attribution was initially misleading because
`MeshRef.closest_to_in_frame` called `_containment` with the guest vertex in its own frame. For
Link3 this produced eight false `lidar_hull_only` contacts (the reference mesh-mesh distance was
0.7-12.5 mm). The fix transforms the guest vertex into the host frame before the winding-number
test. Auditing the fix: the 342 near-set labels changed **0** (all 80 contacts are surface
intersections, none containment); the 1800 replay has zero contacts, and a full reference-only recheck of all 10800 replay rows gives zero label changes; the candidate holdout has 26
surface contacts and zero containment. The candidate artifacts are therefore unaffected; only the
v1 hybrid attribution was invalid and is marked superseded. The corrected contact-side result is
4 `both_hulls` + 12 `arm_box_only`. D's residual distance error is mixed: on 12 separated rows, 6
are arm-hull-dominant (Link2 3.3-10.6 mm, Link3 4.3-5.7 mm), 3 are lidar-hull-dominant (Link3
1.3-2.7 mm), and 3 are small. The far-distance radar-box conclusion must not be extended to near
contact, and the near result must not be reduced to "not radar": both arm-box/hull and lidar-hull
over-approximations contribute.

Choice: **neither C nor D** meets the per-bin 3 mm near-contact requirement. D is better than C,
but not enough. The next minimal task is a local arm Link2/3 patch, plus a local lidar patch only
for the lidar-hull-dominant rows, validated on the same near set and replay; no global
decomposition, no box tightening, no PPO. The Link3<->Link5 candidate remains frozen.

## 10. Stage 4.13b radar round 2: attribution first, then one shared radar patch

The round started by freezing the inputs and separating four label classes: strict surface
intersection, complete containment, clearance <=0.10 mm, and clearance >0.10 mm. The development
near set has 80 strict surface overlaps, 0 observed complete containment, 1 clearance <=0.10 mm
and 261 >0.10 mm, so both 80 and 81 are kept as denominators. Raw STL hosts are not watertight;
their complete containment is unknown and is not reported as separation. The old shorthand
"0 mm contact" is not used for the 1e-4 m tolerance table.

All 111 exact separated rows with MM-D >3 mm were attributed, not a sampled subset. The four
representations use the same transform, atomic constituents, union-min and MeshRef query. Results:
Link2 45/45 arm-side, Link3 13 arm / 10 radar, Link4 12 radar / 2 arm, Link5 13 radar / 2 arm,
Link6 14/14 radar. The monotonic relations HH <= HM/MH <= MM hold on every row (0 violations), and
HH vs the stored D GJK value differs by at most 0.018 mm. The single-side gains are reported but
not summed: the interaction term is material on Link4.

The shared radar cause justified one candidate, not three patches: replace the single global lidar
hull with 95 per-component convex hulls. All 95 components are retained (54 watertight, 41
non-watertight, 48 degenerate); the old hull is not retained; the union is not the old hull. The
asset records the source hash, weld/split/hull settings, seed, versions, per-file hashes and
mapping. Sampled coverage max is 0.049 mm; sampled unsigned overfill is P95/max 2.16/22.15 mm with
inside/outside unknown, and watertight-component outside is P95/max 8.15/51.14 mm. This is an
observation, not a proof, and the candidate is not a deployment asset.

On the fixed set the candidate lowers strict/tolerance/margin false positives 12/20/43 -> 10/14/28
and exact-error P95 19.60 -> 16.69 mm, with 0 false negatives; all 80/81 positives remain. On the
new holdout (seed 20261109, 187/500 references, 36/36 positives) the false positives go 10/11/31 ->
8/9/17 and P95 17.29 -> 12.65 mm. Link6 passes 0-20 on dev and holdout; Link5 passes 0-20 on the
holdout and 10-20 on dev, but dev 3-5/5-10 are 3.11/3.94 mm (just over). Link2 is unchanged and
fails; Link3 and Link4 still fail 10-20. This is a local success, not an overall pass. The 1800
replay reference was rechecked component-aware with 0 changes; D min is 40.12 mm and the candidate
is a subset of D's lidar hull, so no new alarm can occur.

Two tooling defects were found and kept visible. The containment implementation only used the
first mesh vertex; it now uses one representative point per connected component, with deterministic
regression cases, and flags non-watertight hosts unknown. The holdout search writer also omitted
the contact reason string; the original file is kept and marked, and the relabeled file is used for
strict/tolerance tables. Both fixes changed 0 labels, but the historical artifacts are not
pretended to be the corrected ones.

## 11. Stage 4.13b simple boxes: local blocking is not global stability

The user allowed a simplified arm collision cover: a single box per link, cross-section between the
thin shaft and the thickest housing, slightly shorter along the link, and a solid box_link. The
arm boxes are in `assets/ranger_cr10_simple/ranger_cr10_simple.urdf` (full extents) and the matching
MuJoCo XML; the flat USD is the independent PhysX asset. box_link could not be one AABB: its mesh is
a thin shell with a left enclosure and a right mounting plate, and a single AABB puts Link1/Link2
inside it at the default pose (-274/-220 mm). Two boxes preserve the gap for the arm mount. The
default config, reward proxy and frozen assets are untouched.

The test config `RANGER-CR10-TELEOP-SIMPLECOLL` has self-collisions actually enabled on the imported
articulation (checked on the stage, not just the YAML). 77 filtered pairs exclude chassis/steering/
wheel, adjacent arm and AG95 closure pairs; arm-vs-box, arm-vs-lidar/d435 and non-adjacent arm pairs
remain enabled. Local blocking works in slow joint-target tests: the box left enclosure and the
non-adjacent Link3<->Link6 pair stop at contact (on pair distance -0.026 mm / +9.74 mm) while the
off comparison penetrates 24.3 mm / 21.7 mm. But the global behavior fails: self-on hold moves the
base 0.182 m (off 0.029), small sinusoidal motion produces ~189 kN and 0.193 m, and the plate target
stops 69.9 mm early. The plate result is confounded, not a clean pass. The cause still to isolate is
the remaining chassis/adjacent self-contacts or ineffective filtered-pair application; do not call
"self-collision enabled" a validation pass.

The simple distance proxy comparison shows the trade-off: on 1800 poses it removes all 379 old A
margin false alarms, but on the 342 near rows it misses four true Link3 path overlaps (P0021--P0024)
because the shortened Link3 box does not cover that surface region. It is therefore not a reward
proxy. Next work: isolate the global self-collision contacts/filters, restore coverage of those four
Link3 regions, then decide whether the simplified box length is acceptable.

## 12. Stage 4.13b v2: the self-collision "failure" was a filter and test-protocol artifact

The v1 stage-4.13b conclusion -- local blocking works but global self-collision is unstable -- was
reopened because three independent effects had been read as physics failure.

First, the test protocol did not hold the pose it claimed. Zero action with `use_default_offset`
left the joint targets at zero, so the arm moved to a zero pose (the 2.3--2.8 rad "drift" in the
first stability JSON). The 200-step hold also hit the 200-step time-out exactly: the last env step
reset the robot to the origin, so the final base position looked like a 0.18 m displacement. A
`base_contact` termination could additionally teleport the robot on the first large contact. The
v2 tests therefore disable `base_contact`, `bad_orientation` and `time_out`, use
`use_default_offset=False`, and hold the reset home joint targets.

Second, and decisive, the filtered pairs were not effective. The v1 writer looped over body pairs
and called `ClearTargets` for each pair, so a collider that appears in several pairs kept only the
last pair's targets; the 77 recorded pairs collapsed to 21 surviving pairs with one target each.
Accumulating all 78 pairs and 272 targets fixed that but still produced `base_link` vs front/rear
left/right wheels at 83.6/72.1/55.7 kN and gripper-link contacts at 2--5 kN. The contact points are
the wheel/steering positions and the minimum separations (-20.1/-18.5/-145.4 mm) match MuJoCo's
169--254 mm chassis-wheel/steering overlaps. **PhysX ignores collider-level `FilteredPairsAPI` in
this asset; the same pairs applied to rigid-body prims produce zero self-contacts.** The fix is
body-level filters with accumulated targets, not lower stiffness, higher damping, moving box_link
or changing the default pose.

Third, the reported "P0021--P0024 new distances 1.19/4.18/5.18/18.62 mm" were the global minimum
over every arm link and every protected collider, then compared with a reference label for the
Link3-lidar pair. Recomputing per pair: those numbers are Link5-vs-base (P0021--P0023) and
Link4-vs-base (P0024); the Link3 box to lidar is 52.93/51.94/51.60/42.29 mm in both v1 and v2.
The 20 mm global false-safe rows are D0106/D0109/D0157/D0264, not P0021--P0024. The old table is
kept as history; the v2 comparison exports per-row global and row-pair distances and shows both
confusion matrices, so a global true positive cannot be mistaken for pair detection.

The v2 geometry addresses the original joint-overlap concern with evidence rather than a uniform
scale. Link4's box is shortened to the central 48 mm between joint4 and joint5 (30 mm and 47 mm
end retreats); Link5's box is re-centred from the distal side to the central/proximal 80 mm
(55 mm before joint5, 83 mm before joint6); Link1's bottom is raised 10 mm; Link2/3/6 are
unchanged. Default adjacent box distances become 40.0/16.0/25.8/2.0/44.4 mm, and across the 1800
operating poses every adjacent, non-adjacent and protected pair is non-negative. The remaining
coverage gaps are deliberate joint-end retreats, not accidental overfill; the Link4/5 surface
coverage drop is recorded rather than hidden.

box_link was re-measured component-wise. The left enclosure shell/frame ends at a lid at z=0.2555;
the v1 box top of 0.3816 was there only to include small accessories (a bracket, two plates and a
connector). The v2 left box stops at the lid and a small 0.18x0.18x0.1261 m box covers the visible
top accessory (7.8% of the lid footprint); the right mounting plate collision was removed per user
request and is visual only. The freed volume is 0.0525 m3 gross, 0.0484 m3 net of the top box; 494
sampled accessory surface points (mostly p1/p2) remain outside. The free over-box test drives Link2
through the old upper slab (-91.5 mm against v1) and reaches the target with +18.6 mm against v2,
no contact. The old plate target no longer corresponds to a collision; it is recorded as
not_applicable rather than a plate pass.

Two smaller defects were fixed in the candidate build. The AG95 conversion left
`gripper_finger1_inner_knuckle_joint` and `gripper_finger1_finger_tip_joint` with an empty
`physxMimicJoint:rotY:referenceJoint`; all seven now target `gripper_finger1_joint` (gearing,
offset and limits unchanged). The old Link3-Link6 blocking target was also confounded: the
gripper fingers hit the box at steps 72/75 before the target pair at step 111, so the "9.74 mm"
stop was not the target pair. The clean target keeps every arm box in front of the box
(x > -0.02, Link6 x > 0.12) and contacts only Link3-Link6.

The chassis collision needed the same layer audit. The MuJoCo model used an AABB box of the
ranger_base_link mesh (0.4926 m3), while the URDF/PhysX chassis was already the mesh convex hull
(346 faces, 0.3331 m3). The box was an overfilling proxy, not the real collision. A trial scaled
the URDF chassis collision mesh by 0.98 and used the same scaled 346-face hull in MuJoCo (0.3135
m3); across the 1800 poses the arm-chassis minima were Link1 55.3 / Link2 62.2 / Link5 68.3 mm.
The tighter hull made the global simple proxy report a P0024 false-safe (Link5 box +10.6 mm while
the original Link5 mesh still penetrated the unscaled chassis by 66.4 mm). Per user request that
trial was reverted; the current candidate keeps the previous version: URDF/PhysX unscaled hull,
MuJoCo AABB box. The 1800-pose arm-chassis minima are Link1 52.05 / Link2 59.0 / Link5 24.14 mm,
all positive, and the global proxy is back to tolerance FN0/FP164, margin FN4
(D0106/D0109/D0157/D0264); row-pair metrics remain 31/81 and 60/190.

The distance proxy still fails as a reward proxy even after the geometry: row-pair tolerance
misses 31/81, margin misses 60/190. All 20 Link3 misses localise to the distal joint4 retreat
(local x -0.574..-0.610, outside the box end -0.554); 11 Link6 misses remain in the compact wrist
box. Filling the joint gap to remove them would reintroduce adjacent overlap, so it is not done.
Single-env self-collision stability and the three clean blocking tests pass; a small-scale
self-collision-only parallel validation may proceed, but no PPO and no reward-proxy replacement.
