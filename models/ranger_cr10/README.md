# ranger_cr10 / model_9999 — the selected observation-control baseline

Frozen inference checkpoint for the body-relative teleoperation task on branch
`rangercr10-teleop`. This is the policy the repository treats as the current baseline; the
fine-tuned `model_10498` candidates from the SIMPLECOLL experiments are **not** published here
because neither improved same-protocol tracking (see `docs/plans/teleop_plan.md` §43–§46).

## Identity

| item | value |
|---|---|
| file | `models/ranger_cr10/model_9999.pt` |
| SHA-256 | `94ffb7af2f9d0d8864920fb2a38f799938f2b7afffe8c7a75a1050057a5d7ce5` |
| size | 5,125,429 bytes |
| source run | `logs/rsl_rl/ranger_cr10_teleop/2026-11-04_06-57-27_stage410f_ppo/` (stage 4.10f PPO, not versioned) |
| iteration | 9999 |
| task used for training | `RANGER-CR10-TELEOP-WBC` (`RangerCr10TeleopEnvCfg`, original mesh asset, self-collisions off) |
| also evaluated on | `RANGER-CR10-TELEOP-SIMPLECOLL` (v2 box asset, self-collisions on) — fixed-command tracking and B-distance safety |

The checkpoint stores `actor_state_dict`, `critic_state_dict`, `optimizer_state_dict`, `iter`
and `infos`. Only the actor is needed for inference; the critic and optimizer are kept so the
file stays a complete, resumable checkpoint.

## Policy interface (unchanged from training)

| item | value |
|---|---|
| observation | 117 = 39 quantities x 3 history frames: `base_ang_vel`(3), `projected_gravity`(3), `joint_pos`(6, `cr10_joint1..6`), `joint_vel`(6), `actions`(8), `velocity_commands`(3), `pos_commands`(7), `base_lin_vel`(3) |
| action | 8 = `[vx, wz, cr10_joint1..6]`; the gripper is not an action channel |
| actor | MLP 117-512-256-128-8, ELU, Gaussian output, no observation normalization |
| critic | MLP 68-512-256-128-1 (not needed for inference) |
| control rate | 10 Hz policy on 200 Hz physics (`decimation = 20`); the TCP is measured at `cr10_Link6` with a +0.14389 m tool offset, in the `base_link` frame |

## Loading it

The loaders in `scripts/ranger_cr10/` read `params/agent.yaml` **next to the checkpoint**, so this
directory keeps that layout (`params/agent.yaml` is included) and no loader change is needed:

```bash
# fixed-command evaluation (writes metrics, no training)
/home/ubuntu/miniconda3/envs/env_isaaclab5/bin/python \
  scripts/ranger_cr10/rollout_phys_proxy_fixed_commands.py \
  --num_envs 64 --steps 200 --seed 42 --env simplecoll \
  --checkpoint models/ranger_cr10/model_9999.pt \
  --out /tmp/model_9999_eval.json

# the same checkpoint in the original teleop environment
/home/ubuntu/miniconda3/envs/env_isaaclab5/bin/python \
  scripts/ranger_cr10/rollout_phys_proxy_fixed_commands.py \
  --num_envs 64 --steps 200 --seed 42 --env teleop \
  --checkpoint models/ranger_cr10/model_9999.pt \
  --out /tmp/model_9999_teleop.json
```

`params/agent.yaml` carries the architecture and PPO hyper-parameters the runner needs to build
the networks; inference only uses the actor. In Python:

```python
from rsl_rl.runners import OnPolicyRunner
runner = OnPolicyRunner(wrapper, train_cfg, log_dir=None, device=dev)
runner.load("models/ranger_cr10/model_9999.pt")
policy = runner.get_inference_policy(device=dev)   # deterministic mean action
```

## Storage note

`.pt` files are normally Git LFS objects in this repository (`.gitattributes`). The LFS batch
upload endpoint was unreachable from the machine that published this model, so `models/**/*.pt`
is explicitly stored as a regular Git object, together with the `check-added-large-files`
exclusion for this path — the same documented trade-off the repository already makes for the
liDAR meshes and one converted USD. The file is 5.1 MB, far below GitHub's per-file limit, so a
clone needs no `git lfs`. The published object is verified by SHA-256 above; see
`docs/plans/teleop_plan.md` §47 for the round-trip check.
