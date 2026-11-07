# Ranger CR10 simple-collision candidate v2 (test only)

This directory is the independent v2 test candidate loaded by
`RANGER-CR10-TELEOP-SIMPLECOLL`. It does **not** replace the default
`RANGER-CR10-TELEOP-WBC` asset or configuration.

- `ranger_cr10_simple_v2.urdf`: Link1--Link6 collision is one box per link with
  joint-end retreats; `box_link` has the left enclosure body up to its lid and a
  small box for the top accessory. The right mounting plate collision was removed per
  user request (it remains visual only). Mass, inertia and joint parameters are copied
  unchanged from the v1 candidate; only the collision blocks differ.
- The chassis collision is the previous version: URDF/PhysX use the unscaled `ranger_base_link.STL`
  convex hull; MuJoCo `base_link_col` is the AABB box. A 0.98-scaled-hull trial was reverted per
  user request and is not part of the current candidate.
- `config.yaml`: URDF-to-USD conversion settings. `merge_fixed_joints: true`,
  `make_instanceable: false`, `collider_type: convex_hull`, `self_collision: true`.
- Generated files (`ranger_cr10_simple_v2.usd`, `configuration/`, `.asset_hash`,
  `ranger_cr10_simple_v2_flat.usd`) are intentionally **not** tracked. The flat USD is
  ~22.8 MB and is rejected by the repository's 5000 KB large-file rule.

## Rebuild (from the repository root)

The URDF converter appends its generation header to the config in `usd_dir`; running
this config writes only into this directory.

```bash
PY=/home/ubuntu/miniconda3/envs/env_isaaclab5/bin/python
ASSET=source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10_simple_v2

$PY source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10_simple/convert_ranger_cr10.py \
  --config $ASSET/config.yaml

$PY scripts/ranger_cr10/flatten_simple_usd.py \
  --input $ASSET/configuration/ranger_cr10_simple_v2_physics.usd \
  --output $ASSET/ranger_cr10_simple_v2_flat.usd

$PY scripts/ranger_cr10/fix_mimic_references.py \
  --usd $ASSET/ranger_cr10_simple_v2_flat.usd \
  --out-json outputs/stage413/b/simple_collision_v2/mimic_references.json

$PY scripts/ranger_cr10/apply_collision_filters.py \
  --usd $ASSET/ranger_cr10_simple_v2_flat.usd --mode body \
  --out-json outputs/stage413/b/simple_collision_v2/filters_v2.json
```

The flatten step is deliberate: the PhysX filtered-pair targets must exist at the
simulation scope. The filters are applied to **rigid-body prims**; collider-level
relationships were measured not to take effect in this PhysX setup. The 77 filtered
body pairs cover the closed chassis/steering/wheel assembly, the five adjacent arm
pairs and the AG95 closed linkage. Arm vs box_link, arm vs chassis/lidar/D455 and
non-adjacent arm pairs stay enabled.

`fix_mimic_references.py` repairs the two AG95 mimic joints whose
`physxMimicJoint:rotY:referenceJoint` target is empty after conversion; all seven
mimic joints then reference `gripper_finger1_joint`.

## Load path

- env id: `RANGER-CR10-TELEOP-SIMPLECOLL`
- env cfg: `source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/config/ranger_cr10/simple_collision_env_cfg.py`
- USD: `source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10_simple_v2/ranger_cr10_simple_v2_flat.usd`
- self-collision: `physxArticulation:enabledSelfCollisions = true` on the imported articulation

The default `ranger_cr10.usd`, `safety.py`, reward, observation, action and default
deployment configuration are untouched. No PPO is run for this candidate.
