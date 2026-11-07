# Ranger CR10 simple-collision candidate (test only)

This directory is an independent test candidate. It does **not** replace the default
`RANGER-CR10-TELEOP-WBC` asset or configuration.

- `ranger_cr10_simple.urdf` replaces the Link1--Link6 collision mesh with one box per link,
  and gives `box_link` two solid box colliders (left enclosure + right mounting plate).
- `config.yaml` and `convert_ranger_cr10.py` build the USD variant. The generated USDs in
  `configuration/` and `ranger_cr10_simple*.usd` are intentionally **not** tracked
  (the flat USD is ~22 MB); rebuild them locally:

```bash
/home/ubuntu/miniconda3/envs/env_isaaclab5/bin/python \
  source/LeggedManip_Lab/LeggedManip_Lab/assets/ranger_cr10_simple/convert_ranger_cr10.py
```

The independent env is registered as `RANGER-CR10-TELEOP-SIMPLECOLL`. Its USD is flattened
(`ranger_cr10_simple_flat.usd`) and filtered pairs are added by the scripts under
`scripts/ranger_cr10/`; see `outputs/stage413/b/simple_collision_validation/` for the local
validation artifacts. Self-collision is enabled in this candidate; the default reward proxy
(`safety.py`) is unchanged and still uses the old A boxes.
