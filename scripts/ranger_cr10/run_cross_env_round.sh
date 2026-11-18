#!/usr/bin/env bash
# Cross-environment comparison: the same three policies in the original teleop environment
# (original mesh asset, self-collisions off, 15 mm clearance term). The v2 results are reused
# from the attribution round; only these six rollouts are new.
set -euo pipefail
cd /home/ubuntu/locomani/LocoManip_Lab

PY=/home/ubuntu/miniconda3/envs/env_isaaclab5/bin/python
ORIGINAL=models/ranger_cr10/model_9999.pt
WEIGHT2=logs/rsl_rl/ranger_cr10_teleop/2026-11-09_23-55-08_physproxy_long_500/model_10498.pt
WEIGHT0=logs/rsl_rl/ranger_cr10_teleop/2026-11-16_18-41-15_physproxy_weight0_500/model_10498.pt
OUTDIR=outputs/stage413/b/cross_env
mkdir -p "$OUTDIR"

$PY -m py_compile scripts/ranger_cr10/rollout_phys_proxy_fixed_commands.py
echo COMPILE_OK

# Interface smoke in the teleop environment, with one forced reset.
timeout 400 $PY scripts/ranger_cr10/rollout_phys_proxy_fixed_commands.py \
    --num_envs 4 --steps 12 --seed 42 --env teleop --force-reset-step 5 --force-reset-env 0 \
    --checkpoint "$ORIGINAL" --out "$OUTDIR/smoke_teleop_4env.json"
echo SMOKE_OK

for seed in 42 43; do
    for pair in "original:$ORIGINAL" "weight2:$WEIGHT2" "weight0:$WEIGHT0"; do
        label="${pair%%:*}"; ckpt="${pair#*:}"
        echo "=== teleop seed $seed $label ==="
        timeout 2400 $PY scripts/ranger_cr10/rollout_phys_proxy_fixed_commands.py \
            --num_envs 64 --steps 200 --seed "$seed" --env teleop --checkpoint "$ckpt" \
            --out "$OUTDIR/teleop_seed${seed}_${label}.json"
    done
done
echo TELEOP_EVALS_DONE
