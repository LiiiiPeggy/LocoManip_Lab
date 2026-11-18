#!/usr/bin/env bash
# Attribution round: re-run the six fixed-command evaluations with the full reward-term and
# phase instrumentation, then build the attribution summary.
set -euo pipefail
cd /home/ubuntu/locomani/LocoManip_Lab

PY=/home/ubuntu/miniconda3/envs/env_isaaclab5/bin/python
ORIGINAL=logs/rsl_rl/ranger_cr10_teleop/2026-11-04_06-57-27_stage410f_ppo/model_9999.pt
WEIGHT2=logs/rsl_rl/ranger_cr10_teleop/2026-11-09_23-55-08_physproxy_long_500/model_10498.pt
WEIGHT0=logs/rsl_rl/ranger_cr10_teleop/2026-11-16_18-41-15_physproxy_weight0_500/model_10498.pt
OUTDIR=outputs/stage413/b/attribution
mkdir -p "$OUTDIR"

$PY -m py_compile scripts/ranger_cr10/rollout_phys_proxy_fixed_commands.py \
    scripts/ranger_cr10/analyze_fixed_eval.py
echo COMPILE_OK

# Interface smoke: 4 env, 12 steps, one forced reset.
timeout 300 $PY scripts/ranger_cr10/rollout_phys_proxy_fixed_commands.py \
    --num_envs 4 --steps 12 --seed 42 --force-reset-step 5 --force-reset-env 0 \
    --checkpoint "$ORIGINAL" --out "$OUTDIR/smoke_4env.json"
echo SMOKE_OK

for seed in 42 43; do
    for pair in "original:$ORIGINAL" "weight2:$WEIGHT2" "weight0:$WEIGHT0"; do
        label="${pair%%:*}"; ckpt="${pair#*:}"
        echo "=== seed $seed $label ==="
        timeout 1800 $PY scripts/ranger_cr10/rollout_phys_proxy_fixed_commands.py \
            --num_envs 64 --steps 200 --seed "$seed" --checkpoint "$ckpt" \
            --out "$OUTDIR/seed${seed}_${label}.json"
    done
done
echo EVALS_DONE

$PY scripts/ranger_cr10/analyze_fixed_eval.py \
    --eval 42:original="$OUTDIR/seed42_original.json" \
    --eval 42:weight2="$OUTDIR/seed42_weight2.json" \
    --eval 42:weight0="$OUTDIR/seed42_weight0.json" \
    --eval 43:original="$OUTDIR/seed43_original.json" \
    --eval 43:weight2="$OUTDIR/seed43_weight2.json" \
    --eval 43:weight0="$OUTDIR/seed43_weight0.json" \
    --out "$OUTDIR/attribution_summary.json"
echo ATTRIBUTION_DONE
