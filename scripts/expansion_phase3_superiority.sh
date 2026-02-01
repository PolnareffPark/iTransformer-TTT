#!/bin/bash

# ==========================================================
# [Phase 3] Aggressive Superiority (Asset Exchange)
#   - GPU: 1
#   - Expansion 7: Look-back Window (L) Expansion
#   - Goal: Convert VRAM savings into forecasting accuracy
#
# [Usage]
#   Run in background:
#     nohup bash scripts/expansion_phase3_superiority.sh > expansion_phase3_superiority.log 2>&1 &
#
#   Check log:
#     tail -f expansion_phase3_superiority.log
#
# [Termination]
#   Stop ONLY this session (Safe for main benchmark):
#     pkill -f expansion_phase3_superiority.sh
#     pkill -f "run.py.*summary_expansion_phase3_superiority.csv"
#
# [Hardware Note]
#   Current: NVIDIA RTX 3090 (24GB)
#   Future: Can be scaled to A100 (80GB) by increasing batch_size or seq_len.
# ==========================================================

export CUDA_VISIBLE_DEVICES=0

model_name=VG_iTransformer
if [ "$DEBUG" == "1" ]; then
    summary_file=summary_expansion_phase3_superiority_debug.csv
else
    summary_file=summary_expansion_phase3_superiority.csv
fi

# Check for DEBUG mode
if [ "$DEBUG" == "1" ]; then
    echo "!!! DEBUG MODE ENABLED: Running fast checks (1 epoch, limited iters) !!!"
    debug_args="--debug 1 --train_epochs 1 --patience 1"
else
    debug_args="--train_epochs 100 --patience 5"
fi

# Aligned with Phase 1 parameters
common_args="--is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model $model_name \
  --data custom \
  --features M \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --output_subdir ablation/phase3 \
  $debug_args \
  --pooling mean \
  --num_groups 32 \
  --summary_file $summary_file"

for seed in 2021 2022 2023
do
    echo "=== [Seed: $seed] Phase 3.1 Look-back Window (L) Asset Exchange ==="
    # Standard L values: 96, 192, 336, 720
    
    # BASELINE: L=96 (Minimum Asset)
    echo ">>> Running Baseline (L=96) <<<"
    python -u run.py $common_args --model iTransformer --model_id traffic_base_L96_s${seed} --seq_len 96 --seed $seed

    # VG-iT: Expansion to L=720 (Asset Exchange)
    # Testing how VRAM savings from variate-grouping allow for larger temporal context
    for l in 192 336 720
    do
        echo ">>> Testing VG-iT with Expanded L: $l <<<"
        python -u run.py $common_args --model_id traffic_vgit_L${l}_s${seed} --seq_len $l --seed $seed
    done
    
    # Optional: Test Baseline L=720 to show it's either too slow or OOM-prone compared to VG-iT
    echo ">>> Testing Baseline with Expanded L: 720 (Counter-example) <<<"
    python -u run.py $common_args --model iTransformer --model_id traffic_base_L720_s${seed} --seq_len 720 --seed $seed
done

echo "Phase 3 Aggressive Superiority finished. Results: test_results/$summary_file"
