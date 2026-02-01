#!/bin/bash

# ==========================================================
# [Expansion 10] Hardware Universality Profiler
#   - Purpose: Collect resource metrics (VRAM, FLOPs, Latency) across different GPUs
#   - Usage:
#       export CUDA_VISIBLE_DEVICES=1
#       bash scripts/profile_hardware_universality.sh
#   - Target: RTX 3090, A100, etc.
#
# [Usage]
#   Run in background:
#     nohup bash scripts/profile_hardware_universality.sh > hw_profile.log 2>&1 &
#
#   Check log:
#     tail -f hw_profile.log
#
# [Termination]
#   Stop ONLY this session (Safe for main benchmark):
#     pkill -f profile_hardware_universality.sh
#     pkill -f "run.py.*summary_hardware_universality.csv"
# ==========================================================

export CUDA_VISIBLE_DEVICES=0

model_name=VG_iTransformer
if [ "$DEBUG" == "1" ]; then
    summary_file=summary_hardware_universality_debug.csv
else
    summary_file=summary_hardware_universality.csv
fi

# Profiling doesn't need 100 epochs, 1 epoch is enough for VRAM/Speed but we run a few for stability.
# NOTE: This script is for RESOURCE PROFILING (VRAM, FLOPs, Latency).
# It runs only 1 epoch per configuration to capture peak memory and speed.
# Accuracy (MSE/MAE) is NOT the goal here.
# Check for DEBUG mode
if [ "$DEBUG" == "1" ]; then
    echo "!!! DEBUG MODE ENABLED: Running fast checks (1 epoch, limited iters) !!!"
    debug_args="--debug 1 --train_epochs 1 --patience 1"
else
    debug_args="--train_epochs 1 --patience 1" # Default for profiling is still 1 epoch speed test
fi

common_args="--is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 4 \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  $debug_args \
  --pooling mean \
  --num_groups 32 \
  --summary_file $summary_file"

echo "=== [Expansion 10] Hardware Universality Profiling Started ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Scaling N (Variates) up to 20,000 to show hardware independence
for n_vars in 862 1000 2000 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 65000 70000 75000 80000 85000 90000 95000 100000
do
    echo ">>> Hardware Profile: N=$n_vars <<<"
    seed=2021
    
    # 1. VG-iT (Ours)
    echo "Profiling VG-iT..."
    python -u run.py $common_args \
        --model_id hw_vgit_n${n_vars} \
        --enc_in $n_vars --dec_in $n_vars --c_out $n_vars \
        --seed $seed

    # 2. iTransformer (Baseline)
    echo "Profiling Baseline..."
    python -u run.py $common_args \
        --model iTransformer \
        --model_id hw_base_n${n_vars} \
        --enc_in $n_vars --dec_in $n_vars --c_out $n_vars \
        --seed $seed
done

echo "Hardware Profiling Finished. Results: test_results/$summary_file"
echo "Check VRAM and Latency columns to compare efficiency across different GPUs."
