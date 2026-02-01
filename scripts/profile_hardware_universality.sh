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

# Profiling doesn't need 100 epochs, 1 epoch is enough for VRAM/Speed
if [ "$DEBUG" == "1" ]; then
    echo "!!! DEBUG MODE ENABLED: Running fast checks (1 epoch, limited iters) !!!"
    debug_args="--debug 1 --train_epochs 1 --patience 1"
else
    debug_args="--train_epochs 1 --patience 1"
fi

# Common args for profiling (NO label_len to match author)
common_args="--is_training 1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  $debug_args \
  --pooling mean \
  --num_groups 32 \
  --summary_file $summary_file"

echo "=== [Expansion 10] Hardware Universality Profiling Started ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ===========================
# Part 1: Synthetic Stress Test (Scaling N)
# ===========================
echo ""
echo ">>> Part 1: Synthetic Stress Test (Scaling N up to 100K) <<<"
for n_vars in 500 1000 2000 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 65000 70000 75000 80000 85000 90000 95000 100000
do
    echo ">>> Hardware Profile: N=$n_vars <<<"
    seed=2021
    
    # VG-iT (Ours)
    echo "Profiling VG-iT..."
    python -u run.py $common_args \
        --root_path ./dataset/ \
        --data_path stress_${n_vars}.csv \
        --data stress \
        --model $model_name \
        --model_id hw_vgit_n${n_vars} \
        --enc_in $n_vars --dec_in $n_vars --c_out $n_vars \
        --e_layers 4 \
        --learning_rate 0.001 \
        --seed $seed

    # Baseline
    echo "Profiling Baseline..."
    python -u run.py $common_args \
        --root_path ./dataset/ \
        --data_path stress_${n_vars}.csv \
        --data stress \
        --model iTransformer \
        --model_id hw_base_n${n_vars} \
        --enc_in $n_vars --dec_in $n_vars --c_out $n_vars \
        --e_layers 4 \
        --learning_rate 0.001 \
        --seed $seed
done

# ===========================
# Part 2: Real Dataset Profiling (Author's exact settings)
# ===========================
echo ""
echo ">>> Part 2: Real Dataset Profiling (Traffic, ECL, Solar) <<<"

# Traffic (N=862, e_layers=4, lr=0.001)
echo "Profiling Traffic (N=862)..."
python -u run.py $common_args \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --data custom \
    --model $model_name \
    --model_id hw_real_traffic \
    --enc_in 862 --dec_in 862 --c_out 862 \
    --e_layers 4 \
    --learning_rate 0.001 \
    --seed 2021

python -u run.py $common_args \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --data custom \
    --model iTransformer \
    --model_id hw_real_traffic_baseline \
    --enc_in 862 --dec_in 862 --c_out 862 \
    --e_layers 4 \
    --learning_rate 0.001 \
    --seed 2021

# Electricity (N=321, e_layers=3, lr=0.0005)
echo "Profiling Electricity (N=321)..."
python -u run.py $common_args \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --data custom \
    --model $model_name \
    --model_id hw_real_ecl \
    --enc_in 321 --dec_in 321 --c_out 321 \
    --e_layers 3 \
    --learning_rate 0.0005 \
    --seed 2021

python -u run.py $common_args \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --data custom \
    --model iTransformer \
    --model_id hw_real_ecl_baseline \
    --enc_in 321 --dec_in 321 --c_out 321 \
    --e_layers 3 \
    --learning_rate 0.0005 \
    --seed 2021

# Solar (N=137, e_layers=2, lr=0.0005) - Uses --data Solar and .txt file
echo "Profiling Solar (N=137)..."
python -u run.py $common_args \
    --root_path ./dataset/Solar/ \
    --data_path solar_AL.txt \
    --data Solar \
    --model $model_name \
    --model_id hw_real_solar \
    --enc_in 137 --dec_in 137 --c_out 137 \
    --e_layers 2 \
    --learning_rate 0.0005 \
    --seed 2021

python -u run.py $common_args \
    --root_path ./dataset/Solar/ \
    --data_path solar_AL.txt \
    --data Solar \
    --model iTransformer \
    --model_id hw_real_solar_baseline \
    --enc_in 137 --dec_in 137 --c_out 137 \
    --e_layers 2 \
    --learning_rate 0.0005 \
    --seed 2021

echo "Hardware Profiling Finished. Results: test_results/$summary_file"
echo "Check VRAM and Latency columns to compare efficiency across different GPUs."
