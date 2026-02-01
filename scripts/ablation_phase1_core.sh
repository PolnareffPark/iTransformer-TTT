#!/bin/bash

# ==========================================================
# [Phase 1] Core Architecture Ablation Study
#   - GPU: 1 (Shared resources, use for accuracy/VRAM only, ignore time)
#   - Dataset: Traffic (N=862)
#   - Prediction Length (H): 96, 720
#   - Seeds: 2021, 2022, 2025 (3-Seed for statistical significance)
#   - Hyperparameters: e_layers=4, d_model=512, d_ff=512 (Author's default)
# ==========================================================
# [Usage]
#   Run in background:
#     nohup bash scripts/ablation_phase1_core.sh > ablation_phase1_core.log 2>&1 &
#
#   Check log:
#     tail -f ablation_phase1_core.log
#
# [Termination]
#   Stop ONLY this ablation experiment (Safe for main benchmark):
#     pkill -f ablation_phase1_core.sh
#     pkill -f "run.py.*summary_ablation_phase1_core.csv"
# ==========================================================

# Default to GPU 1, but allow override from command line
export CUDA_VISIBLE_DEVICES=0

model_name=VG_iTransformer
if [ "$DEBUG" == "1" ]; then
    summary_file=summary_ablation_phase1_core_debug.csv
else
    summary_file=summary_ablation_phase1_core.csv
fi
# Check for DEBUG mode
if [ "$DEBUG" == "1" ]; then
    echo "!!! DEBUG MODE ENABLED: Running fast checks (1 epoch, limited iters) !!!"
    debug_args="--debug 1 --train_epochs 1 --patience 1"
else
    debug_args="--train_epochs 100 --patience 5"
fi

# Standard config (G=32, Statistical Pooling)
# Formatted for readability
common_args="--is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --output_subdir ablation/phase1 \
  $debug_args \
  --summary_file $summary_file"

for seed in 2021 2022 2023
do
    for h in 96 192 336 720
    do
        # BASELINE (Standard VG-iT)
        echo ">>> [BASE] G=32, H=$h, Seed=$seed <<<"
        python -u run.py $common_args --model_id traffic_base_h${h}_s${seed} --pred_len $h --num_groups 32 --seed $seed

        # 1.1 Systemic Locality (Ablation 1: Shuffling)
        echo ">>> [SHUFFLE] Ablation 1, H=$h, Seed=$seed <<<"
        python -u run.py $common_args --model_id traffic_shuff_h${h}_s${seed} --pred_len $h --num_groups 32 --use_shuffling 1 --seed $seed

        # 1.2 Communication Necessity (Ablation 4: No Global Interaction)
        echo ">>> [NO-INTERACT] Ablation 4, H=$h, Seed=$seed <<<"
        python -u run.py $common_args --model_id traffic_nointer_h${h}_s${seed} --pred_len $h --num_groups 32 --use_global_interact 0 --seed $seed

        # 1.3 Bridge Necessity (Ablation 6: No Gated Bridge)
        # Note: --use_interaction_bridge 0 implements SIMPLE ADDITIVE RESIDUAL (H_local + H_global)
        echo ">>> [BRIDGE] Ablation 6, H=$h, Seed=$seed <<<"
        python -u run.py $common_args --model_id traffic_bridge_h${h}_s${seed} --pred_len $h --num_groups 32 --use_interaction_bridge 0 --seed $seed

        # 1.4 G-Sensitivity (Ablation 5)
        for g in 8 16 64
        do
            echo ">>> [G-SENS] G=$g, H=$h, Seed=$seed <<<"
            python -u run.py $common_args --model_id traffic_g${g}_h${h}_s${seed} --pred_len $h --num_groups $g --seed $seed
        done
    done
done

echo "Phase 1 Core Ablation finished. Results: test_results/$summary_file"
