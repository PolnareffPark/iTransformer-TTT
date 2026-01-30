#!/bin/bash

# Ablation Study for Phase 17 Components
# Dataset: Traffic (N=862)
# Environment: G=128 (Found to be effective in Phase 17 initial tests)

export CUDA_VISIBLE_DEVICES=0

model_name=VG_iTransformer

echo "Starting Phase 17 Ablation Study..."

# 1. Base Fixed Grouping (Control)
echo "Running [1/4] Fixed Grouping (G=128)..."
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_ablation_fixed_g128 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'test' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0001 \
  --train_epochs 1 \
  --num_groups 128 \
  --pooling mean

# 2. Dynamic Partitioning Only (VR Off, Bridge Off)
echo "Running [2/4] Dynamic Partitioning Only (G=128)..."
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_ablation_dynamic_only_g128 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'test' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0001 \
  --train_epochs 1 \
  --num_groups 128 \
  --pooling dynamic \
  --use_variable_resolution 0 \
  --use_interaction_bridge 0

# 3. Dynamic Partitioning + Variable Resolution (Bridge Off)
echo "Running [3/4] Dynamic Partitioning + VR (G=128)..."
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_ablation_dynamic_vr_g128 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'test' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0001 \
  --train_epochs 1 \
  --num_groups 128 \
  --pooling dynamic \
  --use_variable_resolution 1 \
  --use_interaction_bridge 0

# 4. Full Dynamic Grouping (VR On, Bridge On)
echo "Running [4/4] Full Dynamic Grouping (G=128)..."
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_ablation_dynamic_full_g128 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'test' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0001 \
  --train_epochs 1 \
  --num_groups 128 \
  --pooling dynamic \
  --use_variable_resolution 1 \
  --use_interaction_bridge 1

echo "Ablation Study Completed. Results saved in test_results/summary.csv"
