#!/bin/bash

# Standard Benchmark for iTransformer Papers (d_ff = 4 * d_model)
# Target: Beat Standard Baseline & G32_mean
# Dataset: Traffic (N=862)

export CUDA_VISIBLE_DEVICES=0
model_name=VG_iTransformer
py_path=/home/himchan/miniconda3/envs/CTSF/bin/python

echo "Starting iStandard (d_ff=2048) Benchmark & Multi-Token Ablation..."

# 1. Standard iTransformer Baseline (d_ff=2048)
echo "Running [1/4] iTransformer Standard Baseline (d_ff=2048)..."
$py_path -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_standard_baseline_df2048 \
  --model iTransformer \
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
  --d_ff 2048 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --train_epochs 1

# 2. G32 Fixed (Mean) with d_ff=2048
echo "Running [2/4] VG-iT G32 Fixed (df2048)..."
$py_path -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_standard_vgit_g32_fixed_df2048 \
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
  --d_ff 2048 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --train_epochs 1 \
  --num_groups 32 \
  --pooling mean

# 3. Sharp Dynamic v2: G32 + Gumbel + 4-Tokens per Group (df2048)
# Twist: 4 tokens per group handles information bottleneck better at G=32
echo "Running [3/4] VG-iT G32 Sharp v2 (Multi-Token 4, Gumbel, df2048)..."
$py_path -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_standard_vgit_g32_sharp_v2_df2048 \
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
  --d_ff 2048 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --train_epochs 1 \
  --num_groups 32 \
  --pooling dynamic \
  --use_variable_resolution 1 \
  --use_interaction_bridge 1 \
  --partition_strategy gumbel \
  --dynamic_tokens_per_group 4

# 4. Sharp Dynamic v2: G48 + Gumbel + 4-Tokens per Group (df2048)
echo "Running [4/4] VG-iT G48 Sharp v2 (Multi-Token 4, Gumbel, df2048)..."
$py_path -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_standard_vgit_g48_sharp_v2_df2048 \
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
  --d_ff 2048 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --train_epochs 1 \
  --num_groups 48 \
  --pooling dynamic \
  --use_variable_resolution 1 \
  --use_interaction_bridge 1 \
  --partition_strategy gumbel \
  --dynamic_tokens_per_group 4

echo "Ablation Completed. Review summary.csv for absolute Pareto analysis vs Baseline."
