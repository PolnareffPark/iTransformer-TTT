#!/bin/bash

# ==========================================================
# [Usage]
#   Run in background:
#     nohup bash scripts/full_benchmark_vgit.sh > full_benchmark_clean.log 2>&1 &
#
#   Check log:
#     tail -f full_benchmark_clean.log
#
# [Termination]
#   Stop all benchmark processes:
#     pkill -f full_benchmark_vgit.sh
#     pkill -f run.py
# ==========================================================

# Full Comprehensive Benchmark for VG-iT G32 Fixed (Pareto Best)
# Datasets: Traffic (N=862), Electricity (N=321), Solar (N=137)
# Prediction Lengths: 96, 192, 336, 720
# Hyperparameters: EXACTLY matching author's defaults per dataset
# VG-iT specific: G=32, pooling=mean (our best)

export CUDA_VISIBLE_DEVICES=0
py_path=/home/himchan/miniconda3/envs/CTSF/bin/python

pred_lens=(96 192 336 720)
seeds=(2021 2022 2023)

# =============================================================================
# Dataset 1: Traffic (N=862)
# Author settings: e_layers=4, d_model=512, d_ff=512, lr=0.001, batch=16
# =============================================================================
for pl in "${pred_lens[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "=========================================================="
        echo "Running Traffic Benchmark: Pred_Len=$pl, Seed=$seed"
        echo "=========================================================="

        # Baseline
        echo "[1/2] iTransformer Baseline..."
        $py_path -u run.py \
          --is_training 1 \
          --root_path ./dataset/ \
          --data_path traffic.csv \
          --model_id "traffic_baseline_h${pl}_s${seed}" \
          --model iTransformer \
          --data custom \
          --features M \
          --seq_len 96 \
          --pred_len $pl \
          --e_layers 4 \
          --enc_in 862 --dec_in 862 --c_out 862 \
          --d_model 512 --d_ff 512 \
          --batch_size 16 \
          --learning_rate 0.001 \
          --output_subdir benchmarks/baseline \
          --train_epochs 100 --patience 5 \
          --seed $seed

        # VG-iT
        echo "[2/2] VG-iT G32 Fixed..."
        $py_path -u run.py \
          --is_training 1 \
          --root_path ./dataset/ \
          --data_path traffic.csv \
          --model_id "traffic_vgit_g32_h${pl}_s${seed}" \
          --model VG_iTransformer \
          --data custom \
          --features M \
          --seq_len 96 \
          --pred_len $pl \
          --e_layers 4 \
          --enc_in 862 --dec_in 862 --c_out 862 \
          --d_model 512 --d_ff 512 \
          --batch_size 16 \
          --learning_rate 0.001 \
          --output_subdir benchmarks/vgit_g32 \
          --train_epochs 100 --patience 5 \
          --num_groups 32 --pooling mean \
          --seed $seed
    done
done

# =============================================================================
# Dataset 2: Electricity (N=321)
# Author settings: e_layers=3, d_model=512, d_ff=512, lr=0.0005, batch=16
# =============================================================================
for pl in "${pred_lens[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "=========================================================="
        echo "Running Electricity Benchmark: Pred_Len=$pl, Seed=$seed"
        echo "=========================================================="

        # Baseline
        echo "[1/2] iTransformer Baseline..."
        $py_path -u run.py \
          --is_training 1 \
          --root_path ./dataset/ \
          --data_path electricity.csv \
          --model_id "electricity_baseline_h${pl}_s${seed}" \
          --model iTransformer \
          --data custom \
          --features M \
          --seq_len 96 \
          --pred_len $pl \
          --e_layers 3 \
          --enc_in 321 --dec_in 321 --c_out 321 \
          --d_model 512 --d_ff 512 \
          --batch_size 16 \
          --learning_rate 0.0005 \
          --output_subdir benchmarks/baseline \
          --train_epochs 100 --patience 5 \
          --seed $seed

        # VG-iT
        echo "[2/2] VG-iT G32 Fixed..."
        $py_path -u run.py \
          --is_training 1 \
          --root_path ./dataset/ \
          --data_path electricity.csv \
          --model_id "electricity_vgit_g32_h${pl}_s${seed}" \
          --model VG_iTransformer \
          --data custom \
          --features M \
          --seq_len 96 \
          --pred_len $pl \
          --e_layers 3 \
          --enc_in 321 --dec_in 321 --c_out 321 \
          --d_model 512 --d_ff 512 \
          --batch_size 16 \
          --learning_rate 0.0005 \
          --output_subdir benchmarks/vgit_g32 \
          --train_epochs 100 --patience 5 \
          --num_groups 32 --pooling mean \
          --seed $seed
    done
done

# =============================================================================
# Dataset 3: Solar (N=137)
# Author settings: e_layers=2, d_model=512, d_ff=512, lr=0.0005
# CRITICAL: Uses --data Solar (special loader), .txt file, ./dataset/Solar/ path
# NO --label_len (author's script does not use it)
# =============================================================================
for pl in "${pred_lens[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "=========================================================="
        echo "Running Solar Benchmark: Pred_Len=$pl, Seed=$seed"
        echo "=========================================================="

        # Baseline
        echo "[1/2] iTransformer Baseline..."
        $py_path -u run.py \
          --is_training 1 \
          --root_path ./dataset/Solar/ \
          --data_path solar_AL.txt \
          --model_id "solar_baseline_h${pl}_s${seed}" \
          --model iTransformer \
          --data Solar \
          --features M \
          --seq_len 96 \
          --pred_len $pl \
          --e_layers 2 \
          --enc_in 137 --dec_in 137 --c_out 137 \
          --d_model 512 --d_ff 512 \
          --learning_rate 0.0005 \
          --output_subdir benchmarks/baseline \
          --train_epochs 100 --patience 5 \
          --seed $seed

        # VG-iT
        echo "[2/2] VG-iT G32 Fixed..."
        $py_path -u run.py \
          --is_training 1 \
          --root_path ./dataset/Solar/ \
          --data_path solar_AL.txt \
          --model_id "solar_vgit_g32_h${pl}_s${seed}" \
          --model VG_iTransformer \
          --data Solar \
          --features M \
          --seq_len 96 \
          --pred_len $pl \
          --e_layers 2 \
          --enc_in 137 --dec_in 137 --c_out 137 \
          --d_model 512 --d_ff 512 \
          --learning_rate 0.0005 \
          --output_subdir benchmarks/vgit_g32 \
          --train_epochs 100 --patience 5 \
          --num_groups 32 --pooling mean \
          --seed $seed
    done
done

echo "Comprehensive Benchmark Completed. Results saved in test_results/summary.csv"
