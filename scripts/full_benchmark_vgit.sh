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

# Full Comprehensive Benchmark for VG-iT G32 Fixed (👑 Pareto Best)
# Datasets: Traffic (N=862), Electricity (N=321)
# Prediction Lengths: 96, 192, 336, 720
# Standard: d_ff = 2048

export CUDA_VISIBLE_DEVICES=0
py_path=/home/himchan/miniconda3/envs/CTSF/bin/python

datasets=("traffic.csv" "electricity.csv")
pred_lens=(96 192 336 720)
seeds=(2021 2022 2023 2024 2025)

for data_path in "${datasets[@]}"; do
    # Determine model parameters based on dataset
    if [ "$data_path" == "traffic.csv" ]; then
        enc_in=862
    else
        enc_in=321
    fi

    for pl in "${pred_lens[@]}"; do
        for seed in "${seeds[@]}"; do
            echo "=========================================================="
            echo "Running Full Benchmark: Data=$data_path, Pred_Len=$pl, Seed=$seed"
            echo "=========================================================="

            # 1. iTransformer Baseline (Standard df512)
            echo "[1/2] Standard Baseline..."
            $py_path -u run.py \
              --is_training 1 \
              --root_path ./dataset/ \
              --data_path $data_path \
              --model_id "${data_path%.*}_baseline_h${pl}_s${seed}" \
              --model iTransformer \
              --data custom \
              --features M \
              --seq_len 96 \
              --label_len 48 \
              --pred_len $pl \
              --e_layers 4 \
              --enc_in $enc_in \
              --dec_in $enc_in \
              --c_out $enc_in \
              --d_model 512 \
              --d_ff 512 \
              --batch_size 16 \
              --learning_rate 0.001 \
              --output_subdir benchmarks/baseline \
              --train_epochs 100 \
              --patience 5 \
              --seed $seed

            # 2. VG-iT G32 Fixed (Standard df512)
            echo "[2/2] VG-iT G32 Fixed..."
            $py_path -u run.py \
              --is_training 1 \
              --root_path ./dataset/ \
              --data_path $data_path \
              --model_id "${data_path%.*}_vgit_g32_h${pl}_s${seed}" \
              --model VG_iTransformer \
              --data custom \
              --features M \
              --seq_len 96 \
              --label_len 48 \
              --pred_len $pl \
              --e_layers 4 \
              --enc_in $enc_in \
              --dec_in $enc_in \
              --c_out $enc_in \
              --d_model 512 \
              --d_ff 512 \
              --batch_size 16 \
              --learning_rate 0.001 \
              --output_subdir benchmarks/vgit_g32 \
              --train_epochs 100 \
              --patience 5 \
              --num_groups 32 \
              --pooling mean \
              --seed $seed
        done
    done
done

echo "Comprehensive Benchmark Completed. results saved in test_results/summary.csv"
