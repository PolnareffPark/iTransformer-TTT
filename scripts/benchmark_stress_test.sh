#!/bin/bash

# Extreme Scalability Benchmark for VG-iT (Phase 14)
# Measuring: VRAM, FLOPs, Params, Speed (with Warm-up & Sync)

export CUDA_VISIBLE_DEVICES=0
PYTHON_PATH=/home/himchan/miniconda3/envs/CTSF/bin/python

# Scenarios: Increase N from 500 to 5000
N_VARS=(500 1000 2000 3000 4000 5000)
SEQ_LEN=96
PRED_LEN=96
BATCH_SIZE=16
NUM_GROUPS=16

SUMMARY_FILE="./test_results/efficiency_benchmark.csv"
echo "timestamp,model,N,mse,mae,flops_G,params_M,vram_GB" > $SUMMARY_FILE

for N in "${N_VARS[@]}"; do
    echo "=========================================================="
    echo "Stress Test: N = ${N}"
    echo "=========================================================="

    # 1. Baseline iTransformer
    echo "Testing Baseline iTransformer with N=${N}..."
    $PYTHON_PATH -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path stress_${N}.csv \
      --model_id Baseline_N${N} \
      --model iTransformer \
      --data stress \
      --features M \
      --seq_len $SEQ_LEN \
      --pred_len $PRED_LEN \
      --enc_in $N \
      --dec_in $N \
      --c_out $N \
      --train_epochs 1 \
      --batch_size $BATCH_SIZE \
      --gpu 0

    # 2. VG-iTransformer (Hybrid)
    echo "Testing VG-iTransformer (Hybrid) with N=${N}..."
    $PYTHON_PATH -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path stress_${N}.csv \
      --model_id VGIT_N${N} \
      --model VG_iTransformer \
      --data stress \
      --features M \
      --seq_len $SEQ_LEN \
      --pred_len $PRED_LEN \
      --enc_in $N \
      --dec_in $N \
      --c_out $N \
      --num_groups $NUM_GROUPS \
      --train_epochs 1 \
      --batch_size $BATCH_SIZE \
      --gpu 0

done

echo "Extreme scalability benchmark completed. Check ${SUMMARY_FILE}"
