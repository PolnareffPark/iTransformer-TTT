#!/bin/bash

# Multi-Dataset Benchmark for VG-iT Hybrid (Phase 13) - Fixed for Solar Target
# Target: Electricity (321 vars), Solar-Energy (137 vars)

export CUDA_VISIBLE_DEVICES=0
PYTHON_PATH=/home/himchan/miniconda3/envs/CTSF/bin/python

# Configs
SEQ_LEN=96
PRED_LEN=96
BATCH_SIZE=32
EPOCHS=100
PATIENCE=5
NUM_GROUPS=16

# Datasets to test
DATASETS=("electricity.csv" "solar_AL.csv")
VARS=(321 137)
NAMES=("Electricity" "Solar")
TARGETS=("OT" "136") # Solar dataset target is the last column index

for i in "${!DATASETS[@]}"; do
    DATA_PATH=${DATASETS[$i]}
    N_VARS=${VARS[$i]}
    NAME=${NAMES[$i]}
    TARGET=${TARGETS[$i]}

    echo "=========================================================="
    echo "Running Benchmark on ${NAME} (${N_VARS} variables) with target ${TARGET}..."
    echo "=========================================================="

    # 1. Baseline iTransformer
    echo "Running Baseline iTransformer..."
    $PYTHON_PATH -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path $DATA_PATH \
      --model_id ${NAME}_Baseline \
      --model iTransformer \
      --data custom \
      --features M \
      --target $TARGET \
      --seq_len $SEQ_LEN \
      --pred_len $PRED_LEN \
      --enc_in $N_VARS \
      --dec_in $N_VARS \
      --c_out $N_VARS \
      --des 'BaselineBenchmark' \
      --train_epochs $EPOCHS \
      --patience $PATIENCE \
      --batch_size $BATCH_SIZE \
      --learning_rate 0.0001 \
      --gpu 0

    # 2. VG-iTransformer (Hybrid)
    echo "Running VG-iTransformer (Hybrid)..."
    $PYTHON_PATH -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path $DATA_PATH \
      --model_id ${NAME}_VGIT \
      --model VG_iTransformer \
      --data custom \
      --features M \
      --target $TARGET \
      --seq_len $SEQ_LEN \
      --pred_len $PRED_LEN \
      --enc_in $N_VARS \
      --dec_in $N_VARS \
      --c_out $N_VARS \
      --num_groups $NUM_GROUPS \
      --des 'VGITBenchmark' \
      --train_epochs $EPOCHS \
      --patience $PATIENCE \
      --batch_size $BATCH_SIZE \
      --learning_rate 0.0001 \
      --gpu 0

done

echo "Multi-dataset benchmark completed."
