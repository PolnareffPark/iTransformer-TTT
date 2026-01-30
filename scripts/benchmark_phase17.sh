#!/bin/bash

# Phase 17: Dynamic vs Fixed Grouping Benchmark
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CTSF

export CUDA_VISIBLE_DEVICES=0
rm -f test_results/summary.csv

# Standard Configs
ROOT_PATH="./dataset/"
DATA_PATH="traffic.csv"
SEQ_LEN=96
PRED_LEN=96
E_LAYERS=4
D_MODEL=512
D_FF=512
BATCH_SIZE=16
EPOCHS=1 # Benchmark profile

echo "Starting Dynamic Grouping Validation..."

# 1. Baseline (Fixed Grouping G32)
echo "Running Fixed Grouping (G=32)..."
python -u run.py --is_training 1 --root_path $ROOT_PATH --data_path $DATA_PATH \
  --model_id traffic_fixed_g32 --model VG_iTransformer --data custom --features M \
  --seq_len $SEQ_LEN --pred_len $PRED_LEN --e_layers $E_LAYERS --enc_in 862 --dec_in 862 --c_out 862 \
  --d_model $D_MODEL --d_ff $D_FF --batch_size $BATCH_SIZE --train_epochs $EPOCHS --num_groups 32 --pooling mean --itr 1

# 2. PROPOSED (Dynamic Grouping G32)
echo "Running Dynamic Grouping (G=32)..."
python -u run.py --is_training 1 --root_path $ROOT_PATH --data_path $DATA_PATH \
  --model_id traffic_dynamic_g32 --model VG_iTransformer --data custom --features M \
  --seq_len $SEQ_LEN --pred_len $PRED_LEN --e_layers $E_LAYERS --enc_in 862 --dec_in 862 --c_out 862 \
  --d_model $D_MODEL --d_ff $D_FF --batch_size $BATCH_SIZE --train_epochs $EPOCHS --num_groups 32 --pooling dynamic --itr 1

# 3. Baseline (Fixed Grouping G128)
echo "Running Fixed Grouping (G=128)..."
python -u run.py --is_training 1 --root_path $ROOT_PATH --data_path $DATA_PATH \
  --model_id traffic_fixed_g128 --model VG_iTransformer --data custom --features M \
  --seq_len $SEQ_LEN --pred_len $PRED_LEN --e_layers $E_LAYERS --enc_in 862 --dec_in 862 --c_out 862 \
  --d_model $D_MODEL --d_ff $D_FF --batch_size $BATCH_SIZE --train_epochs $EPOCHS --num_groups 128 --pooling mean --itr 1

# 4. PROPOSED (Dynamic Grouping G128)
echo "Running Dynamic Grouping (G=128)..."
python -u run.py --is_training 1 --root_path $ROOT_PATH --data_path $DATA_PATH \
  --model_id traffic_dynamic_g128 --model VG_iTransformer --data custom --features M \
  --seq_len $SEQ_LEN --pred_len $PRED_LEN --e_layers $E_LAYERS --enc_in 862 --dec_in 862 --c_out 862 \
  --d_model $D_MODEL --d_ff $D_FF --batch_size $BATCH_SIZE --train_epochs $EPOCHS --num_groups 128 --pooling dynamic --itr 1

# Plotting
python scripts/plot_pareto_frontier.py

echo "Benchmark Completed."
