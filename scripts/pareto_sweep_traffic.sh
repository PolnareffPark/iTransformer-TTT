#!/bin/bash

# Traffic Dataset Pareto Sweep Script (Phase 16)
# Goal: Collect data points for Pareto Frontier analysis

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CTSF

export CUDA_VISIBLE_DEVICES=0

# Clean old summary
rm -f test_results/summary.csv

# Common Params (Paper Standard)
ROOT_PATH="./dataset/"
DATA_PATH="traffic.csv"
SEQ_LEN=96
PRED_LEN=96
E_LAYERS=4
D_MODEL=512
D_FF=512
BATCH_SIZE=16
EPOCHS=1 # Fast sweep for VRAM/Speed profile + 1-epoch MSE proxy

echo "Starting Pareto Sweep on Traffic (N=862)..."

# 1. Baseline (Full Attention)
echo "Running Baseline (iTransformer)..."
python -u run.py --is_training 1 --root_path $ROOT_PATH --data_path $DATA_PATH \
  --model_id traffic_pareto_baseline --model iTransformer --data custom --features M \
  --seq_len $SEQ_LEN --pred_len $PRED_LEN --e_layers $E_LAYERS --enc_in 862 --dec_in 862 --c_out 862 \
  --d_model $D_MODEL --d_ff $D_FF --batch_size $BATCH_SIZE --train_epochs $EPOCHS --itr 1

# 2. VG-iT Variations (num_groups + pooling)
# 2-1. G=32, Mean (Existing)
echo "Running VG-iT (G=32, Mean)..."
python -u run.py --is_training 1 --root_path $ROOT_PATH --data_path $DATA_PATH \
  --model_id traffic_pareto_vgit_g32_mean --model VG_iTransformer --data custom --features M \
  --seq_len $SEQ_LEN --pred_len $PRED_LEN --e_layers $E_LAYERS --enc_in 862 --dec_in 862 --c_out 862 \
  --d_model $D_MODEL --d_ff $D_FF --batch_size $BATCH_SIZE --train_epochs $EPOCHS --num_groups 32 --pooling mean --itr 1

# 2-2. G=32, Statistical (SMTB)
echo "Running VG-iT (G=32, Statistical)..."
python -u run.py --is_training 1 --root_path $ROOT_PATH --data_path $DATA_PATH \
  --model_id traffic_pareto_vgit_g32_stat --model VG_iTransformer --data custom --features M \
  --seq_len $SEQ_LEN --pred_len $PRED_LEN --e_layers $E_LAYERS --enc_in 862 --dec_in 862 --c_out 862 \
  --d_model $D_MODEL --d_ff $D_FF --batch_size $BATCH_SIZE --train_epochs $EPOCHS --num_groups 32 --pooling statistical --itr 1

# 2-3. G=32, Learnable (Proposed)
echo "Running VG-iT (G=32, Learnable)..."
python -u run.py --is_training 1 --root_path $ROOT_PATH --data_path $DATA_PATH \
  --model_id traffic_pareto_vgit_g32_learn --model VG_iTransformer --data custom --features M \
  --seq_len $SEQ_LEN --pred_len $PRED_LEN --e_layers $E_LAYERS --enc_in 862 --dec_in 862 --c_out 862 \
  --d_model $D_MODEL --d_ff $D_FF --batch_size $BATCH_SIZE --train_epochs $EPOCHS --num_groups 32 --pooling learnable --itr 1

# 2-4. G=64, Learnable (Higher fidelity)
echo "Running VG-iT (G=64, Learnable)..."
python -u run.py --is_training 1 --root_path $ROOT_PATH --data_path $DATA_PATH \
  --model_id traffic_pareto_vgit_g64_learn --model VG_iTransformer --data custom --features M \
  --seq_len $SEQ_LEN --pred_len $PRED_LEN --e_layers $E_LAYERS --enc_in 862 --dec_in 862 --c_out 862 \
  --d_model $D_MODEL --d_ff $D_FF --batch_size $BATCH_SIZE --train_epochs $EPOCHS --num_groups 64 --pooling learnable --itr 1

# 3. Plotting
echo "Generating Pareto Frontier Plots..."
python scripts/plot_pareto_frontier.py

echo "Pareto Sweep Completed."
