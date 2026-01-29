#!/bin/bash
PYTHON_PATH="/home/himchan/miniconda3/envs/CTSF/bin/python"

# 1. Baseline iTransformer (Full Training)
echo "Running Baseline iTransformer Full Training on Traffic..."
$PYTHON_PATH -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id Traffic_862_96_Baseline_Full \
  --model iTransformer \
  --data custom \
  --features M \
  --target "OT" \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 512 \
  --e_layers 2 \
  --train_epochs 100 \
  --patience 5 \
  --batch_size 32 \
  --use_gpu True \
  --gpu 0 \
  --des 'BaselineFull'

# 2. VG-iTransformer (Full Training, G=16)
echo "Running VG-iTransformer Full Training on Traffic..."
$PYTHON_PATH -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id Traffic_862_96_VGIT_Full \
  --model VG_iTransformer \
  --data custom \
  --features M \
  --target "OT" \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 512 \
  --e_layers 2 \
  --num_groups 16 \
  --train_epochs 100 \
  --patience 5 \
  --batch_size 32 \
  --use_gpu True \
  --gpu 0 \
  --des 'VGITFull'
