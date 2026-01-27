#!/bin/bash

# Configuration
PYTHON_PATH="/home/himchan/miniconda3/envs/CTSF/bin/python"
MODEL_ID="ETTh1_96_96"
SEQ_LEN=96
PRED_LEN=96
BATCH_SIZE=32
EPOCHS=100
TTT_LR=0.001
TTT_STEPS=3

echo "========================================================"
echo " [Step 1] Training Standard iTransformer (Baseline)"
echo " Epochs: $EPOCHS, Batch: $BATCH_SIZE"
echo " This sets up the base model for comparison."
echo "========================================================"

$PYTHON_PATH -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id $MODEL_ID \
  --model iTransformer \
  --data ETTh1 \
  --features M \
  --seq_len $SEQ_LEN \
  --label_len 48 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Benchmark' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --train_epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --learning_rate 0.0005 \
  --use_gpu True \
  --gpu 0 \
  --patience 3

echo ""
echo "========================================================"
echo " [Step 2] Evaluating iTTT (Structural Adaptation)"
echo " Applying TTT to the PRE-TRAINED model from Step 1."
echo " TTT LR: $TTT_LR, Steps: $TTT_STEPS, Mode: norm_only"
echo "========================================================"

$PYTHON_PATH -u run.py \
  --is_training 0 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id $MODEL_ID \
  --model iTransformer \
  --data ETTh1 \
  --features M \
  --seq_len $SEQ_LEN \
  --label_len 48 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Benchmark' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --train_epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --use_gpu True \
  --gpu 0 \
  --use_ttt \
  --ttt_lr $TTT_LR \
  --ttt_steps $TTT_STEPS

echo ""
echo "========================================================"
echo " Benchmark Completed."
echo " Check test_results/summary.csv for final comparison."
echo " (Look for the last two entries with des='Benchmark')"
echo "========================================================"
