#!/bin/bash

# iTransformer 논문 표준 하이퍼파라미터 기반 검증 스크립트 (Phase 15)
# 목표: 고부하 설정(e_layers=4, d_model=512)에서 VG-iT의 효율성 초격차 증명

export CUDA_VISIBLE_DEVICES=0

# Conda 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CTSF

# 결과 파일 초기화
RESULT_FILE="./test_results/paper_validation_summary.csv"
mkdir -p ./test_results

# 공통 파라미터 (논문 Traffic 설정 참조)
SEQ_LEN=96
PRED_LEN=96
E_LAYERS=4
D_MODEL=512
D_FF=512
BATCH_SIZE=16
L_RATE=0.001

echo "N=862 (Traffic) Paper-Standard Validation Start..."

# 1. iTransformer (Baseline)
echo "Running iTransformer (Baseline) with Paper Params..."
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_paper_baseline \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len $SEQ_LEN \
  --pred_len $PRED_LEN \
  --e_layers $E_LAYERS \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model $D_MODEL \
  --d_ff $D_FF \
  --batch_size $BATCH_SIZE \
  --learning_rate $L_RATE \
  --train_epochs 1 \
  --itr 1

# 2. VG-iTransformer (Ours - Hybrid)
echo "Running VG-iTransformer (Hybrid) with Paper Params..."
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_paper_vgit \
  --model VG_iTransformer \
  --data custom \
  --features M \
  --seq_len $SEQ_LEN \
  --pred_len $PRED_LEN \
  --e_layers $E_LAYERS \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model $D_MODEL \
  --d_ff $D_FF \
  --batch_size $BATCH_SIZE \
  --learning_rate $L_RATE \
  --train_epochs 1 \
  --num_groups 32 \
  --itr 1

echo "Paper-Standard Validation Finished."
echo "Results saved in ./test_results/summary.csv"
