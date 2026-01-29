#!/bin/bash
PYTHON_PATH="/home/himchan/miniconda3/envs/CTSF/bin/python"
BATCH_SIZE=32
EPOCHS=10
TTT_LR=0.001
TTT_STEPS=3

run_benchmark() {
    DATA_NAME=$1
    DATA_PATH=$2
    ID_TAG=$3
    SEQ_LEN=$4
    PRED_LEN=$5
    ENC_IN=$6
    TARGET=${7:-'OT'}

    MODEL_ID="${ID_TAG}_${SEQ_LEN}_${PRED_LEN}"
    
    echo "========================================================"
    echo " START BENCHMARK: $ID_TAG"
    echo " Data: $DATA_NAME, Path: $DATA_PATH, Target: $TARGET"
    echo " Seq: $SEQ_LEN, Pred: $PRED_LEN, Channels: $ENC_IN"
    echo "========================================================"

    # 1. Train Baseline
    echo " >> Step 1: Training Baseline..."
    $PYTHON_PATH -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path $DATA_PATH \
      --model_id $MODEL_ID \
      --model iTransformer \
      --data $DATA_NAME \
      --features M \
      --target $TARGET \
      --seq_len $SEQ_LEN \
      --label_len 48 \
      --pred_len $PRED_LEN \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in $ENC_IN \
      --dec_in $ENC_IN \
      --c_out $ENC_IN \
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

    # 2. Test TTT
    echo " >> Step 2: Testing TTT (Structural V2)..."
    $PYTHON_PATH -u run.py \
      --is_training 0 \
      --root_path ./dataset/ \
      --data_path $DATA_PATH \
      --model_id $MODEL_ID \
      --model iTransformer \
      --data $DATA_NAME \
      --features M \
      --target $TARGET \
      --seq_len $SEQ_LEN \
      --label_len 48 \
      --pred_len $PRED_LEN \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in $ENC_IN \
      --dec_in $ENC_IN \
      --c_out $ENC_IN \
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

    echo " >> Finished $ID_TAG"
    echo ""
}

# Final Kill-or-Cure Validation (Weather & Traffic)

# Weather (21 cols)
# enc_in=21 (20 features + 1 OT)
run_benchmark "custom" "weather.csv" "Weather" 96 96 21 "OT"

# Traffic (862 cols)
# enc_in=862 (861 sensors + 1 OT)
run_benchmark "custom" "traffic.csv" "Traffic" 96 96 862 "OT"
