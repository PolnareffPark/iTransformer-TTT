export CUDA_VISIBLE_DEVICES=0

/home/himchan/miniconda3/envs/CTSF/bin/python run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model iTransformer \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --train_epochs 1 \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --use_gpu True \
  --use_ttt \
  --ttt_lr 0.001 \
  --ttt_steps 1
