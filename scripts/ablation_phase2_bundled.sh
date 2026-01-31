# ==========================================================
# [Phase 2] Bundled Validation (Noise Robustness & OOM Match)
#   - GPU: 1 (Isolated from main benchmark)
#   - Ablation 2: Gaussian Denoising Effect
#   - Expansion 8: OOM Death Match (Large Scale Stress Test)
#
# [Usage]
#   Run in background:
#     nohup bash scripts/ablation_phase2_bundled.sh > ablation_phase2_bundled.log 2>&1 &
#
#   Check log:
#     tail -f ablation_phase2_bundled.log
#
# [Termination]
#   Stop ONLY this session (Safe for main benchmark):
#     pkill -f ablation_phase2_bundled.sh
#     pkill -f "run.py.*summary_ablation_phase2_bundled.csv"
# ==========================================================

# Default to GPU 1, but allow override from command line
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

model_name=VG_iTransformer
summary_file=summary_ablation_phase2_bundled.csv

# Standard config (G=32, Mean Pooling)
# Aligned with Phase 1 (Author default d_ff=512)
common_args="--is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 3 \
  --pooling mean \
  --num_groups 32 \
  --summary_file $summary_file"

for seed in 2021 2022 2023
do
    echo "=== [Seed: $seed] Phase 2.1 Noise Robustness Study ==="
    for noise in 0.1 0.2 0.5
    do
        echo ">>> Testing Noise STD: $noise <<<"
        # VG-iT (Ours) - G32 Fixed Mean
        python -u run.py $common_args --model_id traffic_vgit_n${noise}_s${seed} --noise_std $noise --seed $seed
        
        # iTransformer (Baseline) - for direct comparison in noisy env
        python -u run.py $common_args --model iTransformer --model_id traffic_base_n${noise}_s${seed} --noise_std $noise --seed $seed
    done

    echo "=== [Seed: $seed] Phase 2.2 OOM Death Match (Stress Test) ==="
    # OOM stress tests: finding the limit where iTransformer dies but VG-iT survives.
    # Range expanded up to N=20,000 for aggressive benchmarking.
    for n_vars in 1000 2000 5000 8000 10000 15000 20000 25000 30000
    do
        echo ">>> Stress Test: N=$n_vars <<<"
        # VG-iT (G=32 - Best Pareto, proving its scalability)
        python -u run.py $common_args --model_id stress_vgit_n${n_vars}_s${seed} --enc_in $n_vars --dec_in $n_vars --c_out $n_vars --num_groups 32 --train_epochs 1 --batch_size 4 --seed $seed
        
        # Baseline (iTransformer) -> Expected to OOM between 5000-10000 on 3090
        python -u run.py $common_args --model iTransformer --model_id stress_base_n${n_vars}_s${seed} --enc_in $n_vars --dec_in $n_vars --c_out $n_vars --train_epochs 1 --batch_size 4 --seed $seed
    done
done

echo "Phase 2 Bundled Validation finished. Results: test_results/$summary_file"
