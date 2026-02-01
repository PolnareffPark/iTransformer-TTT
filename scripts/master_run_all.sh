#!/bin/bash

# ==========================================================
# [Master Script] Sequential Execution of All Experiments
# Target: Single GPU (Device 0)
# Execution Order:
#   1. Full Benchmark (Traffic, ECL, Solar)
#   2. Ablation Phase 1 (Core Mechanisms)
#   3. Ablation Phase 2 (Noise & Robustness)
#   4. Ablation Phase 3 (Superiority & Expansion)
#
#   nohup bash scripts/master_run_all.sh > master_execution.log 2>&1 &
#   tail -f master_execution.log  # 진행 상황 모니터링
# ==========================================================

# Log file for master execution status
LOG_FILE="master_execution.log"

echo "==========================================================" >> $LOG_FILE
echo "Master Execution Started at $(date)" >> $LOG_FILE
echo "==========================================================" >> $LOG_FILE

export CUDA_VISIBLE_DEVICES=0

# Define Python Interpreter properly for all subprocesses
export PYTHON_EXEC="/home/himchan/miniconda3/envs/CTSF/bin/python"

# Progress tracking
# Total Runs Calculation:
# - Full Benchmark: 3 datasets * 4 pred_lens * 3 seeds * 2 models = 72
# - Phase 1: 3 seeds * 4 pred_lens * 7 models = 84
# - Phase 2: 3 seeds * (3 noise * 2 models + 9 stress * 2 models) = 3s * (6 + 18) = 72
# - Phase 3: 3 seeds * (1 base + 3 vgit + 1 base_cntr) = 15
# GRAND TOTAL = 72 + 84 + 72 + 15 = 243
GRAND_TOTAL=243

source scripts/utils_progress.sh
init_progress $GRAND_TOTAL

# Function to run script and check status
run_stage() {
    script_name=$1
    CURRENT_STAGE=$((CURRENT_STAGE + 1))
    
    echo "==========================================================" | tee -a $LOG_FILE
    echo ">> [Progress: ${CURRENT_STAGE}/${TOTAL_STAGES}] Starting Stage: $script_name" | tee -a $LOG_FILE
    echo ">> Timestamp: $(date)" | tee -a $LOG_FILE
    echo "==========================================================" | tee -a $LOG_FILE
    
    # Execute script
    bash scripts/$script_name
    
    if [ $? -eq 0 ]; then
        echo ">> [${CURRENT_STAGE}/${TOTAL_STAGES}] Stage Completed Successfully: $script_name at $(date)" | tee -a $LOG_FILE
    else
        echo "!! [${CURRENT_STAGE}/${TOTAL_STAGES}] Stage FAILED: $script_name at $(date)" | tee -a $LOG_FILE
    fi
    echo "" >> $LOG_FILE
}

# 1. Full Benchmark (Traffic, ECL, Solar)
run_stage "full_benchmark_vgit.sh"

# 2. Ablation Phase 1 (Core Mechanisms)
run_stage "ablation_phase1_core.sh"

# 3. Ablation Phase 2 (Noise & Robustness)
run_stage "ablation_phase2_bundled.sh"

# 4. Ablation Phase 3 (Superiority & Expansion)
run_stage "expansion_phase3_superiority.sh"

echo "==========================================================" >> $LOG_FILE
echo ">> All Stages Finished at $(date)" >> $LOG_FILE
echo "==========================================================" >> $LOG_FILE
