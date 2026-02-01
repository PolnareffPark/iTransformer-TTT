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

# Ensure GPU 0 is used
export CUDA_VISIBLE_DEVICES=0

# Progress tracking
TOTAL_STAGES=4
CURRENT_STAGE=0

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
