#!/bin/bash

# ==========================================================
# [Utils] Global Progress Tracking & Robust Execution
# ==========================================================

PROGRESS_FILE="master_progress.cnt"
TOTAL_RUNS=243 # Default, can be overridden

# Default Python Executable
if [ -z "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python"
fi

init_progress() {
    local total=$1
    echo "0" > $PROGRESS_FILE
    # Use the passed total or default
    if [ ! -z "$total" ]; then
        TOTAL_RUNS=$total
    fi
    echo "[Progress Utils] Initialized. Total Runs: $TOTAL_RUNS"
}

run_python() {
    # 1. Update Global Counter
    if [ ! -f $PROGRESS_FILE ]; then
        echo "0" > $PROGRESS_FILE
    fi
    
    # Read current count safely
    local current=$(cat $PROGRESS_FILE)
    current=$((current + 1))
    echo "$current" > $PROGRESS_FILE
    
    # 2. Log Progress
    echo "----------------------------------------------------------"
    echo ">> [Global: ${current}/${TOTAL_RUNS}] Executing..."
    echo ">> Cmd: $PYTHON_EXEC -u run.py $@"
    echo "----------------------------------------------------------"
    
    # 3. Execute with Error Handling (Continue on Error)
    $PYTHON_EXEC -u run.py "$@"
    local status=$?
    
    if [ $status -ne 0 ]; then
        echo "!!!! [Global: ${current}/${TOTAL_RUNS}] FAILED (Exit Code: $status) !!!!"
        echo "!!!! Note: If this is a Stress Test, OOM is EXPECTED. Continuing... !!!!"
    else
        echo ">> [Global: ${current}/${TOTAL_RUNS}] COMPLETED Successfully."
    fi
    # 4. Do NOT exit, ensuring the benchmark suite continues even if OOM occurs.
}
