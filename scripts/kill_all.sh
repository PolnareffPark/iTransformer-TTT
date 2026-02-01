#!/bin/bash
echo "Stopping all benchmark processes..."
pkill -9 -f "run.py"
pkill -9 -f "master_run_all.sh"
pkill -9 -f "full_benchmark_vgit.sh"
pkill -9 -f "ablation_phase1_core.sh"
pkill -9 -f "ablation_phase2_bundled.sh"
pkill -9 -f "expansion_phase3_superiority.sh"
echo "All processes targeted for termination."
ps aux | grep "run.py" | grep -v grep
