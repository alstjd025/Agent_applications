#!/bin/bash
# Full motivation experiment pipeline
# Usage: bash run_all.sh

set -e

VENV_PY=/home/nxc/mskim/agent/Agent_applications/agent_sglang_concurrent/.venv/bin/python
SCRIPT=/home/nxc/mskim/agent/Agent_applications/agent_motivation_experiment/run_experiment.py
ANALYSIS=/home/nxc/mskim/agent/Agent_applications/agent_motivation_experiment/analyze_motivation.py
RESULTS_DIR=/home/nxc/mskim/agent/Agent_applications/agent_motivation_experiment/results
FIGURES_DIR=/home/nxc/mskim/agent/Agent_applications/agent_motivation_experiment/figures

echo "=== Step 1: Baseline (concurrency=1, 300 jobs) ==="
$VENV_PY $SCRIPT --mode baseline --log-level info --end-index 300

echo ""
echo "=== Step 2: Concurrency Sweep (1,4,8,16,32,64) ==="
$VENV_PY $SCRIPT --mode sweep --log-level info --end-index 300 \
  --concurrency-list 1,4,8,16,32,64

echo ""
echo "=== Step 3: Rate Sweep (60,120,240,360 RPM, replay=5 for sustained load) ==="
$VENV_PY $SCRIPT --mode rate-sweep --log-level info \
  --end-index 300 --replay-count 5 \
  --rate-list 60,120,240,360

echo ""
echo "=== Step 4: Analysis ==="
$VENV_PY $ANALYSIS \
  --results-dir $RESULTS_DIR \
  --output-dir $FIGURES_DIR

echo ""
echo "=== Done! Check figures/ for results ==="
