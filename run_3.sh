#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

PYTHON="/c/Users/ouyan/AppData/Local/Programs/Python/Python312/python"

mkdir -p logs

echo "=========================================="
echo "  Level 3  (max resources, step=3)"
echo "  3 models run in parallel, each using"
echo "  all CPU cores for inner OOS loop"
echo "=========================================="

# OMP/MKL thread count: let numpy/scipy use all cores per worker
export OMP_NUM_THREADS=$(python -c "import os; print(os.cpu_count())")
export MKL_NUM_THREADS=$OMP_NUM_THREADS

$PYTHON regression/regression_all.py --level 3 --step 3 2>&1 | tee logs/level3.log

echo ""
echo "Done. Plots → plots/level3/  |  Results → results/level3_scorecard.csv"
