#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

PYTHON="/c/Users/ouyan/AppData/Local/Programs/Python/Python312/python"

mkdir -p logs

echo "=========================================="
echo "  Levels 1 + 2  (sequential)"
echo "=========================================="

echo ""
echo "--- Level 1 ---"
$PYTHON regression/regression_all.py --level 1 2>&1 | tee logs/level1.log

echo ""
echo "--- Level 2 ---"
$PYTHON regression/regression_all.py --level 2 2>&1 | tee logs/level2.log

echo ""
echo "Done. Plots → plots/  |  Results → results/"
