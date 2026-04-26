#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

PYTHON="/c/Users/ouyan/AppData/Local/Programs/Python/Python312/python"

echo "=========================================="
echo "  Japan Inflation Assemblage – Run All"
echo "=========================================="

echo ""
echo "--- Level 1 ---"
$PYTHON regression/regression_all.py --level 1

echo ""
echo "--- Level 2 ---"
$PYTHON regression/regression_all.py --level 2

echo ""
echo "--- Level 3  [slow: ~700 components] ---"
$PYTHON regression/regression_all.py --level 3

echo ""
echo "All done. Plots saved to plots/"
