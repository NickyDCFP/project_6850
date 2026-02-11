#!/usr/bin/env bash
# Launcher to run semantic analysis in a separate terminal/process.
# Usage: ./run_semantic_analysis.sh smoke
#        ./run_semantic_analysis.sh full

set -euo pipefail
MODE=${1:-smoke}

case "$MODE" in
  smoke)
    echo "Running smoke test (400 samples)"
    python3 semantic_convergence_refactor.py --sample-size 400 --model all-MiniLM-L6-v2 --n-clusters 10
    ;;
  full)
    echo "Running full analysis (no sampling) - make sure you have embedding cache and expect long runtime"
    python3 semantic_convergence_refactor.py --no-sample --model all-MiniLM-L6-v2 --n-clusters 50
    ;;
  *)
    echo "Unknown mode. Use 'smoke' or 'full'"; exit 2
    ;;
esac
