#!/usr/bin/env bash
# Run experiments/test_agaci.py for all datasets and save logs.
# Usage:
#   ./experiments/run_all_datasets.sh [extra args forwarded to test_agaci.py]
# Example:
#   ./experiments/run_all_datasets.sh --aci_train_size 20 --alpha 0.1

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

# Datasets to run (matches test_agaci.py choices)
DATASETS=(Toy)
# DATASETS=(Toy Lorenz Sleep Unemployment Hangzhou Seattle Pacific Electricity)

# DATASETS=(Lorenz Hangzhou Seattle Pacific Electricity)

# Forward any extra args to test_agaci.py
EXTRA_ARGS=("$@")

for d in "${DATASETS[@]}"; do
  echo "============================================================"
  echo "Running test_agaci.py --problem $d"
  echo "Log -> $LOGDIR/test_agaci_${d}.log"
  echo "============================================================"
  "$PY" "$ROOT/experiments/test_agaci.py" --problem "$d" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" 2>&1 | tee "$LOGDIR/test_agaci_${d}.log"
  echo "Finished $d"
  echo
done

echo "All done. Logs saved in: $LOGDIR"