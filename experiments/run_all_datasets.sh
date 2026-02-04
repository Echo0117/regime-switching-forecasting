#!/usr/bin/env bash
# Run experiments/test_agaci.py for all datasets and save logs.
# Usage:
#   ./experiments/run_all_datasets.sh [extra args forwarded to test_agaci.py]
# Examples:
#   ./experiments/run_all_datasets.sh --aci_train_size 20 --alpha 0.1
#   ./experiments/run_all_datasets.sh --data-dir Deep_Switching_State_Space_Model/data/Toy_exp2_V1_0.5_V2_1.0 --use-oracle-switches
#   ./experiments/run_all_datasets.sh --n-models 10  # Train 10 models and average predictions

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
  # Extract n_models from EXTRA_ARGS if present, for log filename
  N_MODELS=""
  for i in "${!EXTRA_ARGS[@]}"; do
    if [[ "${EXTRA_ARGS[$i]}" == "--n-models" ]] && [[ $((i+1)) -lt ${#EXTRA_ARGS[@]} ]]; then
      N_MODELS="_n${EXTRA_ARGS[$((i+1))]}"
      break
    fi
  done

  LOGFILE="$LOGDIR/test_agaci_${d}${N_MODELS}.log"

  echo "============================================================"
  echo "Running test_agaci.py --problem $d ${EXTRA_ARGS[*]}"
  echo "Log -> $LOGFILE"
  echo "============================================================"
  "$PY" "$ROOT/experiments/test_agaci.py" --problem "$d" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" 2>&1 | tee "$LOGFILE"
  echo "Finished $d"
  echo
done

echo "All done. Logs saved in: $LOGDIR"