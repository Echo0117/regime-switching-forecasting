#!/usr/bin/env bash
set -euo pipefail

# --------------------------
# Config (edit as you like)
# --------------------------

# Device: cuda | mps | cpu
DEVICE=${1:-cuda}

# Mixed precision (harmless on CPU/MPS; your Python ignores if not supported)
AMP_FLAG="--amp"

# Path to the real S4 repo root so that models/s4/s4d.py is importable.
# Leave empty ("") to use the Conv1d fallback in our code.
# S4_PATH="${S4_PATH:-}"
S4_PATH="./s4"

# Output CSV (single combined table for *all* datasets)
OUTCSV="experiments/results_all_${DEVICE}.csv"

# Common sweep settings
LAGS="${LAGS:-48}"
ALPHA="${ALPHA:-0.1}"
GAMMA="${GAMMA:-0.01}"
TRAIN_SIZE="${TRAIN_SIZE:-200}"

# Models to run (edit to add/remove)
MODELS=("ds3m")
# MODELS=("S4" "CPD" "MCDropoutGRU" "GPTorchSparse")

# Interval methods to run
# METHODS=("ACI")
METHODS=("ACI" "AgACI" "Naive")
# Datasets to loop
PROBLEMS=(
  "Toy"
  "Sleep"
  "Unemployment"
  "Lorenz"
  # "Hangzhou"
  # "Seattle"
  # "Pacific"
  
  # "Electricity"
)

# --------------------------
# Derived / helper flags
# --------------------------
declare -a S4_PATH_ARG=()
if [[ -n "${S4_PATH}" ]]; then
  S4_PATH_ARG=(--s4-path "${S4_PATH}")
fi

METHOD_ARGS=(--methods "${METHODS[@]}")
MODEL_ARGS=(--models "${MODELS[@]}")

# Ensure output dir exists and start fresh
mkdir -p "$(dirname "${OUTCSV}")"
rm -f "${OUTCSV}"

echo "Device: ${DEVICE}"
echo "Models: ${MODELS[*]}"
echo "Methods: ${METHODS[*]}"
echo "Problems: ${PROBLEMS[*]}"
echo "Output CSV: ${OUTCSV}"
echo ""

FIRST=1
for PROB in "${PROBLEMS[@]}"; do
  TMPCSV="experiments/tmp_${PROB}.csv"
  echo ">>> Running problem: ${PROB}"

  # Build the python command safely in an array
  cmd=(python experiments/run_all_experiments.py
       --problem "${PROB}"
       --lags "${LAGS}"
       --alpha "${ALPHA}"
       --gamma "${GAMMA}"
       --train_size "${TRAIN_SIZE}"
       "${METHOD_ARGS[@]}"
       "${MODEL_ARGS[@]}"
       --device "${DEVICE}" ${AMP_FLAG}
       --csv "${TMPCSV}"
  )

  # Append optional S4 path only if set
  if (( ${#S4_PATH_ARG[@]} )); then
    cmd+=("${S4_PATH_ARG[@]}")
  fi

  # (Optional) echo the command for debugging
  # printf 'CMD:'; printf ' %q' "${cmd[@]}"; printf '\n'

  "${cmd[@]}"

  if [[ "${FIRST}" -eq 1 ]]; then
    mv "${TMPCSV}" "${OUTCSV}"
    FIRST=0
  else
    awk 'NR>1{print}' "${TMPCSV}" >> "${OUTCSV}"
    rm -f "${TMPCSV}"
  fi

  echo ""
done

echo "====================================="
echo " Done. Combined CSV at: ${OUTCSV}"
echo "====================================="
