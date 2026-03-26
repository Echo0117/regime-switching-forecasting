"""
Diagnose time alignment for ALL models (not just DS3M).
"""
import sys
import os
import numpy as np
from sklearn.metrics import mean_squared_error

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

from experiments.generate_forecasting_comparison import run_comparison

def diagnose_model_alignment(dataname, model_name, y_true, y_pred, shift_range=10):
    """
    Check if a model's predictions are aligned with ground truth.

    Parameters
    ----------
    dataname : str
        Dataset name
    model_name : str
        Model name
    y_true : np.ndarray
        Ground truth
    y_pred : np.ndarray
        Model predictions
    shift_range : int
        Range of shifts to check

    Returns
    -------
    best_shift : int
        Best alignment shift
    best_rmse : float
        RMSE at best shift
    """
    if len(y_true) != len(y_pred):
        print(f"    WARNING: Length mismatch - truth: {len(y_true)}, pred: {len(y_pred)}")
        return None, None

    best_shift = 0
    best_rmse = float('inf')

    for shift in range(-shift_range, shift_range + 1):
        if shift == 0:
            y_true_aligned = y_true
            pred_aligned = y_pred
        elif shift > 0:
            # Shift prediction forward (prediction delayed)
            if shift >= len(y_true):
                continue
            y_true_aligned = y_true[:-shift]
            pred_aligned = y_pred[shift:]
        else:  # shift < 0
            # Shift prediction backward (prediction ahead)
            if -shift >= len(y_true):
                continue
            y_true_aligned = y_true[-shift:]
            pred_aligned = y_pred[:shift]

        if len(y_true_aligned) != len(pred_aligned):
            continue

        rmse = np.sqrt(mean_squared_error(y_true_aligned, pred_aligned))

        if rmse < best_rmse:
            best_rmse = rmse
            best_shift = shift

    return best_shift, best_rmse


def diagnose_all_models(dataname, shift_range=10):
    """
    Diagnose alignment for all models on a dataset.
    """
    print(f"\n{'='*70}")
    print(f"Alignment Diagnosis: {dataname}")
    print(f"{'='*70}\n")

    # Run comparison to get predictions
    results = run_comparison(
        dataname=dataname,
        device="cpu",
        verbose=False,
        use_multidim=False,
        ds3m_force_new=False,
    )

    if results is None:
        print(f"Failed to load results for {dataname}")
        return

    y_true = results['y_true']
    predictions = results['predictions']

    print(f"Ground truth length: {len(y_true)}\n")

    # Check each model
    alignment_results = {}
    for model_name in ['AR', 'S4', 'MCD', 'GP', 'CPD', 'DS3M']:
        if model_name not in predictions:
            print(f"[{model_name}] Not available")
            continue

        y_pred = predictions[model_name]
        print(f"[{model_name}] Prediction length: {len(y_pred)}")

        best_shift, best_rmse = diagnose_model_alignment(
            dataname, model_name, y_true, y_pred, shift_range
        )

        if best_shift is not None:
            alignment_results[model_name] = (best_shift, best_rmse)

            # Original RMSE at shift 0
            orig_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            status = "✅ OK" if best_shift == 0 else f"⚠️  SHIFT {best_shift:+d}"
            improvement = f"({orig_rmse:.4f} → {best_rmse:.4f})" if best_shift != 0 else ""

            print(f"    Best shift: {best_shift:+d} | RMSE: {best_rmse:.4f} | {status} {improvement}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY for {dataname}")
    print(f"{'='*70}")
    print(f"{'Model':<10} {'Shift':<10} {'RMSE':<15} {'Status'}")
    print(f"{'-'*70}")

    for model_name in ['AR', 'S4', 'MCD', 'GP', 'CPD', 'DS3M']:
        if model_name in alignment_results:
            shift, rmse = alignment_results[model_name]
            status = "✅ Aligned" if shift == 0 else f"⚠️  Offset {shift:+d}"
            print(f"{model_name:<10} {shift:+d}          {rmse:<15.4f} {status}")
        else:
            print(f"{model_name:<10} {'N/A':<10} {'N/A':<15} N/A")

    print(f"{'='*70}\n")

    return alignment_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Toy",
                        help="Dataset to diagnose (default: Toy)")
    parser.add_argument("--shift-range", type=int, default=10,
                        help="Range of shifts to check (default: 10)")
    parser.add_argument("--all", action="store_true",
                        help="Check all datasets")

    args = parser.parse_args()

    if args.all:
        datasets = [
            "Toy", "Lorenz", "Sleep", "Unemployment",
            "Electricity", "Hangzhou", "Seattle", "Pacific", "Pernod"
        ]

        all_results = {}
        for dataname in datasets:
            try:
                results = diagnose_all_models(dataname, args.shift_range)
                all_results[dataname] = results
            except Exception as e:
                print(f"Error with {dataname}: {e}\n")

        # Global summary
        print("\n" + "="*70)
        print("GLOBAL SUMMARY: All Datasets & Models")
        print("="*70)

        for dataname, results in all_results.items():
            if results:
                misaligned = [f"{m}({s:+d})" for m, (s, r) in results.items() if s != 0]
                if misaligned:
                    print(f"{dataname:<15} ⚠️  Misaligned: {', '.join(misaligned)}")
                else:
                    print(f"{dataname:<15} ✅ All models aligned")

        print("="*70)
    else:
        diagnose_all_models(args.dataset, args.shift_range)
