"""
Diagnostic script to identify and fix lag/shift issues in predictions.

This addresses Charles's feedback from the meeting:
"Why does DS3M seem to be shifted one time step? ... It seems like DS3M is shifted one time, one time step."

Key things to check:
1. How lag features are created (X[t-lag:t] -> y[t])
2. How DS3M aligns predictions with ground truth
3. Cross-correlation between prediction and ground truth
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add project root to path
HERE = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


def cross_correlation_lag(y_true, y_pred, max_lag=10):
    """
    Compute cross-correlation to find optimal lag between prediction and ground truth.

    Returns
    -------
    best_lag : int
        Positive means y_pred should be shifted RIGHT to align with y_true
        Negative means y_pred should be shifted LEFT
    correlations : dict
        Correlation at each lag
    """
    correlations = {}

    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            corr = np.corrcoef(y_true, y_pred)[0, 1]
        elif lag > 0:
            # Shift y_pred right (compare y_pred[:-lag] with y_true[lag:])
            if len(y_true) > lag:
                corr = np.corrcoef(y_true[lag:], y_pred[:-lag])[0, 1]
            else:
                corr = np.nan
        else:
            # Shift y_pred left (compare y_pred[-lag:] with y_true[:lag])
            if len(y_true) > -lag:
                corr = np.corrcoef(y_true[:lag], y_pred[-lag:])[0, 1]
            else:
                corr = np.nan

        correlations[lag] = corr

    # Find best lag
    valid_corrs = {k: v for k, v in correlations.items() if not np.isnan(v)}
    best_lag = max(valid_corrs, key=valid_corrs.get)

    return best_lag, correlations


def diagnose_dataset(dataname):
    """
    Diagnose lag/shift issues for a specific dataset.
    """
    print(f"\n{'='*70}")
    print(f"Diagnosing lag/shift for: {dataname}")
    print(f"{'='*70}")

    # Import required modules
    from experiments.competitor_models import DS3MWrapper, RupturesSegmentedLinear
    from experiments.generate_forecasting_comparison import (
        load_original_data, create_lag_features, standardize_train_test, DATASET_CONFIG
    )
    from sklearn.linear_model import Ridge

    config = DATASET_CONFIG.get(dataname)
    if config is None:
        print(f"Unknown dataset: {dataname}")
        return None

    test_len = config['test_len']
    lags = config['lags']
    dim = config['dim']

    # Load data
    try:
        raw_data = load_original_data(dataname)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # Extract 1D series
    if raw_data.ndim > 1:
        y_1d = raw_data[:, dim].flatten()
    else:
        y_1d = raw_data.flatten()

    # Create lag features
    X, y = create_lag_features(y_1d, lags)
    N = len(y)
    train_end = N - min(test_len, N - 50)

    X_train, y_train = X[:train_end], y[:train_end]
    X_test, y_test = X[train_end:], y[train_end:]

    print(f"Data shape: N={N}, lags={lags}, test_len={len(y_test)}")

    results = {'y_true': y_test}

    # 1. AR baseline
    print("\n1. AR (Ridge) baseline...")
    ar = Ridge(alpha=1.0)
    ar.fit(X_train, y_train)
    pred_ar = ar.predict(X_test)
    results['AR'] = pred_ar

    # 2. CPD (Ruptures)
    print("2. CPD (Ruptures)...")
    try:
        cpd = RupturesSegmentedLinear(penalty=10.0, min_size=max(20, len(y_train)//10))
        cpd.fit(X_train, y_train)
        pred_cpd = cpd.predict(X_test)
        results['CPD'] = pred_cpd
    except Exception as e:
        print(f"   CPD failed: {e}")

    # 3. DS3M (if available)
    print("3. DS3M...")
    try:
        ds3m = DS3MWrapper(
            lags=lags,
            problem=dataname,
            target_dim=dim,
            device='cpu',
            force_new=False,
        )
        ds3m.fit(X, y)
        pred_all = ds3m.predict(X)
        pred_ds3m = pred_all[train_end:]
        results['DS3M'] = pred_ds3m
    except Exception as e:
        print(f"   DS3M failed: {e}")

    # Analyze cross-correlation to detect shift
    print("\n" + "-"*70)
    print("Cross-correlation analysis (detecting optimal lag):")
    print("-"*70)

    for method, pred in results.items():
        if method == 'y_true':
            continue

        pred = np.asarray(pred).flatten()
        y_t = results['y_true'].flatten()

        # Ensure same length
        min_len = min(len(y_t), len(pred))
        y_t = y_t[:min_len]
        pred = pred[:min_len]

        best_lag, correlations = cross_correlation_lag(y_t, pred, max_lag=5)

        print(f"\n{method}:")
        print(f"  Best lag: {best_lag}")
        print(f"  Interpretation: ", end="")

        if best_lag > 0:
            print(f"Predictions are AHEAD by {best_lag} step(s) - shift predictions RIGHT to align")
        elif best_lag < 0:
            print(f"Predictions are BEHIND by {abs(best_lag)} step(s) - shift predictions LEFT to align")
        else:
            print("Predictions are aligned (no shift needed)")

        print(f"  Correlation at each lag:")
        for lag in sorted(correlations.keys()):
            marker = " <-- BEST" if lag == best_lag else ""
            print(f"    lag={lag:+d}: {correlations[lag]:.4f}{marker}")

    # Generate diagnostic plot
    output_dir = Path("figures/lag_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_len = min(100, len(y_test))
    t = np.arange(plot_len)

    fig, axes = plt.subplots(len(results) - 1, 1, figsize=(14, 3 * (len(results) - 1)))
    if len(results) - 1 == 1:
        axes = [axes]

    y_true_plot = results['y_true'][:plot_len]

    for ax, (method, pred) in zip(axes, [(k, v) for k, v in results.items() if k != 'y_true']):
        pred_plot = np.asarray(pred).flatten()[:plot_len]

        ax.plot(t, y_true_plot, 'k-', alpha=0.7, linewidth=1.5, label='Ground Truth')
        ax.plot(t, pred_plot, 'b-', alpha=0.7, linewidth=1.2, label=f'{method} Prediction')

        # Mark peaks to visually check alignment
        from scipy.signal import find_peaks
        peaks_true, _ = find_peaks(y_true_plot, distance=5)
        peaks_pred, _ = find_peaks(pred_plot, distance=5)

        ax.scatter(peaks_true, y_true_plot[peaks_true], c='red', s=50, zorder=5, label='True peaks')
        ax.scatter(peaks_pred, pred_plot[peaks_pred], c='blue', s=50, marker='x', zorder=5, label='Pred peaks')

        # Compute shift
        best_lag, _ = cross_correlation_lag(y_true_plot, pred_plot, max_lag=5)

        ax.set_title(f'{dataname} - {method} (detected lag: {best_lag:+d})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / f"{dataname}_lag_diagnosis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nDiagnostic plot saved to: {save_path}")

    return results


def apply_lag_correction(y_pred, lag):
    """
    Apply lag correction to predictions.

    Parameters
    ----------
    y_pred : np.ndarray
        Predictions to correct
    lag : int
        Detected lag (positive = shift right, negative = shift left)

    Returns
    -------
    y_corrected : np.ndarray
        Corrected predictions
    """
    y_corrected = np.roll(y_pred, lag)

    # Handle edge effects
    if lag > 0:
        y_corrected[:lag] = y_pred[0]  # Fill start with first value
    elif lag < 0:
        y_corrected[lag:] = y_pred[-1]  # Fill end with last value

    return y_corrected


def main():
    """Run diagnostics on key datasets."""
    print("="*70)
    print("LAG/SHIFT DIAGNOSTIC ANALYSIS")
    print("="*70)
    print("""
This script analyzes prediction alignment issues.

Charles's observation: "DS3M seems to be shifted one time step"
"CPD is looking in advance... they're not all shifted in the same direction"

We use cross-correlation to detect the optimal lag between predictions and ground truth.
""")

    # Test on key datasets
    datasets_to_test = ['Electricity', 'Sleep', 'Unemployment']

    # Check which datasets are available
    from experiments.generate_forecasting_comparison import DATASET_CONFIG
    available = [d for d in datasets_to_test if d in DATASET_CONFIG]

    for dataname in available:
        try:
            diagnose_dataset(dataname)
        except Exception as e:
            print(f"Error diagnosing {dataname}: {e}")

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("""
If a model shows consistent lag ≠ 0:

1. For plotting: Apply lag correction using apply_lag_correction()

2. For evaluation: The metrics may be artificially inflated if predictions
   are compared at wrong time indices

3. Root cause investigation:
   - Check how lag features are created in create_lag_features()
   - Verify that X[t] predicts y[t+1] or y[t] (which convention?)
   - Ensure all models use the same convention

4. In the harness:
   - AR/MCD/GP/S4: X[t-lag:t] -> y[t] (predicting current step from history)
   - DS3M: Uses its own data loader with (L, N, D) tensors
   - CPD: Uses same lag features as AR
""")


if __name__ == "__main__":
    main()
