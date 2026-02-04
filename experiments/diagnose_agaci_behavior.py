"""
Diagnostic script to analyze AGACI vs ACI behavior around regime switches.

This script addresses the meeting feedback:
- Charles: "I'd like to understand why AGACI doesn't work as well than ACI"
- Investigate whether different gammas are optimal during regime vs. at regime switch
- Test on simple AR model first (simpler to understand)

Usage:
    python experiments/diagnose_agaci_behavior.py
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

from sklearn.linear_model import Ridge


def generate_ar_with_switch(
    n_total: int = 500,
    switch_point: int = 250,
    ar_coef_before: float = 0.8,
    ar_coef_after: float = 0.3,
    var_before: float = 0.5,
    var_after: float = 2.0,
    seed: int = 42
):
    """
    Generate AR(1) process with a single regime switch.

    Parameters
    ----------
    n_total : int
        Total length of series
    switch_point : int
        Index where regime switch occurs
    ar_coef_before, ar_coef_after : float
        AR(1) coefficients before/after switch
    var_before, var_after : float
        Noise variance before/after switch

    Returns
    -------
    y : np.ndarray
        Generated time series
    regime_labels : np.ndarray
        Regime labels (0 before switch, 1 after)
    """
    np.random.seed(seed)

    y = np.zeros(n_total)
    regime_labels = np.zeros(n_total, dtype=int)

    # Initialize
    y[0] = np.random.randn() * np.sqrt(var_before)

    for t in range(1, n_total):
        if t < switch_point:
            ar_coef = ar_coef_before
            var = var_before
            regime_labels[t] = 0
        else:
            ar_coef = ar_coef_after
            var = var_after
            regime_labels[t] = 1

        y[t] = ar_coef * y[t-1] + np.random.randn() * np.sqrt(var)

    return y, regime_labels


def create_lag_features(data: np.ndarray, lag: int):
    """Create lag features for AR forecasting."""
    X = np.zeros((len(data) - lag, lag))
    y_target = np.zeros(len(data) - lag)

    for i in range(lag, len(data)):
        X[i - lag] = data[i - lag:i]
        y_target[i - lag] = data[i]

    return X, y_target


def compute_aci_intervals(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    alpha: float,
    gamma: float,
    train_size: int
):
    """
    Compute ACI (Adaptive Conformal Inference) intervals.

    Parameters
    ----------
    y_pred : np.ndarray
        Point predictions
    y_true : np.ndarray
        True values
    alpha : float
        Miscoverage rate (e.g., 0.1 for 90% PI)
    gamma : float
        Step size for alpha_t update
    train_size : int
        Number of points to use for initial calibration

    Returns
    -------
    lower, upper : np.ndarray
        Prediction interval bounds
    alpha_t_history : np.ndarray
        History of adaptive alpha values
    """
    n = len(y_true)
    test_size = n - train_size

    # Initialize
    alpha_t = alpha
    alpha_t_history = np.zeros(test_size)

    # Calibration set residuals
    residuals_cal = np.abs(y_true[:train_size] - y_pred[:train_size])

    lower = np.zeros(test_size)
    upper = np.zeros(test_size)

    for t in range(test_size):
        idx = train_size + t

        # Store alpha_t
        alpha_t_history[t] = alpha_t

        # Compute quantile of residuals
        # Use all residuals up to current point
        all_residuals = np.abs(y_true[:idx] - y_pred[:idx])

        # Compute conformal quantile with current alpha_t
        # Ensure alpha_t is bounded
        alpha_t_bounded = np.clip(alpha_t, 0.01, 0.99)
        q = np.quantile(all_residuals, 1 - alpha_t_bounded)

        # Prediction interval
        lower[t] = y_pred[idx] - q
        upper[t] = y_pred[idx] + q

        # Check coverage and update alpha_t
        covered = (y_true[idx] >= lower[t]) and (y_true[idx] <= upper[t])
        err_t = alpha if covered else alpha - 1  # +alpha if covered, -(1-alpha) if not

        # ACI update rule
        alpha_t = alpha_t + gamma * (alpha - (1 if not covered else 0))

    return lower, upper, alpha_t_history


def run_diagnostic():
    """Run full AGACI diagnostic analysis."""
    print("=" * 70)
    print("AGACI vs ACI Diagnostic Analysis")
    print("=" * 70)

    # Generate data with regime switch
    n_total = 500
    switch_point = 250
    train_size = 100

    y, regime_labels = generate_ar_with_switch(
        n_total=n_total,
        switch_point=switch_point,
        var_before=0.5,
        var_after=2.0,
        seed=42
    )

    print(f"\nData generated: n={n_total}, switch at t={switch_point}")
    print(f"Variance before switch: 0.5, after switch: 2.0")

    # Create lag features
    lag = 5
    X, y_target = create_lag_features(y, lag)

    # Split train/test
    X_train, y_train = X[:train_size], y_target[:train_size]
    X_all, y_all = X, y_target

    # Fit AR model
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_all)

    print(f"\nAR model fitted with lag={lag}")

    # Test different gamma values
    gammas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    alpha = 0.1  # 90% PI

    results = {}

    print("\n" + "=" * 70)
    print("Testing different gamma values for ACI")
    print("=" * 70)

    for gamma in gammas:
        lower, upper, alpha_t = compute_aci_intervals(
            y_pred, y_all, alpha, gamma, train_size
        )

        # Compute coverage
        test_indices = np.arange(train_size, len(y_all))
        y_test = y_all[train_size:]
        covered = (y_test >= lower) & (y_test <= upper)
        coverage = np.mean(covered)

        # Coverage before and after switch
        switch_idx_test = switch_point - train_size - lag  # Adjust for lag and train_size
        if switch_idx_test > 0 and switch_idx_test < len(covered):
            coverage_before = np.mean(covered[:switch_idx_test])
            coverage_after = np.mean(covered[switch_idx_test:])
        else:
            coverage_before = coverage_after = coverage

        # Median interval length
        lengths = upper - lower
        median_length = np.median(lengths)

        results[gamma] = {
            'lower': lower,
            'upper': upper,
            'alpha_t': alpha_t,
            'coverage': coverage,
            'coverage_before': coverage_before,
            'coverage_after': coverage_after,
            'median_length': median_length,
            'lengths': lengths,
            'covered': covered
        }

        print(f"\ngamma = {gamma:.4f}:")
        print(f"  Overall coverage: {coverage:.3f}")
        print(f"  Coverage before switch: {coverage_before:.3f}")
        print(f"  Coverage after switch: {coverage_after:.3f}")
        print(f"  Median interval length: {median_length:.3f}")

    # Analyze which gamma is best at different times
    print("\n" + "=" * 70)
    print("Analysis: Which gamma performs best when?")
    print("=" * 70)

    # Find best gamma by coverage (closest to 0.9) before and after switch
    switch_idx_test = switch_point - train_size - lag

    best_before_coverage = 0
    best_before_gamma = None
    best_after_coverage = 0
    best_after_gamma = None

    for gamma, res in results.items():
        # Before switch: want coverage closest to 0.9
        if abs(res['coverage_before'] - 0.9) < abs(best_before_coverage - 0.9):
            best_before_coverage = res['coverage_before']
            best_before_gamma = gamma

        # After switch: want coverage closest to 0.9
        if abs(res['coverage_after'] - 0.9) < abs(best_after_coverage - 0.9):
            best_after_coverage = res['coverage_after']
            best_after_gamma = gamma

    print(f"\nBest gamma BEFORE switch: {best_before_gamma:.4f} (coverage: {best_before_coverage:.3f})")
    print(f"Best gamma AFTER switch: {best_after_gamma:.4f} (coverage: {best_after_coverage:.3f})")

    if best_before_gamma != best_after_gamma:
        print("\n*** IMPORTANT: Different gammas are optimal before and after switch! ***")
        print("This suggests AGACI SHOULD be able to adapt and outperform single-gamma ACI.")
        print("\nPossible reasons AGACI doesn't work as expected:")
        print("  1. BOA learning rate (eta) might be too slow")
        print("  2. Weight adaptation takes too long to catch regime switch")
        print("  3. Gamma range might not include optimal values")
    else:
        print("\n*** Same gamma is optimal before and after switch ***")
        print("In this case, AGACI won't outperform the best single-gamma ACI.")

    # Create diagnostic plots
    output_dir = Path("figures/agaci_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Data with regime switch
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    ax = axes[0]
    time = np.arange(len(y))
    ax.plot(time, y, 'b-', alpha=0.7, label='Data')
    ax.axvline(switch_point, color='red', linestyle='--', label='Regime switch')
    ax.fill_between(time[:switch_point], y.min(), y.max(), alpha=0.1, color='blue', label='Regime 0')
    ax.fill_between(time[switch_point:], y.min(), y.max(), alpha=0.1, color='orange', label='Regime 1')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('AR(1) with Regime Switch (var: 0.5 → 2.0)')
    ax.legend()

    # Plot 2: Coverage over time for different gammas
    ax = axes[1]
    window = 30  # Rolling window for coverage
    test_time = np.arange(train_size + lag, len(y))

    colors = plt.cm.viridis(np.linspace(0, 1, len(gammas)))
    for i, (gamma, res) in enumerate(results.items()):
        # Rolling coverage
        rolling_cov = np.convolve(res['covered'].astype(float),
                                   np.ones(window)/window, mode='valid')
        ax.plot(test_time[:len(rolling_cov)], rolling_cov,
               color=colors[i], label=f'γ={gamma:.3f}', alpha=0.8)

    ax.axhline(0.9, color='black', linestyle=':', label='Target (90%)')
    ax.axvline(switch_point, color='red', linestyle='--', label='Switch')
    ax.set_xlabel('Time')
    ax.set_ylabel('Rolling Coverage')
    ax.set_title(f'Rolling Coverage (window={window}) for Different Gammas')
    ax.legend(loc='lower left', ncol=4, fontsize=8)
    ax.set_ylim([0.5, 1.05])

    # Plot 3: Interval length over time
    ax = axes[2]
    for i, (gamma, res) in enumerate(results.items()):
        ax.plot(test_time[:len(res['lengths'])], res['lengths'],
               color=colors[i], label=f'γ={gamma:.3f}', alpha=0.8)

    ax.axvline(switch_point, color='red', linestyle='--', label='Switch')
    ax.set_xlabel('Time')
    ax.set_ylabel('Interval Length')
    ax.set_title('Prediction Interval Length for Different Gammas')
    ax.legend(loc='upper left', ncol=4, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'aci_gamma_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nDiagnostic plot saved to: {output_dir / 'aci_gamma_comparison.png'}")

    # Plot 4: Alpha_t evolution for different gammas
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (gamma, res) in enumerate(results.items()):
        ax.plot(test_time[:len(res['alpha_t'])], res['alpha_t'],
               color=colors[i], label=f'γ={gamma:.3f}', alpha=0.8)

    ax.axhline(alpha, color='black', linestyle=':', label=f'Target α={alpha}')
    ax.axvline(switch_point, color='red', linestyle='--', label='Switch')
    ax.set_xlabel('Time')
    ax.set_ylabel('α_t')
    ax.set_title('Adaptive α_t Evolution for Different Gammas')
    ax.legend(loc='upper left', ncol=4, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'alpha_t_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Alpha evolution plot saved to: {output_dir / 'alpha_t_evolution.png'}")

    # Summary recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS for AGACI improvement:")
    print("=" * 70)
    print("""
1. If different gammas are optimal at different times:
   - Increase BOA learning rate (eta) for faster adaptation
   - Use constant learning rate schedule (not decaying)
   - Consider wider gamma range to capture optimal values

2. If same gamma is optimal throughout:
   - AGACI won't help in this scenario
   - Focus on choosing the right single gamma value
   - Consider alternative aggregation strategies

3. For regime-switching experiments:
   - Generate data with LARGER variance change (e.g., 0.5 → 4.0)
   - Use MULTIPLE switches to give BOA time to learn
   - Analyze weight dynamics around each switch
""")

    return results


if __name__ == "__main__":
    results = run_diagnostic()
