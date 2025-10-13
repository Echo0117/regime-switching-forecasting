"""
Example script to generate regime switch analysis plots.

Usage:
    python experiments/analyze_regime_switches.py --problem Electricity --method AgACI
"""

import argparse
import os
import sys
import numpy as np

HERE = os.path.dirname(__file__)
PROJ = os.path.abspath(os.path.join(HERE, ".."))
for p in [HERE, PROJ]:
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.utils.regime_switch_analysis import (
    plot_agaci_weights_at_switches,
    plot_coverage_at_switches,
    plot_coverage_vs_length_tradeoff
)
from experiments.utils.acp_utils import aci_intervals
from experiments.utils.ds3m_utils import load_ds3m_data, load_ds3m_model, forecast
from experiments.utils.experiments_utils import _fetch_ds3m_outputs
from experiments.run_all_experiments import _fetch_ds3m_outputs as fetch_outputs


def run_analysis(args):
    """Run regime switch analysis for a given problem."""

    # Load DS3M data and forecasts
    ds = load_ds3m_data(args)

    model = load_ds3m_model(
        ds["directoryBest"],
        ds["x_dim"], ds["y_dim"], ds["h_dim"], ds["z_dim"],
        ds["d_dim"], ds["n_layers"], ds["learning_rate"],
        ds["device"], bidirection=ds["bidirection"],
    )

    res, testForecast_mean, testOriginal, size, d_argmax, uq, lq = forecast(
        model,
        ds["testX"], ds["testY"],
        ds["moments"], ds["d_dim"],
        ds["means"], ds["trend"],
        ds["test_len"], ds["freq"],
        ds["RawDataOriginal"],
        remove_mean=ds["remove_mean"],
        remove_residual=ds["remove_residual"],
    )

    # Get target dimension
    target_dim = int(ds["target_dim"])

    # Extract data for target dimension
    y_true = np.asarray(testOriginal)
    y_pred = np.asarray(testForecast_mean)
    if y_true.ndim > 1:
        y_true = y_true[:, target_dim]
    if y_pred.ndim > 1:
        y_pred = y_pred[:, target_dim]

    d_argmax = np.asarray(d_argmax).reshape(-1)

    # Get DS3M intervals
    y_lq = np.asarray(lq)
    y_uq = np.asarray(uq)
    if y_lq.ndim > 1:
        y_lq = y_lq[:, target_dim]
    if y_uq.ndim > 1:
        y_uq = y_uq[:, target_dim]

    print(f"Analyzing {args.problem}")
    print(f"Test length: {len(y_true)}")
    print(f"Target dimension: {target_dim}")
    print(f"Number of regime switches: {np.sum(np.diff(d_argmax) != 0)}")

    # Prepare data for ACI
    y_full = np.asarray(ds["data"])
    if y_full.ndim == 1:
        y_full = y_full.reshape(-1, 1)

    N = len(y_full)
    test_len = int(ds["test_len"])
    X_dummy = np.zeros((N, 1), dtype=float)

    # Run ACI
    print("\nRunning ACI...")
    y_lowers_aci, y_uppers_aci, tab_alpha_t, gammas = aci_intervals(X_dummy, y_full, args=args)

    # Align intervals to test period for all gammas
    # ACI outputs are for the tail starting at T0
    T0 = args.aci_train_size

    # Create intervals dict for plotting with all gammas
    intervals_dict = {
        'DS3M': (y_lq, y_uq),
    }

    # Add AgACI intervals for each gamma
    for gid, gamma in enumerate(gammas):
        aci_lower = y_lowers_aci[gid]
        aci_upper = y_uppers_aci[gid]

        # For plotting, we need intervals aligned with test data
        # Currently ACI gives us (test_len - T0) intervals
        # We need to pad the beginning with NaN
        test_size_eff = len(aci_lower)
        aci_lower_full = np.full(test_len, np.nan)
        aci_upper_full = np.full(test_len, np.nan)
        aci_lower_full[T0:T0+test_size_eff] = aci_lower
        aci_upper_full[T0:T0+test_size_eff] = aci_upper

        intervals_dict[f'AgACI (γ={gamma:.3f})'] = (aci_lower_full, aci_upper_full)

    # --- Plot 1: Coverage around regime switches ---
    print("\nGenerating coverage plot...")
    save_dir = "figures/regime_analysis"
    os.makedirs(save_dir, exist_ok=True)

    plot_coverage_at_switches(
        intervals_dict,
        y_true,
        d_argmax,
        window_before=10,
        window_after=50,
        save_path=f"{save_dir}/{args.problem}_coverage_at_switches.png"
    )

    # --- Plot 2: AgACI weights (if implemented) ---
    # TODO: Need to implement AgACI with BOA to get weights
    # For now, we can plot the alpha_t values from ACI
    print("\nSkipping AgACI weights plot (AgACI not yet fully implemented)")

    # --- Plot 3: Coverage vs Length tradeoff ---
    print("\nGenerating tradeoff plot...")

    # Compute metrics for each method
    results_dict = {}

    for method_name, (lower, upper) in intervals_dict.items():
        # Only use valid intervals
        valid = ~np.isnan(lower) & ~np.isnan(upper)
        if not np.any(valid):
            continue

        y_true_valid = y_true[valid]
        lower_valid = lower[valid]
        upper_valid = upper[valid]

        coverage = np.mean((y_true_valid >= lower_valid) & (y_true_valid <= upper_valid))
        median_length = np.median(upper_valid - lower_valid)

        results_dict[method_name] = (coverage, median_length)
        print(f"{method_name}: Coverage={coverage:.3f}, MedianLength={median_length:.2f}")

    plot_coverage_vs_length_tradeoff(
        results_dict,
        save_path=f"{save_dir}/{args.problem}_tradeoff.png"
    )

    print(f"\nPlots saved to {save_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="Electricity",
                    choices=["Toy", "Lorenz", "Sleep", "Unemployment", "Hangzhou", "Seattle", "Pacific", "Electricity"])
    ap.add_argument("--aci_train_size", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--tab-gamma", type=float, nargs="*", default=[0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    np.random.seed(args.seed)

    run_analysis(args)


if __name__ == "__main__":
    main()
