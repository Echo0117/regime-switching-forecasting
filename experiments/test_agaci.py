"""
Test script to verify AgACI implementation and generate all analysis plots.

This script:
1. Runs AgACI on a dataset
2. Generates weight analysis plots aligned to regime switches
3. Generates coverage analysis plots
4. Generates coverage vs length tradeoff plots

Usage:
    python experiments/test_agaci.py --problem Electricity
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

from experiments.utils.acp_utils import aci_intervals, agaci_intervals
from experiments.utils.ds3m_utils import load_ds3m_data, load_ds3m_model, forecast, get_full_d_argmax
from experiments.utils.regime_switch_analysis import (
    plot_agaci_weights_at_switches,
    plot_coverage_at_switches,
    plot_coverage_full_timeline,
    plot_coverage_timeline_scatter,
    plot_coverage_vs_length_tradeoff,
    plot_regime_heatmap_full,
    load_timestamps_for_dataset
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="Electricity",
                    choices=["Toy", "Lorenz", "Sleep", "Unemployment", "Hangzhou", "Seattle", "Pacific", "Electricity"])
    ap.add_argument("--aci_train_size", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--tab-gamma", type=float, nargs="*", default=[0.0025, 0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--agaci-eta", type=float, default=0.1, help="Learning rate for AgACI BOA")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d-dim", type=int, default=2, help="Number of regimes (default=2 for simpler experiments)")

    args = ap.parse_args()
    np.random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"Testing AgACI on {args.problem} with d_dim={args.d_dim}")
    print(f"{'='*60}\n")

    # Load DS3M data and forecasts
    ds = load_ds3m_data(args)

    # Override d_dim for simpler experiments
    ds["d_dim"] = args.d_dim

    model = load_ds3m_model(
        ds["directoryBest"],
        ds["x_dim"], ds["y_dim"], ds["h_dim"], ds["z_dim"],
        ds["d_dim"], ds["n_layers"], ds["learning_rate"],
        ds["device"], bidirection=ds["bidirection"],
    )

    # First, get regime indicators for FULL dataset (train+valid+test)
    # We need to run forecast on the entire RawDataOriginal to get complete d_argmax
    data_full = np.asarray(ds["data"])
    if data_full.ndim == 1:
        data_full = data_full[:, None]

    N_full = len(data_full)
    print(f"Full dataset length: {N_full}")

    # Get test portion forecasts for interval construction
    res, testForecast_mean, testOriginal, size, d_argmax_test, uq, lq = forecast(
        model,
        ds["testX"], ds["testY"],
        ds["moments"], ds["d_dim"],
        ds["means"], ds["trend"],
        ds["test_len"], ds["freq"],
        ds["RawDataOriginal"],
        remove_mean=ds["remove_mean"],
        remove_residual=ds["remove_residual"],
    )

    # Get full regime sequence using the proper utility function
    print("Extracting full regime sequence from model...")
    try:
        d_argmax_full = get_full_d_argmax(model, ds)
        print(f"Successfully extracted full regime sequence (length: {len(d_argmax_full)})")
    except Exception as e:
        print(f"Could not extract full regime sequence from model: {e}")
        print("Using fallback: test regimes padded with regime 0 for training period")
        test_len_actual = len(d_argmax_test)
        d_argmax_full = np.zeros(N_full, dtype=int)
        # Place test regimes at the end
        d_argmax_full[-test_len_actual:] = d_argmax_test

    d_argmax_full = np.asarray(d_argmax_full).reshape(-1)

    # Get target dimension
    target_dim = int(ds["target_dim"])
    print(f"Target dimension: {target_dim}")

    # Extract data for target dimension
    y_true = np.asarray(testOriginal)
    if y_true.ndim > 1:
        y_true = y_true[:, target_dim]

    y_lq_ds3m = np.asarray(lq)
    y_uq_ds3m = np.asarray(uq)
    if y_lq_ds3m.ndim > 1:
        y_lq_ds3m = y_lq_ds3m[:, target_dim]
    if y_uq_ds3m.ndim > 1:
        y_uq_ds3m = y_uq_ds3m[:, target_dim]

    d_argmax_test = np.asarray(d_argmax_test).reshape(-1)
    test_len = int(ds["test_len"])

    print(f"Test length: {test_len}")
    print(f"Full dataset length: {N_full}")
    print(f"Number of regime switches (full): {np.sum(np.diff(d_argmax_full) != 0)}")
    print(f"Number of regime switches (test): {np.sum(np.diff(d_argmax_test) != 0)}")

    # Prepare data
    y_full = np.asarray(ds["data"])
    if y_full.ndim == 1:
        y_full = y_full.reshape(-1, 1)

    N = len(y_full)
    X_dummy = np.zeros((N, 1), dtype=float)

    # ---------------------------------------------------
    # 1. Run standard ACI
    # ---------------------------------------------------
    print("\n" + "="*60)
    print("Running ACI...")
    print("="*60)

    y_lowers_aci, y_uppers_aci, tab_alpha_t_aci, gammas_aci = aci_intervals(
        X_dummy, y_full, args=args
    )

    print(f"ACI completed. Got {len(gammas_aci)} experts with gammas: {gammas_aci}")

    # Select best gamma for ACI (simple: use first one for now)
    aci_lower = y_lowers_aci[0]
    aci_upper = y_uppers_aci[0]

    # ---------------------------------------------------
    # 2. Run AgACI
    # ---------------------------------------------------
    print("\n" + "="*60)
    print("Running AgACI...")
    print("="*60)
    print("[WEIGHTS FLOW] Starting AgACI aggregation")
    print(f"[WEIGHTS FLOW] Number of gamma experts: {len(gammas_aci)}")
    print(f"[WEIGHTS FLOW] Gamma values: {gammas_aci}")
    print(f"[WEIGHTS FLOW] Learning rate (eta): {args.agaci_eta}")

    agaci_results = agaci_intervals(
        X_dummy, y_full, basemodel="ds3m", args=args
    )

    agaci_lower = agaci_results['lower']
    agaci_upper = agaci_results['upper']
    weights_lower = agaci_results['weights_lower']
    weights_upper = agaci_results['weights_upper']

    print(f"\n[WEIGHTS FLOW] AgACI completed.")
    print(f"[WEIGHTS FLOW] Output weights_lower shape: {weights_lower.shape}")
    print(f"[WEIGHTS FLOW] Output weights_upper shape: {weights_upper.shape}")
    print(f"[WEIGHTS FLOW] Weights_lower summary:")
    print(f"[WEIGHTS FLOW]   - Initial weights (t=0): {weights_lower[0]}")
    print(f"[WEIGHTS FLOW]   - Final weights (t={len(weights_lower)-1}): {weights_lower[-1]}")
    print(f"[WEIGHTS FLOW]   - Mean weights over time: {np.mean(weights_lower, axis=0)}")
    print(f"[WEIGHTS FLOW]   - Std weights over time: {np.std(weights_lower, axis=0)}")
    print(f"\n  Coverage: {agaci_results['coverage']:.3f}")
    print(f"  Median length: {agaci_results['median_length']:.2f}")

    # Debug: Check if AgACI is different from ACI
    if np.allclose(aci_lower, agaci_lower, rtol=1e-5, equal_nan=True):
        print("  WARNING: AgACI intervals are IDENTICAL to ACI (first gamma)!")
        print("  This suggests AgACI aggregation may not be working correctly.")
    else:
        diff_lower = np.nanmean(np.abs(aci_lower - agaci_lower))
        print(f"  AgACI differs from ACI by avg {diff_lower:.4f} in lower bounds")

    # ---------------------------------------------------
    # 3. Align intervals to test period
    # ---------------------------------------------------
    T0 = args.aci_train_size
    test_size_eff = len(aci_lower)

    # Pad beginning with NaN
    aci_lower_full = np.full(test_len, np.nan)
    aci_upper_full = np.full(test_len, np.nan)
    agaci_lower_full = np.full(test_len, np.nan)
    agaci_upper_full = np.full(test_len, np.nan)

    aci_lower_full[T0:T0+test_size_eff] = aci_lower
    aci_upper_full[T0:T0+test_size_eff] = aci_upper
    agaci_lower_full[T0:T0+test_size_eff] = agaci_lower
    agaci_upper_full[T0:T0+test_size_eff] = agaci_upper

    # ---------------------------------------------------
    # 4. Generate plots
    # ---------------------------------------------------
    save_dir = f"figures/agaci_test/{args.problem}"
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)

    # Plot 0a: Regime heatmap for full dataset (train+valid+test with window indices)
    print("\n0a. Plotting regime heatmap for full dataset (all windows)...")
    test_start_in_full = N_full - test_len  # Where test set starts in full data
    plot_regime_heatmap_full(
        d_argmax=d_argmax_full,
        d_dim=args.d_dim,
        dataname=args.problem,
        timestamps=None,  # Can't use timestamps for full window sequence
        save_path=f"{save_dir}/regime_heatmap_full.png",
        plot_scope="full",
        test_start_idx=test_start_in_full
    )

    # Plot 0b: Regime heatmap for test set only (with real timestamps)
    print("\n0b. Plotting regime heatmap for test set only (with timestamps)...")
    timestamps_test = load_timestamps_for_dataset(args.problem, test_len, from_end=True)
    plot_regime_heatmap_full(
        d_argmax=d_argmax_test,
        d_dim=args.d_dim,
        dataname=args.problem,
        timestamps=timestamps_test,
        save_path=f"{save_dir}/regime_heatmap_test.png",
        plot_scope="test"
    )

    # Plot 1: AgACI weights aligned to switches
    # Align weights to full dataset (pad with NaN for pre-test period)
    print("\n1. Plotting AgACI weights at regime switches...")

    # Weights are only available for test portion after T0
    # Pad to align with full dataset
    weights_lower_full = np.full((len(gammas_aci), N_full), np.nan)
    # Place weights in correct position (test portion starts at N_full - test_len + T0)
    test_start_in_full = N_full - test_len + T0
    weights_lower_full[:, test_start_in_full:test_start_in_full + test_size_eff] = weights_lower.T

    plot_agaci_weights_at_switches(
        agaci_weights=weights_lower_full,  # (n_gammas, N_full) with NaN padding
        d_argmax=d_argmax_full,  # Use full dataset regimes
        gamma_values=gammas_aci,
        # window_before and window_after will be computed adaptively
        adaptive_window=True,  # Enable adaptive window sizing
        save_path=f"{save_dir}/agaci_weights_switches.png"
    )

    # Plot 2: Coverage at switches
    print("\n" + "="*60)
    print("2. Plotting coverage dynamics at regime switches...")
    print("="*60)

    # Compute naive interval (same as DS3M for comparison but labeled differently)
    # Naive = using DS3M quantiles directly without adaptation
    intervals_dict = {
        'Naive': (y_lq_ds3m, y_uq_ds3m),  # DS3M original interval (naive/unadapted)
        'ACI': (aci_lower_full, aci_upper_full),
        'AgACI': (agaci_lower_full, agaci_upper_full),
    }

    print(f"\n>>> CREATING intervals_dict with {len(intervals_dict)} methods: {list(intervals_dict.keys())}")

    # Debug: Check intervals
    print(f"\nInterval dict keys: {list(intervals_dict.keys())}")
    for method_name, (lower, upper) in intervals_dict.items():
        n_valid = np.sum(~np.isnan(lower))
        print(f"  {method_name}: {n_valid}/{len(lower)} valid values, "
              f"range=[{np.nanmin(lower):.2f}, {np.nanmax(upper):.2f}]")
        if method_name == 'ACI' and np.allclose(lower[~np.isnan(lower)],
                                                  agaci_lower_full[~np.isnan(agaci_lower_full)],
                                                  rtol=1e-5):
            print(f"  WARNING: ACI and AgACI intervals are identical!")

    # Plot 2a: Windowed coverage around switches
    plot_coverage_at_switches(
        intervals_dict,
        y_true,
        d_argmax_test,  # Use test regime for coverage analysis
        adaptive_window=True,  # Use adaptive window sizing
        save_path=f"{save_dir}/coverage_at_switches.png"
    )

    # Plot 2b: Full timeline coverage (bar chart)
    print("\n2b. Plotting coverage over full timeline (bar chart)...")
    plot_coverage_full_timeline(
        intervals_dict,
        y_true,
        d_argmax_test,  # Use test regime for coverage analysis
        timestamps=timestamps_test,  # Use same timestamps as test heatmap
        save_path=f"{save_dir}/coverage_full_timeline.png"
    )

    # Plot 2c: Full timeline coverage (scatter plot)
    print("\n2c. Plotting coverage over full timeline (scatter plot)...")
    plot_coverage_timeline_scatter(
        intervals_dict,
        y_true,
        d_argmax_test,  # Use test regime for coverage analysis
        timestamps=timestamps_test,  # Use same timestamps as test heatmap
        save_path=f"{save_dir}/coverage_timeline_scatter.png"
    )

    # Plot 3: Tradeoff plot
    print("3. Plotting coverage vs length tradeoff...")
    results_dict = {}

    for method_name, (lower, upper) in intervals_dict.items():
        valid = ~np.isnan(lower) & ~np.isnan(upper)
        if not np.any(valid):
            continue

        y_true_valid = y_true[valid]
        lower_valid = lower[valid]
        upper_valid = upper[valid]

        coverage = np.mean((y_true_valid >= lower_valid) & (y_true_valid <= upper_valid))
        median_length = np.median(upper_valid - lower_valid)

        results_dict[method_name] = (coverage, median_length)
        print(f"   {method_name}: Coverage={coverage:.3f}, MedianLength={median_length:.2f}")

    plot_coverage_vs_length_tradeoff(
        results_dict,
        save_path=f"{save_dir}/tradeoff.png"
    )

    print(f"\n{'='*60}")
    print(f"All plots saved to: {save_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
