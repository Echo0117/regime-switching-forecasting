"""
Task 3: Uncertainty Quantification - AgACI Analysis

Generates UQ comparison figures for all datasets:
1. Coverage vs Width trade-off plot
2. Interval length at regime switches
3. Coverage at regime switches
4. Recovery comparison

Output: overleaf_upload/figures/task3_uq/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import argparse

# Add project root to path
HERE = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

# Output directory
OUTPUT_DIR = Path("overleaf_upload/figures/task3_uq")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import after path setup
from experiments.utils.ds3m_utils import load_ds3m_data, load_ds3m_model, forecast, get_full_d_argmax
from experiments.utils.acp_utils import aci_intervals, agaci_intervals, run_other_cp_methods
from experiments.utils.experiments_utils import normalize_interval_lengths


# Display names for plot titles (internal name -> paper name)
DISPLAY_NAMES = {
    'Toy_og': 'Toy',
}

def _display_name(dataname):
    return DISPLAY_NAMES.get(dataname, dataname)


def run_task3_for_dataset(dataname, args):
    """Run Task 3 UQ analysis for a single dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataname}")
    print(f"{'='*60}")

    # Create args for ds3m
    class DS3MArgs:
        def __init__(self):
            self.problem = dataname
            self.seed = args.seed
            self.data_dir = None
            self.aci_train_size = args.aci_train_size
            self.alpha = args.alpha
            self.tab_gamma = args.tab_gamma
            self.aci_gamma = args.aci_gamma
            self.agaci_eta = args.agaci_eta
            self.agaci_lr_schedule = args.agaci_lr_schedule
            self.agaci_width_penalty = args.agaci_width_penalty

    ds3m_args = DS3MArgs()

    try:
        # Load DS3M data
        ds = load_ds3m_data(ds3m_args)

        # Load model
        model = load_ds3m_model(
            ds["directoryBest"],
            ds["x_dim"], ds["y_dim"], ds["h_dim"], ds["z_dim"],
            ds["d_dim"], ds["n_layers"], ds["learning_rate"],
            ds["device"], bidirection=ds.get("bidirection", False),
        )

        # Get forecasts
        res, testForecast_mean, testOriginal, size, d_argmax_test, uq, lq = forecast(
            model, ds["testX"], ds["testY"],
            ds["moments"], ds["d_dim"], ds["means"], ds["trend"],
            ds["test_len"], ds["freq"], ds["RawDataOriginal"],
            remove_mean=ds["remove_mean"],
            remove_residual=ds["remove_residual"],
        )

        print(f"  DS3M RMSE: {res['rmse']:.2f}")

    except Exception as e:
        print(f"  Failed to load DS3M: {e}")
        return None

    # Get target dimension data
    target_dim = int(ds.get("target_dim", 0))
    test_len = int(ds["test_len"])

    y_true = np.asarray(testOriginal)
    if y_true.ndim > 1:
        y_true = y_true[:, target_dim]

    y_lq_ds3m = np.asarray(lq)
    y_uq_ds3m = np.asarray(uq)
    if y_lq_ds3m.ndim > 1:
        y_lq_ds3m = y_lq_ds3m[:, target_dim]
    if y_uq_ds3m.ndim > 1:
        y_uq_ds3m = y_uq_ds3m[:, target_dim]

    # Get full regime sequence
    try:
        d_argmax_full = get_full_d_argmax(model, ds)
        d_argmax_test = d_argmax_full[-test_len:]
    except:
        d_argmax_test = d_argmax_test

    # Prepare data for ACI/AgACI
    y_full = np.asarray(ds["data"])
    if y_full.ndim == 1:
        y_full = y_full.reshape(-1, 1)
    N = len(y_full)
    X_dummy = np.zeros((N, 1), dtype=float)

    # Run ACI
    print("  Running ACI...")
    y_lowers_aci, y_uppers_aci, tab_alpha_t_aci, gammas_aci = aci_intervals(
        X_dummy, y_full, args=ds3m_args
    )

    # Run AgACI
    print("  Running AgACI...")
    agaci_results = agaci_intervals(
        X_dummy, y_full, basemodel="ds3m", args=ds3m_args
    )

    agaci_lower = agaci_results['lower']
    agaci_upper = agaci_results['upper']

    print(f"  AgACI Coverage: {agaci_results['coverage']:.3f}")
    print(f"  AgACI Median Length: {agaci_results['median_length']:.2f}")

    # Run other CP methods
    print("  Running other CP methods...")
    try:
        other_cp_results = run_other_cp_methods(
            X_dummy, y_full,
            methods=['Gaussian', 'CP', 'EnbPI'],
            basemodel="ds3m",
            params_basemodel=None,
            args=ds3m_args
        )
    except Exception as e:
        print(f"  Warning: Other CP methods failed: {e}")
        other_cp_results = {}

    # Align intervals to test period
    T0 = args.aci_train_size
    test_size_eff = len(agaci_lower)

    # Pad with NaN
    agaci_lower_full = np.full(test_len, np.nan)
    agaci_upper_full = np.full(test_len, np.nan)
    agaci_lower_full[T0:T0+test_size_eff] = agaci_lower
    agaci_upper_full[T0:T0+test_size_eff] = agaci_upper

    # Fixed-gamma ACI (no oracle selection — fair comparison with AgACI)
    gamma_values = gammas_aci[:, 0]
    best_idx = int(np.argmin(np.abs(gamma_values - args.aci_gamma)))
    best_gamma_val = float(gamma_values[best_idx])
    print(f"  ACI fixed gamma: {best_gamma_val}")

    aci_lower_full = np.full(test_len, np.nan)
    aci_upper_full = np.full(test_len, np.nan)
    aci_lower_full[T0:T0+test_size_eff] = y_lowers_aci[best_idx]
    aci_upper_full[T0:T0+test_size_eff] = y_uppers_aci[best_idx]

    # Build intervals dict
    intervals_dict = {
        'Naive': (y_lq_ds3m, y_uq_ds3m),
        'ACI': (aci_lower_full, aci_upper_full),
        'AgACI': (agaci_lower_full, agaci_upper_full),
    }

    # Add other methods
    for method_name, (lower, upper) in other_cp_results.items():
        lower_full = np.full(test_len, np.nan)
        upper_full = np.full(test_len, np.nan)
        method_test_size = len(lower)
        lower_full[T0:T0+method_test_size] = lower
        upper_full[T0:T0+method_test_size] = upper
        intervals_dict[method_name] = (lower_full, upper_full)

    # Compute results for all methods
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

    # Generate plots
    display = _display_name(dataname)

    # 1. Coverage vs Length trade-off
    plot_tradeoff(results_dict, display, OUTPUT_DIR / f"{dataname}_tradeoff.png")

    # 2. Interval length at switches
    plot_length_at_switches(
        intervals_dict, y_true, d_argmax_test,
        OUTPUT_DIR / f"{dataname}_length_at_switches.png"
    )

    # 3. Coverage at switches
    plot_coverage_at_switches(
        intervals_dict, y_true, d_argmax_test,
        OUTPUT_DIR / f"{dataname}_coverage_at_switches.png"
    )

    # 4. Recovery comparison
    plot_recovery_comparison(
        intervals_dict, y_true, d_argmax_test,
        target_coverage=1.0 - args.alpha,
        save_path=OUTPUT_DIR / f"{dataname}_recovery_comparison.png"
    )

    return {
        'dataset': dataname,
        'agaci_coverage': agaci_results['coverage'],
        'agaci_median_length': agaci_results['median_length'],
        'results': results_dict,
    }


def plot_tradeoff(results_dict, dataname, save_path):
    """Plot coverage vs interval length trade-off."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        'Naive': '#95a5a6',
        'Gaussian': '#3498db',
        'CP': '#2ecc71',
        'EnbPI': '#9b59b6',
        'ACI': '#e67e22',
        'AgACI': '#e74c3c',
    }
    markers = {
        'Naive': 'o',
        'Gaussian': 's',
        'CP': '^',
        'EnbPI': 'v',
        'ACI': 'D',
        'AgACI': '*',
    }

    for method, (coverage, length) in results_dict.items():
        ax.scatter(length, coverage,
                   c=colors.get(method, 'gray'),
                   marker=markers.get(method, 'o'),
                   s=150 if method == 'AgACI' else 100,
                   label=f'{method} ({coverage:.1%}, {length:.1f})',
                   edgecolors='black' if method == 'AgACI' else 'none',
                   linewidths=2 if method == 'AgACI' else 0,
                   zorder=10 if method == 'AgACI' else 5)

    # Add 90% target line
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90% target')

    ax.set_xlabel('Median Interval Length', fontsize=12)
    ax.set_ylabel('Coverage', fontsize=12)
    ax.set_title(f'{dataname}: Coverage vs Interval Length Trade-off', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_length_at_switches(intervals_dict, y_true, d_argmax, save_path, window=10):
    """Plot interval length around regime switches."""
    # Find switch points
    switch_indices = np.where(np.diff(d_argmax) != 0)[0] + 1

    if len(switch_indices) == 0:
        print(f"  No switches found, skipping length plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'Naive': '#95a5a6',
        'Gaussian': '#3498db',
        'CP': '#2ecc71',
        'EnbPI': '#9b59b6',
        'ACI': '#e67e22',
        'AgACI': '#e74c3c',
    }

    t_rel = np.arange(-window, window + 1)

    for method_name, (lower, upper) in intervals_dict.items():
        lengths = upper - lower
        avg_lengths = []

        for t in t_rel:
            vals = []
            for sw in switch_indices:
                idx = sw + t
                if 0 <= idx < len(lengths) and not np.isnan(lengths[idx]):
                    vals.append(lengths[idx])
            avg_lengths.append(np.mean(vals) if vals else np.nan)

        ax.plot(t_rel, avg_lengths,
                color=colors.get(method_name, 'gray'),
                marker='o' if method_name == 'AgACI' else '',
                linewidth=2.5 if method_name == 'AgACI' else 1.5,
                alpha=0.9 if method_name == 'AgACI' else 0.7,
                label=method_name)

    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Switch point')
    ax.set_xlabel('Time relative to switch (t=0 is switch)', fontsize=12)
    ax.set_ylabel('Average Interval Length', fontsize=12)
    ax.set_title(f'Interval Length Dynamics Around Regime Switches\n({len(switch_indices)} switches)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_coverage_at_switches(intervals_dict, y_true, d_argmax, save_path, window=10):
    """Plot coverage around regime switches."""
    switch_indices = np.where(np.diff(d_argmax) != 0)[0] + 1

    if len(switch_indices) == 0:
        print(f"  No switches found, skipping coverage plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'Naive': '#95a5a6',
        'Gaussian': '#3498db',
        'CP': '#2ecc71',
        'EnbPI': '#9b59b6',
        'ACI': '#e67e22',
        'AgACI': '#e74c3c',
    }

    t_rel = np.arange(-window, window + 1)

    for method_name, (lower, upper) in intervals_dict.items():
        avg_coverage = []

        for t in t_rel:
            covered = []
            for sw in switch_indices:
                idx = sw + t
                if 0 <= idx < len(y_true) and not np.isnan(lower[idx]) and not np.isnan(upper[idx]):
                    covered.append(1 if lower[idx] <= y_true[idx] <= upper[idx] else 0)
            avg_coverage.append(np.mean(covered) if covered else np.nan)

        ax.plot(t_rel, avg_coverage,
                color=colors.get(method_name, 'gray'),
                marker='o' if method_name == 'AgACI' else '',
                linewidth=2.5 if method_name == 'AgACI' else 1.5,
                alpha=0.9 if method_name == 'AgACI' else 0.7,
                label=method_name)

    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Switch point')
    ax.axhline(y=0.9, color='red', linestyle=':', alpha=0.5, label='90% target')
    ax.set_xlabel('Time relative to switch (t=0 is switch)', fontsize=12)
    ax.set_ylabel('Coverage Rate', fontsize=12)
    ax.set_title(f'Coverage Dynamics Around Regime Switches\n({len(switch_indices)} switches)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_recovery_comparison(intervals_dict, y_true, d_argmax, target_coverage=0.9, save_path=None, window=20):
    """Plot recovery metrics after regime switches."""
    switch_indices = np.where(np.diff(d_argmax) != 0)[0] + 1

    if len(switch_indices) == 0:
        print(f"  No switches found, skipping recovery plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'Naive': '#95a5a6',
        'Gaussian': '#3498db',
        'CP': '#2ecc71',
        'EnbPI': '#9b59b6',
        'ACI': '#e67e22',
        'AgACI': '#e74c3c',
    }

    recovery_threshold = 0.85 * target_coverage

    recovery_times = {}
    for method_name, (lower, upper) in intervals_dict.items():
        # Compute rolling coverage after switches
        t_post = np.arange(0, window + 1)
        avg_coverage_post = []

        for t in t_post:
            covered = []
            for sw in switch_indices:
                idx = sw + t
                if 0 <= idx < len(y_true) and not np.isnan(lower[idx]) and not np.isnan(upper[idx]):
                    covered.append(1 if lower[idx] <= y_true[idx] <= upper[idx] else 0)
            avg_coverage_post.append(np.mean(covered) if covered else np.nan)

        # Find recovery time
        recovery_time = None
        for i, cov in enumerate(avg_coverage_post):
            if cov is not None and not np.isnan(cov) and cov >= recovery_threshold:
                recovery_time = i
                break

        recovery_times[method_name] = recovery_time if recovery_time is not None else window

        ax.plot(t_post, avg_coverage_post,
                color=colors.get(method_name, 'gray'),
                marker='o' if method_name == 'AgACI' else '',
                linewidth=2.5 if method_name == 'AgACI' else 1.5,
                alpha=0.9 if method_name == 'AgACI' else 0.7,
                label=f'{method_name} (rec={recovery_times[method_name]})')

    ax.axhline(y=target_coverage, color='red', linestyle='--', alpha=0.5, label=f'{target_coverage:.0%} target')
    ax.axhline(y=recovery_threshold, color='orange', linestyle=':', alpha=0.5, label=f'{recovery_threshold:.0%} threshold')
    ax.set_xlabel('Time steps after switch', fontsize=12)
    ax.set_ylabel('Coverage Rate', fontsize=12)
    ax.set_title(f'Coverage Recovery After Regime Switches\n({len(switch_indices)} switches)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def create_summary_table(results):
    """Create and save summary table."""
    rows = []
    for r in results:
        row = {
            'Dataset': r['dataset'],
            'AgACI Coverage': f"{r['agaci_coverage']:.1%}",
            'AgACI Med. Length': f"{r['agaci_median_length']:.1f}",
        }
        # Add other methods
        for method, (cov, length) in r['results'].items():
            if method != 'AgACI':
                row[f'{method} Cov'] = f"{cov:.1%}"
                row[f'{method} Len'] = f"{length:.1f}"
        rows.append(row)

    df = pd.DataFrame(rows)

    print("\n" + "="*80)
    print("Task 3: UQ Summary")
    print("="*80)
    print(df.to_string(index=False))

    # Save to CSV
    csv_path = OUTPUT_DIR / "task3_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary: {csv_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Task 3: UQ Analysis")
    parser.add_argument("--datasets", nargs="+",
                        default=["Toy_og", "Sleep", "Unemployment", "Electricity", "Lorenz"],
                        help="Datasets to process")
    parser.add_argument("--aci-train-size", type=int, default=20,
                        help="ACI calibration window size")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Target miscoverage rate (1-alpha = coverage)")
    parser.add_argument("--tab-gamma", type=float, nargs="*",
                        default=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
                        help="Gamma values for ACI experts (wide grid for diverse experts)")
    parser.add_argument("--aci-gamma", type=float, default=0.01,
                        help="Fixed gamma for single-ACI (Gibbs & Candes 2021)")
    parser.add_argument("--agaci-eta", type=float, default=0.5,
                        help="AgACI BOA learning rate")
    parser.add_argument("--agaci-lr-schedule", type=str, default="constant",
                        help="AgACI learning rate schedule")
    parser.add_argument("--agaci-width-penalty", type=float, default=0.1,
                        help="Width penalty in IS loss (encourages narrower intervals)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("="*70)
    print("Task 3: Uncertainty Quantification Analysis")
    print("="*70)
    print(f"Datasets: {args.datasets}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Target coverage: {1 - args.alpha:.0%}")
    print("="*70)

    results = []
    for dataname in args.datasets:
        try:
            result = run_task3_for_dataset(dataname, args)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    if results:
        create_summary_table(results)

    print(f"\n{'='*70}")
    print(f"Done! All figures saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
