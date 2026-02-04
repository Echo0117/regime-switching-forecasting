"""
Task 2: Change Point Detection - DS3M vs Ruptures Comparison

Generates comparison figures showing:
1. DS3M predictions with detected regimes
2. Ruptures predictions with detected change points
3. Side-by-side comparison metrics

Output: overleaf_upload/figures/task2_cpd/
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
OUTPUT_DIR = Path("overleaf_upload/figures/task2_cpd")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import after path setup
from experiments.utils.ds3m_utils import load_ds3m_data, load_ds3m_model, forecast
from experiments.utils.experiments_utils import load_forecast

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    print("Warning: ruptures not installed. Run: pip install ruptures")


def run_ruptures_cpd(data, test_len, method='Binseg', model='l2', penalty=None, min_size=10, n_bkps=5):
    """Run ruptures change point detection on full data and return breakpoints."""
    if not RUPTURES_AVAILABLE:
        return None

    train_data = data[:-test_len]

    # Normalize using training data statistics
    mean, std = train_data.mean(), train_data.std()
    if std < 1e-8:
        std = 1.0
    data_norm = (data - mean) / std

    # Run ruptures on full data - use Binseg by default as it's more robust
    try:
        if method == 'Binseg':
            algo = rpt.Binseg(model=model, min_size=min_size).fit(data_norm)
            bkps = algo.predict(n_bkps=n_bkps)
        elif method == 'BottomUp':
            algo = rpt.BottomUp(model=model, min_size=min_size).fit(data_norm)
            bkps = algo.predict(n_bkps=n_bkps)
        else:  # Pelt
            algo = rpt.Pelt(model=model, min_size=min_size).fit(data_norm)
            if penalty is None:
                penalty = np.log(len(data_norm)) * 2  # BIC-like penalty
            bkps = algo.predict(pen=penalty)
    except Exception as e:
        print(f"    Ruptures failed: {e}, using fallback")
        # Fallback: just use single segment
        bkps = [len(data_norm)]

    # Count switches in test period
    test_start = len(data) - test_len
    switches_in_test = sum(1 for bp in bkps[:-1] if bp > test_start)

    return {
        'breakpoints': bkps,
        'n_switches': len(bkps) - 1,
        'switches_in_test': switches_in_test,
    }


def plot_ds3m_vs_ruptures_heatmap(dataname, d_argmax, ruptures_bkps, test_len, save_path, d_true=None):
    """Create regime heatmap comparison: Ground Truth (top, if available) + DS3M + Ruptures."""
    import seaborn as sns

    has_gt = d_true is not None
    n_rows = 3 if has_gt else 2
    ratios = [1] * n_rows
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 1.5 * n_rows), sharex=True,
                             gridspec_kw={'height_ratios': ratios, 'hspace': 0.3})

    T = len(d_argmax)
    t = np.arange(T)

    # Get number of regimes
    d_dim = len(np.unique(d_argmax))

    row_idx = 0

    # --- Ground Truth (if available) ---
    if has_gt:
        ax_gt = axes[row_idx]
        gt = d_true[-T:].astype(int)  # align to same test window
        gt_dim = len(np.unique(gt))
        if gt_dim == 2:
            gt_norm = 1 - gt
        else:
            gt_norm = (gt_dim - 1 - gt) / (gt_dim - 1) if gt_dim > 1 else gt.astype(float)
        cmap_gt = plt.get_cmap('RdBu', max(2, gt_dim))
        sns.heatmap(gt_norm.reshape(1, -1), ax=ax_gt, cbar=False,
                    cmap=cmap_gt, vmin=0, vmax=1, linewidth=0)
        n_switches_gt = int(np.sum(np.diff(gt) != 0))
        ax_gt.set_title(f'{dataname} | Ground Truth (Switches: {n_switches_gt})',
                        fontsize=12, fontweight='bold')
        ax_gt.set_yticks([])
        ax_gt.set_ylabel('Truth')
        ax_gt.set_xticks([])
        row_idx += 1

    # --- DS3M Regime Heatmap ---
    ax1 = axes[row_idx]

    # Normalize for heatmap coloring (same as original regime heatmap)
    if d_dim == 2:
        arr_normalized = 1 - d_argmax
    else:
        arr_normalized = (d_dim - 1 - d_argmax) / (d_dim - 1) if d_dim > 1 else d_argmax

    cmap = plt.get_cmap('RdBu', max(2, d_dim))
    sns.heatmap(arr_normalized.reshape(1, -1), ax=ax1, cbar=False,
                cmap=cmap, vmin=0, vmax=1, linewidth=0)

    n_switches_ds3m = np.sum(np.diff(d_argmax) != 0)
    ax1.set_title(f'{dataname} | DS3M discrete states (Switches: {n_switches_ds3m})',
                  fontsize=12, fontweight='bold')
    ax1.set_yticks([])
    ax1.set_ylabel('DS3M')
    ax1.set_xticks([])
    row_idx += 1

    # --- Ruptures Regime Heatmap ---
    ax2 = axes[row_idx]

    if ruptures_bkps is not None and len(ruptures_bkps) > 0:
        # Create regime labels from breakpoints
        ruptures_regimes = np.zeros(T, dtype=int)
        regime_id = 0
        prev_bp = 0

        for bp in ruptures_bkps:
            if bp > T:
                bp = T
            ruptures_regimes[prev_bp:bp] = regime_id
            regime_id += 1
            prev_bp = bp

        # Normalize for coloring
        n_ruptures_regimes = len(np.unique(ruptures_regimes))
        if n_ruptures_regimes == 1:
            ruptures_normalized = np.zeros(T)
        elif n_ruptures_regimes == 2:
            ruptures_normalized = 1 - ruptures_regimes
        else:
            ruptures_normalized = (n_ruptures_regimes - 1 - ruptures_regimes) / (n_ruptures_regimes - 1)

        cmap_rup = plt.get_cmap('RdBu', max(2, n_ruptures_regimes))
        sns.heatmap(ruptures_normalized.reshape(1, -1), ax=ax2, cbar=False,
                    cmap=cmap_rup, vmin=0, vmax=1, linewidth=0)

        n_switches_ruptures = len(ruptures_bkps) - 1
        ax2.set_title(f'{dataname} | Ruptures (Binseg) discrete states (Switches: {n_switches_ruptures})',
                      fontsize=12, fontweight='bold')
    else:
        # No ruptures result - show empty heatmap
        ax2.text(0.5, 0.5, 'Ruptures not available', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title(f'{dataname} | Ruptures (Binseg) discrete states',
                      fontsize=12, fontweight='bold')

    ax2.set_yticks([])
    ax2.set_ylabel('Ruptures')
    ax2.set_xlabel('time')

    # Add x-axis ticks at regular intervals
    tick_interval = max(T // 10, 1)
    tick_positions = np.arange(0, T, tick_interval)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_positions, rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def run_task2_for_dataset(dataname, args):
    """Run Task 2 comparison for a single dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataname}")
    print(f"{'='*60}")

    # Try to load cached DS3M forecast
    cached = load_forecast(dataname)

    if cached is None:
        print(f"  No cached DS3M forecast found for {dataname}")
        print(f"  Attempting to load and run DS3M...")

        # Create args for ds3m
        class DS3MArgs:
            def __init__(self):
                self.problem = dataname
                self.seed = 42
                self.data_dir = None

        ds3m_args = DS3MArgs()

        try:
            ds = load_ds3m_data(ds3m_args)
            model = load_ds3m_model(
                ds["directoryBest"],
                ds["x_dim"], ds["y_dim"], ds["h_dim"], ds["z_dim"],
                ds["d_dim"], ds["n_layers"], ds["learning_rate"],
                ds["device"], bidirection=ds.get("bidirection", False),
            )

            res, y_pred, y_true, size, d_argmax, uq, lq = forecast(
                model, ds["testX"], ds["testY"],
                ds["moments"], ds["d_dim"], ds["means"], ds["trend"],
                ds["test_len"], ds["freq"], ds["RawDataOriginal"],
                remove_mean=ds["remove_mean"],
                remove_residual=ds["remove_residual"],
            )

            ds3m_result = {
                'y_true': y_true,
                'y_pred': y_pred,
                'd_argmax': d_argmax,
                'd_dim': ds["d_dim"],
                'uq': uq,
                'lq': lq,
                'test_len': ds["test_len"],
            }

        except Exception as e:
            print(f"  Failed to run DS3M: {e}")
            return None
    else:
        print(f"  Loaded cached DS3M forecast")

        # Get d_dim safely
        d_dim_val = cached.get('d_dim', None)
        d_argmax = cached.get('d_argmax', None)

        if d_dim_val is None or d_dim_val == np.array(None):
            # Infer from d_argmax
            if d_argmax is not None:
                d_dim_val = len(np.unique(d_argmax))
            else:
                d_dim_val = 2  # default fallback

        ds3m_result = {
            'y_true': cached['y_true'],
            'y_pred': cached['y_pred_mean'],
            'd_argmax': d_argmax if d_argmax is not None else np.zeros(len(cached['y_true']), dtype=int),
            'd_dim': int(d_dim_val),
            'uq': cached.get('y_uq'),
            'lq': cached.get('y_lq'),
            'test_len': int(cached['test_len']),
        }

    test_len = ds3m_result['test_len']

    # Apply offset correction for remove_residual datasets
    y_pred = ds3m_result['y_pred']
    y_true = ds3m_result['y_true']
    if y_pred.ndim > 1:
        y_pred_1d = y_pred[:, 0].copy()
        y_true_1d = y_true[:, 0]
    else:
        y_pred_1d = y_pred.copy()
        y_true_1d = y_true

    offset = y_pred_1d.mean() - y_true_1d.mean()
    y_range = y_true_1d.max() - y_true_1d.min()
    if y_range > 0 and abs(offset) > 0.1 * y_range:
        print(f"  Applying offset correction: {offset:.2f}")
        if y_pred.ndim > 1:
            ds3m_result['y_pred'] = y_pred.copy()
            ds3m_result['y_pred'][:, 0] -= offset
        else:
            ds3m_result['y_pred'] = y_pred - offset

    # Run Ruptures
    ruptures_result = None
    if RUPTURES_AVAILABLE:
        print(f"  Running Ruptures CPD...")
        # Get full data for ruptures
        if y_true.ndim > 1:
            data_1d = y_true[:, 0]
        else:
            data_1d = y_true

        ruptures_result = run_ruptures_cpd(
            data_1d,
            test_len=min(test_len, len(data_1d)),
            method='Binseg',
            model='l2',
            penalty=args.ruptures_penalty,
            min_size=args.ruptures_min_size,
            n_bkps=10,
        )

        if ruptures_result:
            print(f"    Ruptures detected {ruptures_result['n_switches']} change points")
            print(f"    Switches in test: {ruptures_result['switches_in_test']}")

    # Load ground truth regime labels for Toy
    d_true = None
    if dataname.startswith("Toy"):
        gt_path = Path("Deep_Switching_State_Space_Model/data/Toy_exp1/simulation_data_nonlinear_d.csv")
        if gt_path.exists():
            d_true = np.loadtxt(gt_path).astype(int)
            print(f"  Loaded ground truth regimes: {len(d_true)} points, {int(np.sum(np.diff(d_true)!=0))} switches")

    # Generate plots
    # DS3M vs Ruptures regime heatmap comparison
    ruptures_bkps = ruptures_result['breakpoints'] if ruptures_result else None
    plot_ds3m_vs_ruptures_heatmap(
        dataname,
        ds3m_result['d_argmax'],
        ruptures_bkps,
        test_len,
        OUTPUT_DIR / f"{dataname}_regime_comparison.png",
        d_true=d_true,
    )

    # Compute and return metrics
    n_switches_ds3m = np.sum(np.diff(ds3m_result['d_argmax']) != 0)

    result = {
        'dataset': dataname,
        'ds3m_switches': n_switches_ds3m,
    }

    if ruptures_result and ruptures_bkps:
        n_switches_ruptures = len(ruptures_bkps) - 1
        result['ruptures_switches'] = n_switches_ruptures
    else:
        result['ruptures_switches'] = 0

    return result


def create_summary_table(results):
    """Create and save summary table."""
    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("Task 2: DS3M vs Ruptures Summary")
    print("="*80)
    print(df.to_string(index=False))

    # Save to CSV
    csv_path = OUTPUT_DIR / "task2_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary: {csv_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Task 2: CPD Comparison")
    parser.add_argument("--datasets", nargs="+",
                        default=["Toy", "Sleep", "Unemployment", "Electricity", "Lorenz"],
                        help="Datasets to process")
    parser.add_argument("--ruptures-penalty", type=float, default=None,
                        help="Ruptures penalty (auto if None)")
    parser.add_argument("--ruptures-min-size", type=int, default=10,
                        help="Ruptures min segment size")
    args = parser.parse_args()

    print("="*70)
    print("Task 2: Change Point Detection - DS3M vs Ruptures")
    print("="*70)
    print(f"Datasets: {args.datasets}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*70)

    results = []
    for dataname in args.datasets:
        try:
            result = run_task2_for_dataset(dataname, args)
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
