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
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import argparse
from typing import Dict

# Add project root to path
HERE = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

# Output directory
OUTPUT_DIR = Path("overleaf_upload/figures/task2_cpd")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Custom low-saturation red-blue colormap
CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    'soft_redblue',
    ['#6090c0', '#d67575'],  # Even deeper soft blue to even deeper soft red
    N=256
)

# Import after path setup
from experiments.utils.ds3m_utils import load_ds3m_data, load_ds3m_model, forecast
from experiments.utils.experiments_utils import load_forecast

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    print("Warning: ruptures not installed. Run: pip install ruptures")


def run_ruptures_cpd(data, test_len, method='Binseg', model='l2', penalty=None, min_size=5, n_bkps=5):
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
                # Use BIC-like penalty for balanced detection
                penalty = np.log(len(data_norm)) * 0.3
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


def _switches_from_labels(labels: np.ndarray) -> np.ndarray:
    """Return switch indices (where regime label changes)."""
    labels = np.asarray(labels).astype(int)
    return np.where(np.diff(labels) != 0)[0] + 1


def cpd_precision_recall_f1(
    true_switches: np.ndarray,
    pred_switches: np.ndarray,
    tol: int = 5,
) -> Dict[str, float]:
    """
    Compute precision/recall/F1 for changepoint detection with tolerance window.

    Uses greedy one-to-one matching: each predicted switch is matched to the
    closest unmatched true switch within ±tol time steps.

    Parameters
    ----------
    true_switches : np.ndarray
        Ground truth switch indices
    pred_switches : np.ndarray
        Predicted switch indices
    tol : int
        Tolerance window (±tol time steps)

    Returns
    -------
    metrics : dict
        Dictionary with precision, recall, f1, tp, fp, fn
    """
    true_switches = np.asarray(true_switches, dtype=int)
    pred_switches = np.asarray(pred_switches, dtype=int)

    # Edge cases
    if true_switches.size == 0 and pred_switches.size == 0:
        return {
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "tp": 0.0, "fp": 0.0, "fn": 0.0
        }
    if true_switches.size == 0:
        return {
            "precision": 0.0, "recall": 1.0, "f1": 0.0,
            "tp": 0.0, "fp": float(len(pred_switches)), "fn": 0.0
        }
    if pred_switches.size == 0:
        return {
            "precision": 1.0, "recall": 0.0, "f1": 0.0,
            "tp": 0.0, "fp": 0.0, "fn": float(len(true_switches))
        }

    # Greedy matching
    used_true = np.zeros(len(true_switches), dtype=bool)
    tp = 0

    for p in pred_switches:
        # Find closest unmatched true switch
        distances = np.abs(true_switches - p)
        distances[used_true] = 10**9  # Mark used as infinitely far
        j = int(np.argmin(distances))

        if distances[j] <= tol:
            used_true[j] = True
            tp += 1

    fp = len(pred_switches) - tp
    fn = len(true_switches) - tp

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn)
    }


def plot_ds3m_vs_ruptures_heatmap(dataname, d_argmax, ruptures_results_dict, test_len, save_path, d_true=None, metrics=None):
    """
    Create regime heatmap comparison: Ground Truth (if available) + DS3M + Ruptures methods.

    Parameters
    ----------
    ruptures_results_dict : dict
        Dictionary mapping method names (e.g., 'Binseg', 'Pelt') to breakpoints
    """
    import seaborn as sns

    has_gt = d_true is not None
    n_ruptures_methods = len(ruptures_results_dict) if ruptures_results_dict else 0
    n_rows = (1 if has_gt else 0) + 1 + n_ruptures_methods  # GT + DS3M + Ruptures methods

    ratios = [1] * n_rows
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 1.5 * n_rows), sharex=True,
                             gridspec_kw={'height_ratios': ratios, 'hspace': 0.3})

    # Handle single row case
    if n_rows == 1:
        axes = [axes]

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
        cmap_gt = CUSTOM_CMAP
        sns.heatmap(gt_norm.reshape(1, -1), ax=ax_gt, cbar=False,
                    cmap=cmap_gt, vmin=0, vmax=1, linewidth=0)
        n_switches_gt = int(np.sum(np.diff(gt) != 0))

        # Add metrics info to title if available
        extra = ""
        if metrics is not None and "gt" in metrics:
            m = metrics["gt"]
            extra = f" (GT: {int(m.get('n_switches', n_switches_gt))})"

        ax_gt.set_title(f'{dataname} | Ground Truth (Switches: {n_switches_gt}){extra}',
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

    cmap = CUSTOM_CMAP
    sns.heatmap(arr_normalized.reshape(1, -1), ax=ax1, cbar=False,
                cmap=cmap, vmin=0, vmax=1, linewidth=0)

    n_switches_ds3m = np.sum(np.diff(d_argmax) != 0)

    # Add metrics to title if available
    extra = ""
    if metrics is not None and "ds3m" in metrics and has_gt:
        m = metrics["ds3m"]
        extra = f" | P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}"

    ax1.set_title(f'{dataname} | DS3M discrete states (Switches: {n_switches_ds3m}){extra}',
                  fontsize=12, fontweight='bold')
    ax1.set_yticks([])
    ax1.set_ylabel('DS3M')
    ax1.set_xticks([])
    row_idx += 1

    # --- Ruptures Regime Heatmaps (one per method) ---
    if ruptures_results_dict:
        for method_name, ruptures_bkps in ruptures_results_dict.items():
            ax_rup = axes[row_idx]

            if ruptures_bkps is not None and len(ruptures_bkps) > 0:
                # Create regime labels - ALTERNATE between 0 and 1 (red/blue only)
                ruptures_regimes = np.zeros(T, dtype=int)
                regime_id = 0
                prev_bp = 0

                for bp in ruptures_bkps:
                    if bp > T:
                        bp = T
                    if bp > prev_bp:  # Only assign if segment is non-empty
                        ruptures_regimes[prev_bp:bp] = regime_id % 2  # Alternate: 0, 1, 0, 1, ...
                        regime_id += 1
                    prev_bp = bp

                # Count actual switches (transitions between segments)
                n_switches_ruptures = int(np.sum(np.diff(ruptures_regimes) != 0))

                # Simple binary coloring (red/blue only)
                ruptures_normalized = 1 - ruptures_regimes  # 0->1 (blue), 1->0 (red)
                cmap_rup = CUSTOM_CMAP  # Always use 2 colors
                sns.heatmap(ruptures_normalized.reshape(1, -1), ax=ax_rup, cbar=False,
                            cmap=cmap_rup, vmin=0, vmax=1, linewidth=0)

                # Add metrics to title if available
                extra = ""
                metrics_key = f"ruptures_{method_name.lower()}"
                if metrics is not None and metrics_key in metrics and has_gt:
                    m = metrics[metrics_key]
                    extra = f" | P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}"

                ax_rup.set_title(f'{dataname} | Ruptures ({method_name}) discrete states (Switches: {n_switches_ruptures}){extra}',
                          fontsize=12, fontweight='bold')
            else:
                # No ruptures result
                ax_rup.text(0.5, 0.5, f'Ruptures {method_name} not available', ha='center', va='center',
                        transform=ax_rup.transAxes, fontsize=12)
                ax_rup.set_title(f'{dataname} | Ruptures ({method_name}) discrete states',
                          fontsize=12, fontweight='bold')

            ax_rup.set_yticks([])
            ax_rup.set_ylabel(f'{method_name}')

            # Only add x-axis to last row
            if row_idx == n_rows - 1:
                ax_rup.set_xlabel('time')
                tick_interval = max(T // 10, 1)
                tick_positions = np.arange(0, T, tick_interval)
                ax_rup.set_xticks(tick_positions)
                ax_rup.set_xticklabels(tick_positions, rotation=0, fontsize=9)
            else:
                ax_rup.set_xticks([])

            row_idx += 1

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

    # Run Ruptures on FULL series (not just test tail)
    # This is the FIX for the normalization issue
    # Run BOTH Pelt and Binseg for comparison
    ruptures_results = {}
    full_series_1d = None

    if RUPTURES_AVAILABLE:
        methods_str = "Pelt + Binseg" if dataname.startswith("Toy") else "Pelt"
        print(f"  Running Ruptures CPD ({methods_str})...")

        # FIX: Use full series from cache, not just test tail
        # This ensures ruptures has proper train/test split for normalization
        if cached is not None and 'data' in cached and cached['data'] is not None:
            full_data = cached['data']
            if full_data.ndim > 1:
                full_series_1d = full_data[:, 0]
            else:
                full_series_1d = full_data
            print(f"    Using full series: {len(full_series_1d)} points (test_len={test_len})")
        else:
            # Fallback: use y_true if full data not available
            print(f"    Warning: Full series not in cache, using y_true (may cause normalization issues)")
            if y_true.ndim > 1:
                full_series_1d = y_true[:, 0]
            else:
                full_series_1d = y_true

        # Determine which methods to run based on dataset
        if dataname.startswith("Toy"):
            # Toy: run both for comparison
            methods_to_run = ['Binseg', 'Pelt']
        else:
            # All real datasets: use Pelt
            methods_to_run = ['Pelt']

        # Run selected methods
        for method_name in methods_to_run:
            # Use more breakpoints for better detection
            n_bkps_to_use = 80 if method_name == 'Binseg' else 10

            result = run_ruptures_cpd(
                full_series_1d,
                test_len=test_len,
                method=method_name,
                model='l2',
                penalty=args.ruptures_penalty,
                min_size=args.ruptures_min_size,
                n_bkps=n_bkps_to_use,
            )

            if result:
                ruptures_results[method_name] = result
                print(f"    {method_name}: {result['n_switches']} change points total, "
                      f"{result['switches_in_test']} in test")

    # Compute accuracy metrics for Toy dataset
    metrics = None
    d_argmax_test = ds3m_result['d_argmax']
    T_test = len(d_argmax_test)

    # Load ground truth regime labels for Toy
    d_true = None
    if dataname.startswith("Toy"):
        # Try multiple Toy variants - cached forecast may be from different variant
        toy_variants = [
            "Toy_og",  # Original Toy dataset (24 switches in test)
            "Toy_exp1",  # Toy exp1 (1 switch in test)
            "Toy_exp2_V1_0.5_V2_2.0",  # Toy exp2
        ]

        for variant in toy_variants:
            gt_path = Path(f"Deep_Switching_State_Space_Model/data/{variant}/simulation_data_nonlinear_d.csv")
            if gt_path.exists():
                d_true_candidate = np.loadtxt(gt_path).astype(int)
                # Check if length matches cached data
                if cached and 'data' in cached:
                    expected_len = len(cached['data'])
                    if len(d_true_candidate) >= expected_len:
                        d_true = d_true_candidate
                        test_switches = int(np.sum(np.diff(d_true[-T_test:]) != 0))
                        print(f"  Loaded ground truth from {variant}: {len(d_true)} points, "
                              f"{int(np.sum(np.diff(d_true)!=0))} total switches, "
                              f"{test_switches} switches in test")
                        break

    # Determine evaluation window: full series or test-only
    eval_full_series = getattr(args, "eval_full_series", False)

    if d_true is not None:
        if eval_full_series:
            # Evaluate on FULL series (train + test) for better statistics
            # Use cached d_argmax for full series if available
            if cached and 'states' in cached and cached['states'] is not None:
                # states contains full inference results
                # For now, use test-only but mark this as TODO
                print("  Warning: --eval-full-series requested but full DS3M states not in cache.")
                print("           Falling back to test-only evaluation.")
                eval_full_series = False

        if eval_full_series:
            # TODO: Implement full-series evaluation
            # Would need full DS3M d_argmax (not just test)
            pass
        else:
            # Test-only evaluation (current behavior)
            d_true_test = np.asarray(d_true)[-T_test:]
            true_switches = _switches_from_labels(d_true_test)
            ds3m_switches = _switches_from_labels(d_argmax_test)

            metrics = {
                "gt": {"n_switches": float(len(true_switches))}
            }

            # DS3M metrics
            metrics["ds3m"] = cpd_precision_recall_f1(
                true_switches, ds3m_switches, tol=getattr(args, "cpt_tol", 5)
            )
            print(f"  DS3M metrics (test-only, n={len(true_switches)}): "
                  f"P={metrics['ds3m']['precision']:.3f}, R={metrics['ds3m']['recall']:.3f}, "
                  f"F1={metrics['ds3m']['f1']:.3f}")

    # Ruptures metrics (convert absolute breakpoints to test-relative indices)
    # Process each ruptures method
    ruptures_bkps_test_relative_dict = {}

    if d_true is not None and ruptures_results and full_series_1d is not None:
        N_total = len(full_series_1d)
        test_start_idx = N_total - T_test
        true_switches = _switches_from_labels(np.asarray(d_true)[-T_test:])

        for method_name, ruptures_result in ruptures_results.items():
            # Filter breakpoints in test window
            ruptures_bkps_abs = np.array(ruptures_result['breakpoints'][:-1], dtype=int)
            ruptures_bkps_in_test = ruptures_bkps_abs[ruptures_bkps_abs >= test_start_idx]

            # Convert to test-relative indices
            ruptures_switches_test_relative = ruptures_bkps_in_test - test_start_idx

            # Compute metrics
            metrics_key = f"ruptures_{method_name.lower()}"
            metrics[metrics_key] = cpd_precision_recall_f1(
                true_switches, ruptures_switches_test_relative, tol=getattr(args, "cpt_tol", 5)
            )
            print(f"  {method_name} metrics (test-only, n={len(true_switches)}): "
                  f"P={metrics[metrics_key]['precision']:.3f}, R={metrics[metrics_key]['recall']:.3f}, "
                  f"F1={metrics[metrics_key]['f1']:.3f}")

            # For plotting, convert ALL breakpoints to test-relative
            ruptures_bkps_test_relative = []
            for bp in ruptures_result['breakpoints']:
                if bp > test_start_idx:
                    ruptures_bkps_test_relative.append(bp - test_start_idx)
            if ruptures_bkps_test_relative and ruptures_bkps_test_relative[-1] != T_test:
                ruptures_bkps_test_relative.append(T_test)  # Add final endpoint if not present

            ruptures_bkps_test_relative_dict[method_name] = ruptures_bkps_test_relative

    elif ruptures_results:
        # No ground truth - just convert breakpoints for plotting
        N_total = len(full_series_1d) if full_series_1d is not None else T_test
        test_start_idx = N_total - T_test

        for method_name, ruptures_result in ruptures_results.items():
            ruptures_bkps_test_relative = []
            for bp in ruptures_result['breakpoints']:
                if bp > test_start_idx:
                    ruptures_bkps_test_relative.append(bp - test_start_idx)
            if ruptures_bkps_test_relative and ruptures_bkps_test_relative[-1] != T_test:
                ruptures_bkps_test_relative.append(T_test)

            ruptures_bkps_test_relative_dict[method_name] = ruptures_bkps_test_relative

    # Generate plots
    plot_ds3m_vs_ruptures_heatmap(
        dataname,
        d_argmax_test,
        ruptures_bkps_test_relative_dict,
        test_len,
        OUTPUT_DIR / f"{dataname}_regime_comparison.png",
        d_true=d_true,
        metrics=metrics,
    )

    # Compute and return metrics
    n_switches_ds3m = np.sum(np.diff(d_argmax_test) != 0)

    result = {
        'dataset': dataname,
        'ds3m_switches': n_switches_ds3m,
    }

    # Add ruptures switches for each method
    for method_name in ['Binseg', 'Pelt']:
        if method_name in ruptures_results:
            n_switches = ruptures_results[method_name]['switches_in_test']
            result[f'ruptures_{method_name.lower()}_switches'] = n_switches
        else:
            result[f'ruptures_{method_name.lower()}_switches'] = 0

    # Add Toy metrics to result
    if metrics is not None and "ds3m" in metrics:
        result.update({
            'toy_ds3m_precision': metrics["ds3m"]["precision"],
            'toy_ds3m_recall': metrics["ds3m"]["recall"],
            'toy_ds3m_f1': metrics["ds3m"]["f1"],
        })

    # Add ruptures metrics for each method
    for method_name in ['Binseg', 'Pelt']:
        metrics_key = f"ruptures_{method_name.lower()}"
        if metrics is not None and metrics_key in metrics:
            result.update({
                f'toy_{metrics_key}_precision': metrics[metrics_key]["precision"],
                f'toy_{metrics_key}_recall': metrics[metrics_key]["recall"],
                f'toy_{metrics_key}_f1': metrics[metrics_key]["f1"],
            })

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
    parser.add_argument("--ruptures-min-size", type=int, default=5,
                        help="Ruptures min segment size")
    parser.add_argument("--cpt-tol", type=int, default=5,
                        help="Toy CPD: match tolerance in time steps for precision/recall (default: ±5)")
    parser.add_argument("--eval-full-series", action="store_true",
                        help="For Toy: evaluate on full series (train+test) instead of test-only. "
                             "Gives n=77 switches instead of n=1 for better statistics.")
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
