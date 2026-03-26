"""
Task 4: Controlled Synthetic Studies

Two sub-experiments that use synthetic series with **known** switch points to
(i) confirm that adaptive methods react as predicted by theory, and
(ii) disentangle the distinct effects of mean shifts and variance shifts on
    interval calibration.

Experiment 1 – Heaviside mean shift
    Piecewise-constant signal, single mean shift at t=250, constant variance.
    Validates the basic ACI adaptation loop under the simplest possible regime
    change.

Experiment 2 – AR(1) variance shift
    AR(1) series with mean shift (mu1=-20 -> mu2=2) at t=250 and a sweep over
    V1, V2 in {0.5, 2, 10, 20}, producing 16 (V1,V2) combinations.
    Three findings:
      1. Pure mean shift  (V1=V2): adaptation helps but is not critical.
      2. Low->high variance (V1<<V2): hardest case; fast-gamma experts upweighted.
      3. High->low variance (V1>>V2): slow-gamma experts enable smooth contraction.

Output: overleaf_upload/figures/task4_controlled/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import argparse
import traceback

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
HERE = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

OUTPUT_DIR = Path("overleaf_upload/figures/task4_controlled")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Imports from project utilities
# ---------------------------------------------------------------------------
from experiments.utils.ds3m_utils import (
    load_ds3m_data, load_ds3m_model, forecast, get_full_d_argmax,
)
from experiments.utils.acp_utils import (
    aci_intervals, agaci_intervals, run_other_cp_methods,
)
from experiments.utils.experiments_utils import normalize_interval_lengths

# Soft red-blue colormap (consistent with Task 2)
CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    'soft_redblue', ['#6090c0', '#d67575'], N=256,
)

# ---------------------------------------------------------------------------
# Shared plotting style
# ---------------------------------------------------------------------------
METHOD_COLORS = {
    'Naive':    '#95a5a6',
    'Gaussian': '#3498db',
    'CP':       '#2ecc71',
    'EnbPI':    '#9b59b6',
    'ACI':      '#e67e22',
    'AgACI':    '#e74c3c',
}

METHOD_MARKERS = {
    'Naive':    'o',
    'Gaussian': 's',
    'CP':       '^',
    'EnbPI':    'v',
    'ACI':      'D',
    'AgACI':    '*',
}


# ===================================================================
# Helper: build DS3M args object
# ===================================================================
class DS3MArgs:
    """Lightweight namespace expected by ds3m_utils / acp_utils."""
    def __init__(self, problem, args):
        self.problem = problem
        self.seed = args.seed
        self.data_dir = None
        self.aci_train_size = args.aci_train_size
        self.alpha = args.alpha
        self.tab_gamma = args.tab_gamma
        self.aci_gamma = args.aci_gamma
        self.agaci_eta = args.agaci_eta
        self.agaci_lr_schedule = args.agaci_lr_schedule
        self.agaci_width_penalty = args.agaci_width_penalty


# ===================================================================
# Core pipeline: load DS3M, run UQ methods, collect results
# ===================================================================
def run_uq_pipeline(dataname, args):
    """
    Run DS3M + all UQ methods for *dataname*.

    Returns
    -------
    results : dict   with keys  y_true, d_argmax_test, intervals_dict,
                     metrics_dict, agaci_results, test_len
    None on failure.
    """
    ds3m_args = DS3MArgs(dataname, args)

    # ---- load DS3M ----
    try:
        ds = load_ds3m_data(ds3m_args)
        model = load_ds3m_model(
            ds["directoryBest"],
            ds["x_dim"], ds["y_dim"], ds["h_dim"], ds["z_dim"],
            ds["d_dim"], ds["n_layers"], ds["learning_rate"],
            ds["device"], bidirection=ds.get("bidirection", False),
        )
        res, y_pred, y_true, size, d_argmax_test, uq, lq = forecast(
            model, ds["testX"], ds["testY"],
            ds["moments"], ds["d_dim"], ds["means"], ds["trend"],
            ds["test_len"], ds["freq"], ds["RawDataOriginal"],
            remove_mean=ds["remove_mean"],
            remove_residual=ds["remove_residual"],
        )
        print(f"  DS3M RMSE: {res['rmse']:.2f}")
    except Exception as e:
        print(f"  Failed to load DS3M for {dataname}: {e}")
        return None

    target_dim = int(ds.get("target_dim", 0))
    test_len = int(ds["test_len"])

    # flatten to 1-D
    y_true_1d = np.asarray(y_true)
    if y_true_1d.ndim > 1:
        y_true_1d = y_true_1d[:, target_dim]

    y_lq_ds3m = np.asarray(lq)
    y_uq_ds3m = np.asarray(uq)
    if y_lq_ds3m.ndim > 1:
        y_lq_ds3m = y_lq_ds3m[:, target_dim]
    if y_uq_ds3m.ndim > 1:
        y_uq_ds3m = y_uq_ds3m[:, target_dim]

    y_pred_1d = np.asarray(y_pred)
    if y_pred_1d.ndim > 1:
        y_pred_1d = y_pred_1d[:, target_dim]

    # full regime sequence
    try:
        d_argmax_full = get_full_d_argmax(model, ds)
        d_argmax_test = d_argmax_full[-test_len:]
    except Exception:
        pass  # keep d_argmax_test from forecast

    # ---- prepare dummy X for ACI / AgACI ----
    y_full = np.asarray(ds["data"])
    if y_full.ndim == 1:
        y_full = y_full.reshape(-1, 1)
    N = len(y_full)
    X_dummy = np.zeros((N, 1), dtype=float)

    # ---- ACI ----
    print("  Running ACI...")
    y_lowers_aci, y_uppers_aci, tab_alpha_t_aci, gammas_aci = aci_intervals(
        X_dummy, y_full, args=ds3m_args,
    )

    # ---- AgACI ----
    print("  Running AgACI...")
    agaci_results = agaci_intervals(
        X_dummy, y_full, basemodel="ds3m", args=ds3m_args,
    )
    agaci_lower = agaci_results['lower']
    agaci_upper = agaci_results['upper']
    print(f"  AgACI Coverage: {agaci_results['coverage']:.3f}, "
          f"Med.Length: {agaci_results['median_length']:.2f}")

    # ---- other CP methods ----
    print("  Running other CP methods...")
    try:
        other_cp = run_other_cp_methods(
            X_dummy, y_full,
            methods=['Gaussian', 'CP', 'EnbPI'],
            basemodel="ds3m",
            params_basemodel=None,
            args=ds3m_args,
        )
    except Exception as e:
        print(f"  Warning: other CP methods failed: {e}")
        other_cp = {}

    # ---- align to test period ----
    T0 = args.aci_train_size
    test_size_eff = len(agaci_lower)

    def _pad(arr):
        out = np.full(test_len, np.nan)
        out[T0:T0 + len(arr)] = arr
        return out

    agaci_lower_full = _pad(agaci_lower)
    agaci_upper_full = _pad(agaci_upper)

    # Fixed-gamma ACI (no oracle selection — fair comparison with AgACI)
    aci_gamma_target = args.aci_gamma
    gamma_values = gammas_aci[:, 0]  # first column = gamma value per expert
    best_idx = int(np.argmin(np.abs(gamma_values - aci_gamma_target)))
    print(f"  ACI fixed gamma: {float(gamma_values[best_idx])}")

    aci_lower_full = _pad(y_lowers_aci[best_idx])
    aci_upper_full = _pad(y_uppers_aci[best_idx])

    # ---- build intervals dict ----
    intervals_dict = {
        'Naive': (y_lq_ds3m, y_uq_ds3m),
        'ACI':   (aci_lower_full, aci_upper_full),
        'AgACI': (agaci_lower_full, agaci_upper_full),
    }
    for name, (lo, hi) in other_cp.items():
        intervals_dict[name] = (_pad(lo), _pad(hi))

    # ---- compute per-method metrics ----
    metrics_dict = {}
    for name, (lo, hi) in intervals_dict.items():
        valid = ~np.isnan(lo) & ~np.isnan(hi)
        if not np.any(valid):
            continue
        yt = y_true_1d[valid]
        coverage = float(np.mean((yt >= lo[valid]) & (yt <= hi[valid])))
        med_len  = float(np.median(hi[valid] - lo[valid]))
        metrics_dict[name] = (coverage, med_len)

    return dict(
        y_true=y_true_1d,
        y_pred=y_pred_1d,
        d_argmax_test=d_argmax_test,
        intervals_dict=intervals_dict,
        metrics_dict=metrics_dict,
        agaci_results=agaci_results,
        test_len=test_len,
        d_dim=ds["d_dim"],
    )


# ===================================================================
# Plotting helpers
# ===================================================================

def plot_prediction(y_true, y_pred, d_argmax, d_dim, true_switch_idx,
                    title, save_path):
    """DS3M prediction + regime heatmap for a single synthetic dataset."""
    import seaborn as sns

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 5),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.25},
        sharex=True,
    )

    T = len(y_true)
    t = np.arange(T)

    # -- top: prediction vs truth --
    ax = axes[0]
    ax.plot(t, y_true, color='black', lw=1.0, label='True')
    ax.plot(t, y_pred, color='tab:blue', lw=1.0, alpha=0.8, label='DS3M')
    if true_switch_idx is not None:
        ax.axvline(true_switch_idx, color='red', ls='--', lw=1.5, alpha=0.7,
                   label=f'True switch (t={true_switch_idx})')
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # -- bottom: regime heatmap --
    ax2 = axes[1]
    arr = d_argmax.astype(float)
    if d_dim == 2:
        arr = 1 - arr
    else:
        arr = (d_dim - 1 - arr) / max(d_dim - 1, 1)
    sns.heatmap(arr.reshape(1, -1), ax=ax2, cbar=False,
                cmap=CUSTOM_CMAP, vmin=0, vmax=1, linewidth=0)
    n_sw = int(np.sum(np.diff(d_argmax) != 0))
    ax2.set_title(f'DS3M regime states (detected switches: {n_sw})',
                  fontsize=11)
    ax2.set_yticks([])
    ax2.set_xlabel('time')
    tick_iv = max(T // 10, 1)
    ticks = np.arange(0, T, tick_iv)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(ticks, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_uq_comparison(intervals_dict, y_true, y_pred, d_argmax, d_dim,
                       true_switch_idx, title, save_path):
    """
    Full time-series plot with UQ intervals for ALL methods.
    Layout: one subplot per method + a regime heatmap at the bottom.
    Each subplot shows True, DS3M Pred, and the method's interval band.
    """
    import seaborn as sns

    # Show all available methods in a fixed order
    method_order = ['Naive', 'Gaussian', 'CP', 'EnbPI', 'ACI', 'AgACI']
    methods_present = [m for m in method_order if m in intervals_dict]
    n_methods = len(methods_present)
    if n_methods == 0:
        return

    # n_methods subplots + 1 regime heatmap row
    n_rows = n_methods + 1
    height_ratios = [3] * n_methods + [1]
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(14, 2.8 * n_methods + 1.2),
        sharex=True,
        gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.25},
    )

    T = len(y_true)
    t = np.arange(T)

    for ax, mname in zip(axes[:n_methods], methods_present):
        lo, hi = intervals_dict[mname]
        valid = ~np.isnan(lo) & ~np.isnan(hi)

        ax.plot(t, y_true, color='black', lw=0.8, label='True')
        ax.plot(t, y_pred, color='tab:blue', lw=0.7, alpha=0.7, label='DS3M')
        ax.fill_between(t, np.where(valid, lo, np.nan),
                        np.where(valid, hi, np.nan),
                        color=METHOD_COLORS.get(mname, 'gray'),
                        alpha=0.35, label=f'{mname} interval')
        if true_switch_idx is not None:
            ax.axvline(true_switch_idx, color='red', ls='--', lw=1.5,
                       alpha=0.6)

        # compute metrics on valid region
        yt_v = y_true[valid]
        lo_v, hi_v = lo[valid], hi[valid]
        cov = float(np.mean((yt_v >= lo_v) & (yt_v <= hi_v)))
        med = float(np.median(hi_v - lo_v))
        ax.set_title(f'{mname}  (coverage={cov:.1%}, med.length={med:.1f})',
                     fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # ---- regime heatmap at bottom ----
    ax_hm = axes[-1]
    arr = d_argmax.astype(float)
    if d_dim == 2:
        arr = 1 - arr
    else:
        arr = (d_dim - 1 - arr) / max(d_dim - 1, 1)
    sns.heatmap(arr.reshape(1, -1), ax=ax_hm, cbar=False,
                cmap=CUSTOM_CMAP, vmin=0, vmax=1, linewidth=0)
    n_sw = int(np.sum(np.diff(d_argmax) != 0))
    ax_hm.set_title(f'DS3M regime (switches: {n_sw})', fontsize=10)
    ax_hm.set_yticks([])
    ax_hm.set_xlabel('time', fontsize=11)
    tick_iv = max(T // 10, 1)
    ticks = np.arange(0, T, tick_iv)
    ax_hm.set_xticks(ticks)
    ax_hm.set_xticklabels(ticks, fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_coverage_at_switch(intervals_dict, y_true, true_switch_idx,
                            window, title, save_path):
    """Coverage rate in a sliding window around the known switch point."""
    fig, ax = plt.subplots(figsize=(10, 5))
    t_rel = np.arange(-window, window + 1)

    for mname, (lo, hi) in intervals_dict.items():
        covs = []
        for dt in t_rel:
            idx = true_switch_idx + dt
            if 0 <= idx < len(y_true) and not np.isnan(lo[idx]) and not np.isnan(hi[idx]):
                covs.append(1.0 if lo[idx] <= y_true[idx] <= hi[idx] else 0.0)
            else:
                covs.append(np.nan)
        ax.plot(t_rel, covs,
                color=METHOD_COLORS.get(mname, 'gray'),
                marker='o' if mname == 'AgACI' else '',
                linewidth=2.5 if mname == 'AgACI' else 1.5,
                alpha=0.9 if mname == 'AgACI' else 0.7,
                label=mname)

    ax.axvline(0, color='black', ls='--', alpha=0.5, label='Switch')
    ax.axhline(0.9, color='red', ls=':', alpha=0.5, label='90% target')
    ax.set_xlabel('Time relative to switch', fontsize=11)
    ax.set_ylabel('Coverage (point-wise)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_length_at_switch(intervals_dict, y_true, true_switch_idx,
                          window, title, save_path):
    """Interval length around the known switch point."""
    fig, ax = plt.subplots(figsize=(10, 5))
    t_rel = np.arange(-window, window + 1)

    for mname, (lo, hi) in intervals_dict.items():
        lengths = []
        for dt in t_rel:
            idx = true_switch_idx + dt
            if 0 <= idx < len(y_true) and not np.isnan(lo[idx]) and not np.isnan(hi[idx]):
                lengths.append(hi[idx] - lo[idx])
            else:
                lengths.append(np.nan)
        ax.plot(t_rel, lengths,
                color=METHOD_COLORS.get(mname, 'gray'),
                marker='o' if mname == 'AgACI' else '',
                linewidth=2.5 if mname == 'AgACI' else 1.5,
                alpha=0.9 if mname == 'AgACI' else 0.7,
                label=mname)

    ax.axvline(0, color='black', ls='--', alpha=0.5, label='Switch')
    ax.set_xlabel('Time relative to switch', fontsize=11)
    ax.set_ylabel('Interval Length', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_tradeoff(metrics_dict, title, save_path):
    """Coverage vs interval-length scatter (Pareto frontier)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for mname, (cov, med_len) in metrics_dict.items():
        ax.scatter(
            med_len, cov,
            c=METHOD_COLORS.get(mname, 'gray'),
            marker=METHOD_MARKERS.get(mname, 'o'),
            s=170 if mname == 'AgACI' else 100,
            label=f'{mname} ({cov:.1%}, {med_len:.1f})',
            edgecolors='black' if mname == 'AgACI' else 'none',
            linewidths=2 if mname == 'AgACI' else 0,
            zorder=10 if mname == 'AgACI' else 5,
        )
    ax.axhline(0.9, color='red', ls='--', alpha=0.5, label='90% target')
    ax.set_xlabel('Median Interval Length', fontsize=12)
    ax.set_ylabel('Coverage', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


# ===================================================================
# Experiment 1: Heaviside mean shift
# ===================================================================

def run_experiment1(args):
    """
    Experiment 1: piecewise-constant (Heaviside) mean shift, constant variance.
    Dataset: Toy_exp1
    """
    print("\n" + "=" * 70)
    print("Experiment 1: Heaviside Mean Shift")
    print("=" * 70)

    dataname = "Toy_exp1"
    true_switch_idx = 250  # known switch at t=250 within test set

    result = run_uq_pipeline(dataname, args)
    if result is None:
        print("  Experiment 1 FAILED – skipping.")
        return None

    prefix = "exp1_heaviside"

    # 1. DS3M prediction + regime heatmap
    plot_prediction(
        result['y_true'], result['y_pred'],
        result['d_argmax_test'], result['d_dim'],
        true_switch_idx,
        title='Exp 1 – Heaviside: DS3M Prediction & Regime Detection',
        save_path=OUTPUT_DIR / f"{prefix}_prediction.png",
    )

    # 2. UQ interval comparison (all methods + regime heatmap)
    plot_uq_comparison(
        result['intervals_dict'], result['y_true'], result['y_pred'],
        result['d_argmax_test'], result['d_dim'],
        true_switch_idx,
        title='Exp 1 – Heaviside: UQ Interval Comparison',
        save_path=OUTPUT_DIR / f"{prefix}_uq_comparison.png",
    )

    # 3. Coverage at switch
    plot_coverage_at_switch(
        result['intervals_dict'], result['y_true'], true_switch_idx,
        window=args.window,
        title='Exp 1 – Heaviside: Coverage Around Switch',
        save_path=OUTPUT_DIR / f"{prefix}_coverage_at_switch.png",
    )

    # 4. Length at switch
    plot_length_at_switch(
        result['intervals_dict'], result['y_true'], true_switch_idx,
        window=args.window,
        title='Exp 1 – Heaviside: Interval Length Around Switch',
        save_path=OUTPUT_DIR / f"{prefix}_length_at_switch.png",
    )

    # 5. Trade-off scatter
    plot_tradeoff(
        result['metrics_dict'],
        title='Exp 1 – Heaviside: Coverage vs Length Trade-off',
        save_path=OUTPUT_DIR / f"{prefix}_tradeoff.png",
    )

    return result['metrics_dict']


# ===================================================================
# Experiment 2: AR(1) variance shift
# ===================================================================

def _exp2_dataset_name(v1, v2):
    """Build the DS3M dataset name used for exp2 AR variants."""
    return f"Toy_exp2_V1_{v1}_V2_{v2}_ar_mean-20_2"


def run_experiment2(args):
    """
    Experiment 2: AR(1) with mean + variance shift.
    Sweep over V1, V2 in variance_levels.
    """
    print("\n" + "=" * 70)
    print("Experiment 2: AR(1) Variance Shift")
    print("=" * 70)

    variance_levels = args.variance_levels
    true_switch_idx = 250

    all_rows = []       # for summary CSV

    for v1 in variance_levels:
        for v2 in variance_levels:
            dataname = _exp2_dataset_name(v1, v2)
            label = f"V1={v1}, V2={v2}"
            print(f"\n--- {label} ({dataname}) ---")

            # Check that both data and checkpoint exist
            data_dir = Path(f"Deep_Switching_State_Space_Model/data/{dataname}")
            ckpt_dir = Path(f"Deep_Switching_State_Space_Model/results/checkpoints/{dataname}")
            if not data_dir.exists():
                print(f"  Data directory missing: {data_dir} – skipping")
                continue
            if not ckpt_dir.exists():
                print(f"  Checkpoint missing: {ckpt_dir} – skipping")
                continue

            try:
                result = run_uq_pipeline(dataname, args)
            except Exception as e:
                print(f"  Pipeline failed: {e}")
                traceback.print_exc()
                continue

            if result is None:
                continue

            prefix = f"exp2_V1_{v1}_V2_{v2}"

            # --- per-combination figures ---

            # prediction + regime
            plot_prediction(
                result['y_true'], result['y_pred'],
                result['d_argmax_test'], result['d_dim'],
                true_switch_idx,
                title=f'Exp 2 – AR(1) [{label}]: DS3M Prediction',
                save_path=OUTPUT_DIR / f"{prefix}_prediction.png",
            )

            # UQ comparison (all methods + regime heatmap)
            plot_uq_comparison(
                result['intervals_dict'], result['y_true'], result['y_pred'],
                result['d_argmax_test'], result['d_dim'],
                true_switch_idx,
                title=f'Exp 2 – AR(1) [{label}]: UQ Comparison',
                save_path=OUTPUT_DIR / f"{prefix}_uq_comparison.png",
            )

            # coverage at switch
            plot_coverage_at_switch(
                result['intervals_dict'], result['y_true'], true_switch_idx,
                window=args.window,
                title=f'Exp 2 – AR(1) [{label}]: Coverage at Switch',
                save_path=OUTPUT_DIR / f"{prefix}_coverage_at_switch.png",
            )

            # length at switch
            plot_length_at_switch(
                result['intervals_dict'], result['y_true'], true_switch_idx,
                window=args.window,
                title=f'Exp 2 – AR(1) [{label}]: Length at Switch',
                save_path=OUTPUT_DIR / f"{prefix}_length_at_switch.png",
            )

            # collect summary row
            row = {'V1': v1, 'V2': v2, 'dataset': dataname}
            for mname, (cov, med_len) in result['metrics_dict'].items():
                row[f'{mname}_cov'] = cov
                row[f'{mname}_len'] = med_len
            # classify regime type
            if v1 == v2:
                row['shift_type'] = 'pure_mean'
            elif v1 < v2:
                row['shift_type'] = 'low_to_high'
            else:
                row['shift_type'] = 'high_to_low'
            all_rows.append(row)

    # --- summary CSV ---
    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_path = OUTPUT_DIR / "exp2_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Summary CSV: {csv_path}")

        print("\n" + "=" * 80)
        print("Experiment 2: Summary")
        print("=" * 80)
        print(df.to_string(index=False))

    return all_rows


# ===================================================================
# Summary table (both experiments combined)
# ===================================================================

def create_summary_table(exp1_metrics, exp2_rows):
    """Print and save a combined summary."""
    rows = []

    # Experiment 1
    if exp1_metrics:
        row = {'experiment': 'Exp1_Heaviside', 'V1': '-', 'V2': '-',
               'shift_type': 'pure_mean'}
        for m, (c, l) in exp1_metrics.items():
            row[f'{m}_cov'] = f"{c:.1%}"
            row[f'{m}_len'] = f"{l:.1f}"
        rows.append(row)

    # Experiment 2
    if exp2_rows:
        for r in exp2_rows:
            row = {'experiment': 'Exp2_AR', 'V1': r['V1'], 'V2': r['V2'],
                   'shift_type': r['shift_type']}
            for k, v in r.items():
                if k.endswith('_cov'):
                    row[k] = f"{v:.1%}"
                elif k.endswith('_len'):
                    row[k] = f"{v:.1f}"
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = OUTPUT_DIR / "task4_summary.csv"
        df.to_csv(csv_path, index=False)

        print("\n" + "=" * 80)
        print("Task 4: Combined Summary")
        print("=" * 80)
        print(df.to_string(index=False))
        print(f"\nSaved: {csv_path}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Task 4: Controlled Synthetic Studies",
    )

    # experiment selection
    parser.add_argument("--exp", nargs="*", default=["1", "2"],
                        choices=["1", "2"],
                        help="Which experiments to run (default: both)")

    # exp2 variance grid
    parser.add_argument("--variance-levels", type=float, nargs="*",
                        default=[0.5, 2.0, 10.0, 20.0],
                        help="Variance levels for Exp 2 grid")

    # UQ parameters (same defaults as Task 3)
    parser.add_argument("--aci-train-size", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--tab-gamma", type=float, nargs="*",
                        default=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--aci-gamma", type=float, default=0.01,
                        help="Fixed gamma for single-ACI (default: 0.01, Gibbs & Candes 2021)")
    parser.add_argument("--agaci-eta", type=float, default=0.5)
    parser.add_argument("--agaci-lr-schedule", type=str, default="constant")
    parser.add_argument("--agaci-width-penalty", type=float, default=0.1,
                        help="Width penalty in IS loss (encourages narrower intervals)")
    parser.add_argument("--seed", type=int, default=42)

    # plot settings
    parser.add_argument("--window", type=int, default=30,
                        help="Window size around switch for coverage/length plots")

    args = parser.parse_args()

    print("=" * 70)
    print("Task 4: Controlled Synthetic Studies")
    print("=" * 70)
    print(f"Experiments: {args.exp}")
    print(f"Variance levels (Exp 2): {args.variance_levels}")
    print(f"Target coverage: {1 - args.alpha:.0%}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    exp1_metrics = None
    exp2_rows = None

    if "1" in args.exp:
        exp1_metrics = run_experiment1(args)

    if "2" in args.exp:
        exp2_rows = run_experiment2(args)

    create_summary_table(exp1_metrics, exp2_rows)

    print(f"\n{'=' * 70}")
    print(f"Done! All figures saved to: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
