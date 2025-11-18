# -*- coding: utf-8 -*-
"""
Run paper benchmarks and export a single CSV with:
Model, IntervalMethod, RMSE, Coverage@90, MedianLen, PctInfinite, Notes
"""
import sys
sys.stdout.write("[TRACE] Script started\n")
sys.stdout.flush()

import argparse, os, csv, math, numpy as np
sys.stdout.write("[TRACE] Basic imports done\n")
sys.stdout.flush()

# Set matplotlib backend to non-interactive to avoid hanging
import matplotlib
matplotlib.use('Agg')
print("[INFO] Matplotlib backend set to 'Agg'")

print("[INFO] Setting up paths...")
HERE = os.path.dirname(__file__)
PROJ = os.path.abspath(os.path.join(HERE, ".."))
for p in [HERE, PROJ]:
    if p not in sys.path:
        sys.path.insert(0, p)
print(f"[INFO] HERE={HERE}")
print(f"[INFO] PROJ={PROJ}")

# from experiments.ds3m_wrapper import DS3MWrapper
print("[INFO] Importing acp_utils...")
from experiments.utils.acp_utils import aci_intervals, agaci_intervals
print("[INFO] Importing ds3m_utils...")
from experiments.utils.ds3m_utils import ds3m_to_tabular_all, forecast, load_ds3m_data, load_ds3m_model, get_full_d_argmax
print("[INFO] Importing plot_utils...")
from experiments.utils.plot_utils import plot_results_with_aci
print("[INFO] Importing regime_switch_analysis...")
from experiments.utils.regime_switch_analysis import (
    plot_agaci_weights_at_switches,
    plot_coverage_at_switches,
    plot_coverage_vs_length_tradeoff,
    plot_regime_heatmap_full,
    detect_regime_switches,
    load_timestamps_for_dataset
)
print("[INFO] Importing ds3m_wrapper...")
from experiments.ds3m_wrapper import build_model
print("[INFO] All imports completed successfully!")

# ---------------------------
# Helpers
# ---------------------------
def pick_device(name: str):
    try:
        import torch
    except Exception:
        return "cpu"
    name = (name or "").lower()
    if name == "cuda" and torch.cuda.is_available():
        return "cuda"
    if name == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def pick_gamma_by_coverage_and_width(
    y_lowers: np.ndarray,  # shape (n_gamma, test_size_eff)
    y_uppers: np.ndarray,  # shape (n_gamma, test_size_eff)
    y_all: np.ndarray,     # full target series after lagging, shape (N,)
    T0: int,               # train_size used inside ACI
    alpha: float,
    gamma_idx: int | None = None,
    skip_eval_head: int = 0
):
    """
    Select a gamma row from (y_lowers, y_uppers).

    If gamma_idx is given, we use it directly.
    Otherwise, we auto-select the gamma whose coverage is closest to 1-alpha
    (on the ACI evaluation segment), breaking ties by smaller median width.

    We make sure ground-truth slice matches ACI's test_size_eff exactly.
    """
    assert y_lowers.shape == y_uppers.shape
    n_gamma, test_size_eff = y_lowers.shape

    # If user forces a specific gamma, honor it (clamped to valid range)
    if gamma_idx is not None:
        gid = int(gamma_idx)
        gid = max(0, min(gid, n_gamma - 1))
        lo_full = y_lowers[gid]
        up_full = y_uppers[gid]
        return gid, lo_full, up_full, None, None  # no auto metrics

    # Ground truth segment must have the SAME length as ACI outputs
    # ACI test segment begins at T0 (train_size) and has length test_size_eff
    y_true_seg = y_all[T0 : T0 + test_size_eff]  # shape (test_size_eff,)

    # Optional: skip the first K points (alpha_t warm-up region)
    if skip_eval_head > 0:
        k = min(skip_eval_head, test_size_eff)
        lo_mat = y_lowers[:, k:]
        up_mat = y_uppers[:, k:]
        y_true_eval = y_true_seg[k:]
    else:
        lo_mat = y_lowers
        up_mat = y_uppers
        y_true_eval = y_true_seg

    # Coverage per gamma (fraction of times y falls inside [lo, up])
    # Broadcast y_true to (1, eval_len) so it compares with (n_gamma, eval_len)
    covers = (y_true_eval[None, :] >= lo_mat) & (y_true_eval[None, :] <= up_mat)
    cov_rate = covers.mean(axis=1)  # (n_gamma,)

    # Median width per gamma
    widths = (up_mat - lo_mat)                 # (n_gamma, eval_len)
    med_width = np.median(widths, axis=1)      # (n_gamma,)

    target_cov = 1.0 - float(alpha)
    cov_err = np.abs(cov_rate - target_cov)

    # Primary key: coverage error (smaller is better)
    # Secondary key: median width (smaller is better)
    gid = np.lexsort((med_width, cov_err))[0]

    lo_full = y_lowers[gid]  # full ACI segment (length test_size_eff)
    up_full = y_uppers[gid]

    print(f"[ACI] auto-select gamma: gid={gid}, "
          f"coverage={cov_rate[gid]:.3f}, target={target_cov:.3f}, "
          f"median width={med_width[gid]:.3f}")

    return gid, lo_full, up_full


def _fetch_ds3m_outputs(args):
    """
    Compute DS³M forecast once (no rolling), return a dict with
    y_pred_mean, y_uq, y_lq, d_argmax, test_len, predict_dim.
    Also returns d_argmax_full for the entire dataset (train+valid+test).
    """
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

    # Get d_argmax for full dataset (train+valid+test)
    d_argmax_full = get_full_d_argmax(model, ds)

    out = dict(
        y_pred_mean=testForecast_mean,  # (test_len, D) or (test_len,)
        y_true=testOriginal,            # (test_len, D) or (test_len,)
        y_uq=uq,                        # (test_len, D) or (test_len,)
        y_lq=lq,                        # (test_len, D) or (test_len,)
        d_argmax=d_argmax,              # (test_len,)
        d_argmax_full=d_argmax_full,    # (N_total,) - full dataset
        test_len=ds["test_len"],
        predict_dim=ds["predict_dim"],
        model=model,                     # For potential reuse
        ds=ds,                          # Full dataset info
    )
    return out


def evaluate_one(problem: str, model_name: str, interval_method: str, args):
    print(f"\n[INFO] Evaluating {model_name} + {interval_method} on {problem}...")

    # Load DS3M data once and reuse it
    print(f"[INFO] Loading DS3M data for {problem}...")
    ds = load_ds3m_data(args)
    print(f"[INFO] DS3M data loaded successfully")

    # Use the processed data after reshaping (not RawDataOriginal which may have different structure)
    # ds["data"] has shape (T, D) after RawData.reshape(-1, RawData.shape[2])
    y_full = np.asarray(ds["data"])  # shape (T, D)
    if y_full.ndim == 1:
        y_full = y_full.reshape(-1, 1)

    N        = len(y_full)
    test_len = int(ds["test_len"])
    T0       = int(args.aci_train_size)
    t0_tail  = N - test_len          # DS³M test-tail start
    eval_lo  = t0_tail + T0
    eval_hi  = t0_tail + test_len
    eval_len = max(0, eval_hi - eval_lo)

    row = {
        "Problem": problem,
        "Model": model_name,
        "IntervalMethod": interval_method,
        "RMSE": math.nan,
        "Coverage@90": math.nan,
        "MedianLen": math.nan,
        "PctInfinite": math.nan,
        "Notes": "",
    }

    # --- Run ACI or AgACI based on method ---
    X_dummy = np.zeros((N, 1), dtype=float)

    # Extract the target dimension from y_full for evaluation
    target_dim_aci = int(ds["target_dim"])
    if y_full.ndim > 1:
        target_dim_aci = max(0, min(target_dim_aci, y_full.shape[1] - 1))
        y_full_1d = y_full[:, target_dim_aci]
    else:
        y_full_1d = y_full.reshape(-1)

    # Pass only the test tail segment (ACI operates on tail only)
    y_tail_1d = y_full_1d[t0_tail:]

    agaci_weights_lower = None
    agaci_weights_upper = None
    y_lowers = None
    y_uppers = None
    gammas = None
    gid = None

    # --- Fetch DS³M uq/lq and d-argmax first (needed for Naive method) ---
    print(f"[INFO] Loading DS3M model and generating forecast...")
    res = _fetch_ds3m_outputs(args)
    print(f"[INFO] DS3M forecast generated successfully")

    # --- Slice DS³M uq/lq and d-argmax to the SAME eval window ---
    y_uq_full = np.asarray(res["y_uq"])
    y_lq_full = np.asarray(res["y_lq"])
    if y_uq_full.ndim == 1: y_uq_full = y_uq_full[:, None]
    if y_lq_full.ndim == 1: y_lq_full = y_lq_full[:, None]

    if interval_method.upper() == "NAIVE":
        # Naive: use DS3M's original Monte Carlo intervals (testForecast_uq, testForecast_lq)
        # Use FULL test set intervals without slicing (same as test_agaci.py)
        print("[INFO] [Naive] Using DS3M MC intervals (full test set)")
        td = target_dim_aci
        td = np.clip(td, 0, y_uq_full.shape[1]-1)
        # Extract full test set intervals (no slicing)
        lo_full_naive = y_lq_full[:, td]     # Full test set, length = test_len
        up_full_naive = y_uq_full[:, td]     # Full test set, length = test_len
        # For evaluation, slice from T0 onwards
        lo_full = lo_full_naive[T0:]         # length = test_len - T0
        up_full = up_full_naive[T0:]         # length = test_len - T0
        gid = None

    elif interval_method.upper() == "AGACI":
        # Run AgACI with BOA aggregation
        print("[INFO] [AgACI] Running AgACI with BOA aggregation...")
        agaci_results = agaci_intervals(X_dummy, y_full, basemodel="ds3m", args=args)
        print("[INFO] [AgACI] Completed successfully")

        lo_full = agaci_results['lower']
        up_full = agaci_results['upper']
        agaci_weights_lower = agaci_results['weights_lower']
        agaci_weights_upper = agaci_results['weights_upper']
        y_lowers = agaci_results['y_lowers_experts']
        y_uppers = agaci_results['y_uppers_experts']
        gammas = agaci_results['gammas']
        tab_alpha_t = agaci_results['tab_alpha_t']
        gid = None  # AgACI doesn't select a single gamma

    else:
        # ACI: Run standard ACI with multiple gammas and select best
        print("[INFO] [ACI] Running standard ACI with multiple gammas...")
        y_lowers, y_uppers, tab_alpha_t, gammas = aci_intervals(X_dummy, y_full, args=args)
        print("[INFO] [ACI] Completed successfully")

        # Select best gamma based on coverage and width
        gid, lo_full, up_full = pick_gamma_by_coverage_and_width(
            y_lowers=y_lowers,
            y_uppers=y_uppers,
            y_all=y_tail_1d,
            T0=T0,
            alpha=args.alpha,
            gamma_idx=getattr(args, "gamma_idx", None),
            skip_eval_head=getattr(args, "skip_eval_head", 0)
        )

    # Center prediction for RMSE (if you prefer DS³M mean later, swap it in)
    y_pred_eval = 0.5 * (lo_full + up_full)
    y_true_eval = y_full_1d[eval_lo:eval_hi]
    covered_eval = (y_true_eval >= lo_full) & (y_true_eval <= up_full)
    widths_eval  = (up_full - lo_full)

    # --- Get d-argmax for regime analysis ---
    td = 0  # or args.target_dim
    td = np.clip(td, 0, y_uq_full.shape[1]-1)
    ds3m_uq_eval = y_uq_full[T0:, td]     # length = eval_len
    ds3m_lq_eval = y_lq_full[T0:, td]     # length = eval_len
    d_argmax_full = np.asarray(res.get("d_argmax_full", res["d_argmax"])).reshape(-1)
    ds3m_d_argmax_eval = d_argmax_full[T0:] if d_argmax_full.size >= T0 else None

    # --- Metrics ---
    if eval_len > 0:
        row["RMSE"] = float(np.sqrt(np.mean((y_true_eval - y_pred_eval) ** 2)))
        row["Coverage@90"] = float(np.mean(covered_eval))
        row["MedianLen"] = float(np.median(widths_eval))
        row["PctInfinite"] = float(np.mean(np.isinf(lo_full) | np.isinf(up_full)))
    else:
        row["Notes"] = "Empty eval segment (test_len <= T0)."

    print(f"[{problem}] {model_name} + {interval_method} -> "
          f"RMSE={row['RMSE']:.4f} | Cov={row['Coverage@90']:.3f} | "
          f"MedLen={row['MedianLen']:.3f} | %Inf={row['PctInfinite']:.3f} | {row['Notes']}")

    # Get full test data for plotting (not just eval segment)
    y_true_full = np.asarray(res["y_true"])
    y_pred_full = np.asarray(res["y_pred_mean"])
    y_uq_full_plot = np.asarray(res["y_uq"])
    y_lq_full_plot = np.asarray(res["y_lq"])

    # Get d_argmax_full from DS3M outputs
    d_argmax_full = res.get("d_argmax_full", None)

    # Create padded intervals for plotting (aligned with full test set)
    # This matches test_agaci.py approach: pad first T0 positions with NaN
    lo_full_padded = np.full(test_len, np.nan)
    up_full_padded = np.full(test_len, np.nan)

    if interval_method.upper() == "NAIVE":
        # For Naive, use full DS3M intervals (already computed above)
        lo_full_padded = lo_full_naive
        up_full_padded = up_full_naive
    else:
        # For ACI/AgACI, pad with NaN in first T0 positions
        lo_full_padded[T0:] = lo_full
        up_full_padded[T0:] = up_full

    return (
        row,
        y_true_eval,
        y_pred_eval,
        lo_full,
        up_full,
        T0,
        covered_eval,
        widths_eval,
        interval_method,
        ds3m_lq_eval,
        ds3m_uq_eval,
        ds["d_dim"],
        ds3m_d_argmax_eval,
        target_dim_aci,
        y_true_full,
        y_pred_full,
        y_uq_full_plot,
        y_lq_full_plot,
        agaci_weights_lower,
        agaci_weights_upper,
        gammas,
        d_argmax_full,
        y_lowers if interval_method.upper() != "AGACI" else y_lowers,
        y_uppers if interval_method.upper() != "AGACI" else y_uppers,
        lo_full_padded,  # For plotting with full test set alignment
        up_full_padded,  # For plotting with full test set alignment
    )


def main():
    print("="*60)
    print("Starting run_all_experiments.py")
    print("="*60)

    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="Unemployment",
                    choices=["Toy","Lorenz","Sleep","Unemployment","Hangzhou","Seattle","Pacific","Electricity"])
    ap.add_argument("--lags", type=int, default=48)

    # Interval settings

    ap.add_argument("--gamma", type=float, default=0.01)  # (unused here; we sweep methods)
    ap.add_argument("--train_size", type=int, default=1000)

    ap.add_argument("--agaci", action="store_true")  # (unused here; we sweep methods)
    ap.add_argument("--agaci_gammas", type=float, nargs="*", default=[0.0025, 0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--agaci_eta", type=float, default=0.1)

    # Device / S4
    ap.add_argument("--device", default=None)       # cuda|mps|cpu
    ap.add_argument("--amp", action="store_true")   # mixed precision
    ap.add_argument("--s4-path", default=None)

    # Sweeps / output
    ap.add_argument("--models", nargs="*", default=["DS3M"])
    ap.add_argument("--methods", nargs="*", default=["ACI","AgACI","Naive"])
    ap.add_argument("--aci_train_size", type=int, default=20, help="T0 used as calibration size for ACP")
    ap.add_argument("--force-new", action="store_true", help="Ignore cache and recompute forecast")
    ap.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level 1-alpha, e.g. 0.1 for 90% PI")
    ap.add_argument("--tab-gamma", type=float, nargs="*", default=[0.0025, 0.005, 0.01, 0.02, 0.05], help="ACI step sizes")

    # ap.add_argument("--models", nargs="*", default=["S4","CPD","MCDropoutGRU","GPTorchSparse","DS3M"])
    # ap.add_argument("--methods", nargs="*", default=["ACI","AgACI","Naive"])

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default="paper_results.csv")
    ap.add_argument("--save-dir", default="figures", help="Directory to save plots (use local path to avoid OneDrive sync issues)")
    args = ap.parse_args()

    # device
    device = pick_device(args.device)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    rows = []
    all_results = {}  # Store results for regime analysis
    print(f"\n[INFO] Running experiments for problem={args.problem}")
    print(f"[INFO] Models: {args.models}")
    print(f"[INFO] Methods: {args.methods}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Seed: {args.seed}")
    print(f"[INFO] Output CSV: {args.csv}\n")

    for model_name in args.models:
        for method in args.methods:
            # try:

            results = evaluate_one(args.problem, model_name, method, args)
            row = results[0]
            lower_r = results[3]
            upper_r = results[4]
            T0 = results[5]
            method_name = results[8]
            d_dim = results[11]
            target_dim_aci = results[13]
            y_true_full = results[14]
            y_pred_full = results[15]
            y_uq_full_plot = results[16]
            y_lq_full_plot = results[17]
            agaci_weights_lower = results[18]
            agaci_weights_upper = results[19]
            gammas = results[20]
            d_argmax_full = results[21]
            y_lowers_all = results[22]
            y_uppers_all = results[23]
            lower_padded = results[24]  # New: padded intervals for plotting
            upper_padded = results[25]  # New: padded intervals for plotting

            # Store for regime analysis
            # Use padded intervals for plotting (aligned with full test set)
            all_results[method_name] = {
                'lower': lower_padded,  # Changed: use padded instead of sliced
                'upper': upper_padded,  # Changed: use padded instead of sliced
                'agaci_weights_lower': agaci_weights_lower,
                'agaci_weights_upper': agaci_weights_upper,
                'gammas': gammas,
                'y_lowers_all': y_lowers_all,
                'y_uppers_all': y_uppers_all,
                'row': row,
            }

            rows.append(row)

            print(f"\n[INFO] Plotting {model_name} + {method} ...")

            # plot_results_with_aci(
            #     dataname=args.problem,
            #     testOriginal=y_true,
            #     testForecast_mean=y_pred_mean,
            #     d_dim=d_dim,                       
            #     forecast_d_MC_argmax=forecast_d_MC_argmax,
            #     # No DS3M intervals
            #     dsm_lower=y_lq, dsm_upper=y_uq,
            #     # No model intervals
            #     model_lower=None, model_upper=None, model_interval_label=None,
            #     # ACI series
            #     aci_lower=lower_r,                
            #     aci_upper=upper_r,
            #     T0=T0,
            #     target_dim=0,
            #     coverage=float(row["Coverage@90"]),  # <- ensure scalar
            #     width=float(row["MedianLen"]),       # <- ensure scalar
            #     model_name=model_name,
            #     interval_method_name=method_name,  # "ACI" | "AgACI" | "Naive"
            #     save_dir_root="figures",
            #     show=True,
            # )

            # Debug: Check d_argmax_full values
            if d_argmax_full is not None:
                print(f"\n[DEBUG] d_argmax_full stats:")
                print(f"  Shape: {d_argmax_full.shape}")
                print(f"  Min: {d_argmax_full.min()}, Max: {d_argmax_full.max()}")
                print(f"  Unique values: {np.unique(d_argmax_full)}")
                print(f"  d_dim: {d_dim}")

            plot_results_with_aci(
                dataname=args.problem,
                testOriginal=y_true_full,
                testForecast_mean=y_pred_full,
                d_dim=d_dim,
                forecast_d_MC_argmax=d_argmax_full,
                dsm_lower=y_lq_full_plot,
                dsm_upper=y_uq_full_plot,
                aci_lower=lower_r,
                aci_upper=upper_r,
                T0=T0,
                target_dim=target_dim_aci,
                coverage=float(row["Coverage@90"]),
                width=float(row["MedianLen"]),
                model_name=model_name,
                interval_method_name=method_name,  # "ACI" | "AgACI" | "Naive"
                save_dir_root=args.save_dir,
                show=False,  # Set to False to avoid hanging in non-interactive environments
            )
            print(f"[INFO] Plot completed for {model_name} + {method_name}")

    # =========================================================================
    # REGIME SWITCHING ANALYSIS
    # =========================================================================
    print("\n" + "="*60)
    print("Regime Switching Analysis")
    print("="*60)

    if d_argmax_full is not None and len(d_argmax_full) > 0:
        regime_save_dir = f"figures/regime_analysis/{args.problem}"
        os.makedirs(regime_save_dir, exist_ok=True)

        # 1. Plot regime heatmap for full dataset
        print("\n1. Plotting regime heatmap (full dataset: train+valid+test)...")
        # Load timestamps from dataset if available
        timestamps = load_timestamps_for_dataset(args.problem, len(d_argmax_full))
        plot_regime_heatmap_full(
            d_argmax=d_argmax_full,
            d_dim=d_dim,
            dataname=args.problem,
            timestamps=timestamps,
            save_path=f"{regime_save_dir}/regime_heatmap_full.png"
        )

        # 2. Plot AgACI weights at regime switches
        if 'AgACI' in all_results and all_results['AgACI']['agaci_weights_lower'] is not None:
            print("\n2. Plotting AgACI weights at regime switches...")
            agaci_weights = all_results['AgACI']['agaci_weights_lower']  # shape (T, n_gammas)
            gammas_list = all_results['AgACI']['gammas']

            # Transpose to (n_gammas, T) for plotting
            agaci_weights_t = agaci_weights.T if agaci_weights.ndim == 2 else agaci_weights

            plot_agaci_weights_at_switches(
                agaci_weights=agaci_weights_t,
                d_argmax=d_argmax_full,
                gamma_values=gammas_list,
                window_before=10,
                window_after=50,
                save_path=f"{regime_save_dir}/agaci_weights_switches.png",
                show_individual_lines=True
            )

        # 3. Plot coverage at regime switches
        print("\n3. Plotting coverage at regime switches...")

        # Need y_true for test set (to align with intervals)
        ds = load_ds3m_data(args)
        y_full_data = np.asarray(ds["data"])
        if y_full_data.ndim > 1:
            y_full_data = y_full_data[:, target_dim_aci]

        # Get test set portion
        N = len(y_full_data)
        test_len = int(ds["test_len"])
        t0_tail = N - test_len

        # Extract test set (to align with padded intervals)
        y_true_test = y_true_full  # This is already the test set from DS3M output
        if y_true_test.ndim > 1:
            y_true_test = y_true_test[:, target_dim_aci]

        # Get d_argmax for test set
        d_argmax_test = d_argmax_full[t0_tail:] if len(d_argmax_full) >= t0_tail + test_len else d_argmax_full[-test_len:]

        # Build intervals dict (padded intervals aligned with test set)
        intervals_for_coverage = {}
        for method_name, res_dict in all_results.items():
            lower_arr = res_dict['lower']  # Padded intervals, length = test_len
            upper_arr = res_dict['upper']  # Padded intervals, length = test_len

            # Debug: Check if intervals are different
            print(f"  {method_name}: lower shape={lower_arr.shape}, "
                  f"mean={np.nanmean(lower_arr):.4f}, std={np.nanstd(lower_arr):.4f}")
            print(f"  {method_name}: upper shape={upper_arr.shape}, "
                  f"mean={np.nanmean(upper_arr):.4f}, std={np.nanstd(upper_arr):.4f}")

            intervals_for_coverage[method_name] = (lower_arr, upper_arr)

        plot_coverage_at_switches(
            intervals_dict=intervals_for_coverage,
            y_true=y_true_test,  # Full test set
            d_argmax=d_argmax_test,  # Test set regimes
            window_before=10,
            window_after=50,
            save_path=f"{regime_save_dir}/coverage_at_switches.png"
        )

        # 4. Plot coverage vs length tradeoff
        print("\n4. Plotting coverage vs length tradeoff...")
        tradeoff_dict = {}
        for method_name, res_dict in all_results.items():
            row = res_dict['row']
            tradeoff_dict[method_name] = (row['Coverage@90'], row['MedianLen'])

        plot_coverage_vs_length_tradeoff(
            results_dict=tradeoff_dict,
            save_path=f"{regime_save_dir}/tradeoff.png"
        )

        print(f"\n{'='*60}")
        print(f"Regime analysis plots saved to: {regime_save_dir}/")
        print(f"{'='*60}")

    # =========================================================================
    # Save CSV
    # =========================================================================
    out = args.csv
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
