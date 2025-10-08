# -*- coding: utf-8 -*-
"""
Run paper benchmarks and export a single CSV with:
Model, IntervalMethod, RMSE, Coverage@90, MedianLen, PctInfinite, Notes
"""

import argparse, os, sys, csv, math, numpy as np

HERE = os.path.dirname(__file__)
PROJ = os.path.abspath(os.path.join(HERE, ".."))
for p in [HERE, PROJ]:
    if p not in sys.path:
        sys.path.insert(0, p)

# from experiments.ds3m_wrapper import DS3MWrapper
from experiments.utils.acp_utils import aci_intervals
from experiments.utils.ds3m_utils import ds3m_to_tabular_all, forecast, load_ds3m_data, load_ds3m_model
from experiments.utils.plot_utils import plot_results_with_aci
from experiments.ds3m_wrapper import build_model

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
    out = dict(
        y_pred_mean=testForecast_mean,  # (test_len, D) or (test_len,)
        y_true=testOriginal,            # (test_len, D) or (test_len,)
        y_uq=uq,                        # (test_len, D) or (test_len,)
        y_lq=lq,                        # (test_len, D) or (test_len,)
        d_argmax=d_argmax,              # (test_len,)
        test_len=ds["test_len"],
        predict_dim=ds["predict_dim"],
    )
    return out


def evaluate_one(problem: str, model_name: str, interval_method: str, args):
    ds = load_ds3m_data(args)

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

    # --- ACI bounds over the tail AFTER T0 ---
    X_dummy = np.zeros((N, 1), dtype=float)
    y_lowers, y_uppers, tab_alpha_t, gammas = aci_intervals(X_dummy, y_full, args=args)

    # gid = int(getattr(args, "gamma_idx", 0))
    # gid = 0 if gid < 0 or gid >= y_lowers.shape[0] else gid
    # lo_full = y_lowers[gid]   # shape (test_len - T0,)
    # up_full = y_uppers[gid]   # shape (test_len - T0,)

    # Extract the target dimension from y_full for ACI evaluation
    target_dim_aci = int(ds["target_dim"])
    if y_full.ndim > 1:
        target_dim_aci = max(0, min(target_dim_aci, y_full.shape[1] - 1))
        y_full_1d = y_full[:, target_dim_aci]
    else:
        y_full_1d = y_full.reshape(-1)

    # Pass only the test tail segment to pick_gamma (ACI operates on tail only)
    y_tail_1d = y_full_1d[t0_tail:]

    gid, lo_full, up_full= pick_gamma_by_coverage_and_width(
    y_lowers=y_lowers,
    y_uppers=y_uppers,
    y_all=y_tail_1d,
    T0=T0,
    alpha=args.alpha,
    gamma_idx=getattr(args, "gamma_idx", None),
    skip_eval_head=getattr(args, "skip_eval_head", 0)  # e.g., 50
)

    # Center prediction for RMSE (if you prefer DS³M mean later, swap it in)
    y_pred_eval = 0.5 * (lo_full + up_full)
    y_true_eval = y_full_1d[eval_lo:eval_hi]
    covered_eval = (y_true_eval >= lo_full) & (y_true_eval <= up_full)
    widths_eval  = (up_full - lo_full)

    # --- Fetch DS³M uq/lq and d-argmax for comparison and slice to the SAME eval segment ---
 
    ds3m = build_model(args)
    res = getattr(ds3m, "_get_ds3m_forecast", lambda a: None)(args)
    res = _fetch_ds3m_outputs(args)

    # --- Slice DS³M uq/lq and d-argmax to the SAME eval window ---
    y_uq_full = np.asarray(res["y_uq"])
    y_lq_full = np.asarray(res["y_lq"])
    if y_uq_full.ndim == 1: y_uq_full = y_uq_full[:, None]
    if y_lq_full.ndim == 1: y_lq_full = y_lq_full[:, None]
    td = 0  # or args.target_dim
    td = np.clip(td, 0, y_uq_full.shape[1]-1)
    ds3m_uq_eval = y_uq_full[T0:, td]     # length = eval_len
    ds3m_lq_eval = y_lq_full[T0:, td]     # length = eval_len
    d_argmax_full = np.asarray(res["d_argmax"]).reshape(-1)
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
    
    print(f"  ACI gammas: {gammas}, selected gamma index: {gid}, value: {gammas[gid] if gid is not None and 0 <= gid < len(gammas) else 'N/A'}"
          )
    print(f"  ACI alphas: {tab_alpha_t}")
    print(f"  ACI lower bounds: {y_lowers}")
    print(f"  ACI upper bounds: {y_uppers}")
    print(f"  ds3m_lq_eval predictions: {ds3m_lq_eval}")
    print(f"  ds3m_uq_eval predictions: {ds3m_uq_eval}")

    # Get full test data for plotting (not just eval segment)
    y_true_full = np.asarray(res["y_true"])
    y_pred_full = np.asarray(res["y_pred_mean"])
    y_uq_full_plot = np.asarray(res["y_uq"])
    y_lq_full_plot = np.asarray(res["y_lq"])

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
    )


def main():
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
    print(f"Running experiments for problem={args.models} | device={args.methods}")
    
    for model_name in args.models:
        for method in args.methods:
            # try:
            
            row, _, _, lower_r, upper_r, T0, _, _, method_name, _, _, d_dim, _, target_dim_aci, y_true_full, y_pred_full, y_uq_full_plot, y_lq_full_plot = \
            evaluate_one(args.problem, model_name, method, args)
            # print(f"[{args.problem}] {model_name} + {method} -> "
            #       f"RMSE={row['RMSE']:.4f} | Cov={row['Coverage@90']:.3f} | "
            #       f"MedLen={row['MedianLen']:.3f} | %Inf={row['PctInfinite']:.3f} | {row['Notes']}")
            rows.append(row)

            print(f"Plotting {model_name} + {method} ...")

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
            plot_results_with_aci(
                dataname=args.problem,
                testOriginal=y_true_full,
                testForecast_mean=y_pred_full,
                d_dim=d_dim,
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
                save_dir_root="figures",
                show=True,
            )

    out = args.csv
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
