# -*- coding: utf-8 -*-
"""
Run paper benchmarks and export a single CSV with:
Model, IntervalMethod, RMSE, Coverage@90, MedianLen, PctInfinite, Notes
"""

import argparse, os, sys, csv, math, numpy as np
from typing import Dict, Tuple

HERE = os.path.dirname(__file__)
PROJ = os.path.abspath(os.path.join(HERE, ".."))
for p in [HERE, PROJ]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---- Conformal utils (yours) ----
from experiments.competitor_models import DS3MWrapper, GPTorchSparse, MCDropoutGRU, RupturesSegmentedLinear, S4Regressor
from experiments.utils.acp_utils import aci_intervals, agaci_ewa

# ---- DS3M data loader (yours) ----
from experiments.utils.ds3m_utils import load_ds3m_data
from experiments.utils.plot_utils import plot_results_with_aci


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


def make_lag_matrix(y, lags):
    y = np.asarray(y).reshape(-1)
    T = len(y)
    if lags >= T:
        raise ValueError("lags must be < length of series")
    X = np.empty((T - lags, lags), dtype=float)
    for i in range(lags, T):
        X[i - lags, :] = y[i - lags:i]
    y_t = y[lags:]
    return X, y_t


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def median_len(widths):
    w = np.asarray(widths)
    w = w[~np.isinf(w)]
    return float(np.median(w)) if w.size else float("nan")

def _to_col(a):
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2 and a.shape[1] == 1:
        return a
    # if model returns (N, D) with D>1 keep as is, but you likely want first dim
    return a[:, :1]



# ---------------------------
# Builders & Intervals
# ---------------------------
def build_model(name, lags, params):
    name = name.lower()
    device = params.get("device", "cpu")
    if name == "s4":
        return S4Regressor(
            lags=lags,
            d_model=params.get("s4_d_model", 128),
            n_layers=params.get("s4_layers", 4),
            dropout=params.get("s4_dropout", 0.1),
            epochs=params.get("s4_epochs", 10),
            batch=params.get("s4_batch", 64),
            lr=params.get("s4_lr", 1e-3),
            device=device,
            amp=params.get("amp", False),
            s4_path=params.get("s4_path", None),
        )
    if name == "cpd":
        return RupturesSegmentedLinear(
            penalty=params.get("cpd_penalty", 10.0),
            min_size=params.get("cpd_min_size", 20),
            model=params.get("cpd_model", "l2"),
        )
    if name == "mcdropoutgru":
        return MCDropoutGRU(
            lags=lags,
            hidden=params.get("gru_hidden", 128),
            layers=params.get("gru_layers", 2),
            dropout=params.get("gru_dropout", 0.2),
            epochs=params.get("gru_epochs", 10),
            batch=params.get("gru_batch", 128),
            lr=params.get("gru_lr", 1e-3),
            mc_samples=params.get("gru_samples", 30),
            device=device,
            amp=params.get("amp", False),
        )
    if name == "gptorchsparse":
        return GPTorchSparse(
            lags=lags,
            num_inducing=params.get("gp_inducing", 128),
            iters=params.get("gp_iters", 300),
            lr=params.get("gp_lr", 0.01),
            device=device,
        )
    if name == "ds3m":
        return DS3MWrapper(
            lags=lags,
            problem=params.get("problem", args.problem if "args" in params else "Sleep"),
            target_dim=params.get("target_dim", 0),
            train_size=params.get("train_size", 300),
            device=params.get("device", "cpu"),
            use_cache=not params.get("force_new", False),
            force_new=params.get("force_new", False),
        )

def naive_fixed_intervals(residuals, alpha, train_size):
    """
    'Naive' baseline: fixed half-width q = Q_{1-alpha} over calibration window only.
    Returns arrays (lo, up) for t > train_size, all equal to q.
    """
    calib = np.asarray(residuals[:train_size])
    q = float(np.quantile(calib, 1.0 - alpha))
    T = len(residuals) - train_size
    lo = np.full(T, q, dtype=float)
    up = np.full(T, q, dtype=float)
    return lo, up


# ---------------------------
# Metric computation per (model, method)
# ---------------------------
def evaluate_one(problem: str, model_name: str, interval_method: str, args, params) -> Dict[str, float]:
    """
    Returns dict with metrics for CSV.
    """
    # Load dataset (your ds3m_utils)
    ds = load_ds3m_data(args)

    # univariate series
    if "RawDataOriginal" in ds:
        y_full = np.asarray(ds["RawDataOriginal"]).reshape(-1)
    elif "data" in ds and isinstance(ds["data"], np.ndarray):
        y_full = ds["data"][:, 0].reshape(-1)
    else:
        raise RuntimeError("Could not locate univariate series in dataset.")

    # test length (fallback)
    test_len = int(ds.get("test_len", max(30, len(y_full)//5)))

    # Lag features
    # X_all, y_all = make_lag_matrix(y_full, args.lags)
    X_all = ds.get("trainX", None)
    y_all = ds.get("trainY", None)
    N = len(y_all)
    T0 = int(args.train_size)
    print("N, T0:", N, T0)
    if T0 <= 1 or T0 >= N:
        raise ValueError(f"Bad train_size={T0}; must be in (1, N={N})")
    # Train split before test tail
    # train_end = max(1, N - test_len)
    train_end = max(T0, N - test_len)
    X_tr, y_tr = X_all[:train_end], y_all[:train_end]

    print(f"Dataset {problem}: N={N}, lags={args.lags}, train_end={train_end}, test_len={test_len}, T0={T0}")
    # print(f"y_tr mean/std: {np.mean(y_tr):.4f} / {np.std(y_tr):.4f}, len={len(y_tr)}")
    print(f"X_tr: {X_tr}, y_tr: {y_tr}")

    from sklearn.preprocessing import StandardScaler
    x_scaler = StandardScaler().fit(X_tr)
    y_scaler = StandardScaler().fit(_to_col(y_tr))

    X_tr_s  = x_scaler.transform(X_tr)
    X_all_s = x_scaler.transform(X_all)
    y_tr_s  = y_scaler.transform(_to_col(y_tr)).ravel()

    print("N:", N, "test_len:", test_len, "train_end:", train_end)
    print("y_tr mean/std:", np.mean(y_tr), np.std(y_tr), "len:", len(y_tr))
    print("y_all head:", y_all[:5])

    # Build model
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

    # Fit / Predict
    # try:
    reg = build_model(model_name, args.lags, params)
    if isinstance(reg, DS3MWrapper):
        # DS3M expects full prediction; user-provided function should handle everything
        yhat_all_s = reg.predict(X_all_s)
    else:
        reg.fit(X_tr_s, y_tr_s)
        yhat_all_s = reg.predict(X_all_s)

    # except Exception as e:
    #     row["Notes"] = f"ERROR: {e}"
    #     return row
    
    # back to original scale for metrics/plots/ACI
    if isinstance(yhat_all_s, list):  # safety
        yhat_all_s = np.asarray(yhat_all_s)

    # Ensure 2D for inverse_transform
    yhat_all_s = _to_col(yhat_all_s)

    # Back to original scale -> 1D
    yhat_all = y_scaler.inverse_transform(yhat_all_s).ravel()

    print("yhat_all mean/std:", np.mean(yhat_all), np.std(yhat_all))
    print("corr(y_all, yhat_all):", np.corrcoef(y_all, yhat_all)[0,1])

    # Residuals along entire sequence (align with y_all)
    yhat_all = np.asarray(yhat_all).reshape(-1)
    if len(yhat_all) != len(y_all):
        row["Notes"] = f"len(yhat_all)={len(yhat_all)} != len(y_all)={len(y_all)}"
        return row
    
    print("yhat_all mean/std:", np.mean(yhat_all), np.std(yhat_all))
    print("corr(y_all, yhat_all):", np.corrcoef(y_all, yhat_all)[0,1])

    residuals = np.abs(y_all - yhat_all)

    # Intervals on t>T0 (online/calibration window is first T0)
    try:
        if interval_method.lower() == "aci":
            lo_r, up_r = aci_intervals(residuals, alpha=args.alpha, gamma=args.gamma, train_size=T0)
        elif interval_method.lower() == "agaci":
            lo_r, up_r = agaci_ewa(residuals, alpha=args.alpha, train_size=T0,
                                   gammas=args.agaci_gammas, eta=args.agaci_eta)
        elif interval_method.lower() == "naive":
            lo_r, up_r = naive_fixed_intervals(residuals, alpha=args.alpha, train_size=T0)
        else:
            row["Notes"] = f"Unknown interval method {interval_method}"
            return row
    except Exception as e:
        row["Notes"] = f"Interval ERROR: {e}"
        return row

    # Slice test tail within the (T0: end) segment
    y_pred_seg = yhat_all[T0:]
    y_true_seg = y_all[T0:]
    covered = (y_true_seg >= (y_pred_seg - up_r)) & (y_true_seg <= (y_pred_seg + up_r))

    start_test_in_seg = max(0, (N - test_len) - T0)
    covered_test = covered[start_test_in_seg:]
    up_r_test = up_r[start_test_in_seg:]
    y_pred_test = y_pred_seg[start_test_in_seg:]
    y_true_test = y_true_seg[start_test_in_seg:]

    # Metrics
    row["RMSE"] = rmse(y_true_test, y_pred_test)
    row["Coverage@90"] = float(np.mean(covered_test)) if covered_test.size else float("nan")
    widths = 2.0 * up_r_test
    row["MedianLen"] = median_len(widths)
    row["PctInfinite"] = float(np.mean(np.isinf(up_r_test))) if up_r_test.size else float("nan")

    # Optional note for S4 fallback
    if model_name.lower() == "s4" and isinstance(reg, S4Regressor) and reg.using_fallback:
        row["Notes"] = "S4 fallback (Conv1d) used; pass --s4-path to enable real S4D."

    print("llll y_true_test mean/std:", np.mean(y_true_test), np.std(y_true_test))
    print("llll y_pred_test mean/std:", np.mean(y_pred_test), np.std(y_pred_test))
    return row, y_true_test, y_pred_test, lo_r, up_r, T0, covered_test, widths, interval_method


def evaluate_one2(problem: str, model_name: str, interval_method: str, args, params):
    """
    Returns dict with metrics for CSV + artifacts:
    (row, y_true_test, y_pred_test, lo_r, up_r, T0, covered_test, widths, interval_method)

    - Robust to torch.Tensors and 3D inputs (e.g., Toy: (N, D, 1))
    - Keeps your existing model training/pred flow
    - Builds ACI/AgACI via adapters (RF/OLS through models.py OR your four custom models)
    """
    import math
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    # --- local imports (match your project layout) ---
    from utils.ds3m_utils import load_ds3m_data
    from competitor_models import DS3MWrapper, S4Regressor
    from experiments.utils.acp_utils import (
        set_backend_data, set_backend_series, set_backend_model,
        aci_intervals, agaci_ewa
    )

    # # ----------------- helpers -----------------
    # def _np(a):
    #     """Return a as plain NumPy (handles torch.Tensor / lists / np)."""
    #     try:
    #         import torch
    #         if isinstance(a, torch.Tensor):
    #             return a.detach().cpu().numpy()
    #     except Exception:
    #         pass
    #     return np.asarray(a)

    # # ------------- 1) Load dataset -------------
    ds = load_ds3m_data(args)

    # univariate series
    if "RawDataOriginal" in ds:
        y_full = np.asarray(ds["RawDataOriginal"]).reshape(-1)
    elif "data" in ds and isinstance(ds["data"], np.ndarray):
        y_full = ds["data"][:, 0].reshape(-1)
    else:
        raise RuntimeError("Could not locate univariate series in dataset.")

    # test length (fallback)
    test_len = int(ds.get("test_len", max(300, len(y_full)//5)))

    # Lag features
    # X_all, y_all = make_lag_matrix(y_full, args.lags)
    X_all = ds.get("trainX", None)
    y_all = ds.get("trainY", None)
    N = len(y_all)
    T0 = int(args.train_size)
    print("N, T0:", N, T0)
    if T0 <= 1 or T0 >= N:
        raise ValueError(f"Bad train_size={T0}; must be in (1, N={N})")
    # Train split before test tail
    # train_end = max(1, N - test_len)
    train_end = max(T0, N - test_len)
    X_tr, y_tr = X_all[:train_end], y_all[:train_end]

    print(f"Dataset {problem}: N={N}, lags={args.lags}, train_end={train_end}, test_len={test_len}, T0={T0}")
    # print(f"y_tr mean/std: {np.mean(y_tr):.4f} / {np.std(y_tr):.4f}, len={len(y_tr)}")
    print(f"X_tr: {X_tr}, y_tr: {y_tr}")

    from sklearn.preprocessing import StandardScaler
    x_scaler = StandardScaler().fit(X_tr)
    y_scaler = StandardScaler().fit(_to_col(y_tr))

    X_tr_s  = x_scaler.transform(X_tr)
    X_all_s = x_scaler.transform(X_all)
    y_tr_s  = y_scaler.transform(_to_col(y_tr)).ravel()

    print("N:", N, "test_len:", test_len, "train_end:", train_end)
    print("y_tr mean/std:", np.mean(y_tr), np.std(y_tr), "len:", len(y_tr))
    print("y_all head:", y_all[:5])

    # Build model
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

    # Build & run forecaster (kept as before)
    reg = build_model(model_name, args.lags, params)
    if isinstance(reg, DS3MWrapper):
        yhat_all_s = reg.predict(X_all_s)
    else:
        reg.fit(X_tr_s, y_tr_s)
        yhat_all_s = reg.predict(X_all_s)

    # Back to original scale
    yhat_all_s = _to_col(_np(yhat_all_s))
    yhat_all = y_scaler.inverse_transform(yhat_all_s).ravel()
    print("yhat_all shape:", yhat_all.shape, "y_all shape:", y_all.shape)
    residuals = np.abs(y_all - yhat_all)

    # -- 3) Register backend for ACI/AgACI (RF/OLS via models.py OR your 4 models) --
    print(f"Registering backend for model '{model_name}'...")
    model_key = (model_name or "").lower().strip()
    if model_key in {"rf", "ols"}:
        # Original models.py path
        X_backend = X_all.T  # (d, n)
        Y_backend = y_all    # (n,)
        if model_key == "rf":
            basemodel = "RF"
            params_basemodel = {
                "cores": -1,
                "n_estimators": 1000,
                "min_samples_leaf": 1,
                "max_features": 6,
            }
        else:
            basemodel = "OLS"
            params_basemodel = {}
        set_backend_data(
            X_backend, Y_backend,
            basemodel=basemodel,
            params_basemodel=params_basemodel,
            online=True
        )
    else:
        # Custom competitor models path
        set_backend_series(X_all, y_all)
        if model_key == "s4":
            set_backend_model(
                "S4Regressor",
                lags=args.lags,
                device=("cuda" if getattr(args, "device", "cpu") == "cuda" else "cpu"),
                epochs=getattr(args, "s4_epochs", 50),
                batch=getattr(args, "s4_batch", 64),
                lr=getattr(args, "s4_lr", 1e-3),
                weight_decay=getattr(args, "s4_wd", 1e-2),
                n_layers=getattr(args, "s4_layers", 4),
                d_model=getattr(args, "s4_dmodel", 128),
                dropout=getattr(args, "s4_dropout", 0.1),
                amp=(getattr(args, "device", "cpu") == "cuda"),
            )
        elif model_key == "gru":
            set_backend_model(
                "MCDropoutGRU",
                lags=args.lags,
                device=("cuda" if getattr(args, "device", "cpu") == "cuda" else "cpu"),
                epochs=getattr(args, "gru_epochs", 50),
                batch=getattr(args, "gru_batch", 128),
                lr=getattr(args, "gru_lr", 1e-3),
                weight_decay=getattr(args, "gru_wd", 1e-4),
                mc_samples=getattr(args, "gru_mc", 30),
                dropout=getattr(args, "gru_dropout", 0.2),
            )
        elif model_key == "ruptures":
            set_backend_model(
                "RupturesSegmentedLinear",
                penalty=getattr(args, "rupt_penalty", 10.0),
                min_size=getattr(args, "rupt_min_size", 20),
                model=getattr(args, "rupt_model", "l2"),
            )
        elif model_key == "gp":
            set_backend_model(
                "GPTorchSparse",
                lags=args.lags,
                num_inducing=getattr(args, "gp_m", 128),
                iters=getattr(args, "gp_iters", 300),
                lr=getattr(args, "gp_lr", 1e-2),
                device=("cuda" if getattr(args, "device", "cpu") == "cuda" else "cpu"),
            )
        elif model_key == "ds3m":
            set_backend_model(
                "DS3MWrapper",
                lags=args.lags,
                problem=problem,
                train_size=T0,
                device=("cuda" if getattr(args, "device", "cpu") == "cuda" else "cpu"),
                use_cache=True,
                force_new=False,
            )


    # ------------- 4) Build intervals -------------
    print(f"Building {interval_method} intervals (T0={T0})...")
    if interval_method.lower() == "aci":
        lo_r, up_r = aci_intervals(residuals, alpha=args.alpha, gamma=args.gamma, train_size=T0)
    elif interval_method.lower() == "agaci":
        lo_r, up_r = agaci_ewa(
            residuals, alpha=args.alpha, train_size=T0,
            gammas=getattr(args, "agaci_gammas", [0.005, 0.01, 0.02, 0.05]),
            eta=getattr(args, "agaci_eta", 0.1),
        )
    elif interval_method.lower() == "naive":
        lo_r, up_r = naive_fixed_intervals(residuals, alpha=args.alpha, train_size=T0)
    else:
        row["Notes"] = f"Unknown interval method {interval_method}"
        return row

    # -------- 5) Slice test tail; metrics --------
    print("yhat_all mean/std:", np.mean(yhat_all), np.std(yhat_all))
    y_pred_seg = yhat_all[T0:]
    y_true_seg = y_all[T0:]
    covered = (y_true_seg >= (y_pred_seg - up_r)) & (y_true_seg <= (y_pred_seg + up_r))

    start_test_in_seg = max(0, (N - test_len) - T0)
    covered_test = covered[start_test_in_seg:]
    up_r_test = up_r[start_test_in_seg:]
    y_pred_test = y_pred_seg[start_test_in_seg:]
    y_true_test = y_true_seg[start_test_in_seg:]

    row["RMSE"] = rmse(y_true_test, y_pred_test)
    row["Coverage@90"] = float(np.mean(covered_test)) if covered_test.size else float("nan")
    widths = 2.0 * up_r_test
    row["MedianLen"] = median_len(widths)
    row["PctInfinite"] = float(np.mean(np.isinf(up_r_test))) if up_r_test.size else float("nan")

    if (model_key == "s4") and isinstance(reg, S4Regressor) and getattr(reg, "using_fallback", False):
        row["Notes"] = "S4 fallback (Conv1d) used; pass --s4-path to enable real S4D."

    return row, y_true_test, y_pred_test, lo_r, up_r, T0, covered_test, widths, interval_method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="Unemployment",
                    choices=["Toy","Lorenz","Sleep","Unemployment","Hangzhou","Seattle","Pacific","Electricity","Pernod"])
    ap.add_argument("--lags", type=int, default=48)

    # Interval settings
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.01)
    ap.add_argument("--train_size", type=int, default=100)

    ap.add_argument("--agaci", action="store_true")  # (unused here; we sweep methods)
    ap.add_argument("--agaci_gammas", type=float, nargs="*", default=[0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--agaci_eta", type=float, default=0.1)

    # Device / S4
    ap.add_argument("--device", default=None)       # cuda|mps|cpu
    ap.add_argument("--amp", action="store_true")   # mixed precision
    ap.add_argument("--s4-path", default=None)

    # Sweeps / output
    ap.add_argument("--models", nargs="*", default=["S4","CPD","MCDropoutGRU","GPTorchSparse","DS3M"])
    ap.add_argument("--methods", nargs="*", default=["ACI","AgACI","Naive"])

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

    params = dict(
        device=device,
        amp=args.amp,
        s4_path=args.s4_path,
        # tunables (optional to edit)
        s4_d_model=128, s4_layers=4, s4_dropout=0.1, s4_epochs=100, s4_batch=64, s4_lr=1e-3,
        cpd_penalty=10.0, cpd_min_size=20, cpd_model="l2",
        gru_hidden=128, gru_layers=2, gru_dropout=0.2, gru_epochs=100, gru_batch=128, gru_lr=1e-3, gru_samples=30,
        gp_inducing=128, gp_iters=500, gp_lr=0.01,
    )

    rows = []
    for model_name in args.models:
        for method in args.methods:
            row, y_true, y_pred_mean, lower_r, upper_r, T0, coverage, width, method_name = \
            evaluate_one(args.problem, model_name, method, args, params)
            print(f"[{args.problem}] {model_name} + {method} -> "
                  f"RMSE={row['RMSE']:.4f} | Cov={row['Coverage@90']:.3f} | "
                  f"MedLen={row['MedianLen']:.3f} | %Inf={row['PctInfinite']:.3f} | {row['Notes']}")
            rows.append(row)

            # example for S4 (no native intervals)
            # print(f"Plotting results for {args.problem} + {model_name} + {method_name}")
            plot_results_with_aci(
                dataname=args.problem,
                testOriginal=y_true,
                testForecast_mean=y_pred_mean,
                d_dim=None,                       
                forecast_d_MC_argmax=None,
                # No DS3M intervals
                dsm_lower=None, dsm_upper=None,
                # No model intervals
                model_lower=None, model_upper=None, model_interval_label=None,
                # ACI series
                aci_lower=lower_r,                
                aci_upper=upper_r,
                T0=T0,
                target_dim=0,
                coverage=float(row["Coverage@90"]),  # <- ensure scalar
                width=float(row["MedianLen"]),       # <- ensure scalar
                model_name=model_name,
                interval_method_name=method_name,  # "ACI" | "AgACI" | "Naive"
                save_dir_root="figures",
                show=True,
            )

    # # example for MC-Dropout GRU (with model intervals)
    # plot_results_with_aci(
    #     dataname=args.problem,
    #     testOriginal=y_true,
    #     testForecast_mean=y_pred_mean,
    #     d_dim=None,
    #     forecast_d_MC_argmax=None,
    #     model_lower=None,                # (T, D) or (T,)
    #     model_upper=None,
    #     model_interval_label="MC-Dropout 90% PI",
    #     aci_lower=lower_r,
    #     aci_upper=upper_r,
    #     T0=T0,
    #     target_dim=0,
    #     coverage=coverage,
    #     width=width,
    #     model_name="MCDropoutGRU",
    #     interval_method_name=method_name,
    # )

    # # example for GP-Torch Sparse
    # plot_results_with_aci(
    #     dataname=args.problem,
    #     testOriginal=y_true,
    #     testForecast_mean=y_pred_mean,
    #     d_dim=None,
    #     forecast_d_MC_argmax=None,
    #     model_lower=None,              # (T, D)
    #     model_upper=None,
    #     model_interval_label="GP 90% PI",
    #     aci_lower=lower_r,
    #     aci_upper=upper_r,
    #     T0=T0,
    #     target_dim=0,
    #     coverage=coverage,
    #     width=width,
    #     model_name="GPTorchSparse",
    #     interval_method_name=method_name,
    # )
    # write CSV
    out = args.csv
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()