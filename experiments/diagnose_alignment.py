"""
Alignment Diagnostic: measures the exact lag offset between predictions
and ground truth for each model × dataset.

For each (model, dataset), we compute MSE and Pearson-r at lags -5..+5.
Optimal lag ≠ 0 means there's a real alignment bug.
Optimal lag = 0 but visual "tracking delay" means autoregressive artifact (not a bug).

Usage:
    python diagnose_alignment.py
    python diagnose_alignment.py --datasets Toy Sleep Unemployment
    python diagnose_alignment.py --datasets Toy --include-ds3m
"""

import os, sys, argparse, warnings
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from experiments.generate_forecasting_comparison import (
    load_original_data, clean_series, create_lag_features,
    standardize_train_test, DATASET_CONFIG,
)

# ---------------------------------------------------------------------------
# Core diagnostic utilities
# ---------------------------------------------------------------------------

def cross_correlate_lag(y_true, y_pred, max_lag=5):
    """
    Evaluate alignment quality at integer lags in [-max_lag, +max_lag].

    Convention:
      lag > 0  →  pred is AHEAD of truth:  compare pred[lag:] vs truth[:-lag]
      lag < 0  →  pred is BEHIND truth:    compare pred[:lag] vs truth[-lag:]
      lag = 0  →  no shift

    Returns dict  lag → {mse, corr, n}
    """
    results = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            t = y_true[:-lag]
            p = y_pred[lag:]
        elif lag < 0:
            t = y_true[-lag:]
            p = y_pred[:lag]
        else:
            t, p = y_true, y_pred

        n = min(len(t), len(p))
        if n < 3:
            continue
        t, p = t[:n].astype(float), p[:n].astype(float)

        mse = float(np.mean((t - p) ** 2))
        std_t, std_p = np.std(t), np.std(p)
        if std_t > 1e-12 and std_p > 1e-12:
            r = float(np.corrcoef(t, p)[0, 1])
        else:
            r = 0.0
        results[lag] = {"mse": mse, "corr": r, "n": n}
    return results


def find_optimal_lag(lag_results, criterion="mse"):
    """Return (best_lag, best_value) by minimizing MSE or maximizing corr."""
    best_lag, best_val = 0, float("inf") if criterion == "mse" else -float("inf")
    for lag, m in lag_results.items():
        val = m[criterion]
        if (criterion == "mse" and val < best_val) or \
           (criterion == "corr" and val > best_val):
            best_val = val
            best_lag = lag
    return best_lag, best_val


def mse_improvement_pct(lag_results, opt_lag):
    """How much (%) does shifting to opt_lag improve MSE vs lag=0?"""
    mse0 = lag_results.get(0, {}).get("mse", float("nan"))
    mse_opt = lag_results.get(opt_lag, {}).get("mse", float("nan"))
    if np.isnan(mse0) or mse0 == 0:
        return 0.0
    return 100.0 * (mse0 - mse_opt) / mse0


# ---------------------------------------------------------------------------
# Per-dataset diagnosis
# ---------------------------------------------------------------------------

def diagnose_dataset(dataname, include_ds3m=False, device="cpu", max_lag=5):
    """
    Run alignment diagnostic for one dataset.
    Returns dict  model_name → diagnostic_info
    """
    cfg = DATASET_CONFIG.get(dataname)
    if cfg is None:
        print(f"  [SKIP] Unknown dataset: {dataname}")
        return None

    test_len = cfg["test_len"]
    lags = cfg["lags"]
    dim = cfg["dim"]

    # --- load data (same path as generate_forecasting_comparison) ---
    try:
        raw = load_original_data(dataname)
    except Exception as e:
        print(f"  [ERR] {dataname}: {e}")
        return None

    y_1d = raw[:, dim].flatten() if raw.ndim > 1 else raw.flatten()
    y_data = clean_series(y_1d)
    X, y = create_lag_features(y_data, lags)

    N = len(y)
    act_tl = min(test_len, N - 50)
    if act_tl < 20:
        print(f"  [SKIP] {dataname}: too short (N={N})")
        return None
    tr_end = N - act_tl

    X_tr, y_tr = X[:tr_end], y[:tr_end]
    X_te, y_te = X[tr_end:], y[tr_end:]
    X_tr_s, X_te_s, y_tr_s, y_te_s, y_mu, y_sig = standardize_train_test(
        X_tr, X_te, y_tr, y_te
    )

    out = {}

    # 1) AR baseline (Ridge) — should be lag=0 if indexing is correct
    try:
        ar = Ridge(alpha=1.0).fit(X_tr, y_tr)
        p = ar.predict(X_te)
        lr = cross_correlate_lag(y_te, p, max_lag)
        ol, _ = find_optimal_lag(lr)
        out["AR"] = {"optimal_lag": ol, "lag_results": lr,
                     "improvement_%": mse_improvement_pct(lr, ol),
                     "pred_len": len(p), "y_test_len": len(y_te)}
    except Exception as e:
        out["AR"] = {"error": str(e)}

    # 2) Naive persistence baseline (predict y_{t} = y_{t-1})
    #    This SHOULD have optimal lag = 0 by construction; if not, indexing is broken.
    try:
        p_naive = X_te[:, -1]  # last lag = y_{t-1}
        lr = cross_correlate_lag(y_te, p_naive, max_lag)
        ol, _ = find_optimal_lag(lr)
        out["Persist"] = {"optimal_lag": ol, "lag_results": lr,
                          "improvement_%": mse_improvement_pct(lr, ol),
                          "pred_len": len(p_naive), "y_test_len": len(y_te)}
    except Exception as e:
        out["Persist"] = {"error": str(e)}

    # 3) "Shifted-by-1" sanity check — intentionally misalign AR to show contrast
    try:
        p_shifted = ar.predict(X_te)
        lr_shifted = cross_correlate_lag(y_te[:-1], p_shifted[1:], max_lag)
        ol_s, _ = find_optimal_lag(lr_shifted)
        # If the hack helps (ol_s closer to 0), that confirms the bug is real.
        # If ol_s = -1, the hack is INTRODUCING a shift.
        out["AR_hack"] = {"optimal_lag": ol_s, "lag_results": lr_shifted,
                          "improvement_%": mse_improvement_pct(lr_shifted, ol_s),
                          "note": "y_test[:-1] vs pred[1:] (current code hack)"}
    except Exception as e:
        out["AR_hack"] = {"error": str(e)}

    # 4) DS3M (optional — slower)
    if include_ds3m:
        try:
            from experiments.competitor_models import DS3MWrapper
            ds = DS3MWrapper(lags=lags, problem=dataname, target_dim=dim,
                             device=device, test_len_override=test_len)
            ds.fit(X, y)
            p_all = ds.predict(X)
            p_te = p_all[tr_end:]
            mn = min(len(p_te), len(y_te))
            lr = cross_correlate_lag(y_te[:mn], p_te[:mn], max_lag)
            ol, _ = find_optimal_lag(lr)
            out["DS3M"] = {
                "optimal_lag": ol, "lag_results": lr,
                "improvement_%": mse_improvement_pct(lr, ol),
                "pred_len": len(p_te), "y_test_len": len(y_te),
                "ds3m_test_len": ds._test_len,
                "length_match": len(p_te) == len(y_te),
            }
        except Exception as e:
            out["DS3M"] = {"error": str(e)}

    return out


# ---------------------------------------------------------------------------
# Pretty-print report
# ---------------------------------------------------------------------------

def print_report(all_results):
    hdr = f"{'Dataset':<15} {'Model':<10} {'OptLag':<8} {'MSE@0':<12} {'MSE@opt':<12} {'Corr@0':<9} {'Impr%':<8} {'Status'}"
    sep = "-" * len(hdr)

    print("\n" + "=" * len(hdr))
    print("ALIGNMENT DIAGNOSTIC REPORT")
    print("=" * len(hdr))
    print(hdr)
    print(sep)

    issues = []
    for ds, models in all_results.items():
        if models is None:
            continue
        for name, info in models.items():
            if "error" in info:
                print(f"{ds:<15} {name:<10} ERROR: {info['error'][:50]}")
                continue

            ol = info["optimal_lag"]
            lr = info["lag_results"]
            mse0 = lr.get(0, {}).get("mse", float("nan"))
            corr0 = lr.get(0, {}).get("corr", float("nan"))
            mse_opt = lr.get(ol, {}).get("mse", float("nan"))
            imp = info.get("improvement_%", 0)

            if ol != 0 and imp > 1.0:
                status = f"SHIFT={ol:+d}"
                issues.append((ds, name, ol, imp))
            elif ol != 0 and imp <= 1.0:
                status = "~  (negligible)"
            else:
                status = "OK"

            print(f"{ds:<15} {name:<10} {ol:<8} {mse0:<12.4f} {mse_opt:<12.4f} {corr0:<9.4f} {imp:<8.1f} {status}")

    print(sep)

    # --- Summary & interpretation ---
    print("\nINTERPRETATION GUIDE:")
    print("  OptLag=0 for AR & Persist  -> create_lag_features indexing is CORRECT")
    print("  OptLag!=0 for AR & Persist -> create_lag_features has an off-by-one BUG")
    print("  OptLag=0 for DS3M          -> DS3MWrapper alignment is CORRECT")
    print("  OptLag!=0 for DS3M only    -> DS3MWrapper tail placement is misaligned")
    print("  AR_hack OptLag=-1          -> the [1:]/[:-1] hack INTRODUCES a shift (remove it!)")
    print("  AR_hack OptLag=0           -> the hack is compensating for a real shift\n")

    if issues:
        print("ISSUES FOUND:")
        for ds, model, lag, imp in issues:
            d = "ahead" if lag > 0 else "behind"
            print(f"  {ds}/{model}: optimal lag={lag:+d} ({abs(lag)} step(s) {d}), "
                  f"MSE improves by {imp:.1f}% when corrected")
    else:
        print("No significant alignment issues detected.")
        print("   If supervisors see 'visual lag', it's the autoregressive tracking artifact,")
        print("   NOT an indexing bug.  Remove the pred[1:]/y_test[:-1] hack.\n")

    return issues


# ---------------------------------------------------------------------------
# Detailed per-lag table (optional verbose mode)
# ---------------------------------------------------------------------------

def print_lag_table(all_results, dataset, model):
    """Print full lag → MSE/corr table for one (dataset, model)."""
    info = all_results.get(dataset, {}).get(model, {})
    if not info or "error" in info:
        print(f"No data for {dataset}/{model}")
        return
    lr = info["lag_results"]
    print(f"\n  Lag table for {dataset}/{model}:")
    print(f"  {'Lag':<6} {'MSE':<14} {'Corr':<10} {'N'}")
    for lag in sorted(lr):
        m = lr[lag]
        marker = " <-- optimal" if lag == info["optimal_lag"] else ""
        print(f"  {lag:<6} {m['mse']:<14.6f} {m['corr']:<10.6f} {m['n']}{marker}")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Alignment diagnostic")
    p.add_argument("--datasets", nargs="+",
                   default=["Toy", "Sleep", "Unemployment", "Electricity",
                            "Lorenz", "Hangzhou", "Seattle", "Pacific", "Pernod"])
    p.add_argument("--include-ds3m", action="store_true",
                   help="Also diagnose DS3M alignment (slower)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-lag", type=int, default=5)
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print full lag tables")
    args = p.parse_args()

    all_results = {}
    for ds in args.datasets:
        print(f"\n[{ds}] diagnosing...")
        all_results[ds] = diagnose_dataset(
            ds, include_ds3m=args.include_ds3m,
            device=args.device, max_lag=args.max_lag
        )

    issues = print_report(all_results)

    if args.verbose:
        for ds in args.datasets:
            if all_results.get(ds) is None:
                continue
            for model in all_results[ds]:
                print_lag_table(all_results, ds, model)

    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
