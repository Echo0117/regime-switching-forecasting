# ds3m_with_acp.py  ─────────────────────────────────────────────
# -*- coding: utf-8 -*-
import argparse, os
import numpy as np
import os, sys

# from experiments.interval_evaluate_utils import evaluate_intervals

HERE = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from experiments.utils.experiments_utils import (
    load_forecast,
    plot_results_with_ds3m_aci,
    save_forecast,
)
from experiments.utils.acp_utils import agaci_ewa, aci_intervals
from experiments.utils.ds3m_utils import (
    forecast,
    load_ds3m_data,
    load_ds3m_model,
    plot_results,
)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem",
        default="Pacific",
        choices=[
            "Toy",
            "Lorenz",
            "Sleep",
            "Unemployment",
            "Hangzhou",
            "Seattle",
            "Pacific",
            "Electricity",
        ],
    )
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--train_size", type=int, default=100, help="T0 used as calibration size for ACP"
    )
    parser.add_argument(
        "--force_new", default=True, action="store_true", help="Ignore cache and recompute forecast"
    )
    args = parser.parse_args()
    print(args)

    # 0) read cache if exists
    if not args.force_new:
        cached = load_forecast(args.problem)
    else:
        cached = None

    if cached is None:
        # 1) load data and model
        ds = load_ds3m_data(args)
        model = load_ds3m_model(
            ds["directoryBest"],
            ds["x_dim"],
            ds["y_dim"],
            ds["h_dim"],
            ds["z_dim"],
            ds["d_dim"],
            ds["n_layers"],
            ds["learning_rate"],
            ds["device"],
            bidirection=ds["bidirection"],
        )

        # 2) prediction
        (
            res,
            testForecast_mean,
            testOriginal,
            size,
            forecast_d_MC_argmax,
            testForecast_uq,
            testForecast_lq,
        ) = forecast(
            model,
            ds["testX"],
            ds["testY"],
            ds["moments"],
            ds["d_dim"],
            ds["means"],
            ds["trend"],
            ds["test_len"],
            ds["freq"],
            ds["RawDataOriginal"],
            remove_mean=ds["remove_mean"],
            remove_residual=ds["remove_residual"],
        )
        plot_results(
            model,
            testForecast_mean,
            testOriginal,
            size,
            forecast_d_MC_argmax,
            testForecast_uq,
            testForecast_lq,
            ds["RawDataOriginal"],
            ds["d_dim"],
            ds["predict_dim"],
            args.problem,
            ds["figdirectory"],
            ds["device"],
            ds["test_len"],
            z_true=ds.get("z_true", None),
            trend=ds.get("trend", None),
            testX=ds["testX"],
            testY=ds["testY"],
            data=ds["data"],
            states=ds.get("states", None),
            res=res,
            moments=ds["moments"],
            freq=ds["freq"],
            means=ds["means"],
            remove_mean=ds["remove_mean"],
            remove_residual=ds["remove_residual"],
        )

        res_dict = dict(
            y_pred_mean=testForecast_mean,
            y_true=testOriginal,
            size=size,
            d_argmax=forecast_d_MC_argmax,
            y_uq=testForecast_uq,
            y_lq=testForecast_lq,
            res_metric=res,
            problem=args.problem,
            train_size=args.train_size,
            alpha=args.alpha,
            gamma=args.gamma,
            seed=args.seed,
            device=ds["device"],
            test_len=ds["test_len"],
            z_true=ds.get("z_true", None),
            trend=ds.get("trend", None),
            testX=ds["testX"],
            testY=ds["testY"],
            data=ds["data"],
            states=ds.get("states", None),
            res=res,
            moments=ds["moments"],
            freq=ds["freq"],
            means=ds["means"],
            remove_mean=ds["remove_mean"],
            remove_residual=ds["remove_residual"],
            figdirectory=ds["figdirectory"],
            d_dim=ds["d_dim"],
            predict_dim=ds["predict_dim"],
        )

        # 3) save forecast to cache
        save_forecast(res_dict, args.problem)
    else:
        # use cached results
        res_dict = cached

    # ============== 下面做 ACP 区间  ==================
    y_pred_mean = res_dict["y_pred_mean"]  # (T, D)
    y_true = res_dict["y_true"]  # (T, D)
    size = res_dict["size"]
    d_argmax = res_dict["d_argmax"]
    y_uq = res_dict["y_uq"]  # (T, D)
    y_lq = res_dict["y_lq"]  # (T, D
    d_dim = res_dict["d_dim"]  # dimension

    residuals = np.abs(y_true[:, 0] - y_pred_mean[:, 0]).ravel()
    print("[DEBUG] Residual summary: min=%.3f max=%.3f mean=%.3f p90=%.3f p95=%.3f p99=%.3f" % (
        np.min(residuals), np.max(residuals), np.mean(residuals),
        np.quantile(residuals, 0.90),
        np.quantile(residuals, 0.95),
        np.quantile(residuals, 0.99)))


    # calculate residuals
    y_true_1d = y_true[:, 0]
    y_pred_1d = y_pred_mean[:, 0]
    residuals = np.abs(y_true_1d - y_pred_1d)
    T0 = args.train_size

    # use ACI or AgACI
    # use_agaci = True
    # if use_agaci:
    lower_agaci, upper_agaci = agaci_ewa(
        residuals,
        alpha=args.alpha,
        train_size=T0,
        gammas=[0.005, 0.009, 0.011, 0.015, 0.025, 0.04],
        # agg="ewa",
    )
    # else:
    lower_r, upper_r = aci_intervals(
        residuals, alpha=args.alpha, gamma=args.gamma, train_size=T0
    )

    # calculate coverage and width
    y_pred_test = y_pred_1d[T0:]
    y_true_test = y_true_1d[T0:]
    coverage = np.mean(
        (y_true_test >= (y_pred_test - upper_r))
        & (y_true_test <= (y_pred_test + upper_r))
    )
    width = np.mean(2 * upper_r)
    print(f"[ACI/AgACI] coverage={coverage:.3f}, width={width:.3f}")

    # plot results with ACI intervals
    plot_results_with_ds3m_aci(
        args.problem,
        y_true,
        y_pred_mean,
        dsm_lower=y_lq,
        dsm_upper=y_uq,
        d_dim=d_dim,  
        forecast_d_MC_argmax=d_argmax,
        # ACI related
        aci_lower=lower_r,
        aci_upper=upper_r,
        T0=T0,
        target_dim=0,
        coverage=coverage,
        width=width,
    )

        # ===================== Metrics (test segment only) =====================
    # test segment is t >= T0 by construction of ACI
    y_pred_test = y_pred_1d[T0:]
    y_true_test = y_true_1d[T0:]

    def _rmse(a, b):
        return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

    # ---- ACI metrics ----
    rmse_aci = _rmse(y_true_test, y_pred_test)
    cov_aci = float(np.mean((y_true_test >= (y_pred_test - upper_r)) &
                            (y_true_test <= (y_pred_test + upper_r))))
    medlen_aci = float(np.median(2.0 * upper_r))
    pct_inf_aci = float(np.mean(np.isinf(upper_r)))

    # ---- AgACI metrics ----
    cov_agaci = float(np.mean((y_true_test >= (y_pred_test - upper_agaci)) &
                              (y_true_test <= (y_pred_test + upper_agaci))))
    medlen_agaci = float(np.median(2.0 * upper_agaci))
    pct_inf_agaci = float(np.mean(np.isinf(upper_agaci)))
    # RMSE is the same (interval method doesn't change the point forecast)
    rmse_agaci = rmse_aci

    # Pretty print
    print("\n=== DS³M metrics (test window) ===")
    print(f"[ACI]    RMSE={rmse_aci:.6f} | Coverage@{int((1-args.alpha)*100)}%={cov_aci:.3f} | "
          f"MedianLen={medlen_aci:.6f} | %Infinite={pct_inf_aci:.3f}")
    print(f"[AgACI]  RMSE={rmse_agaci:.6f} | Coverage@{int((1-args.alpha)*100)}%={cov_agaci:.3f} | "
          f"MedianLen={medlen_agaci:.6f} | %Infinite={pct_inf_agaci:.3f}")

    # ===================== (Optional) Append to a CSV =====================
    # You can change this path if you like
    out_csv = os.path.join(PROJ_ROOT, "experiments", "results_ds3m_single.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    def _append_row(csv_path, row_vals):
        header = ["Problem","Model","IntervalMethod","RMSE","Coverage@90","MedianLen","PctInfinite"]
        write_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
        with open(csv_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write(",".join(header) + "\n")
            f.write(",".join(str(v) for v in row_vals) + "\n")

    _append_row(out_csv, [args.problem, "ds3m", "ACI",    rmse_aci,   cov_aci,   medlen_aci,   pct_inf_aci])
    _append_row(out_csv, [args.problem, "ds3m", "AgACI",  rmse_agaci, cov_agaci, medlen_agaci, pct_inf_agaci])


if __name__ == "__main__":
    main()
