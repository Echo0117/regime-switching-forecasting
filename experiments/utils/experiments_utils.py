import os, json
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pathlib import Path
import sys
import seaborn as sns

# Custom low-saturation red-blue colormap (matching Task 2)
CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    'soft_redblue',
    ['#85b0d0', '#e59999'],  # Deeper soft blue to deeper soft red
    N=256
)


HERE = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


CACHE_DIR = Path("cached_forecasts")
CACHE_DIR.mkdir(exist_ok=True)


def cache_path(problem: str) -> Path:
    """generate the cache file path for the dataset"""
    return CACHE_DIR / f"{problem}_ds3m_forecast.npz"


def save_forecast(result_dict: dict, problem: str) -> None:
    path = cache_path(problem)
    np.savez_compressed(
        path,
        y_pred_mean=result_dict["y_pred_mean"],
        y_true=result_dict["y_true"],
        size=result_dict["size"],
        d_argmax=result_dict.get("d_argmax"),
        y_uq=result_dict.get("y_uq"),
        y_lq=result_dict.get("y_lq"),
        res_metric=json.dumps(result_dict.get("res_metric", {})),
        figdirectory=result_dict.get("figdirectory", None),
        d_dim=result_dict.get("d_dim", None),
        predict_dim=result_dict.get("predict_dim", None),
        test_len=result_dict.get("test_len", None),
        RawDataOriginal=result_dict.get("RawDataOriginal", None),
        device=result_dict.get("device", None),
        testX=result_dict.get("testX", None),
        testY=result_dict.get("testY", None),
        data=result_dict.get("data", None),
        states=result_dict.get("states", None),
        res=result_dict.get("res", None),
        moments=result_dict.get("moments", None),
        freq=result_dict.get("freq", None),
        means=result_dict.get("means", None),
        remove_mean=result_dict.get("remove_mean", False),
        remove_residual=result_dict.get("remove_residual", False),
        trend=result_dict.get("trend", None),
        z_true=result_dict.get("z_true", None),
    )
    print(f"[CACHE] forecast saved to {path}")


def load_forecast(problem: str):
    """cache file exists, load it; otherwise return None"""
    path = cache_path(problem)
    if path.exists():
        data = np.load(path, allow_pickle=True)
        print(f"[CACHE] loaded cached forecast from {path}")
        return dict(
            y_pred_mean=data["y_pred_mean"],
            y_true=data["y_true"],
            size=int(data["size"]),
            d_argmax=data["d_argmax"] if "d_argmax" in data else None,
            y_uq=data["y_uq"] if "y_uq" in data else None,
            y_lq=data["y_lq"] if "y_lq" in data else None,
            res_metric=(
                json.loads(data["res_metric"].item()) if "res_metric" in data else {}
            ),
            figdirectory=data.get("figdirectory", None),
            d_dim=data.get("d_dim", None),
            predict_dim=data.get("predict_dim", None),
            test_len=data.get("test_len", None),
            RawDataOriginal=data.get("RawDataOriginal", None),
            device=data.get("device", None),
            testX=data.get("testX", None),
            testY=data.get("testY", None),
            data=data.get("data", None),
            states=data.get("states", None),
            res=data.get("res", None),
            moments=data.get("moments", None),
            freq=data.get("freq", None),
            means=data.get("means", None),
            remove_mean=data.get("remove_mean", False),
            remove_residual=data.get("remove_residual", False),
            trend=data.get("trend", None),
            z_true=data.get("z_true", None),
        )
    return None

def plot_results_with_ds3m_aci(
    dataname,
    testOriginal,  # shape (T, D)  (or (T, D, 1) will reshape)
    testForecast_mean,  # shape (T, D)
    d_dim,
    forecast_d_MC_argmax,  # shape (T, D) or (T, 1)
    # ---- dsm related ----
    dsm_lower=None,
    dsm_upper=None,
    # ---- ACI related ----
    aci_lower=None,  # shape (T - T0,)
    aci_upper=None,  # shape (T - T0,)
    T0=None,  # train size T0
    target_dim=0,  # only plot this dimension
    coverage=None,  # coverage ratio
    width=None,  # average width of ACI intervals
):
    # check if testForecast_mean and testOriginal are 2D
    if testOriginal.ndim == 3 and testOriginal.shape[2] == 1:
        testOriginal = testOriginal.reshape(
            testOriginal.shape[0], testOriginal.shape[1]
        )
    if testForecast_mean.ndim == 3 and testForecast_mean.shape[2] == 1:
        testForecast_mean = testForecast_mean.reshape(
            testForecast_mean.shape[0], testForecast_mean.shape[1]
        )

    T = testOriginal.shape[0]
    assert (
        testForecast_mean.shape[0] == T
    ), "Time length mismatch for testForecast_mean/testOriginal"

    # only plot this dimension
    y_true_1d = testOriginal[:, target_dim]
    y_pred_1d = testForecast_mean[:, target_dim]

    td = target_dim
    tt = np.arange(T)

    print(f"[INFO] Plotting results for {dataname} (dim={target_dim})")
    print(f"[INFO] T={T}, target_dim={td}, d_dim={d_dim}")

    plt.figure(figsize=(10, 4))
    # ==================== plot results with DS³M MC intervals ====================

    if dsm_lower is not None and dsm_upper is not None:
        print("[INFO] Plotting DS³M MC intervals...")
        plt.plot(
            tt, y_true_1d, color="black", lw=1.0, label="True"
        )
        if dsm_lower.ndim == 2:
            dl = dsm_lower[:, td]
            du = dsm_upper[:, td]
        else:
            dl = dsm_lower
            du = dsm_upper

        plt.fill_between(
            tt, dl, du, color="gray", alpha=0.35, label="DS³M MC interval"
        )
    # ACI intervals
    if aci_lower is not None and aci_upper is not None and T0 is not None:
        tt_test = np.arange(T0, T)  # ACI coverage only applies to test set
        if len(aci_lower) != T - T0 or len(aci_upper) != T - T0:
            print(
                f"[WARN] ACI interval length {len(aci_lower)} != T-T0={T-T0}, auto-trim might be needed."
            )
            # auto-trim to match T-T0
            min_len = min(len(aci_lower), T - T0)
            aci_lower = aci_lower[:min_len]
            aci_upper = aci_upper[:min_len]
            tt_test = tt_test[:min_len]

        # ==================== plot results with ACI intervals ====================

        plt.plot(
            tt_test,
            y_true_1d[T0 : T0 + len(tt_test)],
            color="black",
            lw=1.0,
            label="True",
        )
        plt.plot(
            tt_test,
            y_pred_1d[T0 : T0 + len(tt_test)],
            color="tab:blue",
            lw=1.2,
            label="Pred",
        )

        # fill ACI intervals
        plt.fill_between(
            tt_test,
            y_pred_1d[T0 : T0 + len(tt_test)] - aci_upper,
            y_pred_1d[T0 : T0 + len(tt_test)] + aci_upper,
            color="orange",
            alpha=0.30,
            label="ACI/AgACI interval",
        )

        plt.title(
            f"{dataname}: Pred vs True with ACI intervals (dim={target_dim}), coverage={coverage:.3f}, width={width:.3f}"
        )
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

        fig_dir = os.path.join("figures", "aci_results")
        os.makedirs(fig_dir, exist_ok=True)

        plt.savefig(
            os.path.join(fig_dir, f"{dataname}_aci_plot_dim{target_dim}.png"),
            dpi=200,
            bbox_inches="tight",
        )

    else:
        print("[INFO] No ACI intervals provided; skipping ACI layer plotting.")

    # ====== optionally: draw regime heatmap ======
    if forecast_d_MC_argmax is not None:
        try:
            # Use exact same style as original DS3M code (main.py:728)
            cmap_states = plt.get_cmap("RdBu", d_dim if d_dim is not None else 2)
            plt.figure(figsize=(10, 1.8))

            # Normalize to [0, 1] range for visualization
            # For d_dim=2: invert using (1 - arr) like original code
            # For d_dim>2: normalize to [0, 1]
            arr = forecast_d_MC_argmax
            if d_dim == 2:
                arr_normalized = 1 - arr
            else:
                arr_normalized = (d_dim - 1 - arr) / (d_dim - 1) if d_dim > 1 else arr

            sns.heatmap(
                arr_normalized.reshape(1, -1),
                linewidth=0,
                cbar=False,
                alpha=1,
                cmap=cmap_states,
                vmin=0,
                vmax=1,
            )
            plt.title("{} DS³M discrete states (regime)".format(dataname))
            plt.yticks([])
            plt.xlabel("time")
            plt.tight_layout()
            plt.show()

            fig_dir = os.path.join("figures", "regime_heatmap")
            plt.savefig(
                os.path.join(fig_dir, f"{dataname}_regime_heatmap.png"),
                dpi=200,
                bbox_inches="tight",
            )
            print(f"[FIG] Saved regime heatmap to {fig_dir}/regime_heatmap.png")
        except Exception as e:
            print(f"[WARN] Plot regime heatmap failed: {e}")


def normalize_interval_lengths(results_dict: dict) -> dict:
    """
    Normalize interval lengths relative to the maximum across all methods.

    This makes interval lengths more interpretable by expressing them as percentages
    of the widest method (typically Naive). The method with the widest intervals = 100%,
    and all others are expressed relative to that.

    Parameters
    ----------
    results_dict : dict
        Dictionary with keys as method names and values as tuples (coverage, median_length)
        Example: {'Naive': (0.95, 47), 'ACI': (0.92, 40), 'AgACI': (0.91, 37)}

    Returns
    -------
    normalized_dict : dict
        Dictionary with keys as method names and values as tuples (coverage, relative_length_pct)
        Example: {'Naive': (0.95, 100.0), 'ACI': (0.92, 85.1), 'AgACI': (0.91, 78.7)}
    """
    if not results_dict:
        return {}

    # Find the maximum median length across all methods
    max_length = max(length for _, length in results_dict.values())

    if max_length == 0:
        # Edge case: if all lengths are 0, return 100% for all
        return {method: (cov, 100.0) for method, (cov, _) in results_dict.items()}

    # Normalize each method's length as percentage of max
    normalized_dict = {}
    for method, (coverage, length) in results_dict.items():
        relative_length_pct = (length / max_length) * 100.0
        normalized_dict[method] = (coverage, relative_length_pct)

    return normalized_dict
