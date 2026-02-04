# experiments/plot_utils.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def _add_generic_time_labels(ax, dataname, n_points):
    """Add generic time labels when real timestamps are not available."""
    if dataname == "Unemployment":
        # Monthly data starting from 1948
        start_year = 1948
        freq = 12
        tick_interval = 12  # Show labels every year
        xticks = np.arange(9, n_points, tick_interval)
        xticklabels = [f"{start_year + i // freq} Jan" for i in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=0, fontsize=8)
    elif dataname in ["Hangzhou", "Seattle"]:
        tick_interval = n_points // 10
        if tick_interval > 0:
            xticks = np.arange(0, n_points, tick_interval)
            xticklabels = [f"t={i}" for i in xticks]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=45, fontsize=8)
    elif dataname == "Pacific":
        tick_interval = 12 * 5  # Show every 5 years
        if tick_interval > 0:
            xticks = np.arange(0, n_points, tick_interval)
            xticklabels = [f"Year {i // 12}" for i in xticks]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=0, fontsize=8)


def plot_results_with_aci(
    dataname: str,
    testOriginal: np.ndarray,              # shape (T, D)
    testForecast_mean: np.ndarray,         # shape (T, D)
    d_dim: int,
    forecast_d_MC_argmax=None,

    # DS³M MC intervals (optional)
    dsm_lower=None,
    dsm_upper=None,

    # ACI intervals
    aci_lower=None,                        # optional (for asymmetric)
    aci_upper=None,                        # upper bound or symmetric radius
    T0=None,                               # train size
    target_dim: int = 0,

    # meta
    coverage: float = None,
    width: float = None,
    model_name: str = None,
    interval_method_name: str = "ACI",
    save_dir_root: str = "figures",
    show: bool = True,
):
    """Unified plotting function for DS³M + ACI results."""
    # ------------------ helpers ------------------
    def _ensure_2d(a):
        if a is None: return None
        a = np.asarray(a)
        if a.ndim == 1:  return a.reshape(-1, 1)
        if a.ndim == 3 and a.shape[2] == 1: return a.reshape(a.shape[0], a.shape[1])
        return a

    def _match_tail(x, T):
        """Trim or pad interval to match last T points."""
        if x is None: return None
        x = np.asarray(x).reshape(-1)
        L = min(len(x), T)
        return x[-L:]

    # ------------------ data prep ------------------
    testOriginal = _ensure_2d(testOriginal)
    testForecast_mean = _ensure_2d(testForecast_mean)
    T, D = testOriginal.shape

    assert 0 <= target_dim < D, f"Invalid target_dim={target_dim} for D={D}"

    y_true = testOriginal[:, target_dim]
    y_pred = testForecast_mean[:, target_dim]
    tt = np.arange(T)

    plt.figure(figsize=(11.5, 4.2))
    plt.plot(tt, y_true, color="black", lw=1.0, alpha=0.65, label="True")
    plt.plot(tt, y_pred, color="tab:blue", lw=1.2, label="Prediction")

    # ------------------ DS³M intervals ------------------
    if dsm_lower is not None and dsm_upper is not None:
        dsm_lower = _ensure_2d(dsm_lower)
        dsm_upper = _ensure_2d(dsm_upper)
        # If already extracted for target dimension (shape: (T, 1)), squeeze it
        # Otherwise, extract the target dimension
        if dsm_lower.shape[1] == 1:
            dsm_lower = dsm_lower[:, 0]
            dsm_upper = dsm_upper[:, 0]
        else:
            dsm_lower = dsm_lower[:, target_dim]
            dsm_upper = dsm_upper[:, target_dim]
        L = min(len(dsm_lower), T)
        plt.fill_between(tt[-L:], dsm_lower[-L:], dsm_upper[-L:], color="gray", alpha=0.25, label="DS³M MC interval")

    # ------------------ ACI intervals ------------------
    if (aci_lower is not None) or (aci_upper is not None):
        aci_lower = np.asarray(aci_lower).reshape(-1) if aci_lower is not None else None
        aci_upper = np.asarray(aci_upper).reshape(-1) if aci_upper is not None else None

        # choose alignment mode
        use_tail_mode = True
        if (T0 is not None) and (0 <= T0 < T):
            expected = T - T0
            if (aci_upper is not None and len(aci_upper) == expected) or \
               (aci_lower is not None and len(aci_lower) == expected):
                use_tail_mode = False

        if use_tail_mode:
            # ---- align to tail ----
            L = 0
            if aci_upper is not None: L = max(L, len(aci_upper))
            if aci_lower is not None: L = max(L, len(aci_lower))
            L = min(L, T)

            tt_tail = np.arange(T - L, T)
            yp_tail = y_pred[-L:]

            if (aci_lower is not None) and (aci_upper is not None) and len(aci_lower) == len(aci_upper):
                plt.fill_between(tt_tail, aci_lower[-L:], aci_upper[-L:], color="orange", alpha=0.3,
                                 label=f"{interval_method_name} interval")
            elif aci_upper is not None:
                r = _match_tail(aci_upper, T)
                plt.fill_between(tt_tail, yp_tail - r, yp_tail + r, color="orange", alpha=0.3,
                                 label=f"{interval_method_name} interval")
        else:
            # ---- align to [T0:T) ----
            test_len = T - T0
            tt_test = np.arange(T0, T)
            yp_test = y_pred[T0:T]

            if (aci_lower is not None) and (aci_upper is not None) and len(aci_lower) == len(aci_upper):
                lo = aci_lower[:test_len]
                up = aci_upper[:test_len]
                plt.fill_between(tt_test, lo, up, color="orange", alpha=0.3,
                                 label=f"{interval_method_name} interval")
            elif aci_upper is not None:
                r = aci_upper[:test_len]
                plt.fill_between(tt_test, yp_test - r, yp_test + r, color="orange", alpha=0.3,
                                 label=f"{interval_method_name} interval")

    # ------------------ title / legend / save ------------------
    title_parts = []
    if model_name: title_parts.append(model_name)
    if dataname: title_parts.append(dataname)
    title_parts.append(f"dim={target_dim}")
    if interval_method_name: title_parts.append(interval_method_name)
    if coverage is not None: title_parts.append(f"cov={coverage:.3f}")
    if width is not None: title_parts.append(f"width={width:.1f}")

    plt.title(" | ".join(title_parts))
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.join(save_dir_root, "aci_results"), exist_ok=True)
    out_path = os.path.join(
        save_dir_root, "aci_results",
        f"{dataname}__{model_name or 'model'}__{interval_method_name}__dim{target_dim}.png"
    )
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"[FIG] Saved: {out_path}")

    if forecast_d_MC_argmax is not None and d_dim is not None:
        try:
            # Debug: Check regime data before plotting
            print(f"\n[DEBUG REGIME HEATMAP]")
            print(f"  forecast_d_MC_argmax shape: {forecast_d_MC_argmax.shape}")
            print(f"  forecast_d_MC_argmax dtype: {forecast_d_MC_argmax.dtype}")
            print(f"  Min: {forecast_d_MC_argmax.min()}, Max: {forecast_d_MC_argmax.max()}")
            print(f"  Unique values: {np.unique(forecast_d_MC_argmax)}")
            print(f"  d_dim: {d_dim}")

            arr = forecast_d_MC_argmax
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr.reshape(-1)
            elif arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr.reshape(-1)
            elif arr.ndim == 2 and arr.shape[0] == T:
                arr = arr[:, 0]  # if (T,D) keep dim 0 for heatmap

            # Use exact same style as original DS3M code (main.py:728)
            # cmap = plt.get_cmap('RdBu', d_dim)
            # sns.heatmap(1-forecast_d_MC_argmax.reshape(1, -1), linewidth=0,
            #             cbar=False, alpha=1, cmap=cmap, vmin=0, vmax=1, ax=ax2)
            cmap_states = plt.get_cmap("RdBu", d_dim if d_dim is not None else 2)
            plt.figure(figsize=(11.5, 1.8))

            # Normalize arr to [0, 1] range for visualization
            # For d_dim=2: invert using (1 - arr) like original code
            # For d_dim>2: normalize to [0, 1] using (d_dim-1-arr)/(d_dim-1)
            if d_dim == 2:
                # Original DS3M style for binary regimes
                arr_normalized = 1 - arr
                vmax_val = 1
            else:
                # For multi-regime case, normalize to [0, 1]
                arr_normalized = (d_dim - 1 - arr) / (d_dim - 1) if d_dim > 1 else arr
                vmax_val = 1

            ax = sns.heatmap(
                arr_normalized.reshape(1, -1),
                linewidth=0,
                cbar=False,
                alpha=1,
                cmap=cmap_states,
                vmin=0,
                vmax=vmax_val,
            )

            # Add time labels for datasets with temporal information
            # Try to load real timestamps from dataset
            n_points = len(arr)
            try:
                from experiments.utils.regime_switch_analysis import load_timestamps_for_dataset
                timestamps = load_timestamps_for_dataset(dataname, n_points)

                if timestamps is not None and len(timestamps) == n_points:
                    # Use real timestamps - show ~12 labels
                    tick_interval = max(1, n_points // 12)
                    xticks = np.arange(0, n_points, tick_interval)
                    xticklabels = [timestamps[i] for i in xticks]
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels, rotation=45, fontsize=8, ha='right')
                else:
                    # Fallback to generic labels
                    _add_generic_time_labels(ax, dataname, n_points)
            except Exception as e:
                # If loading fails, use generic labels
                _add_generic_time_labels(ax, dataname, n_points)

            plt.title(f"{dataname} | {model_name or 'DS3M'} discrete states")
            plt.yticks([])
            plt.xlabel("time")
            plt.tight_layout()

            hdir = os.path.join(save_dir_root, "regime_heatmap")
            os.makedirs(hdir, exist_ok=True)
            out_hm = os.path.join(hdir, f"{dataname}__{model_name}__regime.png")
            plt.savefig(out_hm, dpi=200, bbox_inches="tight")
            if show:
                plt.show()
            plt.close()
            print(f"[FIG] Saved: {out_hm}")
        except Exception as e:
            print(f"[WARN] Plot regime heatmap failed: {e}")

