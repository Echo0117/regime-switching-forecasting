# experiments/plot_utils.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def _ensure_2d(a):
    """Reshape (T, D, 1) -> (T, D). Leave (T, D) as-is."""
    if a is None:
        return None
    if a.ndim == 3 and a.shape[2] == 1:
        return a.reshape(a.shape[0], a.shape[1])
    return a

def _safe_trim(x, target_len):
    """Trim or pad (shouldn't pad) to match target_len."""
    if x is None:
        return None
    m = min(len(x), target_len)
    return x[:m]

def _ensure_2d(a):
    """Make array 2D: (T,) -> (T,1); (T,D,1) -> (T,D)."""
    if a is None:
        return None
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 3 and a.shape[2] == 1:
        return a.reshape(a.shape[0], a.shape[1])
    if a.ndim == 2:
        return a
    # As a last resort, flatten all but time
    return a.reshape(a.shape[0], -1)


def plot_results_with_aci(
    dataname,
    testOriginal,                # (T, D) or (T, D, 1)
    testForecast_mean,           # (T, D) or (T, D, 1)
    d_dim,
    forecast_d_MC_argmax=None,   # (T, D) or (T,) if available (DS3M regimes)

    # ---- DS³M intervals (backward-compatible) ----
    dsm_lower=None,              # same length as T (or None)
    dsm_upper=None,

    # ---- Generic model intervals (new) ----
    model_lower=None,            # e.g., MC-Dropout 5% quantile or GP lower
    model_upper=None,            # e.g., MC-Dropout 95% quantile or GP upper
    model_interval_label=None,   # legend label for model intervals (str)

    # ---- ACI / AgACI / Naive (radius-or-bounds) ----
    aci_lower=None,              # if you compute asym. bounds, you can pass both
    aci_upper=None,              # your current pipeline uses symmetric "upper" as radius
    T0=None,                     # train size
    target_dim=0,
    coverage=None,
    width=None,

    # ---- meta for titles & filenames (new) ----
    model_name=None,             # e.g., "S4", "CPD", "MCDropoutGRU", "GPTorchSparse", "DS3M"
    interval_method_name=None,   # e.g., "ACI", "AgACI", "Naive"
    save_dir_root="figures",     # base folder
    show=True,                   # call plt.show()
):
    """
    Backward compatible with your original DS3M plotter.
    Also supports generic model predictive intervals and method/model labeling.
    """
    testOriginal = _ensure_2d(testOriginal)
    testForecast_mean = _ensure_2d(testForecast_mean)

    if testOriginal is None or testForecast_mean is None:
        raise ValueError("testOriginal/testForecast_mean cannot be None.")
    T, D = testOriginal.shape
    if testForecast_mean.shape[0] != T:
        raise ValueError("Time length mismatch for testForecast_mean/testOriginal")
    if target_dim < 0 or target_dim >= D:
        raise ValueError(f"target_dim={target_dim} is out of range for D={D}")
    # If ACI arrays are longer than the available [T0:] segment, trim them safely
    if (aci_upper is not None) and (T0 is not None):
        test_len = max(0, T - T0)
        if len(aci_upper) != test_len:
            min_len = min(len(aci_upper), test_len)
            aci_upper = aci_upper[:min_len] 
            aci_lower = aci_lower[:min_len] if aci_lower is not None else None

    y_true_1d = testOriginal[:, target_dim]
    y_pred_1d = testForecast_mean[:, target_dim]
    tt = np.arange(T)

    # ---------- Figure 1: Prediction with optional model intervals + ACI ----------
    plt.figure(figsize=(11.5, 4.2))
    # Base truth curve (full horizon)
    plt.plot(tt, y_true_1d, color="black", lw=1.0, alpha=0.65, label="True (all)")

    # ----- DS³M MC intervals (legacy) or Generic Model intervals -----
    drew_model_band = False
    if dsm_lower is not None and dsm_upper is not None:
        dl = dsm_lower[:, target_dim] if dsm_lower.ndim == 2 else dsm_lower
        du = dsm_upper[:, target_dim] if dsm_upper.ndim == 2 else dsm_upper
        if len(dl) == T and len(du) == T:
            plt.fill_between(tt, dl, du, color="gray", alpha=0.25, label="DS³M MC interval")
            drew_model_band = True

    if (model_lower is not None) and (model_upper is not None):
        ml = model_lower[:, target_dim] if model_lower.ndim == 2 else model_lower
        mu = model_upper[:, target_dim] if model_upper.ndim == 2 else model_upper
        if len(ml) == T and len(mu) == T:
            lab = model_interval_label or "Model interval"
            plt.fill_between(tt, ml, mu, color="lightgray", alpha=0.3, label=lab)
            drew_model_band = True

    # ----- ACI (or AgACI/Naive) intervals on test split only -----
    if (aci_lower is not None or aci_upper is not None) and T0 is not None:
        test_len = T - T0
        tt_test = np.arange(T0, T)
        # Backward-compatible: your pipeline uses aci_upper as a symmetric radius "r_t".
        # If both provided, we still use aci_upper as radius; you can customize below if needed.
        # Trim if mismatched length
        if aci_upper is not None:
            aci_upper = _safe_trim(aci_upper, test_len)
            tt_test = tt_test[: len(aci_upper)]

        N_test = len(y_true_1d)          # == len(y_pred_1d) == len(aci_upper)
        T0 = 0
        T  = N_test
        tt_test = np.arange(N_test)

        print(f"[PLOT] ACI on test split t={T0}..{T-1} (len={len(tt_test)}), {tt_test.tolist()}")
        print(f"       y_true[{T0}:{T}]={y_true_1d[T0:T]}")
        print(f"       y_pred[{T0}:{T}]={y_pred_1d[T0:T]}")
        print(f"       aci_upper[:{N_test}]={aci_upper}")

        # print(f"[PLOT] ACI on test split t={T0}..{T-1} (len={len(tt_test)}), {tt_test}")
        # print(f"       y_true[{T0}:{T0+len(tt_test)}]  y_pred[{T0}:{T0+len(tt_test)}], {y_pred_1d[T0 : T0 + len(tt_test)]}")
        # print(f"       y_pred[{T0}:{T0+len(tt_test)}]={y_true_1d[T0 : T0 + len(tt_test)]}")
        # print(f"       aci_upper[:{len(tt_test)}]={aci_upper}") 
        
        # Overlay True + Pred on test window
        plt.plot(tt_test, y_true_1d[T0 : T0 + len(tt_test)], color="black", lw=1.0, label="True (test)")
        plt.plot(tt_test, y_pred_1d[T0 : T0 + len(tt_test)], color="tab:blue", lw=1.2, label="Pred")

        if aci_upper is not None and len(aci_upper) == len(tt_test):
            # symmetric band around y_pred
            lower_band = y_pred_1d[T0 : T0 + len(tt_test)] - aci_upper
            upper_band = y_pred_1d[T0 : T0 + len(tt_test)] + aci_upper
            plt.fill_between(tt_test, lower_band, upper_band, color="orange", alpha=0.30,
                             label=f"{interval_method_name or 'ACI'} interval")

    # Title with meta
    title_bits = [dataname, f"dim={target_dim}"]
    print(f"Plotting results for {model_name}")
    if model_name:
        title_bits.insert(0, model_name)
    if interval_method_name:
        title_bits.append(interval_method_name)
    if coverage is not None:
        title_bits.append(f"coverage={coverage:.3f}")
    if width is not None:
        title_bits.append(f"width={width:.3f}")
    plt.title(" | ".join(title_bits))
    print(f"Plotting title_bits for {title_bits}")
    # plt.title(" | ".join(str(title_bits)))
    plt.legend(loc="upper left")
    plt.tight_layout()

    # Save
    fig_dir = os.path.join(save_dir_root, "aci_results")
    os.makedirs(fig_dir, exist_ok=True)
    tag_model = (model_name or "model").replace(" ", "_")
    tag_method = (interval_method_name or "ACI").replace(" ", "_")
    out_png = os.path.join(fig_dir, f"{dataname}__{tag_model}__{tag_method}__dim{target_dim}.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"[FIG] Saved: {out_png}")

    # ---------- Figure 2 (optional): regime heatmap for DS³M ----------
    if forecast_d_MC_argmax is not None and d_dim is not None:
        try:
            arr = forecast_d_MC_argmax
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr.reshape(-1)
            elif arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr.reshape(-1)
            elif arr.ndim == 2 and arr.shape[0] == T:
                arr = arr[:, 0]  # if (T,D) keep dim 0 for heatmap

            cmap_states = plt.get_cmap("RdBu", d_dim if d_dim is not None else 2)
            plt.figure(figsize=(11.5, 1.8))
            sns.heatmap(
                arr.reshape(1, -1),
                linewidth=0,
                cbar=False,
                alpha=1,
                cmap=cmap_states,
                vmin=0,
                vmax=(d_dim - 1 if d_dim is not None else 1),
            )
            plt.title(f"{dataname} | {model_name or 'DS3M'} discrete states")
            plt.yticks([])
            plt.xlabel("time")
            plt.tight_layout()

            hdir = os.path.join(save_dir_root, "regime_heatmap")
            os.makedirs(hdir, exist_ok=True)
            out_hm = os.path.join(hdir, f"{dataname}__{tag_model}__regime.png")
            plt.savefig(out_hm, dpi=200, bbox_inches="tight")
            if show:
                plt.show()
            plt.close()
            print(f"[FIG] Saved: {out_hm}")
        except Exception as e:
            print(f"[WARN] Plot regime heatmap failed: {e}")
