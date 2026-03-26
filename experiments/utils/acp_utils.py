# experiments/utils/aci_original_adapter.py

import numpy as np
from typing import Tuple, Dict, Optional

# Import the ORIGINAL ACI implementation you pasted:
# Make sure this import path points to the module where fit_predict lives.
from AdaptiveConformalPredictionsTimeSeries.models import fit_predict, fit_predict_ACPs  # adjust path if needed
from AdaptiveConformalPredictionsTimeSeries.agaci import run_agaci

def aci_intervals(
    X: np.ndarray,            # (N, D) features, time-major
    y: np.ndarray,            # (N,) targets
    # alpha: float,                 # miscoverage, e.g. 0.1 for 90% PI
    # tab_gamma: list[float] = [0.005, 0.01, 0.02, 0.05],                 # ACI step
    basemodel: str = "ds3m",      # "RF" or "OLS" 
    params_basemodel: Dict = None,
    # train_size: int = 100,   # T0
    args = None

) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Call ORIGINAL ACI (fit_predict) and convert outputs into:
      y_hat_seg : (N - train_size,) center predictions from RF/OLS
      lo_r      : zeros of shape (N - train_size,)
      up_r      : half-widths q_t from ORIGINAL ACI (upper-lower)/2

    Notes
    -----
    - The original implementation expects X as (d, n), columns=time.
    - Center = (lower + upper)/2 is the original ACI’s mean-regressor center,
      not your DS3M center. If you want DS3M as center, you can still use
      'up_r' from here and replace 'y_hat_seg' with your external predictions.
    """
    y_lowers, y_uppers, tab_alpha_t, gammas = fit_predict_ACPs(X, y, args.alpha, args.tab_gamma, basemodel, params_basemodel, args.aci_train_size, args)

    return y_lowers, y_uppers, tab_alpha_t, gammas


def agaci_intervals(
    X: np.ndarray,
    y: np.ndarray,
    basemodel: str = "ds3m",
    params_basemodel: Dict = None,
    args = None
) -> Dict:
    """
    Run AgACI (Aggregated Adaptive Conformal Inference).

    First runs ACI with multiple gammas, then aggregates using BOA.

    Parameters
    ----------
    X, y : np.ndarray
        Input features and targets
    basemodel : str
        Base model type ("ds3m", "RF", "OLS")
    params_basemodel : dict
        Parameters for base model
    args : argparse.Namespace
        Arguments containing alpha, tab_gamma, aci_train_size, agaci_eta

    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'lower': Aggregated lower bounds
        - 'upper': Aggregated upper bounds
        - 'weights_lower': BOA weights for lower bounds (T, n_gammas)
        - 'weights_upper': BOA weights for upper bounds (T, n_gammas)
        - 'y_lowers_experts': Expert lower bounds (n_gammas, T)
        - 'y_uppers_experts': Expert upper bounds (n_gammas, T)
        - 'gammas': Gamma values used
        - Additional metrics
    """
    # First, run ACI with multiple gammas to get experts
    y_lowers_experts, y_uppers_experts, tab_alpha_t, gammas = fit_predict_ACPs(
        X, y, args.alpha, args.tab_gamma, basemodel, params_basemodel, args.aci_train_size, args
    )

    # Get ground truth for test period (must match the scale of expert bounds)
    test_len = y_lowers_experts.shape[1]  # test_size_eff
    T0 = int(args.aci_train_size)

    if basemodel == "ds3m":
        # IMPORTANT: Use testOriginal from forecast() — the de-normalized ground truth
        # that matches the scale of expert bounds from fit_predict_ACPs.
        # Previously this used ds["data"] which can be in a different (normalized) scale,
        # causing catastrophic coverage failure for real datasets like Hangzhou.
        from experiments.utils.ds3m_utils import load_ds3m_data, load_ds3m_model, forecast

        ds = load_ds3m_data(args)
        model = load_ds3m_model(
            ds["directoryBest"],
            ds["x_dim"], ds["y_dim"], ds["h_dim"], ds["z_dim"],
            ds["d_dim"], ds["n_layers"], ds["learning_rate"],
            ds["device"], bidirection=ds["bidirection"],
        )

        _, _, testOriginal, _, _, _, _ = forecast(
            model,
            ds["testX"], ds["testY"],
            ds["moments"], ds["d_dim"],
            ds["means"], ds["trend"],
            ds["test_len"], ds["freq"],
            ds["RawDataOriginal"],
            remove_mean=ds["remove_mean"],
            remove_residual=ds["remove_residual"],
        )

        y_true_tail = np.asarray(testOriginal, dtype=float)
        if y_true_tail.ndim == 1:
            y_true_tail = y_true_tail[:, None]

        target_dim = int(ds.get("target_dim", 0))
        D = y_true_tail.shape[1]
        target_dim = max(0, min(target_dim, D - 1))

        y_true_test = y_true_tail[T0:T0 + test_len, target_dim]
    else:
        # For RF/OLS, it's simpler
        y_full = np.asarray(y)
        if y_full.ndim > 1:
            target_dim = int(getattr(args, 'target_dim', 0))
            y_full = y_full[:, target_dim]
        y_true_test = y_full[T0:T0 + test_len]

    # Run AgACI aggregation
    eta = float(getattr(args, 'agaci_eta', 0.1))
    use_gradient = bool(getattr(args, 'agaci_gradient', True))
    lr_schedule = str(getattr(args, 'agaci_lr_schedule', 'constant'))
    coverage_loss = bool(getattr(args, 'agaci_coverage_loss', True))
    width_penalty = float(getattr(args, 'agaci_width_penalty', 0.1))

    print(f"  [AgACI] experts={y_lowers_experts.shape[0]}, T={y_lowers_experts.shape[1]}, "
          f"eta={eta}, lr={lr_schedule}, coverage_loss={coverage_loss}")

    agaci_results = run_agaci(
        y_lowers_experts,
        y_uppers_experts,
        y_true_test,
        alpha=args.alpha,
        eta=eta,
        use_gradient=use_gradient,
        lr_schedule=lr_schedule,
        verbose=False,
        coverage_loss=coverage_loss,
        width_penalty=width_penalty,
    )

    # Add expert information to results
    agaci_results['y_lowers_experts'] = y_lowers_experts
    agaci_results['y_uppers_experts'] = y_uppers_experts
    agaci_results['gammas'] = gammas
    agaci_results['tab_alpha_t'] = tab_alpha_t

    return agaci_results


def run_other_cp_methods(
    X: np.ndarray,
    y: np.ndarray,
    methods: list = ['Gaussian', 'CP', 'EnbPI'],
    basemodel: str = "ds3m",
    params_basemodel: Dict = None,
    args = None
) -> Dict:
    """
    Run other conformal prediction methods for comparison.

    Parameters
    ----------
    X, y : np.ndarray
        Input features and targets
    methods : list of str
        List of methods to run. Options: 'Gaussian', 'CP', 'EnbPI', 'EnbPI_Mean'
    basemodel : str
        Base model type ("ds3m", "RF", "OLS")
    params_basemodel : dict
        Parameters for base model
    args : argparse.Namespace
        Arguments containing alpha, aci_train_size

    Returns
    -------
    results : dict
        Dictionary mapping method name to (lower, upper) tuple
        Each lower/upper is a 1D array of shape (test_size_eff,)
    """
    from scipy.stats import norm

    # For DS3M, we use a special path similar to fit_predict_ACPs
    if basemodel == "ds3m":
        from experiments.utils.ds3m_utils import load_ds3m_data, load_ds3m_model, forecast

        # Load DS3M data/model
        if params_basemodel is not None and 'ds' in params_basemodel:
            ds = params_basemodel['ds']
        else:
            ds = load_ds3m_data(args)

        if params_basemodel is not None and 'model' in params_basemodel:
            model = params_basemodel['model']
        else:
            model = load_ds3m_model(
                ds["directoryBest"],
                ds["x_dim"], ds["y_dim"], ds["h_dim"], ds["z_dim"],
                ds["d_dim"], ds["n_layers"], ds["learning_rate"],
                ds["device"], bidirection=ds["bidirection"],
            )

        # Get forecasts
        _, testForecast_mean, testOriginal, _, _, _, _ = forecast(
            model,
            ds["testX"], ds["testY"],
            ds["moments"], ds["d_dim"],
            ds["means"], ds["trend"],
            ds["test_len"], ds["freq"],
            ds["RawDataOriginal"],
            remove_mean=ds["remove_mean"],
            remove_residual=ds["remove_residual"],
        )

        # Extract predictions and ground truth
        y_hat_tail = np.asarray(testForecast_mean, dtype=float)
        if y_hat_tail.ndim == 1:
            y_hat_tail = y_hat_tail[:, None]

        y_true_tail = np.asarray(testOriginal, dtype=float)
        if y_true_tail.ndim == 1:
            y_true_tail = y_true_tail[:, None]

        # Get dimensions
        target_dim = int(ds["target_dim"])
        D = y_hat_tail.shape[1] if y_hat_tail.ndim > 1 else 1
        target_dim = max(0, min(target_dim, D - 1))

        test_len = int(ds["test_len"])
        T0 = int(args.aci_train_size)
        test_size_eff = max(0, test_len - T0)

        if test_size_eff == 0:
            raise ValueError("test_len must be > train_size for DS3M.")

        m = T0 // 2  # Half/half split for train/calibration

        # Initialize outputs for each method
        results = {}
        for method in methods:
            y_lowers = np.empty(test_size_eff, dtype=float)
            y_uppers = np.empty(test_size_eff, dtype=float)

            # Rolling window over test set
            for j in range(test_size_eff):
                cal_lo = j + m
                cal_hi = j + 2*m
                t_test = j + T0

                # Extract calibration data
                y_pred_cal = y_hat_tail[cal_lo:cal_hi, :] if D > 1 else y_hat_tail[cal_lo:cal_hi].reshape(-1, 1)
                y_true_cal = y_true_tail[cal_lo:cal_hi, :]

                # Compute residuals
                res_cal = np.abs(y_true_cal[:, target_dim] - y_pred_cal[:, target_dim])

                # Test point prediction
                y_pred = float(y_hat_tail[t_test, target_dim] if D > 1 else y_hat_tail[t_test])

                # Apply different conformal methods
                if method == "Gaussian":
                    # Gaussian interval: mean ± z * std
                    window = norm.ppf(1 - args.alpha/2) * np.std(res_cal)
                    y_lowers[j] = y_pred - window
                    y_uppers[j] = y_pred + window

                elif method == "CP":
                    # Standard Conformal Prediction
                    window = np.quantile(res_cal, (1 - args.alpha) * (1 + 1/len(res_cal)))
                    y_lowers[j] = y_pred - window
                    y_uppers[j] = y_pred + window

                elif method in ["EnbPI", "EnbPI_Mean"]:
                    # Residual-bootstrap EnbPI for DS3M
                    # Instead of training B models, we bootstrap the residuals
                    B = 50  # Number of bootstrap samples
                    use_mean = (method == "EnbPI_Mean")

                    # Bootstrap residuals to create B quantile estimates
                    bootstrap_quantiles = []
                    for b in range(B):
                        # Resample residuals with replacement
                        boot_idx = np.random.choice(len(res_cal), size=len(res_cal), replace=True)
                        res_boot = res_cal[boot_idx]

                        # Compute quantile for this bootstrap sample
                        q_boot = np.quantile(res_boot, (1 - args.alpha) * (1 + 1/len(res_boot)))
                        bootstrap_quantiles.append(q_boot)

                    # Aggregate bootstrap quantiles
                    if use_mean:
                        # EnbPI_Mean: use mean of bootstrap quantiles
                        window = np.mean(bootstrap_quantiles)
                    else:
                        # EnbPI: use quantile of bootstrap quantiles (more conservative)
                        window = np.quantile(bootstrap_quantiles, 1 - args.alpha)

                    y_lowers[j] = y_pred - window
                    y_uppers[j] = y_pred + window

            results[method] = (y_lowers, y_uppers)

        return results

    else:
        # For RF/OLS, use the original fit_predict function
        params_methods = {
            'online': True,
            'randomized': False,
        }

        # Add EnbPI parameters if needed
        if 'EnbPI' in methods or 'EnbPI_Mean' in methods:
            params_methods['B'] = 50
            params_methods['mean'] = 'EnbPI_Mean' in methods

            if params_basemodel is None:
                params_basemodel = {}
            params_basemodel.setdefault('n_estimators', 100)
            params_basemodel.setdefault('min_samples_leaf', 5)
            params_basemodel.setdefault('max_features', 'sqrt')
            params_basemodel.setdefault('cores', -1)

        # Call fit_predict
        y_lowers, y_uppers, times, times_proc = fit_predict(
            X.T,
            y,
            args.alpha,
            methods,
            params_methods,
            basemodel,
            params_basemodel,
            args.aci_train_size,
            args
        )

        # Package results
        results = {}
        for i, method in enumerate(methods):
            results[method] = (y_lowers[i], y_uppers[i])

        return results
