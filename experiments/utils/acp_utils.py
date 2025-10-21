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

    # Get ground truth for test period
    # For DS3M, need to extract from the tail
    y_full = np.asarray(y)
    if y_full.ndim > 1:
        target_dim = int(getattr(args, 'target_dim', 0))
        y_full = y_full[:, target_dim]

    N = len(y_full)
    test_len = y_lowers_experts.shape[1]  # test_size_eff
    T0 = int(args.aci_train_size)

    # For DS3M, we need the tail starting at t0_tail + T0
    if basemodel == "ds3m":
        from experiments.utils.ds3m_utils import load_ds3m_data
        ds = load_ds3m_data(args)
        data_full = np.asarray(ds["data"])
        if data_full.ndim > 1:
            target_dim = int(ds.get("target_dim", 0))
            data_full = data_full[:, target_dim]

        N_data = len(data_full)
        ds_test_len = int(ds["test_len"])
        t0_tail = N_data - ds_test_len

        # Ground truth for evaluation segment
        y_true_test = data_full[t0_tail + T0: t0_tail + T0 + test_len]
    else:
        # For RF/OLS, it's simpler
        y_true_test = y_full[T0:T0 + test_len]

    # Run AgACI aggregation
    eta = float(getattr(args, 'agaci_eta', 2))
    use_gradient = bool(getattr(args, 'agaci_gradient', True))
    lr_schedule = str(getattr(args, 'agaci_lr_schedule', 'constant'))  # Default to 'constant' for regime-switching

    print(f"\n[ACP_UTILS] Preparing to call AgACI core function")
    print(f"[ACP_UTILS] Expert intervals shapes:")
    print(f"[ACP_UTILS]   - y_lowers_experts: {y_lowers_experts.shape}")
    print(f"[ACP_UTILS]   - y_uppers_experts: {y_uppers_experts.shape}")
    print(f"[ACP_UTILS]   - y_true_test: {y_true_test.shape}")
    print(f"[ACP_UTILS] Parameters:")
    print(f"[ACP_UTILS]   - alpha: {args.alpha}")
    print(f"[ACP_UTILS]   - eta (learning rate): {eta}")
    print(f"[ACP_UTILS]   - use_gradient: {use_gradient}")
    print(f"[ACP_UTILS]   - lr_schedule: {lr_schedule}")
    print(f"[ACP_UTILS] Expert lower bound ranges:")
    for i, gamma in enumerate(gammas):
        print(f"[ACP_UTILS]   - Expert {i} (gamma={gamma[0]:.4f}): "
              f"[{np.min(y_lowers_experts[i]):.2f}, {np.max(y_lowers_experts[i]):.2f}]")

    agaci_results = run_agaci(
        y_lowers_experts,
        y_uppers_experts,
        y_true_test,
        alpha=args.alpha,
        eta=eta,
        use_gradient=use_gradient,
        lr_schedule=lr_schedule,
        verbose=True  # Enable verbose mode for detailed BOA trace
    )

    print(f"\n[ACP_UTILS] AgACI core function returned")
    print(f"[ACP_UTILS] Results keys: {list(agaci_results.keys())}")

    # Add expert information to results
    agaci_results['y_lowers_experts'] = y_lowers_experts
    agaci_results['y_uppers_experts'] = y_uppers_experts
    agaci_results['gammas'] = gammas
    agaci_results['tab_alpha_t'] = tab_alpha_t

    return agaci_results
