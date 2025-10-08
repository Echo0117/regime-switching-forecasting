# experiments/utils/aci_original_adapter.py

import numpy as np
from typing import Tuple, Dict

# Import the ORIGINAL ACI implementation you pasted:
# Make sure this import path points to the module where fit_predict lives.
from AdaptiveConformalPredictionsTimeSeries.models import fit_predict, fit_predict_ACPs  # adjust path if needed

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
