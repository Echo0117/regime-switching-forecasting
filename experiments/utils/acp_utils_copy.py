"""
Drop-in ACI/AgACI that:
  1) Uses ORIGINAL models.py (RF/OLS) when you call set_backend_data(...), OR
  2) Uses YOUR custom regressors (S4Regressor, RupturesSegmentedLinear,
     MCDropoutGRU, GPTorchSparse, DS3MWrapper) when you call set_backend_model(...).

Old API preserved
-----------------
aci_intervals(residuals, alpha=0.1, gamma=0.01, train_size=200)
agaci_ewa(residuals, alpha=0.1, train_size=200, gammas=(...), eta=0.1)

Both return (lower≈zeros, upper=q_t) in residual space, exactly like your code expects.
"""
from __future__ import annotations
from typing import Dict, Iterable, Tuple, Optional, Callable
import numpy as np

from AdaptiveConformalPredictionsTimeSeries.models import fit_predict, fit_predict_ACPs  # type: ignore
from experiments.ds3m_wrapper import DS3MWrapper
from experiments.run_all_experiments import build_model
    



# =============================================================================
# Backend registry
# =============================================================================
# _BACKEND: dict = {
#     "X": None,                  # (d, n), features x time
#     "Y": None,                  # (n,)
#     "basemodel": None,          # "RF"/"OLS" for ORIGINAL; "CUSTOM" for your four
#     "params_basemodel": None,   # dict for ORIGINAL; unused for CUSTOM
#     "online": True,

#     # CUSTOM model ctor -> returns an object with .fit(X, y) and .predict(X)
#     "custom_ctor": None,        # Callable[[], reg]
#     "custom_kwargs": None,      # dict for the ctor
# }


# def set_backend_data(
#     X: np.ndarray,
#     Y: np.ndarray,
#     *,
#     basemodel: str = "RF",
#     params_basemodel: Optional[Dict] = None,
#     online: bool = True,
# ) -> None:
#     """
#     ORIGINAL models.py path (RF/OLS).
#     Call this if you want to run via fit_predict / fit_predict_ACPs.
#     """
#     if params_basemodel is None and basemodel == "RF":
#         params_basemodel = {
#             "cores": -1,
#             "n_estimators": 200,
#             "min_samples_leaf": 1,
#             "max_features": 1.0,
#         }
#     y = np.asarray(Y)
#     if y.ndim == 2 and y.shape[1] == 1:
#         y = y.ravel()
#     _BACKEND.update({
#         "X": np.asarray(X),
#         "Y": y,
#         "basemodel": basemodel,
#         "params_basemodel": params_basemodel,
#         "online": bool(online),
#         "custom_ctor": None,
#         "custom_kwargs": None,
#     })

def _half_len_from_pred_interval(y_lower: np.ndarray, y_upper: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert symmetric PIs [ŷ - w, ŷ + w] to residual bounds (0, w)."""
    y_lower = np.asarray(y_lower); y_upper = np.asarray(y_upper)
    w = 0.5 * (y_upper - y_lower)
    T = w.shape[-1]
    return np.zeros(T, dtype=float), np.asarray(w, dtype=float)


# =============================================================================
# ORIGINAL path → delegate into models.py (RF/OLS)
# =============================================================================
def _aci_original(alpha: float, gamma: float, train_size: int) -> Tuple[np.ndarray, np.ndarray]:
    methods = ["ACP"]
    params_methods = {"gamma": float(gamma), "online": True if _BACKEND.get("online", True) else False}
    y_l, y_u, _, _ = fit_predict(
       X,
        _BACKEND["Y"],
        alpha,
        methods,
        params_methods,
        _BACKEND["basemodel"],
        _BACKEND["params_basemodel"],
        train_size,
    )
    return _half_len_from_pred_interval(y_l[0], y_u[0])


def _agaci_original(alpha: float, train_size: int, gammas: Iterable[float], eta: float) -> Tuple[np.ndarray, np.ndarray]:
    y_l, y_u, _alpha_t, _gs = fit_predict_ACPs(
        _BACKEND["X"],
        _BACKEND["Y"],
        alpha,
        list(gammas),
        _BACKEND["basemodel"],
        _BACKEND["params_basemodel"],
        train_size,
    )
    # EWA aggregation in residual space
    K, T = y_u.shape
    q = 0.5 * (y_u - y_l)
    tau = 1.0 - alpha
    log_w = np.zeros(K)
    agg_q = np.zeros(T)
    # Need absolute residuals for loss—approx from Y and ŷ: but ORIGINAL path
    # doesn't expose ŷ; use the same EWA as your prior code with per-step pinball on q only.
    # (If you want, you can pass residuals to weight by true r_t; left neutral here.)
    for i in range(T):
        w = np.exp(log_w - np.max(log_w)); w /= np.sum(w)
        q_i = q[:, i]
        agg_q[i] = float(np.dot(w, q_i))
        # neutral losses -> keep weights (or plug your residuals here)
    return np.zeros(T), np.clip(agg_q, 0.0, None)


# =============================================================================
# CUSTOM path → mirror ACP from models.py but plug YOUR regressor
# =============================================================================
# def _train_cal_split_idx(train_size: int):
#     idx = np.arange(train_size)
#     n_half = int(np.floor(train_size / 2))
#     return idx[:n_half], idx[n_half:2 * n_half]

# def aci_intervals(
#     Xd: np.ndarray,   # (d, n)
#     y: np.ndarray,    # (n,)
#     alpha: float,
#     gamma: float,
#     train_size: int,
#     args=None,
#     reg=None,  # instance of your regressor with .fit/.predict
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """ACP loop with a custom regressor. Supports fit-once for heavy models."""
#     d, n = Xd.shape
#     test_size = n - train_size
#     if test_size <= 0:
#         raise ValueError(f"train_size={train_size} must be < n={n}")

#     y_lower = np.empty(test_size, dtype=float)
#     y_upper = np.empty(test_size, dtype=float)

#     # classic half/half split inside the rolling window
#     def _split_idx(T0: int):
#         n_half = int(np.floor(T0 / 2))
#         return np.arange(n_half), np.arange(n_half, 2 * n_half)

#     alpha_t = float(alpha)
#     # refit_each = bool(reg_kwargs.pop("refit_each_step", False))

#     # Optionally fit once for heavy models
#     # if not refit_each:
#     #     reg = reg_ctor(**reg_kwargs)
    

#     for i in range(test_size):
#         X_win = Xd[:, i:(train_size + i)].T  # (train_size, d)
#         x_test = Xd[:, (train_size + i)].reshape(1, -1)
#         y_win = y[i:(train_size + i)]

#         idx_tr, idx_cal = _split_idx(train_size)

#         # Make (or reuse) the model
#         # if refit_each:
#         #     reg = reg_ctor(**reg_kwargs)

#         # Fit either every step (cheap models) or only once (heavy)
#         # if (not refit_each and i == 0) or refit_each:
#         reg.predict(X_win[idx_tr, :], y_win[idx_tr], args)

#         # Predict on calibration + test
#         y_pred_cal = np.asarray(reg.predict(X_win[idx_cal, :], args)).reshape(-1)
#         res_cal = np.abs(y_win[idx_cal] - y_pred_cal)
#         y_pred = float(np.asarray(reg.predict(x_test, args)).reshape(-1)[0])

#         # ACI update
#         if alpha_t >= 1.0:
#             lo_i, up_i, err = 0.0, 0.0, 1.0
#         elif alpha_t <= 0.0:
#             lo_i, up_i, err = -np.inf, np.inf, 0.0
#         else:
#             q = float(np.quantile(res_cal, 1.0 - alpha_t, method="higher"))
#             lo_i, up_i = y_pred - q, y_pred + q
#             y_true = y[train_size + i]
#             err = 1.0 - float((lo_i <= y_true) and (y_true <= up_i))

#         alpha_t = alpha_t + gamma * (alpha - err)
#         y_lower[i] = lo_i
#         y_upper[i] = up_i

#     # residual-space (0, q_t)
#     return np.zeros(test_size), 0.5 * (y_upper - y_lower)
from typing import Tuple
import numpy as np

def aci_intervals(
    Xd: np.ndarray,   # (d, n)  注意: 列是时间轴
    y: np.ndarray,    # (n,)
    alpha: float,
    gamma: float,
    train_size: int,
    args=None,
    reg=None,         # 你的回归器实例, 具有 .predict(...) 接口（内部可完成训练/暖启动）
    return_center: bool = False,  # 新增: True 时同时返回中心预测
):
    """
    自适应(滑动)ACP:
      - 窗口长度 = train_size = T0
      - 每步将窗口一分为二: 前半训练, 后半校准
      - 在线更新 alpha_t

    输出 (默认与旧版兼容):
      - return_center=False: 返回 (lo_r, up_r)
      - return_center=True : 返回 (y_hat_test, lo_r, up_r)
        其中 y_hat_test, lo_r, up_r 的长度均为 n - train_size (对应 t >= T0)
        lo_r 恒为 0（保持你“残差区间”的老接口风格）
    """
    d, n = Xd.shape
    test_size = n - train_size
    if test_size <= 0:
        raise ValueError(f"train_size={train_size} must be < n={n}")

    y_lower = np.empty(test_size, dtype=float)
    y_upper = np.empty(test_size, dtype=float)
    y_hat_test = np.empty(test_size, dtype=float)  # 中心预测(从 t=T0 开始)

    # 训练/校准的索引切分: 前半训练, 后半校准
    def _split_idx(T0: int):
        n_half = int(np.floor(T0 / 2))
        return np.arange(n_half), np.arange(n_half, 2 * n_half)

    alpha_t = float(alpha)

    # 滑动窗口
    for i in range(test_size):
        # 取本步窗口与测试点
        X_win = Xd[:, i:(train_size + i)].T        # (train_size, d)
        x_test = Xd[:, (train_size + i)].reshape(1, -1)
        y_win = y[i:(train_size + i)]

        idx_tr, idx_cal = _split_idx(train_size)

        # ====== 训练 / 预测 ======
        # 你的 reg.predict(...) 里通常会在传入带标签的 (X, y) 时完成一次拟合/暖启动，
        # 这里沿用你的接口风格:
        #   - 在训练半窗上“训练/置状态”
        _ = reg.predict(X_win[idx_tr, :], y_win[idx_tr], args)

        #   - 校准半窗预测 -> 残差分位数
        y_pred_cal = np.asarray(reg.predict(X_win[idx_cal, :], y_win[idx_tr], args)).reshape(-1)
        res_cal = np.abs(y_win[idx_cal] - y_pred_cal)

        #   - 测试点中心预测
        y_pred = float(np.asarray(reg.predict(x_test, args)).reshape(-1)[0])
        y_hat_test[i] = y_pred

        # ====== ACI 更新 ======
        if alpha_t >= 1.0:
            lo_i, up_i, err = 0.0, 0.0, 1.0
        elif alpha_t <= 0.0:
            lo_i, up_i, err = -np.inf, np.inf, 0.0
        else:
            # 校准残差的 (1 - alpha_t) 分位数 = 半宽 q_t
            q = float(np.quantile(res_cal, 1.0 - alpha_t, method="higher"))
            lo_i, up_i = y_pred - q, y_pred + q
            y_true = y[train_size + i]
            err = 1.0 - float((lo_i <= y_true) and (y_true <= up_i))

        alpha_t = alpha_t + gamma * (alpha - err)
        y_lower[i] = lo_i
        y_upper[i] = up_i

    # 残差区间格式: (0, q_t)
    lo_r = np.zeros(test_size, dtype=float)
    up_r = 0.5 * (y_upper - y_lower)

    if return_center:
        return y_hat_test, lo_r, up_r
    else:
        return lo_r, up_r


def agaci_ewa(
    Xd: np.ndarray,
    y: np.ndarray,
    alpha: float,
    train_size: int,
    gammas: Iterable[float],
    # eta: float,
    args=None,
    reg=None,  # instance of your regressor with .fit/.predict
) -> Tuple[np.ndarray, np.ndarray]:
    """Run ACP for each gamma; EWA aggregate; honors refit_each_step flag."""
    gammas = list(gammas)
    d, n = Xd.shape
    test_size = n - train_size
    if test_size <= 0:
        raise ValueError("train_size must be < n")

    # ctor = _BACKEND["custom_ctor"]
    # kw = dict(_BACKEND["custom_kwargs"] or {})
    Q = np.zeros((len(gammas), test_size), dtype=float)

    for k, g in enumerate(gammas):
        _, up = aci_intervals(Xd, y, alpha, float(g), train_size, args, reg)
        Q[k, :] = up

    # simple EWA (no residual-driven losses here)
    log_w = np.zeros(len(gammas))
    agg_q = np.zeros(test_size)
    for i in range(test_size):
        w = np.exp(log_w - np.max(log_w)); w /= np.sum(w)
        agg_q[i] = float(np.dot(w, Q[:, i]))
    return np.zeros(test_size), np.clip(agg_q, 0.0, None)

# =============================================================================
# PUBLIC API (unchanged signatures)
# =============================================================================
# def aci_intervals(
#     # residuals: np.ndarray,   # kept for backward-compatibility; not used in ORIGINAL path
#     X,
#     y,
#     alpha: float = 0.1,
#     gamma: float = 0.01,
#     train_size: int = 200,
#     args=None,
#     reg=None  # instance of your regressor with .fit/.predict
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     If basemodel in {"RF","OLS"} -> ORIGINAL models.py
#     If basemodel == "CUSTOM"      -> run ACP loop with your regressor
#     """
#     # _require_data()
#     # ctor = _BACKEND["custom_ctor"]
   
#     return _run_acp_with_reg(X, y, alpha, gamma, train_size, args, reg)

# def agaci_ewa(
#     # residuals: np.ndarray,
#     X,
#     y,
#     alpha: float = 0.1,
#     train_size: int = 200,
#     gammas: Iterable[float] = (0.005, 0.01, 0.02, 0.05),
#     eta: float = 0.1,
#     args=None,
#     reg=None  # instance of your regressor with .fit/.predict
# ) -> Tuple[np.ndarray, np.ndarray]:
#     # _require_data()
#     return _run_multi_gamma_with_reg(X, y, alpha, train_size, gammas, eta, args, reg)
