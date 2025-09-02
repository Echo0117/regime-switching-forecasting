# experiments/acp_utils.py
# -*- coding: utf-8 -*-
"""
Adaptive Conformal Intervals (ACI) and Aggregated ACI (AgACI) for 1-D residual sequences.

This module provides two main functions:

1) aci_intervals(residuals, alpha, gamma, train_size)
   -------------------------------------------------
   A single-γ ACI mechanism that updates the "effective" coverage
   level α_t online as in Gibbs & Candès (2021) and returns
   a stepwise interval [0, q_t] on the absolute residuals |y - ŷ|.
   For direct use in time-series forecasting where the base
   model supplies point predictions ŷ_t, the ACI intervals on residuals
   are typically transformed to prediction intervals:
        [ ŷ_t - q_t , ŷ_t + q_t ].

2) agaci_ewa(residuals, alpha, train_size, gammas, eta, ...)
   ---------------------------------------------------------
   A multi-γ version (AgACI) using online expert aggregation:
   - For each γ in the user-provided grid (gammas), we run ACI to obtain
     a per-step sequence of quantiles q_{γ, t}.
   - We aggregate up to test steps using Exponentially Weighted Average (EWA),
     with a pinball loss on residuals, to get a single aggregated quantile
     q̃_t for each step. The interval is again [0, q̃_t].

   We intentionally keep the aggregator simple, stable, and dependency-free.
   This module does not depend on the original "AdaptiveConformalPredictions-
   TimeSeries" repository; it is suitable for direct integration into your pipeline.

Typical usage in a forecasting pipeline (single dimension):
-----------------------------------------------------------
    # residual series for one dimension:
    resid = np.abs(y_true[:, 0] - y_pred[:, 0])

    # single-gamma ACI:
    lo_r, up_r = aci_intervals(resid, alpha=0.1, gamma=0.01, train_size=200)

    # build final prediction intervals:
    pred_lower = y_pred[train_size:, 0] - up_r
    pred_upper = y_pred[train_size:, 0] + up_r

    # multi-gamma EWA aggregator (AgACI):
    lo_r2, up_r2 = agaci_ewa(resid, alpha=0.1, train_size=200,
                             gammas=[0.005, 0.01, 0.02, 0.05], eta=0.1)

    # final intervals again:
    pred_lower2 = y_pred[train_size:, 0] - up_r2
    pred_upper2 = y_pred[train_size:, 0] + up_r2
"""

import numpy as np
from typing import Tuple, Sequence


# --------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------

def _clip(value: float, a: float, b: float) -> float:
    """Clamp a scalar into the range [a, b]."""
    return min(max(value, a), b)


def _quantile_higher(samples: np.ndarray, q: float) -> float:
    """
    A safe wrapper around np.quantile(..., method='higher') ensuring q∈[0,1].
    The 'higher' method helps guarantee coverage in finite samples.
    """
    q = _clip(q, 0.0, 1.0)
    return float(np.quantile(samples, q, method="higher"))


# --------------------------------------------------------------------------
# 1) Single-γ ACI
# --------------------------------------------------------------------------

def aci_intervals(
    residuals: np.ndarray,
    alpha: float = 0.1,
    gamma: float = 0.01,
    train_size: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a single-gamma Adaptive Conformal Interval on a 1-D residual series.

    Parameters
    ----------
    residuals : np.ndarray, shape (T,)
        Absolute residuals r_t = |y_t - ŷ_t|.
    alpha : float
        Target miscoverage (e.g., 0.1 for 90% coverage).
    gamma : float
        ACI learning rate for the effective α_t update.
    train_size : int
        Number of initial points used as a calibration window (T0).
        Must be strictly less than T.

    Returns
    -------
    lower : np.ndarray, shape (T - train_size,)
        Lower bound for residual intervals (≈ zeros).
    upper : np.ndarray, shape (T - train_size,)
        Upper bound for residual intervals q_t computed at each step.
        Final predictive intervals for the *signal* y are typically:
            [ ŷ_t - q_t, ŷ_t + q_t ].
    """
    r = np.asarray(residuals, dtype=float).ravel()
    T = r.size
    assert train_size < T, "aci_intervals: train_size must be < T (len(residuals))."

    # Initialization
    cal = list(r[:train_size])                 # initial calibration window
    test_len = T - train_size
    lower = np.zeros(test_len)                 # Because residuals ≥ 0
    upper = np.zeros(test_len)
    alpha_t = alpha

    # Online loop
    for i, t in enumerate(range(train_size, T)):
        # 1) Compute the per-step quantile q_t
        #    The "effective" coverage is (1 - alpha_t) clipped to [0,1]
        alpha_t = _clip(alpha_t, 0.0, 1.0)
        q_level = _clip(1.0 - alpha_t, 0.0, 1.0)
        q_t = _quantile_higher(np.array(cal, dtype=float), q_level)

        lower[i] = 0.0
        upper[i] = q_t

        # 2) Feedback update
        # err is 1 if residual is out of the interval, 0 otherwise
        err = 0.0 if r[t] <= q_t else 1.0
        alpha_t = alpha_t + gamma * (alpha - err)

        # 3) Rolling update of the calibration set
        cal.append(r[t])
        if len(cal) > train_size:
            cal.pop(0)

    return lower, upper


# --------------------------------------------------------------------------
# 2) AgACI via EWA (Exponentially Weighted Average)
# --------------------------------------------------------------------------

def _pinball_loss(y: float, q: np.ndarray, tau: float) -> np.ndarray:
    """
    Compute pinball loss for a given scalar y and a vector of quantiles q.
    Loss_j = max( tau*(y - q_j), (tau-1)*(y - q_j) ).
    A standard choice for upper quantile is tau = 1 - alpha.
    """
    u = y - q
    return np.maximum(tau * u, (tau - 1.0) * u)


def agaci_ewa(
    residuals: np.ndarray,
    alpha: float = 0.1,
    train_size: int = 200,
    gammas: Sequence[float] = (0.005, 0.01, 0.02, 0.05),
    eta: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregated Adaptive Conformal Intervals using EWA over multiple γ.

    Steps:
    - For each γ in 'gammas', run ACI to compute per-step upper bounds
      (quantiles) q_{γ, t} for t = train_size,...,T-1.
    - Aggregate these K sequences using Exponentially Weighted Average (EWA),
      with a pinball loss on the residuals. The aggregated quantile at step t
      is q̃_t = Σ_k w_k(t) q_{γ_k,t}, where w_k(t) are adaptive weights.

    Parameters
    ----------
    residuals : np.ndarray, shape (T,)
        Absolute residual series r_t = |y_t - ŷ_t|.
    alpha : float
        Target miscoverage (e.g., 0.1).
    train_size : int
        Calibration window size (T0). Must be < T.
    gammas : Sequence[float]
        Set of gamma learning rates used for the experts (ACI channels).
    eta : float
        EWA learning rate (for the weights).

    Returns
    -------
    lower : np.ndarray, shape (T - train_size,)
        Lower bounds (≈ zeros).
    upper : np.ndarray, shape (T - train_size,)
        Aggregated upper bounds q̃_t for each test step.
    """
    r = np.asarray(residuals, dtype=float).ravel()
    T = r.size
    assert train_size < T, "agaci_ewa: train_size must be < T (len(residuals))."
    test_len = T - train_size
    K = len(gammas)
    assert K >= 1, "agaci_ewa: at least one gamma is required."

    # 1) Per-γ ACI sequences: q_{γ, t}
    expert_quants = np.zeros((K, test_len))  # shape (K, test_len)
    for k, g in enumerate(gammas):
        _, qk = aci_intervals(r, alpha=alpha, gamma=g, train_size=train_size)
        expert_quants[k, :] = qk

    # 2) Online EWA over test_len steps, with pinball loss
    # weights: w_k(0) = 1/K
    log_w = np.zeros(K, dtype=float)
    tau = 1.0 - alpha  # recommended for upper quantiles
    agg_upper = np.zeros(test_len, dtype=float)

    for i in range(test_len):
        # Weighted prediction:
        w = np.exp(log_w - np.max(log_w))  # softmax trick
        w /= np.sum(w)
        q_i = expert_quants[:, i]
        agg_upper[i] = np.dot(w, q_i)

        # Observe residual r[t] and update the weights
        t = train_size + i
        losses = _pinball_loss(y=r[t], q=q_i, tau=tau)
        log_w += -eta * losses  # EWA update in log-space

    # Lower is zero for absolute residual intervals
    lower = np.zeros(test_len, dtype=float)
    upper = np.clip(agg_upper, 0.0, None)
    return lower, upper
