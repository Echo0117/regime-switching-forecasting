# experiments/interval_evaluate.py
# -*- coding: utf-8 -*-
import numpy as np

def trim_to_common_length(y_true, lower, upper):
    """
    Trim (y_true, lower, upper) to the same length = min(len(y), len(lower), len(upper)).
    """
    n = min(len(y_true), len(lower), len(upper))
    if len(y_true) != n or len(lower) != n or len(upper) != n:
        print(f"[WARN] Trimming lengths: y={len(y_true)}, lower={len(lower)}, upper={len(upper)} → {n}")
    return y_true[:n], lower[:n], upper[:n]

def coverage_rate(y_true, lower, upper):
    return np.mean((y_true >= lower) & (y_true <= upper))

def avg_length(lower, upper):
    return float(np.mean(upper - lower))

def median_length(lower, upper):
    return float(np.median(upper - lower))

def interval_score(y_true, lower, upper, alpha=0.1):
    """
    Interval Score（Gneiting & Raftery）
    Sα(l,u;y) = (u - l) + (2/α)*(l - y) * 1(y < l) + (2/α)*(y - u) * 1(y > u)
    """
    L = upper - lower
    under = (y_true < lower)
    over  = (y_true > upper)
    pen = np.zeros_like(y_true, dtype=float)
    pen[under] = (2.0/alpha) * (lower[under] - y_true[under])
    pen[over]  = (2.0/alpha) * (y_true[over] - upper[over])
    return float(np.mean(L + pen))

def winkler_score(y_true, lower, upper, alpha=0.1):
    """
    Winkler Score 对称区间的版本。对于不对称区间，等价于 Interval Score
    """
    return interval_score(y_true, lower, upper, alpha=alpha)

def miscoverage_rate(y_true, lower, upper):
    return 1.0 - coverage_rate(y_true, lower, upper)

def evaluate_intervals(y_true, lower, upper, alpha=0.1, label=""):
    """
    对预测区间进行评价：Coverage, mis-coverage、均/中位宽度、Interval/Winkler Score 等
    自动对齐长度（裁剪到最短）。
    """
    y, lo, up = trim_to_common_length(np.asarray(y_true).ravel(),
                                      np.asarray(lower).ravel(),
                                      np.asarray(upper).ravel())
    cov   = coverage_rate(y, lo, up)
    msc   = miscoverage_rate(y, lo, up)
    avgl  = avg_length(lo, up)
    medl  = median_length(lo, up)
    iscore= interval_score(y, lo, up, alpha=alpha)
    wsc   = winkler_score(y, lo, up, alpha=alpha)

    metrics = dict(
        label=label,
        coverage=cov,
        miscoverage=msc,
        avg_length=avgl,
        median_length=medl,
        interval_score=iscore,
        winkler_score=wsc
    )
    print(f"[EVAL-{label}] coverage={cov:.3f}, avg_len={avgl:.3f}, med_len={medl:.3f}, IS={iscore:.3f}, Winkler={wsc:.3f}")
    return metrics


# import numpy as np

# def interval_coverage(y_true, lower, upper):
#     """Return empirical coverage: fraction of points inside [lower, upper]."""
#     return np.mean((y_true >= lower) & (y_true <= upper))

# def interval_width(lower, upper, agg='mean'):
#     """Return average width (mean or median)."""
#     w = (upper - lower)
#     if agg == 'mean':
#         return np.mean(w)
#     elif agg == 'median':
#         return np.median(w)
#     return np.mean(w)

# def interval_score(y_true, lower, upper, alpha=0.1):
#     """
#     Gneiting & Raftery interval score.
#     Penalizes width and miscoverage. Lower is better.
#     """
#     width = upper - lower
#     below = (y_true < lower).astype(float)
#     above = (y_true > upper).astype(float)
#     score = width + (2.0/alpha) * ((lower - y_true) * below + (y_true - upper) * above)
#     return np.mean(score)

# def winkler_score(y_true, lower, upper, alpha=0.1):
#     """
#     Winkler score for symmetric intervals.
#     """
#     width = upper - lower
#     under = y_true < lower
#     over  = y_true > upper
#     score = np.copy(width)
#     score[under] += (2.0/alpha)*(lower - y_true[under])
#     score[over]  += (2.0/alpha)*(y_true[over] - upper[over])
#     return np.mean(score)

# def side_miscoverage(y_true, lower, upper):
#     """
#     Check left/right miscoverage separately.
#     Returns (p_under, p_over)
#     """
#     p_under = np.mean(y_true < lower)
#     p_over  = np.mean(y_true > upper)
#     return p_under, p_over

# def evaluate_intervals(y_true, lower, upper, alpha=0.1, label=''):
#     """
#     Convenience: compute coverage, width, interval score, Winkler score, and miscoverage sides.
#     """
#     cov   = interval_coverage(y_true, lower, upper)
#     wid   = interval_width(lower, upper, agg='mean')
#     iscor = interval_score(y_true, lower, upper, alpha=alpha)
#     # wsc   = winkler_score(y_true, lower, upper, alpha=alpha)
#     p_under, p_over = side_miscoverage(y_true, lower, upper)
#     return {
#         "label": label,
#         "coverage": cov,
#         "target": 1 - alpha,
#         "width_mean": wid,
#         "interval_score": iscor,
#         # "winkler_score": wsc,
#         "under_mis": p_under,
#         "over_mis": p_over,
#     }
