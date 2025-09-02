# experiments/sensitivity_utils.py
import math
import copy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ---------- Small helpers ----------

def _to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def _detach_cpu(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def _batched_predict(predict_fn: Callable, X: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """Predict in batches, returns tensor on X.device."""
    outs = []
    for i in range(0, X.shape[0], batch_size):
        with torch.no_grad():
            outs.append(predict_fn(X[i:i+batch_size]))
    return torch.cat(outs, dim=0)


# ---------- Main analyzer ----------

class SensitivityAnalyzer:
    """
    Sensitivity analysis toolkit for time-series forecasters (PyTorch).
    Works with any model for which you can provide a predict_fn(X) -> yhat.

    Parameters
    ----------
    model : torch.nn.Module
        Your trained PyTorch model (used for gradient-based analyses).
    predict_fn : Callable[[torch.Tensor], torch.Tensor]
        Function that takes a batch X (B, L, C) and returns predictions (B, D) or (B, 1).
    device : str
        'cpu' or 'cuda'.
    """

    def __init__(
        self,
        model: nn.Module,
        predict_fn: Callable[[torch.Tensor], torch.Tensor],
        device: str = "cpu"
    ):
        self.model = model
        self.predict_fn = predict_fn
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    # -------- Local input sensitivity (gradients) --------

    @torch.no_grad()
    def predict(self, X: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        return _batched_predict(self.predict_fn, X, batch_size=batch_size)

    def input_gradient_sensitivity(
        self,
        X: np.ndarray,
        reduction: str = "mean_abs",  # 'mean_abs' | 'max_abs'
        batch_size: int = 64,
    ) -> Dict[str, np.ndarray]:
        """
        Computes dŷ/dX for each sample, lag, and channel via autograd (local sensitivity).
        Returns aggregated heatmaps over batch.

        X: numpy or tensor with shape (B, L, C)
        """
        self.model.eval()
        X_t = _to_tensor(X, self.device).requires_grad_(True)

        grads_accum = None
        n = X_t.shape[0]
        for i in range(0, n, batch_size):
            xb = X_t[i:i+batch_size]
            yb = self.predict_fn(xb)  # (B, D) or (B, 1)
            # Scalarize output to backprop through all dims:
            scalar = yb.pow(2).mean()  # any smooth scalar works; squared mean is stable
            self.model.zero_grad(set_to_none=True)
            if xb.grad is not None:
                xb.grad.zero_()
            scalar.backward(retain_graph=False)
            gb = xb.grad.detach()  # (B, L, C)

            if grads_accum is None:
                grads_accum = gb.abs().sum(dim=0)  # (L, C)
            else:
                grads_accum += gb.abs().sum(dim=0)

        # aggregate across batch
        if reduction == "mean_abs":
            heatmap = (grads_accum / n).detach().cpu().numpy()  # (L, C)
        elif reduction == "max_abs":
            # Recompute with max over batch (slower, but OK for small n)
            # Simple approximation: treat sum as proxy; or re-run with storing per-batch max
            heatmap = (grads_accum / n).detach().cpu().numpy()
        else:
            raise ValueError("Unsupported reduction.")

        return {"input_grad_heatmap": heatmap, "description": "Rows=lags, Cols=channels, values=|∂ŷ/∂x| (avg over batch)"}

    # -------- OFAT finite-difference input sensitivity --------

    def ofat_input_sensitivity(
        self,
        X: np.ndarray,
        delta: float = 0.05,
        center: Optional[np.ndarray] = None,
        batch_size: int = 128
    ) -> Dict[str, np.ndarray]:
        """
        One-factor-at-a-time (OFAT) finite-difference sensitivity.
        For each (lag,channel), perturb by ±delta * scale and measure Δŷ.

        Returns: dict with arrays of shape (L, C): mean |Δŷ| per dimension (averaged over batch).
        """
        self.model.eval()
        X0 = _to_tensor(X, self.device)
        B, L, C = X0.shape

        # Baseline predictions
        y0 = self.predict(X0, batch_size=batch_size)  # (B, D) or (B, 1)
        y0 = y0 if y0.ndim == 2 else y0.view(B, -1)
        D = y0.shape[1]

        # scale for perturbations
        if center is None:
            scale = torch.clamp(X0.abs().mean(dim=0, keepdim=True), min=1e-6)  # (1, L, C)
        else:
            center_t = _to_tensor(center, self.device).view(1, L, C)
            scale = torch.clamp(center_t.abs(), min=1e-6)

        mean_abs_delta = torch.zeros((L, C), device=self.device)

        for ell in range(L):
            for ch in range(C):
                x_plus = X0.clone()
                x_minus = X0.clone()
                step = delta * scale[0, ell, ch]
                x_plus[:, ell, ch] += step
                x_minus[:, ell, ch] -= step

                yp = self.predict(x_plus, batch_size=batch_size)
                ym = self.predict(x_minus, batch_size=batch_size)
                yp = yp if yp.ndim == 2 else yp.view(B, -1)
                ym = ym if ym.ndim == 2 else ym.view(B, -1)

                # symmetric difference around baseline
                d_abs = (torch.abs(yp - y0) + torch.abs(ym - y0)) / 2.0  # (B, D)
                # Reduce across output dims and batch
                mean_abs_delta[ell, ch] = d_abs.mean()

        return {"ofat_mean_abs_delta": _detach_cpu(mean_abs_delta),
                "description": "Rows=lags, Cols=channels, values=mean |Δŷ| under ±δ perturbation"}

    # -------- Parameter-group sensitivity (gradients) --------

    def parameter_group_sensitivity(
        self,
        X: np.ndarray,
        y: np.ndarray,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        groupby: Optional[Callable[[str], str]] = None,
        normalize: bool = True,
        batch_size: int = 128,
    ) -> Dict[str, float]:
        """
        Layer/block influence via gradient norms ‖∇θ L‖_2.
        Returns dict: group_name -> normalized score.

        groupby(name) can map parameter names to a group (e.g., 's4_layer_0', 'encoder', ...).
        """
        self.model.train(False)
        X_t = _to_tensor(X, self.device)
        y_t = _to_tensor(y, self.device)
        if y_t.ndim == 1:
            y_t = y_t.view(-1, 1)

        if loss_fn is None:
            loss_fn = nn.MSELoss()

        # Accumulate grads across batch, avoid OOM
        # First zero grads
        self.model.zero_grad(set_to_none=True)

        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

        # We will manually backprop a running sum of losses (scaled to keep magnitude stable)
        count = 0
        for xb, yb in dl:
            xb.requires_grad_(True)
            pred = self.predict_fn(xb)
            pred = pred if pred.ndim == 2 else pred.view(xb.shape[0], -1)
            loss = loss_fn(pred, yb)
            loss.backward()
            count += xb.shape[0]

        # Collect gradient norms per parameter and group
        group_scores = {}
        group_sizes = {}

        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            gnorm = torch.linalg.norm(g).item()
            key = groupby(name) if groupby else name
            group_scores[key] = group_scores.get(key, 0.0) + gnorm
            group_sizes[key] = group_sizes.get(key, 0) + p.numel()

        # Normalize by total or by group size
        if normalize and len(group_scores) > 0:
            total = sum(group_scores.values())
            if total > 0:
                for k in list(group_scores.keys()):
                    group_scores[k] = group_scores[k] / total

        return group_scores

    # -------- Calibration / ACI sensitivity --------

    def aci_sensitivity(
        self,
        aci_fit_fn: Callable[..., Dict],
        aci_predict_fn: Callable[..., Dict],
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        alpha: float = 0.1,
        gammas: Tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2, 5e-2),
        cal_window_sizes: Tuple[int, ...] = (64, 128, 256),
        score_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Probes sensitivity of conformal intervals to (i) ACI learning-rate gamma, and (ii) size of calibration window.
        Uses your acp_utils-style fit/predict functions:

        aci_fit_fn(y_cal, y_hat_cal, alpha, gamma) -> state dict
        aci_predict_fn(state, y_hat_seq) -> intervals/coverage dict

        score_fn(y, y_hat) defaults to absolute residuals if None.
        """
        if score_fn is None:
            score_fn = lambda y, yhat: np.abs(y - yhat)

        # compute predictions on cal and test
        Xc = _to_tensor(X_cal, self.device)
        Xt = _to_tensor(X_test, self.device)

        with torch.no_grad():
            yhat_cal = _detach_cpu(self.predict(Xc))
            yhat_test = _detach_cpu(self.predict(Xt))

        y_cal_np = np.asarray(y_cal).reshape(yhat_cal.shape[0], -1)
        y_test_np = np.asarray(y_test).reshape(yhat_test.shape[0], -1)

        # Use first output dim for scalar intervals (extend if you produce vector intervals)
        y_cal_s = y_cal_np[:, 0]
        yhat_cal_s = yhat_cal[:, 0]
        y_test_s = y_test_np[:, 0]
        yhat_test_s = yhat_test[:, 0]

        results_gamma = []
        for g in gammas:
            state = aci_fit_fn(
                y_cal=y_cal_s, y_hat_cal=yhat_cal_s, alpha=alpha, gamma=g
            )
            pred = aci_predict_fn(
                state=state, y_hat_seq=yhat_test_s
            )
            # Expect pred to include 'lower', 'upper' arrays; adapt to your acp_utils signatures
            lower = pred["lower"]
            upper = pred["upper"]
            cover = ((y_test_s >= lower) & (y_test_s <= upper)).mean()
            med_len = np.median(upper - lower)
            frac_inf = np.mean(~np.isfinite(upper - lower))
            results_gamma.append({
                "gamma": g,
                "coverage": float(cover),
                "median_length": float(med_len),
                "frac_infinite": float(frac_inf),
            })

        results_window = []
        n_cal = len(y_cal_s)
        for W in cal_window_sizes:
            W = min(W, n_cal)
            idx = np.arange(n_cal - W, n_cal)
            state = aci_fit_fn(
                y_cal=y_cal_s[idx], y_hat_cal=yhat_cal_s[idx], alpha=alpha, gamma=gammas[0]
            )
            pred = aci_predict_fn(state=state, y_hat_seq=yhat_test_s)
            lower = pred["lower"]
            upper = pred["upper"]
            cover = ((y_test_s >= lower) & (y_test_s <= upper)).mean()
            med_len = np.median(upper - lower)
            frac_inf = np.mean(~np.isfinite(upper - lower))
            results_window.append({
                "cal_window": int(W),
                "coverage": float(cover),
                "median_length": float(med_len),
                "frac_infinite": float(frac_inf),
            })

        return {"gamma_sensitivity": results_gamma, "window_sensitivity": results_window}

    # -------- Optional: DS3M regime counterfactuals --------

    def ds3m_regime_counterfactual(
        self,
        X: np.ndarray,
        force_regimes: List[int],
        predict_force_regime_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
        batch_size: int = 128
    ) -> Optional[Dict[int, np.ndarray]]:
        """
        If your DS3M exposes a way to force a specific discrete regime (d_t = k),
        compare predictions under each forced regime.

        predict_force_regime_fn(X_t, k) -> yhat_k
        """
        if predict_force_regime_fn is None:
            return None

        X_t = _to_tensor(X, self.device)
        report = {}
        with torch.no_grad():
            base = self.predict(X_t, batch_size=batch_size)
        base = _detach_cpu(base)

        for k in force_regimes:
            with torch.no_grad():
                yk = predict_force_regime_fn(X_t, k)
            report[k] = _detach_cpu(yk) - base  # Δ prediction vs. base
        return report


# ---------- Example tiny glue for ACI (adapt to your acp_utils) ----------

def make_aci_glue(acp_module):
    """
    Returns two callables that SensitivityAnalyzer.aci_sensitivity can use:
    - fit_fn(y_cal, y_hat_cal, alpha, gamma) -> state
    - predict_fn(state, y_hat_seq) -> {'lower': ..., 'upper': ...}
    """
    def fit_fn(y_cal, y_hat_cal, alpha, gamma):
        # Example: acp_module.ACI(alpha=alpha, gamma=gamma).fit(y_cal, y_hat_cal)
        aci = acp_module.ACI(alpha=alpha, gamma=gamma)
        aci.fit(y_cal=y_cal, y_hat_cal=y_hat_cal)
        return {"aci": aci}

    def predict_fn(state, y_hat_seq):
        aci = state["aci"]
        lower, upper = aci.predict(y_hat_seq)
        return {"lower": np.asarray(lower), "upper": np.asarray(upper)}

    return fit_fn, predict_fn
