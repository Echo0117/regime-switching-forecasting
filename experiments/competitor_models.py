import os, sys, numpy as np

class BaseReg:
    def fit(self, X, y): raise NotImplementedError
    def predict(self, X): raise NotImplementedError

class S4Regressor(BaseReg):
    """
    Uses the official S4D if available (pass --s4-path to repo root so models/s4/s4d.py is importable).
    Else falls back to a causal Conv1d tower.
    Adds early stopping and per-epoch metrics.
    """
    def __init__(
        self,
        lags=48,
        d_model=128,
        n_layers=4,
        dropout=0.1,
        epochs=50,
        batch=64,
        lr=1e-3,
        weight_decay=1e-2,
        device="cpu",
        amp=False,
        s4_path=None,
        val_frac=0.2,
        patience=10,
        min_delta=0.0,
        grad_clip=1.0,
        verbose=True,
        seed=42,
    ):
        import importlib, random
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.optim as optim

        # Repro
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        # extend sys.path if user passed the repo root
        import os, sys
        if s4_path:
            s4_path = os.path.abspath(s4_path)
            if s4_path not in sys.path:
                sys.path.insert(0, s4_path)

        self.torch, self.nn, self.optim = torch, nn, optim
        self.device = torch.device(device)
        self.amp = bool(amp)
        self.autocast = getattr(torch.cuda.amp, "autocast", None)
        self.scaler = getattr(torch.cuda.amp, "GradScaler", lambda **k: None)(enabled=self.amp)

        self.lags = int(lags); self.d_model = int(d_model)
        self.n_layers = int(n_layers); self.dropout = float(dropout)
        self.epochs = int(epochs); self.batch = int(batch); self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.val_frac = float(val_frac)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.grad_clip = float(grad_clip)
        self.verbose = bool(verbose)

        # try to import S4D
        S4D = None
        for mod in ("models.s4.s4d", "s4.s4d", "s4d"):
            try:
                m = importlib.import_module(mod)
                S4D = getattr(m, "S4D", None)
                if S4D is not None: break
            except Exception:
                pass
        self.using_fallback = S4D is None
        if self.verbose:
            print("[S4] using_fallback:", self.using_fallback)

        if self.using_fallback:
            class CausalConvModel(nn.Module):
                def __init__(self, lags, d_model, n_layers, p):
                    super().__init__()
                    layers = []
                    in_ch = 1
                    for i in range(n_layers):
                        conv = nn.Conv1d(in_ch, d_model, kernel_size=3, padding=2, dilation=2**i)
                        layers += [conv, nn.GELU(), nn.Dropout(p)]
                        in_ch = d_model
                    self.net = nn.Sequential(*layers)
                    self.head = nn.Linear(d_model, 1)
                def forward(self, x):
                    # x: (B, L, 1)
                    x = x.transpose(1, 2)      # -> (B, 1, L)
                    z = self.net(x)            # -> (B, d_model, L)
                    z = z.mean(dim=-1)         # pool over time
                    return self.head(z).squeeze(-1)
            self.model = CausalConvModel(self.lags, self.d_model, self.n_layers, self.dropout).to(self.device)
        else:
            Dropout = nn.Dropout
            def s4_lr(base): return min(1e-3, base)
            S4D_cls = S4D
            class S4Backbone(nn.Module):
                def __init__(self, L, d_model, n_layers, p):
                    super().__init__()
                    self.enc = nn.Linear(1, d_model)
                    self.blocks = nn.ModuleList([
                        S4D_cls(d_model, dropout=p, transposed=True, lr=s4_lr(1e-3))
                        for _ in range(n_layers)
                    ])
                    self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
                    self.drops = nn.ModuleList([Dropout(p) for _ in range(n_layers)])
                    # lightweight residual MLP head (helps nonlinearity)
                    self.head = nn.Sequential(
                        nn.Linear(d_model, d_model),
                        nn.GELU(),
                        nn.Linear(d_model, 1),
                    )
                def forward(self, x):
                    # x: (B, L, 1)
                    x = self.enc(x)                      # (B,L,d_model)
                    x = x.transpose(-1, -2)             # (B,d_model,L)
                    for s4, ln, dp in zip(self.blocks, self.norms, self.drops):
                        z = x
                        z, _ = s4(z)                     # (B,d_model,L)
                        z = dp(z)
                        x = ln((z + x).transpose(-1, -2)).transpose(-1, -2)
                    x = x.transpose(-1, -2)             # (B,L,d_model)
                    x = x[:, -1, :]                     # take last step (next-step forecast)
                    return self.head(x).squeeze(-1)
            self.model = S4Backbone(self.lags, self.d_model, self.n_layers, self.dropout).to(self.device)

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            try: torch.set_float32_matmul_precision("high")
            except Exception: pass

    def _reshape(self, X):
        import numpy as np
        X = np.asarray(X, dtype=np.float32)
        N, D = X.shape
        if D != self.lags:
            raise ValueError(f"Expected input dim == lags ({self.lags}), got {D}")
        return X.reshape(N, self.lags, 1)

    def _make_loaders(self, Xr, y):
        import numpy as np
        torch = self.torch
        # split train/val
        n = len(y)
        n_val = max(1, int(self.val_frac * n))
        n_tr = n - n_val
        X_tr, X_val = Xr[:n_tr], Xr[n_tr:]
        y_tr, y_val = y[:n_tr], y[n_tr:]
        # loaders
        ds_tr = torch.utils.data.TensorDataset(
            torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        ds_val = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(y_val))
        dl_tr = torch.utils.data.DataLoader(
            ds_tr, batch_size=self.batch, shuffle=True,
            pin_memory=(self.device.type=="cuda"))
        dl_val = torch.utils.data.DataLoader(
            ds_val, batch_size=self.batch, shuffle=False,
            pin_memory=(self.device.type=="cuda"))
        return dl_tr, dl_val

    def fit(self, X, y):
        import numpy as np, math
        torch, nn = self.torch, self.nn
        Xr = self._reshape(X)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        dl_tr, dl_val = self._make_loaders(Xr, y)

        # Optimizer (will work for both fallback and S4D; S4D has _optim hints internally)
        opt = self.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            # -------- train --------
            self.model.train()
            train_losses = []
            for xb, yb in dl_tr:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                if self.amp and self.autocast:
                    with self.autocast():
                        pred = self.model(xb)
                        loss = loss_fn(pred, yb)
                    self.scaler.scale(loss).backward()
                    if self.grad_clip is not None and self.grad_clip > 0:
                        self.scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(opt); self.scaler.update()
                else:
                    pred = self.model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    if self.grad_clip is not None and self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    opt.step()
                train_losses.append(loss.detach().item())

            # -------- validate --------
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in dl_val:
                    xb = xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)
                    pred = self.model(xb)
                    vloss = loss_fn(pred, yb)
                    val_losses.append(vloss.item())

            tr_loss = float(np.mean(train_losses)) if len(train_losses) else float("nan")
            va_loss = float(np.mean(val_losses)) if len(val_losses) else float("nan")
            va_rmse = math.sqrt(va_loss) if np.isfinite(va_loss) else float("nan")
            if self.verbose:
                print(f"[S4] epoch {epoch:03d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_RMSE={va_rmse:.4f}")

            # -------- early stopping --------
            if np.isfinite(va_loss) and (best_val - va_loss) > self.min_delta:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    if self.verbose:
                        print(f"[S4] Early stopping at epoch {epoch} (no improve {self.patience} epochs).")
                    break

        # restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X):
        import numpy as np
        torch = self.torch
        Xr = self._reshape(X)
        self.model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(Xr).to(self.device, non_blocking=True)
            out = self.model(xb)
        return out.detach().cpu().numpy()


class RupturesSegmentedLinear(BaseReg):
    """Change-point detection (ruptures) + last-segment OLS on lag features."""
    def __init__(self, penalty=10.0, min_size=20, model="l2"):
        try:
            import ruptures as rpt 
        except Exception as e:
            raise ImportError("pip install ruptures") from e
        from sklearn.linear_model import LinearRegression
        self._LR = LinearRegression
        import ruptures as rpt
        self.rpt = rpt
        self.penalty = float(penalty); self.min_size = int(min_size); self.model = model
        self.regs = []

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y).reshape(-1)
        algo = self.rpt.Pelt(model=self.model, min_size=self.min_size).fit(y)
        bkps = algo.predict(pen=self.penalty)
        starts = [0] + bkps[:-1]; ends = bkps
        self.regs = []
        for s, e in zip(starts, ends):
            reg = self._LR()
            reg.fit(X[s:e], y[s:e])
            self.regs.append(reg)
        return self

    def predict(self, X):
        if not self.regs: raise RuntimeError("Call fit() first.")
        return self.regs[-1].predict(np.asarray(X))


class MCDropoutGRU(BaseReg):
    """MC-Dropout GRU regressor on lag features (sequence -> scalar) with early stopping and logging."""
    def __init__(
        self,
        lags,
        hidden=128,
        layers=2,
        dropout=0.2,
        epochs=50,
        batch=128,
        lr=1e-3,
        weight_decay=1e-4,
        mc_samples=30,
        device="cpu",
        amp=False,
        val_frac=0.2,
        patience=10,
        min_delta=0.0,
        grad_clip=1.0,
        verbose=True,
        seed=42,
    ):
        import random, numpy as np
        import torch, torch.nn as nn, torch.optim as optim

        # Repro
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        self.np = np
        self.torch, self.nn, self.optim = torch, nn, optim
        self.device = torch.device(device)
        self.amp = bool(amp)
        self.autocast = getattr(torch.cuda.amp, "autocast", None)
        self.scaler = getattr(torch.cuda.amp, "GradScaler", lambda **k: None)(enabled=self.amp)

        self.lags = int(lags); self.hidden = int(hidden)
        self.layers = int(layers); self.dropout = float(dropout)
        self.epochs = int(epochs); self.batch = int(batch); self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.mc_samples = int(mc_samples)
        self.val_frac = float(val_frac)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.grad_clip = float(grad_clip)
        self.verbose = bool(verbose)

        class Net(nn.Module):
            def __init__(self, lags, hidden, layers, p):
                super().__init__()
                self.gru = nn.GRU(
                    input_size=1,
                    hidden_size=hidden,
                    num_layers=layers,
                    dropout=p if layers > 1 else 0.0,
                    batch_first=True,
                )
                self.drop = nn.Dropout(p)
                self.head = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, 1),
                )

            def forward(self, x):
                # x: (B, L, 1)
                out, _ = self.gru(x)     # out: (B, L, H)
                last = out[:, -1, :]     # (B, H)
                last = self.drop(last)   # MC at train+test (we keep model.train() at predict)
                return self.head(last).squeeze(-1)

        self.model = Net(self.lags, self.hidden, self.layers, self.dropout).to(self.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            try: torch.set_float32_matmul_precision("high")
            except Exception: pass

    def _reshape(self, X):
        import numpy as np
        X = np.asarray(X, dtype=np.float32)
        N, D = X.shape
        if D != self.lags:
            raise ValueError(f"Expected {self.lags} lags, got {D}")
        return X.reshape(N, self.lags, 1)

    def _make_loaders(self, Xr, y):
        torch = self.torch
        n = len(y)
        n_val = max(1, int(self.val_frac * n))
        n_tr = n - n_val
        X_tr, X_val = Xr[:n_tr], Xr[n_tr:]
        y_tr, y_val = y[:n_tr], y[n_tr:]

        ds_tr = torch.utils.data.TensorDataset(
            torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        ds_val = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(y_val))

        dl_tr = torch.utils.data.DataLoader(
            ds_tr, batch_size=self.batch, shuffle=True,
            pin_memory=(self.device.type == "cuda"))
        dl_val = torch.utils.data.DataLoader(
            ds_val, batch_size=self.batch, shuffle=False,
            pin_memory=(self.device.type == "cuda"))
        return dl_tr, dl_val

    def fit(self, X, y):
        import numpy as np, math
        torch, nn = self.torch, self.nn
        Xr = self._reshape(X)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        dl_tr, dl_val = self._make_loaders(Xr, y)

        opt = self.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, self.epochs + 1):
            # -------- train --------
            self.model.train()
            train_losses = []
            for xb, yb in dl_tr:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                if self.amp and self.autocast:
                    with self.autocast():
                        pred = self.model(xb)
                        loss = loss_fn(pred, yb)
                    self.scaler.scale(loss).backward()
                    if self.grad_clip and self.grad_clip > 0:
                        self.scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(opt); self.scaler.update()
                else:
                    pred = self.model(xb); loss = loss_fn(pred, yb)
                    loss.backward()
                    if self.grad_clip and self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    opt.step()
                train_losses.append(loss.detach().item())

            # -------- validate --------
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in dl_val:
                    xb = xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)
                    pred = self.model(xb)   # dropout inactive in eval, but that's fine for val loss
                    vloss = loss_fn(pred, yb)
                    val_losses.append(vloss.item())

            tr_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            va_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            va_rmse = math.sqrt(va_loss) if np.isfinite(va_loss) else float("nan")
            if self.verbose:
                print(f"[MCDropoutGRU] epoch {epoch:03d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_RMSE={va_rmse:.4f}")

            # -------- early stopping --------
            if np.isfinite(va_loss) and (best_val - va_loss) > self.min_delta:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    if self.verbose:
                        print(f"[MCDropoutGRU] Early stopping at epoch {epoch} (no improve {self.patience} epochs).")
                    break

        # restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X):
        import numpy as np
        torch = self.torch
        Xr = self._reshape(X)
        xb = torch.from_numpy(Xr).to(self.device, non_blocking=True)

        preds = []
        # Keep dropout active for MC sampling
        self.model.train()
        with torch.no_grad():
            for _ in range(self.mc_samples):
                preds.append(self.model(xb).detach().cpu().numpy())
        preds = np.stack(preds, axis=0)  # (S, N)
        return preds.mean(axis=0)        # mean prediction for ACI residuals

class GPTorchSparse(BaseReg):
    """Sparse variational GP (inducing points) via GPyTorch; falls back to sklearn if gpytorch not installed."""
    def __init__(self, lags, num_inducing=128, iters=300, lr=0.01, device="cpu"):
        self.lags = int(lags)
        self.num_inducing = int(num_inducing)
        self.iters = int(iters)
        self.lr = float(lr)
        self.device = device
        try:
            import torch, gpytorch
            self.torch, self.gpytorch = torch, gpytorch
            self._ok = True
        except Exception:
            self._ok = False

    def fit(self, X, y):
        if not self._ok:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
            self._sk = GaussianProcessRegressor(
                kernel=1.0*RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel(1e-3),
                normalize_y=True
            )
            self._sk.fit(np.asarray(X), np.asarray(y).reshape(-1))
            return self

        torch, gpytorch = self.torch, self.gpytorch
        X = torch.from_numpy(np.asarray(X, np.float32)).to(self.device)
        y = torch.from_numpy(np.asarray(y, np.float32)).reshape(-1).to(self.device)
        N, D = X.shape
        Z_idx = torch.linspace(0, N-1, steps=min(self.num_inducing, N)).round().long()
        inducing_points = X[Z_idx, :]

        class SVGP(gpytorch.models.ApproximateGP):
            def __init__(self, inducing_points):
                variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
                variational_strategy = gpytorch.variational.VariationalStrategy(
                    self, inducing_points, variational_distribution, learn_inducing_locations=True
                )
                super().__init__(variational_strategy)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=D)
                )
            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = SVGP(inducing_points).to(self.device)
        self.model.train(); self.likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=self.lr)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=N)

        for _ in range(self.iters):
            optimizer.zero_grad(set_to_none=True)
            output = self.model(X)
            loss = -mll(output, y)
            loss.backward(); optimizer.step()
        return self

    def predict(self, X):
        if not self._ok:
            return self._sk.predict(np.asarray(X))
        torch, gpytorch = self.torch, self.gpytorch
        self.model.eval(); self.likelihood.eval()
        X = torch.from_numpy(np.asarray(X, np.float32)).to(self.device)
        with torch.no_grad():
            preds = self.likelihood(self.model(X))
            mean = preds.mean
        return mean.detach().cpu().numpy()


# competitor_models.py (add this class alongside your other models)
import numpy as np

class DS3MWrapper:
    """
    Thin adapter to use the DS³M forecasting code inside the universal harness.

    - Ignores lag features (DS³M uses its own dataset loader); 'lags' is only used
      to align the prediction vector length to X.shape[0] (the lag-matrix length).
    - Returns a 1D array of length N (same as y_all) where the last `test_len`
      positions are DS³M one-step-ahead means (selected `target_dim`); the head
      part is filled but ignored by the harness because metrics are computed on
      the test tail [N - test_len : N].

    Parameters
    ----------
    lags : int
        Number of lags used by the harness (for alignment only).
    problem : str
        Dataset name in your DS³M repo: {"Toy","Lorenz","Sleep","Unemployment",
        "Hangzhou","Seattle","Pacific","Electricity"}.
    target_dim : int
        Which observation dimension to evaluate/return (default 0).
    train_size : int
        Calibration window T0 you used elsewhere (saved alongside cache).
    device : str
        "cpu" or "cuda" (passed through to DS³M utils).
    use_cache : bool
        If True, try reading a cached DS³M forecast first.
    force_new : bool
        If True, ignore any cache and recompute the DS³M forecast.
    """
    def __init__(self, lags,
                 problem="Sleep",
                 target_dim=0,
                 train_size=20,
                 device="cpu",
                 use_cache=True,
                 force_new=False):
        self.lags = int(lags)
        self.problem = problem
        self.target_dim = int(target_dim)
        self.train_size = int(train_size)
        self.device = device
        self.use_cache = bool(use_cache)
        self.force_new = bool(force_new)

        # filled during fit()
        self._yhat_all = None
        self._N = None
        self._test_len = None

    # --- private: run / load DS³M forecast and align to harness timeline ----
    def _get_ds3m_forecast(self, args):
        # Local imports to avoid hard dependency when not used
        from experiments.utils.ds3m_utils import (
            load_ds3m_data, load_ds3m_model, forecast
        )
        from experiments.utils.experiments_utils import (
            load_forecast, save_forecast
        )

        # Mimic the minimal 'args' needed by load_ds3m_data()
        # class _Args:
        #     def __init__(self, problem, train_size):
        #         self.problem = problem
        #         self.train_size = train_size
        # args = _Args(self.problem, self.train_size)

        # 0) try cache
        cached = None
        # if self.use_cache and not self.force_new:
        #     cached = load_forecast(self.problem)

        if cached is None:
            # 1) load DS³M dataset + trained checkpoint
            ds = load_ds3m_data(args)
            model = load_ds3m_model(
                ds["directoryBest"],
                ds["x_dim"], ds["y_dim"], ds["h_dim"], ds["z_dim"],
                ds["d_dim"], ds["n_layers"], ds["learning_rate"],
                ds["device"], bidirection=ds["bidirection"],
            )
            # 2) one-step prediction on the test block
            res, testForecast_mean, testOriginal, size, d_argmax, uq, lq = forecast(
                model,
                ds["testX"], ds["testY"],
                ds["moments"], ds["d_dim"],
                ds["means"], ds["trend"],
                ds["test_len"], ds["freq"],
                ds["RawDataOriginal"],
                remove_mean=ds["remove_mean"],
                remove_residual=ds["remove_residual"],
            )

            res_dict = dict(
                y_pred_mean=testForecast_mean,
                y_true=testOriginal,
                size=size,
                d_argmax=d_argmax,
                y_uq=uq, y_lq=lq,
                res_metric=res,
                problem=self.problem,
                train_size=self.train_size,
                device=ds["device"],
                test_len=ds["test_len"],
                predict_dim=ds["predict_dim"],
            )
            # 3) save to cache for reuse
            save_forecast(res_dict, self.problem)
        else:
            res_dict = cached

        return res_dict

    def fit(self, X, y, args):
        """
        Produces aligned predictions for the whole lag-matrix timeline.
        X : array of shape (N, lags)
        y : array of shape (N,)
        """
        X = np.asarray(X)
        N = X.shape[0]
        self._N = N

        res = self._get_ds3m_forecast(args)
        test_len = int(res["test_len"])
        self._test_len = test_len

        # testForecast_mean has shape (T_test, D) or (T_test,) – normalize to (T_test, D)
        y_pred_mean = res["y_pred_mean"]
        if y_pred_mean.ndim == 1:
            y_pred_mean = y_pred_mean[:, None]

        # Select target dimension
        td = np.clip(self.target_dim, 0, y_pred_mean.shape[1] - 1)
        y_pred_1d = y_pred_mean[:, td].reshape(-1)

        # Build a length-N vector and place predictions on the tail (the harness
        # computes metrics only on [N - test_len : N], i.e., the test tail).
        yhat_all = np.zeros(N, dtype=np.float32)
        tail = min(test_len, N)  # safety
        yhat_all[-tail:] = y_pred_1d[-tail:]

        self._yhat_all = yhat_all
        return self

    def predict(self, X, args):
        if self._yhat_all is None:
            # If .fit() wasn’t explicitly called, compute on the fly
            self.fit(X, np.zeros(len(X), dtype=np.float32), args)
        # Return 1D vector of length N (same as y_all)
        return self._yhat_all.copy()