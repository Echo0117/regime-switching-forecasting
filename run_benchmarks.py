# experiments/run_benchmarks_gpu.py
# -*- coding: utf-8 -*-
import argparse, os, sys, numpy as np

HERE = os.path.dirname(__file__)
for p in [HERE, os.path.abspath(os.path.join(HERE, ".."))]:
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.utils.acp_utils import aci_intervals, agaci_ewa
try:
    from experiments.utils.ds3m_utils import load_ds3m_data
except Exception as e:
    raise RuntimeError(
        "Cannot import ds3m_utils.load_ds3m_data; keep this script beside your project modules."
    ) from e


# ---------------------------
# Utilities
# ---------------------------
def make_lag_matrix(y, lags):
    y = np.asarray(y).reshape(-1)
    T = len(y)
    if lags >= T:
        raise ValueError("lags must be < length of series")
    X = np.empty((T - lags, lags), dtype=float)
    for i in range(lags, T):
        X[i - lags, :] = y[i - lags:i]
    y_t = y[lags:]
    return X, y_t


def pick_device(name: str):
    import torch
    name = (name or "").lower()
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda":
        print("[warn] CUDA requested but not available; falling back to CPU.")
    if name == "mps":
        print("[warn] MPS requested but not available; falling back to CPU.")
    return torch.device("cpu")


# ---------------------------
# Baselines (PyTorch-first)
# ---------------------------
class S4Regressor:
    """
    Wrapper around S4D from the official repo.
    Pass --s4-path /path/to/state-spaces (repo root) OR make sure PYTHONPATH contains that path,
    so `models/s4/s4d.py` is importable.

    If import fails, we fallback to a causal Conv1d baseline so the pipeline still runs.
    """

    def __init__(
        self,
        lags=48,
        d_model=128,
        n_layers=4,
        dropout=0.1,
        epochs=10,
        batch_size=64,
        lr=1e-3,
        device="cpu",
        amp=False,
        s4_path=None,
    ):
        import importlib
        try:
            import torch
            from torch import nn, optim
        except Exception as e:
            raise ImportError("PyTorch required. pip install torch") from e

        # Mixed precision helpers
        self.amp = bool(amp)
        self.autocast = getattr(torch.cuda.amp, "autocast", None)
        self.GradScaler = getattr(torch.cuda.amp, "GradScaler", None)
        self.scaler = self.GradScaler(enabled=self.amp) if self.GradScaler else None

        # Extend sys.path if user gave --s4-path
        if s4_path:
            s4_path = os.path.abspath(s4_path)
            if s4_path not in sys.path:
                sys.path.insert(0, s4_path)

        # Try to import S4D
        S4D = None
        tried = []
        for mod in ("models.s4.s4d", "s4.s4d", "s4d"):
            try:
                m = importlib.import_module(mod)
                S4D = getattr(m, "S4D", None)
                if S4D is not None:
                    break
            except Exception as ex:
                tried.append((mod, str(ex)))

        self.using_fallback = S4D is None
        self.torch = torch
        self.nn = nn
        self.optim = optim
        self.lags = int(lags)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.device = device

        # Build model
        if self.using_fallback:
            # Fallback: simple causal Conv1d tower
            class CausalConvModel(nn.Module):
                def __init__(self, lags, d_model, n_layers, p):
                    super().__init__()
                    layers = []
                    in_ch = 1
                    for i in range(n_layers):
                        conv = nn.Conv1d(in_ch, d_model, kernel_size=3, padding=2, dilation=2**i)
                        layers += [conv, nn.ReLU(), nn.Dropout(p)]
                        in_ch = d_model
                    self.net = nn.Sequential(*layers)
                    self.head = nn.Linear(d_model, 1)
                def forward(self, x):
                    # x: (B, L, 1)
                    x = x.transpose(1, 2)  # (B, 1, L)
                    z = self.net(x)        # (B, d_model, L)
                    z = z.mean(dim=-1)     # pool over time
                    out = self.head(z).squeeze(-1)
                    return out

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
                        S4D_cls(d_model, dropout=p, transposed=True, lr=s4_lr(lr))
                        for _ in range(n_layers)
                    ])
                    self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
                    self.drops = nn.ModuleList([Dropout(p) for _ in range(n_layers)])
                    self.dec = nn.Linear(d_model, 1)

                def forward(self, x):
                    # x: (B, L, 1)
                    x = self.enc(x)                     # (B, L, d_model)
                    x = x.transpose(-1, -2)            # (B, d_model, L)
                    for s4, ln, dp in zip(self.blocks, self.norms, self.drops):
                        z = x
                        z, _ = s4(z)                    # (B, d_model, L)
                        z = dp(z)
                        x = ln((z + x).transpose(-1, -2)).transpose(-1, -2)
                    x = x.transpose(-1, -2)            # (B, L, d_model)
                    x = x.mean(dim=1)                  # pool
                    out = self.dec(x).squeeze(-1)
                    return out

            self.model = S4Backbone(self.lags, self.d_model, self.n_layers, self.dropout).to(self.device)

        if self.device.type == "cuda":
            # Helps with stable perf on variable-length convs/SSMs
            self.torch.backends.cudnn.benchmark = True
            # Optional: better matmul kernels on Ampere+
            try:
                self.torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    def _reshape(self, X):
        X = np.asarray(X, dtype=np.float32)
        N, D = X.shape
        if D != self.lags:
            raise ValueError(f"S4Regressor expects input dim == lags ({self.lags}), got {D}")
        return X.reshape(N, self.lags, 1)

    def fit(self, X, y):
        torch = self.torch; nn = self.nn
        Xr = self._reshape(X)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        ds = torch.utils.data.TensorDataset(torch.from_numpy(Xr), torch.from_numpy(y))
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=True,
            pin_memory=(self.device.type == "cuda"), num_workers=0
        )
        opt = self.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                if self.amp and self.autocast:
                    with self.autocast():
                        pred = self.model(xb)
                        loss = loss_fn(pred, yb)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(opt)
                    self.scaler.update()
                else:
                    pred = self.model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    opt.step()
        return self

    def predict(self, X):
        torch = self.torch
        Xr = self._reshape(X)
        self.model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(Xr).to(self.device, non_blocking=True)
            if self.amp and self.autocast:
                with self.autocast():
                    out = self.model(xb)
            else:
                out = self.model(xb)
        return out.detach().cpu().numpy()


class RupturesSegmentedLinear:
    """Change-point detection (ruptures) + last-segment OLS on lag features."""
    def __init__(self, penalty=10.0, min_size=20, model="l2"):
        try:
            import ruptures as rpt
        except Exception as e:
            raise ImportError("pip install ruptures") from e
        self.rpt = __import__("ruptures")
        self.penalty = float(penalty)
        self.min_size = int(min_size)
        self.model = model
        from sklearn.linear_model import LinearRegression
        self._LR = LinearRegression
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
        if not self.regs:
            raise RuntimeError("Call fit() first.")
        return self.regs[-1].predict(np.asarray(X))


class MCDropoutRegressor:
    """MC-Dropout MLP on lag features (GPU+AMP capable)."""
    def __init__(
        self, d_in, hidden=128, dropout=0.2, epochs=10, batch_size=128,
        lr=1e-3, mc_samples=30, device="cpu", amp=False
    ):
        try:
            import torch
            from torch import nn, optim
        except Exception as e:
            raise ImportError("PyTorch required for MCDropoutRegressor") from e

        self.torch = __import__("torch")
        self.nn = nn
        self.optim = optim
        self.d_in = int(d_in)
        self.hidden = int(hidden); self.dropout = float(dropout)
        self.epochs = int(epochs); self.batch_size = int(batch_size)
        self.lr = float(lr); self.mc_samples = int(mc_samples)
        self.device = device

        self.amp = bool(amp)
        self.autocast = getattr(self.torch.cuda.amp, "autocast", None)
        self.GradScaler = getattr(self.torch.cuda.amp, "GradScaler", None)
        self.scaler = self.GradScaler(enabled=self.amp) if self.GradScaler else None

        class Net(nn.Module):
            def __init__(self, d_in, h, p):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(p),
                    nn.Linear(h, h),    nn.ReLU(), nn.Dropout(p),
                    nn.Linear(h, 1),
                )
            def forward(self, x): return self.net(x).squeeze(-1)

        self.model = Net(self.d_in, self.hidden, self.dropout).to(self.device)
        if self.device.type == "cuda":
            self.torch.backends.cudnn.benchmark = True

    def fit(self, X, y):
        torch = self.torch; nn = self.nn
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float32).reshape(-1)
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=True,
            pin_memory=(self.device.type == "cuda"), num_workers=0
        )
        opt = self.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                if self.amp and self.autocast:
                    with self.autocast():
                        pred = self.model(xb)
                        loss = loss_fn(pred, yb)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(opt)
                    self.scaler.update()
                else:
                    pred = self.model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    opt.step()
        return self

    def predict(self, X):
        torch = self.torch
        X = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device, non_blocking=True)
        self.model.train()  # keep dropout active
        preds = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                if self.amp and self.autocast:
                    with self.autocast():
                        preds.append(self.model(X).detach().cpu().numpy())
                else:
                    preds.append(self.model(X).detach().cpu().numpy())
        return np.mean(np.stack(preds, axis=0), axis=0)


class GPyTorchRegressor:
    """Exact GP in GPyTorch (CUDA if available). Falls back to sklearn GP if not installed."""
    def __init__(self, lags, max_iters=200, lr=0.1, device="cpu"):
        self.lags = int(lags)
        self.max_iters = int(max_iters)
        self.lr = float(lr)
        self.device = device
        self._ok = False
        try:
            import torch, gpytorch
            self.torch = torch
            self.gpytorch = gpytorch
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

        torch = self.torch; gpytorch = self.gpytorch
        X = torch.from_numpy(np.asarray(X, np.float32)).to(self.device)
        y = torch.from_numpy(np.asarray(y, np.float32)).reshape(-1).to(self.device)
        train_x = X; train_y = y

        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
                )
            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = ExactGPModel(train_x, train_y, likelihood).to(self.device)
        self.likelihood = likelihood
        self.model.train(); self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for _ in range(self.max_iters):
            optimizer.zero_grad(set_to_none=True)
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward(); optimizer.step()
        return self

    def predict(self, X):
        if not self._ok:
            return self._sk.predict(np.asarray(X))
        torch = self.torch; gpytorch = self.gpytorch
        self.model.eval(); self.likelihood.eval()
        X = torch.from_numpy(np.asarray(X, np.float32)).to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self.model(X))
            mean = preds.mean
        return mean.detach().cpu().numpy()


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def build_model(name, lags, params):
    name = name.lower()
    if name == "s4":
        return S4Regressor(
            lags=lags,
            d_model=params.get("s4_d_model", 128),
            n_layers=params.get("s4_layers", 4),
            dropout=params.get("s4_dropout", 0.1),
            epochs=params.get("s4_epochs", 10),
            batch_size=params.get("s4_batch", 64),
            lr=params.get("s4_lr", 1e-3),
            device=params.get("device", "cpu"),
            amp=params.get("amp", False),
            s4_path=params.get("s4_path", None),
        )
    if name == "cpd":
        return RupturesSegmentedLinear(
            penalty=params.get("cpd_penalty", 10.0),
            min_size=params.get("cpd_min_size", 20),
            model=params.get("cpd_model", "l2"),
        )
    if name == "mcdropout":
        return MCDropoutRegressor(
            d_in=lags,
            hidden=params.get("mcd_hidden", 128),
            dropout=params.get("mcd_dropout", 0.2),
            epochs=params.get("mcd_epochs", 10),
            batch_size=params.get("mcd_batch", 128),
            lr=params.get("mcd_lr", 1e-3),
            mc_samples=params.get("mcd_samples", 30),
            device=params.get("device", "cpu"),
            amp=params.get("amp", False),
        )
    if name == "gptorch":
        return GPyTorchRegressor(
            lags=lags,
            max_iters=params.get("gp_iters", 200),
            lr=params.get("gp_lr", 0.1),
            device=params.get("device", "cpu"),
        )
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=params.get("rf_trees", 300),
            min_samples_leaf=params.get("rf_leaf", 1),
            random_state=params.get("seed", 1),
            n_jobs=params.get("rf_jobs", 1),
        )
    if name == "ols":
        return LinearRegression()
    raise ValueError(f"Unknown model '{name}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="Pacific", choices=["Toy","Lorenz","Sleep","Unemployment","Hangzhou","Seattle","Pacific","Electricity"])
    ap.add_argument("--model", default="S4", choices=["S4","CPD","MCDropout","GPTorch","RF","OLS"])
    ap.add_argument("--lags", type=int, default=48)

    # ACI / AgACI
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.01)      # ACI learning rate
    ap.add_argument("--train_size", type=int, default=200)     # ACI calibration window (T0)
    ap.add_argument("--agaci", action="store_true", help="Use AgACI (EWA over gammas)")
    ap.add_argument("--agaci_gammas", type=float, nargs="*", default=[0.005, 0.01, 0.02, 0.05])

    # S4 / GPU / AMP
    ap.add_argument("--s4-path", default=None, help="Path to the state-spaces (S4) repo root (so models/s4/s4d.py is importable)")
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto)")
    ap.add_argument("--amp", action="store_true", help="Use mixed precision on GPU")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Choose device
    device = pick_device(args.device)

    # Seeding
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    ds = load_ds3m_data(args)

    # Try to locate a univariate series from loader output
    if "RawDataOriginal" in ds:
        y_full = np.asarray(ds["RawDataOriginal"]).reshape(-1)
    elif "data" in ds and isinstance(ds["data"], np.ndarray):
        y_full = ds["data"][:, 0].reshape(-1)
    else:
        raise RuntimeError("Could not locate univariate series (RawDataOriginal/data) in load_ds3m_data output.")

    test_len = int(ds.get("test_len", max(60, len(y_full)//5)))

    X_all, y_all = make_lag_matrix(y_full, args.lags)
    N = len(y_all)
    train_end = max(1, N - test_len)
    X_tr, y_tr = X_all[:train_end], y_all[:train_end]

    params = dict(
        device=device,
        seed=args.seed,
        s4_path=args.s4_path,
        amp=args.amp,
    )
    print(f"Device: {device}; AMP: {args.amp}")

    reg = build_model(args.model, args.lags, params)
    reg.fit(X_tr, y_tr)

    yhat_all = reg.predict(X_all)  # shape (N,)

    # ACI on residual magnitudes
    residuals = np.abs(y_all - yhat_all)
    T0 = args.train_size
    if T0 >= N:
        raise ValueError(f"train_size T0={T0} must be < N={N}")

    if args.agaci:
        lo_r, up_r = agaci_ewa(residuals, alpha=args.alpha, train_size=T0,
                               gammas=args.agaci_gammas, eta=0.1)
    else:
        lo_r, up_r = aci_intervals(residuals, alpha=args.alpha, gamma=args.gamma, train_size=T0)

    y_pred_seg = yhat_all[T0:]
    y_true_seg = y_all[T0:]
    covered = (y_true_seg >= (y_pred_seg - up_r)) & (y_true_seg <= (y_pred_seg + up_r))
    cov_all = float(np.mean(covered))
    width_all = float(np.mean(2.0 * up_r))

    start_test_in_seg = max(0, (N - test_len) - T0)
    covered_test = covered[start_test_in_seg:] if start_test_in_seg < len(covered) else np.array([])
    up_r_test = up_r[start_test_in_seg:] if start_test_in_seg < len(up_r) else np.array([])
    cov_test = float(np.mean(covered_test)) if covered_test.size else float("nan")
    width_test = float(np.mean(2.0 * up_r_test)) if up_r_test.size else float("nan")

    print(f"\n=== {args.problem} | {args.model} | lags={args.lags} ===")
    if isinstance(reg, S4Regressor) and reg.using_fallback:
        print("[warning] S4 import failed; using causal Conv1d fallback. Pass --s4-path to actual S4 repo if you need S4.")
    print(f"ACI={'AgACI' if args.agaci else 'ACI'} alpha={args.alpha} T0={T0}")
    print(f"Overall: coverage={cov_all:.3f} avg_width={width_all:.3f} (N_pred={len(y_true_seg)})")
    print(f"Test tail: coverage={cov_test:.3f} avg_width={width_test:.3f} (N_test={len(covered_test)})")


if __name__ == "__main__":
    main()


# # run_benchmarks.py
# # -*- coding: utf-8 -*-
# import argparse, os, sys, numpy as np

# HERE = os.path.dirname(__file__)
# for p in [HERE, os.path.abspath(os.path.join(HERE, ".."))]:
#     if p not in sys.path:
#         sys.path.insert(0, p)

# from acp_utils import aci_intervals, agaci_ewa

# try:
#     from ds3m_utils import load_ds3m_data
# except Exception as e:
#     raise RuntimeError("Cannot import ds3m_utils.load_ds3m_data; keep this script beside your project modules.") from e

# # ---------------------------
# # Utilities
# # ---------------------------

# def make_lag_matrix(y, lags):
#     y = np.asarray(y).reshape(-1)
#     T = len(y)
#     if lags >= T:
#         raise ValueError("lags must be < length of series")
#     X = np.empty((T - lags, lags), dtype=float)
#     for i in range(lags, T):
#         X[i - lags, :] = y[i - lags:i]
#     y_t = y[lags:]
#     return X, y_t

# # ---------------------------
# # Baselines (PyTorch-first)
# # ---------------------------

# class S4Regressor:
#     """
#     Wrapper around S4D from the official repo.

#     Pass --s4-path /path/to/state-spaces (repo root) OR make sure PYTHONPATH contains that path,
#     so that `models/s4/s4d.py` is importable.

#     If import fails, we fallback to a causal Conv1d baseline so the pipeline still runs.
#     """
#     def __init__(self, lags=48, d_model=128, n_layers=4, dropout=0.1,
#                  epochs=10, batch_size=64, lr=1e-3, device="cpu", s4_path=None):
#         import importlib
#         try:
#             import torch
#             from torch import nn, optim
#         except Exception as e:
#             raise ImportError("PyTorch required. pip install torch") from e

#         # Dynamically extend sys.path if user gave --s4-path
#         if s4_path:
#             s4_path = os.path.abspath(s4_path)
#             if s4_path not in sys.path:
#                 sys.path.insert(0, s4_path)

#         # Try to import S4D
#         S4D = None
#         tried = []
#         for mod in ("models.s4.s4d", "s4.s4d", "s4d"):
#             try:
#                 m = importlib.import_module(mod)
#                 S4D = getattr(m, "S4D", None)
#                 if S4D is not None:
#                     break
#             except Exception as ex:
#                 tried.append((mod, str(ex)))

#         self.using_fallback = S4D is None

#         self.torch = torch
#         self.nn = nn
#         self.optim = optim
#         self.lags = int(lags)
#         self.d_model = int(d_model)
#         self.n_layers = int(n_layers)
#         self.dropout = float(dropout)
#         self.epochs = int(epochs)
#         self.batch_size = int(batch_size)
#         self.lr = float(lr)
#         self.device = self.torch.device(device)

#         if self.using_fallback:
#             # Fallback: simple “SSM-ish” causal Conv1d tower (keeps pipeline alive)
#             class CausalConvModel(nn.Module):
#                 def __init__(self, lags, d_model, n_layers, p):
#                     super().__init__()
#                     layers = []
#                     in_ch = 1
#                     for i in range(n_layers):
#                         conv = nn.Conv1d(in_ch, d_model, kernel_size=3, padding=2, dilation=2**i)
#                         layers += [conv, nn.ReLU(), nn.Dropout(p)]
#                         in_ch = d_model
#                     self.net = nn.Sequential(*layers)
#                     self.head = nn.Linear(d_model, 1)
#                 def forward(self, x):  # x: (B, L, 1)
#                     x = x.transpose(1, 2)     # (B, 1, L)
#                     z = self.net(x)           # (B, d_model, L)
#                     z = z.mean(dim=-1)        # pool over time
#                     out = self.head(z).squeeze(-1)
#                     return out
#             self.model = CausalConvModel(self.lags, self.d_model, self.n_layers, self.dropout).to(self.device)
#         else:
#             Dropout = nn.Dropout
#             def s4_lr(base): return min(1e-3, base)
#             S4D_cls = S4D

#             class S4Backbone(nn.Module):
#                 def __init__(self, L, d_model, n_layers, p):
#                     super().__init__()
#                     self.enc = nn.Linear(1, d_model)
#                     self.blocks = nn.ModuleList([
#                         S4D_cls(d_model, dropout=p, transposed=True, lr=s4_lr(lr))
#                         for _ in range(n_layers)
#                     ])
#                     self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
#                     self.drops = nn.ModuleList([Dropout(p) for _ in range(n_layers)])
#                     self.dec = nn.Linear(d_model, 1)
#                 def forward(self, x):           # x: (B, L, 1)
#                     x = self.enc(x)             # (B, L, d_model)
#                     x = x.transpose(-1, -2)     # (B, d_model, L)
#                     for s4, ln, dp in zip(self.blocks, self.norms, self.drops):
#                         z = x
#                         z, _ = s4(z)            # (B, d_model, L)
#                         z = dp(z)
#                         x = ln((z + x).transpose(-1, -2)).transpose(-1, -2)
#                     x = x.transpose(-1, -2)     # (B, L, d_model)
#                     x = x.mean(dim=1)           # pool over L
#                     out = self.dec(x).squeeze(-1)
#                     return out

#             self.model = S4Backbone(self.lags, self.d_model, self.n_layers, self.dropout).to(self.device)

#     def _reshape(self, X):
#         X = np.asarray(X, dtype=np.float32)
#         N, D = X.shape
#         if D != self.lags:
#             raise ValueError(f"S4Regressor expects input dim == lags ({self.lags}), got {D}")
#         return X.reshape(N, self.lags, 1)

#     def fit(self, X, y):
#         torch = self.torch; nn = self.nn
#         Xr = self._reshape(X)
#         y = np.asarray(y, dtype=np.float32).reshape(-1)

#         ds = torch.utils.data.TensorDataset(torch.from_numpy(Xr), torch.from_numpy(y))
#         dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
#         opt = self.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
#         loss_fn = nn.MSELoss()
#         self.model.train()
#         for _ in range(self.epochs):
#             for xb, yb in dl:
#                 xb = xb.to(self.device); yb = yb.to(self.device)
#                 opt.zero_grad()
#                 pred = self.model(xb)
#                 loss = loss_fn(pred, yb)
#                 loss.backward(); opt.step()
#         return self

#     def predict(self, X):
#         torch = self.torch
#         Xr = self._reshape(X)
#         self.model.eval()
#         with torch.no_grad():
#             xb = torch.from_numpy(Xr).to(next(self.model.parameters()).device)
#             return self.model(xb).cpu().numpy()


# class RupturesSegmentedLinear:
#     """Change-point detection (ruptures) + last-segment OLS on lag features."""
#     def __init__(self, penalty=10.0, min_size=20, model="l2"):
#         import ruptures as rpt
#         self.rpt = __import__("ruptures")
#         self.penalty = float(penalty)
#         self.min_size = int(min_size)
#         self.model = model
#         from sklearn.linear_model import LinearRegression
#         self._LR = LinearRegression
#         self.regs = []

#     def fit(self, X, y):
#         X = np.asarray(X); y = np.asarray(y).reshape(-1)
#         algo = self.rpt.Pelt(model=self.model, min_size=self.min_size).fit(y)
#         bkps = algo.predict(pen=self.penalty)
#         starts = [0] + bkps[:-1]; ends = bkps
#         self.regs = []
#         for s, e in zip(starts, ends):
#             reg = self._LR()
#             reg.fit(X[s:e], y[s:e])
#             self.regs.append(reg)
#         return self

#     def predict(self, X):
#         if not self.regs:
#             raise RuntimeError("Call fit() first.")
#         return self.regs[-1].predict(np.asarray(X))


# class MCDropoutRegressor:
#     """MC-Dropout MLP on lag features."""
#     def __init__(self, d_in, hidden=128, dropout=0.2, epochs=10, batch_size=128, lr=1e-3, mc_samples=30, device="cpu"):
#         try:
#             from torch import nn, optim
#         except Exception as e:
#             raise ImportError("PyTorch required for MCDropoutRegressor") from e
#         self.torch = __import__("torch")
#         self.nn = nn
#         self.optim = optim
#         self.d_in = int(d_in)
#         self.hidden = int(hidden); self.dropout = float(dropout)
#         self.epochs = int(epochs); self.batch_size = int(batch_size)
#         self.lr = float(lr); self.mc_samples = int(mc_samples)
#         self.device = self.torch.device(device)

#         class Net(nn.Module):
#             def __init__(self, d_in, h, p):
#                 super().__init__()
#                 self.net = nn.Sequential(
#                     nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(p),
#                     nn.Linear(h, h), nn.ReLU(), nn.Dropout(p),
#                     nn.Linear(h, 1),
#                 )
#             def forward(self, x): return self.net(x).squeeze(-1)
#         self.model = Net(self.d_in, self.hidden, self.dropout).to(self.device)

#     def fit(self, X, y):
#         torch = self.torch; nn = self.nn
#         X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float32).reshape(-1)
#         ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
#         dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
#         opt = self.optim.Adam(self.model.parameters(), lr=self.lr)
#         loss_fn = nn.MSELoss()
#         self.model.train()
#         for _ in range(self.epochs):
#             for xb, yb in dl:
#                 xb = xb.to(self.device); yb = yb.to(self.device)
#                 opt.zero_grad()
#                 pred = self.model(xb)
#                 loss = loss_fn(pred, yb)
#                 loss.backward(); opt.step()
#         return self

#     def predict(self, X):
#         torch = self.torch
#         X = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
#         self.model.train()  # keep dropout active
#         preds = []
#         with torch.no_grad():
#             for _ in range(self.mc_samples):
#                 preds.append(self.model(X).cpu().numpy())
#         return np.mean(np.stack(preds, axis=0), axis=0)


# # PyTorch-first GP (GPyTorch). Falls back to sklearn GP if gpytorch not installed.
# class GPyTorchRegressor:
#     def __init__(self, lags, max_iters=200, lr=0.1, device="cpu"):
#         self.lags = int(lags)
#         self.max_iters = int(max_iters)
#         self.lr = float(lr)
#         self.device = device
#         self._ok = False
#         try:
#             import torch
#             import gpytorch
#             self.torch = torch
#             self.gpytorch = gpytorch
#             self._ok = True
#         except Exception:
#             self._ok = False

#     def fit(self, X, y):
#         if not self._ok:
#             # fallback to sklearn GP
#             from sklearn.gaussian_process import GaussianProcessRegressor
#             from sklearn.gaussian_process.kernels import RBF, WhiteKernel
#             self._sk = GaussianProcessRegressor(kernel=1.0*RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel(1e-3),
#                                                 normalize_y=True)
#             self._sk.fit(np.asarray(X), np.asarray(y).reshape(-1))
#             return self

#         torch = self.torch; gpytorch = self.gpytorch
#         X = torch.from_numpy(np.asarray(X, np.float32))
#         y = torch.from_numpy(np.asarray(y, np.float32)).reshape(-1)
#         train_x = X; train_y = y

#         class ExactGPModel(gpytorch.models.ExactGP):
#             def __init__(self, train_x, train_y, likelihood):
#                 super().__init__(train_x, train_y, likelihood)
#                 self.mean_module = gpytorch.means.ConstantMean()
#                 self.covar_module = gpytorch.kernels.ScaleKernel(
#                     gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
#                 )
#             def forward(self, x):
#                 mean_x = self.mean_module(x)
#                 covar_x = self.covar_module(x)
#                 return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

#         likelihood = gpytorch.likelihoods.GaussianLikelihood()
#         self.model = ExactGPModel(train_x, train_y, likelihood).to(self.device)
#         self.likelihood = likelihood.to(self.device)

#         self.model.train(); self.likelihood.train()
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
#         mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
#         for _ in range(self.max_iters):
#             optimizer.zero_grad()
#             output = self.model(train_x)
#             loss = -mll(output, train_y)
#             loss.backward(); optimizer.step()
#         return self

#     def predict(self, X):
#         if not self._ok:
#             return self._sk.predict(np.asarray(X))
#         torch = self.torch; gpytorch = self.gpytorch
#         self.model.eval(); self.likelihood.eval()
#         X = torch.from_numpy(np.asarray(X, np.float32))
#         with torch.no_grad(), gpytorch.settings.fast_pred_var():
#             preds = self.likelihood(self.model(X))
#             mean = preds.mean
#         return mean.cpu().numpy()


# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression


# def build_model(name, lags, params):
#     name = name.lower()
#     if name == "s4":
#         return S4Regressor(
#             lags=lags,
#             d_model=params.get("s4_d_model", 128),
#             n_layers=params.get("s4_layers", 4),
#             dropout=params.get("s4_dropout", 0.1),
#             epochs=params.get("s4_epochs", 10),
#             batch_size=params.get("s4_batch", 64),
#             lr=params.get("s4_lr", 1e-3),
#             device=params.get("device", "cpu"),
#             s4_path=params.get("s4_path", None),
#         )
#     if name == "cpd":
#         return RupturesSegmentedLinear(
#             penalty=params.get("cpd_penalty", 10.0),
#             min_size=params.get("cpd_min_size", 20),
#             model=params.get("cpd_model", "l2"),
#         )
#     if name == "mcdropout":
#         return MCDropoutRegressor(
#             d_in=lags,
#             hidden=params.get("mcd_hidden", 128),
#             dropout=params.get("mcd_dropout", 0.2),
#             epochs=params.get("mcd_epochs", 10),
#             batch_size=params.get("mcd_batch", 128),
#             lr=params.get("mcd_lr", 1e-3),
#             mc_samples=params.get("mcd_samples", 30),
#             device=params.get("device", "cpu"),
#         )
#     if name == "gptorch":
#         return GPyTorchRegressor(lags=lags, max_iters=params.get("gp_iters", 200),
#                                  lr=params.get("gp_lr", 0.1), device=params.get("device", "cpu"))
#     if name == "rf":
#         return RandomForestRegressor(
#             n_estimators=params.get("rf_trees", 300),
#             min_samples_leaf=params.get("rf_leaf", 1),
#             random_state=params.get("seed", 1),
#             n_jobs=params.get("rf_jobs", 1),
#         )
#     if name == "ols":
#         return LinearRegression()
#     raise ValueError(f"Unknown model '{name}'")


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--problem", default="Pacific",
#                     choices=["Toy","Lorenz","Sleep","Unemployment","Hangzhou","Seattle","Pacific","Electricity"])
#     ap.add_argument("--model", default="S4",
#                     choices=["S4","CPD","MCDropout","GPTorch","RF","OLS"])
#     ap.add_argument("--lags", type=int, default=48)

#     # ACI / AgACI
#     ap.add_argument("--alpha", type=float, default=0.1)
#     ap.add_argument("--gamma", type=float, default=0.01)           # ACI learning rate
#     ap.add_argument("--train_size", type=int, default=200)         # ACI calibration window (T0)
#     ap.add_argument("--agaci", action="store_true", help="Use AgACI (EWA over gammas)")
#     ap.add_argument("--agaci_gammas", type=float, nargs="*", default=[0.005, 0.01, 0.02, 0.05])

#     # S4 convenience
#     ap.add_argument("--s4-path", default=None, help="Path to the state-spaces (S4) repo root (so models/s4/s4d.py is importable)")

#     ap.add_argument("--seed", type=int, default=42)
#     ap.add_argument("--device", default="cpu")
#     args = ap.parse_args()
#     np.random.seed(args.seed)

#     ds = load_ds3m_data(args)
#     # Try to locate a univariate series from loader output
#     if "RawDataOriginal" in ds:
#         y_full = np.asarray(ds["RawDataOriginal"]).reshape(-1)
#     elif "data" in ds and isinstance(ds["data"], np.ndarray):
#         y_full = ds["data"][:, 0].reshape(-1)
#     else:
#         raise RuntimeError("Could not locate univariate series (RawDataOriginal/data) in load_ds3m_data output.")

#     test_len = int(ds.get("test_len", max(60, len(y_full)//5)))
#     X_all, y_all = make_lag_matrix(y_full, args.lags)  # lengths = len - lags
#     N = len(y_all)
#     train_end = max(1, N - test_len)
#     X_tr, y_tr = X_all[:train_end], y_all[:train_end]

#     params = dict(device=args.device, seed=args.seed, s4_path=args.s4_path)
#     reg = build_model(args.model, args.lags, params)
#     reg.fit(X_tr, y_tr)
#     yhat_all = reg.predict(X_all)                      # shape (N,)

#     # ACI on residual magnitudes
#     residuals = np.abs(y_all - yhat_all)
#     T0 = args.train_size
#     if T0 >= N:
#         raise ValueError(f"train_size T0={T0} must be < N={N}")

#     if args.agaci:
#         lo_r, up_r = agaci_ewa(residuals, alpha=args.alpha, train_size=T0,
#                                gammas=args.agaci_gammas, eta=0.1)
#     else:
#         lo_r, up_r = aci_intervals(residuals, alpha=args.alpha, gamma=args.gamma, train_size=T0)

#     y_pred_seg = yhat_all[T0:]
#     y_true_seg = y_all[T0:]
#     covered = (y_true_seg >= (y_pred_seg - up_r)) & (y_true_seg <= (y_pred_seg + up_r))
#     cov_all = float(np.mean(covered))
#     width_all = float(np.mean(2.0 * up_r))

#     start_test_in_seg = max(0, (N - test_len) - T0)
#     covered_test = covered[start_test_in_seg:] if start_test_in_seg < len(covered) else np.array([])
#     up_r_test = up_r[start_test_in_seg:] if start_test_in_seg < len(up_r) else np.array([])
#     cov_test = float(np.mean(covered_test)) if covered_test.size else float("nan")
#     width_test = float(np.mean(2.0 * up_r_test)) if up_r_test.size else float("nan")

#     print(f"\n=== {args.problem} | {args.model} | lags={args.lags} ===")
#     if isinstance(reg, S4Regressor) and reg.using_fallback:
#         print("[warning] S4 import failed; using causal Conv1d fallback. Pass --s4-path to actual S4 repo if you need S4.")
#     print(f"ACI={'AgACI' if args.agaci else 'ACI'} alpha={args.alpha} T0={T0}")
#     print(f"Overall:   coverage={cov_all:.3f}   avg_width={width_all:.3f}   (N_pred={len(y_true_seg)})")
#     print(f"Test tail: coverage={cov_test:.3f}   avg_width={width_test:.3f}  (N_test={len(covered_test)})")


# if __name__ == "__main__":
#     main()
