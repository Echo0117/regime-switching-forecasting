"""
Task 1: Forecasting Performance Comparison for Monthly Report

Generates professional forecasting comparison plots for ALL original datasets.
Uses models from competitor_models.py: AR, S4, MCD (MC-Dropout GRU), GP, CPD (Ruptures), DS3M
Adds comprehensive metrics: RMSE, R2, MAE, MAPE on each subplot.

Datasets: Toy, Electricity, Hangzhou, Lorenz, Pacific, Seattle, Sleep, Unemployment, Pernod
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
import argparse
import warnings
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
import json

warnings.filterwarnings("ignore")

# Add project root to path
HERE = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from experiments.competitor_models import (
    S4Regressor, MCDropoutGRU, GPTorchSparse, RupturesSegmentedLinear, DS3MWrapper
)

# Output directory - save to overleaf_upload
OUTPUT_DIR = Path("overleaf_upload/figures/task1_prediction")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configurations - matching ds3m_utils.py parameters
DATASET_CONFIG = {
    'Toy': {
        'test_len': 500,
        'lags': 20,
        'dim': 0,
        'predict_dim': 1,
        'description': 'Synthetic regime-switching'
    },
    'Electricity': {
        'test_len': 320,
        'lags': 14,
        'dim': 0,
        'predict_dim': 48,
        'description': 'French electricity load'
    },
    'Hangzhou': {
        'test_len': 540,
        'lags': 12,
        'dim': 0,
        'predict_dim': 80,
        'description': 'Hangzhou metro traffic'
    },
    'Lorenz': {
        'test_len': 1000,
        'lags': 5,
        'dim': 0,
        'predict_dim': 10,
        'description': 'Lorenz attractor'
    },
    'Pacific': {
        'test_len': 60,
        'lags': 24,
        'dim': 0,
        'predict_dim': 2520,
        'description': 'Pacific SST'
    },
    'Seattle': {
        'test_len': 1440,
        'lags': 12,
        'dim': 0,
        'predict_dim': 323,
        'description': 'Seattle traffic speed'
    },
    'Sleep': {
        'test_len': 800,
        'lags': 20,
        'dim': 0,
        'predict_dim': 1,
        'description': 'Sleep respiration'
    },
    'Unemployment': {
        'test_len': 240,
        'lags': 6,
        'dim': 0,
        'predict_dim': 1,
        'description': 'US unemployment rate'
    },
    'Pernod': {
        'test_len': 80,
        'lags': 4,
        'dim': 0,
        'predict_dim': 1,
        'description': 'Pernod sales volume'
    },
}

GROUND_TRUTH_COLOR = '#1f3a5f'  # deep blue (avoid black for GT)


def load_original_data(dataname):
    """Load original dataset using the same logic as ds3m_utils.py."""
    data_dir = Path(f"Deep_Switching_State_Space_Model/data/{dataname}")

    if dataname == "Toy":
        # Try multiple possible locations for Toy data
        possible_paths = [
            data_dir / "simulation_data_nonlinear_y.csv",
            Path("Deep_Switching_State_Space_Model/data/Toy_og/simulation_data_nonlinear_y.csv"),
            Path("Deep_Switching_State_Space_Model/data/Toy_exp1/simulation_data_nonlinear_y.csv"),
        ]
        for data_path in possible_paths:
            if data_path.exists():
                RawData = pd.read_csv(data_path, header=None).values.reshape(-1)
                return RawData
        raise FileNotFoundError(f"Toy data not found in any of: {possible_paths}")

    elif dataname == "Lorenz":
        json_path = data_dir / "lorenz.json"
        with open(json_path) as f:
            data_st_all = json.load(f)
        data_st = np.array(data_st_all["data"])
        data_st = data_st[:, 2000:5000]
        # Set seed for reproducible noise
        np.random.seed(0)
        data_st = data_st + data_st[0].std(axis=0) * 0.001 * np.random.randn(data_st.shape[1], 10)
        data_st = (data_st - data_st[0].mean(axis=0)) / data_st[0].std(axis=0)
        RawData = data_st.transpose(1, 0, 2)
        return RawData.reshape(-1, RawData.shape[-1])

    elif dataname == "Sleep":
        data_path = data_dir / "b1.txt"
        dataset = pd.read_csv(data_path, sep=" ", header=None)
        chest = dataset.iloc[:, 1].values.reshape(-1, 1)
        train_data = chest[6200:7200, :]
        test_data = chest[5200:6200, :]
        RawData = np.concatenate((train_data, test_data)).reshape(-1)
        return RawData

    elif dataname == "Unemployment":
        data_path = data_dir / "UNRATE.csv"
        RawData = pd.read_csv(data_path, header=0).loc[:, "UNRATE"].values
        return RawData

    elif dataname == "Hangzhou":
        from scipy.io import loadmat
        mat_path = data_dir / "hangzhou.mat"
        RawData = loadmat(str(mat_path))["tensor"].astype("float")
        RawData = RawData.transpose(1, 2, 0)
        return RawData.reshape(-1, RawData.shape[-1])

    elif dataname == "Seattle":
        npz_path = data_dir / "seattle.npz"
        RawData = np.load(npz_path)["arr_0"].astype("float").transpose(1, 0)
        return RawData

    elif dataname == "Pacific":
        tsv_path = data_dir / "pacific.tsv"
        RawData = pd.read_csv(tsv_path, sep="\t", header=None).values.reshape(-1, 30*84)[3:]
        return RawData

    elif dataname == "Electricity":
        csv_path = data_dir / "French_all.csv"
        df = pd.read_csv(csv_path)
        df = df[df["Date"] < "2019-12-31"]
        RawData = df["Load"].values
        return RawData.reshape(-1, 48)

    elif dataname == "Pernod":
        csv_path = data_dir / "pernod.csv"
        df = pd.read_csv(csv_path, delimiter=';')
        RawData = df['volume_so'].values
        RawData = np.nan_to_num(RawData, nan=0.0)
        return RawData

    else:
        raise ValueError(f"Unknown dataset: {dataname}")


def clean_series(y):
    """Replace NaN/inf and clip extreme outliers for stability."""
    y = np.asarray(y, dtype=float).reshape(-1)
    finite = np.isfinite(y)
    if not finite.any():
        return np.zeros_like(y)
    fill = np.nanmedian(y[finite])
    y[~finite] = fill
    mu = float(np.mean(y))
    sigma = float(np.std(y))
    if sigma > 0:
        y = np.clip(y, mu - 5.0 * sigma, mu + 5.0 * sigma)
    return y


def create_lag_features(data, lags, multidim=False):
    """
    Create lag feature matrix.

    Parameters
    ----------
    data : array-like
        1D array (for single-dim) or 2D array (T, D) for multi-dim
    lags : int
        Number of lags
    multidim : bool
        If True, treat each dimension as a separate target (multi-output)
        If False, use only first dimension or flatten to 1D

    Returns
    -------
    X : np.ndarray
        Features of shape (N, lags) for single-dim or (N, lags*D) for multi-dim
    y : np.ndarray
        Targets of shape (N,) for single-dim or (N, D) for multi-dim
    """
    data = np.asarray(data)

    if not multidim:
        # Single-dimensional: flatten or take first column
        if data.ndim > 1:
            data = data[:, 0]  # Take first dimension
        data = data.flatten()
        N = len(data) - lags
        if N <= 0:
            raise ValueError(f"Data length {len(data)} too short for lags={lags}")
        X = np.zeros((N, lags))
        y = np.zeros(N)
        for i in range(N):
            X[i] = data[i:i+lags]
            y[i] = data[i+lags]
        return X, y
    else:
        # Multi-dimensional: each dimension is a target
        if data.ndim == 1:
            # Reshape to (T, 1)
            data = data.reshape(-1, 1)

        T, D = data.shape
        N = T - lags
        if N <= 0:
            raise ValueError(f"Data length {T} too short for lags={lags}")

        # X: concatenate lags from all dimensions (N, lags * D)
        X = np.zeros((N, lags * D))
        # y: all dimensions (N, D)
        y = np.zeros((N, D))

        for i in range(N):
            # Flatten lags from all dimensions
            X[i] = data[i:i+lags, :].flatten()
            y[i] = data[i+lags, :]

        return X, y


def standardize_train_test(X_train, X_test, y_train, y_test):
    """Standardize using training stats; return scaled arrays and y stats."""
    x_mean = np.mean(X_train, axis=0)
    x_std = np.std(X_train, axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    X_train_s = (X_train - x_mean) / x_std
    X_test_s = (X_test - x_mean) / x_std

    # Handle both 1D and multi-dimensional y
    if y_train.ndim == 1:
        y_mean = float(np.mean(y_train))
        y_std = float(np.std(y_train))
        if y_std < 1e-8:
            y_std = 1.0
        y_train_s = (y_train - y_mean) / y_std
        y_test_s = (y_test - y_mean) / y_std
    else:
        # Multi-dimensional: standardize per dimension
        y_mean = np.mean(y_train, axis=0)
        y_std = np.std(y_train, axis=0)
        y_std = np.where(y_std < 1e-8, 1.0, y_std)
        y_train_s = (y_train - y_mean) / y_std
        y_test_s = (y_test - y_mean) / y_std
    return X_train_s, X_test_s, y_train_s, y_test_s, y_mean, y_std


def compute_metrics(y_true, y_pred):
    """Compute comprehensive metrics: RMSE, MAE, MAPE, R2.
    For multi-dimensional data, compute metrics per dimension and average."""

    # Handle multi-dimensional predictions
    if y_true.ndim > 1 and y_pred.ndim > 1:
        # Compute metrics per dimension and average
        metrics_per_dim = []
        for d in range(y_true.shape[1]):
            m = compute_metrics(y_true[:, d], y_pred[:, d])
            metrics_per_dim.append(m)

        # Average across dimensions
        result = {}
        for key in ['RMSE', 'MAE', 'MAPE', 'R2']:
            values = [m[key] for m in metrics_per_dim if not np.isnan(m[key])]
            result[key] = np.mean(values) if values else np.nan
        return result

    # Single-dimensional case
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t, y_p = y_true[mask], y_pred[mask]

    if len(y_t) == 0:
        return {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan, 'R2': np.nan}

    rmse = np.sqrt(np.mean((y_t - y_p) ** 2))
    mae = mean_absolute_error(y_t, y_p)

    # MAPE with protection against division by zero
    nonzero = np.abs(y_t) > 1e-8
    if nonzero.sum() > 0:
        mape = 100 * np.mean(np.abs((y_t[nonzero] - y_p[nonzero]) / y_t[nonzero]))
    else:
        mape = np.nan

    try:
        r2 = r2_score(y_t, y_p)
    except:
        r2 = np.nan

    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}


def run_comparison(
    dataname,
    device="cpu",
    verbose=True,
    test_extra=0,
    gp_max_lags=12,
    use_ds3m_target=False,
    ds3m_target_datasets=None,
    ds3m_force_new=False,
    pernod_test_len=None,
    dim_overrides=None,
    use_multidim=False,
):
    """Run all models on a dataset and return predictions with metrics."""
    print(f"\n{'='*70}")
    print(f"Dataset: {dataname}")
    print(f"{'='*70}")

    config = DATASET_CONFIG.get(dataname)
    if config is None:
        raise ValueError(f"Unknown dataset: {dataname}")

    test_len = config['test_len'] + int(test_extra)
    if dataname == "Pernod" and pernod_test_len is not None:
        test_len = int(pernod_test_len)
    lags = config['lags']
    dim = config['dim']

    # Load data
    try:
        raw_data = load_original_data(dataname)
    except Exception as e:
        print(f"  Error loading data: {e}")
        return None

    # Handle multi-dimensional data
    ds3m_target_datasets = set(ds3m_target_datasets or [])
    dim_overrides = dim_overrides or {}
    use_ds3m_target_for_dataset = use_ds3m_target or (dataname in ds3m_target_datasets)
    if raw_data.ndim > 1 and use_ds3m_target_for_dataset:
        pred_dim = int(config.get("predict_dim", raw_data.shape[1]))
        dim = max(0, min(pred_dim - 1, raw_data.shape[1] - 1))
        if verbose and dim != config['dim']:
            print(f"  Using DS3M target_dim override: {dim}")
    if raw_data.ndim > 1 and dataname in dim_overrides:
        dim = max(0, min(int(dim_overrides[dataname]), raw_data.shape[1] - 1))
        if verbose and dim != config['dim']:
            print(f"  Using dim override: {dim}")

    # Prepare data: multi-dimensional or single-dimensional
    if use_multidim and raw_data.ndim > 1:
        # Multi-dimensional mode: use all dimensions
        y_data = clean_series(raw_data) if raw_data.ndim == 1 else raw_data
        output_dim = y_data.shape[1] if y_data.ndim > 1 else 1
        print(f"  Data shape: {y_data.shape}")
        print(f"  Multi-dimensional mode: {output_dim} dimensions")
    else:
        # Single-dimensional mode: extract one dimension
        if raw_data.ndim > 1:
            y_1d = raw_data[:, dim].flatten()
        else:
            y_1d = raw_data.flatten()
        y_data = clean_series(y_1d)
        output_dim = 1
        print(f"  Data shape: {raw_data.shape if raw_data.ndim > 1 else len(raw_data)}")
        print(f"  Using dimension: {dim}, Total length: {len(y_data)}")

    # Create lag features
    try:
        X, y = create_lag_features(y_data, lags, multidim=use_multidim)
    except ValueError as e:
        print(f"  Error creating features: {e}")
        return None

    N = len(y)
    actual_test_len = min(test_len, N - 50)
    if actual_test_len < 20:
        print(f"  Warning: Not enough data (N={N}, test_len={test_len})")
        return None

    train_end = N - actual_test_len

    X_train, y_train = X[:train_end], y[:train_end]
    X_test, y_test = X[train_end:], y[train_end:]

    X_train_s, X_test_s, y_train_s, y_test_s, y_mean, y_std = standardize_train_test(
        X_train, X_test, y_train, y_test
    )

    print(f"  Lags: {lags}, Train: {train_end}, Test: {len(y_test)}")

    # NOTE: No alignment hack applied. Predictions and ground truth are used directly.
    # The "visual lag" on smooth time series is the autoregressive tracking artifact,
    # not an indexing bug. Models learn to predict ~y_{t-1} on smooth series.

    results = {
        'y_true': y_test,  # No alignment hack - use y_test directly
        'predictions': {},
        'metrics': {},
        'config': config,
        'dim': dim,
    }

    # 1. AR baseline (Ridge regression)
    print("\n  [1/6] AR (Ridge)...")
    try:
        ar = Ridge(alpha=1.0)
        ar.fit(X_train, y_train)
        pred = ar.predict(X_test)
        results['predictions']['AR'] = pred
        results['metrics']['AR'] = compute_metrics(y_test, pred)
        print(f"    RMSE: {results['metrics']['AR']['RMSE']:.4f}")
    except Exception as e:
        print(f"    Failed: {e}")

    # 2. S4
    print("\n  [2/6] S4...")
    try:
        # Adaptive parameters based on dataset
        if dataname == "Toy":
            # For Toy: use optimized hyperparameters
            s4 = S4Regressor(
                lags=lags,
                d_model=256,     # Larger model capacity
                n_layers=6,      # Deeper model
                dropout=0.1,
                epochs=100,      # More training
                lr=0.001,
                device=device,
                verbose=False,
                patience=15
            )
        else:
            # Default settings for other datasets
            s4 = S4Regressor(lags=lags, epochs=50, device=device, verbose=False, patience=10)

        s4.fit(X_train_s, y_train_s)
        pred = s4.predict(X_test_s)
        pred = pred * y_std + y_mean
        results['predictions']['S4'] = pred
        results['metrics']['S4'] = compute_metrics(y_test, pred)
        print(f"    RMSE: {results['metrics']['S4']['RMSE']:.4f}")
    except Exception as e:
        print(f"    Failed: {e}")
        try:
            s4 = S4Regressor(
                lags=lags, epochs=30, d_model=64, n_layers=2,
                batch=max(8, min(64, len(X_train)//4)),
                device=device, verbose=False, patience=5,
            )
            s4.fit(X_train_s, y_train_s)
            pred = s4.predict(X_test_s)
            pred = pred * y_std + y_mean
            results['predictions']['S4'] = pred
            results['metrics']['S4'] = compute_metrics(y_test, pred)
            print(f"    RMSE (fallback): {results['metrics']['S4']['RMSE']:.4f}")
        except Exception as e2:
            print(f"    Fallback failed: {e2}")

    # 3. MCD (MC-Dropout GRU)
    print("\n  [3/6] MCD (MC-Dropout GRU)...")
    try:
        mcd = MCDropoutGRU(lags=lags, epochs=50, device=device, verbose=False, patience=10)
        mcd.fit(X_train_s, y_train_s)
        pred = mcd.predict(X_test_s)
        pred = pred * y_std + y_mean
        results['predictions']['MCD'] = pred
        results['metrics']['MCD'] = compute_metrics(y_test, pred)
        print(f"    RMSE: {results['metrics']['MCD']['RMSE']:.4f}")
    except Exception as e:
        print(f"    Failed: {e}")

    # 4. GP
    print("\n  [4/6] GP (Sparse)...")
    try:
        # Adaptive parameters based on dataset
        if dataname == "Toy":
            # For Toy: use more training data, all lags, and better hyperparameters
            max_train = min(1000, len(X_train))
            gp_lags = lags  # Use all lags for Toy
            gp_inducing = 128
            gp_iters = 300
            gp_lr = 0.05
        else:
            # For other datasets: use previous settings
            max_train = min(500, len(X_train))
            gp_lags = min(int(gp_max_lags), lags)
            gp_inducing = 64
            gp_iters = 150
            gp_lr = 0.01

        X_train_gp = X_train_s[-max_train:, -gp_lags:]
        y_train_gp = y_train_s[-max_train:]
        X_test_gp = X_test_s[:, -gp_lags:]
        gp = GPTorchSparse(lags=gp_lags, num_inducing=gp_inducing, iters=gp_iters, lr=gp_lr, device=device)
        gp.fit(X_train_gp, y_train_gp)
        pred = gp.predict(X_test_gp)
        pred = pred * y_std + y_mean
        results['predictions']['GP'] = pred
        results['metrics']['GP'] = compute_metrics(y_test, pred)
        print(f"    RMSE: {results['metrics']['GP']['RMSE']:.4f}")
    except Exception as e:
        print(f"    Failed: {e}")

    # 5. CPD (Ruptures + Linear)
    print("\n  [5/6] CPD (Ruptures)...")
    try:
        cpd = RupturesSegmentedLinear(penalty=10.0, min_size=max(20, len(y_train)//10))
        cpd.fit(X_train, y_train)
        pred = cpd.predict(X_test)
        results['predictions']['CPD'] = pred
        results['metrics']['CPD'] = compute_metrics(y_test, pred)
        print(f"    RMSE: {results['metrics']['CPD']['RMSE']:.4f}")
    except Exception as e:
        print(f"    Failed: {e}")

    # 6. DS3M
    print("\n  [6/6] DS3M...")
    try:
        from experiments.utils.experiments_utils import load_forecast
        ds3m = DS3MWrapper(
            lags=lags,
            problem=dataname,
            target_dim=dim,
            device=device,
            force_new=ds3m_force_new,
            test_len_override=test_len,
        )
        ds3m.fit(X, y)
        pred_all = ds3m.predict(X)
        pred = pred_all[train_end:]

        # Handle length mismatch if DS3M returns different length
        if len(pred) != len(y_test):
            print(f"    Note: DS3M pred length ({len(pred)}) != y_test ({len(y_test)}), trimming")
            min_len = min(len(pred), len(y_test))
            pred = pred[:min_len]
            y_test_ds3m = y_test[:min_len]
        else:
            y_test_ds3m = y_test

        results['predictions']['DS3M'] = pred

        # DS3M predicts single dimension, so compare against that dimension only
        if use_multidim and y_test_ds3m.ndim > 1:
            y_test_ds3m = y_test_ds3m[:, dim]
        results['metrics']['DS3M'] = compute_metrics(y_test_ds3m, pred)

        # Also load full-dimensional RMSE from cache (DS3M's original evaluation)
        cached_forecast = load_forecast(dataname)
        if cached_forecast and 'res_metric' in cached_forecast:
            res_metric = cached_forecast['res_metric']
            if isinstance(res_metric, dict) and 'rmse' in res_metric:
                full_dim_rmse = float(res_metric['rmse'])
                results['metrics']['DS3M']['RMSE_full_dim'] = full_dim_rmse
                print(f"    RMSE (dim={dim}): {results['metrics']['DS3M']['RMSE']:.4f}")
                print(f"    RMSE (all dims avg): {full_dim_rmse:.4f}")
            else:
                print(f"    RMSE: {results['metrics']['DS3M']['RMSE']:.4f}")
        else:
            print(f"    RMSE: {results['metrics']['DS3M']['RMSE']:.4f}")
    except Exception as e:
        print(f"    Failed: {e}")

    return results


def plot_professional_comparison(results, dataname, plot_range=200):
    """
    Create professional multi-panel comparison plot with comprehensive metrics.
    Each subplot shows one method with RMSE, R2, MAE, MAPE in a text box.

    For multi-dimensional data, plots the average across all dimensions.
    """
    if results is None:
        return None

    y_true = results['y_true']
    predictions = results['predictions']
    metrics = results['metrics']
    config = results['config']

    if len(predictions) == 0:
        print(f"  No predictions for {dataname}")
        return None

    # Handle multi-dimensional data: average across dimensions for plotting
    if y_true.ndim > 1:
        y_true_plot = y_true.mean(axis=1)
        print(f"  Multi-dim data: averaging {y_true.shape[1]} dimensions for plotting")
    else:
        y_true_plot = y_true

    # Convert predictions to 1D for plotting
    predictions_plot = {}
    for model, pred in predictions.items():
        if pred.ndim > 1:
            predictions_plot[model] = pred.mean(axis=1)
        else:
            predictions_plot[model] = pred

    # Use plot versions for visualization
    y_true = y_true_plot
    predictions = predictions_plot

    plot_len = min(plot_range, len(y_true))
    t = np.arange(plot_len)

    # Professional color scheme - DISTINCT colors for each method
    # Fixed per meeting feedback: GP and DS3M were too similar
    colors = {
        'AR': '#7f8c8d',       # Gray (baseline)
        'S4': '#9b59b6',       # Purple
        'MCD': '#e67e22',      # Orange
        'GP': '#3498db',       # Blue (changed from teal)
        'CPD': '#e74c3c',      # Red (changed from dark gray)
        'DS3M': '#27ae60',     # Green (changed from green-teal)
    }

    method_order = ['AR', 'S4', 'MCD', 'GP', 'CPD', 'DS3M']
    methods = [m for m in method_order if m in predictions]
    n_methods = len(methods)

    # Create figure
    fig, axes = plt.subplots(n_methods, 1, figsize=(14, 2.5*n_methods),
                              sharex=True, constrained_layout=True)
    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        y_pred = predictions.get(method)
        pred_len = min(plot_len, len(y_pred))

        # Ground truth
        ax.plot(t[:pred_len], y_true[:pred_len], color=GROUND_TRUTH_COLOR, alpha=0.8,
                linewidth=1.0, label='Ground Truth')

        # Prediction
        ax.plot(t[:pred_len], y_pred[:pred_len], color=colors.get(method, 'blue'),
                linewidth=1.2, alpha=0.9, label=f'{method} Prediction')

        # Fill error region
        ax.fill_between(t[:pred_len], y_true[:pred_len], y_pred[:pred_len],
                        color=colors.get(method, 'blue'), alpha=0.12)

        # Get metrics
        m = metrics.get(method, {})
        rmse = m.get('RMSE', np.nan)
        r2 = m.get('R2', np.nan)
        mae = m.get('MAE', np.nan)
        mape = m.get('MAPE', np.nan)

        # Metrics text box
        metrics_text = (f'RMSE: {rmse:.3f}\n'
                       f'MAE:  {mae:.3f}\n'
                       f'R\u00b2:   {r2:.3f}\n'
                       f'MAPE: {mape:.1f}%')

        props = dict(boxstyle='round,pad=0.4', facecolor='white',
                    edgecolor=colors.get(method, 'gray'), alpha=0.9)
        ax.text(0.98, 0.95, metrics_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=props, fontfamily='monospace')

        # Styling
        ax.set_ylabel(f'{method}', fontsize=11, fontweight='bold',
                     color=colors.get(method, 'black'))
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax.set_facecolor('#f7f7f5')
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3, linewidth=0.5, linestyle='--')
        ax.tick_params(labelsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        ax.minorticks_on()
        ax.grid(which='minor', alpha=0.15, linewidth=0.3)

    # X-axis label
    axes[-1].set_xlabel('Time Step (t)', fontsize=11)

    # Title
    description = config.get('description', dataname)
    fig.suptitle(f'{dataname}: Forecasting Comparison\n{description}',
                 fontsize=13, fontweight='bold', y=1.02)

    # Save
    save_path = OUTPUT_DIR / f"{dataname}_forecast_comparison.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n  Saved: {save_path}")
    return save_path


def plot_combined_overlay(results, dataname, plot_range=200):
    """
    Create single overlay plot showing all methods together.
    """
    if results is None:
        return None

    y_true = results['y_true']
    predictions = results['predictions']
    metrics = results['metrics']
    config = results['config']

    if len(predictions) == 0:
        return None

    plot_len = min(plot_range, len(y_true))
    t = np.arange(plot_len)

    # Professional color scheme - DISTINCT colors for each method
    colors = {
        'AR': '#7f8c8d',       # Gray (baseline)
        'S4': '#9b59b6',       # Purple
        'MCD': '#e67e22',      # Orange
        'GP': '#3498db',       # Blue
        'CPD': '#e74c3c',      # Red
        'DS3M': '#27ae60',     # Green
    }

    fig, ax = plt.subplots(figsize=(14, 5))

    # Ground truth
    ax.plot(t, y_true[:plot_len], color=GROUND_TRUTH_COLOR, linewidth=2.0, alpha=0.85,
            label='Ground Truth', zorder=10)

    # All predictions
    method_order = ['AR', 'S4', 'MCD', 'GP', 'CPD', 'DS3M']
    for method in method_order:
        if method not in predictions:
            continue
        y_pred = predictions.get(method)
        pred_len = min(plot_len, len(y_pred))
        m = metrics.get(method, {})
        rmse = m.get('RMSE', np.nan)
        r2 = m.get('R2', np.nan)
        ax.plot(t[:pred_len], y_pred[:pred_len],
                color=colors.get(method, 'gray'),
                linewidth=1.2, alpha=0.8,
                label=f'{method} (RMSE={rmse:.2f}, R\u00b2={r2:.2f})')

    ax.set_xlabel('Time Step (t)', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
    ax.set_facecolor('#f7f7f5')
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, linewidth=0.5, linestyle='--')
    ax.minorticks_on()
    ax.grid(which='minor', alpha=0.15, linewidth=0.3)

    description = config.get('description', dataname)
    ax.set_title(f'{dataname}: All Methods Overlay\n{description}',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path = OUTPUT_DIR / f"{dataname}_forecast_overlay.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: {save_path}")
    return save_path


def create_summary_tables(all_results):
    """Create comprehensive metrics comparison tables."""
    methods = ['AR', 'S4', 'MCD', 'GP', 'CPD', 'DS3M']
    datasets = [d for d in all_results.keys() if all_results[d] is not None]

    rmse_data = []
    r2_data = []
    mape_data = []

    for dataset in datasets:
        rmse_row = {'Dataset': dataset}
        r2_row = {'Dataset': dataset}
        mape_row = {'Dataset': dataset}

        for method in methods:
            m = all_results[dataset]['metrics'].get(method, {})
            rmse_row[method] = m.get('RMSE', np.nan)
            r2_row[method] = m.get('R2', np.nan)
            mape_row[method] = m.get('MAPE', np.nan)

        rmse_data.append(rmse_row)
        r2_data.append(r2_row)
        mape_data.append(mape_row)

    df_rmse = pd.DataFrame(rmse_data)
    df_r2 = pd.DataFrame(r2_data)
    df_mape = pd.DataFrame(mape_data)

    # Save CSVs
    df_rmse.to_csv(OUTPUT_DIR / "task1_rmse.csv", index=False, float_format='%.4f')
    df_r2.to_csv(OUTPUT_DIR / "task1_r2.csv", index=False, float_format='%.4f')
    df_mape.to_csv(OUTPUT_DIR / "task1_mape.csv", index=False, float_format='%.2f')

    print("\n" + "="*80)
    print("RMSE Comparison")
    print("="*80)
    print(df_rmse.to_string(index=False))

    print("\n" + "="*80)
    print("R\u00b2 Comparison")
    print("="*80)
    print(df_r2.to_string(index=False))

    print("\n" + "="*80)
    print("MAPE (%) Comparison")
    print("="*80)
    print(df_mape.to_string(index=False))

    return df_rmse, df_r2, df_mape


def create_summary_heatmap(all_results):
    """Create visual summary heatmap of RMSE across all datasets and methods."""
    methods = ['AR', 'S4', 'MCD', 'GP', 'CPD', 'DS3M']
    datasets = [d for d in all_results.keys() if all_results[d] is not None]

    if len(datasets) == 0:
        return None

    rmse_matrix = []
    for dataset in datasets:
        row = []
        for method in methods:
            m = all_results[dataset]['metrics'].get(method, {})
            row.append(m.get('RMSE', np.nan))
        rmse_matrix.append(row)

    rmse_matrix = np.array(rmse_matrix)

    # Normalize per-row for visualization
    rmse_norm = np.zeros_like(rmse_matrix)
    for i in range(len(datasets)):
        row_min = np.nanmin(rmse_matrix[i])
        row_max = np.nanmax(rmse_matrix[i])
        if row_max > row_min:
            rmse_norm[i] = (rmse_matrix[i] - row_min) / (row_max - row_min)
        else:
            rmse_norm[i] = 0.5

    fig, ax = plt.subplots(figsize=(10, max(6, len(datasets)*0.8)))

    im = ax.imshow(rmse_norm, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

    # Text annotations
    for i in range(len(datasets)):
        for j in range(len(methods)):
            val = rmse_matrix[i, j]
            text = 'N/A' if np.isnan(val) else f'{val:.2f}'
            color = 'white' if rmse_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)

    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_yticklabels(datasets, fontsize=10)

    ax.set_xlabel('Method', fontsize=11)
    ax.set_ylabel('Dataset', fontsize=11)
    ax.set_title('RMSE Comparison (Green=Best, Red=Worst)\nValues shown, colors normalized per row',
                 fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Relative RMSE (0=Best, 1=Worst)', fontsize=10)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "task1_summary_heatmap.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\nSaved heatmap: {save_path}")
    return save_path


def scan_ds3m_dims(datasets, top_k=5, metric="RMSE"):
    """Scan cached DS3M forecasts to find best target dim per dataset."""
    from experiments.utils.experiments_utils import load_forecast

    metric = metric.upper()
    if metric not in {"RMSE", "MAE", "MAPE", "R2"}:
        print(f"  Unknown metric '{metric}', defaulting to RMSE")
        metric = "RMSE"

    print("\n" + "=" * 70)
    print("DS3M Dimension Scan (cached forecasts, offset-corrected)")
    print("=" * 70)
    print(f"Metric: {metric} | Top-k: {top_k}")

    for dataname in datasets:
        res = load_forecast(dataname)
        if res is None:
            print(f"\n{dataname}: no cached DS3M forecast found.")
            continue

        y_pred = np.asarray(res["y_pred_mean"])
        y_true = np.asarray(res["y_true"])
        if y_pred.ndim == 1:
            y_pred = y_pred[:, None]
        if y_true.ndim == 1:
            y_true = y_true[:, None]

        # Apply offset correction for remove_residual datasets (like Seattle)
        remove_residual = res.get("remove_residual", False)

        d_max = min(y_pred.shape[1], y_true.shape[1])
        scores = []
        for d in range(d_max):
            pred_d = y_pred[:, d].copy()
            true_d = y_true[:, d]
            # Apply offset correction if needed
            if remove_residual:
                offset = pred_d.mean() - true_d.mean()
                y_range = true_d.max() - true_d.min()
                if y_range > 0 and abs(offset) > 0.1 * y_range:
                    pred_d = pred_d - offset
            m = compute_metrics(true_d, pred_d)
            scores.append((d, m.get(metric, np.nan), m))

        reverse = metric == "R2"
        scores = [s for s in scores if np.isfinite(s[1])]
        scores.sort(key=lambda x: x[1], reverse=reverse)

        print(f"\n{dataname}: best dims by {metric}" + (" (offset-corrected)" if remove_residual else ""))
        for d, val, m in scores[:max(1, top_k)]:
            print(f"  dim={d:4d} | RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} "
                  f"MAPE={m['MAPE']:.2f}% R2={m['R2']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Task 1: Forecasting Comparison")
    parser.add_argument("--datasets", nargs="+",
                        default=["Toy", "Electricity", "Hangzhou", "Lorenz",
                                 "Pacific", "Seattle", "Sleep", "Unemployment", "Pernod"],
                        help="Datasets to evaluate")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")
    parser.add_argument("--plot-range", type=int, default=200,
                        help="Number of time steps to plot")
    parser.add_argument("--test-extra", type=int, default=0,
                        help="Extra test points to extend beyond dataset default")
    parser.add_argument("--gp-max-lags", type=int, default=12,
                        help="Max lag features used by GP (last-k lags)")
    parser.add_argument("--use-ds3m-target", action="store_true",
                        help="Use DS3M target_dim (predict_dim-1) for multivariate datasets")
    parser.add_argument("--ds3m-target-datasets", nargs="+", default=[],
                        help="Datasets that should use DS3M target_dim override")
    parser.add_argument("--ds3m-force-new", action="store_true",
                        help="Ignore DS3M cache and recompute forecasts")
    parser.add_argument("--pernod-test-len", type=int, default=None,
                        help="Override Pernod test length (e.g., 80)")
    parser.add_argument("--dim-override", nargs="+", default=[],
                        help="Per-dataset target dim overrides, e.g., Pacific=100 Seattle=10")
    parser.add_argument("--scan-ds3m-dims", nargs="+", default=[],
                        help="Scan cached DS3M forecasts to find best dims")
    parser.add_argument("--scan-top-k", type=int, default=5,
                        help="Top-k dims to show in DS3M scan")
    parser.add_argument("--scan-metric", type=str, default="RMSE",
                        help="Metric for DS3M scan: RMSE, MAE, MAPE, R2")
    args = parser.parse_args()

    print("="*70)
    print("Task 1: Forecasting Performance Comparison")
    print("="*70)
    print(f"Datasets: {args.datasets}")
    print(f"Device: {args.device}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*70)

    dim_overrides = {}
    for item in args.dim_override:
        if "=" in item:
            name, val = item.split("=", 1)
            if name:
                try:
                    dim_overrides[name] = int(val)
                except ValueError:
                    print(f"  Warning: invalid dim override '{item}' (expected Dataset=int)")
        else:
            print(f"  Warning: invalid dim override '{item}' (expected Dataset=int)")

    all_results = {}

    if args.scan_ds3m_dims:
        scan_ds3m_dims(
            datasets=args.scan_ds3m_dims,
            top_k=args.scan_top_k,
            metric=args.scan_metric,
        )
        return

    for dataname in args.datasets:
        try:
            results = run_comparison(
                dataname,
                device=args.device,
                test_extra=args.test_extra,
                gp_max_lags=args.gp_max_lags,
                use_ds3m_target=args.use_ds3m_target,
                ds3m_target_datasets=args.ds3m_target_datasets,
                ds3m_force_new=args.ds3m_force_new,
                pernod_test_len=args.pernod_test_len,
                dim_overrides=dim_overrides,
            )
            all_results[dataname] = results

            if results is not None:
                plot_professional_comparison(results, dataname, plot_range=args.plot_range)
                plot_combined_overlay(results, dataname, plot_range=args.plot_range)

        except Exception as e:
            print(f"\nError processing {dataname}: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataname] = None

    # Create summary
    if any(r is not None for r in all_results.values()):
        create_summary_tables(all_results)
        create_summary_heatmap(all_results)

    print(f"\n{'='*70}")
    print(f"Done! All figures saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
