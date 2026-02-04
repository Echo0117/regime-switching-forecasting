"""
Diagnostic script for DS3M performance on Toy data.

This addresses Charles's concern from the meeting:
"if the toy is the one generated with DS3M and it doesn't work best on DS3M,
we're gonna lose a lot of readers"

The script:
1. Loads Toy data and trained DS3M model
2. Compares DS3M with AR baseline on same data
3. Analyzes potential issues (data alignment, model convergence, etc.)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

# Add project root to path
HERE = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


def load_toy_data(data_dir="Toy_exp1"):
    """Load Toy data directly from CSV files."""
    base_path = Path(f"Deep_Switching_State_Space_Model/data/{data_dir}")

    # Load raw data
    y = np.loadtxt(base_path / "simulation_data_nonlinear_y.csv")
    d = np.loadtxt(base_path / "simulation_data_nonlinear_d.csv")
    z = np.loadtxt(base_path / "simulation_data_nonlinear_z.csv")

    return {
        'y': y,
        'd': d,  # Ground truth regime labels
        'z': z,  # Latent variable
        'data_dir': str(base_path)
    }


def create_lag_features(data, lags):
    """Create lag features for AR model."""
    X = np.zeros((len(data) - lags, lags))
    y = np.zeros(len(data) - lags)
    for i in range(lags, len(data)):
        X[i - lags] = data[i - lags:i]
        y[i - lags] = data[i]
    return X, y


def run_ar_baseline(y, lags=20, test_len=500):
    """Run AR baseline on Toy data."""
    X, y_target = create_lag_features(y, lags)

    train_end = len(y_target) - test_len
    X_train, y_train = X[:train_end], y_target[:train_end]
    X_test, y_test = X[train_end:], y_target[train_end:]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        'y_pred': y_pred,
        'y_test': y_test,
        'rmse': rmse,
        'r2': r2,
        'model': model
    }


def run_ds3m_forecast(data_dir="Toy_exp1"):
    """Run DS3M forecast on Toy data."""
    from experiments.utils.ds3m_utils import load_ds3m_data, load_ds3m_model, forecast

    # Create args object
    class Args:
        def __init__(self, problem, data_dir=None):
            self.problem = problem
            self.data_dir = data_dir
            self.seed = 42
            self.train_size = 100

    args = Args("Toy", f"Deep_Switching_State_Space_Model/data/{data_dir}")

    print(f"Loading DS3M data from: {args.data_dir}")
    ds = load_ds3m_data(args)

    print(f"\nDS3M data loaded:")
    print(f"  trainX shape: {ds['trainX'].shape}")
    print(f"  testX shape: {ds['testX'].shape}")
    print(f"  test_len: {ds['test_len']}")
    print(f"  timestep: {ds['timestep']}")

    # Load model
    model = load_ds3m_model(
        ds["directoryBest"],
        ds["x_dim"], ds["y_dim"], ds["h_dim"], ds["z_dim"],
        ds["d_dim"], ds["n_layers"], ds["learning_rate"],
        ds["device"], bidirection=ds.get("bidirection", False),
    )

    # Run forecast
    res, testForecast_mean, testOriginal, size, d_argmax, uq, lq = forecast(
        model,
        ds["testX"], ds["testY"],
        ds["moments"], ds["d_dim"],
        ds["means"], ds["trend"],
        ds["test_len"], ds["freq"],
        ds["RawDataOriginal"],
        remove_mean=ds.get("remove_mean", False),
        remove_residual=ds.get("remove_residual", False),
    )

    # Extract predictions
    y_pred = testForecast_mean.flatten()
    y_true = testOriginal.flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        'y_pred': y_pred,
        'y_true': y_true,
        'rmse': rmse,
        'r2': r2,
        'd_argmax': d_argmax,
        'ds': ds,
        'metrics': res
    }


def diagnose_toy():
    """Run full diagnostic on Toy data."""
    print("="*70)
    print("DS3M on Toy Data Diagnostic")
    print("="*70)

    # List available Toy datasets
    data_path = Path("Deep_Switching_State_Space_Model/data")
    toy_dirs = sorted([d.name for d in data_path.iterdir() if d.name.startswith("Toy")])

    print(f"\nAvailable Toy datasets: {len(toy_dirs)}")
    for d in toy_dirs[:10]:
        print(f"  - {d}")
    if len(toy_dirs) > 10:
        print(f"  ... and {len(toy_dirs) - 10} more")

    # Test on Toy_exp1 (main dataset)
    data_dir = "Toy_exp1"
    print(f"\n{'='*70}")
    print(f"Testing on: {data_dir}")
    print(f"{'='*70}")

    # Load raw data
    toy_data = load_toy_data(data_dir)
    y = toy_data['y']
    d_true = toy_data['d']

    print(f"\nRaw data info:")
    print(f"  y length: {len(y)}")
    print(f"  y range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  Unique regimes in d: {np.unique(d_true)}")
    print(f"  Regime distribution: {np.bincount(d_true.astype(int))}")

    # Run AR baseline
    print("\n" + "-"*70)
    print("1. AR Baseline")
    print("-"*70)

    ar_results = run_ar_baseline(y, lags=20, test_len=500)
    print(f"  RMSE: {ar_results['rmse']:.4f}")
    print(f"  R²:   {ar_results['r2']:.4f}")

    # Run DS3M
    print("\n" + "-"*70)
    print("2. DS3M")
    print("-"*70)

    try:
        ds3m_results = run_ds3m_forecast(data_dir)
        print(f"  RMSE: {ds3m_results['rmse']:.4f}")
        print(f"  R²:   {ds3m_results['r2']:.4f}")
        print(f"  Original metrics from DS3M: {ds3m_results['metrics']}")

        # Regime detection accuracy
        test_len = len(ds3m_results['y_pred'])
        d_pred = ds3m_results['d_argmax']
        d_test = d_true[-test_len:].astype(int)

        if len(d_pred) == len(d_test):
            regime_acc = np.mean(d_pred == d_test)
            # Check if we need to flip (0 and 1 might be swapped)
            regime_acc_flipped = np.mean((1 - d_pred) == d_test)
            regime_acc = max(regime_acc, regime_acc_flipped)
            print(f"  Regime detection accuracy: {regime_acc:.3f}")

    except Exception as e:
        print(f"  DS3M failed: {e}")
        import traceback
        traceback.print_exc()
        ds3m_results = None

    # Compare
    print("\n" + "-"*70)
    print("3. Comparison")
    print("-"*70)

    if ds3m_results:
        ar_rmse = ar_results['rmse']
        ds3m_rmse = ds3m_results['rmse']

        if ds3m_rmse < ar_rmse:
            print(f"  ✅ DS3M is BETTER (RMSE: {ds3m_rmse:.4f} vs AR: {ar_rmse:.4f})")
            improvement = (ar_rmse - ds3m_rmse) / ar_rmse * 100
            print(f"     Improvement: {improvement:.1f}%")
        else:
            print(f"  ❌ AR is BETTER (RMSE: {ar_rmse:.4f} vs DS3M: {ds3m_rmse:.4f})")
            print(f"\n  INVESTIGATION NEEDED:")
            print(f"  - Check if DS3M model converged properly")
            print(f"  - Verify data alignment between AR and DS3M")
            print(f"  - Consider retraining DS3M with more epochs")

    # Create diagnostic plot
    output_dir = Path("figures/ds3m_toy_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    plot_len = min(200, len(ar_results['y_test']))
    t = np.arange(plot_len)

    # Plot 1: Full time series with regimes
    ax = axes[0]
    ax.plot(y, 'b-', alpha=0.7, label='Full time series')

    # Color by regime
    for regime in np.unique(d_true):
        mask = d_true == regime
        ax.scatter(np.where(mask)[0], y[mask], c=f'C{int(regime)}',
                  s=5, alpha=0.5, label=f'Regime {int(regime)}')

    ax.axvline(len(y) - 500, color='red', linestyle='--', label='Test start')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(f'Toy Data: Full Time Series ({data_dir})')
    ax.legend(loc='upper right')

    # Plot 2: AR predictions
    ax = axes[1]
    ax.plot(t, ar_results['y_test'][:plot_len], 'k-', alpha=0.7, linewidth=1.5, label='Ground Truth')
    ax.plot(t, ar_results['y_pred'][:plot_len], 'b-', alpha=0.7, linewidth=1.2, label='AR Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(f'AR Baseline (RMSE: {ar_results["rmse"]:.4f}, R²: {ar_results["r2"]:.4f})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 3: DS3M predictions (if available)
    ax = axes[2]
    if ds3m_results:
        y_true_ds3m = ds3m_results['y_true'][:plot_len]
        y_pred_ds3m = ds3m_results['y_pred'][:plot_len]

        ax.plot(t[:len(y_true_ds3m)], y_true_ds3m, 'k-', alpha=0.7, linewidth=1.5, label='Ground Truth')
        ax.plot(t[:len(y_pred_ds3m)], y_pred_ds3m, 'g-', alpha=0.7, linewidth=1.2, label='DS3M Prediction')
        ax.set_title(f'DS3M (RMSE: {ds3m_results["rmse"]:.4f}, R²: {ds3m_results["r2"]:.4f})')
    else:
        ax.text(0.5, 0.5, 'DS3M failed to load', transform=ax.transAxes,
               ha='center', va='center', fontsize=14)
        ax.set_title('DS3M (Failed)')

    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / f"{data_dir}_diagnosis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nDiagnostic plot saved to: {save_path}")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("""
If DS3M performs worse than AR on Toy data:

1. **Check model training**:
   - Verify the checkpoint exists and loaded correctly
   - May need to retrain with more epochs
   - Check if training converged (validation loss decreasing)

2. **Check data alignment**:
   - Ensure Toy data format matches what DS3M expects
   - Verify train/test split is consistent

3. **Consider the data generation**:
   - Toy data may be generated with AR dynamics, not switching dynamics
   - If Toy data is basically AR, then AR should perform similarly or better
   - Check if the Toy data actually has regime switches

4. **For the paper**:
   - Consider using Toy data with more pronounced regime switches
   - Use variance ratio > 4:1 (e.g., 0.5 vs 2.0 or 0.5 vs 4.0)
   - Show that DS3M excels when regime switches are clear
""")

    return ar_results, ds3m_results


if __name__ == "__main__":
    ar_results, ds3m_results = diagnose_toy()
