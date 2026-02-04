"""
Test script to verify AgACI implementation and generate all analysis plots.

This script:
1. Runs AgACI on a dataset
2. Generates weight analysis plots aligned to regime switches
3. Generates coverage analysis plots
4. Generates coverage vs length tradeoff plots

Usage:
    python experiments/test_agaci.py --problem Electricity
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
PROJ = os.path.abspath(os.path.join(HERE, ".."))
for p in [HERE, PROJ]:
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.utils.plot_utils import plot_results_with_aci
from experiments.utils.acp_utils import aci_intervals, agaci_intervals, run_other_cp_methods
from experiments.utils.ds3m_utils import load_ds3m_data, load_ds3m_model, forecast, get_full_d_argmax, evaluation

# Import ruptures only if available (optional dependency)
try:
    from experiments.utils.ruptures_utils import ruptures_forecast_sequence
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    ruptures_forecast_sequence = None

from experiments.utils.regime_switch_analysis import (
    plot_agaci_weights_at_switches,
    plot_coverage_at_switches,
    plot_coverage_vs_length_tradeoff,
    plot_regime_heatmap_full,
    plot_individual_switch_trajectories,
    plot_length_at_switches,
    compute_recovery_metrics,
    plot_recovery_comparison,
    load_timestamps_for_dataset
)
from experiments.utils.lag_compensation import apply_lag_compensation, print_lag_info, get_window


def load_ground_truth_regimes_toy(custom_regime_path=None, data_dir=None):
    """Load ground truth regime labels for Toy data from CSV.

    Args:
        custom_regime_path: Optional custom path to regime CSV file
        data_dir: Optional data directory path

    Returns:
        np.ndarray: Ground truth regime labels (full dataset: 2001 points for d, 2000 for y)
    """
    from pathlib import Path

    if custom_regime_path is not None:
        d_csv_path = Path(custom_regime_path)
        print(f"Using custom regime file: {d_csv_path}")
    elif data_dir is not None:
        d_csv_path = Path(data_dir) / "simulation_data_nonlinear_d.csv"
    else:
        toy_data_dir = Path("Deep_Switching_State_Space_Model/data/Toy")
        d_csv_path = toy_data_dir / "simulation_data_nonlinear_d.csv"

    if not d_csv_path.exists():
        raise FileNotFoundError(
            f"Ground truth regime file not found: {d_csv_path}\n"
            "Oracle switches are only available for Toy data with single-switch test set."
        )

    d_true = pd.read_csv(d_csv_path, header=None).values.flatten()
    print(f"✓ Loaded ground truth regimes from {d_csv_path}")
    print(f"  - Total length: {len(d_true)}")
    print(f"  - Unique regimes: {np.unique(d_true)}")
    print(f"  - Number of switches: {np.sum(np.diff(d_true) != 0)}")

    return d_true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="Electricity",
                    choices=["Toy", "Lorenz", "Sleep", "Unemployment", "Hangzhou", "Seattle", "Pacific", "Electricity", "Pernod"])
    ap.add_argument("--aci_train_size", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--tab-gamma", type=float, nargs="*", default=[0.001, 0.01, 0.02, 0.5, 0.99])
    ap.add_argument("--agaci-eta", type=float, default=0.5, help="Learning rate for AgACI BOA (increased from 0.1 to 0.5 for faster adaptation)")
    ap.add_argument("--agaci-lr-schedule", type=str, default="constant",
                    choices=["constant", "sqrt", "log", "poly025", "poly06"],
                    help="Learning rate schedule for AgACI BOA (default: constant, best for regime-switching)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d-dim", type=int, default=2, help="Number of regimes (default=2 for simpler experiments)")
    ap.add_argument("--use-oracle-switches", action="store_true", default=False,
                    help="Use ground truth (oracle) regime switches for Toy data (default: False, uses model predictions)")
    ap.add_argument("--data-dir", type=str, default=None,
                    help="Custom data directory (e.g., 'Deep_Switching_State_Space_Model/data/Toy_exp2_V1_0.5_V2_1.0')")
    ap.add_argument("--model-idx", type=int, default=None,
                    help="Specific model index to use (for multi-model processing)")
    ap.add_argument("--save-coverage-csv", action="store_true", default=False,
                    help="Save coverage/length data to CSV for aggregation")
    ap.add_argument("--use-lag-compensation", action="store_true", default=False,
                    help="Apply lag compensation to align coverage with regime switches")

    # Ruptures parameters (plug-in alternative to DS3M)
    ap.add_argument("--regime-method", type=str, default="ds3m",
                    choices=["ds3m", "ruptures"],
                    help="Regime detection method: 'ds3m' (deep switching model) or 'ruptures' (changepoint detection)")
    ap.add_argument("--ruptures-method", type=str, default="Pelt",
                    choices=["Pelt", "Binseg", "BottomUp", "Window"],
                    help="Ruptures detection algorithm (only used if --regime-method=ruptures)")
    ap.add_argument("--ruptures-model", type=str, default="rbf",
                    choices=["l1", "l2", "rbf", "normal", "ar"],
                    help="Ruptures cost function (only used if --regime-method=ruptures)")
    ap.add_argument("--ruptures-min-size", type=int, default=10,
                    help="Minimum segment size for ruptures (only used if --regime-method=ruptures)")
    ap.add_argument("--ruptures-penalty", type=float, default=None,
                    help="Penalty for changepoint detection (auto if None, only used if --regime-method=ruptures)")
    ap.add_argument("--ruptures-forecast-method", type=str, default="ar",
                    choices=["ar", "mean", "median", "last", "gru", "s4", "linear"],
                    help="Forecasting method for ruptures (only used if --regime-method=ruptures)")
    ap.add_argument("--ruptures-ar-lag", type=int, default=5,
                    help="AR lag for ruptures forecasting (only used if --regime-method=ruptures)")

    args = ap.parse_args()
    np.random.seed(args.seed)

    # Determine dataset identifier from data_dir if provided
    if hasattr(args, 'data_dir') and args.data_dir is not None:
        dataset_id = os.path.basename(args.data_dir)
    else:
        dataset_id = args.problem

    print(f"\n{'='*60}")
    print(f"Testing AgACI on {args.problem}")
    print(f"Regime detection: {args.regime_method.upper()}")
    if args.regime_method == "ruptures":
        print(f"Ruptures method: {args.ruptures_method} ({args.ruptures_model})")
    else:
        print(f"d_dim: {args.d_dim}")
    if args.data_dir:
        print(f"Using custom data: {dataset_id}")
    if args.model_idx is not None:
        print(f"Using specific model: model_{args.model_idx}")
    print(f"{'='*60}\n")

    # ==================================================================================
    # BRANCH: DS3M or RUPTURES
    # ==================================================================================

    if args.regime_method == "ruptures":
        # ===== RUPTURES PIPELINE =====
        if not RUPTURES_AVAILABLE:
            print(f"\n{'='*60}")
            print("ERROR: Ruptures library not installed")
            print(f"{'='*60}")
            print("Please install ruptures: pip install ruptures")
            print("Or use DS3M method instead: --regime-method ds3m")
            print(f"{'='*60}\n")
            sys.exit(1)

        print(f"{'='*60}")
        print("RUPTURES REGIME DETECTION + FORECASTING")
        print(f"{'='*60}\n")

        # Load DS3M data structure (for data loading only, not for model)
        ds = load_ds3m_data(args)

        # Extract data
        data_full = np.asarray(ds["data"])
        target_dim = int(ds.get("target_dim", 0))
        if data_full.ndim > 1:
            data_full = data_full[:, target_dim]

        test_len = int(ds["test_len"])
        train_size = len(data_full) - test_len

        print(f"Data: Total={len(data_full)}, Train={train_size}, Test={test_len}")

        # Normalize for ruptures
        mean = np.mean(data_full[:train_size])
        std = np.std(data_full[:train_size])
        data_norm = (data_full - mean) / std

        # Run ruptures
        print(f"\nRunning ruptures ({args.ruptures_method})...")
        print(f"  Forecast method: {args.ruptures_forecast_method}")
        print(f"  AR lag: {args.ruptures_ar_lag}")

        # Setup model kwargs for neural models
        forecast_model_kwargs = {}
        if args.ruptures_forecast_method in ["gru", "s4"]:
            import torch
            forecast_model_kwargs = {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'epochs': 30,
                'patience': 5,
                'hidden': 64,  # GRU hidden size
                'd_model': 64,  # S4 model dimension
                'batch': 32,
                'verbose': False
            }
            print(f"  Using device: {forecast_model_kwargs['device']}")

        ruptures_results = ruptures_forecast_sequence(
            data_norm,
            train_size=train_size,
            test_size=test_len,
            method=args.ruptures_method,
            model=args.ruptures_model,
            min_size=args.ruptures_min_size,
            penalty=args.ruptures_penalty,
            forecast_method=args.ruptures_forecast_method,
            ar_lag=args.ruptures_ar_lag,
            forecast_model_kwargs=forecast_model_kwargs
        )

        # Denormalize
        testForecast_mean_raw = ruptures_results['forecasts'] * std + mean
        lq = ruptures_results['lower_quantiles'] * std + mean
        uq = ruptures_results['upper_quantiles'] * std + mean

        # Get test data and regime labels
        testOriginal = data_full[train_size:train_size + test_len]
        d_argmax_test = ruptures_results['regime_labels_test']
        d_argmax_full = ruptures_results['regime_labels_full']

        # Compute RMSE
        rmse = np.sqrt(np.mean((testForecast_mean_raw - testOriginal) ** 2))
        res = {'rmse': rmse, 'mape': np.mean(np.abs((testOriginal - testForecast_mean_raw) / testOriginal)) * 100}

        print(f"  Detected regimes: {ruptures_results['n_regimes']}")
        print(f"  Switches in test: {np.sum(np.diff(d_argmax_test) != 0)}")
        print(f"  RMSE: {rmse:.4f}")

        # Format for compatibility with rest of code
        testForecast_mean = testForecast_mean_raw.reshape(-1, 1)
        testOriginal = testOriginal.reshape(-1, 1)
        size = len(testOriginal)

        # Store in ds for conformal prediction
        ds["d_dim"] = ruptures_results['n_regimes']
        ds["moments"] = None  # Not used for ruptures

    else:
        # ===== DS3M PIPELINE =====
        print(f"{'='*60}")
        print("DS3M REGIME DETECTION + FORECASTING")
        print(f"{'='*60}\n")

        # Load DS3M data and forecasts
        ds = load_ds3m_data(args)

        # Override d_dim for simpler experiments
        ds["d_dim"] = args.d_dim

        # Load specific model if model-idx is provided
        if args.model_idx is not None:
            print(f"\n{'='*60}")
            print(f"Loading specific model: model_{args.model_idx}")
            print(f"{'='*60}\n")

            import torch
            import sys as _sys
            _sys.path.insert(0, 'Deep_Switching_State_Space_Model/src')

            # Load the specific model
            model_path = os.path.join(ds["directoryBest"], f"best_model{args.model_idx}.tar")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            from DSSSMCode import DSSSM
            model = DSSSM(
                ds["x_dim"], ds["y_dim"], ds["h_dim"], ds["z_dim"],
                ds["d_dim"], ds["n_layers"], ds["device"], ds["bidirection"]
            ).to(ds["device"])

            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"  Loaded from: {model_path}")

            # Get forecasts
            res, testForecast_mean, testOriginal, size, d_argmax_test, uq, lq = forecast(
                model,
                ds["testX"], ds["testY"],
                ds["moments"], ds["d_dim"],
                ds["means"], ds["trend"],
                ds["test_len"], ds["freq"],
                ds["RawDataOriginal"],
                remove_mean=ds["remove_mean"],
                remove_residual=ds["remove_residual"],
            )

            print(f"  Model {args.model_idx} RMSE: {res['rmse']:.4f}")
            print(f"  Model {args.model_idx} MAPE: {res['mape']:.4f}")

        else:
            # Single model (original behavior: load best.tar)
            try:
                model = load_ds3m_model(
                    ds["directoryBest"],
                    ds["x_dim"], ds["y_dim"], ds["h_dim"], ds["z_dim"],
                    ds["d_dim"], ds["n_layers"], ds["learning_rate"],
                    ds["device"], bidirection=ds["bidirection"],
                )
            except FileNotFoundError as e:
                print(f"\n❌ Error: Model not found for {dataset_id}")
                print(f"   Checkpoint path: {ds['directoryBest']}/best.tar")
                print(f"\nOptions:")
                print(f"   1. Train DS3M model first:")
                print(f"      cd Deep_Switching_State_Space_Model")
                print(f"      python train.py --dataset {dataset_id}")
                print(f"   2. Use ruptures instead:")
                print(f"      python experiments/test_agaci.py --problem {args.problem} --regime-method ruptures")
                raise

            # Get test portion forecasts for interval construction
            res, testForecast_mean, testOriginal, size, d_argmax_test, uq, lq = forecast(
                model,
                ds["testX"], ds["testY"],
                ds["moments"], ds["d_dim"],
                ds["means"], ds["trend"],
                ds["test_len"], ds["freq"],
                ds["RawDataOriginal"],
                remove_mean=ds["remove_mean"],
                remove_residual=ds["remove_residual"],
            )

        # Get full regime sequence for DS3M
        print("Extracting full regime sequence from model...")
        try:
            d_argmax_full = get_full_d_argmax(model, ds)
            print(f"Successfully extracted full regime sequence (length: {len(d_argmax_full)})")
        except Exception as e:
            print(f"Could not extract full regime sequence from model: {e}")
            print("Using fallback: test regimes padded with regime 0 for training period")
            data_full = np.asarray(ds["data"])
            if data_full.ndim == 1:
                data_full = data_full[:, None]
            N_full = len(data_full)
            test_len_actual = len(d_argmax_test)
            d_argmax_full = np.zeros(N_full, dtype=int)
            d_argmax_full[-test_len_actual:] = d_argmax_test

    # ==================================================================================
    # COMMON PROCESSING (both DS3M and RUPTURES)
    # ==================================================================================

    print(f"\n{'='*60}")
    print(f"PREPARING DATA FOR CONFORMAL PREDICTION")
    print(f"{'='*60}\n")

    # Extract test length (already defined in both branches, but ensure it's set)
    if args.regime_method == "ruptures":
        test_len = len(testOriginal)
    else:
        test_len = int(ds["test_len"])

    # Get target dimension
    target_dim = int(ds.get("target_dim", 0))
    print(f"Target dimension: {target_dim}")
    print(f"Test length: {test_len}")

    # Extract data for target dimension
    # Note: For ruptures, data is already extracted to target_dim in the branch above
    # So we need to check the actual shape before indexing
    y_true = np.asarray(testOriginal)
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        # Multi-dimensional, need to extract target_dim
        y_true = y_true[:, target_dim]
    elif y_true.ndim > 1 and y_true.shape[1] == 1:
        # Already single dimension, just flatten
        y_true = y_true.flatten()

    y_lq_ds3m = np.asarray(lq)
    y_uq_ds3m = np.asarray(uq)
    if y_lq_ds3m.ndim > 1 and y_lq_ds3m.shape[1] > 1:
        y_lq_ds3m = y_lq_ds3m[:, target_dim]
    elif y_lq_ds3m.ndim > 1 and y_lq_ds3m.shape[1] == 1:
        y_lq_ds3m = y_lq_ds3m.flatten()

    if y_uq_ds3m.ndim > 1 and y_uq_ds3m.shape[1] > 1:
        y_uq_ds3m = y_uq_ds3m[:, target_dim]
    elif y_uq_ds3m.ndim > 1 and y_uq_ds3m.shape[1] == 1:
        y_uq_ds3m = y_uq_ds3m.flatten()

    test_len = int(ds["test_len"])

    # Extract test portion of d_argmax from full sequence
    # d_argmax_full has shape (N_full,) where N_full = train + valid + test
    # We want the last test_len points
    d_argmax_test = d_argmax_full[-test_len:]
    print(f"Extracted d_argmax_test from full sequence: {len(d_argmax_test)} points")

    # Debug: print regime distribution
    print(f"\n{'='*60}")
    print("MODEL REGIME DISTRIBUTION IN TEST SET:")
    print(f"{'='*60}")
    unique, counts = np.unique(d_argmax_test, return_counts=True)
    for regime, count in zip(unique, counts):
        print(f"  Regime {regime}: {count}/{len(d_argmax_test)} points ({100*count/len(d_argmax_test):.1f}%)")
    print(f"  Detected switches: {np.sum(np.diff(d_argmax_test) != 0)}")
    print(f"First 30 regimes: {d_argmax_test[:30]}")
    print(f"Last 30 regimes: {d_argmax_test[-30:]}")
    print(f"{'='*60}\n")

    # ---------------------------------------------------
    # Load ground truth regimes for Toy data if using oracle switches
    # ---------------------------------------------------
    d_true_full = None
    d_true_test = None

    if args.use_oracle_switches:
        if args.problem != "Toy":
            print(f"\n{'='*60}")
            print(f"WARNING: --use-oracle-switches is only supported for Toy data.")
            print(f"Current dataset: {args.problem}")
            print(f"Falling back to model-predicted regimes (d_argmax).")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print("USING ORACLE (GROUND TRUTH) REGIME SWITCHES")
            print(f"{'='*60}")
            try:
                d_true_full = load_ground_truth_regimes_toy(data_dir=args.data_dir)

                # Extract test portion of ground truth regimes
                # d_true has 2001 points (includes initial state), y has 2000 points
                # Test set is last test_len points
                # Match d_argmax_test indexing
                d_true_test = d_true_full[-test_len:]

                print(f"\n  - Ground truth test regimes extracted: {len(d_true_test)} points")
                print(f"  - Oracle switches in test set: {np.sum(np.diff(d_true_test) != 0)}")
                print(f"{'='*60}\n")

            except FileNotFoundError as e:
                print(f"\n{'='*60}")
                print(f"ERROR: {e}")
                print(f"Falling back to model-predicted regimes (d_argmax).")
                print(f"{'='*60}\n")
                d_true_full = None
                d_true_test = None

    # Determine which regimes to use for analysis
    if d_true_full is not None and d_true_test is not None and args.use_oracle_switches is True:
        regimes_full_for_analysis = d_true_full
        regimes_test_for_analysis = d_true_test
        regime_source = "ORACLE (Ground Truth)"
    else:
        regimes_full_for_analysis = d_argmax_full
        regimes_test_for_analysis = d_argmax_test
        regime_source = "MODEL (d_argmax)"

    print(f"\n{'='*60}")
    print(f"REGIME SOURCE FOR ANALYSIS: {regime_source}")
    print(f"{'='*60}\n")

    # Print lag compensation info if enabled
    if args.use_lag_compensation:
        print(f"\n{'='*60}")
        print("LAG COMPENSATION ENABLED")
        print(f"{'='*60}")
        print_lag_info(args.problem)
        print(f"{'='*60}\n")

    # Get full dataset length
    data_full = np.asarray(ds["data"])
    if data_full.ndim == 1:
        data_full = data_full[:, None]
    N_full = len(data_full)

    print(f"Test length: {test_len}")
    print(f"Full dataset length: {N_full}")
    print(f"\nRegime switches ({regime_source}):")
    print(f"  - Full dataset: {np.sum(np.diff(regimes_full_for_analysis) != 0)} switches")
    print(f"  - Test set: {np.sum(np.diff(regimes_test_for_analysis) != 0)} switches")

    # Also print model switches for comparison when using oracle
    if args.use_oracle_switches and d_true_full is not None:
        print(f"\nModel switches (d_argmax) for comparison:")
        print(f"  - Full dataset: {np.sum(np.diff(d_argmax_full) != 0)} switches")
        print(f"  - Test set: {np.sum(np.diff(d_argmax_test) != 0)} switches")

    # Prepare data
    y_full = np.asarray(ds["data"])
    if y_full.ndim == 1:
        y_full = y_full.reshape(-1, 1)

    N = len(y_full)
    X_dummy = np.zeros((N, 1), dtype=float)

    # ---------------------------------------------------
    # 1. Run ACI/AgACI
    # ---------------------------------------------------
    print("\n" + "="*60)
    print("Running ACI...")
    print("="*60)

    y_lowers_aci, y_uppers_aci, tab_alpha_t_aci, gammas_aci = aci_intervals(
        X_dummy, y_full, args=args
    )

    print(f"ACI completed. Got {len(gammas_aci)} experts with gammas: {gammas_aci}")

    # Create dictionary of ACI results by gamma for later use
    # Extract unique gamma values (gammas_aci is (n_gammas, T) but all columns are same)
    aci_results_by_gamma = {}
    for i in range(len(gammas_aci)):
        gamma_val = gammas_aci[i, 0]  # Extract first value (all are same)
        aci_results_by_gamma[gamma_val] = (y_lowers_aci[i], y_uppers_aci[i])

    # Select first gamma for main "ACI" comparison
    aci_lower = y_lowers_aci[0]
    aci_upper = y_uppers_aci[0]

    print("\n" + "="*60)
    print("Running AgACI...")
    print("="*60)

    agaci_results = agaci_intervals(
        X_dummy, y_full, basemodel="ds3m", args=args
    )

    agaci_lower = agaci_results['lower']
    agaci_upper = agaci_results['upper']

    print(f"\n  Coverage: {agaci_results['coverage']:.3f}")
    print(f"  Median length: {agaci_results['median_length']:.2f}")

    # ---------------------------------------------------
    # 2.5. Run other conformal prediction methods
    # ---------------------------------------------------
    print("\n" + "="*60)
    print("Running other CP methods (Gaussian, CP, EnbPI)...")
    print("="*60)

    # Note: EnbPI not yet supported for DS3M basemodel
    other_methods = ['Gaussian', 'CP', 'EnbPI']
    try:
        other_cp_results = run_other_cp_methods(
            X_dummy, y_full,
            methods=other_methods,
            basemodel="ds3m",
            params_basemodel=None,  # Will be loaded inside function
            args=args
        )
        print(f"\nSuccessfully ran {len(other_cp_results)} additional CP methods:")
        for method_name, (lower, upper) in other_cp_results.items():
            test_size_eff = len(lower)
            print(f"  {method_name}: {test_size_eff} predictions")
    except Exception as e:
        print(f"\nWarning: Failed to run other CP methods: {e}")
        print("Continuing with ACI/AgACI only...")
        other_cp_results = {}

    # ---------------------------------------------------
    # 3. Align intervals to test period
    # ---------------------------------------------------
    T0 = args.aci_train_size
    test_size_eff = len(aci_lower)

    # Pad beginning with NaN
    aci_lower_full = np.full(test_len, np.nan)
    aci_upper_full = np.full(test_len, np.nan)
    agaci_lower_full = np.full(test_len, np.nan)
    agaci_upper_full = np.full(test_len, np.nan)

    aci_lower_full[T0:T0+test_size_eff] = aci_lower
    aci_upper_full[T0:T0+test_size_eff] = aci_upper
    agaci_lower_full[T0:T0+test_size_eff] = agaci_lower
    agaci_upper_full[T0:T0+test_size_eff] = agaci_upper

    # Align other CP methods to full test period
    other_cp_full = {}
    for method_name, (lower, upper) in other_cp_results.items():
        lower_full = np.full(test_len, np.nan)
        upper_full = np.full(test_len, np.nan)
        method_test_size = len(lower)
        lower_full[T0:T0+method_test_size] = lower
        upper_full[T0:T0+method_test_size] = upper
        other_cp_full[method_name] = (lower_full, upper_full)

    # ---------------------------------------------------
    # 4. Generate plots with dynamic folder names
    # ---------------------------------------------------
    # Create descriptive folder suffix with key parameters
    gamma_min = min(args.tab_gamma)
    gamma_max = max(args.tab_gamma)

    # Add regime method to folder name
    if args.regime_method == "ruptures":
        param_suffix = f"{args.regime_method}_{args.ruptures_method}_{args.ruptures_model}"
        param_suffix += f"_eta{args.agaci_eta:.2f}_lr{args.agaci_lr_schedule}"
        param_suffix += f"_gamma{gamma_min:.4f}-{gamma_max:.4f}_alpha{args.alpha:.2f}"
    else:
        param_suffix = f"ds3m_eta{args.agaci_eta:.2f}_lr{args.agaci_lr_schedule}"
        param_suffix += f"_gamma{gamma_min:.4f}-{gamma_max:.4f}_alpha{args.alpha:.2f}_ddim{args.d_dim}"

    # Add model index to folder name if specific model is used
    if args.model_idx is not None:
        param_suffix += f"_model{args.model_idx}"

    # Add oracle indicator to folder name if using ground truth
    if args.use_oracle_switches and d_true_full is not None:
        param_suffix += "_oracle"

    # Put parameter info in folder name instead of file name
    save_dir = f"figures/agaci_test/{dataset_id}_{param_suffix}"
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*60)
    print("Generating plots...")
    print(f"Parameter configuration: {param_suffix}")
    print(f"Save directory: {save_dir}")
    print("="*60)

    # Plot 0a: Regime heatmap for full dataset (train+valid+test with window indices)
    print("\n0a. Plotting regime heatmap for full dataset (all windows)...")
    test_start_in_full = N_full - test_len  # Where test set starts in full data
    plot_regime_heatmap_full(
        d_argmax=regimes_full_for_analysis,  # Use oracle if available, else model
        d_dim=args.d_dim,
        dataname=args.problem,
        timestamps=None,  # Can't use timestamps for full window sequence
        save_path=f"{save_dir}/regime_heatmap_full.png",
        plot_scope="full",
        test_start_idx=test_start_in_full
    )

    # Plot 0b: Regime heatmap for test set only (with real timestamps)
    print("\n0b. Plotting regime heatmap for test set only (with timestamps)...")
    timestamps_test = load_timestamps_for_dataset(args.problem, test_len, from_end=True)
    plot_regime_heatmap_full(
        d_argmax=regimes_test_for_analysis,  # Use oracle if available, else model
        d_dim=args.d_dim,
        dataname=args.problem,
        timestamps=timestamps_test,
        save_path=f"{save_dir}/regime_heatmap_test.png",
        plot_scope="test"
    )

    # Plot 1: AgACI weights aligned to switches
    # Align weights to full dataset (pad with NaN for pre-test period)
    print("\n1. Plotting AgACI weights at regime switches...")

    # Weights are only available for test portion after T0
    # Extract weights from agaci_results
    weights_lower = agaci_results['weights_lower']  # Shape: (T, n_gammas)

    # Pad to align with full dataset
    weights_lower_full = np.full((len(gammas_aci), N_full), np.nan)
    # Place weights in correct position (test portion starts at N_full - test_len + T0)
    test_start_in_full = N_full - test_len + T0
    weights_lower_full[:, test_start_in_full:test_start_in_full + test_size_eff] = weights_lower.T

    plot_agaci_weights_at_switches(
        agaci_weights=weights_lower_full,  # (n_gammas, N_full) with NaN padding
        d_argmax=regimes_full_for_analysis,  # Use oracle if available, else model
        gamma_values=gammas_aci,
        # window_before and window_after will be computed adaptively
        adaptive_window=False,  # Enable adaptive window sizing
        save_path=f"{save_dir}/agaci_weights_switches.png"
    )

    # Plot 2: Coverage at switches
    print("\n" + "="*60)
    print("2. Plotting coverage dynamics at regime switches...")
    print("="*60)

    # Align individual ACI results to full test period
    aci_by_gamma_full = {}
    for gamma_val, (lower, upper) in aci_results_by_gamma.items():
        lower_full = np.full(test_len, np.nan)
        upper_full = np.full(test_len, np.nan)
        lower_full[T0:T0+test_size_eff] = lower
        upper_full[T0:T0+test_size_eff] = upper
        aci_by_gamma_full[gamma_val] = (lower_full, upper_full)

    # Compute naive interval (same as DS3M for comparison but labeled differently)
    # Naive = using DS3M quantiles directly without adaptation
    intervals_dict = {
        'Naive': (y_lq_ds3m, y_uq_ds3m),  # DS3M original interval (naive/unadapted)
        'AgACI': (agaci_lower_full, agaci_upper_full),
    }

    # Add individual ACI results for each gamma
    for gamma_val, (lower_full, upper_full) in aci_by_gamma_full.items():
        intervals_dict[f'ACI (γ={gamma_val:.4f})'] = (lower_full, upper_full)

    # Add other CP methods (Gaussian, CP, EnbPI)
    for method_name, (lower_full, upper_full) in other_cp_full.items():
        intervals_dict[method_name] = (lower_full, upper_full)

    print(f"\n>>> CREATING intervals_dict with {len(intervals_dict)} methods: {list(intervals_dict.keys())}")

    # Debug: Check intervals
    print(f"\nInterval dict keys: {list(intervals_dict.keys())}")
    for method_name, (lower, upper) in intervals_dict.items():
        n_valid = np.sum(~np.isnan(lower))
        print(f"  {method_name}: {n_valid}/{len(lower)} valid values, "
              f"range=[{np.nanmin(lower):.2f}, {np.nanmax(upper):.2f}]")

    # Save coverage/interval data to CSV if requested
    if args.save_coverage_csv:
        print("\n" + "="*60)
        print("Saving coverage/interval data to CSV...")
        print("="*60)

        # Find regime switches in test set
        d_regimes_for_csv = d_true_test if d_true_test is not None else d_argmax_test
        switch_indices = np.where(np.diff(d_regimes_for_csv) != 0)[0] + 1

        print(f"Found {len(switch_indices)} regime switches in test set")

        # For each switch, extract window of data (-5 to +5)
        window_before = 5
        window_after = 5

        csv_rows = []
        for switch_idx in switch_indices:
            # Skip switches too close to boundaries (can't get full window)
            if switch_idx < window_before or switch_idx + window_after >= len(y_true):
                print(f"  Skipping switch at {switch_idx} (too close to boundary)")
                continue

            # Define window around switch
            start_idx = switch_idx - window_before
            end_idx = switch_idx + window_after + 1

            # Extract time indices relative to switch
            for t_abs in range(start_idx, end_idx):
                t_rel = t_abs - switch_idx  # Relative time: -10 to +10

                # Compute coverage and interval length for each method
                row = {
                    'switch_idx': switch_idx,
                    't_relative': t_rel,
                    't_absolute': t_abs,
                    'y_true': y_true[t_abs]
                }

                for method_name, (lower, upper) in intervals_dict.items():
                    # Coverage: 1 if interval covers y_true, 0 otherwise
                    if not np.isnan(lower[t_abs]) and not np.isnan(upper[t_abs]):
                        covered = 1 if (lower[t_abs] <= y_true[t_abs] <= upper[t_abs]) else 0
                        length = upper[t_abs] - lower[t_abs]
                    else:
                        covered = np.nan
                        length = np.nan

                    row[f'{method_name}_coverage'] = covered
                    row[f'{method_name}_length'] = length
                    row[f'{method_name}_lower'] = lower[t_abs]
                    row[f'{method_name}_upper'] = upper[t_abs]

                csv_rows.append(row)

        # Save to CSV
        csv_df = pd.DataFrame(csv_rows)
        model_suffix = f"_model{args.model_idx}" if args.model_idx is not None else ""
        csv_path = f"{save_dir}/coverage_data{model_suffix}.csv"
        csv_df.to_csv(csv_path, index=False)
        print(f"✓ Saved coverage data to {csv_path}")
        print(f"  - Total rows: {len(csv_rows)}")
        print(f"  - Switches: {len(switch_indices)}")
        print(f"  - Methods: {list(intervals_dict.keys())}")
        print("="*60 + "\n")

    # Plot 2a: Windowed coverage around switches
    print("\n2a. Plotting windowed coverage at regime switches...")

    # Get dataset-specific window size
    dataset_name_for_config = args.problem
    window_before = get_window(dataset_name_for_config)
    window_after = window_before  # Use same window size before and after

    print(f"\n  Dataset name for config: {dataset_name_for_config}")
    print(f"  Dataset-specific window: before=±{window_before}, after=±{window_after}")

    # Apply lag compensation if requested
    if args.use_lag_compensation:
        d_argmax_test_for_plot = apply_lag_compensation(d_argmax_test, args.problem)
        print(f"  ✓ Applied lag compensation to regime indicators")
    else:
        d_argmax_test_for_plot = d_argmax_test

    # Plot using model predicted regimes
    print("\n  Generating coverage plot with MODEL predicted regimes...")
    plot_coverage_at_switches(
        intervals_dict,
        y_true,
        d_argmax_test_for_plot,  # Use potentially compensated regimes
        window_before=window_before,
        window_after=window_after,
        adaptive_window=False,  # Use dataset-specific window
        save_path=f"{save_dir}/coverage_at_switches_model.png"
    )

    # Plot using oracle regimes (if available)
    if d_true_test is not None:
        # Apply lag compensation to oracle regimes too if requested
        if args.use_lag_compensation:
            d_true_test_for_plot = apply_lag_compensation(d_true_test, args.problem)
        else:
            d_true_test_for_plot = d_true_test

        print("\n  Generating coverage plot with ORACLE regimes...")
        plot_coverage_at_switches(
            intervals_dict,
            y_true,
            d_true_test_for_plot,  # Use potentially compensated oracle regimes
            window_before=window_before,
            window_after=window_after,
            adaptive_window=False,  # Use dataset-specific window
            save_path=f"{save_dir}/coverage_at_switches_oracle.png"
        )
    else:
        print("\n  Oracle regimes not available, skipping oracle coverage plot.")

    # Plot 2a2: Windowed interval length around switches
    print("\n2a2. Plotting interval length at regime switches...")

    # Plot using model predicted regimes - NORMALIZED (relative)
    print("\n  Generating NORMALIZED length plot with MODEL predicted regimes...")
    plot_length_at_switches(
        intervals_dict,
        y_true,
        d_argmax_test_for_plot,  # Use potentially compensated
        window_before=window_before,
        window_after=window_after,
        adaptive_window=False,  # Use dataset-specific window
        normalize=True,  # Relative length (percentage)
        save_path=f"{save_dir}/length_at_switches_model_normalized.png"
    )

    # Plot using model predicted regimes - ABSOLUTE
    print("\n  Generating ABSOLUTE length plot with MODEL predicted regimes...")
    plot_length_at_switches(
        intervals_dict,
        y_true,
        d_argmax_test_for_plot,  # Use potentially compensated
        window_before=window_before,
        window_after=window_after,
        adaptive_window=False,  # Use dataset-specific window
        normalize=False,  # Absolute length
        save_path=f"{save_dir}/length_at_switches_model_absolute.png"
    )

    # Plot using oracle regimes (if available)
    if d_true_test is not None:
        # Normalized version
        print("\n  Generating NORMALIZED length plot with ORACLE regimes...")
        plot_length_at_switches(
            intervals_dict,
            y_true,
            d_true_test_for_plot,  # Use potentially compensated oracle
            window_before=window_before,
            window_after=window_after,
            adaptive_window=False,  # Use dataset-specific window
            normalize=True,
            save_path=f"{save_dir}/length_at_switches_oracle_normalized.png"
        )

        # Absolute version
        print("\n  Generating ABSOLUTE length plot with ORACLE regimes...")
        plot_length_at_switches(
            intervals_dict,
            y_true,
            d_true_test_for_plot,  # Use potentially compensated oracle
            window_before=window_before,
            window_after=window_after,
            adaptive_window=False,  # Use dataset-specific window
            normalize=False,
            save_path=f"{save_dir}/length_at_switches_oracle_absolute.png"
        )
    else:
        print("\n  Oracle regimes not available, skipping oracle length plot.")

    # Plot 2d: Individual switch trajectories
    print("\n2d. Plotting individual switch trajectories...")
    # Compute window sizes for this plot
    from experiments.utils.regime_switch_analysis import compute_adaptive_window
    wb, wa = compute_adaptive_window(regimes_test_for_analysis, percentile=50)
    plot_individual_switch_trajectories(
        intervals_dict,
        y_true,
        regimes_test_for_analysis,  # Use oracle if available, else model
        window_before=wb,
        window_after=wa,
        save_path=f"{save_dir}/individual_switch_trajectories.png"
    )

    # Plot 2f: Recovery metrics
    print("\n2f. Computing and plotting coverage recovery metrics...")
    recovery_metrics = compute_recovery_metrics(
        intervals_dict,
        y_true,
        regimes_test_for_analysis,  # Use oracle if available, else model
        target_coverage=1.0 - args.alpha,  # 0.9 for alpha=0.1
        recovery_threshold=0.85,  # 85% of target
        window_size=10
    )

    plot_recovery_comparison(
        recovery_metrics,
        save_path=f"{save_dir}/recovery_comparison.png"
    )

    # Plot 3: Tradeoff plot
    print("\n3. Plotting coverage vs length tradeoff...")
    results_dict = {}

    for method_name, (lower, upper) in intervals_dict.items():
        valid = ~np.isnan(lower) & ~np.isnan(upper)
        if not np.any(valid):
            continue

        y_true_valid = y_true[valid]
        lower_valid = lower[valid]
        upper_valid = upper[valid]

        coverage = np.mean((y_true_valid >= lower_valid) & (y_true_valid <= upper_valid))
        median_length = np.median(upper_valid - lower_valid)

        results_dict[method_name] = (coverage, median_length)
        print(f"   {method_name}: Coverage={coverage:.3f}, MedianLength={median_length:.2f}")

    plot_coverage_vs_length_tradeoff(
        results_dict,
        save_path=f"{save_dir}/tradeoff.png"
    )

    print(f"\n{'='*60}")
    print(f"All plots saved to: {save_dir}/")
    print(f"Configuration: {param_suffix}")
    print(f"{'='*60}\n")

    # Plot: Time series with prediction intervals (like aci_results plots)
    print("\n" + "="*60)
    print("Generating time series plots with prediction intervals...")
    print("="*60)

    # We have the following data available:
    # - testOriginal: true values (shape: (test_len, D))
    # - testForecast_mean: DS3M predictions (shape: (test_len, D))  
    # - y_lq_ds3m, y_uq_ds3m: DS3M MC intervals (shape: (test_len,) for target_dim)
    # - intervals_dict: all conformal intervals

    # Generate plots for AgACI and other key methods
    methods_to_plot = ['AgACI', 'CP', 'ACI (γ=0.0100)', 'Naive', 'Gaussian','EnbPI']

    for method_name in methods_to_plot:
        if method_name not in intervals_dict:
            print(f"  Skipping {method_name} (not in intervals_dict)")
            continue
        
        lower_full, upper_full = intervals_dict[method_name]
        
        # Compute coverage and median width for this method
        valid_mask = ~np.isnan(lower_full) & ~np.isnan(upper_full)
        if np.sum(valid_mask) == 0:
            print(f"  Skipping {method_name} (no valid intervals)")
            continue
        
        y_valid = y_true[valid_mask]
        lower_valid = lower_full[valid_mask]
        upper_valid = upper_full[valid_mask]
        
        coverage = np.mean((y_valid >= lower_valid) & (y_valid <= upper_valid))
        median_width = np.median(upper_valid - lower_valid)
        
        print(f"  Plotting {method_name}: coverage={coverage:.3f}, width={median_width:.2f}")
        
        # Call the plotting function
        # Note: target_dim=0 because data has already been extracted to target dimension above
        plot_results_with_aci(
            dataname=args.problem,
            testOriginal=testOriginal,
            testForecast_mean=testForecast_mean,
            d_dim=args.d_dim,
            forecast_d_MC_argmax=d_argmax_test,
            dsm_lower=y_lq_ds3m.reshape(-1, 1),  # Reshape to (T, 1) for compatibility
            dsm_upper=y_uq_ds3m.reshape(-1, 1),
            aci_lower=lower_full,
            aci_upper=upper_full,
            T0=T0,
            target_dim=0,  # Always 0 since we've already extracted the target dimension
            coverage=coverage,
            width=median_width,
            model_name="DS3M",
            interval_method_name=method_name,
            save_dir_root=save_dir,
            show=False
        )

        print(f"\n✅ Time series plots saved to {save_dir}/aci_results/")


if __name__ == "__main__":
    main()
