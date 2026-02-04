#!/usr/bin/env python3
"""
Generate synthetic data for Experiment 2: Controlled Variance Study with AR/ARMA Time Series

"one switch. Before a time series with homogeneous variance v_1, after v_2.
 It could AR time series..."

This script generates test data with:
- ONE switch point (regime change)
- AR(1) or ARMA(1,1) TIME SERIES (not independent Gaussian noise!)
- CONTROLLED homogeneous variance before (V1) and after (V2) the switch
- Systematically vary V1 and V2 to study their effects on coverage and detection

Experiment Design:
------------------
Goal: Disentangle the effect of variance from the effect of regime switches

Generate multiple test sets with different (V1, V2) combinations:
- V1 = homogeneous variance before switch (e.g., 0.1, 0.5, 1.0, 2.0)
- V2 = homogeneous variance after switch  (e.g., 0.1, 0.5, 1.0, 2.0)

This creates a grid of scenarios to understand:
1. How does low→high variance affect switch detection?
2. How does high→low variance affect switch detection?
3. What is the effect of constant variance with a mean shift?

AR(1) Model:
------------
y_t = μ + φ(y_{t-1} - μ) + ε_t
where:
- μ = mean (0.0 before switch, 3.0 after)
- φ = AR coefficient (0.5 for moderate autocorrelation)
- ε_t ~ N(0, σ_ε²) with σ_ε = σ * sqrt(1 - φ²)

ARMA(1,1) Model:
----------------
y_t = μ + φ(y_{t-1} - μ) + ε_t + θε_{t-1}
where:
- μ = mean (0.0 before switch, 3.0 after)
- φ = AR coefficient (0.5 for moderate autocorrelation)
- θ = MA coefficient (0.3 for smoothness)
- ε_t ~ N(0, σ_ε²) with adjusted σ_ε for target variance

Both models ensure:
- Temporal autocorrelation
- Homogeneous variance within each regime
- Smooth time series (not jumpy white noise)

Usage:
------
python experiments/generate_experiment2_data.py

Output:
-------
Saves multiple .npz files in experiments/data/experiment2/:
- exp2_V1_0.1_V2_0.1.npz
- exp2_V1_0.1_V2_0.5.npz
- ...
- exp2_V1_2.0_V2_2.0.npz

Each file contains:
- y_test: test observations (T,) - AR time series
- switch_idx: index where switch occurs
- regime_labels: true regime for each point (0 or 1)
- V1: variance before switch (homogeneous)
- V2: variance after switch (homogeneous)
- mean1: mean before switch
- mean2: mean after switch
- ar_coef: AR(1) coefficient used (0.5)
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt


def generate_arma_timeseries(length, mean, std, ar_coef=0.5, ma_coef=0.3, seed=None):
    """
    Generate ARMA(1,1) time series with specified mean and std.

    ARMA(1,1) model: y_t = mean + ar_coef * (y_{t-1} - mean) + epsilon_t + ma_coef * epsilon_{t-1}
    where epsilon_t ~ N(0, sigma_eps^2)

    This combines:
    - AR(1): Autoregressive component using previous value
    - MA(1): Moving Average component using previous error

    Parameters
    ----------
    length : int
        Length of time series
    mean : float
        Target mean
    std : float
        Target standard deviation (marginal variance)
    ar_coef : float
        AR(1) coefficient (0 < ar_coef < 1 for stationarity)
        Default: 0.5 for moderate autocorrelation
    ma_coef : float
        MA(1) coefficient (controls smoothness)
        Default: 0.3
    seed : int, optional
        Random seed

    Returns
    -------
    y : np.ndarray, shape (length,)
        Generated ARMA time series

    Notes
    -----
    For ARMA(1,1), the variance relationship is more complex:
    var(y) ≈ sigma_eps^2 * (1 + ma_coef^2 + 2*ar_coef*ma_coef) / (1 - ar_coef^2)

    We approximate sigma_eps to achieve target std, though exact calculation is complex.
    """
    if seed is not None:
        np.random.seed(seed)

    # Approximate sigma_eps for ARMA(1,1)
    # This is a simplified calculation; exact formula is more complex
    variance_multiplier = (1 + ma_coef**2 + 2*ar_coef*ma_coef) / (1 - ar_coef**2)
    sigma_eps = std / np.sqrt(variance_multiplier)

    # Generate innovations
    eps = np.random.normal(0, sigma_eps, length)

    # Initialize
    y = np.zeros(length)
    y[0] = mean + eps[0]  # Start at mean

    # Generate ARMA(1,1) process
    for t in range(1, length):
        # AR component: ar_coef * (y_{t-1} - mean)
        ar_term = ar_coef * (y[t-1] - mean)

        # MA component: ma_coef * epsilon_{t-1}
        ma_term = ma_coef * eps[t-1]

        # ARMA(1,1): y_t = mean + ar_term + epsilon_t + ma_term
        y[t] = mean + ar_term + eps[t] + ma_term

    return y


def generate_ar_timeseries(length, mean, std, ar_coef=0.5, seed=None):
    """
    Generate AR(1) time series with specified mean and std.

    AR(1) model: y_t = mean + ar_coef * (y_{t-1} - mean) + epsilon_t
    where epsilon_t ~ N(0, sigma_eps^2)

    Parameters
    ----------
    length : int
        Length of time series
    mean : float
        Target mean
    std : float
        Target standard deviation (marginal variance)
    ar_coef : float
        AR(1) coefficient (0 < ar_coef < 1 for stationarity)
        Default: 0.5 for moderate autocorrelation
    seed : int, optional
        Random seed

    Returns
    -------
    y : np.ndarray, shape (length,)
        Generated AR time series
    """
    if seed is not None:
        np.random.seed(seed)

    # For AR(1): var(y) = sigma_eps^2 / (1 - ar_coef^2)
    # So: sigma_eps = std * sqrt(1 - ar_coef^2)
    sigma_eps = std * np.sqrt(1 - ar_coef**2)

    # Generate innovations
    eps = np.random.normal(0, sigma_eps, length)

    # Initialize
    y = np.zeros(length)
    y[0] = mean + eps[0]  # Start at mean

    # Generate AR(1) process
    for t in range(1, length):
        y[t] = mean + ar_coef * (y[t-1] - mean) + eps[t]

    return y


def generate_one_switch_data(
    T: int = 200,
    switch_idx: int = 100,
    mean1: float = 0.0,
    mean2: float = 3.0,
    V1: float = 0.5,
    V2: float = 1.0,
    ar_coef: float = 0.5,
    ma_coef: float = 0.0,
    seed: int = None,
    model_type: str = "AR"
):
    """
    Generate test data with one switch and controlled variances using AR or ARMA time series.

    Following Marianne's requirement:
    "one switch. Before a time series with homogeneous variance v_1, after v_2."

    Parameters
    ----------
    T : int
        Total length of test sequence
    switch_idx : int
        Index where switch occurs (0-indexed)
    mean1, mean2 : float
        Mean values before and after switch
    V1, V2 : float
        Standard deviations before and after switch (homogeneous within regime)
    ar_coef : float
        AR(1) coefficient for temporal autocorrelation (default=0.5)
    ma_coef : float
        MA(1) coefficient for moving average component (default=0.0)
        Only used if model_type="ARMA"
    seed : int, optional
        Random seed for reproducibility
    model_type : str
        Type of model to use: "AR" or "ARMA" (default="AR")

    Returns
    -------
    y_test : np.ndarray, shape (T,)
        Generated observations (AR or ARMA time series)
    regime_labels : np.ndarray, shape (T,)
        Regime indicators (0 before switch, 1 after)
    """
    if model_type.upper() == "ARMA":
        # Generate ARMA time series before switch (homogeneous variance V1)
        y_before = generate_arma_timeseries(
            length=switch_idx,
            mean=mean1,
            std=V1,
            ar_coef=ar_coef,
            ma_coef=ma_coef,
            seed=seed
        )

        # Generate ARMA time series after switch (homogeneous variance V2)
        seed_after = None if seed is None else seed + 10000
        y_after = generate_arma_timeseries(
            length=T - switch_idx,
            mean=mean2,
            std=V2,
            ar_coef=ar_coef,
            ma_coef=ma_coef,
            seed=seed_after
        )
    else:  # Default to AR
        # Generate AR time series before switch (homogeneous variance V1)
        y_before = generate_ar_timeseries(
            length=switch_idx,
            mean=mean1,
            std=V1,
            ar_coef=ar_coef,
            seed=seed
        )

        # Generate AR time series after switch (homogeneous variance V2)
        # Use different seed to avoid correlation between regimes
        seed_after = None if seed is None else seed + 10000
        y_after = generate_ar_timeseries(
            length=T - switch_idx,
            mean=mean2,
            std=V2,
            ar_coef=ar_coef,
            seed=seed_after
        )

    # Concatenate
    y_test = np.concatenate([y_before, y_after])

    # Regime labels
    regime_labels = np.zeros(T, dtype=int)
    regime_labels[switch_idx:] = 1

    return y_test, regime_labels


def generate_all_experiment2_data(
    base_data_dir: str = "Deep_Switching_State_Space_Model/data",
    original_toy_dir: str = "Deep_Switching_State_Space_Model/data/Toy og",
    T_test: int = 500,
    switch_idx: int = 250,
    mean1: float = -21.03,
    mean2: float = 1.54,
    variance_levels: list = None,
    ar_coef: float = 0.5,
    ma_coef: float = 0.0,
    model_type: str = "AR",
    seed: int = 42
):
    """
    Generate all combinations of (V1, V2) for Experiment 2 in DS3M format.

    "one switch. Before a time series with homogeneous variance v_1, after v_2."

    Output format matches Deep_Switching_State_Space_Model/data/Toy og/ exactly:
    - 5 CSV files per dataset
    - Training set (first 1500 points): unchanged from original Toy
    - Test set (last 500 points): AR or ARMA time series with controlled variance

    Parameters
    ----------
    base_data_dir : str
        Base directory for DS3M data
    original_toy_dir : str
        Directory containing original Toy data to copy training set from
    T_test : int
        Length of test sequence (500 points)
    switch_idx : int
        Where the switch occurs within test set (250 = middle)
    mean1, mean2 : float
        Mean values for two regimes
    variance_levels : list of float
        List of variance levels to test (will create all pairs)
        Default: [0.1, 0.5, 1.0, 2.0]
    ar_coef : float
        AR(1) coefficient for temporal autocorrelation (default=0.5)
    ma_coef : float
        MA(1) coefficient for moving average component (default=0.0)
        Only used if model_type="ARMA"
    model_type : str
        Type of model to use: "AR" or "ARMA" (default="AR")
    seed : int
        Random seed base (each combination gets seed+offset)
    """
    if variance_levels is None:
        variance_levels = [0.1, 0.5, 1.0, 2.0]

    model_name = f"{model_type.upper()}(1,1)" if model_type.upper() == "ARMA" else f"{model_type.upper()}(1)"

    print("=" * 80)
    print(f"Generating Experiment 2 Data: {model_name} Time Series with Controlled Variance")
    print("=" * 80)
    print(f"Training set: First 1500 points (from {original_toy_dir})")
    print(f"Test set: Last {T_test} points ({model_name} time series)")
    print(f"Switch at: t={switch_idx} (within test set)")
    print(f"Mean before switch: {mean1}")
    print(f"Mean after switch: {mean2}")
    print(f"Variance levels: {variance_levels}")
    print(f"AR(1) coefficient: {ar_coef} (temporal autocorrelation)")
    if model_type.upper() == "ARMA":
        print(f"MA(1) coefficient: {ma_coef} (moving average)")
    print(f"Total combinations: {len(variance_levels)**2}")
    print()

    # Load original training data (first 1500 points) from Toy og
    print("Loading original Toy training data...")
    y_train = pd.read_csv(f'{original_toy_dir}/simulation_data_nonlinear_y.csv', header=None).values.flatten()[:1500]
    d_train = pd.read_csv(f'{original_toy_dir}/simulation_data_nonlinear_d.csv', header=None).values.flatten()[:1500]
    z_train = pd.read_csv(f'{original_toy_dir}/simulation_data_nonlinear_z.csv', header=None).values.flatten()[:1500]

    # Load forecasted files (these are only for test set)
    s_dsarf = pd.read_csv(f'{original_toy_dir}/Toy_s_forecasted_dsarf.csv', header=None).values.flatten()
    s_snlds = pd.read_csv(f'{original_toy_dir}/Toy_s_forecasted_snlds.csv', header=None).values.flatten()

    metadata_list = []
    count = 0

    for V1 in variance_levels:
        for V2 in variance_levels:
            # Generate AR or ARMA time series test data
            y_test, regime_labels_test = generate_one_switch_data(
                T=T_test,
                switch_idx=switch_idx,
                mean1=mean1,
                mean2=mean2,
                V1=V1,
                V2=V2,
                ar_coef=ar_coef,
                ma_coef=ma_coef,
                model_type=model_type,
                seed=seed + count
            )

            # Combine training and test data
            y_full = np.concatenate([y_train, y_test])

            # For d (regime labels): train + test
            # d file has 2001 rows (add one more at the end)
            d_test = regime_labels_test
            d_full = np.concatenate([d_train, d_test, [d_test[-1]]])

            # For z (latent states): use same structure as original
            # z file has 2001 rows
            z_test = y_test  # Use observations as latent states for simplicity
            z_full = np.concatenate([z_train, z_test, [z_test[-1]]])

            # Create output directory
            model_suffix = f"_{model_type.lower()}" if model_type.upper() == "ARMA" else f"_{model_type.lower()}"
            mean_suffix = f"_mean{mean1:.0f}_{mean2:.0f}"
            dir_name = f"Toy_exp2_V1_{V1:.1f}_V2_{V2:.1f}{model_suffix}{mean_suffix}"
            output_dir = os.path.join(base_data_dir, dir_name)
            os.makedirs(output_dir, exist_ok=True)

            # Save 5 CSV files in DS3M format
            np.savetxt(f'{output_dir}/simulation_data_nonlinear_y.csv', y_full, fmt='%.18e')
            np.savetxt(f'{output_dir}/simulation_data_nonlinear_d.csv', d_full, fmt='%.18e')
            np.savetxt(f'{output_dir}/simulation_data_nonlinear_z.csv', z_full, fmt='%.18e')
            np.savetxt(f'{output_dir}/Toy_s_forecasted_dsarf.csv', s_dsarf, fmt='%.18e')
            np.savetxt(f'{output_dir}/Toy_s_forecasted_snlds.csv', s_snlds, fmt='%.18e')

            print(f"[{count+1:2d}] V1={V1:.1f}, V2={V2:.1f} → {dir_name}/")

            metadata_list.append({
                'dir_name': dir_name,
                'V1': V1,
                'V2': V2,
                'mean1': mean1,
                'mean2': mean2,
                'switch_idx': switch_idx,
                'T_test': T_test,
                'model_type': model_type,
                'ar_coef': ar_coef,
                'ma_coef': ma_coef if model_type.upper() == "ARMA" else None,
                'seed': seed + count
            })

            count += 1

    # Save metadata summary
    metadata_file = os.path.join(base_data_dir, "experiment2_metadata.txt")
    with open(metadata_file, 'w') as f:
        f.write(f"Experiment 2: {model_name} Time Series with Controlled Variance - Metadata\n")
        f.write("=" * 80 + "\n\n")
        f.write("Following Marianne's requirement:\n")
        f.write("'one switch. Before a time series with homogeneous variance v_1, after v_2.'\n\n")
        f.write(f"Total test sets generated: {len(metadata_list)}\n")
        f.write(f"Training set: First 1500 points (from original Toy)\n")
        f.write(f"Test length (T): {T_test}\n")
        f.write(f"Switch index (within test): {switch_idx}\n")
        f.write(f"Mean before switch (mean1): {mean1}\n")
        f.write(f"Mean after switch (mean2): {mean2}\n")
        f.write(f"Variance levels: {variance_levels}\n")
        f.write(f"Model type: {model_type.upper()}\n")
        f.write(f"AR(1) coefficient: {ar_coef} (temporal autocorrelation)\n")
        if model_type.upper() == "ARMA":
            f.write(f"MA(1) coefficient: {ma_coef} (moving average)\n")
            f.write(f"Model: y_t = μ + {ar_coef}(y_{{t-1}} - μ) + ε_t + {ma_coef}ε_{{t-1}}\n\n")
        else:
            f.write(f"Model: y_t = μ + {ar_coef}(y_{{t-1}} - μ) + ε_t\n\n")
        f.write("Directories:\n")
        f.write("-" * 80 + "\n")
        for meta in metadata_list:
            f.write(f"{meta['dir_name']}: V1={meta['V1']:.1f}, V2={meta['V2']:.1f}\n")

    print()
    print("=" * 80)
    print(f"✓ Generated {len(metadata_list)} test datasets in DS3M format")
    print(f"✓ Saved to: {base_data_dir}/Toy_exp2_*/")
    print(f"✓ Metadata: {metadata_file}")
    print("=" * 80)

    return metadata_list


def visualize_experiment2_samples(
    base_data_dir: str = "Deep_Switching_State_Space_Model/data",
    save_fig_path: str = None
):
    """
    Visualize a sample of generated data to verify it looks correct.
    Shows only the TEST SET (last 500 points).

    Parameters
    ----------
    base_data_dir : str
        Base directory containing Toy_exp2_* directories
    save_fig_path : str, optional
        Path to save the visualization
    """
    # Select interesting cases to visualize
    cases_to_plot = [
        ('Toy_exp2_V1_0.1_V2_0.1', 'Low→Low variance'),
        ('Toy_exp2_V1_0.1_V2_2.0', 'Low→High variance'),
        ('Toy_exp2_V1_2.0_V2_0.1', 'High→Low variance'),
        ('Toy_exp2_V1_1.0_V2_1.0', 'Constant variance'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.ravel()

    for idx, (dirname, title) in enumerate(cases_to_plot):
        dirpath = os.path.join(base_data_dir, dirname)

        if not os.path.exists(dirpath):
            axes[idx].text(0.5, 0.5, f"Directory not found:\n{dirname}",
                          ha='center', va='center', transform=axes[idx].transAxes)
            continue

        # Load data
        y_full = pd.read_csv(f'{dirpath}/simulation_data_nonlinear_y.csv', header=None).values.flatten()
        d_full = pd.read_csv(f'{dirpath}/simulation_data_nonlinear_d.csv', header=None).values.flatten()

        # Extract test set (last 500 points)
        y_test = y_full[1500:]
        d_test = d_full[1500:2000]

        # Extract variance levels from directory name
        parts = dirname.split('_')
        V1 = float(parts[3])
        V2 = float(parts[5])

        mean1 = 0.0
        mean2 = 3.0
        switch_idx = 250

        # Plot
        ax = axes[idx]
        t = np.arange(len(y_test))

        # Color by regime
        regime0_mask = (d_test == 0)
        regime1_mask = (d_test == 1)

        ax.scatter(t[regime0_mask], y_test[regime0_mask],
                  c='C0', s=10, alpha=0.6, label='Regime 0')
        ax.scatter(t[regime1_mask], y_test[regime1_mask],
                  c='C1', s=10, alpha=0.6, label='Regime 1')

        # Mark switch
        ax.axvline(switch_idx, color='red', linestyle='--', linewidth=2,
                  label=f'Switch (t={switch_idx})')

        # Add mean lines
        ax.axhline(mean1, color='C0', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.axhline(mean2, color='C1', linestyle=':', linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Observation', fontsize=10)
        ax.set_title(f'{title}\n(V1={V1:.1f}, V2={V2:.1f}, μ1={mean1:.1f}, μ2={mean2:.1f})',
                    fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_fig_path:
        os.makedirs(os.path.dirname(save_fig_path) or ".", exist_ok=True)
        plt.savefig(save_fig_path, dpi=200, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {save_fig_path}")
    else:
        plt.show()

    plt.close()


def visualize_full_dataset(
    dirname: str = "Toy_exp2_V1_0.5_V2_1.0",
    base_data_dir: str = "Deep_Switching_State_Space_Model/data",
    save_fig_path: str = None
):
    """
    Visualize the FULL dataset (training + test) to see the complete structure.

    Parameters
    ----------
    dirname : str
        Name of directory to visualize (e.g., 'Toy_exp2_V1_0.5_V2_1.0')
    base_data_dir : str
        Base directory containing Toy_exp2_* directories
    save_fig_path : str, optional
        Path to save the visualization
    """
    dirpath = os.path.join(base_data_dir, dirname)

    if not os.path.exists(dirpath):
        print(f"Directory not found: {dirpath}")
        return

    # Load data
    y_full = pd.read_csv(f'{dirpath}/simulation_data_nonlinear_y.csv', header=None).values.flatten()
    d_full = pd.read_csv(f'{dirpath}/simulation_data_nonlinear_d.csv', header=None).values.flatten()[:2000]

    # Extract variance levels from directory name
    parts = dirname.split('_')
    V1 = float(parts[3])
    V2 = float(parts[5])

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # Plot 1: Full dataset with regime colors
    t = np.arange(len(y_full))

    # Color by regime
    regime0_mask = (d_full == 0)
    regime1_mask = (d_full == 1)

    ax1.scatter(t[regime0_mask], y_full[regime0_mask],
              c='C0', s=5, alpha=0.6, label='Regime 0 (μ=0.0)')
    ax1.scatter(t[regime1_mask], y_full[regime1_mask],
              c='C1', s=5, alpha=0.6, label='Regime 1 (μ=3.0)')

    # Mark training/test boundary
    ax1.axvline(1500, color='purple', linestyle='--', linewidth=2.5,
               label='Train/Test Split', alpha=0.8)

    # Mark switch in training (at t=750)
    ax1.axvline(750, color='green', linestyle=':', linewidth=2,
               label='Training Switch', alpha=0.6)

    # Mark switch in test (at t=1750, which is 250 in test set)
    ax1.axvline(1750, color='red', linestyle='--', linewidth=2.5,
               label='Test Switch', alpha=0.8)

    # Add mean lines
    ax1.axhline(0.0, color='C0', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.axhline(3.0, color='C1', linestyle=':', linewidth=1.5, alpha=0.5)

    # Add shaded regions for train/test
    ax1.axvspan(0, 1500, alpha=0.1, color='blue', label='Training Set')
    ax1.axvspan(1500, 2000, alpha=0.1, color='orange', label='Test Set')

    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Observation', fontsize=12)
    ax1.set_title(f'Full Dataset: {dirname}\n'
                 f'Training: Heaviside (no variance) | Test: AR(0.5) with V1={V1:.1f}, V2={V2:.1f}',
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Zoomed views side by side
    ax2_left = plt.subplot(2, 2, 3)
    ax2_right = plt.subplot(2, 2, 4)

    # Left: Training set (first 1500 points)
    y_train = y_full[:1500]
    d_train = d_full[:1500]
    t_train = np.arange(len(y_train))

    regime0_train = (d_train == 0)
    regime1_train = (d_train == 1)

    ax2_left.scatter(t_train[regime0_train], y_train[regime0_train],
                    c='C0', s=5, alpha=0.6, label='Regime 0')
    ax2_left.scatter(t_train[regime1_train], y_train[regime1_train],
                    c='C1', s=5, alpha=0.6, label='Regime 1')
    ax2_left.axvline(750, color='green', linestyle='--', linewidth=2,
                    label='Switch (t=750)')
    ax2_left.axhline(0.0, color='C0', linestyle=':', linewidth=1.5, alpha=0.5)
    ax2_left.axhline(3.0, color='C1', linestyle=':', linewidth=1.5, alpha=0.5)
    ax2_left.set_xlabel('Time', fontsize=11)
    ax2_left.set_ylabel('Observation', fontsize=11)
    ax2_left.set_title('Training Set: Heaviside (σ=0)', fontsize=12, fontweight='bold')
    ax2_left.legend(loc='upper right', fontsize=9)
    ax2_left.grid(True, alpha=0.3)

    # Right: Test set (last 500 points)
    y_test = y_full[1500:]
    d_test = d_full[1500:2000]
    t_test = np.arange(len(y_test))

    regime0_test = (d_test == 0)
    regime1_test = (d_test == 1)

    ax2_right.scatter(t_test[regime0_test], y_test[regime0_test],
                     c='C0', s=10, alpha=0.6, label='Regime 0')
    ax2_right.scatter(t_test[regime1_test], y_test[regime1_test],
                     c='C1', s=10, alpha=0.6, label='Regime 1')
    ax2_right.axvline(250, color='red', linestyle='--', linewidth=2,
                     label='Switch (t=250)')
    ax2_right.axhline(0.0, color='C0', linestyle=':', linewidth=1.5, alpha=0.5)
    ax2_right.axhline(3.0, color='C1', linestyle=':', linewidth=1.5, alpha=0.5)
    ax2_right.set_xlabel('Time (within test set)', fontsize=11)
    ax2_right.set_ylabel('Observation', fontsize=11)
    ax2_right.set_title(f'Test Set: AR(0.5) with σ1={V1:.1f}, σ2={V2:.1f}',
                       fontsize=12, fontweight='bold')
    ax2_right.legend(loc='upper right', fontsize=9)
    ax2_right.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_fig_path:
        os.makedirs(os.path.dirname(save_fig_path) or ".", exist_ok=True)
        plt.savefig(save_fig_path, dpi=200, bbox_inches='tight')
        print(f"\n✓ Full dataset visualization saved to: {save_fig_path}")
    else:
        plt.show()

    plt.close()


def load_experiment2_data(dirname: str, base_data_dir: str = "Deep_Switching_State_Space_Model/data"):
    """
    Helper function to load a specific Experiment 2 dataset.

    Parameters
    ----------
    dirname : str
        Name of directory (e.g., 'Toy_exp2_V1_0.5_V2_1.0')
    base_data_dir : str
        Base directory containing the data directories

    Returns
    -------
    data_dict : dict
        Dictionary with keys: y_full, y_test, d_full, d_test, z_full, V1, V2
    """
    dirpath = os.path.join(base_data_dir, dirname)

    # Load all 5 CSV files
    y_full = pd.read_csv(f'{dirpath}/simulation_data_nonlinear_y.csv', header=None).values.flatten()
    d_full = pd.read_csv(f'{dirpath}/simulation_data_nonlinear_d.csv', header=None).values.flatten()
    z_full = pd.read_csv(f'{dirpath}/simulation_data_nonlinear_z.csv', header=None).values.flatten()

    # Extract variance levels from directory name
    parts = dirname.split('_')
    V1 = float(parts[3])
    V2 = float(parts[5])

    return {
        'y_full': y_full,
        'y_test': y_full[1500:],
        'd_full': d_full,
        'd_test': d_full[1500:2000],
        'z_full': z_full,
        'V1': V1,
        'V2': V2,
        'mean1': 0.0,
        'mean2': 3.0,
        'switch_idx': 250
    }


if __name__ == "__main__":
    import sys

    # Choose model type: "AR" or "ARMA"
    # To generate ARMA data, run: python generate_experiment2_data.py ARMA
    model_type = "ARMA" if len(sys.argv) > 1 and sys.argv[1].upper() == "ARMA" else "AR"

    if model_type == "ARMA":
        print("=" * 80)
        print("GENERATING ARMA(1,1) TIME SERIES DATA")
        print("=" * 80)
        ma_coef = 0.3  # MA coefficient for ARMA model
    else:
        print("=" * 80)
        print("GENERATING AR(1) TIME SERIES DATA")
        print("=" * 80)
        ma_coef = 0.0  # Not used for AR model

    # Generate all Experiment 2 data in DS3M format with AR or ARMA time series
    metadata = generate_all_experiment2_data(
        base_data_dir="Deep_Switching_State_Space_Model/data",
        original_toy_dir="Deep_Switching_State_Space_Model/data/Toy og",
        T_test=500,
        switch_idx=250,
        mean1=-20.0,   # Mean before switch
        mean2=2.0,     # Mean after switch
        variance_levels=[0.5, 2.0, 10.0, 20.0],  # 包含低方差(0.5~2.0)和高方差(10.0~20.0)
        ar_coef=0.5,  # AR(1) coefficient for temporal autocorrelation
        ma_coef=ma_coef,  # MA(1) coefficient for ARMA model
        model_type=model_type,
        seed=42
    )

    # Create visualizations
    print("\nGenerating sample visualization...")
    visualize_experiment2_samples(
        base_data_dir="Deep_Switching_State_Space_Model/data",
        save_fig_path="figures/experiment2_ar_samples.png"
    )

    print("\nGenerating full dataset visualization...")
    visualize_full_dataset(
        dirname="Toy_exp2_V1_0.5_V2_20.0_ar_mean-20_2",
        base_data_dir="Deep_Switching_State_Space_Model/data",
        save_fig_path="figures/experiment2_full_dataset.png"
    )

    print("\n" + "=" * 80)
    if model_type == "ARMA":
        print("✓ All done! Generated ARMA(1,1) time series with controlled variance in DS3M format")
    else:
        print("✓ All done! Generated AR(1) time series with controlled variance in DS3M format")
    print("=" * 80)
    print("\nKey features:")
    if model_type == "ARMA":
        print("  ✓ ARMA(1,1) time series (not white noise)")
        print("  ✓ AR component: Temporal autocorrelation (φ = 0.5)")
        print("  ✓ MA component: Smoothing effect (θ = 0.3)")
    else:
        print("  ✓ AR(1) time series (not white noise)")
        print("  ✓ Temporal autocorrelation: ACF(lag=1) ≈ 0.5")
    print("  ✓ Homogeneous variance within each regime")
    print("  ✓ Follows Marianne's requirement")
    print("  ✓ DS3M format: 5 CSV files per dataset")
    print("  ✓ Training set (first 1500 points) from original Toy")
    print(f"  ✓ Test set (last 500 points) with {model_type}(1{',' if model_type == 'ARMA' else ''}{1 if model_type == 'ARMA' else ''}) time series")
    print("\nNext steps:")
    print("1. Verify AR/ARMA properties: python experiments/verify_ar_timeseries.py")
    print("2. Run DS3M training on these datasets")
    print("3. Run ACP methods on the test sets")
    print("4. Compare coverage & detection across different (V1, V2) combinations")
    print("5. Analyze: How does variance affect switch detection?")
    print("\nTo generate ARMA data instead of AR:")
    print("  python experiments/generate_experiment2_data.py ARMA")
    print("=" * 80)
