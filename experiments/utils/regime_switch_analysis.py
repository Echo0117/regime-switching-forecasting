"""
Analysis tools for ACI/AgACI performance around regime switches.

Based on meeting notes:
1. AgACI weight dynamics aligned to regime switches
2. Coverage analysis around regime switches
3. Coverage vs interval length tradeoff plots
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import seaborn as sns


def detect_regime_switches(d_argmax: np.ndarray) -> np.ndarray:
    """
    Detect regime switch points from d_argmax sequence.

    Parameters
    ----------
    d_argmax : np.ndarray, shape (T,)
        Discrete regime indicators over time

    Returns
    -------
    switch_indices : np.ndarray
        Indices where regime switches occur
    """
    if len(d_argmax) < 2:
        return np.array([], dtype=int)

    # Find where regime changes
    switches = np.where(np.diff(d_argmax) != 0)[0] + 1
    return switches


def align_to_switches(
    data: np.ndarray,
    switch_indices: np.ndarray,
    window_before: int = 10,
    window_after: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align time series data to regime switch points.

    Parameters
    ----------
    data : np.ndarray, shape (T,) or (T, D)
        Time series data to align
    switch_indices : np.ndarray
        Indices of regime switches
    window_before : int
        Number of time steps before switch to include
    window_after : int
        Number of time steps after switch to include

    Returns
    -------
    aligned_data : np.ndarray, shape (n_switches, window_before + window_after + 1, ...)
        Data aligned to switches (trial 0 = switch point)
    valid_mask : np.ndarray, shape (n_switches,)
        Boolean mask indicating which switches had enough data
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    T = len(data)
    window_size = window_before + window_after + 1
    n_switches = len(switch_indices)

    aligned = []
    valid = []

    for switch_idx in switch_indices:
        start_idx = switch_idx - window_before
        end_idx = switch_idx + window_after + 1

        # Check if we have enough data
        if start_idx >= 0 and end_idx <= T:
            aligned.append(data[start_idx:end_idx])
            valid.append(True)
        else:
            # Pad with NaN if not enough data
            segment = np.full((window_size, data.shape[1]), np.nan)
            valid.append(False)
            aligned.append(segment)

    aligned_data = np.array(aligned)  # shape: (n_switches, window_size, D)
    valid_mask = np.array(valid)

    if data.shape[1] == 1:
        aligned_data = aligned_data.squeeze(-1)

    return aligned_data, valid_mask


def plot_agaci_weights_at_switches(
    agaci_weights: np.ndarray,
    d_argmax: np.ndarray,
    gamma_values: List[float],
    window_before: int = 10,
    window_after: int = 50,
    save_path: Optional[str] = None,
    show_individual_lines: bool = True
):
    """
    Plot AgACI weights aligned to regime switches.

    Following tutor's sketch: Each subplot shows one gamma, with each regime switch
    as a separate line overlaid on the same plot.

    Parameters
    ----------
    agaci_weights : np.ndarray, shape (n_gammas, T)
        BOA weights over time for each gamma
    d_argmax : np.ndarray, shape (T,)
        Regime indicators across FULL dataset (train+valid+test)
    gamma_values : list of float
        Gamma values corresponding to each row
    window_before : int
        Time steps before switch
    window_after : int
        Time steps after switch
    save_path : str, optional
        Path to save figure
    show_individual_lines : bool
        If True, show each switch as a separate line (per tutor's sketch)
    """
    import os
    switches = detect_regime_switches(d_argmax)

    if len(switches) == 0:
        print("No regime switches detected")
        return

    n_gammas = agaci_weights.shape[0]
    n_switches = len(switches)

    print(f"Found {n_switches} regime switches in the dataset")

    fig, axes = plt.subplots(1, n_gammas, figsize=(6 * n_gammas, 5), squeeze=False)
    axes = axes.flatten()

    time_axis = np.arange(-window_before, window_after + 1)
    # Use a qualitative palette for better distinction between many switch lines.
    # Prefer seaborn/tab palettes for up to 20 distinct colors, otherwise fall
    # back to a continuous colormap.
    if n_switches <= 10:
        colors = sns.color_palette('tab10', n_switches)
    elif n_switches <= 20:
        colors = sns.color_palette('tab20', n_switches)
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n_switches))

    for gamma_idx, (ax, gamma) in enumerate(zip(axes, gamma_values)):
        weights = agaci_weights[gamma_idx]

        # Align to switches
        aligned, valid = align_to_switches(weights, switches, window_before, window_after)

        n_valid = np.sum(valid)
        if n_valid == 0:
            ax.text(0.5, 0.5, 'No valid switches', ha='center', va='center',
                   transform=ax.transAxes)
            continue

        # Plot each switch as a separate line (as per tutor's sketch)
        if show_individual_lines:
            for switch_idx, (traj, is_valid) in enumerate(zip(aligned, valid)):
                if is_valid:
                    ax.plot(time_axis, traj, alpha=0.6, color=colors[switch_idx],
                           linewidth=1.5, label=f'Switch {switch_idx+1}')

        # Also show mean
        aligned_valid = aligned[valid]
        mean_weight = np.nanmean(aligned_valid, axis=0)
        ax.plot(time_axis, mean_weight, 'k-', linewidth=2.5,
               label=f'Mean ({n_valid} switches)', zorder=10)

        # Mark switch point
        ax.axvline(0, color='r', linestyle='--', linewidth=2, alpha=0.7,
                  label='Regime switch', zorder=5)

        ax.set_xlabel('Time relative to switch (t)', fontsize=11)
        ax.set_ylabel('Weight', fontsize=11)
        ax.set_title(f'γ = {gamma[0]}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Only show legend if few switches
        if n_valid <= 5:
            ax.legend(fontsize=8, loc='best')

    plt.suptitle('AgACI Weight Dynamics Around Regime Switches',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_coverage_at_switches(
    intervals_dict: dict,
    y_true: np.ndarray,
    d_argmax: np.ndarray,
    window_before: int = 10,
    window_after: int = 50,
    save_path: Optional[str] = None
):
    """
    Plot coverage around regime switches for different methods.

    Parameters
    ----------
    intervals_dict : dict
        Dictionary with keys as method names ('OSCP', 'ACI', 'AgACI', etc.)
        and values as tuples (lower_bounds, upper_bounds), each shape (T,)
    y_true : np.ndarray, shape (T,)
        Ground truth values
    d_argmax : np.ndarray, shape (T,)
        Regime indicators
    window_before, window_after : int
        Window size around switches
    save_path : str, optional
        Path to save figure
    """
    switches = detect_regime_switches(d_argmax)

    if len(switches) == 0:
        print("No regime switches detected")
        return

    print(f"\nFound {len(switches)} regime switches at indices: {switches}")
    print(f"Data length: {len(d_argmax)}, Window: [{window_before}, {window_after}]")
    print(f"Valid switch range for alignment: [{window_before}, {len(d_argmax) - window_after - 1}]")

    plt.figure(figsize=(10, 6))
    time_axis = np.arange(-window_before, window_after + 1)

    # Define colors for known methods, use colormap for AgACI gammas
    base_colors = {'DS3M': 'C4', 'Naive': 'C0', 'ACI': 'C2', 'AgACI': 'C3'}

    # Separate AgACI methods and assign colors using a colormap
    agaci_methods = {k: v for k, v in intervals_dict.items() if 'AgACI' in k and k != 'AgACI'}

    # Assign colors to AgACI methods using a colormap
    if agaci_methods:
        agaci_cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(agaci_methods)))
        agaci_colors = {name: agaci_cmap[i] for i, name in enumerate(agaci_methods.keys())}
    else:
        agaci_colors = {}

    colors = {**base_colors, **agaci_colors}

    print(f"Plotting coverage for methods: {list(intervals_dict.keys())}")
    print(f"Number of methods in intervals_dict: {len(intervals_dict)}")

    for method_name, (lower, upper) in intervals_dict.items():
        print(f"Processing method: {method_name}")
        # Compute coverage at each time step
        # Check for NaN values
        n_nan_lower = np.sum(np.isnan(lower))
        n_nan_upper = np.sum(np.isnan(upper))
        if n_nan_lower > 0 or n_nan_upper > 0:
            print(f"Warning: {method_name} has {n_nan_lower} NaN in lower, {n_nan_upper} NaN in upper")

        covered = (y_true >= lower) & (y_true <= upper)

        # Align coverage to switches
        aligned_coverage, valid = align_to_switches(covered.astype(float), switches,
                                                     window_before, window_after)
        n_valid_switches = np.sum(valid)
        aligned_coverage = aligned_coverage[valid]

        print(f"  {method_name}: {n_valid_switches}/{len(valid)} valid switches for alignment")

        if len(aligned_coverage) == 0:
            print(f"Warning: No valid switches for method {method_name}, skipping")
            continue

        # Compute mean coverage across switches
        mean_coverage = np.nanmean(aligned_coverage, axis=0)
        std_coverage = np.nanstd(aligned_coverage, axis=0)

        color = colors.get(method_name, None)

        # Use different line styles for better visibility
        linestyle = '-'
        if 'Naive' in method_name:
            linestyle = ':'
            linewidth = 3
        elif 'ACI' in method_name and 'AgACI' not in method_name:
            linestyle = '--'
            linewidth = 2.5
        else:
            linestyle = '-'
            linewidth = 2

        plt.plot(time_axis, mean_coverage, label=method_name, linewidth=linewidth,
                color=color, linestyle=linestyle)
        # plt.fill_between(time_axis,
        #                 mean_coverage - std_coverage,
        #                 mean_coverage + std_coverage,
        #                 alpha=0.15, color=color)

    # Mark switch point
    plt.axvline(0, color='r', linestyle='--', linewidth=2, label='Regime switch')

    plt.xlabel('Time relative to switch')
    plt.ylabel('Coverage rate')
    plt.title('Coverage dynamics around regime switches')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_coverage_full_timeline(
    intervals_dict: dict,
    y_true: np.ndarray,
    d_argmax: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Plot coverage over the full timeline (entire test set).

    Parameters
    ----------
    intervals_dict : dict
        Dictionary with keys as method names ('Naive', 'ACI', 'AgACI', etc.)
        and values as tuples (lower_bounds, upper_bounds), each shape (T,)
    y_true : np.ndarray, shape (T,)
        Ground truth values
    d_argmax : np.ndarray, shape (T,)
        Regime indicators
    timestamps : np.ndarray, optional
        Timestamps for x-axis. If None, uses indices
    save_path : str, optional
        Path to save figure
    """
    switches = detect_regime_switches(d_argmax)

    fig, ax = plt.subplots(figsize=(16, 6))

    # Define colors
    base_colors = {'DS3M': 'C4', 'Naive': 'C0', 'ACI': 'C2', 'AgACI': 'C3'}
    agaci_methods = {k: v for k, v in intervals_dict.items() if 'AgACI' in k and k != 'AgACI'}
    if agaci_methods:
        agaci_cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(agaci_methods)))
        agaci_colors = {name: agaci_cmap[i] for i, name in enumerate(agaci_methods.keys())}
    else:
        agaci_colors = {}
    colors = {**base_colors, **agaci_colors}

    # Plot coverage for each method
    for method_name, (lower, upper) in intervals_dict.items():
        # Compute coverage at each time step
        covered = (y_true >= lower) & (y_true <= upper)

        # Handle NaN values
        covered_clean = np.where(np.isnan(lower) | np.isnan(upper), np.nan, covered.astype(float))

        color = colors.get(method_name, None)
        ax.plot(covered_clean, label=method_name, linewidth=2, color=color, alpha=0.8)

    # Mark regime switches with vertical lines
    for switch_idx in switches:
        ax.axvline(switch_idx, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Add horizontal line at target coverage (0.9)
    ax.axhline(0.9, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Target (90%)')

    # Set x-axis labels
    if timestamps is not None and len(timestamps) == len(y_true):
        tick_interval = max(1, len(y_true) // 12)
        xticks = np.arange(0, len(y_true), tick_interval)
        xticklabels = [str(timestamps[i]) for i in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45, fontsize=9, ha='right')
        ax.set_xlabel('Time', fontsize=11)
    else:
        ax.set_xlabel('Time step', fontsize=11)

    ax.set_ylabel('Coverage (1 = covered, 0 = not covered)', fontsize=11)
    ax.set_title('Coverage Over Full Timeline with Regime Switches', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_coverage_vs_length_tradeoff(
    results_dict: dict,
    save_path: Optional[str] = None
):
    """
    Plot coverage vs interval length tradeoff for different methods.

    Parameters
    ----------
    results_dict : dict
        Dictionary with keys as method names and values as tuples
        (coverage, median_length)
    save_path : str, optional
        Path to save figure
    """
    plt.figure(figsize=(8, 6))

    colors = {'OSCP': 'C0', 'Naive': 'C1', 'ACI': 'C2', 'AgACI': 'C3'}
    markers = {'OSCP': 'o', 'Naive': 's', 'ACI': '^', 'AgACI': 'D'}

    for method_name, (coverage, length) in results_dict.items():
        color = colors.get(method_name, None)
        marker = markers.get(method_name, 'o')
        plt.scatter(length, coverage, s=200, marker=marker, color=color,
                   label=method_name, edgecolor='black', linewidth=1.5, zorder=3)

    # Add target coverage line
    plt.axhline(0.9, color='gray', linestyle='--', linewidth=1,
               label='Target coverage (90%)', zorder=1)

    # Annotate regions
    ax = plt.gca()
    ax.text(0.95, 0.95, 'Ideal\n(high cov, short len)',
           transform=ax.transAxes, ha='right', va='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.text(0.95, 0.05, 'Useless & bad\n(low cov, long len)',
           transform=ax.transAxes, ha='right', va='bottom',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

    plt.xlabel('Median interval length')
    plt.ylabel('Coverage rate')
    plt.title('Coverage vs Interval Length Tradeoff')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()

def plot_regime_heatmap_full(
    d_argmax: np.ndarray,
    d_dim: int,
    dataname: str,
    timestamps: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    plot_scope: str = "full",
    test_start_idx: Optional[int] = None
):
    """
    Plot regime heatmap for dataset with regime switches marked.

    Parameters
    ----------
    d_argmax : np.ndarray, shape (T,)
        Regime indicators for dataset
    d_dim : int
        Number of regimes
    dataname : str
        Dataset name (for title and time formatting)
    timestamps : np.ndarray, optional
        Timestamps for x-axis. If None, uses indices or dataset-specific defaults
    save_path : str, optional
        Path to save figure
    plot_scope : str, optional
        'full' = plot all data (train+valid+test, may use window indices)
        'test' = plot only test set (can use real timestamps)
        Default: 'full'
    test_start_idx : int, optional
        Index where test set starts in d_argmax (only used when plot_scope='full' to mark boundary)
    """
    switches = detect_regime_switches(d_argmax)
    n_switches = len(switches)

    fig, ax = plt.subplots(figsize=(16, 2))
    regime_map = d_argmax.reshape(1, -1)
    im = ax.imshow(regime_map, aspect='auto', cmap='tab10', interpolation='nearest')

    # Mark regime switches
    for switch_idx in switches:
        ax.axvline(switch_idx, color='red', linestyle='--', linewidth=1, alpha=0.7)

    # Mark train/test boundary if plot_scope='full' and test_start_idx provided
    if plot_scope == "full" and test_start_idx is not None:
        ax.axvline(test_start_idx, color='yellow', linestyle='-', linewidth=2, alpha=0.8,
                   label=f'Test start (index={test_start_idx})')
        ax.legend(loc='upper right', fontsize=9)

    # Add time labels based on timestamps or dataset
    n_points = len(d_argmax)
    if timestamps is not None and len(timestamps) == n_points:
        # Use provided timestamps
        # Show ~10-15 labels
        tick_interval = max(1, n_points // 12)
        xticks = np.arange(0, n_points, tick_interval)
        xticklabels = [str(timestamps[i]) for i in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45, fontsize=9, ha='right')
        ax.set_xlabel('Time', fontsize=11)
    elif plot_scope == "test" and dataname == "Unemployment":
        # Test set with real timestamps (monthly data)
        # For test set, we can use real timestamps if provided
        if timestamps is not None:
            tick_interval = max(1, n_points // 12)
            xticks = np.arange(0, n_points, tick_interval)
            xticklabels = [str(timestamps[i]) for i in xticks]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=45, fontsize=9, ha='right')
            ax.set_xlabel('Time', fontsize=11)
        else:
            # Fallback: show generic time indices
            ax.set_xlabel('Test set index', fontsize=11)
    elif plot_scope == "full":
        # Full dataset: use window indices
        tick_interval = n_points // 10
        if tick_interval > 0:
            xticks = np.arange(0, n_points, tick_interval)
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{i}" for i in xticks], rotation=0, fontsize=9)
        ax.set_xlabel('Window index', fontsize=11)
    else:
        ax.set_xlabel('Time (t)', fontsize=11)

    ax.set_ylabel('Regime', fontsize=11)
    scope_label = "Test Set" if plot_scope == "test" else "Full Dataset (train+valid+test)"
    ax.set_title(f'{dataname}: Regime Switches - {scope_label} (d_dim={d_dim}, {n_switches} switches)',
                 fontsize=12, fontweight='bold')
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, label='Regime ID', orientation='vertical')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {save_path}")
        plt.close()
    else:
        plt.show()


def load_timestamps_for_dataset(dataname: str, data_length: int, from_end: bool = False) -> Optional[np.ndarray]:
    """
    Load timestamps for a given dataset if available.

    Note: DS3M uses sliding windows, so d_argmax_full length (1506 for Unemployment)
    is different from raw data length (879 for Unemployment). Timestamps only work
    when data_length matches the raw data file length.

    Parameters
    ----------
    dataname : str
        Dataset name
    data_length : int
        Expected length of timestamps (number of regime predictions)
    from_end : bool, optional
        If True, return the last N timestamps (for test set).
        If False, return the first N timestamps (for train set).
        Default: False

    Returns
    -------
    timestamps : np.ndarray or None
        Array of timestamps (as strings or datetime objects), or None if not available
    """
    import pandas as pd

    try:
        if dataname == "Unemployment":
            # Load UNRATE dataset
            df = pd.read_csv("Deep_Switching_State_Space_Model/data/Unemployment/UNRATE.csv")
            dates = pd.to_datetime(df['DATE'])
            # Format as "YYYY-MM" for cleaner display
            timestamps = dates.dt.strftime('%Y-%m').values

            if len(timestamps) >= data_length:
                if from_end:
                    return timestamps[-data_length:]  # Last N timestamps (for test set)
                else:
                    return timestamps[:data_length]   # First N timestamps
            else:
                print(f"Warning: Dataset has {len(timestamps)} timestamps but need {data_length}")
                return None
                
        # Add more datasets as needed
        # elif dataname == "Hangzhou":
        #     ...
        
    except Exception as e:
        print(f"Warning: Could not load timestamps for {dataname}: {e}")
        return None
    
    return None
