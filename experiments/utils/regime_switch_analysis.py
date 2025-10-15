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

        # Check if we have enough data and no NaN values
        if start_idx >= 0 and end_idx <= T:
            segment = data[start_idx:end_idx]
            # Check if segment contains NaN values
            has_nan = np.any(np.isnan(segment))
            if not has_nan:
                aligned.append(segment)
                valid.append(True)
            else:
                # If segment has NaN, mark as invalid
                aligned.append(segment)
                valid.append(False)
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


def compute_adaptive_window(
    d_argmax: np.ndarray,
    percentile: float = 50.0
) -> Tuple[int, int]:
    """
    Compute adaptive window size based on regime length distribution.

    Parameters
    ----------
    d_argmax : np.ndarray
        Regime indicators
    percentile : float
        Percentile of regime lengths to use (default: 50 = median)

    Returns
    -------
    window_before : int
        Number of steps before switch
    window_after : int
        Number of steps after switch
    """
    switches = detect_regime_switches(d_argmax)

    if len(switches) < 2:
        # Not enough switches, use defaults
        return 10, 50

    # Compute regime lengths (distance between consecutive switches)
    regime_lengths = np.diff(switches)

    # Use percentile of regime lengths
    typical_length = np.percentile(regime_lengths, percentile)

    # Window sizes: typically want to see about half a regime before/after
    window_before = max(5, int(typical_length * 0.3))  # 30% before
    window_after = max(10, int(typical_length * 0.7))   # 70% after

    # Cap maximum to avoid too large windows
    window_before = min(window_before, 50)
    window_after = min(window_after, 150)

    return window_before, window_after


def plot_agaci_weights_at_switches(
    agaci_weights: np.ndarray,
    d_argmax: np.ndarray,
    gamma_values: List[float],
    window_before: Optional[int] = None,
    window_after: Optional[int] = None,
    save_path: Optional[str] = None,
    show_individual_lines: bool = True,
    adaptive_window: bool = True
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
    window_before : int, optional
        Time steps before switch. If None and adaptive_window=True, computed automatically.
    window_after : int, optional
        Time steps after switch. If None and adaptive_window=True, computed automatically.
    save_path : str, optional
        Path to save figure
    show_individual_lines : bool
        If True, show each switch as a separate line (per tutor's sketch)
    adaptive_window : bool
        If True and window sizes not provided, compute adaptive window based on regime lengths
    """
    import os
    switches = detect_regime_switches(d_argmax)

    # Compute adaptive window if not provided
    if adaptive_window and (window_before is None or window_after is None):
        computed_before, computed_after = compute_adaptive_window(d_argmax, percentile=50)
        if window_before is None:
            window_before = computed_before
        if window_after is None:
            window_after = computed_after

        print(f"\n[ADAPTIVE WINDOW] Computed window sizes based on regime lengths:")
        print(f"[ADAPTIVE WINDOW]   window_before: {window_before}")
        print(f"[ADAPTIVE WINDOW]   window_after: {window_after}")
        print(f"[ADAPTIVE WINDOW]   Total window: {window_before + window_after + 1} time steps")

        # Show regime length statistics
        if len(switches) >= 2:
            regime_lengths = np.diff(switches)
            print(f"[ADAPTIVE WINDOW] Regime length statistics:")
            print(f"[ADAPTIVE WINDOW]   Min: {np.min(regime_lengths)}, Max: {np.max(regime_lengths)}")
            print(f"[ADAPTIVE WINDOW]   Mean: {np.mean(regime_lengths):.1f}, Median: {np.median(regime_lengths):.1f}")
    else:
        # Use defaults if not provided
        if window_before is None:
            window_before = 10
        if window_after is None:
            window_after = 50

    if len(switches) == 0:
        print("No regime switches detected")
        return

    n_gammas = agaci_weights.shape[0]
    n_switches = len(switches)

    print(f"\n[WEIGHT PLOT DEBUG] Found {n_switches} regime switches in the dataset")
    print(f"[WEIGHT PLOT DEBUG] Switches at indices: {switches[:20]}..." if n_switches > 20 else f"[WEIGHT PLOT DEBUG] Switches at indices: {switches}")
    print(f"[WEIGHT PLOT DEBUG] agaci_weights shape: {agaci_weights.shape}")
    print(f"[WEIGHT PLOT DEBUG] Data length (d_argmax): {len(d_argmax)}")

    # Check how many weights are non-NaN
    for gamma_idx in range(n_gammas):
        n_valid_weights = np.sum(~np.isnan(agaci_weights[gamma_idx]))
        valid_range = np.where(~np.isnan(agaci_weights[gamma_idx]))[0]
        if len(valid_range) > 0:
            print(f"[WEIGHT PLOT DEBUG] Gamma {gamma_idx}: {n_valid_weights} valid weights in range [{valid_range[0]}, {valid_range[-1]}]")
        else:
            print(f"[WEIGHT PLOT DEBUG] Gamma {gamma_idx}: All NaN")

    fig, axes = plt.subplots(1, n_gammas, figsize=(6 * n_gammas, 5), squeeze=False)
    axes = axes.flatten()

    time_axis = np.arange(-window_before, window_after + 1)

    # Define distinctive colors for switch trajectories
    # These will be used for valid switches only
    distinct_colors = [
        '#e41a1c',  # Red
        '#377eb8',  # Blue
        '#4daf4a',  # Green
        '#984ea3',  # Purple
        '#ff7f00',  # Orange
        '#ffff33',  # Yellow
        '#a65628',  # Brown
        '#f781bf',  # Pink
        '#999999',  # Gray
        '#66c2a5',  # Teal
    ]

    for gamma_idx, (ax, gamma) in enumerate(zip(axes, gamma_values)):
        weights = agaci_weights[gamma_idx]

        # Align to switches
        aligned, valid = align_to_switches(weights, switches, window_before, window_after)

        n_valid = np.sum(valid)
        print(f"[WEIGHT PLOT DEBUG] Gamma {gamma_idx}: {n_valid}/{len(valid)} valid switches after alignment")

        # Check which switches are valid
        valid_switches = switches[valid]
        invalid_switches = switches[~valid]
        if len(valid_switches) > 0:
            print(f"[WEIGHT PLOT DEBUG]   Valid switches at: {valid_switches[:10]}..." if len(valid_switches) > 10 else f"[WEIGHT PLOT DEBUG]   Valid switches at: {valid_switches}")
        if len(invalid_switches) > 0 and len(invalid_switches) <= 10:
            print(f"[WEIGHT PLOT DEBUG]   Invalid switches at: {invalid_switches}")
        elif len(invalid_switches) > 10:
            print(f"[WEIGHT PLOT DEBUG]   {len(invalid_switches)} invalid switches")

        if n_valid == 0:
            ax.text(0.5, 0.5, 'No valid switches', ha='center', va='center',
                   transform=ax.transAxes)
            continue

        # Plot each switch as a separate line (as per tutor's sketch)
        if show_individual_lines:
            valid_color_idx = 0  # Track color index for valid switches only
            for switch_idx, (traj, is_valid) in enumerate(zip(aligned, valid)):
                if is_valid:
                    # Use color based on valid_color_idx, cycling through distinct_colors
                    color = distinct_colors[valid_color_idx % len(distinct_colors)]
                    switch_position = switches[switch_idx]  # Actual position in full data
                    ax.plot(time_axis, traj, alpha=0.7, color=color,
                           linewidth=2.0, label=f'Switch at t={switch_position}')
                    valid_color_idx += 1

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

        # Show legend if not too many switches
        if n_valid <= 10:
            ax.legend(fontsize=9, loc='best', framealpha=0.9)
        elif n_valid <= 20:
            ax.legend(fontsize=7, loc='best', ncol=2, framealpha=0.9)

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
    window_before: Optional[int] = None,
    window_after: Optional[int] = None,
    save_path: Optional[str] = None,
    adaptive_window: bool = True
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
    window_before : int, optional
        Window size before switches. If None and adaptive_window=True, computed automatically.
    window_after : int, optional
        Window size after switches. If None and adaptive_window=True, computed automatically.
    save_path : str, optional
        Path to save figure
    adaptive_window : bool
        If True and window sizes not provided, compute adaptive window
    """
    switches = detect_regime_switches(d_argmax)

    # Compute adaptive window if not provided
    if adaptive_window and (window_before is None or window_after is None):
        # Create a padded d_argmax for full dataset if needed
        # For this function, d_argmax is usually from test set only
        computed_before, computed_after = compute_adaptive_window(d_argmax, percentile=50)
        if window_before is None:
            window_before = computed_before
        if window_after is None:
            window_after = computed_after
        print(f"\n[COVERAGE PLOT] Using adaptive window: before={window_before}, after={window_after}")
    else:
        if window_before is None:
            window_before = 10
        if window_after is None:
            window_after = 50

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
    Plot coverage over the full timeline using bar chart (entire test set).

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

    # Plot coverage for each method using bar chart
    for method_idx, (method_name, (lower, upper)) in enumerate(intervals_dict.items()):
        # Compute coverage at each time step
        covered = (y_true >= lower) & (y_true <= upper)

        # Handle NaN values
        covered_clean = np.where(np.isnan(lower) | np.isnan(upper), np.nan, covered.astype(float))

        color = colors.get(method_name, None)

        # Add vertical offset for each method (stacked visualization)
        y_position = method_idx  # 1.0 spacing between methods

        # Use bar visualization for binary data
        time_indices = np.arange(len(covered_clean))

        # Create bars for coverage status
        for t in range(len(covered_clean)):
            if not np.isnan(covered_clean[t]):
                if covered_clean[t] == 1:
                    # Covered: draw filled bar
                    ax.barh(y_position, 1, left=t, height=0.8, color=color,
                           alpha=0.7, edgecolor='none')
                else:
                    # Not covered: draw empty/outline bar
                    ax.barh(y_position, 1, left=t, height=0.8, color='white',
                           alpha=0.9, edgecolor=color, linewidth=0.5)

        # Add method label on the left
        ax.text(-len(covered_clean)*0.02, y_position, method_name,
               va='center', ha='right', fontsize=11, fontweight='bold', color=color)

    # Mark regime switches with vertical lines spanning all methods
    n_methods = len(intervals_dict)
    for switch_idx in switches:
        ax.axvline(switch_idx, color='red', linestyle='--', linewidth=1.5, alpha=0.4,
                  ymin=0, ymax=1)

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

    # Adjust y-axis to fit all methods
    ax.set_ylim([-0.5, n_methods - 0.2])
    ax.set_ylabel('Method', fontsize=11)
    ax.set_title('Coverage Over Full Timeline with Regime Switches\n(Filled bar = covered, Empty bar = not covered)',
                fontsize=13, fontweight='bold')

    # Remove default y-ticks since we have method labels
    ax.set_yticks([])

    # Add legend for bars
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='gray', alpha=0.7, label='Covered (prediction interval contains true value)'),
        Patch(facecolor='white', edgecolor='gray', linewidth=0.5,
              label='Not covered (true value outside interval)'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
              alpha=0.4, label='Regime switch')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95)

    ax.grid(True, alpha=0.2, axis='x')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_coverage_timeline_scatter(
    intervals_dict: dict,
    y_true: np.ndarray,
    d_argmax: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Plot coverage over the full timeline using scatter plot (○/× for each point).

    This is complementary to plot_coverage_full_timeline which uses bars.
    Scatter plot shows individual time points more clearly.

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

    # Plot coverage for each method using scatter plot
    for method_idx, (method_name, (lower, upper)) in enumerate(intervals_dict.items()):
        # Compute coverage at each time step
        covered = (y_true >= lower) & (y_true <= upper)

        # Handle NaN values
        covered_clean = np.where(np.isnan(lower) | np.isnan(upper), np.nan, covered.astype(float))

        color = colors.get(method_name, None)

        # Add vertical offset for each method (stacked visualization)
        y_position = method_idx  # 1.0 spacing between methods

        # Use scatter plot for binary data
        time_indices = np.arange(len(covered_clean))

        # Separate covered and not covered points
        covered_mask = covered_clean == 1
        not_covered_mask = covered_clean == 0

        # Plot covered points (filled circles)
        if np.any(covered_mask):
            covered_times = time_indices[covered_mask]
            ax.scatter(covered_times, np.full(len(covered_times), y_position),
                      marker='o', s=30, color=color, alpha=0.8, edgecolors='none',
                      zorder=3)

        # Plot not covered points (X markers)
        if np.any(not_covered_mask):
            not_covered_times = time_indices[not_covered_mask]
            ax.scatter(not_covered_times, np.full(len(not_covered_times), y_position),
                      marker='x', s=50, color=color, alpha=0.9, linewidths=2,
                      zorder=3)

        # Add horizontal reference line for this method
        ax.axhline(y_position, color='gray', linestyle='-', linewidth=0.3, alpha=0.2, zorder=0)

        # Add method label on the left
        ax.text(-len(covered_clean)*0.02, y_position, method_name,
               va='center', ha='right', fontsize=11, fontweight='bold', color=color)

    # Mark regime switches with vertical lines spanning all methods
    n_methods = len(intervals_dict)
    for switch_idx in switches:
        ax.axvline(switch_idx, color='red', linestyle='--', linewidth=1.5, alpha=0.4,
                  ymin=0, ymax=1, zorder=1)

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

    # Adjust y-axis to fit all methods
    ax.set_ylim([-0.5, n_methods - 0.2])
    ax.set_ylabel('Method', fontsize=11)
    ax.set_title('Coverage Over Full Timeline (Scatter Plot)\n(○ = covered, × = not covered)',
                fontsize=13, fontweight='bold')

    # Remove default y-ticks since we have method labels
    ax.set_yticks([])

    # Add legend for scatter markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
              markersize=8, linestyle='', label='Covered (true value in interval)'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='gray',
              markeredgecolor='gray', markersize=10, markeredgewidth=2,
              linestyle='', label='Not covered (true value outside interval)'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
              alpha=0.4, label='Regime switch')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95)

    ax.grid(True, alpha=0.2, axis='x')
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
