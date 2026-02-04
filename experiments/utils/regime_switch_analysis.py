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


def detect_regime_switches(
    d_argmax: np.ndarray,
    min_gap: int = 1
) -> np.ndarray:
    """
    Detect regime switch points from d_argmax sequence.

    Parameters
    ----------
    d_argmax : np.ndarray, shape (T,)
        Discrete regime indicators over time

    min_gap : int
        Minimum distance (in timesteps) required between successive switches.
        Use values >1 to suppress rapid flip-flops caused by noise.

    Returns
    -------
    switch_indices : np.ndarray
        Indices where regime switches occur
    """
    if len(d_argmax) < 2:
        return np.array([], dtype=int)

    # Find where regime changes
    switches = np.where(np.diff(d_argmax) != 0)[0] + 1

    if len(switches) == 0 or min_gap <= 1:
        return switches

    filtered = [switches[0]]
    for idx in switches[1:]:
        if idx - filtered[-1] >= min_gap:
            filtered.append(idx)

    return np.array(filtered, dtype=int)


def align_to_switches(
    data: np.ndarray,
    switch_indices: np.ndarray,
    window_before: int = 5,
    window_after: int = 5,
    cut_at_next_switch: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align time series data to regime switch points.

    When cut_at_next_switch=True, cut the window if another
    switch occurs within window_after. Use min(window_after, time_to_next_switch - 1).

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
    cut_at_next_switch : bool
        If True, cut window when next switch occurs

    Returns
    -------
    aligned_data : list of np.ndarray
        Data aligned to switches. Each element may have different length if cut_at_next_switch=True
    valid_mask : np.ndarray, shape (n_switches,)
        Boolean mask indicating which switches had enough data
    actual_lengths : np.ndarray, shape (n_switches,)
        Actual length of each aligned segment after cutting
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    T = len(data)
    n_switches = len(switch_indices)

    aligned = []
    valid = []
    actual_lengths = []

    for i, switch_idx in enumerate(switch_indices):
        start_idx = switch_idx - window_before

        # Determine end index based on next switch
        if cut_at_next_switch and i < n_switches - 1:
            # Time until next switch
            next_switch = switch_indices[i + 1]
            time_to_next = next_switch - switch_idx
            # Use min(window_after, time_to_next_switch - 1)
            effective_window_after = min(window_after, time_to_next - 1)
        else:
            effective_window_after = window_after

        end_idx = switch_idx + effective_window_after + 1
        actual_length = window_before + effective_window_after + 1

        # Check if we have enough data and no NaN values
        if start_idx >= 0 and end_idx <= T:
            segment = data[start_idx:end_idx]
            # Check if segment contains NaN values
            has_nan = np.any(np.isnan(segment))
            if not has_nan:
                aligned.append(segment if data.shape[1] > 1 else segment.squeeze(-1))
                valid.append(True)
                actual_lengths.append(actual_length)
            else:
                # If segment has NaN, mark as invalid
                aligned.append(segment if data.shape[1] > 1 else segment.squeeze(-1))
                valid.append(False)
                actual_lengths.append(actual_length)
        else:
            # Not enough data before or after
            valid.append(False)
            # Still create a segment but mark as invalid
            safe_start = max(0, start_idx)
            safe_end = min(T, end_idx)
            segment = data[safe_start:safe_end]
            aligned.append(segment if data.shape[1] > 1 else segment.squeeze(-1))
            actual_lengths.append(len(segment))

    valid_mask = np.array(valid)
    actual_lengths = np.array(actual_lengths)

    return aligned, valid_mask, actual_lengths


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
        return 5, 5

    # Compute regime lengths (distance between consecutive switches)
    regime_lengths = np.diff(switches)

    # Use percentile of regime lengths
    typical_length = np.percentile(regime_lengths, percentile)

    # Window sizes: typically want to see about half a regime before/after
    window_before = max(5, int(typical_length * 0.3))  # 30% before
    window_after = max(5, int(typical_length * 0.7))   # 70% after

    # Cap maximum to avoid too large windows
    window_before = min(window_before, 5)
    window_after = min(window_after, 5)

    return window_before, window_after


def plot_agaci_weights_at_switches(
    agaci_weights: np.ndarray,
    d_argmax: np.ndarray,
    gamma_values: List[float],
    window_before: Optional[int] = None,
    window_after: Optional[int] = None,
    save_path: Optional[str] = None,
    show_individual_lines: bool = True,
    adaptive_window: bool = False,
    standardize_ylim: bool = True,
    cut_at_next_switch: bool = True
):
    """
    Plot AgACI weights aligned to regime switches.

    Each subplot shows one gamma, with each regime switch
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
        If True, show each switch as a separate line
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
            window_before = 5
        if window_after is None:
            window_after = 5

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

    # First pass: collect all valid aligned trajectories to compute global y-limits
    all_valid_weights = []
    if standardize_ylim:
        for gamma_idx in range(n_gammas):
            weights = agaci_weights[gamma_idx]
            aligned, valid, _ = align_to_switches(weights, switches, window_before, window_after, cut_at_next_switch)
            for traj, is_valid in zip(aligned, valid):
                if is_valid:
                    all_valid_weights.extend(traj.flatten())

        if len(all_valid_weights) > 0:
            global_ymin = np.nanmin(all_valid_weights)
            global_ymax = np.nanmax(all_valid_weights)
            # Add 10% padding
            y_range = global_ymax - global_ymin
            global_ylim = (global_ymin - 0.1 * y_range, global_ymax + 0.1 * y_range)
        else:
            global_ylim = None
    else:
        global_ylim = None

    for gamma_idx, (ax, gamma) in enumerate(zip(axes, gamma_values)):
        weights = agaci_weights[gamma_idx]

        # Align to switches with cutting enabled
        aligned, valid, actual_lengths = align_to_switches(weights, switches, window_before, window_after, cut_at_next_switch)

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

        # Plot each switch as a separate line
        if show_individual_lines:
            valid_color_idx = 0  # Track color index for valid switches only
            for switch_idx, (traj, is_valid, actual_len) in enumerate(zip(aligned, valid, actual_lengths)):
                if is_valid:
                    # Use color based on valid_color_idx, cycling through distinct_colors
                    color = distinct_colors[valid_color_idx % len(distinct_colors)]
                    switch_position = switches[switch_idx]  # Actual position in full data
                    # Create time axis for this specific trajectory
                    traj_time_axis = np.arange(-window_before, -window_before + actual_len)
                    ax.plot(traj_time_axis, traj, alpha=0.7, color=color,
                           linewidth=2.0, label=f'Switch at t={switch_position}')
                    valid_color_idx += 1

        # Compute mean with variable-length trajectories
        # We need to pad to common length for averaging
        max_len = window_before + window_after + 1
        padded_trajs = []
        for traj, is_valid in zip(aligned, valid):
            if is_valid:
                padded = np.full(max_len, np.nan)
                padded[:len(traj)] = traj
                padded_trajs.append(padded)

        if len(padded_trajs) > 0:
            padded_trajs = np.array(padded_trajs)
            mean_weight = np.nanmean(padded_trajs, axis=0)
            # Count how many trajectories contribute to each time point
            n_contributors = np.sum(~np.isnan(padded_trajs), axis=0)
            # Only plot where we have at least one contributor
            time_axis = np.arange(-window_before, window_after + 1)
            valid_time_mask = n_contributors > 0
            ax.plot(time_axis[valid_time_mask], mean_weight[valid_time_mask], 'k-', linewidth=2.5,
                   label=f'Mean ({n_valid} switches)', zorder=10)

        # Mark switch point
        ax.axvline(0, color='r', linestyle='--', linewidth=2, alpha=0.7,
                  label='Regime switch', zorder=5)

        ax.set_xlabel('Time relative to switch (t)', fontsize=11)
        ax.set_ylabel('Weight', fontsize=11)
        ax.set_title(f'γ = {gamma[0]}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Apply standardized y-limits if requested
        if standardize_ylim and global_ylim is not None:
            ax.set_ylim(global_ylim)

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
    adaptive_window: bool = False,
    show_error_bars: bool = True,
    cut_at_next_switch: bool = True,
    precomputed_coverage: Optional[dict] = None
):
    """
    Plot coverage around regime switches for different methods (AVERAGED over all switches).

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
            window_before = 5
        if window_after is None:
            window_after = 5

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

        # Check if we have precomputed coverage for this method
        if precomputed_coverage is not None and method_name in precomputed_coverage:
            # Use precomputed coverage (for multi-model case)
            covered = precomputed_coverage[method_name]
            print(f"  Using precomputed multi-model coverage for {method_name}")
        else:
            # Compute coverage at each time step (standard case)
            # Check for NaN values
            n_nan_lower = np.sum(np.isnan(lower))
            n_nan_upper = np.sum(np.isnan(upper))
            if n_nan_lower > 0 or n_nan_upper > 0:
                print(f"Warning: {method_name} has {n_nan_lower} NaN in lower, {n_nan_upper} NaN in upper")

            covered = (y_true >= lower) & (y_true <= upper)

        # Align coverage to switches with cutting
        aligned_coverage, valid, actual_lengths = align_to_switches(covered.astype(float), switches,
                                                     window_before, window_after, cut_at_next_switch)
        n_valid_switches = np.sum(valid)

        print(f"  {method_name}: {n_valid_switches}/{len(valid)} valid switches for alignment")

        if n_valid_switches == 0:
            print(f"Warning: No valid switches for method {method_name}, skipping")
            continue

        # Pad aligned coverage to common length for averaging
        max_len = window_before + window_after + 1
        padded_coverage = []
        for traj, is_valid in zip(aligned_coverage, valid):
            if is_valid:
                padded = np.full(max_len, np.nan)
                padded[:len(traj)] = traj
                padded_coverage.append(padded)

        padded_coverage = np.array(padded_coverage)

        # Compute mean coverage across switches
        mean_coverage = np.nanmean(padded_coverage, axis=0)
        std_coverage = np.nanstd(padded_coverage, axis=0)
        # Standard error for error bars
        n_contributors = np.sum(~np.isnan(padded_coverage), axis=0)
        stderr_coverage = std_coverage / np.sqrt(np.maximum(n_contributors, 1))

        time_axis = np.arange(-window_before, window_after + 1)
        # Only plot where we have contributors
        valid_time_mask = n_contributors > 0

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

        plt.plot(time_axis[valid_time_mask], mean_coverage[valid_time_mask],
                label=method_name, linewidth=linewidth,
                color=color, linestyle=linestyle)

        # Add error bars (standard error)
        if show_error_bars:
            plt.fill_between(time_axis[valid_time_mask],
                            mean_coverage[valid_time_mask] - stderr_coverage[valid_time_mask],
                            mean_coverage[valid_time_mask] + stderr_coverage[valid_time_mask],
                            alpha=0.2, color=color)

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


def plot_coverage_vs_length_tradeoff(
    results_dict: dict,
    save_path: Optional[str] = None,
    normalize: bool = True
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
    normalize : bool, optional (default=True)
        If True, normalize interval lengths as percentage of maximum.
        This makes values more interpretable (e.g., "80%" instead of "40").
    """
    # Normalize interval lengths if requested
    if normalize:
        from experiments.utils.experiments_utils import normalize_interval_lengths
        plot_dict = normalize_interval_lengths(results_dict)
        xlabel = 'Relative Interval Length (%)'
        xlabel_note = '\n(100% = widest method, lower is better)'
    else:
        plot_dict = results_dict
        xlabel = 'Median Interval Length'
        xlabel_note = ''

    plt.figure(figsize=(8, 6))

    colors = {'OSCP': 'C0', 'Naive': 'C1', 'ACI': 'C2', 'AgACI': 'C3'}
    markers = {'OSCP': 'o', 'Naive': 's', 'ACI': '^', 'AgACI': 'D'}

    for method_name, (coverage, length) in plot_dict.items():
        color = colors.get(method_name, None)
        marker = markers.get(method_name, 'o')
        plt.scatter(length, coverage, s=200, marker=marker, color=color,
                   label=method_name, edgecolor='black', linewidth=1.5, zorder=3)

    # Add target coverage line
    plt.axhline(0.9, color='gray', linestyle='--', linewidth=1,
               label='Target coverage (90%)', zorder=1)

    # Annotate regions - FIX: "Ideal" is TOP-LEFT, not top-right
    ax = plt.gca()
    ax.text(0.05, 0.95, 'IDEAL\n(high coverage\n+ short intervals)',
           transform=ax.transAxes, ha='left', va='top', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.4, edgecolor='darkgreen', linewidth=2))
    ax.text(0.95, 0.05, 'Poor\n(low coverage\n+ long intervals)',
           transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.4, edgecolor='darkred', linewidth=2))
    ax.text(0.95, 0.95, 'Naive\n(high coverage but\nuninformative intervals)',
           transform=ax.transAxes, ha='right', va='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4, edgecolor='orange', linewidth=1.5))

    plt.xlabel(xlabel + xlabel_note, fontsize=11)
    plt.ylabel('Coverage Rate', fontsize=11)
    plt.title('Coverage vs Interval Length Trade-off', fontsize=13, fontweight='bold')
    plt.legend(loc='center left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])

    # Set x-axis limits with some padding
    if plot_dict:
        lengths = [length for _, length in plot_dict.values()]
        x_min, x_max = min(lengths), max(lengths)
        x_range = x_max - x_min
        plt.xlim([max(0, x_min - 0.1 * x_range), x_max + 0.1 * x_range])

    plt.tight_layout()

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


def plot_individual_switch_trajectories(
    intervals_dict: dict,
    y_true: np.ndarray,
    d_argmax: np.ndarray,
    window_before: int = 5,
    window_after: int = 5,
    save_path: Optional[str] = None,
    cut_at_next_switch: bool = True
):
    """
    Plot individual switch trajectory timelines (before averaging).

    Show each switch's coverage timeline separately.

    Parameters
    ----------
    intervals_dict : dict
        Dictionary with method names and (lower, upper) bounds
    y_true : np.ndarray
        Ground truth values
    d_argmax : np.ndarray
        Regime indicators
    window_before, window_after : int
        Window sizes
    save_path : str, optional
        Path to save figure
    cut_at_next_switch : bool
        Whether to cut windows at next switch
    """
    switches = detect_regime_switches(d_argmax)

    if len(switches) == 0:
        print("No regime switches detected")
        return

    # Choose one method to display (e.g., AgACI)
    method_name = 'AgACI' if 'AgACI' in intervals_dict else list(intervals_dict.keys())[0]
    lower, upper = intervals_dict[method_name]

    covered = (y_true >= lower) & (y_true <= upper)
    aligned_coverage, valid, actual_lengths = align_to_switches(
        covered.astype(float), switches, window_before, window_after, cut_at_next_switch
    )

    # Plot each valid switch
    valid_switches = switches[valid]
    n_valid = len(valid_switches)

    if n_valid == 0:
        print("No valid switches to plot")
        return

    # Create subplots
    n_cols = min(5, n_valid)
    n_rows = (n_valid + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    axes = axes.flatten()

    plot_idx = 0
    for i, (switch_idx, traj, is_valid, actual_len) in enumerate(zip(switches, aligned_coverage, valid, actual_lengths)):
        if not is_valid:
            continue

        ax = axes[plot_idx]
        time_axis = np.arange(-window_before, -window_before + actual_len)

        # Plot coverage (1 = covered, 0 = not covered)
        ax.plot(time_axis, traj, 'o-', markersize=4, linewidth=1.5)
        ax.axvline(0, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Switch')
        ax.axhline(0.9, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Target')

        ax.set_xlabel('Time relative to switch', fontsize=9)
        ax.set_ylabel('Coverage', fontsize=9)
        ax.set_title(f'Switch at t={switch_idx}', fontsize=10)
        ax.set_ylim([-0.1, 1.1])
        ax.grid(True, alpha=0.3)

        if plot_idx == 0:
            ax.legend(fontsize=8)

        plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Individual Switch Trajectories: {method_name}\n(Each subplot shows coverage around one regime switch)',
                fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
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


def compute_recovery_metrics(
    intervals_dict: dict,
    y_true: np.ndarray,
    d_argmax: np.ndarray,
    target_coverage: float = 0.9,
    recovery_threshold: float = 0.85,
    window_size: int = 5
):
    """
    Compute recovery time metrics for each method after regime switches.

    For each switch, measure how long it takes for coverage to recover above
    a threshold value (e.g., 85% of target coverage).

    Parameters
    ----------
    intervals_dict : dict
        Dictionary with method names and (lower, upper) bounds
    y_true : np.ndarray
        Ground truth values
    d_argmax : np.ndarray
        Regime indicators
    target_coverage : float
        Target coverage rate (default 0.9 for 90%)
    recovery_threshold : float
        Threshold for considering coverage "recovered" (default 0.85 = 85% of target)
    window_size : int
        Window size for computing rolling coverage (default 10)

    Returns
    -------
    metrics : dict
        Dictionary with method names and recovery statistics:
        - 'mean_recovery_time': Average time to recover
        - 'median_recovery_time': Median time to recover
        - 'recovery_times': List of recovery times for each switch
        - 'failed_recoveries': Number of switches that didn't recover within window
    """
    switches = detect_regime_switches(d_argmax)

    if len(switches) == 0:
        print("No regime switches detected")
        return {}

    threshold = target_coverage * recovery_threshold
    metrics = {}

    print(f"\n{'='*80}")
    print(f"COVERAGE RECOVERY ANALYSIS")
    print(f"{'='*80}")
    print(f"Target coverage: {target_coverage:.1%}")
    print(f"Recovery threshold: {threshold:.1%} ({recovery_threshold:.0%} of target)")
    print(f"Window size: {window_size}")
    print(f"Number of switches: {len(switches)}")
    print(f"{'='*80}\n")

    for method_name, (lower, upper) in intervals_dict.items():
        # Compute binary coverage
        covered = (y_true >= lower) & (y_true <= upper)
        valid_mask = ~np.isnan(lower) & ~np.isnan(upper)

        recovery_times = []
        failed_recoveries = 0

        for switch_idx in switches:
            # Look at window after switch
            start_idx = switch_idx
            end_idx = min(switch_idx + window_size * 3, len(covered))  # Look up to 3x window

            if start_idx >= len(covered):
                continue

            # Compute rolling coverage after switch
            recovered = False
            recovery_time = None

            for t in range(start_idx, end_idx):
                # Compute coverage in rolling window [t, t+window_size]
                window_end = min(t + window_size, len(covered))
                window_mask = valid_mask[t:window_end]

                if np.sum(window_mask) == 0:
                    continue

                window_coverage = np.mean(covered[t:window_end][window_mask])

                if window_coverage >= threshold:
                    recovery_time = t - start_idx
                    recovered = True
                    break

            if recovered:
                recovery_times.append(recovery_time)
            else:
                failed_recoveries += 1

        # Compute statistics
        if len(recovery_times) > 0:
            mean_recovery = np.mean(recovery_times)
            median_recovery = np.median(recovery_times)
        else:
            mean_recovery = np.nan
            median_recovery = np.nan

        metrics[method_name] = {
            'mean_recovery_time': mean_recovery,
            'median_recovery_time': median_recovery,
            'recovery_times': recovery_times,
            'failed_recoveries': failed_recoveries,
            'n_switches': len(switches),
            'recovery_rate': len(recovery_times) / len(switches) if len(switches) > 0 else 0
        }

        # Print summary
        print(f"{method_name}:")
        print(f"  Mean recovery time: {mean_recovery:.1f} steps")
        print(f"  Median recovery time: {median_recovery:.1f} steps")
        print(f"  Recovery rate: {len(recovery_times)}/{len(switches)} "
              f"({100*len(recovery_times)/len(switches):.1f}%)")
        print(f"  Failed recoveries: {failed_recoveries}")
        print()

    return metrics


def plot_length_at_switches(
    intervals_dict: dict,
    y_true: np.ndarray,
    d_argmax: np.ndarray,
    window_before: Optional[int] = None,
    window_after: Optional[int] = None,
    save_path: Optional[str] = None,
    adaptive_window: bool = False,
    show_error_bars: bool = True,
    cut_at_next_switch: bool = True,
    normalize: bool = True
):
    """
    Plot median interval length around regime switches for different methods.

    Similar to plot_coverage_at_switches but shows interval length instead of coverage.
    This helps understand how different gamma values adjust interval width in response
    to regime switches.

    Parameters
    ----------
    intervals_dict : dict
        Dictionary with keys as method names and values as (lower_bounds, upper_bounds)
    y_true : np.ndarray, shape (T,)
        Ground truth values (not used but kept for API consistency)
    d_argmax : np.ndarray, shape (T,)
        Regime indicators
    window_before : int, optional
        Window size before switches
    window_after : int, optional
        Window size after switches
    save_path : str, optional
        Path to save figure
    adaptive_window : bool
        If True, compute adaptive window sizes
    show_error_bars : bool
        If True, show standard error bars
    cut_at_next_switch : bool
        If True, truncate trajectories at next switch
    normalize : bool
        If True, normalize interval lengths as percentage of widest method (default=True)
    """
    switches = detect_regime_switches(d_argmax)

    # Compute adaptive window if not provided
    if adaptive_window and (window_before is None or window_after is None):
        computed_before, computed_after = compute_adaptive_window(d_argmax, percentile=50)
        if window_before is None:
            window_before = computed_before
        if window_after is None:
            window_after = computed_after
        print(f"\n[LENGTH PLOT] Using adaptive window: before={window_before}, after={window_after}")
    else:
        if window_before is None:
            window_before = 5
        if window_after is None:
            window_after = 5

    if len(switches) == 0:
        print("No regime switches detected")
        return

    print(f"\n[LENGTH PLOT] Found {len(switches)} regime switches")
    print(f"Data length: {len(d_argmax)}, Window: [{window_before}, {window_after}]")

    plt.figure(figsize=(10, 6))
    time_axis = np.arange(-window_before, window_after + 1)

    # Define colors
    base_colors = {'DS3M': 'C4', 'Naive': 'C0', 'ACI': 'C2', 'AgACI': 'C3'}

    # Separate ACI methods by gamma
    aci_gamma_methods = {k: v for k, v in intervals_dict.items() if 'ACI (γ=' in k}

    # Assign colors to ACI gamma methods
    if aci_gamma_methods:
        n_gamma = len(aci_gamma_methods)
        gamma_cmap = plt.cm.viridis(np.linspace(0.2, 0.9, n_gamma))
        gamma_colors = {name: gamma_cmap[i] for i, name in enumerate(aci_gamma_methods.keys())}
    else:
        gamma_colors = {}

    colors = {**base_colors, **gamma_colors}

    print(f"Plotting interval length for methods: {list(intervals_dict.keys())}")

    # First pass: compute all lengths and find maximum for normalization
    all_lengths = {}
    max_length = 0
    for method_name, (lower, upper) in intervals_dict.items():
        lengths = upper - lower
        all_lengths[method_name] = lengths
        method_max = np.nanmax(lengths)
        if method_max > max_length:
            max_length = method_max

    if normalize and max_length > 0:
        print(f"Normalizing interval lengths relative to max={max_length:.4f}")

    for method_name, (lower, upper) in intervals_dict.items():
        print(f"Processing method: {method_name}")

        # Get precomputed interval lengths
        lengths = all_lengths[method_name]

        # Normalize if requested
        if normalize and max_length > 0:
            lengths = (lengths / max_length) * 100.0  # Convert to percentage
            print(f"  Normalized lengths (first 5): {lengths[:5]}")

        # Check for NaN
        n_nan = np.sum(np.isnan(lengths))
        if n_nan > 0:
            print(f"Warning: {method_name} has {n_nan} NaN values")

        # Align lengths to switches with cutting
        aligned_lengths, valid, actual_lengths = align_to_switches(
            lengths, switches, window_before, window_after, cut_at_next_switch
        )
        n_valid_switches = np.sum(valid)

        print(f"  {method_name}: {n_valid_switches}/{len(valid)} valid switches")

        if n_valid_switches == 0:
            print(f"Warning: No valid switches for {method_name}, skipping")
            continue
        print(normalize and f"  Max length (normalized): {np.nanmax(lengths):.4f}")
        print(f"  Actual lengths of aligned trajectories (first 5): {actual_lengths[:5]}")
        # Pad aligned lengths to common length
        max_len = window_before + window_after + 1
        padded_lengths = []
        for traj, is_valid in zip(aligned_lengths, valid):
            if is_valid:
                padded = np.full(max_len, np.nan)
                padded[:len(traj)] = traj
                padded_lengths.append(padded)

        padded_lengths = np.array(padded_lengths)

        # Compute mean and std
        mean_length = np.nanmean(padded_lengths, axis=0)
        std_length = np.nanstd(padded_lengths, axis=0)
        n_contributors = np.sum(~np.isnan(padded_lengths), axis=0)
        stderr_length = std_length / np.sqrt(np.maximum(n_contributors, 1))

        time_axis = np.arange(-window_before, window_after + 1)
        valid_time_mask = n_contributors > 0

        color = colors.get(method_name, None)

        # Line styles
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

        plt.plot(time_axis[valid_time_mask], mean_length[valid_time_mask],
                label=method_name, linewidth=linewidth,
                color=color, linestyle=linestyle)

        # Add error bars
        if show_error_bars:
            plt.fill_between(time_axis[valid_time_mask],
                           mean_length[valid_time_mask] - stderr_length[valid_time_mask],
                           mean_length[valid_time_mask] + stderr_length[valid_time_mask],
                           alpha=0.2, color=color)

    # Mark switch point
    plt.axvline(0, color='r', linestyle='--', linewidth=2, label='Regime switch')

    plt.xlabel('Time relative to switch', fontsize=11)
    if normalize:
        plt.ylabel('Relative Interval Length (%)', fontsize=11)
        plt.title('Interval Length Dynamics Around Regime Switches\n'
                 'Normalized as percentage of widest method',
                 fontsize=12, fontweight='bold')
    else:
        plt.ylabel('Interval length', fontsize=11)
        plt.title('Interval Length Dynamics Around Regime Switches\n'
                 'Shows how different gamma values adjust interval width',
                 fontsize=12, fontweight='bold')
    plt.legend(loc='best', fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_recovery_comparison(
    recovery_metrics: dict,
    save_path: Optional[str] = None
):
    """
    Visualize recovery time comparison across methods with enhanced details.

    Creates a 4-panel visualization showing:
    - Recovery time vs gamma (with error bars)
    - Recovery rate comparison
    - Box plot of recovery time distributions
    - Summary table

    Parameters
    ----------
    recovery_metrics : dict
        Output from compute_recovery_metrics()
    save_path : str, optional
        Path to save figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)

    # Separate ACI gamma methods from others
    aci_gamma_methods = {}
    other_methods = {}

    for method_name, metrics in recovery_metrics.items():
        if 'ACI (γ=' in method_name:
            gamma_str = method_name.split('γ=')[1].rstrip(')')
            aci_gamma_methods[float(gamma_str)] = (method_name, metrics)
        else:
            other_methods[method_name] = metrics

    # ========== Panel 1: Recovery time vs gamma with error bars ==========
    ax1 = fig.add_subplot(gs[0, 0])
    if aci_gamma_methods:
        gamma_values = sorted(aci_gamma_methods.keys())
        mean_times = []
        median_times = []
        std_times = []

        for gamma in gamma_values:
            _, metrics = aci_gamma_methods[gamma]
            mean_times.append(metrics['mean_recovery_time'])
            median_times.append(metrics['median_recovery_time'])
            if len(metrics['recovery_times']) > 0:
                std_times.append(np.std(metrics['recovery_times']))
            else:
                std_times.append(0)

        mean_times = np.array(mean_times)
        std_times = np.array(std_times)

        ax1.errorbar(gamma_values, mean_times, yerr=std_times,
                    fmt='o-', linewidth=2, markersize=8, capsize=5,
                    label='Mean ± Std', color='C0')
        ax1.plot(gamma_values, median_times, 's--', linewidth=2, markersize=8,
                label='Median', color='C1', alpha=0.7)

        ax1.set_xlabel('Gamma value (learning rate)', fontsize=11)
        ax1.set_ylabel('Recovery time (steps)', fontsize=11)
        ax1.set_title('Recovery Time vs Gamma\n(Lower is better - faster adaptation)',
                     fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')

        # Add value labels
        for i, (g, m) in enumerate(zip(gamma_values, mean_times)):
            ax1.text(g, m, f'{m:.1f}', ha='left', va='bottom', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No ACI gamma methods found',
                ha='center', va='center', transform=ax1.transAxes)

    # ========== Panel 2: Recovery rate bar chart ==========
    ax2 = fig.add_subplot(gs[0, 1])

    method_names = []
    recovery_rates = []
    colors = []

    # Add other methods first
    for method_name, metrics in other_methods.items():
        method_names.append(method_name)
        recovery_rates.append(metrics['recovery_rate'] * 100)
        if 'AgACI' in method_name:
            colors.append('C3')
        elif 'Naive' in method_name:
            colors.append('C0')
        else:
            colors.append('C2')

    # Add ACI gamma methods
    if aci_gamma_methods:
        gamma_cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(aci_gamma_methods)))
        for i, (gamma, (method_name, metrics)) in enumerate(sorted(aci_gamma_methods.items())):
            method_names.append(f'γ={gamma:.4f}')
            recovery_rates.append(metrics['recovery_rate'] * 100)
            colors.append(gamma_cmap[i])

    x_pos = np.arange(len(method_names))
    bars = ax2.bar(x_pos, recovery_rates, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, recovery_rates)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)

    ax2.set_xlabel('Method', fontsize=11)
    ax2.set_ylabel('Recovery success rate (%)', fontsize=11)
    ax2.set_title('Coverage Recovery Success Rate\n(Higher is better)',
                 fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(method_names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylim([0, 105])
    ax2.axhline(100, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Target: 100%')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()

    # ========== Panel 3: Box plot of recovery time distributions ==========
    ax3 = fig.add_subplot(gs[1, :])

    if aci_gamma_methods:
        recovery_time_data = []
        box_labels = []
        box_colors = []

        # Collect data
        gamma_cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(aci_gamma_methods)))
        for i, (gamma, (method_name, metrics)) in enumerate(sorted(aci_gamma_methods.items())):
            if len(metrics['recovery_times']) > 0:
                recovery_time_data.append(metrics['recovery_times'])
                box_labels.append(f'γ={gamma:.4f}')
                box_colors.append(gamma_cmap[i])

        # Create box plot
        bp = ax3.boxplot(recovery_time_data, labels=box_labels, patch_artist=True,
                        showmeans=True, meanline=True,
                        boxprops=dict(alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        meanprops=dict(color='blue', linestyle='--', linewidth=2))

        # Color boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)

        ax3.set_xlabel('Method (Gamma value)', fontsize=11)
        ax3.set_ylabel('Recovery time (steps)', fontsize=11)
        ax3.set_title('Distribution of Recovery Times\n(Red line = median, Blue dashed = mean)',
                     fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_xticks(range(1, len(box_labels) + 1))
        ax3.set_xticklabels(box_labels, rotation=45, ha='right')

    # ========== Panel 4: Summary table ==========
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    # Prepare table data
    if aci_gamma_methods:
        table_data = []
        headers = ['Gamma', 'Mean (steps)', 'Median (steps)', 'Std (steps)',
                  'Success Rate', 'Total Switches', 'Failed']

        for gamma, (method_name, metrics) in sorted(aci_gamma_methods.items()):
            if len(metrics['recovery_times']) > 0:
                mean_t = np.mean(metrics['recovery_times'])
                median_t = np.median(metrics['recovery_times'])
                std_t = np.std(metrics['recovery_times'])
            else:
                mean_t = median_t = std_t = 0

            row = [
                f'{gamma:.4f}',
                f'{mean_t:.1f}',
                f'{median_t:.1f}',
                f'{std_t:.1f}',
                f'{metrics["recovery_rate"]*100:.1f}%',
                f'{metrics["n_switches"]}',
                f'{metrics["failed_recoveries"]}'
            ]
            table_data.append(row)

        table = ax4.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Color header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color rows alternately
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

    plt.suptitle('Comprehensive Recovery Time Analysis',
                fontsize=14, fontweight='bold', y=0.995)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()
