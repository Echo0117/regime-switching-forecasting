"""
Standalone test for regime switch visualization.
Tests the plotting functions with synthetic data (no DS3M/torch required).
"""

import numpy as np
import os
import sys

HERE = os.path.dirname(__file__)
PROJ = os.path.abspath(os.path.join(HERE, ".."))
for p in [HERE, PROJ]:
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.utils.regime_switch_analysis import (
    plot_agaci_weights_at_switches,
    plot_coverage_at_switches,
    plot_coverage_vs_length_tradeoff
)


def generate_synthetic_data(n_timesteps=1000, d_dim=2, n_switches=5, n_gammas=5):
    """
    Generate synthetic data for testing visualization.

    Parameters
    ----------
    n_timesteps : int
        Total length of time series
    d_dim : int
        Number of regimes
    n_switches : int
        Number of regime switches to create
    n_gammas : int
        Number of gamma values (experts)

    Returns
    -------
    dict with synthetic data
    """
    np.random.seed(42)

    # Create regime sequence with n_switches
    d_argmax = np.zeros(n_timesteps, dtype=int)
    switch_points = np.sort(np.random.choice(
        np.arange(100, n_timesteps - 100),
        size=n_switches,
        replace=False
    ))

    # Assign regimes
    for i, switch_idx in enumerate(switch_points):
        if i + 1 < len(switch_points):
            d_argmax[switch_idx:switch_points[i+1]] = (i + 1) % d_dim
        else:
            d_argmax[switch_idx:] = (i + 1) % d_dim

    # Generate synthetic ground truth
    y_true = np.sin(np.linspace(0, 10, n_timesteps)) + 0.1 * np.random.randn(n_timesteps)

    # Add regime-dependent noise
    for regime in range(d_dim):
        mask = d_argmax == regime
        y_true[mask] += regime * 0.5  # Offset for each regime

    # Generate AgACI weights (evolving over time, changing at switches)
    gamma_values = [0.0025, 0.005, 0.01, 0.02, 0.05][:n_gammas]
    agaci_weights = np.zeros((n_gammas, n_timesteps))

    for gamma_idx in range(n_gammas):
        # Base weight decreases with gamma
        base_weight = 1.0 / (gamma_idx + 1)
        weights = np.full(n_timesteps, base_weight)

        # Add dynamics around switches
        for switch_idx in switch_points:
            # Weight increases before switch (detecting change)
            if switch_idx > 50:
                weights[switch_idx-50:switch_idx] += 0.1 * np.linspace(0, 1, 50)
            # Weight drops after switch (recalibrating)
            if switch_idx + 100 < n_timesteps:
                weights[switch_idx:switch_idx+100] *= np.linspace(0.5, 1.0, 100)

        # Normalize
        agaci_weights[gamma_idx] = weights / weights.sum() * n_timesteps

    # Generate prediction intervals
    test_len = n_timesteps // 2
    test_start = n_timesteps - test_len

    # Method 1: Narrow but low coverage
    lower_narrow = y_true[test_start:] - 0.3
    upper_narrow = y_true[test_start:] + 0.3

    # Method 2: Wide but high coverage
    lower_wide = y_true[test_start:] - 1.5
    upper_wide = y_true[test_start:] + 1.5

    # Method 3: Adaptive (changes at switches)
    lower_adaptive = y_true[test_start:] - 0.5
    upper_adaptive = y_true[test_start:] + 0.5
    for switch_idx in switch_points:
        if switch_idx >= test_start:
            local_idx = switch_idx - test_start
            # Widen intervals around switches
            if local_idx > 0:
                window = slice(max(0, local_idx-20), min(test_len, local_idx+20))
                lower_adaptive[window] -= 0.3
                upper_adaptive[window] += 0.3

    return {
        'd_argmax': d_argmax,
        'y_true': y_true[test_start:],
        'agaci_weights': agaci_weights,
        'gamma_values': gamma_values,
        'switch_points': switch_points,
        'intervals': {
            'Narrow': (lower_narrow, upper_narrow),
            'Wide': (lower_wide, upper_wide),
            'Adaptive': (lower_adaptive, upper_adaptive),
        }
    }


def main():
    print("="*60)
    print("Testing Regime Switch Visualization (Standalone)")
    print("="*60)

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    data = generate_synthetic_data(
        n_timesteps=1000,
        d_dim=2,  # 2 regimes as per user's request
        n_switches=5,
        n_gammas=5
    )

    print(f"  Total timesteps: {len(data['d_argmax'])}")
    print(f"  Number of switches: {len(data['switch_points'])}")
    print(f"  Switch locations: {data['switch_points']}")
    print(f"  Number of gammas: {len(data['gamma_values'])}")
    print(f"  Gamma values: {data['gamma_values']}")

    # Setup output directory
    save_dir = "figures/regime_test_standalone"
    os.makedirs(save_dir, exist_ok=True)

    # Test 1: Plot regime heatmap
    print("\n1. Plotting regime heatmap...")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 2))
    regime_map = data['d_argmax'].reshape(1, -1)
    im = ax.imshow(regime_map, aspect='auto', cmap='tab10', interpolation='nearest')

    # Mark regime switches
    for switch_idx in data['switch_points']:
        ax.axvline(switch_idx, color='red', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_xlabel('Time (t)', fontsize=11)
    ax.set_ylabel('Regime', fontsize=11)
    ax.set_title(f'Regime Switches (d_dim=2, {len(data["switch_points"])} switches)',
                 fontsize=12, fontweight='bold')
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, label='Regime ID', orientation='vertical')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/regime_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"   Saved: {save_dir}/regime_heatmap.png")
    plt.close()

    # Test 2: Plot AgACI weights at switches
    print("\n2. Plotting AgACI weights at regime switches...")
    plot_agaci_weights_at_switches(
        agaci_weights=data['agaci_weights'],
        d_argmax=data['d_argmax'],
        gamma_values=data['gamma_values'],
        window_before=10,
        window_after=50,
        save_path=f"{save_dir}/agaci_weights_switches.png",
        show_individual_lines=True  # Show each switch as separate line
    )

    # Test 3: Plot coverage at switches
    print("\n3. Plotting coverage at regime switches...")
    plot_coverage_at_switches(
        intervals_dict=data['intervals'],
        y_true=data['y_true'],
        d_argmax=data['d_argmax'][500:],  # Test portion only
        window_before=10,
        window_after=50,
        save_path=f"{save_dir}/coverage_at_switches.png"
    )

    # Test 4: Plot coverage vs length tradeoff
    print("\n4. Plotting coverage vs length tradeoff...")
    results_dict = {}
    for method_name, (lower, upper) in data['intervals'].items():
        coverage = np.mean((data['y_true'] >= lower) & (data['y_true'] <= upper))
        median_length = np.median(upper - lower)
        results_dict[method_name] = (coverage, median_length)
        print(f"   {method_name}: Coverage={coverage:.3f}, MedianLength={median_length:.2f}")

    plot_coverage_vs_length_tradeoff(
        results_dict=results_dict,
        save_path=f"{save_dir}/tradeoff.png"
    )

    print("\n" + "="*60)
    print(f"All plots saved to: {save_dir}/")
    print("="*60)
    print("\nVisualization")
    print("  ✓ d_dim=2 (2 regimes)")
    print("  ✓ Each subplot = one gamma value")
    print("  ✓ Each colored line = one regime switch")
    print("  ✓ X-axis = time relative to switch")
    print("  ✓ Y-axis = weight for that gamma")
    print("  ✓ Red dashed line = regime switch point (t=0)")
    print("  ✓ Black line = mean across all switches")


if __name__ == "__main__":
    main()
