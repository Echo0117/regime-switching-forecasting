"""
Quick test to verify regime visualization integration works with real DS3M data.
"""
import sys
import os
import argparse

HERE = os.path.dirname(__file__)
PROJ = os.path.abspath(os.path.join(HERE, ".."))
for p in [HERE, PROJ]:
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.utils.ds3m_utils import load_ds3m_data, load_ds3m_model, get_full_d_argmax
from experiments.utils.regime_switch_analysis import detect_regime_switches
import matplotlib.pyplot as plt

def main():
    # Create minimal args
    class Args:
        problem = "Unemployment"
        seed = 42
        device = None

    args = Args()

    print("="*60)
    print("Testing Regime Visualization with Real DS3M Data")
    print("="*60)

    # Load data and model
    print("\nLoading DS3M data and model...")
    ds = load_ds3m_data(args)
    print(f"  d_dim: {ds['d_dim']}")
    print(f"  Data shape: {ds['data'].shape}")

    model = load_ds3m_model(
        ds["directoryBest"],
        ds["x_dim"], ds["y_dim"], ds["h_dim"], ds["z_dim"],
        ds["d_dim"], ds["n_layers"], ds["learning_rate"],
        ds["device"], bidirection=ds["bidirection"],
    )
    print("  Model loaded successfully!")

    # Get d_argmax for full dataset
    print("\nGetting regime indicators for full dataset...")
    d_argmax_full = get_full_d_argmax(model, ds)
    print(f"  d_argmax_full shape: {d_argmax_full.shape}")
    print(f"  Unique regimes: {sorted(set(d_argmax_full.tolist()))}")

    # Detect switches
    switches = detect_regime_switches(d_argmax_full)
    print(f"\n  Detected {len(switches)} regime switches")
    print(f"  Switch locations: {switches[:10]}..." if len(switches) > 10 else f"  Switch locations: {switches}")

    # Plot regime heatmap
    print("\nPlotting regime heatmap...")
    save_dir = "figures/regime_integration_test"
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 2))
    regime_map = d_argmax_full.reshape(1, -1)
    im = ax.imshow(regime_map, aspect='auto', cmap='tab10', interpolation='nearest')

    # Mark regime switches
    for switch_idx in switches:
        ax.axvline(switch_idx, color='red', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_xlabel('Time (t)', fontsize=11)
    ax.set_ylabel('Regime', fontsize=11)
    ax.set_title(f'{args.problem}: Regime Switches (d_dim={ds["d_dim"]}, {len(switches)} switches)',
                 fontsize=12, fontweight='bold')
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, label='Regime ID', orientation='vertical')
    plt.tight_layout()

    save_path = f"{save_dir}/regime_heatmap.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
