"""
Quick multi-dimensional evaluation on key datasets.
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

from experiments.generate_forecasting_comparison import run_comparison

# Output directory
OUTPUT_DIR = Path("experiments/figures_multidim_comparison")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Key datasets for quick comparison
DATASETS = [
    "Toy",        # 1D synthetic
    "Lorenz",     # 10D synthetic
    "Sleep",      # 1D real
    "Electricity", # 8D real
]

DATASET_DIMS = {
    "Toy": 1,
    "Lorenz": 10,
    "Sleep": 1,
    "Electricity": 8,
}

MODELS = ["AR", "S4", "MCD", "GP", "CPD", "DS3M"]

def run_quick_comparison():
    """Run multi-dimensional evaluation on key datasets."""
    print("="*80)
    print("Quick Multi-Dimensional Evaluation")
    print("="*80)

    all_results = []

    for dataname in DATASETS:
        print(f"\n{'='*80}")
        print(f"Processing: {dataname} (dims={DATASET_DIMS.get(dataname, '?')})")
        print(f"{'='*80}")

        try:
            results = run_comparison(
                dataname=dataname,
                device="cpu",
                verbose=True,
                use_multidim=True,
                ds3m_force_new=False,
            )

            if results is not None:
                row = {
                    'Dataset': dataname,
                    'Dimensions': DATASET_DIMS.get(dataname, 1),
                }

                for model in MODELS:
                    if model in results['metrics']:
                        metrics = results['metrics'][model]
                        row[f'{model}_RMSE'] = metrics.get('RMSE', np.nan)
                        if model == 'DS3M' and 'RMSE_full_dim' in metrics:
                            row['DS3M_RMSE_full_dim'] = metrics['RMSE_full_dim']
                    else:
                        row[f'{model}_RMSE'] = np.nan

                all_results.append(row)
                print(f"\n✅ {dataname} completed")

        except Exception as e:
            print(f"\n❌ {dataname} error: {e}")
            import traceback
            traceback.print_exc()

    return all_results


def create_comparison_table(all_results):
    """Create and display comparison table."""
    df = pd.DataFrame(all_results)
    rmse_cols = ['Dataset', 'Dimensions'] + [f'{m}_RMSE' for m in MODELS]
    rmse_df = df[rmse_cols].copy()
    rmse_df.columns = ['Dataset', 'Dims'] + MODELS

    # Save to CSV
    csv_path = OUTPUT_DIR / "multidim_rmse_quick.csv"
    rmse_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n✅ Saved: {csv_path}")

    # Print table
    print("\n" + "="*80)
    print("Multi-Dimensional RMSE Comparison")
    print("="*80)
    print(rmse_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print("="*80)

    return rmse_df


def create_quick_heatmap(rmse_df):
    """Create quick heatmap."""
    datasets = rmse_df['Dataset'].values
    data_matrix = rmse_df[MODELS].values

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use log scale for better visualization
    data_log = np.log10(data_matrix + 1e-6)
    im = ax.imshow(data_log, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels(MODELS, fontsize=12, fontweight='bold')
    ax.set_yticklabels(datasets, fontsize=11)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log₁₀(RMSE)', rotation=270, labelpad=20, fontsize=11)

    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(MODELS)):
            val = data_matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if data_log[i, j] < np.nanmedian(data_log) else 'black'
                ax.text(j, i, f'{val:.3f}', ha="center", va="center",
                       color=text_color, fontsize=10, fontweight='bold')

    ax.set_title('Multi-Dimensional RMSE Comparison (Quick)\n(Lower is Better)',
                 fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "multidim_rmse_heatmap_quick.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


def main():
    print("Starting quick multi-dimensional comparison...")
    print(f"Output directory: {OUTPUT_DIR}")

    all_results = run_quick_comparison()

    if not all_results:
        print("\n❌ No results collected!")
        return

    rmse_df = create_comparison_table(all_results)
    create_quick_heatmap(rmse_df)

    print("\n" + "="*80)
    print(f"✅ Done! Results saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
