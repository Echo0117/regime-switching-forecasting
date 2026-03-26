"""
Run multi-dimensional evaluation on all datasets and generate comparison figures.
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

# All datasets (ordered by type)
DATASETS = [
    # Synthetic
    "Toy",
    "Lorenz",
    # Real-world
    "Sleep",
    "Unemployment",
    "Electricity",
    "Hangzhou",
    "Seattle",
]

# Dataset dimensions (for reference)
DATASET_DIMS = {
    "Toy": 1,
    "Lorenz": 10,
    "Sleep": 1,
    "Unemployment": 1,
    "Electricity": 8,
    "Hangzhou": 73,
    "Seattle": 166,
}

# Models to compare
MODELS = ["AR", "S4", "MCD", "GP", "CPD", "DS3M"]

def run_all_datasets():
    """Run multi-dimensional evaluation on all datasets."""
    print("="*80)
    print("Multi-Dimensional Evaluation - All Datasets")
    print("="*80)

    all_results = []

    for dataname in DATASETS:
        print(f"\n{'='*80}")
        print(f"Processing: {dataname} (dims={DATASET_DIMS.get(dataname, '?')})")
        print(f"{'='*80}")

        try:
            # Run with multi-dimensional evaluation
            results = run_comparison(
                dataname=dataname,
                device="cpu",
                verbose=True,
                use_multidim=True,
                ds3m_force_new=False,
            )

            if results is not None:
                # Extract metrics
                row = {
                    'Dataset': dataname,
                    'Dimensions': DATASET_DIMS.get(dataname, 1),
                }

                for model in MODELS:
                    if model in results['metrics']:
                        metrics = results['metrics'][model]
                        row[f'{model}_RMSE'] = metrics.get('RMSE', np.nan)
                        row[f'{model}_MAE'] = metrics.get('MAE', np.nan)
                        row[f'{model}_R2'] = metrics.get('R2', np.nan)

                        # For DS3M, also store full-dim RMSE if available
                        if model == 'DS3M' and 'RMSE_full_dim' in metrics:
                            row['DS3M_RMSE_full_dim'] = metrics['RMSE_full_dim']
                    else:
                        row[f'{model}_RMSE'] = np.nan
                        row[f'{model}_MAE'] = np.nan
                        row[f'{model}_R2'] = np.nan

                all_results.append(row)

                print(f"\n✅ {dataname} completed")
            else:
                print(f"\n❌ {dataname} failed")

        except Exception as e:
            print(f"\n❌ {dataname} error: {e}")
            import traceback
            traceback.print_exc()

    return all_results


def create_rmse_comparison_table(all_results):
    """Create RMSE comparison table."""
    df = pd.DataFrame(all_results)

    # Create RMSE-only table
    rmse_cols = ['Dataset', 'Dimensions'] + [f'{m}_RMSE' for m in MODELS]
    rmse_df = df[rmse_cols].copy()

    # Rename columns for cleaner display
    rmse_df.columns = ['Dataset', 'Dims'] + MODELS

    # Save to CSV
    csv_path = OUTPUT_DIR / "multidim_rmse_comparison.csv"
    rmse_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n✅ Saved CSV: {csv_path}")

    # Print formatted table
    print("\n" + "="*80)
    print("Multi-Dimensional RMSE Comparison (averaged across dimensions)")
    print("="*80)
    print(rmse_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print("="*80)

    return rmse_df


def create_rmse_heatmap(rmse_df):
    """Create RMSE heatmap visualization."""
    # Prepare data for heatmap (datasets x models)
    datasets = rmse_df['Dataset'].values
    data_matrix = rmse_df[MODELS].values

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use log scale for better visualization (RMSE can vary widely)
    data_log = np.log10(data_matrix + 1e-6)  # Add small epsilon to avoid log(0)

    # Create heatmap
    im = ax.imshow(data_log, cmap='RdYlGn_r', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels(MODELS, fontsize=12, fontweight='bold')
    ax.set_yticklabels(datasets, fontsize=11)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add colorbar with original scale labels
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log₁₀(RMSE)', rotation=270, labelpad=20, fontsize=11)

    # Add text annotations with actual RMSE values
    for i in range(len(datasets)):
        for j in range(len(MODELS)):
            val = data_matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if data_log[i, j] < np.nanmedian(data_log) else 'black'
                ax.text(j, i, f'{val:.3f}', ha="center", va="center",
                       color=text_color, fontsize=9, fontweight='bold')

    ax.set_title('Multi-Dimensional RMSE Comparison\n(Lower is Better)',
                 fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "multidim_rmse_heatmap.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved heatmap: {save_path}")


def create_model_ranking_plot(rmse_df):
    """Create bar plot showing average ranking of each model."""
    # Calculate ranks per dataset (1 = best, lower is better)
    ranks_matrix = rmse_df[MODELS].rank(axis=1, method='average')

    # Average rank per model
    avg_ranks = ranks_matrix.mean(axis=0)
    avg_ranks_sorted = avg_ranks.sort_values()

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c', '#95a5a6']
    bars = ax.barh(avg_ranks_sorted.index, avg_ranks_sorted.values, color=colors[:len(avg_ranks_sorted)])

    # Add value labels
    for i, (model, rank) in enumerate(avg_ranks_sorted.items()):
        ax.text(rank + 0.1, i, f'{rank:.2f}', va='center', fontweight='bold', fontsize=11)

    ax.set_xlabel('Average Rank (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Ranking Across All Datasets\n(Multi-Dimensional RMSE)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim([0, len(MODELS) + 0.5])
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()

    plt.tight_layout()
    save_path = OUTPUT_DIR / "multidim_model_ranking.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved ranking plot: {save_path}")


def create_dataset_difficulty_plot(rmse_df):
    """Create plot showing dataset difficulty (median RMSE across models)."""
    # Calculate median RMSE per dataset
    median_rmse = rmse_df[MODELS].median(axis=1)

    # Create dataframe with dataset info
    difficulty_df = pd.DataFrame({
        'Dataset': rmse_df['Dataset'],
        'Dimensions': rmse_df['Dims'],
        'Median_RMSE': median_rmse
    }).sort_values('Median_RMSE')

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Color by number of dimensions
    colors = ['#3498db' if d == 1 else '#e74c3c' if d <= 10 else '#9b59b6'
              for d in difficulty_df['Dimensions']]

    bars = ax.barh(difficulty_df['Dataset'], difficulty_df['Median_RMSE'], color=colors)

    # Add dimension labels
    for i, (dataset, dims, rmse) in enumerate(zip(difficulty_df['Dataset'],
                                                    difficulty_df['Dimensions'],
                                                    difficulty_df['Median_RMSE'])):
        ax.text(rmse + rmse*0.05, i, f'({dims}D)', va='center', fontsize=9)

    ax.set_xlabel('Median RMSE Across Models', fontsize=12, fontweight='bold')
    ax.set_title('Dataset Difficulty Ranking\n(Lower RMSE = Easier to Predict)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='1D'),
        Patch(facecolor='#e74c3c', label='2-10D'),
        Patch(facecolor='#9b59b6', label='>10D')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "multidim_dataset_difficulty.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved difficulty plot: {save_path}")


def main():
    print("Starting multi-dimensional comparison...")
    print(f"Output directory: {OUTPUT_DIR}")

    # Run evaluations
    all_results = run_all_datasets()

    if not all_results:
        print("\n❌ No results collected!")
        return

    # Create comparison table
    rmse_df = create_rmse_comparison_table(all_results)

    # Create visualizations
    print("\nGenerating visualizations...")
    create_rmse_heatmap(rmse_df)
    create_model_ranking_plot(rmse_df)
    create_dataset_difficulty_plot(rmse_df)

    print("\n" + "="*80)
    print(f"✅ Done! All results saved to: {OUTPUT_DIR}")
    print("="*80)
    print("\nGenerated files:")
    print(f"  1. {OUTPUT_DIR / 'multidim_rmse_comparison.csv'}")
    print(f"  2. {OUTPUT_DIR / 'multidim_rmse_heatmap.png'}")
    print(f"  3. {OUTPUT_DIR / 'multidim_model_ranking.png'}")
    print(f"  4. {OUTPUT_DIR / 'multidim_dataset_difficulty.png'}")
    print("="*80)


if __name__ == "__main__":
    main()
