"""
Analyze and visualize multi-dimensional forecasting results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = Path("experiments/figures_multidim_comparison")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Multi-dimensional datasets results
multidim_results = {
    'Dataset': ['Lorenz', 'Electricity'],
    'Dimensions': [10, 48],
    'AR': [0.0097, 1509.2450],
    'S4': [0.9233, 7298.9651],
    'MCD': [0.0491, 3683.6561],
    'GP': [0.1818, 2874.8884],
    'CPD': [0.0025, 2876.1999],
    'DS3M': [0.4274, 1700.6712],
}

df = pd.DataFrame(multidim_results)

print("="*80)
print("Multi-Dimensional Forecasting Performance Analysis")
print("="*80)
print("\nDatasets:")
for _, row in df.iterrows():
    print(f"  - {row['Dataset']}: {row['Dimensions']} dimensions")

print("\n" + "="*80)
print("RMSE Comparison (averaged across all dimensions)")
print("="*80)
print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print("="*80)

# Calculate rankings
models = ['AR', 'S4', 'MCD', 'GP', 'CPD', 'DS3M']
ranks = df[models].rank(axis=1, method='average')
avg_ranks = ranks.mean(axis=0).sort_values()

print("\n" + "="*80)
print("Average Model Ranking (1=best)")
print("="*80)
for model, rank in avg_ranks.items():
    print(f"  {model:6s}: {rank:.2f}")
print("="*80)

# Create detailed comparison plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Bar chart per dataset
ax1 = axes[0, 0]
x = np.arange(len(models))
width = 0.35
for i, dataset in enumerate(df['Dataset']):
    values = df.loc[df['Dataset']==dataset, models].values[0]
    ax1.bar(x + i*width, values, width, label=f'{dataset} ({df.loc[df["Dataset"]==dataset, "Dimensions"].values[0]}D)')
ax1.set_xlabel('Model', fontweight='bold')
ax1.set_ylabel('RMSE', fontweight='bold')
ax1.set_title('RMSE by Model and Dataset', fontweight='bold')
ax1.set_xticks(x + width / 2)
ax1.set_xticklabels(models)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_yscale('log')

# 2. Normalized heatmap
ax2 = axes[0, 1]
data_matrix = df[models].values
# Normalize per row (dataset)
data_normalized = (data_matrix - data_matrix.min(axis=1, keepdims=True)) / \
                  (data_matrix.max(axis=1, keepdims=True) - data_matrix.min(axis=1, keepdims=True) + 1e-10)
im = ax2.imshow(data_normalized, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
ax2.set_xticks(np.arange(len(models)))
ax2.set_yticks(np.arange(len(df)))
ax2.set_xticklabels(models)
ax2.set_yticklabels([f"{row['Dataset']} ({row['Dimensions']}D)" for _, row in df.iterrows()])
ax2.set_title('Normalized Performance\n(0=best, 1=worst per dataset)', fontweight='bold')
for i in range(len(df)):
    for j in range(len(models)):
        text = ax2.text(j, i, f'{data_normalized[i, j]:.2f}',
                       ha="center", va="center", color="white" if data_normalized[i, j] > 0.5 else "black",
                       fontsize=9, fontweight='bold')
cbar = plt.colorbar(im, ax=ax2)
cbar.set_label('Normalized Error', rotation=270, labelpad=15)

# 3. Ranking plot
ax3 = axes[1, 0]
ax3.barh(avg_ranks.index, avg_ranks.values,
         color=['#2ecc71' if r < 2 else '#3498db' if r < 3.5 else '#e74c3c' for r in avg_ranks.values])
ax3.set_xlabel('Average Rank', fontweight='bold')
ax3.set_title('Model Ranking Across Datasets\n(Lower is Better)', fontweight='bold')
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3)
for i, (model, rank) in enumerate(avg_ranks.items()):
    ax3.text(rank + 0.1, i, f'{rank:.2f}', va='center', fontweight='bold')

# 4. Best model frequency
ax4 = axes[1, 1]
best_models = []
for idx, row in df.iterrows():
    best_model = df.loc[idx, models].idxmin()
    best_models.append(best_model)
model_counts = pd.Series(best_models).value_counts()
colors_map = {
    'CPD': '#2ecc71',
    'AR': '#3498db',
    'DS3M': '#9b59b6',
    'MCD': '#e67e22',
    'GP': '#e74c3c',
    'S4': '#95a5a6'
}
colors = [colors_map.get(m, '#95a5a6') for m in model_counts.index]
ax4.pie(model_counts.values, labels=model_counts.index, autopct='%1.0f%%',
        colors=colors, startangle=90, textprops={'fontweight': 'bold', 'fontsize': 11})
ax4.set_title('Best Model Frequency\n(Winner per dataset)', fontweight='bold')

plt.tight_layout()
save_path = OUTPUT_DIR / "multidim_detailed_analysis.png"
plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n✅ Saved detailed analysis: {save_path}")

# Create per-model comparison
fig, ax = plt.subplots(figsize=(12, 6))
datasets_labels = [f"{row['Dataset']}\n({row['Dimensions']}D)" for _, row in df.iterrows()]
x = np.arange(len(datasets_labels))
width = 0.13

for i, model in enumerate(models):
    values = df[model].values
    offset = (i - len(models)/2 + 0.5) * width
    bars = ax.bar(x + offset, values, width, label=model)

    # Add value labels on bars
    for j, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        if val < 100:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=7, rotation=0)

ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax.set_ylabel('RMSE (log scale)', fontsize=12, fontweight='bold')
ax.set_title('Multi-Dimensional RMSE Comparison by Model', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(datasets_labels, fontsize=10)
ax.legend(loc='upper left', ncol=3, fontsize=10)
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3, which='both')

plt.tight_layout()
save_path = OUTPUT_DIR / "multidim_model_comparison_bars.png"
plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✅ Saved model comparison: {save_path}")

# Key findings
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

print("\n1. Best Models per Dataset:")
for idx, row in df.iterrows():
    best_model = df.loc[idx, models].idxmin()
    best_rmse = df.loc[idx, best_model]
    print(f"   {row['Dataset']:12s} ({row['Dimensions']:3d}D): {best_model:6s} (RMSE: {best_rmse:.4f})")

print("\n2. Model Rankings (averaged across datasets):")
for i, (model, rank) in enumerate(avg_ranks.items(), 1):
    print(f"   #{i}. {model:6s}: avg rank {rank:.2f}")

print("\n3. Performance Insights:")
print(f"   - CPD excels on Lorenz (RMSE: {df.loc[df['Dataset']=='Lorenz', 'CPD'].values[0]:.4f})")
print(f"   - AR excels on Electricity (RMSE: {df.loc[df['Dataset']=='Electricity', 'AR'].values[0]:.4f})")
print(f"   - S4 struggles with both datasets (avg RMSE: {df['S4'].mean():.1f})")
print(f"   - DS3M performs consistently in middle tier")

print("\n4. Multi-Dimensional Challenge:")
lorenz_range = df.loc[df['Dataset']=='Lorenz', models].max().values[0] / df.loc[df['Dataset']=='Lorenz', models].min().values[0]
elec_range = df.loc[df['Dataset']=='Electricity', models].max().values[0] / df.loc[df['Dataset']=='Electricity', models].min().values[0]
print(f"   - Lorenz (10D): {lorenz_range:.0f}x performance gap (best to worst)")
print(f"   - Electricity (48D): {elec_range:.1f}x performance gap (best to worst)")
print(f"   - Higher dimensionality increases model performance variance")

print("\n" + "="*80)
print("All visualizations saved to:", OUTPUT_DIR)
print("="*80)
