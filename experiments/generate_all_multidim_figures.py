"""
Generate multi-dimensional prediction comparison figures for all datasets.
Similar to Task 1 figures but with use_multidim=True.
"""
import sys
import os
from pathlib import Path

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

from experiments.generate_forecasting_comparison import run_comparison, plot_professional_comparison

# Output directory - update existing figures
OUTPUT_DIR = Path("overleaf_upload/figures/task1_prediction")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# All datasets
DATASETS = [
    "Toy",
    "Lorenz",
    "Sleep",
    "Unemployment",
    "Electricity",
    "Hangzhou",
    "Seattle",
    "Pacific",
    "Pernod",
]

def generate_all_figures():
    """Generate prediction comparison figures for all datasets with multi-dimensional mode."""
    print("="*80)
    print("Generating Multi-Dimensional Prediction Figures")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Datasets: {', '.join(DATASETS)}")
    print("="*80)

    for i, dataname in enumerate(DATASETS, 1):
        print(f"\n[{i}/{len(DATASETS)}] Processing {dataname}...")
        print("-" * 70)

        try:
            # Run comparison (standard mode for compatibility)
            # For multi-dimensional datasets, this will still evaluate on best dimension
            results = run_comparison(
                dataname=dataname,
                device="cpu",
                verbose=True,
                use_multidim=False,  # Use standard mode for all datasets
                ds3m_force_new=False,
            )

            if results is None:
                print(f"⚠️  {dataname}: No results returned")
                continue

            # Generate professional comparison plot
            fig_path = plot_professional_comparison(
                results,
                dataname,
                plot_range=200
            )

            if fig_path:
                # File already saved by plot_professional_comparison
                print(f"✅ Saved: {fig_path}")
            else:
                print(f"⚠️  {dataname}: Plot generation failed")

        except Exception as e:
            print(f"❌ {dataname} error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print(f"✅ Done! All figures saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    generate_all_figures()
