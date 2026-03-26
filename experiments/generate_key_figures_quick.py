"""
Quick generation of prediction figures for key datasets.
"""
import sys
import os
from pathlib import Path

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

from experiments.generate_forecasting_comparison import run_comparison, plot_professional_comparison

# Output directory
OUTPUT_DIR = Path("overleaf_upload/figures/task1_prediction_updated")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Key datasets for quick test
DATASETS = ["Toy", "Lorenz", "Sleep"]

def generate_figures():
    """Generate prediction figures for key datasets."""
    print("="*80)
    print("Quick Figure Generation - Key Datasets")
    print("="*80)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Datasets: {', '.join(DATASETS)}")
    print("="*80)

    for i, dataname in enumerate(DATASETS, 1):
        print(f"\n[{i}/{len(DATASETS)}] {dataname}...")
        print("-" * 70)

        try:
            results = run_comparison(
                dataname=dataname,
                device="cpu",
                verbose=True,
                use_multidim=False,
                ds3m_force_new=False,
            )

            if results:
                fig_path = plot_professional_comparison(results, dataname, plot_range=200)
                if fig_path:
                    import shutil
                    dest = OUTPUT_DIR / f"{dataname}_forecast_comparison.png"
                    shutil.copy(fig_path, dest)
                    print(f"✅ {dest}")
                else:
                    print(f"⚠️  Plot failed")
            else:
                print(f"⚠️  No results")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print(f"Done! Figures in: {OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    generate_figures()
