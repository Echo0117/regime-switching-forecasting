"""
Test multi-dimensional evaluation on Task 1 (forecasting comparison)
"""
import sys
import os
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

from experiments.generate_forecasting_comparison import run_comparison

print("="*70)
print("Testing multi-dimensional evaluation on Lorenz dataset")
print("="*70)

# Run with multi-dimensional mode on Lorenz (3 dimensions)
results = run_comparison(
    dataname="Lorenz",
    device="cpu",
    verbose=True,
    use_multidim=True,
    ds3m_force_new=False,
)

if results is not None:
    print("\n" + "="*70)
    print("Results Summary (Multi-dimensional evaluation)")
    print("="*70)
    for model_name, metrics in results['metrics'].items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
else:
    print("\nTest failed!")
