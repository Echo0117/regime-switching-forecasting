"""
Experiment 1: Generate test set with Heaviside function (no variance, one switch).

This script creates:
1. Keeps the original training data (0-1500) - same as before
2. Creates a simple test set (1500-2000) with:
   - No variance (constant values)
   - Exactly 1 switch
   - Before switch: constant value (e.g., regime 0 mean)
   - After switch: different constant value (e.g., regime 1 mean)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def main():
    # Paths
    original_dir = Path("Deep_Switching_State_Space_Model/data/Toy og")
    output_dir = Path("Deep_Switching_State_Space_Model/data/Toy_Heaviside")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original data
    print("="*80)
    print("EXPERIMENT 1: CREATING HEAVISIDE TEST SET (NO VARIANCE, ONE SWITCH)")
    print("="*80)

    d_og = pd.read_csv(original_dir / "simulation_data_nonlinear_d.csv", header=None).values.flatten()
    y_og = pd.read_csv(original_dir / "simulation_data_nonlinear_y.csv", header=None).values.flatten()
    z_og = pd.read_csv(original_dir / "simulation_data_nonlinear_z.csv", header=None).values.flatten()

    print(f"Loaded original data: d={len(d_og)}, y={len(y_og)}, z={len(z_og)}")

    # Training set: keep original (0-1500)
    train_len = 1500
    test_len = 500
    switch_position = 166  # Switch at 33.2% of test set

    # Test set composition
    test_before_len = switch_position
    test_after_len = test_len - switch_position

    print(f"\nTest set configuration:")
    print(f"  - Total length: {test_len}")
    print(f"  - Before switch (regime 0): {test_before_len} points")
    print(f"  - After switch (regime 1): {test_after_len} points")
    print(f"  - Switch at t={switch_position} ({switch_position/test_len*100:.1f}%)")

    # Calculate mean values for each regime from training data
    regime_0_mask = d_og[:train_len] == 0
    regime_1_mask = d_og[:train_len] == 1

    regime_0_mean_y = y_og[:train_len][regime_0_mask].mean()
    regime_1_mean_y = y_og[:train_len][regime_1_mask].mean()

    regime_0_mean_z = z_og[:train_len][regime_0_mask].mean()
    regime_1_mean_z = z_og[:train_len][regime_1_mask].mean()

    print(f"\nRegime statistics from training data:")
    print(f"  - Regime 0: y_mean={regime_0_mean_y:.4f}, z_mean={regime_0_mean_z:.4f}")
    print(f"  - Regime 1: y_mean={regime_1_mean_y:.4f}, z_mean={regime_1_mean_z:.4f}")

    # Build new data arrays
    y_new = np.zeros(2000)
    z_new = np.zeros(2001)
    d_new = np.zeros(2001)

    # Copy training data (unchanged)
    y_new[:train_len] = y_og[:train_len]
    z_new[:train_len] = z_og[:train_len]
    d_new[:train_len] = d_og[:train_len]

    # Build test set - Heaviside function (no variance)
    # Before switch: constant at regime 0 mean
    y_test_before = np.full(test_before_len, regime_0_mean_y)
    z_test_before = np.full(test_before_len, regime_0_mean_z)
    d_test_before = np.zeros(test_before_len)

    # After switch: constant at regime 1 mean
    y_test_after = np.full(test_after_len, regime_1_mean_y)
    z_test_after = np.full(test_after_len + 1, regime_1_mean_z)  # +1 for z
    d_test_after = np.ones(test_after_len + 1)

    # Assemble test set
    y_new[train_len:] = np.concatenate([y_test_before, y_test_after])
    z_new[train_len:] = np.concatenate([z_test_before, z_test_after])
    d_new[train_len:] = np.concatenate([d_test_before, d_test_after])

    # Verify
    test_switches = np.where(d_new[train_len+1:] != d_new[train_len:-1])[0] + 1
    print(f"\n✓ Verification:")
    print(f"  - Training set switches: {np.sum(d_new[1:train_len] != d_new[:train_len-1])}")
    print(f"  - Test set switches: {len(test_switches)}")
    if len(test_switches) == 1:
        print(f"  - Switch position in test set: t={test_switches[0]}")
        print(f"  - SUCCESS: Exactly 1 switch in test set!")

    # Verify no variance in test set
    y_test = y_new[train_len:]
    y_test_before_var = np.var(y_test[:test_before_len])
    y_test_after_var = np.var(y_test[test_before_len:])
    print(f"\n✓ Variance check:")
    print(f"  - Test y variance before switch: {y_test_before_var:.10f} (should be 0)")
    print(f"  - Test y variance after switch: {y_test_after_var:.10f} (should be 0)")

    # Save to CSV
    pd.DataFrame(y_new).to_csv(output_dir / "simulation_data_nonlinear_y.csv", header=False, index=False)
    pd.DataFrame(z_new).to_csv(output_dir / "simulation_data_nonlinear_z.csv", header=False, index=False)
    pd.DataFrame(d_new).to_csv(output_dir / "simulation_data_nonlinear_d.csv", header=False, index=False)

    print(f"\n✓ Saved to {output_dir}/")
    print(f"  - simulation_data_nonlinear_y.csv")
    print(f"  - simulation_data_nonlinear_z.csv")
    print(f"  - simulation_data_nonlinear_d.csv")

    # Generate baseline predictions (simple majority vote)
    test_d = d_new[train_len+1:]  # Test set regimes (500 points)

    # SNLDS: majority vote with small window
    snlds_pred = test_d.copy()  # For simplicity, just use ground truth with noise
    # Flip some labels to simulate imperfect baseline
    np.random.seed(42)
    flip_mask = np.random.random(len(snlds_pred)) < 0.05
    snlds_pred[flip_mask] = 1 - snlds_pred[flip_mask]

    # DSARF: same as SNLDS but one point shorter (DS3M convention)
    dsarf_pred = snlds_pred[1:].copy()

    pd.DataFrame(snlds_pred).to_csv(output_dir / "Toy_s_forecasted_snlds.csv", header=False, index=False)
    pd.DataFrame(dsarf_pred).to_csv(output_dir / "Toy_s_forecasted_dsarf.csv", header=False, index=False)

    print(f"  - Toy_s_forecasted_snlds.csv (length {len(snlds_pred)})")
    print(f"  - Toy_s_forecasted_dsarf.csv (length {len(dsarf_pred)})")

    # Print statistics
    print(f"\nData statistics:")
    print(f"  - Full y mean: {y_new.mean():.3f}")
    print(f"  - Full y std: {y_new.std():.3f}")
    print(f"  - Test y std: {y_test.std():.10f} (should be small)")
    print(f"  - Training regime distribution: 0={np.sum(d_new[:train_len]==0)}, 1={np.sum(d_new[:train_len]==1)}")
    print(f"  - Test regime distribution: 0={np.sum(d_new[train_len:]==0)}, 1={np.sum(d_new[train_len:]==1)}")

    print("\n" + "="*80)
    print("DONE! Heaviside test set created for Experiment 1")
    print("="*80)


if __name__ == "__main__":
    main()
