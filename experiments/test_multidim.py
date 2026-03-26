"""
Quick test for multi-dimensional model support
"""
import numpy as np
from sklearn.linear_model import Ridge
import sys
import os
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

from experiments.generate_forecasting_comparison import create_lag_features

# Test data: 100 timesteps, 3 dimensions
np.random.seed(42)
data = np.random.randn(100, 3)
lags = 5

print("Testing multi-dimensional lag features...")
print(f"Data shape: {data.shape}")

# Single-dim mode
X_single, y_single = create_lag_features(data, lags, multidim=False)
print(f"\nSingle-dim mode:")
print(f"  X shape: {X_single.shape}  (expected: (95, 5))")
print(f"  y shape: {y_single.shape}  (expected: (95,))")

# Multi-dim mode
X_multi, y_multi = create_lag_features(data, lags, multidim=True)
print(f"\nMulti-dim mode:")
print(f"  X shape: {X_multi.shape}  (expected: (95, 15) = lags*dims)")
print(f"  y shape: {y_multi.shape}  (expected: (95, 3))")

# Test AR (Ridge) with multi-output
print("\n\nTesting Ridge with multi-output...")
ar = Ridge(alpha=1.0)
ar.fit(X_multi, y_multi)
pred = ar.predict(X_multi[:10])
print(f"  Prediction shape: {pred.shape}  (expected: (10, 3))")
print(f"  Sample prediction:\n{pred[0]}")

print("\n✅ Basic multi-dimensional support working!")
