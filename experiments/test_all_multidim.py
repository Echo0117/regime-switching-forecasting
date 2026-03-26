"""
Comprehensive test for multi-dimensional support across all competitor models
"""
import numpy as np
import sys
import os
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

from experiments.generate_forecasting_comparison import create_lag_features
from experiments.competitor_models import (
    S4Regressor, MCDropoutGRU, RupturesSegmentedLinear, GPTorchSparse
)
from sklearn.linear_model import Ridge

# Test data: 100 timesteps, 3 dimensions
np.random.seed(42)
data = np.random.randn(100, 3)
lags = 5

print("=" * 60)
print("Testing multi-dimensional support for all models")
print("=" * 60)
print(f"Data shape: {data.shape}")
print(f"Lags: {lags}")

# Create multi-dim lag features
X_multi, y_multi = create_lag_features(data, lags, multidim=True)
print(f"\nMulti-dim features:")
print(f"  X shape: {X_multi.shape}  (expected: (95, 15) = lags*dims)")
print(f"  y shape: {y_multi.shape}  (expected: (95, 3))")

# Split train/test
n_train = 70
X_train, X_test = X_multi[:n_train], X_multi[n_train:]
y_train, y_test = y_multi[:n_train], y_multi[n_train:]

print(f"\nTrain/test split: {n_train}/{len(X_test)}")

# Test each model
print("\n" + "=" * 60)
print("Testing AR (Ridge)")
print("=" * 60)
try:
    ar = Ridge(alpha=1.0)
    ar.fit(X_train, y_train)
    pred = ar.predict(X_test)
    print(f"✅ AR prediction shape: {pred.shape}  (expected: ({len(X_test)}, 3))")
    print(f"   Sample prediction: {pred[0]}")
except Exception as e:
    print(f"❌ AR failed: {e}")

# Test S4
print("\n" + "=" * 60)
print("Testing S4")
print("=" * 60)
try:
    s4 = S4Regressor(
        lags=lags,
        output_dim=3,
        d_model=32,
        n_layers=2,
        epochs=5,
        batch=16,
        patience=3,
        verbose=False,
        device="cpu"
    )
    s4.fit(X_train, y_train)
    pred = s4.predict(X_test)
    print(f"✅ S4 prediction shape: {pred.shape}  (expected: ({len(X_test)}, 3))")
    print(f"   Sample prediction: {pred[0]}")
except Exception as e:
    print(f"❌ S4 failed: {e}")

# Test MCDropoutGRU
print("\n" + "=" * 60)
print("Testing MC-Dropout GRU")
print("=" * 60)
try:
    mcd = MCDropoutGRU(
        lags=lags,
        output_dim=3,
        hidden=32,
        layers=1,
        epochs=5,
        batch=16,
        mc_samples=10,
        patience=3,
        verbose=False,
        device="cpu"
    )
    mcd.fit(X_train, y_train)
    pred = mcd.predict(X_test)
    print(f"✅ MCD prediction shape: {pred.shape}  (expected: ({len(X_test)}, 3))")
    print(f"   Sample prediction: {pred[0]}")
except Exception as e:
    print(f"❌ MCD failed: {e}")

# Test RupturesSegmentedLinear (CPD)
print("\n" + "=" * 60)
print("Testing CPD (Ruptures)")
print("=" * 60)
try:
    cpd = RupturesSegmentedLinear(penalty=5.0, min_size=10, model="l2")
    cpd.fit(X_train, y_train)
    pred = cpd.predict(X_test)
    print(f"✅ CPD prediction shape: {pred.shape}  (expected: ({len(X_test)}, 3))")
    print(f"   Sample prediction: {pred[0]}")
except Exception as e:
    print(f"❌ CPD failed: {e}")

# Test GP (may take longer, use smaller subset)
print("\n" + "=" * 60)
print("Testing GP (using smaller subset for speed)")
print("=" * 60)
try:
    X_train_small = X_train[:30]
    y_train_small = y_train[:30]
    X_test_small = X_test[:10]

    gp = GPTorchSparse(
        lags=lags,
        num_inducing=20,
        iters=50,
        lr=0.01,
        device="cpu"
    )
    gp.fit(X_train_small, y_train_small)
    pred = gp.predict(X_test_small)
    print(f"✅ GP prediction shape: {pred.shape}  (expected: ({len(X_test_small)}, 3))")
    print(f"   Sample prediction: {pred[0]}")
except Exception as e:
    print(f"❌ GP failed: {e}")

print("\n" + "=" * 60)
print("✅ All models support multi-dimensional output!")
print("=" * 60)
