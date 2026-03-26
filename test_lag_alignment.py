"""
Test to understand the lag feature alignment issue.
"""
import numpy as np

def create_lag_features_test(data, lags):
    """Simplified version of create_lag_features."""
    data = np.asarray(data).flatten()
    N = len(data) - lags
    X = np.zeros((N, lags))
    y = np.zeros(N)
    for i in range(N):
        X[i] = data[i:i+lags]
        y[i] = data[i+lags]
    return X, y

# Simple test data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
data = np.arange(10)
lags = 2

X, y = create_lag_features_test(data, lags)

print("Original data:", data)
print(f"\nWith lags={lags}:")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"\nX and y contents:")
for i in range(len(X)):
    print(f"X[{i}] = {X[i]} -> y[{i}] = {y[i]} (should be data[{i+lags}])")

# Now simulate train/test split
test_len = 2
train_end = len(y) - test_len

X_train, y_train = X[:train_end], y[:train_end]
X_test, y_test = X[train_end:], y[train_end:]

print(f"\n{'='*60}")
print(f"Train/Test split with test_len={test_len}, train_end={train_end}")
print(f"{'='*60}")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

print(f"\nTest data details:")
for i in range(len(X_test)):
    abs_idx = train_end + i
    print(f"X_test[{i}] = {X_test[i]} -> y_test[{i}] = {y_test[i]} (data[{abs_idx + lags}])")

print(f"\ny_test corresponds to original data indices: {[train_end + lags + i for i in range(len(y_test))]}")
print(f"y_test values: {y_test}")

# Simulate a perfect predictor
class PerfectPredictor:
    """Predicts exactly the target value."""
    def predict(self, X):
        # Given X[i] = [data[j], data[j+1]], predict data[j+lags]
        # For simplicity, just return the last value + 1
        # In reality, models learn the pattern
        return X[:, -1] + 1  # Next value after the last lag

predictor = PerfectPredictor()
pred = predictor.predict(X_test)

print(f"\n{'='*60}")
print("Perfect predictor test:")
print(f"{'='*60}")
print(f"Predictions: {pred}")
print(f"Ground truth (y_test): {y_test}")
print(f"Match? {np.allclose(pred, y_test)}")

# Now check what happens if we use the WRONG ground truth
print(f"\n{'='*60}")
print("What if models predict one step earlier?")
print(f"{'='*60}")

# If models actually predict data[j+lags-1] instead of data[j+lags]
class OffByOnePredictor:
    """Predicts one step earlier."""
    def predict(self, X):
        return X[:, -1]  # Returns last lag value, not next value

off_predictor = OffByOnePredictor()
pred_off = off_predictor.predict(X_test)

print(f"Off-by-one predictions: {pred_off}")
print(f"Ground truth (y_test): {y_test}")
print(f"Match? {np.allclose(pred_off, y_test)}")
print(f"\nIf we shift pred forward by 1:")
if len(pred_off) > 1:
    print(f"pred_off[1:] = {pred_off[1:]}")
    print(f"y_test[:-1] = {y_test[:-1]}")
    print(f"Better match? No, because pred_off itself is wrong")
