"""
Tune Ruptures hyperparameters on Toy dataset to minimize MAE.
"""
import os
import sys
import numpy as np
from pathlib import Path

HERE = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from experiments.utils.experiments_utils import load_forecast
import ruptures as rpt

def cpd_precision_recall_f1(true_switches, pred_switches, tol=5):
    """Compute precision/recall/F1/MAE for changepoint detection."""
    true_switches = np.asarray(true_switches, dtype=int)
    pred_switches = np.asarray(pred_switches, dtype=int)
    
    if true_switches.size == 0 or pred_switches.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mae": np.nan}
    
    # Greedy matching
    used_true = np.zeros(len(true_switches), dtype=bool)
    tp = 0
    matched_errors = []
    
    for p in pred_switches:
        distances = np.abs(true_switches - p)
        distances[used_true] = 10**9
        j = int(np.argmin(distances))
        
        if distances[j] <= tol:
            used_true[j] = True
            tp += 1
            matched_errors.append(distances[j])
    
    fp = len(pred_switches) - tp
    fn = len(true_switches) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mae = float(np.mean(matched_errors)) if matched_errors else np.nan
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "mae": mae
    }

def _switches_from_labels(labels):
    """Return switch indices (where regime label changes)."""
    labels = np.asarray(labels).astype(int)
    return np.where(np.diff(labels) != 0)[0] + 1

# Load Toy data
print("Loading Toy dataset...")
cached = load_forecast("Toy")
full_series_1d = cached['data'][:, 0] if cached['data'].ndim > 1 else cached['data']
test_len = cached['test_len']

# Load ground truth
gt_path = Path("Deep_Switching_State_Space_Model/data/Toy_og/simulation_data_nonlinear_d.csv")
d_true = np.loadtxt(gt_path).astype(int)
d_true_test = d_true[-test_len:]
gt_switches = _switches_from_labels(d_true_test)

print(f"Full series: {len(full_series_1d)} points")
print(f"Test length: {test_len}")
print(f"True switches in test: {len(gt_switches)}")

# Normalize using training data statistics
train_data = full_series_1d[:-test_len]
mean, std = train_data.mean(), train_data.std()
data_norm = (full_series_1d - mean) / std

# Get test start index
test_start = len(full_series_1d) - test_len

print("\n" + "="*70)
print("Testing different Binseg configurations")
print("="*70)

best_mae = float('inf')
best_config = None

# Test different n_bkps values
for n_bkps in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
    for min_size in [3, 5, 7, 10]:
        try:
            algo = rpt.Binseg(model='l2', min_size=min_size).fit(data_norm)
            bkps = algo.predict(n_bkps=n_bkps)
            
            # Extract switches in test period
            test_bkps = [bp for bp in bkps[:-1] if bp > test_start]
            test_switches = np.array([bp - test_start for bp in test_bkps])
            
            # Compute metrics
            metrics = cpd_precision_recall_f1(gt_switches, test_switches, tol=5)
            
            if not np.isnan(metrics['mae']) and metrics['mae'] < best_mae:
                best_mae = metrics['mae']
                best_config = {
                    'method': 'Binseg',
                    'n_bkps': n_bkps,
                    'min_size': min_size,
                    'metrics': metrics,
                    'n_detected': len(test_switches)
                }
            
            print(f"n_bkps={n_bkps:3d}, min_size={min_size:2d}: "
                  f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
                  f"F1={metrics['f1']:.3f}, MAE={metrics['mae']:.2f}, "
                  f"detected={len(test_switches)}")
        except Exception as e:
            print(f"n_bkps={n_bkps:3d}, min_size={min_size:2d}: Failed - {e}")

print("\n" + "="*70)
print("Testing different Pelt configurations")
print("="*70)

for penalty in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
    for min_size in [3, 5, 7, 10]:
        try:
            algo = rpt.Pelt(model='l2', min_size=min_size).fit(data_norm)
            bkps = algo.predict(pen=penalty)
            
            # Extract switches in test period
            test_bkps = [bp for bp in bkps[:-1] if bp > test_start]
            test_switches = np.array([bp - test_start for bp in test_bkps])
            
            # Compute metrics
            metrics = cpd_precision_recall_f1(gt_switches, test_switches, tol=5)
            
            if not np.isnan(metrics['mae']) and metrics['mae'] < best_mae:
                best_mae = metrics['mae']
                best_config = {
                    'method': 'Pelt',
                    'penalty': penalty,
                    'min_size': min_size,
                    'metrics': metrics,
                    'n_detected': len(test_switches)
                }
            
            print(f"pen={penalty:5.1f}, min_size={min_size:2d}: "
                  f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
                  f"F1={metrics['f1']:.3f}, MAE={metrics['mae']:.2f}, "
                  f"detected={len(test_switches)}")
        except Exception as e:
            print(f"pen={penalty:5.1f}, min_size={min_size:2d}: Failed - {e}")

print("\n" + "="*70)
print("BEST CONFIGURATION")
print("="*70)
print(f"Method: {best_config['method']}")
if 'n_bkps' in best_config:
    print(f"n_bkps: {best_config['n_bkps']}")
    print(f"min_size: {best_config['min_size']}")
else:
    print(f"penalty: {best_config['penalty']}")
    print(f"min_size: {best_config['min_size']}")
print(f"\nMetrics:")
print(f"  Precision: {best_config['metrics']['precision']:.4f}")
print(f"  Recall: {best_config['metrics']['recall']:.4f}")
print(f"  F1: {best_config['metrics']['f1']:.4f}")
print(f"  MAE: {best_config['metrics']['mae']:.4f}")
print(f"  Detected switches: {best_config['n_detected']} (true: {len(gt_switches)})")
print("="*70)

# Compare with current DS3M
print("\n" + "="*70)
print("DS3M Current Performance")
print("="*70)
d_argmax = cached['d_argmax']
ds3m_switches = _switches_from_labels(d_argmax)
ds3m_metrics = cpd_precision_recall_f1(gt_switches, ds3m_switches, tol=5)
print(f"Precision: {ds3m_metrics['precision']:.4f}")
print(f"Recall: {ds3m_metrics['recall']:.4f}")
print(f"F1: {ds3m_metrics['f1']:.4f}")
print(f"MAE: {ds3m_metrics['mae']:.4f}")
print(f"Detected switches: {len(ds3m_switches)} (true: {len(gt_switches)})")
print("="*70)

print(f"\nImprovement: {ds3m_metrics['mae'] - best_mae:.4f} reduction in MAE")
