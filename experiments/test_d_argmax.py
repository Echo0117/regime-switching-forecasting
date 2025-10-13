#!/usr/bin/env python
"""Quick test to check d_argmax values for Unemployment dataset."""

import sys
import argparse
sys.path.insert(0, '/Users/pr059704/Library/CloudStorage/OneDrive-PERNODRICARD/Documents/phd/code/regime-switching-forecasting')

import numpy as np
import torch
from experiments.utils.ds3m_utils import load_ds3m_data, get_full_d_argmax

# Create args object
parser = argparse.ArgumentParser()
parser.add_argument('--problem', default='Unemployment')
parser.add_argument('--lags', type=int, default=6)
parser.add_argument('--device', default='cpu')
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args([])
args.problem = 'Unemployment'
args.lags = 6
args.device = 'cpu'

# Load data
ds = load_ds3m_data(args)

print(f"Dataset d_dim: {ds['d_dim']}")
print(f"Train shape: {ds['trainX'].shape}")
print(f"Valid shape: {ds['validX'].shape}")
print(f"Test shape: {ds['testX'].shape}")

# Load model
model_path = "Deep_Switching_State_Space_Model/trained_model/Unemployment"
model = torch.load(model_path, map_location="cpu")
print(f"\nModel loaded from: {model_path}")

# Get d_argmax for full dataset
d_argmax_full = get_full_d_argmax(model, ds)

print(f"\nd_argmax_full stats:")
print(f"  Shape: {d_argmax_full.shape}")
print(f"  Type: {type(d_argmax_full)}, dtype: {d_argmax_full.dtype}")
print(f"  Min: {d_argmax_full.min()}, Max: {d_argmax_full.max()}")
print(f"  Unique values: {np.unique(d_argmax_full)}")
print(f"  Number of unique values: {len(np.unique(d_argmax_full))}")

# Check if there's a mismatch
if d_argmax_full.max() >= ds['d_dim']:
    print(f"\n⚠️  WARNING: d_argmax_full contains values >= d_dim!")
    print(f"  Expected range: [0, {ds['d_dim']-1}]")
    print(f"  Actual range: [{d_argmax_full.min()}, {d_argmax_full.max()}]")
