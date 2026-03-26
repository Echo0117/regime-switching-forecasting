"""
Trace DS3M forecast indices to find the exact alignment issue.
This script adds detailed debugging to understand the index mapping.
"""
import sys
import os
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

def trace_seattle_indices():
    """
    Trace Seattle dataset indices step by step to find the misalignment.
    """
    print("="*70)
    print("TRACING SEATTLE INDICES")
    print("="*70)

    # Load Seattle data
    from experiments.utils.ds3m_utils import load_ds3m_data

    args_dict = {
        'dataname': 'Seattle',
        'device': 'cpu',
        'seed': 42,
    }

    # Create a simple namespace object
    class Args:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
        def __getattr__(self, name):
            return None  # Return None for missing attributes

    args = Args(args_dict)

    print("\n1. Loading DS3M data...")
    ds = load_ds3m_data(args)

    print(f"\n2. Dataset info:")
    print(f"   RawDataOriginal shape: {ds['RawDataOriginal'].shape}")
    print(f"   test_len: {ds['test_len']}")
    print(f"   freq: {ds.get('freq', 'N/A')}")
    print(f"   timestep: {ds.get('timestep', 'N/A')}")
    print(f"   remove_residual: {ds.get('remove_residual', False)}")

    # Calculate indices
    raw_shape = ds['RawDataOriginal'].shape
    if len(raw_shape) == 3:
        total_len = raw_shape[0] * raw_shape[1]
        print(f"   Total length (flattened): {total_len}")
        test_actual_len = int(ds['test_len'] / ds.get('freq', 1)) * ds.get('freq', 1)
        print(f"   Test actual length: {test_actual_len}")
        test_start_idx = total_len - test_actual_len
        test_end_idx = total_len
        print(f"   testOriginal would span: [{test_start_idx}, {test_end_idx})")
    else:
        total_len = raw_shape[0]
        test_start_idx = total_len - ds['test_len']
        test_end_idx = total_len
        print(f"   testOriginal would span: [{test_start_idx}, {test_end_idx})")

    # Check if trend exists
    if ds.get('trend') is not None:
        print(f"\n3. Residual transformation applied:")
        print(f"   trend shape: {ds['trend'].shape}")
        print(f"   trend contains: y[0] to y[{ds['trend'].shape[0]-1}]")

        # Calculate trend_tail indices in current code
        test_len = ds['test_len']
        trend = ds['trend']
        print(f"\n4. Current trend_tail calculation:")
        print(f"   Code: trend[-test_len-1:-1]")
        print(f"   Indices: trend[{len(trend)-test_len-1}:{len(trend)-1}]")
        print(f"   This is: y[{len(trend)-test_len-1}] to y[{len(trend)-2}]")

        print(f"\n5. Correct trend_tail should be:")
        print(f"   For testOriginal [{test_start_idx}, {test_end_idx})")
        print(f"   Need trend: y[{test_start_idx-1}] to y[{test_end_idx-2}]")
        print(f"   Indices: trend[{test_start_idx-1}:{test_end_idx-1}]")

        current_start = len(trend) - test_len - 1
        current_end = len(trend) - 1
        correct_start = test_start_idx - 1
        correct_end = test_end_idx - 1

        offset = current_start - correct_start
        print(f"\n6. Offset calculation:")
        print(f"   Current start: {current_start}")
        print(f"   Correct start: {correct_start}")
        print(f"   Offset: {offset} steps")

    # Check test_data shapes
    if 'test_data_Y' in ds:
        print(f"\n7. Test data info:")
        print(f"   test_data_Y shape: {ds['test_data_Y'].shape}")
        print(f"   test_data_X shape: {ds['test_data_X'].shape}")

    print("\n" + "="*70)

def trace_electricity_indices():
    """
    Trace Electricity dataset indices.
    """
    print("\n" + "="*70)
    print("TRACING ELECTRICITY INDICES")
    print("="*70)

    from experiments.utils.ds3m_utils import load_ds3m_data

    class Args:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    args = Args({'dataname': 'Electricity', 'device': 'cpu', 'seed': 42})

    print("\n1. Loading DS3M data...")
    ds = load_ds3m_data(args)

    print(f"\n2. Dataset info:")
    print(f"   RawDataOriginal shape: {ds['RawDataOriginal'].shape}")
    print(f"   test_len: {ds['test_len']}")
    print(f"   timestep: {ds.get('timestep', 'N/A')}")
    print(f"   remove_residual: {ds.get('remove_residual', False)}")

    raw_shape = ds['RawDataOriginal'].shape
    total_len = raw_shape[0]
    test_len = ds['test_len']
    timestep = ds.get('timestep', 0)

    print(f"   Total length: {total_len}")
    print(f"   Test indices: [{total_len - test_len}, {total_len})")

    print(f"\n3. Test data extraction:")
    print(f"   test_data_Y = data_Y[-test_len - timestep:]")
    print(f"   = data_Y[-{test_len + timestep}:]")
    print(f"   Length: {test_len + timestep}")

    print(f"\n4. Expected testForecast length: {test_len}")
    print(f"   If testForecast uses first {timestep} as init:")
    print(f"   Forecast would span: data_Y[{timestep}:{test_len + timestep}]")
    print(f"   Corresponding to original: y[{total_len - test_len}:{total_len}]")

    print(f"\n5. But testOriginal is:")
    print(f"   RawDataOriginal[-test_len:] = y[{total_len - test_len}:{total_len}]")

    print(f"\n6. Potential offset:")
    print(f"   If DS3M's forecast actually corresponds to a different window,")
    print(f"   we get misalignment.")

    print("\n" + "="*70)

if __name__ == "__main__":
    trace_seattle_indices()
    trace_electricity_indices()
