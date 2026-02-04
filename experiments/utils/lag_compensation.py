"""
Simple lag compensation utility.

Usage:
    from experiments.utils.lag_compensation import apply_lag_compensation

    # In any plotting function:
    d_argmax_compensated = apply_lag_compensation(d_argmax, dataset_name)
"""

import numpy as np


# Measured lags from cross-dataset experiments
DATASET_LAGS = {
    "Electricity": -8.0,
    "Sleep": -8.0,
    "Hangzhou": -7.0,
    "Seattle": -6.0,
    "Unemployment": -7.0,
    "Lorenz": -1.5,
}

# Recommended window sizes - unified to 5 for all datasets
# This provides consistent tight windows around regime switches
DATASET_WINDOWS = {
    "Electricity": 5,
    "Sleep": 5,
    "Hangzhou": 5,
    "Seattle": 5,
    "Unemployment": 5,
    "Lorenz": 5,
    "Pacific": 5,
}


def apply_lag_compensation(d_argmax, dataset_name=None, lag=None, inplace=False):
    """
    Apply lag compensation to regime indicators.

    This shifts regime indicators to align coverage minima with switch points.

    Parameters
    ----------
    d_argmax : np.ndarray
        Regime indicators (0, 1, 2, ...)
    dataset_name : str, optional
        Dataset name to look up measured lag. If None, uses 'lag' parameter.
    lag : float, optional
        Manual lag value. If None, looks up dataset_name in DATASET_LAGS.
    inplace : bool
        If True, modify array in place. Default False (return copy).

    Returns
    -------
    d_argmax_compensated : np.ndarray
        Lag-compensated regime indicators

    Examples
    --------
    >>> # Automatic compensation based on dataset
    >>> d_comp = apply_lag_compensation(d_argmax, "Electricity")

    >>> # Manual lag
    >>> d_comp = apply_lag_compensation(d_argmax, lag=-8)

    >>> # No compensation (returns copy)
    >>> d_comp = apply_lag_compensation(d_argmax, lag=0)
    """
    # Determine lag
    if lag is None:
        if dataset_name is None:
            # Default: no compensation
            lag = 0
        else:
            # Look up dataset lag
            lag = DATASET_LAGS.get(dataset_name, 0)

    # Convert to int for shift
    shift = -int(lag)

    if shift == 0:
        # No compensation needed
        return d_argmax if inplace else d_argmax.copy()

    # Apply shift
    if inplace:
        d_compensated = d_argmax
        d_compensated[:] = np.roll(d_argmax, shift)
    else:
        d_compensated = np.roll(d_argmax, shift)

    # Handle edge effects - preserve original values at boundaries
    if shift > 0:
        d_compensated[:shift] = d_argmax[:shift]
    elif shift < 0:
        d_compensated[shift:] = d_argmax[shift:]

    return d_compensated


def get_lag(dataset_name):
    """
    Get measured lag for a dataset.

    Parameters
    ----------
    dataset_name : str
        Dataset name

    Returns
    -------
    lag : float
        Measured lag (negative means coverage min before switch)
    """
    return DATASET_LAGS.get(dataset_name, 0)


def get_window(dataset_name):
    """
    Get recommended window size for a dataset.

    The window size is chosen based on switch frequency:
    - Frequent switches (e.g., Electricity, Sleep): small window (±5)
    - Moderate switches (e.g., Seattle, Hangzhou): medium window (±20-30)
    - Sparse switches (e.g., Unemployment, Lorenz): large window (±30)
    Parameters
    ----------
    dataset_name : str
        Dataset name

    Returns
    -------
    window : int
        Recommended window size (in timesteps around each switch)
    """
    # Direct lookup
    if dataset_name in DATASET_WINDOWS:
        return DATASET_WINDOWS[dataset_name]

    # Default
    return 20


def print_lag_info(dataset_name):
    """Print lag compensation and window information for a dataset."""
    lag = get_lag(dataset_name)
    shift = -int(lag)
    window = get_window(dataset_name)

    print(f"\nLag compensation for {dataset_name}:")
    print(f"  Measured lag: {lag} timesteps")
    print(f"  Compensation: shift regimes by {shift} steps")
    print(f"  Recommended window: ±{window} timesteps")

    if shift == 0:
        print(f"  → No compensation needed (already aligned)")
    elif shift > 0:
        print(f"  → Shifts switches later by {shift} steps")
        print(f"  → Example: switch at t=100 → effective switch at t={100+shift}")
    else:
        print(f"  → Shifts switches earlier by {abs(shift)} steps")
        print(f"  → Example: switch at t=100 → effective switch at t={100+shift}")
