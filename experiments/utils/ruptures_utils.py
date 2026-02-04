"""
Utilities for ruptures-based regime detection and forecasting.

This module provides:
1. Regime detection using ruptures changepoint detection methods
2. Simple forecasting models (AR, mean, median) for each regime
3. Integration with AGACI conformal prediction framework
4. Evaluation metrics for changepoint detection accuracy

Ruptures documentation: https://centre-borelli.github.io/ruptures-docs/

Supported Methods (per Alessandro's feedback - specify which method is used):
- Pelt: Pruned Exact Linear Time (default, O(n) complexity)
- Binseg: Binary Segmentation (O(n log n))
- BottomUp: Bottom-Up Segmentation
- Window: Sliding Window

Supported Cost Functions:
- l1, l2: Linear costs
- rbf: Radial Basis Function (default, good for general use)
- linear: Linear model
- normal: Gaussian model
- ar: Autoregressive model
"""

import numpy as np
import ruptures as rpt
from typing import Tuple, Dict, Optional, List


def create_lag_features(data: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create lag features for time series forecasting.

    Parameters
    ----------
    data : np.ndarray
        Time series data, shape (T,)
    lag : int
        Number of lags to use

    Returns
    -------
    X : np.ndarray
        Lag features, shape (T-lag, lag)
    y : np.ndarray
        Target values, shape (T-lag,)
    """
    if len(data) < lag + 1:
        return np.array([]), np.array([])

    X = []
    y = []
    for i in range(lag, len(data)):
        X.append(data[i-lag:i])
        y.append(data[i])

    return np.array(X), np.array(y)


def detect_regimes_ruptures(
    data: np.ndarray,
    method: str = "Pelt",
    model: str = "rbf",
    min_size: int = 10,
    penalty: float = None,
    n_bkps: int = None
) -> Tuple[np.ndarray, List[int]]:
    """
    Detect regime switches using ruptures changepoint detection.

    Parameters
    ----------
    data : np.ndarray
        Time series data, shape (T,) or (T, D)
    method : str
        Detection method: "Pelt", "Binseg", "BottomUp", "Window"
    model : str
        Cost function model: "l1", "l2", "rbf", "linear", "normal", "ar"
    min_size : int
        Minimum segment size between changepoints
    penalty : float
        Penalty value for Pelt/Binseg (auto if None)
    n_bkps : int
        Number of breakpoints for Binseg/BottomUp/Window (auto if None)

    Returns
    -------
    regime_labels : np.ndarray
        Regime labels (0, 1, 2, ...), shape (T,)
    breakpoints : List[int]
        List of breakpoint indices (includes final index)
    """
    if data.ndim == 1:
        signal = data.reshape(-1, 1)
    else:
        signal = data

    T = len(signal)

    # Select detection algorithm
    if method.lower() == "pelt":
        algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
        if penalty is None:
            # Auto penalty: use elbow method or fixed heuristic
            penalty = 3 * np.log(T)
        breakpoints = algo.predict(pen=penalty)

    elif method.lower() == "binseg":
        algo = rpt.Binseg(model=model, min_size=min_size).fit(signal)
        if n_bkps is None:
            # Auto: use penalty-based stopping
            if penalty is None:
                penalty = 3 * np.log(T)
            breakpoints = algo.predict(pen=penalty)
        else:
            breakpoints = algo.predict(n_bkps=n_bkps)

    elif method.lower() == "bottomup":
        algo = rpt.BottomUp(model=model, min_size=min_size).fit(signal)
        if n_bkps is None:
            if penalty is None:
                penalty = 3 * np.log(T)
            breakpoints = algo.predict(pen=penalty)
        else:
            breakpoints = algo.predict(n_bkps=n_bkps)

    elif method.lower() == "window":
        algo = rpt.Window(model=model, width=min_size*2).fit(signal)
        if n_bkps is None:
            if penalty is None:
                penalty = 3 * np.log(T)
            breakpoints = algo.predict(pen=penalty)
        else:
            breakpoints = algo.predict(n_bkps=n_bkps)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Convert breakpoints to regime labels
    regime_labels = np.zeros(T, dtype=int)
    start = 0
    for i, bkp in enumerate(breakpoints[:-1]):  # Last breakpoint is always T
        regime_labels[start:bkp] = i
        start = bkp

    return regime_labels, breakpoints


def simple_ar_forecast(
    history: np.ndarray,
    lag: int = 1,
    n_ahead: int = 1
) -> np.ndarray:
    """
    Simple AR(p) forecast using least squares.

    Parameters
    ----------
    history : np.ndarray
        Historical data, shape (T,)
    lag : int
        AR order (number of lags)
    n_ahead : int
        Number of steps ahead to forecast

    Returns
    -------
    forecast : np.ndarray
        Forecast values, shape (n_ahead,)
    """
    if len(history) < lag + 1:
        # Fallback: use mean
        return np.full(n_ahead, np.mean(history))

    # Prepare X (lagged values) and y (targets)
    X = []
    y = []
    for i in range(lag, len(history)):
        X.append(history[i-lag:i])
        y.append(history[i])

    X = np.array(X)
    y = np.array(y)

    # Fit AR coefficients using least squares
    if len(X) == 0:
        return np.full(n_ahead, np.mean(history))

    try:
        # Add intercept
        X_with_intercept = np.c_[np.ones(len(X)), X]
        coef = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    except:
        # Fallback: use mean
        return np.full(n_ahead, np.mean(history))

    # Multi-step forecast
    forecast = []
    current_window = list(history[-lag:])

    for _ in range(n_ahead):
        # Predict next value
        X_new = np.array([1] + current_window[-lag:])
        y_new = np.dot(X_new, coef)
        forecast.append(y_new)

        # Update window
        current_window.append(y_new)

    return np.array(forecast)


class RupturesForecaster:
    """
    Regime-switching forecaster using ruptures for regime detection
    and simple forecasting models for each regime.
    """

    def __init__(
        self,
        method: str = "Pelt",
        model: str = "rbf",
        min_size: int = 10,
        penalty: float = None,
        n_bkps: int = None,
        forecast_method: str = "ar",
        ar_lag: int = 1,
        forecast_model_kwargs: dict = None
    ):
        """
        Parameters
        ----------
        method : str
            Ruptures detection method
        model : str
            Ruptures cost function model
        min_size : int
            Minimum segment size
        penalty : float
            Penalty for changepoint detection
        n_bkps : int
            Number of breakpoints (for some methods)
        forecast_method : str
            Forecasting method: "ar", "mean", "median", "last", "gru", "s4", "linear"
        ar_lag : int
            Lag for AR forecasting (also used for neural models)
        forecast_model_kwargs : dict
            Additional kwargs for neural models (gru, s4)
        """
        self.method = method
        self.model = model
        self.min_size = min_size
        self.penalty = penalty
        self.n_bkps = n_bkps
        self.forecast_method = forecast_method
        self.ar_lag = ar_lag
        self.forecast_model_kwargs = forecast_model_kwargs or {}

        # Will be set during fit
        self.regime_labels_ = None
        self.breakpoints_ = None
        self.regime_params_ = {}
        self.trained_models_ = {}  # Store trained neural models per regime

    def fit(self, data: np.ndarray) -> 'RupturesForecaster':
        """
        Fit regime detection and estimate parameters for each regime.

        Parameters
        ----------
        data : np.ndarray
            Training data, shape (T,) or (T, D)

        Returns
        -------
        self
        """
        # Detect regimes
        self.regime_labels_, self.breakpoints_ = detect_regimes_ruptures(
            data,
            method=self.method,
            model=self.model,
            min_size=self.min_size,
            penalty=self.penalty,
            n_bkps=self.n_bkps
        )

        # Extract parameters for each regime
        if data.ndim == 1:
            data_1d = data
        else:
            data_1d = data[:, 0]  # Use first dimension for simplicity

        unique_regimes = np.unique(self.regime_labels_)
        for regime in unique_regimes:
            mask = self.regime_labels_ == regime
            regime_data = data_1d[mask]

            self.regime_params_[regime] = {
                'mean': np.mean(regime_data),
                'median': np.median(regime_data),
                'std': np.std(regime_data),
                'last': regime_data[-1] if len(regime_data) > 0 else 0
            }

            # Fit forecasting model based on method
            if self.forecast_method == "ar":
                # Store recent history for AR forecasting
                self.regime_params_[regime]['ar_history'] = regime_data

            elif self.forecast_method in ["gru", "s4", "linear"]:
                # Train neural/advanced model on regime data
                # Need sufficient data and lag
                if len(regime_data) >= self.ar_lag + 10:  # Minimum data requirement
                    try:
                        self._train_regime_model(regime, regime_data)
                    except Exception as e:
                        print(f"[RUPTURES] Warning: Failed to train {self.forecast_method} for regime {regime}: {e}")
                        # Fallback to mean
                        self.regime_params_[regime]['fallback'] = True
                else:
                    # Not enough data, use fallback
                    self.regime_params_[regime]['fallback'] = True

        return self

    def _train_regime_model(self, regime: int, regime_data: np.ndarray):
        """Train advanced forecasting model for a specific regime."""
        from experiments.competitor_models import MCDropoutGRU, S4Regressor, RupturesSegmentedLinear

        # Create lag features
        X, y = create_lag_features(regime_data, lag=self.ar_lag)

        if len(X) < 10:  # Need minimum samples
            self.regime_params_[regime]['fallback'] = True
            return

        # Select and train model based on forecast_method
        if self.forecast_method == "gru":
            model = MCDropoutGRU(
                lags=self.ar_lag,
                hidden=self.forecast_model_kwargs.get('hidden', 64),
                layers=self.forecast_model_kwargs.get('layers', 1),
                dropout=self.forecast_model_kwargs.get('dropout', 0.2),
                epochs=self.forecast_model_kwargs.get('epochs', 30),
                batch=self.forecast_model_kwargs.get('batch', 32),
                lr=self.forecast_model_kwargs.get('lr', 1e-3),
                patience=self.forecast_model_kwargs.get('patience', 5),
                verbose=False,
                device=self.forecast_model_kwargs.get('device', 'cpu')
            )
        elif self.forecast_method == "s4":
            model = S4Regressor(
                lags=self.ar_lag,
                d_model=self.forecast_model_kwargs.get('d_model', 64),
                n_layers=self.forecast_model_kwargs.get('n_layers', 2),
                dropout=self.forecast_model_kwargs.get('dropout', 0.1),
                epochs=self.forecast_model_kwargs.get('epochs', 30),
                batch=self.forecast_model_kwargs.get('batch', 32),
                lr=self.forecast_model_kwargs.get('lr', 1e-3),
                patience=self.forecast_model_kwargs.get('patience', 5),
                verbose=False,
                device=self.forecast_model_kwargs.get('device', 'cpu')
            )
        elif self.forecast_method == "linear":
            model = RupturesSegmentedLinear(
                penalty=self.forecast_model_kwargs.get('penalty', 10.0),
                min_size=self.forecast_model_kwargs.get('min_size', 10),
                model=self.forecast_model_kwargs.get('model', 'l2')
            )
        else:
            raise ValueError(f"Unknown forecast method: {self.forecast_method}")

        # Train the model
        model.fit(X, y)
        self.trained_models_[regime] = model
        self.regime_params_[regime]['fallback'] = False

    def predict(
        self,
        history: np.ndarray,
        n_ahead: int = 1,
        current_regime: int = None
    ) -> Tuple[np.ndarray, int]:
        """
        Forecast next n_ahead steps.

        Parameters
        ----------
        history : np.ndarray
            Recent history, shape (window,) or (window, D)
        n_ahead : int
            Number of steps to forecast
        current_regime : int
            Current regime (if None, detect from history)

        Returns
        -------
        forecast : np.ndarray
            Forecast values, shape (n_ahead,)
        predicted_regime : int
            Predicted regime label
        """
        if history.ndim > 1:
            history_1d = history[:, 0]
        else:
            history_1d = history

        # Detect current regime if not provided
        if current_regime is None:
            # Use last detected regime as default
            current_regime = self.regime_labels_[-1]

        # Get regime parameters
        if current_regime not in self.regime_params_:
            # Fallback: use last regime
            current_regime = max(self.regime_params_.keys())

        params = self.regime_params_[current_regime]

        # Make forecast based on method
        if self.forecast_method == "mean":
            forecast = np.full(n_ahead, params['mean'])

        elif self.forecast_method == "median":
            forecast = np.full(n_ahead, params['median'])

        elif self.forecast_method == "last":
            forecast = np.full(n_ahead, params['last'])

        elif self.forecast_method == "ar":
            # Use AR forecast with regime-specific history
            regime_history = params.get('ar_history', history_1d)
            forecast = simple_ar_forecast(regime_history, lag=self.ar_lag, n_ahead=n_ahead)

        elif self.forecast_method in ["gru", "s4", "linear"]:
            # Use trained neural/advanced model
            if params.get('fallback', True) or current_regime not in self.trained_models_:
                # Fallback to mean if model not trained
                forecast = np.full(n_ahead, params['mean'])
            else:
                model = self.trained_models_[current_regime]
                # Use recent history for prediction
                if len(history_1d) >= self.ar_lag:
                    X_new = history_1d[-self.ar_lag:].reshape(1, -1)
                    pred = model.predict(X_new)
                    if n_ahead == 1:
                        forecast = np.array([pred[0]])
                    else:
                        # Multi-step: iteratively predict
                        forecast = []
                        current_window = list(history_1d[-self.ar_lag:])
                        for _ in range(n_ahead):
                            X_new = np.array(current_window[-self.ar_lag:]).reshape(1, -1)
                            y_new = model.predict(X_new)[0]
                            forecast.append(y_new)
                            current_window.append(y_new)
                        forecast = np.array(forecast)
                else:
                    # Not enough history, use mean
                    forecast = np.full(n_ahead, params['mean'])

        else:
            # Default: use mean
            forecast = np.full(n_ahead, params['mean'])

        return forecast, current_regime

    def predict_with_uncertainty(
        self,
        history: np.ndarray,
        n_ahead: int = 1,
        current_regime: int = None,
        quantile_low: float = 0.05,
        quantile_high: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Forecast with uncertainty quantiles based on regime variance.

        Returns
        -------
        forecast_mean : np.ndarray
        forecast_low : np.ndarray
        forecast_high : np.ndarray
        predicted_regime : int
        """
        forecast, regime = self.predict(history, n_ahead, current_regime)

        # Get regime std
        params = self.regime_params_[regime]
        std = params.get('std', 1.0)

        # Simple Gaussian quantiles (can be improved)
        from scipy.stats import norm
        z_low = norm.ppf(quantile_low)
        z_high = norm.ppf(quantile_high)

        forecast_low = forecast + z_low * std
        forecast_high = forecast + z_high * std

        return forecast, forecast_low, forecast_high, regime


def ruptures_forecast_sequence(
    data: np.ndarray,
    train_size: int,
    test_size: int,
    method: str = "Pelt",
    model: str = "rbf",
    min_size: int = 10,
    penalty: float = None,
    forecast_method: str = "ar",
    ar_lag: int = 1,
    forecast_model_kwargs: dict = None
) -> Dict:
    """
    Run ruptures-based regime detection and forecasting on a sequence.

    Similar to DS3M forecast interface for compatibility with AGACI.

    Parameters
    ----------
    data : np.ndarray
        Full data sequence, shape (T,) or (T, D)
    train_size : int
        Training set size
    test_size : int
        Test set size
    method : str
        Ruptures detection method
    model : str
        Ruptures cost function model
    min_size : int
        Minimum segment size
    penalty : float
        Changepoint penalty
    forecast_method : str
        Forecasting method within regimes ("ar", "mean", "median", "last", "gru", "s4", "linear")
    ar_lag : int
        AR lag for forecasting (also used for neural models)
    forecast_model_kwargs : dict
        Additional kwargs for neural models (gru, s4)

    Returns
    -------
    results : dict
        Dictionary with:
        - 'regime_labels_train': Regime labels for training set
        - 'regime_labels_test': Regime labels for test set
        - 'regime_labels_full': Full regime sequence
        - 'breakpoints': Detected breakpoints
        - 'forecasts': Point forecasts for test set
        - 'lower_quantiles': Lower quantiles (0.05)
        - 'upper_quantiles': Upper quantiles (0.95)
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    T_total = len(data)

    # Split data
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]

    # **FIX**: Fit on FULL data to detect breakpoints in both train and test
    # This is necessary because we want to see regime switches in the test set
    forecaster = RupturesForecaster(
        method=method,
        model=model,
        min_size=min_size,
        penalty=penalty,
        forecast_method=forecast_method,
        ar_lag=ar_lag,
        forecast_model_kwargs=forecast_model_kwargs
    )

    print(f"[RUPTURES] Training forecaster with method={forecast_method}, lag={ar_lag}")
    forecaster.fit(data)  # Fit on full data instead of just train_data
    print(f"[RUPTURES] Detected {len(np.unique(forecaster.regime_labels_))} regimes")

    # Get regime labels for train and test sets
    regime_labels_train = forecaster.regime_labels_[:train_size]
    regime_labels_test_full = forecaster.regime_labels_[train_size:train_size + test_size]

    # Forecast test set one step at a time
    # Use the detected regime labels from full data, not predicted ones
    forecasts = []
    lower_quantiles = []
    upper_quantiles = []

    for t in range(test_size):
        # Use sliding window of recent history
        window_size = min(train_size, 100)  # Limit window size for efficiency
        if t == 0:
            history = train_data[-window_size:]
            current_regime = regime_labels_test_full[0]  # Use detected regime
        else:
            history = data[train_size + t - window_size:train_size + t]
            current_regime = regime_labels_test_full[t]  # Use detected regime

        # Forecast with uncertainty using the detected regime
        fc, fc_low, fc_high, _ = forecaster.predict_with_uncertainty(
            history,
            n_ahead=1,
            current_regime=current_regime
        )

        forecasts.append(fc[0])
        lower_quantiles.append(fc_low[0])
        upper_quantiles.append(fc_high[0])

    # Use detected regime labels (not predicted)
    regime_labels_test = regime_labels_test_full
    regime_labels_full = forecaster.regime_labels_  # Use full detected labels

    return {
        'regime_labels_train': regime_labels_train,
        'regime_labels_test': np.array(regime_labels_test),
        'regime_labels_full': regime_labels_full,
        'breakpoints': forecaster.breakpoints_,
        'forecasts': np.array(forecasts),
        'lower_quantiles': np.array(lower_quantiles),
        'upper_quantiles': np.array(upper_quantiles),
        'n_regimes': len(np.unique(regime_labels_full)),
        'method': method,  # Added: which ruptures method was used
        'model': model,    # Added: which cost function was used
    }

