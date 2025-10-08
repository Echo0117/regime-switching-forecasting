from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from experiments.config import config

def prepare_data(DataPath, brand, test_len, valid_ratio, device):
    trainX, trainY, validX, validY, testX, testY, RawDataOriginal, scaler_x,  scaler_y = build_flat_input(
        csv_path=DataPath,
        target_brand=brand,
        test_len=test_len,
        valid_ratio=valid_ratio,
        device=device
    )
    x_dim = trainX.shape[1]
    y_dim = trainY.shape[1]

    # DS3M uses (T, B, F) → keep B=1, let T=N
    trainX = trainX.unsqueeze(0)
    trainY = trainY.unsqueeze(0)
    validX = validX.unsqueeze(0)
    validY = validY.unsqueeze(0)
    testX  = testX.unsqueeze(0)
    testY  = testY.unsqueeze(0)

    return trainX, trainY, validX, validY, testX, testY, RawDataOriginal, scaler_x,  scaler_y, x_dim, y_dim

def build_flat_input(
    csv_path: str,
    target_brand: str = None,
    target_col: str = config["dataset"]["dependent_variable"],
    time_col: str = "year_week",
    brand_col: str = "brand_name",
    exog_cols=None,
    test_len: int = 52,
    valid_ratio: float = 0.25,
    sep: str = ";",
    device: str = "cpu"
):
    """
    Return trainX, trainY, validX, validY, testX, testY as torch tensors
    (no time windows, just flat X -> y).
    """
    df = pd.read_csv(csv_path, sep=sep)
    df = df.fillna(0.0)

    if target_brand is not None:
        df = df[df[brand_col] == target_brand].copy()
        if df.empty:
            raise ValueError(f"No rows found for brand='{target_brand}'")

    df = df.sort_values(time_col).reset_index(drop=True)

    # Infer exogenous columns
    if exog_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exog_cols = [c for c in numeric_cols if c != target_col]

    y = df[target_col].astype(float).to_numpy().reshape(-1, 1)

    X = df[exog_cols].copy().values

    N = len(y)
    train_end = N - test_len
    train_len = int(train_end * (1 - valid_ratio))

    Xs, ys, scaler_x, scaler_y = normalize_data(X, y)

    # splits
    trainX, validX, testX = Xs[:train_len], Xs[train_len:train_end], Xs[train_end:]
    trainY, validY, testY = ys[:train_len], ys[train_len:train_end], ys[train_end:]

    # convert to torch
    trainX = torch.tensor(trainX, dtype=torch.float32, device=device)
    validX = torch.tensor(validX, dtype=torch.float32, device=device)
    testX  = torch.tensor(testX,  dtype=torch.float32, device=device)

    trainY = torch.tensor(trainY, dtype=torch.float32, device=device)
    validY = torch.tensor(validY, dtype=torch.float32, device=device)
    testY  = torch.tensor(testY,  dtype=torch.float32, device=device)

    RawDataOriginal = y.reshape(-1, 1, 1)  # shape (T, 1, 1) to mimic old 3D style
    return trainX, trainY, validX, validY, testX, testY, RawDataOriginal, scaler_x, scaler_y


def normalize_data(X_t: np.ndarray, Y_t: np.ndarray):
    scaler_x = AbsoluteMedianScaler()
    scaler_y = AbsoluteMedianScaler()
    X_t_normalized = scaler_x.fit_transform(X_t)
    Y_t_normalized = scaler_y.fit_transform(Y_t)

    return X_t_normalized, Y_t_normalized, scaler_x, scaler_y


class AbsoluteMedianScaler(BaseEstimator, TransformerMixin):
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fit the scaler by calculating the mean and std of each feature.

        Parameters:
        X (np.ndarray): The input data to fit, shape (n_samples, n_features).
        y (np.ndarray, optional): The target values (ignored).

        Returns:
        MeanStdScaler: The fitted scaler.
        """
        self.means_ = np.mean(X, axis=0)
        self.stds_ = np.std(X, axis=0)
        # prevent division by zero
        self.stds_[self.stds_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize the data using mean and std.

        Parameters:
        X (np.ndarray): The input data to transform.

        Returns:
        np.ndarray: The transformed data.
        """
        return (X - self.means_) / self.stds_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Revert the standardization back to original scale.

        Parameters:
        X (np.ndarray): The standardized data to inverse.

        Returns:
        np.ndarray: The original data.
        """
        return X * self.stds_ + self.means_