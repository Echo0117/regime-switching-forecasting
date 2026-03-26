# %%
import warnings
import os
import copy
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import matplotlib
import argparse
from pathlib import Path
import os, sys

HERE = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


CACHE_DIR = Path("cached_forecasts")
CACHE_DIR.mkdir(exist_ok=True)


from Deep_Switching_State_Space_Model.src.DSSSMCode import *
from Deep_Switching_State_Space_Model.src.utils import *
import torch.utils.data
import torch.utils
import torch
import seaborn as sns
import matplotlib

matplotlib.use("Agg")  # Set backend before importing pyplot
warnings.filterwarnings("ignore")


# %%
def load_ds3m_data(args):
    remove_mean = False
    remove_residual = False
    longterm = False
    bidirection = False

    if args.seed is not None:
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # %%
    dataname = args.problem
    print(dataname)
    retry = 1

    # Determine data directory and dataset identifier
    if hasattr(args, 'data_dir') and args.data_dir is not None:
        # Custom data directory (e.g., Deep_Switching_State_Space_Model/data/Toy_exp2_V1_0.5_V2_1.0)
        data_dir = args.data_dir
        # Extract dataset identifier from directory name
        dataset_id = os.path.basename(data_dir)
    else:
        # Default data directory
        data_dir = f"Deep_Switching_State_Space_Model/data/{dataname}"
        dataset_id = dataname

    print(f"Data directory: {data_dir}")
    print(f"Dataset ID: {dataset_id}")

    if dataname == "Toy" or dataset_id.startswith('Toy'):

        freq = 1
        test_len = 500
        timestep = 20
        predict_dim = 1

        x_dim = predict_dim  # Dimension of x
        y_dim = predict_dim  # Dimension of y #equal to predict_len
        h_dim = 10  # Dimension of the hidden states in RNN
        z_dim = 2  # Dimension of the latent variable z
        d_dim = 2  # Dimension of the latent variable d
        n_layers = 1  # Number of the layers of the RNN
        clip = 10  # Gradient clips
        learning_rate = 1e-3  # Learning rate
        batch_size = 64  # Batch size
        n_epochs = 100  # Number of epochs for training

        data_path = f"{data_dir}/simulation_data_nonlinear_y.csv"
        print(f"Loading data from: {data_path}")

        RawDataOriginal = pd.read_csv(data_path, header=None).values
        RawDataOriginal = RawDataOriginal.reshape(-1, 1, 1)

    # %%
    if dataname == "Lorenz":

        freq = 1
        test_len = 1000
        timestep = 5
        predict_dim = 10
        look_back = timestep

        # hyperparameters
        x_dim = predict_dim  # Dimension of x
        y_dim = predict_dim  # Dimension of y #equal to predict_len
        h_dim = 20  # Dimension of the hidden states in RNN
        z_dim = 3  # Dimension of the latent variable z
        d_dim = 2  # Dimension of the latent variable d
        n_layers = 1  # Number of the layers of the RNN
        clip = 10  # Gradient clips
        learning_rate = 1e-3  # Learning rate
        batch_size = 64  # Batch size
        n_epochs = 300

        import json

        f = open("Deep_Switching_State_Space_Model/data/Lorenz/lorenz.json")
        data_st_all = json.load(f)
        D = len(data_st_all["data"][0][0])
        factors_true = torch.FloatTensor(data_st_all["factors"])
        z_true = torch.FloatTensor(data_st_all["latents"])
        z_true = z_true[:, 2000:5000]
        data_st = np.array(data_st_all["data"])
        data_st = data_st[:, 2000:5000]
        # Set seed for reproducible noise
        np.random.seed(0)
        torch.manual_seed(0)
        data_st = data_st + data_st[0].std(axis=0) * 0.001 * np.random.randn(
            data_st.shape[1], 10
        )  # added noise
        data_st = (data_st - data_st[0].mean(axis=0)) / data_st[0].std(axis=0)  # added
        states = np.zeros(z_true.numpy().shape[0:2])
        states[z_true.numpy()[:, :, 0] > 0] = 1
        states = torch.LongTensor(states)
        RawDataOriginal = data_st.transpose(1, 0, 2)

    # %%
    if dataname == "Sleep":

        freq = 1
        timestep = 200
        predict_dim = 1
        test_len = 1000 - timestep

        x_dim = predict_dim  # Dimension of x
        y_dim = predict_dim  # Dimension of y #equal to predict_len
        h_dim = 10  # Dimension of the hidden states in RNN
        z_dim = 1  # Dimension of the latent variable z
        d_dim = 2  # Dimension of the latent variable d
        n_layers = 1  # Number of the layers of the RNN
        clip = 10  # Gradient clips
        learning_rate = 1e-3  # Learning rate
        batch_size = 64  # Batch size
        n_epochs = 100
        dataset = pd.read_csv(
            "Deep_Switching_State_Space_Model/data/Sleep/b1.txt", sep=" ", header=None
        )
        chest = dataset.iloc[:, 1].values.reshape(-1, 1)
        train_data = chest[6200:7200, :]
        test_data = chest[5200:6200, :]
        RawDataOriginal = np.concatenate((train_data, test_data)).reshape(-1, 1, 1)
    # %%
    if dataname == "Unemployment":
        freq = 1
        test_len = 20 * 12
        timestep = 6
        predict_dim = 1

        x_dim = predict_dim  # Dimension of x
        y_dim = predict_dim  # Dimension of y #equal to predict_len
        h_dim = 10  # Dimension of the hidden states in RNN
        z_dim = 2  # Dimension of the latent variable z
        d_dim = 2  # Dimension of the latent variable d
        n_layers = 1  # Number of the layers of the RNN
        clip = 10  # Gradient clips
        learning_rate = 1e-3  # Learning rate
        batch_size = 64  # Batch size
        n_epochs = 500  # Number of epochs for training

        RawDataOriginal = (
            pd.read_csv(
                "Deep_Switching_State_Space_Model/data/Unemployment/UNRATE.csv",
                header=0,
            )
            .loc[:, "UNRATE"]
            .values
        )
        RawDataOriginal = RawDataOriginal.reshape(-1, 1, 1)
        torch.manual_seed(2)

    # %%
    if dataname == "Hangzhou":

        remove_mean = True
        bidirection = True

        freq = 108
        test_len = 5 * freq
        timestep = 12  # freq #freq*1
        predict_dim = 80

        # hyperparameters
        x_dim = predict_dim  # Dimension of x
        y_dim = predict_dim  # Dimension of y #equal to predict_len
        h_dim = predict_dim  # Dimension of the hidden states in RNN
        z_dim = 40  # int(predict_dim/2) # Dimension of the latent variable z
        d_dim = 2  # Dimension of the latent variable d
        n_layers = 1  # Number of the layers of the RNN
        clip = 10  # Gradient clips
        learning_rate = 1e-3  # Learning rate
        batch_size = 64  # 512 # Batch size
        n_epochs = 100

        from scipy.io import loadmat

        RawDataOriginal = loadmat(
            "Deep_Switching_State_Space_Model/data/Hangzhou/hangzhou.mat"
        )["tensor"].astype("float")
        RawDataOriginal = RawDataOriginal.transpose(1, 2, 0)
        RawDataOriginal.shape

    # %%
    if dataname == "Seattle":
        remove_residual = True
        bidirection = True

        freq = 288
        test_len = 5 * freq
        timestep = 12  # 24 #int(freq/3)
        predict_dim = 323

        # hyperparameters
        x_dim = predict_dim  # Dimension of x
        y_dim = predict_dim  # Dimension of y #equal to predict_len
        h_dim = 40  # Dimension of the hidden states in RNN
        z_dim = 10  # Dimension of the latent variable z
        d_dim = 2  # Dimension of the latent variable d
        n_layers = 1  # Number of the layers of the RNN
        clip = 10  # Gradient clips
        learning_rate = 1e-2  # Learning rate
        batch_size = 64  # Batch size
        n_epochs = 100

        # Dataset Preprocessing
        RawDataOriginal = (
            np.load("Deep_Switching_State_Space_Model/data/Seattle/seattle.npz")[
                "arr_0"
            ]
            .astype("float")
            .transpose(1, 0)
        )
        RawDataOriginal = RawDataOriginal.reshape((-1, freq, RawDataOriginal.shape[1]))
        RawDataOriginal.shape

    # %%
    if dataname == "Pacific":

        remove_mean = True
        bidirection = True

        freq = 12
        test_len = 5 * freq
        timestep = 24  # 12#freq*3
        predict_dim = 2520

        # hyperparameters
        x_dim = predict_dim  # Dimension of x
        y_dim = predict_dim  # Dimension of y #equal to predict_len
        h_dim = 200  # Dimension of the hidden states in RNN
        z_dim = 50  # Dimension of the latent variable z
        d_dim = 2  # Dimension of the latent variable d
        n_layers = 1  # Number of the layers of the RNN
        clip = 10  # Gradient clips
        learning_rate = 1e-3  # Learning rate
        batch_size = 64  # Batch size
        n_epochs = 100

        # Dataset Preprocessing
        RawDataOriginal = pd.read_csv(
            "Deep_Switching_State_Space_Model/data/Pacific/pacific.tsv",
            sep="\t",
            header=None,
        ).values.reshape(-1, 30 * 84)[3:]
        RawDataOriginal = RawDataOriginal.reshape(-1, 12, RawDataOriginal.shape[1])
        RawDataOriginal.shape

    # %%
    if dataname == "Electricity":

        freq = 1
        timestep = 14
        predict_dim = 48
        test_len = 320
        DataPath = "Deep_Switching_State_Space_Model/data/Electricity/French_all.csv"

        x_dim = predict_dim  # Dimension of x
        y_dim = predict_dim  # Dimension of y #equal to predict_len
        h_dim = predict_dim  # Dimension of the hidden states in RNN
        z_dim = 10  # Dimension of the latent variable z
        d_dim = 2  # Dimension of the latent variable d
        n_layers = 1  # Number of the layers of the RNN
        clip = 10  # Gradient clips
        learning_rate = 1e-3  # Learning rate
        batch_size = 64  # 256 # Batch size
        n_epochs = 100  # Number of epochs for training

        RawDataOriginal = pd.read_csv(DataPath)
        RawDataOriginal = RawDataOriginal[RawDataOriginal["Date"] < "2019-12-31"]
        RawDataOriginal = RawDataOriginal["Load"].values
        RawDataOriginal = RawDataOriginal.reshape(-1, 1, predict_dim)

    # %%
    if dataname == "Pernod":

        freq = 1
        timestep = 4  # Weekly data, use 4 weeks lookback
        predict_dim = 1  # Univariate: volume_so only
        test_len = 260  # ~5 years of weekly data for testing
        DataPath = "Deep_Switching_State_Space_Model/data/Pernod/pernod.csv"

        # These match the trained model checkpoint
        x_dim = 1  # Dimension of x (must match rnn_forward input)
        y_dim = 1  # Dimension of y (same as x_dim for autoregressive)
        h_dim = 50  # Dimension of the hidden states in RNN
        z_dim = 10  # Dimension of the latent variable z
        d_dim = 2  # Dimension of the latent variable d
        n_layers = 1  # Number of the layers of the RNN
        clip = 10  # Gradient clips
        learning_rate = 1e-3  # Learning rate
        batch_size = 64  # Batch size
        n_epochs = 100  # Number of epochs for training

        df = pd.read_csv(DataPath, delimiter=';')
        feature_cols = ['volume_so']
        RawDataOriginal = df[feature_cols].values
        # Handle missing values (replace with 0)
        RawDataOriginal = np.nan_to_num(RawDataOriginal, nan=0.0)
        RawDataOriginal = RawDataOriginal.reshape(-1, 1, predict_dim)
        pernod_target_dim = 0

    # %%
    if remove_mean:
        means = np.expand_dims(
            np.mean(RawDataOriginal[: -int(test_len / freq), :, :], axis=0), axis=0
        )
    else:
        means = 0
    RawData = RawDataOriginal - means

    RawData = RawData.reshape(-1, RawData.shape[2])
    data = RawData

    # For most datasets, X and Y are the same (autoregressive)
    data_X = data
    data_Y = data
    separate_XY = False

    if remove_residual:
        trend = data[0:-1, :]
        data = data[1:, :] - trend
        if separate_XY:
            trend_X = data_X[0:-1, :]
            data_X = data_X[1:, :] - trend_X
            trend_Y = data_Y[0:-1, :]
            data_Y = data_Y[1:, :] - trend_Y

    # Split into train and test data
    # Dataset-specific fixes for alignment issues:
    # - Electricity: Use [(-test_len-timestep-1):-1] to fix -1 shift
    # - Other datasets: Use standard [-test_len - timestep :] indexing

    if dataname == "Unemployment":
        length = len(data) - test_len
        train_len = int(length)
        train_data_X = data_X[:train_len]
        valid_data_X = data_X[:train_len]
        test_data_X = data_X[-test_len - timestep :]
        train_data_Y = data_Y[:train_len]
        valid_data_Y = data_Y[:train_len]
        test_data_Y = data_Y[-test_len - timestep :]

    elif dataname == "Lorenz":
        train_len = 1000 + timestep
        valid_len = len(data) - train_len - test_len
        train_data_X = data_X[:train_len]
        valid_data_X = data_X[(train_len) : (train_len + valid_len)]

        test_data_X = data_X[-test_len - timestep :]
        train_data_Y = data_Y[:train_len]
        valid_data_Y = data_Y[(train_len) : (train_len + valid_len)]
        test_data_X = data_X[-test_len - timestep :]

    elif dataname == "Sleep":
        train_len = 1000
        train_data_X = data_X[:train_len]
        valid_data_X = data_X[:train_len]
        test_data_X = data_X[-test_len - timestep :]
        train_data_Y = data_Y[:train_len]
        valid_data_Y = data_Y[:train_len]
        test_data_Y = data_Y[-test_len - timestep :]
    elif dataname == "Electricity":
        # Special fix for Electricity: exclude last element to fix -1 alignment
        length = len(data) - test_len
        train_len = int(length * 0.75)
        valid_len = int(length * 0.25)

        train_data_X = data_X[:train_len]
        valid_data_X = data_X[(train_len) : (train_len + valid_len)]
        # test_data_X = data_X[(-test_len - timestep - 1):-1]
        test_data_X = data_X[-test_len - timestep :]
        train_data_Y = data_Y[:train_len]
        valid_data_Y = data_Y[(train_len) : (train_len + valid_len)]
        # test_data_Y = data_Y[(-test_len - timestep - 1):-1]
        test_data_X = data_X[-test_len - timestep :]
    else:
        length = len(data) - test_len
        train_len = int(length * 0.75)
        valid_len = int(length * 0.25)

        train_data_X = data_X[:train_len]
        valid_data_X = data_X[(train_len) : (train_len + valid_len)]
        test_data_X = data_X[-test_len - timestep :]
        train_data_Y = data_Y[:train_len]
        valid_data_Y = data_Y[(train_len) : (train_len + valid_len)]
        test_data_Y = data_Y[-test_len - timestep :]

    # %%
    # Normalize the dataset
    # For Pernod, normalize X and Y separately since they have different dimensions
    if separate_XY:
        moments_X = normalize_moments(train_data_X)
        train_data_X = normalize_fit(train_data_X, moments_X)
        valid_data_X = normalize_fit(valid_data_X, moments_X)
        test_data_X = normalize_fit(test_data_X, moments_X)

        moments_Y = normalize_moments(train_data_Y)
        train_data_Y = normalize_fit(train_data_Y, moments_Y)
        valid_data_Y = normalize_fit(valid_data_Y, moments_Y)
        test_data_Y = normalize_fit(test_data_Y, moments_Y)
        moments = moments_X  # Use X moments for compatibility
    else:
        moments = normalize_moments(train_data_X)
        train_data_X = normalize_fit(train_data_X, moments)
        valid_data_X = normalize_fit(valid_data_X, moments)
        test_data_X = normalize_fit(test_data_X, moments)
        train_data_Y = train_data_X
        valid_data_Y = valid_data_X
        test_data_Y = test_data_X

    # Create training and test dataset
    if separate_XY:
        trainX, _ = create_dataset2(train_data_X, timestep)
        _, trainY = create_dataset2(train_data_Y, timestep)
        validX, _ = create_dataset2(valid_data_X, timestep)
        _, validY = create_dataset2(valid_data_Y, timestep)
        testX, _ = create_dataset2(test_data_X, timestep)
        _, testY = create_dataset2(test_data_Y, timestep)
    else:
        trainX, trainY = create_dataset2(train_data_X, timestep)
        validX, validY = create_dataset2(valid_data_X, timestep)
        testX, testY = create_dataset2(test_data_X, timestep)

    trainX = np.transpose(trainX, (1, 0, 2))
    validX = np.transpose(validX, (1, 0, 2))
    testX = np.transpose(testX, (1, 0, 2))
    print("3D size(X):", trainX.shape, validX.shape, testX.shape)

    trainY = np.transpose(trainY, (1, 0, 2))
    validY = np.transpose(validY, (1, 0, 2))
    testY = np.transpose(testY, (1, 0, 2))
    print("3D size(Y):", trainY.shape, validY.shape, testY.shape)

    trainX = torch.from_numpy(trainX).float()
    validX = torch.from_numpy(validX).float()
    testX = torch.from_numpy(testX).float()
    trainY = torch.from_numpy(trainY).float()
    validY = torch.from_numpy(validY).float()
    testY = torch.from_numpy(testY).float()

    # %%
    # Use dataset_id for checkpoint directory (includes exp2 info if present)
    directoryBest = os.path.join(
        "Deep_Switching_State_Space_Model", "results", "checkpoints", dataset_id
    )
    figdirectory = os.path.join("figures")
    if not os.path.exists(directoryBest):
        os.makedirs(directoryBest)
    if not os.path.exists(figdirectory):
        os.makedirs(figdirectory)
    figdirectory = figdirectory + "/" + dataname + "_"

    # %%
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainX = trainX.to(device)
    validX = validX.to(device)
    testX = testX.to(device)
    trainY = trainY.to(device)
    validY = validY.to(device)
    testY = testY.to(device)

    # Set target_dim based on the dataset - default to 0 for single-dim, or predict_dim-1 for multi-dim
    # For Electricity with 48 dimensions, this would be 47
    target_dim = predict_dim - 1 if predict_dim > 1 else 0
    if dataname == "Pernod":
        target_dim = pernod_target_dim

    return {
        "trainX": trainX,
        "trainY": trainY,
        "validX": validX,
        "validY": validY,
        "testX": testX,
        "testY": testY,
        "RawDataOriginal": RawDataOriginal,
        "means": means,
        "moments": moments,
        "freq": freq,
        "timestep": timestep,
        "predict_dim": predict_dim,
        "test_len": test_len,
        "x_dim": x_dim,
        "y_dim": y_dim,
        "h_dim": h_dim,
        "z_dim": z_dim,
        "d_dim": d_dim,
        "n_layers": n_layers,
        "trend": trend if remove_residual else None,
        "device": device,
        "directoryBest": directoryBest,
        "figdirectory": figdirectory,
        "states": states if dataname == "Lorenz" else None,
        "dataname": dataname,
        "remove_mean": remove_mean,
        "remove_residual": remove_residual,
        "longterm": longterm,
        "bidirection": bidirection,
        "data": data,
        "z_true": z_true if dataname == "Lorenz" else None,
        "learning_rate": learning_rate,
        "target_dim": target_dim,
    }


# %%
# Training
start = time.time()


def forecast(
    model,
    testX,
    testY,
    moments,
    d_dim,
    means,
    trend,
    test_len,
    freq,
    RawDataOriginal,
    remove_mean=False,
    remove_residual=False,
    forecaststep=1,
    MC_S=200,
    dataname=None,
):

    forecast_MC, forecast_d_MC, forecast_z_MC = model._forecastingMultiStep(
        testX, testY, forecaststep, MC_S
    )

    if forecaststep == 1:
        all_testForecast = normalize_invert(
            forecast_MC.squeeze(1).transpose(1, 0, 2), moments
        )
    else:
        all_testForecast = normalize_invert(
            forecast_MC.squeeze(2).transpose(1, 0, 2), moments
        )

    testY_inversed = normalize_invert(testY.cpu().numpy().transpose(1, 0, 2), moments)
    size = testY_inversed.shape[0]

    forecast_d_MC_argmax = []
    for i in range(d_dim):
        forecast_d_MC_argmax.append(np.sum(forecast_d_MC[:, -1, :, :] == i, axis=0))
    forecast_d_MC_argmax = np.argmax(np.array(forecast_d_MC_argmax), axis=0).reshape(-1)

    if remove_mean:
        testForecast_mean = np.mean(all_testForecast, axis=1) + np.tile(
            means[0, :, :], (int(test_len / freq), 1)
        )
        testForecast_uq = np.quantile(all_testForecast, 0.95, axis=1) + np.tile(
            means[0, :, :], (int(test_len / freq), 1)
        )
        testForecast_lq = np.quantile(all_testForecast, 0.05, axis=1) + np.tile(
            means[0, :, :], (int(test_len / freq), 1)
        )
    elif remove_residual:
        # Residual reconstruction: r_t = y_t - y_{t-1}, so y_t = r_t + y_{t-1}
        # Dataset-specific fixes for alignment:
        # - Seattle: +2 delay, shift trend_tail forward by 1 (partial fix)
        if dataname == "Seattle":
            # Shift forward by 1 to reduce +2 delay to +1, then extract correct portion
            trend_tail = trend[-test_len:, :]
        else:
            trend_tail = trend[-test_len-1:-1, :]

        testForecast_mean = np.mean(all_testForecast, axis=1) + trend_tail
        testForecast_uq = (
            np.quantile(all_testForecast, 0.95, axis=1) + trend_tail
        )
        testForecast_lq = (
            np.quantile(all_testForecast, 0.05, axis=1) + trend_tail
        )
    else:
        testForecast_mean = np.mean(all_testForecast, axis=1)
        testForecast_uq = np.quantile(all_testForecast, 0.95, axis=1)
        testForecast_lq = np.quantile(all_testForecast, 0.05, axis=1)

    # Extract ground truth with dataset-specific alignment
    # Seattle: +1 shift remaining, adjust testOriginal to match forecast
    if dataname == "Seattle":
        # Shift testOriginal back by 1 to match forecast at positions [6623:8063]
        # Instead of y[6624:8064], extract y[6623:8063]
        test_periods = int(test_len / freq)
        all_flattened = RawDataOriginal.reshape(-1, RawDataOriginal.shape[2])
        start_idx = len(all_flattened) - test_len - 1
        testOriginal = all_flattened[start_idx : start_idx + test_len, :]
    else:
        testOriginal = RawDataOriginal[-int(test_len / freq) :, :, :].reshape(
            -1, RawDataOriginal.shape[2]
        )
    # print(testForecast_mean.shape, testOriginal.shape)

    # Evaluation results
    res = evaluation(testForecast_mean.T, testOriginal.T)

    return (
        res,
        testForecast_mean,
        testOriginal,
        size,
        forecast_d_MC_argmax,
        testForecast_uq,
        testForecast_lq,
    )

def get_full_d_argmax(model, ds, forecaststep=1, MC_S=200):
    """
    Get d_argmax (regime indicators) for the entire dataset (train+valid+test).

    Parameters
    ----------
    model : DSSSM
        Trained DS3M model
    ds : dict
        Dataset dictionary from load_ds3m_data
    forecaststep : int
        Forecast horizon (typically 1)
    MC_S : int
        Number of Monte Carlo samples

    Returns
    -------
    d_argmax_full : np.ndarray, shape (N_total,)
        Regime indicators for full dataset (train+valid+test combined)
    """
    d_dim = ds["d_dim"]

    # Concatenate all splits
    X_full = torch.cat([ds["trainX"], ds["validX"], ds["testX"]], dim=1)  # (L, N_total, D)
    Y_full = torch.cat([ds["trainY"], ds["validY"], ds["testY"]], dim=1)  # (L, N_total, D)

    # Run forecasting to get regime indicators
    with torch.no_grad():
        forecast_MC, forecast_d_MC, forecast_z_MC = model._forecastingMultiStep(
            X_full, Y_full, forecaststep, MC_S
        )

    # Extract d_argmax (most likely regime at each timestep)
    forecast_d_MC_argmax = []
    for i in range(d_dim):
        forecast_d_MC_argmax.append(np.sum(forecast_d_MC[:, -1, :, :] == i, axis=0))
    d_argmax_full = np.argmax(np.array(forecast_d_MC_argmax), axis=0).reshape(-1)

    return d_argmax_full

# %%
# if restore == False:


# def retrain(
#     model,
#     optimizer,
#     trainX,
#     trainY,
#     validX,
#     validY,
#     testX,
#     testY,
#     epoch,
#     batch_size,
#     n_epochs,
#     x_dim,
#     y_dim,
#     h_dim,
#     z_dim,
#     d_dim,
#     n_layers,
#     learning_rate,
#     dataname,
#     directoryBest,
#     retry=1,
#     device="cpu",
# ):
#     unique_states = 0

#     while unique_states < 2:
#         # Init model
#         model = DSSSM(
#             x_dim, y_dim, h_dim, z_dim, d_dim, n_layers, device, bidirection
#         ).to(device)
#         # Optimizer
#         total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         print("The total number of parameters:", total_params)

#         optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#         scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
#         early_stopping = EarlyStopping(20, verbose=True)
#         loss_train_list, loss_valid_list, loss_test_list = [], [], []
#         best_validation = 1e5
#         best_validation_temp = 1e5

#         for i in range(retry):
#             print("n_epochs", n_epochs)
#             for epoch in range(1, n_epochs + 1):

#                 # Training
#                 (
#                     all_d_t_sampled_train,
#                     all_z_t_sampled_train,
#                     loss_train,
#                     all_d_posterior_train,
#                     all_z_posterior_mean_train,
#                 ) = train(model, optimizer, trainX, trainY, epoch, batch_size, n_epochs)

#                 # Validation
#                 if dataname in ["Unemployment", "Sleep"]:
#                     loss_valid = loss_train
#                 else:
#                     (
#                         all_d_t_sampled_valid,
#                         all_z_t_sampled_valid,
#                         loss_valid,
#                         all_d_posterior_valid,
#                         all_z_posterior_mean_valid,
#                     ) = test(model, validX, validY, epoch, "valid")

#                 # Testing
#                 (
#                     all_d_t_sampled_test,
#                     all_z_t_sampled_test,
#                     loss_test,
#                     all_d_posterior_test,
#                     all_z_posterior_mean_test,
#                 ) = test(model, testX, testY, epoch, "test")
#                 loss_train_list.append(loss_train)
#                 loss_valid_list.append(loss_valid)
#                 loss_test_list.append(loss_test)

#                 if loss_valid < best_validation:
#                     best_validation = copy.deepcopy(loss_valid)
#                     torch.save(
#                         {
#                             "epoch": epoch,
#                             "model_state_dict": model.state_dict(),
#                             "optimizer_state_dict": optimizer.state_dict(),
#                             "loss": loss_train,
#                         },
#                         os.path.join(directoryBest, "best_temp.tar"),
#                     )

#                 # Learning rate scheduler
#                 scheduler.step(loss_valid)
#                 print("Learning rate:", optimizer.param_groups[0]["lr"])

#                 # Early stopping
#                 loss_valid_average = np.average(loss_valid_list)
#                 early_stopping(loss_valid_average, model)
#                 if early_stopping.early_stop:
#                     print("Early stopping")
#                     break

#             print("Running Time:", time.time() - start)

#             if best_validation < best_validation_temp:
#                 best_validation_temp = best_validation
#                 PATH = os.path.join(directoryBest, "best.tar")
#                 checkpoint = torch.load(os.path.join(directoryBest, "best_temp.tar"))
#                 model.load_state_dict(checkpoint["model_state_dict"])
#                 optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#                 torch.save(
#                     {
#                         "epoch": checkpoint["epoch"],
#                         "model_state_dict": model.state_dict(),
#                         "optimizer_state_dict": optimizer.state_dict(),
#                         "loss": checkpoint["loss"],
#                     },
#                     PATH,
#                 )

#         # 训练后检查离散状态数
#         _, _, _, _, forecast_d_MC_argmax, _, _ = forecast(
#             model, testX, testY, forecaststep=1, MC_S=200
#         )
#         unique_states = len(np.unique(forecast_d_MC_argmax))

#     os.remove(os.path.join(directoryBest, "best_temp.tar"))


# %%
# Reload the parameters
def load_ds3m_model(
    directoryBest,
    x_dim,
    y_dim,
    h_dim,
    z_dim,
    d_dim,
    n_layers,
    learning_rate,
    device,
    bidirection=False,
):
    # check if checkpoint exists. if not, load best.tar
    if os.path.exists(os.path.join(directoryBest, "checkpoint.tar")):
        PATH = os.path.join(directoryBest, "checkpoint.tar")
    else:
        PATH = os.path.join(directoryBest, "best.tar")

    model = DSSSM(x_dim, y_dim, h_dim, z_dim, d_dim, n_layers, device, bidirection).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    checkpoint = torch.load(PATH, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]

    print("Epoch:", epoch)

    return model
