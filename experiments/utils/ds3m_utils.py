# %%
import warnings
import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import time
import matplotlib
from pathlib import Path
import os, sys


from experiments.config import config
from experiments.utils.pernod_loader import DataPreprocessing, create_dataset_exog_only, make_exog_windows_seq_target, standardize_X_by_train
HERE = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


CACHE_DIR = Path("cached_forecasts")
CACHE_DIR.mkdir(exist_ok=True)


from Deep_Switching_State_Space_Model.src.DSSSMCode import *
from Deep_Switching_State_Space_Model.src.utils import *
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
    
    import numpy as np
    args.seed=42
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

    if dataname == "Toy":

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

        RawDataOriginal = pd.read_csv(
            "Deep_Switching_State_Space_Model/data/Toy/simulation_data_nonlinear_y.csv",
            header=None,
        ).values
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
    # if dataname == "Pernod":
    #     freq = 1
    #     timestep = 14
    #     test_len = 30
    #     predict_dim = 1
    #     h_dim = predict_dim  # Dimension of the hidden states in RNN
    #     z_dim = 10  # Dimension of the latent variable z
    #     d_dim = 2  # Dimension of the latent variable d
    #     n_layers = 1  # Number of the layers of the RNN
    #     clip = 10  # Gradient clips
    #     learning_rate = 1e-3  # Learning rate

    #     DataPath = "Deep_Switching_State_Space_Model/data/Pernod/pernod.csv"

    #     data_preprocessing = DataPreprocessing(
    #         DataPath,
    #         config["dataset"]["brand"],
    #         config["dataset"]["dependent_variable"],
    #         config["dataset"]["independent_variables_X"],
    #         config["dataset"]["independent_variables_Z"],
    #     )

    #     _, _, RawDataOriginal = data_preprocessing.preprocess(normalization=True)
        
    #     # RawDataOriginal = RawDataOriginal["volume_so"].values
        
    #     x_dim = len(config["dataset"]["independent_variables_Z"])  # Dimension of x
    #     y_dim = predict_dim  # Dimension of y #equal to predict_len

    #     RawDataOriginal = RawDataOriginal.reshape(-1, 1, predict_dim)
    if dataname == "Pernod":
        freq = 1
        timestep = 1
        test_len = 30
        predict_dim = 1   # 仍然单变量预测 Y
        y_dim = predict_dim

        # 读取并做预处理（建议 normalization=True，让 X/Z/Y 都在同一归一化空间）
        DataPath = "Deep_Switching_State_Space_Model/data/Pernod/pernod.csv"
        data_preprocessing = DataPreprocessing(
            DataPath,
            config["dataset"]["brand"],
            config["dataset"]["dependent_variable"],
            config["dataset"]["independent_variables_X"],  # 控制变量X
            config["dataset"]["independent_variables_Z"],  # 营销变量Z
        )
        X_t, Z_t, Y_t = data_preprocessing.preprocess(normalization=True)  # -> numpy

        # -------- 只用外生特征作为输入，不包含任何 y 的滞后 --------
        # 如果没有 X 或 Z，替换为 shape=(T,0) 的空数组并在后面报错更友好
        import numpy as np
        X_t = np.asarray(X_t, dtype=np.float32) if X_t is not None else np.zeros((len(Y_t), 0), np.float32)
        Z_t = np.asarray(Z_t, dtype=np.float32) if Z_t is not None else np.zeros((len(Y_t), 0), np.float32)
        inp = np.concatenate([X_t, Z_t], axis=1)  # (T, n_x+n_z)
        if inp.shape[1] == 0:
            raise ValueError("No exogenous features found. Provide X/Z columns to run exogenous-only forecasting.")

        x_dim = inp.shape[1]      # 输入维度 = 外生特征维数
        h_dim = max(16, min(64, x_dim * 2))  # 隐层可按需调整
        z_dim = 10
        d_dim = 2
        n_layers = 1
        clip = 10
        learning_rate = 1e-3

        # 目标 Y（注意：不把 Y 拼进输入）
        y_vec = np.asarray(Y_t, dtype=np.float32).reshape(-1)      # (T,)
        RawDataOriginal = y_vec.reshape(-1, 1, 1)                   # (T,1,1) 只用于后续评估/画图


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

    if remove_residual:
        trend = data[0:-1, :]
        data = data[1:, :] - trend

    # Split into train and test data
    if dataname == "Unemployment":
        length = len(data) - test_len
        train_len = int(length)
        train_data = data[:train_len]
        valid_data = data[:train_len]
        test_data = data[(-test_len - timestep - 1) : -1]

    elif dataname == "Lorenz":
        train_len = 1000 + timestep
        valid_len = len(data) - train_len - test_len
        train_data = data[:train_len]
        valid_data = data[(train_len) : (train_len + valid_len)]
        test_data = data[(-test_len - timestep - 1) : -1]

    elif dataname == "Sleep":
        train_len = 1000
        train_data = data[:train_len]
        valid_data = data[:train_len]
        test_data = data[-test_len - timestep :]
    else:
        length = len(data) - test_len
        train_len = int(length * 0.75)
        valid_len = int(length * 0.25)

        train_data = data[:train_len]
        valid_data = data[(train_len) : (train_len + valid_len)]
        test_data = data[(-test_len - timestep - 1) : -1]

    
    if dataname == "Pernod":
        # Split indices: keep the same policy as before: (train+valid) then fixed test_len
        N = len(y_vec)
        train_end = N - test_len                 # end of train+valid block
        length = train_end
        train_len = int(length * 0.75)
        valid_len = length - train_len
        assert train_len + valid_len == train_end

        # Normalize y using moments from train only; apply to full y
        moments = normalize_moments(y_vec[:train_len])
        y_all_s = normalize_fit(y_vec, moments)               # (N,)

        # Standardize X using stats from [0:train_end), apply to full X
        X_all_s, x_mu, x_sd = standardize_X_by_train(inp, train_end)  # (N, x_dim)

        # Build windows inside each partition
        trX, trY = make_exog_windows_seq_target(X_all_s, y_all_s, timestep, 0,            train_len)
        vaX, vaY = make_exog_windows_seq_target(X_all_s, y_all_s, timestep, train_len,    train_end)
        teX, teY = make_exog_windows_seq_target(X_all_s, y_all_s, timestep, train_end,    N)

        # Move to (seq_len, batch, feat) and (1, batch, 1)
        def to_model_axes(X_win, Y_win):
            if X_win.shape[0] == 0:
                X_torch = torch.empty((timestep, 0, x_dim), dtype=torch.float32)
                Y_torch = torch.empty((1, 0, 1), dtype=torch.float32)
                return X_torch, Y_torch
            X_win = np.transpose(X_win, (1, 0, 2))          # (k, B, x_dim)
            Y_win = Y_win.reshape(-1, 1, 1)                 # (B,1,1)
            Y_win = np.transpose(Y_win, (1, 0, 2))          # (1, B, 1)
            return torch.from_numpy(X_win).float(), torch.from_numpy(Y_win).float()

        trainX, trainY = to_model_axes(trX, trY)
        validX, validY = to_model_axes(vaX, vaY)
        testX,  testY  = to_model_axes(teX, teY)

        # Device placement
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainX = trainX.to(device); trainY = trainY.to(device)
        validX = validX.to(device); validY = validY.to(device)
        testX  = testX.to(device);  testY  = testY.to(device)

        # # Prepare common returns used elsewhere
        # directoryBest = os.path.join("Deep_Switching_State_Space_Model", "results", "checkpoints", dataname)
        # figdirectory = os.path.join("figures")
        # os.makedirs(directoryBest, exist_ok=True)
        # os.makedirs(figdirectory, exist_ok=True)
        # figdirectory = f"{figdirectory}/{dataname}_"

        # # 与原来一致的切分策略
        # length = len(y_vec) - test_len
        # train_len = int(length * 0.75)
        # valid_len = int(length * 0.25)

        # train_in  = inp[:train_len]
        # valid_in  = inp[train_len:train_len+valid_len]
        # test_in   = inp[(-test_len - timestep - 1) : -1]

        # train_y   = y_vec[:train_len]
        # valid_y   = y_vec[train_len:train_len+valid_len]
        # test_y    = y_vec[(-test_len - timestep - 1) : -1]
        # # # 只对 y 做 normalize_moments/normalize_fit（保持和你现有评估一致）
        # moments = normalize_moments(train_y)
        # train_y_n = normalize_fit(train_y, moments)
        # valid_y_n = normalize_fit(valid_y, moments)
        # test_y_n  = normalize_fit(test_y,  moments)
        # # train_y_n = train_y
        # # valid_y_n = valid_y
        # # test_y_n  = test_y
        # trainX, trainY = make_exog_windows_seq_target(train_in, train_y_n, timestep, stride=1)
        # validX, validY = make_exog_windows_seq_target(valid_in, valid_y_n, timestep, stride=1)
        # testX,  testY  = make_exog_windows_seq_target(test_in,  test_y_n,  timestep, stride=1)

        # # 转置到 (seq_len, batch, feat) 以匹配你现有的模型训练接口
        # trainX = np.transpose(trainX, (1, 0, 2))
        # validX = np.transpose(validX, (1, 0, 2))
        # testX  = np.transpose(testX,  (1, 0, 2))

        # trainY = np.transpose(trainY, (1, 0, 2))
        # validY = np.transpose(validY, (1, 0, 2))
        # testY  = np.transpose(testY,  (1, 0, 2))

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # trainX = torch.from_numpy(trainX).float().to(device)
        # validX = torch.from_numpy(validX).float().to(device)
        # testX  = torch.from_numpy(testX ).float().to(device)
        # trainY = torch.from_numpy(trainY).float().to(device)
        # validY = torch.from_numpy(validY).float().to(device)
        # testY  = torch.from_numpy(testY ).float().to(device)

        # %%
    else:
        # Normalize the dataset
        moments = normalize_moments(train_data)
        train_data = normalize_fit(train_data, moments)
        valid_data = normalize_fit(valid_data, moments)
        test_data = normalize_fit(test_data, moments)

        # Create training and test dataset
        trainX, trainY = create_dataset2(train_data, timestep)
        validX, validY = create_dataset2(valid_data, timestep)
        testX, testY = create_dataset2(test_data, timestep)

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
    directoryBest = os.path.join(
        "Deep_Switching_State_Space_Model", "results", "checkpoints", dataname
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
        
        "X_all_s": X_all_s if X_all_s is not None else None,      # np.ndarray shape (N, x_dim) standardized
        "y_all_s": y_all_s if y_all_s is not None else None,      # np.ndarray shape (N,)   standardized
        "train_end": train_end if train_end is not None else None  # int, e.g. N - test_len
    }


# %%
# Training
start = time.time()

def forecast(
    model,
    X_all_s,
    y_all_s,
    train_end,
    testX,              # kept for API compatibility; not used by exog-only path
    testY,              # kept for API compatibility; only used for size/moments invert
    moments,
    d_dim,
    means,
    trend,
    test_len,
    freq,
    RawDataOriginal,
    remove_mean=False,
    remove_residual=False,
    forecaststep=1,     # ignored for multi-step; we use test_len
    MC_S=200,
):
    import numpy as np
    import torch

    device = next(model.parameters()).device

    # Slice history / future (numpy) -> torch with expected shapes
    x_hist_np = X_all_s[:train_end]                          # (T_hist, x_dim)
    y_hist_np = y_all_s[:train_end]                          # (T_hist,)
    x_fut_np  = X_all_s[train_end : train_end + test_len]    # (H, x_dim)

    x_hist = torch.tensor(x_hist_np, dtype=torch.float32, device=device).unsqueeze(1)   # (T_hist,1,x_dim)
    y_hist = torch.tensor(y_hist_np, dtype=torch.float32, device=device).reshape(-1,1,1) # (T_hist,1,1)
    x_fut  = torch.tensor(x_fut_np,  dtype=torch.float32, device=device).unsqueeze(1)   # (H,1,x_dim)

    # Run multi-step exogenous-only forecast
    forecast_MC, forecast_d_MC, forecast_z_MC = model._forecastingMultiStep(
        x_hist, y_hist, step=test_len, S=MC_S, x_fut=x_fut
    )
    # forecast_MC: (S, step, B, y_dim) with B=1, y_dim=1 in your setup

    # Monte Carlo mean -> (step, B, y_dim)
    y_pred_s = forecast_MC.mean(axis=0)          # (step, 1, 1)

    # Invert normalization to original y space -> (step, 1, 1)
    y_pred_inv = normalize_invert(y_pred_s, moments)

    # For convenience, collapse (step, 1, 1) -> (step, 1)
    testForecast_mean = y_pred_inv.reshape(test_len, -1)

    # Also compute 5%/95% quantiles across MC -> (step, 1, 1) -> (step, 1)
    testForecast_uq = normalize_invert(np.quantile(forecast_MC, 0.95, axis=0), moments).reshape(test_len, -1)
    testForecast_lq = normalize_invert(np.quantile(forecast_MC, 0.05, axis=0), moments).reshape(test_len, -1)

    # Discrete state summary: take the last step’s most likely d across MC paths
    # forecast_d_MC shape depends on your implementation; this is a robust fallback:
    # try to read the last future-step samples and pick the mode across MC.
    try:
        # Example expected shape: (S, T_total, B, 1) of state indices
        last_d_samples = forecast_d_MC[:, -1, :, :].astype(int).squeeze(-1).squeeze(-1)  # (S,)
        # Mode across MC
        vals, counts = np.unique(last_d_samples, return_counts=True)
        d_mode = vals[np.argmax(counts)]
        forecast_d_MC_argmax = np.array([d_mode], dtype=int)  # (B=1,)
    except Exception:
        # Fallback: zeros if the structure differs
        forecast_d_MC_argmax = np.zeros((1,), dtype=int)

    # Build ground truth segment for evaluation in original space: (test_len, y_dim)
    testOriginal = RawDataOriginal[-int(test_len / freq):, :, :].reshape(-1, RawDataOriginal.shape[2])

    # Optionally add back mean/trend if your pipeline removed them
    if remove_mean:
        add_back = np.tile(means[0, :, :], (int(test_len / freq), 1))
        testForecast_mean = testForecast_mean + add_back
        testForecast_uq   = testForecast_uq   + add_back
        testForecast_lq   = testForecast_lq   + add_back
    elif remove_residual and trend is not None:
        add_back = trend[-test_len:, :]
        testForecast_mean = testForecast_mean + add_back
        testForecast_uq   = testForecast_uq   + add_back
        testForecast_lq   = testForecast_lq   + add_back

    # Evaluate (transpose to match your evaluation() expectations)
    res = evaluation(testForecast_mean.T, testOriginal.T)

    size = testOriginal.shape[0]  # = test_len

    return (
        res,
        testForecast_mean,     # (test_len, y_dim)
        testOriginal,          # (test_len, y_dim)
        size,
        forecast_d_MC_argmax,  # (B=1,)
        testForecast_uq,       # (test_len, y_dim)
        testForecast_lq,       # (test_len, y_dim)
    )

# def forecast(
#     model,
#     X_all_s,
#     y_all_s,
#     train_end,
#     testX,
#     testY,
#     moments,
#     d_dim,
#     means,
#     trend,
#     test_len,
#     freq,
#     RawDataOriginal,
#     remove_mean=False,
#     remove_residual=False,
#     forecaststep=1,
#     MC_S=200,
# ):
    
#     x_hist = X_all_s[:train_end]

#     y_hist = y_all_s[:train_end]

#     x_fut = X_all_s[train_end: train_end + test_len]

#     # forecast_MC, forecast_d_MC, forecast_z_MC = model._forecastingMultiStep(
#     #     testX, testY, forecaststep, MC_S
#     # )
#     forecast_MC, forecast_d_MC, forecast_z_MC = model._forecastingMultiStep(
#         x_hist, y_hist, step=test_len, S=200, x_fut=x_fut, use_lag=False, use_exog=True
#     )


#     if forecaststep == 1:
#         all_testForecast = normalize_invert(
#             forecast_MC.squeeze(1).transpose(1, 0, 2), moments
#         )
#     else:
#         all_testForecast = normalize_invert(
#             forecast_MC.squeeze(2).transpose(1, 0, 2), moments
#         )

#     testY_inversed = normalize_invert(testY.cpu().numpy().transpose(1, 0, 2), moments)
#     size = testY_inversed.shape[0]

#     forecast_d_MC_argmax = []
#     for i in range(d_dim):
#         forecast_d_MC_argmax.append(np.sum(forecast_d_MC[:, -1, :, :] == i, axis=0))
#     forecast_d_MC_argmax = np.argmax(np.array(forecast_d_MC_argmax), axis=0).reshape(-1)

#     if remove_mean:
#         testForecast_mean = np.mean(all_testForecast, axis=1) + np.tile(
#             means[0, :, :], (int(test_len / freq), 1)
#         )
#         testForecast_uq = np.quantile(all_testForecast, 0.95, axis=1) + np.tile(
#             means[0, :, :], (int(test_len / freq), 1)
#         )
#         testForecast_lq = np.quantile(all_testForecast, 0.05, axis=1) + np.tile(
#             means[0, :, :], (int(test_len / freq), 1)
#         )
#     elif remove_residual:
#         testForecast_mean = np.mean(all_testForecast, axis=1) + trend[-test_len:, :]
#         testForecast_uq = (
#             np.quantile(all_testForecast, 0.95, axis=1) + trend[-test_len:, :]
#         )
#         testForecast_lq = (
#             np.quantile(all_testForecast, 0.05, axis=1) + trend[-test_len:, :]
#         )
#     else:
#         testForecast_mean = np.mean(all_testForecast, axis=1)
#         testForecast_uq = np.quantile(all_testForecast, 0.95, axis=1)
#         testForecast_lq = np.quantile(all_testForecast, 0.05, axis=1)

#     testOriginal = RawDataOriginal[-int(test_len / freq) :, :, :].reshape(
#         -1, RawDataOriginal.shape[2]
#     )
#     # print(testForecast_mean.shape, testOriginal.shape)

#     # Evaluation results
#     res = evaluation(testForecast_mean.T, testOriginal.T)

#     return (
#         res,
#         testForecast_mean,
#         testOriginal,
#         size,
#         forecast_d_MC_argmax,
#         testForecast_uq,
#         testForecast_lq,
#     )


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
    PATH = os.path.join(directoryBest, "checkpoint.tar")

    # arch = checkpoint.get("arch", {})
    # if arch and arch.get("x_dim") != x_dim:
    #     raise ValueError(f"Checkpoint x_dim={arch.get('x_dim')} != current x_dim={x_dim}")


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


# %%


def plot_results(
    model,
    testForecast_mean,
    testOriginal,
    size,
    forecast_d_MC_argmax,
    testForecast_uq,
    testForecast_lq,
    RawDataOriginal,
    d_dim,
    predict_dim,
    dataname,
    figdirectory,
    device,
    test_len,
    z_true=None,
    trend=None,
    testX=None,
    testY=None,
    data=None,
    states=None,
    res=None,
    moments=None,
    freq=None,
    means=None,
    remove_mean=False,
    remove_residual=False,
    forecaststep=1,
    MC_S=200,
):

    my_cmap = matplotlib.cm.get_cmap("rainbow")
    cmap = plt.get_cmap("RdBu", d_dim)

    # %%
    if dataname == "Toy":

        d_original = pd.read_csv(
            "Deep_Switching_State_Space_Model/data/Toy/simulation_data_nonlinear_d.csv",
            header=None,
        ).values.reshape(-1)[-test_len:]
        z_original = pd.read_csv(
            "Deep_Switching_State_Space_Model/data/Toy/simulation_data_nonlinear_z.csv",
            header=None,
        ).values.reshape(-1)[-test_len:]

        DSARF = pd.read_csv(
            "Deep_Switching_State_Space_Model/data/Toy/Toy_s_forecasted_dsarf.csv",
            header=None,
        ).values
        DSARF = np.concatenate((np.array([[1]]), DSARF))
        SNLDS = pd.read_csv(
            "Deep_Switching_State_Space_Model/data/Toy/Toy_s_forecasted_snlds.csv",
            header=None,
        ).values

        DS3M = forecast_d_MC_argmax  # d(state) value
        acc = accuracy_score(d_original[-size:], DS3M)
        if acc < 0.5:
            DS3M = 1 - DS3M
        DSARF = DSARF
        SNLDS = 1 - SNLDS[-size:]

        inferX = (
            torch.from_numpy(np.expand_dims(data[(-test_len - 1) : -1], axis=0))
            .float()
            .to(device)
        )
        inferY = (
            torch.from_numpy(np.expand_dims(data[-test_len:], axis=0))
            .float()
            .to(device)
        )
        (
            all_d_t_sampled_plot_test,
            all_z_t_sampled_test,
            loss_test,
            all_d_posterior_test,
            all_z_posterior_mean_test,
        ) = test(model, inferX, inferY, 0, "test")
        d_infer = all_d_t_sampled_plot_test[:, 1, 0]
        d_infer = d_infer

        xticks_int = 10
        fig, (ax1, ax4, ax3, ax5, ax6) = plt.subplots(
            5,
            1,
            figsize=(20, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 0.5, 0.5, 0.5, 0.5]},
        )

        ######
        _ = ax1.plot(
            np.arange(size * predict_dim) / predict_dim,
            testOriginal.reshape(-1),
            label="original $y_t$",
        )
        _ = ax1.plot(
            np.arange(size * predict_dim) / predict_dim,
            testForecast_mean.reshape(-1),
            color="red",
            label="forecasted mean for $y_t$",
        )
        _ = ax1.plot(
            np.arange(size * predict_dim) / predict_dim,
            testForecast_uq.reshape(-1),
            color="grey",
            alpha=0.8,
        )
        _ = ax1.plot(
            np.arange(size * predict_dim) / predict_dim,
            testForecast_lq.reshape(-1),
            color="grey",
            alpha=0.8,
        )
        _ = ax1.fill_between(
            np.arange(size * predict_dim) / predict_dim,
            testForecast_uq.reshape(-1),
            testForecast_lq.reshape(-1),
            color="grey",
            alpha=0.2,
            label="90% confidence interval",
        )
        _ = ax1.set_xlim(0, size)
        _ = ax1.legend()
        _ = ax1.set_title("Prediction for $y_t$")

        ######
        _ = sns.heatmap(
            d_original[-size:].reshape(1, -1),
            linewidth=0,
            cbar=False,
            alpha=1,
            cmap=cmap,
            vmin=0,
            vmax=1,
            ax=ax4,
        )
        _ = ax4.set_xticks(np.round(np.arange(0, size, xticks_int)))
        _ = ax4.set_xticklabels(np.round(np.arange(0, size, xticks_int)))
        _ = ax4.set_ylabel("True", fontweight="bold", c="red")
        _ = ax4.set_yticks([])

        ######
        _ = sns.heatmap(
            DS3M.reshape(1, -1),
            linewidth=0,
            cbar=False,
            alpha=1,
            cmap=cmap,
            vmin=0,
            vmax=1,
            ax=ax3,
        )
        _ = ax3.set_xticks(np.round(np.arange(0, size, xticks_int)))
        _ = ax3.set_xticklabels(np.round(np.arange(0, size, xticks_int)))
        _ = ax3.set_ylabel("DS$^3$M", fontweight="bold")
        _ = ax3.set_yticks([])

        _ = sns.heatmap(
            DSARF.reshape(1, -1),
            linewidth=0,
            cbar=False,
            alpha=1,
            cmap=cmap,
            vmin=0,
            vmax=1,
            ax=ax5,
        )
        _ = ax5.set_xticks(np.round(np.arange(0, size, xticks_int)))
        _ = ax5.set_xticklabels(np.round(np.arange(0, size, xticks_int)))
        _ = ax5.set_ylabel("DSARF", fontweight="bold")
        _ = ax5.set_yticks([])

        _ = sns.heatmap(
            SNLDS.reshape(1, -1),
            linewidth=0,
            cbar=False,
            alpha=1,
            cmap=cmap,
            vmin=0,
            vmax=1,
            ax=ax6,
        )
        _ = ax6.set_xticks(np.round(np.arange(0, size, xticks_int)))
        _ = ax6.set_xticklabels(np.round(np.arange(0, size, xticks_int)))
        _ = ax6.set_ylabel("SNLDS", fontweight="bold")
        _ = ax6.set_yticks([])
        plt.savefig(figdirectory + "Prediction.png", format="png")
        plt.show()

        save_results_to_csv(
            dataname, d_original[-size:], DS3M, d_infer, res, type_val=""
        )

    # %%
    if dataname == "Lorenz":

        trainX2 = torch.from_numpy(RawDataOriginal[:1000, :, :]).float().to(device)
        trainY2 = torch.from_numpy(RawDataOriginal[1:1001, :, :]).float().to(device)
        (
            all_d_t_sampled_plot_test,
            all_z_t_sampled_test,
            loss_test,
            all_d_posterior_test,
            all_z_posterior_mean_test,
        ) = test(model, trainX2, trainY2, 0, "test")
        latents = z_true[0, 1:1001, :]
        categories = all_d_t_sampled_plot_test[:, 1:, :].reshape(-1)
        colormap = np.array(["r", "b", "g", "y"])
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(latents[:, 0], latents[:, 1], latents[:, 2], c=colormap[categories])
        plt.savefig(figdirectory + "Prediction.png", format="png")
        plt.show()

        testX2 = (
            torch.from_numpy(RawDataOriginal[(-test_len - 1) : -1, :, :])
            .float()
            .to(device)
        )
        testY2 = torch.from_numpy(RawDataOriginal[-test_len:, :, :]).float().to(device)
        (
            all_d_t_sampled_plot_test,
            all_z_t_sampled_test,
            loss_test,
            all_d_posterior_test,
            all_z_posterior_mean_test,
        ) = test(model, testX2, testY2, 0, "test")
        categories = all_d_t_sampled_plot_test[:, 1:, :].reshape(-1)

        save_results_to_csv(
            dataname,
            states.numpy().reshape(-1)[-testX.shape[1] :],
            forecast_d_MC_argmax,
            categories,
            res,
            type_val="",
        )

    # %%
    if dataname == "Sleep":

        xticks_int = 100

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(20, 4), sharex=True, gridspec_kw={"height_ratios": [3, 0.5]}
        )
        ax1.plot(
            np.arange(size * predict_dim) / predict_dim,
            testOriginal.reshape(-1),
            label="original",
        )
        ax1.plot(
            np.arange(size * predict_dim) / predict_dim,
            testForecast_mean.reshape(-1),
            color="red",
            label="forecasted mean",
        )
        ax1.plot(
            np.arange(size * predict_dim) / predict_dim,
            testForecast_uq.reshape(-1),
            color="grey",
            alpha=0.8,
        )
        ax1.plot(
            np.arange(size * predict_dim) / predict_dim,
            testForecast_lq.reshape(-1),
            color="grey",
            alpha=0.8,
        )
        ax1.fill_between(
            np.arange(size * predict_dim) / predict_dim,
            testForecast_uq.reshape(-1),
            testForecast_lq.reshape(-1),
            color="grey",
            alpha=0.2,
        )
        ax1.legend()

        sns.heatmap(
            1 - forecast_d_MC_argmax.reshape(1, -1),
            linewidth=0,
            cbar=False,
            alpha=1,
            cmap=cmap,
            vmin=0,
            vmax=1,
            ax=ax2,
        )
        ax2.set_xticks(np.round(np.arange(0, size, xticks_int)))
        ax2.set_xticklabels(np.round(np.arange(0, size, xticks_int)))
        ax5 = ax2.twinx()

        plt.savefig(figdirectory + "Prediction.png", format="png")
        plt.show()

        save_rmse_mape(dataname, res)

    # %%
    if dataname == "Unemployment":

        xticklabels = list()
        for i in np.arange(2002, 2022):
            xticklabels.append(str(i) + " Jan")

        xticks_int = 100
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(20, 4), sharex=True, gridspec_kw={"height_ratios": [3, 0.5]}
        )

        _ = ax1.plot(testOriginal.reshape(-1), label="original")
        _ = ax1.plot(testForecast_mean.reshape(-1), color="red", label="predict")
        _ = ax1.plot(testForecast_uq.reshape(-1), color="grey", alpha=0.8)
        _ = ax1.plot(testForecast_lq.reshape(-1), color="grey", alpha=0.8)
        _ = ax1.fill_between(
            np.arange(size),
            testForecast_uq.reshape(-1),
            testForecast_lq.reshape(-1),
            color="grey",
            alpha=0.4,
        )
        _ = ax1.legend()

        _ = sns.heatmap(
            1 - forecast_d_MC_argmax.reshape(1, -1),
            linewidth=0,
            cbar=False,
            alpha=1,
            cmap=cmap,
            vmin=0,
            vmax=1,
            ax=ax2,
        )

        _ = ax2.set_xticks(np.arange(9, test_len, 12))
        _ = ax2.set_xticklabels(xticklabels)
        _ = plt.xticks(rotation=0)
        _ = plt.suptitle("%s" % (dataname))
        plt.savefig(figdirectory + "Prediction.png", format="png")
        plt.show()

        save_rmse_mape(dataname, res)

    # %%
    if dataname == "Hangzhou":
        for station in [0, 40]:
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(10, 4),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 0.5]},
            )
            ax1.plot(testOriginal[:, station], label="original")
            ax1.plot(testForecast_mean[:, station], label="predict")
            ax1.plot(testForecast_uq[:, station], color="grey", alpha=0.8)
            ax1.plot(testForecast_lq[:, station], color="grey", alpha=0.8)
            ax1.fill_between(
                np.arange(size),
                testForecast_uq[:, station],
                testForecast_lq[:, station],
                color="grey",
                alpha=0.4,
            )
            ax1.legend()
            sns.heatmap(
                1 - forecast_d_MC_argmax.reshape(1, -1),
                linewidth=0,
                cbar=False,
                alpha=1,
                cmap=cmap,
                vmin=0,
                vmax=1,
                ax=ax2,
            )

            plt.suptitle("%s #%i" % (dataname, station))
            plt.savefig(figdirectory + "Station %i" % (station) + ".png", format="png")
            plt.show()

        save_rmse_mape(dataname, res)

        print("Long term:")
        (
            res,
            testForecast_mean,
            testOriginal,
            size,
            forecast_d_MC_argmax,
            testForecast_uq,
            testForecast_lq,
        ) = forecast(
            model,
            testX[:, 0:1, :],
            testY[:, 0:1, :],
            moments,
            d_dim,
            means,
            trend,
            test_len,
            freq,
            RawDataOriginal,
            remove_mean,
            remove_residual,
            forecaststep,
            MC_S,
        )

        save_rmse_mape(dataname, res, "Long-term")

    # %%
    if dataname == "Seattle":
        for station in [0, 322]:
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(10, 4),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 0.5]},
            )
            ax1.plot(testOriginal[:, station], label="original")
            ax1.plot(testForecast_mean[:, station], label="predict")
            ax1.plot(testForecast_uq[:, station], color="grey", alpha=0.8)
            ax1.plot(testForecast_lq[:, station], color="grey", alpha=0.8)
            ax1.fill_between(
                np.arange(size),
                testForecast_uq[:, station],
                testForecast_lq[:, station],
                color="grey",
                alpha=0.4,
            )
            ax1.legend()
            sns.heatmap(
                forecast_d_MC_argmax.reshape(1, -1),
                linewidth=0,
                cbar=False,
                alpha=1,
                cmap=cmap,
                vmin=0,
                vmax=1,
                ax=ax2,
            )

            plt.suptitle("%s #%i" % (dataname, station))
            plt.savefig(figdirectory + "Station %i" % (station) + ".png", format="png")
            plt.show()

        save_rmse_mape(dataname, res)

        print("Long term:")
        (
            res,
            testForecast_mean,
            testOriginal,
            size,
            forecast_d_MC_argmax,
            testForecast_uq,
            testForecast_lq,
        ) = forecast(
            model,
            testX[:, 0:1, :],
            testY[:, 0:1, :],
            moments,
            d_dim,
            means,
            trend,
            test_len,
            freq,
            RawDataOriginal,
            remove_mean,
            remove_residual,
            forecaststep,
            MC_S,
        )

        save_rmse_mape(dataname, res, "Long-term")

    # %%
    if dataname == "Pacific":

        for station in [0, 840]:
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(10, 4),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 0.5]},
            )

            ax1.plot(testOriginal[:, station], label="original")
            ax1.plot(testForecast_mean[:, station], label="predict")
            ax1.plot(testForecast_uq[:, station], color="grey", alpha=0.8)
            ax1.plot(testForecast_lq[:, station], color="grey", alpha=0.8)
            ax1.fill_between(
                np.arange(size),
                testForecast_uq[:, station],
                testForecast_lq[:, station],
                color="grey",
                alpha=0.4,
            )
            ax1.legend()

            sns.heatmap(
                forecast_d_MC_argmax.reshape(1, -1),
                linewidth=0,
                cbar=False,
                alpha=1,
                cmap=cmap,
                vmin=0,
                vmax=1,
                ax=ax2,
            )

            plt.suptitle("%s #%i" % (dataname, station))
            plt.savefig(figdirectory + "Station %i" % (station) + ".png", format="png")
            plt.show()

        save_rmse_mape(dataname, res)

        print("Long term:")
        (
            res,
            testForecast_mean,
            testOriginal,
            size,
            forecast_d_MC_argmax,
            testForecast_uq,
            testForecast_lq,
        ) = forecast(
            model,
            testX[:, 0:1, :],
            testY[:, 0:1, :],
            moments,
            d_dim,
            means,
            trend,
            test_len,
            freq,
            RawDataOriginal,
            remove_mean,
            remove_residual,
            forecaststep,
            MC_S,
        )

        save_rmse_mape(dataname, res, "Long-term")

    # %%
    if dataname == "Electricity":
        for station in [0, 24]:
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(10, 4),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 0.5]},
            )
            ax1.plot(testOriginal[:, station], label="original")
            ax1.plot(testForecast_mean[:, station], label="predict")
            ax1.plot(testForecast_uq[:, station], color="grey", alpha=0.8)
            ax1.plot(testForecast_lq[:, station], color="grey", alpha=0.8)
            ax1.fill_between(
                np.arange(size),
                testForecast_uq[:, station],
                testForecast_lq[:, station],
                color="grey",
                alpha=0.4,
            )
            ax1.legend()
            sns.heatmap(
                forecast_d_MC_argmax.reshape(1, -1),
                linewidth=0,
                cbar=False,
                alpha=1,
                cmap=cmap,
                vmin=0,
                vmax=1,
                ax=ax2,
            )

            plt.suptitle("%s #%i" % (dataname, station))
            plt.savefig(figdirectory + "Station %i" % (station) + ".png", format="png")
            plt.show()

        save_rmse_mape(dataname, res)
