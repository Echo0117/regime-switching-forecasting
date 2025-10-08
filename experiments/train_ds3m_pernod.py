# %%
import random
import sys
import warnings
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib

HERE = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
    
from Deep_Switching_State_Space_Model.src.DSSSMCode import *
from Deep_Switching_State_Space_Model.src.utils import *
import torch
import matplotlib

from experiments.config import config
from experiments.utils.pernod_loader import prepare_data
from experiments.utils.plot_utils import plot_forecasts_and_regimes, plot_switches_vs_events

matplotlib.use('Agg')  # Set backend before importing pyplot
warnings.filterwarnings("ignore")

seed = 3
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------
# 1) Config & globals (keep original variable names)
# ---------------------------------------------------------------------
DataPath     = config["dataset"]["path"]
brand_list   = config["dataset"]["brand_list"]
exog_cols    = config["dataset"]["independent_variables"]
test_len     = config["modelTraining"]["test_len"]
freq         = config["modelTraining"]["freq"]
timestep     = config["modelTraining"]["timestep"]
predict_dim  = config["modelTraining"]["predict_dim"]
z_dim        = config["modelTraining"]["z_dim"]
d_dim        = config["modelTraining"]["d_dim"]
h_dim        = config["modelTraining"]["h_dim"]
n_layers     = config["modelTraining"]["n_layers"]
clip         = config["modelTraining"]["clip"]
learning_rate= config["modelTraining"]["learning_rate"]
batch_size   = config["modelTraining"]["batch_size"]
n_epochs     = config["modelTraining"]["n_epochs"]
valid_ratio  = config["modelTraining"]["valid_ratio"]
bidirection  = config["modelTraining"]["bidirection"]
event_cols   = config["event_cols"]
dataname     = "Pernod"

# Output dirs (keep names)
directoryBest = os.path.join("results", "checkpoints", dataname)
figdirectory = os.path.join("figures", "pernod")
os.makedirs(directoryBest, exist_ok=True)
os.makedirs(figdirectory, exist_ok=True)
figdirectory = figdirectory + '/' + dataname + '_'

# Final device override (keep your original behavior)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# 2) Functions (keep parameter/return names consistent with your usage)
# ---------------------------------------------------------------------



def init_model(x_dim, y_dim, h_dim, z_dim, d_dim, n_layers, device, bidirection):
    model = DSSSM(x_dim, y_dim, h_dim, z_dim, d_dim, n_layers, device, bidirection).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The total number of parameters:", total_params)
    return model

def forecast(model, testX, testY, scaler_y, forecaststep=1, MC_S=200):

    forecast_MC, forecast_d_MC, forecast_z_MC = model._forecastingMultiStep(
        testX, testY, forecaststep, MC_S)

    if forecaststep == 1:
        all_testForecast = scaler_y.inverse_transform(
            forecast_MC.squeeze(1).transpose(1, 0, 2))
    else:
        all_testForecast = scaler_y.inverse_transform(
            forecast_MC.squeeze(2).transpose(1, 0, 2))

    testY_inversed = scaler_y.inverse_transform(
        testY.cpu().numpy().transpose(1, 0, 2))
    size = testY_inversed.shape[0]

    forecast_d_MC_argmax = []
    for i in range(d_dim):
        forecast_d_MC_argmax.append(
            np.sum(forecast_d_MC[:, -1, :, :] == i, axis=0))
    forecast_d_MC_argmax = np.argmax(
        np.array(forecast_d_MC_argmax), axis=0).reshape(-1)

    testForecast_mean = np.mean(all_testForecast, axis=1)
    testForecast_uq = np.quantile(all_testForecast, 0.95, axis=1)
    testForecast_lq = np.quantile(all_testForecast, 0.05, axis=1)

    testOriginal = RawDataOriginal[-int(test_len/freq):, :, :].reshape(-1, RawDataOriginal.shape[2])
    # print(testForecast_mean.shape, testOriginal.shape)

    # Evaluation results
    res = evaluation(testForecast_mean.T, testOriginal.T)

    return res, testForecast_mean, testOriginal, size, forecast_d_MC_argmax, testForecast_uq, testForecast_lq

def train_until_min_states(model, trainX, trainY, validX, validY, testX, testY,
                           learning_rate, n_epochs, batch_size,
                           directoryBest, dataname, scaler_y):
    # uses global: device, d_dim, test_len, freq, RawDataOriginal, scaler
    scheduler_patience = 10
    early_patience = 20

    unique_states = 0

    while unique_states < 1:
        # model = DSSSM(x_dim, y_dim, h_dim, z_dim, d_dim,
        #               n_layers, device, bidirection).to(device)
        # total_params = sum(p.numel()
        #                    for p in model.parameters() if p.requires_grad)
        # print("The total number of parameters:", total_params)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                              threshold=1e-4, threshold_mode='rel', min_lr=1e-4)
        early_stopping = EarlyStopping(early_patience, verbose=True)
        
        loss_train_list, loss_valid_list, loss_test_list = [], [], []
        best_validation = 1e5
        best_validation_temp = 1e5

        # move to device (keeps original pattern)
        trX, trY = trainX.to(device), trainY.to(device)
        vaX, vaY = validX.to(device), validY.to(device)
        teX, teY = testX.to(device), testY.to(device)

        for epoch in range(1, n_epochs + 1):
            # Training
            all_d_t_sampled_train, all_z_t_sampled_train, loss_train, all_d_posterior_train, all_z_posterior_mean_train = train(
                model, optimizer, trX, trY, epoch, batch_size, n_epochs
            )

            # Validation (keep your special-case)
            # if dataname in ['Unemployment', 'Sleep']:
            #     loss_valid = loss_train
            # else:
            all_d_t_sampled_valid, all_z_t_sampled_valid, loss_valid, all_d_posterior_valid, all_z_posterior_mean_valid = test(
                model, vaX, vaY, epoch, "valid"
            )

            # Testing
            all_d_t_sampled_test, all_z_t_sampled_test, loss_test, all_d_posterior_test, all_z_posterior_mean_test = test(
                model, teX, teY, epoch, "test"
            )

            loss_train_list.append(loss_train)
            loss_valid_list.append(loss_valid)
            loss_test_list.append(loss_test)

            # Save best_temp
            if (loss_valid < best_validation):
                best_validation = copy.deepcopy(loss_valid)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_train,
                }, os.path.join(directoryBest, 'best_temp.tar'))

            # LR scheduler
            # print("Scheduler:", type(scheduler).__name__)
            scheduler.step(loss_valid)
            print("Learning rate:", optimizer.param_groups[0]['lr'])

            # Early stopping
            loss_valid_average = np.average(loss_valid_list)
            early_stopping(loss_valid_average, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # keep best across trials
        if best_validation < best_validation_temp:
            best_validation_temp = best_validation
            PATH = os.path.join(directoryBest, 'best.tar')
            checkpoint = torch.load(os.path.join(directoryBest, 'best_temp.tar'), map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            torch.save({
                'epoch': checkpoint['epoch'],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': checkpoint['loss'],
            }, PATH)

        # check number of unique states
        _, _, _, _, forecast_d_MC_argmax, _, _ = forecast(
            model, teX, teY, scaler_y, forecaststep=1, MC_S=200
        )
        unique_states = len(np.unique(forecast_d_MC_argmax))
        print("Unique regimes on test:", unique_states)

    os.remove(os.path.join(directoryBest, 'best_temp.tar'))

    # # cleanup
    # temp = os.path.join(directoryBest, 'best_temp.tar')
    # if os.path.exists(temp):
    #     os.remove(temp)


def do_plots_and_save(res, testForecast_mean, testOriginal, size,
                      forecast_d_MC_argmax, testForecast_uq, testForecast_lq,
                      brand, figdirectory, d_dim, DataPath, test_len):
    # color maps (keep names)
    my_cmap = matplotlib.cm.get_cmap('rainbow')
    cmap = plt.get_cmap('RdBu', d_dim)

    # 1) Forecast + regimes plots
    plot_forecasts_and_regimes(
        testOriginal=testOriginal,
        testForecast_mean=testForecast_mean,
        testForecast_uq=testForecast_uq,
        testForecast_lq=testForecast_lq,
        size=size,
        forecast_d_MC_argmax=forecast_d_MC_argmax,
        brand_name=brand,
        figdirectory=figdirectory,
        cmap=cmap
    )

    # Save RMSE/MAPE exactly as before
    save_rmse_mape("Pernod", res)

    # 2) Regime switches vs events (test set)
    df = pd.read_csv(DataPath, sep=";")
    dfb_test = df[df["brand_name"] == brand].sort_values("year_week").iloc[-test_len:]
    regime_ids = np.asarray(forecast_d_MC_argmax).reshape(-1)

    plot_switches_vs_events(
        dfb=dfb_test,
        regime_ids=regime_ids,
        event_cols=["is_christmas","is_easter","is_last_week_of_the_year"],
        time_col="year_week",
        save_path=f"{figdirectory}/regimes/{brand}_regime_switches.png"
    )


# ---------------------------------------------------------------------
# 3) Main flow 
# ---------------------------------------------------------------------

def main():
    global trainX, trainY, validX, validY, testX, testY, RawDataOriginal, scaler, x_dim, y_dim

    for brand in brand_list:   # loop over all brands
        print(f"\n============================")
        print(f"Running pipeline for brand: {brand}")
        print(f"============================")

        # --- Data ---
        trainX, trainY, validX, validY, testX, testY, RawDataOriginal, scaler_x,  scaler_y, x_dim, y_dim = prepare_data(
            DataPath, brand, test_len, valid_ratio, device
        )

        print(trainX.shape, trainY.shape)
        print(validX.shape, validY.shape)
        print(testX.shape, testY.shape)

        # --- Model ---
        print(f"Input dim: {x_dim}, Output dim: {y_dim}")
        print(f"Hidden dim: {h_dim}, Latent dim: {z_dim}, Regime dim: {d_dim}, N layers: {n_layers}")
        print(f"Batch size: {batch_size}, Epochs: {n_epochs}, Learning rate: {learning_rate}, Bidirectional: {bidirection}")
        model = init_model(x_dim, y_dim, h_dim, z_dim, d_dim, n_layers, device, bidirection)

        # Train until at least 2 unique states appear on test
        train_until_min_states(
            model, trainX, trainY, validX, validY, testX, testY,
            learning_rate, n_epochs, batch_size, directoryBest, dataname, scaler_y
        )

        # Load best and forecast
        checkpoint = torch.load(os.path.join(directoryBest, 'best.tar'), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        res, testForecast_mean, testOriginal, size, forecast_d_MC_argmax, testForecast_uq, testForecast_lq = forecast(
            model, testX.to(device), testY.to(device), scaler_y, forecaststep=1, MC_S=200
        )

        # --- Plots + save metrics ---
        do_plots_and_save(
            res, testForecast_mean, testOriginal, size,
            forecast_d_MC_argmax, testForecast_uq, testForecast_lq,
            brand,  # pass as list for plotting utils
            figdirectory, d_dim, DataPath, test_len
        )

if __name__ == "__main__":
    main()
