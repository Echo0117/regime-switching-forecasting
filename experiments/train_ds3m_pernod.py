# experiments/train_ds3m_pernod.py
# -*- coding: utf-8 -*-
import os, sys, time, argparse, torch

HERE = os.path.dirname(__file__)
PROJ = os.path.abspath(os.path.join(HERE, ".."))
for p in [HERE, PROJ]:
    if p not in sys.path: sys.path.insert(0, p)

from experiments.utils.ds3m_utils import load_ds3m_data
from Deep_Switching_State_Space_Model.src.DSSSMCode import DSSSM, EarlyStopping, train, test
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="Pernod")
    ap.add_argument("--pernod-csv", default="Deep_Switching_State_Space_Model/data/pernod/pernod.csv")
    ap.add_argument("--pernod-date", default="date")
    ap.add_argument("--pernod-value", default="value")
    ap.add_argument("--pernod-freq", default=None, help="e.g., D|W|MS; None=auto")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pernod-brand-col", type=str, default="brand",
    help="Column name for brand filter in Pernod dataset")
    ap.add_argument("--pernod-brand", type=str, default="absolut",
        help="Brand name to filter in Pernod dataset")
    ap.add_argument("--pernod-sep", type=str, default=";",
        help="CSV separator)")
    
    # --- Pernod advanced options ---
    ap.add_argument("--pernod-resample", type=str, default=None,
        choices=[None, "raw", "D", "W", "MS"],
        help="Resample rule for target & features. None=>infer, 'raw' => no resample, 'W' weekly Monday, 'MS' month start")
    ap.add_argument("--pernod-agg", type=str, default="mean",
        choices=["mean", "sum"], help="Aggregation when resampling (mean/sum)")
    ap.add_argument("--pernod-fillna", type=str, default="ffill",
        help="Missing value handling: 'ffill'/'bfill'/'none' or a number (e.g. '0')")
    ap.add_argument("--pernod-normalize", default=True,
        help="Apply AbsoluteMedianScaler to Y (and X/Z if provided)")
    ap.add_argument("--pernod-drop-zero-y",default=False,
        help="Drop rows where Y==0 before normalization (robust for sales-like targets)")
    ap.add_argument("--pernod-x-cols", type=str, default="",
        help="Comma separated control columns for X (e.g. 'january,february,...')")
    ap.add_argument("--pernod-z-cols", type=str, default="",
        help="Comma separated marketing columns for Z")


    args = ap.parse_args()

    # make args visible to loader
    setattr(args, "pernod_csv", args.pernod_csv)
    setattr(args, "pernod_date", args.pernod_date)
    setattr(args, "pernod_value", args.pernod_value)
    setattr(args, "pernod_freq", args.pernod_freq)

    ds = load_ds3m_data(args)
    device = ds["device"]
    x_dim, y_dim = ds["x_dim"], ds["y_dim"]
    h_dim, z_dim, d_dim = ds["h_dim"], ds["z_dim"], ds["d_dim"]
    n_layers = ds["n_layers"]
    lr = ds["learning_rate"]
    directoryBest = ds["directoryBest"]

    # model = DSSSM(x_dim=x_np.shape[1], y_dim=1, h_dim=h_dim, z_dim=z_dim, d_dim=d_dim,
    #             n_layers=n_layers, device=device, dataname="Pernod").to(device)
    model = DSSSM(x_dim, y_dim, h_dim, z_dim, d_dim, n_layers, device, bidirection=ds["bidirection"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=10)
    early = EarlyStopping(patience=20, verbose=True)

    trainX, trainY = ds["trainX"], ds["trainY"]
    validX, validY = ds["validX"], ds["validY"]
    testX,  testY  = ds["testX"],  ds["testY"]

    best_valid = 1e12
    start = time.time()

    print(f"Training {args.problem} for up to {args.epochs} epochs")
    print(f"Architecture: x_dim={x_dim} y_dim={y_dim} h_dim={h_dim} z_dim={z_dim} d_dim={d_dim} n_layers={n_layers}")

    for epoch in range(1, args.epochs + 1):
        _, _, loss_tr, _, _ = train(model, opt, trainX, trainY, epoch, batch_size=64, n_epochs=args.epochs)
        # For small univariate sets, validation == training is acceptable fallback
        if ds["dataname"] in ["Unemployment", "Sleep", "Pernod"]:
            loss_val = loss_tr
        else:
            _, _, loss_val, _, _ = test(model, validX, validY, epoch, "valid")
        _, _, loss_te, _, _ = test(model, testX, testY, epoch, "test")

        scheduler.step(loss_val)
        early(loss_val, model)

        print(f"[{epoch:03d}] train={loss_tr:.4f}  valid={loss_val:.4f}  test={loss_te:.4f} "
              f"lr={opt.param_groups[0]['lr']:.2e}")

        # Save "best_temp"
        if loss_val < best_valid:
            best_valid = loss_val
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": float(loss_tr),
                "arch": dict(x_dim=x_dim, y_dim=y_dim, h_dim=h_dim, z_dim=z_dim,
                 d_dim=d_dim, n_layers=n_layers),
            }, os.path.join(directoryBest, "best_temp.tar"))

        if early.early_stop:
            print("Early stopping.")
            break

    # Promote best_temp -> checkpoint.tar (your loader expects 'checkpoint.tar')
    ck_best = torch.load(os.path.join(directoryBest, "best_temp.tar"), map_location="cpu")
    model.load_state_dict(ck_best["model_state_dict"])
    opt.load_state_dict(ck_best["optimizer_state_dict"])
    # torch.save({
    #     "epoch": ck_best["epoch"],
    #     "model_state_dict": model.state_dict(),
    #     "optimizer_state_dict": opt.state_dict(),
    #     "loss": ck_best["loss"],
    # }, os.path.join(directoryBest, "checkpoint.tar"))
    torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": opt.state_dict(),
    "arch": dict(x_dim=x_dim, y_dim=y_dim, h_dim=h_dim, z_dim=z_dim,
                 d_dim=d_dim, n_layers=n_layers),
}, os.path.join(directoryBest, "checkpoint.tar"))
    
    torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": opt.state_dict(),
    "arch": dict(x_dim=x_dim, y_dim=y_dim, h_dim=h_dim, z_dim=z_dim, d_dim=d_dim, n_layers=n_layers),
    # 新增：记录训练用到的特征名顺序
    # "feature_names": x_cols + z_cols,  # 列表，按训练拼接顺序
    }, os.path.join(directoryBest, "checkpoint.tar"))


    
    print(f"Saved {os.path.join(directoryBest, 'checkpoint.tar')}  (elapsed {time.time()-start:.1f}s)")

if __name__ == "__main__":
    main()