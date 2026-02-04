# import numpy as np
# from experiments.utils.ds3m_utils import load_ds3m_data, load_ds3m_model, forecast
# from experiments.utils.experiments_utils import save_forecast


# class BaseReg:
#     def fit(self, X, y): raise NotImplementedError
#     def predict(self, X): raise NotImplementedError


# class DS3MWrapper:
#     """
#     Thin adapter to use the DS³M forecasting code inside the universal harness.

#     Important:
#     - This wrapper DOES NOT consume (X, y) passed to .predict(...).
#       It mirrors your DS3M pipeline by calling load_ds3m_data(args) to obtain
#       normalized tensors (trainX/validX/testX, etc.) and then runs forecast(...)
#       on the prepared test tensors.
#     - It returns the DS³M one-step-ahead mean forecast for the test horizon.
#       If predict_dim > 1, it selects the column `target_dim`.

#     Parameters
#     ----------
#     lags : int
#         Unused here; present only for harness compatibility.
#     problem : str
#         Dataset name in your DS³M repo: {"Toy","Lorenz","Sleep","Unemployment",
#         "Hangzhou","Seattle","Pacific","Electricity"}.
#     target_dim : int
#         Which observation dimension to return (default 0).
#     train_size : int
#         Kept for compatibility / caching metadata (not used by DS³M).
#     device : str
#         "cpu" or "cuda" (DS³M utils choose device internally as well).
#     use_cache : bool
#         If True, try reading a cached DS³M forecast first (disabled by default).
#     force_new : bool
#         If True, ignore any cache and recompute the DS³M forecast.
#     """

#     def __init__(self, lags,
#                  problem=None,
#                  target_dim=0,
#                  train_size=200,
#                  device="cpu",
#                  use_cache=False,
#                  force_new=False):
#         self.lags = int(lags)
#         self.problem = problem
#         self.target_dim = int(target_dim)
#         self.train_size = int(train_size)
#         self.device = device
#         self.use_cache = bool(use_cache)
#         self.force_new = bool(force_new)

#     def _get_ds3m_forecast(self, args):
#         """
#         Load DS³M data/model the same way your training/eval pipeline does and
#         run forecast(...) on the prepared test tensors.
#         """
#         # 1) load DS³M dataset tensors and metadata
#         ds = load_ds3m_data(args)  # returns dict with trainX/validX/testX/testY as torch tensors

#         # 2) load DS³M checkpoint / model
#         model = load_ds3m_model(
#             ds["directoryBest"],
#             ds["x_dim"], ds["y_dim"], ds["h_dim"], ds["z_dim"],
#             ds["d_dim"], ds["n_layers"], ds["learning_rate"],
#             ds["device"], bidirection=ds["bidirection"],
#         )

#         # 3) one-step prediction on the prepared test tensors
#         #    NOTE: forecast(...) expects testX/testY as torch tensors from load_ds3m_data
#         res, testForecast_mean, testOriginal, size, d_argmax, uq, lq = forecast(
#             model,
#             ds["testX"], ds["testY"],   # ✅ torch tensors prepared in load_ds3m_data
#             ds["moments"], ds["d_dim"],
#             ds["means"], ds["trend"],
#             ds["test_len"], ds["freq"],
#             ds["RawDataOriginal"],
#             remove_mean=ds["remove_mean"],
#             remove_residual=ds["remove_residual"],
#         )


#         res_dict = dict(
#             y_pred_mean=testForecast_mean,   # np.ndarray, shape (T_test, D)
#             y_true=testOriginal,             # np.ndarray, shape (T_test, D)
#             size=size,
#             d_argmax=d_argmax,
#             y_uq=uq, y_lq=lq,
#             res_metric=res,
#             problem=self.problem,
#             train_size=self.train_size,
#             device=ds["device"],
#             test_len=ds["test_len"],
#             predict_dim=ds["predict_dim"],
#         )

#         # Optional caching (disabled by default)
#         # if self.use_cache and not self.force_new:
#         #     save_forecast(res_dict, self.problem)

#         return res_dict
    
#     def predict(self, X, y, args, *, test_offset = None):
#         """
#         Return a vector of length len(X). If test_offset is provided, slice the
#         precomputed DS³M test-horizon forecast starting at that offset.
#         """
#         X = np.asarray(X)
#         m = X.shape[0] if X.ndim >= 1 else 0
#         if m <= 0:
#             return np.array([], dtype=np.float32)

#         res = self._get_ds3m_forecast(args)

#         y_pred_mean = np.asarray(res["y_pred_mean"])  # (T_test, D) or (T_test,)
#         if y_pred_mean.ndim == 1:
#             y_pred_mean = y_pred_mean[:, None]

#         td = np.clip(self.target_dim, 0, y_pred_mean.shape[1] - 1)
#         series = y_pred_mean[:, td].astype(np.float32).reshape(-1)   # length = T_test

#         # If caller gave an offset into the DS³M test horizon, slice from there.
#         if test_offset is None:
#             start = 0
#         else:
#             start = int(test_offset)
#         start = max(0, min(start, series.shape[0]))  # clamp

#         tail = series[start : start + m]
#         if tail.shape[0] == m:
#             return tail
#         # pad if we ran past the end
#         if tail.size == 0:
#             pad_val = 0.0
#         else:
#             pad_val = float(tail[-1])
#         pad = np.full(m - tail.shape[0], pad_val, dtype=np.float32)
#         return np.concatenate([tail, pad])
        
#     # def predict(self, X, y, args):
#     #     """
#     #     Returns a vector whose length matches len(X) so ACI can subtract residuals
#     #     on calibration blocks and read a single value on test points.
#     #     """
#     #     X = np.asarray(X)
#     #     m = X.shape[0] if X.ndim >= 1 else 0
#     #     if m <= 0:
#     #         return np.array([], dtype=np.float32)

#     #     res = self._get_ds3m_forecast(args)

#     #     y_pred_mean = np.asarray(res["y_pred_mean"])  # (T_test, D) or (T_test,)
#     #     if y_pred_mean.ndim == 1:
#     #         y_pred_mean = y_pred_mean[:, None]

#     #     td = np.clip(self.target_dim, 0, y_pred_mean.shape[1] - 1)
#     #     series = y_pred_mean[:, td].astype(np.float32).reshape(-1)  # (T_test,)

#     #     # Enforce ACI contract: output length == len(X_block)
#     #     if series.shape[0] >= m:
#     #         return series[:m]
#     #     else:
#     #         pad = np.full(m - series.shape[0], series[-1] if series.size else 0.0, dtype=np.float32)
#     #         return np.concatenate([series, pad])

# def build_model(args):
#     return DS3MWrapper(
#         lags=getattr(args, "lags", 0),
#         problem=getattr(args, "problem", None),
#         target_dim=0,
#         train_size=getattr(args, "aci_train_size", 200),
#         device=("cuda" if getattr(args, "device", "cpu") == "cuda" else "cpu"),
#         use_cache=False,
#         force_new=False,
#     )



import numpy as np
from experiments.utils.ds3m_utils import load_ds3m_data, load_ds3m_model, forecast
from experiments.utils.experiments_utils import save_forecast

class DS3MWrapper:
    """
    DS³M 适配器：内部自己加载 DS³M 的数据与模型，只使用 load_ds3m_data(args) 里构造的 testX/testY。
    新增：
      - init_alignment(total_len, train_size, args): 自动计算对齐基准与 DS³M 测试区起点
      - predict_cal(X_block, y_prefix, args): 预测校准段（自动切片）
      - predict_test(x_row, y_prefix, args): 预测单个测试点（自动切片）
      - predict(X, y, args, test_offset=None): 仍保留；若不给 test_offset，按内部游标切
    """

    def __init__(self, lags, problem=None, target_dim=0,
                 train_size=200, device="cpu", use_cache=False, force_new=False):
        self.lags = int(lags)
        self.problem = problem
        self.target_dim = int(target_dim)
        self.train_size = int(train_size)
        self.device = device
        self.use_cache = bool(use_cache)
        self.force_new = bool(force_new)

        # for auto-alignment / cursor
        self._aligned = False
        self._ds = None
        self._series = None       # DS³M test forecast 1-D (length = test_len)
        self._t0_test = None      # 全序列上 DS³M 测试区的首索引 = N - test_len
        self._cursor = 0          # 当前游标（相对 DS³M 测试区）
        self._cal_len = None      # 记录 half-window 长度 m
        self._T0 = None           # 训练窗口长度 T0
        self._N  = None           # 全序列长度 N

    # ------- internal helpers -------
    def _ensure_ds3m_forecast(self, args):
        if self._series is not None:
            return

        ds = load_ds3m_data(args)
        model = load_ds3m_model(
            ds["directoryBest"],
            ds["x_dim"], ds["y_dim"], ds["h_dim"], ds["z_dim"],
            ds["d_dim"], ds["n_layers"], ds["learning_rate"],
            ds["device"], bidirection=ds["bidirection"],
        )
        res, testForecast_mean, testOriginal, size, d_argmax, uq, lq = forecast(
            model,
            ds["testX"], ds["testY"],
            ds["moments"], ds["d_dim"],
            ds["means"], ds["trend"],
            ds["test_len"], ds["freq"],
            ds["RawDataOriginal"],
            remove_mean=ds["remove_mean"],
            remove_residual=ds["remove_residual"],
        )
        y_pred_mean = np.asarray(testForecast_mean)  # (T_test, D) 或 (T_test,)
        if y_pred_mean.ndim == 1:
            y_pred_mean = y_pred_mean[:, None]
        td = np.clip(self.target_dim, 0, y_pred_mean.shape[1] - 1)
        self._series = y_pred_mean[:, td].astype(np.float32).reshape(-1)  # len = test_len
        self._ds = ds

    def _slice_from(self, start, length):
        """从 DS³M 测试序列 _series 的 start 位置取 length 个点，自动 clamp + pad。"""
        s = max(0, int(start))
        if self._series is None:
            return np.zeros(length, dtype=np.float32)
        end = s + int(length)
        if s >= len(self._series):
            return np.full(length, float(self._series[-1]), dtype=np.float32)
        out = self._series[s:end]
        if len(out) < length:
            pad_val = float(out[-1]) if len(out) > 0 else 0.0
            out = np.concatenate([out, np.full(length - len(out), pad_val, dtype=np.float32)])
        return out

    # ------- public API -------
    def init_alignment(self, total_len: int, train_size: int, args):
        """
        初始化 ACI 时间轴对齐：
          - total_len = 全序列长度 N（即 ACI 看到的 y_all 长度）
          - train_size = T0（ACI 窗口长度）
        自动计算：
          t0_test = N - ds_test_len
          并把游标置为 0（游标用于 predict_* 自动前进）
        """
        self._ensure_ds3m_forecast(args)
        self._N = int(total_len)
        self._T0 = int(train_size)
        self._cal_len = self._T0 // 2
        ds_test_len = int(self._ds["test_len"])
        self._t0_test = self._N - ds_test_len  # DS³M 测试区在全序列上的起点
        self._cursor = 0
        self._aligned = True
        # 可选：打印对齐信息
        print(f"[DS3MWrapper] aligned: N={self._N}, T0={self._T0}, test_len={ds_test_len}, t0_test={self._t0_test}")

    def predict_cal(self, X_block, y_prefix, args):
        """
        预测校准段：长度应为 m = T0//2。
        自动计算应该取的 DS³M 偏移，并与内部游标同步。
        """
        if not self._aligned:
            raise RuntimeError("Call init_alignment(N, T0, args) before predict_cal/predict_test.")
        self._ensure_ds3m_forecast(args)

        m = int(self._cal_len)
        # 期望拿到长度 m 的一段
        want = m if np.ndim(X_block) == 2 else int(np.asarray(X_block).shape[0])  # 保险
        want = m if want <= 0 else want

        # 计算“理论偏移” = (当前 i 的 cal 起点) - t0_test
        # 我们用“游标驱动”的方式：保证每轮调用顺序固定：warmup -> cal(m) -> test(1)
        # 这样 cal 就从当前游标开始取 m 个，随后把游标推进 m
        out = self._slice_from(self._cursor, want)
        self._cursor += want
        return out

    def predict_test(self, x_row, y_prefix, args):
        """
        预测单个测试点（长度=1）。自动从当前游标取 1 个，并推进游标。
        """
        if not self._aligned:
            raise RuntimeError("Call init_alignment(N, T0, args) before predict_cal/predict_test.")
        self._ensure_ds3m_forecast(args)
        out = self._slice_from(self._cursor, 1)
        self._cursor += 1
        return out

    def predict(self, X, y, args, *, test_offset = None):
        """
        兼容旧接口：
          - 若给了 test_offset，则从该位置切 length=len(X) 的一段返回；
          - 若没给 test_offset：
              * 若已 init_alignment，则按内部游标切（游标自动前进）；
              * 否则默认从 0 开始切。
        """
        X = np.asarray(X)
        want = X.shape[0] if X.ndim >= 1 else 0
        self._ensure_ds3m_forecast(args)
        if want <= 0:
            return np.array([], dtype=np.float32)

        if test_offset is not None:
            return self._slice_from(test_offset, want)

        if self._aligned:
            out = self._slice_from(self._cursor, want)
            self._cursor += want
            return out

        # 未对齐也可用：从 0 开始
        return self._slice_from(0, want)
    
    
def build_model(args):
    return DS3MWrapper(
        lags=getattr(args, "lags", 0),
        problem=getattr(args, "problem", None),
        target_dim=0,
        train_size=getattr(args, "aci_train_size", 200),
        device=("cuda" if getattr(args, "device", "cpu") == "cuda" else "cpu"),
        use_cache=False,
        force_new=False,
    )
