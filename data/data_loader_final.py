# ccpo/data/data_loader_final.py (수정 완료된 전체 코드)

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple, Optional, List, Union, Dict, Literal
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from utils.evaluation_utils import datestr


# -----------------------
# Dataset
# -----------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, unsqueeze_y: bool = False):
        self.X = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y)
        if unsqueeze_y and y_t.ndim == 2:
            y_t = y_t.unsqueeze(1)  # [N, d] -> [N, 1, d]
        self.y = y_t

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------
# Data Loader
# -----------------------
class TimeSeriesDataLoader:
    def __init__(
        self,
        base_path: str = "./data/",
        num_assets: int = 10,
    ):
        # <--- MODIFIED SECTION START --->
        def get_data_path(num_assets: int, base_dir: Union[str, Path] = ".") -> Path:
            allowed_assets = {5, 10, 30, 49}
            if num_assets not in allowed_assets:
                raise ValueError(f"num_assets must be one of {sorted(allowed_assets)}")

            # data_type을 'daily'로 고정
            return Path(base_dir) / f"industry_{num_assets}_daily.csv"

        self.data_path = get_data_path(num_assets, base_path)
        # <--- MODIFIED SECTION END --->

        self.raw_data: Optional[pd.DataFrame] = None
        self.data: Optional[pd.DataFrame] = None # Processed data (resampled, filled)
        self.scaler: Optional[StandardScaler] = None  # fit ONLY on "model train"

    # -------- I/O --------
    def load_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        df = pd.read_csv(self.data_path, index_col=0, parse_dates=True).sort_index()
        self.raw_data = df
        print(f"Loaded data: {df.shape} ({df.index.min()} ~ {df.index.max()})")
        return df

    # -------- Resample Helper --------
    def resample_frequency(
        self,
        data: pd.DataFrame,
        frequency: Literal['daily', 'weekly']
    ) -> pd.DataFrame:

        if frequency == 'daily':
            print("Using original 'daily' data.")
            self.data = data # Store processed data
            return data

        elif frequency == 'weekly':
            resampled = (1+data).resample('W-FRI').prod() - 1

        else:
            raise ValueError("frequency must be 'daily' or 'weekly'")

        if len(resampled) == 0:
              raise ValueError(f"Resampling to {frequency} ('W-FRI') resulted in empty DataFrame.")

        print(f"Resampled data to {frequency} ('W-FRI'): {resampled.shape}")
        self.data = resampled # Store processed data
        return resampled

    # -------- Scaling (fit/transform 분리) --------
    def fit_scaler(self, fit_df: pd.DataFrame):
        """스케일러는 반드시 '모델 Train(=K 시작 이전)' 구간으로만 fit."""
        if fit_df.empty:
             print("Warning: Cannot fit scaler on empty DataFrame.")
             self.scaler = None
             return
        self.scaler = StandardScaler().fit(fit_df.values)
        print(f"Scaler fitted on data range: {fit_df.index.min().date()} ~ {fit_df.index.max().date()}")


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None:
            print("Scaler not fitted, returning original data.")
            return df.copy()
        arr = self.scaler.transform(df.values)
        return pd.DataFrame(arr, index=df.index, columns=df.columns)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:

        if self.scaler is None:
            return data  # 스케일러 미사용 시 패스

        if data.ndim == 1:
            out = self.scaler.inverse_transform(data.reshape(1, -1))
            return out.reshape(-1)  # (features,)
        elif data.ndim == 2:
            return self.scaler.inverse_transform(data)
        elif data.ndim == 3:
            n, h, f = data.shape
            out2d = self.scaler.inverse_transform(data.reshape(n * h, f))
            return out2d.reshape(n, h, f)
        else:
            raise ValueError(f"inverse_transform: unsupported ndim={data.ndim}")

    # -------- Sequence 생성 (forecast_horizon=1 고정) --------
    def create_sequences(
        self,
        data: pd.DataFrame, # Changed input to DataFrame to keep index
        lookback: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
        """
        horizon=1 고정.
        X: [N, lookback, d], y: [N, d]
        pred_dates: 각 y 시점의 날짜
        """
        values = data.values
        dates = data.index
        X, y, pred_dates = [], [], []
        # Check if there's enough data to create even one sequence
        if len(values) <= lookback:
             print(f"Warning: Not enough data ({len(values)} points) to create sequences with lookback {lookback}.")
             # Return empty arrays with correct shapes
             num_features = values.shape[1] if values.ndim > 1 else 1
             return np.empty((0, lookback, num_features)), np.empty((0, num_features)), []

        for i in range(lookback, len(values)):
            X.append(values[i - lookback : i])
            y.append(values[i])  # horizon=1
            pred_dates.append(dates[i])
        X = np.array(X)
        y = np.array(y)
        print(f"Created sequences - X: {X.shape}, y: {y.shape}")
        return X, y, pred_dates

    # =========================
    # 날짜 기준: Model(train/valid/test) + Opt(y_K/y_V only, raw scale)
    # =========================
    def create_all_by_dates(
        self,
        lookback: int,
        # --- Arguments without defaults first ---
        train_end_date: str,
        val_end_date: str,
        # --- Arguments with defaults after (SyntaxError FIX) ---
        train_start_date: Optional[str] = None, # <-- 인자 추가됨
        test_end_date: Optional[str] = None,
        batch_size: int = 32,
        shuffle_train: bool = True,
        use_scaler: bool = True,
        frequency: Literal['daily', 'weekly'] = 'daily', # frequency 인자는 create_all에서 처리됨
    ) -> Dict[str, object]:
        if self.raw_data is None:
            self.load_data()

        # 리샘플링된 데이터 사용 (create_all에서 self.data에 저장됨)
        if self.data is None:
             raise RuntimeError("self.data not set. Call resample_frequency before create_all_by_dates.")
        data_to_use = self.data

        print("\n[Date mode] Model Split Setup")

        # 날짜 파싱
        train_start = pd.to_datetime(train_start_date) if train_start_date else None
        train_end = pd.to_datetime(train_end_date)
        val_end   = pd.to_datetime(val_end_date)
        test_end  = pd.to_datetime(test_end_date) if test_end_date else data_to_use.index[-1]


        # 데이터 시작점 결정
        if train_start:
            start_date = train_start
            print(f"Using provided train_start_date: {start_date.date()}")
        else:
            start_date = data_to_use.index[0]
            print(f"No train_start_date provided. Using earliest data point: {start_date.date()}")

        # lookback을 고려한 실제 데이터 시작 인덱스 찾기 (get_loc 오류 수정됨)
        mask_after_start = data_to_use.index >= start_date
        potential_start_loc = np.argmax(mask_after_start) if mask_after_start.any() else -1

        if potential_start_loc == -1:
            raise ValueError(f"No data found on or after train_start_date: {start_date}")

        effective_start_loc = max(lookback, potential_start_loc)

        if len(data_to_use) <= effective_start_loc:
             raise ValueError(f"Data length {len(data_to_use)} after start date is not sufficient for lookback {lookback} (effective start index: {effective_start_loc}).")

        effective_start_date = data_to_use.index[effective_start_loc]

        print(f"Effective data start date (considering lookback {lookback}): {effective_start_date.date()}")

        # 올바른 시작점에서 데이터 필터링
        filtered_data_to_use = data_to_use[data_to_use.index >= effective_start_date]


        if use_scaler:

            fit_df = data_to_use.loc[effective_start_date:train_end]
            self.fit_scaler(fit_df) # fit_scaler handles empty check

        # 2) 모델용 시퀀스 (필터링된 데이터 사용)
        data_scaled = self.transform(filtered_data_to_use)
        X_s, y_s, dates = self.create_sequences(data_scaled, lookback=lookback)
        dates = pd.to_datetime(dates)

        # 3) 최적화용 시퀀스 (필터링된 데이터 사용)
        _, y_raw, dates_raw = self.create_sequences(filtered_data_to_use, lookback)
        dates_raw = pd.to_datetime(dates_raw) # Convert to datetime
        if not (len(dates) == len(dates_raw) and np.all(dates == dates_raw)):
             print("Warning: Scaled/Raw sequence dates seem misaligned after filtering.")
             if len(dates) == len(dates_raw):
                 common_dates = dates.intersection(dates_raw)
                 print(f"Aligning based on {len(common_dates)} common dates.")
                 scale_mask = pd.Series(dates).isin(common_dates)
                 raw_mask = pd.Series(dates_raw).isin(common_dates)
                 X_s = X_s[scale_mask.values] if len(X_s) > 0 else X_s
                 y_s = y_s[scale_mask.values] if len(y_s) > 0 else y_s
                 dates = dates[scale_mask.values] if len(dates) > 0 else dates
                 y_raw = y_raw[raw_mask.values] if len(y_raw) > 0 else y_raw
                 dates_raw = dates_raw[raw_mask.values] if len(dates_raw) > 0 else dates_raw
                 if not np.all(dates == dates_raw):
                     raise AssertionError("Date alignment failed.")
             else:
                  raise AssertionError("Scaled/Raw sequence dates misaligned and lengths differ.")

        # 4) masks
        dates_idx = pd.DatetimeIndex(dates)
        train_mask = dates_idx <= train_end
        valid_mask = (dates_idx > train_end) & (dates_idx <= val_end)
        test_mask = (dates_idx > val_end) & (dates_idx <= test_end)


        # 5) 모델용 분할 → DataLoader
        X_tr, y_tr, d_tr = (X_s[train_mask], y_s[train_mask], dates_idx[train_mask]) if train_mask.any() else (np.empty((0, lookback, data_scaled.shape[1] if data_scaled.ndim > 1 else 1)), np.empty((0, data_scaled.shape[1] if data_scaled.ndim > 1 else 1)), pd.DatetimeIndex([]))
        X_va, y_va, d_va = (X_s[valid_mask], y_s[valid_mask], dates_idx[valid_mask]) if valid_mask.any() else (np.empty((0, lookback, data_scaled.shape[1] if data_scaled.ndim > 1 else 1)), np.empty((0, data_scaled.shape[1] if data_scaled.ndim > 1 else 1)), pd.DatetimeIndex([]))
        X_te, y_te, d_te = (X_s[test_mask],  y_s[test_mask],  dates_idx[test_mask])  if test_mask.any()  else (np.empty((0, lookback, data_scaled.shape[1] if data_scaled.ndim > 1 else 1)), np.empty((0, data_scaled.shape[1] if data_scaled.ndim > 1 else 1)), pd.DatetimeIndex([]))


        model_train_loader = DataLoader(TimeSeriesDataset(X_tr, y_tr, unsqueeze_y=True), batch_size=batch_size, shuffle=shuffle_train)
        model_valid_loader = DataLoader(TimeSeriesDataset(X_va, y_va, unsqueeze_y=True), batch_size=batch_size, shuffle=False)
        model_test_loader  = DataLoader(TimeSeriesDataset(X_te, y_te, unsqueeze_y=True), batch_size=batch_size, shuffle=False)

        # 6) 최적화용 분할
        y_K_shape_dim = y_raw.shape[1] if y_raw.ndim > 1 else (0 if y_raw.ndim == 1 and y_raw.size == 0 else 1) # Handle empty or 1d y_raw
        y_K = y_raw[valid_mask] if valid_mask.any() else np.empty((0, y_K_shape_dim))
        y_V = y_raw[test_mask]  if test_mask.any()  else np.empty((0, y_K_shape_dim))
        t_K = np.array(dates_idx[valid_mask]) if valid_mask.any() else np.array([])
        t_V = np.array(dates_idx[test_mask])  if test_mask.any()  else np.array([])


        print("\n[Date mode] Actual Model Split Used:")
        print(f"Train: {len(X_tr)} ({d_tr.min().date() if len(d_tr)>0 else 'N/A'} ~ {d_tr.max().date() if len(d_tr)>0 else 'N/A'})")
        print(f"Valid(K): {len(X_va)} ({d_va.min().date() if len(d_va)>0 else 'N/A'} ~ {d_va.max().date() if len(d_va)>0 else 'N/A'})")
        print(f"Test (V): {len(X_te)} ({d_te.min().date() if len(d_te)>0 else 'N/A'} ~ {d_te.max().date() if len(d_te)>0 else 'N/A'})")

        return {
            "model": {
                "train_loader": model_train_loader,
                "valid_loader": model_valid_loader,
                "test_loader":  model_test_loader,
                "dates": {"train": np.array(d_tr), "valid": np.array(d_va), "test": np.array(d_te)},
            },
            "opt": {
                "y_K": y_K, "dates_K": t_K,
                "y_V": y_V, "dates_V": t_V,
            },
            "scaler": self.scaler,
        }

    # =========================
    # 개수 기준: Model(train_len / K / V) + Opt(y_K/y_V only, raw scale)
    # =========================
    def create_all_by_counts(
        self,
        lookback: int,
        train_len: int,  # 모델 Train 시퀀스 개수(= K 시작 이전)
        K: int,          # Valid(K) 시퀀스 개수
        V: int,          # Test(V) 시퀀스 개수
        start_idx: int = 0,
        batch_size: int = 32,
        shuffle_train: bool = True,
        use_scaler: bool = True,
        frequency: Literal['daily', 'weekly'] = 'daily', # frequency 인자는 create_all에서 처리됨
    ) -> Dict[str, object]:
        if self.raw_data is None:
            self.load_data()

        if self.data is None:
             self.resample_frequency(self.raw_data, frequency)
        data_to_use = self.data

        # 필요한 총 원본 데이터 길이 계산
        total_sequence_len = train_len + K + V
        total_raw_len_needed = lookback + total_sequence_len

        end_idx = start_idx + total_raw_len_needed

        if end_idx > len(data_to_use):
            raise ValueError(
                f"Not enough data: need {total_raw_len_needed} raw points starting from index {start_idx}, "
                f"but only {len(data_to_use) - start_idx} available (after resampling)"
            )

        # 1) window
        window_df = data_to_use.iloc[start_idx:end_idx]

        # 2) scaler fit
        if use_scaler:
            fit_df = window_df.iloc[: (lookback + train_len)]
            self.fit_scaler(fit_df) # fit_scaler handles empty check

        # 3) 모델용 시퀀스
        scaled_df = self.transform(window_df)
        X_all, y_all, dates_all = self.create_sequences(scaled_df, lookback=lookback)

        # 4) 최적화용 시퀀스
        _, yr, dtr = self.create_sequences(window_df, lookback)
        if len(dates_all) != len(dtr):
             raise AssertionError("Scaled/Raw sequence length mismatch in count mode.")
        dates_all = pd.to_datetime(dates_all)
        dtr = pd.to_datetime(dtr)
        if not np.all(dates_all == dtr):
            raise AssertionError("Scaled/Raw sequence dates misaligned in count mode.")


        # 생성된 시퀀스 수 확인
        if len(X_all) < total_sequence_len:
            raise ValueError(
                f"Sequence count {len(X_all)} is less than required train_len+K+V={total_sequence_len}."
            )

        # 5) 인덱스 스플릿
        tr_slice = slice(0, train_len)
        va_slice = slice(train_len, train_len + K)
        te_slice = slice(train_len + K, total_sequence_len)

        # 모델용 분할 → DataLoader
        X_tr, y_tr = X_all[tr_slice], y_all[tr_slice]
        X_va, y_va = X_all[va_slice], y_all[va_slice]
        X_te, y_te = X_all[te_slice], y_all[te_slice]

        d_tr = np.array(dates_all[tr_slice])
        d_va = np.array(dates_all[va_slice])
        d_te = np.array(dates_all[te_slice])

        model_train_loader = DataLoader(TimeSeriesDataset(X_tr, y_tr, unsqueeze_y=True), batch_size=batch_size, shuffle=shuffle_train)
        model_valid_loader = DataLoader(TimeSeriesDataset(X_va, y_va, unsqueeze_y=True), batch_size=batch_size, shuffle=False)
        model_test_loader  = DataLoader(TimeSeriesDataset(X_te, y_te, unsqueeze_y=True), batch_size=batch_size, shuffle=False)

        # 최적화용 분할
        y_K = yr[va_slice]
        y_V = yr[te_slice]
        t_K = np.array(dtr[va_slice])
        t_V = np.array(dtr[te_slice])

        print("\n[Count mode] Model Split")
        print(f"Raw data start idx: {start_idx}, lookback: {lookback}")
        print(f"Sequence lengths: train_len={train_len}, K={K}, V={V}")
        print(f"Train seq: {len(X_tr)} ({datestr(d_tr[0]) if len(d_tr)>0 else 'N/A'} ~ {datestr(d_tr[-1]) if len(d_tr)>0 else 'N/A'})")
        print(f"Valid(K) seq: {len(X_va)} ({datestr(d_va[0]) if len(d_va)>0 else 'N/A'} ~ {datestr(d_va[-1]) if len(d_va)>0 else 'N/A'})")
        print(f"Test (V) seq: {len(X_te)} ({datestr(d_te[0]) if len(d_te)>0 else 'N/A'} ~ {datestr(d_te[-1]) if len(d_te)>0 else 'N/A'})")

        return {
            "model": {
                "train_loader": model_train_loader,
                "valid_loader": model_valid_loader,
                "test_loader":  model_test_loader,
                "dates": {"train": d_tr, "valid": d_va, "test": d_te},
            },
            "opt": {
                "y_K": y_K, "dates_K": t_K,
                "y_V": y_V, "dates_V": t_V,
            },
            "scaler": self.scaler,
        }

    def _require_keys(self, kwargs: Dict, keys: List[str]):
        """Helper to check for required keys in kwargs."""
        missing = [k for k in keys if k not in kwargs]
        if missing:
            raise ValueError(f"Missing required arguments for mode: {missing}")


    # =========================
    # Wrapper (create_all) - 수정됨
    # =========================
    def create_all(
        self,
        mode: Literal["counts", "dates"],
        **kwargs
    ):
        """
        mode에 따라 적절한 스플릿 함수를 호출한다.

        [resample_freq] (kwargs로 전달)
        - 'daily' (기본값) : 리샘플링 안함 (원본 데이터 사용)
        - 'weekly'        : 주간 리샘플링 ('W-FRI' - 금요일 기준)

        - mode='counts' -> create_all_by_counts(...)
            필수: lookback, train_len, K, V
            선택: start_idx, batch_size, shuffle_train, use_scaler, resample_freq
        - mode='dates'  -> create_all_by_dates(...)
            필수: lookback, train_end_date, val_end_date
            선택: train_start_date, test_end_date, batch_size, shuffle_train, use_scaler, resample_freq
        """

        # 1. 'resample_freq' 인자 추출 (기본값 'daily')
        frequency = kwargs.get("resample_freq", 'daily')

        if frequency not in ['daily', 'weekly']:
            raise ValueError(f"resample_freq must be 'daily' or 'weekly', but got {frequency}")

        if self.raw_data is None:
             self.load_data()
        self.resample_frequency(self.raw_data, frequency)
        
        if mode == "counts":
            self._require_keys(kwargs, ["lookback", "train_len", "K", "V"])
            return self.create_all_by_counts(
                lookback=kwargs["lookback"],
                train_len=kwargs["train_len"],
                K=kwargs["K"],
                V=kwargs["V"],
                start_idx=kwargs.get("start_idx", 0),
                batch_size=kwargs.get("batch_size", 32),
                shuffle_train=kwargs.get("shuffle_train", True),
                use_scaler=kwargs.get("use_scaler", True),
            )
            
        elif mode == "dates":
            self._require_keys(kwargs, ["lookback", "train_end_date", "val_end_date"])
            train_start_date = kwargs.get("train_start_date", None)

            return self.create_all_by_dates(
                lookback=kwargs["lookback"],

                train_end_date=kwargs["train_end_date"],
                val_end_date=kwargs["val_end_date"],
                train_start_date=train_start_date,
                test_end_date=kwargs.get("test_end_date", None),
                batch_size=kwargs.get("batch_size", 32),
                shuffle_train=kwargs.get("shuffle_train", True),
                use_scaler=kwargs.get("use_scaler", True),
            )
        else:
            raise ValueError("mode must be 'counts' or 'dates'")
        
class SimpleTimeSeriesDataLoader(TimeSeriesDataLoader):
    """
    Data Loader for 2-Split Mode: Train(K) / Test(V).
    - The 'Train' split is used for both Model Training and Calibration(K).
    - No separate Validation split.
    """

    # def create_2split_by_counts(
    #     self,
    #     lookback: int,
    #     k_len: int,  # Length for Train(K)
    #     v_len: int,  # Length for Test(V)
    #     start_idx: int = 0,
    #     batch_size: int = 32,
    #     shuffle_train: bool = True,
    #     use_scaler: bool = True,
    # ) -> Dict[str, object]:
        
    #     # 1. Prepare Data
    #     if self.raw_data is None: self.load_data()
    #     data_to_use = self.data if self.data is not None else self.raw_data

    #     total_len = k_len + v_len
    #     total_raw_needed = lookback + total_len
    #     end_idx = start_idx + total_raw_needed

    #     # Check data availability
    #     if end_idx > len(data_to_use):
    #         raise ValueError(f"Not enough data. Needed {total_raw_needed}, available {len(data_to_use)-start_idx}")

    #     # 2. Slicing Window
    #     window_df = data_to_use.iloc[start_idx:end_idx]
        
    #     # 3. Fit Scaler (Fit ONLY on Train(K) part)
    #     if use_scaler:
    #         fit_df = window_df.iloc[: (lookback + k_len)]
    #         self.fit_scaler(fit_df)

    #     scaled_df = self.transform(window_df)
    #     X_all, y_all, dates_all = self.create_sequences(scaled_df, lookback=lookback)
        
    #     # Raw Data (for Optimization / Calibration)
    #     _, y_raw, dates_raw = self.create_sequences(window_df, lookback=lookback)
        
    #     dates_all = pd.to_datetime(dates_all)
    #     dates_raw = pd.to_datetime(dates_raw)

    #     # 5. Split into Train(K) and Test(V)
    #     k_slice = slice(0, k_len)
    #     v_slice = slice(k_len, total_len)

    #     # Model Data
    #     X_k, y_k = X_all[k_slice], y_all[k_slice]
    #     X_v, y_v = X_all[v_slice], y_all[v_slice]
    #     d_k = dates_all[k_slice]
    #     d_v = dates_all[v_slice]

    #     # Optimization Data (Raw)
    #     y_opt_k = y_raw[k_slice]
    #     y_opt_v = y_raw[v_slice]
    #     d_opt_k = dates_raw[k_slice]
    #     d_opt_v = dates_raw[v_slice]

    #     # 6. Create Loaders
    #     train_loader = DataLoader(TimeSeriesDataset(X_k, y_k, unsqueeze_y=True), batch_size=batch_size, shuffle=shuffle_train)
    #     test_loader  = DataLoader(TimeSeriesDataset(X_v, y_v, unsqueeze_y=True), batch_size=batch_size, shuffle=False)
        
    #     print(f"\n[2-Split Mode] Counts Split Result:")
    #     print(f"Train(K): {len(X_k)} sequences ({d_k[0].date()} ~ {d_k[-1].date()})")
    #     print(f"Test(V) : {len(X_v)} sequences ({d_v[0].date()} ~ {d_v[-1].date()})")

    #     return {
    #         "model": {
    #             "train_loader": train_loader,
    #             "valid_loader": None,  # No separate validation set
    #             "test_loader":  test_loader,
    #             "dates": {"train": d_k, "valid": None, "test": d_v},
    #         },
    #         "opt": {
    #             "y_K": y_opt_k, "dates_K": d_opt_k,
    #             "y_V": y_opt_v, "dates_V": d_opt_v,
    #         },
    #         "scaler": self.scaler,
    #     }
    
    def create_2split_by_counts(
        self,
        lookback: int,
        k_len: int,  # Length for Train(K)
        v_len: int,  # Length for Test(V)
        start_idx: int = 0,
        batch_size: int = 32,
        shuffle_train: bool = True,
        use_scaler: bool = True,
    ) -> Dict[str, object]:
        
        # 1. Prepare Data
        if self.raw_data is None: self.load_data()
        data_to_use = self.data if self.data is not None else self.raw_data


        available_len = len(data_to_use) - start_idx
        min_needed_for_k = lookback + k_len

        if available_len < min_needed_for_k:
            raise ValueError(
                f"Not enough data for Train(K). Needed {min_needed_for_k}, "
                f"available {available_len} (start_idx={start_idx})"
            )

        total_raw_needed = lookback + k_len + v_len
        
        if available_len < total_raw_needed:
            adjusted_v_len = available_len - min_needed_for_k
            v_len = adjusted_v_len
        
        total_len = k_len + v_len
        end_idx = start_idx + lookback + total_len

        # 2. Slicing Window
        window_df = data_to_use.iloc[start_idx:end_idx]
        
        if use_scaler:
            fit_df = window_df.iloc[: (lookback + k_len)]
            self.fit_scaler(fit_df)

        scaled_df = self.transform(window_df)
        X_all, y_all, dates_all = self.create_sequences(scaled_df, lookback=lookback)
        
        # Raw Data (for Optimization / Calibration)
        _, y_raw, dates_raw = self.create_sequences(window_df, lookback=lookback)
        
        dates_all = pd.to_datetime(dates_all)
        dates_raw = pd.to_datetime(dates_raw)

        # 5. Split into Train(K) and Test(V)
        k_slice = slice(0, k_len)
        v_slice = slice(k_len, total_len)

        # Model Data
        X_k, y_k = X_all[k_slice], y_all[k_slice]
        X_v, y_v = X_all[v_slice], y_all[v_slice]
        d_k = dates_all[k_slice]
        d_v = dates_all[v_slice]

        y_opt_k = y_raw[k_slice]
        y_opt_v = y_raw[v_slice]
        d_opt_k = dates_raw[k_slice]
        d_opt_v = dates_raw[v_slice]

        train_loader = DataLoader(TimeSeriesDataset(X_k, y_k, unsqueeze_y=True), batch_size=batch_size, shuffle=shuffle_train)
        test_loader  = DataLoader(TimeSeriesDataset(X_v, y_v, unsqueeze_y=True), batch_size=batch_size, shuffle=False)
        
        print(f"\n[2-Split Mode] Counts Split Result:")
        print(f"Train(K): {len(X_k)} sequences ({d_k[0].date() if len(d_k)>0 else 'N/A'} ~ {d_k[-1].date() if len(d_k)>0 else 'N/A'})")
        print(f"Test(V) : {len(X_v)} sequences ({d_v[0].date() if len(d_v)>0 else 'N/A'} ~ {d_v[-1].date() if len(d_v)>0 else 'N/A'})")

        return {
            "model": {
                "train_loader": train_loader,
                "valid_loader": None,  # No separate validation set
                "test_loader":  test_loader,
                "dates": {"train": d_k, "valid": None, "test": d_v},
            },
            "opt": {
                "y_K": y_opt_k, "dates_K": d_opt_k,
                "y_V": y_opt_v, "dates_V": d_opt_v,
            },
            "scaler": self.scaler,
        }

    def create_2split_by_dates(
        self,
        lookback: int,
        k_end_date: str, # End date for Train(K)
        v_end_date: str, # End date for Test(V)
        batch_size: int = 32,
        shuffle_train: bool = True,
        use_scaler: bool = True,
        train_start_date: Optional[str] = None,
    ) -> Dict[str, object]:
        
        if self.raw_data is None: self.load_data()
        data_to_use = self.data if self.data is not None else self.raw_data

        k_end = pd.to_datetime(k_end_date)
        v_end = pd.to_datetime(v_end_date)
        
        # Determine Start Date
        if train_start_date:
            start_date = pd.to_datetime(train_start_date)
        else:
            start_date = data_to_use.index[0]

        # 1. Fit Scaler (Fit ONLY on Train(K) part)
        if use_scaler:
            # Fit on data from start_date up to k_end
            fit_mask = (data_to_use.index >= start_date) & (data_to_use.index <= k_end)
            fit_df = data_to_use[fit_mask]
            self.fit_scaler(fit_df)

        # 2. Sequence Creation
        filtered_data = data_to_use[data_to_use.index >= start_date]
        
        # Model Data (Scaled)
        scaled_data = self.transform(filtered_data)
        X_s, y_s, dates = self.create_sequences(scaled_data, lookback)
        
        # Opt Data (Raw)
        _, y_raw, dates_raw = self.create_sequences(filtered_data, lookback)
        
        dates = pd.to_datetime(dates)
        dates_raw = pd.to_datetime(dates_raw)
        
        # 3. Create Masks
        k_mask = (dates <= k_end)
        v_mask = (dates > k_end) & (dates <= v_end)

        # 4. Split
        X_k, y_k = X_s[k_mask], y_s[k_mask]
        X_v, y_v = X_s[v_mask], y_s[v_mask]
        
        y_opt_k = y_raw[k_mask]
        y_opt_v = y_raw[v_mask]
        
        # 5. Loaders
        train_loader = DataLoader(TimeSeriesDataset(X_k, y_k, unsqueeze_y=True), batch_size=batch_size, shuffle=shuffle_train)
        test_loader  = DataLoader(TimeSeriesDataset(X_v, y_v, unsqueeze_y=True), batch_size=batch_size, shuffle=False)

        print(f"\n[2-Split Mode] Dates Split Result:")
        print(f"Train(K): ~ {k_end.date()} ({len(X_k)} seqs)")
        print(f"Test(V) : ~ {v_end.date()} ({len(X_v)} seqs)")

        return {
            "model": {
                "train_loader": train_loader,
                "valid_loader": None,
                "test_loader":  test_loader,
                "dates": {"train": dates[k_mask], "valid": None, "test": dates[v_mask]},
            },
            "opt": {
                "y_K": y_opt_k, "dates_K": dates_raw[k_mask],
                "y_V": y_opt_v, "dates_V": dates_raw[v_mask],
            },
            "scaler": self.scaler,
        }

    # Override the wrapper to route to new methods
    def create_all(self, mode: Literal["counts", "dates"], **kwargs):
        # Handle resampling frequency
        frequency = kwargs.get("resample_freq", 'daily')
        if self.raw_data is None: self.load_data()
        self.resample_frequency(self.raw_data, frequency)

        if mode == "counts":
            return self.create_2split_by_counts(
                lookback=kwargs["lookback"],
                k_len=kwargs["K"],   
                v_len=kwargs["V"],   
                start_idx=kwargs.get("start_idx", 0),
                batch_size=kwargs.get("batch_size", 32),
                shuffle_train=kwargs.get("shuffle_train", True),
                use_scaler=kwargs.get("use_scaler", True),
            )
        elif mode == "dates":
            return self.create_2split_by_dates(
                lookback=kwargs["lookback"],
                k_end_date=kwargs["k_end_date"], 
                v_end_date=kwargs["v_end_date"], 
                train_start_date=kwargs.get("train_start_date"),
                batch_size=kwargs.get("batch_size", 32),
                shuffle_train=kwargs.get("shuffle_train", True),
                use_scaler=kwargs.get("use_scaler", True),
            )
        else:
            raise ValueError("Mode must be 'counts' or 'dates'")