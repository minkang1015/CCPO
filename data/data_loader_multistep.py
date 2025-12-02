"""
Multi-step data loader for daily input with multi-horizon prediction.

Unlike TimeSeriesDataLoader (data_loader_final.py) which can resample:
- Uses daily *raw* data only (no resampling for inputs)
- Predicts multiple horizons (t+1, ..., t+H)
- Aggregates predictions (mean or last) for weekly/monthly evaluation (separate step)
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Literal, Dict
from torch.utils.data import DataLoader

from data.data_loader_final import TimeSeriesDataset, TimeSeriesDataLoader


class DataLoaderMultiStep(TimeSeriesDataLoader):
    """
    Multi-step prediction data loader.
    """

    def __init__(self, base_path: str = "./data/", num_assets: int = 10):
        """
        Args:
            base_path: directory containing industry_{num_assets}_daily.csv
            num_assets: Number of assets [5, 10, 30, 49]
        """
        super().__init__(base_path=base_path, num_assets=num_assets)

    # ------------------------------------------------------------------
    # Multi-step sequence creation
    # ------------------------------------------------------------------
    def create_sequences_multistep(
        self,
        data: pd.DataFrame,
        lookback: int,
        horizon: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
        """
        Create multi-step sequences from daily data.

        Args:
            data: [T, d] DataFrame (index: dates)
            lookback: Number of past days to use as input
            horizon: Number of future days to predict

        Returns:
            X: [N, lookback, d]
            y: [N, horizon, d]
            pred_dates: list of last target date for each sequence (y[:, -1, :])
        """
        values = data.values
        dates = data.index
        X, y, pred_dates = [], [], []

        min_length = lookback + horizon
        if len(values) < min_length:
            print(
                f"[MultiStep] Warning: Not enough data ({len(values)}) "
                f"for lookback={lookback}, horizon={horizon}"
            )
            num_features = values.shape[1] if values.ndim > 1 else 1
            return (
                np.empty((0, lookback, num_features)),
                np.empty((0, horizon, num_features)),
                [],
            )

        # i: 현재 윈도우의 "입력 끝" 인덱스 (predict window 시작점)
        for i in range(lookback, len(values) - horizon + 1):
            X.append(values[i - lookback : i])        # [lookback, d]
            y.append(values[i : i + horizon])         # [horizon, d]
            pred_dates.append(dates[i + horizon - 1]) # horizon 마지막 날짜

        X = np.array(X)
        y = np.array(y)
        print(f"[MultiStep] Created sequences - X: {X.shape}, y: {y.shape}, horizon={horizon}")
        return X, y, pred_dates

    # ------------------------------------------------------------------
    # counts mode
    # ------------------------------------------------------------------
    def create_all_by_counts(
        self,
        lookback: int,
        k_len: int,             # Length for Train(K)
        v_len: int,             # Length for Test(V)
        horizon: int,
        start_idx: int = 0,
        batch_size: int = 32,
        shuffle_train: bool = True,
        use_scaler: bool = True,
    ) -> Dict[str, object]:
        """
        2-split multi-step counts mode: Train(K) / Test(V).

        - Input data: ALWAYS daily raw_data (no resampling)
        - Model:
            X_K, y_K: [k_len, lookback, d], [k_len, horizon, d]
            X_V, y_V: [v_len, lookback, d], [v_len, horizon, d]
        - Opt:
            y_K, y_V: raw-scale multi-step targets [k_len, horizon, d], [v_len, horizon, d]
        """
        # 1. Data 준비
        if self.raw_data is None:
            self.load_data()
        data_to_use = self.raw_data  

        total_len = k_len + v_len
        total_raw_needed = lookback + total_len + horizon - 1
        end_idx = start_idx + total_raw_needed

        if end_idx > len(data_to_use):
            raise ValueError(
                f"[MultiStep] Not enough data. Needed {total_raw_needed}, "
                f"available {len(data_to_use) - start_idx}"
            )

        window_df = data_to_use.iloc[start_idx:end_idx]

        if use_scaler:
            fit_len = lookback + k_len + horizon - 1
            fit_df = window_df.iloc[:fit_len]
            self.fit_scaler(fit_df)

        # 4. Multi-step 시퀀스 생성 (scaled / raw)
        scaled_df = self.transform(window_df)
        X_all, y_all, dates_all = self.create_sequences_multistep(
            scaled_df, lookback=lookback, horizon=horizon
        )
        _, y_raw, dates_raw = self.create_sequences_multistep(
            window_df, lookback=lookback, horizon=horizon
        )

        dates_all = pd.to_datetime(dates_all)
        dates_raw = pd.to_datetime(dates_raw)

        if len(X_all) != total_len:
            raise ValueError(
                f"[MultiStep] Got {len(X_all)} sequences but expected {total_len} (k_len+v_len)."
            )
        if len(dates_all) != len(dates_raw) or not np.array_equal(dates_all, dates_raw):
            raise AssertionError("[MultiStep] Scaled/Raw sequence dates misaligned in count mode.")

        # 5. Train(K) / Test(V) 슬라이스
        k_slice = slice(0, k_len)
        v_slice = slice(k_len, total_len)

        X_k, y_k = X_all[k_slice], y_all[k_slice]
        X_v, y_v = X_all[v_slice], y_all[v_slice]
        d_k = dates_all[k_slice]
        d_v = dates_all[v_slice]

        y_opt_k = y_raw[k_slice]
        y_opt_v = y_raw[v_slice]
        d_opt_k = dates_raw[k_slice]
        d_opt_v = dates_raw[v_slice]

        # 6. DataLoader 생성 (y는 [N, H, d] → unsqueeze_y=False)
        train_loader = DataLoader(
            TimeSeriesDataset(X_k, y_k, unsqueeze_y=False),
            batch_size=batch_size,
            shuffle=shuffle_train,
        )
        test_loader = DataLoader(
            TimeSeriesDataset(X_v, y_v, unsqueeze_y=False),
            batch_size=batch_size,
            shuffle=False,
        )

        print(f"\n[MultiStep Count Mode] Result:")
        print(f"Train(K): {len(X_k)} sequences ({d_k[0].date()} ~ {d_k[-1].date()})")
        print(f"Test(V) : {len(X_v)} sequences ({d_v[0].date()} ~ {d_v[-1].date()})")

        return {
            "model": {
                "train_loader": train_loader,
                "valid_loader": None,  # No separate validation set
                "test_loader":  test_loader,
                "dates": {"train": d_k, "valid": None, "test": d_v},
            },
            "opt": {
                "y_K": y_opt_k, "dates_K": d_opt_k,   # [k_len, horizon, d]
                "y_V": y_opt_v, "dates_V": d_opt_v,   # [v_len, horizon, d]
            },
            "scaler": self.scaler,
            "horizon": horizon,
        }

    # ------------------------------------------------------------------
    # dates 모드: 2-split (K / V) 멀티스텝
    # ------------------------------------------------------------------
    def create_all_by_dates(
        self,
        lookback: int,
        horizon: int,
        k_end_date: str,  # End date for Train(K)  (기준: horizon 마지막 날짜)
        v_end_date: str,  # End date for Test(V)
        batch_size: int = 32,
        shuffle_train: bool = True,
        use_scaler: bool = True,
        train_start_date: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        2-split multi-step dates mode: Train(K) / Test(V) by dates.

        - dates는 모두 "horizon 마지막 타깃 날짜" 기준으로 마스킹
        """
        if self.raw_data is None:
            self.load_data()
        data_to_use = self.raw_data  # ALWAYS daily

        k_end = pd.to_datetime(k_end_date)
        v_end = pd.to_datetime(v_end_date)

        # Start date
        if train_start_date:
            start_date = pd.to_datetime(train_start_date)
        else:
            start_date = data_to_use.index[0]

        # 1. Scaler fit (Train(K) 기간까지만)
        if use_scaler:
            fit_mask = (data_to_use.index >= start_date) & (data_to_use.index <= k_end)
            fit_df = data_to_use[fit_mask]
            self.fit_scaler(fit_df)

        # 2. Sequence Creation (start_date 이후 데이터로)
        filtered_data = data_to_use[data_to_use.index >= start_date]

        scaled_data = self.transform(filtered_data)
        X_s, y_s, dates = self.create_sequences_multistep(
            scaled_data, lookback=lookback, horizon=horizon
        )
        _, y_raw, dates_raw = self.create_sequences_multistep(
            filtered_data, lookback=lookback, horizon=horizon
        )

        dates = pd.to_datetime(dates)
        dates_raw = pd.to_datetime(dates_raw)

        # 3. Masks (horizon 마지막 날짜 기준)
        k_mask = dates <= k_end
        v_mask = (dates > k_end) & (dates <= v_end)

        # 4. Split
        X_k, y_k = X_s[k_mask], y_s[k_mask]
        X_v, y_v = X_s[v_mask], y_s[v_mask]

        y_opt_k = y_raw[k_mask]
        y_opt_v = y_raw[v_mask]

        d_k = dates[k_mask]
        d_v = dates[v_mask]
        d_opt_k = dates_raw[k_mask]
        d_opt_v = dates_raw[v_mask]

        # 5. Loaders
        train_loader = DataLoader(
            TimeSeriesDataset(X_k, y_k, unsqueeze_y=False),
            batch_size=batch_size,
            shuffle=shuffle_train,
        )
        test_loader = DataLoader(
            TimeSeriesDataset(X_v, y_v, unsqueeze_y=False),
            batch_size=batch_size,
            shuffle=False,
        )

        print(f"\n[MultiStep 2-Split Dates Mode] Result:")
        print(f"Train(K): ~ {k_end.date()} ({len(X_k)} seqs)")
        print(f"Test(V) : ~ {v_end.date()} ({len(X_v)} seqs)")

        return {
            "model": {
                "train_loader": train_loader,
                "valid_loader": None,
                "test_loader":  test_loader,
                "dates": {"train": d_k, "valid": None, "test": d_v},
            },
            "opt": {
                "y_K": y_opt_k, "dates_K": d_opt_k,
                "y_V": y_opt_v, "dates_V": d_opt_v,
            },
            "scaler": self.scaler,
            "horizon": horizon,
        }

    # ------------------------------------------------------------------
    # Horizon 자동 추론 + Wrapper
    # ------------------------------------------------------------------
    def _infer_horizon_from_freq(self, frequency: str) -> int:
        """resample_freq(=평가 주기)로부터 기본 horizon 길이 추론."""
        f = frequency.lower()
        if f in ["d", "day", "daily"]:
            return 1
        elif f in ["w", "week", "weekly"]:
            return 5   # 예: 주간이면 t+1~t+5
        elif f in ["m", "month", "monthly"]:
            return 20  # 예: 월간이면 t+1~t+20
        else:
            raise ValueError(f"Cannot infer horizon from resample_freq={frequency}")

    def create_all(self, mode: Literal["counts", "dates"], **kwargs):
        # resample_freq는 여기서 horizon 추론용으로만 사용 (입력 resample X)
        resample_freq = kwargs.get("resample_freq", "daily")

        if mode == "counts":
            # 필수 키 체크 (horizon은 optional)
            for key in ["lookback", "K", "V"]:
                if key not in kwargs:
                    raise ValueError(f"Missing required argument for counts mode: {key}")

            lookback = kwargs["lookback"]
            k_len    = kwargs["K"]
            v_len    = kwargs["V"]

            horizon  = kwargs.get("horizon", None)
            if horizon is None:
                horizon = self._infer_horizon_from_freq(resample_freq)
                print(f"[MultiStep] horizon not provided; inferred horizon={horizon} from resample_freq={resample_freq}")

            return self.create_all_by_counts(
                lookback=lookback,
                k_len=k_len,
                v_len=v_len,
                horizon=horizon,
                start_idx=kwargs.get("start_idx", 0),
                batch_size=kwargs.get("batch_size", 32),
                shuffle_train=kwargs.get("shuffle_train", True),
                use_scaler=kwargs.get("use_scaler", True),
            )

        elif mode == "dates":
            for key in ["lookback", "k_end_date", "v_end_date"]:
                if key not in kwargs:
                    raise ValueError(f"Missing required argument for dates mode: {key}")

            lookback    = kwargs["lookback"]
            k_end_date  = kwargs["k_end_date"]
            v_end_date  = kwargs["v_end_date"]

            horizon  = kwargs.get("horizon", None)
            if horizon is None:
                horizon = self._infer_horizon_from_freq(resample_freq)
                print(f"[MultiStep] horizon not provided; inferred horizon={horizon} from resample_freq={resample_freq}")

            return self.create_all_by_dates(
                lookback=lookback,
                horizon=horizon,
                k_end_date=k_end_date,
                v_end_date=v_end_date,
                train_start_date=kwargs.get("train_start_date"),
                batch_size=kwargs.get("batch_size", 32),
                shuffle_train=kwargs.get("shuffle_train", True),
                use_scaler=kwargs.get("use_scaler", True),
            )

        else:
            raise ValueError("mode must be 'counts' or 'dates'")

    # ------------------------------------------------------------------
    # Horizon aggregation / resample (예측 후 평가용)
    # ------------------------------------------------------------------
    def aggregate_predictions(
        self,
        predictions: np.ndarray,
        method: str = "mean",
    ) -> np.ndarray:
        """
        Aggregate multi-horizon predictions.

        Args:
            predictions: [N, horizon, d] or [horizon, d]
            method: "mean" or "last"

        Returns:
            Aggregated predictions [N, d] or [d]
        """
        if method == "mean":
            return predictions.mean(axis=-2)   # horizon 축 평균
        elif method == "last":
            return predictions[..., -1, :]     # 마지막 스텝만
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def resample_frequency(
        self,
        data: pd.DataFrame,
        frequency: str,
    ) -> pd.DataFrame:
        """
        예측 결과(DataFrame)를 weekly/monthly로 resample할 때 사용.
        입력 features에는 사용하지 않는다.

        Args:
            data: daily index를 가진 DataFrame
            frequency: "daily", "weekly", "monthly"

        Returns:
            Resampled DataFrame
        """
        freq = frequency.lower()
        if freq in ["d", "day", "daily"]:
            return data
        elif freq in ["w", "week", "weekly"]:
            return data.resample("W-FRI").last().dropna()
        elif freq in ["m", "month", "monthly"]:
            return data.resample("M").last().dropna()
        else:
            raise ValueError(f"Unknown frequency: {frequency}")

    def resample_to_target_frequency(
        self,
        daily_predictions: List[pd.DataFrame],
        target_frequency: str = "weekly",
    ) -> pd.DataFrame:
        """
        Resample daily predictions to target frequency for evaluation.

        Args:
            daily_predictions: List of DataFrames with *daily* predictions
            target_frequency: "weekly" or "monthly"

        Returns:
            Resampled DataFrame
        """
        all_preds = pd.concat(daily_predictions)
        return self.resample_frequency(all_preds, target_frequency)
