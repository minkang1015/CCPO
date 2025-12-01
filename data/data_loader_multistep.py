"""
Multi-step data loader for daily input with multi-horizon prediction.

Unlike data_loader_final.py which resamples to weekly/monthly:
- Uses daily input data (no resampling)
- Predicts multiple horizons (t+1, ..., t+H)
- Aggregates predictions (mean or last) for weekly/monthly evaluation

Example:
    Daily: lookback=252 days -> predict t+1 (horizon=1)
    Weekly: lookback=252 days -> predict t+1~t+5 (horizon=5) -> mean/last
    Monthly: lookback=252 days -> predict t+1~t+20 (horizon=20) -> mean/last
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler


class DataLoaderMultiStep:
    """
    Multi-step prediction data loader.
    
    Key differences from data_loader_final:
    1. No resampling - always uses daily data
    2. create_sequences returns [N, lookback, d] -> [N, horizon, d]
    3. Supports aggregation (mean/last) for evaluation
    """
    
    def __init__(self, data_path: str, num_assets: int = 10):
        """
        Args:
            data_path: Path to data directory
            num_assets: Number of assets [5, 10, 30, 49]
        """
        self.data_path = data_path
        self.num_assets = num_assets
        self.scaler = StandardScaler()
        self.raw_data = None
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load daily industry return data."""
        import os
        
        filename = f"industry_{self.num_assets}_daily.csv"
        filepath = os.path.join(self.data_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        self.raw_data = df
        print(f"Loaded daily data: {df.shape} ({df.index[0].date()} to {df.index[-1].date()})")
    
    def fit_scaler(self, data: pd.DataFrame):
        """Fit scaler on training data."""
        self.scaler.fit(data.values)
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale data using fitted scaler."""
        scaled_values = self.scaler.transform(data.values)
        return pd.DataFrame(scaled_values, index=data.index, columns=data.columns)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform numpy array (compatible with TimeSeriesDataLoader).
        
        Supports:
        - 1D: [d,]
        - 2D: [N, d]
        - 3D: [N, horizon, d]
        """
        if data.ndim == 1:
            return self.scaler.inverse_transform(data.reshape(1, -1)).flatten()
        elif data.ndim == 2:
            return self.scaler.inverse_transform(data)
        elif data.ndim == 3:
            n, h, f = data.shape
            out2d = self.scaler.inverse_transform(data.reshape(n * h, f))
            return out2d.reshape(n, h, f)
        else:
            raise ValueError(f"inverse_transform: unsupported ndim={data.ndim}")
    
    def inverse_transform_ndarray(self, data: np.ndarray) -> np.ndarray:
        """Alias for inverse_transform (backward compatibility)."""
        return self.inverse_transform(data)
    
    def create_sequences_multistep(
        self,
        data: pd.DataFrame,
        lookback: int,
        horizon: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
        """
        Create multi-step sequences from daily data.
        
        Args:
            data: Daily return data [T, d]
            lookback: Number of past days to use as input (e.g., 252)
            horizon: Number of future days to predict (e.g., 1 for daily, 5 for weekly, 20 for monthly)
        
        Returns:
            X: [N, lookback, d] - Input sequences
            y: [N, horizon, d] - Target sequences (multi-step)
            pred_dates: List of target end dates (date of y[:, -1, :])
        """
        values = data.values
        dates = data.index
        X, y, pred_dates = [], [], []
        
        # Need lookback past + horizon future
        min_length = lookback + horizon
        if len(values) < min_length:
            print(f"Warning: Not enough data ({len(values)} points) for lookback={lookback}, horizon={horizon}")
            num_features = values.shape[1] if values.ndim > 1 else 1
            return np.empty((0, lookback, num_features)), np.empty((0, horizon, num_features)), []
        
        for i in range(lookback, len(values) - horizon + 1):
            X.append(values[i - lookback : i])           # [lookback, d]
            y.append(values[i : i + horizon])            # [horizon, d]
            pred_dates.append(dates[i + horizon - 1])    # Last date of prediction horizon
        
        X = np.array(X)  # [N, lookback, d]
        y = np.array(y)  # [N, horizon, d]
        
        print(f"Created multi-step sequences - X: {X.shape}, y: {y.shape}, horizon: {horizon}")
        return X, y, pred_dates
    
    def create_all_by_counts(
        self,
        train_len: int,
        test_len: int,
        lookback: int,
        horizon: int,
        train_start_idx: int = 0
    ):
        """
        Create train/test split by sequence counts (for rolling windows).
        
        Args:
            train_len: Number of training sequences
            test_len: Number of test sequences
            lookback: Lookback window
            horizon: Prediction horizon
            train_start_idx: Starting index for train period
        
        Returns:
            Dictionary with train/test data and metadata
        """
        # Use daily data directly (no resampling)
        daily_data = self.raw_data
        
        # Calculate date ranges
        # Total sequences needed
        total_sequences = train_len + test_len
        total_samples_needed = train_start_idx + lookback + total_sequences + horizon - 1
        
        if total_samples_needed > len(daily_data):
            raise ValueError(
                f"Not enough data: need {total_samples_needed} samples, "
                f"but only have {len(daily_data)}"
            )
        
        # Split data
        train_end_idx = train_start_idx + lookback + train_len
        test_end_idx = train_end_idx + test_len + horizon - 1
        
        train_data = daily_data.iloc[train_start_idx:train_end_idx]
        test_data = daily_data.iloc[train_start_idx:test_end_idx]  # Includes train for lookback
        
        # Fit scaler on train data
        self.fit_scaler(train_data)
        
        # Scale data
        train_scaled = self.transform(train_data)
        test_scaled = self.transform(test_data)
        
        # Create sequences
        X_train, Y_train, train_dates = self.create_sequences_multistep(
            train_scaled, lookback, horizon
        )
        X_test, Y_test, test_dates = self.create_sequences_multistep(
            test_scaled, lookback, horizon
        )
        
        # Get raw returns for portfolio evaluation
        # For test period: extract the actual dates we're predicting
        test_returns_raw = []
        for date in test_dates:
            # Get the data for this prediction window
            date_idx = daily_data.index.get_loc(date)
            # Get horizon-length window ending at this date
            window_data = daily_data.iloc[date_idx - horizon + 1 : date_idx + 1]
            test_returns_raw.append(window_data)
        
        result = {
            'X_train': X_train,
            'Y_train': Y_train,
            'X_test': X_test,
            'Y_test': Y_test,
            'train_dates': train_dates,
            'test_dates': test_dates,
            'test_returns_raw': test_returns_raw,  # List of DataFrames for each test sample
            'scaler': self.scaler,
            'lookback': lookback,
            'horizon': horizon,
            'frequency': 'daily',  # Always daily for multi-step loader
        }
        
        print(f"\n=== Multi-step Data Split (counts mode) ===")
        print(f"Horizon: {horizon} days")
        print(f"Train: {len(X_train)} sequences (dates: {train_dates[0].date()} to {train_dates[-1].date()})")
        print(f"Test: {len(X_test)} sequences (dates: {test_dates[0].date()} to {test_dates[-1].date()})")
        
        return result
    
    def aggregate_predictions(
        self,
        predictions: np.ndarray,
        method: str = "mean"
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
            return predictions.mean(axis=-2)  # Average over horizon dimension
        elif method == "last":
            return predictions[..., -1, :]    # Take last horizon step
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def resample_frequency(
        self,
        data: pd.DataFrame,
        frequency: str
    ) -> pd.DataFrame:
        """
        Resample data to target frequency.
        
        Compatible with TimeSeriesDataLoader interface.
        Multi-step loader uses daily data, so this is mainly for compatibility.
        
        Args:
            data: Input DataFrame (daily)
            frequency: Target frequency ("daily", "weekly", "monthly")
        
        Returns:
            Resampled DataFrame
        """
        if frequency.lower() in ["d", "day", "daily"]:
            return data
        elif frequency.lower() in ["w", "week", "weekly"]:
            return data.resample('W-FRI').last().dropna()
        elif frequency.lower() in ["m", "month", "monthly"]:
            return data.resample('M').last().dropna()
        else:
            raise ValueError(f"Unknown frequency: {frequency}")
    
    def resample_to_target_frequency(
        self,
        daily_predictions: List[pd.DataFrame],
        target_frequency: str = "weekly"
    ) -> pd.DataFrame:
        """
        Resample daily predictions to target frequency for evaluation.
        
        Args:
            daily_predictions: List of DataFrames with daily predictions
            target_frequency: "weekly" or "monthly"
        
        Returns:
            Resampled DataFrame
        """
        # Concatenate all predictions
        all_preds = pd.concat(daily_predictions)
        
        # Resample using unified method
        return self.resample_frequency(all_preds, target_frequency)