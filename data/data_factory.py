from typing import Dict, Any
from data.data_loader_final import TimeSeriesDataLoader, SimpleTimeSeriesDataLoader
from data.data_loader_multistep import DataLoaderMultiStep

def get_dataset(cfg) -> Dict[str, Any]:
    """
    Factory function to create the dataset dictionary.
    """
    
    norm_method = getattr(cfg, "NORM_METHOD", "scaling")
    use_scaler = (norm_method == "scaling")

    common_kwargs = {
        "lookback": cfg.LOOKBACK,
        "batch_size": cfg.BATCH_SIZE,
        "resample_freq": cfg.FREQUENCY,
        "use_scaler": use_scaler,  # If use RevIN scaling would be False
        "shuffle_train": True
    }

    # Case 1: Single-Step Forecasting
    if getattr(cfg, "PREDICTION_MODE", "single") == "single":
        loader = SimpleTimeSeriesDataLoader(base_path=cfg.DATA_PATH, num_assets=cfg.NUM_ASSETS)
        mode_kwargs = {
            "K": cfg.TRAIN_K_LEN, "V": cfg.TEST_V_LEN, "start_idx": 0
        } if cfg.MODE == "counts" else {
            "k_end_date": cfg.K_END_DATE, "v_end_date": cfg.V_END_DATE,
        }
        return loader.create_all(mode=cfg.MODE, **{**common_kwargs, **mode_kwargs})

    # Case 2: Multi-step Forecasting
    elif getattr(cfg, "PREDICTION_MODE", "multi") == "multi":
        loader = DataLoaderMultiStep(base_path=cfg.DATA_PATH, num_assets=cfg.NUM_ASSETS, horizon=cfg.HORIZON)
        mode_kwargs = {
            "K": cfg.TRAIN_K_LEN, "V": cfg.TEST_V_LEN, "start_idx": 0
        } if cfg.MODE == "counts" else {
            "k_end_date": cfg.K_END_DATE, "v_end_date": cfg.V_END_DATE,
        }
        return loader.create_all(mode=cfg.MODE, **{**common_kwargs, **mode_kwargs})
    
    else:
        raise ValueError(f"Unsupported PREDICTION_MODE: {cfg.PREDICTION_MODE}")