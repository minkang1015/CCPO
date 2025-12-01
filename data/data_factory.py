from typing import Dict, Any
from data.data_loader_final import TimeSeriesDataLoader, SimpleTimeSeriesDataLoader
from data.data_loader_multistep import DataLoaderMultiStep

def get_dataset(cfg) -> Dict[str, Any]:
    """
    Factory function to create the dataset dictionary based on SPLIT_MODE.
    Handles parameter mapping for both 2-split and 3-split modes.
    """
    
    # Common arguments
    common_kwargs = {
        "lookback": cfg.LOOKBACK,
        "batch_size": cfg.BATCH_SIZE,
        "resample_freq": cfg.FREQUENCY,
        "use_scaler": True,
        "shuffle_train": True
    }

    # ==========================================
    # Case 1: Single-Step Forecasting
    # ==========================================
    if getattr(cfg, "PREDICTION_TYPE", "single") == "single":
        print(f">> Initializing Single Step Prediction Loader (Train(K) / Test(V))...")
        loader = SimpleTimeSeriesDataLoader(base_path=cfg.DATA_PATH, num_assets=cfg.NUM_ASSETS)
        
        mode_kwargs = {}
        if cfg.MODE == "counts":
            # Map config variables to 2-split args
            mode_kwargs = {
                "K": cfg.TRAIN_K_LEN,   # K length (Train)
                "V": cfg.TEST_V_LEN,    # V length (Test)
                "start_idx": 0          # Default start
            }
        elif cfg.MODE == "dates":
            mode_kwargs = {
                "k_end_date": cfg.K_END_DATE,
                "v_end_date": cfg.V_END_DATE,
            }
        
        # Combine and Run
        final_kwargs = {**common_kwargs, **mode_kwargs}
        return loader.create_all(mode=cfg.MODE, **final_kwargs)

    # ==========================================
    # Case 2: Multi-step Forecasting
    # ==========================================
    elif getattr(cfg, "PREDICTION_TYPE", "multi") == "multi":
        print(f">> Initializing Multi Step Prediction Loader (Train(K) / Test(V))...")
        loader = DataLoaderMultiStep(base_path=cfg.DATA_PATH, num_assets=cfg.NUM_ASSETS, horizon=cfg.HORIZON)
        
        mode_kwargs = {}
        if cfg.MODE == "counts":
            # Map config variables to multi-step args
            mode_kwargs = {
                "K": cfg.TRAIN_K_LEN,   # K length (Train)
                "V": cfg.TEST_V_LEN,    # V length (Test)
                "start_idx": 0          # Default start
            }
        elif cfg.MODE == "dates":
            mode_kwargs = {
                "k_end_date": cfg.K_END_DATE,
                "v_end_date": cfg.V_END_DATE,
            }
            
        final_kwargs = {**common_kwargs, **mode_kwargs}
        return loader.create_all(mode=cfg.MODE, **final_kwargs)
    
    else:
        raise ValueError(f"Unsupported PREDICTION_TYPE: {cfg.PREDICTION_TYPE}. Choose either 'single' or 'multi'.")