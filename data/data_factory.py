from typing import Dict, Any
from data.data_loader_final import TimeSeriesDataLoader, SimpleTimeSeriesDataLoader

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
    # Case 1: 2-Split Mode (Train(K) / Test(V))
    # ==========================================
    if getattr(cfg, "SPLIT_MODE", "3_split") == "2_split":
        print(f">> Initializing 2-Split Loader (Train(K) / Test(V))...")
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
                # "train_start_date": ... (Optional)
            }
        
        # Combine and Run
        final_kwargs = {**common_kwargs, **mode_kwargs}
        return loader.create_all(mode=cfg.MODE, **final_kwargs)

    # ==========================================
    # Case 2: 3-Split Mode (Train / Valid / Test)
    # ==========================================
    else:
        print(f">> Initializing 3-Split Loader (Original)...")
        loader = TimeSeriesDataLoader(base_path=cfg.DATA_PATH, num_assets=cfg.NUM_ASSETS)
        
        mode_kwargs = {}
        if cfg.MODE == "counts":
            # Map config variables to 3-split args
            mode_kwargs = {
                "train_len": cfg.TRAIN_LENGTH,
                "K": cfg.LEN_K,
                "V": cfg.LEN_V,
            }
        elif cfg.MODE == "dates":
            mode_kwargs = {
                "train_end_date": cfg.TRAIN_END_DATES,
                "val_end_date": cfg.VALID_END_DATES,
                "test_end_date": cfg.TEST_END_DATES,
            }
            
        final_kwargs = {**common_kwargs, **mode_kwargs}
        return loader.create_all(mode=cfg.MODE, **final_kwargs)