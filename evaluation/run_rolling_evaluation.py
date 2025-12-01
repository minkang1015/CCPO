import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pandas.tseries.offsets import DateOffset
from configs import config_revised as config
from data.data_loader_final import TimeSeriesDataLoader
from utils.portfolios import Portfolio
from utils.evaluation_utils import DirectLogger, aggregate_and_save_results
from evaluation.evaluation_runners import run_cpp_direct, run_ccpo_rolling_counts, run_ccpo_rolling_dates

def run_rolling_evaluation(
    data_path: str = None,
    frequency: str = None,
    lookback: int = None,
    alpha: float = None,
    cfg: config = config
):

    frequency = frequency or cfg.FREQUENCY
    lookback = lookback or cfg.LOOKBACK
    alpha = alpha or cfg.ALPHA
    
    prediction_mode = getattr(cfg, "PREDICTION_MODE", "single")

    if prediction_mode == "single":
        if cfg.MODE == "counts":
            k_info = f"TrK{cfg.ROLLING.SINGLE.COUNTS.TRAIN_K_LEN}" # Train+K merged len
            v_info = f"V{cfg.ROLLING.SINGLE.COUNTS.V_LEN}"
        else: # dates
            k_info = f"TrK{cfg.ROLLING.SINGLE.DATES.K_PERIOD_OFFSET}"
            v_info = f"V{cfg.ROLLING.SINGLE.DATES.V_PERIOD_OFFSET}"
    else:
        if cfg.MODE == "counts":
            k_info = f"K{cfg.ROLLING.MULTI.COUNTS.TRAIN_K_LEN}"
            v_info = f"V{cfg.ROLLING.MULTI.COUNTS.V_LEN}"
        else: # dates
            k_info = f"K{cfg.ROLLING.MULTI.DATES.K_PERIOD_OFFSET}"
            v_info = f"V{cfg.ROLLING.MULTI.DATES.V_PERIOD_OFFSET}"
    
    
    timestamp = datetime.now().strftime("%m%d%H%M")
    result_folder = os.path.join(
        os.path.dirname(__file__),
        "..", "results",
        f"run_rolling_{cfg.ROLLING.WINDOW_TYPE}_{cfg.MODE}_{prediction_mode}_{alpha}_{cfg.NUM_ASSETS}_assets_{cfg.SEED}_{k_info}K_{v_info}V"
    )
    os.makedirs(result_folder, exist_ok=True)

    log_file = os.path.join(result_folder, "rolling_log.txt")
    logger = DirectLogger(log_file)
    logger.log_header(title=f"Rolling Evaluation ({cfg.ROLLING.WINDOW_TYPE} / {cfg.MODE} / {prediction_mode})")

    original_stdout = sys.stdout
    sys.stdout = logger

    try:
        print(f"🎯 Rolling Window Evaluation ({cfg.ROLLING.WINDOW_TYPE} / {cfg.MODE})")
        print(f"   Prediction Mode: {prediction_mode}")
        print(f"   Frequency: {frequency}, Lookback: {lookback}, Alpha: {alpha}")

        # Load Full Data
        loader = TimeSeriesDataLoader(base_path=cfg.DATA_PATH, num_assets=cfg.NUM_ASSETS)
        loader.load_data()
        full_data_resampled = loader.resample_frequency(loader.raw_data, frequency)
        total_len, n_assets = full_data_resampled.shape
        asset_names = full_data_resampled.columns.tolist()
        print(f"\nFull data loaded: {total_len} periods, {n_assets} assets")
        print(f"  [{full_data_resampled.index.min().date()} ~ {full_data_resampled.index.max().date()}]")

        # Setup Methods
        cpp_methods = cfg.CPP.METHODS
        ccpo_methods = ["CCPO-CCO"]
        baseline_methods = ["Equal-Weight"]
        all_methods = cpp_methods + ccpo_methods + baseline_methods
        portfolios = {m: Portfolio(name=m) for m in all_methods}

        window_definitions = []

        # ==============================================================================
        # WINDOW GENERATION LOGIC
        # ==============================================================================
        
        # ---------------------
        # CASE A: Single Step Prediction
        # ---------------------
        if prediction_mode == "single":
            cfg_roll = cfg.ROLLING.SINGLE  # Load Single Rolling Config
            print(f"\n[Single Step Mode] Generating windows...")
            
            if cfg.MODE == "counts":
                k_len = cfg_roll.COUNTS.TRAIN_K_LEN
                v_len = cfg_roll.COUNTS.V_LEN
                step_size = cfg_roll.COUNTS.STEP_SIZE
                
                print(f"  Train(K) Len: {k_len}, Test(V) Len: {v_len}, Step: {step_size}")
                
                # Raw index logic
                current_idx = 0
                while True:
                    k_end_idx = current_idx + k_len + lookback 
                    v_end_idx = k_end_idx + v_len
                    
                    if v_end_idx > total_len:
                        break
                        
                    window_definitions.append({
                        "mode": "counts",
                        "prediction": "single",
                        "k_start_raw_idx": current_idx, 
                        "k_end_raw_idx": k_end_idx,     
                        "v_start_raw_idx": k_end_idx,
                        "v_end_raw_idx": v_end_idx,
                        "train_len": k_len, 
                        "K_len": 0,         
                        "V_len": v_len
                    })
                    current_idx += step_size
                    
            elif cfg.MODE == "dates":
                 k_period = cfg_roll.DATES.K_PERIOD_OFFSET 
                 v_period = cfg_roll.DATES.V_PERIOD_OFFSET
                 step_period = cfg_roll.DATES.STEP_OFFSET
                 
                 print(f"  Train(K) Period: {k_period}, Test(V) Period: {v_period}, Step: {step_period}")
                 
                 k_offset = pd.tseries.frequencies.to_offset(k_period)
                 v_offset = pd.tseries.frequencies.to_offset(v_period)
                 step_offset = pd.tseries.frequencies.to_offset(step_period)
                 
                 start_date = pd.to_datetime(cfg_roll.DATES.ROLLING_START_DATE)
                 max_date = full_data_resampled.index[-1]
                 
                 current_start = start_date
                 while True:
                     k_end_date = current_start + k_offset
                     v_end_date = k_end_date + v_offset
                     
                     if v_end_date > max_date:
                         break
                         
                     window_definitions.append({
                        "mode": "dates",
                        "prediction": "single",
                        "k_start_date": current_start,
                        "k_end_date": k_end_date,
                        "v_start_date": k_end_date,
                        "v_end_date": v_end_date,
                        "train_start_date": current_start,
                        "train_end_date": k_end_date 
                     })
                     current_start += step_offset

        # ---------------------
        # CASE B: Multi Step Prediction
        # ---------------------
        else:
            cfg_roll = cfg.ROLLING.MULTI # Load Multi Rolling Config
            print(f"\n[Multi-step Mode] Generating windows...")

            if cfg.MODE == "counts":
                initial_train_len = cfg_roll.COUNTS.TRAIN_K_LEN
                k_raw_len = cfg_roll.COUNTS.TRAIN_K_LEN
                v_raw_len = cfg_roll.COUNTS.V_LEN
                step_size = cfg_roll.COUNTS.STEP_SIZE
                
                print(f"  Init Train: {initial_train_len}, K: {k_raw_len}, V: {v_raw_len}, Step: {step_size}")

                initial_train_raw_len = lookback + initial_train_len
                fixed_train_start_idx = 0
                current_k_start_idx = fixed_train_start_idx + initial_train_raw_len

                while True:
                    k_end_idx = current_k_start_idx + k_raw_len
                    v_end_idx = k_end_idx + v_raw_len
                    
                    if v_end_idx > total_len:
                        break
                    
                    if cfg.ROLLING.WINDOW_TYPE == "expanding":
                        train_start_idx = fixed_train_start_idx
                        train_end_idx = current_k_start_idx
                        current_train_len = train_end_idx - train_start_idx - lookback
                    else: # sliding
                        current_train_len = initial_train_len           
                        train_end_idx = current_k_start_idx
                        train_start_idx = train_end_idx - (lookback + current_train_len)

                    window_definitions.append({
                        "mode": "counts",
                        "prediction": "multi",
                        "train_start_idx": train_start_idx,
                        "train_len": current_train_len,
                        "K_len": k_raw_len,
                        "V_len": v_raw_len,
                        "k_start_raw_idx": current_k_start_idx,
                        "k_end_raw_idx": k_end_idx,
                        "v_start_raw_idx": k_end_idx,
                        "v_end_raw_idx": v_end_idx,
                    })
                    current_k_start_idx += step_size

            elif cfg.MODE == "dates":
                train_offset = pd.tseries.frequencies.to_offset(cfg_roll.DATES.K_PERIOD_OFFSET)
                k_offset = pd.tseries.frequencies.to_offset(cfg_roll.DATES.K_PERIOD_OFFSET)
                v_offset = pd.tseries.frequencies.to_offset(cfg_roll.DATES.V_PERIOD_OFFSET)
                step_offset = pd.tseries.frequencies.to_offset(cfg_roll.DATES.STEP_OFFSET)
                
                print(f"  Train Off: {train_offset}, K Off: {k_offset}, V Off: {v_offset}, Step: {step_offset}")

                first_possible_train_start = pd.to_datetime(cfg_roll.DATES.ROLLING_START_DATE)
                first_k_end = first_possible_train_start + train_offset + k_offset
                first_v_start = first_k_end

                current_v_start_date = first_v_start
                while True:
                    v_end_date = current_v_start_date + v_offset
                    if v_end_date > full_data_resampled.index[-1] + pd.Timedelta(days=1):
                        break

                    k_end_date = current_v_start_date
                    train_end_date = k_end_date - k_offset

                    if cfg.ROLLING.WINDOW_TYPE == "expanding":
                        train_start_date = first_possible_train_start
                    else:
                        train_start_date = train_end_date - train_offset

                    window_definitions.append({
                        "mode": "dates",
                        "prediction": "multi",
                        "train_start_date": train_start_date,
                        "train_end_date": train_end_date,
                        "k_start_date": train_end_date,
                        "k_end_date": k_end_date,
                        "v_start_date": k_end_date,
                        "v_end_date": v_end_date
                    })
                    current_v_start_date += step_offset

        print(f"\nTotal valid windows defined: {len(window_definitions)}")

        # ==============================================================================
        # MAIN LOOP
        # ==============================================================================
        for i, window in enumerate(window_definitions):
            window_num = i + 1
            print(f"\n{'='*80}")
            print(f"RUNNING WINDOW {window_num}/{len(window_definitions)} ({window['prediction']} / {cfg.MODE})")

            # 1) Slice Data
            if window["mode"] == "counts":
                k_s, k_e = window["k_start_raw_idx"], window["k_end_raw_idx"]
                v_s, v_e = window["v_start_raw_idx"], window["v_end_raw_idx"]
                
                K_data_raw = full_data_resampled.iloc[k_s : k_e]
                V_data_raw = full_data_resampled.iloc[v_s : v_e]
                
                print(f"  Idx: K=[{k_s}:{k_e}], V=[{v_s}:{v_e}]")
                
            else: # dates
                k_start, k_end = window["k_start_date"], window["k_end_date"]
                v_start, v_end = window["v_start_date"], window["v_end_date"]
                
                K_data_raw = full_data_resampled.loc[k_start : k_end - pd.Timedelta(nanoseconds=1)]
                V_data_raw = full_data_resampled.loc[v_start : v_end - pd.Timedelta(nanoseconds=1)]
                
                print(f"  Dates: K=[{k_start.date()} ~ {k_end.date()}], V=[{v_start.date()} ~ {v_end.date()}]")

            if K_data_raw.empty or V_data_raw.empty:
                print("⚠️ Skipping empty window.")
                continue

            K_returns_raw = K_data_raw.values
            V_returns_raw = V_data_raw.values
            V_dates = V_data_raw.index

            # 2) Run CPP
            window_results = {}
            for cpp_method in cpp_methods:
                cpp_res = run_cpp_direct(
                    K_returns=K_returns_raw,
                    L_returns=None,
                    V_returns=V_returns_raw,
                    method=cpp_method,
                    alpha=alpha,
                )
                window_results[cpp_method] = cpp_res

            # 3) Run CCPO
            if window["mode"] == "counts":
                ccpo_res = run_ccpo_rolling_counts(
                    data_path=cfg.DATA_PATH, lookback=lookback, alpha=alpha,
                    model_train_len=window["train_len"],
                    K_len=window.get("K_len", 0), 
                    V_len=window["V_len"],
                    start_idx=window.get("train_start_idx", window["k_start_raw_idx"]), 
                    V_dates=V_dates,
                    V_returns_raw=V_returns_raw,
                    cfg=cfg
                )
            else:
                ccpo_res = run_ccpo_rolling_dates(
                    data_path=cfg.DATA_PATH, lookback=lookback, alpha=alpha,
                    train_start_date=window["train_start_date"],
                    train_end_date=window["train_end_date"],
                    K_end_date=window["k_end_date"],
                    V_dates=V_dates,
                    V_returns_raw=V_returns_raw,
                    cfg=cfg
                )
            window_results["CCPO-CCO"] = ccpo_res

            # 4) Equal-Weight
            if n_assets > 0:
                equal_w = np.ones(n_assets) / n_assets
                window_results["Equal-Weight"] = {
                    'weights': equal_w, 'threshold_post': None, 'status': 'optimal', 'solve_time': 0.0
                }

            # 5) Aggregate to Portfolio
            for method in all_methods:
                result = window_results.get(method)
                if not result or result.get('status') != 'optimal':
                    continue
                    
                if method == 'CCPO-CCO':
                    portfolio_item_list = result.get('portfolios', [])
                    solve_time_avg = (result.get('optimization_time', 0.0)) / len(portfolio_item_list) if portfolio_item_list else 0.0
                    
                    for i_p, item in enumerate(portfolio_item_list):
                         if i_p < len(V_dates):
                             d = V_dates[i_p]
                             w = item['weights']
                             if w is not None:
                                 portfolios[method].add_period(
                                     date=d, weight=w,
                                     realized_return=float(w @ V_returns_raw[i_p]),
                                     solve_time=solve_time_avg,
                                     threshold_post=item.get('threshold')
                                 )
                else:
                    w = result.get('weights')
                    if w is not None:
                         for d, ret in zip(V_dates, V_returns_raw):
                             portfolios[method].add_period(
                                 date=d, weight=w,
                                 realized_return=float(w @ ret),
                                 solve_time=result.get('solve_time', 0.0)/len(V_dates),
                                 threshold_post=result.get('threshold_post')
                             )

        # Final Save
        aggregate_and_save_results(
            portfolios=portfolios, result_folder=result_folder,
            asset_names=asset_names, prefix="rolling_agg",
            cfg=cfg, results=None
        )
        print(f"📝 Log saved to: {log_file}")
        return { "portfolios": portfolios, "result_folder": result_folder }

    finally:
        sys.stdout = original_stdout