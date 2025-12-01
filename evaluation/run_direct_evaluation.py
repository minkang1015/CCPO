import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from configs import config_revised as config
from data.data_factory import get_dataset
from utils.portfolios import Portfolio
from utils.evaluation_utils import DirectLogger, aggregate_and_save_results
from evaluation.evaluation_runners import run_cpp_direct, run_ccpo_direct

def run_direct_evaluation(
    data_path: str = None, 
    frequency: str = None,
    lookback: int = None,
    alpha: float = None,
    cfg: config = config
):
    """
    Direct evaluation using data_factory to support both 3-Split and 2-Split modes.
    """
    # 0) Config Load
    frequency = frequency or cfg.FREQUENCY
    lookback = lookback or cfg.LOOKBACK
    alpha = alpha or cfg.ALPHA
    
    prediction_mode = getattr(cfg, "PREDICTION_MODE", "single")

    # Logger Setup
    timestamp = datetime.now().strftime("%m%d%H%M")
    result_folder = os.path.join(
        os.path.dirname(__file__),
        "..", "results",
        f"run_direct_{cfg.MODE}_{prediction_mode}_{timestamp}"
    )
    os.makedirs(result_folder, exist_ok=True)

    log_file = os.path.join(result_folder, "direct_log.txt")
    logger = DirectLogger(log_file)
    logger.log_header(title=f"Direct Evaluation (MODE={cfg.MODE}, PREDICTION={prediction_mode})")

    original_stdout = sys.stdout
    sys.stdout = logger

    try:
        print(f"🎯 Direct Evaluation (Single Split, MODE={cfg.MODE}, PREDICTION={prediction_mode})")
        print("\nConfiguration:")
        print(f"  Frequency: {frequency}")
        print(f"  Lookback={lookback}, Alpha={alpha}\n")

        # -----------------------------------------------------------
        # 1) Fetch Data using Factory
        # -----------------------------------------------------------
        print(">> Fetching Dataset via Factory...")
        # get_dataset handles all logic for 2-split/3-split and kwargs
        dataset_res = get_dataset(cfg)
        
        # Extract Optimization Data (Raw Returns)
        K_returns_raw = dataset_res['opt']['y_K']
        V_returns_raw = dataset_res['opt']['y_V']
        K_dates = pd.DatetimeIndex(dataset_res['opt']['dates_K'])
        V_dates = pd.DatetimeIndex(dataset_res['opt']['dates_V'])
        
        # Check Assets (Get from columns if available, or load temp)
        # Assuming asset names are consistent, we load a small temp loader just for names if needed
        # Or better, we assume a standard set or derived from loaded data if exposed.
        # Ideally, we should get asset names from the factory result or loader, but loader is hidden.
        # We will quickly load raw head to get names.
        
        temp_df = pd.read_csv(os.path.join(cfg.DATA_PATH, f"industry_{cfg.NUM_ASSETS}_daily.csv"), index_col=0, nrows=2)
        asset_names = temp_df.columns.tolist()
        n_assets = len(asset_names)

        print("\nData Split Info:")
        print(f"  Assets ({n_assets}): {asset_names}")
        print(f"  K Period (Calib/Train): {len(K_returns_raw)} obs [{K_dates.min().date() if len(K_dates)>0 else 'N/A'} ~ {K_dates.max().date() if len(K_dates)>0 else 'N/A'}]")
        print(f"  V Period (Test):        {len(V_returns_raw)} obs [{V_dates.min().date() if len(V_dates)>0 else 'N/A'} ~ {V_dates.max().date() if len(V_dates)>0 else 'N/A'}]\n")
        
        if len(K_returns_raw) == 0 or len(V_returns_raw) == 0:
            print("❌ Error: K or V data is empty. Check your dates/counts config.")
            return {}

        # -----------------------------------------------------------
        # 2) Setup Portfolios
        # -----------------------------------------------------------
        cpp_methods = cfg.CPP.METHODS
        ccpo_methods = ["CCPO-CCO"] 
        baseline_methods = ["Equal-Weight"]
        all_methods = cpp_methods + ccpo_methods + baseline_methods

        portfolios = {m: Portfolio(name=m) for m in all_methods}
        results = {}

        print("=" * 80)
        print("RUNNING EXPERIMENTS")
        print("=" * 80 + "\n")

        # -----------------------------------------------------------
        # 3) Run CPP (Conformal Prediction Programming)
        # -----------------------------------------------------------
        for cpp_method in cpp_methods:
            cpp_res = run_cpp_direct(
                K_returns=K_returns_raw,
                L_returns=None,
                V_returns=V_returns_raw,
                method=cpp_method,
                alpha=alpha,
            )
            results[cpp_method] = cpp_res

            if cpp_res.get("status") == "optimal":
                weights = cpp_res["weights"]
                threshold = cpp_res["threshold_post"]
                for date, asset_ret in zip(V_dates, V_returns_raw):
                    portfolios[cpp_method].add_period(
                        date=date, weight=weights,
                        realized_return=float(weights @ asset_ret),
                        solve_time=cpp_res.get("solve_time", 0.0) / len(V_dates) if len(V_dates)>0 else 0.0,
                        threshold_post=threshold,
                    )

        # -----------------------------------------------------------
        # 4) Run CCPO
        # -----------------------------------------------------------
        # Note: run_ccpo_direct might internally use standard loader logic. 
        # Ideally, we should refactor run_ccpo_direct to accept data directly, 
        # but for now we pass cfg and let it handle or use the factory if updated.
        # If run_ccpo_direct is not updated to use factory, it might reload data.
        # Assuming run_ccpo_direct is compatible or we rely on cfg settings.
        
        ccpo_res = run_ccpo_direct(
            data_path=cfg.DATA_PATH,
            lookback=lookback,
            alpha=alpha,
            cfg=cfg
        )
        results["CCPO-CCO"] = ccpo_res

        if ccpo_res.get("status") == "optimal" and "portfolios" in ccpo_res:
            for pinfo in ccpo_res["portfolios"]:
                date = pinfo["date"]
                weights = pinfo["weights"]
                threshold = pinfo["threshold"]
                try:
                    # Find matching date in V_dates
                    idx_loc = V_dates.get_loc(date)
                    # get_loc might return slice or int
                    if isinstance(idx_loc, slice):
                         asset_ret_raw = V_returns_raw[idx_loc][0] # Take first if duplicate (rare)
                    else:
                         asset_ret_raw = V_returns_raw[idx_loc]
                         
                    realized_return = float(weights @ asset_ret_raw)
                    portfolios["CCPO-CCO"].add_period(
                        date=date, weight=weights,
                        realized_return=realized_return,
                        solve_time=ccpo_res.get("optimization_time", 0.0) / len(V_dates) if len(V_dates)>0 else 0.0,
                        threshold_post=threshold,
                    )
                except KeyError:
                    pass # Date mismatch (e.g. rolling vs direct alignment)
                except Exception as e:
                    print(f"  ⚠️ CCPO Log Error {date.date()}: {e}")

            print(f"    ✅ CCPO completed. Processed {len(portfolios['CCPO-CCO'])} V periods.")

        # -----------------------------------------------------------
        # 5) Run Baseline (Equal Weight)
        # -----------------------------------------------------------
        print("  Running Equal-Weight...")
        if n_assets > 0:
            equal_w = np.ones(n_assets) / n_assets
            for date, asset_ret in zip(V_dates, V_returns_raw):
                portfolios["Equal-Weight"].add_period(
                    date=date, weight=equal_w,
                    realized_return=float(equal_w @ asset_ret),
                    solve_time=0.0, threshold_post=None,
                )
            print(f"    ✅ Completed {len(V_dates)} periods.")

        # Save
        aggregate_and_save_results(
            portfolios=portfolios, result_folder=result_folder,
            asset_names=asset_names, prefix="direct",
            cfg=cfg, results=results
        )
        print(f"📝 Log saved to: {log_file}")

        return { "portfolios": portfolios, "result_folder": result_folder }

    finally:
        sys.stdout = original_stdout