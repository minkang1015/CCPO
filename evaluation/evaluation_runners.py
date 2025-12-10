import numpy as np
import pandas as pd
from datetime import datetime
import time # For timing
from typing import Dict, Optional, Any
import torch
import traceback
from configs import config_revised as config
from data.data_loader_final import TimeSeriesDataLoader, SimpleTimeSeriesDataLoader 
from data.data_loader_multistep import DataLoaderMultiStep
from data.data_factory import get_dataset                      
import cpp.solver as cpp_solver
from layers.multi_cp_new import SPCI_and_EnbPI 
from evaluation.run_ccpo import CCPOPortfolioOptimizer
from utils.evaluation_utils import _build_create_all_kwargs

# ============================================================================
# CPP RUNNER
# ============================================================================

def run_cpp_direct(
    K_returns: np.ndarray,
    L_returns: Optional[np.ndarray], # Note: L is not used in the new rolling logic for CCPO Calib.
    V_returns: np.ndarray,
    method: str,
    alpha: float
) -> Dict:
    """
    Run a single CPP method. K_returns are used for optimization.
    L_returns are for optional calibration (ignored in current rolling).
    V_returns are for out-of-sample coverage check (informational).
    """
    K, n_assets = K_returns.shape
    L = 0 if L_returns is None else L_returns.shape[0]
    V = V_returns.shape[0]

    print(f"  Running {method}...")
    print(f"    Optimization Data (K): {K} periods, {n_assets} assets")
    print(f"    Test Data (V): {V} periods") # V is only for coverage check here

    training_Ys = [K_returns[i, :] for i in range(K)]
    x_dim = n_assets + 1

    def f(x, Y):
        s = x[n_assets]
        portfolio_return = sum(x[i] * Y[i] for i in range(n_assets))
        return s - portfolio_return

    def J(x):
        return -x[n_assets]

    hs = [lambda x, i=i: -x[i] for i in range(n_assets)]
    gs = [lambda x: sum(x[i] for i in range(n_assets)) - 1]

    start_time = time.time()
    try:
        solution, _ = cpp_solver.solve( # Ignoring internal solver time reporting for now
            x_dim=x_dim, delta=alpha, training_Ys=training_Ys,
            hs=hs, gs=gs, f=f, J=J, method=method,
            omega=config.CPP.OMEGA if method == 'SAA' else None,
            time_limit=config.CPP.TIME_LIMIT
        )
        solve_time = time.time() - start_time

        if isinstance(solution, str):
            print(f"    ❌ Failed: {solution}")
            return {'status': solution, 'weights': None, 'solve_time': solve_time}

        weights = np.array(solution[:n_assets])
        threshold_opt = solution[n_assets] # Threshold from optimization

        # --- Calibration Step Removed for CPP in Rolling ---
        threshold_post = threshold_opt

        if V > 0:
            portfolio_returns_V = V_returns @ weights
            coverage_post = float(np.mean(portfolio_returns_V >= threshold_post))
            print(f"    ✅ Opt Success. Threshold: {threshold_post:.6f}, OOS Coverage (on V): {coverage_post:.3f}, Time: {solve_time:.2f}s")
        else:
            coverage_post = np.nan
            print(f"    ✅ Opt Success. Threshold: {threshold_post:.6f}, No V data for OOS Coverage. Time: {solve_time:.2f}s")


        return {
            'weights': weights,
            'threshold_post': threshold_post, # Threshold used for V period evaluation
            'coverage_post': coverage_post, # Informational coverage on V
            'solve_time': solve_time,
            'status': 'optimal'
        }

    except Exception as e:
        solve_time = time.time() - start_time
        print(f"    ❌ CPP solver error: {e}")
        print(traceback.format_exc()) # Print full traceback for debugging
        return {'status': f'error: {str(e)}', 'weights': None, 'solve_time': solve_time}


# ============================================================================
# CCPO RUNNER
# ============================================================================

def run_ccpo_direct(
    data_path: str,
    lookback: int,
    alpha: float,
    # V_dates, V_returns removed, will be obtained from loader
    cfg: config = config
) -> Dict[str, Any]:
    """
    Run CCPO method for direct evaluation (single split).
    Uses data_factory to support both single step forecasting and multi step forecasting automatically.
    """
    prediction_mode = getattr(cfg, "PREDICRTION_MODE", "single")
    print(f"  Running CCPO-CCO (MODE={prediction_mode})...")
    print(f"    Lookback={lookback}, Alpha={alpha}")

    start_time_total = time.time()
    
    # [MODIFIED] Use Factory instead of manual loader creation
    # This handles the complex switching logic between Simple/Final loaders and args
    try:
        res = get_dataset(cfg)
        
        # Extract components
        train_loader = res['model']['train_loader']
        valid_loader = res['model']['valid_loader'] 
        test_loader  = res['model']['test_loader']
        scaler = res['scaler']
        
        # Loader for referencing methods (though we have the data already)
        # We need a loader instance for the conformal predictor internal calls (e.g. resample utils)
        # We can create a dummy one or use the one from factory if exposed.
        # Since factory returns dict, let's create a temp loader for Utils.
        temp_loader = TimeSeriesDataLoader(base_path=data_path, num_assets=cfg.NUM_ASSETS)

        # These are RAW returns, not scaled
        V_returns_raw = res['opt']['y_V']
        V_dates = pd.DatetimeIndex(res['opt']['dates_V'])
        n_assets = V_returns_raw.shape[1] if V_returns_raw.ndim > 1 else (1 if V_returns_raw.size > 0 else 0)


        # Check if loaders are empty
        if len(train_loader.dataset) == 0:
             print("    ⚠️ Warning: Train loader is empty. Check config TRAIN settings.")
        
        # For 2-split, valid_loader is None. We use train_loader as Calibration set.
        if prediction_mode == "single":
            print("    [Single Step] Using Single Period Forecasting Loaders.")
            loader_k = train_loader
        else:
            if valid_loader is None or len(valid_loader.dataset) == 0:
                print("    ⚠️ Warning: Validation (K) loader is empty or None in 3-split mode.")
            loader_k = valid_loader


        # 2. Initialize optimizer
        optimizer = CCPOPortfolioOptimizer(
            alpha=alpha, model_cls=cfg.CCPO.MODEL_CLASS,
            device=cfg.DEVICE, r=cfg.CCPO.LOW_RANK_R,
            use_local_ellipsoid=cfg.CCPO.USE_LOCAL_ELLIPSOID,
            bins=cfg.CCPO.QRF_BINS, n_estimators=cfg.CCPO.QRF_N_ESTIMATORS,
            max_d=cfg.CCPO.QRF_MAX_DEPTH, criterion=cfg.CCPO.CRITERION
        )

        # 3. Train models and Calibrate
        print(f"    Training {cfg.CCPO.B} bootstrap models...")
        start_time_calib = time.time()

        # Data needs to be Tensors
        X_train, Y_train = train_loader.dataset.X, train_loader.dataset.y
        X_predict, Y_predict = test_loader.dataset.X, test_loader.dataset.y 


        conformal_predictor = SPCI_and_EnbPI(
            X_train, X_predict,
            Y_train, Y_predict,
            model_cls=cfg.CCPO.MODEL_CLASS, loader=temp_loader, scaler=scaler,
            device=cfg.DEVICE, r=cfg.CCPO.LOW_RANK_R,
            use_local_ellipsoid=cfg.CCPO.USE_LOCAL_ELLIPSOID,
            bins=cfg.CCPO.QRF_BINS,
            max_d=cfg.CCPO.QRF_MAX_DEPTH,
            n_estimators=cfg.CCPO.QRF_N_ESTIMATORS,
            criterion=cfg.CCPO.CRITERION
        )

        results = conformal_predictor.fit_bootstrap_models_online_multistep(
            B=cfg.CCPO.B, batch_size=cfg.CCPO.BATCH_SIZE, EPOCHS=cfg.CCPO.EPOCHS,
            lr=cfg.CCPO.LEARNING_RATE, path=cfg.CCPO.WEIGHTS_PATH
        )

        print(f"    Calibrating conformal prediction intervals...")
        conformal_predictor.compute_Widths_Ensemble_online(
            alpha=alpha, smallT=False, use_SPCI=cfg.CCPO.USE_SPCI,
            past_window=cfg.CCPO.PAST_WINDOW, random_state=cfg.SEED
        )
        calibration_time = time.time() - start_time_calib

        mean_coverage_calib, mean_volume_calib, coverage_seq, volume_seq, radius_seq = conformal_predictor.get_results()
        
        if not radius_seq: 
             raise ValueError("Calibration failed: Radius sequence is empty.")
        radius = float(np.mean(radius_seq))
        cov_matrix = conformal_predictor.global_cov 

        print(f"    ✅ Calibration done - Calib Set Coverage: {mean_coverage_calib:.3f}, Radius: {radius:.6f}, Time: {calibration_time:.2f}s")

        # 4. Optimize portfolio for each period in V
        print(f"    Optimizing portfolio for each of {len(V_dates)} test periods (V)...")
        start_time_opt = time.time()
        portfolios_list = []
        
        if prediction_mode == "single":
            mu_pred_raw = results["test"]["y_pred"].squeeze(1).cpu().numpy()        # [n_test, n_assets]
        elif prediction_mode == "multi":
            mu_pred_raw = (1 + conformal_predictor.test_pred_raw).prod(dim=1).sub(1).detach().cpu().numpy() # [n_test, n_assets]
        else:
            raise ValueError(f"Unknown prediction mode: {prediction_mode}")
        
        for v_idx, v_date in enumerate(V_dates):
            # mu_pred_raw matches V_dates length
            current_mu = mu_pred_raw[v_idx] if v_idx < len(mu_pred_raw) else np.zeros(n_assets)
            current_radius = radius_seq[v_idx] if v_idx < len(radius_seq) else radius

            opt_result = optimizer.optimize_portfolio_socp(
                mu_hat=current_mu, cov_matrix=cov_matrix, radius=current_radius,
                gamma=cfg.CCPO.GAMMA, formulation=cfg.CCPO.FORMULATION
            )

            if opt_result['status'] == 'optimal':
                portfolios_list.append({
                    'date': v_date,
                    'weights': opt_result['weights'],
                    'threshold': opt_result['threshold'], 
                })
            else:
                print(f"      ⚠️  Optimization failed for {v_date.date()}: {opt_result['status']}")
                portfolios_list.append({
                    'date': v_date,
                    'weights': np.ones(n_assets) / n_assets if n_assets > 0 else np.array([]),
                    'threshold': None,
                })

        optimization_time = time.time() - start_time_opt
        total_time = time.time() - start_time_total
        print(f"    ✅ Completed {len(portfolios_list)}/{len(V_dates)} V periods. Opt Time: {optimization_time:.2f}s, Total Time: {total_time:.2f}s")

        return {
            'portfolios': portfolios_list, 
            'status': 'optimal',
            'coverage': mean_coverage_calib, 
            'volume': mean_volume_calib,     
            'threshold': radius,             
            'calibration_time': calibration_time, 
            'optimization_time': optimization_time
        }, results

    except Exception as e:
        total_time = time.time() - start_time_total
        print(f"    ❌ CCPO (direct) error: {e}")
        print(traceback.format_exc())
        return {'status': f'error: {str(e)}', 'portfolios': [], 'total_time': total_time}


# ============================================================================
# CCPO RUNNER (Rolling - COUNTS)
# ============================================================================

def run_ccpo_rolling_counts(
    data_path: str,
    lookback: int,
    alpha: float,
    model_train_len: int,   
    K_len: int,             
    V_len: int,             
    start_idx: int,         
    V_dates: pd.DatetimeIndex,
    V_returns_raw: np.ndarray, 
    cfg: config = config
) -> Dict[str, Any]:
    """
    Run CCPO method for one window in rolling evaluation (MODE=counts).
    Handles both 3-Split and 2-Split logic.
    """
    prediction_mode = getattr(cfg, "PREDICTION_MODE", "single")
    print(f"  Running CCPO-CCO (Rolling Counts, SPLIT={prediction_mode})...")
    print(f"    TrainLen={model_train_len}, KLen={K_len}, VLen={V_len}, StartIdx={start_idx}")

    start_time_total = time.time()
    n_assets = V_returns_raw.shape[1] if V_returns_raw.ndim > 1 else (1 if V_returns_raw.size > 0 else 0)

    try:
        if prediction_mode == "single":
            loader = SimpleTimeSeriesDataLoader(base_path=data_path, num_assets=cfg.NUM_ASSETS)
            res = loader.create_all(
                mode="counts",
                lookback=lookback,
                K=model_train_len, # Map passed TrainLen to K (as they are same)
                V=V_len,
                start_idx=start_idx,
                batch_size=cfg.CCPO.BATCH_SIZE,
                shuffle_train=True,
                use_scaler=True,
                resample_freq=cfg.FREQUENCY
            )
            train_loader = res['model']['train_loader']
            
        else:
            loader = DataLoaderMultiStep(base_path=data_path, num_assets=cfg.NUM_ASSETS)
            res = loader.create_all(
                mode="counts",
                lookback=lookback,
                K=model_train_len,   # 전체 Train(K) 시퀀스 개수
                V=V_len,
                horizon=cfg.CCPO.HORIZON,
                start_idx=start_idx,
                batch_size=cfg.CCPO.BATCH_SIZE,
                shuffle_train=True,
                use_scaler=True,
                resample_freq=cfg.FREQUENCY
            )
            train_loader = res['model']['train_loader']
            loader_k = train_loader

        test_loader = res['model']['test_loader']
        scaler = res['scaler']

        if len(train_loader.dataset) == 0:
             raise ValueError("Train data loader is empty.")

        # 2. Initialize optimizer
        optimizer = CCPOPortfolioOptimizer(
            alpha=alpha, model_cls=cfg.CCPO.MODEL_CLASS,
            device=cfg.DEVICE, r=cfg.CCPO.LOW_RANK_R,
            use_local_ellipsoid=cfg.CCPO.USE_LOCAL_ELLIPSOID
        )

        # 3. Train & Calibrate
        print(f"    Training {cfg.CCPO.B} bootstrap models...")
        start_time_calib = time.time()

        X_train, Y_train = train_loader.dataset.X, train_loader.dataset.y
        X_predict, Y_predict = test_loader.dataset.X, test_loader.dataset.y

        conformal_predictor = SPCI_and_EnbPI(
            X_train, X_predict,
            Y_train, Y_predict,
            model_cls=cfg.CCPO.MODEL_CLASS, loader=loader, scaler=scaler,
            device=cfg.DEVICE, r=cfg.CCPO.LOW_RANK_R,
            use_local_ellipsoid=cfg.CCPO.USE_LOCAL_ELLIPSOID,
            bins=cfg.CCPO.QRF_BINS,
            max_d=cfg.CCPO.QRF_MAX_DEPTH,
            n_estimators=cfg.CCPO.QRF_N_ESTIMATORS,
            criterion=cfg.CCPO.CRITERION
        )

        results = conformal_predictor.fit_bootstrap_models_online_multistep(
            B=cfg.CCPO.B, batch_size=cfg.CCPO.BATCH_SIZE, EPOCHS=cfg.CCPO.EPOCHS,
            lr=cfg.CCPO.LEARNING_RATE, path=cfg.CCPO.WEIGHTS_PATH, loss_aggregation=cfg.CCPO.LOSS_AGG, cp_residual_mode='aggregated'
        )

        print(f"    Calibrating conformal prediction intervals...")
        conformal_predictor.compute_Widths_Ensemble_online(
            alpha=alpha, smallT=False, use_SPCI=cfg.CCPO.USE_SPCI,
            past_window=cfg.CCPO.PAST_WINDOW, random_state=cfg.SEED
        )
        calibration_time = time.time() - start_time_calib

        mean_coverage_calib, _, coverage_seq, volume_seq, radius_seq = conformal_predictor.get_results()
        if not radius_seq: raise ValueError("Calibration failed: Radius sequence is empty.")
        
        radius = float(np.mean(radius_seq))
        cov_matrix = conformal_predictor.global_cov

        print(f"    ✅ Calibration done - Calib Set Coverage: {mean_coverage_calib:.3f}, Radius: {radius:.6f}, Time: {calibration_time:.2f}s")

        # 4. Optimize
        print(f"    Optimizing portfolio for each of {len(V_dates)} test periods (V)...")
        start_time_opt = time.time()
        portfolios_list = []

        if prediction_mode == "single":
            mu_pred_raw = results["test"]["y_pred"].squeeze(1).cpu().numpy()        # [n_test, n_assets]
        elif prediction_mode == "multi":
            mu_pred_raw = (1 + conformal_predictor.test_pred_raw).prod(dim=1).sub(1).detach().cpu().numpy() # [n_test, n_assets]
        else:
            raise ValueError(f"Unknown prediction mode: {prediction_mode}")
        
        for v_idx, v_date in enumerate(V_dates):
            current_mu = mu_pred_raw[v_idx] if v_idx < len(mu_pred_raw) else np.zeros(n_assets)
            current_radius = radius_seq[v_idx] if v_idx < len(radius_seq) else radius

            opt_result = optimizer.optimize_portfolio_socp(
                mu_hat=current_mu, cov_matrix=cov_matrix, radius=current_radius,
                gamma=cfg.CCPO.GAMMA, formulation=cfg.CCPO.FORMULATION
            )
            
            if opt_result['status'] == 'optimal':
                portfolios_list.append({
                    'date': v_date, 'weights': opt_result['weights'],
                    'threshold': opt_result['threshold']
                })
            else:
                portfolios_list.append({
                    'date': v_date,
                    'weights': np.ones(n_assets) / n_assets if n_assets > 0 else np.array([]),
                    'threshold': None
                })

        optimization_time = time.time() - start_time_opt
        total_time = time.time() - start_time_total
        print(f"    ✅ Completed {len(portfolios_list)}/{len(V_dates)} V periods. Opt Time: {optimization_time:.2f}s, Total Time: {total_time:.2f}s")

        return {
            'portfolios': portfolios_list,
            'status': 'optimal',
            'calibration_time': calibration_time,
            'optimization_time': optimization_time,
            'coverage_seq': coverage_seq,
            'radius_seq': radius_seq,
            'volume_seq': volume_seq
        }, results

    except Exception as e:
        total_time = time.time() - start_time_total
        print(f"    ❌ CCPO (rolling_counts) error: {e}")
        print(traceback.format_exc())
        return {'status': f'error: {str(e)}', 'portfolios': [], 'total_time': total_time}


# ============================================================================
# CCPO RUNNER (Rolling - DATES)
# ============================================================================

def run_ccpo_rolling_dates(
    data_path: str,
    lookback: int,
    alpha: float,
    train_start_date: pd.Timestamp, 
    train_end_date: pd.Timestamp,   
    K_end_date: pd.Timestamp,       
    V_dates: pd.DatetimeIndex,
    V_returns_raw: np.ndarray, 
    cfg: config = config
) -> Dict[str, Any]:
    """
    Run CCPO method for one window in rolling evaluation (MODE=dates).
    Handles both single step prediction and multi step prediction logic.
    """
    prediction_mode = getattr(cfg, "PREDICTION_MODE", "single")
    print(f"  Running CCPO-CCO (Rolling Dates, SPLIT={prediction_mode})...")

    start_time_total = time.time()
    n_assets = V_returns_raw.shape[1] if V_returns_raw.ndim > 1 else (1 if V_returns_raw.size > 0 else 0)

    try:
        # 1. Load data
        if prediction_mode == "single":
            # In 2-split: train_end_date is effectively the end of K
            loader = SimpleTimeSeriesDataLoader(base_path=data_path, num_assets=cfg.NUM_ASSETS)
            res = loader.create_all(
                mode="dates",
                lookback=lookback,
                k_end_date=train_end_date.strftime('%Y-%m-%d'), 
                v_end_date=V_dates.max().strftime('%Y-%m-%d'), # Infer V end from provided V_dates
                train_start_date=train_start_date.strftime('%Y-%m-%d'),
                batch_size=cfg.CCPO.BATCH_SIZE,
                shuffle_train=True,
                use_scaler=True,
                resample_freq=cfg.FREQUENCY
            )
            train_loader = res['model']['train_loader']

        else:
            loader = DataLoaderMultiStep(base_path=data_path, num_assets=cfg.NUM_ASSETS)
            res = loader.create_all(
                mode="dates",
                lookback=lookback,
                k_end_date=train_end_date.strftime('%Y-%m-%d'), 
                v_end_date=V_dates.max().strftime('%Y-%m-%d'), # Infer V end from provided V_dates
                horizon=cfg.CCPO.HORIZON,
                train_start_date=train_start_date.strftime('%Y-%m-%d'),
                batch_size=cfg.CCPO.BATCH_SIZE,
                shuffle_train=True,
                use_scaler=True,
                resample_freq=cfg.FREQUENCY
            )
            train_loader = res['model']['train_loader']

        test_loader = res['model']['test_loader']
        scaler = res['scaler'] 

        if len(train_loader.dataset) == 0:
             raise ValueError("Train data loader is empty.")


        # 2. Initialize optimizer
        optimizer = CCPOPortfolioOptimizer(
            alpha=alpha, model_cls=cfg.CCPO.MODEL_CLASS,
            device=cfg.DEVICE, r=cfg.CCPO.LOW_RANK_R,
            use_local_ellipsoid=cfg.CCPO.USE_LOCAL_ELLIPSOID
        )

        # 3. Train & Calibrate
        print(f"    Training {cfg.CCPO.B} bootstrap models...")
        start_time_calib = time.time()

        X_train, Y_train = train_loader.dataset.X, train_loader.dataset.y
        X_predict, Y_predict = test_loader.dataset.X, test_loader.dataset.y

        conformal_predictor = SPCI_and_EnbPI(
            X_train, X_predict,
            Y_train, Y_predict,
            model_cls=cfg.CCPO.MODEL_CLASS, loader=loader, scaler=scaler,
            device=cfg.DEVICE, r=cfg.CCPO.LOW_RANK_R,
            use_local_ellipsoid=cfg.CCPO.USE_LOCAL_ELLIPSOID,
            bins=cfg.CCPO.QRF_BINS,
            max_d=cfg.CCPO.QRF_MAX_DEPTH,
            n_estimators=cfg.CCPO.QRF_N_ESTIMATORS,
            criterion=cfg.CCPO.CRITERION
        )

        results = conformal_predictor.fit_bootstrap_models_online_multistep(
            B=cfg.CCPO.B, batch_size=cfg.CCPO.BATCH_SIZE, EPOCHS=cfg.CCPO.EPOCHS,
            lr=cfg.CCPO.LEARNING_RATE, path=cfg.CCPO.WEIGHTS_PATH, 
            loss_aggregation=cfg.CCPO.LOSS_AGG, cp_residual_mode='aggregated'
        )

        print(f"    Calibrating conformal prediction intervals...")
        conformal_predictor.compute_Widths_Ensemble_online(
            alpha=alpha, smallT=False, use_SPCI=cfg.CCPO.USE_SPCI,
            past_window=cfg.CCPO.PAST_WINDOW, random_state=cfg.SEED
        )
        calibration_time = time.time() - start_time_calib

        mean_coverage_calib, _, coverage_seq, volume_seq, radius_seq = conformal_predictor.get_results()
        if not radius_seq: raise ValueError("Calibration failed: Radius sequence is empty.")
        
        radius = float(np.mean(radius_seq))
        cov_matrix = conformal_predictor.global_cov

        print(f"    ✅ Calibration done - Calib Set Coverage: {mean_coverage_calib:.3f}, Radius: {radius:.6f}, Time: {calibration_time:.2f}s")

        # 4. Optimize
        print(f"    Optimizing portfolio for each of {len(V_dates)} test periods (V)...")
        start_time_opt = time.time()
        portfolios_list = []

        if prediction_mode == "single":
            mu_pred_raw = results["test"]["y_pred"].squeeze(1).cpu().numpy()        # [n_test, n_assets]
        elif prediction_mode == "multi":
            mu_pred_raw = (1 + conformal_predictor.test_pred_raw).prod(dim=1).sub(1).detach().cpu().numpy() # [n_test, n_assets]
        else:
            raise ValueError(f"Unknown prediction mode: {prediction_mode}")
        
        for v_idx, v_date in enumerate(V_dates):
            current_mu = mu_pred_raw[v_idx] if v_idx < len(mu_pred_raw) else np.zeros(n_assets)
            current_radius = radius_seq[v_idx] if v_idx < len(radius_seq) else radius

            opt_result = optimizer.optimize_portfolio_socp(
                mu_hat=current_mu, cov_matrix=cov_matrix, radius=current_radius,
                gamma=cfg.CCPO.GAMMA, formulation=cfg.CCPO.FORMULATION
            )

            if opt_result['status'] == 'optimal':
                portfolios_list.append({
                    'date': v_date, 'weights': opt_result['weights'],
                    'threshold': opt_result['threshold']
                })
            else:
                portfolios_list.append({
                    'date': v_date,
                    'weights': np.ones(n_assets) / n_assets if n_assets > 0 else np.array([]),
                    'threshold': None
                })

        optimization_time = time.time() - start_time_opt
        total_time = time.time() - start_time_total
        print(f"    ✅ Completed {len(portfolios_list)}/{len(V_dates)} V periods. Opt Time: {optimization_time:.2f}s, Total Time: {total_time:.2f}s")

        return {
            'portfolios': portfolios_list,
            'status': 'optimal',
            'calibration_time': calibration_time,
            'optimization_time': optimization_time,
            'coverage_seq': coverage_seq,
            'radius_seq': radius_seq,
            'volume_seq': volume_seq
        }, results

    except Exception as e:
        total_time = time.time() - start_time_total
        print(f"    ❌ CCPO (rolling_dates) error: {e}")
        print(traceback.format_exc())
        return {'status': f'error: {str(e)}', 'portfolios': [], 'total_time': total_time}