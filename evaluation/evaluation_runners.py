import numpy as np
import pandas as pd
from datetime import datetime
import time # For timing
from typing import Dict, Optional, Any
import torch
import traceback
from configs import config_revised as config
from data.data_loader_final import TimeSeriesDataLoader
import cpp.solver as cpp_solver
from layers.multi_cp import SPCI_and_EnbPI 
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
    Run CCPO method for direct evaluation (single split) using config settings.
    Uses Train/Valid(K)/Test(V) split defined in config.
    """
    print(f"  Running CCPO-CCO (Direct Single Split)...")
    print(f"    Lookback={lookback}, Alpha={alpha}")

    start_time_total = time.time()
    loader = TimeSeriesDataLoader(base_path=config.DATA_PATH, num_assets=cfg.NUM_ASSETS) # Use config DATA_PATH

    try:
        # 1. Load data based on config's direct split settings
        create_kwargs = _build_create_all_kwargs(cfg) # Use helper for direct split
        print(f"    loader.create_all kwargs (direct): {create_kwargs}")
        res = loader.create_all(**create_kwargs)

        train_loader = res['model']['train_loader']
        valid_loader = res['model']['valid_loader'] # Used for Calibration (K period)
        test_loader  = res['model']['test_loader']  # Used for final testing (V period)
        scaler = res['scaler']

        # These are RAW returns, not scaled
        V_returns_raw = res['opt']['y_V']
        V_dates = pd.DatetimeIndex(res['opt']['dates_V'])
        n_assets = V_returns_raw.shape[1] if V_returns_raw.ndim > 1 else (1 if V_returns_raw.size > 0 else 0)


        # Check if loaders are empty, which might indicate issues with split dates/lengths
        if len(train_loader.dataset) == 0:
             print("    ⚠️ Warning: Train loader is empty. Check config TRAIN settings.")
             # return {'status': 'error: Empty train loader', 'portfolios': []} # Or allow continuation?
        if len(valid_loader.dataset) == 0:
             print("    ⚠️ Warning: Validation (K) loader is empty. Check config VALID/K settings.")


        # 2. Initialize optimizer
        optimizer = CCPOPortfolioOptimizer(
            alpha=alpha, model_cls=cfg.CCPO.MODEL_CLASS,
            device=cfg.DEVICE, r=cfg.CCPO.LOW_RANK_R,
            use_local_ellipsoid=cfg.CCPO.USE_LOCAL_ELLIPSOID,
            bins=cfg.CCPO.QRF_BINS, n_estimators=cfg.CCPO.QRF_N_ESTIMATORS,
            max_d=cfg.CCPO.QRF_MAX_DEPTH, criterion=cfg.CCPO.CRITERION
        )

        # 3. Train models and Calibrate using Train and Valid(K) loaders
        print(f"    Training {cfg.CCPO.B} bootstrap models...")
        start_time_calib = time.time()

        # Data needs to be Tensors
        X_train, Y_train = train_loader.dataset.X, train_loader.dataset.y
        X_valid, Y_valid = valid_loader.dataset.X, valid_loader.dataset.y # K data
        X_predict, Y_predict = test_loader.dataset.X, test_loader.dataset.y # V data

        # Handle cases where loaders might be empty
        if X_train.nelement() == 0 or X_valid.nelement() == 0:
             raise ValueError("Train or Validation data is empty, cannot proceed.")
        # Allow X_predict to be empty
        if X_predict.nelement() == 0:
             print("    Note: Test data (V) is empty, using validation data as placeholder for predictor init.")
             X_predict, Y_predict = X_valid.clone(), Y_valid.clone()


        conformal_predictor = SPCI_and_EnbPI(
            X_train, X_valid, X_predict,
            Y_train, Y_valid, Y_predict,
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
            patience=cfg.CCPO.PATIENCE, valid_mode=True # Use validation set for early stopping
        )

        print(f"    Calibrating conformal prediction intervals (using K period)...")
        conformal_predictor.compute_Widths_Ensemble_online(
            alpha=alpha, smallT=False, use_SPCI=cfg.CCPO.USE_SPCI,
            past_window=cfg.CCPO.PAST_WINDOW, random_state=cfg.SEED
        )
        calibration_time = time.time() - start_time_calib

        mean_coverage_calib, mean_volume_calib, coverage_seq, volume_seq, radius_seq = conformal_predictor.get_results()
        if not radius_seq: # Handle empty radius sequence
             raise ValueError("Calibration failed: Radius sequence is empty.")
        radius = float(np.mean(radius_seq)) # Use mean radius calibrated on K
        cov_matrix = conformal_predictor.global_cov # Or local if configured

        print(f"    ✅ Calibration done - Calib Set Coverage: {mean_coverage_calib:.3f}, Radius: {radius:.6f}, Time: {calibration_time:.2f}s")

        # 4. Optimize portfolio for each period in V using calibrated results
        print(f"    Optimizing portfolio for each of {len(V_dates)} test periods (V)...")
        start_time_opt = time.time()
        portfolios_list = []
        mu_pred_raw = results['test']['preds'].squeeze(1).cpu().numpy()
        
        # # We need the full raw data series to get lookback windows for V period predictions
        # full_returns_raw = loader.resample_frequency(loader.raw_data, cfg.FREQUENCY)
        # full_returns_values = full_returns_raw.values
        # full_returns_dates = full_returns_raw.index

        # if len(V_dates) == 0:
        #      print("    Note: No V periods to optimize for.")

        for v_idx, v_date in enumerate(V_dates):
            # try:
            #     # Find index in the full RAW series
            #     date_idx = full_returns_dates.get_loc(v_date)
            # except KeyError:
            #     date_idx = full_returns_dates.get_indexer([v_date], method='nearest')[0]
            #     print(f"      Warning: Date {v_date.date()} not found exactly, using nearest: {full_returns_dates[date_idx].date()}")

            # if date_idx < lookback:
            #     print(f"      ⚠️  Skipping {v_date.date()}: not enough history ({date_idx} < {lookback})")
            #     continue

            # # Get lookback window (RAW data), scale it for prediction
            # X_test_raw = full_returns_values[date_idx - lookback: date_idx]
            # if scaler:
            #      X_test_scaled = scaler.transform(X_test_raw)
            # else:
            #      X_test_scaled = X_test_raw # No scaling

            # X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(0).to(cfg.DEVICE)

            # # Predict using ensemble
            # predictions = []
            # with torch.no_grad():
            #      for b in range(cfg.CCPO.B):
            #           model = conformal_predictor.models[b]
            #           model.eval()
            #           pred_scaled = model(X_test_tensor) # Prediction is scaled
            #           # Remove sequence length dim if present (e.g., LSTM)
            #           if pred_scaled.ndim == 3 and pred_scaled.shape[1] == 1:
            #                pred_scaled = pred_scaled.squeeze(1)
            #           elif pred_scaled.ndim != 2:
            #                print(f"      Warning: Unexpected prediction shape {pred_scaled.shape}")
            #           predictions.append(pred_scaled)

            # mean_pred_scaled = torch.stack(predictions).mean(dim=0) # Shape: (1, n_assets)
            # mu_pred_raw = scaler.inverse_transform(mean_pred_scaled.cpu().numpy()).flatten()

            # Optimize portfolio using mu_hat (raw), cov_matrix (raw), radius (calibrated)
            opt_result = optimizer.optimize_portfolio_socp(
                mu_hat=mu_pred_raw[v_idx], cov_matrix=cov_matrix, radius=radius_seq[v_idx],
                gamma=cfg.CCPO.GAMMA, formulation=cfg.CCPO.FORMULATION
            )

            if opt_result['status'] == 'optimal':
                portfolios_list.append({
                    'date': v_date,
                    'weights': opt_result['weights'],
                    'threshold': opt_result['threshold'], # Threshold determined by optimization
                    # 'mu_pred': mu_pred_raw # Optional: store prediction
                })
            else:
                print(f"      ⚠️  Optimization failed for {v_date.date()}: {opt_result['status']}")
                # Fallback to equal weight
                portfolios_list.append({
                    'date': v_date,
                    'weights': np.ones(n_assets) / n_assets if n_assets > 0 else np.array([]),
                    'threshold': None,
                })

        optimization_time = time.time() - start_time_opt
        total_time = time.time() - start_time_total
        print(f"    ✅ Completed {len(portfolios_list)}/{len(V_dates)} V periods. Opt Time: {optimization_time:.2f}s, Total Time: {total_time:.2f}s")

        # Return results needed for aggregate_and_save_results
        return {
            'portfolios': portfolios_list, # List of weights/thresholds per V period
            'status': 'optimal',
            # Include calibration stats (informational)
            'coverage': mean_coverage_calib, # Coverage on K period
            'volume': mean_volume_calib,     # Volume on K period
            'threshold': radius,             # Calibrated radius used for Opt
            'coverage_seq': coverage_seq,    # Optional: detailed sequences
            'volume_seq': volume_seq,
            'radius_seq': radius_seq,
            'calibration_time': calibration_time, # Timing info
            'optimization_time': optimization_time
        }

    except Exception as e:
        total_time = time.time() - start_time_total
        print(f"    ❌ CCPO (direct) error: {e}")
        print(traceback.format_exc())
        return {'status': f'error: {str(e)}', 'portfolios': [], 'total_time': total_time}


# ============================================================================
# CCPO RUNNER (Rolling - COUNTS)
# ============================================================================

def run_ccpo_rolling_counts(
    data_path: str, # Usually config.DATA_PATH
    lookback: int,
    alpha: float,
    # --- Rolling Period Info ---
    model_train_len: int,   # Length of Train period (before K)
    K_len: int,             # Length of Calib (K) period
    V_len: int,             # Length of Test (V) period
    start_idx: int,         # Raw data start index for the Train+K+V window
    # --- V period actual data (for optimization/eval) ---
    V_dates: pd.DatetimeIndex,
    V_returns_raw: np.ndarray, # RAW returns for V period
    cfg: config = config
) -> Dict[str, Any]:
    """
    Run CCPO method for one window in rolling evaluation (MODE=counts).
    Uses Train / K / V lengths relative to start_idx.
    """
    print(f"  Running CCPO-CCO (Rolling Window - Counts)...")
    print(f"    TrainLen={model_train_len}, KLen(Calib)={K_len}, VLen(Test)={V_len}, StartIdx={start_idx}")

    start_time_total = time.time()
    loader = TimeSeriesDataLoader(base_path=config.DATA_PATH, num_assets=cfg.NUM_ASSETS)
    n_assets = V_returns_raw.shape[1] if V_returns_raw.ndim > 1 else (1 if V_returns_raw.size > 0 else 0)

    try:
        # 1. Load data specifically for this window's Train and K periods
        #    V period data (V_returns_raw) is already provided.
        res = loader.create_all(
            mode="counts",
            lookback=cfg.LOOKBACK,
            train_len=model_train_len, # Model Train length
            K=K_len,                 # Model Valid/Calib (K) length
            V=V_len,                     # We handle V manually using V_returns_raw
            start_idx=start_idx,       # Starting index for the raw data window
            batch_size=cfg.CCPO.BATCH_SIZE, # Use CCPO batch size for model loading
            shuffle_train=True,
            use_scaler=True,
            resample_freq=cfg.FREQUENCY # Should match overall frequency
        )

        train_loader = res['model']['train_loader']
        valid_loader = res['model']['valid_loader'] # K period for Calibration
        test_loader = res['model']['test_loader']
        scaler = res['scaler'] # Scaler fitted on Train period raw data

        if len(train_loader.dataset) == 0 or len(valid_loader.dataset) == 0:
             raise ValueError("Train or Validation (K) data loader is empty for this window.")

        # 2. Initialize optimizer
        optimizer = CCPOPortfolioOptimizer(
            alpha=alpha, model_cls=cfg.CCPO.MODEL_CLASS,
            device=cfg.DEVICE, r=cfg.CCPO.LOW_RANK_R,
            use_local_ellipsoid=cfg.CCPO.USE_LOCAL_ELLIPSOID
        )

        # 3. Train models and Calibrate using Train and Valid(K) loaders
        print(f"    Training {cfg.CCPO.B} bootstrap models...")
        start_time_calib = time.time()

        X_train, Y_train = train_loader.dataset.X, train_loader.dataset.y
        X_valid, Y_valid = valid_loader.dataset.X, valid_loader.dataset.y # K data
        X_predict, Y_predict = test_loader.dataset.X, test_loader.dataset.y # V data
                
        
        conformal_predictor = SPCI_and_EnbPI(
            X_train, X_valid, X_predict,
            Y_train, Y_valid, Y_predict,
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
            patience=cfg.CCPO.PATIENCE, valid_mode=True
        )

        print(f"    Calibrating conformal prediction intervals (using K period)...")
        conformal_predictor.compute_Widths_Ensemble_online(
            alpha=alpha, smallT=False, use_SPCI=cfg.CCPO.USE_SPCI,
            past_window=cfg.CCPO.PAST_WINDOW, random_state=cfg.SEED
        )
        calibration_time = time.time() - start_time_calib

        mean_coverage_calib, _, _, _, radius_seq = conformal_predictor.get_results()
        if not radius_seq: raise ValueError("Calibration failed: Radius sequence is empty.")
        
        radius = float(np.mean(radius_seq))
        cov_matrix = conformal_predictor.global_cov

        print(f"    ✅ Calibration done - Calib Set Coverage: {mean_coverage_calib:.3f}, Radius: {radius:.6f}, Time: {calibration_time:.2f}s")

        # 4. Optimize portfolio for each period in V
        print(f"    Optimizing portfolio for each of {len(V_dates)} test periods (V)...")
        start_time_opt = time.time()
        portfolios_list = []

        mu_pred_raw = results["test"]["y_pred"].squeeze(1).cpu().numpy()
        
        # Need full raw data series for lookback windows during V period
        # full_returns_raw = loader.resample_frequency(loader.raw_data, cfg.FREQUENCY)
        # full_returns_values = full_returns_raw.values
        # full_returns_dates = full_returns_raw.index

        # if len(V_dates) == 0:
        #      print("    Note: No V periods to optimize for.")

        for v_idx, v_date in enumerate(V_dates):
        #     try:
        #         date_idx = full_returns_dates.get_loc(v_date)
        #     except KeyError:
        #         date_idx = full_returns_dates.get_indexer([v_date], method='nearest')[0]
        #         print(f"      Warning: Date {v_date.date()} not found exactly, using nearest: {full_returns_dates[date_idx].date()}")

        #     if date_idx < lookback:
        #         print(f"      ⚠️  Skipping {v_date.date()}: not enough history ({date_idx} < {lookback})")
        #         continue

        #     X_test_raw = full_returns_values[date_idx - lookback: date_idx]
        #     if scaler: X_test_scaled = scaler.transform(X_test_raw)
        #     else: X_test_scaled = X_test_raw
        #     X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(0).to(cfg.DEVICE)

        #     predictions = []
        #     with torch.no_grad():
        #         for b in range(cfg.CCPO.B):
        #             model = conformal_predictor.models[b]
        #             model.eval()
        #             pred_scaled = model(X_test_tensor)
        #             if pred_scaled.ndim == 3 and pred_scaled.shape[1] == 1: pred_scaled = pred_scaled.squeeze(1)
        #             predictions.append(pred_scaled)

        #     mean_pred_scaled = torch.stack(predictions).mean(dim=0)
        #     mu_pred_raw = scaler.inverse_transform(mean_pred_scaled.cpu().numpy()).flatten()

            # print(f'{v_idx} prediction: {mu_pred_raw[v_idx]}, radius: {radius_seq[v_idx]}')
            opt_result = optimizer.optimize_portfolio_socp(
                mu_hat=mu_pred_raw[v_idx], cov_matrix=cov_matrix, radius=radius_seq[v_idx],
                gamma=cfg.CCPO.GAMMA, formulation=cfg.CCPO.FORMULATION
            )
            
            # print(f'optimal weights at {v_idx}: {opt_result["weights"]}')
            
            
            if opt_result['status'] == 'optimal':
                portfolios_list.append({
                    'date': v_date, 'weights': opt_result['weights'],
                    'threshold': opt_result['threshold']
                })
            else:
                print(f"      ⚠️  Optimization failed for {v_date.date()}: {opt_result['status']}")
                portfolios_list.append({
                    'date': v_date,
                    'weights': np.ones(n_assets) / n_assets if n_assets > 0 else np.array([]),
                    'threshold': None
                })

        optimization_time = time.time() - start_time_opt
        total_time = time.time() - start_time_total
        print(f"    ✅ Completed {len(portfolios_list)}/{len(V_dates)} V periods. Opt Time: {optimization_time:.2f}s, Total Time: {total_time:.2f}s")

        # Return only essential results for rolling aggregation
        return {
            'portfolios': portfolios_list,
            'status': 'optimal',
            # Optionally include calibration stats if needed for window summary
            'coverage_calib': mean_coverage_calib,
            'radius': radius_seq,
            'calibration_time': calibration_time,
            'optimization_time': optimization_time,
            'cov_matrix': cov_matrix
        }

    except Exception as e:
        total_time = time.time() - start_time_total
        print(f"    ❌ CCPO (rolling_counts) error: {e}")
        print(traceback.format_exc())
        return {'status': f'error: {str(e)}', 'portfolios': [], 'total_time': total_time}


# ============================================================================
# CCPO RUNNER (Rolling - DATES)
# ============================================================================

def run_ccpo_rolling_dates(
    data_path: str, # Usually config.DATA_PATH
    lookback: int,
    alpha: float,
    # --- Rolling Period Info ---
    train_start_date: pd.Timestamp, # Start of Train period
    train_end_date: pd.Timestamp,   # End of Train period
    K_end_date: pd.Timestamp,       # End of Calib (K) period
    # V_end_date is implicitly end of V_dates
    # --- V period actual data (for optimization/eval) ---
    V_dates: pd.DatetimeIndex,
    V_returns_raw: np.ndarray, # RAW returns for V period
    cfg: config = config
) -> Dict[str, Any]:
    """
    Run CCPO method for one window in rolling evaluation (MODE=dates).
    Uses Train / K / V date ranges.
    """
    # K period starts right after Train period ends
    K_start_date = train_end_date + pd.Timedelta(days=1) # Or appropriate offset for frequency

    print(f"  Running CCPO-CCO (Rolling Window - Dates)...")
    print(f"    Train: [{train_start_date.date()} ~ {train_end_date.date()}]")
    print(f"    K(Calib): [{K_start_date.date()} ~ {K_end_date.date()}]")
    print(f"    V(Test): [{V_dates.min().date()} ~ {V_dates.max().date()}] ({len(V_dates)} periods)")

    start_time_total = time.time()
    loader = TimeSeriesDataLoader(base_path=config.DATA_PATH, num_assets=cfg.NUM_ASSETS)
    n_assets = V_returns_raw.shape[1] if V_returns_raw.ndim > 1 else (1 if V_returns_raw.size > 0 else 0)


    try:
        # 1. Load data specifically for this window's Train and K periods
        res = loader.create_all(
            mode="dates",
            lookback=cfg.LOOKBACK,
            # Pass ALL date boundaries
            train_start_date=train_start_date.strftime('%Y-%m-%d'), # Use the specific start
            train_end_date=train_end_date.strftime('%Y-%m-%d'),   # Model Train end
            val_end_date=K_end_date.strftime('%Y-%m-%d'),         # Model Valid/Calib (K) end
            test_end_date=None, # V is handled manually, loader doesn't need test period
            batch_size=cfg.CCPO.BATCH_SIZE,
            shuffle_train=True,
            use_scaler=True,
            resample_freq=cfg.FREQUENCY
        )

        train_loader = res['model']['train_loader']
        valid_loader = res['model']['valid_loader'] # K period for Calibration
        test_loader = res['model']['test_loader']
        scaler = res['scaler'] # Scaler fitted on Train period raw data

        if len(train_loader.dataset) == 0 or len(valid_loader.dataset) == 0:
             raise ValueError("Train or Validation (K) data loader is empty for this window.")


        # 2. Initialize optimizer
        optimizer = CCPOPortfolioOptimizer(
            alpha=alpha, model_cls=cfg.CCPO.MODEL_CLASS,
            device=cfg.DEVICE, r=cfg.CCPO.LOW_RANK_R,
            use_local_ellipsoid=cfg.CCPO.USE_LOCAL_ELLIPSOID
        )

        # 3. Train models and Calibrate using Train and Valid(K) loaders
        print(f"    Training {cfg.CCPO.B} bootstrap models...")
        start_time_calib = time.time()

        X_train, Y_train = train_loader.dataset.X, train_loader.dataset.y
        X_valid, Y_valid = valid_loader.dataset.X, valid_loader.dataset.y # K data
        X_predict, Y_predict = test_loader.dataset.X, test_loader.dataset.y # V data

        conformal_predictor = SPCI_and_EnbPI(
            X_train, X_valid, X_predict,
            Y_train, Y_valid, Y_predict,
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
            patience=cfg.CCPO.PATIENCE, valid_mode=True
        )

        print(f"    Calibrating conformal prediction intervals (using K period)...")
        conformal_predictor.compute_Widths_Ensemble_online(
            alpha=alpha, smallT=False, use_SPCI=cfg.CCPO.USE_SPCI,
            past_window=cfg.CCPO.PAST_WINDOW, random_state=cfg.SEED
        )
        calibration_time = time.time() - start_time_calib

        mean_coverage_calib, _, _, _, radius_seq = conformal_predictor.get_results()
        if not radius_seq: raise ValueError("Calibration failed: Radius sequence is empty.")
        # radius = float(np.mean(radius_seq))
        cov_matrix = conformal_predictor.global_cov
        radius = float(np.mean(radius_seq))
        
        print(f"    ✅ Calibration done - Calib Set Coverage: {mean_coverage_calib:.3f}, Radius: {radius:.6f}, Time: {calibration_time:.2f}s")

        # 4. Optimize portfolio for each period in V
        print(f"    Optimizing portfolio for each of {len(V_dates)} test periods (V)...")
        start_time_opt = time.time()
        portfolios_list = []

        mu_pred_raw = results["test"]["y_pred"].squeeze(1).cpu().numpy()
        
        # full_returns_raw = loader.resample_frequency(loader.raw_data, cfg.FREQUENCY)
        # full_returns_values = full_returns_raw.values
        # full_returns_dates = full_returns_raw.index

        # if len(V_dates) == 0:
        #      print("Note: No V periods to optimize for.")

        for v_idx, v_date in enumerate(V_dates):
            # try:
            #     date_idx = full_returns_dates.get_loc(v_date)
            # except KeyError:
            #     date_idx = full_returns_dates.get_indexer([v_date], method='nearest')[0]
            #     print(f"      Warning: Date {v_date.date()} not found exactly, using nearest: {full_returns_dates[date_idx].date()}")

            # if date_idx < lookback:
            #     print(f"      ⚠️  Skipping {v_date.date()}: not enough history ({date_idx} < {lookback})")
            #     continue

            # X_test_raw = full_returns_values[date_idx - lookback: date_idx]
            # if scaler: X_test_scaled = scaler.transform(X_test_raw)
            # else: X_test_scaled = X_test_raw
            # X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(0).to(cfg.DEVICE)

            # predictions = []
            # with torch.no_grad():
            #     for b in range(cfg.CCPO.B):
            #         model = conformal_predictor.models[b]
            #         model.eval()
            #         pred_scaled = model(X_test_tensor)
            #         if pred_scaled.ndim == 3 and pred_scaled.shape[1] == 1: pred_scaled = pred_scaled.squeeze(1)
            #         predictions.append(pred_scaled)

            # mean_pred_scaled = torch.stack(predictions).mean(dim=0)
            # mu_pred_raw = scaler.inverse_transform(mean_pred_scaled.cpu().numpy()).flatten()

            opt_result = optimizer.optimize_portfolio_socp(
                mu_hat=mu_pred_raw[v_idx], cov_matrix=cov_matrix, radius=radius_seq[v_idx],
                gamma=cfg.CCPO.GAMMA, formulation=cfg.CCPO.FORMULATION
            )

            if opt_result['status'] == 'optimal':
                portfolios_list.append({
                    'date': v_date, 'weights': opt_result['weights'],
                    'threshold': opt_result['threshold']
                })
            else:
                print(f"⚠️  Optimization failed for {v_date.date()}: {opt_result['status']}")
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
            'coverage_calib': mean_coverage_calib,
            'radius': radius_seq[v_idx],
            'calibration_time': calibration_time,
            'optimization_time': optimization_time,
            'cov_matrix': cov_matrix
        }

    except Exception as e:
        total_time = time.time() - start_time_total
        print(f"    ❌ CCPO (rolling_dates) error: {e}")
        print(traceback.format_exc())
        return {'status': f'error: {str(e)}', 'portfolios': [], 'total_time': total_time}