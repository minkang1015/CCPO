"""
Run CCPO (Conformal Prediction + Portfolio Optimization) with rolling windows
Following run_cpp.py structure with time series prediction + conformal calibration + SOCP optimization
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from datetime import datetime
import torch
import cvxpy as cp
from data.data_loader_final import TimeSeriesDataLoader
from utils.portfolios import Portfolio
from utils.metrics import calculate_portfolio_metrics, compare_methods, print_portfolio_metrics
from utils.visualization import create_all_plots
from utils.evaluate import generate_rolling_splits, print_rolling_splits
from layers.cp_utils import set_seed, train_models, compute_residuals
from layers.multi_cp import SPCI_and_EnbPI
from layers.predictors import MLP, DLinear, LSTMModel
from configs.config_revised import CCPO as config_cp


# Logging utility (same as run_cpp)
class Logger:
    """Logger that writes to both console and file"""
    def __init__(self, log_file='./results/ccpo_log.txt'):
        self.log_file = log_file
        self.terminal = sys.stdout
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
    def write(self, message):
        """Write to both terminal and file"""
        self.terminal.write(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message)
    
    def flush(self):
        """Flush both terminal and file"""
        self.terminal.flush()
    
    def log_header(self):
        """Write a header with timestamp"""
        header = f"\n{'='*80}\n"
        header += f"CCPO Experiment Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"{'='*80}\n"
        self.write(header)


class CCPOPortfolioOptimizer:
    """
    CCPO-based portfolio optimizer with 3 steps:
    Step 1: Time series prediction (Bootstrap ensemble)
    Step 2: Conformal calibration (Ellipsoid construction)
    Step 3: SOCP portfolio optimization
    """
    
    def __init__(self, 
                 alpha: float = 0.1,
                 model_cls=DLinear,
                 device=None,
                 r: int = None,
                 use_local_ellipsoid: bool = False,
                 bins: int = 10,
                 n_estimators: int = 50,
                 max_d: int = 5,
                 criterion: str = 'squared_error'
                 ):     # 수정
        """
        Args:
            alpha: Miscoverage rate (e.g., 0.1 for 90% coverage)
            model_cls: Time series model class (DLinear, LSTM, MLP)
            device: Torch device
            r: Low-rank approximation for covariance
            use_local_ellipsoid: Whether to use local covariance
        """
        self.alpha = alpha
        self.model_cls = model_cls
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.r = r
        self.use_local_ellipsoid = use_local_ellipsoid
        self.bins = bins
        self.n_estimators = n_estimators
        self.max_d = max_d
        self.criterion = criterion
        
    def fit_and_calibrate(self,
                         X_K: np.ndarray,
                         X_L: np.ndarray,
                         X_V: np.ndarray,
                         y_K: np.ndarray,
                         y_L: np.ndarray,
                         y_V: np.ndarray,
                         loader: TimeSeriesDataLoader,
                         scaler,
                         B: int = 30,
                         batch_size: int = 32,
                         EPOCHS: int = 100,
                         lr: float = 1e-3,
                         path: str = './weights/',
                         patience: int = 10) -> Dict:
        """
        Step 1 & 2: Fit time series models and calibrate with conformal prediction
        
        Args:
            X_K, y_K: Training data (K samples)
            X_L, y_L: Calibration data (L samples)
            X_V, y_V: Validation data (V samples)
            loader: Data loader for preprocessing
            scaler: Fitted scaler
            B: Number of bootstrap models
            
        Returns:
            result: {
                'mu_pred_L': predicted mean on L set,
                'mu_pred_V': predicted mean on V set,
                'cov_matrix': covariance matrix,
                'radius': conformal radius,
                'coverage': empirical coverage on V set,
                'y_pred_V': predictions on V set,
                'y_true_V': true values on V set
            }
        """
        X_K_t = torch.FloatTensor(X_K)
        y_K_t = torch.FloatTensor(y_K).unsqueeze(1) if y_K.ndim == 2 else torch.FloatTensor(y_K).unsqueeze(1).unsqueeze(2)
        X_L_t = torch.FloatTensor(X_L)
        y_L_t = torch.FloatTensor(y_L).unsqueeze(1) if y_L.ndim == 2 else torch.FloatTensor(y_L).unsqueeze(1).unsqueeze(2)
        X_V_t = torch.FloatTensor(X_V)
        y_V_t = torch.FloatTensor(y_V).unsqueeze(1) if y_V.ndim == 2 else torch.FloatTensor(y_V).unsqueeze(1).unsqueeze(2)
        
        # Initialize conformal predictor
        conformal_predictor = SPCI_and_EnbPI(
            X_K_t, X_L_t, X_V_t,
            y_K_t, y_L_t, y_V_t,
            model_cls=self.model_cls,
            loader=loader,
            scaler=scaler,
            device=self.device,
            r=self.r,
            use_local_ellipsoid=self.use_local_ellipsoid,
            bins=self.bins,
            n_estimators=self.n_estimators,
            max_d=self.max_d,
            criterion=self.criterion
        )
        
        # Fit bootstrap models
        print("  Fitting bootstrap models...")
        results_fit = conformal_predictor.fit_bootstrap_models_online_multistep(
            B=B,
            batch_size=batch_size,
            EPOCHS=EPOCHS,
            lr=lr,
            path=path,
            patience=patience,
            valid_mode=True
        )
        
        # Compute prediction intervals
        print("  Computing conformal prediction intervals...")
        conformal_predictor.compute_Widths_Ensemble_online(
            alpha=self.alpha,
            smallT=False,
            use_SPCI=config_cp.USE_SPCI,
            past_window=config_cp.PAST_WINDOW,
            random_state=config_cp.SEED
        )
        
        # Get results
        mean_coverage, mean_volume, coverage_seq, volume_seq, radius_seq = conformal_predictor.get_results()
        
        
        # mu_pred_L = results_fit['valid']['y_pred']
        # mu_pred_V = results_fit['test']['y_pred']
        mu_pred_L = conformal_predictor.valid_pred.mean(dim=0).squeeze().detach().cpu().numpy()  # (d,)
        mu_pred_V = conformal_predictor.test_pred.mean(dim=0).squeeze().detach().cpu().numpy()   # (d,)
        cov_matrix = conformal_predictor.global_cov          

        y_pred_V = conformal_predictor.test_pred.mean(dim=0).squeeze().detach().cpu().numpy()
        y_true_V = conformal_predictor.Y_predict.squeeze().numpy()
        
        return {
            'mu_pred_L': mu_pred_L,
            'mu_pred_V': mu_pred_V,
            'cov_matrix': cov_matrix,
            'radius': radius_seq,
            'coverage': mean_coverage,
            'volume': mean_volume,
            'y_pred_V': y_pred_V,
            'y_true_V': y_true_V,
            'status': 'optimal'
        }
    
    def optimize_portfolio_socp(self,
                               mu_hat: np.ndarray,
                               cov_matrix: np.ndarray,
                               radius: float,
                               gamma: float = 1.0,
                               formulation: str = 'cco', # 'cco' or 'target'
                               s0: float = None) -> Dict:
        """
        Step 3: SOCP portfolio optimization
        
        CCO formulation: max s  s.t. gamma * mu^T w - sqrt(q) * ||L^T w||_2 >= s
        Target formulation: max mu^T w  s.t. mu^T w - sqrt(q) * ||L^T w||_2 >= s0
        
        Args:
            mu_hat: Expected return vector (d,)
            cov_matrix: Covariance matrix (d, d)
            radius: Conformal radius (q)
            gamma: Risk preference factor (higher = more risk-averse)
            formulation: 'cco' or 'target'
            s0: Threshold for target formulation
            
        Returns:
            result: {
                'weights': optimal weights,
                'objective_value': objective value,
                'status': solver status
            }
        """
        d = len(mu_hat)
        
        try:
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # If not positive definite, use eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(cov_matrix)
            eigvals = np.maximum(eigvals, 1e-6) 
            L = eigvecs @ np.diag(np.sqrt(eigvals))
        
        # CVXPY variables
        w = cp.Variable(d)
        s = cp.Variable()  # Threshold variable
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Budget constraint
            w >= 0           # Long-only
        ]

        if formulation == 'cco':
            # CCO formulation: max s  s.t. mu^T w - radius * ||L^T w||_2 >= s
            # Note: radius from CP already equals sqrt(q), not q itself
            constraints.append(
                gamma * mu_hat @ w - cp.norm(L.T @ w, 2) * radius >= s
            )

            objective = cp.Maximize(s)
        else:
            if s0 is None:
                raise ValueError("s0 must be provided for target formulation")
            
            constraints.append(
                mu_hat @ w - cp.norm(L.T @ w, 2) * radius >= s0
            )
            objective = cp.Maximize(mu_hat @ w)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status in ['optimal', 'optimal_inaccurate']:
                result = {
                    'weights': w.value,
                    'objective_value': problem.value,
                    'status': 'optimal'
                }
                # For CCO formulation, the threshold is the objective value (s)
                if formulation == 'cco':
                    result['threshold'] = problem.value  # This is s* (optimal threshold)
                else:
                    result['threshold'] = s0  # For target formulation, threshold is given
                return result
            else:
                return {
                    'weights': None,
                    'objective_value': None,
                    'threshold': None,
                    'status': problem.status
                }
        except Exception as e:
            print(f"  ❌ SOCP solver error: {e}")
            return {
                'weights': None,
                'objective_value': None,
                'threshold': None,
                'status': f'error: {str(e)}'
            }