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
from layers.multi_cp_new import SPCI_and_EnbPI
from layers.predictors import MLP, DLinear, LSTMModel
from configs.config_revised import CCPO as config_cp
from configs.config_revised import SEED, PREDICTION_MODE


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
                 ): 
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