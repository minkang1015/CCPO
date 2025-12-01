import pandas as pd
import numpy as np
import math
import time as time
from layers.cp_utils import *
from sklearn.neighbors import NearestNeighbors
import warnings
from sklearn_quantile import RandomForestQuantileRegressor, SampleRandomForestQuantileRegressor
from numpy.lib.stride_tricks import sliding_window_view
warnings.filterwarnings("ignore")
import torch
from torch.utils.data import TensorDataset, DataLoader
from layers.cp_utils import (
    train_models, make_bootstrap_loader, compute_residuals,
    strided_app, binning, binning_use_RF_quantile_regr, ellipsoid_volume
)

class SPCI_and_EnbPI():
    """
    EnbPI with proper Leave-One-Out (LOO) methodology using Bootstrap.
    
    Key features:
    1. Uses X_train (in-sample) for both training and LOO calibration
    2. Uses X_test (out-of-sample) for final evaluation
    3. Bootstrap B models with replacement sampling
    4. LOO: predict each train sample using models that didn't see it
    """
    def __init__(self, X_train, X_test, Y_train, Y_test, model_cls, loader, scaler=None, device=None, r=None, bins=10, n_estimators=50, max_d=5, criterion='squared_error', use_local_ellipsoid=False):
        """
        Args:
            X_train: Training/calibration data (in-sample) - shape [n_train, seq_len, feature_dim]
            X_test: Test data (out-of-sample) - shape [n_test, seq_len, feature_dim]
            Y_train: Training labels - shape [n_train, pred_len, d]
            Y_test: Test labels - shape [n_test, pred_len, d]
            model_cls: Model class for ensemble
            loader: Data loader configuration
            scaler: Optional data scaler
            device: Torch device (cuda/cpu)
            r: Rank for covariance matrix approximation
            bins: Number of bins for quantile binning
            n_estimators: Number of trees in QRF
            max_d: Max depth for QRF
            criterion: Split criterion for QRF
            use_local_ellipsoid: Whether to use local covariance estimation
        """
        self.model_cls = model_cls
        self.X_train = X_train  # In-sample (for training & calibration)
        self.X_test = X_test    # Out-of-sample (for evaluation)
        self.Y_train = Y_train
        self.Y_test = Y_test
        
        n_train = len(self.X_train)
        n_test = len(self.X_test)
        self.d = self.Y_train.shape[2]  # dimension
        
        self.scaler = scaler
        self.loader = loader 
        
        self.models = []
        self.bootstrap_indices_list = []  # Store which samples each bootstrap used
        self.device = device
        self.r = r
        
        if self.device is None:
           self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
           
        # local ellipsoid
        self.use_local_ellipsoid = use_local_ellipsoid
        self.local_ellipsoid_idx = 0
        self.cov_matrix_ls = []
        
        # ensemble / residual / pred placeholders
        self.Ensemble_online_resid = None
        self.Ensemble_train_interval_centers = np.ones((n_train, self.d)) * np.inf
        self.Ensemble_test_interval_centers = np.ones((n_test, self.d)) * np.inf
        
        self.pred_len = None
        self.d = None
        self.train_et = None  # Non-conformity scores from LOO on train
        self.test_et = None   # Non-conformity scores on test
        self.all_et = None    # Combined
        
        # QRF / binning / hyperparams setting
        self.bins = bins
        self.beta_hat_bins = []
        self.n_estimators = n_estimators
        self.max_d = max_d
        self.criterion = criterion
        self.weigh_residuals = False
        self.c = 0.995
        self.T1 = None
        
        # 결과 저장용
        self.Width_Ensemble = None
        self.global_cov = None
        self.global_cov_inv = None
        
        
    def fit_bootstrap_models_online_multistep(self, B, batch_size=64, EPOCHS=100, lr=1e-3, path='./weights/', loss_aggregation='mean', cp_residual_mode='aggregated'):
        """
        Train B bootstrap estimators and compute LOO residuals for calibration.
        
        Bootstrap LOO methodology:
        1. Train B models using bootstrap sampling (with replacement)
        2. For each train sample i, predict using only models that didn't see i
        3. Compute residuals for all train samples via LOO
        4. Train final models on full train set for test predictions
        
        This gives us n_train calibration residuals efficiently!
        
        Args:
            B: Number of bootstrap models
            batch_size: Batch size for training
            EPOCHS: Training epochs
            lr: Learning rate
            path: Path to save model weights
            patience: Early stopping patience
            loss_aggregation: 'mean' or 'last' - how to aggregate multi-step loss
            cp_residual_mode: 'all_horizon' or 'aggregated' - how to use residuals for CP
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        n_train = len(self.X_train)
        
        print(f"Starting Bootstrap training on {n_train} train samples with B={B} models...")
        start_time = time.time()
        
        # === STEP 1: Bootstrap Training ===
        train_dataset = TensorDataset(
            self.X_train, 
            self.Y_train, 
            torch.zeros(len(self.X_train)), 
            torch.zeros(len(self.X_train))
        )
        
        # Create B bootstrap loaders (with replacement sampling)
        bootstrap_loaders = make_bootstrap_loader(train_dataset, B=B, batch_size=batch_size)
        
        # Train B bootstrap models
        print(f"Training {B} bootstrap models...")
        models, bootstrap_indices_list = train_models(
            self.model_cls, 
            bootstrap_loaders,
            EPOCHS=EPOCHS,
            lr=lr, 
            path=path, 
            loss_aggregation=loss_aggregation,
            cp_residual_mode ='mean'
        )
        
        self.models = models
        self.bootstrap_indices_list = [indices for _, indices in bootstrap_loaders]
        
        print(f"Bootstrap training completed in {time.time() - start_time:.2f}s")
        
        # === STEP 2: LOO Prediction on Train Set ===
        print(f"Computing LOO predictions for {n_train} train samples...")
        start_loo = time.time()
        
        train_pred_loo = []
        train_resid_loo = []
        
        for i in range(n_train):
            if i % 100 == 0:
                print(f"  LOO progress: {i}/{n_train}...")
            
            # Find models that didn't use sample i in training
            valid_model_indices = []
            for b, bootstrap_indices in enumerate(self.bootstrap_indices_list):
                if i not in bootstrap_indices:
                    valid_model_indices.append(b)
            
            # If all models used sample i (rare but possible), use all models
            if len(valid_model_indices) == 0:
                valid_model_indices = list(range(B))
            
            # Predict sample i using valid models
            X_i = self.X_train[i:i+1].to(device)
            Y_i = self.Y_train[i:i+1].to(device)
            
            predictions = []
            with torch.no_grad():
                for b in valid_model_indices:
                    model = self.models[b]
                    model.eval()
                    pred = model(X_i)
                    if pred.ndim == 3 and pred.shape[1] == 1:
                        pred = pred.squeeze(1)
                    predictions.append(pred)
            
            # Average predictions from valid models
            pred_i = torch.stack(predictions).mean(dim=0)  # [1, horizon, d] or [1, d] - SCALED
            
            # ✅ Convert to raw space for residual computation
            if hasattr(self.loader, 'inverse_transform'):
                pred_i_np = pred_i.detach().cpu().numpy()
                Y_i_np = Y_i.detach().cpu().numpy()
                pred_i_raw = self.loader.inverse_transform(pred_i_np)
                Y_i_raw = self.loader.inverse_transform(Y_i_np)
                resid_i = torch.from_numpy(Y_i_raw - pred_i_raw).float().to(device)
                pred_i = torch.from_numpy(pred_i_raw).float().to(device)
            else:
                resid_i = Y_i - pred_i  # Fallback: use scaled residuals
            
            train_pred_loo.append(pred_i)
            train_resid_loo.append(resid_i)
        
        # Concatenate all LOO predictions/residuals
        train_pred_concat = torch.cat(train_pred_loo, dim=0).to(device)  # [N, horizon, d] or [N, d]
        train_resid_concat = torch.cat(train_resid_loo, dim=0).to(device)  # [N, horizon, d] or [N, d]
        
        # Apply CP residual mode
        if cp_residual_mode == 'aggregated' and train_pred_concat.ndim == 3:
            # Aggregate multi-step predictions/residuals
            if loss_aggregation == 'last':
                self.train_pred = train_pred_concat[:, -1, :]  # [N, d]
                self.train_resid = train_resid_concat[:, -1, :]  # [N, d]
                print(f"CP Residual Mode: Using LAST horizon step for calibration")
            else:  # 'mean'
                self.train_pred = train_pred_concat.mean(dim=1)  # [N, d]
                self.train_resid = train_resid_concat.mean(dim=1)  # [N, d]
                print(f"CP Residual Mode: Using MEAN across horizon for calibration")
        else:
            # Use all horizon steps (flatten to [N*horizon, d])
            self.train_pred = train_pred_concat
            self.train_resid = train_resid_concat
            if train_pred_concat.ndim == 3:
                print(f"CP Residual Mode: Using ALL horizon steps ({train_pred_concat.shape[1]} steps) for calibration")
        
        print(f"LOO predictions completed in {time.time() - start_loo:.2f}s")
        
        # === STEP 3: Test Set Prediction ===
        # Use all B models for test predictions (already returns raw predictions/residuals)
        print(f"Computing test predictions...")
        test_dataset = TensorDataset(
            self.X_test,
            self.Y_test,
            torch.zeros(len(self.X_test)),
            torch.zeros(len(self.X_test))
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        result_test = compute_residuals(
            model_type=self.model_cls,
            valid_loader=None,
            test_loader=test_loader,
            models=self.models,
            device=device,
            loader=self.loader
        )
        
        # compute_residuals already returns inverse-transformed (raw) predictions/residuals
        test_pred_raw = result_test["test"]["y_pred"].to(device)
        test_resid_raw = result_test["test"]["resid"].to(device)
        
        # Apply CP residual mode to test predictions/residuals
        if cp_residual_mode == 'aggregated' and test_pred_raw.ndim == 3:
            if loss_aggregation == 'last':
                self.test_pred = test_pred_raw[:, -1, :]  # [N_test, d]
                self.test_resid = test_resid_raw[:, -1, :]  # [N_test, d]
            else:  # 'mean'
                self.test_pred = test_pred_raw.mean(dim=1)  # [N_test, d]
                self.test_resid = test_resid_raw.mean(dim=1)  # [N_test, d]
        else:
            self.test_pred = test_pred_raw
            self.test_resid = test_resid_raw
        
        # Extract dimensions
        if self.train_pred.ndim == 3:
            n_train, self.pred_len, self.d = self.train_pred.shape
        else:
            n_train, self.d = self.train_pred.shape
            self.pred_len = 1
        
        if self.test_pred.ndim == 3:
            n_test = self.test_pred.shape[0]
        else:
            n_test = self.test_pred.shape[0]
        
        # === STEP 4: Compute Non-conformity Scores ===
        print(f"Computing non-conformity scores for {n_train} train samples (LOO)...")
        self.get_test_et = False
        # Reshape: handle both [N, d] and [N, horizon, d]
        train_resid_flat = self.train_resid.reshape(-1, self.d).detach().cpu().numpy()
        self.train_et = self.get_et(train_resid_flat)
        
        print(f"Computing non-conformity scores for {n_test} test samples...")
        self.get_test_et = True
        test_resid_flat = self.test_resid.reshape(-1, self.d).detach().cpu().numpy()
        self.test_et = self.get_et(test_resid_flat)
        
        self.all_et = np.concatenate([self.train_et, self.test_et], axis=0)
        
        result = {
            "train": {
                "y_pred": self.train_pred,
                "resid": self.train_resid
            },
            "test": {
                "y_pred": self.test_pred,
                "resid": self.test_resid
            }
        }
        
        print(f"✅ Total train samples with LOO residuals: {len(self.train_et)}")
        print(f"✅ Total test samples: {len(self.test_et)}")
        print(f"✅ Total time: {time.time() - start_time:.2f}s")
        
        return result
    
    def update_covariance_with_past_residuals(self, past_test_residuals):
        """
        Update global covariance by including past test residuals from previous rolling windows.
        
        Use case: In expanding window scenarios where we retrain periodically
        - Window 1: Train [0~15Y] -> get 780 train residuals
        - Window 2: Train [0~20Y] -> get 1040 train residuals + 260 past test residuals from W1
        - Window 3: Train [0~25Y] -> get 1300 train residuals + 520 past test residuals from W1-W2
        
        Benefits:
        - More samples -> more accurate covariance estimation
        - Captures distribution evolution over time
        - Better uncertainty quantification
        
        Args:
            past_test_residuals: [n_past_test, d] array of test residuals from previous windows
        """
        if past_test_residuals is None or len(past_test_residuals) == 0:
            print("⚠️  No past test residuals provided. Using only current train residuals.")
            return
        
        # Combine current train residuals with past test residuals
        train_resid_np = self.train_resid.reshape(-1, self.d).detach().cpu().numpy()
        all_residuals = np.vstack([train_resid_np, past_test_residuals])
        
        print(f"🔄 Updating covariance: {len(train_resid_np)} train + {len(past_test_residuals)} past test = {len(all_residuals)} total residuals")
        
        # Recompute global covariance with all residuals
        self.global_cov, self.global_cov_inv = self.get_rank_approx(np.cov(all_residuals.T))
        
        # Recompute non-conformity scores for train samples using updated covariance
        self.train_et = self._compute_mahalanobis_scores(train_resid_np)
        
        # Update all_et if test_et already computed
        if hasattr(self, 'test_et') and self.test_et is not None:
            self.all_et = np.concatenate([self.train_et, self.test_et], axis=0)
        
        print(f"✅ Covariance updated. Train non-conformity scores recomputed.")
    
    def _compute_mahalanobis_scores(self, residuals):
        """
        Helper method to compute Mahalanobis scores given residuals and current global covariance.
        
        Args:
            residuals: [n, d] array
        
        Returns:
            scores: [n,] array of non-conformity scores
        """
        scores = []
        for i in range(len(residuals)):
            score = np.sqrt(
                np.matmul(residuals[i], np.matmul(self.global_cov_inv, residuals[i].T))
            )
            scores.append(score)
        return np.array(scores)

    def get_local_ellipsoid(self):
        """Compute local covariance matrix using KNN on past residuals."""
        if self.use_local_ellipsoid and self.get_test_et:
            idx = self.local_ellipsoid_idx
            # Use past train + test samples
            X_prev = np.vstack([self.X_train[idx:], self.X_test[:idx]])
            max_past = min(1000, len(X_prev))
            X_prev = X_prev[-max_past:]
            
            n_neighbors = int(0.1 * max_past)
            knn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_prev)
            
            neighbors = knn.kneighbors(self.X_test[idx].reshape(1, -1), return_distance=False).reshape(-1)
            Cov_neighbor = np.cov(self.Ensemble_online_resid[idx:][neighbors].T)
            lamb = 0.95
            local_cov = lamb * Cov_neighbor + (1 - lamb) * self.global_cov
            cov_now, inv_cov_now = self.get_rank_approx(local_cov)
            self.cov_matrix_ls.append(cov_now)
            self.local_ellipsoid_idx += 1
            if self.local_ellipsoid_idx % 25 == 0:
                print(X_prev.shape)
                print(f'Local Ellipsoid {self.local_ellipsoid_idx} computed')
        return inv_cov_now

    def get_rank_approx(self, A):
        """Compute rank-r approximation of covariance matrix."""
        r = self.r
        if r is not None:
            # Rank r approximation via SVD
            u, s, v = np.linalg.svd(A, full_matrices=False)
            Ur = u[:, :r]
            Sr = np.diag(s[:r])
            Vr = v[:r, :]
            Ar = np.dot(Ur, np.dot(Sr, Vr))
            S_inv = np.diag(1 / s[:r])
            Ar_pseudo_inverse = np.dot(Vr.T, np.dot(S_inv, Ur.T))
        else:
            Ar = A
            Ar_pseudo_inverse = np.linalg.inv(A)
        return Ar, Ar_pseudo_inverse

    def get_et(self, residuals):
        """
        Compute non-conformity scores as Mahalanobis distances.
        
        e_t = sqrt(residual^T @ Sigma^{-1} @ residual)
        
        Args:
            residuals: Array of shape [n, d] where d is dimension
            
        Returns:
            Array of non-conformity scores [n,]
        """
        if self.get_test_et is False:
            # Compute global covariance from train residuals
            global_cov, global_inv = self.get_rank_approx(np.cov(residuals.T))
            self.global_cov = global_cov
            self.global_cov_inv = global_inv
            print(f"Global covariance computed from {len(residuals)} train residuals (LOO)")
        
        # Compute non-conformity scores
        nonconform_scores = []
        for i in range(len(residuals)):
            if self.use_local_ellipsoid is False:
                cov_mat_est_inv = self.global_cov_inv
            else:
                if self.get_test_et is False:
                    cov_mat_est_inv = self.global_cov_inv
                else:
                    cov_mat_est_inv = self.get_local_ellipsoid()
            
            # Mahalanobis distance = sqrt(x^T Sigma^{-1} x)
            score = np.sqrt(
                np.matmul(residuals[i], np.matmul(cov_mat_est_inv, residuals[i].T))
            )
            nonconform_scores.append(score)
        
        return np.array(nonconform_scores)

    def compute_Widths_Ensemble_online(self, alpha, stride=1, smallT=True, past_window=100, use_SPCI=False, quantile_regr='RF', random_state=None):
        """
        Compute prediction intervals using train non-conformity scores.
        
        Key advantage of LOO: We now have n_train samples instead of just a subset
        for computing quantiles, leading to tighter and more accurate intervals.
        
        Args:
            alpha: Miscoverage level (e.g., 0.05 for 95% coverage)
            stride: Prediction stride
            smallT: Whether to use limited past window
            past_window: Size of past window for quantile computation
            use_SPCI: Whether to use SPCI (quantile regression) instead of EnbPI
            quantile_regr: Type of quantile regressor ('RF' for random forest)
            random_state: Random seed
        """
        self.random_state = random_state
        self.alpha = alpha
        n_train = len(self.X_train)
        
        self.past_window = past_window
        if smallT:
            # Use at most past_window number of train residuals
            n_train = min(self.past_window, len(self.X_train))
        
        out_sample_predict = self.Ensemble_test_interval_centers
        start = time.time()
        
        if use_SPCI:
            s = stride
            stride = 1
        
        # Rolling window of residuals
        # Use train non-conformity scores (LOO computed!)
        resid_strided = strided_app(self.all_et[len(self.X_train) - n_train:-1], n_train, stride)
        
        print(f'Shape of slided e_t lists is {resid_strided.shape}')
        print(f'Using {n_train} train samples for quantile computation (LOO method)')
        
        num_unique_resid = resid_strided.shape[0]
        width_left = np.zeros(num_unique_resid)
        width_right = np.zeros(num_unique_resid)
        
        self.QRF_ls = []
        self.i_star_ls = []
        self.radius_ls = []
        
        for i in range(num_unique_resid):
            if use_SPCI:
                remainder = i % s
                if remainder == 0:
                    # Update QRF
                    past_resid = resid_strided[i, :]
                    n2 = self.past_window
                    resid_pred = self.multi_step_QRF(past_resid, i, s, n2)
                
                rfqr = self.QRF_ls[remainder]
                i_star = self.i_star_ls[remainder]
                wid_all = rfqr.predict(resid_pred)
                num_mid = int(len(wid_all) / 2)
                wid_left = wid_all[i_star]
                wid_right = wid_all[num_mid + i_star]
                width_left[i] = wid_left
                width_right[i] = wid_right
            else:
                # EnbPI: Use empirical quantiles
                past_resid = resid_strided[i, :]
                cov_mat = self.global_cov if self.use_local_ellipsoid is False else self.cov_matrix_ls[i]
                beta_hat_bin = binning(past_resid, cov_mat, alpha, self.bins)
                self.beta_hat_bins.append(beta_hat_bin)
                
                # Compute quantiles
                width_left[i] = np.percentile(past_resid, math.ceil(100 * beta_hat_bin))
                width_right[i] = np.percentile(past_resid, math.ceil(100 * (1 - alpha + beta_hat_bin)))
            
            num_print = int(num_unique_resid / 20)
            radius = width_right[i] - width_left[i]
            
            if num_print == 0:
                print(f'Radius of Ellipsoid at test {i} is {radius:.6f}')
            else:
                if i % num_print == 0:
                    print(f'Radius of Ellipsoid at test {i} is {radius:.6f}')
            
            self.radius_ls.append(radius)
        
        print(f'Finished computing {num_unique_resid} unique Prediction Intervals, took {time.time() - start:.2f} secs.')
        
        Ntest = len(out_sample_predict)
        width_left = np.repeat(width_left, stride)[:Ntest]
        width_right = np.repeat(width_right, stride)[:Ntest]
        
        Width_Ensemble = pd.DataFrame(np.c_[width_left, width_right], columns=['lower', 'upper'])
        self.Width_Ensemble = Width_Ensemble

    def get_results(self):
        """Compute coverage and average ellipsoid volume."""
        covered_or_not, rolling_size = [], []
        
        for i in range(len(self.test_et)):
            et = self.test_et[i]
            lower, upper = self.Width_Ensemble.iloc[i, 0], self.Width_Ensemble.iloc[i, 1]
            
            covered_or_not.append((et <= upper) and (et >= lower))
            
            if self.use_local_ellipsoid:
                if i < len(self.cov_matrix_ls):
                    cov_mat = self.cov_matrix_ls[i]
                else:
                    cov_mat = self.global_cov
            else:
                cov_mat = self.global_cov
            
            upper_v = ellipsoid_volume(cov_mat, upper)
            lower_v = ellipsoid_volume(cov_mat, lower)
            
            rolling_size.append(upper_v - lower_v)
        
        self.coverages_all = covered_or_not
        self.width_all = rolling_size
        mean_cov, mean_size = np.mean(covered_or_not), np.mean(rolling_size)
        
        print(f'Average Coverage is {mean_cov:.3f}, Average Ellipsoid Volume is {mean_size:.2e}')
        print(f'Coverage computed using {len(self.train_et)} LOO train samples')
        
        return mean_cov, mean_size, covered_or_not, rolling_size, self.radius_ls

    '''
        Multi-step QRF methods (for SPCI)
    '''

    def multi_step_QRF(self, past_resid, i, s, n2):
        """Train multi-step QRF with the most recent residuals."""
        num = len(past_resid)
        resid_pred = past_resid[-n2:].reshape(1, -1)
        residX = sliding_window_view(past_resid[:num - s + 1], window_shape=n2)
        self.cov_matrix = self.global_cov if self.use_local_ellipsoid is False else self.cov_matrix_ls[i]
        
        for k in range(s):
            residY = past_resid[n2 + k:num - (s - k - 1)]
            self.train_QRF(residX, residY)
            if i == 0:
                # Initial training, append QRF to QRF_ls
                self.QRF_ls.append(self.rfqr)
                self.i_star_ls.append(self.i_star)
            else:
                # Retraining, update QRF to QRF_ls
                self.QRF_ls[k] = self.rfqr
                self.i_star_ls[k] = self.i_star
        
        return resid_pred

    def train_QRF(self, residX, residY):
        """Train quantile random forest for SPCI."""
        alpha = self.alpha
        beta_ls = np.linspace(start=0, stop=alpha, num=self.bins)
        full_alphas = np.append(beta_ls, 1 - alpha + beta_ls)
        
        self.common_params = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_d,
            criterion=self.criterion,
            n_jobs=-1,
            random_state=self.random_state
        )
        
        if residX[:-1].shape[0] > 10000:
            self.rfqr = SampleRandomForestQuantileRegressor(
                **self.common_params, q=full_alphas
            )
        else:
            self.rfqr = RandomForestQuantileRegressor(
                **self.common_params, q=full_alphas
            )
        
        sample_weight = None
        if self.weigh_residuals:
            sample_weight = self.c ** np.arange(len(residY), 0, -1)
        
        if self.T1 is not None:
            self.T1 = min(self.T1, len(residY))
            self.i_star, _, _, _ = binning_use_RF_quantile_regr(
                self.rfqr, self.cov_matrix, residX[-(self.T1 + 1):-1], 
                residY[-self.T1:], residX[-1], beta_ls, sample_weight
            )
        else:
            self.i_star, _, _, _ = binning_use_RF_quantile_regr(
                self.rfqr, self.cov_matrix, residX[:-1], residY, 
                residX[-1], beta_ls, sample_weight
            )