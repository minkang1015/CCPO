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
from layers.cp_utils import train_models, make_bootstrap_loader, compute_residuals

class SPCI_and_EnbPI():
    def __init__(self, X_train, X_valid, X_predict, Y_train, Y_valid, Y_predict, model_cls, loader, scaler=None, device=None, r=None, bins=10, n_estimators=50, max_d=5, criterion='squared_error', use_local_ellipsoid=False):
        self.model_cls = model_cls
        self.X_train = X_train
        self.X_valid = X_valid
        self.X_predict = X_predict
        self.Y_train = Y_train
        self.Y_valid = Y_valid
        
        n, n1 = len(self.X_valid), len(self.X_predict)
        self.d = self.Y_train.shape[2]  # dimension
        self.Y_predict = Y_predict
        self.scaler = scaler
        self.loader = loader 
        
        self.models = []
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
        self.Ensemble_train_interval_centers = np.ones((n,self.d))*np.inf
        self.Ensemble_pred_interval_centers = np.ones((n1,self.d))*np.inf
        
        self.pred_len = None
        self.d = None
        self.valid_et = None
        self.test_et = None
        self.all_et = None
        
        # QRF / binning / hyperparams setting
        self.bins = bins
        self.beta_hat_bins = []
        self.n_estimators = n_estimators
        self.max_d = max_d
        self.criterion = criterion # {'absolute_error', 'squared_error', 'friedman_mse', 'poisson'}
        self.weigh_residuals = False
        self.c = 0.995
        self.T1 = None
        
        self.Width_Ensemble = None
        self.global_cov = None
        self.global_cov_inv = None
        
        
    def fit_bootstrap_models_online_multistep(self, B, batch_size=64, EPOCHS=100, lr=1e-3, path='./weights/'):
        '''
          Train B bootstrap estimators from subsets of (X_train, Y_train), 
          compute aggregated predictors, and compute the residuals
          '''
          
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        train_dataset = TensorDataset(self.X_train, self.Y_train, torch.zeros(len(self.X_train)), torch.zeros(len(self.X_train)))
        valid_dataset = TensorDataset(self.X_valid, self.Y_valid, torch.zeros(len(self.X_valid)), torch.zeros(len(self.X_valid)))
        test_dataset = TensorDataset(self.X_predict, self.Y_predict, torch.zeros(len(self.X_predict)), torch.zeros(len(self.X_predict)))
        
            
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        bootstrap_loaders = make_bootstrap_loader(train_dataset, B=B, batch_size=batch_size)

        models, indices_ls = train_models(self.model_cls, bootstrap_loaders, valid_loader, EPOCHS=EPOCHS, lr=lr, path=path)
        self.models = models

        result = compute_residuals(
            model_type=self.model_cls,                 
            valid_loader=valid_loader,
            test_loader=test_loader,
            models=self.models,  
            device=device,
            loader = self.loader
        )

        valid_pred  = result["valid"]["y_pred"].to(device)   # [n_valid, pred_len, d]
        test_pred   = result["test"]["y_pred"].to(device)    # [n_test,  pred_len, d]
        valid_resid = result["valid"]["resid"].to(device)    
        test_resid  = result["test"]["resid"].to(device)

        self.valid_pred = valid_pred
        self.test_pred  = test_pred
        self.valid_resid = valid_resid
        self.test_resid  = test_resid

        n_valid, self.pred_len, self.d = valid_pred.shape
        n_test  = test_pred.shape[0]

        self.get_test_et = False
        self.valid_et = self.get_et(valid_resid.reshape(-1, self.d).detach().cpu().numpy())
        
        self.get_test_et = True
        self.test_et  = self.get_et(test_resid.reshape(-1,  self.d).detach().cpu().numpy())
        self.all_et   = np.concatenate([self.valid_et, self.test_et], axis=0)
        
        return result

    def get_local_ellipsoid(self):
        
        if self.use_local_ellipsoid and self.get_test_et:
            idx = self.local_ellipsoid_idx
            X_prev = np.vstack([self.X_valid[idx:], self.X_predict[:idx]])
            max_past = min(1000, len(X_prev))
            X_prev = X_prev[-max_past:]
            
            n_neighbors = int(0.1*max_past)
            knn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_prev)
            
            neighbors = knn.kneighbors(self.X_predict[idx].reshape(1, -1), return_distance=False).reshape(-1)
            Cov_neighbor = np.cov(self.Ensemble_online_resid[idx:][neighbors].T)
            lamb = 0.95
            local_cov = lamb * Cov_neighbor + (1-lamb) * self.global_cov
            cov_now, inv_cov_now = self.get_rank_approx(local_cov) 
            self.cov_matrix_ls.append(cov_now)
            self.local_ellipsoid_idx += 1
            if self.local_ellipsoid_idx % 25 == 0:
                print(X_prev.shape)
                print(f'Local Ellipsoid {self.local_ellipsoid_idx} computed')
        return inv_cov_now

    def get_rank_approx(self, A):
        r = self.r
        if r is not None:
            u, s, v = np.linalg.svd(A, full_matrices=False)
            Ur = u[:, :r]; Sr = np.diag(s[:r]); Vr = v[:r, :]
            Ar = np.dot(Ur, np.dot(Sr, Vr))
            S_inv = np.diag(1 / s[:r])  
            Ar_pseudo_inverse = np.dot(Vr.T, np.dot(S_inv, Ur.T))
        else:
            Ar = A; Ar_pseudo_inverse = np.linalg.inv(A)
        return Ar, Ar_pseudo_inverse

    def get_et(self, residuals):
        if self.get_test_et is False:
            global_cov, global_inv = self.get_rank_approx(np.cov(residuals.T))
            self.global_cov = global_cov
            self.global_cov_inv = global_inv
            
        nonconform_scores = []
        for i in range(len(residuals)):
            if self.use_local_ellipsoid is False:
                cov_mat_est_inv = self.global_cov_inv
            else:
                if self.get_test_et is False:
                    cov_mat_est_inv = self.global_cov_inv
                else:
                    cov_mat_est_inv = self.get_local_ellipsoid()
            nonconform_scores.append(np.sqrt(
                np.matmul(residuals[i], np.matmul(cov_mat_est_inv, residuals[i].T))))
        return np.array(nonconform_scores)

    def compute_Widths_Ensemble_online(self, alpha, stride=1, smallT=True, past_window=100, use_SPCI=False, quantile_regr='RF', random_state=None):
        '''
            stride: control how many steps we predict ahead
            smallT: if True, we would only start with the last n number of LOO residuals, rather than use the full length T ones. Used in change detection
                NOTE: smallT can be important if time-series is very dynamic, in which case training MORE data may actaully be worse (because quantile longer)
                HOWEVER, if fit quantile regression, set it to be FALSE because we want to have many training pts for the quantile regressor
            use_SPCI: if True, we fit conditional quantile to compute the widths, rather than simply using empirical quantile
        '''
        self.random_state = random_state
        self.alpha = alpha
        n1 = len(self.X_valid)
        self.past_window = past_window 
        if smallT:
            n1 = min(self.past_window, len(self.X_valid))
        out_sample_predict = self.Ensemble_pred_interval_centers
        start = time.time()
        if use_SPCI:
            s = stride
            stride = 1
        resid_strided = strided_app(self.all_et[len(self.X_valid) - n1:-1], n1, stride)
        
        print(f'Shape of slided e_t lists is {resid_strided.shape}')
        num_unique_resid = resid_strided.shape[0]
        width_left = np.zeros(num_unique_resid)
        width_right = np.zeros(num_unique_resid)
       
        # NOTE:
        self.QRF_ls = []
        self.i_star_ls = []
        self.radius_ls = []
        
        for i in range(num_unique_resid):
            if use_SPCI:
                remainder = i % s
                if remainder == 0:
                    past_resid = resid_strided[i, :]
                    n2 = self.past_window
                    resid_pred = self.multi_step_QRF(past_resid, i, s, n2)
                
                rfqr= self.QRF_ls[remainder]
                i_star = self.i_star_ls[remainder]
                wid_all = rfqr.predict(resid_pred)
                num_mid = int(len(wid_all)/2)
                wid_left = wid_all[i_star]
                wid_right = wid_all[num_mid+i_star]
                width_left[i] = wid_left
                width_right[i] = wid_right    
                
            else:
                past_resid = resid_strided[i, :]
                cov_mat = self.global_cov if self.use_local_ellipsoid is False else self.cov_matrix_ls[i]
                beta_hat_bin = binning(past_resid, cov_mat, alpha, self.bins)
                self.beta_hat_bins.append(beta_hat_bin)
                width_left[i] = np.percentile(
                    past_resid, math.ceil(100 * beta_hat_bin))
                width_right[i] = np.percentile(
                    past_resid, math.ceil(100 * (1 - alpha + beta_hat_bin)))
            num_print = int(num_unique_resid / 20)
            
            
            if num_print == 0:
                print(
                        f'Radius of Ellipsoid at test {i} is {width_right[i]-width_left[i]}')
            else:
                if i % num_print == 0:
                    print(
                        f'Radius of Ellipsoid at test {i} is {width_right[i]-width_left[i]}')
                    
            self.radius_ls.append(width_right[i]-width_left[i])
            
        print(
            f'Finish Computing {num_unique_resid} unique Prediction Intervals, took {time.time()-start} secs.')
        Ntest = len(out_sample_predict)
        width_left = np.repeat(width_left, stride)[:Ntest]
        width_right = np.repeat(width_right, stride)[:Ntest]
        Width_Ensemble = pd.DataFrame(np.c_[width_left,width_right], columns=['lower', 'upper'])
        self.Width_Ensemble = Width_Ensemble

    def get_results(self):
        covered_or_not, rolling_size = [], []
        for i in range(len(self.test_et)):
            et = self.test_et[i]
            lower, upper = self.Width_Ensemble.iloc[i, 0], self.Width_Ensemble.iloc[i, 1]
            
            covered_or_not.append((et <= upper) and (et >= lower))
            cov_mat = self.global_cov if self.use_local_ellipsoid is False else self.cov_matrix_ls[i]
                        
            upper_v = ellipsoid_volume(cov_mat, upper)
            lower_v = ellipsoid_volume(cov_mat, lower)
            
            rolling_size.append(upper_v - lower_v)
            
            if self.use_local_ellipsoid:
                if i < len(self.cov_matrix_ls):
                    cov_mat = self.cov_matrix_ls[i]
                else:
                    cov_mat = self.global_cov
            else:
                cov_mat = self.global_cov
                
        self.coverages_all = covered_or_not
        self.width_all = rolling_size
        mean_cov, mean_size = np.mean(covered_or_not), np.mean(rolling_size)
        print(f'Average Coverage is {mean_cov:.3f}, Average Ellipsoid Volume is {mean_size:.2e}')
        return mean_cov, mean_size, covered_or_not, rolling_size, self.radius_ls

    '''
        Get Multi-step QRF
    '''

    def multi_step_QRF(self, past_resid, i, s, n2):
        '''
            Train multi-step QRF with the most recent residuals
            i: prediction index
            s: num of multi-step, same as stride
            n2: past window w
        '''
        num = len(past_resid)
        resid_pred = past_resid[-n2:].reshape(1, -1)
        residX = sliding_window_view(past_resid[:num-s+1], window_shape=n2)
        self.cov_matrix = self.global_cov if self.use_local_ellipsoid is False else self.cov_matrix_ls[i]
        for k in range(s):
            residY = past_resid[n2+k:num-(s-k-1)]
            self.train_QRF(residX, residY)
            if i == 0:
                self.QRF_ls.append(self.rfqr)
                self.i_star_ls.append(self.i_star)
            else:
                self.QRF_ls[k] = self.rfqr
                self.i_star_ls[k] = self.i_star
        return resid_pred

    def train_QRF(self, residX, residY):
        alpha = self.alpha
        beta_ls = np.linspace(start=0, stop=alpha, num=self.bins)
        full_alphas = np.append(beta_ls, 1 - alpha + beta_ls)
        self.common_params = dict(n_estimators = self.n_estimators,
                                  max_depth = self.max_d,
                                  criterion = self.criterion,
                                  n_jobs = -1,
                                  random_state = self.random_state)
        
        if residX[:-1].shape[0] > 10000:
            self.rfqr = SampleRandomForestQuantileRegressor(
                **self.common_params, q=full_alphas)
        else:
            self.rfqr = RandomForestQuantileRegressor(
                **self.common_params, q=full_alphas)
        sample_weight = None
        if self.weigh_residuals:
            sample_weight = self.c ** np.arange(len(residY), 0, -1)
        if self.T1 is not None:
            self.T1 = min(self.T1, len(residY))
            self.i_star, _, _, _ = binning_use_RF_quantile_regr(
                self.rfqr, self.cov_matrix, residX[-(self.T1+1):-1], residY[-self.T1:], residX[-1], beta_ls, sample_weight)
        else:
            self.i_star, _, _, _ = binning_use_RF_quantile_regr(
                self.rfqr, self.cov_matrix, residX[:-1], residY, residX[-1], beta_ls, sample_weight)
    
    