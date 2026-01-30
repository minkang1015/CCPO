import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
import math
import time
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import os
from layers.predictors import get_predictor, MLP, DLinear, LSTMModel
import torch.optim as optim
import random
import sys

def set_seed(seed: int):
    """
    Set the random seed for reproducibility across various libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def ellipsoid_volume(covariance_matrix, r):
    # Only compute volume of the ellipsoid along the first r dimensions
    # where r is the number of dimensions with non-zero eigenvalues
    eigenvalues = np.linalg.eigvals(covariance_matrix)
    eigenvalues = np.real(np.sort(eigenvalues)[::-1])
    eps = 1e-6
    num_r = np.sum(eigenvalues > eps)
    det_sigma = np.prod(eigenvalues[:num_r])
    constant_cd = np.pi**(num_r/2) / math.gamma(num_r/2 + 1)  # Volume constant for d-dimensional sphere
    volume = constant_cd * r**num_r * np.sqrt(det_sigma)
    return volume

def adjust_alpha_t(alpha_t, alpha, errs, gamma=0.005, method='simple'):
    if method == 'simple':
        return alpha_t+gamma*(alpha-errs[-1])
    else:
        t = len(errs)
        errs = np.array(errs)
        w_s_ls = np.array([0.95**(t-i) for i in range(t)])  
        return alpha_t+gamma*(alpha-w_s_ls.dot(errs))

def ave_cov_width(df, Y):
    coverage_res = ((np.array(df['lower']) <= Y) & (
        np.array(df['upper']) >= Y)).mean()
    print(f'Average Coverage is {coverage_res}')
    width_res = (df['upper'] - df['lower']).mean()
    print(f'Average Width is {width_res}')
    return [coverage_res, width_res]

window_size = 300

def rolling_avg(x, window=window_size):
    return np.convolve(x, np.ones(window)/window)[(window-1):-window]

def generate_bootstrap_samples(n, m, B):
    '''
      Return: B-by-m matrix, where row b gives the indices for b-th bootstrap sample
    '''
    samples_idx = np.zeros((B, m), dtype=int)
    for b in range(B):
        sample_idx = np.random.choice(n, m)
        samples_idx[b, :] = sample_idx
    return(samples_idx)

def make_bootstrap_loader(dataset, B=30, replace=True, batch_size=64):
    """
    B: num bootstrap models
    return: List of DataLoader for each bootstrap subset
    """
    T = len(dataset)  
    bootstrap_loaders = []
    for b in range(B):
        sampled_indices = np.random.choice(T, size=T, replace=replace)
        subset = Subset(dataset, sampled_indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        bootstrap_loaders.append((loader, set(sampled_indices)))
    return bootstrap_loaders

def strided_app(a, L, S):
    nrows = ((a.shape[0] - L) // S) + 1
    shape = (nrows, L) + a.shape[1:]
    strides = (S * a.strides[0],) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def binning(past_resid, cov_mat_est, alpha, bins = 5):
    beta_ls = np.linspace(start=0, stop=alpha, num=bins)
    sizes = np.zeros(bins)
    for i in range(bins):
        width = np.percentile(past_resid, math.ceil(100 * (1 - alpha + beta_ls[i]))) - \
            np.percentile(past_resid, math.ceil(100 * beta_ls[i]))
        sizes[i] = ellipsoid_volume(cov_mat_est, width)
    i_star = np.argmin(sizes)
    return beta_ls[i_star]

def binning_use_RF_quantile_regr(quantile_regr, cov_mat_est, Xtrain, Ytrain, feature, beta_ls, sample_weight=None):
    feature = feature.reshape(1, -1)
    low_high_pred = quantile_regr.fit(Xtrain, Ytrain, sample_weight).predict(feature)
    num_mid = int(len(low_high_pred)/2)
    low_pred, high_pred = low_high_pred[:num_mid], low_high_pred[num_mid:]
    width = (high_pred-low_pred).flatten()
    width = [ellipsoid_volume(cov_mat_est, w) for w in width]
    i_star = np.argmin(width)
    wid_left, wid_right = low_pred[i_star], high_pred[i_star]
    return i_star, beta_ls[i_star], wid_left, wid_right

def train_models(
    model_cls,
    data_loader,
    EPOCHS: int = 100,
    lr: float = 1e-3,
    path: str = './weights/',
    loss_aggregation: str = 'mean',        # 'mean' or 'last'
    norm_method: str = 'scaling'           # 'scaling' or 'revin'
):
    os.makedirs(path, exist_ok=True)
    models = []
    indices_ls = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.MSELoss()

    for i, (loader_b, indices_b) in enumerate(data_loader):
        sample_x, sample_y, _, _ = next(iter(loader_b))
        
        # 설정 객체 (Predictor가 기대하는 형식)
        class ModelConfig:
            def __init__(self, x, y):
                self.enc_in = x.shape[2]
                self.c_out = y.shape[2]
                self.seq_len = x.shape[1]
                self.pred_len = y.shape[1]
                self.dropout = 0.1
                self.moving_avg = 25 # For DLinear
        
        configs = ModelConfig(sample_x, sample_y)
        
        # Factory를 통한 모델 생성 (RevIN 자동 적용)
        model_b = get_predictor(model_cls, configs, norm_method=norm_method).to(device)
            
        optimizer = optim.Adam(model_b.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        model_name = model_cls.__name__
        model_save_path = f"{path}/{model_name}_model_b{i}.pt"

        for epoch in range(EPOCHS):
            model_b.train()
            total_train_loss = 0.0

            for X_batch, y_batch, _, _ in loader_b:
                if model_cls == MLP and norm_method == 'scaling':
                    # RevIN을 안쓰고 순수 MLP인 경우 예외적으로 flatten 처리
                    if not hasattr(model_b, 'revin'):
                        X_batch = X_batch.float().to(device).view(X_batch.size(0), -1)

                X_batch = X_batch.float().to(device)
                y_batch = y_batch.float().to(device)
              
                optimizer.zero_grad()
                preds = model_b(X_batch)
                
                # multi step loss aggregation
                if preds.ndim == 3 and y_batch.ndim == 3:
                    if loss_aggregation == 'last':
                        loss = criterion(preds[:, -1, :], y_batch[:, -1, :])
                    else: # 'mean'
                        loss = criterion(preds, y_batch)
                else:
                    loss = criterion(preds, y_batch)

                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            scheduler.step()
            if (epoch+1) % 20 == 0:
                print(f"Model {i} ({model_name}), Epoch {epoch+1}, Loss: {total_train_loss/len(loader_b):.4f}")

        print(f"Saving {model_name}_{i} weights.")
        torch.save(model_b.state_dict(), model_save_path)
        
        models.append(model_b)
        indices_ls.append(indices_b)

    return models, indices_ls

def compute_residuals(model_type, valid_loader, test_loader, models, loader, device="cpu", norm_method="scaling"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    def prep_inputs(X, y):
        # RevIN을 사용하거나 Wrapper가 있는 경우 모델이 3D 입력을 기대함
        # MLP 베이스 모델만 사용하는 scaling 모드일 때만 flatten 처리
        if model_type == "MLP" and norm_method == "scaling":
             return X.float().to(device).view(X.size(0), -1), y.float().to(device)
        return X.float().to(device), y.float().to(device)

    def gather_targets(loader):
        ys = []
        for _, y, _, _ in loader:
            ys.append(y)
        return torch.cat(ys, dim=0)

    def inverse(tensor_data, data_loader_instance):
        # scaling 모드이고 외부 스케일러가 존재할 때만 역변환 수행
        if norm_method == "scaling" and hasattr(data_loader_instance, 'inverse_transform'):
            numpy_data = tensor_data.detach().cpu().numpy()
            inversed_data = data_loader_instance.inverse_transform(numpy_data)
            return torch.from_numpy(inversed_data).float()
        # revin 모드이거나 스케일러가 없으면 이미 원본 스케일임
        return tensor_data

    # 타깃 데이터 수집
    if valid_loader is not None:
        Yv = gather_targets(valid_loader)
    Yt = gather_targets(test_loader)

    Pv_list, Pt_list = [], []
    
    with torch.no_grad():
        for m in models:
            m.eval()
            m.to(device)

            if valid_loader is not None:
                outs_v = []
                for Xb, yb, _, _ in valid_loader:
                    Xb, _ = prep_inputs(Xb, yb)
                    outs_v.append(m(Xb).detach().cpu())
                Pv_list.append(torch.cat(outs_v, dim=0))

            outs_t = []
            for Xb, yb, _, _ in test_loader: 
                Xb, _ = prep_inputs(Xb, yb)
                outs_t.append(m(Xb).detach().cpu())
            Pt_list.append(torch.cat(outs_t, dim=0))

    # 테스트 세트 잔차 계산
    Pt = torch.stack(Pt_list).mean(dim=0)
    Yt_inv = inverse(Yt, loader)
    Pt_inv = inverse(Pt, loader)
    Rt = Yt_inv - Pt_inv

    if valid_loader is not None:
        Pv = torch.stack(Pv_list).mean(dim=0)
        Yv_inv = inverse(Yv, loader)
        Pv_inv = inverse(Pv, loader)
        Rv = Yv_inv - Pv_inv
        return {
            "valid": {"y_true": Yv_inv, "y_pred": Pv_inv, "resid": Rv},
            "test":  {"y_true": Yt_inv, "y_pred": Pt_inv, "resid": Rt}
        }
    else:
        return {
            "valid": {"y_true": None, "y_pred": None, "resid": None},
            "test":  {"y_true": Yt_inv, "y_pred": Pt_inv, "resid": Rt}
        }