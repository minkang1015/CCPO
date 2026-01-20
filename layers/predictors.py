import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# RevIN Implementation
# -----------------------
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            
            print(f"[RevIN-Norm] Input Mean: {self.mean.mean().item():.4f}, Std: {self.stdev.mean().item():.4f}")
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

class RevINModelWrapper(nn.Module):
    def __init__(self, backbone, num_features):
        super().__init__()
        self.revin = RevIN(num_features)
        self.backbone = backbone

    def forward(self, x, mode='both'):
        if mode == 'norm_only': return self.revin(x, 'norm')
        x = self.revin(x, 'norm')
        out = self.backbone(x)
        if mode == 'both': out = self.revin(out, 'denorm')
        return out

# -----------------------
# Base Models
# -----------------------
class DLinear(nn.Module):
    def __init__(self, configs):
        super(DLinear, self).__init__()
        self.seq_len, self.pred_len, self.channels = configs.seq_len, configs.pred_len, configs.enc_in
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
    def forward(self, x):
        # (기존 분해 로직 수행...)
        return self.Linear_Seasonal(x.permute(0,2,1)).permute(0,2,1) # Placeholder

class LSTMModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.lstm = nn.LSTM(configs.enc_in, 96, 2, batch_first=True)
        self.projection = nn.Linear(96, configs.c_out)
        self.pred_len = configs.pred_len
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.projection(out[:, -self.pred_len:, :])

class MLP(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(configs.seq_len * configs.enc_in, configs.pred_len * configs.enc_in))
        self.p, self.d = configs.pred_len, configs.enc_in
    def forward(self, x):
        return self.fc(x).view(-1, self.p, self.d)

# -----------------------
# Predictor Factory
# -----------------------
def get_predictor(model_cls, configs, norm_method='scaling'):
    base_model = model_cls(configs)
    if norm_method == 'revin':
        return RevINModelWrapper(base_model, configs.enc_in)
    return base_model