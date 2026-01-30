import torch
from layers.predictors import LSTMModel, DLinear, MLP

# ==========================================
# PREDICTION MODE SETTING
# ==========================================
PREDICTION_MODE = "single"  # ["single", "multi"]

# ----  EVALUATION MODE ----
EVALUATION_MODE = "rolling"  # ["direct", "rolling"]

# ---- Data Mode ----
MODE = "counts"              # ["dates", "counts"]

# ==========================================
# NORMALIZATION SETTING
# ==========================================
NORM_METHOD = "revin"

# ---- Base Settings ----
DATA_PATH = "./data"
FREQUENCY = "weekly"
LOOKBACK = 52
NUM_ASSETS = 30
ALPHA = 0.05
SEED = 2025
BATCH_SIZE = 32 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 1. Mode Configuration (Direct)
# ==========================================
TRAIN_END_DATES = "2015-12-31" 
VALID_END_DATES = "2020-12-31" 
TEST_END_DATES  = "2023-12-31" 

TRAIN_LENGTH = 52 * 10 
LEN_K = 52 * 10        
LEN_V = 52 * 10        

TRAIN_K_LEN = 520
TEST_V_LEN = 104

K_END_DATE = "2020-12-31" 
V_END_DATE = "2023-12-31" 

# ==========================================
# ROLLING Configuration
# ==========================================
class ROLLING:
    WINDOW_TYPE = "expanding"
    class SINGLE:
        class COUNTS:
            TRAIN_K_LEN = 520
            V_LEN = 104
            STEP_SIZE = 104
        class DATES:
            K_PERIOD_OFFSET = "15Y"
            V_PERIOD_OFFSET = "1Y"   
            STEP_OFFSET = "1Y"       
            ROLLING_START_DATE = "2000-01-01"
            ROLLING_END_DATE = None        
    class MULTI:
        class COUNTS:
            TRAIN_K_LEN = 520
            V_LEN = 104
            STEP_SIZE = 104
        class DATES:
            K_PERIOD_OFFSET = "15Y"
            V_PERIOD_OFFSET = "1Y"   
            STEP_OFFSET = "1Y"       
            ROLLING_START_DATE = "2000-01-01"
            ROLLING_END_DATE = None

class CPP:
    METHODS = ['CPP-MIP', 'SAA']
    OMEGA = 0.03
    TIME_LIMIT = 2
    M = 0.99
    m = -M
    zeta = 1e-6

class CCPO:
    MODEL_CLASS = LSTMModel
    LOW_RANK_R = int(0.8 * NUM_ASSETS)      
    USE_LOCAL_ELLIPSOID = False     
    B = 20
    BATCH_SIZE = 32    
    EPOCHS = 50     
    LEARNING_RATE = 1e-3       
    WEIGHTS_PATH = f"./weights/ccpo_{NORM_METHOD}"     
    USE_SPCI = True     
    PAST_WINDOW = 52  
    GAMMA = 1.0     
    FORMULATION = "target"
    if FORMULATION == "target":
        S0 = -3.0
    else:
        S0 = None
    QRF_BINS = 10
    QRF_N_ESTIMATORS = 50
    QRF_MAX_DEPTH = 5
    CRITERION = "squared_error"
    LOSS_AGG = "mean"
    HORIZON = 5

def get_periods_per_year(freq: str) -> int:
    f = (freq or "").lower()
    if f in ("w", "week", "weekly"): return 52
    if f in ("d", "day", "daily"): return 252
    if f in ("m", "month", "monthly"): return 12
    return 52