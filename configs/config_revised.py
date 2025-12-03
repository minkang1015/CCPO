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

# ---- Base Settings ----
DATA_PATH = "./data"
FREQUENCY = "weekly"
LOOKBACK = 52
NUM_ASSETS = 5
ALPHA = 0.01
SEED = 2025
BATCH_SIZE = 32 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 1. Mode Configuration (Direct)
# ==========================================

# ---- 'dates' Mode ----
TRAIN_END_DATES = "2015-12-31" # Model Train End
VALID_END_DATES = "2020-12-31" # Model Valid(K) End
TEST_END_DATES  = "2023-12-31" # Model Test(V) End

# ---- 'counts' Mode ----
TRAIN_LENGTH = 52 * 10 # The number of Model Train Sequences
LEN_K = 52 * 10        # The number of Model Valid(K) Sequences
LEN_V = 52 * 10        # The number of Model Test(V) Sequences


# ---- 'counts' Mode (2-Split) ----
# Train data serves as Calibration(K) data
TRAIN_K_LEN = 520
TEST_V_LEN = 156

# ---- 'dates' Mode (2-Split) ----
K_END_DATE = "2020-12-31" # End of Train(K)
V_END_DATE = "2023-12-31" # End of Test(V)


# ==========================================
# ROLLING Configuration
# ==========================================
class ROLLING:
    WINDOW_TYPE = "sliding"   # ["sliding", "expanding"]

    # ----------------------------------------
    # [Option A] Signle Step Rolling Settings
    # ----------------------------------------
    # Used when PRDICTION_MODE = "single"
    class SINGLE:
        class COUNTS:
            TRAIN_K_LEN = 520
            V_LEN = 156
            STEP_SIZE = 156

        class DATES:
            K_PERIOD_OFFSET = "15Y"         # Merged Train(K) Offset
            V_PERIOD_OFFSET = "1Y"   
            STEP_OFFSET = "1Y"       
            ROLLING_START_DATE = "2000-01-01"
            ROLLING_END_DATE = None        

    # ----------------------------------------
    # [Option B] Multi Step Rolling Settings
    # ----------------------------------------
    # Used when PRDICTION_MODE = "multi"
    class MULTI:
        class COUNTS:
            TRAIN_K_LEN = 520
            V_LEN = 156
            STEP_SIZE = 156

        class DATES:
            K_PERIOD_OFFSET = "15Y"         # Merged Train(K) Offset
            V_PERIOD_OFFSET = "1Y"   
            STEP_OFFSET = "1Y"       
            ROLLING_START_DATE = "2000-01-01"
            ROLLING_END_DATE = None

# ---- CPP Configuration ----
class CPP:
    METHODS = ['CPP-MIP', 'SAA']
    OMEGA = 0.03
    TIME_LIMIT = 360
    M = 0.99
    m = -M
    zeta = 1e-6

# ---- CCPO Configuration ----
class CCPO:
    MODEL_CLASS = LSTMModel
    LOW_RANK_R = int(0.8 * NUM_ASSETS)      
    USE_LOCAL_ELLIPSOID = False     
    B = 5
    BATCH_SIZE = 32    
    EPOCHS = 50     
    LEARNING_RATE = 1e-4        
    WEIGHTS_PATH = "./weights/ccpo"     
    USE_SPCI = True     
    PAST_WINDOW = 52    
    GAMMA = 1.0     
    FORMULATION = "cco"
    QRF_BINS = 10
    QRF_N_ESTIMATORS = 50
    QRF_MAX_DEPTH = 5
    CRITERION = "squared_error"
    LOSS_AGG = "mean"    # ["mean", "last"]     # mean in single, last in multi
    HORIZON = 5     # Used only for Multi-Step Prediction (The number of steps to predict)


# ---- Utility Function ----
def get_periods_per_year(freq: str) -> int:
    f = (freq or "").lower()
    if f in ("w", "week", "weekly"): return 52
    if f in ("d", "day", "daily"): return 252
    if f in ("m", "month", "monthly"): return 12
    return 52
