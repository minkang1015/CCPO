import torch
from layers.predictors import LSTMModel, DLinear, MLP

# ==========================================
# SPLIT MODE SETTING
# ==========================================
# "3_split": Original (Train / Valid(K) / Test(V))
# "2_split": New (Train(K) / Test(V)) -> Train data is also used for Calibration(K)
SPLIT_MODE = "2_split"  # ["2_split", "3_split"]

# ----  EVALUATION MODE ----
EVALUATION_MODE = "rolling"  # ["direct", "rolling"]

# ---- Data Mode ----
MODE = "counts"              # ["dates", "counts"]

# ---- Base Settings ----
DATA_PATH = "./data"
FREQUENCY = "weekly"
LOOKBACK = 52
NUM_ASSETS = 10     # [5, 10, 30, 49]
ALPHA = 0.05
SEED = 2025
BATCH_SIZE = 32 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 1. 3-Split Mode Configuration (Direct)
# ==========================================
# Used when SPLIT_MODE = "3_split"

# ---- 'dates' Mode ----
TRAIN_END_DATES = "2015-12-31" # Model Train End
VALID_END_DATES = "2020-12-31" # Model Valid(K) End
TEST_END_DATES  = "2023-12-31" # Model Test(V) End

# ---- 'counts' Mode ----
TRAIN_LENGTH = 52 * 10 # The number of Model Train Sequences
LEN_K = 52 * 10        # The number of Model Valid(K) Sequences
LEN_V = 52 * 10        # The number of Model Test(V) Sequences

# ==========================================
# 2. 2-Split Mode Configuration (Direct)
# ==========================================
# Used when SPLIT_MODE = "2_split"

# ---- 'counts' Mode (2-Split) ----
# Train data serves as Calibration(K) data
TRAIN_K_LEN = 1300  # Length for Train(K)
TEST_V_LEN = 260    # Length for Test(V)

# ---- 'dates' Mode (2-Split) ----
K_END_DATE = "2020-12-31" # End of Train(K)
V_END_DATE = "2023-12-31" # End of Test(V)


# ==========================================
# ROLLING Configuration
# ==========================================
class ROLLING:
    WINDOW_TYPE = "sliding"   # ["sliding", "expanding"]

    # ----------------------------------------
    # [Option A] 3-Split Rolling Settings
    # ----------------------------------------
    # Used when SPLIT_MODE = "3_split"
    class SPLIT_3:
        class COUNTS:
            MODEL_TRAIN_LEN = 52 * 15       # Train Length
            K_LEN = 52 * 10                 # K(Calib) Length
            V_LEN = int(52 * 4)             # V(Test) Length
            STEP_SIZE = int(52 * 4)         # Step size

        class DATES:
            MODEL_TRAIN_OFFSET = "10Y" 
            K_PERIOD_OFFSET = "5Y"   
            V_PERIOD_OFFSET = "1Y"   
            STEP_OFFSET = "1Y"       
            ROLLING_START_DATE = "2000-01-01"
            ROLLING_END_DATE = None           

    # ----------------------------------------
    # [Option B] 2-Split Rolling Settings
    # ----------------------------------------
    # Used when SPLIT_MODE = "2_split"
    # Note: TRAIN_K_LEN defines the merged Train+K period length.
    class SPLIT_2:
        class COUNTS:
            TRAIN_K_LEN = 52 * 15           # Merged Train(K) Length
            V_LEN = int(52 * 4)             # V(Test) Length
            STEP_SIZE = int(52 * 4)         # Step size

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
    TIME_LIMIT = 60
    M = 0.99
    m = -M
    zeta = 1e-6

# ---- CCPO Configuration ----
class CCPO:
    MODEL_CLASS = LSTMModel
    LOW_RANK_R = 8      
    USE_LOCAL_ELLIPSOID = False     
    B = 5       
    BATCH_SIZE = 32    
    EPOCHS = 30     
    LEARNING_RATE = 1e-4        
    WEIGHTS_PATH = "./weights/ccpo"     
    PATIENCE = 10
    USE_SPCI = True     
    PAST_WINDOW = 52    
    GAMMA = 1.0     
    FORMULATION = "cco"
    QRF_BINS = 10
    QRF_N_ESTIMATORS = 50
    QRF_MAX_DEPTH = 5
    CRITERION = "squared_error"


# ---- Utility Function ----
def get_periods_per_year(freq: str) -> int:
    f = (freq or "").lower()
    if f in ("w", "week", "weekly"): return 52
    if f in ("d", "day", "daily"): return 252
    if f in ("m", "month", "monthly"): return 12
    return 52