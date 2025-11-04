import torch
from layers.predictors import LSTMModel, DLinear, MLP


# ----  EVALUATION MODE ----
EVALUATION_MODE = "rolling"  # ["direct", "rolling"]

# ---- Data Mode ----
MODE = "counts"                  # ["dates", "counts"]

# ---- Base Settings ----
DATA_PATH = "./data"
FREQUENCY = "weekly"
LOOKBACK = 52
NUM_ASSETS = 10     # [5, 10, 30, 49]
ALPHA = 0.05
SEED = 2025
BATCH_SIZE = 32 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



# EVALUATION_MODE = "direct" 
# ---- 'dates' Mode Configuration ----
TRAIN_END_DATES = "2015-12-31" # Model Train End
VALID_END_DATES = "2020-12-31" # Model Valid(K) End
TEST_END_DATES  = "2023-12-31" # Model Test(V) End

# ---- 'counts' Mode Configuration ----
TRAIN_LENGTH = 780 # The number of Model Train Sequences
LEN_K = 520        # The number of Model Valid(K) Sequences
LEN_V = 260        # The number of Model Test(V) Sequences



# EVALUATION_MODE = "rolling" 
class ROLLING:
    WINDOW_TYPE = "expanding"   # ["sliding", "expanding"]

    # --- 'counts' Mode Rolling Configuration ---
    class COUNTS:
        MODEL_TRAIN_LEN = 52 * 15       # Train Length for Training Forecasting Model
        K_LEN = 52 * 10                 # K(Calib) Length
        V_LEN = int(52 * 4)                  # V(Test) Length
        STEP_SIZE = int(52 * 4)              # Window Seting(Should be same as V_LEN)

    # --- 'dates' Mode Rolling Configuration ---
    class DATES:
        # Train Length for Training Forecasting Model
        MODEL_TRAIN_OFFSET = "10Y" # EX) 10Y

        # K(Calib) Length
        K_PERIOD_OFFSET = "5Y"   # EX) 5Y

        # V(Test) Length
        V_PERIOD_OFFSET = "1Y"   # EX) 1Y

        # Window Seting
        STEP_OFFSET = "1Y"       # Window Seting(Should be same as V_PERIOD_OFFSET)

        # Rolling Window Start/End Dates
        ROLLING_START_DATE = "2000-01-01" # First Period for Training Model
        ROLLING_END_DATE = None           # Last period for V(if None, set as last date in dataset)

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
    LOW_RANK_R = 8      # Low-Rank Approximation for Cov Matrix
    USE_LOCAL_ELLIPSOID = False     # Use Local Ellipsoid
    B = 5       # The Number of Bootstrap Models
    BATCH_SIZE = 32    # Batch Size
    EPOCHS = 30     # Epochs
    LEARNING_RATE = 1e-4        # Learning Rate
    WEIGHTS_PATH = "./weights/ccpo"     # Path Model Weights Saved
    PATIENCE = 10
    USE_SPCI = True     # True
    PAST_WINDOW = 52    # Lookback Window for Training
    GAMMA = 1.0     # Risk Preference
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