import sys
import os
import warnings
warnings.filterwarnings('ignore')

# --- Path Setup ---
# Add the 'ccpo' directory to the Python path to ensure module imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Imports ---
from configs import config_revised as config
from layers.cp_utils import set_seed

# Runners
from evaluation.run_direct_evaluation import run_direct_evaluation
from evaluation.run_rolling_evaluation import run_rolling_evaluation

if __name__ == "__main__":
    # 1. Set Seed for reproducibility
    set_seed(config.SEED)

    # 2. Retrieve and Validate Configuration
    prediction_mode = getattr(config, "PREDICTION_TYPE", "single").lower()
    data_mode = config.MODE.lower()
    eval_mode = config.EVALUATION_MODE.lower()

    # --- Validation Block ---
    if prediction_mode not in ["single", "multi"]:
        raise ValueError(f"Invalid SPLIT_MODE: '{prediction_mode}'. Must be 'single' or 'multi'.")
    
    if data_mode not in ["counts", "dates"]:
        raise ValueError(f"Invalid MODE: '{data_mode}'. Must be 'counts' or 'dates'.")
        
    if eval_mode not in ["direct", "rolling"]:
        raise ValueError(f"Invalid EVALUATION_MODE: '{eval_mode}'. Must be 'direct' or 'rolling'.")
    # ------------------------

    # 3. Print Execution Status
    print(f"\n{'='*60}")
    print(f"🚀 CCPO Execution Started")
    print(f"{'='*60}")
    print(f" 🔹 Evaluation Type : {eval_mode.upper()}") # Direct or Rolling
    print(f" 🔹 Data Mode       : {data_mode.upper()}")  # Counts or Dates
    print(f" 🔹 Split Mode      : {prediction_mode.upper()}") # 2_SPLIT or 3_SPLIT
    print(f" 🔹 Asset Count     : {config.NUM_ASSETS}")
    print(f" 🔹 Device          : {config.DEVICE}")
    print(f"{'='*60}\n")

    # 4. Run Evaluation based on Mode
    try:
        if eval_mode == "direct":
            results = run_direct_evaluation(
                data_path=config.DATA_PATH,
                frequency=config.FREQUENCY,
                lookback=config.LOOKBACK,
                alpha=config.ALPHA,
                cfg=config
            )
        elif eval_mode == "rolling":
            results = run_rolling_evaluation(
                data_path=config.DATA_PATH,
                frequency=config.FREQUENCY,
                lookback=config.LOOKBACK,
                alpha=config.ALPHA,
                cfg=config
            )
        
        print("\n✅ Main Execution Completed Successfully!")
        
    except Exception as e:
        print(f"\n❌ Execution Failed: {str(e)}")
        # Raise the error again to see the full traceback
        raise e