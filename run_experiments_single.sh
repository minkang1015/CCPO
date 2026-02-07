BASE_DIR="."
CONFIG_FILE="${BASE_DIR}/configs/config_revised.py"
MAIN_FILE="${BASE_DIR}/main.py"

# 설정 파일 백업
cp "$CONFIG_FILE" "${CONFIG_FILE}.bak"

restore_config() {
    echo -e "\n 원본 설정 파일로 복구 중..."
    mv "${CONFIG_FILE}.bak" "$CONFIG_FILE"
    echo "복구 완료."
}
trap restore_config EXIT

# === 실험 파라미터 설정 ===
ALPHAS=(0.05)                 # alpha
SEEDS=(2025)                  # seed
WINDOW_TYPES=("sliding")    # Window Type
ASSETS=(5 10 30 49)           # NUM_ASSETS

K_LENS=(520)                  # 15Y, 10Y, 5Y
V_LENS=(104)                  # 3Y, 2Y, 1Y
BOOTSTRAPS=(20)               # the number of bootstrap models

NORM_METHODS=("scaling") 

echo "🚀 Run CCPO-CCO..."
echo "📂 : $CONFIG_FILE"

for alpha in "${ALPHAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for window in "${WINDOW_TYPES[@]}"; do
      for v_len in "${V_LENS[@]}"; do
        for b_val in "${BOOTSTRAPS[@]}"; do
          for norm in "${NORM_METHODS[@]}"; do
            for k_len in "${K_LENS[@]}"; do
              for asset in "${ASSETS[@]}"; do
                
                echo "--------------------------------------------------------------------------------------------------------"
                echo "▶ Run: Alpha=$alpha | Seed=$seed | Win=$window | Norm=$norm | Asset=$asset | B=$b_val | K=$k_len | V=$v_len"
                echo "--------------------------------------------------------------------------------------------------------"

                # 1. 기본 파라미터 수정
                sed -i "s/^ALPHA = .*/ALPHA = ${alpha}/" "$CONFIG_FILE"
                sed -i "s/^SEED = .*/SEED = ${seed}/" "$CONFIG_FILE"
                sed -i "s/^NUM_ASSETS = .*/NUM_ASSETS = ${asset}/" "$CONFIG_FILE"
                
                # 2. Window Type 수정
                sed -i "s/WINDOW_TYPE = \"[^\"]*\"/WINDOW_TYPE = \"${window}\"/" "$CONFIG_FILE"

                # 3. [추가됨] Normalization Method 수정
                sed -i "s/^NORM_METHOD = .*/NORM_METHOD = \"${norm}\"/" "$CONFIG_FILE"

                # 4. Rolling 관련 파라미터 수정 (들여쓰기 고려)
                sed -i "s/\( *\)TRAIN_K_LEN = .*/\1TRAIN_K_LEN = ${k_len}/" "$CONFIG_FILE"
                sed -i "s/\( *\)V_LEN = .*/\1V_LEN = ${v_len}/" "$CONFIG_FILE"
                sed -i "s/\( *\)STEP_SIZE = .*/\1STEP_SIZE = ${v_len}/" "$CONFIG_FILE"
                sed -i "s/\( *\)B = .*/\1B = ${b_val}/" "$CONFIG_FILE"

                # 실행
                python "$MAIN_FILE"

                sleep 1
              done
            done
          done
        done
      done
    done
  done
done

echo ""
echo "✨ All Done."