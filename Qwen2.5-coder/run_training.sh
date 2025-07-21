#!/bin/bash

# Qwen2.5-coder-7B 모델 파인튜닝 스크립트 (QLoRA + T4 최적화)
# 사용법: ./run_training.sh [mode]
#   mode: 
#     'prompt' - 프롬프트 기반 코드 생성 모드 (기본값)
# 

# 학습 모드 설정
# 1. 명령줄 인수를 우선 확인
if [ ! -z "$1" ]; then
    MODE="$1"
# 2. 명령줄 인수가 없으면 환경 설정 파일 확인
elif [ -f "$(dirname "$0")/training_config.env" ]; then
    source "$(dirname "$0")/training_config.env"
    MODE="${TRAINING_MODE:-prompt}"
# 3. 기본값 사용
else
    MODE="prompt"
fi

# 유효한 모드 확인
if [ "$MODE" != "prompt" ]; then
    echo "⚠️ 경고: 유효하지 않은 모드 '$MODE'. 기본값 'prompt'로 설정합니다."
    MODE="prompt"
fi

# 체크포인트 디렉토리 설정
CHECKPOINT_DIR="./checkpoints/prompt-finetuned"

# 체크포인트 자동 감지 및 재개 설정
RESUME_ARGUMENT=""

# 체크포인트 자동 감지 함수
function detect_latest_checkpoint() {
    if [ ! -d "$1" ]; then
        echo ""
        return
    fi
    
    # 체크포인트 디렉토리에서 가장 최신 체크포인트 찾기 (checkpoint- 접두어 기준)
    local latest_dir=$(find "$1" -type d -name "checkpoint-*" | sort -V | tail -n 1)
    
    if [ -n "$latest_dir" ]; then
        echo "$latest_dir"
    else
        echo ""
    fi
}

# 체크포인트 설정
if [ -z "$RESUME_CHECKPOINT" ] || [ "$RESUME_CHECKPOINT" = "none" ]; then
    echo "🆕 체크포인트 없이 처음부터 학습을 시작합니다."
    RESUME_ARGUMENT=""
elif [ "$RESUME_CHECKPOINT" = "auto" ]; then
    # 자동 체크포인트 감지
    DETECTED_CHECKPOINT=$(detect_latest_checkpoint "$CHECKPOINT_DIR")
    
    if [ -n "$DETECTED_CHECKPOINT" ]; then
        echo "🔄 자동으로 감지된 체크포인트: $DETECTED_CHECKPOINT"
        RESUME_ARGUMENT="--resume_from_checkpoint $DETECTED_CHECKPOINT"
    else
        echo "🆕 자동 체크포인트 감지 실패. 처음부터 학습을 시작합니다."
        RESUME_ARGUMENT=""
    fi
else
    # 사용자 지정 체크포인트
    if [ -d "$RESUME_CHECKPOINT" ]; then
        echo "🔄 지정된 체크포인트에서 학습 재개: $RESUME_CHECKPOINT"
        RESUME_ARGUMENT="--resume_from_checkpoint $RESUME_CHECKPOINT"
    else
        echo "⚠️ 지정된 체크포인트를 찾을 수 없습니다: $RESUME_CHECKPOINT"
        echo "🆕 처음부터 학습을 시작합니다."
        RESUME_ARGUMENT=""
    fi
fi

# 필수 패키지 확인 함수
function check_package() {
    python -c "import $1" &>/dev/null
    return $?
}

# 필수 패키지 확인
MISSING_PACKAGES=()
for package in torch transformers datasets accelerate peft trl; do
    if ! check_package $package; then
        MISSING_PACKAGES+=($package)
    fi
done

# 누락된 패키지가 있으면 설치
if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "📦 필요한 패키지 설치 중: ${MISSING_PACKAGES[*]}"
    pip install ${MISSING_PACKAGES[*]}
fi

# 훈련 함수
function run_training() {
    # 환경 변수 설정
    export TOKENIZERS_PARALLELISM=false
    export WANDB_DISABLED=true
    
    # 디렉토리 생성
    mkdir -p "$CHECKPOINT_DIR"
    
    # 학습 시작
    echo "🚀 Qwen2.5-coder 학습 시작 (모드: $MODE)"
    
    # 현재 시간을 이름에 포함하여 로그 파일 생성
    LOG_FILE="./logs/training_${MODE}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p ./logs
    
    # 학습 실행 (백그라운드로 실행하고 로그 저장)
    python Q_train.py --config config.yaml $RESUME_ARGUMENT | tee "$LOG_FILE"
    
    TRAINING_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAINING_EXIT_CODE -eq 0 ]; then
        echo "✅ 학습이 성공적으로 완료되었습니다!"
    else
        echo "❌ 학습 중 오류가 발생했습니다 (종료 코드: $TRAINING_EXIT_CODE)"
    fi
    
    return $TRAINING_EXIT_CODE
}

# 학습 실행
echo "🚀 선택한 모드($MODE)로 파인튜닝을 시작합니다..."
run_training

# 종료 코드
exit $?
