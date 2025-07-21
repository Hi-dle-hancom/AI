#!/bin/bash

# DeepSeek Coder 6.7B 모델 파인튜닝 스크립트 (T4 GPU 최적화)
# 사용법: ./run_training.sh [mode]
#   mode: 
#     'complete' - 코드 자동완성 학습 모드 (FIM 형식 지원)
#     'prompt' - 일반 프롬프트 기반 코드 생성 모드 
#     'comment' - 주석 기반 코드 생성 모드
#     'error_fix' - 오류 코드 설명 및 수정 모드
# 
# 지원하는 데이터 형식:
#   - 일반 프롬프트 기반: {"prompt": "명령 텍스트", "completion": "지시에 따른 코드"}
#   - 주석 키워드 기반: {"comment": "주석 텍스트", "code": "주석에 맞는 코드"}
#   - 코드 자동완성: {"content": "코드 내용"}
#   - 에러 수정 형식: {"error_context": "오류 코드", "fixed_code_snippet": "수정된 코드"}

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
if [ "$MODE" != "complete" ] && [ "$MODE" != "prompt" ] && [ "$MODE" != "comment" ] && [ "$MODE" != "error_fix" ]; then
    echo "⚠️ 경고: 유효하지 않은 모드 '$MODE'. 기본값 'prompt'로 설정합니다."
    MODE="prompt"
fi

# 체크포인트 재개 설정
# 명령줄 두 번째 인수가 있으면 체크포인트 경로로 사용
if [ ! -z "$2" ]; then
    RESUME_CHECKPOINT="$2"
# 아니면 환경 설정 파일의 RESUME_CHECKPOINT 사용
elif [ ! -z "${RESUME_CHECKPOINT}" ]; then
    # training_config.env에서 이미 로드됨
    echo "체크포인트 경로(환경변수): ${RESUME_CHECKPOINT}"
else
    RESUME_CHECKPOINT=""
fi

# 학습 모드 출력
echo "학습 모드: $MODE"

# wandb 비활성화 (오류 방지)
export WANDB_DISABLED=true

echo "##################################################"
if [ "$MODE" = "complete" ]; then
    echo "# DeepSeek Coder 6.7B 자동완성(FIM) 학습 시작 #"
    echo "##################################################"
    echo ""
    echo "학습 태그: autocomplete, fill_in_middle"
    echo "학습 모드: 코드 자동완성 학습 (FIM 형식 지원)"
    echo "저장 경로: output/complete-finetuned"
    echo "지원 기능: Fill-in-the-Middle 포맷 및 다양한 데이터 형식 지원"
elif [ "$MODE" = "prompt" ]; then
    echo "# DeepSeek Coder 6.7B 프롬프트 기반 코드 생성 학습 시작 #"
    echo "##################################################"
    echo ""
    echo "학습 태그: prompt_generation"
    echo "학습 모드: 일반 프롬프트 기반 코드 생성 학습"
    echo "저장 경로: output/prompt-finetuned"
    echo "지원 기능: 프롬프트 기반 코드 생성 및 명령 이행 기능 강화"
elif [ "$MODE" = "comment" ]; then
    echo "# DeepSeek Coder 6.7B 주석 기반 코드 생성 학습 시작 #"
    echo "##################################################"
    echo ""
    echo "학습 태그: comment_generation"
    echo "학습 모드: 주석 기반 코드 생성 학습"
    echo "저장 경로: output/comment-finetuned"
    echo "지원 기능: 주석 및 문서화 기반 코드 자동 생성"
elif [ "$MODE" = "error_fix" ]; then
    echo "# DeepSeek Coder 6.7B 오류 수정 학습 시작 #"
    echo "##################################################"
    echo ""
    echo "학습 태그: error_correction"
    echo "학습 모드: 오류 코드 설명 및 수정 학습"
    echo "저장 경로: output/error-fix-finetuned"
    echo "지원 기능: 코드 디버깅, 오류 설명, 코드 수정 기능"
fi

echo ""
echo "--------------------------------------------------"

set +e

echo "====================================================="
echo "wandb 비활성화 (오류 해결)"
export WANDB_DISABLED=true

# tokenizers 병렬 처리 경고 제거
export TOKENIZERS_PARALLELISM=false

echo "====================================================="
echo "시작 시간: $(date)"

# Python 실행 파일 찾기
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python이 설치되어 있지 않습니다!"
    exit 1
fi

echo "Python 명령어: $PYTHON_CMD"

# GPU 정보 확인
echo "GPU 정보:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "nvidia-smi를 찾을 수 없습니다."
fi

echo "Python 버전: $($PYTHON_CMD --version)"
echo "확인 중: 필요한 패키지들..."

# 패키지 확인 함수
check_package() {
    if $PYTHON_CMD -c "import $1" &> /dev/null; then
        echo "✅ $1: $($PYTHON_CMD -c "import $1; print($1.__version__)" 2>/dev/null || echo "설치됨")"
    else
        echo "❌ $1: 설치되지 않음"
        return 1
    fi
}

# 필수 패키지 확인
MISSING_PACKAGES=()
for package in torch transformers datasets accelerate peft trl; do
    if ! check_package $package; then
        MISSING_PACKAGES+=($package)
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "❌ 다음 패키지들이 누락되었습니다: ${MISSING_PACKAGES[*]}"
    echo "requirements.txt로 설치하세요: $PYTHON_CMD -m pip install -r requirements.txt"
    exit 1
fi

# 학습 모드별 안내
if [ "$MODE" = "complete" ]; then
    echo "===================================================="
    echo "자동완성(FIM 형식 포함) 모델 학습 시작"
    echo "===================================================="
elif [ "$MODE" = "prompt" ]; then
    echo "===================================================="
    echo "프롬프트 기반 코드 생성 모델 학습 시작"
    echo "===================================================="
elif [ "$MODE" = "comment" ]; then
    echo "===================================================="
    echo "주석 기반 코드 생성 모델 학습 시작"
    echo "===================================================="
else
    echo "===================================================="
    echo "오류 수정 모델 학습 시작"
    echo "===================================================="
fi

# 데이터 경로 확인
DATA_PATH="./data/train.jsonl"
echo "데이터 경로: $DATA_PATH"

# config.yaml 파일 확인 및 생성
if [ ! -f "config.yaml" ]; then
    echo "⚠️ config.yaml이 없습니다. 기본 설정으로 생성합니다..."
    cat > config.yaml << EOF
# T4 환경 최적화 설정
model_name: "deepseek-ai/deepseek-coder-6.7b-instruct"
torch_dtype: "float16"

# 양자화 설정 (T4 호환)
load_in_4bit: true
bnb_4bit_compute_dtype: "float16"
bnb_4bit_quant_type: "nf4"
bnb_4bit_use_double_quant: true

# LoRA 설정
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1
lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 학습 설정 (T4 메모리 제한 고려)
batch_size: 1
grad_acc: 8
learning_rate: 2e-4
num_epochs: 3
max_length: 1024
warmup_ratio: 0.1
lr_scheduler: "cosine"
weight_decay: 0.01

# 최적화 설정
fp16: true
bf16: false
gradient_checkpointing: true
dataloader_num_workers: 4
remove_unused_columns: false

# 저장 설정
save_steps: 500
eval_steps: 500
logging_steps: 50
save_total_limit: 3

# 데이터 설정
data_path: "$DATA_PATH"
EOF
    echo "✅ config.yaml 생성 완료"
fi

# 로그 디렉토리 구성 (절대경로 사용)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGS_DIR="$SCRIPT_DIR/logs"

# 로그 디렉토리 생성 확인
mkdir -p "$LOGS_DIR"
echo "로그 디렉토리: $LOGS_DIR"

# 모델 학습 실행 함수
run_training() {
    local log_suffix
    
    if [ "$MODE" = "complete" ]; then
        echo "=== 코드 자동완성 학습 시작 (FIM 형식 지원) ($(date)) ==="
        log_suffix="complete"
    elif [ "$MODE" = "error_fix" ]; then
        echo "=== 코드 오류 설명 및 수정 학습 시작 ($(date)) ==="
        log_suffix="error_fix"
    elif [ "$MODE" = "comment" ]; then
        echo "=== 주석 기반 코드 생성 학습 시작 ($(date)) ==="
        log_suffix="comment"
    else
        echo "=== 프롬프트 기반 코드 생성 학습 시작 ($(date)) ==="
        log_suffix="prompt"
    fi
    
    # 로그 파일명 생성 (현재 시간 포함)
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    
    # 체크포인트 경로가 있으면 재개 인자 추가
    if [ ! -z "$RESUME_CHECKPOINT" ]; then
        echo "체크포인트에서 학습 재개: $RESUME_CHECKPOINT"
        LOG_FILE="$LOGS_DIR/training_${log_suffix}_resume_${TIMESTAMP}.log"
        $PYTHON_CMD train.py --config config.yaml --mode $MODE --resume_from_checkpoint "$RESUME_CHECKPOINT" 2>&1 | tee "$LOG_FILE"
    else
        # 체크포인트가 없으면 처음부터 학습
        LOG_FILE="$LOGS_DIR/training_${log_suffix}_${TIMESTAMP}.log"
        $PYTHON_CMD train.py --config config.yaml --mode $MODE 2>&1 | tee "$LOG_FILE"
    fi
    
    TRAINING_EXIT_CODE=$?
    if [ $TRAINING_EXIT_CODE -ne 0 ]; then
        echo "학습이 중단되었습니다 (exit code: $TRAINING_EXIT_CODE)."
    else
        echo "학습이 성공적으로 완료되었습니다!"
    fi
}

# 재시도 없이 학습 1회만 실행
run_training

# 완료 메시지
if [ "$MODE" = "complete" ]; then
    echo "=================================================="
    echo "자동완성(FIM 형식 포함) 모델 학습 완료"
    echo "완료 시간: $(date)"
    echo "=================================================="
    echo "모델 저장 위치: ../models/autocomplete-finetuned/final_model"
    echo "inference.py에서 이 모델을 사용하려면: python inference.py --mode autocomplete --model ../models/autocomplete-finetuned/final_model"
    echo "FIM 형식으로 추론하려면 적절한 입력 형식을 사용하세요."
elif [ "$MODE" = "error_fix" ]; then
    echo "=================================================="
    echo "코드 오류 설명 및 수정 모델 학습 완료"
    echo "완료 시간: $(date)"
    echo "=================================================="
    echo "모델 저장 위치: ../models/error-fix-finetuned/final_model"
    echo "inference.py에서 이 모델을 사용하려면: python inference.py --mode error_fix --model ../models/error-fix-finetuned/final_model"
elif [ "$MODE" = "comment" ]; then
    echo "=================================================="
    echo "주석 기반 코드 생성 모델 학습 완료"
    echo "완료 시간: $(date)"
    echo "=================================================="
    echo "모델 저장 위치: ../models/comment-finetuned/final_model"
    echo "inference.py에서 이 모델을 사용하려면: python inference.py --mode comment --model ../models/comment-finetuned/final_model"
else
    echo "=================================================="
    echo "일반 프롬프트 학습 완료"
    echo "완료 시간: $(date)"
    echo "=================================================="
    echo "모델 저장 위치: ../models/prompt-finetuned/final_model"
    echo "inference.py에서 이 모델을 사용하려면: python inference.py --mode prompt --model ../models/prompt-finetuned/final_model"
fi

# 최종 GPU 상태
echo "최종 GPU 상태:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
fi

# 스팟 인스턴스 모니터링 종료
if [ ! -z "$SPOT_MONITOR_PID" ]; then
    kill $SPOT_MONITOR_PID 2>/dev/null || true
fi