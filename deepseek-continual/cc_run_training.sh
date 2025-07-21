#!/bin/bash

# DeepSeek Coder 6.7B 모델 지속학습 스크립트 (A10G GPU 최적화)
# 사용법: ./run_training.sh [mode]
#   mode: 
#     'complete' - [제1차] 코드 자동완성 학습 모드 (FIM 형식 지원)
#     'prompt' - [제2차] 일반 프롬프트 기반 코드 생성 모드 
#     'comment' - [제3차] 주석 기반 코드 생성 모드
#     'error_fix' - [제4차] 오류 코드 설명 및 수정 모드
# 
# 지원하는 데이터 형식:
#   - 일반 프롬프트 기반: {"messages": [{"role": "user", "content": "명령 텍스트"}, {"role": "assistant", "content": "지시에 따른 코드"}]}
#   - 주석 키워드 기반: {"prompt": "# 주석 텍스트", "completion": "주석에 맞는 코드"}
#   - 코드 자동완성 FIM (Fill-in-the-Middle)
#   - 에러 설명/수정 형식

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

# 체크포인트 자동 감지 및 재개 설정
# 체크포인트 경로 설정 - AWS 환경에 맞게 수정
# 모드에 해당하는 체크포인트 및 모델 디렉토리 설정
case "$MODE" in
    "complete")
        CHECKPOINT_DIR="./checkpoints/autocomplete-finetuned"
        ;;
    "prompt")
        CHECKPOINT_DIR="./checkpoints/prompt-finetuned"
        ;;
    "comment")
        CHECKPOINT_DIR="./checkpoints/comment-finetuned"
        ;;
    "error_fix")
        CHECKPOINT_DIR="./checkpoints/error-fix-finetuned"
        ;;
    *)
        CHECKPOINT_DIR="./checkpoints/prompt-finetuned"
        ;;
esac

# 최종 모델 경로 설정
case "$MODE" in
    "complete")
        FINAL_MODEL_DIR="../models/autocomplete-finetuned/final_model"
        ;;
    "prompt")
        FINAL_MODEL_DIR="../models/prompt-finetuned/final_model"
        ;;
    "comment")
        FINAL_MODEL_DIR="../models/comment-finetuned/final_model"
        ;;
    "error_fix")
        FINAL_MODEL_DIR="../models/error-fix-finetuned/final_model"
        ;;
    *)
        FINAL_MODEL_DIR="../models/prompt-finetuned/final_model"  # 기본값
        ;;
esac

# 명령줄 두 번째 인수가 있으면 해당 경로 사용
if [ ! -z "$2" ]; then
    RESUME_CHECKPOINT="$2"
    echo "👉 체크포인트 경로(명령줄 지정): ${RESUME_CHECKPOINT}"
# 환경 변수가 설정되어 있으면 사용
elif [ ! -z "${RESUME_CHECKPOINT}" ] && [ "${RESUME_CHECKPOINT}" != "auto" ]; then
    echo "👉 체크포인트 경로(환경변수 지정): ${RESUME_CHECKPOINT}"
# 환경변수가 'auto'로 설정되어 있으면 최신 체크포인트 자동 감지
elif [ "${RESUME_CHECKPOINT}" = "auto" ]; then
    echo "🔎 최신 체크포인트 검색 중..."
    
    # 체크포인트 디렉토리 존재 확인
    mkdir -p "$CHECKPOINT_DIR"
    
    # 체크포인트 파일은 checkpoint-*.pt 형태로 저장됨
    LATEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -maxdepth 1 -name "checkpoint-*.pt" | sort -V | tail -n 1)
    
    if [ ! -z "$LATEST_CHECKPOINT" ]; then
        # 최신 체크포인트 발견
        RESUME_CHECKPOINT="$LATEST_CHECKPOINT"
        echo "👏 최신 체크포인트 발견: $RESUME_CHECKPOINT"
    elif [ -d "$FINAL_MODEL_DIR" ]; then
        # 최신 체크포인트 없음, 최종 모델 사용
        RESUME_CHECKPOINT="$FINAL_MODEL_DIR"
        echo "💾 최종 모델을 체크포인트로 사용: $RESUME_CHECKPOINT"
    else
        # 체크포인트나 최종 모델 없음
        RESUME_CHECKPOINT=""
        echo "🔍 저장된 체크포인트가 없습니다. 처음부터 학습합니다."
    fi
else
    RESUME_CHECKPOINT=""
    echo "🌟 새 학습 시작: 체크포인트 없음"
fi

# 학습 모드 출력
echo "👀 선택된 학습 모드: $MODE"

# wandb 비활성화 (오류 방지)
export WANDB_DISABLED=true

echo "##################################################"
if [ "$MODE" = "complete" ]; then
    echo "# DeepSeek Coder 6.7B 자동완성(FIM) 지속학습 시작 #"
    echo "##################################################"
    echo ""
    echo "학습 태그: autocomplete, fill_in_middle (지속학습)"
    echo "학습 모드: [1차] 코드 자동완성 학습 (FIM 형식 지원)"
    echo "저장 경로: ../models/autocomplete-finetuned"
    echo "지원 기능: Fill-in-the-Middle 포맷 및 다양한 데이터 형식 지원"
elif [ "$MODE" = "prompt" ]; then
    echo "# DeepSeek Coder 6.7B 프롬프트 생성 지속학습 시작 #"
    echo "##################################################"
    echo ""
    echo "학습 태그: prompt_generation (지속학습)"
    echo "학습 모드: [2차] 일반 프롬프트 기반 코드 생성 학습"
    echo "저장 경로: ../models/prompt-finetuned"
    echo "지원 기능: 프롬프트 기반 코드 생성 및 명령 이행 기능 강화"
elif [ "$MODE" = "comment" ]; then
    echo "# DeepSeek Coder 6.7B 주석 기반 생성 지속학습 시작 #"
    echo "##################################################"
    echo ""
    echo "학습 태그: comment_generation (지속학습)"
    echo "학습 모드: [3차] 주석 기반 코드 생성 학습"
    echo "저장 경로: ../models/comment-finetuned"
    echo "지원 기능: 주석 및 문서화 기반 코드 자동 생성 강화"
elif [ "$MODE" = "error_fix" ]; then
    echo "# DeepSeek Coder 6.7B 오류 수정 지속학습 시작   #"
    echo "##################################################"
    echo ""
    echo "학습 태그: error_correction (지속학습)"
    echo "학습 모드: [4차] 오류 코드 설명 및 수정 학습"
    echo "저장 경로: ../models/error-fix-finetuned"
    echo "지원 기능: 코드 디버깅, 오류 설명, 자동 코드 수정 기능 강화"
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

# AWS 스팟 인스턴스 중단 감지
echo "AWS 스팟 인스턴스 중단 감지 도구 독립 실행 시작..."
if [ -f "test_environment.py" ]; then
    $PYTHON_CMD test_environment.py > ../logs/spot_monitor_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    SPOT_MONITOR_PID=$!
    echo "✅ 스팟 인스턴스 중단 감지 실행 완료 (PID: $SPOT_MONITOR_PID)"
else
    echo "⚠️ test_environment.py 가 존재하지 않습니다!"
fi

# 학습 모드별 안내
if [ "$MODE" = "complete" ]; then
    echo "==================================================="
    echo "자동완성(FIM 형식 포함) 모델 지속학습 시작"
    echo "==================================================="
elif [ "$MODE" = "prompt" ]; then
    echo "==================================================="
    echo "프롬프트 모델 지속학습 시작"
    echo "==================================================="
elif [ "$MODE" = "comment" ]; then
    echo "==================================================="
    echo "주석 모델 지속학습 시작"
    echo "==================================================="
else
    echo "==================================================="
    echo "오류 수정 모델 지속학습 시작"
    echo "==================================================="
fi

# 데이터 경로 확인
DATA_PATH="/home/ubuntu/deepseek-continual/data/train.jsonl"
echo "데이터 경로: $DATA_PATH"

# config.yaml 파일 확인 및 처리
if [ ! -f "config.yaml" ]; then
    echo "⚠️ config.yaml이 없습니다. 기본 설정으로 생성합니다..."
    cat > config.yaml << EOF
# DeepSeek-Coder 6.7B Instruct T4 환경 최적화 설정
model_name: "/home/ubuntu/deepseek-coder/model_cache/deepseek-coder-6.7b-instruct"  # instruct 모델 사용
data_path: "/home/ubuntu/deepseek-continual/data/train.jsonl"

# T4 최적화 학습 하이퍼파라미터
learning_rate: 2e-4
batch_size: 4  # 더 효율적인 학습을 위해 배치 크기 증가
grad_acc: 2  # 배치 크기가 증가함에 따라 그래디언트 어큐문레이션을 조정 (총 교육량 유지)
max_length: 1024
num_epochs: 3
lr_scheduler: "cosine"
warmup_ratio: 0.1
weight_decay: 0.01

# T4 호환 최적화 설정
fp16: true
bf16: false
gradient_checkpointing: true
torch_dtype: "float16"

# 양자화 설정 (T4 호환)
load_in_4bit: true
bnb_4bit_compute_dtype: "float16"
bnb_4bit_quant_type: "nf4"

# 출력 및 저장 설정
output_dir: "/home/ubuntu/deepseek-continual/scripts/checkpoints/prompt-finetuned/"
save_steps: 10    # I/O 부하 및 tqdm 진행바 문제 고려
eval_steps: 50    # 평가 빈도 최적화
logging_steps: 5  # 로그 출력 빈도
save_total_limit: 3
evaluation_strategy: "steps"
save_strategy: "steps"
load_best_model_at_end: true
metric_for_best_model: "eval_loss"

# 추가 설정
dataloader_num_workers: 4
remove_unused_columns: false

# 저장 설정
save_steps: 500
eval_steps: 500
logging_steps: 50
save_total_limit: 3

# 데이터 설정
data_path: "$DATA_PATH"

# wandb 동기화 비활성화
report_to: []
wandb: false
EOF
    echo "✅ config.yaml 생성 완료"
else
    echo "🔧 기존 config.yaml 파일을 사용하되, wandb 설정을 도구를 위해 비활성화합니다..."
    
    # config.yaml에 wandb 설정 추가/수정
    if grep -q "wandb" config.yaml; then
        # wandb 관련 설정이 이미 있는 경우 수정
        sed -i '' '/wandb/d' config.yaml
        echo "wandb: false" >> config.yaml
        echo "report_to: []" >> config.yaml
    else
        # wandb 관련 설정이 없는 경우 추가
        echo "# wandb 동기화 비활성화" >> config.yaml
        echo "wandb: false" >> config.yaml
        echo "report_to: []" >> config.yaml
    fi
    
    # data_path 없으면 추가
    if ! grep -q "data_path" config.yaml; then
        echo "# 데이터 경로" >> config.yaml
        echo "data_path: \"$DATA_PATH\"" >> config.yaml
    fi
    
    echo "✅ config.yaml 업데이트 완료"
fi

# 모델 학습 실행 함수
run_training() {
    local log_suffix
    
    if [ "$MODE" = "complete" ]; then
        echo "=== [1차 지속학습] 코드 자동완성 학습 시작 (FIM 형식 지원) ($(date)) ==="
        log_suffix="autocomplete"
    elif [ "$MODE" = "error_fix" ]; then
        echo "=== [4차 지속학습] 코드 오류 설명 및 수정 학습 시작 (지속학습) ($(date)) ==="
        log_suffix="errorfix"
    elif [ "$MODE" = "comment" ]; then
        echo "=== [3차 지속학습] 주석 기반 코드 생성 학습 시작 (지속학습) ($(date)) ==="
        log_suffix="comment"
    else
        echo "=== [2차 지속학습] 일반 프롬프트 학습 시작 (지속학습) ($(date)) ==="
        log_suffix="prompt"
    fi
    
    # CUDA 메모리 최적화 환경변수 설정
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
    echo "CUDA 메모리 단편화 방지 설정 적용"
    
    # 학습 준비
    echo "학습 준비 완료"
    
    # 체크포인트 경로가 있으면 재개 인자 추가
    if [ ! -z "$RESUME_CHECKPOINT" ]; then
        echo "🔥 체크포인트에서 학습 재개: $RESUME_CHECKPOINT"
        $PYTHON_CMD cc_train.py --config config.yaml --mode $MODE --resume_from_checkpoint "$RESUME_CHECKPOINT" 2>&1 | tee ../logs/training_${log_suffix}_resume_$(date +%Y%m%d_%H%M%S).log
    else
        # 체크포인트가 없으면 처음부터 학습
        $PYTHON_CMD cc_train.py --config config.yaml --mode $MODE 2>&1 | tee ../logs/training_${log_suffix}_$(date +%Y%m%d_%H%M%S).log
    fi
    
    TRAINING_EXIT_CODE=$?
    if [ $TRAINING_EXIT_CODE -ne 0 ]; then
        echo "⚠️ 학습이 중단되었습니다 (exit code: $TRAINING_EXIT_CODE). 10초 후 재시도..."
        sleep 10
        return 1
    else
        echo "✅ 학습이 성공적으로 완료되었습니다!"
        return 0
    fi
}

# 학습 실행
echo "🚀 선택한 모드($MODE)로 지속학습을 시작합니다..."
run_training

# 완료 메시지
if [ "$MODE" = "complete" ]; then
    echo "=================================================="
    echo "자동완성(FIM 형식 포함) 모델 학습 완료"
    echo "완료 시간: $(date)"
    echo "=================================================="
    echo "모델 저장 위치: ../models/autocomplete-finetuned/final_model"
    echo "inference.py에서 이 모델을 사용하려면: python3 inference.py --mode autocomplete --model ../models/autocomplete-finetuned/final_model"
    echo "FIM 형식으로 추론하려면 적절한 입력 형식을 사용하세요."
elif [ "$MODE" = "error_fix" ]; then
    echo "=================================================="
    echo "코드 오류 설명 및 수정 모델 학습 완료"
    echo "완료 시간: $(date)"
    echo "=================================================="
    echo "모델 저장 위치: ../models/error-fix-finetuned/final_model"
    echo "inference.py에서 이 모델을 사용하려면: python3 inference.py --mode error_fix --model ../models/error-fix-finetuned/final_model"
elif [ "$MODE" = "comment" ]; then
    echo "=================================================="
    echo "주석 기반 코드 생성 모델 학습 완료"
    echo "완료 시간: $(date)"
    echo "=================================================="
    echo "모델 저장 위치: ../models/comment-finetuned/final_model"
    echo "inference.py에서 이 모델을 사용하려면: python3 inference.py --mode comment --model ../models/comment-finetuned/final_model"
else
    echo "=================================================="
    echo "일반 프롬프트 학습 완료"
    echo "완료 시간: $(date)"
    echo "=================================================="
    echo "모델 저장 위치: ../models/prompt-finetuned/final_model"
    echo "inference.py에서 이 모델을 사용하려면: python3 inference.py --mode prompt --model ../models/prompt-finetuned/final_model"
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