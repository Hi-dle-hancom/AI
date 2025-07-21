#!/bin/bash

# DeepSeek 훈련 환경 설정 스크립트

set -e

echo "====================================================="
echo "DeepSeek 훈련 환경 설정 시작"
echo "====================================================="

# 작업 디렉토리 확인
WORK_DIR="$(pwd)"
echo "작업 디렉토리: $WORK_DIR"

# Python 설치 확인 및 설치
echo "1. Python 설치 확인..."
if ! command -v python3 &> /dev/null; then
    echo "Python3가 설치되어 있지 않습니다. 설치를 시작합니다..."
    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv
    echo "✅ Python3 설치 완료"
else
    echo "✅ Python3가 이미 설치되어 있습니다: $(python3 --version)"
fi

# pip 업그레이드
echo "2. pip 업그레이드..."
python3 -m pip install --upgrade pip

# 가상 환경 생성 및 활성화
echo "3. 가상 환경(Hidle) 생성..."
if [ ! -d "Hidle" ]; then
    echo "가상 환경을 생성합니다..."
    python3 -m venv Hidle
    echo "✅ 가상 환경 생성 완료"
else
    echo "✅ 가상 환경이 이미 존재합니다"
fi

echo "4. 가상 환경 활성화..."
source Hidle/bin/activate
echo "✅ 가상 환경 활성화됨: $(which python)"

# 의존성 설치
echo "5. 필요한 패키지 설치..."
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    echo "requirements.txt에서 패키지 설치 중..."
    pip install -r requirements.txt
    echo "✅ 패키지 설치 완료"
else
    echo "⚠️ requirements.txt 파일이 존재하지 않습니다. 지정된 버전의 기본 패키지를 설치합니다."
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 transformers==4.31.0 datasets==2.12.0 accelerate==0.21.0 peft==0.4.0 trl==0.7.1 sentencepiece==0.1.97 protobuf==3.20.0 bitsandbytes==0.41.0 scipy==1.10.0 wandb==0.15.5
    echo "✅ 기본 패키지 설치 완료"
fi

# CUDA 설치 확인
echo "6. CUDA 설치 확인..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU 감지됨:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | cut -c2- || echo "미확인")
    echo "CUDA 버전: $CUDA_VERSION"
else
    echo "⚠️ nvidia-smi를 찾을 수 없습니다. CPU 모드로 실행됩니다."
fi

# 올바른 PyTorch 버전 확인
echo "7. PyTorch GPU 지원 확인..."
python -c "import torch; print('CUDA 사용 가능:', torch.cuda.is_available(), '/ 장치 수:', torch.cuda.device_count(), '/ 현재 장치:', torch.cuda.current_device(), '/ 장치 이름:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# 데이터 디렉토리 확인
echo "8. 데이터 디렉토리 확인..."
if [ ! -d "data" ]; then
    echo "⚠️ 'data' 디렉토리가 존재하지 않습니다. 데이터셋 파일을 저장할 디렉토리를 생성합니다."
    mkdir -p data
    echo "✅ 'data' 디렉토리 생성 완료"
fi

# 출력 디렉토리 확인
if [ ! -d "output" ]; then
    echo "⚠️ 'output' 디렉토리가 존재하지 않습니다. 모델 체크포인트를 저장할 디렉토리를 생성합니다."
    mkdir -p output
    echo "✅ 'output' 디렉토리 생성 완료"
fi

# 전체 패키지 설치 확인
echo "9. 패키지 설치 확인..."
python -c "
import sys
import torch
import transformers
import datasets
import accelerate
import peft
try:
    import bitsandbytes
    bnb_available = True
except ImportError:
    bnb_available = False

print(f'✅ Python: {sys.version.split()[0]}')
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ Datasets: {datasets.__version__}')
print(f'✅ Accelerate: {accelerate.__version__}')
print(f'✅ PEFT: {peft.__version__}')
print(f'✅ BitsAndBytes: {\"설치됨\" if bnb_available else \"설치되지 않음\"}')
print(f'✅ CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA 디바이스: {torch.cuda.get_device_name(0)}')
    print(f'✅ 가용 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')
"

# AWS 스팟 인스턴스 중단 감지 설정 확인
echo "10. AWS 스팟 인스턴스 환경 확인..."
if [ -f "test_environment.py" ]; then
    echo "✅ 스팟 인스턴스 중단 감지 스크립트가 존재합니다."
else
    echo "⚠️ 스팟 인스턴스 중단 감지 스크립트가 없습니다."
    # 스크립트 생성
    cat > test_environment.py << 'EOF'
#!/usr/bin/env python3
"""
AWS 스팟 인스턴스 중단 감지 스크립트
"""

import os
import time
import signal
import threading
import requests
import logging
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

def check_spot_termination():
    """스팟 인스턴스 중단 신호를 확인하고 학습 프로세스에 SIGTERM 신호를 보냔"""
    try:
        # 스팟 인스턴스 중단 신호 업데이트 확인
        url = "http://169.254.169.254/latest/meta-data/spot/instance-action"
        response = requests.get(url, timeout=2)

        if response.status_code == 200:
            # 중단 신호가 있음 - train.py 프로세스에 SIGTERM 전송
            log.warning("스팟 인스턴스 중단 신호 감지: %s", response.text)

            # train.py 프로세스 시그널 전송
            result = subprocess.run(["pgrep", "-f", "train\.py"], capture_output=True, text=True)
            if result.returncode == 0:
                pid = int(result.stdout.strip())
                log.info("train.py 프로세스 (PID: %s)에 SIGTERM 신호 전송", pid)
                os.kill(pid, signal.SIGTERM)
                return True
    except requests.RequestException:
        # EC2 메타데이터 API에 접근할 수 없음 - 일반 인스턴스일 수 있음
        pass
    except Exception as e:
        log.error("스팟 인스턴스 중단 신호 확인 중 오류: %s", e)

    return False

def monitoring_thread_func():
    """지속적으로 스팟 인스턴스 중단을 감시하는 백그라운드 스레드"""
    log.info("스팟 인스턴스 중단 감지 도구 시작")

    while True:
        if check_spot_termination():
            log.warning("스팟 인스턴스 중단 감지 - 학습 중단 및 체크포인트 저장 프로세스 시작")
            break

        # 5초마다 확인
        time.sleep(5)

if __name__ == "__main__":
    # 백그라운드 스레드로 스팟 인스턴스 중단 감시 시작
    monitoring_thread = threading.Thread(target=monitoring_thread_func, daemon=True)
    monitoring_thread.start()

    log.info("메인 프로세스 실행 중 (중단하려면 Ctrl+C 를 누르세요)")
    try:
        # 메인 스레드는 계속 실행
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        log.info("사용자에 의해 중지됨")

    log.info("스팟 인스턴스 중단 감지 도구 종료")
EOF
    chmod +x test_environment.py
    echo "✅ 스팟 인스턴스 중단 감지 스크립트 생성 완료"
fi

echo "====================================================="
echo "환경 설정 완료!"
echo "====================================================="
echo "이제 다음 명령어로 훈련을 시작할 수 있습니다:"
echo "./run_training.sh"
echo ""
echo "AWS 스팟 인스턴스 중단 감지 스크립트를 백그라운드로 실행하려면:"
echo "python test_environment.py &"
echo ""
echo "가상 환경을 사용하는 경우 먼저 활성화하세요:"
echo "source Hidle/bin/activate"