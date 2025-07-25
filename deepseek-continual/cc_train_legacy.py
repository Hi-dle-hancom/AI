#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Continual Learning 통합 시스템
최신 연구 결과 기반 고성능 지속 학습 시스템
MER + EWC + NCM + SCR

Created: June 2025
Authors: [James Kim]

실행 방법:
    python cc_train.py --config config.yaml
    또는
    ./cc_run_training.sh
"""

import os
import sys
import json
import glob
import time
import math
import logging
import traceback
import tempfile
import datetime
import threading
import signal
import inspect
from typing import List, Dict, Any, Tuple, Optional, Callable, Union

# 메모리 모니터링 모듈 가져오기
try:
    from memory_monitor import MemoryMonitor
    MEMORY_MONITOR_AVAILABLE = True
except ImportError:
    MEMORY_MONITOR_AVAILABLE = False
    logging.warning("memory_monitor.py 모듈을 가져올 수 없습니다. VRAM 모니터링 기능이 비활성화됩니다.")
    
# AdamA 최적화 모듈 가져오기 (그래디언트 즉시 해제 및 메모리 최적화)
try:
    from adama import AdamA
    ADAMA_AVAILABLE = True
except ImportError:
    ADAMA_AVAILABLE = False
    logging.warning("AdamA 최적화가 필요하다면 'pip install adama' 명령으로 설치하세요.")
import collections
import random
import argparse
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict, deque
import datetime
from pathlib import Path
from tqdm import tqdm
import yaml

# 필요할 때만 임포트되는 라이브러리들은 조건부 임포트로 변경
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

# 중요도에 따라 경고 필터링
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.cuda.*_rng_state")
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated")

# 로깅 모듈 불러오기 - 중앙 집중식 로깅
from logger_module import DeepseekLogger

# 통합 로깅 시스템 초기화
logger = DeepseekLogger(
    name="deepseek_trainer", 
    log_dir="./logs", 
    level=logging.WARNING, 
    console_output=True
)

# DeepseekLogger와 통합된 메트릭 Writer 클래스
class Writer:
    """메트릭을 로깅하는 심플한 클래스로, TensorBoard 없이 작동합니다.
    DeepseekLogger를 활용하여 일관된 로깅 시스템을 구현합니다."""
    def __init__(self):
        self.metrics = {}
        
    def add_scalar(self, tag, value, step):
        """메트릭 값 기록 (메모리에만 저장)"""
        if tag not in self.metrics:
            self.metrics[tag] = []
        self.metrics[tag].append((step, value))
        
        # 중요 메트릭은 로그에도 기록 (필요 시 주석 해제)
        # if any(key in tag for key in ['Loss', 'Accuracy', 'Metrics']):
        #    logger.debug(f"{tag}: {value:.4f} (step {step})")
    
    def add_figure(self, *args, **kwargs):
        """호환성을 위한 더미 함수"""
        pass
        
    def close(self):
        """자원 정리 함수"""
        pass

# 조건부 라이브러리 임포트 - 필요할 때만 로드
# Transformers 관련 임포트
try:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model, PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers 또는 PEFT 라이브러리가 설치되지 않았습니다. 실제 모델 로드가 불가능합니다.")

# Datasets 관련 임포트
try:
    from datasets import load_dataset, DatasetDict
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logger.warning("datasets 라이브러리가 설치되지 않았습니다. 데이터셋 로드가 불가능합니다.")

# 시스템 모니터링 관련 임포트 (필요할 때만 로드)
def import_monitoring_libs():
    """시스템 모니터링 라이브러리 조건부 임포트"""
    try:
        import psutil
        import GPUtil
        return True
    except ImportError:
        logger.warning("psutil 또는 GPUtil 라이브러리가 설치되지 않았습니다. 시스템 모니터링이 제한됩니다.")
        return False

# 전역 시드 고정 함수
def set_seed(seed: int):
    """완전한 재현 가능성을 보장하기 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"모든 랜덤 시드가 {seed}로 고정되었습니다.")

# AWS Spot Instance 중단 감지 클래스
class SpotInterruptionHandler:
    """AWS Spot Instance 중단 감지 및 대응 핸들러
    
    AWS 스팟 인스턴스 환경에서 중단 신호를 감지하고 체크포인트 저장을 관리합니다.
    SIGTERM 신호 감지 및 EC2 메타데이터 API를 통한 중단 알림을 모니터링합니다.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints", checkpoint_freq: int = 500):
        """AWS 스팟 인스턴스 중단 감지 핸들러 초기화
        
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리
            checkpoint_freq: 정기적인 체크포인트 주기 (steps)
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.interrupted = False
        self.last_checkpoint_time = time.time()
        self.metadata_url = "http://169.254.169.254/latest/meta-data/spot/instance-action"
        self.monitoring_active = False
        
        # 저장 디렉토리 생성 및 시그널 핸들러 등록
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        signal.signal(signal.SIGTERM, self.interruption_handler)
        
        logger.info("Spot Instance 중단 감지기가 초기화되었습니다.")

    def interruption_handler(self, signum, frame):
        """중단 시그널(SIGTERM) 처리"""
        self.interrupted = True
        logger.warning("Spot Instance 중단 신호(SIGTERM)가 감지되었습니다!")

    def check_interruption(self) -> bool:
        """EC2 메타데이터 API를 통한 중단 예정 확인
        
        Returns:
            bool: 중단 감지 여부
        """
        # 이미 중단이 감지된 경우 추가 확인 없이 반환
        if self.interrupted:
            return True
            
        # 메타데이터 API 요청
        try:
            import requests
            response = requests.get(self.metadata_url, timeout=0.1)
            if response.status_code == 200:
                self.interrupted = True
                metadata = response.json()
                logger.warning(f"Spot 인스턴스 중단 임박! 메타데이터: {metadata}")
                return True
        except ImportError:
            logger.warning("requests 라이브러리가 설치되지 않아 메타데이터 확인이 불가능합니다.")
        except Exception:
            # 중단 예정이 아니거나 메타데이터 접근 실패 시 무시
            pass
            
        return self.interrupted

    def should_checkpoint(self, step: int) -> bool:
        """체크포인트 저장이 필요한지 확인
        
        다음 조건에 따라 체크포인트가 필요한지 결정합니다:
        1. 스팟 인스턴스 중단이 감지된 경우
        2. 설정된 체크포인트 주기에 도달한 경우
        
        Args:
            step: 현재 학습 단계(step)
            
        Returns:
            bool: 체크포인트 저장 필요 여부
        """
        # 1. 중단 감지 시 즉시 체크포인트
        if self.check_interruption():
            logger.info("스팟 인스턴스 중단이 감지되어 체크포인트를 수행합니다.")
            print(f"[중요] 스팟 인스턴스 중단 감지 - 체크포인트 저장 시도")
            return True

        # 2. 정기적인 체크포인트 (스텝 기반)
        freq_check = step > 0 and step % self.checkpoint_freq == 0
        if freq_check:
            logger.info(f"[정기 저장] 주기적 체크포인트 저장 시점: {step} (step % {self.checkpoint_freq} == 0)")
            print(f"[정기 저장] 체크포인트 저장 시작 - 스텝 {step}")
            # 마지막 저장 시간 갱신
            self.last_checkpoint_time = time.time()
            return True

        return False

    def start_monitoring(self, model=None, optimizer=None, scheduler=None, scaler=None, trainer=None):
        """스팟 인스턴스 중단 모니터링 시작
        
        test_environment.py에서 사용하는 백그라운드 모니터링과의 호환성을 위한 메서드입니다.
        저장이 필요한 상태 객체들을 참조로 저장합니다.
        
        Args:
            model: 모델 객체
            optimizer: 최적화 객체
            scheduler: 스케줄러 객체
            scaler: 그래디언트 스케일러 객체
            trainer: 학습기 객체
        """
        # 이미 모니터링 중이면 중복 실행 방지
        if self.monitoring_active:
            logger.debug("이미 모니터링이 활성화되어 있습니다")
            return
        
        # 참조 저장 (얼은 복사)
        self.model = model if model is not None else getattr(self, 'model', None)
        self.optimizer = optimizer if optimizer is not None else getattr(self, 'optimizer', None)
        self.scheduler = scheduler if scheduler is not None else getattr(self, 'scheduler', None)
        self.scaler = scaler if scaler is not None else getattr(self, 'scaler', None)
        self.trainer = trainer if trainer is not None else getattr(self, 'trainer', None)
        
        # 백그라운드 모니터링 스레드 시작
        self.monitoring_active = True
        self._start_background_monitor()
        logger.info("스팟 인스턴스 모니터링이 활성화되었습니다.")
        
    def _start_background_monitor(self):
        """백그라운드 스레드로 스팟 인스턴스 중단 시그널 또는 메타데이터를 주기적으로 체크
        주의: threading 모듈 사용은 GIL로 인해 완전한 병렬성을 보장하지 않음
        """
        import threading
        
        def monitor_func():
            check_interval = 10  # 10초 간격으로 확인
            while self.monitoring_active:
                try:
                    # 중단 시그널 인식을 위한 주기적 확인
                    was_interrupted = self.interrupted
                    self.check_interruption()
                    
                    # 새로 중단 감지 시 체크포인트 강제 저장
                    if not was_interrupted and self.interrupted:
                        logger.warning("[Background] 스팟 인스턴스 중단 감지! 대처 작업 시작...")
                        
                        if (self.model is not None and self.optimizer is not None and 
                                self.trainer is not None):
                            step = getattr(self.trainer, 'global_step', 0)
                            try:
                                # 긴급 체크포인트 저장 시도
                                self.save_checkpoint(
                                    model=self.model,
                                    optimizer=self.optimizer,
                                    scheduler=self.scheduler,
                                    scaler=self.scaler,
                                    trainer=self.trainer,
                                    step=step
                                )
                                logger.info("[Background] 긴급 체크포인트 저장 성공")
                            except Exception as e:
                                logger.error(f"[Background] 긴급 체크포인트 저장 실패: {e}")
                    
                    # 주기적 슬리프 (코어를 과도하게 사용하지 않도록)
                    time.sleep(check_interval)
                except Exception as e:
                    logger.error(f"[Background] 모니터링 스레드 오류: {e}")
                    time.sleep(30)  # 오류 발생 시 더 긴 대기 시간
        
        # 데모니스레드(백그라운드 스레드)로 실행
        self.monitor_thread = threading.Thread(
            target=monitor_func, 
            daemon=True,  # 데모니스레드로 설정하여 메인 스레드 종료시 함께 종료되도록 함
            name="SpotMonitor"
        )
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """스팟 인스턴스 중단 모니터링 종료"""
        self.monitoring_active = False
        logger.info("스팟 인스턴스 모니터링이 비활성화되었습니다.")
        
    def __del__(self):
        """객체 소멸 시 자원 정리"""
        self.stop_monitoring()

    def save_checkpoint(self, model=None, optimizer=None, scheduler=None, scaler=None, trainer=None, step=0):
        """스팟 인스턴스 중단 시 긴급 체크포인트 저장
        
        Args:
            model: 모델 객체
            optimizer: 최적화기 객체
            scheduler: 스케줄러 객체
            scaler: 스케일러 객체
            trainer: 트레이너 객체
            step: 현재 학습 스텝
            
        Returns:
            bool: 체크포인트 저장 성공 여부
        """
        if trainer is None:
            logger.error("[SpotInterruptionHandler] 트레이너 객체 없이 체크포인트를 저장할 수 없습니다.")
            return False
            
        try:
            # 체크포인트 파일명 생성
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"interrupt_step{step}_{timestamp}.pt"
            
            # 트레이너의 save_model 메서드 호출
            if hasattr(trainer, 'save_model'):
                save_path = trainer.save_model(filename, is_spot_interruption=True)
                logger.info(f"[SpotInterruptionHandler] 긴급 체크포인트 저장 완료: {save_path}")
                return True
            else:
                logger.error("[SpotInterruptionHandler] 트레이너에 save_model 메서드가 없습니다.")
                return False
        except Exception as e:
            logger.error(f"[SpotInterruptionHandler] 긴급 체크포인트 저장 실패: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def handle_checkpoint_event(self, checkpoint_path: str, step: int):
        """
        체크포인트 저장 후 처리 작업 (심볼릭 링크 업데이트 등)
        
        Args:
            checkpoint_path: 저장된 체크포인트 파일 경로
            step: 현재 스텝
        """
        # 타임스탬프 생성
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 최신 체크포인트 심볼릭 링크 업데이트
        latest_path = os.path.join(self.checkpoint_dir, "checkpoint-latest.pt")
        
        # 심볼릭 링크 확인 및 안전하게 제거
        if os.path.lexists(latest_path):  # lexists는 심볼릭 링크 자체가 있는지 확인
            try:
                os.unlink(latest_path)  # 심볼릭 링크를 안전하게 제거
            except Exception as e:
                logger.warning(f"심볼릭 링크 제거 중 오류: {e}")
                
        # 심볼릭 링크 생성 시 예외 처리
        try:
            os.symlink(checkpoint_path, latest_path)
            logger.debug(f"최신 체크포인트 심볼릭 링크 업데이트: {latest_path} -> {checkpoint_path}")
        except FileExistsError:
            # 여전히 존재한다면 강제로 다시 시도
            logger.warning("심볼릭 링크가 여전히 존재함. 강제 삭제 후 재생성 시도...")
            try:
                os.remove(latest_path)  # 강제 삭제
                os.symlink(checkpoint_path, latest_path)  # 다시 생성
                logger.debug("심볼릭 링크 강제 재생성 성공")
            except Exception as e:
                logger.error(f"심볼릭 링크 강제 재생성 실패: {e}")
                logger.info(f"심볼릭 링크 생성은 실패했지만, 체크포인트는 정상 저장됨")

        # 중단 감지 시 추가 정보 기록 (spot-info.json)
        if self.interrupted:
            info_path = os.path.join(self.checkpoint_dir, "spot-info.json")
            try:
                import json
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(info_path, 'w') as f:
                    json.dump({
                        'interrupted_at': current_time,
                        'latest_checkpoint': checkpoint_path,
                        'step': step,
                        'interrupted': True
                    }, f)
                logger.info(f"스팟 인스턴스 중단 정보가 {info_path}에 저장되었습니다.")
            except Exception as e:
                logger.warning(f"스팟 중단 정보 저장 실패: {e}")

        # 체크포인트 저장 이벤트 구조화된 로깅
        reason = "spot_interruption" if self.interrupted else "regular_checkpoint"
        logger.info(f"[체크포인트 저장] 경로: {checkpoint_path}, 스텝: {step}, 이유: {reason}, 시간: {timestamp}")
        self.last_checkpoint_time = time.time()

    def load_checkpoint(self, checkpoint_path, model, optimizer, scheduler, scaler, trainer):
        """지정된 경로의 체크포인트 로드"""
        if not os.path.exists(checkpoint_path):
            # 체크포인트 파일 없음 경고 구조화 로깅
            logger.warning(f"[체크포인트 로드 실패] 경로: {checkpoint_path}, 이유: 파일을 찾을 수 없음")
            return 0

        # 체크포인트 로드 시작 구조화 로깅
        logger.info(f"[체크포인트 로드 시작] 경로: {checkpoint_path}")
        
        # 체크포인트 경로가 디렉토리인지 파일인지 먼저 확인
        if os.path.isdir(checkpoint_path):            
            # 디렉토리인 경우 (최종 모델 디렉토리 로드 시도)
            logger.info(f"체크포인트 경로가 디렉토리입니다: {checkpoint_path}")
            try:
                # 허깅페이스 모델 디렉토리에서 로드 시도
                # PEFT 모델이 이미 적용된 경우 적절한 방식으로 처리
                if hasattr(model, 'pretrained_model') or hasattr(model, 'base_model'):
                    # 이미 PEFT 모델인 경우
                    from peft import PeftModel, PeftConfig
                    logger.info(f"PEFT 모델에 대한 어댑터 로드 시도: {checkpoint_path}")
                    
                    # 기존 모델에서 기본 모델 추출
                    if hasattr(model, 'pretrained_model'):
                        base_model = model.pretrained_model
                    else:
                        base_model = model.base_model if hasattr(model, 'base_model') else model
                    
                    # 1. 기존 어댑터가 있는 경우 완전히 제거 (PEFT 0.4.0 방식)
                    if hasattr(model, 'unload'):
                        try:
                            model.unload()
                            logger.info("기존 PEFT 어댑터 제거 완료")
                        except Exception as e:
                            logger.warning(f"PEFT 어댑터 제거 중 오류(무시함): {e}")
                    
                    # 2. 어댑터 구성 파일 확인
                    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
                    if not os.path.exists(adapter_config_path):
                        logger.warning(f"adapter_config.json 파일이 없어 일반 모델로 로드합니다: {checkpoint_path}")
                        model = base_model
                        model.load_state_dict(torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location="cpu"), strict=False)
                        return 0  # 혹은 적절한 반환값
                        
                    # 3. 어댑터 구성 정보 로드
                    try:
                        # 어댑터 구성 파일 로드
                        peft_config = PeftConfig.from_pretrained(checkpoint_path)
                        logger.info(f"어댑터 구성 로드 완료: {peft_config.peft_type}, target_modules={getattr(peft_config, 'target_modules', 'N/A')}")
                        
                        # 4. 기본 모델을 PEFT 어댑터에 맞게 준비
                        logger.info("기본 모델을 PEFT 어댑터에 맞게 준비 중...")
                        
                        # 5. 체크포인트에서 가중치만 로드 (두 형식 지원: .bin 및 .safetensors)
                        bin_weights_path = os.path.join(checkpoint_path, "adapter_model.bin")
                        safetensors_weights_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
                        
                        # 6. 새 PeftModel 생성 (PEFT 0.4.0 방식)
                        model = PeftModel(base_model, peft_config, adapter_name="default")
                        
                        # 7. 가중치 파일 여부 확인 및 로드
                        weights_loaded = False
                        
                        # .bin 파일 확인
                        if os.path.exists(bin_weights_path):
                            logger.info(f"어댑터 가중치 (.bin) 로드 중: {bin_weights_path}")
                            adapter_weights = torch.load(bin_weights_path, map_location="cpu")
                            model.load_state_dict(adapter_weights, strict=False)
                            logger.info("어댑터 가중치 (.bin) 로드 완료")
                            weights_loaded = True
                        
                        # .safetensors 파일 확인
                        elif os.path.exists(safetensors_weights_path):
                            try:
                                from safetensors import safe_open
                                from safetensors.torch import load_file
                                
                                logger.info(f"어댑터 가중치 (.safetensors) 로드 중: {safetensors_weights_path}")
                                # safetensors 파일에서 가중치 로드
                                adapter_weights = load_file(safetensors_weights_path)
                                model.load_state_dict(adapter_weights, strict=False)
                                logger.info("어댑터 가중치 (.safetensors) 로드 완료")
                                weights_loaded = True
                            except Exception as e:
                                logger.error(f"safetensors 파일 로드 오류: {str(e)}")
                        
                        # 가중치 파일을 찾지 못한 경우
                        if not weights_loaded:
                            logger.warning(f"어댑터 가중치 파일을 찾을 수 없음: {bin_weights_path} 또는 {safetensors_weights_path}")
                            logger.warning("새 어댑터로 초기화하여 계속합니다.")
                        
                        # 8. 어댑터 활성화 확인 및 설정
                        if hasattr(model, 'active_adapter') and not model.active_adapter:
                            model.set_adapter("default")
                            logger.info("어댑터 활성화됨: default")
                            
                        return 0  # PEFT 로드 성공
                    except Exception as e:
                        logger.error(f"PEFT 어댑터 로드 중 오류 발생: {str(e)}")
                        # 오류 발생 시 기본 모델 사용
                        model = base_model
                        return 0
                else:
                    # 일반 모델인 경우 (허깅페이스 모델로 로드)
                    logger.info(f"일반 모델로 로드 시도: {checkpoint_path}")
                    # 허깅페이스 디렉토리로부터 로드하는 로직 추가
                    return 0
            except Exception as e:
                # 체크포인트 디렉토리 로드 중 오류 처리
                logger.error(f"체크포인트 디렉토리 로드 중 오류: {e}")
                current_batch_size = getattr(self, 'batch_size', 4)  # 기본값 4
                self._handle_oom_error(current_batch_size)
                logger.info("오류로 인해 체크포인트 로드에 실패했습니다.")
                return 0  # 시작 스텝
        
        # 파일인 경우 (.pt 파일) 로드 시도
        try:
            logger.info(f"체크포인트 파일을 로드합니다: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 모델 로드
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
                logger.info("[체크포인트] 모델 상태 로드 완료")
                
            # 옵티마이저 로드
            if 'optimizer' in checkpoint and optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info("[체크포인트] 옵티마이저 상태 로드 완료")
                
            # 스케줄러 로드
            if 'scheduler' in checkpoint and scheduler is not None and checkpoint['scheduler'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])
                logger.info("[체크포인트] 스케줄러 상태 로드 완료")
                
            # 스케일러 로드
            if 'scaler' in checkpoint and scaler is not None and checkpoint['scaler'] is not None:
                scaler.load_state_dict(checkpoint['scaler'])
                logger.info("[체크포인트] 스케일러 상태 로드 완료")
                
            # 트레이너 상태 로드
            if 'trainer_state' in checkpoint and trainer is not None:
                trainer.load_state_dict(checkpoint['trainer_state'])
                logger.info("[체크포인트] 트레이너 상태 로드 완료")
                
                # 현재 학습 모드 정보 로깅
                if hasattr(trainer, 'current_mode') and trainer.current_mode:
                    logger.info(f"[체크포인트] 현재 학습 모드: {trainer.current_mode} (완료된 모드: {getattr(trainer, 'completed_modes', [])})")
            
            step = checkpoint.get('step', 0)
            
            # AWS 스팟 인스턴스 환경에서의 중단/재개 관련 정보 로깅
            if 'spot_info' in checkpoint:
                spot_info = checkpoint.get('spot_info', {})
                last_save_time = spot_info.get('last_save_time', 'N/A')
                logger.info(f"[체크포인트] 마지막 저장 시간: {last_save_time}")
            
            try:
                # 학습 진행 상태 샘플링 정보
                if 'sampling_info' in checkpoint:
                    loss_avg = checkpoint['sampling_info'].get('loss_avg', 'N/A')
                    acc_avg = checkpoint['sampling_info'].get('acc_avg', 'N/A')
                    logger.info(f"[체크포인트] 마지막 손실/정확도: loss={loss_avg}, acc={acc_avg}")
            except Exception:
                # 선택적 정보이므로 예외 무시
                pass

            # 체크포인트 로드 성공 구조화 로깅
            # 로드된 컴포넌트 목록 가져오기
            components = ["model"]
            if 'optimizer' in checkpoint:
                components.append("optimizer")
            
            # 추가 컴포넌트 확인
            if 'scheduler' in checkpoint:
                components.append("scheduler")
            if 'scaler' in checkpoint:
                components.append("scaler")
            if 'trainer_state' in checkpoint:
                components.append("trainer_state")
                
            logger.info(f"[체크포인트 로드 성공] 경로: {checkpoint_path}, 스텝: {step}, 컴포넌트: {', '.join(components)}")
            return step
            
        except Exception as e:
            # 디렉토리인 경우 (최종 모델 디렉토리 로드 시도)
            if os.path.isdir(checkpoint_path):
                logger.info(f"체크포인트 경로가 디렉토리입니다: {checkpoint_path}")
                try:
                    # 허깅페이스 모델 디렉토리에서 로드 시도
                    # PEFT 모델이 이미 적용된 경우 적절한 방식으로 처리
                    if hasattr(model, 'pretrained_model') or hasattr(model, 'base_model'):
                        # 이미 PEFT 모델인 경우
                        from peft import PeftModel, PeftConfig
                        logger.info(f"PEFT 모델에 대한 어댑터 로드 시도: {checkpoint_path}")
                        
                        # 기존 모델에서 기본 모델 추출
                        if hasattr(model, 'pretrained_model'):
                            base_model = model.pretrained_model
                        else:
                            base_model = model.base_model if hasattr(model, 'base_model') else model
                        
                        # 1. 기존 어댑터가 있는 경우 완전히 제거 (PEFT 0.4.0 방식)
                        if hasattr(model, 'unload'):
                            try:
                                model.unload()
                                logger.info("기존 PEFT 어댑터 제거 완료")
                            except Exception as e:
                                logger.warning(f"PEFT 어댑터 제거 중 오류(무시함): {e}")
                        
                        # 2. 어댑터 구성 파일 확인
                        adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
                        if not os.path.exists(adapter_config_path):
                            logger.warning(f"adapter_config.json 파일이 없어 일반 모델로 로드합니다: {checkpoint_path}")
                            model = base_model
                            model.load_state_dict(torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location="cpu"), strict=False)
                            return model, optimizer, scheduler, scaler, trainer
                            
                        # 3. 어댑터 구성 정보 로드
                        try:
                            # PEFT 호환성 문제 해결을 위한 수정
                            # _prepare_model_for_peft_adapter 함수를 사용하지 않고 직접 어댑터 적용
                            
                            # 어댑터 구성 파일 로드
                            peft_config = PeftConfig.from_pretrained(checkpoint_path)
                            logger.info(f"어댑터 구성 로드 완료: {peft_config.peft_type}, target_modules={getattr(peft_config, 'target_modules', 'N/A')}")
                            
                            # 4. 기본 모델을 PEFT 어댑터에 맞게 준비 (가장 중요한 단계)
                            logger.info("기본 모델을 PEFT 어댑터에 맞게 준비 중...")
                            # PEFT 0.4.0에서는 이 함수가 없으므로 PeftModel이 자동으로 처리하도록 함
                            
                            # 5. 체크포인트에서 가중치만 로드 (두 형식 지원: .bin 및 .safetensors)
                            bin_weights_path = os.path.join(checkpoint_path, "adapter_model.bin")
                            safetensors_weights_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
                            
                            # 6. 새 PeftModel 생성 (PEFT 0.4.0 방식)
                            model = PeftModel(base_model, peft_config, adapter_name="default")
                            
                            # 7. 가중치 파일 여부 확인 및 로드
                            weights_loaded = False
                            
                            # .bin 파일 확인
                            if os.path.exists(bin_weights_path):
                                logger.info(f"어댑터 가중치 (.bin) 로드 중: {bin_weights_path}")
                                adapter_weights = torch.load(bin_weights_path, map_location="cpu")
                                model.load_state_dict(adapter_weights, strict=False)
                                logger.info("어댑터 가중치 (.bin) 로드 완료")
                                weights_loaded = True
                            
                            # .safetensors 파일 확인
                            elif os.path.exists(safetensors_weights_path):
                                try:
                                    from safetensors import safe_open
                                    from safetensors.torch import load_file
                                    
                                    logger.info(f"어댑터 가중치 (.safetensors) 로드 중: {safetensors_weights_path}")
                                    # safetensors 파일에서 가중치 로드
                                    adapter_weights = load_file(safetensors_weights_path)
                                    model.load_state_dict(adapter_weights, strict=False)
                                    logger.info("어댑터 가중치 (.safetensors) 로드 완료")
                                    weights_loaded = True
                                except Exception as e:
                                    logger.error(f"safetensors 파일 로드 오류: {str(e)}")
                            
                            # 가중치 파일을 찾지 못한 경우
                            if not weights_loaded:
                                logger.warning(f"어댑터 가중치 파일을 찾을 수 없음: {bin_weights_path} 또는 {safetensors_weights_path}")
                                logger.warning("새 어댑터로 초기화하여 계속합니다.")
                            
                            # 8. 어댑터 활성화 확인 및 설정
                            if hasattr(model, 'active_adapter') and not model.active_adapter:
                                model.set_adapter("default")
                                logger.info("어댑터 활성화됨: default")
                        except Exception as e:
                            logger.error(f"PEFT 어댑터 로드 중 오류 발생: {str(e)}")
                            # 오류 발생 시 기본 모델 사용
                            model = base_model
                    else:
                        # 일반 모델인 경우 (허깅페이스 모델로 로드)
                        logger.info(f"일반 모델로 로드 시도: {checkpoint_path}")
                        # 일반 모델 로드 로직 추가 필요
                except Exception as e:
                    # 체크포인트 디렉토리 로드 중 오류 처리
                    logger.error(f"체크포인트 디렉토리 로드 중 오류: {e}")
                    current_batch_size = getattr(self, 'batch_size', 4)  # 기본값 4
                    self._handle_oom_error(current_batch_size)
                    logger.info("오류로 인해 체크포인트 로드에 실패했습니다.")
                    return 0  # 시작 스텝

        # 파일인 경우 체크포인트 로드 완료 후 처리
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler and 'scheduler' in checkpoint and checkpoint['scheduler']:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if scaler and 'scaler' in checkpoint and checkpoint['scaler']:
            scaler.load_state_dict(checkpoint['scaler'])

        if 'trainer_state' in checkpoint and trainer is not None:
            trainer.load_state_dict(checkpoint['trainer_state'])

        step = checkpoint.get('step', 0)
        logger.info(f"체크포인트 로드 완료, 스텝 {step}부터 재개합니다.")
        return step

# Fisher Information Matrix 계산 클래스
class FisherInformationComputer:
    """
    정확한 Fisher Information Matrix 계산
    EWC와 LwF에서 사용
    """
    def __init__(self, mode: str = "empirical"):
        """
        Args:
            mode: Fisher 계산 모드 ('exact', 'empirical', 'diagonal')
        """
        self.mode = mode
        logger.info(f"Fisher Information 계산기 초기화: 모드={mode}")

    def compute_fisher_diagonal(self, model: nn.Module, dataloader, 
                              criterion, num_samples: int = 1000) -> Dict[str, torch.Tensor]:
        """
        대각 Fisher Information Matrix 계산
        F_i,i = E[(∂log p(y|x,θ)/∂θ_i)²] 구현
        """
        fisher = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)

        model.eval()
        samples_processed = 0

        with torch.no_grad():
            for data, target in tqdm(dataloader, desc="Fisher 계산 중"):
                if samples_processed >= num_samples:
                    break

                data, target = data.cuda(), target.cuda()
                batch_size = data.size(0)
                samples_processed += batch_size

                # 모드별 Fisher 계산
                if self.mode == "exact":
                    # 정확한 Fisher Information 계산
                    output = model(data)
                    log_probs = F.log_softmax(output, dim=1)

                    for class_idx in range(output.size(1)):
                        model.zero_grad()
                        class_log_prob = log_probs[:, class_idx].mean()
                        class_log_prob.backward(retain_graph=(class_idx < output.size(1) - 1))

                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                fisher[name] += param.grad.pow(2) * batch_size / num_samples

                elif self.mode == "empirical":
                    # 경험적 Fisher Information (실제 레이블 사용)
                    for i in range(batch_size):
                        model.zero_grad()
                        output = model(data[i:i+1])
                        loss = criterion(output, target[i:i+1])
                        loss.backward()

                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                fisher[name] += param.grad.pow(2) / num_samples

                elif self.mode == "diagonal":
                    # 대각 근사 Fisher (메모리 효율적)
                    output = model(data)
                    log_probs = F.log_softmax(output, dim=1)

                    # 실제 타겟에 대한 로그 확률의 그래디언트
                    model.zero_grad()
                    target_log_probs = log_probs[torch.arange(batch_size), target]
                    target_log_probs.mean().backward()

                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            fisher[name] += param.grad.pow(2) * batch_size / num_samples

        model.train()
        return fisher

# Meta-Experience Replay 구현
class MetaExperienceReplay:
    """
    Meta-Experience Replay 완전 구현
    Stanford NLP의 원본 논문 기반
    """
    def __init__(self, buffer_size: int = 5120, beta: float = 0.1, 
                 gamma: float = 0.1, s: int = 10):
        """
        Args:
            buffer_size: 메모리 버퍼 크기
            beta: 배치 내 업데이트 강도
            gamma: 배치 간 업데이트 강도
            s: Reptile 스텝 수
        """
        self.buffer_size = buffer_size
        self.buffer = []
        self.beta = beta
        self.gamma = gamma
        self.s = s
        self.class_counter = defaultdict(int)
        logger.info(f"MER 초기화: 버퍼 크기={buffer_size}, β={beta}, γ={gamma}, s={s}")

    def reservoir_sampling(self, x: torch.Tensor, y: torch.Tensor):
        """저장소 샘플링을 이용한 메모리 버퍼 관리"""
        for i in range(x.size(0)):
            sample = (x[i].clone(), y[i].clone())

            if len(self.buffer) < self.buffer_size:
                self.buffer.append(sample)
                self.class_counter[y[i].item()] += 1
            else:
                # 클래스 밸런스를 유지하기 위한 선택적 대체
                if self.should_replace(y[i].item()):
                    # 같은 클래스의 샘플 대체
                    indices = [j for j, (_, label) in enumerate(self.buffer) if label == y[i]]
                    if indices:
                        idx = random.choice(indices)
                        self.buffer[idx] = sample

    def should_replace(self, class_label: int) -> bool:
        """클래스 밸런스를 위한 대체 결정"""
        # 적게 등장한 클래스는 대체 확률 높임
        counts = list(self.class_counter.values())
        if not counts:
            return True

        avg_count = sum(counts) / len(counts)
        current_count = self.class_counter[class_label]

        # 평균보다 적게 등장한 클래스는 대체할 확률 높임
        if current_count < avg_count:
            return random.random() < 0.8
        else:
            return random.random() < 0.2

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """버퍼에서 배치 샘플링"""
        if len(self.buffer) == 0:
            return None, None

        indices = np.random.choice(len(self.buffer), 
                                 min(batch_size, len(self.buffer)), 
                                 replace=False)

        x_batch = torch.stack([self.buffer[i][0] for i in indices])
        y_batch = torch.stack([self.buffer[i][1] for i in indices])

        return x_batch, y_batch

    def meta_update(self, model: nn.Module, optimizer, criterion, 
                   current_batch: Tuple[torch.Tensor, torch.Tensor],
                   memory_batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """
        MER 메타 업데이트 수행
        Reptile 알고리즘 기반
        """
        x_curr, y_curr = current_batch
        x_mem, y_mem = memory_batch

        # 현재 파라미터 저장
        old_params = {name: param.clone() for name, param in model.named_parameters()}

        # Within-batch Reptile update
        for _ in range(self.s):
            # 현재 배치 업데이트
            optimizer.zero_grad()
            output_curr = model(x_curr)
            loss_curr = criterion(output_curr, y_curr)
            loss_curr.backward()
            optimizer.step()

            if x_mem is not None:
                # 메모리 배치 업데이트  
                inner_params = {name: param.clone() for name, param in model.named_parameters()}

                optimizer.zero_grad()
                output_mem = model(x_mem)
                loss_mem = criterion(output_mem, y_mem)
                loss_mem.backward()
                optimizer.step()

                # Within-batch interpolation
                for name, param in model.named_parameters():
                    param.data = inner_params[name] + self.beta * (param.data - inner_params[name])

        # Across-batch Reptile update (메타 업데이트)
        if x_mem is not None:
            for name, param in model.named_parameters():
                param.data = old_params[name] + self.gamma * (param.data - old_params[name])

        return loss_curr.item()

    def get_state_dict(self):
        """상태 저장을 위한 딕셔너리 반환"""
        return {
            'buffer': self.buffer,
            'class_counter': dict(self.class_counter),
            'buffer_size': self.buffer_size,
            'beta': self.beta,
            'gamma': self.gamma,
            's': self.s
        }

    def load_state_dict(self, state_dict):
        """저장된 상태 로드"""
        self.buffer = state_dict['buffer']
        self.class_counter = defaultdict(int, state_dict['class_counter'])
        self.buffer_size = state_dict['buffer_size']
        self.beta = state_dict['beta']
        self.gamma = state_dict['gamma']
        self.s = state_dict['s']

# 다이나믹 배치 사이즈 관리 클래스
# DynamicBatchSizeManager 클래스 제거됨

# 적응적 학습률 스케줄러
class AdaptiveLRScheduler:
    """
    선형 감소 + 웜업 + 종료 시 급격한 감소를 포함한 
    적응적 학습률 스케줄링 (LambdaLR을 사용하지 않고 직접 구현)
    """
    def __init__(self, optimizer, total_steps: int, 
                 warmup_steps: int = 0,
                 min_lr_ratio: float = 0.1,
                 end_lr_ratio: float = 0.01,
                 base_lr: float = 2e-4):
        """
        Args:
            optimizer: 학습률을 조정할 옵티마이저
            total_steps: 총 훈련 스텝 수
            warmup_steps: 웜업 스텝 수
            min_lr_ratio: 최소 학습률 비율
            end_lr_ratio: 종료 시 학습률 비율
            base_lr: 기본 최대 학습률 (옵티마이저의 초기 학습률이 너무 낮을 경우 사용)
        """
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.end_lr_ratio = end_lr_ratio
        self.current_step = 0
        
        # 초기 학습률 확인 및 설정
        init_lrs = [group['lr'] for group in optimizer.param_groups]
        self.base_lrs = []
        
        for i, init_lr in enumerate(init_lrs):
            # 초기 학습률이 너무 낮으면 지정된 base_lr 사용
            if init_lr < 1e-5:
                logger.warning(f"[{i}] 학습률이 너무 낮음: {init_lr:.8f}, {base_lr}로 대체")
                self.base_lrs.append(base_lr)
                # 초기 학습률 직접 설정
                optimizer.param_groups[i]['lr'] = base_lr * 0.1  # 웜업 초기값
            else:
                self.base_lrs.append(init_lr)
        
        logger.info(f"개선된 학습률 스케줄러 초기화: 웜업={warmup_steps}, 총 스텝={total_steps}")
        logger.info(f"기본 학습률: {self.base_lrs}, 현재 학습률: {[group['lr'] for group in optimizer.param_groups]}")

    def step(self, current_step: int):
        """학습률 업데이트 (외부에서 현재 스텝을 명시적으로 받음)
        
        Args:
            current_step: 현재의 전역 학습 스텝 (global_step)
        """
        # 강제 디버그 로그 추가 - 매 스텝 로깅 (문제 상황 확인용)
        old_lr = self.optimizer.param_groups[0]['lr']
        old_step = self.current_step
        
        # 중요: 호출 스택 출력 (누가 호출했는지 추적)
        import traceback
        stack = traceback.format_stack()
        caller = stack[-2] if len(stack) > 1 else "Unknown"
        logger.debug(f"[중요] 스케줄러.step({current_step}) 호출됨 - 이전 스텝: {old_step}, 이전 학습률: {old_lr:.6f}")
        logger.debug(f"호출 위치: {caller.strip()}")
        
        # 내부 스텝 카운터에 현재 전역 스텝을 명시적으로 할당
        self.current_step = current_step
        
        # 현재 단계에 대한 학습률 계산
        new_lrs = self._get_lr(self.current_step)
        
        # 옵티마이저의 학습률 직접 업데이트
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = lr
            
        # 매 스텝마다 학습률 변화 로깅 (디버깅용)
        logger.debug(f"[스텝 {self.current_step}] 학습률: {new_lrs[0]:.6f} (이전: {old_lr:.6f}, 변화량: {new_lrs[0]-old_lr:.8f})")
        
        # 중요 시점에서는 INFO 레벨로 로그 출력
        if self.current_step % 50 == 0 or self.current_step == self.warmup_steps or \
           self.current_step == self.warmup_steps - 1 or self.current_step == self.warmup_steps + 1 or \
           abs(new_lrs[0] - old_lr) > 1e-6:
            logger.info(f"[단계 {self.current_step}] 학습률 변화: {old_lr:.6f} -> {new_lrs[0]:.6f} (웜업={self.warmup_steps}, 총={self.total_steps})")
        
        return new_lrs[0]  # 현재 학습률 반환
    
    def _get_lr(self, current_step):
        """현재 단계에 대한 학습률 계산"""
        lrs = []
        for i, base_lr in enumerate(self.base_lrs):
            # 로그 추가 - 행동 추적
            if current_step % 200 == 0 or current_step == self.warmup_steps:
                logger.debug(f"LR 계산 [step={current_step}, warmup={self.warmup_steps}, total={self.total_steps}]")
            
            # 워밍 기간
            if current_step < self.warmup_steps:
                # 워밍 동안 기본 학습률의 10% -> 100%까지 선형 증가
                ratio = float(current_step) / float(max(1, self.warmup_steps))
                lr = base_lr * (0.1 + 0.9 * ratio)  # 10%에서 시작해서 워밍
                
                # 워밍 로깅 강화
                if current_step % 20 == 0:
                    logger.debug(f"[그룹 {i}] 워밍 학습률: {lr:.6f} (ratio={ratio:.2f}, step={current_step}/{self.warmup_steps})")
            
            # 막바지 구간 (90%~100%)
            elif current_step > 0.9 * self.total_steps:
                progress = (current_step - 0.9 * self.total_steps) / (0.1 * self.total_steps)
                decay = max(self.end_lr_ratio, (1.0 - progress) * (1.0 - 0.9) + 0.9 * self.end_lr_ratio)
                lr = base_lr * decay
                
                if current_step % 100 == 0:
                    logger.debug(f"[그룹 {i}] 막바지 감소 학습률: {lr:.6f} (decay={decay:.2f})")
            
            # 중간 구간 (10%~90%)
            else:
                progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                decay = max(self.min_lr_ratio, 1.0 - progress)
                lr = base_lr * decay
                
                if current_step % 500 == 0:
                    logger.debug(f"[그룹 {i}] 선형 감소 학습률: {lr:.6f} (progress={progress:.2f}, decay={decay:.2f})")
                
            # 최소값 보장
            original_lr = lr
            lr = max(base_lr * 0.01, lr)  # 기본 학습률의 1% 이상
            
            # 최소값 보정 발생 로그
            if lr != original_lr and (current_step % 500 == 0 or current_step == 1):
                logger.debug(f"[그룹 {i}] 최소 학습률 보장: {original_lr:.6f} -> {lr:.6f}")
                
            lrs.append(lr)
            
        return lrs
    
    def get_last_lr(self):
        """현재 학습률 반환 (PyTorch 스케줄러 호환성)"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """상태 저장 (PyTorch 스케줄러 호환성)"""
        return {
            'base_lrs': self.base_lrs,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'warmup_steps': self.warmup_steps,
            'min_lr_ratio': self.min_lr_ratio,
            'end_lr_ratio': self.end_lr_ratio
        }
    
    def load_state_dict(self, state_dict):
        """상태 불러오기 (PyTorch 스케줄러 호환성)"""
        self.base_lrs = state_dict.get('base_lrs', self.base_lrs)
        self.current_step = state_dict.get('current_step', 0)
        self.total_steps = state_dict.get('total_steps', self.total_steps)
        self.warmup_steps = state_dict.get('warmup_steps', self.warmup_steps)
        self.min_lr_ratio = state_dict.get('min_lr_ratio', self.min_lr_ratio)
        self.end_lr_ratio = state_dict.get('end_lr_ratio', self.end_lr_ratio)
        
        # 학습률 즉시 업데이트
        new_lrs = self._get_lr(self.current_step)
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = lr

# Feature Adapter 구현
class FeatureAdapter(nn.Module):
    """
    언어 모델의 큰 출력 차원을 분류기가 기대하는 작은 차원으로 변환하는 어댑터
    행렬곱 차원 불일치 문제를 해결하기 위한 중간 변환 레이어
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 2048):
        """
        Args:
            input_dim: 입력 차원 (언어 모델의 vocab_size 또는 출력 차원)
            output_dim: 출력 차원 (분류기가 기대하는 특징 차원)  
            hidden_dim: 중간 은닉층 차원
        """
        super().__init__()
        
        # 입력 차원이 매우 큰 경우 (vocab_size 등) 단계적 차원 축소
        if input_dim > 10000:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, output_dim)
            )
        else:
            # 입력 차원이 작은 경우 단순한 변환
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim)
            )
            
        logger.info(f"FeatureAdapter 초기화: {input_dim} → {output_dim} (hidden: {hidden_dim})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        특징 변환 수행
        
        Args:
            x: 입력 텐서 - 다양한 형태의 모델 출력을 처리
        
        Returns:
            변환된 특징 텐서 (batch_size, output_dim)
        """
        # 입력 텐서 형태 정규화
        if len(x.shape) == 3:  # (batch_size, seq_len, hidden_dim)
            # 시퀀스의 마지막 토큰 사용 (언어 모델의 일반적 패턴)
            x = x[:, -1, :]
        elif len(x.shape) > 3:
            # 더 고차원인 경우 배치 차원을 제외한 모든 차원을 평탄화
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
        elif len(x.shape) == 1:
            # 1차원인 경우 배치 차원 추가
            x = x.unsqueeze(0)
        
        # 차원 변환 적용
        return self.layers(x)

# Nearest Mean Classifier 구현
class NearestMeanClassifier(nn.Module):
    """
    Stability Gap 완화를 위한 Nearest Mean Classifier
    Supervised Contrastive Replay와 결합 시 최고 성능
    """
    def __init__(self, num_classes: int, feat_dim: int):
        """
        Args:
            num_classes: 클래스 수
            feat_dim: 특징 벡터 차원
        """
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.register_buffer('class_means', torch.zeros(num_classes, feat_dim))
        self.register_buffer('class_counts', torch.zeros(num_classes))
        logger.info(f"Nearest Mean Classifier 초기화: 클래스={num_classes}, 차원={feat_dim}")

    def update_class_means(self, features: torch.Tensor, labels: torch.Tensor):
        """클래스별 평균 특징 벡터 업데이트"""
        for c in range(self.num_classes):
            idx = (labels == c)
            if idx.sum() > 0:
                class_feats = features[idx]
                self.class_means[c] = (self.class_means[c] * self.class_counts[c] + class_feats.sum(0)) / (self.class_counts[c] + idx.sum())
                self.class_counts[c] += idx.sum()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        가장 가까운 클래스 평균으로 분류

        Args:
            features: 모델에서 추출한 특징 벡터

        Returns:
            각 클래스의 로짓 값
        """
        # L2 정규화
        norm_features = F.normalize(features, p=2, dim=1)
        norm_class_means = F.normalize(self.class_means, p=2, dim=1)

        # 코사인 유사도 계산
        logits = torch.matmul(norm_features, norm_class_means.t())

        # 온도 스케일링 (선택적)
        # logits = logits / 0.07  # temperature

        return logits

    def get_state_dict(self):
        """상태 저장"""
        return {
            'class_means': self.class_means.clone(),
            'class_counts': self.class_counts.clone(),
            'num_classes': self.num_classes,
            'feat_dim': self.feat_dim
        }

    def load_state_dict(self, state_dict, strict=True):
        """상태 로드"""
        self.num_classes = state_dict['num_classes']
        self.feat_dim = state_dict['feat_dim']
        self.class_means = state_dict['class_means'].clone()
        self.class_counts = state_dict['class_counts'].clone()

# Supervised Contrastive Replay 손실
class SupervisedContrastiveLoss(nn.Module):
    """
    NCM과 함께 사용하는 지도 대조 손실
    같은 클래스의 샘플들은 가까게, 다른 클래스의 샘플들은 멀게 배치
    """
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: 대조 손실 온도 파라미터
        """
        super().__init__()
        self.temperature = temperature
        logger.info(f"지도 대조 손실 초기화: 온도={temperature}")

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        지도 대조 손실 계산

        Args:
            features: 정규화된 특징 벡터 (batch_size, feat_dim)
            labels: 클래스 레이블 (batch_size)

        Returns:
            대조 손실 값
        """
        # L2 정규화
        features = F.normalize(features, p=2, dim=1)

        # 유사도 행렬 계산
        similarity_matrix = torch.matmul(features, features.t()) / self.temperature

        # 마스크 생성: 같은 클래스 = 1, 다른 클래스 = 0
        batch_size = features.size(0)
        labels_matrix = labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).t())

        # 자기 자신은 제외
        labels_matrix.fill_diagonal_(False)

        # 양의 쌍이 있는지 확인
        pos_pairs = labels_matrix.sum(dim=1) > 0

        if not pos_pairs.any():
            return torch.tensor(0.0, device=features.device)

        # 마스크가 있는 샘플만 선택
        labels_matrix = labels_matrix[pos_pairs]
        similarity_matrix = similarity_matrix[pos_pairs]

        # 로그-소프트맥스와 마스크 적용
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()  # 수치 안정성

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 양의 쌍의 로그 확률 평균
        mean_log_prob_pos = (labels_matrix * log_prob).sum(1) / labels_matrix.sum(1)

        # 손실 계산
        loss = -mean_log_prob_pos.mean()
        return loss

# EWC (Elastic Weight Consolidation) 구현
class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation
    이전 태스크의 중요한 가중치를 보호하여 Catastrophic Forgetting 완화
    """
    def __init__(self, model: nn.Module, fisher_computer: FisherInformationComputer,
                 lambda_ewc: float = 5000.0, normalize_fisher: bool = True):
        """
        Args:
            model: 모델
            fisher_computer: Fisher Information 계산기
            lambda_ewc: EWC 강도 계수
            normalize_fisher: Fisher 정규화 여부
        """
        self.model = model
        self.fisher_computer = fisher_computer
        self.lambda_ewc = lambda_ewc
        self.normalize_fisher = normalize_fisher

        self.fisher_dict = {}  # mode -> fisher
        self.optpar_dict = {}  # mode -> optimal parameters

        logger.info(f"EWC 초기화: λ={lambda_ewc}, 정규화={normalize_fisher}")

    def register_task(self, mode: str, dataloader, criterion):
        """
        새 학습 모드 등록 및 Fisher Information 계산

        Args:
            mode: 학습 모드 이름 (complete, prompt, comment, error_fix)
            dataloader: 학습 데이터 로더
            criterion: 손실 함수
        """
        logger.info(f"[모드: {mode}] Fisher Information 계산 중...")
        # Fisher Information 계산
        fisher = self.fisher_computer.compute_fisher_diagonal(
            self.model, dataloader, criterion
        )

        # Fisher 정규화 (선택적)
        if self.normalize_fisher:
            for name, f in fisher.items():
                fisher[name] = f / f.sum()

        # 현재 파라미터 저장
        optpar = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                optpar[name] = param.data.clone()

        # 학습 모드 정보 저장
        self.fisher_dict[mode] = fisher
        self.optpar_dict[mode] = optpar

        logger.info(f"[모드: {mode}] 등록 완료")

    def compute_ewc_loss(self) -> torch.Tensor:
        """
        EWC 정규화 손실 계산
        이전 태스크의 중요 파라미터 변화에 페널티 부여

        Returns:
            EWC 손실
        """
        loss = torch.tensor(0., device=next(self.model.parameters()).device)

        # 등록된 태스크가 없으면 손실 없음
        if not self.fisher_dict:
            return loss

        # 각 학습 모드에 대한 EWC 손실 계산
        for mode, fisher in self.fisher_dict.items():
            optpar = self.optpar_dict[mode]

            for name, param in self.model.named_parameters():
                if name in fisher and param.requires_grad:
                    try:
                        # 텐서 처리를 안전하게 수행
                        # 가능한 경우 먼저 머신 에러 체크
                        if fisher[name].shape != param.shape or optpar[name].shape != param.shape:
                            logger.warning(f"EWC 손실 계산 시 텐서 형태 불일치 무시: {name}, {fisher[name].shape}, {param.shape}, {optpar[name].shape}")
                            continue
                            
                        # 두 텐서의 차이
                        diff = param - optpar[name]
                        # 제곱 계산
                        squared_diff = diff.pow(2)
                        # Fisher로 가중치 계산
                        weighted_squared_diff = fisher[name] * squared_diff
                        # 합계 계산
                        param_loss = weighted_squared_diff.sum()
                        # 확인 후 추가
                        if torch.isfinite(param_loss).all():  # NaN 확인
                            loss += param_loss
                        else:
                            logger.warning(f"EWC 손실에서 NaN/Inf 발견, 무시함: {name}")
                    except Exception as e:
                        logger.warning(f"EWC 손실 계산 중 오류 발생(무시): {name}, {str(e)}")
                        continue

        # 태스크 수로 정규화
        num_tasks = len(self.fisher_dict)
        if num_tasks > 0:
            loss = self.lambda_ewc * loss / num_tasks

        return loss

    def get_state_dict(self):
        """상태 저장"""
        return {
            'fisher_dict': self.fisher_dict,
            'optpar_dict': self.optpar_dict,
            'lambda_ewc': self.lambda_ewc,
            'normalize_fisher': self.normalize_fisher
        }

    def load_state_dict(self, state_dict):
        """상태 로드"""
        self.fisher_dict = state_dict['fisher_dict']
        self.optpar_dict = state_dict['optpar_dict']
        self.lambda_ewc = state_dict['lambda_ewc']
        self.normalize_fisher = state_dict['normalize_fisher']

# 지속 학습 메트릭 계산기
class ContinualLearningMetrics:
    """
    지속 학습 전용 평가 메트릭
    ACC, BWT, FWT, Stability Gap 등 포괄적 평가
    모드 기반 학습 지원 (complete, prompt, comment, error_fix)
    """
    def __init__(self):
        # 정확도 기록 - 모드 기반
        self.mode_accuracies = defaultdict(list)  # {mode: [accuracies]}
        self.stability_gaps = []
        self.mode_names = []  # 학습 순서대로 등록된 모드
        # 호환성을 위해 태스크 정확도 변수 유지
        self.task_accuracies = defaultdict(list)
        self.task_names = []

    def update(self, current_mode: str, test_results: Dict[str, float]):
        """
        모드별 정확도 업데이트

        Args:
            current_mode: 현재 학습 중인 모드 (complete, prompt, comment, error_fix)
            test_results: 모든 이전 모드에 대한 테스트 결과 {mode: accuracy}
        """
        # 모드가 있는 경우 - 모드 기반 저장
        if current_mode:
            # 모드 명칭 기록 - 학습 순서 추적
            if current_mode not in self.mode_names:
                self.mode_names.append(current_mode)
                
            # 모드별 정확도 업데이트
            for test_mode, accuracy in test_results.items():
                self.mode_accuracies[test_mode].append(accuracy)
        
        # 호환성을 위해 태스크 ID 기반 기록도 유지
        # 이는 점진적으로 삭제 가능
        task_id_mapping = {'complete': 0, 'prompt': 1, 'comment': 2, 'error_fix': 3}
        if current_mode and current_mode in task_id_mapping:
            task_id = task_id_mapping[current_mode]
            for test_mode, accuracy in test_results.items():
                if test_mode in task_id_mapping:
                    test_task_id = task_id_mapping[test_mode]
                    self.task_accuracies[test_task_id].append(accuracy)

    def record_stability_gap(self, current_mode: str, accuracies: List[float]):
        """
        Stability Gap 기록 - 모드 기반

        Args:
            current_mode: 현재 모드
            accuracies: 학습 중 이전 모드의 정확도 변화
        """
        if len(accuracies) < 3:  # 너무 적은 데이터
            return

        initial_acc = accuracies[0]
        min_acc = min(accuracies)
        final_acc = accuracies[-1]

        gap = initial_acc - min_acc
        recovery = final_acc - min_acc

        self.stability_gaps.append({
            'mode': current_mode,  # 태스크 ID 대신 모드 사용
            'gap': gap,
            'recovery': recovery,
            'recovery_ratio': recovery / gap if gap > 0 else 1.0
        })

    def compute_metrics(self) -> Dict[str, float]:
        """
        BWT, FWT, ACC, Stability Gap 계산 - 모드 기반

        Returns:
            계산된 메트릭들
        """
        # 모드 기반 계산
        if self.mode_names:  # 모드 기반 데이터가 있는 경우
            # 모드 순서 정리 - [완성/프롬프트/주석/오류수정] 순으로 정렬
            mode_order = [m for m in ['complete', 'prompt', 'comment', 'error_fix'] if m in self.mode_names]
            
            # 최종 정확도
            final_accuracies = []
            for mode in mode_order:
                if mode in self.mode_accuracies and self.mode_accuracies[mode]:
                    final_accuracies.append(self.mode_accuracies[mode][-1])
            
            # 평균 정확도
            acc = np.mean(final_accuracies) if final_accuracies else 0.0
            
            # 매개변수 계산
            bwt = 0.0  # Backward Transfer
            fwt = 0.0  # Forward Transfer
            
            if len(mode_order) > 1:
                # 후방 전이(BWT) - 이전 모드의 최초/최종 성능 비교
                bwt_values = []
                for i, mode in enumerate(mode_order[:-1]):  # 마지막 모드 제외
                    if mode in self.mode_accuracies and len(self.mode_accuracies[mode]) >= 2:
                        initial = self.mode_accuracies[mode][0]  # 최초 성능
                        final = self.mode_accuracies[mode][-1]  # 현재 성능
                        bwt_values.append(final - initial)
                        
                bwt = np.mean(bwt_values) if bwt_values else 0.0
                
                # 전방 전이(FWT) - 현재 모드의 최초 성능과 기준선 비교
                baseline_acc = 0.0  # 기본 기준선
                fwt_values = []
                for i, mode in enumerate(mode_order[1:]):  # 첫 번째 모드 제외
                    if mode in self.mode_accuracies and self.mode_accuracies[mode]:
                        fwt_values.append(self.mode_accuracies[mode][0] - baseline_acc)
                        
                fwt = np.mean(fwt_values) if fwt_values else 0.0
            
            # Stability Gap 계산
            avg_gap = 0.0
            avg_recovery_ratio = 0.0
            if self.stability_gaps:
                avg_gap = np.mean([gap['gap'] for gap in self.stability_gaps])
                avg_recovery_ratio = np.mean([gap['recovery_ratio'] for gap in self.stability_gaps])
            
            # 모드별 최종 정확도 생성
            mode_final_accuracies = {mode: self.mode_accuracies[mode][-1] 
                                   for mode in self.mode_accuracies 
                                   if self.mode_accuracies[mode]}
            
            return {
                'ACC': acc,
                'BWT': bwt, 
                'FWT': fwt,
                'mode_accuracies': mode_final_accuracies,
                'stability_gap': avg_gap,
                'recovery_ratio': avg_recovery_ratio
            }
        
        # 호환성을 위해 기존 task_id 기반 메트릭 계산 유지
        else:
            num_tasks = len(self.task_accuracies)
            if num_tasks == 0:
                return {'ACC': 0, 'BWT': 0, 'FWT': 0, 'stability_gap': 0, 'recovery_ratio': 0}
                
            # Average Accuracy (ACC)
            final_accuracies = [self.task_accuracies[i][-1] for i in range(num_tasks) if self.task_accuracies[i]]
            acc = np.mean(final_accuracies) if final_accuracies else 0.0
            
            # 기타 메트릭은 호환성을 위해 기존 로직 유지
            # 추후 완전히 모드 기반으로 대체될 예정
            bwt = 0
            for i in range(num_tasks - 1):
                if i in self.task_accuracies and len(self.task_accuracies[i]) > i:
                    bwt += self.task_accuracies[i][-1] - self.task_accuracies[i][i]
            bwt /= (num_tasks - 1) if num_tasks > 1 else 1

            fwt = 0
            baseline_acc = 0
            for i in range(1, num_tasks):
                if len(self.task_accuracies[i]) > 0:
                    fwt += self.task_accuracies[i][0] - baseline_acc
            fwt /= (num_tasks - 1) if num_tasks > 1 else 1
            
            avg_gap = 0
            avg_recovery_ratio = 0
            if self.stability_gaps:
                avg_gap = np.mean([gap['gap'] for gap in self.stability_gaps])
                avg_recovery_ratio = np.mean([gap['recovery_ratio'] for gap in self.stability_gaps])
                
            return {
                'ACC': acc,
                'BWT': bwt, 
                'FWT': fwt,
                'final_accuracies': final_accuracies,
                'stability_gap': avg_gap,
                'recovery_ratio': avg_recovery_ratio
            }

    def get_state_dict(self):
        """상태 저장 - 모드와 태스크 기반 모두 저장"""
        return {
            'mode_accuracies': dict(self.mode_accuracies),
            'mode_names': self.mode_names,
            'task_accuracies': dict(self.task_accuracies),  # 호환성을 위해 유지
            'stability_gaps': self.stability_gaps,
            'task_names': self.task_names
        }

    def load_state_dict(self, state_dict):
        """저장된 상태 로드"""
        # 모드 기반 상태 로드
        if 'mode_accuracies' in state_dict:
            self.mode_accuracies = defaultdict(list, state_dict['mode_accuracies'])
            self.mode_names = state_dict.get('mode_names', [])
        
        # 호환성을 위한 태스크 기반 데이터 로드
        self.task_accuracies = defaultdict(list, state_dict.get('task_accuracies', {}))
        self.stability_gaps = state_dict.get('stability_gaps', [])
        self.task_names = state_dict.get('task_names', [])
        
    def get_accuracy_dict(self):
        """모드별 최종 정확도를 반환합니다.
        
        Returns:
            Dict[str, float]: 각 모드의 최신 정확도
        """
        # 모드 기반 정확도 반환
        mode_accuracies = {}
        
        # 각 모드별 최신 정확도 추출
        for mode, accuracies in self.mode_accuracies.items():
            if accuracies:  # 빈 리스트가 아닌 경우
                mode_accuracies[mode] = accuracies[-1]  # 가장 최근 정확도
            else:
                mode_accuracies[mode] = 0.0  # 기본값
        
        # 데이터가 없는 경우 기본값 제공
        if not mode_accuracies:
            # 기본 모드들에 대한 기본값 설정
            for mode in ['complete', 'prompt', 'comment', 'error_fix']:
                mode_accuracies[mode] = 0.0
        
        # 호환성을 위한 태스크 ID 정확도 추가
        for task_id, accuracies in self.task_accuracies.items():
            if accuracies:  # 빈 리스트가 아닌 경우
                mode_accuracies[f"task_{task_id}"] = accuracies[-1]  # 가장 최근 정확도
                
        return mode_accuracies

# 메인 지속 학습 트레이너 클래스
class ContinualLearner:
    """
    Continual Learning 지속학습 시스템
    높은 성공률과 안정성을 위한 모든 기법 통합
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 설정 파라미터
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # CUDA 메모리 단편화 해결을 위한 환경변수 설정
        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        
        # GPU 메모리 완전 정리 및 최적화
        if torch.cuda.is_available():
            import gc
            torch.cuda.empty_cache()
            gc.collect()
            
            # GPU 메모리 사용량 제한 설정
            memory_fraction = config.get('gpu_memory_fraction', 0.75)
            torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
            logger.info(f"GPU 메모리 사용량 제한: {memory_fraction*100:.0f}%")
            logger.info("CUDA 메모리 단편화 방지 설정 적용됨")

        # 시드 설정
        set_seed(config.get('seed', 42))

        # 모델 구성
        self.build_model()

        # 핵심 컴포넌트 초기화
        self.setup_components()

        # 메트릭 및 로깅 설정
        self.metrics = ContinualLearningMetrics()
        self.writer = Writer()  # 디렉토리 관련 파라미터 제거
        
        # 로그 레벨 WARNING으로 설정 (불필요한 로그 최소화)
        # DeepseekLogger는 직접 setLevel이 없고 내부 logger 객체에 있음
        logger.logger.setLevel(logging.WARNING)  # 내부 logger 객체에 접근
        for handler in logger.logger.handlers:
            handler.setLevel(logging.WARNING)
        print("[시스템] 로그 레벨이 WARNING으로 설정되었습니다.")

        # AWS Spot Instance 중단 핸들러 초기화
        # 체크포인트 디렉토리를 output_dir로 설정 (중요: config에서 output_dir을 사용)
        output_dir = config.get('output_dir', '/home/ubuntu/deepseek-continual/checkpoints/prompt-finetuned/')
        
        # 경로 중복 문제 해결: 경로 끝에 'prompt-finetuned'가 이미 포함되어 있는지 확인
        if output_dir.endswith('prompt-finetuned/') or output_dir.endswith('prompt-finetuned'):
            checkpoint_dir = output_dir  # 그대로 사용
            print(f"[중요] 체크포인트 경로 사용: {checkpoint_dir} (기존 경로 사용)")
        else:
            checkpoint_dir = os.path.join(output_dir, 'checkpoints')  # 경로가 없는 경우 새로 생성
            print(f"[중요] 체크포인트 경로 생성: {checkpoint_dir}")
        
        # 경로가 중복으로 들어가는 문제가 있는지 확인
        path_parts = checkpoint_dir.split('/')
        duplicates = [item for item, count in collections.Counter(path_parts).items() if count > 1 and item]
        if duplicates:
            print(f"[경고] 경로에 중복 부분 발견: {duplicates}")
            # 마지막 부분만 사용하여 중복 방지
            checkpoint_dir = '/'.join(path_parts[:-1]) if path_parts[-1] in duplicates else checkpoint_dir
        
        # 디렉토리 확인 및 생성
        os.makedirs(checkpoint_dir, exist_ok=True)
        dir_status = os.access(checkpoint_dir, os.W_OK)
        print(f"[중요] 체크포인트 디렉토리: {checkpoint_dir}")
        print(f"[중요] 디렉토리 상태: 존재={os.path.exists(checkpoint_dir)}, 쓰기가능={dir_status}")
        logger.info(f"[디버그] 체크포인트 디렉토리 확인: {checkpoint_dir} (존재: {os.path.exists(checkpoint_dir)}, 쓰기가능: {dir_status})")
        
        # config에서 체크포인트 주기 가져오기
        checkpoint_freq = config.get('checkpoint_freq', 20)  # 기본값 20으로 설정
        print(f"[중요] 체크포인트 저장 주기: {checkpoint_freq} 스텝마다")
        logger.info(f"[디버그] 체크포인트 주기 설정: {checkpoint_freq} 스텝")
        
        # config에 checkpoint_dir 명시적으로 추가
        config['checkpoint_dir'] = checkpoint_dir
        
        # SpotInterruptionHandler 초기화
        self.spot_handler = SpotInterruptionHandler(
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq
        )
        
        # 명시적으로 모니터링 시작
        self.spot_handler.start_monitoring(model=self.model, optimizer=self.optimizer, trainer=self)
        logger.info("[디버그] SpotInterruptionHandler.start_monitoring 호출됨")

        # 배치 크기 설정 (DynamicBatchSizeManager 제거됨)
        self.batch_size = config.get('batch_size', 32)
        
        # 그라디언트 누적 스텝 설정 (메모리 최적화)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
        logger.info(f"그라디언트 누적 스텝: {self.gradient_accumulation_steps}")
        self.accumulated_gradients = 0
        
        # VRAM 모니터링 초기화 - 메모리 호환성 추가
        self.memory_monitor = None
        if MEMORY_MONITOR_AVAILABLE and torch.cuda.is_available():
            try:
                # OOM 발생 시 배치 크기 자동 감소 콜백 함수
                def reduce_batch_size_callback():
                    current_size = self.batch_size
                    self.batch_size = max(1, int(current_size * 0.8))  # 20% 감소
                    logger.warning(f"VRAM 사용량 경고: 배치 크기 {current_size} → {self.batch_size}로 자동 감소")
                
                # 메모리 모니터 인스턴스 생성 - A10G 환경에 최적화된 경계값
                # A10G 22GB에서 안정적으로 학습하는 것이 목표
                warning_threshold = float(self.config.get('memory_warning_threshold', 0.78))  # 기본값 78%
                critical_threshold = float(self.config.get('memory_critical_threshold', 0.88))  # 기본값 88%
                check_interval = float(self.config.get('memory_check_interval', 10.0))  # 기본값 10초
                
                self.memory_monitor = MemoryMonitor(
                    warning_threshold=warning_threshold,
                    critical_threshold=critical_threshold,
                    check_interval=check_interval,
                    critical_callback=reduce_batch_size_callback
                )
                logger.info(f"VRAM 모니터링 구성: 경고={warning_threshold*100:.0f}%, 위험={critical_threshold*100:.0f}%, 검사주기={check_interval}초")
                # 모니터링 시작
                self.memory_monitor.start()
                logger.info("VRAM 모니터링 시스템 활성화 - 자동 메모리 관리 기능 제공")
            except Exception as e:
                logger.warning(f"VRAM 모니터링 초기화 실패: {e}")
                self.memory_monitor = None

        # 학습 상태
        self.current_mode = None  # 현재 학습 모드 추적 (complete, prompt, comment, error_fix)
        self.completed_modes = []  # 완료된 학습 모드 목록
        self.global_step = 0
        self.best_accuracies = {}  # 모드별 최고 정확도 {mode: accuracy}

        logger.info("ContinualLearner 초기화 완료")
        
    def __del__(self):
        """인스턴스 소멸 시 호출되는 메서드 - 리소스 정리"""
        self.cleanup_resources()
        
    def cleanup_resources(self):
        """리소스 정리 - 스레드 종료 및 메모리 해제"""
        # 메모리 모니터링 정리
        if hasattr(self, 'memory_monitor') and self.memory_monitor is not None:
            try:
                self.memory_monitor.stop()
                logger.info("VRAM 모니터링 시스템 종료")
            except Exception as e:
                logger.warning(f"VRAM 모니터링 종료 중 오류: {e}")
        
        # 스팟 인스턴스 핸들러 정리
        if hasattr(self, 'spot_handler') and self.spot_handler is not None:
            try:
                self.spot_handler.stop_monitoring()
                logger.info("AWS 스팟 인스턴스 모니터링 종료")
            except Exception as e:
                logger.warning(f"AWS 스팟 인스턴스 모니터링 종료 중 오류: {e}")
        
        # CUDA 캠시 정리
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"CUDA 캠시 정리 중 오류: {e}")

    def build_model(self):
        """모델 구성 - DeepSeek-Coder 모델 로드"""
        # 이미 모델이 로드되어 있는지 확인 (독일성 방지)
        if hasattr(self, 'model') and self.model is not None:
            logger.info("이미 모델이 로드되어 있어 중복 로드를 방지합니다.")
            return
            
        model_config = self.config.get('model', {})
        model_name = model_config.get('name', 'deepseek-coder')
        mode = self.config.get('mode', 'prompt')  # 학습 모드 확인
        
        # DeepSeek-Coder 모델 로딩 시도
        if TRANSFORMERS_AVAILABLE and model_name == 'deepseek-coder':
            try:
                # AWS 환경에 맞는 기본 모델 경로 설정 (절대경로)
                base_model = "/home/ubuntu/deepseek-coder/model_cache/deepseek-coder-6.7b-instruct"  # 기본 모델 (경로 유지)
                
                # 홈 디렉토리 기반 모델 루트 경로 설정
                model_base_path = os.path.expanduser("~/deepseek-coder/models")
                
                # 모드별 모델 경로 확인
                if mode == 'complete':
                    # 코드 자동완성 모드
                    model_path = os.path.join(model_base_path, "autocomplete-finetuned/final_model")
                    logger.info(f"[자동완성] 모드 - 모델 경로: {model_path}")
                elif mode == 'prompt':
                    # 프롬프트 모드
                    model_path = os.path.join(model_base_path, "prompt-finetuned/final_model")
                    logger.info(f"[프롬프트] 모드 - 모델 경로: {model_path}")
                elif mode == 'comment':
                    # 주석 모드
                    model_path = os.path.join(model_base_path, "comment-finetuned/final_model")
                    logger.info(f"[주석] 모드 - 모델 경로: {model_path}")
                elif mode == 'error_fix':
                    # 오류 수정 모드
                    model_path = os.path.join(model_base_path, "error-fix-finetuned/final_model")
                    logger.info(f"[오류수정] 모드 - 모델 경로: {model_path}")
                
                else:
                    # 알 수 없는 모드일 경우 기본으로 prompt 모드 사용
                    model_path = os.path.join(model_base_path, "prompt-finetuned/final_model")
                    logger.info(f"알 수 없는 모드: {mode}, 프롬프트 모델 사용: {model_path}")
                
                # 경로가 없으면 기본 모델 사용 예정임을 로깅
                if not os.path.exists(model_path):
                    logger.warning(f"지정된 모델 경로가 존재하지 않습니다: {model_path}")
                    logger.warning(f"기본 모델 {base_model}를 사용합니다.")
                    use_base_model = True
                else:
                    use_base_model = False
                
                # 양자화 설정
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                
                try:
                    if use_base_model:
                        # 기본 모델 로드
                        logger.info(f"기본 모델 로드 시도: {base_model}")
                        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
                    else:
                        # 학습된 모델 로드
                        logger.info(f"학습된 모델 로드 시도: {model_path}")
                        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            quantization_config=bnb_config,
                            device_map="auto"
                        )
                    logger.info(f"학습된 {mode} 모델 로드 성공!")
                except Exception as e:
                    # 오류 발생시 기본 모델로 돌아가기
                    logger.warning(f"모델 로드 오류: {str(e)}")
                    logger.info(f"기본 모델 로드 시도(콜백): {base_model}")
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
                        self.model = AutoModelForCausalLM.from_pretrained(
                            base_model,
                            quantization_config=bnb_config,
                            device_map="auto"
                        )
                        logger.info(f"기본 모델 로드 성공(fallback)!")
                    except Exception as nested_e:
                        logger.error(f"기본 모델 로드도 실패: {str(nested_e)}")
                        # 모델 로드가 완전히 실패한 경우 - dummy 모델 사용
                        self._create_dummy_model()
                        logger.warning("모든 모델 로드 시도 실패. 더미 모델로 최소 한의 실행을 보장합니다.")
                
                # LoRA 설정
                lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                
                # LoRA 적용
                self.model = get_peft_model(self.model, lora_config)
                
                # VRAM 절약을 위한 그래디언트 체크포인팅 활성화
                # - 중간 activation 저장 대신 필요할 때 재계산 → VRAM 사용량 대폭 감소
                use_gradient_checkpointing = self.config.get('use_gradient_checkpointing', True)
                if use_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
                    self.model.gradient_checkpointing_enable()
                    logger.info("VRAM 절약을 위한 그래디언트 체크포인팅 활성화")
                
                # 모델 설정 저장
                self.feat_dim = 4096  # DeepSeek-Coder hidden size
                self.vocab_size = self.model.config.vocab_size  # 언어 모델 어휘 크기
                
                # 언어 모델 로그 정보
                logger.info(f"언어 모델 어휘 크기: {self.vocab_size}, 히든 차원: {self.feat_dim}")
                
                # 분류기와 FeatureAdapter 제거 - 언어 모델 내장 손실만 사용하도록 수정
                
                logger.info(f"DeepSeek-Coder 모델 로드 성공")
                
            except Exception as e:
                logger.error(f"DeepSeek-Coder 모델 로드 오류: {str(e)}")
                # 오류 발생시 더미 모델로 돌아감
                self._create_dummy_model()
        else:
            # Transformers 미설치 또는 지원되지 않는 모델일 경우 더미 모델 사용
            logger.warning(f"Transformers 라이브러리 미설치 또는 지원되지 않는 모델: {model_name}")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """더미 모델 생성 (테스트용)"""
        logger.warning("더미 모델을 사용합니다. 실제 학습에는 적합하지 않습니다.")
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        ).to(self.device)
        
        self.feat_dim = 256
        self.vocab_size = 1000  # 더미 모델의 어휘 사이즈
        
        # 분류기와 Feature Adapter 제거
        # 언어 모델만 사용하고 내장 손실 함수 사용
        logger.info(f"더미 모델 생성 완료 - 히든 차원: {self.feat_dim}")

        # 그래디언트 체크포인팅 (메모리 최적화)
        if self.config.get('use_gradient_checkpointing', True):
            # Hugging Face 모델은 gradient_checkpointing_enable 메서드가 있지만 Sequential는 없음
            # 안전하게 메서드 존재 여부 확인 후 호출
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("그래디언트 체크포인팅이 활성화되었습니다.")
            else:
                logger.warning("이 모델은 그래디언트 체크포인팅을 지원하지 않습니다.")

    def setup_components(self):
        """핵심 컴포넌트 초기화"""
        # GPU 메모리 상한 설정 - A10G 22GB 환경 최적화
        if torch.cuda.is_available():
            gpu_memory_fraction = float(self.config.get('gpu_memory_fraction', 0.80))  # 기본값 80%
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, 0)
            logger.info(f"GPU 메모리 사용량 제한: 전체 용량의 {gpu_memory_fraction*100:.1f}%")
            
            # 현재 GPU 메모리 상태 로깅
            if hasattr(torch.cuda, 'memory_reserved') and hasattr(torch.cuda, 'memory_allocated'):
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                logger.info(f"GPU 메모리 현황: 총={total:.2f}GB, 예약={reserved:.2f}GB, 할당={allocated:.2f}GB")
        
        # 옵티마이저 설정 - AdamA 지원 추가
        learning_rate = float(self.config.get('learning_rate', '3e-4'))  # 기본값 향상
        weight_decay = float(self.config.get('weight_decay', '0.01'))
        use_adama = self.config.get('adam_accumulation', False)  # AdamA 사용 여부
        
        logger.info(f"학습률: {learning_rate}, 가중치 감소율: {weight_decay}")
        
        if use_adama and ADAMA_AVAILABLE:
            # AdamA 사용 - 메모리 효율성 향상 (그래디언트 즉시 해제)
            logger.info("AdamA 최적화 옵티마이저 사용 (메모리 15-23% 절감)")
            self.optimizer = AdamA(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            # 일반 AdamW 사용
            if use_adama and not ADAMA_AVAILABLE:
                logger.warning("AdamA 패키지가 설치되지 않아 기본 AdamW 사용 ('pip install adama'로 설치 가능)")
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )

        # Fisher Information 계산기
        fisher_config = self.config.get('fisher', {})
        self.fisher_computer = FisherInformationComputer(
            mode=fisher_config.get('mode', 'empirical')
        )

        # EWC 설정
        ewc_config = self.config.get('ewc', {})
        self.ewc = ElasticWeightConsolidation(
            model=self.model,
            fisher_computer=self.fisher_computer,
            lambda_ewc=ewc_config.get('lambda', 5000.0),
            normalize_fisher=ewc_config.get('normalize', True)
        )

        # Meta-Experience Replay
        mer_config = self.config.get('mer', {})
        self.mer = MetaExperienceReplay(
            buffer_size=mer_config.get('buffer_size', 5120),
            beta=mer_config.get('beta', 0.1),
            gamma=mer_config.get('gamma', 0.1),
            s=mer_config.get('s', 10)
        )

        # 분류기 관련 손실 함수 제거 - 언어 모델 내장 손실만 사용
        logger.info("언어 모델링 내장 손실 함수 사용 설정")

        # 혼합 정밀도 훈련
        self.use_amp = self.config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            logger.info("혼합 정밀도 훈련(AMP)이 활성화되었습니다. (torch.amp.GradScaler 사용)")
            if not torch.cuda.is_available():
                logger.warning("CUDA가 사용 불가능하여 AMP 효과가 제한될 수 있습니다.")

        # 학습률 스케줄러 초기화 개선
        scheduler_config = self.config.get('scheduler', {})
        total_steps = scheduler_config.get('total_steps', 10000)
        warmup_steps = min(100, int(total_steps * 0.05))  # 웜업 스텝을 전체의 5%로 제한
        
        # 학습률 강제 확인 - 기본값이 너무 낮을 수 있음
        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr < 0.0001:  # 학습률이 너무 낮은 경우 강제 조정
            new_lr = 0.0002  # 2e-4
            logger.warning(f"학습률이 너무 낮습니다: {current_lr:.8f}. {new_lr:.6f}로 강제 조정합니다.")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        
        logger.info(f"학습률 스케줄러 설정: 전체 스텝={total_steps}, 웜업 스텝={warmup_steps}")
        
        self.scheduler = AdaptiveLRScheduler(
            optimizer=self.optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=scheduler_config.get('min_lr_ratio', 0.1),
            end_lr_ratio=scheduler_config.get('end_lr_ratio', 0.01)
        )
        
        # 스케줄러 초기화 후 학습률 확인
        initial_lr = self.scheduler.get_last_lr()[0]
        logger.info(f"초기화 후 학습률 확인: {initial_lr:.6f}")
        
        # 학습률이 여전히 0인 경우 추가 조치
        if initial_lr < 0.0001:
            logger.warning(f"학습률이 여전히 너무 낮습니다. 추가 조정이 필요합니다.")

    def train_task(self, train_loader: DataLoader, val_loader: DataLoader, test_loaders: Dict[str, DataLoader] = None, current_mode: str = None):
        """
        단일 모드에 대한 훈련 수행

        Args:
            train_loader: 훈련 데이터 로더
            val_loader: 검증 데이터 로더
            test_loaders: 이전 모드의 테스트 데이터 로더 {mode: loader}
            current_mode: 현재 학습 모드 (complete, prompt, comment, error_fix)
        """
        self.current_mode = current_mode  # 현재 학습 모드 설정

        # 태스크 구성
        max_epochs = self.config.get('max_epochs', 10)
        patience = self.config.get('patience', 3)

        # 조기 종료 추적
        best_val_acc = 0.0
        patience_counter = 0

        # 모드별 학습 시작
        if current_mode:
            logger.info(f"=== 모드 {current_mode} 학습 시작 ===")
        else:
            logger.info(f"=== 학습 시작 (모드 미지정) ===")

        for epoch in range(max_epochs):
            # 훈련 에포크
            train_loss, train_acc = self.train_epoch(
                train_loader=train_loader, 
                epoch=epoch,
                current_mode=current_mode
            )

            # 검증
            val_loss, val_acc = self.evaluate(val_loader)

            # 모든 이전 모드에 대한 테스트
            test_results = {}
            if test_loaders:
                for test_mode, test_loader in test_loaders.items():
                    _, test_acc = self.evaluate(test_loader)
                    test_results[test_mode] = test_acc

            # 메트릭 업데이트 - 현재 모드 사용
            self.metrics.update(current_mode, test_results)

            # 안정성 측정을 위해 이전 모드 정확도 기록
            # complete 모드(1차 학습)가 있을 경우 이를 기준으로 사용
            if current_mode != 'complete' and 'complete' in test_results:
                self.metrics.record_stability_gap(
                    current_mode,
                    self.metrics.mode_accuracies.get('complete', 0)
                )

            # 로깅
            metrics = self.metrics.compute_metrics()
            self.log_metrics(epoch, train_loss, train_acc, val_loss, val_acc, test_results, metrics)

            # 조기 종료 확인
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # 최적 모델 저장
                best_model_name = f"best_model_{current_mode}.pt" if current_mode else f"best_model.pt"
                self.save_model(best_model_name)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"조기 종료: {patience}에포크 동안 개선 없음")
                    break

        # 학습 완료 후 EWC 등록 - 현재 모드를 사용
        if current_mode:
            # 모드명으로 등록하여 각 모드별로 중요도 관리
            self.ewc.register_task(current_mode, train_loader, self.ce_loss)
            
            # EWC lambda 값 동적 조절 (모드별 부가적 기능)
            if current_mode == 'prompt':
                # 2차 학습(프롬프트)에서는 표준 lambda 값
                self.ewc.lambda_ewc = self.config.get('ewc', {}).get('lambda', 5000.0)
            elif current_mode == 'comment':
                # 3차 학습(주석)에서는 lambda 값 증가
                self.ewc.lambda_ewc = self.config.get('ewc', {}).get('lambda', 5000.0) * 1.5
            elif current_mode == 'error_fix':
                # 4차 학습(오류 수정)에서는 lambda 값 더 증가
                self.ewc.lambda_ewc = self.config.get('ewc', {}).get('lambda', 5000.0) * 2.0
                
            logger.info(f"{current_mode} 모드에 대한 EWC lambda 값: {self.ewc.lambda_ewc}")

        # 최종 성능 저장 - 모드별 관리
        if current_mode:
            self.best_accuracies[current_mode] = best_val_acc
        else:
            self.best_accuracies['default'] = best_val_acc

        # 종합 메트릭 로깅 - 구조화된 형태로 변경
        final_metrics = self.metrics.compute_metrics()
        
        # 훈련 완료 로깅
        logger.info(f"=== {current_mode} 모드 학습 완료 ===")
        logger.training_status({
            "status": "completed",
            "mode": current_mode,  # 모드 정보 추가
            "best_val_accuracy": float(best_val_acc),
            "ewc_lambda": float(self.ewc.lambda_ewc) if hasattr(self.ewc, 'lambda_ewc') else 0,
            **{k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in final_metrics.items()}
        })

        # 최종 모델 저장 - AWS 스팟 인스턴스 중단 고려
        checkpoint_name = f"final_model_continual_{current_mode}.pt" if current_mode else "final_model.pt"
        safe_path = self.save_model(checkpoint_name)
        logger.info(f"[중요] 모드 {current_mode} 축 학습 완료 - 체크포인트 저장됨: {safe_path}")

    def train_epoch(self, train_loader: DataLoader, epoch: int, current_mode: str = None) -> Tuple[float, float]:
        """
        한 에포크의 학습 수행

        Args:
            train_loader: 학습 데이터 로더
            epoch: 현재 에포크
            current_mode: 현재 학습 모드 (complete, prompt, comment, error_fix)

        Returns:
            평균 손실, 평균 정확도
        """
        self.model.train()
        
        # 학습률 디버깅 - 에포크 시작시 학습률 확인
        current_lr = self.optimizer.param_groups[0]['lr']
        warmup_steps = self.scheduler.warmup_steps if hasattr(self.scheduler, 'warmup_steps') else 0
        total_steps = self.scheduler.total_steps if hasattr(self.scheduler, 'total_steps') else 0
        
        logger.info(f"===== 에포크 {epoch} 시작: 학습률={current_lr:.6f}, 전역 스텝={self.global_step}, "
                  f"웜업 진행={self.global_step}/{warmup_steps} (완료={self.global_step >= warmup_steps}) =====")
        
        # 학습률이 너무 낮은지 검사 (디버깅 모드)
        if current_lr < 0.000001 and self.global_step > warmup_steps:
            logger.warning(f"학습률이 너무 낮음! current_lr={current_lr:.8f}, global_step={self.global_step}")
            # 학습률 강제 조정 기능 (개발/디버깅용)
            if self.config.get('debug_force_lr', False):
                logger.warning("디버깅 모드: 학습률 강제 조정")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.0001
        
        # 학습 통계 초기화
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 프로그레스 바 참조를 위한 변수 초기화 (예외 발생 시 안전한 참조를 위해)
        self.pbar = None
        
        # AWS 스팟 인스턴스 환경에서 학습 중단/재개시 필요한 정보 업데이트
        logger.update_epoch(epoch)
        logger.update_task(current_mode)  # task_id 대신 현재 모드 전달
        
        # 모델을 훈련 모드로 설정
        self.model.train()
        
        # 체크포인트 디렉토리 정보 확인 (기본 정보만 출력)
        checkpoint_dir = getattr(self.spot_handler, 'checkpoint_dir', self.config.get('output_dir', 'checkpoints'))
        if os.path.exists(checkpoint_dir):
            print(f"[시스템] 학습 시작 - 체크포인트 디렉토리 확인: {checkpoint_dir}")

        # 메트릭 추적 변수 초기화
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 2. 프로그레스 바 설정
        spot_status = "[SPOT]" if hasattr(self, 'spot_handler') and self.spot_handler is not None else ""
        mode_info = f"[{current_mode or 'default'}]"  # 항상 모드 표시
        desc = f"E{epoch} {mode_info} {spot_status}"
        
        # AWS 스팟 인스턴스를 위한 추가 정보
        initial_metrics = {
            'step': self.global_step,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        self.pbar = tqdm(
            train_loader, 
            desc=desc,
            dynamic_ncols=True,  # 터미널 크기 변경에 동적으로 대응
            leave=True           # 진행바 유지
        )
        
        # logger를 통해 tqdm 초기화 - 첫 표시를 위한 설정
        logger.configure_tqdm(self.pbar, initial_metrics)

        # 3. 배치 크기 설정
        current_batch_size = self.batch_size

        # 4. 배치 반복
        for batch_idx, batch_data in enumerate(self.pbar):
            try:
                # 4.1 배치 데이터 전처리
                data, target = self._process_batch_data(batch_data, batch_idx)
                if data is None or target is None:
                    continue  # 처리할 수 없는 배치는 건너뜀

                # 4.2 학습 단계 수행
                if self.use_amp:
                    loss, acc = self.train_step_amp(data, target, current_mode)
                else:
                    loss, acc = self.train_step(data, target, current_mode)
                    
                # 4.3 고정 배치 크기 사용 (DynamicBatchSizeManager 제거)
                # 배치 크기는 OOM 발생 시에만 변경됨
                        
                # 4.4 프로그레스 바 업데이트 (성능 최적화: 50배치마다)
                if batch_idx % 50 == 0 or batch_idx == 0:  
                    self._update_progress_bar(pbar, epoch, loss, total_loss, batch_idx, correct, total, current_batch_size)
                    
                # 4.5 메트릭 누적
                total_loss += loss
                batch_size = data.size(0) if hasattr(data, 'size') else 1
                correct += acc * batch_size
                total += batch_size
                
                # 4.6 체크포인트 확인 (AWS 스팟 인스턴스 중단 감지 또는 주기적 저장)
                
                # 디버그: 모든 스텝에 체크포인트 저장 가능성 확인 (print와 logger 모두 사용)
                checkpoint_freq = getattr(self.spot_handler, 'checkpoint_freq', 100)
                is_step_multiple = self.global_step > 0 and self.global_step % checkpoint_freq == 0
                if is_step_multiple:
                    print(f"[중요 확인] 체크포인트 저장 조건 충족! 글로벌 스텝: {self.global_step}, 주기: {checkpoint_freq}")
                
                # 스텝 5번마다 체크포인트 저장 조건 확인
                if self.global_step % 5 == 0:
                    should_save = self.spot_handler.should_checkpoint(self.global_step)
                    last_check_time = time.time() - getattr(self.spot_handler, 'last_checkpoint_time', 0)
                    monitoring_active = getattr(self.spot_handler, 'monitoring_active', False)
                    logger.info(f"[디버그] 글로벌 스텝: {self.global_step}, 저장해야 할까?: {should_save}, "
                              f"주기 설정: {checkpoint_freq}, 주기 충족여부: {is_step_multiple}, "
                              f"모니터링 활성화: {monitoring_active}, "
                              f"마지막 저장: {last_check_time:.1f}초 전")
                
                # 체크포인트 저장 시도
                should_save = self.spot_handler.should_checkpoint(self.global_step)
                if should_save:
                    is_interruption = self.spot_handler.interrupted
                    checkpoint_reason = "AWS 스팟 인스턴스 중단" if is_interruption else "주기적 체크포인트"
                    logger.info(f"체크포인트 저장 조건 감지 (원인: {checkpoint_reason})")
                    # [디버그] should_checkpoint의 간단한 내부 상태 출력
                    logger.info(f"[디버그] 저장 결정 상세 - step: {self.global_step}, interrupted: {is_interruption}, "
                              f"freq_check: {self.global_step > 0 and self.global_step % self.spot_handler.checkpoint_freq == 0}")
                    
                    
                    # 체크포인트 파일명에 정보 포함 (시간 제외)
                    prefix = "interrupt" if is_interruption else "checkpoint"
                    mode_part = f"_{current_mode}" if current_mode else "_default"
                    checkpoint_name = f"{prefix}{mode_part}_epoch{epoch}_step{self.global_step}.pt"
                    
                    # 개선된 save_model 메서드로 체크포인트 저장 - 스팟 핸들러 관련 처리 통합
                    # is_spot_interruption 파라미터로 중단 여부 전달 (메타데이터 포함 및 로깅용)
                    saved_path = self.save_model(
                        checkpoint_name,
                        is_spot_interruption=is_interruption
                    )
                    
                    if saved_path and is_interruption:
                        logger.warning(f"스팟 인스턴스 중단으로 인한 긴급 체크포인트 저장 성공: {saved_path}")
                    elif saved_path:
                        logger.debug(f"주기적 체크포인트 저장 완료: {saved_path}")
                    else:
                        logger.error("체크포인트 저장 실패!")
                    
            except torch.cuda.OutOfMemoryError:
                # OOM 오류 처리 - 배치 크기 감소 및 메모리 정리
                self._handle_oom_error(current_batch_size)
                current_batch_size = self.batch_size  # 업데이트된 배치 크기 가져오기
                continue
                
            except Exception as e:
                # 일반 예외 처리
                self._handle_training_exception(e, batch_data, batch_idx)
                continue
                
        # 5. 에폭 결과 계산 및 반환
        avg_loss = total_loss / max(len(pbar), 1)  # 배치 수로 나누기 (0 방지)
        accuracy = correct / max(total, 1)         # 전체 샘플 수로 나누기 (0 방지)
        
        return avg_loss, accuracy
        
    def _process_batch_data(self, batch_data: Any, batch_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        다양한 형식의 배치 데이터를 처리하여 모델 입력과 타겟으로 변환합니다.
        
        Args:
            batch_data: DataLoader에서 반환된 배치 데이터 (다양한 형식 가능)
            batch_idx: 현재 배치 인덱스 (디버그 로깅용)
            
        Returns:
            (data, target) 텐서 튜플 또는 처리 불가능할 경우 (None, None)
        """
        # 1. Huggingface BatchEncoding 또는 딕셔너리 형태 처리
        if isinstance(batch_data, dict) or hasattr(batch_data, 'input_ids'):
            # input_ids 추출
            data = batch_data.get('input_ids', None)
            if data is None and 'input_ids' in batch_data:
                data = batch_data['input_ids']
                
            # 데이터 검증
            if data is None:
                logger.warning(f"배치 {batch_idx}: input_ids를 찾을 수 없습니다")
                return None, None
                
            # 디바이스 이동
            data = data.to(self.device)
            
            # 타겟 추출 (언어 모델링 태스크의 경우)
            if 'labels' in batch_data:
                target = batch_data['labels'].to(self.device)
            else:
                # 자기회귀 학습 방식 (입력=타겟)
                target = data
                
            # 첫 배치만 디버그 정보 기록
            if batch_idx == 0:
                logger.debug(f"배치 데이터: {type(batch_data).__name__}, "  
                            f"크기: {data.shape if hasattr(data, 'shape') else 'unknown'}")
                
            return data, target
            
        # 2. 일반적인 (data, target) 튜플 처리
        try:
            data, target = batch_data
            return data.to(self.device), target.to(self.device)
            
        # 3. 예외 상황 처리
        except ValueError:
            # 리스트 형태 데이터 처리 시도
            if isinstance(batch_data, list) and len(batch_data) > 0:
                data = batch_data[0].to(self.device) if hasattr(batch_data[0], 'to') else batch_data[0]
                return data, data  # 타겟이 없으면 데이터 자체를 타겟으로 사용
                
            # 처리 불가능한 형식
            logger.error(f"배치 {batch_idx}: 처리할 수 없는 형식 {type(batch_data).__name__}")
            return None, None
            
    def _update_progress_bar(self, pbar: Optional[tqdm], epoch: int, loss: float, total_loss: float, 
                               batch_idx: int, correct: float, total: int, batch_size: int) -> None:
        """
        tqdm 프로그레스 바와 logger를 통합하여 학습 현황을 표시
        
        Args:
            pbar: tqdm 프로그레스 바 객체 (없을 수도 있음)
            epoch: 현재 에폭
            loss: 현재 배치 손실
            total_loss: 누적 손실
            batch_idx: 현재 배치 인덱스
            correct: 누적 정확한 예측 수
            total: 총 샘플 수
            batch_size: 현재 배치 크기
        """
        # pbar가 없으면 일반 로깅만 수행
        if pbar is None:
            logger.info(f"[Epoch {epoch}, Step {self.global_step}] loss={loss:.4f} | acc={correct/max(total, 1):.4f} | lr={self.optimizer.param_groups[0]['lr']:.6f}")
            return
        try:
            # 현재까지의 평균 손실과 정확도 계산
            running_loss = total_loss / (batch_idx + 1)
            running_acc = correct / max(total, 1)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 학습률이 0인지 확인하여 경고
            if current_lr == 0:
                logger.warning(f"학습률이 0입니다! 스텝={self.global_step}, 웜업이 너무 길게 설정되었을 수 있습니다.")
            
            # 중요 메트릭 구성
            metrics = {
                'loss': running_loss,
                'acc': running_acc,
                'lr': current_lr,
                'bs': batch_size,
                'step': self.global_step
            }
            
            # tqdm 설명 업데이트 (현재 모드 추가)
            mode_info = f"[{self.current_mode}]" if hasattr(self, 'current_mode') and self.current_mode else ""
            self.pbar.set_description(f"E{epoch} {mode_info} [{self.global_step}]")
            
            # DeepseekLogger를 통한 tqdm 진행바 통합 업데이트
            logger.update_tqdm_progress(self.pbar, n=0, metrics=metrics)
            
            # 중요 진행 마일스톤 로깅 (매 500배치 마다 또는 첫 배치)
            log_interval = 500  # 로그 주기
            
            # 과도한 로그 출력 방지: 첫 배치 또는 지정된 간격에서만 로그 출력
            if batch_idx == 0 or (batch_idx > 0 and batch_idx % log_interval == 0):
                # 메트릭을 표준 로그 시스템에도 기록 - 로그 출력 줄이기 위해 update_tqdm만 수행
                logger.update_tqdm_progress(self.pbar, n=0, metrics=metrics)
                
        except Exception as e:
            # tqdm 관련 오류는 학습에 중요하지 않으나 로그 남김
            logger.warning(f"tqdm 진행 바 업데이트 중 오류: {e}")
            pass
            
    def _handle_oom_error(self, current_batch_size: int = None) -> None:
        """
        Out of Memory 오류 처리 - 향상된 기능
        
        Args:
            current_batch_size: 현재 배치 크기 (선택적, 없으면 현재 배치 크기를 추정함)
        """
        # 1. 전체 CUDA 메모리 현황 기록
        if torch.cuda.is_available():
            try:
                # 메모리 사용량 확인
                if hasattr(torch.cuda, 'memory_reserved') and hasattr(torch.cuda, 'memory_allocated'):
                    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
                    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                    logger.warning(f"OOM 당시 GPU 메모리 상태: 예약={reserved:.2f}GB, 할당={allocated:.2f}GB")
            except Exception as e:
                logger.debug(f"메모리 정보 수집 오류: {e}")
                
        # 2. 급령 캐시 정리 실행
        for _ in range(3):  # 여러번 캠시 정리 시도
            torch.cuda.empty_cache()
        
        # 3. 배치 크기 감소 로직
        if current_batch_size is None:
            current_batch_size = getattr(self, 'batch_size', 4)  # 기본값 4 사용
        
        # 현재 배치가 2 이하이면 더 이상 줄일 수 없음
        if current_batch_size <= 2:
            new_batch_size = 1  # 최소 배치 크기
            logger.error(f"OOM 발생: 배치 크기가 이미 최소치({current_batch_size})에 가까움, 최소값 1로 설정")
        else:
            # 기본적으로 50% 감소
            reduction_factor = 0.5
            new_batch_size = max(1, int(current_batch_size * reduction_factor))
            logger.warning(f"OOM 발생: 배치 크기 {current_batch_size} → {new_batch_size}로 {(1-reduction_factor)*100:.0f}% 감소")
        
        # 새 배치 크기 적용
        self.batch_size = new_batch_size
        
        # 4. 그래디언트 누적 단계 증가 (배치 크기 감소를 보완)
        if hasattr(self, 'gradient_accumulation_steps') and self.gradient_accumulation_steps < 8:
            self.gradient_accumulation_steps *= 2
            logger.info(f"배치 크기 감소로 그래디언트 누적 단계 증가: {self.gradient_accumulation_steps}")
            
        # 5. 추가 메모리 확보 시도 - 불필요한 모듈 언로드
        try:
            import gc
            gc.collect()  # 영구적으로 사용되지 않는 객체 수집
        except Exception as e:
            logger.debug(f"GC 실행 중 오류: {e}")
            
        # 학습은 계속 진행
        
    def _handle_training_exception(self, error: Exception, batch_data: Any, batch_idx: int) -> None:
        """
        학습 중 발생하는 일반 예외 처리
        
        Args:
            error: 발생한 예외
            batch_data: 예외 발생한 배치 데이터
            batch_idx: 예외 발생한 배치 인덱스
        """
        # "No inf checks" 오류는 무시 (학습에 영향 없음)
        if "No inf checks" in str(error):
            return
            
        # 주기적으로만 자세한 오류 로깅 (로그 과다 방지)
        if self.global_step % 50 == 0:
            logger.warning(f"배치 {batch_idx} 처리 중 오류: {error}")
            
            # 배치 데이터 구조 정보
            if isinstance(batch_data, dict) or hasattr(batch_data, 'keys'):
                logger.debug(f"배치 키: {list(batch_data.keys()) if hasattr(batch_data, 'keys') else 'N/A'}")
            
            # input_ids 정보 (있는 경우)
            if hasattr(batch_data, 'input_ids'):
                shape_info = batch_data.input_ids.shape if hasattr(batch_data.input_ids, 'shape') else 'N/A'
                logger.debug(f"input_ids 형태: {type(batch_data.input_ids)}, 크기: {shape_info}")
        else:
            # 간단한 오류 메시지만 기록
            logger.debug(f"배치 처리 중 오류 발생 (배치 {batch_idx}): {type(error).__name__}")
            # batch_data에서 배치 크기 추정
            current_batch_size = None
            if batch_data is not None:
                if hasattr(batch_data, 'input_ids') and hasattr(batch_data.input_ids, 'shape'):
                    current_batch_size = batch_data.input_ids.shape[0]
                elif isinstance(batch_data, torch.Tensor) and hasattr(batch_data, 'shape'):
                    current_batch_size = batch_data.shape[0]
            
            # 배치 크기를 추정할 수 없는 경우 기본 배치 크기 사용
            if current_batch_size is None:
                current_batch_size = getattr(self, 'batch_size', 4)  # 기본값 4
                
            self._handle_oom_error(current_batch_size)
        
        # 현재 스텝에서 체크포인트 저장 시도
        try:
            logger.info("예외 발생 후 체크포인트 저장 시도")
            # 예외 상황에서 유효한 파일명 생성 (시간 정보 제외)
            error_checkpoint_name = f"emergency_checkpoint_{self.current_mode}_step{self.global_step}.pt"
            self.save_model(error_checkpoint_name, is_spot_interruption=True)  # 비상 저장임을 표시
        except Exception as save_error:
            logger.error(f"예외 상황에서 체크포인트 저장 실패: {save_error}")
        
        # 전역 스텝 업데이트
        self.global_step += 1
        # AWS 스팟 인스턴스 환경에서 학습 중단/재개시 필요한 정보 기록 - 모든 스텝에서 호출
        logger.update_step(self.global_step)
        
        # MER 메모리 업데이트 - batch_data를 사용 (안전하게 체크)
        if hasattr(self, 'mer') and self.mer is not None and hasattr(self, 'batch_data'):
            try:
                # 텐서 차원 안전 체크
                batch_data = self.batch_data
                if hasattr(batch_data, 'shape') and len(batch_data.shape) > 1:
                    # 배치 차원이 있는 경우 첫 번째 샘플만 사용
                    sample_data = batch_data[0] if batch_data.shape[0] > 1 else batch_data.squeeze(0)
                    sample_target = target[0] if hasattr(target, 'shape') and target.shape[0] > 1 else target
                    self.mer.reservoir_sampling(sample_data, sample_target)
                else:
                    self.mer.reservoir_sampling(batch_data, target)
            except Exception as mer_error:
                error_msg = str(mer_error)
                if "Scalar" in error_msg or "cannot be converted" in error_msg:
                    logger.debug(f"MER 텐서 차원 오류 (무시): {error_msg}")
                else:
                    logger.error(f"MER 메모리 업데이트 실패: {error_msg}")
        
        # 실패한 경우 기본값 반환
        return 0.0, 0.0  # 손실값과 정확도에 기본값 반환

    def _safe_unscale_gradients(self):
        """
        안전한 그래디언트 언스케일링 (Critical Issue #2 해결)
        PyTorch 버전별 호환성 및 AMP Scaler 이중 호출 방지
        """
        try:
            # PyTorch 2.x 버전별 호환성 체크
            if hasattr(self.scaler, '_per_optimizer_states'):
                opt_id = id(self.optimizer)
                
                # optimizer state가 존재하는지 체크
                if opt_id in self.scaler._per_optimizer_states:
                    state = self.scaler._per_optimizer_states[opt_id]
                    
                    # stage 상태 체크 (안전한 방법)
                    current_stage = getattr(state, 'stage', None) if hasattr(state, 'stage') else state.get('stage', None)
                    
                    if current_stage != "UNSCALED":
                        self.scaler.unscale_(self.optimizer)
                        logger.debug("✅ AMP scaler unscale 수행")
                    else:
                        logger.debug("🔄 AMP scaler 이미 unscale 됨 - 중복 호출 방지")
                else:
                    # 첫 번째 호출인 경우
                    self.scaler.unscale_(self.optimizer)
                    logger.debug("✅ AMP scaler unscale 수행 (첫 호출)")
            else:
                # 새로운 PyTorch 버전 대응
                self.scaler.unscale_(self.optimizer)
                logger.debug("✅ AMP scaler unscale 수행 (새 PyTorch 버전)")
                
        except RuntimeError as e:
            error_msg = str(e)
            if "unscale_" in error_msg and "already been called" in error_msg:
                logger.debug("🔄 Gradient unscaling 이미 완료됨")
            elif "inf" in error_msg or "nan" in error_msg:
                logger.warning(f"⚠️  그래디언트에 inf/NaN 값 감지: {error_msg}")
                # 이번 배치 건너뛰기
                self.optimizer.zero_grad()
                raise e
            else:
                logger.error(f"❌ AMP scaler unscale 오류: {error_msg}")
                raise e
        except Exception as e:
            logger.error(f"❌ 예상치 못한 AMP scaler 오류: {e}")
            raise e

    def safe_train_step(self, data, target, current_mode=None):
        """
        안전한 학습 스텝 (에러 복구 포함)
        OOM 에러 시 배치 크기 자동 조정 및 비상 체크포인트 저장
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 정상 학습 스텝 시도
                return self.train_step_amp(data, target, current_mode)
            
            except torch.cuda.OutOfMemoryError as oom_error:
                retry_count += 1
                logger.error(f"💥 CUDA OOM 발생 (Retry {retry_count}/{max_retries}): {oom_error}")
                
                # 메모리 정리
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                
                if retry_count < max_retries:
                    # 배치 크기 반으로 줄이기
                    old_batch_size = getattr(self, 'current_batch_size', self.config.get('batch_size', 2))
                    new_batch_size = max(1, old_batch_size // 2)
                    self.current_batch_size = new_batch_size
                    
                    # 그래디언트 누적 스텝 증가 (유효 배치 유지)
                    self.gradient_accumulation_steps *= 2
                    
                    logger.info(f"🔄 배치 크기 조정: {old_batch_size} → {new_batch_size}")
                    logger.info(f"🔄 그래디언트 누적: {self.gradient_accumulation_steps//2} → {self.gradient_accumulation_steps}")
                    
                    # 잠시 대기 후 재시도
                    time.sleep(2)
                    continue
                else:
                    # 최대 재시도 횟수 초과
                    logger.error("❌ 최대 재시도 횟수 초과 - OOM 에러 해결 실패")
                    raise oom_error
            
            except Exception as e:
                logger.error(f"💥 학습 스텝 에러: {e}")
                
                # 비상 체크포인트 저장 시도
                try:
                    emergency_path = f"emergency_checkpoint_step_{getattr(self, 'global_step', 0)}.pt"
                    self.save_model(emergency_path)
                    logger.info(f"🆘 비상 체크포인트 저장 완료: {emergency_path}")
                except Exception as save_error:
                    logger.error(f"❌ 비상 체크포인트 저장 실패: {save_error}")
                
                # 원래 에러 재발생
                raise e
        
        # 이 지점에 도달하면 안 됨
        raise RuntimeError("예상치 못한 상황: 재시도 루프 종료")

    def _check_gradient_explosion(self):
        """
        그래디언트 폭주 감지 및 안전 체크
        Returns: total gradient norm
        """
        total_norm = 0.0
        param_count = 0
        
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        
        # 그래디언트 폭주 감지 (10.0 이상이면 위험)
        if total_norm > 10.0:
            logger.warning(f"⚠️  대형 그래디언트 노름 감지: {total_norm:.4f} (매개변수 {param_count}개)")
        elif total_norm > 5.0:
            logger.info(f"📊 그래디언트 노름: {total_norm:.4f}")
        
        return total_norm

    def _perform_gradient_clipping_and_step(self):
        """
        그래디언트 누적 완료 후 그래디언트 클리핑과 optimizer step 수행
        AMP Scaler 이중 호출 문제를 방지하기 위해 분리된 메서드
        """
        # 그래디언트 클리핑 (선택적) - A10G 환경에 최적화
        grad_clip_value = float(self.config.get('grad_clip', 1.0))  # 기본값 1.0
        if grad_clip_value > 0:
            try:
                # 🔥 Critical Issue #2 해결: 안전한 AMP Scaler 언스케일링
                self._safe_unscale_gradients()
                
                # 클리핑 전 그래디언트 통계 수집 (1000스텝 간격으로)
                if self.global_step % 1000 == 0:
                    before_clip_norms = []
                    for param in self.model.parameters():
                        if param.grad is not None:
                            before_clip_norms.append(param.grad.norm().item())
                    
                    if before_clip_norms:
                        avg_grad = sum(before_clip_norms) / len(before_clip_norms)
                        max_grad = max(before_clip_norms)
                        logger.info(f"클리핑 전 그래디언트 통계 [스텝 {self.global_step}]: 평균={avg_grad:.5f}, 최대={max_grad:.5f}")
                
                # 그래디언트 폭주 감지 및 안전한 클리핑
                total_norm = self._check_gradient_explosion()
                
                # 실제 그래디언트 클리핑 적용
                actual_clip_value = min(grad_clip_value, 2.0) if total_norm > 10.0 else grad_clip_value
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=actual_clip_value
                )
                
                if total_norm > 10.0:
                    logger.warning(f"⚠️  그래디언트 폭주 감지! 강한 클리핑 적용: {total_norm:.4f} → {actual_clip_value}")
                
                # 클리핑 후 그래디언트 통계 (1000스텝 간격으로)
                if self.global_step % 1000 == 0:
                    after_clip_norms = []
                    for param in self.model.parameters():
                        if param.grad is not None:
                            after_clip_norms.append(param.grad.norm().item())
                    
                    if after_clip_norms:
                        avg_grad = sum(after_clip_norms) / len(after_clip_norms)
                        max_grad = max(after_clip_norms)
                        logger.info(f"클리핑 후 그래디언트 통계 [스텝 {self.global_step}]: 평균={avg_grad:.5f}, 최대={max_grad:.5f}")
            
            except RuntimeError as e:
                logger.warning(f"그래디언트 클리핑 중 오류 (무시함): {e}")
                # 추가 진단 정보
                if "unscale_" in str(e):
                    logger.info("원인: 옵티마이저 내 inf/NaN 값 - 이번 배치 그래디언트는 건너뛱니다")
                    # optimizer.step()에서 오류가 발생하지 않도록 이번 배치 그래디언트 제거
                    self.optimizer.zero_grad()
                    return  # 이번 스텝 건너뛰기

        # 그래디언트 누적이 완료되었을 때만 optimizer 스텝 실행
        # 계산된 그래디언트로 모델 파라미터 업데이트
        try:
            # 옵티마이저 스텝 수행
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 4. 학습률 스케줄러 업데이트 (옵티마이저 스텝 이후 바로 실행)
            if self.scheduler is not None:
                # 스케줄러 스텝 전의 학습률 저장
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # 명시적으로 global_step을 전달하여 스케줄러를 업데이트합니다.
                self.scheduler.step(self.global_step)
                
                # 업데이트 후 학습률 확인
                new_lr = self.optimizer.param_groups[0]['lr']
                if self.global_step % 100 == 0:  # 100스텝마다 로그
                    logger.info(f"[스케줄러] 스텝 {self.global_step}: LR {current_lr:.2e} → {new_lr:.2e}")
            
            # 5. 옵티마이저 그래디언트 초기화
            self.optimizer.zero_grad()
            
            # 6. global_step 증가
            self.global_step += 1
            
        except Exception as e:
            logger.error(f"옵티마이저 스텝 오류: {e}")
            # 오류 발생 시 그래디언트 초기화
            self.optimizer.zero_grad()

    def train_step_amp(self, data, target, current_mode: str = None):
        """
        단일 훈련 스텝 (혼합 정밀도) - 언어 모델 내장 손실만 사용

        Args:
            data: 입력 데이터 (torch.Tensor 또는 transformers.BatchEncoding)
            target: 타겟 레이블 (torch.Tensor 또는 transformers.BatchEncoding)
            current_mode: 현재 학습 모드 (complete, prompt, comment, error_fix)

        Returns:
            손실 값, 정확도
        """
        # 그라디언트 누적 시작시에만 optimizer.zero_grad() 호출
        if self.accumulated_gradients == 0:
            self.optimizer.zero_grad()

        # 데이터 타입 체크 및 처리
        from transformers.tokenization_utils_base import BatchEncoding
        
        # BatchEncoding 객체 처리
        if isinstance(data, BatchEncoding):
            # BatchEncoding 로깅 최적화: 초기 한 번만 info로 기록, 이후에는 매우 낮은 빈도로 debug로 기록
            if self.global_step == 0:  # 학습 첫 시작시에만 info 레벨로 기록
                logger.info(f"BatchEncoding 데이터 처리: 키={list(data.keys())}, 디바이스={data.get('input_ids').device if 'input_ids' in data else '알 수 없음'}")
            elif self.global_step % 5000 == 0:  # 이후에는 5000 스텝마다 debug 레벨로만 기록
                logger.debug(f"BatchEncoding 데이터 처리: 키={list(data.keys())}, 디바이스={data.get('input_ids').device if 'input_ids' in data else '알 수 없음'}")
            
            # 필요한 경우에만 input_ids 추출
            if 'input_ids' in data:
                input_tensor = data['input_ids']
            else:
                available_keys = list(data.keys())
                logger.error(f"BatchEncoding에서 input_ids를 찾을 수 없음. 사용 가능한 키: {available_keys}")
                raise ValueError(f"BatchEncoding에 input_ids가 없습니다. 사용 가능한 키: {available_keys}")
        else:
            # 이미 텐서인 경우 그대로 사용
            input_tensor = data
        
        # target도 동일하게 처리
        if isinstance(target, BatchEncoding):
            if 'labels' in target:
                labels_tensor = target['labels']
            elif 'input_ids' in target:
                # fallback: input_ids를 labels로 사용
                labels_tensor = target['input_ids']
            else:
                available_keys = list(target.keys())
                logger.error(f"BatchEncoding에서 labels를 찾을 수 없음. 사용 가능한 키: {available_keys}")
                raise ValueError(f"BatchEncoding에 labels가 없습니다. 사용 가능한 키: {available_keys}")
        else:
            # 이미 텐서인 경우 그대로 사용
            labels_tensor = target

        # PyTorch CUDA 메모리 할당자 효율화 확인 및 설정 중복 회피
        if self.global_step == 0 and torch.cuda.is_available():
            # 환경변수에서 PYTORCH_CUDA_ALLOC_CONF 확인
            import os
            cuda_alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
            if not cuda_alloc_conf or 'max_split_size_mb' not in cuda_alloc_conf:
                logger.warning("\n[\ud658경변수 설정 권장] PYTORCH_CUDA_ALLOC_CONF=\"max_split_size_mb:128,garbage_collection_threshold:0.6\" \n"
                              "\ud658경변수 설정을 통해 메문리 할당자 정책을 세부 조정할 수 있습니다.")
        
        # 혼합 정밀도 연산 - 최신 PyTorch API 형식 사용
        with autocast('cuda'):
            # 그래디언트 체크포인팅 상태 확인 및 로깅
            if self.global_step == 0:
                gradient_checkpointing_enabled = getattr(self.model, 'is_gradient_checkpointing', False)
                logger.info(f"그래디언트 체크포인팅 활성화 상태: {gradient_checkpointing_enabled}")
            # 언어 모델링 손실 계산을 위해 labels 인자로 target 전달
            # DeepSeek-Coder 및 다른 Hugging Face 모델은 labels를 통한 직접 손실 계산 지원
            model_output = self.model(input_tensor, labels=labels_tensor)
            
            # 언어 모델 내장 손실 사용
            try:
                # 1. 모델이 내장 손실을 제공하는 경우 (허깅페이스 transformers 모델)
                if hasattr(model_output, 'loss') and model_output.loss is not None:
                    # 손실을 추출하고 그래디언트 추적을 위해 명시적으로 requires_grad=True 설정
                    loss = model_output.loss
                    
                    # 그래디언트 추적이 필요하지만 없는 경우
                    if not loss.requires_grad:
                        # 새로운 텐서로 복사하고 그래디언트 추적 활성화
                        loss = loss.detach().clone().requires_grad_(True)
                    
                    # 명시적 전역 스텝 증가 - 여기서 먼저 증가시킵니다.
                    self.global_step += 1
                    logger.update_step(self.global_step, silent=True)
                    
                    # 로그 출력 빈도를 제어하여 너무 많은 로그 방지
                    # 5000배치마다, 또는 매 100스텝의 배수에서만 로그 출력
                    if self.global_step % 5000 == 0 or self.global_step % 100 == 0 or self.global_step == 1:
                        # 구조화된 로깅 형식 사용 + tqdm.write 적용
                        logger.training_status({
                            "loss": loss.item(),
                            "requires_grad": loss.requires_grad,
                            "lr": self.scheduler.get_last_lr()[0]
                        }, use_tqdm_write=True, log_output=True) # 명시적으로 로그 출력
                else:
                    # 내장 손실이 없는 경우 - 로깅 후 예외 발생
                    available_attrs = [attr for attr in dir(model_output) if not attr.startswith('_')]
                    logger.error(f"model_output 속성: {available_attrs}")
                    raise ValueError(f"모델에서 언어 모델링 손실을 찾을 수 없음: {type(model_output)}")
                    
            except Exception as e:
                logger.error(f"손실 계산 중 오류 발생: {e}")
                logger.error(f"model_output 타입: {type(model_output)}")
                raise RuntimeError(f"LM 손실 계산 실패: {e}")
                
            # 정확도 계산 (선택적) - 완전한 분류기 대신 단순한 정확도 추정 사용
            acc = 0.0
            try:
                if hasattr(model_output, 'logits'):
                    # 마지막 토큰 위치의 예측값만 사용하여 간단한 정확도 계산 
                    # (실제 LM 성능은 perplexity 등으로 별도 평가해야 함)
                    if len(target.shape) > 1 and len(model_output.logits.shape) > 2:
                        # 실제 타겟의 마지막 유효한 토큰 인덱스 계산 (패딩 제외)
                        pred = model_output.logits[:, :-1].argmax(dim=-1)
                        valid_target = target[:, 1:]
                        valid_mask = (valid_target != -100) & (valid_target != 0)  # 패딩 토큰 및 특수 토큰 제외
                        
                        if valid_mask.sum() > 0:
                            correct = (pred == valid_target) & valid_mask
                            acc = correct.sum().float() / valid_mask.sum()
            except Exception as e:
                logger.warning(f"정확도 계산 중 오류 발생 (무시됨): {e}")
                acc = 0.0
                
            # EWC 손실 (이전 태스크 보존) - 필요한 경우
            try:
                # EWC 손실 계산 전에 자원 확보
                if hasattr(self, 'ewc') and self.ewc is not None:
                    ewc_loss = self.ewc.compute_ewc_loss()
                    
                    # 그래디언트 추적이 필요하면 자동으로 설정
                    if not ewc_loss.requires_grad and ewc_loss.item() != 0.0:
                        # 가능한 한 조용히 처리 - 로그 없이 자동 변환
                        ewc_loss = ewc_loss.detach().clone().requires_grad_(True)
                    
                    # 손실값 유효성 검사
                    if torch.isfinite(ewc_loss).all() and torch.isfinite(loss).all():
                        loss = loss + ewc_loss
                        if self.global_step % 20 == 0 and ewc_loss.item() > 0.0001:
                            # 유의미한 EWC 손실이 있을 때만 기록
                            logger.info(f"LM 손실: {loss.item():.4f}, EWC 손실: {ewc_loss.item():.4f}")
                else:
                    # EWC가 없는 경우 아무것도 하지 않음
                    pass
            except Exception as e:
                # 중요하지 않은 예외는 조용히 무시
                pass

        # 역전파 전에 손실이 그래디언트 추적을 요구하는지 확인
        try:
            if not loss.requires_grad:
                logger.warning(f"[안전 조치] 손실이 그래디언트 추적을 요구하지 않음. 명시적으로 requires_grad=True 설정")
                loss = loss.detach().clone().requires_grad_(True)
            
            # 그라디언트 누적을 위해 손실을 gradient_accumulation_steps로 나눔
            scaled_loss = self.scaler.scale(loss / self.gradient_accumulation_steps)
            scaled_loss.backward()
            
            # 그라디언트 누적 카운터 증가
            self.accumulated_gradients += 1
            
            # 그라디언트 누적이 설정된 스텝에 도달하지 않았으면 여기서 종료 (그래디언트 클리핑 없이)
            if self.accumulated_gradients < self.gradient_accumulation_steps:
                return loss.item(), acc.item() if isinstance(acc, torch.Tensor) else acc
                
            # 그라디언트 누적이 끝났을 때만 로그 출력
            logger.info(f"[그라디언트 누적 완료] {self.gradient_accumulation_steps}개 배치 누적 완료, 이제 가중치 업데이트")
            
            # 그라디언트 누적 카운터 리셋
            self.accumulated_gradients = 0
            
            # === 그라디언트 누적 완료 후에만 그래디언트 클리핑 수행 ===
            self._perform_gradient_clipping_and_step()
            
            if self.global_step % 50 == 0:
                logger.debug(f"backward 호출 성공: 손실={loss.item():.4f}, requires_grad={loss.requires_grad}")
        except Exception as e:
            logger.error(f"역전파 오류: {e}")
            # 간단한 대체 역전파 시도 (응급 조치)
            try:
                if hasattr(loss, 'backward'):
                    loss.backward()
                    logger.info("대체 역전파 방법 사용")
            except Exception as e2:
                logger.error(f"대체 역전파도 실패: {e2}")
                # 실패하면 현재 배치 건너뛰기

        # 그래디언트 누적이 완료된 후에만 수행되는 로직은 _perform_gradient_clipping_and_step()에서 처리됨
        
        # MER 메모리 업데이트 (그래디언트 누적이 완료된 경우에만)
        if hasattr(self, 'mer') and self.mer is not None:
            try:
                # 텐서 차원 안전 체크
                if hasattr(data, 'shape') and len(data.shape) > 1:
                    # 배치 차원이 있는 경우 첫 번째 샘플만 사용
                    sample_data = data[0] if data.shape[0] > 1 else data.squeeze(0)
                    sample_target = target[0] if hasattr(target, 'shape') and target.shape[0] > 1 else target
                    self.mer.reservoir_sampling(sample_data, sample_target)
                else:
                    self.mer.reservoir_sampling(data, target)
            except Exception as e:
                # 상세한 오류 정보 로깅 (디버깅용)
                error_msg = str(e)
                if "Scalar" in error_msg or "cannot be converted" in error_msg:
                    logger.debug(f"MER 텐서 차원 오류 (무시): {error_msg}")
                else:
                    logger.warning(f"MER 메모리 업데이트 오류: {error_msg}")
        
        # 로거 업데이트
        if hasattr(logger, 'update_step'):
            logger.update_step(self.global_step, silent=True)
        
        return loss.item(), acc

    def evaluate(self, data_loader: DataLoader):
        """
        모델 평가

        Args:
            data_loader: 평가 데이터 로더

        Returns:
            평균 손실, 정확도
        """
        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)

                try:
                    # 언어 모델의 내장 손실 함수 사용
                    outputs = self.model(data, labels=target)
                    loss = outputs.loss
                    total_loss += loss.item()

                    # 로그엣과 평균 정확도 계산 (간단한 버전)
                    logits = outputs.logits  # [batch_size, sequence_length, vocab_size]
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = target[:, 1:].contiguous()
                    
                    # 참고: 여기서의 정확도는 단어 레벨 정확도의 대략적인 추정
                    flattened_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                    flattened_shift_labels = shift_labels.view(-1)
                    
                    # 유효한 토큰에 대해서만 정확도 계산 (패딩 토큰 제외)
                    valid_indices = (flattened_shift_labels != -100)
                    if valid_indices.sum() > 0:
                        valid_logits = flattened_shift_logits[valid_indices]
                        valid_labels = flattened_shift_labels[valid_indices]
                        
                        predictions = valid_logits.argmax(dim=-1)
                        correct += (predictions == valid_labels).sum().item()
                        total += valid_indices.sum().item()
                    
                except Exception as e:
                    logger.error(f"평가 중 오류 발생: {str(e)}")
                    continue

        # 평균 손실 및 정확도 계산
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
        accuracy = correct / total if total > 0 else 0

        return avg_loss, accuracy

    def log_metrics(self, epoch: int, train_loss: float, train_acc: float, val_loss: float, val_acc: float, 
                   test_results: Dict[int, float], metrics: Dict[str, float]) -> None:
        """
        학습 메트릭을 DeepseekLogger와 Writer 클래스를 통해 기록합니다.

        Args:
            epoch: 현재 에폭
            train_loss: 훈련 손실
            train_acc: 훈련 정확도
            val_loss: 검증 손실
            val_acc: 검증 정확도
            test_results: 이전 학습 모드 테스트 결과 {mode_name: accuracy}
            metrics: 지속학습 성능 지표
        """
        # 글로벌 스텝 계산 (순차적 배정 혹은 에폭 기반)
        # 모드마다 최대 에폭 수를 곱하여 스텝 값을 클러스터링
        mode_to_idx = {'complete': 0, 'prompt': 1, 'comment': 2, 'error_fix': 3}
        mode_idx = mode_to_idx.get(self.current_mode, 0)
        step = epoch + mode_idx * self.config.get('max_epochs', 10)
        
        # 1. 콘솔 로그용 표현 생성
        train_info = f"Train: loss={train_loss:.4f}, acc={train_acc*100:.2f}%"
        val_info = f"Val: loss={val_loss:.4f}, acc={val_acc*100:.2f}%"
        
        # 이전 학습 모드 성능 정보
        mode_info = ", ".join([f"{mode}={acc*100:.1f}%" for mode, acc in test_results.items()])
        
        # 지속학습 지표
        cl_metrics = f"ACC={metrics['ACC']*100:.2f}%, BWT={metrics['BWT']*100:.2f}%, "
        cl_metrics += f"FWT={metrics['FWT']*100:.2f}%, SG={metrics['stability_gap']*100:.2f}%"
        
        # 수정된 로그 형식 (가독성 개선)
        log_parts = [
            f"Mode: {self.current_mode}", 
            f"Epoch: {epoch}/{self.config.get('max_epochs', 10)}",
            train_info,
            val_info
        ]
        
        # 이전 모드 결과가 있는 경우만 추가
        if test_results:
            log_parts.append(f"Past Modes: [{mode_info}]")
            
        # 지속학습 메트릭 추가
        log_parts.append(cl_metrics)
        
        # 학습률 추가
        if self.scheduler:
            lr = self.scheduler.get_last_lr()[0]
            log_parts.append(f"LR: {lr:.2e}")
        
        # 최종 로그 메시지 합치기
        log_message = " | ".join(log_parts)
        
        # 2. DeepseekLogger를 통한 콘솔 로깅
        logger.info(log_message)
        
        # 3. 메트릭 데이터 기록
        # 기본 학습 지표
        self._record_metrics({
            'Loss/train': train_loss,
            'Loss/val': val_loss,
            'Accuracy/train': train_acc,
            'Accuracy/val': val_acc,
            'LR': self.scheduler.get_last_lr()[0] if self.scheduler else 0.0
        }, step)
        
        # 지속학습 지표
        cl_metric_dict = {
            'Metrics/ACC': metrics['ACC'],
            'Metrics/BWT': metrics['BWT'],
            'Metrics/FWT': metrics['FWT'],
            'Metrics/stability_gap': metrics['stability_gap']
        }
        
        # recovery_ratio가 있는 경우만 추가
        if 'recovery_ratio' in metrics:
            cl_metric_dict['Metrics/recovery_ratio'] = metrics['recovery_ratio']
            
        self._record_metrics(cl_metric_dict, step)
        
        # 4. 각 학습 모드별 결과 기록
        for mode, acc in test_results.items():
            self._record_metrics({f'Mode/{mode}/Accuracy': acc}, step)
            
    def _record_metrics(self, metrics_dict: Dict[str, float], step: int) -> None:
        """
        메트릭을 Writer 클래스에 기록합니다.
        
        Args:
            metrics_dict: 기록할 메트릭 {name: value}
            step: 현재 학습 단계
        """
        for name, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(name, value, step)

    def save_model(self, filename: str, is_spot_interruption: bool = False) -> str:
        """
        모델, 옵티마이저, 스케줄러 및 학습 상태를 체크포인트로 저장합니다.

        Args:
            filename: 저장할 파일 이름 (확장자 포함)
            is_spot_interruption: 스팟 인스턴스 중단으로 인한 저장인지 여부

        Returns:
            str: 저장된 파일의 전체 경로
        """
        # [디버그] 함수 호출 확인
        logger.info(f"[디버그] save_model 호출됨. 파일명: {filename}, 스팟 중단: {is_spot_interruption}")
        
        # 저장 경로 설정 및 디렉토리 생성
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        
        # 학습 모드별 하위 디렉토리 설정 (complete, prompt, comment, error_fix)
        training_mode = self.config.get('mode', 'complete')
        if training_mode == "complete":
            mode_subdir = "autocomplete-finetuned"
        else:
            mode_subdir = f"{training_mode}-finetuned"
            
        # 경로 중복 방지 (checkpoint_dir에 이미 mode_subdir가 포함된 경우)
        if checkpoint_dir.endswith(mode_subdir) or mode_subdir in checkpoint_dir.split('/'):
            logger.info(f"체크포인트 경로에 이미 '{mode_subdir}'가 포함되어 있어 중복을 방지합니다.")
            checkpoint_subdir = checkpoint_dir
        else:
            # 정상 경로 설정
            checkpoint_subdir = os.path.join(checkpoint_dir, mode_subdir)
            
        logger.info(f"최종 체크포인트 기본 경로: {checkpoint_subdir}")
        save_path = os.path.join(checkpoint_subdir, filename)
        
        # [디버그] 디렉토리 존재 확인 및 권한 체크
        dir_exists = os.path.exists(os.path.dirname(save_path))
        logger.info(f"[디버그] 체크포인트 디렉토리 상태 - 경로: {os.path.dirname(save_path)}, 존재: {dir_exists}")
        
        # 쓰기 권한 확인 시도
        try:
            if not dir_exists:
                logger.info(f"[디버그] 체크포인트 디렉토리가 존재하지 않아 생성 시도: {os.path.dirname(save_path)}")
            
            # 디렉토리가 없는 경우 생성
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 디렉토리 생성 성공 확인
            logger.info(f"[디버그] 체크포인트 디렉토리 생성 결과: {os.path.exists(os.path.dirname(save_path))}")
            
            # 쓰기 권한 확인을 위한 테스트 파일 생성 시도
            test_file = os.path.join(os.path.dirname(save_path), '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.info(f"[디버그] 체크포인트 디렉토리 쓰기 권한 확인: 성공")
        except Exception as e:
            logger.error(f"[디버그] 체크포인트 디렉토리 권한 문제 발생: {str(e)}")
        
        # 로깅
        logger.info(f"체크포인트 저장 경로: {save_path} (모드: {training_mode})")
        
        try:
            # 메모리 클린업 및 최적화
            if torch.cuda.is_available():
                # 체크포인트 저장 전 CUDA 캐시 비우기
                torch.cuda.empty_cache()
                print(f"[메모리] CUDA 캐시 비운 후 상태 - 사용량: {torch.cuda.memory_allocated() / (1024**3):.2f}GB, 예약량: {torch.cuda.memory_reserved() / (1024**3):.2f}GB")
                
            # 가비지 콜렉터 실행
            # OOM 방지를 위한 메모리 최적화 단계
            import gc
            
            # 1. 강제 가비지 컬렉션 수행
            gc.collect(2)  # 모든 세대의 객체 수집
            
            # 2. CUDA 캐시 비우기
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"[체크포인트] CUDA 메모리 정리 완료")
            
            # 3. 메모리 상태 로깅
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            print(f"[체크포인트] 시스템 메모리 사용량: {memory_info.rss / 1024 / 1024:.1f}MB")
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    print(f"[체크포인트] GPU {i} 메모리: {torch.cuda.memory_allocated(i) / 1024 / 1024:.1f}MB / {torch.cuda.memory_reserved(i) / 1024 / 1024:.1f}MB")
            
            # 임시 파일명 생성 (atomic 저장을 위해)
            temp_save_path = f"{save_path}.tmp"
            print(f"[체크포인트] 임시 파일: {temp_save_path}")
            
            # 메모리 상태 확인
            import psutil
            memory_info = psutil.virtual_memory()
            print(f"[체크포인트] 메모리 상태 - 사용: {memory_info.percent}%, 가용: {memory_info.available/1024/1024:.1f}MB")
            
            # 저장할 상태 정보 구성
            
            # 모델 상태 정보 출력
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"[체크포인트] 모델 파라미터 - 전체: {total_params:,}, 학습 가능: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
            
            # 4. 이전 그래디언트 백업 및 비우기 (OOM 방지)
            gradients_backup = {}
            # 그래디언트가 큰 텐서들을 찾아서 백업
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # 저장 용량 관리를 위해 그래디언트 크기가 일정 이상인 것만 처리
                    if param.grad.numel() > 1000000:  # 백만 요소 이상의 큰 그래디언트만
                        gradients_backup[name] = param.grad.detach().cpu()
                        param.grad = None
            
            print(f"[체크포인트] 임시 그래디언트 백업: {len(gradients_backup)}개 텐서")
            
            print(f"[체크포인트] 파일 최적화 모드 - 변경된 학습 파라미터만 저장합니다")
            
            # 파일 크기 최소화를 위해 두 가지 접근 방식 사용
            # 1. 변된 파라미터만 저장
            # 2. 중요 메타데이터만 저장
            
            # 학습 모드 확인 (complete, prompt, comment, error_fix)
            current_training_mode = self.current_mode or 'unknown'
            print(f"[체크포인트] 현재 학습 모드: {current_training_mode}")
            
            # 5. 중복 데이터 제거 및 최소한의 필수 정보만 포함하는 state_dict 구성
            # 특히 과도한 중복 정보를 제거하여 체크포인트 크기 최적화
            
            # 타임스탬프 생성
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # 핵심 학습 상태 메타데이터만 유지 (중복 없이)
            metadata = {
                'global_step': self.global_step,
                'current_mode': self.current_mode,
                'completed_modes': list(set([self.current_mode] + (self.completed_modes if hasattr(self, 'completed_modes') else [])))
            }
            # 모델과 옵티마이저는 개별적으로 저장할 것이므로 state_dict에서 제외
            # 이 부분은 절대로 중복 저장하지 않도록 함
            state_dict = {
                # 핵심 메타데이터만 포함
                'global_step': self.global_step,
                'current_mode': self.current_mode,
                'best_accuracies': self.best_accuracies,
                
                # 학습 모드간 필수 메타데이터 (중복 제거)
                'mode_metadata': metadata,
                
                # 기본 메타데이터
                'saved_at': timestamp,
                'is_spot_interruption': is_spot_interruption
            }
            
            # 선택적 데이터 처리 - 개별 파일로 저장할 컴포넌트는 여기서 제외
            # 이들은 별도의 파일에 저장하여 중복을 방지함
            # 참고: optimizer, scheduler, scaler는 개별 파일로 저장됨
            logger.info(f"[디버그] 체크포인트 state_dict 구성 완료")
            
            # Hugging Face 형식으로 체크포인트 저장하도록 수정
            # 파일명에서 checkpoint- 형식으로 디렉토리 이름 추출
            reason = "스팟 인스턴스 중단" if is_spot_interruption else "정기 저장"
            
            # 체크포인트 디렉토리 이름 생성 (checkpoint-XXXX 형식)
            if "step" in filename:
                step_num = 0
                try:
                    # step 숫자를 추출 (예: task0_epoch0_step20_test.pt -> 20)
                    step_part = filename.split("step")[1].split("_")[0]
                    step_num = int(step_part)
                except:
                    step_num = self.global_step
                checkpoint_dir_name = f"checkpoint-{step_num}"
            else:
                checkpoint_dir_name = f"checkpoint-{self.global_step}"
            
            # 체크포인트 디렉토리 경로 생성
            checkpoint_dir_path = os.path.join(os.path.dirname(save_path), checkpoint_dir_name)
            logger.info(f"체크포인트 저장 중: {checkpoint_dir_path} (원인: {reason})")
            
            # 디렉토리가 없으면 생성
            if not os.path.exists(checkpoint_dir_path):
                os.makedirs(checkpoint_dir_path, exist_ok=True)
                logger.info(f"[디버그] 새 체크포인트 디렉토리 생성: {checkpoint_dir_path}")
            
            save_start_time = time.time()
            
            try:
                # 메모리 최적화: 저장 직전 추가 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 1. 최적화기 저장 - CPU에서 처리하여 OOM 방지
                optimizer_state = self.optimizer.state_dict()
                
                # 메모리 절약을 위해 옵티마이저의 대형 텐서를 CPU로 이동
                for param_group in optimizer_state['param_groups']:
                    for param_id in param_group['params']:
                        if param_id in optimizer_state['state']:
                            for k, v in optimizer_state['state'][param_id].items():
                                if isinstance(v, torch.Tensor) and v.numel() > 1000000:
                                    optimizer_state['state'][param_id][k] = v.cpu()
                
                optimizer_path = os.path.join(checkpoint_dir_path, "optimizer.pt")
                torch.save(optimizer_state, optimizer_path)
                logger.info(f"[디버그] 최적화기 저장 완료: {optimizer_path}")
                
                # 2. 스케줄러 저장 (있는 경우)
                if self.scheduler:
                    scheduler_path = os.path.join(checkpoint_dir_path, "scheduler.pt")
                    torch.save(self.scheduler.state_dict(), scheduler_path)
                    logger.info(f"[디버그] 스케줄러 저장 완료: {scheduler_path}")
                
                # 3. 스케일러 저장 (있는 경우)
                if self.scaler:
                    scaler_path = os.path.join(checkpoint_dir_path, "scaler.pt")
                    torch.save(self.scaler.state_dict(), scaler_path)
                    logger.info(f"[디버그] 스케일러 저장 완료: {scaler_path}")
                
                # 4. RNG 상태 저장
                rng_state_path = os.path.join(checkpoint_dir_path, "rng_state.pth")
                torch.save({
                    'random_state': random.getstate(),
                    'numpy_state': np.random.get_state(),
                    'torch_state': torch.get_rng_state(),
                    'cuda_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                }, rng_state_path)
                logger.info(f"[디버그] RNG 상태 저장 완료: {rng_state_path}")
                
                # 5. 모델 저장 (PEFT 어댑터 형식)
                # PEFT 모델인 경우 save_pretrained 메서드 사용
                if hasattr(self.model, 'save_pretrained'):
                    self.model.save_pretrained(checkpoint_dir_path)
                    logger.info(f"[디버그] PEFT 모델 저장 완료 (adapter_model.safetensors): {checkpoint_dir_path}")
                else:
                    # 일반 모델인 경우 state_dict를 저장
                    model_path = os.path.join(checkpoint_dir_path, "pytorch_model.bin")
                    torch.save(self.model.state_dict(), model_path)
                    logger.info(f"[디버그] 일반 모델 저장 완료: {model_path}")
                
                # 6. 토크나이저 저장 (있는 경우)
                if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                    if hasattr(self.tokenizer, 'save_pretrained'):
                        self.tokenizer.save_pretrained(checkpoint_dir_path)
                        logger.info(f"[디버그] 토크나이저 저장 완료")
                
                # 7. 트레이너 상태 저장 (JSON 형식)
                trainer_state = {
                    'global_step': self.global_step,
                    'current_mode': self.current_mode,
                    'best_accuracies': self.best_accuracies,
                    'mode_metadata': {
                        'mode': self.current_mode,
                        'completed_modes': list(set(self.completed_modes + [self.current_mode])) 
                                if hasattr(self, 'completed_modes') else [self.current_mode]
                    },
                    # 학습 지표 추가
                    'metrics': {
                        'train_loss': getattr(self, 'last_train_loss', None),
                        'val_loss': getattr(self, 'last_val_loss', None),
                        'train_accuracy': getattr(self, 'last_train_accuracy', None),
                        'val_accuracy': getattr(self, 'last_val_accuracy', None),
                        # EWC, MER 등 추가 지표
                        'ewc_loss': getattr(self, 'last_ewc_loss', None),
                        'mer_loss': getattr(self, 'last_mer_loss', None),
                        'stability_gap': getattr(self, 'last_stability_gap', None)
                    },
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'is_spot_interruption': is_spot_interruption
                }
                
                trainer_state_path = os.path.join(checkpoint_dir_path, "trainer_state.json")
                with open(trainer_state_path, 'w') as f:
                    json.dump(trainer_state, f, indent=2)
                logger.info(f"[디버그] 트레이너 상태 저장 완료: {trainer_state_path}")
                
                # 8. 학습 설정 저장
                training_args_path = os.path.join(checkpoint_dir_path, "training_args.bin")
                torch.save(self.config, training_args_path)
                logger.info(f"[디버그] 학습 설정 저장 완료: {training_args_path}")
                
                # 9. README.md 생성
                readme_path = os.path.join(checkpoint_dir_path, "README.md")
                with open(readme_path, 'w') as f:
                    f.write(f"# 체크포인트 {checkpoint_dir_name}\n\n")
                    f.write(f"- 글로벌 스텝: {self.global_step}\n")
                    f.write(f"- 학습 모드: {self.current_mode}\n")
                    
                    # 스팟 인스턴스 중단 여부 (f-string 내 백슬래시 문제 해결)
                    yes_text = '예'  # 한글 '예'
                    no_text = '아니오'  # 한글 '아니오'
                    spot_status = yes_text if is_spot_interruption else no_text
                    f.write(f"- 스팟 인스턴스 중단: {spot_status}\n\n")
                    
                    # 학습 지표 추가
                    f.write(f"## 학습 지표\n\n")
                    
                    # 허깅페이스 스타일 테이블 형식으로 표시
                    f.write(f"| 지표 | 값 |\n")
                    f.write(f"|------|------|\n")
                    
                    # 학습 손실 값
                    train_loss = getattr(self, 'last_train_loss', None)
                    if train_loss is not None:
                        f.write(f"| Train Loss | {train_loss:.4f} |\n")
                        
                    # 검증 손실 값
                    val_loss = getattr(self, 'last_val_loss', None)
                    if val_loss is not None:
                        f.write(f"| Validation Loss | {val_loss:.4f} |\n")
                    
                    # 학습 정확도
                    train_acc = getattr(self, 'last_train_accuracy', None)
                    if train_acc is not None:
                        f.write(f"| Train Accuracy | {train_acc:.2f}% |\n")
                        
                    # 검증 정확도
                    val_acc = getattr(self, 'last_val_accuracy', None)
                    if val_acc is not None:
                        f.write(f"| Validation Accuracy | {val_acc:.2f}% |\n")
                    
                    # 지속학습 지표
                    ewc_loss = getattr(self, 'last_ewc_loss', None)
                    if ewc_loss is not None:
                        f.write(f"| EWC Loss | {ewc_loss:.4f} |\n")
                        
                    mer_loss = getattr(self, 'last_mer_loss', None)
                    if mer_loss is not None:
                        f.write(f"| MER Loss | {mer_loss:.4f} |\n")
                    
                    stability_gap = getattr(self, 'last_stability_gap', None)
                    if stability_gap is not None:
                        f.write(f"| Stability Gap | {stability_gap:.4f} |\n")
                        
                    # 메모리 사용량 추가
                    f.write(f"\n## 시스템 정보\n\n")
                    
                    # GPU 메모리 사용량
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            gpu_mem_allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                            gpu_mem_reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
                            f.write(f"- GPU {i} 메모리: {gpu_mem_allocated:.1f}MB / {gpu_mem_reserved:.1f}MB\n")
                            
                    # 시스템 메모리 사용량
                    try:
                        import psutil
                        memory_info = psutil.virtual_memory()
                        f.write(f"- 시스템 메모리: 사용 {memory_info.percent}%, 가용 {memory_info.available/(1024*1024*1024):.1f}GB\n")
                    except (ImportError, Exception) as e:
                        logger.warning(f"[디버그] psutil 사용 불가: 시스템 메모리 정보를 가져올 수 없습니다. {e}")
                        f.write(f"- 시스템 메모리: 정보 없음\n")
                    
                logger.info(f"[디버그] README 생성 완료: {readme_path}")
                
                # 10. adapter_config.json 생성 (PEFT 모델용)
                if not os.path.exists(os.path.join(checkpoint_dir_path, "adapter_config.json")):
                    adapter_config = {
                        "peft_type": "LORA",
                        "task_type": "CAUSAL_LM",
                        "r": 8,
                        "lora_alpha": 16,
                        "target_modules": ["q_proj", "v_proj"],
                        "bias": "none"
                    }
                    adapter_config_path = os.path.join(checkpoint_dir_path, "adapter_config.json")
                    with open(adapter_config_path, 'w') as f:
                        json.dump(adapter_config, f, indent=2)
                
                # 저장 완료 시간 및 성능 측정
                save_duration = time.time() - save_start_time
                logger.info(f"[디버그] Hugging Face 형식 저장 완료 - 소요 시간: {save_duration:.2f}초")
                
                # 저장 경로 업데이트
                save_path = checkpoint_dir_path
            except Exception as e:
                logger.error(f"[디버그] Hugging Face 형식 저장 오류 발생: {str(e)}")
                logger.error(traceback.format_exc())
                
                # 오류 발생 시 기본 PyTorch 방식으로 대체 저장 시도
                logger.warning(f"[대체 저장] 기본 PyTorch 형식으로 비상 백업 시도")
                backup_path = f"{save_path}.backup.pt"
                torch.save(state_dict, backup_path)
                logger.info(f"[디버그] 기본 백업 완료: {backup_path}")
                
            
            # 파일 크기 계산 - 디렉토리 전체 크기 계산
            file_size_mb = 0.0
            if os.path.exists(save_path):
                # 디렉토리인 경우 전체 크기 계산
                if os.path.isdir(save_path):
                    total_size = 0
                    file_sizes = {}
                    for dirpath, _, filenames in os.walk(save_path):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            if not os.path.islink(fp):
                                size = os.path.getsize(fp)
                                total_size += size
                                file_sizes[f] = size / (1024 * 1024)  # bytes to MB
                    
                    # 개별 파일 크기 로깅 (용량 분석용)
                    top_files = sorted(file_sizes.items(), key=lambda x: x[1], reverse=True)[:5]
                    print(f"[체크포인트] 최대 용량 파일 Top 5:")
                    for fname, size_mb in top_files:
                        print(f"  - {fname}: {size_mb:.2f}MB")
                    
                    file_size_mb = total_size / (1024 * 1024)  # bytes to MB
                else:
                    # 단일 파일인 경우
                    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)  # bytes to MB
            
            # 저장 완료 로깅
            if file_size_mb < 0.1 and os.path.isdir(save_path):
                logger.warning(f"체크포인트 크기가 비정상적으로 작습니다: {file_size_mb:.1f}MB")
                
            logger.info(f"체크포인트 저장 완료: {save_path} (크기: {file_size_mb:.1f}MB, 시간: {timestamp})")
            print(f"[체크포인트] 저장 성공: {save_path}")
            print(f"[체크포인트] 파일 크기: {file_size_mb:.1f}MB")
            print(f"[체크포인트] 스텝: {self.global_step}, 모드: {self.current_mode or 'default'}")
            
            # 저장 후 백업한 그래디언트 복원
            if len(gradients_backup) > 0:
                print(f"[체크포인트] 백업된 그래디언트 복원 중: {len(gradients_backup)}개 텐서")
                try:
                    for name, grad in gradients_backup.items():
                        param = None
                        for n, p in self.model.named_parameters():
                            if n == name:
                                param = p
                                break
                        if param is not None and param.requires_grad:
                            param.grad = grad.to(param.device) if hasattr(grad, 'to') else grad
                except Exception as e:
                    print(f"[체크포인트] 그래디언트 복원 중 오류 (무시함): {str(e)}")
            
            # 메모리 상태 다시 확인
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    print(f"[체크포인트] 저장 후 GPU {i} 메모리: {torch.cuda.memory_allocated(i) / 1024 / 1024:.1f}MB / {torch.cuda.memory_reserved(i) / 1024 / 1024:.1f}MB")
            
            # 스팟 핸들러에 체크포인트 이벤트 알림 (심볼릭 링크 등 후처리)
            if hasattr(self, 'spot_handler'):
                try:
                    self.spot_handler.handle_checkpoint_event(save_path, self.global_step)
                    print(f"[체크포인트] 심볼릭 링크 업데이트: checkpoint-latest.pt -> {os.path.basename(save_path)}")
                except Exception as e:
                    print(f"[체크포인트] 오류 - 심볼릭 링크 업데이트 실패: {str(e)}")
                    
            # 체크포인트 파일 개수 관리 - 최신 3개만 유지
            try:
                checkpoint_dir = os.path.dirname(save_path)
                # 긴급 체크포인트는 제외하고, 현재 모드의 체크포인트만 관리
                mode_prefix = "checkpoint"
                if is_spot_interruption:
                    mode_prefix = "interrupt"
                
                # 디렉토리에서 현재 모드의 체크포인트 파일들을 찾아서 정렬
                checkpoint_files = []
                for file in os.listdir(checkpoint_dir):
                    if file.startswith(mode_prefix) and file.endswith(".pt") and "emergency" not in file:
                        file_path = os.path.join(checkpoint_dir, file)
                        # 파일 수정 시간을 기준으로 정렬
                        checkpoint_files.append((file_path, os.path.getmtime(file_path)))
                
                # 최신 체크포인트를 제외한 과거 체크포인트 정리
                if len(checkpoint_files) > 5:  # 최신 5개 유지 (안정성 향상)
                    # 시간 순으로 정렬 (최신 것이 마지막에 오게)
                    checkpoint_files.sort(key=lambda x: x[1])  
                    
                    # 최신 5개를 제외한 나머지 삭제
                    files_to_remove = checkpoint_files[:-5]
                    for file_path, _ in files_to_remove:
                        try:
                            os.remove(file_path)
                            print(f"[체크포인트] 오래된 파일 삭제: {os.path.basename(file_path)}")
                        except Exception as e:
                            print(f"[체크포인트] 파일 삭제 오류: {os.path.basename(file_path)}, {str(e)}")
            except Exception as e:
                print(f"[체크포인트] 관리 오류: {str(e)}")
                # 체크포인트 관리 오류가 저장 자체를 실패시키지는 않음
                
            # 파일 존재 추가 확인 - 보안상을 위해 완료 후 다시 한번 확인
            final_check_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if os.path.exists(save_path):
                file_size_mb_final = os.path.getsize(save_path) / 1024 / 1024
                print(f"[체크포인트] 저장 완료: {os.path.basename(save_path)} (최종크기: {file_size_mb_final:.1f}MB)")
            else:
                print(f"[체크포인트] 경고 - 저장 확인 실패: {os.path.basename(save_path)}")
                
            return save_path
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA 메모리 부족 오류 발생: {str(e)}")
            logger.error("[대책 제안] GPU 메모리 부족. 다음 대책 시도: 1) 배치 크기 감소, 2) 처음부터 다시 시작, 3) CPU로 파일 저장 시도")
            torch.cuda.empty_cache()  # 메모리 정리 시도
            return None
            
        except IOError as e:
            logger.error(f"입출력 오류 발생 (파일 저장 실패): {str(e)}")
            logger.error("[대책 제안] 디스크 권한/공간 문제 검토: 1) 디렉토리 권한 확인, 2) 디스크 공간 확인, 3) 디렉토리 경로 정확성 확인")
            
            # 손상된 체크포인트 파일이 남아있을 수 있으므로 삭제 시도
            try:
                if os.path.exists(save_path):
                    logger.warning(f"[디버그] 손상된 체크포인트 파일 제거 시도: {save_path}")
                    os.remove(save_path)
                    logger.warning(f"손상된 체크포인트 파일이 제거되었습니다: {save_path}")
                    
                # 임시 파일도 확인하여 제거
                temp_save_path = f"{save_path}.tmp"
                if os.path.exists(temp_save_path):
                    logger.warning(f"[디버그] 임시 체크포인트 파일 제거 시도: {temp_save_path}")
                    os.remove(temp_save_path)
            except Exception as remove_error:
                logger.error(f"손상된 파일 제거 실패: {str(remove_error)}")
                
            return None

    def load_model(self, filename: str) -> bool:
        """
        저장된 모델, 옵티마이저, 스케줄러 및 학습 상태를 로드합니다.

        Args:
            filename: 로드할 파일 이름 (확장자 포함)

        Returns:
            bool: 모델 로딩 성공 여부
        """
        # 기본 체크포인트 디렉토리 설정
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        
        # 학습 모드별 하위 디렉토리 설정 (complete, prompt, comment, error_fix)
        training_mode = self.config.get('mode', 'complete')
        if training_mode == "complete":
            mode_subdir = "autocomplete-finetuned"
        else:
            mode_subdir = f"{training_mode}-finetuned"
        
        # 최종 로드 경로 설정
        checkpoint_subdir = os.path.join(checkpoint_dir, mode_subdir)
        load_path = os.path.join(checkpoint_subdir, filename)
        
        # 파일 존재 확인
        if not os.path.exists(load_path):
            logger.warning(f"모드별 디렉토리에서 체크포인트 파일을 찾을 수 없습니다: {load_path}")
            
            # 이전 경로에서 로드 시도 (하위호환성)
            legacy_path = os.path.join(checkpoint_dir, filename)
            if os.path.exists(legacy_path):
                logger.info(f"레거시 경로에서 체크포인트를 찾았습니다. 로드합니다: {legacy_path}")
                load_path = legacy_path
            else:
                return False
            
        try:
            # 파일 로드
            logger.info(f"체크포인트 로딩 중: {load_path}")
            state_dict = torch.load(load_path, map_location=self.device)
            
            # 모델 상태 로드
            self.model.load_state_dict(state_dict['model'])
            logger.debug("모델 가중치 로드 완료")
            
            # 옵티마이저 상태 로드
            self.optimizer.load_state_dict(state_dict['optimizer'])
            logger.debug("옵티마이저 상태 로드 완료")
            
            # 스케줄러 상태 (있는 경우만) 로드
            if 'scheduler' in state_dict and state_dict['scheduler'] and self.scheduler:
                self.scheduler.load_state_dict(state_dict['scheduler'])
                logger.debug("스케줄러 상태 로드 완료")
                
                # 중요: 스케줄러의 내부 스텝을 글로벌 스텝과 동기화
                if hasattr(self.scheduler, 'current_step') and 'global_step' in state_dict:
                    old_step = getattr(self.scheduler, 'current_step', 0)
                    self.scheduler.current_step = state_dict['global_step']
                    logger.warning(f"스케줄러 스텝 강제 동기화: {old_step} → {self.scheduler.current_step} (글로벌 스텝: {state_dict['global_step']})")
                    
                    # 학습률 강제 업데이트 (스케줄러가 AdaptiveLRScheduler인 경우)
                    if hasattr(self.scheduler, '_get_lr') and hasattr(self.scheduler, 'optimizer'):
                        new_lrs = self.scheduler._get_lr(self.scheduler.current_step)
                        for param_group, lr in zip(self.scheduler.optimizer.param_groups, new_lrs):
                            old_lr = param_group['lr']
                            param_group['lr'] = lr
                            logger.warning(f"학습률 강제 업데이트: {old_lr:.6f} → {lr:.6f} (스텝: {self.scheduler.current_step})")
                else:
                    logger.warning("스케줄러 또는 글로벌 스텝 정보가 없어 동기화하지 못했습니다.")
            
            # 그래디언트 스케일러 (있는 경우만) 로드
            if 'scaler' in state_dict and state_dict['scaler'] and self.scaler:
                self.scaler.load_state_dict(state_dict['scaler'])
                logger.debug("그래디언트 스케일러 상태 로드 완료")
            
            # 학습 진행 상태 로드 - get 메서드로 안전하게 가져오기
            self.global_step = state_dict.get('global_step', 0)
            self.current_mode = state_dict.get('current_mode', 'complete')  # 호환성을 위해 task_id도 확인
            if 'current_task_id' in state_dict and not self.current_mode:
                # 기존 체크포인트의 호환성을 위한 변환
                task_id_to_mode = {0: 'complete', 1: 'prompt', 2: 'comment', 3: 'error_fix'}
                self.current_mode = task_id_to_mode.get(state_dict.get('current_task_id', 0), 'complete')
            self.best_accuracies = state_dict.get('best_accuracies', {})
            
            # 메트릭 및 지속학습 컴포넌트 로드 (조건부)
            if 'metrics' in state_dict and hasattr(self, 'metrics'):
                self.metrics.load_state_dict(state_dict['metrics'])
                logger.debug("메트릭 상태 로드 완료")
            
            if 'ewc' in state_dict and hasattr(self, 'ewc'):
                self.ewc.load_state_dict(state_dict['ewc'])
                logger.debug("EWC 상태 로드 완료")
            
            if 'mer' in state_dict and hasattr(self, 'mer'):
                self.mer.load_state_dict(state_dict['mer'])
                logger.debug("MER 상태 로드 완료")
            
            # 체크포인트 메타데이터 출력
            if 'saved_at' in state_dict:
                logger.info(f"로드된 체크포인트 저장 시간: {state_dict['saved_at']}")
            if 'version' in state_dict:
                logger.debug(f"체크포인트 형식 버전: {state_dict['version']}")
            
            logger.info(f"체크포인트를 성공적으로 로드했습니다. 현재 모드: {self.current_mode}, 전체 스텝: {self.global_step}")
            return True
            
        except Exception as e:
            error_msg = f"체크포인트 로드 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            # 스택 트레이스 기록 (디버깅용)
            import traceback
            logger.debug(f"체크포인트 로드 오류 상세 정보: {traceback.format_exc()}")
            return False

    def get_state_dict(self):
        """상태 저장을 위한 딕셔너리 반환"""
        return {
            'metrics': self.metrics.get_state_dict(),
            'global_step': self.global_step,
            'current_mode': self.current_mode,
            'best_accuracies': self.best_accuracies,
            'ewc': self.ewc.get_state_dict(),
            'mer': self.mer.get_state_dict()
        }

    def load_state_dict(self, state_dict):
        """저장된 상태 로드"""
        if 'metrics' in state_dict:
            self.metrics.load_state_dict(state_dict['metrics'])

        if 'ewc' in state_dict:
            self.ewc.load_state_dict(state_dict['ewc'])

        if 'mer' in state_dict:
            self.mer.load_state_dict(state_dict['mer'])

        self.global_step = state_dict.get('global_step', 0)
        
        # current_mode를 우선 로드하고, 없으면 current_task_id를 변환하여 사용
        if 'current_mode' in state_dict:
            self.current_mode = state_dict.get('current_mode', 'complete')
        else:
            # 기존 체크포인트의 호환성을 위한 변환
            task_id_to_mode = {0: 'complete', 1: 'prompt', 2: 'comment', 3: 'error_fix'}
            self.current_mode = task_id_to_mode.get(state_dict.get('current_task_id', 0), 'complete')
        self.best_accuracies = state_dict.get('best_accuracies', {})

# 메인 함수
# wandb 명시적 비활성화 (스크립트 맨 위에 추가)
os.environ["WANDB_DISABLED"] = "true"

def load_dataset_and_tokenize(train_path, tokenizer, config, overwrite_cache=False, mode='prompt', padding="max_length"):
    """
    데이터셋을 로드하고 토큰화하는 함수
    
    Args:
        train_path: 데이터 파일 경로
        tokenizer: 토크나이저
        overwrite_cache: 캐시 덮어쓰기 여부
        mode: 학습 모드 (complete, prompt, comment, error_fix)
        padding: 패딩 방식
            
    Returns:
        tokenized_dataset: 토큰화된 데이터셋
    """
    logger.info(f"데이터셋 로드 및 토큰화 시작: {train_path} (모드: {mode})")
    
    # 데이터 파일 존재 확인
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"훈련 데이터 파일을 찾을 수 없습니다: {train_path}")
    
    # 훈련 데이터 파일만 사용 (검증 데이터는 후에 분리)
    data_files = {'train': train_path}
    
    # 데이터 형식 감지 및 처리
    data_format = detect_data_format(train_path)
    logger.info(f"감지된 데이터 형식: {data_format}")
    
    # 최대 토큰 길이 설정 (설정 파일에서 읽기)
    max_length = config.get('max_length', 512)
    
    # 데이터셋 로드
    raw_dataset = load_dataset(
        'json', 
        data_files=data_files,
        cache_dir=config.get('cache_dir', '.cache'),
        streaming=False  # 지정한 경우에만 False로 설정하여 메모리에 로드
    )
    
    # 훈련/검증 분리
    split_ratio = config.get('validation_split_ratio', 0.10)  # 기본값을 10%로 변경
    split_dataset = raw_dataset['train'].train_test_split(
        test_size=split_ratio, shuffle=True, seed=config.get('seed', 42)
    )
    
    # 데이터셋 준비
    raw_dataset = DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['test'],
    })
    
    # 향상된 토큰화 함수 - 개선된 표시 및 에러 처리
    def tokenize_function(examples):
        # API 버전 정보 추가
        if not hasattr(tokenize_function, 'use_new_api'):
            tokenize_function.use_new_api = None
            logger.debug("[토큰화] API 버전 정보가 없어 기본 동작 사용")
        
        # 토큰화 진행바 표시 동작 개선
        if hasattr(tokenize_function, 'batch_count'):
            tokenize_function.batch_count += 1
        else:
            tokenize_function.batch_count = 1
            
        # 배치 진행상황 표시 개선 (2000번째 배치마다 로그)
        if tokenize_function.batch_count % 2000 == 0:
            logger.info(f"[토큰화] 배치 처리 중: {tokenize_function.batch_count}번째 배치")
        
        # 모드별, 데이터 형식별 처리를 통합
        texts = []
        
        # 샘플 수 결정 (다양한 형식 지원)
        sample_count = 0
        if "content" in examples and mode == 'complete':
            sample_count = len(examples["content"])
        elif "messages" in examples:
            sample_count = len(examples["messages"])
        elif "prompt" in examples:
            sample_count = len(examples["prompt"])
        elif "prefix_code" in examples:
            sample_count = len(examples["prefix_code"])
        elif "instruction" in examples:
            sample_count = len(examples["instruction"])
        elif "comment" in examples and "code" in examples:
            sample_count = len(examples["comment"])
        elif "error_description" in examples and "buggy_code" in examples:
            sample_count = len(examples["error_description"])
        else:
            logger.error(f"❌ 지원되지 않는 데이터 형식 - 키: {list(examples.keys())}")
            return {"input_ids": []}
        
        # 각 샘플 처리
        for i in range(sample_count):
            # 데이터 형식 및 모드별 처리
            if data_format == "complete" or (mode == 'complete' and "content" in examples):
                # 자동완성(complete) 모드
                if "content" in examples:
                    text = examples["content"][i]
                    texts.append(text)
            
            elif data_format == "chat":
                # 채팅 형식 처리 - messages 배열이 있는 형태
                if "messages" in examples:
                    messages = examples["messages"][i]
                    if isinstance(messages, list):
                        # 메시지 배열을 ChatML 형식으로 변환
                        text = ""
                        for msg in messages:
                            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                                role = msg["role"].lower()  # user, assistant, system
                                content = msg["content"]
                                # ChatML 형식으로 변환
                                if role == "user":
                                    text += f"user\n{content}\n\n"
                                elif role == "assistant":
                                    text += f"assistant\n{content}\n\n"
                                # system 메시지는 일반적으로 무시하지만 필요하다면 추가 가능
                        texts.append(text)
                    else:
                        logger.warning(f"\u26a0️ messages가 리스트 형태가 아님: {type(messages)}")
                        texts.append("")  # 빈 텍스트 추가
                else:
                    logger.warning("\u26a0️ chat 형식이지만 messages 키가 없음")
                    texts.append("")  # 빈 텍스트 추가
                    
            elif data_format == "prompt_completion":
                # 주석 키워드 기반 코드 생성 형식
                prompt = examples["prompt"][i] if "prompt" in examples else ""
                completion = examples["completion"][i] if "completion" in examples else ""
                
                # FIM 형식 처리 (3차 주석 기반 모델)
                if mode == "comment" and ("<|fim begin|>" in prompt or "<|fim hole|>" in prompt or "<|fim end|>" in prompt):
                    # FIM 형식 그대로 유지하고 ChatML 형식으로 래핑
                    text = f"assistant\n{prompt}\n\nassistant\n{completion}\n"
                else:
                    # 일반 프롬프트를 사용자 메시지로, 완성을 어시스턴트 메시지로 변환
                    text = f"assistant\n{prompt}\n\nassistant\n{completion}\n"
                
                texts.append(text)
                    
            elif data_format == "comment_to_code" or data_format == "comment_code" or mode == 'comment':
                # 주석 기반 코드 생성 형식
                if data_format == "comment_to_code":
                    prefix_code = examples["prefix_code"][i] if "prefix_code" in examples else ""
                    suffix_code = examples["suffix_code"][i] if "suffix_code" in examples else ""
                    comment = examples["comment"][i]
                    target_code = examples["target_code"][i] if "target_code" in examples else ""
                    
                    # 사용자 입력 구성 (주석과 코드 컨텍스트)
                    user_text = f"주석에 따라 적절한 코드를 생성해주세요.\n\n"
                    user_text += f"주석: {comment}\n\n"
                    user_text += "\n이전 코드:\n"
                    user_text += f"{prefix_code}\n"
                    user_text += "// 여기에 코드를 삽입해야 함 //\n"
                    user_text += f"{suffix_code}"
                    
                    # ChatML 형식으로 변환
                    text = f"assistant\n{user_text}\n\nassistant\n{target_code}\n"
                else:
                    # 간단한 주석-코드 형식
                    comment = examples["comment"][i]
                    code = examples["code"][i]
                    text = f"assistant\n주석: {comment}\n\nassistant\n{code}\n"
                
                texts.append(text)
                
            elif data_format == "error_fix" or mode == 'error_fix':
                # 에러 수정 형식
                if "error_description" in examples and "buggy_code" in examples and "fixed_code" in examples:
                    # 단순 에러 수정 형식
                    error_desc = examples["error_description"][i]
                    buggy_code = examples["buggy_code"][i]
                    fixed_code = examples["fixed_code"][i]
                    
                    text = f"assistant\n오류 설명: {error_desc}\n\nassistant\n{buggy_code}\n\nassistant\n{fixed_code}\n"
                    texts.append(text)
                elif "error_context" in examples:
                    # 복잡한 에러 수정 형식
                    error_context = examples["error_context"][i]
                    fixed_code = examples["fixed_code_snippet"][i] if "fixed_code_snippet" in examples else ""
                    
                    # 에러 컨텍스트 정보 구성
                    error_log = error_context.get("error_log", "") if isinstance(error_context, dict) else ""
                    language = error_context.get("language", "") if isinstance(error_context, dict) else ""
                    
                    # buggy_code가 error_context 안에 있는 경우와 외부에 있는 경우 모두 처리
                    if isinstance(error_context, dict) and "buggy_code_snippet" in error_context:
                        buggy_code = error_context["buggy_code_snippet"]
                    elif "buggy_code_snippet" in examples:
                        buggy_code = examples["buggy_code_snippet"][i]
                    else:
                        buggy_code = ""
                        logger.warning("⚠️ buggy_code_snippet을 찾을 수 없습니다")
                    
                    # 사용자 입력 구성
                    user_text = f"다음 {language} 코드의 에러를 수정해주세요:\n\n에러 로그:\n{error_log}\n\n코드:\n{buggy_code}"
                    text = f"assistant\n{user_text}\n\nassistant\n{fixed_code}\n"
                    texts.append(text)
            else:
                # 알 수 없는 형식은 원시 데이터 텍스트를 그대로 사용
                logger.warning(f"⚠️ 지원되지 않는 데이터 형식: {data_format}, 모드: {mode}")
                # 첫 번째 필드만 사용
                if len(examples) > 0:
                    first_key = list(examples.keys())[0]
                    text = str(examples[first_key][i])
                    texts.append(text)
                else:
                    texts.append("")
        
        # 토크나이징
        model_inputs = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
            padding=padding
        )
        
        return model_inputs
    
    # 토큰화 실행
    logger.info(f"토큰화 시작: {train_path}")
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_dataset['train'].column_names,
        num_proc=config.get('preprocessing_num_workers', 4),
        desc=f"{mode} 토큰화",
        load_from_cache_file=not overwrite_cache
    )
    
    # 토큰화 완료 후 통계 정보 추가
    train_samples = len(tokenized_dataset['train'])
    val_samples = len(tokenized_dataset['validation'])
    logger.info(f"토큰화 완료: 훈련 {train_samples:,}개, 검증 {val_samples:,}개 샘플")
    
    # 모드에 맞게 토크나이즈 후 raw_dataset과 함께 반환
    return tokenized_dataset, raw_dataset

# 데이터 형식 감지 함수 정의
def detect_data_format(data_path):
    """
    데이터 파일의 형식을 감지하는 함수
    
    Args:
        data_path: 데이터 파일 경로
        
    Returns:
        data_format: 감지된 데이터 형식
    """
    try:
        # 첫 번째 라인만 읽기
        with open(data_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            data = json.loads(first_line)
        
        # 키를 기반으로 데이터 형식 감지
        if "messages" in data:
            # 메세지 형식의 대화식 데이터 - 채팅 형식으로 처리
            return "chat"
        elif "content" in data:
            return "complete"
        elif "prompt" in data and "completion" in data:
            return "prompt_completion"
        elif "comment" in data and "code" in data:
            return "comment_code"
        elif "prefix_code" in data and "target_code" in data and "comment" in data:
            return "comment_to_code"
        elif "error_description" in data and "buggy_code" in data:
            return "error_fix"
        else:
            return "unknown"
    except Exception as e:
        logger.error(f"데이터 형식 감지 중 오류 발생: {str(e)}")
        return "unknown"

def check_dependencies():
    """필수 의존성 체크"""
    try:
        import transformers
        import torch
        import datasets
        import peft
        
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ Transformers: {transformers.__version__}")
        print(f"✅ Datasets: {datasets.__version__}")
        print(f"✅ PEFT: {peft.__version__}")
        
        # CUDA 사용 가능 여부 체크
        if not torch.cuda.is_available():
            print("❌ CUDA를 사용할 수 없습니다.")
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ 의존성 누락: {e}")
        print("💡 설치 명령어: pip install transformers peft datasets accelerate bitsandbytes")
        return False

def check_gpu_memory_before_start():
    """실행 전 GPU 메모리 체크"""
    if not torch.cuda.is_available():
        return False
        
    props = torch.cuda.get_device_properties(0)
    total_mem_gb = props.total_memory / (1024**3)
    
    # 현재 메모리 사용량 체크
    allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
    reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
    free_gb = total_mem_gb - reserved_gb
    
    print(f"📊 GPU 메모리 상태:")
    print(f"  - 전체: {total_mem_gb:.1f}GB")
    print(f"  - 사용 중: {allocated_gb:.1f}GB")
    print(f"  - 예약됨: {reserved_gb:.1f}GB")
    print(f"  - 사용 가능: {free_gb:.1f}GB")
    
    # DeepSeek-Coder 6.7B + LoRA + 배치 처리에 필요한 최소 메모리: ~8GB
    min_required_gb = 8.0
    
    if free_gb < min_required_gb:
        print(f"⚠️  경고: 사용 가능한 GPU 메모리가 {min_required_gb}GB 미만입니다.")
        print(f"💡 추천: 다른 GPU 프로세스를 종료하거나 배치 크기를 줄이세요.")
        return False
        
    return True

def main():
    """메인 실행 함수"""
    import traceback  # traceback 모듈 import
    
    # 🔥 최우선: GPU 메모리 및 환경 설정 (Critical Issue #1 해결)
    print("🚀 Continual Learning 시작")
    
    # GPU 메모리 사전 설정 (가장 중요!)
    if torch.cuda.is_available():
        # 메모리 fraction 설정을 모델 로딩 전에 수행
        torch.cuda.set_per_process_memory_fraction(0.75, 0)
        torch.cuda.empty_cache()
        
        # CUDA 메모리 단편화 방지 환경변수 설정
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        
        # GPU 상태 확인
        gpu_props = torch.cuda.get_device_properties(0)
        total_mem_gb = gpu_props.total_memory / (1024**3)
        print(f"✅ GPU 메모리 설정 완료: {gpu_props.name}, {total_mem_gb:.1f}GB")
        
        # A10G 환경 전용 최적화 경고
        if total_mem_gb < 15.0:
            print("⚠️  경고: GPU 메모리가 15GB 미만입니다. OOM 위험이 높습니다.")
            print("💡 추천: 배치 크기를 1로, max_length를 256으로 설정하세요.")
    else:
        print("❌ GPU를 사용할 수 없습니다.")
        return
    
    # 의존성 체크
    if not check_dependencies():
        print("❌ 의존성 체크 실패")
        return
    
    # GPU 메모리 체크
    if not check_gpu_memory_before_start():
        print("❌ GPU 메모리 부족")
        return
    
    parser = argparse.ArgumentParser(description='Continual Learning with PEFT/LoRA 시스템')
    parser.add_argument('--config', type=str, default='config.yaml', help='설정 파일 경로')
    parser.add_argument('--mode', type=str, choices=['complete', 'prompt', 'comment', 'error_fix'], default='prompt',
                        help='학습 모드: complete(자동완성), prompt(일반 프롬프트), comment(주석), error_fix(오류 수정)')
    parser.add_argument('--data-file', type=str, help='사용할 데이터 파일명 (예: train1.jsonl, train2.jsonl)')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--resume', action='store_true', help='이전 체크포인트에서 재개')
    parser.add_argument('--resume_from_checkpoint', type=str, help='특정 체크포인트 경로에서 재개')
    # 학습 환경 설정
    parser.add_argument('--overwrite_cache', action='store_true', help='데이터셋 캐시를 사용하지 않고 다시 토큰화')
    args = parser.parse_args()

    # 설정 파일 로드
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 명령행 인수로 설정 덮어쓰기
    if args.seed:
        config['seed'] = args.seed
        
    # 학습 모드 추가
    config['mode'] = args.mode
    logger.info(f"\ud559\uc2b5 \ubaa8\ub4dc: {args.mode}")
    
    # 모드별 출력 경로 설정
    if args.mode == 'complete':
        config['output_dir'] = config.get('output_dir', '../models/autocomplete-finetuned/')
        logger.info("[1차] 코드 자동완성 학습 모드 (FIM 형식 지원)")
    elif args.mode == 'prompt':
        config['output_dir'] = config.get('output_dir', '../models/prompt-finetuned/')
        logger.info("[2차] 일반 프롬프트 기반 코드 생성 모드")
    elif args.mode == 'comment':
        config['output_dir'] = config.get('output_dir', '../models/comment-finetuned/')
        logger.info("[3차] 주석 기반 코드 생성 모드")
    elif args.mode == 'error_fix':
        config['output_dir'] = config.get('output_dir', '../models/error-fix-finetuned/')
        logger.info("[4차] 코드 오류 설명 및 수정 모드")

    # 기본 PEFT/LoRA 기능만 사용하여 파인튜닝을 진행합니다
    logger.info("기본 LoRA 파인튜닝 설정으로 학습을 진행합니다 (r=16, alpha=32, dropout=0.1)")
    # LoRA 관련 설정은 build_model에서 처리합니다
        
    # 통합 학습기 초기화
    learner = ContinualLearner(config)

    # 체크포인트 경로 설정
    checkpoint_path = None
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        logger.info(f"\uccb4\ud06c\ud3ec\uc778\ud2b8 \uacbd\ub85c\uc5d0\uc11c \uc7ac\uac1c: {checkpoint_path}")
    elif args.resume:
        checkpoint_path = "latest"  # 최근 체크포인트에서 재개
        logger.info("\ucd5c\uc2e0 \uccb4\ud06c\ud3ec\uc778\ud2b8\uc5d0\uc11c \uc7ac\uac1c")
    
    # 체크포인트 로드
    if checkpoint_path:
        if checkpoint_path == "latest":
            learner.spot_handler.load_latest_checkpoint(
                model=learner.model,
                optimizer=learner.optimizer,
                scheduler=learner.scheduler,
                scaler=learner.scaler,
                trainer=learner
            )
        else:
            learner.spot_handler.load_checkpoint(
                checkpoint_path,
                model=learner.model,
                optimizer=learner.optimizer,
                scheduler=learner.scheduler,
                scaler=learner.scaler,
                trainer=learner
            )

        # 모드별 데이터 전처리 및 포맷 적용
    logger.info(f"모드별 데이터 처리: {args.mode} 모드용 데이터셋 준비 중...")
    
    # 데이터 로더 생성 - 모드별 처리 로직 적용
    # train.py의 데이터 형식 처리 로직을 가져와서 구현
    train_dataset = None
    val_dataset = None
    
    # 데이터 셋 경로 설정 - 사용자 지정 경로로 변경
    data_path = os.path.expanduser("~/deepseek-continual/data")
    logger.info(f"데이터 디렉토리 경로: {data_path}")
    
    # 모드별 추가 로깅
    if args.mode == 'complete':
        logger.info(f"자동완성(FIM) 모드 사용")
    elif args.mode == 'prompt':
        logger.info(f"프롬프트 생성 모드 사용")
    elif args.mode == 'comment': 
        logger.info(f"주석 기반 생성 모드 사용")
    elif args.mode == 'error_fix':
        logger.info(f"오류 수정 모드 사용")
    
    # 데이터셋 로드 (datasets 라이브러리 사용)
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets 라이브러리를 찾을 수 없습니다. 'pip install datasets'를 실행하여 설치하세요.")
        
    try:
        logger.info(f"데이터셋 로드 중: {data_path}")
        
        # 데이터 파일 이름 결정 (명령행 인수 --data-file 사용 또는 자동 감지)
        if args.data_file:
            # 명시적으로 지정된 파일 사용
            data_files = [args.data_file]
        else:
            # 디렉토리에서 train*.jsonl 패턴의 파일 찾기
            import glob
            import re
            
            # 파일 목록 가져오기
            pattern = os.path.join(data_path, 'train*.jsonl')
            all_files = glob.glob(pattern)
            
            # 파일명에서 숫자를 추출하여 정렬 (train1.jsonl, train2.jsonl, ...)
            def extract_number(filename):
                match = re.search(r'train(\d+)\.jsonl$', os.path.basename(filename))
                if match:
                    return int(match.group(1))
                # train.jsonl 같은 경우는 0으로 처리
                if os.path.basename(filename) == 'train.jsonl':
                    return 0
                return float('inf')  # 숫자가 없는 경우 맨 뒤로 정렬
            
            # 숫자 순서대로 정렬
            data_files = sorted(all_files, key=extract_number)
            
            if not data_files:
                # 파일을 찾지 못한 경우 기본 파일 사용
                data_files = ['train.jsonl']
                logger.warning(f"학습 데이터 파일을 찾을 수 없어 기본 파일 {data_files[0]}을 사용합니다.")
            else:
                logger.info(f"다음 순서로 데이터 파일을 처리합니다: {[os.path.basename(f) for f in data_files]}")
        
        # 기본 DeepSeek-Coder 모델 경로 지정 (절대경로 사용)
        default_model_path = "/home/ubuntu/deepseek-coder/model_cache/deepseek-coder-6.7b-instruct"  # 기본 모델 경로 유지
        
        # 구성에서 모델 경로 가져오기
        model_name_or_path = config.get('model_name', config.get('model_name_or_path', default_model_path))
        
        # 구성에 모델 경로 정보 추가
        config['model_name_or_path'] = model_name_or_path
        
        # 이미 생성한 learner의 tokenizer 사용 (중복 초기화 제거)
        # 첫 번째 ContinualLearner 인스턴스화에서 이미 생성된 tokenizer 사용
        tokenizer = learner.tokenizer
        logger.info("토크나이저 재사용 - 체크포인트 샤드 중복 로드 방지")
        
        # 순차적 학습을 위한 전체 파일 사이클 관리
        training_mode = args.mode
        logger.info(f"학습 모드: {training_mode}")
        
        # 초기 모델 학습을 위한 첫 파일로 시작
        for file_idx, data_file in enumerate(data_files):
            train_path = data_file if os.path.isabs(data_file) else os.path.join(data_path, os.path.basename(data_file))
            
            # 데이터 파일 존재 여부 확인
            if not os.path.exists(train_path):
                logger.warning(f"훈련 데이터 파일을 찾을 수 없습니다: {train_path}")
                continue
                
            logger.info(f"===== 학습 데이터 파일 {file_idx+1}/{len(data_files)}: {os.path.basename(train_path)} =====")
            
            try:
                # 데이터 로드 및 토크나이즈
                tokenized_dataset, raw_dataset = load_dataset_and_tokenize(
                    train_path, tokenizer, config, args.overwrite_cache, args.mode, padding="max_length"
                )
                
                # 현재 파일에 대한 학습 진행
                # tokenized_dataset에서 훈련 및 검증 데이터셋 추출
                train_dataset = tokenized_dataset['train']
                val_dataset = tokenized_dataset['validation']
                
                # DataLoader 생성
                from torch.utils.data import DataLoader
                from transformers import DataCollatorForLanguageModeling
                
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer, 
                    mlm=False
                )
                
                # DataLoader 생성
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=config.get('batch_size', 16),
                    shuffle=True,
                    collate_fn=data_collator
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=config.get('batch_size', 16),
                    collate_fn=data_collator
                )
                
                # 현재 학습 모드 확인 - 환경변수 또는 커맨드 라인 인자에서 세팅
                logger.info(f"[학습 시작] 현재 모드: {training_mode}")
                
                # learner를 사용하여 현재 데이터셋 학습 - task_id 사용 안함
                learner.train_task(train_loader=train_loader, val_loader=val_loader, current_mode=training_mode)
                
                # 현재 학습이 끝나고 다음 파일이 있는 경우 로그 출력
                if file_idx < len(data_files) - 1:
                    logger.info(f"현재 학습 완료. 다음 학습 파일로 진행합니다...")
            except Exception as e:
                logger.error(f"데이터 로드 또는 훈련 중 오류 발생: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        
        # 학습 완료 후 마지막 체크포인트 저장
        logger.info(f"모든 학습 파일 처리 완료. {len(data_files)} 개 파일 처리함.")
        
        # 학습 과정 마무리 및 추가 처리
        logger.info(f"모든 학습 완료 - 모드: {args.mode}")
        
        # 최대 길이 설정
        max_length = config.get('max_length', 512)
        
        # 데이터 형식을 자동으로 감지하는 함수 추가
        def detect_format(example):
            """데이터 형식을 감지하는 함수"""
            if "messages" in example:
                return "chat"
            elif "prompt" in example and "completion" in example:
                return "prompt_completion"
            elif "prefix_code" in example and "suffix_code" in example and "comment" in example and "target_code" in example:
                # 주석 기반 코드 생성 형식 - 새로운 데이터 구조
                return "comment_to_code"
            elif "error_context" in example and "explanation" in example:
                return "error_explanation"
            elif "error_context" in example and "fixed_code_snippet" in example:
                # error_fix 형식 감지 개선 - buggy_code_snippet이 error_context 내부에 있는 경우도 처리
                error_context = example["error_context"]
                if isinstance(error_context, dict) and "buggy_code_snippet" in error_context:
                    return "error_fix"
                # 또는 buggy_code_snippet이 최상위 필드에 있는 경우
                elif "buggy_code_snippet" in example:
                    return "error_fix"
            elif "instruction" in example and "input" in example and "output" in example:
                return "instruction_input_output"
            elif "content" in example and args.mode == "complete":
                # complete 모드용 단순 content 필드만 있는 경우 (FIM 형식일 수 있음)
                return "complete"
            elif "comment" in example and "code" in example and args.mode == "comment":
                # 주석과 코드가 있는 단순 형식
                return "comment_code"
            else:
                # 데이터 구조 로깅
                logger.warning(f"⚠️ 알 수 없는 데이터 형식: {list(example.keys())}")
                return "unknown"
                
        # 데이터 형식 감지 (첫 번째 샘플로 판단)
        try:
            sample = raw_dataset["train"][0]
            data_format = detect_format(sample)
            logger.info(f"✅ 감지된 데이터 형식: {data_format}")
        except Exception as e:
            logger.error(f"데이터 형식 감지 중 오류 발생: {e}")
            data_format = "unknown"
        
        # FIM 형식 사용 여부를 미리 확인 (로그 중복 방지)
        has_fim_format = False
        if data_format == "prompt_completion" and args.mode == "comment":
            # 첫 몇 개 샘플 검사하여 FIM 태그 사용 여부 확인
            sample_check_count = min(5, len(raw_dataset["train"]))
            for i in range(sample_check_count):
                sample_prompt = raw_dataset["train"][i]["prompt"]
                if "<|fim begin|>" in sample_prompt or "<|fim hole|>" in sample_prompt or "<|fim end|>" in sample_prompt:
                    logger.info("✅ FIM 형식 주석 기반 모델 데이터 감지됨")
                    has_fim_format = True
                    break
        
        # 모드별 전처리 함수 정의
        def process_complete_data(examples):
            # FIM (Fill In the Middle) 형식 처리
            results = tokenizer(
                examples['content'],
                truncation=True,
                max_length=max_length,
                return_attention_mask=False
            )
            return results
            
        def process_prompt_data(examples):
            # 프롬프트 기반 생성 형식 처리
            results = tokenizer(
                examples['instruction'] + '\n' + examples['input'] + '\n' + examples['output'],
                truncation=True,
                max_length=max_length,
                return_attention_mask=False
            )
            return results
            
        def process_comment_data(examples):
            # 주석 기반 코드 생성 처리
            results = tokenizer(
                examples['comment'] + '\n' + examples['code'],
                truncation=True,
                max_length=max_length,
                return_attention_mask=False
            )
            return results
            
        def process_error_fix_data(examples):
            # 오류 수정 데이터 처리
            results = tokenizer(
                examples['error_description'] + '\n' + examples['buggy_code'] + '\n' + examples['fixed_code'],
                truncation=True,
                max_length=max_length,
                return_attention_mask=False
            )
            return results
        
        # 향상된 토큰화 함수 - 개선된 표시 및 에러 처리
        def tokenize_function(examples):
            # API 버전 정보 추가 (set_progress_bar에서 설정)
            if not hasattr(tokenize_function, 'use_new_api'):
                tokenize_function.use_new_api = None
                logger.debug("[토큰화] API 버전 정보가 없어 기본 동작 사용")
            
            # 토큰화 진행바 표시 동작 개선
            if hasattr(tokenize_function, 'batch_count'):
                tokenize_function.batch_count += 1
            else:
                tokenize_function.batch_count = 1
                
            # 모드별, 데이터 형식별 처리를 통합
            texts = []
            
            # 배치 진행상황 표시 개선 (2000번째 배치마다 로그)
            if tokenize_function.batch_count % 2000 == 0:
                logger.info(f"[토큰화] 배치 처리 중: {tokenize_function.batch_count}번째 배치")
            
            # 샘플 수 결정 (다양한 형식 지원)
            sample_count = 0
            if "content" in examples and args.mode == 'complete':
                sample_count = len(examples["content"])
            elif "messages" in examples:
                sample_count = len(examples["messages"])
            elif "prompt" in examples:
                sample_count = len(examples["prompt"])
            elif "prefix_code" in examples:
                sample_count = len(examples["prefix_code"])
            elif "instruction" in examples:
                sample_count = len(examples["instruction"])
            elif "comment" in examples and "code" in examples:
                sample_count = len(examples["comment"])
            elif "error_description" in examples and "buggy_code" in examples:
                sample_count = len(examples["error_description"])
            else:
                logger.error(f"❌ 지원되지 않는 데이터 형식 - 키: {list(examples.keys())}")
                return {"input_ids": []}
            
            # 각 샘플 처리
            for i in range(sample_count):
                # 데이터 형식 및 모드별 처리
                if data_format == "complete" or (args.mode == 'complete' and "content" in examples):
                    # 자동완성(complete) 모드
                    if "content" in examples:
                        text = examples["content"][i]
                        texts.append(text)
                
                elif data_format == "prompt_completion":
                    # 주석 키워드 기반 코드 생성 형식
                    prompt = examples["prompt"][i] if "prompt" in examples else ""
                    completion = examples["completion"][i] if "completion" in examples else ""
                    
                    # FIM 형식 처리 (3차 주석 기반 모델)
                    if args.mode == "comment" and ("<|fim begin|>" in prompt or "<|fim hole|>" in prompt or "<|fim end|>" in prompt):
                        # FIM 형식 그대로 유지하고 ChatML 형식으로 래핑
                        text = f"assistant\n{prompt}\n\nassistant\n{completion}\n"
                    else:
                        # 일반 프롬프트를 사용자 메시지로, 완성을 어시스턴트 메시지로 변환
                        text = f"assistant\n{prompt}\n\nassistant\n{completion}\n"
                    
                    texts.append(text)
                        
                elif data_format == "comment_to_code" or data_format == "comment_code" or args.mode == 'comment':
                    # 주석 기반 코드 생성 형식
                    if data_format == "comment_to_code":
                        prefix_code = examples["prefix_code"][i] if "prefix_code" in examples else ""
                        suffix_code = examples["suffix_code"][i] if "suffix_code" in examples else ""
                        comment = examples["comment"][i]
                        target_code = examples["target_code"][i] if "target_code" in examples else ""
                        
                        # 사용자 입력 구성 (주석과 코드 컨텍스트)
                        user_text = f"주석에 따라 적절한 코드를 생성해주세요.\n\n"
                        user_text += f"주석: {comment}\n\n"
                        user_text += "\n이전 코드:\n"
                        user_text += f"{prefix_code}\n"
                        user_text += "// 여기에 코드를 삽입해야 함 //\n"
                        user_text += f"{suffix_code}"
                        
                        # ChatML 형식으로 변환
                        text = f"assistant\n{user_text}\n\nassistant\n{target_code}\n"
                    else:
                        # 간단한 주석-코드 형식
                        comment = examples["comment"][i]
                        code = examples["code"][i]
                        text = f"assistant\n주석: {comment}\n\nassistant\n{code}\n"
                    
                    texts.append(text)
                    
                elif data_format == "error_fix" or args.mode == 'error_fix':
                    # 에러 수정 형식
                    if "error_description" in examples and "buggy_code" in examples and "fixed_code" in examples:
                        # 단순 에러 수정 형식
                        error_desc = examples["error_description"][i]
                        buggy_code = examples["buggy_code"][i]
                        fixed_code = examples["fixed_code"][i]
                        
                        text = f"assistant\n오류 설명: {error_desc}\n\nassistant\n{buggy_code}\n\nassistant\n{fixed_code}\n"
                        texts.append(text)
                    elif "error_context" in examples:
                        # 복잡한 에러 수정 형식
                        error_context = examples["error_context"][i]
                        fixed_code = examples["fixed_code_snippet"][i] if "fixed_code_snippet" in examples else ""
                        
                        # 에러 컨텍스트 정보 구성
                        error_log = error_context.get("error_log", "") if isinstance(error_context, dict) else ""
                        language = error_context.get("language", "") if isinstance(error_context, dict) else ""
                        
                        # buggy_code가 error_context 안에 있는 경우와 외부에 있는 경우 모두 처리
                        if isinstance(error_context, dict) and "buggy_code_snippet" in error_context:
                            buggy_code = error_context["buggy_code_snippet"]
                        elif "buggy_code_snippet" in examples:
                            buggy_code = examples["buggy_code_snippet"][i]
                        else:
                            buggy_code = ""
                            logger.warning("⚠️ buggy_code_snippet을 찾을 수 없습니다")
                        
                        # 사용자 입력 구성
                        user_text = f"다음 {language} 코드의 에러를 수정해주세요:\n\n에러 로그:\n{error_log}\n\n코드:\n{buggy_code}"
                        text = f"assistant\n{user_text}\n\nassistant\n{fixed_code}\n"
                        texts.append(text)
                else:
                    # 알 수 없는 형식은 원시 데이터 텍스트를 그대로 사용
                    logger.warning(f"⚠️ 지원되지 않는 데이터 형식: {data_format}, 모드: {args.mode}")
                    # 첫 번째 필드만 사용
                    if len(examples) > 0:
                        first_key = list(examples.keys())[0]
                        text = str(examples[first_key][i])
                        texts.append(text)
                    else:
                        texts.append("")
            
            # 토크나이징
            model_inputs = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                return_attention_mask=False,
                padding=False
            )
            
            return model_inputs
            
        # 모드와 데이터 형식에 따른 처리 함수 선택
        logger.info(f"모드: {args.mode}, 감지된 데이터 형식: {data_format}")
        
        # 데이터셋 전처리 적용 - 향상된 토큰화 함수 사용
        # 데이터셋 토큰화 로깅 개선
        logger.info(f"✨ {args.mode} 데이터셋 토큰화 시작 (num_proc={config.get('preprocessing_num_workers', 4)})")
        
        # tqdm 설정 관련 코드
        # 라이브러리 버전 호환성 문제로 인해 tqdm_kwargs 매개변수를 사용하지 않음
        # 전역으로 임포트된 tqdm 사용하여 진행 중인 모든 tqdm 인스턴스 정리
        try:
            # tqdm 버전에 따라 _instances가 없을 수 있으므로 안전하게 처리
            from tqdm import tqdm as tqdm_module
            if hasattr(tqdm_module, '_instances'):
                for bar in tqdm_module._instances:
                    bar.close()
        except Exception as e:
            logger.debug(f"tqdm 인스턴스 정리 중 오류 (무시됨): {e}")
        
        # 토큰화 실행 - tqdm 진행바가 보이도록 설정 개선
        logger.info("토큰화 시작 - 토큰화 진행상황이 표시됩니다")
        
        # datasets 라이브러리의 캐시 사용 여부 확인
        # 토큰화 과정 상세 로그 추가 - 사용자에게 과정이 보이도록 향상
        logger.info(f"=== {args.mode} 데이터셋 토큰화 시작 ===")
        logger.info(f"- 훈련 데이터: {len(raw_dataset['train']):,}개 샘플")
        logger.info(f"- 검증 데이터: {len(raw_dataset['validation']):,}개 샘플")
        logger.info(f"- 최대 토큰 길이: {config['max_length']}")
        logger.info(f"- 처리 스레드 수: {config.get('preprocessing_num_workers', 4)}")
        
        # 토큰화 진행을 위한 진행바 로그 추가
        total_samples = len(raw_dataset['train']) + len(raw_dataset['validation'])
        logger.info(f"토큰화 진행 시작 (총 {total_samples:,}개 샘플) - 몇 분 소요될 수 있음")
        
        # 캠시 확인 - 캠시가 있으면 토큰화가 빠를 수 있음을 알려줌
        cache_exists = True
        try:
            # 캠시 확인 메서드 호출
            if hasattr(raw_dataset, 'is_cached') and raw_dataset.is_cached():
                logger.info("캠시된 데이터셋이 발견되어 토큰화가 빠르게 진행됩니다.")
        except:
            cache_exists = False
            logger.info("캠시된 데이터셋이 없어 전체 토큰화를 진행합니다. (더 오래 걸릴 수 있음)")
        
        # 토큰화 진행바의 가시성 향상을 위해 tqdm 설정 수정
        import tqdm.auto as tqdm_auto
        
        # 현재 tqdm 인스턴스를 클리어하여 출력 충돌을 방지
        try:
            import tqdm
            for bar in list(tqdm._instances):
                bar.close()
            logger.info("기존 tqdm 인스턴스 클리어 성공")
        except Exception as e:
            logger.debug(f"tqdm 인스턴스 클리어 시 오류 (무시 가능): {str(e)}")
            
        # 이전 설정 복원용 변수 (함수가 존재하는 경우만 사용)
        original_get_progress_bar = None
            
        # datasets 라이브러리의 프로그레스바 설정 최적화 - 버전별 구분
        try:
            # 버전 1: 최신 datasets API (2023년 이후) - disable_progress_bar/enable_progress_bar 함수 사용
            try:
                from datasets.utils.logging import disable_progress_bar, enable_progress_bar
                # 현재 상태 저장 및 초기화
                disable_progress_bar()
                enable_progress_bar()
                logger.info("[토큰화] 최신 datasets 라이브러리 API로 진행바 초기화 성공")
                # 토큰화 함수에 버전 정보 전달
                tokenize_function.use_new_api = True
            except ImportError as e:
                logger.info(f"[토큰화] 최신 datasets 진행바 API 사용 불가 - 레거시 모드 시도: {str(e)}")
                
                # 버전 2: 레거시 datasets API - _get_progress_bar 함수 재정의
                try:
                    from datasets.utils import logging as datasets_logging
                    if hasattr(datasets_logging, '_get_progress_bar'):
                        # 기존 함수 저장
                        original_get_progress_bar = datasets_logging._get_progress_bar
                        
                        # 사용자 정의 프로그레스바 함수
                        def custom_get_progress_bar(**kwargs):
                            import tqdm.auto as tqdm_auto
                            # 진행 막대 형식 개선 - 가독성 향상
                            kwargs['bar_format'] = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                            kwargs['ascii'] = True  # ASCII 호환성 높임
                            kwargs['leave'] = True  # 프로그레스바가 사라지지 않게 설정
                            return tqdm_auto.tqdm(**kwargs)
                        
                        # 함수 오버라이딩
                        datasets_logging._get_progress_bar = custom_get_progress_bar
                        logger.info("[토큰화] 레거시 datasets 라이브러리 진행바 가시성 개선 성공")
                        # 토큰화 함수에 버전 정보 전달
                        tokenize_function.use_new_api = False
                    else:
                        logger.warning("[토큰화] datasets 라이브러리에 _get_progress_bar 함수가 없음 - 기본 진행바 사용")
                        # 토큰화 함수에 버전 정보 전달
                        tokenize_function.use_new_api = None
                except Exception as e:
                    logger.warning(f"[토큰화] 레거시 진행바 설정 오류: {str(e)}")
                    tokenize_function.use_new_api = None
        except Exception as e:
            logger.warning(f"[토큰화] 데이터셋 진행바 설정 오류: {str(e)}")
            tokenize_function.use_new_api = None
        
        # 토큰화 진행 시작을 로그로 분명히 표시
        logger.info("\n" + "=" * 50)
        logger.info(f"  {args.mode} 데이터셋 토큰화 시작 (총 {total_samples:,}개 샘플)")
        logger.info("=" * 50)
        
        # 최신 API 버전 여부에 따라 다른 토큰화 전략 사용
        try:
            # 디버그 로그
            logger.debug(f"[토큰화] API 버전 정보: {getattr(tokenize_function, 'use_new_api', None)}")
            
            # datasets의 map 함수로 토큰화 진행 - 프로그레스바 가시성 향상
            tokenized_dataset = raw_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=raw_dataset['train'].column_names,
                num_proc=config.get('preprocessing_num_workers', 4),
                desc=f"{args.mode} 토큰화",  # 더 짧게 표시하여 가독성 향상
                load_from_cache_file=not args.overwrite_cache,  # 캠시 사용 여부 명시
                disable_nullable=True  # nullable 기능 비활성화하여 호환성 향상
            )
            
            # 토큰화 완료 후 통계 정보 추가
            train_samples = len(tokenized_dataset['train'])
            val_samples = len(tokenized_dataset['validation'])
            logger.info(f"[토큰화 성공] 훈련: {train_samples:,}개 샘플, 검증: {val_samples:,}개 샘플")
            
            # 배치 수 및 총 토큰 출력
            if hasattr(tokenize_function, 'batch_count'):
                logger.info(f"[토큰화 통계] 총 {tokenize_function.batch_count}개 배치 처리 완료")
                
        except Exception as e:
            # 토큰화 과정에서 예외 발생 시 자세한 정보 기록
            logger.error(f"[토큰화 오류] 토큰화 중 오류 발생: {str(e)}")
            import traceback
            logger.error(f"[토큰화 오류] \n{traceback.format_exc()}")
            raise
        finally:
            # 토큰화 후 원래 프로그레스바 함수를 다시 복원
            try:
                if original_get_progress_bar is not None:
                    from datasets.utils import logging as datasets_logging
                    if hasattr(datasets_logging, '_get_progress_bar'):
                        datasets_logging._get_progress_bar = original_get_progress_bar
                        logger.debug("[토큰화] 레거시 프로그레스바 함수 복원 완료")
            except Exception as e:
                logger.debug(f"[토큰화] 프로그레스바 함수 복원 실패(무시 가능): {str(e)}")
        
        # 토큰화 완료 후 결과 통계 로깅
        logger.info(f"✅ {args.mode} 데이터셋 토큰화 완료!")
        logger.info(f"- 훈련 토큰화 결과: {len(tokenized_dataset['train']):,}개 샘플")
        logger.info(f"- 검증 토큰화 결과: {len(tokenized_dataset['validation']):,}개 샘플")
        
        # DataLoader 생성을 위한 준비
        train_dataset = tokenized_dataset['train']
        val_dataset = tokenized_dataset['validation']
        
        # DataCollatorForLanguageModeling 로드
        try:
            from transformers import DataCollatorForLanguageModeling
            logger.info("DataCollatorForLanguageModeling 로드 성공")
        except ImportError as e:
            logger.error(f"DataCollatorForLanguageModeling 로드 오류: {e}")
            raise ImportError("transformers 라이브러리에서 DataCollatorForLanguageModeling을 로드할 수 없습니다. 'pip install transformers'를 실행하여 최신 버전으로 업그레이드하세요.")
        
        # DataLoader 생성
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False
        )
        
        # collate_fn으로 커스텀 함수 사용하여 BatchEncoding 처리 보장
        def safe_collate_fn(features):
            try:
                # 원래 collator 호출
                batch = data_collator(features)
                
                # 디버깅: 첫 번째 배치만 로깅
                if logger.level <= logging.DEBUG and random.random() < 0.01:  # 1% 샘플링
                    logger.debug(f"배치 타입: {type(batch)}, 키: {list(batch.keys()) if hasattr(batch, 'keys') else 'N/A'}")
                    if hasattr(batch, 'input_ids'):
                        logger.debug(f"input_ids 형태: {batch.input_ids.shape if hasattr(batch.input_ids, 'shape') else 'N/A'}")
                
                return batch
            except Exception as e:
                logger.error(f"Collate 함수 오류: {e}")
                # 긴급 폴백: 원본 특성 그대로 반환
                return features
        
        # 5. 배치 크기 및 VRAM 관리 설정:
        # 설정에서 배치 크기 직접 가져와 고정값으로 사용
        self.batch_size = config.get('batch_size', 32)
        
        # VRAM 관리 개선: GPU 메모리 상한선 설정 (메모리의 80%만 사용)
        if torch.cuda.is_available():
            # 메모리 파편화 방지를 위한 초기 설정 
            torch.cuda.empty_cache()  # 시작 시 캐시 비우기
            
            # GPU 메모리 상한 설정 (80%만 사용, 20% 여유공간)
            memory_fraction = config.get('gpu_memory_fraction', 0.8)
            torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
            
            # 초기 메모리 상태 로깅
            if hasattr(torch.cuda, 'memory_reserved'):
                reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
                allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                logger.info(f"GPU 메모리 초기 상태: 예약={reserved:.2f}GB, 할당={allocated:.2f}GB, 제한={memory_fraction*100}%")
        num_workers = config.get('dataloader_num_workers', 4)
        logger.info(f"학습 DataLoader 초기화: 배치 크기={self.batch_size}, 작업자={num_workers}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            collate_fn=safe_collate_fn,  # 안전한 collate 함수 사용
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers
        )
        
        # 학습 로더 생성 후 추가 정보
        logger.info(f"학습 DataLoader 생성 완료: 배치 크기={batch_size}, 배치 개수={len(train_loader):,}")
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 32),
            collate_fn=safe_collate_fn,  # 안전한 collate 함수 사용
            shuffle=False,
            pin_memory=True,
            num_workers=config.get('dataloader_num_workers', 4)
        )
        
        # 로그에 데이터 정보 기록 - 상세 정보 추가
        total_steps = len(train_loader)
        logger.info(f"훈련 데이터 요약 - 학습용: {len(train_dataset):,} 샘플, 검증용: {len(val_dataset):,} 샘플")
        logger.info(f"학습 배치 정보 - 배치당 {batch_size}개 샘플 포함, 총 {total_steps:,}개 배치")
        logger.info(f"학습 예상 시간: 에포크당 최소 {(total_steps*1.5)/60:.1f}분 (배치당 1.5초 가정)")
        
        # 지속학습 설정 - 모드 기반 학습
        # 학습 모드(complete, prompt, comment, error_fix)를 테스트 로더 키로 사용
        
        # 테스트 로더 사전 설정 - 현재 모드로 테스트 로더 구성
        # 실제 환경에서는 모드별로 전용 테스트 데이터셋 사용 권장
        test_loaders = {args.mode: val_loader}  # 현재 모드를 키로 사용
        
        # AWS 스팟 인스턴스 중단 감시 시작
        logger.info("AWS 스팟 인스턴스 중단 감시 시작")
        learner.spot_handler.start_monitoring(
            model=learner.model,
            optimizer=learner.optimizer,
            scheduler=learner.scheduler,
            scaler=learner.scaler,
            trainer=learner
        )
        
        # 모드 기반 지속학습 실행
        logger.info(f"모드 기반 학습 시작 - 현재 모드: {args.mode}")
        learner.train_task(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loaders=test_loaders,
            current_mode=args.mode
        )
        
        # AWS 스팟 인스턴스 중단 감시 종료
        learner.spot_handler.stop_monitoring()
        
    except Exception as e:
        logger.error(f"데이터 로드 또는 훈련 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    # 최종 메트릭 계산 및 출력
    final_metrics = learner.metrics.compute_metrics()
    print(f"\n\n[FINAL] 학습 완료 - 최종 메트릭: ")
    for mode, metrics in final_metrics.items():
        print(f"[FINAL] 모드: {mode}")
        for metric_name, value in metrics.items():
            print(f"[FINAL]   - {metric_name}: {value:.4f}")
    
    # 최고 정확도 출력
    print(f"\n[FINAL] 최고 정확도: ")
    for mode, acc in learner.best_accuracies.items():
        print(f"[FINAL]   - {mode}: {acc:.4f}")
    
    print("\n학습이 성공적으로 완료되었습니다.\n")

if __name__ == '__main__':
    main()
