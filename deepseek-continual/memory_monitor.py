"""
메모리 모니터링 및 관리 시스템
- VRAM 사용량 실시간 모니터링
- 임계치 도달 시 자동 메모리 정리
- OOM 예방을 위한 선제적 조치
"""

import os
import time
import threading
import logging
from typing import Optional, Callable

import torch

# GPUtil 사용 가능 여부 확인 (선택적 의존성)
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logging.warning("GPUtil 라이브러리를 찾을 수 없습니다. 기본 PyTorch 메모리 모니터링만 사용됩니다.")

# 로깅 설정
logger = logging.getLogger(__name__)

class MemoryMonitor:
    """
    GPU 메모리 사용량 모니터링 및 관리 시스템
    - 별도 스레드로 실행되어 주기적으로 메모리 사용량 체크
    - 임계치 초과 시 자동으로 torch.cuda.empty_cache() 호출
    - 위험 수준 도달 시 콜백 함수 실행 (예: 배치 크기 감소)
    """

    def __init__(
        self, 
        warning_threshold: float = 0.8,  # 경고 임계치 (80%)
        critical_threshold: float = 0.9,  # 위험 임계치 (90%)
        check_interval: float = 5.0,     # 초 단위 확인 주기
        device_id: int = 0,              # 모니터링할 GPU 장치 ID
        critical_callback: Optional[Callable] = None,  # 위험 상태 콜백
        enabled: bool = True             # 모니터링 활성화 여부
    ):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval
        self.device_id = device_id
        self.critical_callback = critical_callback
        self.enabled = enabled and torch.cuda.is_available()
        
        # 내부 상태
        self.monitoring_thread = None
        self.keep_monitoring = False
        self.last_memory_usage = 0.0
        
        # GPU 정보 초기화
        if self.enabled:
            self._log_gpu_info()
    
    def _log_gpu_info(self) -> None:
        """현재 GPU 정보 로깅"""
        try:
            device_name = torch.cuda.get_device_name(self.device_id)
            total_memory = torch.cuda.get_device_properties(self.device_id).total_memory / (1024**3)
            logger.info(f"GPU 정보: {device_name}, 총 메모리: {total_memory:.2f}GB")
            
            if GPUTIL_AVAILABLE:
                gpus = GPUtil.getGPUs()
                if self.device_id < len(gpus):
                    gpu = gpus[self.device_id]
                    logger.info(f"GPUtil 정보: {gpu.name}, 메모리 사용률: {gpu.memoryUtil:.2%}, "
                               f"사용 메모리: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB")
        except Exception as e:
            logger.warning(f"GPU 정보 로깅 중 오류: {e}")
    
    def start(self) -> None:
        """모니터링 스레드 시작"""
        if not self.enabled:
            logger.info("GPU 메모리 모니터링이 비활성화 되었습니다.")
            return
            
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            logger.warning("메모리 모니터링 스레드가 이미 실행 중입니다.")
            return
            
        self.keep_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="VRAM-Monitor"
        )
        self.monitoring_thread.start()
        logger.info(f"GPU 메모리 모니터링 시작: 경고 임계값={self.warning_threshold:.1%}, "
                  f"위험 임계값={self.critical_threshold:.1%}, 확인 주기={self.check_interval}초")
    
    def stop(self) -> None:
        """모니터링 스레드 정지"""
        self.keep_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
            logger.info("GPU 메모리 모니터링 종료")
    
    def _get_memory_usage(self) -> float:
        """현재 GPU 메모리 사용률 확인 (0.0 ~ 1.0)"""
        try:
            if GPUTIL_AVAILABLE:
                # GPUtil로 메모리 사용률 확인 (더 정확한 정보 제공)
                gpus = GPUtil.getGPUs()
                if self.device_id < len(gpus):
                    return gpus[self.device_id].memoryUtil
            
            # PyTorch로 메모리 사용률 추정
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self.device_id)
                reserved = torch.cuda.memory_reserved(self.device_id)
                total = torch.cuda.get_device_properties(self.device_id).total_memory
                
                # 할당된 메모리와 예약된 메모리 중 큰 값 사용
                usage = max(allocated, reserved) / total
                return usage
                
        except Exception as e:
            logger.warning(f"메모리 사용률 확인 중 오류: {e}")
        
        return 0.0  # 확인 실패 시 0 반환
    
    def _clean_memory_if_needed(self, usage: float) -> None:
        """필요 시 메모리 정리 수행"""
        if usage > self.critical_threshold:
            # 위험 수준: 강제 메모리 정리 및 콜백 호출
            logger.warning(f"GPU 메모리 사용률 위험 수준: {usage:.1%}, 강제 정리 수행")
            torch.cuda.empty_cache()
            
            # 위험 수준 콜백 실행
            if self.critical_callback is not None:
                try:
                    self.critical_callback()
                except Exception as e:
                    logger.error(f"메모리 위험 콜백 실행 중 오류: {e}")
            
        elif usage > self.warning_threshold:
            # 경고 수준: 메모리 정리
            logger.info(f"GPU 메모리 사용률 경고 수준: {usage:.1%}, 캐시 정리")
            torch.cuda.empty_cache()
    
    def _monitoring_loop(self) -> None:
        """메모리 모니터링 메인 루프"""
        while self.keep_monitoring:
            try:
                # 메모리 사용률 확인
                usage = self._get_memory_usage()
                
                # 이전 사용률과 큰 차이가 있을 때만 기록
                if abs(usage - self.last_memory_usage) > 0.05:
                    logger.info(f"GPU 메모리 사용률: {usage:.1%}")
                    self.last_memory_usage = usage
                
                # 필요시 메모리 정리
                self._clean_memory_if_needed(usage)
                
            except Exception as e:
                logger.error(f"메모리 모니터링 중 오류: {e}")
            
            # 체크 주기만큼 대기
            time.sleep(self.check_interval)

def configure_pytorch_memory_allocator() -> None:
    """
    PyTorch 메모리 할당자 최적화 (환경 변수 설정)
    - 최적의 설정을 위해서는 스크립트 시작 전 환경 변수 설정 필요
    """
    try:
        # 환경 변수가 이미 설정되어 있는지 확인
        existing_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
        
        if not existing_conf:
            # 권장 설정
            # - max_split_size_mb: 메모리 단편화 방지 (작은 값일수록 효율적)
            # - garbage_collection_threshold: 메모리 회수 임계값 (낮을수록 적극적)
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.6'
            logger.info("PyTorch 메모리 할당자 최적화 설정 완료")
        else:
            logger.info(f"이미 PyTorch 메모리 할당자 설정 있음: {existing_conf}")
    except Exception as e:
        logger.warning(f"PyTorch 메모리 할당자 설정 중 오류: {e}")

# 모듈 로드 시 PyTorch 메모리 할당자 설정
configure_pytorch_memory_allocator()
