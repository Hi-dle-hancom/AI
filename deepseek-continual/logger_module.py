# logger_module.py
import logging
import sys
import os
import time
from typing import Dict, Any, Optional, Union
import threading

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

class TqdmStreamHandler(logging.StreamHandler):
    """tqdm과 호환되는 스트림 핸들러"""
    
    def emit(self, record):
        try:
            msg = self.format(record)
            if TQDM_AVAILABLE:
                tqdm.write(msg)
            else:
                # tqdm 없는 경우 일반적인 방법으로 출력
                stream = self.stream
                stream.write(msg + self.terminator)
                self.flush()
        except Exception:
            self.handleError(record)


class DeepseekLogger:
    """DeepSeek-Coder 프로젝트를 위한 통합 로깅 시스템
    tqdm과 통합되어 진행 막대를 방해하지 않고 로깅"""
    
    def __init__(self, name: str = "deepseek_trainer", 
                 log_dir: Optional[str] = None,
                 level: int = logging.INFO,
                 console_output: bool = True):
        """
        로거 초기화
        
        Args:
            name: 로거 이름
            log_dir: 로그 디렉토리 (None이면 현재 디렉토리에 생성)
            level: 로깅 레벨
            console_output: 콘솔 출력 여부
        """
        self.name = name
        self.level = level  # level 속성 저장
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # 기존 핸들러 제거
        
        # 로그 디렉토리 생성
        if log_dir is None:
            log_dir = './logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # 로그 파일 경로 설정
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        # 표준 로그 포맷터 정의
        standard_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 파일 핸들러 추가
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(standard_formatter)
        file_handler.setLevel(level)
        self.logger.addHandler(file_handler)
        
        # tqdm 호환 콘솔 핸들러 추가 (옵션)
        if console_output:
            # TqdmStreamHandler를 사용해 tqdm 진행 바와 충돌 방지
            console_handler = TqdmStreamHandler(sys.stdout)
            console_handler.setFormatter(standard_formatter)
            console_handler.setLevel(level)
            self.logger.addHandler(console_handler)
        
        self.step = 0
        self.epoch = 0
        self.task_id = None
    
    def update_step(self, step: int, silent: bool = False):
        """현재 학습 단계 업데이트
        
        Args:
            step: 현재 학습 스텝
            silent: True이면 업데이트 시 로그 출력을 하지 않음 (특히 tqdm 사용 시 유용)
        """
        self.step = step
        
    def update_epoch(self, epoch: int):
        """현재 에폭 업데이트"""
        self.epoch = epoch
        
    def update_task(self, task_id: int):
        """현재 태스크 업데이트"""
        self.task_id = task_id
    
    # 기본 로깅 함수들
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
        
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
        
    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
        
    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
        
    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)
    
    # AWS 스팟 인스턴스 관련 로깅
    def spot_instance_event(self, event_type: str, message: str):
        """스팟 인스턴스 이벤트 로깅"""
        self.warning(f"[SPOT_{event_type.upper()}] {message}")
    
    # 학습 관련 로깅 (구조화된 형태)
    def training_status(self, metrics: Dict[str, Any], use_tqdm_write: bool = True, 
                         tqdm_instance: 'tqdm' = None, update_tqdm: bool = True, log_output: bool = True):
        """학습 상태 로깅 - 구조화된 형태로 학습 메트릭 출력 및 tqdm 업데이트
        
        Args:
            metrics: 학습 메트릭 디크션너리 (예: loss, accuracy, lr 등)
            use_tqdm_write: True면 tqdm과 호환되는 방식으로 출력
            tqdm_instance: 진행 상황 표시 막대를 업데이트할 tqdm 인스턴스
            update_tqdm: tqdm_instance가 제공된 경우 상태를 tqdm 막대에 업데이트할지 여부
        """
        # 컨텍스트 정보 구성
        context_parts = []
        if self.epoch is not None:
            context_parts.append(f"Epoch {self.epoch}")
        if self.task_id is not None:
            context_parts.append(f"Task {self.task_id}")
        if self.step is not None:
            context_parts.append(f"Step {self.step}")
        
        context = "[" + ", ".join(context_parts) + "]" if context_parts else ""
        
        # AWS 스팟 인스턴스 학습을 위한 메트릭 출력 포맷팅
        metrics_parts = []
        
        # 중요한 메트릭 먼저 정렬 (다른 메트릭은 이후에 정렬)
        priority_metrics = ["loss", "acc", "accuracy", "lr"]
        priority_values = {}
        
        # 우선순위 메트릭 출력
        for key in priority_metrics:
            if key in metrics:
                value = metrics[key]
                # 메트릭 출력 형식 조정 (부동소수점 포맷팅)
                if isinstance(value, float):
                    formatted_value = f"{value:.6f}" if key == "lr" else f"{value:.4f}"
                else:
                    formatted_value = str(value)
                metrics_parts.append(f"{key}={formatted_value}")
                priority_values[key] = formatted_value
        
        # 나머지 메트릭 출력
        for key, value in metrics.items():
            if key not in priority_metrics:
                # 메트릭 출력 형식 조정
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                metrics_parts.append(f"{key}={formatted_value}")
                
        metrics_str = " | ".join(metrics_parts)
        
        # 최종 로그 메시지 구성
        log_message = f"{context} {metrics_str}" if context else metrics_str
        
        # tqdm 인스턴스 업데이트 (제공된 경우)
        if tqdm_instance is not None and update_tqdm and TQDM_AVAILABLE:
            # tqdm 진행 막대에 중요 메트릭 표시
            postfix_dict = {}
            if "loss" in priority_values:
                postfix_dict["loss"] = priority_values["loss"]
            if "acc" in priority_values or "accuracy" in priority_values:
                postfix_dict["acc"] = priority_values.get("acc", priority_values.get("accuracy", "N/A"))
            if "lr" in priority_values:
                postfix_dict["lr"] = priority_values["lr"]
                
            # 추가 중요 메트릭 포함
            for key, value in metrics.items():
                if key not in priority_metrics and key not in ["step", "epoch"] and key.startswith("val_"):
                    if isinstance(value, float):
                        postfix_dict[key] = f"{value:.4f}"
                    else:
                        postfix_dict[key] = str(value)
            
            # tqdm 설명 및 진행 상황 업데이트
            tqdm_instance.set_postfix(**postfix_dict)
            
        # 로깅 (tqdm과 충돌하지 않도록 내장된 TqdmStreamHandler 사용)
        # log_output이 True인 경우에만 실제 로그 출력
        if log_output:
            self.info(log_message)
    
    # 특수 로깅 함수들
    def model_info(self, model_name: str, params: int, info: Dict[str, Any] = None):
        """모델 정보 로깅"""
        info_str = ""
        if info:
            info_str = ", " + ", ".join([f"{k}: {v}" for k, v in info.items()])
        self.info(f"모델: {model_name}, 파라미터: {params:,}{info_str}")
    
    def batch_info(self, batch_size: int, shapes: Dict[str, Any], memory_usage: float = None):
        """배치 정보 로깅 (첫 배치나 주기적으로만 호출)"""
        memory_str = ""
        if memory_usage:
            memory_str = f", 메모리: {memory_usage:.2f}GB"
        shapes_str = ", ".join([f"{k}: {v}" for k, v in shapes.items()])
        self.debug(f"배치 크기: {batch_size}{memory_str}, 형태: {shapes_str}")
    
    def adapter_status(self, status: str, path: Optional[str] = None):
        """어댑터 상태 로깅"""
        if path:
            self.info(f"어댑터 {status}: {path}")
        else:
            self.info(f"어댑터 {status}")
    
    def checkpoint_event(self, data: Dict[str, Any], level: int = logging.INFO):
        """체크포인트 이벤트 로깅
        
        Args:
            data: 체크포인트 이벤트 데이터 디크셔너리
                (action, path 및 기타 로깅할 정보 포함)
            level: 로깅 레벨
        """
        # 필수 요소 확인
        action = data.get("action", "unknown")
        path = data.get("path", "unknown")
        step = data.get("step")
        
        # 기본 메시지 구성
        step_info = f" (스텝 {step})" if step is not None else ""
        msg = f"체크포인트 {action}: {path}{step_info}"
        
        # 추가 정보가 있는 경우 추가
        extra_info = ", ".join([f"{k}: {v}" for k, v in data.items() 
                            if k not in ("action", "path", "step") and v is not None])
        
        if extra_info:
            msg += f" [{extra_info}]"
            
        # 지정된 레벨로 로깅
        self.logger.log(level, msg)
        
    def configure_tqdm(self, tqdm_instance, metrics: Dict[str, Any] = None):
        """tqdm 인스턴스 구성 및 초기화
        
        Args:
            tqdm_instance: 구성할 tqdm 인스턴스
            metrics: 초기 메트릭 (없으면 기본 설정만)
        
        Returns:
            구성된 tqdm 인스턴스
        """
        if not TQDM_AVAILABLE:
            return tqdm_instance
            
        # 기본 설정 (진행 표시 색상 및 형식)
        tqdm_instance.set_description(f"학습중")
        
        # 초기 메트릭 표시 (있는 경우)
        if metrics:
            self.training_status(metrics, tqdm_instance=tqdm_instance, update_tqdm=True)
            
        return tqdm_instance
        
    def update_tqdm_progress(self, tqdm_instance, n=1, metrics: Dict[str, Any] = None):
        """tqdm 진행 상황 및 메트릭 업데이트
        
        Args:
            tqdm_instance: 업데이트할 tqdm 인스턴스
            n: 증가할 진행 단위 수
            metrics: 표시할 메트릭
        """
        if not TQDM_AVAILABLE or tqdm_instance is None:
            return
            
        # 진행 상황 업데이트
        tqdm_instance.update(n)
        
        # 메트릭 업데이트
        if metrics:
            self.training_status(metrics, tqdm_instance=tqdm_instance, update_tqdm=True)