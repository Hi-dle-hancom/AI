#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
메모리 관리자

GPU 메모리 관리, OOM 처리, 메모리 최적화를 담당합니다.
"""

import os
import gc
import torch
import logging
import psutil
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class MemoryManager:
    """GPU 메모리 관리 및 OOM 처리를 담당하는 매니저"""
    
    def __init__(self, config: Dict[str, Any]):
        """메모리 매니저 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.gpu_memory_fraction = config.get('gpu_memory_fraction', 0.75)
        self.min_batch_size = 1
        
        # CUDA 메모리 설정 초기화
        self.setup_cuda_memory()
    
    def setup_cuda_memory(self):
        """CUDA 메모리 환경 설정"""
        try:
            # CUDA 메모리 단편화 방지 환경변수 설정
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
            
            if torch.cuda.is_available():
                # GPU 메모리 fraction 설정
                torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)
                
                # 초기 CUDA 캐시 정리
                torch.cuda.empty_cache()
                
                logger.info(f"CUDA 메모리 설정 완료: {self.gpu_memory_fraction*100:.0f}% 제한")
                
        except Exception as e:
            logger.error(f"CUDA 메모리 설정 실패: {e}")
    
    def get_memory_info(self) -> Dict[str, float]:
        """현재 메모리 사용량 정보 반환
        
        Returns:
            Dict[str, float]: 메모리 정보 (GB 단위)
        """
        memory_info = {}
        
        try:
            # 시스템 메모리
            system_memory = psutil.virtual_memory()
            memory_info['system_used'] = system_memory.used / (1024**3)
            memory_info['system_available'] = system_memory.available / (1024**3)
            memory_info['system_percent'] = system_memory.percent
            
            # GPU 메모리
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    memory_info[f'gpu_{i}_allocated'] = allocated
                    memory_info[f'gpu_{i}_reserved'] = reserved
                    
        except Exception as e:
            logger.error(f"메모리 정보 수집 실패: {e}")
            
        return memory_info
    
    def log_memory_status(self, context: str = ""):
        """메모리 상태 로깅
        
        Args:
            context: 로깅 컨텍스트
        """
        try:
            memory_info = self.get_memory_info()
            
            if context:
                logger.info(f"[{context}] 메모리 상태:")
            else:
                logger.info("메모리 상태:")
                
            # 시스템 메모리
            if 'system_percent' in memory_info:
                logger.info(f"  시스템: {memory_info['system_percent']:.1f}% 사용 "
                          f"({memory_info['system_used']:.1f}GB / {memory_info['system_available']:.1f}GB 가용)")
            
            # GPU 메모리
            for key in memory_info:
                if key.startswith('gpu_') and key.endswith('_allocated'):
                    gpu_id = key.split('_')[1]
                    allocated = memory_info[key]
                    reserved = memory_info.get(f'gpu_{gpu_id}_reserved', 0)
                    logger.info(f"  GPU {gpu_id}: {allocated:.2f}GB 할당 / {reserved:.2f}GB 예약")
                    
        except Exception as e:
            logger.error(f"메모리 상태 로깅 실패: {e}")
    
    def cleanup_memory(self, aggressive: bool = False):
        """메모리 정리
        
        Args:
            aggressive: 적극적인 정리 여부
        """
        try:
            # 가비지 컬렉션
            if aggressive:
                # 모든 세대 수집
                for generation in range(3):
                    gc.collect(generation)
            else:
                gc.collect()
            
            # CUDA 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if aggressive:
                    torch.cuda.synchronize()
                    # 모든 GPU에 대해 정리
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            torch.cuda.empty_cache()
            
            logger.debug("메모리 정리 완료")
            
        except Exception as e:
            logger.error(f"메모리 정리 실패: {e}")
    
    def handle_oom_error(self, current_batch_size: int = None) -> Tuple[int, int]:
        """OOM 에러 처리
        
        Args:
            current_batch_size: 현재 배치 크기
            
        Returns:
            Tuple[int, int]: (새로운 배치 크기, 새로운 그래디언트 누적 스텝)
        """
        try:
            # 메모리 상태 로깅
            if torch.cuda.is_available():
                try:
                    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
                    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                    logger.warning(f"OOM 당시 GPU 메모리 상태: 예약={reserved:.2f}GB, 할당={allocated:.2f}GB")
                except Exception as e:
                    logger.debug(f"OOM 시 메모리 상태 확인 실패: {e}")
            
            # 적극적인 메모리 정리
            self.cleanup_memory(aggressive=True)
            
            # 배치 크기 조정
            if current_batch_size is None:
                current_batch_size = self.config.get('batch_size', 2)
            
            new_batch_size = max(self.min_batch_size, current_batch_size // 2)
            
            # 그래디언트 누적 스텝 조정 (유효 배치 크기 유지)
            current_grad_steps = self.config.get('gradient_accumulation_steps', 8)
            new_grad_steps = current_grad_steps * 2
            
            logger.warning(f"OOM 발생: 배치 크기 {current_batch_size} → {new_batch_size}로 50% 감소")
            logger.info(f"그래디언트 누적 스텝: {current_grad_steps} → {new_grad_steps}로 증가")
            
            return new_batch_size, new_grad_steps
            
        except Exception as e:
            logger.error(f"OOM 처리 중 오류: {e}")
            return self.min_batch_size, self.config.get('gradient_accumulation_steps', 8) * 2
    
    def check_memory_availability(self, required_gb: float = 2.0) -> bool:
        """메모리 가용성 체크
        
        Args:
            required_gb: 필요한 메모리 (GB)
            
        Returns:
            bool: 메모리 충분 여부
        """
        try:
            if torch.cuda.is_available():
                # GPU 메모리 체크
                props = torch.cuda.get_device_properties(0)
                total_memory = props.total_memory / (1024**3)
                allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
                available_memory = total_memory - allocated_memory
                
                if available_memory < required_gb:
                    logger.warning(f"GPU 메모리 부족: {available_memory:.1f}GB 가용 (필요: {required_gb:.1f}GB)")
                    return False
            
            # 시스템 메모리 체크
            system_memory = psutil.virtual_memory()
            available_system_gb = system_memory.available / (1024**3)
            
            if available_system_gb < required_gb:
                logger.warning(f"시스템 메모리 부족: {available_system_gb:.1f}GB 가용 (필요: {required_gb:.1f}GB)")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"메모리 가용성 체크 실패: {e}")
            return False
    
    def optimize_for_training(self):
        """학습을 위한 메모리 최적화"""
        try:
            # 초기 메모리 정리
            self.cleanup_memory(aggressive=True)
            
            # CUDA 설정 최적화
            if torch.cuda.is_available():
                # 메모리 풀 설정
                torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)
                
                # 동기화
                torch.cuda.synchronize()
            
            # 메모리 상태 로깅
            self.log_memory_status("학습 최적화 후")
            
            logger.info("학습용 메모리 최적화 완료")
            
        except Exception as e:
            logger.error(f"학습용 메모리 최적화 실패: {e}")
    
    def get_optimal_batch_size(self, model_size_gb: float = 1.0) -> int:
        """최적 배치 크기 추정
        
        Args:
            model_size_gb: 모델 크기 (GB)
            
        Returns:
            int: 추정된 최적 배치 크기
        """
        try:
            if not torch.cuda.is_available():
                return self.config.get('batch_size', 2)
            
            # GPU 메모리 정보
            props = torch.cuda.get_device_properties(0)
            total_memory_gb = props.total_memory / (1024**3)
            allocated_memory_gb = torch.cuda.memory_allocated(0) / (1024**3)
            available_memory_gb = (total_memory_gb * self.gpu_memory_fraction) - allocated_memory_gb
            
            # 배치당 예상 메모리 사용량 (모델 크기의 3-4배)
            memory_per_batch = model_size_gb * 3.5
            
            # 최적 배치 크기 계산
            optimal_batch_size = max(1, int(available_memory_gb / memory_per_batch))
            
            # 설정된 최대값과 비교
            max_batch_size = self.config.get('batch_size', 2)
            optimal_batch_size = min(optimal_batch_size, max_batch_size)
            
            logger.info(f"최적 배치 크기 추정: {optimal_batch_size} "
                       f"(가용 메모리: {available_memory_gb:.1f}GB, 배치당 예상: {memory_per_batch:.1f}GB)")
            
            return optimal_batch_size
            
        except Exception as e:
            logger.error(f"최적 배치 크기 추정 실패: {e}")
            return self.config.get('batch_size', 2)
    
    def monitor_memory_usage(self, threshold_percent: float = 90.0) -> bool:
        """메모리 사용량 모니터링
        
        Args:
            threshold_percent: 경고 임계값 (%)
            
        Returns:
            bool: 임계값 초과 여부
        """
        try:
            memory_info = self.get_memory_info()
            
            # GPU 메모리 체크
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated_key = f'gpu_{i}_allocated'
                    reserved_key = f'gpu_{i}_reserved'
                    
                    if allocated_key in memory_info and reserved_key in memory_info:
                        props = torch.cuda.get_device_properties(i)
                        total_memory_gb = props.total_memory / (1024**3)
                        usage_percent = (memory_info[reserved_key] / total_memory_gb) * 100
                        
                        if usage_percent > threshold_percent:
                            logger.warning(f"GPU {i} 메모리 사용량 높음: {usage_percent:.1f}% "
                                         f"({memory_info[reserved_key]:.1f}GB / {total_memory_gb:.1f}GB)")
                            return True
            
            # 시스템 메모리 체크
            if 'system_percent' in memory_info:
                if memory_info['system_percent'] > threshold_percent:
                    logger.warning(f"시스템 메모리 사용량 높음: {memory_info['system_percent']:.1f}%")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"메모리 모니터링 실패: {e}")
            return False
