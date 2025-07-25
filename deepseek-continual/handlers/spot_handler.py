#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AWS Spot Instance 중단 감지 및 대응 핸들러

AWS 스팟 인스턴스 환경에서 중단 신호를 감지하고 체크포인트 저장을 관리합니다.
SIGTERM 신호 감지 및 EC2 메타데이터 API를 통한 중단 알림을 모니터링합니다.
"""

import os
import time
import signal
import threading
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


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
        self.monitor_thread = threading.Thread(target=self._background_monitor, daemon=True)
        self.monitor_thread.start()
        
        logger.info("백그라운드 스팟 인스턴스 모니터링이 시작되었습니다")

    def stop_monitoring(self):
        """스팟 인스턴스 모니터링 중단"""
        if self.monitoring_active:
            self.monitoring_active = False
            logger.info("스팟 인스턴스 모니터링이 중단되었습니다")

    def _background_monitor(self):
        """백그라운드에서 스팟 인스턴스 중단을 모니터링"""
        while self.monitoring_active:
            try:
                if self.check_interruption():
                    logger.warning("[Background] 스팟 인스턴스 중단 감지!")
                    
                    # 트레이너가 있는 경우 긴급 체크포인트 저장 시도
                    if self.trainer and hasattr(self.trainer, 'save_model'):
                        try:
                            # 현재 스텝 정보 가져오기
                            current_step = getattr(self.trainer, 'global_step', 0)
                            
                            # 긴급 체크포인트 저장 시도
                            if hasattr(self.trainer, 'emergency_checkpoint_save'):
                                success = self.trainer.emergency_checkpoint_save(current_step)
                                if success:
                                    logger.info("[Background] 긴급 체크포인트 저장 성공")
                                else:
                                    logger.error(f"[Background] 긴급 체크포인트 저장 실패: {e}")
                            else:
                                # 일반 저장 메서드 사용
                                import time
                                timestamp = time.strftime("%Y%m%d_%H%M%S")
                                filename = f"interrupt_step{current_step}_{timestamp}.pt"
                                saved_path = self.trainer.save_model(filename, is_spot_interruption=True)
                                logger.info(f"[Background] 긴급 체크포인트 저장 완료: {saved_path}")
                        except Exception as e:
                            logger.error(f"[Background] 긴급 체크포인트 저장 실패: {e}")
                    
                    # 모니터링 중단
                    self.monitoring_active = False
                    break
                
                # 5초마다 체크
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"[Background] 모니터링 오류: {e}")
                time.sleep(10)  # 오류 시 더 긴 대기

    def emergency_checkpoint_save(self, step: int) -> bool:
        """스팟 인스턴스 중단 시 긴급 체크포인트 저장
        
        Args:
            step: 현재 학습 스텝
            
        Returns:
            bool: 체크포인트 저장 성공 여부
        """
        try:
            # 트레이너 객체가 있는 경우 저장 시도
            if hasattr(self, 'trainer') and self.trainer:
                import time
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"interrupt_step{step}_{timestamp}.pt"
                save_path = self.trainer.save_model(filename, is_spot_interruption=True)
                logger.info(f"[SpotInterruptionHandler] 긴급 체크포인트 저장 완료: {save_path}")
                return True
            else:
                logger.error("[SpotInterruptionHandler] 트레이너 객체를 찾을 수 없어 긴급 저장을 수행할 수 없습니다")
                return False
                
        except Exception as e:
            logger.error(f"[SpotInterruptionHandler] 긴급 체크포인트 저장 실패: {e}")
            return False

    def handle_checkpoint_event(self, checkpoint_path: str, step: int):
        """
        체크포인트 저장 후 처리 작업 (심볼릭 링크 업데이트 등)
        """
        try:
            # 심볼릭 링크 업데이트
            checkpoint_dir = os.path.dirname(checkpoint_path)
            latest_link = os.path.join(checkpoint_dir, "checkpoint-latest.pt")
            
            # 기존 심볼릭 링크 제거
            if os.path.islink(latest_link):
                os.unlink(latest_link)
            elif os.path.exists(latest_link):
                os.remove(latest_link)
            
            # 새로운 심볼릭 링크 생성
            relative_path = os.path.basename(checkpoint_path)
            os.symlink(relative_path, latest_link)
            
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # 체크포인트 저장 이벤트 구조화된 로깅
            checkpoint_reason = "spot_interruption" if self.interrupted else "periodic"
            logger.info(f"[체크포인트 저장] 경로: {checkpoint_path}, 스텝: {step}, 이유: {checkpoint_reason}, 시간: {timestamp}")
            
        except Exception as e:
            logger.error(f"체크포인트 후처리 작업 실패: {e}")
