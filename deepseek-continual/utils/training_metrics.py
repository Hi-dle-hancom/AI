#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
학습 지표 추적 및 로깅

학습 중 발생하는 다양한 지표들을 추적하고 로깅합니다.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class TrainingMetrics:
    """학습 지표 추적 및 관리"""
    
    def __init__(self, config: Dict[str, Any]):
        """학습 지표 매니저 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.metrics = defaultdict(list)
        self.best_metrics = {}
        self.current_metrics = {}
        
        # 이동 평균을 위한 윈도우
        self.window_size = config.get('metrics_window_size', 100)
        self.moving_averages = defaultdict(lambda: deque(maxlen=self.window_size))
        
        # 시간 추적
        self.start_time = time.time()
        self.step_times = deque(maxlen=100)
        
        logger.info("학습 지표 매니저 초기화 완료")
    
    def update_metric(self, name: str, value: float, step: int):
        """지표 업데이트
        
        Args:
            name: 지표 이름
            value: 지표 값
            step: 현재 스텝
        """
        try:
            # 현재 지표 저장
            self.current_metrics[name] = value
            
            # 전체 히스토리에 추가
            self.metrics[name].append((step, value))
            
            # 이동 평균 업데이트
            self.moving_averages[name].append(value)
            
            # 최고 지표 업데이트 (손실은 최소, 정확도는 최대)
            if name.endswith('_loss') or name.endswith('_error'):
                if name not in self.best_metrics or value < self.best_metrics[name]:
                    self.best_metrics[name] = value
            else:
                if name not in self.best_metrics or value > self.best_metrics[name]:
                    self.best_metrics[name] = value
                    
        except Exception as e:
            logger.error(f"지표 업데이트 실패 ({name}): {e}")
    
    def get_moving_average(self, name: str) -> Optional[float]:
        """이동 평균 계산
        
        Args:
            name: 지표 이름
            
        Returns:
            Optional[float]: 이동 평균 값
        """
        try:
            if name in self.moving_averages and self.moving_averages[name]:
                return sum(self.moving_averages[name]) / len(self.moving_averages[name])
            return None
        except Exception as e:
            logger.error(f"이동 평균 계산 실패 ({name}): {e}")
            return None
    
    def update_step_time(self, step_duration: float):
        """스텝 시간 업데이트
        
        Args:
            step_duration: 스텝 소요 시간 (초)
        """
        self.step_times.append(step_duration)
    
    def get_average_step_time(self) -> float:
        """평균 스텝 시간 계산
        
        Returns:
            float: 평균 스텝 시간 (초)
        """
        if self.step_times:
            return sum(self.step_times) / len(self.step_times)
        return 0.0
    
    def estimate_remaining_time(self, current_step: int, total_steps: int) -> str:
        """남은 시간 추정
        
        Args:
            current_step: 현재 스텝
            total_steps: 전체 스텝
            
        Returns:
            str: 남은 시간 (형식화된 문자열)
        """
        try:
            if current_step >= total_steps or not self.step_times:
                return "완료"
            
            avg_step_time = self.get_average_step_time()
            remaining_steps = total_steps - current_step
            remaining_seconds = remaining_steps * avg_step_time
            
            # 시간 형식화
            hours = int(remaining_seconds // 3600)
            minutes = int((remaining_seconds % 3600) // 60)
            seconds = int(remaining_seconds % 60)
            
            if hours > 0:
                return f"{hours}시간 {minutes}분"
            elif minutes > 0:
                return f"{minutes}분 {seconds}초"
            else:
                return f"{seconds}초"
                
        except Exception as e:
            logger.error(f"남은 시간 추정 실패: {e}")
            return "알 수 없음"
    
    def get_training_progress(self, current_step: int, total_steps: int) -> Dict[str, Any]:
        """학습 진행 상황 요약
        
        Args:
            current_step: 현재 스텝
            total_steps: 전체 스텝
            
        Returns:
            Dict[str, Any]: 진행 상황 정보
        """
        try:
            progress = {
                'current_step': current_step,
                'total_steps': total_steps,
                'progress_percent': (current_step / total_steps) * 100 if total_steps > 0 else 0,
                'elapsed_time': time.time() - self.start_time,
                'average_step_time': self.get_average_step_time(),
                'estimated_remaining': self.estimate_remaining_time(current_step, total_steps),
                'current_metrics': self.current_metrics.copy(),
                'best_metrics': self.best_metrics.copy()
            }
            
            # 이동 평균 추가
            progress['moving_averages'] = {}
            for name in self.moving_averages:
                avg = self.get_moving_average(name)
                if avg is not None:
                    progress['moving_averages'][name] = avg
            
            return progress
            
        except Exception as e:
            logger.error(f"학습 진행 상황 요약 실패: {e}")
            return {}
    
    def log_progress(self, current_step: int, total_steps: int, log_interval: int = 10):
        """진행 상황 로깅
        
        Args:
            current_step: 현재 스텝
            total_steps: 전체 스텝
            log_interval: 로깅 간격
        """
        try:
            if current_step % log_interval != 0:
                return
            
            progress = self.get_training_progress(current_step, total_steps)
            
            # 기본 진행 정보
            logger.info(f"스텝 {current_step}/{total_steps} "
                       f"({progress['progress_percent']:.1f}%) - "
                       f"평균 {progress['average_step_time']:.2f}초/스텝 - "
                       f"남은 시간: {progress['estimated_remaining']}")
            
            # 현재 지표
            if progress['current_metrics']:
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in progress['current_metrics'].items()])
                logger.info(f"현재 지표: {metrics_str}")
            
            # 최고 지표 (100스텝마다)
            if current_step % (log_interval * 10) == 0 and progress['best_metrics']:
                best_str = ", ".join([f"{k}: {v:.4f}" for k, v in progress['best_metrics'].items()])
                logger.info(f"최고 지표: {best_str}")
                
        except Exception as e:
            logger.error(f"진행 상황 로깅 실패: {e}")
    
    def check_gradient_explosion(self, model) -> float:
        """그래디언트 폭주 감지
        
        Args:
            model: 모델 객체
            
        Returns:
            float: 그래디언트 노름
        """
        try:
            total_norm = 0.0
            param_count = 0
            
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            total_norm = total_norm ** (1. / 2)
            
            # 그래디언트 폭주 감지
            if total_norm > 10.0:
                logger.warning(f"⚠️  대형 그래디언트 노름 감지: {total_norm:.4f} (매개변수 {param_count}개)")
            elif total_norm > 5.0:
                logger.info(f"📊 그래디언트 노름: {total_norm:.4f}")
            
            # 지표 업데이트
            self.update_metric('gradient_norm', total_norm, 0)
            
            return total_norm
            
        except Exception as e:
            logger.error(f"그래디언트 폭주 감지 실패: {e}")
            return 0.0
    
    def save_metrics(self, filepath: str):
        """지표를 파일로 저장
        
        Args:
            filepath: 저장할 파일 경로
        """
        try:
            import json
            
            metrics_data = {
                'metrics': dict(self.metrics),
                'best_metrics': self.best_metrics,
                'current_metrics': self.current_metrics,
                'total_training_time': time.time() - self.start_time,
                'average_step_time': self.get_average_step_time()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"학습 지표 저장 완료: {filepath}")
            
        except Exception as e:
            logger.error(f"학습 지표 저장 실패: {e}")
    
    def load_metrics(self, filepath: str) -> bool:
        """파일에서 지표 로드
        
        Args:
            filepath: 로드할 파일 경로
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            import json
            
            with open(filepath, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            
            self.metrics = defaultdict(list, metrics_data.get('metrics', {}))
            self.best_metrics = metrics_data.get('best_metrics', {})
            self.current_metrics = metrics_data.get('current_metrics', {})
            
            logger.info(f"학습 지표 로드 완료: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"학습 지표 로드 실패: {e}")
            return False
    
    def reset_metrics(self):
        """모든 지표 초기화"""
        self.metrics.clear()
        self.best_metrics.clear()
        self.current_metrics.clear()
        self.moving_averages.clear()
        self.step_times.clear()
        self.start_time = time.time()
        
        logger.info("학습 지표 초기화 완료")
    
    def get_summary_report(self) -> str:
        """요약 보고서 생성
        
        Returns:
            str: 요약 보고서
        """
        try:
            total_time = time.time() - self.start_time
            avg_step_time = self.get_average_step_time()
            
            report = [
                "=" * 50,
                "학습 지표 요약 보고서",
                "=" * 50,
                f"총 학습 시간: {total_time/3600:.2f}시간",
                f"평균 스텝 시간: {avg_step_time:.2f}초",
                ""
            ]
            
            # 최고 지표
            if self.best_metrics:
                report.append("최고 지표:")
                for name, value in self.best_metrics.items():
                    report.append(f"  {name}: {value:.4f}")
                report.append("")
            
            # 현재 지표
            if self.current_metrics:
                report.append("현재 지표:")
                for name, value in self.current_metrics.items():
                    report.append(f"  {name}: {value:.4f}")
                report.append("")
            
            # 이동 평균
            report.append("이동 평균:")
            for name in self.moving_averages:
                avg = self.get_moving_average(name)
                if avg is not None:
                    report.append(f"  {name}: {avg:.4f}")
            
            report.append("=" * 50)
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"요약 보고서 생성 실패: {e}")
            return "요약 보고서 생성 실패"
