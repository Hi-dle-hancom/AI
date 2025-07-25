#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
체크포인트 관리자

모델, 옵티마이저, 스케줄러 및 학습 상태의 저장/로드/정리를 담당합니다.
"""

import os
import time
import datetime
import torch
import logging
import psutil
import gc
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class CheckpointManager:
    """체크포인트 저장, 로드, 정리를 담당하는 매니저"""
    
    def __init__(self, config: Dict[str, Any]):
        """체크포인트 매니저 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.max_checkpoints = 5  # 최대 보관할 체크포인트 수
        
    def get_checkpoint_path(self, filename: str) -> str:
        """체크포인트 저장 경로 생성
        
        Args:
            filename: 저장할 파일 이름
            
        Returns:
            str: 전체 저장 경로
        """
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        
        # 학습 모드별 하위 디렉토리 설정
        training_mode = self.config.get('mode', 'complete')
        if training_mode == "complete":
            mode_subdir = "autocomplete-finetuned"
        else:
            mode_subdir = f"{training_mode}-finetuned"
            
        # 경로 중복 방지
        if checkpoint_dir.endswith(mode_subdir) or mode_subdir in checkpoint_dir.split('/'):
            logger.info(f"체크포인트 경로에 이미 '{mode_subdir}'가 포함되어 있어 중복을 방지합니다.")
            checkpoint_subdir = checkpoint_dir
        else:
            checkpoint_subdir = os.path.join(checkpoint_dir, mode_subdir)
            
        return os.path.join(checkpoint_subdir, filename)
    
    def prepare_save_directory(self, save_path: str) -> bool:
        """저장 디렉토리 준비 및 권한 확인
        
        Args:
            save_path: 저장할 파일의 전체 경로
            
        Returns:
            bool: 디렉토리 준비 성공 여부
        """
        try:
            dir_path = os.path.dirname(save_path)
            
            # 디렉토리 생성
            os.makedirs(dir_path, exist_ok=True)
            
            # 쓰기 권한 확인
            test_file = os.path.join(dir_path, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            
            logger.info(f"체크포인트 디렉토리 준비 완료: {dir_path}")
            return True
            
        except Exception as e:
            logger.error(f"체크포인트 디렉토리 준비 실패: {e}")
            return False
    
    def cleanup_memory(self):
        """메모리 정리 및 최적화"""
        # 가비지 컬렉션
        gc.collect(2)
        
        # CUDA 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # 메모리 상태 로깅
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            logger.debug(f"시스템 메모리 사용량: {memory_info.rss / 1024 / 1024:.1f}MB")
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                    reserved = torch.cuda.memory_reserved(i) / 1024 / 1024
                    logger.debug(f"GPU {i} 메모리: {allocated:.1f}MB / {reserved:.1f}MB")
        except Exception as e:
            logger.debug(f"메모리 상태 로깅 실패: {e}")
    
    def create_state_dict(self, trainer, is_spot_interruption: bool = False) -> Dict[str, Any]:
        """저장할 상태 딕셔너리 생성
        
        Args:
            trainer: 학습기 객체
            is_spot_interruption: 스팟 인스턴스 중단 여부
            
        Returns:
            Dict[str, Any]: 상태 딕셔너리
        """
        # 타임스탬프 생성
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 핵심 학습 상태 메타데이터
        metadata = {
            'global_step': trainer.global_step,
            'current_mode': trainer.current_mode,
            'completed_modes': list(set([trainer.current_mode] + 
                                      (trainer.completed_modes if hasattr(trainer, 'completed_modes') else [])))
        }
        
        # 상태 딕셔너리 구성
        state_dict = {
            # 핵심 메타데이터
            'global_step': trainer.global_step,
            'current_mode': trainer.current_mode,
            'best_accuracies': getattr(trainer, 'best_accuracies', {}),
            
            # 학습 모드간 필수 메타데이터
            'mode_metadata': metadata,
            
            # 기본 메타데이터
            'saved_at': timestamp,
            'is_spot_interruption': is_spot_interruption
        }
        
        return state_dict
    
    def save_model_components(self, save_path: str, trainer, state_dict: Dict[str, Any]):
        """모델 컴포넌트들을 개별 파일로 저장
        
        Args:
            save_path: 기본 저장 경로
            trainer: 학습기 객체
            state_dict: 상태 딕셔너리
        """
        checkpoint_dir_path = os.path.dirname(save_path)
        
        # 1. 모델 저장 (Hugging Face 형식)
        try:
            model_dir = os.path.join(checkpoint_dir_path, "model")
            os.makedirs(model_dir, exist_ok=True)
            trainer.model.save_pretrained(model_dir)
            trainer.tokenizer.save_pretrained(model_dir)
            logger.info(f"모델 저장 완료: {model_dir}")
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
        
        # 2. 옵티마이저 상태 저장
        try:
            if hasattr(trainer, 'optimizer') and trainer.optimizer:
                optimizer_path = os.path.join(checkpoint_dir_path, "optimizer.pt")
                torch.save(trainer.optimizer.state_dict(), optimizer_path)
                logger.info(f"옵티마이저 저장 완료: {optimizer_path}")
        except Exception as e:
            logger.error(f"옵티마이저 저장 실패: {e}")
        
        # 3. 스케줄러 상태 저장
        try:
            if hasattr(trainer, 'scheduler') and trainer.scheduler:
                scheduler_path = os.path.join(checkpoint_dir_path, "scheduler.pt")
                torch.save(trainer.scheduler.state_dict(), scheduler_path)
                logger.info(f"스케줄러 저장 완료: {scheduler_path}")
        except Exception as e:
            logger.error(f"스케줄러 저장 실패: {e}")
        
        # 4. AMP 스케일러 저장
        try:
            if hasattr(trainer, 'scaler') and trainer.scaler:
                scaler_path = os.path.join(checkpoint_dir_path, "scaler.pt")
                torch.save(trainer.scaler.state_dict(), scaler_path)
                logger.info(f"AMP 스케일러 저장 완료: {scaler_path}")
        except Exception as e:
            logger.error(f"AMP 스케일러 저장 실패: {e}")
        
        # 5. 학습 설정 저장
        try:
            training_args_path = os.path.join(checkpoint_dir_path, "training_args.bin")
            torch.save(trainer.config, training_args_path)
            logger.info(f"학습 설정 저장 완료: {training_args_path}")
        except Exception as e:
            logger.error(f"학습 설정 저장 실패: {e}")
    
    def create_readme(self, checkpoint_dir_path: str, trainer, is_spot_interruption: bool):
        """README.md 파일 생성
        
        Args:
            checkpoint_dir_path: 체크포인트 디렉토리 경로
            trainer: 학습기 객체
            is_spot_interruption: 스팟 인스턴스 중단 여부
        """
        try:
            readme_path = os.path.join(checkpoint_dir_path, "README.md")
            checkpoint_dir_name = os.path.basename(checkpoint_dir_path)
            
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(f"# 체크포인트 {checkpoint_dir_name}\n\n")
                f.write(f"- 글로벌 스텝: {trainer.global_step}\n")
                f.write(f"- 학습 모드: {trainer.current_mode}\n")
                
                # 스팟 인스턴스 중단 여부
                spot_status = '예' if is_spot_interruption else '아니오'
                f.write(f"- 스팟 인스턴스 중단: {spot_status}\n\n")
                
                # 학습 지표 추가
                f.write(f"## 학습 지표\n\n")
                f.write(f"| 지표 | 값 |\n")
                f.write(f"|------|------|\n")
                
                # 현재 손실값 (있는 경우)
                if hasattr(trainer, 'current_loss'):
                    f.write(f"| 현재 손실 | {trainer.current_loss:.4f} |\n")
                
                # 최고 정확도 (있는 경우)
                if hasattr(trainer, 'best_accuracies') and trainer.best_accuracies:
                    for mode, acc in trainer.best_accuracies.items():
                        f.write(f"| {mode} 최고 정확도 | {acc:.4f} |\n")
                
                f.write(f"\n저장 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
        except Exception as e:
            logger.error(f"README.md 생성 실패: {e}")
    
    def cleanup_old_checkpoints(self, checkpoint_dir: str, is_spot_interruption: bool):
        """오래된 체크포인트 정리
        
        Args:
            checkpoint_dir: 체크포인트 디렉토리
            is_spot_interruption: 스팟 인스턴스 중단 여부
        """
        try:
            # 긴급 체크포인트는 제외하고 정리
            mode_prefix = "interrupt" if is_spot_interruption else "checkpoint"
            
            # 현재 모드의 체크포인트 파일들 찾기
            checkpoint_files = []
            for file in os.listdir(checkpoint_dir):
                if file.startswith(mode_prefix) and file.endswith(".pt") and "emergency" not in file:
                    file_path = os.path.join(checkpoint_dir, file)
                    checkpoint_files.append((file_path, os.path.getmtime(file_path)))
            
            # 최신 N개만 유지
            if len(checkpoint_files) > self.max_checkpoints:
                # 시간 순으로 정렬 (최신 것이 마지막)
                checkpoint_files.sort(key=lambda x: x[1])
                
                # 오래된 파일들 삭제
                files_to_remove = checkpoint_files[:-self.max_checkpoints]
                for file_path, _ in files_to_remove:
                    try:
                        os.remove(file_path)
                        logger.info(f"오래된 체크포인트 삭제: {os.path.basename(file_path)}")
                    except Exception as e:
                        logger.error(f"체크포인트 삭제 실패: {os.path.basename(file_path)}, {e}")
                        
        except Exception as e:
            logger.error(f"체크포인트 정리 실패: {e}")
    
    def save_checkpoint(self, trainer, filename: str, is_spot_interruption: bool = False) -> str:
        """체크포인트 저장 메인 메서드
        
        Args:
            trainer: 학습기 객체
            filename: 저장할 파일 이름
            is_spot_interruption: 스팟 인스턴스 중단 여부
            
        Returns:
            str: 저장된 파일의 전체 경로
        """
        # 저장 경로 설정
        save_path = self.get_checkpoint_path(filename)
        
        # 디렉토리 준비
        if not self.prepare_save_directory(save_path):
            raise RuntimeError("체크포인트 디렉토리 준비 실패")
        
        # 메모리 정리
        self.cleanup_memory()
        
        try:
            # 임시 파일로 저장 (atomic 저장)
            temp_save_path = f"{save_path}.tmp"
            
            # 상태 딕셔너리 생성
            state_dict = self.create_state_dict(trainer, is_spot_interruption)
            
            # 메인 체크포인트 파일 저장
            torch.save(state_dict, temp_save_path)
            
            # 원자적 이동
            os.rename(temp_save_path, save_path)
            
            # 모델 컴포넌트들 개별 저장
            self.save_model_components(save_path, trainer, state_dict)
            
            # README 생성
            checkpoint_dir_path = os.path.dirname(save_path)
            self.create_readme(checkpoint_dir_path, trainer, is_spot_interruption)
            
            # 오래된 체크포인트 정리
            self.cleanup_old_checkpoints(os.path.dirname(save_path), is_spot_interruption)
            
            # 파일 크기 확인
            file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
            logger.info(f"체크포인트 저장 완료: {save_path} (크기: {file_size_mb:.1f}MB)")
            
            return save_path
            
        except Exception as e:
            # 임시 파일 정리
            temp_save_path = f"{save_path}.tmp"
            if os.path.exists(temp_save_path):
                try:
                    os.remove(temp_save_path)
                except:
                    pass
            
            logger.error(f"체크포인트 저장 실패: {e}")
            raise e
    
    def load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """체크포인트 로드
        
        Args:
            checkpoint_path: 체크포인트 파일 경로
            
        Returns:
            Optional[Dict[str, Any]]: 로드된 상태 딕셔너리
        """
        try:
            if not os.path.exists(checkpoint_path):
                logger.error(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
                return None
            
            # 체크포인트 로드
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # 파일 크기 확인
            file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
            logger.info(f"체크포인트 로드 완료: {checkpoint_path} (크기: {file_size_mb:.1f}MB)")
            
            return state_dict
            
        except Exception as e:
            logger.error(f"체크포인트 로드 실패: {e}")
            return None
    
    def find_latest_checkpoint(self, checkpoint_dir: str) -> Optional[str]:
        """최신 체크포인트 찾기
        
        Args:
            checkpoint_dir: 체크포인트 디렉토리
            
        Returns:
            Optional[str]: 최신 체크포인트 파일 경로
        """
        try:
            if not os.path.exists(checkpoint_dir):
                return None
            
            checkpoint_files = []
            for file in os.listdir(checkpoint_dir):
                if file.endswith(".pt") and ("checkpoint" in file or "interrupt" in file):
                    file_path = os.path.join(checkpoint_dir, file)
                    checkpoint_files.append((file_path, os.path.getmtime(file_path)))
            
            if not checkpoint_files:
                return None
            
            # 최신 파일 반환
            checkpoint_files.sort(key=lambda x: x[1], reverse=True)
            return checkpoint_files[0][0]
            
        except Exception as e:
            logger.error(f"최신 체크포인트 찾기 실패: {e}")
            return None
