#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepSeek-Coder 지속 학습 파이프라인 (모듈화 버전)

AWS A10G 22GB GPU 환경에서 스팟 인스턴스를 활용한 안정적인 지속 학습을 위한
모듈화된 파이프라인입니다.
"""

import os
import sys
import yaml
import torch
import logging
import argparse
import warnings
from typing import Dict, Any, Optional
from datetime import datetime

# 모듈 임포트
from handlers.spot_handler import SpotInterruptionHandler
from managers.checkpoint_manager import CheckpointManager
from managers.memory_manager import MemoryManager
from processors.dataset_processor import DatasetProcessor
from model_builders.model_builder import ModelBuilder
from utils.training_metrics import TrainingMetrics

# 외부 라이브러리
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import gc

# 경고 억제
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 로깅 설정
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class ContinualLearner:
    """지속 학습 메인 클래스 (모듈화 버전)"""
    
    def __init__(self, config_path: str):
        """지속 학습기 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self.load_config(config_path)
        self.setup_environment()
        
        # 모듈 초기화
        self.memory_manager = MemoryManager(self.config)
        self.checkpoint_manager = CheckpointManager(self.config)
        self.spot_handler = SpotInterruptionHandler(
            checkpoint_dir=self.config.get('checkpoint_dir', 'checkpoints'),
            checkpoint_freq=self.config.get('checkpoint_freq', 500)
        )
        self.model_builder = ModelBuilder(self.config)
        self.training_metrics = TrainingMetrics(self.config)
        
        # 컴포넌트 초기화
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None
        
        logger.info("지속 학습기 초기화 완료 (모듈화 버전)")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"설정 파일 로드 완료: {config_path}")
            return config
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            raise e
    
    def setup_environment(self):
        """환경 설정"""
        try:
            # CUDA 메모리 환경 설정
            self.memory_manager.setup_cuda_environment()
            
            # 출력 디렉토리 생성
            output_dir = self.config.get('output_dir', 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            # 체크포인트 디렉토리 생성
            checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            logger.info("환경 설정 완료")
            
        except Exception as e:
            logger.error(f"환경 설정 실패: {e}")
            raise e
    
    def build_model_and_tokenizer(self):
        """모델과 토크나이저 빌드"""
        try:
            logger.info("모델과 토크나이저 빌드 시작...")
            
            # 모델과 토크나이저 빌드
            self.model, self.tokenizer = self.model_builder.build_model_and_tokenizer()
            
            # 모델 정보 출력
            model_info = self.model_builder.get_model_info(self.model)
            logger.info(f"모델 정보: {model_info}")
            
            # 메모리 정보 출력
            memory_info = self.memory_manager.get_memory_info()
            logger.info(f"메모리 정보: {memory_info}")
            
        except Exception as e:
            logger.error(f"모델 빌드 실패: {e}")
            raise e
    
    def prepare_datasets(self):
        """데이터셋 준비"""
        try:
            logger.info("데이터셋 준비 시작...")
            
            # 데이터셋 프로세서 초기화
            dataset_processor = DatasetProcessor(self.config, self.tokenizer)
            
            # 데이터 경로
            data_path = self.config.get('data_path')
            if not data_path or not os.path.exists(data_path):
                raise ValueError(f"데이터 파일을 찾을 수 없습니다: {data_path}")
            
            # 모드별 데이터셋 처리
            mode = self.config.get('mode', 'complete')
            self.train_dataset, self.eval_dataset = dataset_processor.process_dataset(data_path, mode)
            
            if self.train_dataset is None:
                raise ValueError("훈련 데이터셋 처리 실패")
            
            # 데이터셋 유효성 검사
            if not dataset_processor.validate_dataset(self.train_dataset):
                raise ValueError("훈련 데이터셋 유효성 검사 실패")
            
            # 데이터셋 통계
            train_stats = dataset_processor.get_dataset_statistics(self.train_dataset)
            logger.info(f"훈련 데이터셋 통계: {train_stats}")
            
            if self.eval_dataset:
                eval_stats = dataset_processor.get_dataset_statistics(self.eval_dataset)
                logger.info(f"검증 데이터셋 통계: {eval_stats}")
            
        except Exception as e:
            logger.error(f"데이터셋 준비 실패: {e}")
            raise e
    
    def build_optimizer_and_scheduler(self):
        """옵티마이저와 스케줄러 빌드"""
        try:
            logger.info("옵티마이저와 스케줄러 빌드 시작...")
            
            # 옵티마이저 빌드
            self.optimizer = self.model_builder.build_optimizer(self.model)
            
            # 총 학습 스텝 계산
            num_epochs = self.config.get('num_epochs', 3)
            batch_size = self.config.get('batch_size', 2)
            gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 8)
            
            steps_per_epoch = len(self.train_dataset) // (batch_size * gradient_accumulation_steps)
            total_steps = steps_per_epoch * num_epochs
            
            # 스케줄러 빌드
            self.scheduler = self.model_builder.build_scheduler(self.optimizer, total_steps)
            
            # AMP 스케일러 빌드
            self.scaler = self.model_builder.build_scaler()
            
            logger.info(f"옵티마이저/스케줄러 빌드 완료 (총 {total_steps} 스텝)")
            
        except Exception as e:
            logger.error(f"옵티마이저/스케줄러 빌드 실패: {e}")
            raise e
    
    def setup_trainer(self):
        """Trainer 설정"""
        try:
            logger.info("Trainer 설정 시작...")
            
            # 훈련 인수 설정
            training_args = TrainingArguments(
                output_dir=self.config.get('output_dir', 'output'),
                num_train_epochs=self.config.get('num_epochs', 3),
                per_device_train_batch_size=self.config.get('batch_size', 2),
                per_device_eval_batch_size=self.config.get('eval_batch_size', 2),
                gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 8),
                learning_rate=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 0.01),
                warmup_ratio=self.config.get('warmup_ratio', 0.05),
                lr_scheduler_type=self.config.get('lr_scheduler', 'cosine'),
                save_steps=self.config.get('save_steps', 500),
                save_total_limit=self.config.get('save_total_limit', 5),
                evaluation_strategy="steps" if self.eval_dataset else "no",
                eval_steps=self.config.get('eval_steps', 500) if self.eval_dataset else None,
                logging_steps=self.config.get('logging_steps', 10),
                fp16=self.config.get('use_amp', True),
                dataloader_num_workers=self.config.get('dataloader_num_workers', 4),
                remove_unused_columns=False,
                report_to=None,  # 외부 로깅 비활성화
                load_best_model_at_end=True if self.eval_dataset else False,
                metric_for_best_model="eval_loss" if self.eval_dataset else None,
                greater_is_better=False if self.eval_dataset else None,
            )
            
            # 데이터 콜레이터
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
            
            # Trainer 초기화
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                data_collator=data_collator,
                optimizers=(self.optimizer, self.scheduler)
            )
            
            logger.info("Trainer 설정 완료")
            
        except Exception as e:
            logger.error(f"Trainer 설정 실패: {e}")
            raise e
    
    def load_checkpoint_if_exists(self):
        """기존 체크포인트 로드"""
        try:
            checkpoint_path = self.checkpoint_manager.find_latest_checkpoint()
            if checkpoint_path:
                logger.info(f"체크포인트 발견: {checkpoint_path}")
                
                # 체크포인트 로드
                success = self.checkpoint_manager.load_checkpoint(
                    checkpoint_path, self.model, self.optimizer, 
                    self.scheduler, self.scaler
                )
                
                if success:
                    logger.info("체크포인트 로드 완료")
                    return True
                else:
                    logger.warning("체크포인트 로드 실패")
            
            return False
            
        except Exception as e:
            logger.error(f"체크포인트 로드 중 오류: {e}")
            return False
    
    def train(self):
        """메인 훈련 함수"""
        try:
            logger.info("=== 지속 학습 시작 ===")
            
            # 1. 모델과 토크나이저 빌드
            self.build_model_and_tokenizer()
            
            # 2. 데이터셋 준비
            self.prepare_datasets()
            
            # 3. 옵티마이저와 스케줄러 빌드
            self.build_optimizer_and_scheduler()
            
            # 4. Trainer 설정
            self.setup_trainer()
            
            # 5. 기존 체크포인트 로드
            checkpoint_loaded = self.load_checkpoint_if_exists()
            
            # 6. 스팟 인스턴스 모니터링 시작
            self.spot_handler.start_monitoring(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                trainer=self.trainer
            )
            
            # 7. 훈련 시작
            logger.info("훈련 시작...")
            
            if checkpoint_loaded:
                # 체크포인트에서 재개
                self.trainer.train(resume_from_checkpoint=True)
            else:
                # 처음부터 시작
                self.trainer.train()
            
            # 8. 최종 모델 저장
            final_model_path = self.checkpoint_manager.save_checkpoint(
                self.model, self.tokenizer, self.optimizer, 
                self.scheduler, self.scaler, 
                step=self.trainer.state.global_step,
                is_final=True
            )
            
            logger.info(f"훈련 완료! 최종 모델 저장: {final_model_path}")
            
        except KeyboardInterrupt:
            logger.info("사용자에 의한 훈련 중단")
            self.emergency_save()
        except Exception as e:
            logger.error(f"훈련 중 오류 발생: {e}")
            self.emergency_save()
            raise e
        finally:
            # 정리 작업
            self.cleanup()
    
    def emergency_save(self):
        """긴급 저장"""
        try:
            logger.info("긴급 저장 시작...")
            
            if self.trainer and self.model:
                emergency_path = self.checkpoint_manager.save_checkpoint(
                    self.model, self.tokenizer, self.optimizer,
                    self.scheduler, self.scaler,
                    step=getattr(self.trainer.state, 'global_step', 0),
                    is_emergency=True
                )
                logger.info(f"긴급 저장 완료: {emergency_path}")
            
        except Exception as e:
            logger.error(f"긴급 저장 실패: {e}")
    
    def cleanup(self):
        """정리 작업"""
        try:
            logger.info("정리 작업 시작...")
            
            # 스팟 인스턴스 모니터링 중단
            if self.spot_handler:
                self.spot_handler.stop_monitoring()
            
            # 메모리 정리
            if self.memory_manager:
                self.memory_manager.cleanup_memory()
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 가비지 컬렉션
            gc.collect()
            
            logger.info("정리 작업 완료")
            
        except Exception as e:
            logger.error(f"정리 작업 중 오류: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="DeepSeek-Coder 지속 학습 (모듈화 버전)")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="설정 파일 경로"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="체크포인트에서 재개"
    )
    
    args = parser.parse_args()
    
    try:
        # 지속 학습기 초기화
        learner = ContinualLearner(args.config)
        
        # 훈련 시작
        learner.train()
        
    except Exception as e:
        logger.error(f"메인 함수 실행 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
