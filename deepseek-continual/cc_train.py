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
from transformers import (
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    AutoTokenizer, AutoModelForCausalLM
)
from datasets import Dataset
from torch.utils.data import DataLoader
import gc
import torch.cuda.amp as amp

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
    
    def __init__(self, config_or_path):
        """지속 학습기 초기화
        
        Args:
            config_or_path: 설정 딕셔너리 또는 설정 파일 경로
        """
        if isinstance(config_or_path, dict):
            self.config = config_or_path
        else:
            self.config = self.load_config(config_or_path)
        
        # 모듈 초기화 (환경 설정 전에 먼저 수행)
        self.memory_manager = MemoryManager(self.config)
        self.checkpoint_manager = CheckpointManager(self.config)
        self.spot_handler = SpotInterruptionHandler(
            checkpoint_dir=self.config.get('checkpoint_dir', 'checkpoints'),
            checkpoint_freq=self.config.get('checkpoint_freq', 500)
        )
        self.model_builder = ModelBuilder(self.config)
        self.training_metrics = TrainingMetrics(self.config)
        
        # 환경 설정 (모듈 초기화 후에 수행)
        self.setup_environment()
        
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
            self.memory_manager.setup_cuda_memory()
            
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
    
    def prepare_datasets(self, data_file: str = None):
        """데이터셋 준비
        
        Args:
            data_file: 특정 데이터 파일명 (선택사항)
        """
        try:
            logger.info("데이터셋 준비 시작...")
            
            # 데이터셋 프로세서 초기화
            dataset_processor = DatasetProcessor(self.config, self.tokenizer)
            
            # 데이터 경로 및 파일 찾기
            data_path = self.config.get('data_path')
            if not data_path:
                raise ValueError("데이터 경로가 설정되지 않았습니다")
            
            # 훈련 파일들 찾기
            training_files = dataset_processor.find_training_files(data_path, data_file)
            if not training_files:
                raise ValueError(f"훈련 데이터 파일을 찾을 수 없습니다: {data_path}")
            
            # 첫 번째 파일로 데이터셋 처리 (다중 파일 지원은 추후 구현)
            current_file = training_files[0]
            logger.info(f"현재 처리 중인 데이터 파일: {os.path.basename(current_file)}")
            
            # 모드별 데이터셋 처리
            mode = self.config.get('mode', 'complete')
            self.train_dataset, self.eval_dataset = dataset_processor.process_dataset(current_file, mode)
            
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
            num_epochs = int(self.config.get('num_epochs', 3))
            batch_size = int(self.config.get('batch_size', 2))
            gradient_accumulation_steps = int(self.config.get('gradient_accumulation_steps', 8))
            
            logger.info(f"학습 설정: 에포크={num_epochs}, 배치={batch_size}, 누적={gradient_accumulation_steps}")
            
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
    
    def train(self, data_file: str = None):
        """메인 훈련 함수 - 레거시 호환 다중 파일 순차 처리
        
        Args:
            data_file: 특정 데이터 파일명 (선택사항)
        """
        try:
            logger.info("=== 지속 학습 시작 ===")
            
            # 1. 모델과 토크나이저 빌드
            self.build_model_and_tokenizer()
            
            # 2. 옵티마이저와 스케줄러 빌드 (데이터셋 준비 전에 수행)
            self.build_optimizer_and_scheduler()
            
            # 3. 기존 체크포인트 로드
            checkpoint_loaded = self.load_checkpoint_if_exists()
            
            # 4. 스팟 인스턴스 모니터링 시작
            self.spot_handler.start_monitoring(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                trainer=self
            )
            
            # 5. 다중 파일 순차 처리 (레거시 호환)
            self.train_multiple_files(data_file)
            
            logger.info("모든 훈련 완료!")
            
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
    
    def train_multiple_files(self, data_file: str = None):
        """다중 파일 순차 처리 (레거시 호환)
        
        Args:
            data_file: 특정 데이터 파일명 (선택사항)
        """
        try:
            # 데이터셋 프로세서 초기화
            dataset_processor = DatasetProcessor(self.config, self.tokenizer)
            
            # 데이터 경로 및 파일 찾기
            data_path = self.config.get('data_path')
            if not data_path:
                raise ValueError("데이터 경로가 설정되지 않았습니다")
            
            # 훈련 파일들 찾기
            training_files = dataset_processor.find_training_files(data_path, data_file)
            if not training_files:
                raise ValueError(f"훈련 데이터 파일을 찾을 수 없습니다: {data_path}")
            
            logger.info(f"총 {len(training_files)}개 파일을 순차적으로 처리합니다")
            
            # 각 파일에 대해 순차적으로 훈련
            for file_idx, current_file in enumerate(training_files):
                logger.info(f"===== 파일 {file_idx+1}/{len(training_files)}: {os.path.basename(current_file)} =====")
                
                try:
                    # 현재 파일로 데이터셋 처리
                    mode = self.config.get('mode', 'complete')
                    train_dataset, eval_dataset = dataset_processor.process_dataset(current_file, mode)
                    
                    if train_dataset is None:
                        logger.warning(f"파일 {current_file} 처리 실패, 건너뜀")
                        continue
                    
                    # 데이터셋 통계
                    train_stats = dataset_processor.get_dataset_statistics(train_dataset)
                    logger.info(f"훈련 데이터셋 통계: {train_stats}")
                    
                    # DataLoader 생성
                    from torch.utils.data import DataLoader
                    from transformers import DataCollatorForLanguageModeling
                    
                    data_collator = DataCollatorForLanguageModeling(
                        tokenizer=self.tokenizer, 
                        mlm=False
                    )
                    
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=self.config.get('batch_size', 16),
                        shuffle=True,
                        collate_fn=data_collator
                    )
                    
                    val_loader = None
                    if eval_dataset:
                        val_loader = DataLoader(
                            eval_dataset,
                            batch_size=self.config.get('batch_size', 16),
                            collate_fn=data_collator
                        )
                    
                    # 현재 파일에 대해 훈련 수행 (레거시 train_task 방식)
                    self.train_task(train_loader, val_loader, current_mode=mode)
                    
                    # 파일별 체크포인트 저장
                    if file_idx < len(training_files) - 1:  # 마지막 파일이 아닌 경우
                        checkpoint_path = self.checkpoint_manager.save_checkpoint(
                            self.model, self.tokenizer, self.optimizer,
                            self.scheduler, self.scaler,
                            step=getattr(self, 'global_step', file_idx),
                            is_intermediate=True
                        )
                        logger.info(f"중간 체크포인트 저장: {checkpoint_path}")
                    
                except Exception as e:
                    logger.error(f"파일 {current_file} 처리 중 오류: {e}")
                    continue
            
            # 최종 모델 저장
            final_model_path = self.checkpoint_manager.save_checkpoint(
                self.model, self.tokenizer, self.optimizer,
                self.scheduler, self.scaler,
                step=getattr(self, 'global_step', len(training_files)),
                is_final=True
            )
            logger.info(f"최종 모델 저장: {final_model_path}")
            
        except Exception as e:
            logger.error(f"다중 파일 처리 중 오류: {e}")
            raise e
    
    def train_task(self, train_loader, val_loader, test_loaders=None, current_mode: str = None):
        """단일 모드에 대한 훈련 수행 (레거시 호환)
        
        Args:
            train_loader: 훈련 데이터 로더
            val_loader: 검증 데이터 로더
            test_loaders: 이전 모드의 테스트 데이터 로더 {mode: loader}
            current_mode: 현재 학습 모드 (complete, prompt, comment, error_fix)
        """
        try:
            # 현재 모드 설정
            self.current_mode = current_mode
            
            # 훈련 설정
            max_epochs = self.config.get('max_epochs', 3)
            patience = self.config.get('patience', 2)
            
            # 조기 종료 추적
            best_val_acc = 0.0
            patience_counter = 0
            
            # 전역 스텝 초기화
            if not hasattr(self, 'global_step'):
                self.global_step = 0
            
            logger.info(f"=== 모드 {current_mode} 학습 시작 ===")
            logger.info(f"최대 에포크: {max_epochs}, 인내: {patience}")
            
            for epoch in range(max_epochs):
                logger.info(f"\n--- 에포크 {epoch+1}/{max_epochs} ---")
                
                # 훈련 에포크
                train_loss, train_acc = self.train_epoch(
                    train_loader=train_loader,
                    epoch=epoch,
                    current_mode=current_mode
                )
                
                # 검증
                val_loss, val_acc = 0.0, 0.0
                if val_loader:
                    val_loss, val_acc = self.evaluate(val_loader)
                
                # 로깅
                logger.info(f"에포크 {epoch+1} 결과:")
                logger.info(f"  훈련 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                if val_loader:
                    logger.info(f"  검증 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                
                # 조기 종료 확인
                if val_loader and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    logger.info(f"  새로운 최고 검증 정확도: {best_val_acc:.4f}")
                else:
                    patience_counter += 1
                    logger.info(f"  인내 카운터: {patience_counter}/{patience}")
                
                # 인내 초과 시 조기 종료
                if patience_counter >= patience:
                    logger.info(f"  인내 초과로 조기 종료")
                    break
                
                # 에포크별 체크포인트 저장
                if (epoch + 1) % self.config.get('checkpoint_freq', 1) == 0:
                    checkpoint_path = self.checkpoint_manager.save_checkpoint(
                        self.model, self.tokenizer, self.optimizer,
                        self.scheduler, self.scaler,
                        step=self.global_step,
                        is_epoch_end=True
                    )
                    logger.info(f"  에포크 체크포인트 저장: {checkpoint_path}")
            
            logger.info(f"=== 모드 {current_mode} 학습 완료 ===")
            logger.info(f"최고 검증 정확도: {best_val_acc:.4f}")
            
        except Exception as e:
            logger.error(f"훈련 태스크 중 오류: {e}")
            raise e
    
    def train_epoch(self, train_loader, epoch: int, current_mode: str = None):
        """단일 에포크 훈련
        
        Args:
            train_loader: 훈련 데이터 로더
            epoch: 현재 에포크 번호
            current_mode: 현재 학습 모드
            
        Returns:
            Tuple[float, float]: (train_loss, train_acc)
        """
        try:
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            
            # 진행 상황 추적
            num_batches = len(train_loader)
            log_interval = max(1, num_batches // 10)  # 10% 마다 로깅
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    # 배치 처리
                    loss, correct, batch_size = self.train_step(batch, current_mode)
                    
                    total_loss += loss
                    total_correct += correct
                    total_samples += batch_size
                    
                    # 전역 스텝 증가
                    self.global_step += 1
                    
                    # 주기적 로깅
                    if (batch_idx + 1) % log_interval == 0:
                        current_loss = total_loss / (batch_idx + 1)
                        current_acc = total_correct / total_samples if total_samples > 0 else 0.0
                        logger.info(f"    배치 {batch_idx+1}/{num_batches} - Loss: {current_loss:.4f}, Acc: {current_acc:.4f}")
                    
                    # 스팟 인스턴스 중단 확인
                    if self.spot_handler.check_interruption():
                        logger.warning("스팟 인스턴스 중단 감지, 긴급 저장 수행")
                        self.emergency_save()
                        break
                        
                except Exception as e:
                    logger.error(f"배치 {batch_idx} 처리 중 오류: {e}")
                    # OOM 오류 처리
                    if "out of memory" in str(e).lower():
                        self.memory_manager.handle_oom_error()
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
            
            # 에포크 평균 계산
            avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
            avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
            
            return avg_loss, avg_acc
            
        except Exception as e:
            logger.error(f"에포크 훈련 중 오류: {e}")
            raise e
    
    def train_step(self, batch, current_mode: str = None):
        """단일 배치 훈련 스텝
        
        Args:
            batch: 훈련 배치
            current_mode: 현재 학습 모드
            
        Returns:
            Tuple[float, int, int]: (loss, correct_count, batch_size)
        """
        try:
            # 데이터 GPU로 이동
            input_ids = batch['input_ids'].to(self.model.device)
            attention_mask = batch['attention_mask'].to(self.model.device)
            labels = batch['labels'].to(self.model.device)
            
            batch_size = input_ids.size(0)
            
            # 그래디언트 초기화
            self.optimizer.zero_grad()
            
            # 포워드 패스
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            
            # 백워드 패스
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 스케줄러 업데이트
            if self.scheduler:
                self.scheduler.step()
            
            # 정확도 계산 (간단한 방식)
            with torch.no_grad():
                predictions = torch.argmax(outputs.logits, dim=-1)
                # 레이블이 -100이 아닌 위치만 비교
                valid_mask = (labels != -100)
                correct = ((predictions == labels) & valid_mask).sum().item()
                valid_tokens = valid_mask.sum().item()
            
            return loss.item(), correct, valid_tokens
            
        except Exception as e:
            logger.error(f"훈련 스텝 중 오류: {e}")
            raise e
    
    def evaluate(self, eval_loader):
        """모델 평가
        
        Args:
            eval_loader: 평가 데이터 로더
            
        Returns:
            Tuple[float, float]: (eval_loss, eval_acc)
        """
        try:
            self.model.eval()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            
            with torch.no_grad():
                for batch in eval_loader:
                    # 데이터 GPU로 이동
                    input_ids = batch['input_ids'].to(self.model.device)
                    attention_mask = batch['attention_mask'].to(self.model.device)
                    labels = batch['labels'].to(self.model.device)
                    
                    # 포워드 패스
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                    
                    # 정확도 계산
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    valid_mask = (labels != -100)
                    correct = ((predictions == labels) & valid_mask).sum().item()
                    valid_tokens = valid_mask.sum().item()
                    
                    total_loss += loss.item()
                    total_correct += correct
                    total_samples += valid_tokens
            
            # 평균 계산
            avg_loss = total_loss / len(eval_loader) if len(eval_loader) > 0 else 0.0
            avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
            
            return avg_loss, avg_acc
            
        except Exception as e:
            logger.error(f"평가 중 오류: {e}")
            return 0.0, 0.0
    
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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["complete", "prompt", "comment", "error_fix"],
        help="훈련 모드 (complete: 자동완성, prompt: 프롬프트, comment: 주석, error_fix: 오류수정)"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        help="사용할 데이터 파일명 (예: train1.jsonl, train2.jsonl)"
    )
    
    args = parser.parse_args()
    
    try:
        # 지속 학습기 초기화
        learner = ContinualLearner(args.config)
        
        # 명령행 인수로 모드 오버라이드
        if args.mode:
            learner.config['mode'] = args.mode
            logger.info(f"명령행에서 모드 설정: {args.mode}")
        
        # 훈련 시작
        learner.train(getattr(args, 'data_file', None))
        
    except Exception as e:
        logger.error(f"메인 함수 실행 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
