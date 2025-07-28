#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
모델 빌더

모델 로드, LoRA 설정, 옵티마이저/스케줄러 빌드를 담당합니다.
"""

import os
import torch
import logging
from typing import Dict, Any, Optional, Tuple
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    get_scheduler
)
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = None
    get_peft_model = None
    TaskType = None
from torch.optim import AdamW
from torch.cuda.amp import GradScaler

logger = logging.getLogger(__name__)


class ModelBuilder:
    """모델, 토크나이저, 옵티마이저 빌드를 담당하는 빌더"""
    
    def __init__(self, config: Dict[str, Any]):
        """모델 빌더 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.model_name = config.get('model_name', 'deepseek-ai/deepseek-coder-6.7b-instruct')
        self.use_4bit = config.get('use_4bit', True)
        self.use_lora = config.get('use_lora', True)
        
        logger.info(f"모델 빌더 초기화 완료 (모델: {self.model_name})")
    
    def get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """4비트 양자화 설정 생성
        
        Returns:
            Optional[BitsAndBytesConfig]: 양자화 설정
        """
        if not self.use_4bit:
            return None
        
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("4비트 양자화 설정 생성 완료")
            return bnb_config
            
        except Exception as e:
            logger.error(f"양자화 설정 생성 실패: {e}")
            return None
    
    def load_tokenizer(self) -> AutoTokenizer:
        """토크나이저 로드
        
        Returns:
            AutoTokenizer: 로드된 토크나이저
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            
            # 패딩 토큰 설정
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            logger.info(f"토크나이저 로드 완료: {self.model_name}")
            return tokenizer
            
        except Exception as e:
            logger.error(f"토크나이저 로드 실패: {e}")
            raise e
    
    def load_base_model(self) -> AutoModelForCausalLM:
        """기본 모델 로드
        
        Returns:
            AutoModelForCausalLM: 로드된 모델
        """
        try:
            # 양자화 설정
            quantization_config = self.get_quantization_config()
            
            # 모델 로드 설정
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "device_map": "auto",
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            # 그래디언트 체크포인팅 설정
            if self.config.get('use_gradient_checkpointing', True):
                model_kwargs["use_cache"] = False
            
            # 모델 로드
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # 그래디언트 체크포인팅 활성화
            if self.config.get('use_gradient_checkpointing', True):
                model.gradient_checkpointing_enable()
                logger.info("그래디언트 체크포인팅 활성화")
            
            logger.info(f"기본 모델 로드 완료: {self.model_name}")
            return model
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise e
    
    def apply_lora(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """LoRA 적용
        
        Args:
            model: 기본 모델
            
        Returns:
            AutoModelForCausalLM: LoRA가 적용된 모델
        """
        if not self.use_lora or not PEFT_AVAILABLE:
            if not PEFT_AVAILABLE:
                logger.warning("PEFT 라이브러리가 없어 LoRA를 사용할 수 없습니다")
            return model
        
        try:
            # LoRA 설정
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.get('lora_r', 16),
                lora_alpha=self.config.get('lora_alpha', 32),
                lora_dropout=self.config.get('lora_dropout', 0.1),
                target_modules=self.config.get('lora_target_modules', [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]),
                bias="none",
            )
            
            # LoRA 적용
            model = get_peft_model(model, lora_config)
            
            # 학습 가능한 파라미터 수 계산
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            
            logger.info(f"LoRA 적용 완료")
            logger.info(f"학습 가능한 파라미터: {trainable_params:,} / {total_params:,} "
                       f"({trainable_params/total_params*100:.2f}%)")
            
            return model
            
        except Exception as e:
            logger.error(f"LoRA 적용 실패: {e}")
            raise e
    
    def build_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """모델과 토크나이저 빌드
        
        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: 모델과 토크나이저
        """
        try:
            # 토크나이저 로드
            tokenizer = self.load_tokenizer()
            
            # 기본 모델 로드
            model = self.load_base_model()
            
            # LoRA 적용
            model = self.apply_lora(model)
            
            # 모델을 학습 모드로 설정
            model.train()
            
            logger.info("모델과 토크나이저 빌드 완료")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"모델 빌드 실패: {e}")
            raise e
    
    def build_optimizer(self, model: AutoModelForCausalLM) -> torch.optim.Optimizer:
        """옵티마이저 빌드
        
        Args:
            model: 모델
            
        Returns:
            torch.optim.Optimizer: 옵티마이저
        """
        try:
            # 학습 가능한 파라미터만 선택
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            
            # AdamA 사용 시도 (실패 시 AdamW 사용)
            optimizer_name = self.config.get('optimizer', 'adamw')
            learning_rate = float(self.config.get('learning_rate', 1e-4))
            weight_decay = float(self.config.get('weight_decay', 0.01))
            
            logger.info(f"옵티마이저 설정: {optimizer_name}, LR: {learning_rate}, WD: {weight_decay}")
            
            if optimizer_name.lower() == 'adama' and self.config.get('adam_accumulation', False):
                try:
                    # AdamA 시도 (현재는 AdamW로 대체)
                    optimizer = AdamW(
                        trainable_params,
                        lr=learning_rate,
                        weight_decay=weight_decay,
                        betas=(0.9, 0.999),
                        eps=1e-8
                    )
                    logger.info("AdamA 모드 AdamW 옵티마이저 생성 완료")
                except Exception as e:
                    # AdamW로 폴백
                    optimizer = AdamW(
                        trainable_params,
                        lr=learning_rate,
                        weight_decay=weight_decay
                    )
                    logger.info(f"AdamW 옵티마이저 생성 완료 (AdamA 폴백: {e})")
            else:
                # 기본 AdamW
                optimizer = AdamW(
                    trainable_params,
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
                logger.info("AdamW 옵티마이저 생성 완료")
            
            return optimizer
            
        except Exception as e:
            logger.error(f"옵티마이저 빌드 실패: {e}")
            raise e
    
    def build_scheduler(self, optimizer: torch.optim.Optimizer, num_training_steps: int):
        """스케줄러 빌드
        
        Args:
            optimizer: 옵티마이저
            num_training_steps: 총 학습 스텝 수
            
        Returns:
            학습률 스케줄러
        """
        try:
            scheduler_type = self.config.get('lr_scheduler', 'cosine')
            warmup_ratio = float(self.config.get('warmup_ratio', 0.05))
            num_warmup_steps = int(num_training_steps * warmup_ratio)
            
            logger.info(f"스케줄러 설정: {scheduler_type}, 워밍업 비율: {warmup_ratio}")
            
            scheduler = get_scheduler(
                name=scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            
            logger.info(f"{scheduler_type} 스케줄러 생성 완료 "
                       f"(워밍업: {num_warmup_steps}/{num_training_steps} 스텝)")
            
            return scheduler
            
        except Exception as e:
            logger.error(f"스케줄러 빌드 실패: {e}")
            raise e
    
    def build_scaler(self) -> Optional[GradScaler]:
        """AMP 스케일러 빌드
        
        Returns:
            Optional[GradScaler]: AMP 스케일러
        """
        if not self.config.get('use_amp', True):
            return None
        
        try:
            scaler = GradScaler()
            logger.info("AMP 스케일러 생성 완료")
            return scaler
            
        except Exception as e:
            logger.error(f"AMP 스케일러 생성 실패: {e}")
            return None
    
    def load_finetuned_model(self, model_path: str, tokenizer: AutoTokenizer) -> Optional[AutoModelForCausalLM]:
        """파인튜닝된 모델 로드
        
        Args:
            model_path: 모델 경로
            tokenizer: 토크나이저
            
        Returns:
            Optional[AutoModelForCausalLM]: 로드된 모델
        """
        try:
            if not os.path.exists(model_path):
                logger.warning(f"파인튜닝된 모델을 찾을 수 없습니다: {model_path}")
                return None
            
            # 양자화 설정
            quantization_config = self.get_quantization_config()
            
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "device_map": "auto",
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            # 파인튜닝된 모델 로드
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            # 그래디언트 체크포인팅 활성화
            if self.config.get('use_gradient_checkpointing', True):
                model.gradient_checkpointing_enable()
            
            logger.info(f"파인튜닝된 모델 로드 완료: {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"파인튜닝된 모델 로드 실패: {e}")
            return None
    
    def get_model_info(self, model: AutoModelForCausalLM) -> Dict[str, Any]:
        """모델 정보 수집
        
        Args:
            model: 모델
            
        Returns:
            Dict[str, Any]: 모델 정보
        """
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            info = {
                'model_name': self.model_name,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'trainable_percentage': (trainable_params / total_params) * 100,
                'use_4bit': self.use_4bit,
                'use_lora': self.use_lora,
                'device': str(next(model.parameters()).device),
                'dtype': str(next(model.parameters()).dtype)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"모델 정보 수집 실패: {e}")
            return {}
