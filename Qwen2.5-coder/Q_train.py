#!/usr/bin/env python3
"""
Qwen2.5-coder-7B-instruct 전용 QLoRA 파인튜닝 스크립트
- Prompt 모드 전용 파인튜닝
- Qwen2.5 토큰화 형식 지원: <|im_start|>role\ncontent<|im_end|>
- T4 GPU 최적화
- AWS 스팟 인스턴스 중단 자동 감지 및 체크포인트 관리
"""

import os
import sys
import json
import yaml
import logging
import argparse
import tempfile
import inspect
import re
from datetime import datetime
from typing import Dict, Any, Optional, List

import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# TRL 임포트 (버전별 처리)
try:
    from trl import SFTTrainer
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    print("⚠️ TRL 없음 - 기본 Trainer 사용")

# 로깅 설정
# logs 디렉토리 생성
os.makedirs('logs', exist_ok=True)

# 현재 시간을 로그 파일명에 추가
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/qwen_training_{current_time}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename, mode='a')
    ]
)
log = logging.getLogger(__name__)

class QwenTrainer:
    """Qwen2.5-coder-7B-instruct prompt 모드용 QLoRA 파인튜닝 클래스"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.cfg = self.load_config(config_path)
        self.model = None
        self.tok = None
        self.train_ds = None
        self.eval_ds = None
        
        # T4 환경 최적화 설정 적용
        self.optimize_for_t4()
        
        log.info(f"🚀 Qwen2.5-coder-7B-instruct 파인튜닝 초기화 완료")
    
    def get_int_config(self, key, default=0):
        """config에서 값을 가져와 int로 변환하는 헬퍼 함수"""
        value = self.cfg.get(key, default)
        if isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                log.warning(f"⚠️ 경고: {key} 값 '{value}'을 정수로 변환할 수 없습니다. 기본값 {default} 사용")
                value = default
        return value
    
    def get_float_config(self, key, default=0.0):
        """config에서 값을 가져와 float로 변환하는 헬퍼 함수"""
        value = self.cfg.get(key, default)
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                log.warning(f"⚠️ 경고: {key} 값 '{value}'을 실수로 변환할 수 없습니다. 기본값 {default} 사용")
                value = default
        return value
        
    def optimize_for_t4(self):
        """T4 GPU 환경에 맞는 최적화 설정"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            compute_cap = torch.cuda.get_device_capability(0)
            
            log.info(f"🖥️ GPU: {gpu_name}")
            log.info(f"🔧 Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
            
            # T4는 Compute Capability 7.5이므로 bfloat16 미지원
            if compute_cap[0] < 8:
                log.info("⚠️ bfloat16 미지원 GPU - float16 사용")
                self.cfg["bnb_4bit_compute_dtype"] = "float16"
                self.cfg["torch_dtype"] = "float16"
                self.cfg["fp16"] = True
                self.cfg["bf16"] = False
            
            # T4 메모리 제한 고려 (15GB)
            if "T4" in gpu_name:
                log.info("🔧 T4 환경 최적화 적용")
                self.cfg["batch_size"] = 1
                self.cfg["grad_acc"] = max(8, self.get_int_config("grad_acc", 4))
                self.cfg["max_length"] = min(1024, self.get_int_config("max_length", 2048))
    
    @staticmethod
    def default_cfg():
        """T4 환경 최적화된 기본 설정"""
        return {
            # 모델 설정 - Qwen 모델 HuggingFace ID
            "model_name": "Qwen/Qwen2.5-coder-7B-instruct",
            "torch_dtype": "float16",
            
            # 양자화 설정 (T4 최적화)
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            
            # LoRA 설정 - Qwen 모델에 맞는 타겟 모듈 설정
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            
            # 학습 설정 (T4 메모리 제한 고려)
            "batch_size": 1,
            "grad_acc": 8,
            "learning_rate": 2e-4,
            "num_epochs": 3,
            "max_length": 1024,
            "warmup_ratio": 0.1,
            "lr_scheduler": "cosine",
            "weight_decay": 0.01,
            
            # 최적화 설정
            "fp16": True,
            "bf16": False,
            "gradient_checkpointing": True,
            "dataloader_num_workers": 4,
            "remove_unused_columns": False,
            
            # 저장 설정 (save_steps: 10)
            "save_steps": 10,
            "eval_steps": 50,
            "logging_steps": 5,
            "save_total_limit": 5,
            
            # 데이터 설정
            "data_path": "../data/train.jsonl"
        }
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로딩"""
        default_config = self.default_cfg()
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                default_config.update(user_config)
                log.info(f"✅ 설정 파일 로딩: {config_path}")
            except Exception as e:
                log.warning(f"⚠️ 설정 파일 로딩 실패, 기본값 사용: {e}")
        else:
            log.info("📝 기본 설정 사용")
        
        return default_config
        
    def load_model(self):
        """T4 최적화된 모델 및 토크나이저 로딩"""
        log.info("🔄 모델 로딩 시작")
        log.info("▶️  베이스 모델 4-bit 로드")
        
        # 양자화 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.cfg["load_in_4bit"],
            bnb_4bit_compute_dtype=getattr(torch, self.cfg["bnb_4bit_compute_dtype"]),
            bnb_4bit_quant_type=self.cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=self.cfg["bnb_4bit_use_double_quant"],
        )
        
        # 토크나이저 로딩
        model_path = self.cfg["model_name"]
        
        # HuggingFace 모델 ID 확인
        is_huggingface_id = '/' not in model_path
        
        log.info(f"토크나이저 로드 중: {model_path} (HuggingFace ID: {is_huggingface_id})")
        
        try:
            self.tok = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="right",
                local_files_only=False
            )
            log.info(f"토크나이저 로딩 성공")
        except Exception as e:
            log.error(f"토크나이저 로딩 오류: {e}")
            raise  # 오류 발생 시 즉시 중단
        
        # 패딩 토큰 설정 - Qwen 모델은 패딩 토큰 설정이 필요할 수 있음
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
            self.tok.pad_token_id = self.tok.eos_token_id
            
        # FIM 토큰 확인 - Qwen에는 FIM 토큰이 있는지 확인
        fim_tokens = ['<|fim_prefix|>', '<|fim_middle|>', '<|fim_suffix|>']
        missing_tokens = []
        
        for token in fim_tokens:
            if token not in self.tok.get_vocab():
                missing_tokens.append(token)
        
        if missing_tokens:
            log.warning(f"⚠️ 다음 FIM 토큰이 토크나이저에 없습니다: {missing_tokens}")
            log.info("FIM 토큰을 토크나이저에 추가합니다.")
            # 토큰 추가하기
            self.tok.add_tokens(missing_tokens)
            
        # 모델 로딩
        log.info(f"모델 로드 중: {model_path}")
        
        # 메모리 상태 확인
        if torch.cuda.is_available():
            log.info(f"CUDA 메모리 상태 (모델 로드 전): 사용 중: {torch.cuda.memory_allocated()/1024**3:.2f}GB, 예약됨: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
        
        import time
        start_time = time.time()
        log.info("모델 로딩 시작 - 이 과정은 몇 분 소요될 수 있습니다...")
        
        try:
            # 로딩 설정 업데이트 - 더 느리지만 메모리를 효율적으로 사용
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=getattr(torch, self.cfg["torch_dtype"]),
                use_cache=False,
                local_files_only=False,
                low_cpu_mem_usage=True,  # 메모리 사용량 최적화
                offload_folder="model_cache"  # 로컬 오프로드 폴더 지정
            )
            end_time = time.time()
            log.info(f"모델 로딩 성공! 소요 시간: {(end_time-start_time):.2f}초")
            
            if torch.cuda.is_available():
                log.info(f"CUDA 메모리 상태 (모델 로드 후): 사용 중: {torch.cuda.memory_allocated()/1024**3:.2f}GB, 예약됨: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
                
        except Exception as e:
            log.error(f"모델 로딩 오류: {e}")
            import traceback
            log.error(f"상세 오류: {traceback.format_exc()}")
            raise  # 오류 발생 시 즉시 중단
            
        # 토크나이저에 새 토큰을 추가했다면 모델의 임베딩 크기 조정
        if missing_tokens:
            log.info(f"모델 임베딩 크기 조정 (FIM 토큰 추가)")
            self.model.resize_token_embeddings(len(self.tok))
        
        # k-bit 학습을 위한 모델 준비
        self.model = prepare_model_for_kbit_training(
            self.model, 
            use_gradient_checkpointing=self.cfg["gradient_checkpointing"]
        )
        log.info("✅ 그래디언트 체크포인팅으로 k-bit 학습을 위한 모델 준비 완료.")
        
        # LoRA 설정 및 적용
        lora_config = LoraConfig(
            r=self.cfg["lora_r"],
            lora_alpha=self.cfg["lora_alpha"],
            target_modules=self.cfg["lora_target_modules"],
            lora_dropout=self.cfg["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        log.info("✅ 모델 로딩 완료")

    def load_data(self):
        """데이터 로딩 및 전처리"""
        log.info("🔄 데이터 로딩 및 전처리 시작")
        log.info(f"▶️  데이터 로딩: {self.cfg['data_path']}")
        
        # JSONL 파일 로딩
        dataset = load_dataset("json", data_files=self.cfg["data_path"], split="train")
        
        log.info(f"📊 데이터셋 컬럼: {dataset.column_names}")
        log.info(f"📊 데이터셋 크기: {len(dataset)}개 샘플")
        
        # 데이터 형식을 감지하고 적절한 변환 함수 선택
        def detect_format(example):
            """데이터 형식을 감지하는 함수 - 필수 형식만 지원"""
            if "system" in example and "user" in example and "assistant" in example:
                return "system_user_assistant"  # 시스템/사용자/어시스턴트 메시지 포맷
            elif "meta" in example and "user" in example and "assistant" in example:
                return "meta_user_assistant"  # 메타데이터와 함께 사용자/어시스턴트 메시지 형식
            elif "prompt" in example and "completion" in example:
                return "prompt_completion"  # 표준 프롬프트-완성 형식 (기본 형식)
            else:
                # 데이터 구조 로깅 - 지원되지 않는 형식
                log.warning(f"⚠️ 지원되지 않는 데이터 형식: {list(example.keys())}")
                return "unknown"
        
        # 샘플을 검사하여 데이터 형식 감지
        sample = dataset[0]
        data_format = detect_format(sample)
        log.info(f"✅ 감지된 데이터 형식: {data_format}")
        
        # 토크나이징 함수
        def tokenize_function(examples):
            # 데이터 형식에 따라 Qwen 형식으로 변환
            texts = []
            
            # 샘플 수 결정
            if "user" in examples and "assistant" in examples:
                sample_count = len(examples["user"])  # system, user, assistant는 같은 길이를 가정
            elif "prompt" in examples and "completion" in examples:
                sample_count = len(examples["prompt"])
            else:
                log.error("❌ 지원되지 않는 데이터 형식")
                raise ValueError("지원되지 않는 데이터 형식입니다.")
            
            # Qwen2.5 형식으로 변환하는 공통 함수
            def format_qwen_message(system_msg, user_msg, assistant_msg):
                text = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                text += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                text += f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
                return text
                
            for i in range(sample_count):
                # 데이터 형식별 처리
                if data_format == "system_user_assistant":
                    system = examples["system"][i].strip()
                    user = examples["user"][i].strip()
                    assistant = examples["assistant"][i].strip()
                    
                    # FIM 토큰 처리 (Fill-in-the-Middle 형식 지원)
                    if "<|fim_prefix|>" in user and "<|fim_middle|>" in user and "<|fim_suffix|>" in user:
                        log.debug("🔍 FIM 형식 감지됨")
                    
                    # 공통 함수를 통한 변환
                    texts.append(format_qwen_message(system, user, assistant))
                
                elif data_format == "meta_user_assistant":
                    # 메타데이터와 함께 사용자/어시스턴트 메시지 형식
                    meta = examples["meta"][i]
                    user = examples["user"][i].strip()
                    assistant = examples["assistant"][i].strip()
                    
                    # 메타데이터에서 시스템 메시지 추출 또는 기본값 사용
                    if "persona_name" in meta and "scenario" in meta:
                        system_msg = f"You are {meta['persona_name']} who helps users with {meta['scenario']}"
                    else:
                        system_msg = "You are a helpful coding assistant."
                    
                    # 공통 함수를 통한 변환
                    texts.append(format_qwen_message(system_msg, user, assistant))
                
                elif data_format == "prompt_completion":
                    # 프롬프트-완성 형식을 Qwen 메세지 형식으로 변환
                    prompt = examples["prompt"][i].strip()
                    completion = examples["completion"][i].strip()
                    
                    # 기본 시스템 메시지 사용
                    system_msg = "You are a helpful Python coding assistant."
                    
                    # FIM 토큰 처리 (prompt 모드에서만 필요)
                    if "<|fim_prefix|>" in prompt and "<|fim_middle|>" in prompt and "<|fim_suffix|>" in prompt:
                        log.debug("🔍 FIM 형식 감지됨")
                    
                    # 공통 함수를 통한 변환
                    texts.append(format_qwen_message(system_msg, prompt, completion))
                
                else:
                    log.warning(f"⚠️ 지원되지 않는 데이터 형식: {data_format}")
                    # 대체 사용할 임시 데이터
                    texts.append("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there!<|im_end|>")
            # 토크나이징
            model_inputs = self.tok(
                texts,
                truncation=True,
                padding=False,
                max_length=self.cfg["max_length"],
                return_tensors=None
            )
            
            # labels = input_ids (causal LM)
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            
            return model_inputs
        
        # 데이터셋 토크나이징
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="토크나이징 중"
        )
        
        log.info("✅ 데이터 토크나이징 완료")
        
        # 학습/검증 분할
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        self.train_ds = split_dataset["train"]
        self.eval_ds = split_dataset["test"]
        
        log.info(f"📊 학습 데이터: {len(self.train_ds)}개")
        log.info(f"📊 검증 데이터: {len(self.eval_ds)}개")
        
    def t_args(self) -> TrainingArguments:
        """학습 인자 생성 - prompt 모드 전용"""
        # 학습 경로 설정 - 체크포인트와 출력용
        output_dir = "../scripts/checkpoints"
        
        # 최종 모델 경로
        self.final_model_dir = "../models/prompt-finetuned"
        
        # 최소한의 안전한 인자만 사용 (타입 변환 추가)
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.get_int_config("batch_size", 1),
            per_device_eval_batch_size=self.get_int_config("batch_size", 1),
            gradient_accumulation_steps=self.get_int_config("grad_acc", 8),
            learning_rate=self.get_float_config("learning_rate", 2e-4),
            num_train_epochs=self.get_float_config("num_epochs", 3),
            warmup_ratio=self.get_float_config("warmup_ratio", 0.1),
            lr_scheduler_type=self.cfg["lr_scheduler"],
            weight_decay=self.get_float_config("weight_decay", 0.01),
            fp16=self.cfg["fp16"],
            bf16=self.cfg["bf16"],
            gradient_checkpointing=self.cfg["gradient_checkpointing"],
            logging_steps=self.get_int_config("logging_steps", 5),
            evaluation_strategy="steps",
            eval_steps=self.get_int_config("eval_steps", 50),
            save_strategy="steps",
            save_steps=self.get_int_config("save_steps", 10),
            save_total_limit=self.get_int_config("save_total_limit", 5),
            remove_unused_columns=self.cfg["remove_unused_columns"],
            dataloader_num_workers=self.get_int_config("dataloader_num_workers", 4),
            report_to="none",  # 빠른 시작을 위해 보고 기능 비활성화
            # 추가 필요 설정
            push_to_hub=False,
            group_by_length=True  # 비슷한 길이의 시퀀스를 함께 배치하여 패딩 최소화
        )

    def find_latest_checkpoint(self, checkpoint_dir):
        """가장 최신 체크포인트 경로 찾기
        
        Args:
            checkpoint_dir: 체크포인트 디렉토리
            
        Returns:
            str: 최신 체크포인트 경로 또는 None
        """
        if not os.path.exists(checkpoint_dir):
            log.warning(f"⚠️ 체크포인트 디렉토리가 없습니다: {checkpoint_dir}")
            return None
        
        # 체크포인트 디렉토리 내에서 checkpoint-* 패턴을 찾음
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
        if not checkpoints:
            log.warning(f"⚠️ 체크포인트를 찾을 수 없습니다: {checkpoint_dir}")
            return None
            
        # 체크포인트 번호로 정렬
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
        
        log.info(f"🔍 가장 최신 체크포인트 발견: {latest_checkpoint}")
        return latest_checkpoint

    def train(self):
        """모델 학습 실행"""
        log.info("🚀 학습 시작")

        # 모델, 데이터 로드
        self.load_model()
        self.load_data()

        # 훈련 인자 생성
        args = self.t_args()

        # 데이터 콜레이터 설정
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tok,
            mlm=False  # CLM 사용
        )

        # SpotInterruptionHandler 인스턴스 생성 (AWS 스팟 인스턴스 중단 감지)
        spot_handler = SpotInterruptionHandler()
        spot_handler.start_monitoring()
        log.info("✅ AWS 스팟 인스턴스 중단 감지 활성화")

        try:
            # TRL 사용 가능 여부에 따라 Trainer 선택
            if TRL_AVAILABLE:
                log.info("✅ TRL SFTTrainer 사용")
                trainer = SFTTrainer(
                    model=self.model,
                    args=args,
                    train_dataset=self.train_ds,
                    eval_dataset=self.eval_ds,
                    tokenizer=self.tok,
                    data_collator=data_collator,
                    max_seq_length=self.cfg["max_length"]
                )
            else:
                log.info("✅ 기본 Trainer 사용")
                trainer = Trainer(
                    model=self.model,
                    args=args,
                    train_dataset=self.train_ds,
                    eval_dataset=self.eval_ds,
                    tokenizer=self.tok,
                    data_collator=data_collator
                )

            # 체크포인트 자동 감지 및 학습 재개 설정
            checkpoint_path = None
            
            # 환경 변수에서 auto 설정 감지
            resume_setting = os.environ.get('RESUME_CHECKPOINT', '').strip().lower()
            auto_resume = (resume_setting == 'auto')
            
            if auto_resume:
                log.info("🔍 체크포인트 자동 감지 모드 활성화")
                checkpoint_path = self.find_latest_checkpoint(args.output_dir)
                if checkpoint_path:
                    log.info(f"✅ 체크포인트 자동 감지 완료: {checkpoint_path}")
                else:
                    log.info("💡 체크포인트가 없어서 처음부터 학습을 시작합니다.")
            elif args.resume_from_checkpoint is not None:
                # 명시적으로 지정된 체크포인트 사용
                log.info(f"✅ 체크포인트 재개: {args.resume_from_checkpoint}")
                checkpoint_path = args.resume_from_checkpoint

            # 학습 시작
            log.info(f"🚀 학습 시작 (resume_from_checkpoint={checkpoint_path})")
            train_result = trainer.train(resume_from_checkpoint=checkpoint_path)
            
            # 최종 모델 저장
            log.info(f"✅ 학습 완료, 저장 중... 출력 경로: {self.final_model_dir}")
            trainer.save_state()
            
            # 최종 모델 저장 디렉토리 생성
            os.makedirs(self.final_model_dir, exist_ok=True)
            
            # 모델 저장 (peft 형태로)
            self.model.save_pretrained(self.final_model_dir)
            self.tok.save_pretrained(self.final_model_dir)
            
            # config.json 파일에 학습 정보 추가
            config_path = os.path.join(self.final_model_dir, "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    
                    # 학습 정보 추가
                    training_info = {
                        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model_name": self.cfg["model_name"],
                        "data_path": self.cfg["data_path"],
                        "training_loss": float(train_result.training_loss),
                        "training_runtime": float(train_result.metrics['train_runtime']),
                        "training_samples": len(self.train_ds),
                    }
                    
                    config_data["training_info"] = training_info
                    
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config_data, f, indent=2, ensure_ascii=False)
                        
                    log.info("✅ 학습 정보가 config.json에 추가되었습니다.")
                except Exception as e:
                    log.warning(f"⚠️ config.json 업데이트 오류: {e}")
            
            # 학습 결과 로그
            log.info("✅ 학습 완료!")
            log.info(f"✅ 최종 loss: {train_result.training_loss}")
            log.info(f"✅ 총 학습 시간: {train_result.metrics['train_runtime']}초")
            log.info(f"✅ 최종 모델 저장 위치: {self.final_model_dir}")
            log.info(f"📦 HuggingFace 모델 ID: {self.cfg['model_name']}")
            log.info(f"📂 데이터 경로: {self.cfg['data_path']}")
            log.info(f"📝 체크포인트 경로: {args.output_dir}")
            log.info("👍 새로운 Qwen2.5-Coder-7B 학습 완료")
            
        except Exception as e:
            log.error(f"❌ 학습 중 오류 발생: {e}")
            import traceback
            log.error(traceback.format_exc())
            raise
        finally:
            # 스팟 인스턴스 모니터링 중지
            spot_handler.stop_monitoring()


class SpotInterruptionHandler:
    """AWS 스팟 인스턴스 중단 감지 및 처리 클래스
    
    스팟 인스턴스 중단 감지 시 체크포인트 저장 후 종료하는 기능 제공
    """
    
    def __init__(self):
        self.monitor_thread = None
        self.should_stop = False
        self.meta_endpoint = "http://169.254.169.254/latest/meta-data/spot/instance-action"
        self.check_interval = 5  # 5초마다 체크
        
        # SIGTERM 핸들러 등록
        import signal
        self.original_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._handle_sigterm)
    
    def _handle_sigterm(self, signum, frame):
        """시그턴 신호 처리"""
        log.warning("🚨 SIGTERM 신호 감지! 체크포인트 저장 후 종료합니다.")
        
        # 원래 핸들러 호출
        if callable(self.original_handler):
            self.original_handler(signum, frame)
    
    def check_interruption(self):
        """스팟 인스턴스 중단 예고 감지"""
        try:
            import requests
            # AWS 메타데이터 엔드포인트 요청
            response = requests.get(self.meta_endpoint, timeout=0.5)
            
            if response.status_code == 200:
                log.warning("⚠️ AWS 스팟 인스턴스 중단 감지!")
                
                # 메인 프로세스에 SIGTERM 신호 보내기
                import signal
                os.kill(os.getpid(), signal.SIGTERM)
                return True
                
            return False
        except:
            return False
    
    def monitor(self):
        """백그라운드에서 주기적으로 중단 감지"""
        log.info("🔍 AWS 스팟 인스턴스 중단 모니터링 시작")
        import time
        
        while not self.should_stop:
            if self.check_interruption():
                break
                
            # 5초마다 체크
            time.sleep(self.check_interval)
                
        log.info("🛑 스팟 인스턴스 모니터링 종료")
    
    def start_monitoring(self):
        """모니터링 시작"""
        import threading
        self.should_stop = False
        
        # AWS 환경인지 체크
        try:
            import requests
            requests.get("http://169.254.169.254/", timeout=0.2)
            is_aws = True
        except:
            is_aws = False
        
        if not is_aws:
            log.warning("⚠️ AWS 환경이 아닙니다. 스팟 인스턴스 모니터링은 비활성화됩니다.")
            return
        
        self.monitor_thread = threading.Thread(target=self.monitor, daemon=True)
        self.monitor_thread.start()
        log.info("✅ 스팟 인스턴스 모니터링 시작")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        if self.monitor_thread is None:
            return
            
        self.should_stop = True
        try:
            self.monitor_thread.join(timeout=1.0)  # 최대 1초 대기
        except:
            pass
        
        self.monitor_thread = None
        log.info("✅ 스팟 인스턴스 모니터링 중지")


def main():
    """메인 함수 - Qwen2.5-coder-7B-instruct prompt 모드 파인튜닝 시작"""
    parser = argparse.ArgumentParser(description="Qwen2.5-coder-7B-instruct Prompt 모드 QLoRA 파인튜닝")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="설정 파일 경로"
    )
    args = parser.parse_args()
    
    # 파인튜닝 실행
    trainer = QwenTrainer(config_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()