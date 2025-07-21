#!/usr/bin/env python3
"""
DeepSeek-Coder 6.7B QLoRA 파인튜닝 스크립트 (완전 호환성 보장)
- 모든 TRL/Transformers 버전 지원
- T4 GPU 최적화
- save_steps: 10 지원
"""

import os
import sys
import json
import yaml
import logging
import argparse
import tempfile
import inspect
from datetime import datetime
from typing import Dict, Any, Optional

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
log_filename = f"logs/training_{current_time}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename, mode='a')
    ]
)
log = logging.getLogger(__name__)

class DeepSeekTrainer:
    """DeepSeek-Coder 6.7B QLoRA 파인튜닝 클래스 (완전 호환성)"""
    
    def __init__(self, config_path: str = "config.yaml", mode: str = "prompt"):
        self.mode = mode
        self.cfg = self.load_config(config_path)
        self.model = None
        self.tok = None
        self.train_ds = None
        self.eval_ds = None
        
        # T4 환경 최적화 설정 적용
        self.optimize_for_t4()
        
        log.info(f"🚀 DeepSeek-Coder 6.7B 파인튜닝 초기화 완료 (모드: {mode})")
    
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
            # 모델 설정 - 로컬 절대 경로 사용
            "model_name": "/home/ubuntu/deepseek-coder/model_cache/deepseek-coder-6.7b-instruct",
            "torch_dtype": "float16",
            
            # 양자화 설정 (T4 최적화)
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            
            # LoRA 설정
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            
            # 학습 설정 (T4 메모리 제한 고려)
            "batch_size": 1,
            "grad_acc": 8,
            "learning_rate": 2e-4,
            "num_epochs": 3,
            "max_length": 2048,
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
            "data_path": "./data/train.jsonl"
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
        
        # 경로 확인
        if not model_path.startswith('/'):
            log.warning(f"⚠️ 경고: 절대 경로가 아닙니다. '/' 추가: {model_path}")
            model_path = '/' + model_path
        
        # 디렉토리 존재 확인
        if os.path.exists(model_path):
            log.info(f"✅ 모델 경로 확인 성공: {model_path}")
            is_local_path = True
        else:
            log.warning(f"⚠️ 경고: 모델 경로가 존재하지 않습니다: {model_path}")
            # 그래도 로컬 경로로 설정 (Hugging Face로 인식 방지)
            is_local_path = True
        
        log.info(f"토크나이저 로드 중: {model_path} (로컬 경로: {is_local_path})")
        
        # config.json 파일 확인
        config_path = os.path.join(model_path, 'config.json')
        tokenizer_path = os.path.join(model_path, 'tokenizer.json')
        if os.path.exists(config_path) and os.path.exists(tokenizer_path):
            log.info(f"✅ 모델 파일 확인: config.json, tokenizer.json 발견")
        else:
            files = os.listdir(model_path) if os.path.exists(model_path) else []
            log.warning(f"⚠️ 모델 파일 정보: {files}")
        
        try:
            self.tok = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="right",
                local_files_only=True
            )
            log.info(f"토크나이저 로딩 성공")
        except Exception as e:
            log.error(f"토크나이저 로딩 오류: {e}")
            raise  # 오류 발생 시 즉시 중단
        
        # 패딩 토큰 설정
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
            self.tok.pad_token_id = self.tok.eos_token_id
        
        # 모델 로딩
        log.info(f"모델 로드 중: {model_path}")
        
        # 모델 파일 확인
        files = os.listdir(model_path)
        log.info(f"모델 디렉토리 파일 목록: {files}")
        
        # 샤드 파일 확인
        shard_files = [f for f in files if 'model.safetensors.index.json' in f or 
                      ('model' in f and '.safetensors' in f) or 
                      ('pytorch_model' in f and '.bin' in f)]
        log.info(f"모델 샤드 파일: {shard_files}")
        
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
                local_files_only=True,
                low_cpu_mem_usage=True,  # 메모리 사용량 최적화
                offload_folder="offload_folder"  # 필요시 디스크로 오프로드
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
        
        # 첫 샘플 내용 로깅 (디버깅용)
        if len(dataset) > 0:
            first_sample = dataset[0]
            if "messages" in first_sample and len(first_sample["messages"]) > 0:
                log.debug(f"첫 번째 샘플 메시지 구조: {first_sample['messages'][0].keys()}")
        
        # 데이터 형식을 감지하고 적절한 변환 함수 선택
        def detect_format(example):
            """데이터 형식을 감지하는 함수"""
            if "messages" in example:
                # messages 배열이 있고, 배열 형식이며, 각 메시지는 role과 content를 가짐
                if (len(example["messages"]) > 0 and 
                    isinstance(example["messages"], list) and 
                    isinstance(example["messages"][0], dict) and 
                    "role" in example["messages"][0] and 
                    "content" in example["messages"][0]):
                    return "chat"
            
            elif "prompt" in example and "completion" in example:
                return "prompt_completion"
                
            elif "prefix_code" in example and "suffix_code" in example and "comment" in example and "target_code" in example:
                # 주석 기반 코드 생성 형식
                return "comment_to_code"
                
            elif "error_context" in example and "explanation" in example:
                return "error_explanation"
                
            elif "error_context" in example and "fixed_code_snippet" in example:
                # error_fix 형식 감지
                error_context = example["error_context"]
                if isinstance(error_context, dict) and "buggy_code_snippet" in error_context:
                    return "error_fix"
                # 또는 buggy_code_snippet이 최상위 필드에 있는 경우
                elif "buggy_code_snippet" in example:
                    return "error_fix"
                    
            elif "instruction" in example and "input" in example and "output" in example:
                return "instruction_input_output"
                
            else:
                # 데이터 구조 로깅
                log.warning(f"\u26a0️ 알 수 없는 데이터 형식: {list(example.keys())}")
                return "unknown"
        
        # 샘플을 검사하여 데이터 형식 감지
        sample = dataset[0]
        data_format = detect_format(sample)
        log.info(f"✅ 감지된 데이터 형식: {data_format}")
        
        # FIM 형식 사용 여부를 미리 확인 (로그 중복 방지)
        has_fim_format = False
        if data_format == "prompt_completion" and self.mode == "comment":
            # 첫 몇 개 샘플 검사하여 FIM 태그 사용 여부 확인
            sample_check_count = min(5, len(dataset))
            for i in range(sample_check_count):
                sample_prompt = dataset[i]["prompt"]
                if "<|fim begin|>" in sample_prompt or "<|fim hole|>" in sample_prompt or "<|fim end|>" in sample_prompt:
                    log.info("✅ FIM 형식 주석 기반 모델 데이터 감지됨")
                    has_fim_format = True
                    break
        
        # 토크나이징 함수
        def tokenize_function(examples):
            # 데이터 형식에 따라 ChatML 형식으로 변환
            texts = []
            
            # 샘플 수 결정 (다양한 형식 지원)
            if "messages" in examples:
                sample_count = len(examples["messages"])
            elif "prompt" in examples:
                sample_count = len(examples["prompt"])
            elif "prefix_code" in examples:
                sample_count = len(examples["prefix_code"])
            elif "error_context" in examples:
                sample_count = len(examples["error_context"])
            elif "instruction" in examples:
                sample_count = len(examples["instruction"])
            else:
                log.error("❌ 지원되지 않는 데이터 형식")
                raise ValueError("지원되지 않는 데이터 형식입니다.")
                
            # 데이터 형식 로깅
            log.info(f"🔢 샘플 수: {sample_count}, 형식: {data_format}")
            
            for i in range(sample_count):
                # 데이터 형식별 처리
                if data_format == "chat":
                    # 채팅 형식 (messages 구조) - Qwen2.5 모델 형식 지원
                    messages = examples["messages"][i]
                    chat_text = ""
                    
                    # 모든 메시지를 ChatML 형식으로 변환
                    for msg in messages:
                        role = msg["role"]
                        content = msg["content"]
                        chat_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                    
                    texts.append(chat_text.strip())
                
                elif data_format == "prompt_completion":
                    # 주석 키워드 기반 코드 생성 형식
                    prompt = examples["prompt"][i]
                    completion = examples["completion"][i]
                    
                    # FIM 형식 처리 (3차 주석 기반 모델)
                    if self.mode == "comment" and ("<|fim begin|>" in prompt or "<|fim hole|>" in prompt or "<|fim end|>" in prompt):
                        # FIM 형식 그대로 유지하고 ChatML 형식으로 래핑
                        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>"
                        texts.append(text)
                    else:
                        # 일반 프롬프트를 사용자 메시지로, 완성을 어시스턴트 메시지로 변환
                        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>"
                    
                    texts.append(text)
                
                elif data_format == "error_explanation":
                    # 에러 설명 형식
                    error_context = examples["error_context"][i]
                    explanation = examples["explanation"][i]
                    
                    # 에러 컨텍스트 정보 구성
                    error_log = error_context.get("error_log", "")
                    code_snippet = error_context.get("code_snippet", "")
                    language = error_context.get("language", "")
                    
                    # 사용자 입력 구성
                    user_text = f"다음 {language} 코드의 에러를 설명해주세요:\n\n에러 로그:\n{error_log}\n\n코드:\n{code_snippet}"
                    
                    # ChatML 형식으로 변환
                    text = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n{explanation}<|im_end|>"
                    texts.append(text)
                
                elif data_format == "comment_to_code":
                    # 주석 기반 코드 생성 형식 (새로운 데이터 구조)
                    prefix_code = examples["prefix_code"][i]
                    suffix_code = examples["suffix_code"][i]
                    comment = examples["comment"][i]
                    # instruction이 없는 경우 주석을 활용
                    instruction = examples.get("instruction", [comment] * sample_count)[i]
                    target_code = examples["target_code"][i]
                    
                    # 사용자 입력 구성 (주석과 코드 컨텍스트)
                    user_text = f"주석에 따라 적절한 코드를 생성해주세요.\n\n"
                    user_text += f"주석: {comment}\n\n"
                    if instruction and instruction != comment:
                        user_text += f"지시사항: {instruction}\n\n"
                    user_text += "\n이전 코드:\n"
                    user_text += f"{prefix_code}\n"
                    user_text += "// 여기에 코드를 삽입해야 함 //\n"
                    user_text += f"{suffix_code}"
                    
                    # ChatML 형식으로 변환
                    text = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n{target_code}<|im_end|>"
                    texts.append(text)
                
                elif data_format == "error_fix":
                    # 에러 수정 형식
                    error_context = examples["error_context"][i]
                    fixed_code = examples["fixed_code_snippet"][i]
                    
                    # 에러 컨텍스트 정보 구성
                    error_log = error_context.get("error_log", "")
                    language = error_context.get("language", "")
                    title = error_context.get("title", "")
                    description = error_context.get("description", "")
                    
                    # buggy_code가 error_context 안에 있는 경우와 외부에 있는 경우 모두 처리
                    if "buggy_code_snippet" in error_context:
                        buggy_code = error_context["buggy_code_snippet"]
                    elif "buggy_code_snippet" in examples:
                        buggy_code = examples["buggy_code_snippet"][i]
                    else:
                        buggy_code = ""
                        log.warning("\u26a0️ buggy_code_snippet을 찾을 수 없습니다")
                    
                    # 사용자 입력 구성 - 제목과 설명 추가
                    user_text = f"다음 {language} 코드의 에러를 수정해주세요:\n"
                    if title:
                        user_text += f"\n오류: {title}"
                    if description:
                        user_text += f"\n설명: {description}"
                    user_text += f"\n\n에러 로그:\n{error_log}\n\n코드:\n{buggy_code}"
                    
                    # ChatML 형식으로 변환
                    text = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n{fixed_code}<|im_end|>"
                    texts.append(text)
                    
                elif data_format == "instruction_input_output":
                    # 기존 instruction, input, output 형식
                    instruction = examples["instruction"][i].strip()
                    input_text = examples["input"][i].strip()
                    output_text = examples["output"][i].strip()
                    
                    # 태그 정보 추가 (있다면)
                    tags = examples.get("tags", [None] * len(examples["instruction"]))[i]
                    if tags:
                        if isinstance(tags, str):
                            tags = [tags]
                        tag_info = f"Task Type: {', '.join(tags)}\n"
                        if input_text:
                            user_text = f"{tag_info}Instruction: {instruction}\nInput: {input_text}"
                        else:
                            user_text = f"{tag_info}Instruction: {instruction}"
                    else:
                        if input_text:
                            user_text = f"Instruction: {instruction}\nInput: {input_text}"
                        else:
                            user_text = f"Instruction: {instruction}"
                    
                    # ChatML 형식
                    text = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
                    texts.append(text)
                
                else:
                    log.error(f"❌ 지원되지 않는 데이터 형식: {data_format}")
                    raise ValueError(f"지원되지 않는 데이터 형식: {data_format}")
            
            # FIM 형식 특수 처리 (prompt에 <|fim begin|>, <|fim hole|>, <|fim end|> 태그가 있는 경우)
            if data_format == "prompt_completion":
                for i in range(len(texts)):
                    prompt = examples["prompt"][i]
                    if "<｜fim begin｜>" in prompt and "<｜fim hole｜>" in prompt and "<｜fim end｜>" in prompt:
                        # FIM 형식의 특수 처리
                        completion = examples["completion"][i]
                        
                        # fim begin과 fim hole 사이의 텍스트를 추출 (prefix)
                        prefix = prompt.split("<｜fim hole｜>")[0].replace("<｜fim begin｜>", "")
                        # fim hole과 fim end 사이의 텍스트를 추출 (suffix)
                        suffix = prompt.split("<｜fim hole｜>")[1].split("<｜fim end｜>")[0]
                        
                        # FIM 형식의 프롬프트로 직접 변환
                        text = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{completion}"
                        texts[i] = text
            
            # 토크나이징
            model_inputs = self.tok(
                texts,
                truncation=True,
                padding=True,  # True로 변경하여 모든 시퀀스가 동일 길이를 갖도록 함
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
        """완전 호환 학습 인자 생성"""
        # 절대 경로 사용으로 변경 (AWS 스팟 인스턴스 중단 시 안전한 체크포인트 저장을 위함)
        # 학습 경로 설정 - 체크포인트는 원래 경로에 저장
        scripts_dir = "/home/ubuntu/deepseek-coder/scripts"
        
        # 학습 체크포인트와 출력 경로
        output_dirs = {
            "complete": f"{scripts_dir}/output/autocomplete-finetuned",
            "prompt": f"{scripts_dir}/output/prompt-finetuned",
            "comment": f"{scripts_dir}/output/comment-finetuned",
            "error_fix": f"{scripts_dir}/output/error-fix-finetuned"
        }
        
        # 최종 모델 경로 (추론용)
        models_dir = "../models"
        final_model_dirs = {
            "complete": f"{models_dir}/autocomplete-finetuned",
            "prompt": f"{models_dir}/prompt-finetuned",
            "comment": f"{models_dir}/comment-finetuned",
            "error_fix": f"{models_dir}/error-fix-finetuned"
        }
        
        # 학습 경로 (체크포인트와 출력 디렉토리)
        output_dir = output_dirs.get(self.mode, f"{scripts_dir}/output/prompt-finetuned")
        
        # 최종 모델 경로 (inference.py에서 사용할 경로) 별도 저장
        self.final_model_dir = final_model_dirs.get(self.mode, f"{models_dir}/prompt-finetuned")
        
        # 최소한의 안전한 인자만 사용 (타입 변환 추가)
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.get_int_config("batch_size", 4),  # config.yaml에서는 4
            per_device_eval_batch_size=self.get_int_config("batch_size", 4),  # config.yaml에서는 4
            gradient_accumulation_steps=self.get_int_config("grad_acc", 2),  # config.yaml에서는 2
            learning_rate=self.get_float_config("learning_rate", 2e-4),
            num_train_epochs=self.get_float_config("num_epochs", 3),
            warmup_ratio=self.get_float_config("warmup_ratio", 0.1),
            lr_scheduler_type=self.cfg["lr_scheduler"],
            weight_decay=self.get_float_config("weight_decay", 0.01),
            fp16=self.cfg["fp16"],
            bf16=self.cfg["bf16"],
            gradient_checkpointing=self.cfg["gradient_checkpointing"],
            dataloader_num_workers=self.get_int_config("dataloader_num_workers", 4),
            save_steps=self.get_int_config("save_steps", 10),  # config.yaml에서는 10
            eval_steps=self.get_int_config("eval_steps", 50),  # config.yaml에서는 50
            logging_steps=self.get_int_config("logging_steps", 5),  # config.yaml에서는 5
            save_total_limit=self.get_int_config("save_total_limit", 3),
            # 지원되지 않는 인자 제거: evaluation_strategy, save_strategy, load_best_model_at_end, metric_for_best_model
            remove_unused_columns=self.cfg["remove_unused_columns"],
            report_to="none",  # None이 아닌 "none"으로 설정하여 모든 로깅 비활성화
            run_name=f"deepseek-coder-{self.mode}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    def train(self):
        """모델 학습 실행"""
        log.info(f"🚀 학습 시작: {self.mode} 모드")
        
        # 학습 인자 준비
        args = self.t_args()
        
        # 체크포인트 감지 및 자동 재개 기능
        checkpoint_dir = args.output_dir
        resume_from_checkpoint = None
        
        # 기존 체크포인트 확인
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint-")]
            
            if checkpoints:
                # 체크포인트 번호 기준으로 정렬
                sorted_checkpoints = sorted(
                    checkpoints,
                    key=lambda x: int(x.split("-")[1]) if len(x.split("-")) > 1 and x.split("-")[1].isdigit() else 0
                )
                
                if sorted_checkpoints:
                    latest_checkpoint = sorted_checkpoints[-1]
                    resume_from_checkpoint = os.path.join(checkpoint_dir, latest_checkpoint)
                    log.info(f"🔄 체크포인트 감지: {resume_from_checkpoint}에서 학습 재개")
        
        # 데이터 콜레이터
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tok,
            mlm=False,  # Causal LM
            pad_to_multiple_of=8
        )
        
        # Trainer 설정 - resume_from_checkpoint는 train() 메서드에만 전달
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_ds,
            eval_dataset=self.eval_ds,
            data_collator=data_collator,
            tokenizer=self.tok
        )
        
        log.info("✅ Trainer 초기화 완료, 체크포인트 감지 활성화")
        
        # 학습 실행
        try:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            log.info("✅ 학습 완료")
            
            # 1. 최종 모델 저장 - args.output_dir 내의 final_model 경로에 저장
            output_model_path = os.path.join(args.output_dir, "final_model")
            trainer.save_model(output_model_path)
            self.tok.save_pretrained(output_model_path)
            
            log.info(f"✅ 최종 모델 저장 완료: {output_model_path}")
            
            # 2. inference.py 호환을 위해 모델을 ../models 경로로도 복사
            import shutil
            final_model_path = os.path.join(self.final_model_dir, "final_model")
            
            # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
            
            if os.path.exists(output_model_path) and output_model_path != final_model_path:
                try:
                    # 기존 디렉토리 삭제 후 복사
                    if os.path.exists(final_model_path):
                        shutil.rmtree(final_model_path)
                    
                    # 모델 디렉토리 복사
                    shutil.copytree(output_model_path, final_model_path)
                    log.info(f"✅ inference.py 호환을 위해 모델 복사 완료: {final_model_path}")
                except Exception as e:
                    log.error(f"❌ 모델 복사 중 오류 발생: {e}")
            
            return final_model_path
            
        except Exception as e:
            log.error(f"❌ 학습 중 오류 발생: {e}")
            raise
    
    def run(self):
        """전체 학습 파이프라인 실행"""
        try:
            self.load_model()
            self.load_data()
            final_model_path = self.train()
            
            log.info("🎉 학습 파이프라인 완료!")
            return final_model_path
            
        except Exception as e:
            log.error(f"❌ 학습 파이프라인 실패: {e}")
            raise

def parse_arguments():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description="DeepSeek-Coder 6.7B QLoRA 파인튜닝 (완전 호환성)")
    parser.add_argument("--config", type=str, default="config.yaml", help="설정 파일 경로")
    parser.add_argument("--mode", type=str, choices=["complete", "prompt", "comment", "error_fix"], default="prompt", help="학습 모드")
    return parser.parse_args()

def main():
    """메인 함수"""
    args = parse_arguments()
    
    mode_descriptions = {
        "complete": "[제1차] 코드 자동완성 전용 (FIM 형식)",
        "prompt": "[제2차] 일반 프롬프트 기반 코드 생성",
        "comment": "[제3차] 주석 기반 코드 생성",
        "error_fix": "[제4차] 오류 코드 설명 및 수정"
    }
    
    log.info("=" * 53)
    log.info("DeepSeek Coder 6.7B 완전 호환성 파인튜닝 시작")
    log.info("=" * 53)
    log.info(f"시작 시간: {datetime.now().strftime('%a %b %d %H:%M:%S %Z %Y')}")
    log.info(f"학습 모드: [{args.mode}] {mode_descriptions.get(args.mode, args.mode)}")
    
    # 환경 정보 출력
    log.info(f"Python 명령어: {sys.executable}")
    if torch.cuda.is_available():
        gpu_info = f"{torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory // 1024**2}, {torch.cuda.memory_allocated() // 1024**2}"
        log.info(f"GPU 정보:\n{gpu_info}")
    
    log.info(f"Python 버전: Python {sys.version.split()[0]}")
    
    # 패키지 버전 확인
    log.info("확인 중: 필요한 패키지들...")
    packages = ["torch", "transformers", "datasets", "accelerate", "peft"]
    for pkg in packages:
        try:
            module = __import__(pkg)
            version = getattr(module, "__version__", "unknown")
            log.info(f"✅ {pkg}: {version}")
        except ImportError:
            log.error(f"❌ {pkg}: 설치되지 않음")
    
    try:
        # 학습 실행
        trainer = DeepSeekTrainer(config_path=args.config, mode=args.mode)
        final_model_path = trainer.run()
        
        log.info("=" * 53)
        log.info("학습이 성공적으로 완료되었습니다!")
        log.info("=" * 53)
        log.info(f"모델 저장 위치: {final_model_path}")
        log.info(f"inference.py에서 이 모델을 사용하려면: python inference.py --mode {args.mode} --model {final_model_path}")
        
    except KeyboardInterrupt:
        log.info("⏹️ 사용자에 의해 학습이 중단되었습니다.")
    except Exception as e:
        log.error(f"❌ 학습 실패: {e}")
        sys.exit(1)
    finally:
        # 최종 GPU 상태 출력
        if torch.cuda.is_available():
            log.info("최종 GPU 상태:")
            os.system("nvidia-smi")

if __name__ == "__main__":
    main()