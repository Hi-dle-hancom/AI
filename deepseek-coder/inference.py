#!/usr/bin/env python3
"""
DeepSeek-Coder 실시간 코드 어시스트 인퍼런스 스크립트 (T4 환경 최적화)
지원 기능:
- 사용자 입력 기반 프럼프트 응답
- 오류 수정 및 작성 지원
- 코드 자동완성
- T4 GPU 전용 최적화
"""

import os
import sys
import json
import argparse
import torch
import readline
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

# ────────── T4 환경 최적화 설정 ──────────
MODEL_CONFIG = {
    "model_id": "deepseek-ai/deepseek-coder-6.7b-instruct",  # 기본 모델
    "model_path": "../models/prompt-finetuned/final_model",  # train.py [2차] 학습 출력 경로
    "device": "cuda" if torch.cuda.is_available() else "cpu", 
    "dtype": torch.float16,  # T4에서 bfloat16 대신 float16 사용
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.95,
    "load_in_4bit": True,  # T4 메모리 효율성을 위한 4-bit 양자화
}

def check_gpu_compatibility():
    """GPU 호환성 확인"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        compute_capability = torch.cuda.get_device_capability(0)
        print(f"🖥️ GPU: {gpu_name}")
        print(f"🔧 Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
        
        # bfloat16 지원 확인
        if torch.cuda.is_bf16_supported():
            print("✅ bfloat16 지원됨")
        else:
            print("⚠️ bfloat16 미지원 - float16 사용")
            
        return True
    else:
        print("❌ CUDA 사용 불가 - CPU 모드")
        return False

class CodeAssistant:
    """코드 어시스트 추론 클래스 (T4 최적화)"""
    
    def __init__(self, config=None, mode="code-gen"):
        self.config = config or MODEL_CONFIG
        self.mode = mode
        
        # GPU 호환성 확인
        self.cuda_available = check_gpu_compatibility()
        
        # T4 환경에서 dtype 조정
        if self.cuda_available:
            # T4에서는 bfloat16 대신 float16 사용
            self.config["dtype"] = torch.float16
        else:
            self.config["dtype"] = torch.float32
            self.config["device"] = "cpu"
        
        # 모드에 따라 다른 모델 선택
        model_path = self._get_model_path()
        self.config["model_path"] = model_path
        
        self.model = None
        self.tokenizer = None
        self.load_model()
        
    def _get_model_path(self):
        """모드에 따른 모델 경로 결정"""
        if self.mode == "autocomplete":
            model_path = "../models/autocomplete-finetuned/final_model"
            if not os.path.exists(model_path):
                print(f"\n⚠️ [제1차] 자동완성 전용 모델을 찾을 수 없어 기본 모델을 사용합니다.")
                model_path = self.config["model_path"]
        elif self.mode == "comment":
            model_path = "../models/comment-finetuned/final_model"
            if not os.path.exists(model_path):
                print(f"\n⚠️ [제3차] 주석 기반 코드 생성 모델을 찾을 수 없어 기본 모델을 사용합니다.")
                model_path = self.config["model_path"]
        elif self.mode == "error_fix":
            model_path = "../models/error-fix-finetuned/final_model"
            if not os.path.exists(model_path):
                print(f"\n⚠️ [제4차] 오류 코드 수정 전용 모델을 찾을 수 없어 기본 모델을 사용합니다.")
                model_path = self.config["model_path"]
        else:  # prompt 모드(기본값)
            model_path = self.config["model_path"]  # prompt 모델이 기본값
        
        return model_path
        
    def load_model(self):
        """T4 환경 최적화된 모델과 토크나이저 로딩"""
        print(f"\n📂 모델 로딩 시작 (T4 최적화)")
        model_path = self.config.get('model_path')
        model_id = self.config.get('model_id')
        
        # 로컬 파인튜닝 모델이 있으면 먼저 시도
        try_path = model_path if os.path.exists(model_path) else model_id
        print(f"🔍 모델 경로: {try_path}")
        print(f"🖥️ 디바이스: {self.config['device']}")
        print(f"🔢 데이터 타입: {self.config['dtype']}")
        
        try:
            # 토크나이저 로딩
            self.tokenizer = AutoTokenizer.from_pretrained(
                try_path, 
                trust_remote_code=True
            )
            
            # T4 환경 최적화된 모델 로딩
            if self.cuda_available and self.config.get("load_in_4bit", False):
                # 4-bit 양자화로 메모리 효율성 향상
                from transformers import BitsAndBytesConfig
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,  # T4 호환
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    try_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                print("✅ 4-bit 양자화 모델 로딩 완료")
            else:
                # 일반 로딩
                self.model = AutoModelForCausalLM.from_pretrained(
                    try_path,
                    torch_dtype=self.config['dtype'],
                    device_map="auto" if self.config['device'] == "cuda" else None,
                    trust_remote_code=True
                )
                
                # CPU의 경우 명시적 이동
                if self.config['device'] == "cpu":
                    self.model = self.model.to("cpu")
                    
            self.model.eval()
            
            # 메모리 사용량 확인
            if self.cuda_available:
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"📊 GPU 메모리 사용량: {memory_used:.2f}GB / {memory_total:.2f}GB")
            
            print("✅ 모델 로딩 완료")
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            print("💡 해결 방안:")
            print("   1. config.yaml에서 bnb_4bit_compute_dtype: float16으로 설정")
            print("   2. torch_dtype: float16으로 설정")
            print("   3. 메모리 부족 시 per_device_train_batch_size 줄이기")
            sys.exit(1)
    
    def get_int_config(self, key, default=0):
        """config에서 값을 가져와 int로 변환하는 헬퍼 함수"""
        value = self.config.get(key, default)
        if isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                print(f"⚠️ 경고: {key} 값 '{value}'을 정수로 변환할 수 없습니다. 기본값 {default} 사용")
                value = default
        return value
    
    def get_float_config(self, key, default=0.0):
        """config에서 값을 가져와 float로 변환하는 헬퍼 함수"""
        value = self.config.get(key, default)
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                print(f"⚠️ 경고: {key} 값 '{value}'을 실수로 변환할 수 없습니다. 기본값 {default} 사용")
                value = default
        return value
    
    def build_prompt(self, user_input, mode="prompt"):
        """ChatML 형식으로 프럼프트 구성 (다양한 데이터 구조 지원)"""
        system_msg = "You are an expert coding assistant."
        
        # 다양한 입력 형식 처리
        if isinstance(user_input, dict):
            # 1. 채팅 형식 (messages 구조)
            if "messages" in user_input:
                messages = user_input["messages"]
                chat_text = ""
                
                # 시스템 메시지 추가
                chat_text += f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                
                # 모든 메시지를 ChatML 형식으로 변환
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    chat_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                
                # 어시스턴트 응답 시작 태그 추가
                chat_text += "<|im_start|>assistant\n"
                return chat_text
            
            # 2. 주석 키워드 기반 코드 생성 (prompt-completion)
            elif "prompt" in user_input:
                prompt = user_input["prompt"]
                
                # FIM 형식 특수 처리
                if "<｜fim begin｜>" in prompt and "<｜fim hole｜>" in prompt and "<｜fim end｜>" in prompt:
                    # fim begin과 fim hole 사이의 텍스트를 추출 (prefix)
                    prefix = prompt.split("<｜fim hole｜>")[0].replace("<｜fim begin｜>", "")
                    # fim hole과 fim end 사이의 텍스트를 추출 (suffix)
                    suffix = prompt.split("<｜fim hole｜>")[1].split("<｜fim end｜>")[0]
                    
                    # FIM 스페셜 토큰 형식으로 구성
                    return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|"
                else:
                    # 일반 프롬프트 처리
                    return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # 3. 에러 설명 형식
            elif "error_context" in user_input and "explanation" in user_input.get("error_context", {}):
                error_context = user_input["error_context"]
                error_log = error_context.get("error_log", "")
                code_snippet = error_context.get("code_snippet", "")
                language = error_context.get("language", "")
                
                user_text = f"다음 {language} 코드의 에러를 설명해주세요:\n\n에러 로그:\n{error_log}\n\n코드:\n{code_snippet}"
                return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
            
            # 4. 에러 수정 형식
            elif "error_context" in user_input and "buggy_code_snippet" in user_input:
                error_context = user_input["error_context"]
                buggy_code = user_input["buggy_code_snippet"]
                error_log = error_context.get("error_log", "")
                language = error_context.get("language", "")
                
                user_text = f"다음 {language} 코드의 에러를 수정해주세요:\n\n에러 로그:\n{error_log}\n\n코드:\n{buggy_code}"
                return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
            
            # 5. 기존 instruction/input 형식
            elif "instruction" in user_input:
                instruction = user_input.get("instruction", "").strip()
                input_text = user_input.get("input", "").strip()
                
                # instruction과 input을 합쳐서 user 메시지로 구성
                if input_text:
                    user_text = f"Instruction: {instruction}\nInput: {input_text}"
                else:
                    user_text = f"Instruction: {instruction}"
                
                # ChatML 형식 프럼프트
                if mode == "error_fix":
                    return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\nPlease fix the following code:\n\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
                elif mode == "autocomplete":
                    return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\nComplete the following code:\n\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
                elif mode == "comment":
                    return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\nGenerate code based on this comment:\n\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
                else:  # prompt
                    return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
            
            # 기타 딕셔너리 형태의 입력
            else:
                user_text = json.dumps(user_input, ensure_ascii=False)
        else:
            # 기존 문자열 입력 처리 (하위 호환성)
            user_text = str(user_input)
        
        # 기본 ChatML 형식 프럼프트
        if mode == "error_fix":
            return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\nPlease fix the following code:\n\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
        elif mode == "autocomplete":
            return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\nComplete the following code:\n\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
        elif mode == "comment":
            return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\nGenerate code based on this comment:\n\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
        else:  # prompt
            return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    
    def generate(self, user_input, mode="prompt"):
        """T4 최적화된 텍스트 생성"""
        # 다양한 데이터 형식 확인
        is_fim_format = False
        if isinstance(user_input, dict) and "prompt" in user_input:
            prompt_str = user_input["prompt"]
            if "<｜fim begin｜>" in prompt_str and "<｜fim hole｜>" in prompt_str and "<｜fim end｜>" in prompt_str:
                is_fim_format = True
                
        # 프롬프트 구성
        prompt = self.build_prompt(user_input, mode)
        
        # T4 메모리 제한을 고려한 토큰 길이 조정
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        if self.config['device'] == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # 모드별 생성 파라미터 조정
        if is_fim_format:
            # FIM 용 특수 설정
            max_new_tokens = min(512, self.get_int_config('max_tokens', 512))
            temp = 0.2  # 더 결정적으로
        elif mode == "autocomplete":
            max_new_tokens = min(128, self.get_int_config('max_tokens', 512))  # 자동완성은 짧게
            temp = 0.3  # 더 결정적으로
        elif mode == "comment":
            max_new_tokens = min(256, self.get_int_config('max_tokens', 512))
            temp = self.get_float_config('temperature', 0.6)
        else:  # prompt, error_fix
            max_new_tokens = self.get_int_config('max_tokens', 512)
            temp = self.get_float_config('temperature', 0.6)
            
        with torch.no_grad():
            # T4 메모리 효율성을 위한 생성 설정
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(mode != "autocomplete" and not is_fim_format),
                top_p=self.get_float_config('top_p', 0.95),
                temperature=temp,
                top_k=self.get_int_config('top_k', 50),
                repetition_penalty=self.get_float_config('repetition_penalty', 1.1),
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,  # 캐시 사용으로 속도 향상
            )
        
        # 프롬프트 길이 이후 응답 부분만 추출
        input_length = inputs['input_ids'].shape[1]
        generated = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        # 생성된 텍스트 처리 및 반환
        return self._clean_output(generated)
    
    def _clean_output(self, text):
        """생성된 텍스트에서 ChatML 태그 및 불필요한 부분 제거"""
        # ChatML 태그 제거
        text = text.replace("<|im_start|>assistant\n", "").replace("<|im_end|>", "")
        
        # FIM 태그 처리 (만약 있을 경우)
        if "<|fim_middle|>" in text:
            text = text.replace("<|fim_middle|>", "")
        if "<|fim_prefix|>" in text:
            text = text.replace("<|fim_prefix|>", "")
        if "<|fim_suffix|>" in text:
            text = text.replace("<|fim_suffix|>", "")
        
        # 불필요한 반복 제거
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip() and not line.startswith('<|'):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()

def parse_arguments():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description="DeepSeek-Coder 코드 어시스트 추론 (T4 최적화)")
    parser.add_argument(
        "--model", 
        type=str, 
        default="../models/prompt-finetuned/final_model", 
        help="모델 경로 또는 모델 ID"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["prompt", "autocomplete", "comment", "error_fix"], 
        default="prompt",
        help="코드 생성 모드"
    )
    parser.add_argument(
        "--temp", 
        type=float, 
        default=0.7,
        help="온도 설정"
    )
    parser.add_argument(
        "--no-4bit", 
        action="store_true",
        help="4-bit 양자화 비활성화"
    )
    return parser.parse_args()

def main():
    """메인 함수 (Instruction, Input 지원)"""
    args = parse_arguments()
    
    # 설정 구성
    config = MODEL_CONFIG.copy()
    config["temperature"] = args.temp
    config["model_path"] = args.model
    config["load_in_4bit"] = not args.no_4bit
    
    mode_names = {
        "prompt": "[제2차] 일반 프롬프트 기반 코드 생성",
        "autocomplete": "[제1차] 코드 자동완성", 
        "comment": "[제3차] 주석 기반 코드 생성",
        "error_fix": "[제4차] 오류 코드 설명 및 수정"
    }
    
    print(f"\n🤖 DeepSeek-Coder 코드 어시스트 ({mode_names.get(args.mode, args.mode)})")
    print("📝 종료하려면 'exit' 또는 'quit'을 입력하세요.")
    print("💡 T4 환경에 최적화된 설정으로 실행됩니다.")
    print("💡 Instruction과 Input을 분리해서 입력하려면 'instruction:' 형식을 사용하세요.\n")
    
    # 코드 어시스트 초기화
    assistant = CodeAssistant(config, mode=args.mode)
    
    # REPL 루프
    while True:
        try:
            prompts = {
                "comment": "💬 주석 입력 (코드 생성): ",
                "autocomplete": "💬 코드 입력 (자동완성): ",
                "error_fix": "💬 오류 코드 입력 (수정): ",
                "prompt": "💬 프롬프트 (instruction: 형식 가능): "
            }
            
            user_input = input(prompts.get(args.mode, prompts["prompt"]))
                
            if user_input.lower() in ["exit", "quit"]:
                print("👋 코드 어시스트를 종료합니다.")
                break
                
            if not user_input.strip():
                continue
            
            # instruction: 형식 파싱
            parsed_input = user_input
            if "instruction:" in user_input.lower():
                try:
                    parts = user_input.split("instruction:", 1)[1].strip()
                    if "input:" in parts.lower():
                        instruction_part, input_part = parts.split("input:", 1)
                        parsed_input = {
                            "instruction": instruction_part.strip(),
                            "input": input_part.strip()
                        }
                    else:
                        parsed_input = {
                            "instruction": parts.strip(),
                            "input": ""
                        }
                except:
                    # 파싱 실패 시 원본 사용
                    parsed_input = user_input
                
            print("\n⏳ 생성 중...")
            output = assistant.generate(parsed_input, mode=args.mode)
            print(f"\n📄 출력:\n{output}\n")
            
        except KeyboardInterrupt:
            print("\n👋 코드 어시스트를 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            if "out of memory" in str(e).lower():
                print("💡 메모리 부족 - 더 짧은 입력을 시도하거나 --no-4bit 옵션을 제거하세요.")

if __name__ == "__main__":
    main()