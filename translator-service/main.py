# translator-service/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import re

# --- 모델 및 토크나이저 전역 변수 ---
model = None
tokenizer = None

# --- FastAPI 앱 초기화 ---
app = FastAPI(title="Translation Service", version="1.0.0")

# --- Pydantic 모델 ---
class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translated_text: str

# --- 서버 시작 이벤트 핸들러 ---
@app.on_event("startup")
def load_model():
    global model, tokenizer
    print("번역 모델 로딩 시작...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model_name = "Qwen/Qwen2.5-Coder-7B"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    # ✨ 1. 토크나이저에 pad_token 추가 (경고 메시지 제거)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ 번역 모델 로딩 완료.")

# --- API 엔드포인트 ---
@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="모델이 아직 초기화되지 않았습니다.")
    
    try:
        # ✨ 2. 모델이 번역에만 집중하도록 프롬프트 강화
        # - 역할과 임무를 명확히 하고, 코드 생성을 금지시킴
        # - 출력 형식을 지정하여 파싱이 쉽도록 유도
        prompt = f"""
[TASK]
You are a machine translator. Your ONLY job is to translate the given Korean text into English.
DO NOT generate code. DO NOT add explanations.
Translate the following text accurately for a software development context.

[KOREAN TEXT]
{request.text}

[ENGLISH TRANSLATION]
"""
        
        # ✨ 3. attention_mask를 함께 전달 (경고 메시지 제거)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150, # 번역이므로 최대 길이 제한
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id, # pad_token_id 명시
            eos_token_id=tokenizer.eos_token_id
        )

        # 입력 부분을 제외하고 생성된 텍스트만 추출
        generated_text_only = tokenizer.decode(output_sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # ✨ 4. 불필요한 앞뒤 공백 및 줄바꿈 제거
        # - 모델이 추가적인 내용을 생성했더라도 첫 줄만 사용하도록 처리
        clean_text = generated_text_only.strip()
        first_line = clean_text.split('\n')[0]

        return TranslationResponse(translated_text=first_line)

    except Exception as e:
        print(f"번역 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))