import json
import itertools
import random

def build_system_prompt(persona):
    """템플릿 구성요소로 시스템 프롬프트 생성"""
    return " ".join([
        persona["role_context"],
        persona["tone_context"],
        persona["task_instruction"],
        persona["output_requirement"]
    ])

def generate_training_samples(code_samples, template_config, output_path):
    """파인튜닝 데이터 생성"""
    # 템플릿 설정 로드
    personas = template_config["personas"]
    
    training_data = []
    
    # 각 코드 샘플에 대해
    for code in code_samples:
        # 랜덤 페르소나 선택
        persona = random.choice(personas)
        
        # 수식어 랜덤 선택 (옵션)
        code_output = random.choice(template_config["code_output_structure"])
        explanation = random.choice(template_config["explanation_style"])
        
        # 시스템 프롬프트 생성
        system_prompt = build_system_prompt(persona)
        
        # 훈련 데이터 레코드 생성 (Qwen 형식)
        record = {
            "messages": [
                {
                    "role": "system", 
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": code["input"]
                },
                {
                    "role": "assistant",
                    "content": code["output"]
                }
            ],
            "meta": {
                "code_output_structure": code_output,
                "explanation_style": explanation
            }
        }
        
        training_data.append(record)
    
    # 저장
    with open(output_path, "w", encoding="utf-8") as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")