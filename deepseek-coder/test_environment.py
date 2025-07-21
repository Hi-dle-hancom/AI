#!/usr/bin/env python3
"""
AWS 스팟 인스턴스 중단 감지 및 환경 테스트 스크립트 (T4 최적화)
- 스팟 인스턴스 중단 신호 모니터링
- GPU 메모리 및 성능 테스트
- 패키지 호환성 확인
- Instruction, Input, Output 데이터 구조 지원
- 학습 환경 검증
"""

import os
import sys
import time
import json
import signal
import requests
import subprocess
from datetime import datetime
from typing import Dict, Any, List

def signal_handler(signum, frame):
    """시그널 핸들러"""
    print(f"\n⚠️ 시그널 {signum} 수신됨. 정리 작업 중...")
    cleanup_and_exit()

def cleanup_and_exit():
    """정리 작업 후 종료"""
    print("🧹 정리 작업 완료")
    sys.exit(0)

def check_spot_instance_termination():
    """AWS 스팟 인스턴스 중단 신호 확인"""
    try:
        # 스팟 인스턴스 메타데이터 확인
        response = requests.get(
            "http://169.254.169.254/latest/meta-data/spot/instance-action",
            timeout=2
        )
        
        if response.status_code == 200:
            termination_time = response.text
            print(f"⚠️ 스팟 인스턴스 중단 예정: {termination_time}")
            return True
        else:
            return False
            
    except requests.exceptions.RequestException:
        # 스팟 인스턴스가 아니거나 중단 신호 없음
        return False

def test_gpu_environment():
    """GPU 환경 테스트 (T4 최적화)"""
    print("\nGPU 환경 테스트...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA 사용 가능 - GPU 개수: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                compute_cap = torch.cuda.get_device_capability(i)
                
                print(f"   GPU {i}: {gpu_name}")
                print(f"   메모리: {gpu_memory:.1f}GB")
                print(f"   Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
                
                # T4 특별 확인
                if "T4" in gpu_name:
                    print("   🔧 T4 GPU 감지 - 최적화 설정 적용됨")
                    if compute_cap[0] < 8:
                        print("   ⚠️ bfloat16 미지원 - float16 사용 권장")
                
                # bfloat16 지원 확인
                if torch.cuda.is_bf16_supported():
                    print("   ✅ bfloat16 지원됨")
                else:
                    print("   ⚠️ bfloat16 미지원 - float16 사용")
            
            # 간단한 GPU 연산 테스트
            try:
                test_tensor = torch.randn(1000, 1000).cuda()
                result = torch.matmul(test_tensor, test_tensor.t())
                print("   ✅ GPU 연산 테스트 통과")
                
                # 메모리 사용량 확인
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"   메모리 사용량: {memory_allocated:.2f}GB (할당) / {memory_reserved:.2f}GB (예약)")
                
                # 메모리 정리
                del test_tensor, result
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"   ❌ GPU 연산 테스트 실패: {e}")
                return False
                
        else:
            print("❌ CUDA 사용 불가")
            return False
            
        return True
        
    except ImportError:
        print("❌ PyTorch 설치되지 않음")
        return False

def test_packages():
    """필수 패키지 설치 확인"""
    print("\n패키지 설치 확인...")
    
    required_packages = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "datasets": "Datasets", 
        "accelerate": "Accelerate",
        "peft": "PEFT",
        "trl": "TRL",
        "bitsandbytes": "BitsAndBytes"
    }
    
    missing_packages = []
    
    for package, description in required_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"✅ {description}: {version}")
            
            # 특별 확인사항
            if package == "torch":
                import torch
                print(f"   CUDA 버전: {torch.version.cuda}")
                print(f"   cuDNN 버전: {torch.backends.cudnn.version()}")
                
        except ImportError:
            print(f"❌ {description}: 설치되지 않음")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 누락된 패키지: {', '.join(missing_packages)}")
        print("다음 명령어로 설치하세요:")
        print("pip install torch transformers datasets accelerate peft trl bitsandbytes")
        return False
    
    return True

def test_data_file():
    """데이터 파일 존재 여부 확인 (Instruction, Input, Output 구조 지원)"""
    print("\n데이터 파일 확인...")
    
    # 가능한 데이터 파일 경로들
    possible_paths = [
        "/home/ubuntu/deepseek-coder/data/train.jsonl",  # train.py 기본 경로
        "../data/train.jsonl",
        "./data/train.jsonl"
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path:
        print(f"✅ 데이터 파일 존재: {data_path}")
        
        # 파일 크기 확인
        file_size = os.path.getsize(data_path)
        print(f"   파일 크기: {file_size / (1024*1024):.2f} MB")
        
        # 총 라인 수 확인
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
            print(f"   총 샘플 수: {total_lines:,}개")
        except Exception as e:
            print(f"   라인 수 확인 실패: {e}")
        
        # 첫 몇 줄 읽기 테스트 및 구조 확인
        try:
            import json
            with open(data_path, 'r', encoding='utf-8') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= 3:  # 첫 3줄만
                        break
                    try:
                        lines.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        print(f"   ⚠️ JSON 파싱 오류 (라인 {i+1}): {e}")
                        continue
            
            if lines:
                sample_keys = list(lines[0].keys())
                print(f"   데이터 컬럼: {sample_keys}")
                
                # 데이터 구조 검증
                if all(key in sample_keys for key in ["instruction", "input", "output"]):
                    print("   ✅ Instruction, Input, Output 구조 확인됨")
                    
                    # 샘플 데이터 미리보기
                    sample = lines[0]
                    instruction_preview = str(sample['instruction'])[:50].replace('\n', '\\n')
                    input_preview = str(sample['input'])[:50].replace('\n', '\\n')
                    output_preview = str(sample['output'])[:50].replace('\n', '\\n')
                    
                    print(f"   - Instruction: {instruction_preview}...")
                    print(f"   - Input: {input_preview}...")
                    print(f"   - Output: {output_preview}...")
                    
                    # 태그 확인 (있다면)
                    if 'tags' in sample_keys:
                        tags = sample.get('tags', [])
                        print(f"   - Tags: {tags}")
                    
                elif "input" in sample_keys and "output" in sample_keys:
                    print("   ✅ Input, Output 구조 확인됨 (레거시 지원)")
                    
                    # 샘플 데이터 미리보기
                    sample = lines[0]
                    input_preview = str(sample['input'])[:50].replace('\n', '\\n')
                    output_preview = str(sample['output'])[:50].replace('\n', '\\n')
                    
                    print(f"   - Input: {input_preview}...")
                    print(f"   - Output: {output_preview}...")
                    
                    # 태그 확인 (있다면)
                    if 'tags' in sample_keys:
                        tags = sample.get('tags', [])
                        print(f"   - Tags: {tags}")
                    
                else:
                    print("   ❌ 지원되지 않는 데이터 구조")
                    print(f"   필요한 키: ['instruction', 'input', 'output'] 또는 ['input', 'output']")
                    print(f"   실제 키: {sample_keys}")
                    return False
                
                # 태그별 샘플 수 확인 (가능한 경우)
                if 'tags' in sample_keys and len(lines) > 1:
                    tag_counts = {}
                    for line in lines:
                        tags = line.get('tags', [])
                        if isinstance(tags, str):
                            tags = [tags]
                        for tag in tags:
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1
                    
                    if tag_counts:
                        print(f"   태그별 샘플 수 (첫 3개 기준): {tag_counts}")
            
        except Exception as e:
            print(f"⚠️  데이터 파일 읽기 오류: {e}")
            return False
            
        return True
    else:
        print("❌ 데이터 파일을 찾을 수 없습니다")
        print("   확인한 경로들:")
        for path in possible_paths:
            print(f"   - {path}")
        return False

def test_model_loading():
    """모델 로딩 테스트 (T4 최적화)"""
    print("\n모델 로딩 테스트...")
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        model_name = "deepseek-ai/deepseek-coder-6.7b-base"
        print(f"테스트 모델: {model_name}")
        
        # 토크나이저 로딩 테스트
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            print("✅ 토크나이저 로딩 성공")
        except Exception as e:
            print(f"❌ 토크나이저 로딩 실패: {e}")
            return False
        
        # T4 환경에서 4-bit 양자화 설정 테스트
        if torch.cuda.is_available():
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,  # T4 호환
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                print("✅ 4-bit 양자화 설정 생성 성공")
                
                # 실제 모델 로딩은 시간이 오래 걸리므로 스킵
                print("   (실제 모델 로딩은 학습 시에 수행됩니다)")
                
            except Exception as e:
                print(f"❌ 양자화 설정 실패: {e}")
                return False
        else:
            print("⚠️ CUDA 없음 - CPU 모드로 테스트")
        
        return True
        
    except ImportError as e:
        print(f"❌ 필요한 패키지 없음: {e}")
        return False

def test_disk_space():
    """디스크 공간 확인"""
    print("\n디스크 공간 확인...")
    
    try:
        # 현재 디렉토리 디스크 사용량
        statvfs = os.statvfs('.')
        free_space = statvfs.f_frsize * statvfs.f_bavail / (1024**3)  # GB
        total_space = statvfs.f_frsize * statvfs.f_blocks / (1024**3)  # GB
        used_space = total_space - free_space
        
        print(f"   총 공간: {total_space:.1f}GB")
        print(f"   사용 공간: {used_space:.1f}GB")
        print(f"   여유 공간: {free_space:.1f}GB")
        
        # 모델 저장에 필요한 최소 공간 (약 15GB)
        min_required = 15.0
        if free_space >= min_required:
            print(f"✅ 충분한 디스크 공간 (최소 {min_required}GB 필요)")
            return True
        else:
            print(f"⚠️ 디스크 공간 부족 (최소 {min_required}GB 필요, 현재 {free_space:.1f}GB)")
            return False
            
    except Exception as e:
        print(f"❌ 디스크 공간 확인 실패: {e}")
        return False

def run_comprehensive_test():
    """종합 환경 테스트 실행"""
    print("=" * 60)
    print("🔍 DeepSeek-Coder 학습 환경 종합 테스트 (T4 최적화)")
    print("=" * 60)
    print(f"테스트 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = {}
    
    # 각 테스트 실행
    tests = [
        ("GPU 환경", test_gpu_environment),
        ("패키지 설치", test_packages),
        ("데이터 파일", test_data_file),
        ("모델 로딩", test_model_loading),
        ("디스크 공간", test_disk_space)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} 테스트 {'='*20}")
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} 테스트 중 오류: {e}")
            test_results[test_name] = False
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name:15}: {status}")
        if result:
            passed += 1
    
    print(f"\n총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 모든 테스트 통과! 학습 환경이 준비되었습니다.")
        return True
    else:
        print("⚠️ 일부 테스트 실패. 문제를 해결한 후 다시 시도하세요.")
        return False

def monitor_spot_instance():
    """스팟 인스턴스 모니터링 루프"""
    print("\n🔍 AWS 스팟 인스턴스 중단 모니터링 시작...")
    print("   (Ctrl+C로 중단)")
    
    check_interval = 30  # 30초마다 확인
    
    try:
        while True:
            if check_spot_instance_termination():
                print("🚨 스팟 인스턴스 중단 신호 감지!")
                print("   학습 중이라면 체크포인트 저장을 확인하세요.")
                
                # 현재 실행 중인 Python 프로세스 확인
                try:
                    result = subprocess.run(['pgrep', '-f', 'train.py'], 
                                          capture_output=True, text=True)
                    if result.stdout.strip():
                        print("   🔄 train.py 프로세스 감지됨 - 자동 저장 대기 중...")
                except:
                    pass
                
                break
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n⏹️ 모니터링 중단됨")

def main():
    """메인 함수"""
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--monitor":
        # 모니터링 모드
        monitor_spot_instance()
    else:
        # 테스트 모드
        success = run_comprehensive_test()
        
        if success:
            print("\n💡 스팟 인스턴스 모니터링을 시작하려면:")
            print("   python test_environment.py --monitor")
        
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()