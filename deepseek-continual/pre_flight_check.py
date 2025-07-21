#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
실행 전 체크리스트 스크립트
Critical Issues 해결 상태 및 환경 준비 상태 검증

실행 방법:
    python pre_flight_check.py
"""

import os
import sys
import yaml
import subprocess
import torch
from pathlib import Path

def check_gpu_environment():
    """GPU 환경 체크"""
    print("🔍 GPU 환경 체크")
    
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다.")
        return False
    
    # GPU 정보
    props = torch.cuda.get_device_properties(0)
    total_mem_gb = props.total_memory / (1024**3)
    
    print(f"✅ GPU: {props.name}")
    print(f"✅ CUDA 버전: {torch.version.cuda}")
    print(f"✅ PyTorch 버전: {torch.__version__}")
    print(f"✅ 총 메모리: {total_mem_gb:.1f}GB")
    
    # A10G 환경 체크
    if "A10G" in props.name or total_mem_gb > 20:
        print("✅ A10G 22GB 환경 감지됨")
    else:
        print(f"⚠️  A10G가 아닌 GPU 감지: {props.name} ({total_mem_gb:.1f}GB)")
        print("💡 메모리 부족 시 배치 크기를 1로, max_length를 256으로 설정하세요.")
    
    return True

def check_dependencies():
    """의존성 체크"""
    print("\n🔍 의존성 체크")
    
    required_packages = {
        'torch': '2.0.0',
        'transformers': '4.30.0',
        'datasets': '2.10.0',
        'peft': '0.4.0',
        'accelerate': '0.20.0',
        'bitsandbytes': '0.39.0'
    }
    
    missing_packages = []
    
    for package, min_version in required_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: 설치되지 않음")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n💡 설치 명령어:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_config_file():
    """config.yaml 파일 체크"""
    print("\n🔍 config.yaml 체크")
    
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ config.yaml 파일을 찾을 수 없습니다: {config_path}")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Critical Issues 해결 상태 체크
        batch_size = config.get('batch_size', 1)
        max_length = config.get('max_length', 1024)
        gpu_memory_fraction = config.get('gpu_memory_fraction', 0.8)
        
        print(f"📊 배치 크기: {batch_size}")
        print(f"📊 최대 길이: {max_length}")
        print(f"📊 GPU 메모리 제한: {gpu_memory_fraction*100:.0f}%")
        
        # Critical Issue #3 체크
        if batch_size < 2:
            print("⚠️  경고: 배치 크기가 2 미만입니다. BatchNorm/LayerNorm 불안정 가능성")
            print("💡 추천: batch_size를 2 이상으로 설정")
        else:
            print("✅ 배치 크기 설정 양호")
        
        if max_length > 512:
            print("⚠️  경고: max_length가 512보다 큽니다. 메모리 부족 가능성")
            print("💡 추천: max_length를 512 이하로 설정")
        else:
            print("✅ 시퀀스 길이 설정 양호")
        
        return True
        
    except Exception as e:
        print(f"❌ config.yaml 파일 읽기 실패: {e}")
        return False

def check_data_files():
    """데이터 파일 체크"""
    print("\n🔍 데이터 파일 체크")
    
    data_paths = [
        os.path.expanduser("~/deepseek-continual/data/train.jsonl"),
        "data/train.jsonl",
        "/home/ubuntu/deepseek-continual/data/train.jsonl"
    ]
    
    data_found = False
    for data_path in data_paths:
        if os.path.exists(data_path):
            file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
            print(f"✅ 데이터 파일 발견: {data_path} ({file_size_mb:.1f}MB)")
            
            # 샘플 라인 수 체크
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                print(f"📊 총 샘플 수: {line_count:,}개")
                data_found = True
                break
            except Exception as e:
                print(f"⚠️  파일 읽기 오류: {e}")
    
    if not data_found:
        print("❌ 학습 데이터 파일을 찾을 수 없습니다.")
        print("💡 다음 경로 중 하나에 train.jsonl 파일을 배치하세요:")
        for path in data_paths:
            print(f"  - {path}")
        return False
    
    return True

def check_output_directories():
    """출력 디렉토리 체크"""
    print("\n🔍 출력 디렉토리 체크")
    
    output_dirs = [
        "checkpoints",
        "../logs",
        "../models"
    ]
    
    for dir_path in output_dirs:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"✅ 디렉토리 생성: {dir_path}")
            except Exception as e:
                print(f"❌ 디렉토리 생성 실패: {dir_path} - {e}")
                return False
        else:
            print(f"✅ 디렉토리 존재: {dir_path}")
    
    return True

def check_memory_settings():
    """메모리 설정 체크"""
    print("\n🔍 메모리 설정 체크")
    
    # CUDA 메모리 설정 확인
    cuda_alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
    
    if 'expandable_segments:True' in cuda_alloc_conf:
        print("✅ CUDA 메모리 단편화 방지 설정 활성화됨")
    else:
        print("⚠️  CUDA 메모리 단편화 방지 설정이 없습니다.")
        print("💡 환경변수 설정: export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128'")
    
    # GPU 메모리 사용량 체크
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
        
        if allocated_gb > 1.0:
            print(f"⚠️  GPU 메모리가 이미 {allocated_gb:.1f}GB 사용 중입니다.")
            print("💡 다른 GPU 프로세스를 종료하는 것을 권장합니다.")
        else:
            print("✅ GPU 메모리 사용량 양호")
    
    return True

def run_all_checks():
    """모든 체크 실행"""
    print("🚀 Continual Learning 실행 전 체크리스트")
    print("=" * 50)
    
    checks = [
        ("GPU 환경", check_gpu_environment),
        ("의존성", check_dependencies),
        ("Config 파일", check_config_file),
        ("데이터 파일", check_data_files),
        ("출력 디렉토리", check_output_directories),
        ("메모리 설정", check_memory_settings)
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed_checks += 1
        except Exception as e:
            print(f"❌ {check_name} 체크 중 오류: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 체크 결과: {passed_checks}/{total_checks} 통과")
    
    if passed_checks == total_checks:
        print("🎉 모든 체크 통과! 학습을 시작할 수 있습니다.")
        print("\n💡 권장 실행 순서:")
        print("1. 소규모 테스트: python test_small_batch.py --samples 100")
        print("2. 중간 테스트: python test_small_batch.py --samples 1000")
        print("3. 전체 학습: ./cc_run_training.sh comment")
        return True
    else:
        print("❌ 일부 체크가 실패했습니다. 위의 문제점들을 해결한 후 다시 시도하세요.")
        return False

def main():
    success = run_all_checks()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
