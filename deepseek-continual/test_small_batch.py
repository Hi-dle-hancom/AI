#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
소규모 테스트 스크립트 - Critical Issues 해결 검증용
A10G 22GB 환경에서 안전한 단계적 테스트를 위한 스크립트

실행 방법:
    python test_small_batch.py --samples 100
    python test_small_batch.py --samples 1000
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path

def create_test_data(input_file, output_file, num_samples):
    """원본 데이터에서 지정된 수만큼 샘플 추출"""
    print(f"📝 테스트 데이터 생성: {num_samples}개 샘플")
    
    if not os.path.exists(input_file):
        print(f"❌ 원본 데이터 파일을 찾을 수 없습니다: {input_file}")
        return False
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < num_samples:
            print(f"⚠️  원본 데이터가 {num_samples}개보다 적습니다. 전체 {len(lines)}개 사용")
            num_samples = len(lines)
        
        # 첫 N개 라인 추출
        test_lines = lines[:num_samples]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(test_lines)
        
        print(f"✅ 테스트 데이터 생성 완료: {output_file} ({num_samples}개 샘플)")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 데이터 생성 실패: {e}")
        return False

def run_memory_check():
    """메모리 상태 체크"""
    print("🔍 GPU 메모리 상태 체크")
    
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다.")
        return False
    
    # GPU 정보 출력
    props = torch.cuda.get_device_properties(0)
    total_mem_gb = props.total_memory / (1024**3)
    allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
    reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
    free_gb = total_mem_gb - reserved_gb
    
    print(f"📊 GPU: {props.name}")
    print(f"📊 전체 메모리: {total_mem_gb:.1f}GB")
    print(f"📊 사용 중: {allocated_gb:.1f}GB")
    print(f"📊 예약됨: {reserved_gb:.1f}GB")
    print(f"📊 사용 가능: {free_gb:.1f}GB")
    
    # 메모리 부족 경고
    if free_gb < 8.0:
        print("⚠️  경고: 사용 가능한 GPU 메모리가 8GB 미만입니다.")
        print("💡 추천: 다른 GPU 프로세스를 종료하거나 더 작은 배치 크기를 사용하세요.")
        return False
    
    return True

def run_test(num_samples, mode='comment'):
    """실제 테스트 실행"""
    print(f"🚀 테스트 실행: {num_samples}개 샘플, {mode} 모드")
    
    # 원본 데이터 파일 경로
    original_data = os.path.expanduser("~/deepseek-continual/data/train.jsonl")
    test_data = f"test_data_{num_samples}.jsonl"
    
    # 1. 테스트 데이터 생성
    if not create_test_data(original_data, test_data, num_samples):
        return False
    
    # 2. 메모리 체크
    if not run_memory_check():
        print("⚠️  메모리 부족으로 인해 테스트를 건너뜁니다.")
        return False
    
    # 3. cc_train.py 실행
    cmd = f"python cc_train.py --config config.yaml --mode {mode} --data-file {test_data}"
    print(f"🔧 실행 명령어: {cmd}")
    
    try:
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)  # 30분 타임아웃
        
        if result.returncode == 0:
            print("✅ 테스트 성공!")
            print("📋 출력:")
            print(result.stdout[-1000:])  # 마지막 1000자만 출력
            return True
        else:
            print("❌ 테스트 실패!")
            print("📋 에러:")
            print(result.stderr[-1000:])  # 마지막 1000자만 출력
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ 테스트 타임아웃 (30분 초과)")
        return False
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {e}")
        return False
    finally:
        # 테스트 데이터 파일 정리
        if os.path.exists(test_data):
            os.remove(test_data)
            print(f"🗑️  테스트 데이터 파일 정리: {test_data}")

def main():
    parser = argparse.ArgumentParser(description='Continual Learning 소규모 테스트')
    parser.add_argument('--samples', type=int, default=100, 
                       help='테스트할 샘플 수 (기본값: 100)')
    parser.add_argument('--mode', type=str, default='comment',
                       choices=['complete', 'prompt', 'comment', 'error_fix'],
                       help='테스트할 모드 (기본값: comment)')
    parser.add_argument('--skip-memory-check', action='store_true',
                       help='메모리 체크 건너뛰기')
    
    args = parser.parse_args()
    
    print("🧪 Continual Learning 소규모 테스트 시작")
    print(f"📊 테스트 설정: {args.samples}개 샘플, {args.mode} 모드")
    
    # 권장 테스트 단계
    if args.samples > 1000:
        print("⚠️  경고: 1000개 이상의 샘플은 메모리 부족을 일으킬 수 있습니다.")
        print("💡 추천: 먼저 100개, 그 다음 1000개로 단계적으로 테스트하세요.")
        
        response = input("계속하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("테스트 취소됨")
            return
    
    # 테스트 실행
    success = run_test(args.samples, args.mode)
    
    if success:
        print("🎉 테스트 완료! 다음 단계로 진행할 수 있습니다.")
        if args.samples == 100:
            print("💡 다음: python test_small_batch.py --samples 1000")
        elif args.samples == 1000:
            print("💡 다음: 전체 데이터셋으로 실제 학습 시작")
    else:
        print("❌ 테스트 실패. 설정을 확인하고 다시 시도하세요.")
        print("💡 해결방안:")
        print("  1. GPU 메모리 정리: nvidia-smi 확인 후 다른 프로세스 종료")
        print("  2. 배치 크기 감소: config.yaml에서 batch_size를 1로 설정")
        print("  3. 시퀀스 길이 단축: config.yaml에서 max_length를 256으로 설정")

if __name__ == "__main__":
    main()
