# Hancom AI Project Scripts

이 저장소는 Hancom AI 프로젝트의 스크립트 모음입니다.

## 프로젝트 구조

### 1. deepseek-coder/
DeepSeek-Coder 모델 파인튜닝 파이프라인
- **4단계 학습 체계**:
  - [제1차] `complete`: 자동완성(FIM)
  - [제2차] `prompt`: 일반 프롬프트 기반 코드 생성
  - [제3차] `comment`: 주석 기반 코드 생성
  - [제4차] `error_fix`: 오류 코드 설명 및 수정

### 2. deepseek-continual/
DeepSeek 지속학습(Continual Learning) 파이프라인
- AWS 스팟 인스턴스 환경 최적화
- 자동 체크포인트 감지 및 재개 기능
- 메모리 효율적인 학습 설정

### 3. Qwen2.5-coder/
Qwen2.5-Coder 모델 관련 스크립트

## 주요 특징

### AWS 스팟 인스턴스 지원
- 스팟 인스턴스 중단 감지 및 자동 처리
- 학습 중단 시 안전한 체크포인트 저장
- 재시작 시 자동 체크포인트 탐지 및 재개

### 메모리 최적화
- A10G 22GB 환경 최적화
- CUDA 메모리 단편화 방지
- AMP(Automatic Mixed Precision) 안정성 개선

## 사용법

각 프로젝트 폴더의 개별 README 또는 스크립트 내 주석을 참조하세요.

## 환경 요구사항

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- Transformers 4.30+

## 라이선스

이 프로젝트는 HancomAI 아카데미의 내부 프로젝트입니다.
