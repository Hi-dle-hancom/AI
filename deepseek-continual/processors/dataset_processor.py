#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
데이터셋 처리기 (완전판)

데이터셋 로드, 토큰화, 전처리를 담당합니다.
"""

import os
import json
import glob
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class DatasetProcessor:
    """데이터셋 처리 및 토큰화를 담당하는 프로세서"""
    
    def __init__(self, config: Dict[str, Any], tokenizer: AutoTokenizer):
        """데이터셋 프로세서 초기화
        
        Args:
            config: 설정 딕셔너리
            tokenizer: 토크나이저
        """
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = config.get('max_length', 512)
        self.num_proc = config.get('dataloader_num_workers', 4)
        
        logger.info(f"데이터셋 프로세서 초기화 완료 (max_length: {self.max_length})")
    
    def find_training_files(self, data_path: str, data_file: str = None) -> List[str]:
        """훈련 데이터 파일들 찾기
        
        Args:
            data_path: 데이터 디렉토리 경로
            data_file: 특정 데이터 파일명 (선택사항)
            
        Returns:
            List[str]: 훈련 데이터 파일 경로 리스트
        """
        try:
            if data_file:
                # 명시적으로 지정된 파일 사용
                if os.path.isabs(data_file):
                    return [data_file]
                else:
                    return [os.path.join(data_path, data_file)]
            
            # 디렉토리에서 train*.jsonl 패턴의 파일 찾기
            if os.path.isfile(data_path):
                # data_path가 파일인 경우
                return [data_path]
            
            if not os.path.isdir(data_path):
                logger.error(f"데이터 경로가 존재하지 않습니다: {data_path}")
                return []
            
            # 파일 목록 가져오기
            pattern = os.path.join(data_path, 'train*.jsonl')
            all_files = glob.glob(pattern)
            
            if not all_files:
                # train*.jsonl이 없으면 train.jsonl 찾기
                train_file = os.path.join(data_path, 'train.jsonl')
                if os.path.exists(train_file):
                    all_files = [train_file]
            
            if not all_files:
                logger.error(f"훈련 데이터 파일을 찾을 수 없습니다: {pattern}")
                return []
            
            # 파일명에서 숫자를 추출하여 정렬 (train1.jsonl, train2.jsonl, ...)
            def extract_number(filename):
                match = re.search(r'train(\d+)\.jsonl$', os.path.basename(filename))
                if match:
                    return int(match.group(1))
                # train.jsonl 같은 경우는 0으로 처리
                if os.path.basename(filename) == 'train.jsonl':
                    return 0
                return float('inf')  # 숫자가 없는 경우 맨 뒤로 정렬
            
            # 숫자 순서대로 정렬
            data_files = sorted(all_files, key=extract_number)
            
            logger.info(f"다음 순서로 데이터 파일을 처리합니다: {[os.path.basename(f) for f in data_files]}")
            return data_files
            
        except Exception as e:
            logger.error(f"훈련 파일 찾기 실패: {e}")
            return []
    
    def detect_data_format(self, data_path: str) -> str:
        """데이터 형식 자동 감지"""
        try:
            if not os.path.exists(data_path):
                logger.error(f"데이터 파일을 찾을 수 없습니다: {data_path}")
                return 'unknown'
            
            if data_path.endswith('.jsonl'):
                return 'jsonl'
            elif data_path.endswith('.json'):
                return 'json'
            elif data_path.endswith('.csv'):
                return 'csv'
            
            # 내용 기반 감지
            with open(data_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                try:
                    json.loads(first_line)
                    return 'jsonl'
                except json.JSONDecodeError:
                    pass
                
                if ',' in first_line and not first_line.startswith('{'):
                    return 'csv'
            
            return 'unknown'
            
        except Exception as e:
            logger.error(f"데이터 형식 감지 실패: {e}")
            return 'unknown'
    
    def load_raw_dataset(self, data_path: str) -> Optional[Dataset]:
        """원시 데이터셋 로드"""
        try:
            data_format = self.detect_data_format(data_path)
            logger.info(f"감지된 데이터 형식: {data_format}")
            
            if data_format == 'jsonl':
                dataset = load_dataset('json', data_files=data_path, split='train')
            elif data_format == 'json':
                dataset = load_dataset('json', data_files=data_path, split='train')
            elif data_format == 'csv':
                dataset = load_dataset('csv', data_files=data_path, split='train')
            else:
                logger.error(f"지원하지 않는 데이터 형식: {data_format}")
                return None
            
            logger.info(f"데이터셋 로드 완료: {len(dataset)}개 샘플")
            return dataset
            
        except Exception as e:
            logger.error(f"데이터셋 로드 실패: {e}")
            return None
    
    def format_sample_by_mode(self, sample: Dict[str, Any], mode: str) -> str:
        """모드별 샘플 포맷팅"""
        try:
            if mode == 'complete':
                # 자동완성 모드 (FIM)
                if 'prefix' in sample and 'suffix' in sample and 'middle' in sample:
                    return f"<|fim_prefix|>{sample['prefix']}<|fim_suffix|>{sample['suffix']}<|fim_middle|>{sample['middle']}"
                else:
                    return sample.get('text', '')
            
            elif mode == 'prompt':
                # 프롬프트 기반 코드 생성
                if 'instruction' in sample and 'output' in sample:
                    return f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['output']}"
                elif 'input' in sample and 'output' in sample:
                    return f"### Input:\n{sample['input']}\n\n### Output:\n{sample['output']}"
                else:
                    return sample.get('text', '')
            
            elif mode == 'comment':
                # 주석 기반 코드 생성
                if 'comment' in sample and 'code' in sample:
                    return f"# {sample['comment']}\n{sample['code']}"
                elif 'instruction' in sample and 'output' in sample:
                    return f"# {sample['instruction']}\n{sample['output']}"
                else:
                    return sample.get('text', '')
            
            elif mode == 'error_fix':
                # 오류 수정
                if 'error_code' in sample and 'fixed_code' in sample:
                    return f"### Error Code:\n{sample['error_code']}\n\n### Fixed Code:\n{sample['fixed_code']}"
                elif 'input' in sample and 'output' in sample:
                    return f"### Error Code:\n{sample['input']}\n\n### Fixed Code:\n{sample['output']}"
                else:
                    return sample.get('text', '')
            
            else:
                return sample.get('text', '')
                
        except Exception as e:
            logger.error(f"샘플 포맷팅 실패 ({mode}): {e}")
            return ''
    
    def tokenize_function(self, examples: Dict[str, List], mode: str):
        """토큰화 함수"""
        try:
            texts = []
            for i in range(len(examples[list(examples.keys())[0]])):
                sample = {key: examples[key][i] for key in examples.keys()}
                formatted_text = self.format_sample_by_mode(sample, mode)
                texts.append(formatted_text)
            
            # 토큰화
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors=None
            )
            
            # 레이블 설정 (input_ids와 동일)
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
            
        except Exception as e:
            logger.error(f"토큰화 실패: {e}")
            return {}
    
    def process_dataset(self, data_path: str, mode: str) -> Tuple[Optional[Dataset], Optional[Dataset]]:
        """데이터셋 처리 메인 함수"""
        try:
            # 원시 데이터셋 로드
            raw_dataset = self.load_raw_dataset(data_path)
            if raw_dataset is None:
                return None, None
            
            # 훈련/검증 분할
            if len(raw_dataset) > 1000:
                split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
                train_dataset = split_dataset['train']
                eval_dataset = split_dataset['test']
            else:
                # 작은 데이터셋의 경우 전체를 훈련용으로 사용
                train_dataset = raw_dataset
                eval_dataset = None
            
            # 토큰화
            logger.info(f"{mode} 토큰화 시작...")
            
            tokenized_train = train_dataset.map(
                lambda examples: self.tokenize_function(examples, mode),
                batched=True,
                num_proc=self.num_proc,
                remove_columns=train_dataset.column_names,
                desc=f"{mode} 토큰화 (num_proc={self.num_proc})"
            )
            
            tokenized_eval = None
            if eval_dataset is not None:
                tokenized_eval = eval_dataset.map(
                    lambda examples: self.tokenize_function(examples, mode),
                    batched=True,
                    num_proc=self.num_proc,
                    remove_columns=eval_dataset.column_names,
                    desc=f"{mode} 검증 토큰화 (num_proc={self.num_proc})"
                )
            
            logger.info(f"토큰화 완료 - 훈련: {len(tokenized_train)}개, 검증: {len(tokenized_eval) if tokenized_eval else 0}개")
            
            return tokenized_train, tokenized_eval
            
        except Exception as e:
            logger.error(f"데이터셋 처리 실패: {e}")
            return None, None
    
    def validate_dataset(self, dataset: Dataset) -> bool:
        """데이터셋 유효성 검사"""
        try:
            if dataset is None or len(dataset) == 0:
                logger.error("데이터셋이 비어있습니다")
                return False
            
            # 필수 컬럼 확인
            required_columns = ['input_ids', 'attention_mask', 'labels']
            for col in required_columns:
                if col not in dataset.column_names:
                    logger.error(f"필수 컬럼 누락: {col}")
                    return False
            
            # 샘플 데이터 확인
            sample = dataset[0]
            for col in required_columns:
                if not isinstance(sample[col], list) or len(sample[col]) == 0:
                    logger.error(f"잘못된 데이터 형식: {col}")
                    return False
            
            # 토큰 길이 확인
            max_len = max(len(sample['input_ids']) for sample in dataset[:100])
            if max_len > self.max_length:
                logger.warning(f"일부 샘플이 최대 길이를 초과합니다: {max_len} > {self.max_length}")
            
            logger.info(f"데이터셋 유효성 검사 통과: {len(dataset)}개 샘플")
            return True
            
        except Exception as e:
            logger.error(f"데이터셋 유효성 검사 실패: {e}")
            return False
    
    def get_dataset_statistics(self, dataset: Dataset) -> Dict[str, Any]:
        """데이터셋 통계 정보"""
        try:
            if dataset is None:
                return {}
            
            # 기본 통계
            stats = {
                'total_samples': len(dataset),
                'columns': dataset.column_names
            }
            
            # 토큰 길이 통계
            if 'input_ids' in dataset.column_names:
                lengths = [len(sample['input_ids']) for sample in dataset[:1000]]  # 샘플링
                stats['token_length'] = {
                    'min': min(lengths),
                    'max': max(lengths),
                    'mean': sum(lengths) / len(lengths),
                    'samples_checked': len(lengths)
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"데이터셋 통계 계산 실패: {e}")
            return {}
    
    def save_processed_dataset(self, dataset: Dataset, save_path: str):
        """처리된 데이터셋 저장"""
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            dataset.save_to_disk(save_path)
            logger.info(f"데이터셋 저장 완료: {save_path}")
            
        except Exception as e:
            logger.error(f"데이터셋 저장 실패: {e}")
    
    def load_processed_dataset(self, load_path: str) -> Optional[Dataset]:
        """처리된 데이터셋 로드"""
        try:
            if not os.path.exists(load_path):
                logger.warning(f"처리된 데이터셋을 찾을 수 없습니다: {load_path}")
                return None
            
            dataset = Dataset.load_from_disk(load_path)
            logger.info(f"처리된 데이터셋 로드 완료: {load_path} ({len(dataset)}개 샘플)")
            return dataset
            
        except Exception as e:
            logger.error(f"처리된 데이터셋 로드 실패: {e}")
            return None
