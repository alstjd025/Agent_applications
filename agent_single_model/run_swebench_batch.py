#!/usr/bin/env python3
"""
SWE-bench Lite 배치 실행 스크립트
- 300개 문제 순차 실행
- 각 문제별 end-to-end latency 측정
- CSV로 메트릭 저장
- 10분 timeout
- 완료된 문제 skip
"""
import os
import sys
import time
import signal
import argparse
from datetime import datetime
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

# 로컬 모듈 import
from swe_agent_single import agent, AgentState
from metrics_tracker import MetricsTracker
from vllm_logger import VLLMLogParser, MockVLLMLogParser
from agent_logger import AgentLogger


class TimeoutException(Exception):
    """Timeout 예외"""
    pass


def timeout_handler(signum, frame):
    """Timeout signal handler"""
    raise TimeoutException("Task timeout (10 minutes)")


class SWEBenchRunner:
    """SWE-bench Lite 배치 실행기"""
    
    def __init__(
        self,
        csv_path: str,
        error_log_path: str,
        vllm_log_path: Optional[str] = None,
        max_iterations: int = 5,
        timeout_seconds: int = 600  # 10분
    ):
        """
        Args:
            csv_path: 메트릭 CSV 저장 경로
            error_log_path: 에러 로그 저장 경로
            vllm_log_path: vLLM 서버 로그 경로 (optional)
            max_iterations: agent 최대 반복 횟수
            timeout_seconds: 문제당 최대 실행 시간 (초)
        """
        self.csv_path = csv_path
        self.error_log_path = error_log_path
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        
        # 메트릭 트래커 초기화 (KV cache 모니터링 포함)
        self.metrics_tracker = MetricsTracker(
            csv_path,
            vllm_base_url="http://localhost:8001"
        )
        
        # Agent 로거 초기화
        agent_log_dir = os.path.join(os.path.dirname(csv_path), 'agent_logs')
        self.agent_logger = AgentLogger(agent_log_dir)
        print(f"[Info] Agent logs will be saved to: {agent_log_dir}")
        
        # vLLM 로그 파서 초기화 (optional)
        if vllm_log_path and os.path.exists(vllm_log_path):
            self.vllm_parser = VLLMLogParser(vllm_log_path)
            self.vllm_parser.start()
        else:
            print("[Warning] vLLM log parsing disabled (using client-side metrics only)")
            self.vllm_parser = MockVLLMLogParser()
        
        # 에러 로그 파일 초기화
        os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
        with open(error_log_path, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Batch run started at {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n\n")
    
    def run_single_task(self, task: dict) -> dict:
        """
        단일 SWE-bench 문제 실행
        
        Args:
            task: SWE-bench 문제 dict
        
        Returns:
            실행 결과 dict
        """
        task_id = task['instance_id']
        
        # 메트릭 트래커에 task 등록
        self.metrics_tracker.start_task(task_id)
        
        # Agent 로거에 task 등록
        self.agent_logger.start_task(
            task_id=task_id,
            problem_statement=task['problem_statement'],
            repo=task['repo']
        )
        
        # Agent 초기 상태 구성
        initial_state: AgentState = {
            "task_id": task_id,
            "problem_statement": task['problem_statement'],
            "repo": task['repo'],
            "plan": "",
            "code": "",
            "debug_result": "",
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "history": [],
            "metrics_tracker": self.metrics_tracker,
            "agent_logger": self.agent_logger,
        }
        
        # Timeout 설정
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout_seconds)
        
        try:
            overall_start = time.time()
            
            # Agent 실행
            result = agent.invoke(initial_state)
            
            overall_end = time.time()
            total_time = overall_end - overall_start
            
            # Timeout 해제
            signal.alarm(0)
            
            # 결과 반환
            success = 'PASS' in result.get('debug_result', '').upper()
            
            # Agent 로거에 최종 결과 기록
            self.agent_logger.log_final_result(
                success=success,
                total_time=total_time,
                iterations=result['iteration']
            )
            
            return {
                'task_id': task_id,
                'success': success,
                'total_time': total_time,
                'iterations': result['iteration'],
                'error': None
            }
        
        except TimeoutException:
            signal.alarm(0)
            error_msg = f"Timeout after {self.timeout_seconds} seconds"
            self._log_error(task_id, error_msg)
            self.agent_logger.log_error(error_msg)
            
            return {
                'task_id': task_id,
                'success': False,
                'total_time': self.timeout_seconds,
                'iterations': -1,
                'error': error_msg
            }
        
        except Exception as e:
            signal.alarm(0)
            error_msg = f"Error: {type(e).__name__}: {str(e)}"
            self._log_error(task_id, error_msg)
            self.agent_logger.log_error(error_msg)
            
            return {
                'task_id': task_id,
                'success': False,
                'total_time': -1,
                'iterations': -1,
                'error': error_msg
            }
    
    def _log_error(self, task_id: str, error_msg: str):
        """에러 로그 기록"""
        with open(self.error_log_path, 'a') as f:
            f.write(f"[{datetime.now().isoformat()}] {task_id}\n")
            f.write(f"  {error_msg}\n\n")
        
        print(f"\n[ERROR] {task_id}: {error_msg}")
    
    def run_batch(self, dataset, start_index: int = 0, end_index: Optional[int] = None):
        """
        배치 실행
        
        Args:
            dataset: SWE-bench 데이터셋
            start_index: 시작 인덱스
            end_index: 종료 인덱스 (None이면 끝까지)
        """
        # 완료된 문제 로드
        completed_tasks = MetricsTracker.load_completed_tasks(self.csv_path)
        print(f"[Info] Already completed tasks: {len(completed_tasks)}")
        
        # 실행할 문제 선택
        if end_index is None:
            end_index = len(dataset)
        
        # Hugging Face Dataset의 select() 메서드 사용
        tasks_to_run = dataset.select(range(start_index, end_index))
        
        # 통계
        stats = {
            'total': len(tasks_to_run),
            'completed': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'timeout': 0,
            'error': 0
        }
        
        # Progress bar
        pbar = tqdm(tasks_to_run, desc="Processing SWE-bench Lite", unit="task")
        
        for task in pbar:
            task_id = task['instance_id']
            
            # 이미 완료된 문제는 skip
            if task_id in completed_tasks:
                stats['skipped'] += 1
                pbar.set_postfix({
                    'success': stats['success'],
                    'failed': stats['failed'],
                    'skipped': stats['skipped'],
                    'status': 'SKIP'
                })
                continue
            
            # 문제 실행
            result = self.run_single_task(task)
            
            # 통계 업데이트
            stats['completed'] += 1
            
            if result['error']:
                if 'Timeout' in result['error']:
                    stats['timeout'] += 1
                else:
                    stats['error'] += 1
                stats['failed'] += 1
                status = 'TIMEOUT' if 'Timeout' in result['error'] else 'ERROR'
            elif result['success']:
                stats['success'] += 1
                status = '✓'
            else:
                stats['failed'] += 1
                status = '✗'
            
            # Progress bar 업데이트
            pbar.set_postfix({
                'success': stats['success'],
                'failed': stats['failed'],
                'skipped': stats['skipped'],
                'time': f"{result['total_time']:.1f}s",
                'status': status
            })
        
        pbar.close()
        
        # 최종 통계 출력
        self._print_summary(stats)
        
        # vLLM 로그 파서 종료
        self.vllm_parser.stop()
    
    def _print_summary(self, stats: dict):
        """최종 통계 출력"""
        print(f"\n{'='*60}")
        print("BATCH EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total tasks: {stats['total']}")
        print(f"Completed: {stats['completed']}")
        print(f"  - Success: {stats['success']} ({stats['success']/max(stats['completed'], 1)*100:.1f}%)")
        print(f"  - Failed: {stats['failed']} ({stats['failed']/max(stats['completed'], 1)*100:.1f}%)")
        print(f"    - Timeout: {stats['timeout']}")
        print(f"    - Error: {stats['error']}")
        print(f"Skipped (already done): {stats['skipped']}")
        print(f"\nResults saved to: {self.csv_path}")
        print(f"Error log saved to: {self.error_log_path}")
        print(f"Agent logs saved to: {self.agent_logger.log_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Run SWE-bench Lite batch evaluation")
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/workspace/vllm/ms_local/results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--vllm-log',
        type=str,
        default='/workspace/vllm/ms_local/agent_single_model/logs/server.log',
        help='vLLM server log path'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=5,
        help='Maximum iterations per task'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=600,
        help='Timeout per task in seconds (default: 600 = 10 minutes)'
    )
    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='Start index in dataset'
    )
    parser.add_argument(
        '--end-index',
        type=int,
        default=None,
        help='End index in dataset (None = all)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='princeton-nlp/SWE-bench_Lite',
        help='Dataset to use'
    )
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 파일 경로
    csv_path = os.path.join(args.output_dir, 'metrics.csv')
    error_log_path = os.path.join(args.output_dir, 'errors.log')
    
    print(f"\n{'='*60}")
    print("SWE-BENCH LITE BATCH EVALUATION")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Timeout: {args.timeout}s")
    print(f"Output: {csv_path}")
    print(f"Error log: {error_log_path}")
    print(f"{'='*60}\n")
    
    # 데이터셋 로드
    print("Loading dataset...")
    dataset = load_dataset(args.dataset, split='test')
    print(f"Loaded {len(dataset)} tasks\n")
    
    # Runner 초기화
    runner = SWEBenchRunner(
        csv_path=csv_path,
        error_log_path=error_log_path,
        vllm_log_path=args.vllm_log,
        max_iterations=args.max_iterations,
        timeout_seconds=args.timeout
    )
    
    # 배치 실행
    try:
        runner.run_batch(
            dataset,
            start_index=args.start_index,
            end_index=args.end_index
        )
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user")
        print(f"Results saved to: {csv_path}")
        sys.exit(0)


if __name__ == "__main__":
    main()