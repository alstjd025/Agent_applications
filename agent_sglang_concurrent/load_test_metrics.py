"""
Load Test용 메트릭 수집 및 저장
multiprocessing 환경에서 안전하게 CSV에 기록
"""
import csv
import os
import time
import threading
from datetime import datetime
from typing import Optional, Dict
from multiprocessing import Lock


class LoadTestMetricsCollector:
    """Load test 메트릭 수집 및 CSV 저장"""
    
    def __init__(self, csv_path: str, experiment_id: str, request_rate: float):
        """
        Args:
            csv_path: CSV 파일 경로
            experiment_id: 실험 ID
            request_rate: 요청 비율 (tasks/min)
        """
        self.csv_path = csv_path
        self.experiment_id = experiment_id
        self.request_rate = request_rate
        
        # 파일 lock (multiprocessing safe)
        self.lock = Lock()
        
        # CSV 헤더
        self.fieldnames = [
            # 기본 정보
            'task_id',
            'experiment_id',
            'request_rate',
            'submit_time',
            'queue_wait_time',
            
            # Agent 실행 정보
            'iteration',
            'agent',
            'start_time',
            'end_time',
            'latency',
            
            # Token 정보
            'input_tokens',
            'output_tokens',
            'first_token_latency',
            'decode_speed_tps',
            
            # 서버 상태
            'gpu_memory_mb',
            'kv_cache_usage_pct',
            'concurrent_tasks',
            'server_running_reqs',
            'server_waiting_reqs',
            
            # 기타
            'transition_time',
            'success',
            'error_msg',
            'timestamp'
        ]
        
        # CSV 초기화
        if not os.path.exists(csv_path):
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
        
        # Task 상태 추적
        self.current_task_id = None
        self.current_iteration = 0
        self.last_agent_end_time = None
        self.task_submit_time = None
        self.task_start_time = None
    
    def start_task(self, task_id: str, submit_time: float):
        """
        새 task 시작
        
        Args:
            task_id: Task ID
            submit_time: 제출 시각
        """
        self.current_task_id = task_id
        self.current_iteration = 0
        self.last_agent_end_time = None
        self.task_submit_time = submit_time
        self.task_start_time = None
    
    def start_iteration(self):
        """새 iteration 시작"""
        self.last_agent_end_time = None
    
    def next_iteration(self):
        """다음 iteration으로 이동"""
        self.current_iteration += 1
        self.last_agent_end_time = None
    
    def record_agent_call(
        self,
        agent_name: str,
        start_time: float,
        end_time: float,
        input_tokens: int,
        output_tokens: int,
        first_token_time: Optional[float],
        concurrent_tasks: int,
        server_state: Dict,
        gpu_memory_mb: Optional[int] = None,
        success: Optional[bool] = None,
        error_msg: str = ""
    ):
        """
        Agent 호출 메트릭 기록
        
        Args:
            agent_name: Agent 이름
            start_time: 시작 시간
            end_time: 종료 시간
            input_tokens: 입력 토큰 수
            output_tokens: 출력 토큰 수
            first_token_time: 첫 토큰 시간
            concurrent_tasks: 동시 실행 task 수
            server_state: 서버 상태 (from LoadMonitor)
            gpu_memory_mb: GPU 메모리 사용량
            success: 성공 여부
            error_msg: 에러 메시지
        """
        # Task 시작 시간 기록 (첫 agent call)
        if self.task_start_time is None:
            self.task_start_time = start_time
        
        latency = end_time - start_time
        
        # Queue wait time 계산
        queue_wait_time = None
        if self.task_submit_time and self.task_start_time:
            queue_wait_time = self.task_start_time - self.task_submit_time
        
        # First token latency
        first_token_latency = None
        if first_token_time:
            first_token_latency = first_token_time - start_time
        
        # Decode speed
        decode_speed_tps = None
        if first_token_time and output_tokens > 0:
            decode_time = end_time - first_token_time
            if decode_time > 0:
                decode_speed_tps = output_tokens / decode_time
        
        # Transition time
        transition_time = None
        if self.last_agent_end_time:
            transition_time = start_time - self.last_agent_end_time
        
        self.last_agent_end_time = end_time
        
        # 레코드 생성
        record = {
            'task_id': self.current_task_id,
            'experiment_id': self.experiment_id,
            'request_rate': self.request_rate,
            'submit_time': self.task_submit_time,
            'queue_wait_time': round(queue_wait_time, 4) if queue_wait_time else None,
            
            'iteration': self.current_iteration,
            'agent': agent_name,
            'start_time': start_time,
            'end_time': end_time,
            'latency': round(latency, 4),
            
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'first_token_latency': round(first_token_latency, 4) if first_token_latency else None,
            'decode_speed_tps': round(decode_speed_tps, 2) if decode_speed_tps else None,
            
            'gpu_memory_mb': gpu_memory_mb,
            'kv_cache_usage_pct': round(server_state.get('kv_cache_usage_pct', 0), 2),
            'concurrent_tasks': concurrent_tasks,
            'server_running_reqs': server_state.get('running_requests', 0),
            'server_waiting_reqs': server_state.get('waiting_requests', 0),
            
            'transition_time': round(transition_time, 4) if transition_time else None,
            'success': success,
            'error_msg': error_msg,
            'timestamp': datetime.now().isoformat()
        }
        
        # CSV에 쓰기 (lock 사용)
        with self.lock:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(record)
    
    @staticmethod
    def load_completed_tasks(csv_path: str) -> set:
        """완료된 task_id 목록 로드"""
        completed = set()
        if not os.path.exists(csv_path):
            return completed
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['agent'] == 'debugging' and row.get('success') == 'True':
                        completed.add(row['task_id'])
        except Exception:
            pass
        
        return completed


if __name__ == "__main__":
    # 테스트
    import tempfile
    
    temp_csv = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
    temp_csv.close()
    
    collector = LoadTestMetricsCollector(
        csv_path=temp_csv.name,
        experiment_id="test-001",
        request_rate=5.0
    )
    
    # 테스트 데이터
    collector.start_task("test_task_1", submit_time=time.time())
    
    time.sleep(0.1)
    
    collector.record_agent_call(
        agent_name="planning",
        start_time=time.time(),
        end_time=time.time() + 1,
        input_tokens=100,
        output_tokens=200,
        first_token_time=time.time() + 0.1,
        concurrent_tasks=3,
        server_state={'kv_cache_usage_pct': 45.2, 'running_requests': 3, 'waiting_requests': 0}
    )
    
    print(f"Test CSV created: {temp_csv.name}")
    
    # 내용 확인
    with open(temp_csv.name, 'r') as f:
        print(f.read())
    
    os.unlink(temp_csv.name)