"""
메트릭 수집 및 CSV 저장 모듈
각 agent call마다 latency, token 정보, GPU 메모리 등을 기록
"""
import csv
import os
import time
import requests
import re
from datetime import datetime
from typing import Optional, Dict, Any
import GPUtil


class KVCacheMonitor:
    """vLLM의 KV cache 사용량 모니터"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.metrics_url = f"{base_url}/metrics"
    
    def get_kv_cache_usage(self) -> Optional[float]:
        """
        KV cache 사용률 반환 (0-100%)
        
        Returns:
            KV cache 사용률 (%), 실패시 None
        """
        try:
            response = requests.get(self.metrics_url, timeout=2)
            response.raise_for_status()
            
            metrics_text = response.text
            
            # KV cache 사용률 파싱
            match = re.search(r'vllm:gpu_cache_usage_perc\s+([\d.]+)', metrics_text)
            if match:
                return float(match.group(1)) * 100  # 0-1 → 0-100%
            
            # 대안: blocks 기반 계산
            blocks_used_match = re.search(r'vllm:gpu_cache_usage_blocks\s+([\d.]+)', metrics_text)
            blocks_total_match = re.search(r'vllm:gpu_cache_total_blocks\s+([\d.]+)', metrics_text)
            
            if blocks_used_match and blocks_total_match:
                blocks_used = float(blocks_used_match.group(1))
                blocks_total = float(blocks_total_match.group(1))
                if blocks_total > 0:
                    return (blocks_used / blocks_total) * 100
            
            return None
        
        except Exception as e:
            # 메트릭 엔드포인트가 없거나 접근 불가시
            return None


class MetricsTracker:
    """Agent 실행 메트릭을 추적하고 CSV로 저장"""
    
    def __init__(self, csv_path: str, vllm_base_url: str = "http://localhost:8001"):
        """
        Args:
            csv_path: 메트릭을 저장할 CSV 파일 경로
            vllm_base_url: vLLM 서버 URL
        """
        self.csv_path = csv_path
        self.current_task_id = None
        self.current_iteration = 0
        self.last_agent_end_time = None
        
        # KV cache 모니터 초기화
        self.kv_monitor = KVCacheMonitor(vllm_base_url)
        
        # CSV 헤더
        self.fieldnames = [
            'task_id',
            'iteration',
            'agent',
            'start_time',
            'end_time',
            'latency',
            'input_tokens',
            'output_tokens',
            'first_token_latency',
            'decode_speed_tps',
            'gpu_memory_mb',
            'kv_cache_usage_pct',
            'transition_time',
            'success',
            'error_msg',
            'timestamp'
        ]
        
        # CSV 파일 초기화 (없으면 헤더 작성)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
    
    def start_task(self, task_id: str):
        """새로운 task 시작"""
        self.current_task_id = task_id
        self.current_iteration = 0
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
        input_tokens: int = 0,
        output_tokens: int = 0,
        first_token_time: Optional[float] = None,
        success: Optional[bool] = None,
        error_msg: str = ""
    ):
        """
        Agent 호출 메트릭 기록
        
        Args:
            agent_name: agent 이름 (planning, coding, debugging)
            start_time: 시작 시간 (time.time())
            end_time: 종료 시간 (time.time())
            input_tokens: 입력 토큰 수 (prompt)
            output_tokens: 생성된 토큰 수 (response)
            first_token_time: 첫 토큰 생성 시간 (prefill 근사치)
            success: 성공 여부 (debugging agent의 경우)
            error_msg: 에러 메시지
        """
        latency = end_time - start_time
        
        # First token latency 계산 (prefill 근사)
        first_token_latency = None
        if first_token_time is not None:
            first_token_latency = first_token_time - start_time
        
        # Decode speed 계산
        decode_speed_tps = None
        if first_token_time is not None and output_tokens > 0:
            decode_time = end_time - first_token_time
            if decode_time > 0:
                decode_speed_tps = output_tokens / decode_time
        
        # GPU 메모리 측정 (vLLM 전체 할당량 - 변화 적음)
        gpu_memory_mb = self._get_gpu_memory()
        
        # KV cache 사용률 측정 (실제 사용량)
        kv_cache_usage_pct = self.kv_monitor.get_kv_cache_usage()
        
        # Transition time 계산 (이전 agent 끝 → 현재 agent 시작)
        transition_time = None
        if self.last_agent_end_time is not None:
            transition_time = start_time - self.last_agent_end_time
        
        # 다음 transition을 위해 저장
        self.last_agent_end_time = end_time
        
        # 메트릭 레코드 생성
        record = {
            'task_id': self.current_task_id,
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
            'kv_cache_usage_pct': round(kv_cache_usage_pct, 2) if kv_cache_usage_pct is not None else None,
            'transition_time': round(transition_time, 4) if transition_time else None,
            'success': success,
            'error_msg': error_msg,
            'timestamp': datetime.now().isoformat()
        }
        
        # CSV에 추가 (즉시 저장 - 중단되어도 데이터 유실 방지)
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(record)
        
        return record
    
    def _get_gpu_memory(self) -> Optional[int]:
        """
        현재 GPU 메모리 사용량 측정 (MB)
        
        Note: vLLM은 시작 시 --gpu-memory-utilization만큼 메모리를 미리 할당하므로,
        이 값은 거의 변하지 않습니다. 실제로는 내부 KV cache 사용량이 변하지만,
        GPUtil은 프로세스의 전체 할당량만 반환합니다.
        """
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                # 첫 번째 GPU 메모리 사용량
                return int(gpus[0].memoryUsed)
            return None
        except Exception as e:
            print(f"Warning: Failed to get GPU memory: {e}")
            return None
    
    @staticmethod
    def load_completed_tasks(csv_path: str) -> set:
        """
        이미 완료된 task_id 목록 로드 (재시작시 skip용)
        
        Returns:
            완료된 task_id의 set
        """
        completed = set()
        if not os.path.exists(csv_path):
            return completed
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # debugging agent에서 success=True인 경우만 완료로 간주
                    if row['agent'] == 'debugging' and row['success'] == 'True':
                        completed.add(row['task_id'])
        except Exception as e:
            print(f"Warning: Failed to load completed tasks: {e}")
        
        return completed


class StreamingTokenTracker:
    """
    스트리밍 응답에서 첫 토큰 시간과 총 토큰 수를 추적
    (OpenAI API streaming 사용시)
    """
    
    def __init__(self):
        self.start_time = None
        self.first_token_time = None
        self.token_count = 0
        self.full_response = ""
    
    def reset(self):
        """새로운 요청을 위해 리셋"""
        self.start_time = time.time()
        self.first_token_time = None
        self.token_count = 0
        self.full_response = ""
    
    def on_token(self, token: str):
        """토큰 수신시 호출"""
        if self.first_token_time is None:
            self.first_token_time = time.time()
        
        self.token_count += 1
        self.full_response += token
    
    def get_metrics(self) -> Dict[str, Any]:
        """메트릭 반환"""
        return {
            'first_token_time': self.first_token_time,
            'tokens_generated': self.token_count,
            'full_response': self.full_response
        }


if __name__ == "__main__":
    # 테스트
    import tempfile
    
    temp_csv = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
    temp_csv.close()
    
    tracker = MetricsTracker(temp_csv.name)
    
    # 예시 기록
    tracker.start_task("test_task_1")
    
    start = time.time()
    time.sleep(0.1)
    first_token = time.time()
    time.sleep(0.2)
    end = time.time()
    
    tracker.record_agent_call(
        agent_name="planning",
        start_time=start,
        end_time=end,
        input_tokens=150,
        output_tokens=100,
        first_token_time=first_token
    )
    
    print(f"Test metrics saved to: {temp_csv.name}")
    
    # 완료된 task 로드 테스트
    completed = MetricsTracker.load_completed_tasks(temp_csv.name)
    print(f"Completed tasks: {completed}")
    
    os.unlink(temp_csv.name)