"""
vLLM 서버 로그 파싱 모듈
서버 로그에서 request별 prefill/decode 시간 추출
"""
import re
import time
from typing import Dict, Optional
from threading import Thread
from collections import defaultdict


class VLLMLogParser:
    """
    vLLM 서버 로그를 실시간으로 파싱하여 메트릭 추출
    
    vLLM 로그 예시:
    INFO:     Received request abc123: prompt_tokens=150, ...
    INFO:     Finished request abc123: output_tokens=512, generation_time=10.5s
    """
    
    def __init__(self, log_file_path: str):
        """
        Args:
            log_file_path: vLLM 서버 로그 파일 경로
        """
        self.log_file_path = log_file_path
        self.metrics = defaultdict(dict)
        self.is_running = False
        self.parser_thread = None
    
    def start(self):
        """로그 파싱 스레드 시작"""
        if self.is_running:
            return
        
        self.is_running = True
        self.parser_thread = Thread(target=self._parse_loop, daemon=True)
        self.parser_thread.start()
        print(f"[VLLMLogParser] Started parsing {self.log_file_path}")
    
    def stop(self):
        """로그 파싱 스레드 종료"""
        self.is_running = False
        if self.parser_thread:
            self.parser_thread.join(timeout=5)
    
    def _parse_loop(self):
        """로그 파일을 tail -f 방식으로 읽으면서 파싱"""
        try:
            with open(self.log_file_path, 'r') as f:
                # 파일 끝으로 이동
                f.seek(0, 2)
                
                while self.is_running:
                    line = f.readline()
                    if not line:
                        time.sleep(0.1)
                        continue
                    
                    self._parse_line(line)
        except FileNotFoundError:
            print(f"[VLLMLogParser] Log file not found: {self.log_file_path}")
        except Exception as e:
            print(f"[VLLMLogParser] Error: {e}")
    
    def _parse_line(self, line: str):
        """
        로그 라인 파싱
        
        예상 패턴:
        - "Received request {request_id}"
        - "Generated {num_tokens} tokens in {time}s"
        - 등등 (vLLM 버전에 따라 다를 수 있음)
        """
        # request_id 추출
        # 예: "INFO ... request_id=abc123" 또는 "request abc123:"
        request_id_match = re.search(r'request[_\s]id[=:\s]+([a-zA-Z0-9\-]+)', line)
        if not request_id_match:
            return
        
        request_id = request_id_match.group(1)
        
        # Prompt tokens
        prompt_tokens_match = re.search(r'prompt[_\s]tokens[=:\s]+(\d+)', line)
        if prompt_tokens_match:
            self.metrics[request_id]['prompt_tokens'] = int(prompt_tokens_match.group(1))
        
        # Output tokens
        output_tokens_match = re.search(r'output[_\s]tokens[=:\s]+(\d+)', line)
        if output_tokens_match:
            self.metrics[request_id]['output_tokens'] = int(output_tokens_match.group(1))
        
        # Generation time
        gen_time_match = re.search(r'generation[_\s]time[=:\s]+([\d.]+)', line)
        if gen_time_match:
            self.metrics[request_id]['generation_time'] = float(gen_time_match.group(1))
        
        # Prefill time (vLLM 일부 버전에서 제공)
        prefill_match = re.search(r'prefill[_\s]time[=:\s]+([\d.]+)', line)
        if prefill_match:
            self.metrics[request_id]['prefill_time'] = float(prefill_match.group(1))
        
        # Decode time
        decode_match = re.search(r'decode[_\s]time[=:\s]+([\d.]+)', line)
        if decode_match:
            self.metrics[request_id]['decode_time'] = float(decode_match.group(1))
    
    def get_metrics(self, request_id: str) -> Optional[Dict]:
        """
        특정 request_id의 메트릭 반환
        
        Returns:
            {
                'prompt_tokens': int,
                'output_tokens': int,
                'generation_time': float,
                'prefill_time': float (optional),
                'decode_time': float (optional)
            }
        """
        return self.metrics.get(request_id)
    
    def clear_metrics(self, request_id: str):
        """메트릭 삭제 (메모리 정리)"""
        if request_id in self.metrics:
            del self.metrics[request_id]


class MockVLLMLogParser:
    """
    로그 파싱이 불가능할 경우 사용하는 Mock
    (vLLM 버전이 로그를 제공하지 않거나 형식이 다를 경우)
    """
    
    def start(self):
        print("[MockVLLMLogParser] Log parsing disabled (using client-side metrics only)")
    
    def stop(self):
        pass
    
    def get_metrics(self, request_id: str) -> None:
        return None
    
    def clear_metrics(self, request_id: str):
        pass


if __name__ == "__main__":
    # 테스트
    import tempfile
    
    # 임시 로그 파일 생성
    temp_log = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log')
    temp_log.write("INFO: Received request abc123: prompt_tokens=150\n")
    temp_log.write("INFO: Finished request abc123: output_tokens=512, generation_time=10.5\n")
    temp_log.flush()
    temp_log.close()
    
    # 파서 테스트
    parser = VLLMLogParser(temp_log.name)
    parser.start()
    
    # 로그 파싱 대기
    time.sleep(1)
    
    # 메트릭 확인
    metrics = parser.get_metrics('abc123')
    print(f"Parsed metrics: {metrics}")
    
    parser.stop()
    
    import os
    os.unlink(temp_log.name)