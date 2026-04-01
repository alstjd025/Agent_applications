"""
Agent 입출력 로깅 모듈
각 agent의 입력 prompt와 출력 response를 읽기 쉬운 형식으로 로깅
"""
import os
from datetime import datetime
from typing import Optional


class AgentLogger:
    """Agent 입출력을 파일에 로깅"""
    
    def __init__(self, log_dir: str):
        """
        Args:
            log_dir: 로그 파일을 저장할 디렉토리
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.current_task_id = None
        self.current_log_file = None
        self.current_iteration = 0
    
    def start_task(self, task_id: str, problem_statement: str, repo: str):
        """
        새로운 task 시작
        
        Args:
            task_id: SWE-bench task ID
            problem_statement: 문제 설명
            repo: 저장소 이름
        """
        self.current_task_id = task_id
        self.current_iteration = 0
        
        # 파일명에서 특수문자 제거
        safe_filename = task_id.replace('/', '_').replace('\\', '_')
        log_path = os.path.join(self.log_dir, f"{safe_filename}.log")
        
        self.current_log_file = log_path
        
        # 로그 파일 초기화
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"TASK: {task_id}\n")
            f.write(f"REPO: {repo}\n")
            f.write(f"STARTED: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("PROBLEM STATEMENT:\n")
            f.write("-" * 80 + "\n")
            f.write(problem_statement + "\n")
            f.write("-" * 80 + "\n\n")
    
    def start_iteration(self, iteration: int):
        """새로운 iteration 시작"""
        self.current_iteration = iteration
        
        if self.current_log_file:
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"ITERATION: {iteration}\n")
                f.write("=" * 80 + "\n\n")
    
    def log_agent_call(
        self,
        agent_name: str,
        input_prompt: str,
        output_response: str,
        latency: float,
        input_tokens: int,
        output_tokens: int
    ):
        """
        Agent 호출 로깅
        
        Args:
            agent_name: agent 이름 (planning, coding, debugging)
            input_prompt: 입력 prompt
            output_response: 출력 response
            latency: 실행 시간 (초)
            input_tokens: 입력 토큰 수
            output_tokens: 출력 토큰 수
        """
        if not self.current_log_file:
            return
        
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            # Agent 이름 (대문자)
            f.write(f"[{agent_name.upper()} AGENT - INPUT]\n")
            f.write("-" * 80 + "\n")
            f.write(input_prompt + "\n")
            f.write("-" * 80 + "\n")
            f.write(f"Input tokens: {input_tokens}\n\n")
            
            f.write(f"[{agent_name.upper()} AGENT - OUTPUT]\n")
            f.write("-" * 80 + "\n")
            f.write(output_response + "\n")
            f.write("-" * 80 + "\n")
            f.write(f"Output tokens: {output_tokens}\n")
            f.write(f"Latency: {latency:.2f}s\n\n")
    
    def log_final_result(self, success: bool, total_time: float, iterations: int):
        """
        최종 결과 로깅
        
        Args:
            success: 성공 여부
            total_time: 총 실행 시간
            iterations: 총 반복 횟수
        """
        if not self.current_log_file:
            return
        
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("FINAL RESULT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Success: {'✓ PASS' if success else '✗ FAIL'}\n")
            f.write(f"Total iterations: {iterations}\n")
            f.write(f"Total time: {total_time:.2f}s\n")
            f.write(f"Completed: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n")
    
    def log_error(self, error_msg: str):
        """에러 로깅"""
        if not self.current_log_file:
            return
        
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("ERROR\n")
            f.write("=" * 80 + "\n")
            f.write(error_msg + "\n")
            f.write("=" * 80 + "\n")


if __name__ == "__main__":
    # 테스트
    import tempfile
    
    temp_dir = tempfile.mkdtemp()
    logger = AgentLogger(temp_dir)
    
    # Task 시작
    logger.start_task(
        "django__django-12345",
        "There is a bug in the authentication system...",
        "django/django"
    )
    
    # Iteration 0
    logger.start_iteration(0)
    
    # Planning agent
    logger.log_agent_call(
        agent_name="planning",
        input_prompt="You are a senior engineer...\n\nBug: Authentication fails",
        output_response="Plan:\n1. Check auth.py\n2. Fix the login function",
        latency=10.5,
        input_tokens=150,
        output_tokens=200
    )
    
    # Coding agent
    logger.log_agent_call(
        agent_name="coding",
        input_prompt="Implement the fix...",
        output_response="def login(user):\n    # Fixed code here\n    pass",
        latency=15.2,
        input_tokens=250,
        output_tokens=300
    )
    
    # Debugging agent
    logger.log_agent_call(
        agent_name="debugging",
        input_prompt="Review this code...",
        output_response="PASS",
        latency=2.1,
        input_tokens=400,
        output_tokens=10
    )
    
    # Final result
    logger.log_final_result(success=True, total_time=27.8, iterations=1)
    
    # 결과 확인
    log_file = os.path.join(temp_dir, "django__django-12345.log")
    print(f"Test log created at: {log_file}")
    print("\nContent preview:")
    with open(log_file, 'r') as f:
        print(f.read()[:500])
    
    import shutil
    shutil.rmtree(temp_dir)