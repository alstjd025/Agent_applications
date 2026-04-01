#!/usr/bin/env python3
"""
SWE-bench Lite 배치 실행 스크립트 (병렬 + request-rate 기반 시작 스케줄링 + multi-run)
- 300개 문제 병렬 실행 (in-flight 제한 없음: executor 큐잉은 가능)
- request rate(x per minute)로 "task 시작" 속도 제어
- CSV로 메트릭 저장 (rate별 파일 분리)
- 완료된 문제 skip (각 CSV별로 독립)
- task별 LLM 인스턴스 생성( shared_llm 제거 )
- optional server.log를 run별로 분리 저장(export)
"""

import os
import sys
import time
import argparse
import threading
from datetime import datetime
from typing import Optional, List

from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 로컬 모듈
from swe_agent_single import agent, AgentState, make_llm, is_pass_verdict
from metrics_tracker import MetricsTracker
from vllm_logger import VLLMLogParser, MockVLLMLogParser
from agent_logger import AgentLogger


class SWEBenchRunner:
    """SWE-bench Lite 배치 실행기"""

    def __init__(
        self,
        csv_path: str,
        error_log_path: str,
        agent_log_root_dir: str,
        server_log_path: Optional[str] = None,
        server_export_log_path: Optional[str] = None,
        server_base_url: str = "http://localhost:30000",
        max_iterations: int = 5,
    ):
        self.csv_path = csv_path
        self.error_log_path = error_log_path
        self.max_iterations = max_iterations
        self.server_base_url = server_base_url

        # run별 agent log dir
        self.agent_log_dir = agent_log_root_dir
        os.makedirs(self.agent_log_dir, exist_ok=True)
        print(f"[Info] Agent logs will be saved to: {self.agent_log_dir}")

        # 에러 로그 파일 초기화
        os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
        with open(error_log_path, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Batch run started at {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n\n")

        # CSV 헤더 보장
        _ = MetricsTracker(
            self.csv_path,
            server_base_url=self.server_base_url,
            enable_server_metrics=False,
        )

        # vLLM 로그 파서 (optional) + export
        self.vllm_parser = None
        if server_log_path and os.path.exists(server_log_path):
            try:
                self.vllm_parser = VLLMLogParser(
                    server_log_path,
                    export_log_path=server_export_log_path,
                    start_at_end=True,
                )
                self.vllm_parser.start()
            except Exception as e:
                print(f"[Warning] Failed to start VLLMLogParser: {e}")
                self.vllm_parser = MockVLLMLogParser()
                self.vllm_parser.start()
        else:
            print("[Info] Server log parsing disabled (using client-side metrics only)")
            self.vllm_parser = MockVLLMLogParser()
            self.vllm_parser.start()

    def _log_error(self, task_id: str, error_msg: str):
        with open(self.error_log_path, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {task_id}\n")
            f.write(f"  {error_msg}\n\n")
        print(f"\n[ERROR] {task_id}: {error_msg}")

    def run_single_task(self, task: dict) -> dict:
        """단일 SWE-bench 문제 실행"""
        task_id = task["instance_id"]
        print(f"[START] thread={threading.get_ident()} task_id={task_id}")

        # ✅ task 전용 MetricsTracker / AgentLogger 생성
        metrics_tracker = MetricsTracker(
            self.csv_path,
            server_base_url=self.server_base_url,
            enable_server_metrics=False,
        )

        agent_logger = AgentLogger(self.agent_log_dir)
        metrics_tracker.start_task(task_id)
        agent_logger.start_task(
            task_id=task_id,
            problem_statement=task["problem_statement"],
            repo=task["repo"],
        )

        # ✅ task 전용 LLM 인스턴스 (shared_llm 제거)
        llm = make_llm(seed=42, temperature=0.7)

        initial_state: AgentState = {
            "task_id": task_id,
            "problem_statement": task["problem_statement"],
            "repo": task["repo"],
            "plan": "",
            "code": "",
            "debug_result": "",
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "history": [],
            "metrics_tracker": metrics_tracker,
            "agent_logger": agent_logger,
            "llm": llm,  # ✅ 핵심
        }

        try:
            overall_start = time.time()
            result = agent.invoke(initial_state)
            overall_end = time.time()
            total_time = overall_end - overall_start

            success = is_pass_verdict(result.get("debug_result", ""))

            agent_logger.log_final_result(
                success=success,
                total_time=total_time,
                iterations=result.get("iteration", -1),
            )

            return {
                "task_id": task_id,
                "success": success,
                "total_time": total_time,
                "iterations": result.get("iteration", -1),
                "error": None,
            }

        except Exception as e:
            error_msg = f"Error: {type(e).__name__}: {str(e)}"
            self._log_error(task_id, error_msg)
            agent_logger.log_error(error_msg)

            return {
                "task_id": task_id,
                "success": False,
                "total_time": -1,
                "iterations": -1,
                "error": error_msg,
            }

    def run_batch_parallel(
        self,
        dataset,
        start_index: int = 0,
        end_index: Optional[int] = None,
        request_rate_per_min: float = 60.0,
        max_workers: Optional[int] = None,
    ):
        """
        병렬 실행 + request rate 기반 시작 스케줄링
        - request_rate_per_min = 60이면 1초마다 1개 submit
        - in-flight 제한은 두지 않음(요구사항)
        """
        completed_tasks = MetricsTracker.load_completed_tasks(self.csv_path)
        print(f"[Info] Already completed tasks (from this CSV): {len(completed_tasks)}")

        if end_index is None:
            end_index = len(dataset)

        tasks_slice = dataset.select(range(start_index, end_index))
        tasks_to_run = [t for t in tasks_slice if t["instance_id"] not in completed_tasks]

        stats = {
            "total_in_range": len(tasks_slice),
            "skipped": len(tasks_slice) - len(tasks_to_run),
            "to_run": len(tasks_to_run),
            "submitted": 0,
            "completed": 0,
            "success": 0,
            "failed": 0,
            "error": 0,
        }

        if request_rate_per_min <= 0:
            raise ValueError("request_rate_per_min must be > 0")
        interval = 60.0 / request_rate_per_min

        if max_workers is None:
            max_workers = max(32, min(300, len(tasks_to_run) if len(tasks_to_run) > 0 else 32))

        print(
            f"[Info] Parallel run: request_rate={request_rate_per_min}/min (interval={interval:.3f}s), "
            f"max_workers={max_workers}, in_flight=UNLIMITED(queueing allowed)"
        )

        pbar = tqdm(total=len(tasks_to_run), desc="Completed SWE-bench Lite", unit="task")

        futures = []
        next_submit_time = time.monotonic()

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                # 1) rate에 맞춰 submit
                for task in tasks_to_run:
                    now = time.monotonic()
                    sleep_s = next_submit_time - now
                    if sleep_s > 0:
                        time.sleep(sleep_s)

                    fut = ex.submit(self.run_single_task, task)
                    futures.append(fut)

                    stats["submitted"] += 1
                    next_submit_time += interval

                # 2) 완료 수집
                for fut in as_completed(futures):
                    result = fut.result()
                    stats["completed"] += 1

                    if result.get("error"):
                        stats["failed"] += 1
                        stats["error"] += 1
                        status = "ERROR"
                    elif result.get("success"):
                        stats["success"] += 1
                        status = "✓"
                    else:
                        stats["failed"] += 1
                        status = "✗"

                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "success": stats["success"],
                            "failed": stats["failed"],
                            "submitted": stats["submitted"],
                            "last_time": f"{result.get('total_time', -1):.1f}s",
                            "status": status,
                        }
                    )
        finally:
            pbar.close()
            # parser stop
            if self.vllm_parser:
                self.vllm_parser.stop()

        self._print_summary_parallel(stats, request_rate_per_min, max_workers)

    def _print_summary_parallel(self, stats: dict, request_rate_per_min: float, max_workers: int):
        print(f"\n{'='*60}")
        print("BATCH EXECUTION SUMMARY (PARALLEL)")
        print(f"{'='*60}")
        print(f"Tasks in range: {stats['total_in_range']}")
        print(f"Skipped (already done): {stats['skipped']}")
        print(f"To run: {stats['to_run']}")
        print(f"Submitted: {stats['submitted']}")
        print(f"Completed: {stats['completed']}")
        if stats["completed"] > 0:
            print(f"  - Success: {stats['success']} ({stats['success']/stats['completed']*100:.1f}%)")
            print(f"  - Failed: {stats['failed']} ({stats['failed']/stats['completed']*100:.1f}%)")
        else:
            print("  - Success: 0 (0.0%)")
            print("  - Failed: 0 (0.0%)")
        print(f"  - Error: {stats['error']}")
        print(f"Request rate: {request_rate_per_min}/min")
        print(f"Executor max_workers: {max_workers}")
        print(f"\nResults saved to: {self.csv_path}")
        print(f"Error log saved to: {self.error_log_path}")
        print(f"Agent logs saved to: {self.agent_log_dir}")
        print(f"{'='*60}\n")


def _parse_rate_list(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    rates: List[float] = []
    for p in parts:
        rates.append(float(p))
    if not rates:
        raise ValueError("request-rate-per-min-list is empty")
    return rates


def _safe_tag(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s).strip("_")


def main():
    parser = argparse.ArgumentParser(
        description="Run SWE-bench Lite batch evaluation (parallel + rate-limited starts + multi-run)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/nxc/mskim/agent/Agent_applications/agent_sglang_concurrent/results",
    )
    parser.add_argument("--server-log", type=str, default=None)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite")

    # 단일/멀티 rate
    parser.add_argument("--request-rate-per-min", type=float, default=None)
    parser.add_argument("--request-rate-per-min-list", type=str, default=None)

    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--run-id", type=str, default=None)

    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--server-base-url", type=str, default="http://localhost:30000")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    dataset = load_dataset(args.dataset, split="test")
    print(f"Loaded {len(dataset)} tasks\n")

    # rate list
    if args.request_rate_per_min_list:
        rate_list = _parse_rate_list(args.request_rate_per_min_list)
    else:
        rate_list = [args.request_rate_per_min if args.request_rate_per_min is not None else 60.0]

    repeat = max(1, args.repeat)
    run_id = _safe_tag(args.run_id) if args.run_id else None

    print(f"\n{'='*60}")
    print("SWE-BENCH LITE MULTI-RUN")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Rates: {rate_list} /min")
    print(f"Repeat per rate: {repeat}")
    print(f"Max workers: {args.max_workers}")
    print(f"Server base url: {args.server_base_url}")
    if run_id:
        print(f"Run ID: {run_id}")
    print(f"Output dir: {args.output_dir}")
    print(f"{'='*60}\n")

    for rate in rate_list:
        for rep in range(1, repeat + 1):
            tag_parts = [f"rpm{rate:g}", f"rep{rep}"]
            if run_id:
                tag_parts.append(run_id)
            tag = "_".join(tag_parts)

            csv_path = os.path.join(args.output_dir, f"metrics_{tag}.csv")
            error_log_path = os.path.join(args.output_dir, f"errors_{tag}.log")
            server_export_log_path = os.path.join(args.output_dir, f"server_{tag}.log")
            agent_log_root_dir = os.path.join(args.output_dir, "agent_logs", tag)

            print(f"\n{'='*60}")
            print(f"RUN: {tag}")
            print(f"{'='*60}")
            print(f"Request rate: {rate}/min")
            print(f"CSV: {csv_path}")
            print(f"Error log: {error_log_path}")
            print(f"Server export log: {server_export_log_path}")
            print(f"Agent log dir: {agent_log_root_dir}")
            print(f"{'='*60}\n")

            runner = SWEBenchRunner(
                csv_path=csv_path,
                error_log_path=error_log_path,
                agent_log_root_dir=agent_log_root_dir,
                server_log_path=args.server_log,
                server_export_log_path=server_export_log_path,
                server_base_url=args.server_base_url,
                max_iterations=args.max_iterations,
            )

            try:
                runner.run_batch_parallel(
                    dataset,
                    start_index=args.start_index,
                    end_index=args.end_index,
                    request_rate_per_min=rate,
                    max_workers=args.max_workers,
                )
            except KeyboardInterrupt:
                print("\n\n[!] Interrupted by user")
                print(f"Partial results saved to: {csv_path}")
                sys.exit(0)


if __name__ == "__main__":
    main()
