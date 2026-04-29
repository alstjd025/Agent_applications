#!/usr/bin/env python3
"""
Motivation Experiment Runner.

Supports four modes:
  1. baseline      – concurrency=1, collect per-job baseline latencies
  2. sweep         – concurrency sweep {1,2,4,8,16,24,32,48,64}
  3. rate-sweep    – RPM sweep {3,6,12,30,60}
  4. poisson-sweep – Poisson arrival with λ sweep {0.005,0.01,0.02,0.05,0.1,0.2}

Each job is a synthetic multi-call LLM chain (N in {5,30})
using SWE-bench Lite problem statements as initial context.

Duration-based: sweep/rate-sweep/poisson-sweep run for a fixed wall-clock
duration (--duration-min, default 60).  Jobs are generated on-the-fly from
a cycling pool of SWE tasks with replay IDs.  Between experiments the
SGLang server is restarted for clean KV cache state.
"""

import csv
import os
import sys
import time
import json
import argparse
import threading
import random
import subprocess
from datetime import datetime
from typing import Optional, List, Dict

from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from synthetic_coding_agent import (
    agent, ChainState, create_chain_state, make_llm, count_tokens,
    SYSTEM_PROMPT, BASE_URL, MODEL_ID, PER_CALL_TIMEOUT,
)
from metrics_tracker import MetricsTracker
from agent_logger import AgentLogger
from vllm_logger import VLLMLogParser, MockVLLMLogParser


# ---------------------------------------------------------------------------
# Chain-length sampling
# ---------------------------------------------------------------------------
CHAIN_MIN = 5
CHAIN_MAX = 30


def sample_chain_length(rng: random.Random) -> int:
    """Sample a chain length uniformly from [CHAIN_MIN, CHAIN_MAX]."""
    return rng.randint(CHAIN_MIN, CHAIN_MAX)


# ---------------------------------------------------------------------------
# Baseline latency loading
# ---------------------------------------------------------------------------
def load_baseline_latencies(baseline_dir: str) -> Dict[str, float]:
    """Load per-task baseline latency from a baseline run's metrics.csv.

    Returns:
        dict mapping base_instance_id -> baseline_latency_seconds.
        Only includes successfully completed jobs.
    """
    csv_path = os.path.join(baseline_dir, "metrics.csv")
    latencies: Dict[str, float] = {}

    if not os.path.exists(csv_path):
        print(f"WARNING: baseline metrics.csv not found at {csv_path}")
        return latencies

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("agent") == "job_summary" and row.get("job_completed") == "True":
                task_id = row["task_id"]
                latency = float(row["latency"]) if row.get("latency") else None
                if latency is not None:
                    # Strip replay suffix if present (e.g. astropy__astropy-12907__replay01)
                    base_id = task_id.split("__replay")[0] if "__replay" in task_id else task_id
                    latencies[base_id] = latency

    print(f"Loaded {len(latencies)} baseline latencies from {baseline_dir}")
    return latencies


# ---------------------------------------------------------------------------
# SGLang server control
# ---------------------------------------------------------------------------
class SGLangServerController:
    """Control the SGLang server on NXC7 via SSH.

    - start(): Launch the server and wait until it's ready.
    - stop(): Kill the server.
    """

    def __init__(
        self,
        ssh_host: str = "NXC7",
        server_start_cmd: str = "cd /home/nxclab/sglang/ms_dev/expctl && python3 run_pd_experiment.py --mode single --single-port 31000",
        server_stop_cmd: str = "bash /home/nxclab/sglang/ms_dev/stop_servers.sh",
        tmux_session: str = "sglang",
        local_port: int = 8080,
        poll_interval: int = 5,
        start_timeout: int = 600,
    ):
        self.ssh_host = ssh_host
        self.server_start_cmd = server_start_cmd
        self.server_stop_cmd = server_stop_cmd
        self.tmux_session = tmux_session
        self.local_port = local_port
        self.poll_interval = poll_interval
        self.start_timeout = start_timeout

    def _ssh(self, cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["ssh", self.ssh_host, cmd],
            capture_output=True, text=True, timeout=timeout,
        )

    def start(self) -> bool:
        """Start the SGLang server. Returns True if server becomes ready."""
        print(f"[SGLangServer] Starting server on {self.ssh_host}...")
        try:
            self._ssh(f"tmux send-keys -t {self.tmux_session} '{self.server_start_cmd}' Enter",
                      timeout=15)
        except Exception as e:
            print(f"[SGLangServer] Failed to send start command: {e}")
            return False

        # Poll until server is ready
        import requests
        deadline = time.time() + self.start_timeout
        while time.time() < deadline:
            try:
                resp = requests.get(f"http://localhost:{self.local_port}/v1/models", timeout=3)
                if resp.status_code == 200:
                    print(f"[SGLangServer] Server ready ({time.time():.0f})")
                    return True
            except Exception:
                pass
            time.sleep(self.poll_interval)

        print(f"[SGLangServer] Server failed to start within {self.start_timeout}s")
        return False

    def stop(self) -> bool:
        """Stop the SGLang server."""
        print(f"[SGLangServer] Stopping server on {self.ssh_host}...")
        try:
            result = self._ssh(self.server_stop_cmd, timeout=30)
            print(f"[SGLangServer] stop_servers.sh exit={result.returncode}")
            # Give it a moment
            time.sleep(3)
            return True
        except Exception as e:
            print(f"[SGLangServer] Failed to stop server: {e}")
            return False


# ---------------------------------------------------------------------------
# Task pool for duration-based experiments
# ---------------------------------------------------------------------------
class CyclingTaskPool:
    """Generate tasks on-the-fly from a cycling pool of SWE problems.

    Each task gets a unique replay ID and nonce to prevent KV cache reuse
    while keeping the system prompt identical across all requests.
    """

    def __init__(self, dataset, baseline_latencies: Dict[str, float],
                 rng: random.Random, tau: float = 2.0):
        self.dataset = dataset
        self.baseline_latencies = baseline_latencies
        self.rng = rng
        self.tau = tau
        self._replay_counter = 0
        self._lock = threading.Lock()

    def next_task(self) -> Optional[dict]:
        """Get the next task from the cycling pool.

        Returns None if the task has no baseline latency (shouldn't happen
        with a properly filtered dataset).
        """
        with self._lock:
            base_idx = self._replay_counter % len(self.dataset)
            replay_num = self._replay_counter // len(self.dataset) + 1
            chain_length = sample_chain_length(self.rng)
            self._replay_counter += 1

        task = self.dataset[base_idx]
        base_task_id = task["instance_id"]
        logical_id = f"{base_task_id}__replay{replay_num:02d}"
        nonce = f"replay{replay_num:02d}-idx{base_idx:04d}-{base_task_id}"

        # Per-job timeout from baseline latency × τ
        baseline_latency = self.baseline_latencies.get(base_task_id)
        job_timeout_sec = baseline_latency * self.tau if baseline_latency else 0

        return {
            "instance_id": logical_id,
            "base_instance_id": base_task_id,
            "problem_statement": task["problem_statement"],
            "repo": task.get("repo", ""),
            "replay_index": replay_num,
            "logical_index": base_idx,
            "nonce": nonce,
            "chain_length": chain_length,
            "job_timeout_sec": job_timeout_sec,
            "baseline_latency": baseline_latency,
        }


# ---------------------------------------------------------------------------
# Logical task builder (for baseline mode — fixed task set)
# ---------------------------------------------------------------------------
def build_logical_tasks(dataset, replay_count: int, rng: random.Random) -> List[dict]:
    """Build logical task list from the dataset for baseline mode."""
    tasks = []
    base_size = len(dataset)
    for replay_idx in range(replay_count):
        replay_num = replay_idx + 1
        for base_idx, task in enumerate(dataset):
            base_task_id = task["instance_id"]
            logical_id = f"{base_task_id}__replay{replay_num:02d}"
            logical_position = replay_idx * base_size + base_idx
            nonce = f"replay{replay_num:02d}-idx{logical_position:04d}-{base_task_id}"

            tasks.append({
                "instance_id": logical_id,
                "base_instance_id": base_task_id,
                "problem_statement": task["problem_statement"],
                "repo": task.get("repo", ""),
                "replay_index": replay_num,
                "logical_index": logical_position,
                "nonce": nonce,
                "chain_length": sample_chain_length(rng),
                "job_timeout_sec": 0,  # No job timeout in baseline
                "baseline_latency": None,
            })
    return tasks


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------
class MotivationExperimentRunner:
    """Run motivation experiment with synthetic multi-call chains."""

    def __init__(
        self,
        csv_path: str,
        error_log_path: str,
        agent_log_root_dir: str,
        tbt_jsonl_path: str,
        server_base_url: str = "http://localhost:8080",
        max_iterations: int = 1,
        log_level: str = "quiet",
    ):
        self.csv_path = csv_path
        self.error_log_path = error_log_path
        self.server_base_url = server_base_url
        self.max_iterations = max_iterations
        self.log_level = log_level
        self.tbt_jsonl_path = tbt_jsonl_path
        self._console_lock = threading.Lock()
        self._pbar: Optional[tqdm] = None
        self._server_terminated = threading.Event()

        self.agent_log_dir = agent_log_root_dir
        os.makedirs(self.agent_log_dir, exist_ok=True)

        os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
        with open(error_log_path, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Experiment run started at {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n\n")

        # CSV header
        _ = MetricsTracker(
            self.csv_path,
            server_base_url=self.server_base_url,
            enable_server_metrics=True,
            tbt_jsonl_path=self.tbt_jsonl_path,
        )

        # vLLM log parser (optional)
        self.vllm_parser = MockVLLMLogParser()
        self.vllm_parser.start()

    def _console_write(self, message: str) -> None:
        with self._console_lock:
            if self._pbar is not None:
                self._pbar.write(message)
            else:
                print(message)

    def _log_error(self, job_id: str, error_msg: str):
        with open(self.error_log_path, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {job_id}\n")
            f.write(f"  {error_msg}\n\n")
        self._console_write(f"[ERROR] {job_id}: {error_msg}")

    def run_single_job(self, task: dict) -> dict:
        """Execute a single multi-call chain job."""
        job_id = task["instance_id"]

        metrics_tracker = MetricsTracker(
            self.csv_path,
            server_base_url=self.server_base_url,
            enable_server_metrics=True,
            tbt_jsonl_path=self.tbt_jsonl_path,
        )
        metrics_tracker.start_task(job_id)

        agent_logger = AgentLogger(self.agent_log_dir)
        agent_logger.start_task(
            task_id=job_id,
            problem_statement=task["problem_statement"],
            repo=task.get("repo", ""),
        )

        llm = make_llm(base_url=f"{self.server_base_url}/v1", model_id=MODEL_ID)

        job_submit_time = time.time()
        job_timeout_sec = task.get("job_timeout_sec", 0)

        initial_state = create_chain_state(
            job_id=job_id,
            problem_statement=task["problem_statement"],
            chain_length=task["chain_length"],
            nonce=task["nonce"],
            metrics_tracker=metrics_tracker,
            agent_logger=agent_logger,
            console_write=self._console_write,
            llm=llm,
            log_level=self.log_level,
            job_timeout_sec=job_timeout_sec,
            job_start_time=job_submit_time,
        )

        # Attach server terminated event
        initial_state["server_terminated_event"] = self._server_terminated

        try:
            result = agent.invoke(initial_state)
            job_end_time = time.time()

            job_completed = result.get("job_completed", False)
            call_index = result.get("call_index", 0)
            chain_length = task["chain_length"]
            error_msg = result.get("error_msg", "")
            is_job_timeout = result.get("is_job_timeout", False)
            is_server_terminated = result.get("is_server_terminated", False)

            total_in = result.get("total_input_tokens", 0)
            total_out = result.get("total_output_tokens", 0)

            agent_logger.log_final_result(
                success=job_completed,
                total_time=job_end_time - job_submit_time,
                iterations=call_index,
            )

            # Record job summary
            metrics_tracker.record_job_summary(
                job_id=job_id,
                chain_length=chain_length,
                calls_completed=call_index - 1 if not job_completed and call_index > 0 else call_index,
                job_completed=job_completed,
                job_submit_time=job_submit_time,
                job_end_time=job_end_time,
                total_input_tokens=total_in,
                total_output_tokens=total_out,
                wasted_input_tokens=total_in if not job_completed else 0,
                wasted_output_tokens=total_out if not job_completed else 0,
                error_msg=error_msg,
                is_job_timeout=is_job_timeout,
                job_timeout_sec=job_timeout_sec if job_timeout_sec > 0 else None,
                is_server_terminated=is_server_terminated,
            )

            return {
                "job_id": job_id,
                "success": job_completed,
                "total_time": job_end_time - job_submit_time,
                "calls_completed": call_index,
                "chain_length": chain_length,
                "error": error_msg if error_msg else None,
                "is_job_timeout": is_job_timeout,
                "is_server_terminated": is_server_terminated,
            }

        except Exception as e:
            error_msg = f"Error: {type(e).__name__}: {str(e)}"
            self._log_error(job_id, error_msg)

            return {
                "job_id": job_id,
                "success": False,
                "total_time": -1,
                "calls_completed": 0,
                "chain_length": task["chain_length"],
                "error": error_msg,
                "is_job_timeout": False,
                "is_server_terminated": self._server_terminated.is_set(),
            }

    def signal_server_terminated(self):
        """Signal all in-flight jobs that the server has been terminated."""
        self._server_terminated.set()
        self._console_write("[Runner] Server terminated signal sent to all jobs")

    def run_baseline(self, tasks: List[dict]):
        """Run baseline at concurrency=1 (fixed task set, no time limit)."""
        self._run_with_concurrency(tasks, concurrency=1)

    def run_concurrency_sweep(
        self,
        tasks: List[dict],
        concurrency_levels: List[int],
    ):
        """Run experiment at multiple concurrency levels (fixed task set)."""
        for level in concurrency_levels:
            self._console_write(f"\n{'='*60}")
            self._console_write(f"CONCURRENCY LEVEL: {level}")
            self._console_write(f"{'='*60}")

            self._run_with_concurrency(tasks, concurrency=level)

    def run_rate_sweep_duration(
        self,
        task_pool: CyclingTaskPool,
        rates: List[float],
        duration_min: float,
    ):
        """Run rate-sweep with duration-based execution and server restart."""
        for rate in rates:
            self._console_write(f"\n{'='*60}")
            self._console_write(f"RATE: {rate} RPM (duration={duration_min}min)")
            self._console_write(f"{'='*60}")

            self._run_with_rate_duration(task_pool, rate, duration_min)

    def run_poisson_sweep_duration(
        self,
        task_pool: CyclingTaskPool,
        lambdas: List[float],
        duration_min: float,
    ):
        """Run poisson-sweep with duration-based execution and server restart."""
        for lam in lambdas:
            self._console_write(f"\n{'='*60}")
            self._console_write(f"POISSON λ={lam} (duration={duration_min}min)")
            self._console_write(f"{'='*60}")

            self._run_with_poisson_duration(task_pool, lam, duration_min)

    def _run_with_concurrency(self, tasks: List[dict], concurrency: int):
        """Run all tasks with a fixed concurrency cap (baseline mode)."""
        completed_jobs = MetricsTracker.load_completed_tasks(self.csv_path)
        tasks_to_run = [t for t in tasks if t["instance_id"] not in completed_jobs]

        stats = {
            "submitted": 0, "completed": 0,
            "success": 0, "failed": 0, "error": 0,
        }

        self._pbar = tqdm(total=len(tasks_to_run),
                          desc=f"Concurrency={concurrency}", unit="job")

        pending = set()

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            task_iter = iter(tasks_to_run)

            for _ in range(min(concurrency, len(tasks_to_run))):
                try:
                    task = next(task_iter)
                except StopIteration:
                    break
                fut = ex.submit(self.run_single_job, task)
                pending.add(fut)
                stats["submitted"] += 1

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)

                for fut in done:
                    result = fut.result()
                    stats["completed"] += 1
                    if result.get("error"):
                        stats["error"] += 1
                        stats["failed"] += 1
                    elif result.get("success"):
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1

                    self._pbar.update(1)

                while len(pending) < concurrency:
                    try:
                        task = next(task_iter)
                    except StopIteration:
                        break
                    fut = ex.submit(self.run_single_job, task)
                    pending.add(fut)
                    stats["submitted"] += 1

        if self._pbar:
            self._pbar.close()
            self._pbar = None

        self._print_summary(stats, mode="concurrency", level=concurrency)

    def _run_with_rate_duration(
        self,
        task_pool: CyclingTaskPool,
        rate_per_min: float,
        duration_min: float,
    ):
        """Run tasks at a fixed rate for a fixed duration.

        Uses CyclingTaskPool to generate tasks on-the-fly.
        No concurrency cap — all submitted tasks run in parallel.
        """
        self._server_terminated.clear()

        interval = 60.0 / rate_per_min
        duration_sec = duration_min * 60.0
        max_workers = 256

        stats = {
            "submitted": 0, "completed": 0,
            "success": 0, "failed": 0, "error": 0,
            "job_timeout": 0, "server_terminated": 0,
        }

        self._pbar = tqdm(total=0, desc=f"RPM={rate_per_min:.0f}", unit="job",
                          bar_format="{desc}: {n} submitted, {postfix}")

        pending = set()
        experiment_start = time.monotonic()
        next_submit_time = time.monotonic()

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                while True:
                    now = time.monotonic()
                    elapsed = now - experiment_start

                    # Check duration — signal BEFORE breaking so in-flight
                    # jobs see the event while the executor is still alive
                    if elapsed >= duration_sec:
                        self.signal_server_terminated()
                        break

                    # Submit tasks at the specified rate
                    while now >= next_submit_time and (now - experiment_start) < duration_sec:
                        task = task_pool.next_task()
                        if task is None:
                            break
                        fut = ex.submit(self.run_single_job, task)
                        pending.add(fut)
                        stats["submitted"] += 1
                        next_submit_time += interval
                        now = time.monotonic()

                    if pending:
                        timeout = max(0.1, min(1.0, next_submit_time - time.monotonic()))
                        done, pending = wait(pending, timeout=timeout,
                                             return_when=FIRST_COMPLETED)

                        for fut in done:
                            result = fut.result()
                            self._update_stats(stats, result)

                    # Update progress
                    if self._pbar:
                        self._pbar.total = stats["submitted"]
                        self._pbar.postfix = (
                            f"{stats['completed']} done, "
                            f"{stats['success']} ok, "
                            f"{stats['failed']} fail, "
                            f"{stats['job_timeout']} τ-timeout, "
                            f"{stats['server_terminated']} srv-kill"
                        )
                        self._pbar.refresh()

                    if not pending and now >= next_submit_time:
                        # Wait for next submit time
                        sleep_s = next_submit_time - time.monotonic()
                        if sleep_s > 0 and (time.monotonic() - experiment_start) < duration_sec:
                            time.sleep(min(sleep_s, 1.0))
        finally:
            # Wait briefly for in-flight jobs to finish
            time.sleep(3)
            if self._pbar:
                self._pbar.close()
                self._pbar = None
            MetricsTracker.shutdown_all_writers()

        self._print_summary(stats, mode="rate", level=rate_per_min)

    def _run_with_poisson_duration(
        self,
        task_pool: CyclingTaskPool,
        lam: float,
        duration_min: float,
    ):
        """Run tasks with Poisson arrival (rate λ) for a fixed duration.

        Inter-arrival times are exponentially distributed with mean 1/λ seconds.
        """
        self._server_terminated.clear()

        duration_sec = duration_min * 60.0
        max_workers = 256
        rng = random.Random(42)  # For Poisson inter-arrival sampling

        stats = {
            "submitted": 0, "completed": 0,
            "success": 0, "failed": 0, "error": 0,
            "job_timeout": 0, "server_terminated": 0,
        }

        self._pbar = tqdm(total=0, desc=f"λ={lam}", unit="job",
                          bar_format="{desc}: {n} submitted, {postfix}")

        pending = set()
        experiment_start = time.monotonic()
        # First inter-arrival time
        next_submit_time = experiment_start + rng.expovariate(lam)

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                while True:
                    now = time.monotonic()
                    elapsed = now - experiment_start

                    # Check duration — signal BEFORE breaking so in-flight
                    # jobs see the event while the executor is still alive
                    if elapsed >= duration_sec:
                        self.signal_server_terminated()
                        break

                    # Submit tasks according to Poisson process
                    while now >= next_submit_time and (now - experiment_start) < duration_sec:
                        task = task_pool.next_task()
                        if task is None:
                            break
                        fut = ex.submit(self.run_single_job, task)
                        pending.add(fut)
                        stats["submitted"] += 1
                        next_submit_time += rng.expovariate(lam)
                        now = time.monotonic()

                    if pending:
                        timeout = max(0.1, min(1.0, next_submit_time - time.monotonic()))
                        done, pending = wait(pending, timeout=timeout,
                                             return_when=FIRST_COMPLETED)

                        for fut in done:
                            result = fut.result()
                            self._update_stats(stats, result)

                    # Update progress
                    if self._pbar:
                        self._pbar.total = stats["submitted"]
                        self._pbar.postfix = (
                            f"{stats['completed']} done, "
                            f"{stats['success']} ok, "
                            f"{stats['failed']} fail, "
                            f"{stats['job_timeout']} τ-timeout, "
                            f"{stats['server_terminated']} srv-kill"
                        )
                        self._pbar.refresh()

                    if not pending and time.monotonic() < next_submit_time:
                        sleep_s = next_submit_time - time.monotonic()
                        if sleep_s > 0 and (time.monotonic() - experiment_start) < duration_sec:
                            time.sleep(min(sleep_s, 1.0))
        finally:
            time.sleep(3)
            if self._pbar:
                self._pbar.close()
                self._pbar = None
            MetricsTracker.shutdown_all_writers()

        self._print_summary(stats, mode="poisson", level=lam)

    @staticmethod
    def _update_stats(stats: dict, result: dict):
        stats["completed"] += 1
        if result.get("error"):
            stats["failed"] += 1
            stats["error"] += 1
        elif result.get("success"):
            stats["success"] += 1
        else:
            stats["failed"] += 1

        if result.get("is_job_timeout"):
            stats["job_timeout"] += 1
        if result.get("is_server_terminated"):
            stats["server_terminated"] += 1

    def _print_summary(self, stats: dict, mode: str, level):
        print(f"\n{'='*60}")
        print(f"SUMMARY ({mode}={level})")
        print(f"{'='*60}")
        print(f"Submitted: {stats['submitted']}")
        print(f"Completed: {stats['completed']}")
        if stats["completed"] > 0:
            print(f"  Success: {stats['success']} ({stats['success']/stats['completed']*100:.1f}%)")
            print(f"  Failed:  {stats['failed']} ({stats['failed']/stats['completed']*100:.1f}%)")
        print(f"  Errors:  {stats['error']}")
        if "job_timeout" in stats:
            print(f"  τ-timeout: {stats['job_timeout']}")
        if "server_terminated" in stats:
            print(f"  Server killed: {stats['server_terminated']}")
        print(f"Results: {self.csv_path}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Run directory setup
# ---------------------------------------------------------------------------
def setup_run_dir(base_dir: str, tag: str, resume_dir: str = None) -> dict:
    """Create a run directory with standard subdirectories."""
    if resume_dir:
        run_dir = resume_dir
        print(f"Resuming from existing directory: {run_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(base_dir, f"{tag}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    return {
        "run_dir": run_dir,
        "csv_path": os.path.join(run_dir, "metrics.csv"),
        "error_log": os.path.join(run_dir, "errors.log"),
        "tbt_jsonl": os.path.join(run_dir, "tbt_events.jsonl"),
        "agent_logs": os.path.join(run_dir, "agent_logs"),
    }


def write_run_config(run_dir: str, config: dict):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Motivation Experiment: Throughput Peak vs. Goodput Collapse"
    )

    # Mode
    parser.add_argument(
        "--mode",
        choices=["baseline", "sweep", "rate-sweep", "poisson-sweep", "single"],
        required=True,
        help="Experiment mode",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/nxc/mskim/agent/Agent_applications/agent_motivation_experiment/results",
    )

    # Dataset
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--replay-count", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)

    # Chain configuration
    parser.add_argument("--chain-min", type=int, default=5)
    parser.add_argument("--chain-max", type=int, default=30)

    # Server
    parser.add_argument("--server-base-url", type=str, default="http://localhost:8080")

    # Duration (for sweep/rate-sweep/poisson-sweep)
    parser.add_argument(
        "--duration-min",
        type=float,
        default=60,
        help="Duration of each experiment in minutes (for rate/poisson sweep)",
    )

    # Job timeout (τ)
    parser.add_argument(
        "--tau",
        type=float,
        default=2.0,
        help="Job timeout multiplier: job_timeout = baseline_latency × τ",
    )

    # Baseline directory (for loading baseline latencies)
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default=None,
        help="Path to baseline run directory (for loading per-job baseline latencies)",
    )

    # Concurrency sweep
    parser.add_argument(
        "--concurrency-list",
        type=str,
        default="1,2,4,8,16,24,32,48,64",
        help="Comma-separated concurrency levels for --mode sweep",
    )

    # Rate sweep
    parser.add_argument(
        "--rate-list",
        type=str,
        default="3,6,12,30,60",
        help="Comma-separated RPM rates for --mode rate-sweep",
    )

    # Poisson sweep
    parser.add_argument(
        "--lambda-list",
        type=str,
        default="0.005,0.01,0.02,0.05,0.1,0.2",
        help="Comma-separated Poisson λ values for --mode poisson-sweep",
    )

    # SGLang server control
    parser.add_argument(
        "--sglang-ssh-host",
        type=str,
        default="NXC7",
        help="SSH host alias for SGLang server",
    )
    parser.add_argument(
        "--sglang-start-cmd",
        type=str,
        default="cd /home/nxclab/sglang/ms_dev/expctl && python3 run_pd_experiment.py --mode single --single-port 31000",
        help="Command to start the SGLang server",
    )
    parser.add_argument(
        "--sglang-stop-cmd",
        type=str,
        default="bash /home/nxclab/sglang/ms_dev/stop_servers.sh",
        help="Command to stop the SGLang server",
    )
    parser.add_argument(
        "--sglang-tmux-session",
        type=str,
        default="sglang",
        help="tmux session name on the remote server",
    )
    parser.add_argument(
        "--no-server-restart",
        action="store_true",
        help="Skip server restart between experiments (use if server is already running)",
    )

    # Single mode
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--rpm", type=float, default=None)
    parser.add_argument("--lambda-val", type=float, default=None)

    # Resume
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Path to an existing run directory to resume from.",
    )

    # Misc
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", choices=["quiet", "info", "debug"], default="quiet")

    args = parser.parse_args()

    # Chain range
    global CHAIN_MIN, CHAIN_MAX
    CHAIN_MIN = args.chain_min
    CHAIN_MAX = args.chain_max

    concurrency_list = [int(x.strip()) for x in args.concurrency_list.split(",")]
    rate_list = [float(x.strip()) for x in args.rate_list.split(",")]
    lambda_list = [float(x.strip()) for x in args.lambda_list.split(",")]

    rng = random.Random(args.seed)

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.dataset, split="test")
    print(f"Loaded {len(dataset)} tasks")

    # Check system prompt token count
    sp_tokens = count_tokens(SYSTEM_PROMPT)
    print(f"System prompt tokens: {sp_tokens}")
    print(f"Chain lengths: [{CHAIN_MIN}, {CHAIN_MAX}] (uniform integer)")
    print(f"Mode: {args.mode}")
    print(f"τ = {args.tau}")

    # Load baseline latencies
    baseline_latencies = {}
    if args.baseline_dir:
        baseline_latencies = load_baseline_latencies(args.baseline_dir)

    # SGLang server controller
    sglang_ctrl = SGLangServerController(
        ssh_host=args.sglang_ssh_host,
        server_start_cmd=args.sglang_start_cmd,
        server_stop_cmd=args.sglang_stop_cmd,
        tmux_session=args.sglang_tmux_session,
    )

    # Verify server connectivity for baseline mode
    if args.mode == "baseline":
        try:
            import requests
            resp = requests.get(f"{args.server_base_url}/v1/models", timeout=5)
            models = resp.json()
            model_ids = [m["id"] for m in models.get("data", [])]
            print(f"Server models: {model_ids}")
        except Exception as e:
            print(f"WARNING: Cannot reach server at {args.server_base_url}: {e}")
            print("Make sure the SGLang server is running!")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- BASELINE MODE ----
    if args.mode == "baseline":
        tasks = build_logical_tasks(dataset, args.replay_count, rng)
        if args.end_index:
            tasks = tasks[args.start_index:args.end_index]
        print(f"Total jobs: {len(tasks)}")

        paths = setup_run_dir(args.output_dir, "baseline", resume_dir=args.resume_dir)
        write_run_config(paths["run_dir"], {
            "mode": "baseline",
            "concurrency": 1,
            "chain_range": [CHAIN_MIN, CHAIN_MAX],
            "dataset": args.dataset,
            "replay_count": args.replay_count,
            "total_jobs": len(tasks),
            "server_base_url": args.server_base_url,
            "seed": args.seed,
            "created_at": datetime.now().isoformat(),
        })

        runner = MotivationExperimentRunner(
            csv_path=paths["csv_path"],
            error_log_path=paths["error_log"],
            agent_log_root_dir=paths["agent_logs"],
            tbt_jsonl_path=paths["tbt_jsonl"],
            server_base_url=args.server_base_url,
            log_level=args.log_level,
        )
        runner.run_baseline(tasks)

    # ---- CONCURRENCY SWEEP MODE ----
    elif args.mode == "sweep":
        tasks = build_logical_tasks(dataset, args.replay_count, rng)
        if args.end_index:
            tasks = tasks[args.start_index:args.end_index]
        print(f"Total jobs: {len(tasks)}")

        for level in concurrency_list:
            tag = f"concurrency_{level}"
            paths = setup_run_dir(args.output_dir, tag, resume_dir=args.resume_dir)
            write_run_config(paths["run_dir"], {
                "mode": "sweep",
                "concurrency": level,
                "chain_range": [CHAIN_MIN, CHAIN_MAX],
                "dataset": args.dataset,
                "replay_count": args.replay_count,
                "total_jobs": len(tasks),
                "server_base_url": args.server_base_url,
                "seed": args.seed,
                "created_at": datetime.now().isoformat(),
            })

            runner = MotivationExperimentRunner(
                csv_path=paths["csv_path"],
                error_log_path=paths["error_log"],
                agent_log_root_dir=paths["agent_logs"],
                tbt_jsonl_path=paths["tbt_jsonl"],
                server_base_url=args.server_base_url,
                log_level=args.log_level,
            )
            runner._run_with_concurrency(tasks, concurrency=level)

    # ---- RATE SWEEP MODE (duration-based) ----
    elif args.mode == "rate-sweep":
        if not baseline_latencies:
            print("ERROR: --baseline-dir is required for rate-sweep mode")
            sys.exit(1)

        task_pool = CyclingTaskPool(dataset, baseline_latencies, rng, tau=args.tau)
        sweep_start = time.monotonic()
        total_experiments = len(rate_list)

        for i, rate in enumerate(rate_list):
            # Sweep progress
            elapsed_sofar = time.monotonic() - sweep_start
            est_per_exp = elapsed_sofar / (i + 1) if i > 0 else args.duration_min * 60 + 360
            est_remaining = est_per_exp * (total_experiments - i - 1)
            print(f"\n{'='*60}")
            print(f"  RATE SWEEP [{i+1}/{total_experiments}] rate={rate} RPM | "
                  f"~{est_remaining/3600:.1f}h remaining")
            print(f"{'='*60}")

            # Restart server between experiments
            if not args.no_server_restart:
                print(f"[Server] Restarting SGLang server...")
                sglang_ctrl.stop()
                time.sleep(2)
                if not sglang_ctrl.start():
                    print(f"ERROR: Failed to start server for rate={rate}")
                    sys.exit(1)

            tag = f"rpm_{rate:g}"
            paths = setup_run_dir(args.output_dir, tag)
            write_run_config(paths["run_dir"], {
                "mode": "rate-sweep",
                "request_rate_per_min": rate,
                "duration_min": args.duration_min,
                "tau": args.tau,
                "chain_range": [CHAIN_MIN, CHAIN_MAX],
                "dataset": args.dataset,
                "server_base_url": args.server_base_url,
                "seed": args.seed,
                "baseline_dir": args.baseline_dir,
                "created_at": datetime.now().isoformat(),
            })

            runner = MotivationExperimentRunner(
                csv_path=paths["csv_path"],
                error_log_path=paths["error_log"],
                agent_log_root_dir=paths["agent_logs"],
                tbt_jsonl_path=paths["tbt_jsonl"],
                server_base_url=args.server_base_url,
                log_level=args.log_level,
            )
            runner.run_rate_sweep_duration(
                task_pool=task_pool,
                rates=[rate],
                duration_min=args.duration_min,
            )

            time.sleep(2)

    # ---- POISSON SWEEP MODE (duration-based) ----
    elif args.mode == "poisson-sweep":
        if not baseline_latencies:
            print("ERROR: --baseline-dir is required for poisson-sweep mode")
            sys.exit(1)

        task_pool = CyclingTaskPool(dataset, baseline_latencies, rng, tau=args.tau)
        sweep_start = time.monotonic()
        total_experiments = len(lambda_list)

        for i, lam in enumerate(lambda_list):
            # Sweep progress
            elapsed_sofar = time.monotonic() - sweep_start
            est_per_exp = elapsed_sofar / (i + 1) if i > 0 else args.duration_min * 60 + 360
            est_remaining = est_per_exp * (total_experiments - i - 1)
            print(f"\n{'='*60}")
            print(f"  POISSON SWEEP [{i+1}/{total_experiments}] λ={lam} | "
                  f"~{est_remaining/3600:.1f}h remaining")
            print(f"{'='*60}")

            # Restart server between experiments
            if not args.no_server_restart:
                print(f"[Server] Restarting SGLang server...")
                sglang_ctrl.stop()
                time.sleep(2)
                if not sglang_ctrl.start():
                    print(f"ERROR: Failed to start server for λ={lam}")
                    sys.exit(1)

            tag = f"poisson_{lam:g}"
            paths = setup_run_dir(args.output_dir, tag)
            write_run_config(paths["run_dir"], {
                "mode": "poisson-sweep",
                "lambda": lam,
                "duration_min": args.duration_min,
                "tau": args.tau,
                "chain_range": [CHAIN_MIN, CHAIN_MAX],
                "dataset": args.dataset,
                "server_base_url": args.server_base_url,
                "seed": args.seed,
                "baseline_dir": args.baseline_dir,
                "created_at": datetime.now().isoformat(),
            })

            runner = MotivationExperimentRunner(
                csv_path=paths["csv_path"],
                error_log_path=paths["error_log"],
                agent_log_root_dir=paths["agent_logs"],
                tbt_jsonl_path=paths["tbt_jsonl"],
                server_base_url=args.server_base_url,
                log_level=args.log_level,
            )
            runner.run_poisson_sweep_duration(
                task_pool=task_pool,
                lambdas=[lam],
                duration_min=args.duration_min,
            )

            time.sleep(2)

    # ---- SINGLE MODE ----
    elif args.mode == "single":
        if args.lambda_val is not None and baseline_latencies:
            # Poisson single run
            task_pool = CyclingTaskPool(dataset, baseline_latencies, rng, tau=args.tau)
            paths = setup_run_dir(args.output_dir, "single", resume_dir=args.resume_dir)
            config = {
                "mode": "single-poisson",
                "lambda": args.lambda_val,
                "duration_min": args.duration_min,
                "tau": args.tau,
                "chain_range": [CHAIN_MIN, CHAIN_MAX],
                "dataset": args.dataset,
                "server_base_url": args.server_base_url,
                "seed": args.seed,
                "created_at": datetime.now().isoformat(),
            }
            write_run_config(paths["run_dir"], config)

            runner = MotivationExperimentRunner(
                csv_path=paths["csv_path"],
                error_log_path=paths["error_log"],
                agent_log_root_dir=paths["agent_logs"],
                tbt_jsonl_path=paths["tbt_jsonl"],
                server_base_url=args.server_base_url,
                log_level=args.log_level,
            )
            runner._run_with_poisson_duration(task_pool, args.lambda_val, args.duration_min)

        elif args.rpm is not None and baseline_latencies:
            # Rate single run
            task_pool = CyclingTaskPool(dataset, baseline_latencies, rng, tau=args.tau)
            paths = setup_run_dir(args.output_dir, "single", resume_dir=args.resume_dir)
            config = {
                "mode": "single-rate",
                "request_rate_per_min": args.rpm,
                "duration_min": args.duration_min,
                "tau": args.tau,
                "chain_range": [CHAIN_MIN, CHAIN_MAX],
                "dataset": args.dataset,
                "server_base_url": args.server_base_url,
                "seed": args.seed,
                "created_at": datetime.now().isoformat(),
            }
            write_run_config(paths["run_dir"], config)

            runner = MotivationExperimentRunner(
                csv_path=paths["csv_path"],
                error_log_path=paths["error_log"],
                agent_log_root_dir=paths["agent_logs"],
                tbt_jsonl_path=paths["tbt_jsonl"],
                server_base_url=args.server_base_url,
                log_level=args.log_level,
            )
            runner._run_with_rate_duration(task_pool, args.rpm, args.duration_min)

        else:
            # Concurrency single run (legacy)
            tasks = build_logical_tasks(dataset, args.replay_count, rng)
            if args.end_index:
                tasks = tasks[args.start_index:args.end_index]

            paths = setup_run_dir(args.output_dir, "single", resume_dir=args.resume_dir)
            config = {
                "mode": "single",
                "concurrency": args.concurrency,
                "chain_range": [CHAIN_MIN, CHAIN_MAX],
                "dataset": args.dataset,
                "replay_count": args.replay_count,
                "total_jobs": len(tasks),
                "server_base_url": args.server_base_url,
                "seed": args.seed,
                "created_at": datetime.now().isoformat(),
            }
            write_run_config(paths["run_dir"], config)

            runner = MotivationExperimentRunner(
                csv_path=paths["csv_path"],
                error_log_path=paths["error_log"],
                agent_log_root_dir=paths["agent_logs"],
                tbt_jsonl_path=paths["tbt_jsonl"],
                server_base_url=args.server_base_url,
                log_level=args.log_level,
            )
            runner._run_with_concurrency(tasks, concurrency=args.concurrency)


if __name__ == "__main__":
    main()
