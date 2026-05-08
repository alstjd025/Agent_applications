"""Shared workload interfaces for motivation experiments."""

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Any
import threading


@dataclass
class RunContext:
    server_base_url: str
    seed: int
    log_level: str
    metrics_tracker: Any
    agent_logger: Any
    console_write: Callable[[str], None]
    server_terminated_event: threading.Event
    job_start_time: float
    parallel_calls_path: Optional[str] = None


@dataclass
class JobResult:
    job_id: str
    success: bool
    total_time: float
    calls_completed: int
    chain_length: int
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    error: Optional[str] = None
    is_job_timeout: bool = False
    is_server_terminated: bool = False
    job_timeout_sec: Optional[float] = None
    transition_time: Optional[float] = None
    is_rejected: bool = False
    rejection_reason: str = ""


@dataclass
class TaskLogInfo:
    task_id: str
    problem_statement: str = ""
    repo: str = ""


class BaseWorkload(Protocol):
    name: str

    def load_dataset(self, args, workload_config: dict):
        ...

    def build_baseline_tasks(self, dataset, replay_count: int, rng, args, workload_config: dict) -> list[dict]:
        ...

    def create_task_pool(self, dataset, baseline_latencies: dict[str, float], rng, args, workload_config: dict):
        ...

    def run_job(self, task: dict, context: RunContext) -> JobResult:
        ...

    def task_log_info(self, task: dict) -> TaskLogInfo:
        ...

    def metadata(self, args, workload_config: dict) -> dict:
        ...

    def reproducibility_config(self, args, workload_config: dict) -> dict:
        ...
