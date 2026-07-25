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
    # HALO: Project Halo request-level client wiring. Off by default.
    # When True, run_job passes the per-request halo_*_slo fields into
    # make_llm so every chat.completions request carries them in
    # extra_body. There is NO job pre-registration (the 2026-05-19
    # request-level refactor removed POST /halo/programs). See
    # workloads/halo_helpers.py and workloads/AGENTS.md.
    halo_enabled: bool = False
    halo_ttft_slo: Optional[float] = None
    halo_tbt_slo: Optional[float] = None
    halo_e2e_slo: Optional[float] = None
    # Transcript recording (literal-replay capture). When set, every LLM
    # call's full prompt + solo timings are appended to this JSONL so the
    # codingagent_request_level_poisson workload can replay them verbatim.
    transcript_record_path: Optional[str] = None
    # When True, the request-level workload disables every client-side
    # abort (τ / TTFT / idle timeout) so slow requests run to completion
    # and are measured rather than killed. Honored by the request-level
    # workload only. Defaults to False.
    disable_timeouts: bool = False
    # Engine wire protocol. "chat" (default) -> ChatOpenAI /v1/chat/completions
    # (SGLang). "completions" -> LlumnixCompletionsLLM /v1/completions (the
    # Llumnix gateway serves only completions). Passed into make_llm(api=...).
    api: str = "chat"
    # Optional model-id override (e.g. meta-llama/Meta-Llama-3-8B-Instruct for
    # the Llumnix deployment). When None, each workload's own MODEL_ID is used.
    model: Optional[str] = None
    # Output token cap for the completions API (vLLM defaults to 16). When
    # None, make_llm falls back to DEFAULT_MAX_TOKENS. Ignored by the chat API.
    max_tokens: Optional[int] = None
    # EDF scheduling (EXP-15): this request's SLO budget in ms. When set, the
    # completions client sends `priority = now_ms + slo_budget_ms` (an absolute
    # deadline) and the engine's priority policy schedules earliest-deadline
    # first. The mixed workload injects the per-class budget per request via
    # dataclasses.replace before delegating. None (default) = send no priority,
    # which is what FIFO needs and what SJF/SRPF want (those derive the
    # priority inside the engine from prompt length / remaining prefill).
    slo_budget_ms: Optional[int] = None
    # DeadlineScheduler (Niyama port): per-request ABSOLUTE SLO spec, e.g.
    # {"ttft_ms": 5000, "tbt_ms": 50} or {"e2e_ms": 20000}. The completions
    # client folds this into `priority = relative first-token-equivalent SLO
    # (ms)` via the tier rule (TTFT present -> interactive; E2E only -> TTLT
    # converted by e2e - out*tbt; none -> best-effort). Canonical rule:
    # patches/vllm-sched/slo_tier.py. Injected per class by the mixed workload.
    slo_spec: Optional[dict] = None
    # How the completions client stamps `priority`: "none" (send nothing —
    # FIFO/SJF/SRPF), "edf" (absolute deadline = now_ms + slo_budget_ms), or
    # "deadline" (relative first-token SLO from slo_spec, for DeadlineScheduler).
    priority_mode: str = "none"


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
