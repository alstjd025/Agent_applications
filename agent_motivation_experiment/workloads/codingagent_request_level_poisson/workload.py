"""Request-level Poisson coding-agent workload.

Unlike the multi-call job workloads (swe_bench_coding*), this workload
treats every recorded agent LLM call as an **independent request**. The
runner submits these requests as a Poisson process (lambda = requests/sec),
matching the SGLang server's request-level Halo admission gate.

Input is a transcript JSONL produced by a concurrency-1 baseline run of
`swe_bench_coding` with `--record-transcript`. Each line carries the full
prompt of one agent call plus its solo (concurrency-1) timings, which
serve as the per-request goodput baseline. Each request is replayed
verbatim — the literal prompt bytes — so server-side prefix-cache
behavior stays realistic.

There is no "job" concept here: one request = one unit. Goodput is
per-request (e2e / TTFT / TBT vs the recorded baseline x tau); see
analysis_scripts/request_level/parse_request_metrics.py.
"""

import json
import os
import threading
import time
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from workloads.base import JobResult, RunContext, TaskLogInfo
from workloads.swe_bench_coding.agent import (
    MODEL_ID,
    count_tokens,
    invoke_with_tracking,
    make_llm,
)


def _deserialize_messages(serialized: list) -> list:
    """Rebuild langchain message objects from {role, content} dicts."""
    cls_by_role = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }
    out = []
    for m in serialized:
        cls = cls_by_role.get(m.get("role", "user"), HumanMessage)
        out.append(cls(content=m.get("content", "")))
    return out


def _resolve_transcript_path(args, workload_config: dict) -> str:
    """Resolve the transcript JSONL path from CLI arg or workload config."""
    path = getattr(args, "transcript_file", None) or workload_config.get(
        "transcript_file"
    )
    if not path:
        raise ValueError(
            "codingagent_request_level_poisson requires a transcript file: "
            "pass --transcript-file <path> (or set 'transcript_file' in "
            "--workload-config). Produce one with a swe_bench_coding "
            "--mode baseline run plus --record-transcript."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"transcript file not found: {path}")
    return path


def load_transcript(path: str) -> list[dict]:
    """Load transcript JSONL records (one recorded agent call per line)."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("messages") and rec.get("request_id"):
                records.append(rec)
    if not records:
        raise ValueError(f"transcript file has no usable records: {path}")
    return records


class CyclingRequestPool:
    """Emit recorded requests on-the-fly from a deterministic cycling pool.

    The transcript records are replayed in order; once exhausted the pool
    cycles, bumping a replay counter so each emission has a unique
    `instance_id` (the prompt bytes stay identical for literal replay).
    """

    def __init__(self, records: list[dict], tau: float = 3.0):
        self.records = records
        self.tau = tau
        self._counter = 0
        self._lock = threading.Lock()

    def next_task(self) -> Optional[dict]:
        with self._lock:
            idx = self._counter % len(self.records)
            replay_num = self._counter // len(self.records) + 1
            self._counter += 1
        return _make_task(self.records[idx], replay_num, self.tau)


def _make_task(record: dict, replay_num: int, tau: float) -> dict:
    """Build a runner task dict for one transcript record emission."""
    request_id = record["request_id"]
    baseline_e2e_s = record.get("baseline_e2e_s")
    job_timeout_sec = baseline_e2e_s * tau if baseline_e2e_s else 0
    return {
        "instance_id": f"{request_id}__r{replay_num:02d}",
        "request_id": request_id,
        "src_job_id": record.get("src_job_id", ""),
        "stage": record.get("stage", ""),
        "nonce": record.get("nonce", ""),
        "messages": record["messages"],
        "baseline_ttft_s": record.get("baseline_ttft_s"),
        "baseline_tbt_mean_ms": record.get("baseline_tbt_mean_ms"),
        "baseline_e2e_s": baseline_e2e_s,
        "recorded_input_tokens": record.get("recorded_input_tokens"),
        "recorded_output_tokens": record.get("recorded_output_tokens"),
        "replay_index": replay_num,
        "job_timeout_sec": job_timeout_sec,
    }


class Workload:
    name = "codingagent_request_level_poisson"

    def load_dataset(self, args, workload_config: dict):
        path = _resolve_transcript_path(args, workload_config)
        records = load_transcript(path)
        print(f"Loaded {len(records)} recorded requests from transcript: {path}")
        self._transcript_path = path
        return records

    def build_baseline_tasks(
        self, dataset, replay_count: int, rng, args, workload_config: dict
    ) -> list[dict]:
        """Replay every recorded request once (per replay) at concurrency 1.

        Useful as a request-level re-baseline / verification pass; the
        transcript already carries solo baselines from its own recording.
        """
        tasks: list[dict] = []
        for replay_idx in range(max(1, replay_count)):
            for record in dataset:
                tasks.append(_make_task(record, replay_idx + 1, args.tau))
        return tasks

    def create_task_pool(
        self, dataset, baseline_latencies, rng, args, workload_config: dict
    ) -> CyclingRequestPool:
        # baseline_latencies is unused: per-request baselines live in the
        # transcript records themselves.
        return CyclingRequestPool(records=dataset, tau=args.tau)

    def run_job(self, task: dict, context: RunContext) -> JobResult:
        job_id = task["instance_id"]
        job_submit_time = context.job_start_time
        job_timeout_sec = task.get("job_timeout_sec", 0) or 0
        stage = task.get("stage", "")
        messages = _deserialize_messages(task["messages"])

        # --disable-request-timeouts: drop every client-side abort so a
        # slow request runs to completion (measured, not killed). The
        # HTTP client timeout is kept large only as a dead-connection
        # safety net.
        disable_to = context.disable_timeouts
        if disable_to:
            job_timeout_sec = 0

        halo_on = context.halo_enabled
        llm = make_llm(
            base_url=f"{context.server_base_url}/v1",
            model_id=context.model or MODEL_ID,
            seed=context.seed,
            api=context.api,
            max_tokens=context.max_tokens,
            halo_ttft_slo=context.halo_ttft_slo if halo_on else None,
            halo_tbt_slo=context.halo_tbt_slo if halo_on else None,
            halo_e2e_slo=context.halo_e2e_slo if halo_on else None,
            timeout=3600.0 if disable_to else None,
        )

        # Minimal single-call state for the shared invoke path. chain_length=1
        # so invoke_with_tracking records total_calls_expected=1 and never
        # tries to advance a chain.
        state = {
            "job_id": job_id,
            "chain_length": 1,
            "nonce": task.get("nonce", ""),
            "metrics_tracker": context.metrics_tracker,
            "agent_logger": context.agent_logger,
            "console_write": context.console_write,
            "llm": llm,
            "transcript_record_path": None,
            "job_timeout_sec": job_timeout_sec,
            "job_start_time": job_submit_time,
            "server_terminated_event": context.server_terminated_event,
            "is_job_timeout": False,
            "is_server_terminated": False,
            "is_rejected": False,
            "rejection_reason": "",
            "last_call_error_msg": "",
        }
        if disable_to:
            # None disables the TTFT / idle aborts in invoke_with_tracking.
            state["per_call_timeout"] = None
            state["idle_timeout"] = None

        response = invoke_with_tracking(
            messages, 1, state, stage, agent_label="request"
        )
        job_end_time = time.time()
        success = response is not None

        full_input = " ".join(
            m.content if hasattr(m, "content") else str(m) for m in messages
        )
        input_tokens = count_tokens(full_input)
        output_tokens = count_tokens(response) if response else 0

        context.agent_logger.log_final_result(
            success=success,
            total_time=job_end_time - job_submit_time,
            iterations=1,
        )

        return JobResult(
            job_id=job_id,
            success=success,
            total_time=job_end_time - job_submit_time,
            calls_completed=1 if success else 0,
            chain_length=1,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            error=state.get("last_call_error_msg") or None,
            is_rejected=bool(state.get("is_rejected", False)),
            rejection_reason=state.get("rejection_reason", ""),
            is_job_timeout=bool(state.get("is_job_timeout", False)),
            is_server_terminated=bool(state.get("is_server_terminated", False)),
            job_timeout_sec=job_timeout_sec if job_timeout_sec > 0 else None,
        )

    def task_log_info(self, task: dict) -> TaskLogInfo:
        return TaskLogInfo(
            task_id=task["instance_id"],
            problem_statement=f"request-level replay (stage={task.get('stage', '')})",
            repo="",
        )

    def metadata(self, args, workload_config: dict) -> dict:
        return {
            "name": self.name,
            "transcript_file": getattr(self, "_transcript_path", None),
            "arrival": "request-level Poisson (lambda = requests/sec)",
            "replay_mode": "literal prompt replay",
        }

    def reproducibility_config(self, args, workload_config: dict) -> dict:
        return {
            "client_seed": args.seed,
            "llm_request_seed": args.seed,
            "temperature": 0.0,
            "top_p": 1.0,
            "transcript_file": getattr(self, "_transcript_path", None),
            "request_order": (
                "transcript order, cycled deterministically; Poisson arrival "
                "times are sampled from random.Random(seed)"
            ),
            "server_requirement": (
                "Baselines in the transcript are tied to a specific "
                "(GPU type x TP size); re-record the transcript when the "
                "server hardware or parallelism changes."
            ),
        }
