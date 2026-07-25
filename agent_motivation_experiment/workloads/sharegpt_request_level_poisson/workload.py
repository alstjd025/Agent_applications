"""ShareGPT request-level Poisson workload (direct).

Each task is **one independent LLM request**, submitted as a Poisson
process (lambda = requests/sec) — no job/chain concept.

This workload is single-step: it downloads the ShareGPT dataset from
HuggingFace and turns every conversation turn into a request that is sent
directly. There is **no transcript / record step and no solo baseline** —
goodput is evaluated against **absolute SLO thresholds** (post-hoc by the
analysis script), not `baseline x tau`, so no per-request baseline is
needed. Consequently `tau` is unused here and there is no `baseline x tau`
job timeout.

Flattening: each conversation `c` with human turns `1..K` produces `K`
requests; request `k`'s prompt is the conversation prefix
`[u_1, a_1, ..., u_{k-1}, a_{k-1}, u_k]`, where the `a_*` turns are the
recorded ShareGPT gpt responses (deterministic context -> realistic
prefix-cache behavior). No system prompt is injected — ShareGPT is
replayed as-is.

All client-side aborts are disabled: every request runs to completion and
is measured rather than killed (the HTTP client timeout is kept large
only as a dead-connection safety net). `metrics.csv` rows have
`agent == "request"`.
"""

import random
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
from workloads.sharegpt_request_level_poisson.sharegpt import (
    DEFAULT_HF_DATA_FILE,
    DEFAULT_HF_REPO,
    DEFAULT_HF_REVISION,
    DEFAULT_MIN_HUMAN_TURNS,
    DEFAULT_NUM_CONVERSATIONS,
    download_sharegpt,
    flatten_sharegpt,
)

# HTTP client timeout (seconds). Not an SLO abort — only a dead-connection
# safety net, since every client-side SLO/idle abort is disabled.
_HTTP_SAFETY_TIMEOUT_S = 3600.0


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


def _make_task(record: dict, replay_num: int) -> dict:
    """Build a runner task dict for one flattened ShareGPT request.

    There is no baseline and no `baseline x tau` timeout, so
    `job_timeout_sec` is always 0 (every request runs to completion).
    """
    request_id = record["request_id"]
    return {
        "instance_id": f"{request_id}__r{replay_num:02d}",
        "request_id": request_id,
        "conv_index": record.get("conv_index"),
        "turn_index": record.get("turn_index"),
        "messages": record["messages"],
        "replay_index": replay_num,
        "job_timeout_sec": 0,
    }


class CyclingRequestPool:
    """Emit flattened ShareGPT requests from a deterministic cycling pool.

    Records are replayed in order; once exhausted the pool cycles, bumping
    a replay counter so each emission has a unique `instance_id` (the
    prompt bytes stay identical for literal replay).
    """

    def __init__(self, records: list[dict]):
        self.records = records
        self._counter = 0
        self._lock = threading.Lock()

    def next_task(self) -> Optional[dict]:
        with self._lock:
            idx = self._counter % len(self.records)
            replay_num = self._counter // len(self.records) + 1
            self._counter += 1
        return _make_task(self.records[idx], replay_num)


class Workload:
    name = "sharegpt_request_level_poisson"

    def __init__(self):
        self._hf_repo: Optional[str] = None
        self._hf_data_file: Optional[str] = None
        self._hf_revision: Optional[str] = None
        self._sample_seed: Optional[int] = None
        self._num_conversations = 0
        self._num_requests = 0

    def load_dataset(self, args, workload_config: dict):
        repo = workload_config.get("hf_repo", DEFAULT_HF_REPO)
        data_file = workload_config.get("hf_data_file", DEFAULT_HF_DATA_FILE)
        revision = workload_config.get("hf_revision", DEFAULT_HF_REVISION)
        num_conversations = int(
            workload_config.get("num_conversations", DEFAULT_NUM_CONVERSATIONS)
        )
        min_human_turns = int(
            workload_config.get("min_human_turns", DEFAULT_MIN_HUMAN_TURNS)
        )
        max_human_turns = workload_config.get("max_human_turns")
        max_human_turns = int(max_human_turns) if max_human_turns else None
        sample_seed = int(workload_config.get("sample_seed", args.seed))

        print(f"[sharegpt] downloading {repo}/{data_file} @ {revision} ...")
        raw = download_sharegpt(repo, data_file, revision=revision)
        records = flatten_sharegpt(
            raw, num_conversations, min_human_turns, max_human_turns, sample_seed
        )
        conv_count = len({r["conv_index"] for r in records})
        print(
            f"[sharegpt] flattened {conv_count} conversations into "
            f"{len(records)} requests (sample_seed={sample_seed})"
        )

        self._hf_repo = repo
        self._hf_data_file = data_file
        self._hf_revision = revision
        self._sample_seed = sample_seed
        self._num_conversations = conv_count
        self._num_requests = len(records)
        return records

    def build_baseline_tasks(
        self, dataset, replay_count: int, rng, args, workload_config: dict
    ) -> list[dict]:
        """Replay every flattened request once per replay at concurrency 1.

        There is no transcript output and no goodput baseline; this is just
        a solo-characterization pass (its `metrics.csv` is handy for
        picking absolute SLO thresholds).
        """
        tasks: list[dict] = []
        for replay_idx in range(max(1, replay_count)):
            for record in dataset:
                tasks.append(_make_task(record, replay_idx + 1))
        return tasks

    def create_task_pool(
        self, dataset, baseline_latencies, rng, args, workload_config: dict
    ) -> CyclingRequestPool:
        # baseline_latencies is unused — this workload has no baseline.
        return CyclingRequestPool(records=dataset)

    def run_job(self, task: dict, context: RunContext) -> JobResult:
        job_id = task["instance_id"]
        job_submit_time = context.job_start_time
        messages = _deserialize_messages(task["messages"])

        halo_on = context.halo_enabled
        llm = make_llm(
            base_url=f"{context.server_base_url}/v1",
            model_id=context.model or MODEL_ID,
            seed=context.seed,
            api=context.api,
            max_tokens=context.max_tokens,
            # Halo SLO fields are forwarded verbatim; with absolute SLOs
            # these carry absolute thresholds (server slo_mode decides).
            halo_ttft_slo=context.halo_ttft_slo if halo_on else None,
            halo_tbt_slo=context.halo_tbt_slo if halo_on else None,
            halo_e2e_slo=context.halo_e2e_slo if halo_on else None,
            timeout=_HTTP_SAFETY_TIMEOUT_S,
            slo_budget_ms=context.slo_budget_ms,
            slo_spec=context.slo_spec,
            priority_mode=context.priority_mode,
        )

        # Minimal single-call state for the shared invoke path. chain_length=1
        # so invoke_with_tracking records total_calls_expected=1 and never
        # tries to advance a chain. Every client-side abort is disabled:
        #   job_timeout_sec=0 -> no job timeout
        #   per_call_timeout=None -> no TTFT abort
        #   idle_timeout=None -> no idle abort
        # so each request runs to completion and is measured, not killed.
        state = {
            "job_id": job_id,
            "chain_length": 1,
            "nonce": "",
            "metrics_tracker": context.metrics_tracker,
            "agent_logger": context.agent_logger,
            "console_write": context.console_write,
            "llm": llm,
            "transcript_record_path": None,
            "job_timeout_sec": 0,
            "job_start_time": job_submit_time,
            "server_terminated_event": context.server_terminated_event,
            "per_call_timeout": None,
            "idle_timeout": None,
            "is_job_timeout": False,
            "is_server_terminated": False,
            "is_rejected": False,
            "rejection_reason": "",
            "last_call_error_msg": "",
        }

        response = invoke_with_tracking(
            messages, 1, state, "", agent_label="request"
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
            job_timeout_sec=None,
        )

    def task_log_info(self, task: dict) -> TaskLogInfo:
        return TaskLogInfo(
            task_id=task["instance_id"],
            problem_statement=(
                f"ShareGPT request-level "
                f"(conv={task.get('conv_index')}, turn={task.get('turn_index')})"
            ),
            repo="",
        )

    def metadata(self, args, workload_config: dict) -> dict:
        return {
            "name": self.name,
            "hf_repo": self._hf_repo,
            "hf_data_file": self._hf_data_file,
            "hf_revision": self._hf_revision,
            "num_conversations": self._num_conversations,
            "num_requests": self._num_requests,
            "sample_seed": self._sample_seed,
            "arrival": "request-level Poisson (lambda = requests/sec)",
            "goodput_model": "absolute SLO thresholds (no baseline, tau unused)",
        }

    def reproducibility_config(self, args, workload_config: dict) -> dict:
        return {
            "client_seed": args.seed,
            "llm_request_seed": args.seed,
            "temperature": 0.0,
            "top_p": 1.0,
            "sample_seed": self._sample_seed,
            "hf_repo": self._hf_repo,
            "hf_data_file": self._hf_data_file,
            "hf_revision": self._hf_revision,
            "request_order": (
                "flattened ShareGPT order, cycled deterministically; Poisson "
                "arrival times are sampled from random.Random(seed)"
            ),
            "conversation_flattening": (
                "each conversation -> one request per human turn; request k's "
                "prompt is the conversation prefix [u1,a1,...,u_{k-1},a_{k-1},u_k] "
                "with the recorded ShareGPT gpt turns as assistant context"
            ),
            "aborts": "all client-side aborts disabled (no SLO/TTFT/idle timeout)",
        }
