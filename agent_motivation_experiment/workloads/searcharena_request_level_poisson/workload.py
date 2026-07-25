"""Search Arena deep-research request-level Poisson workload (direct).

Each task is **one independent LLM request**, submitted as a Poisson
process (lambda = requests/sec) — no job/chain concept. Structurally this
mirrors `sharegpt_request_level_poisson` (direct, no transcript, no solo
baseline, absolute-SLO goodput, all client-side aborts disabled); the
only difference is the request content: a reconstructed deep-research
*synthesis* request ("K research notes + question -> comprehensive
answer") whose input length sits between the chat (~0.7k tok) and SWE
(~22k tok) workloads — measured mean 4,055 / p95 7,914 with the default
K in [2, 12] (a fixed ~910-tok deep-research system prompt is cached by
the engine's prefix cache; ~3,145 tok of notes+question is new prefill).
See `searcharena.py` and this folder's AGENTS.md for the dataset
structure and the reconstruction rationale.

Unlike sharegpt, tasks are lightweight index specs: the question/notes
text pools stay resident once per process and the prompt is assembled on
demand in `run_job` (identical bytes for the same request_id across
replays and runs).
"""

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
from workloads.searcharena_request_level_poisson.searcharena import (
    DEFAULT_HF_DATA_FILE,
    DEFAULT_HF_REPO,
    DEFAULT_HF_REVISION,
    DEFAULT_K_MAX,
    DEFAULT_K_MIN,
    DEFAULT_LANGUAGE,
    DEFAULT_NOTE_MAX_CHARS,
    DEFAULT_NOTE_MIN_CHARS,
    DEFAULT_NUM_REQUESTS,
    DEFAULT_QUESTION_MAX_CHARS,
    assemble_messages,
    build_pools,
    download_search_arena,
    sample_request_specs,
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


def _make_task(spec: dict, replay_num: int) -> dict:
    """Build a runner task dict for one assembled request spec.

    There is no baseline and no `baseline x tau` timeout, so
    `job_timeout_sec` is always 0 (every request runs to completion).
    """
    request_id = spec["request_id"]
    return {
        "instance_id": f"{request_id}__r{replay_num:02d}",
        "request_id": request_id,
        "question_idx": spec["question_idx"],
        "note_idxs": spec["note_idxs"],
        "replay_index": replay_num,
        "job_timeout_sec": 0,
    }


class CyclingSpecPool:
    """Emit request specs from a deterministic cycling pool.

    Specs are replayed in order; once exhausted the pool cycles, bumping
    a replay counter so each emission has a unique `instance_id` (the
    prompt bytes stay identical for literal replay).
    """

    def __init__(self, specs: list[dict]):
        self.specs = specs
        self._counter = 0
        self._lock = threading.Lock()

    def next_task(self) -> Optional[dict]:
        with self._lock:
            idx = self._counter % len(self.specs)
            replay_num = self._counter // len(self.specs) + 1
            self._counter += 1
        return _make_task(self.specs[idx], replay_num)


class Workload:
    name = "searcharena_request_level_poisson"

    def __init__(self):
        self._questions: list[dict] = []
        self._notes: list[dict] = []
        self._hf_repo: Optional[str] = None
        self._hf_data_file: Optional[str] = None
        self._hf_revision: Optional[str] = None
        self._sample_seed: Optional[int] = None
        self._language = DEFAULT_LANGUAGE
        self._k_min = DEFAULT_K_MIN
        self._k_max = DEFAULT_K_MAX
        self._num_requests = 0

    def load_dataset(self, args, workload_config: dict):
        repo = workload_config.get("hf_repo", DEFAULT_HF_REPO)
        data_file = workload_config.get("hf_data_file", DEFAULT_HF_DATA_FILE)
        revision = workload_config.get("hf_revision", DEFAULT_HF_REVISION)
        language = workload_config.get("language", DEFAULT_LANGUAGE)
        num_requests = int(
            workload_config.get("num_requests", DEFAULT_NUM_REQUESTS)
        )
        k_min = int(workload_config.get("k_min", DEFAULT_K_MIN))
        k_max = int(workload_config.get("k_max", DEFAULT_K_MAX))
        note_min_chars = int(
            workload_config.get("note_min_chars", DEFAULT_NOTE_MIN_CHARS)
        )
        note_max_chars = int(
            workload_config.get("note_max_chars", DEFAULT_NOTE_MAX_CHARS)
        )
        question_max_chars = int(
            workload_config.get("question_max_chars", DEFAULT_QUESTION_MAX_CHARS)
        )
        sample_seed = int(workload_config.get("sample_seed", args.seed))

        print(f"[searcharena] downloading {repo}/{data_file} @ {revision} ...")
        parquet_path = download_search_arena(repo, data_file, revision=revision)
        questions, notes = build_pools(
            parquet_path,
            language=language,
            note_min_chars=note_min_chars,
            note_max_chars=note_max_chars,
            question_max_chars=question_max_chars,
        )
        specs = sample_request_specs(
            questions, notes, num_requests, k_min, k_max, sample_seed
        )
        print(
            f"[searcharena] pools: {len(questions)} questions / {len(notes)} "
            f"notes ({language}); assembled {len(specs)} request specs "
            f"(K in [{k_min},{k_max}] log-uniform, sample_seed={sample_seed})"
        )

        self._questions = questions
        self._notes = notes
        self._hf_repo = repo
        self._hf_data_file = data_file
        self._hf_revision = revision
        self._sample_seed = sample_seed
        self._language = language
        self._k_min = k_min
        self._k_max = k_max
        self._num_requests = len(specs)
        return specs

    def build_baseline_tasks(
        self, dataset, replay_count: int, rng, args, workload_config: dict
    ) -> list[dict]:
        """Replay every request spec once per replay at concurrency 1.

        There is no transcript output and no goodput baseline; this is
        just a solo-characterization pass (its `metrics.csv` is handy for
        picking absolute SLO thresholds).
        """
        tasks: list[dict] = []
        for replay_idx in range(max(1, replay_count)):
            for spec in dataset:
                tasks.append(_make_task(spec, replay_idx + 1))
        return tasks

    def create_task_pool(
        self, dataset, baseline_latencies, rng, args, workload_config: dict
    ) -> CyclingSpecPool:
        # baseline_latencies is unused — this workload has no baseline.
        return CyclingSpecPool(specs=dataset)

    def run_job(self, task: dict, context: RunContext) -> JobResult:
        job_id = task["instance_id"]
        job_submit_time = context.job_start_time
        serialized = assemble_messages(task, self._questions, self._notes)
        messages = _deserialize_messages(serialized)

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

        full_input = " ".join(m["content"] for m in serialized)
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
                f"SearchArena deep-research synthesis "
                f"(question_idx={task.get('question_idx')}, "
                f"K={len(task.get('note_idxs') or [])})"
            ),
            repo="",
        )

    def metadata(self, args, workload_config: dict) -> dict:
        return {
            "name": self.name,
            "hf_repo": self._hf_repo,
            "hf_data_file": self._hf_data_file,
            "hf_revision": self._hf_revision,
            "language": self._language,
            "num_questions": len(self._questions),
            "num_notes": len(self._notes),
            "num_requests": self._num_requests,
            "k_min": self._k_min,
            "k_max": self._k_max,
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
                "spec index order, cycled deterministically; Poisson arrival "
                "times are sampled from random.Random(seed)"
            ),
            "request_construction": (
                "request i = fixed synthesis system prompt + K research notes "
                "+ 1 user question; K log-uniform in [k_min,k_max], notes = "
                "search-grounded assistant answers sampled without "
                "replacement from other conversations, all drawn from "
                "random.Random(sample_seed*1_000_003+i) -> byte-identical "
                "across replays and runs"
            ),
            "aborts": "all client-side aborts disabled (no SLO/TTFT/idle timeout)",
        }
