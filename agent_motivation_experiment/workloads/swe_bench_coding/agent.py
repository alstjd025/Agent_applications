"""
Synthetic Stage-Based Coding Agent for Motivation Experiment.

Simulates a realistic multi-call coding agent (like Claude Code) that progresses
through stages: Understand -> Locate -> Plan -> Implement -> Verify -> [Debug loop].

Key design goals for the "Illusion of Efficiency" motivation experiment:
  - Server throughput peaks while job-level goodput collapses under concurrency
  - Tool result simulation drives realistic context growth (the main source in real agents)
  - System prompt is IDENTICAL across all calls within a session
  - Variable chain length: randint(5, 30) -- stage sequence adapts to fill chain_length
  - All-or-nothing: job complete only if ALL calls succeed
  - Per-call timeout: 120s
  - Nonce per call: prevent KV cache reuse across replays
  - temperature=0.0, seed=42: for reproducibility
  - TBT measurement via streaming: using summarize_tbt_ms from metrics_tracker
  - Progressive context growth: conversation history accumulates, tool results injected
"""

import hashlib
import json
import os
import time
import uuid
import random
import threading
from types import SimpleNamespace
from typing import TypedDict, Optional, Any, List, Callable

import requests
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from metrics_tracker import summarize_tbt_ms
from workloads.swe_bench_coding.prompts.system_prompt import SYSTEM_PROMPT
from workloads.swe_bench_coding.prompts.stage_prompts import AgentStage, STAGE_PROMPTS
from workloads.swe_bench_coding.prompts.simulated_artifacts import (
    SIMULATED_SEARCH_RESULT,
    SIMULATED_FILE_CONTENTS,
    SIMULATED_TEST_OUTPUT_PASS,
    SIMULATED_TEST_OUTPUT_FAIL,
    FILE_ORDER,
    get_file_content_for_locate,
    get_test_result_for_verify,
)


# ---------------------------------------------------------------------------
# Server configuration
# ---------------------------------------------------------------------------
BASE_URL = "http://localhost:8080/v1"
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
PER_CALL_TIMEOUT = 120  # seconds — TTFT timeout: no first token within this time
IDLE_TIMEOUT = 60  # seconds — idle timeout: no chunk for this duration during streaming
# Completions API (Llumnix gateway) default output cap. vLLM's /v1/completions
# defaults max_tokens to 16, so we MUST set an explicit cap for this path.
DEFAULT_MAX_TOKENS = 2048


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------
try:
    _tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    _tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Approximate token count using cl100k_base encoding."""
    try:
        return len(_tokenizer.encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------
def make_llm(
    base_url: str = BASE_URL,
    model_id: str = MODEL_ID,
    seed: int = 42,
    *,
    api: str = "chat",
    max_tokens: Optional[int] = None,
    halo_ttft_slo: Optional[float] = None,
    halo_tbt_slo: Optional[float] = None,
    halo_e2e_slo: Optional[float] = None,
    halo_bypass: bool = False,
    timeout: Optional[float] = None,
):
    """Create a ChatOpenAI instance with proper timeout settings.

    HALO request-level wiring (see ms_dev/halo_dev/halo_api_reference.md):
      - halo_ttft_slo / halo_tbt_slo / halo_e2e_slo: per-request SLO
        fields. When set, attached to every chat.completions request via
        `extra_body` so the server's request-level admission gate can
        evaluate each request independently. With slo_mode=ratio these
        are slowdown ratios vs the solo-run baseline; halo_e2e_slo is
        always a ratio.
      - halo_bypass: marks server-internal traffic; real clients leave it
        False.

    `timeout` is the HTTP client timeout in seconds; None uses
    PER_CALL_TIMEOUT. Pass a large value to keep only a dead-connection
    safety net when client-side aborts are disabled.

    There is NO job pre-registration and no halo_job_id anymore (the
    2026-05-19 request-level refactor removed POST /halo/programs). When
    all SLO fields are None, no extra_body is injected — works against a
    non-Halo server unchanged.

    `api` selects the wire protocol:
      - "chat" (default): LangChain ChatOpenAI -> POST /v1/chat/completions.
      - "completions": LlumnixCompletionsLLM -> POST /v1/completions (the
        Llumnix gateway only serves completions; chat returns HTTP 400). The
        adapter renders the message list with the Llama-3 chat template and
        streams SSE, exposing the same .stream()/.invoke() -> .content contract
        so invoke_with_tracking is unchanged.
    """
    if api == "completions":
        return LlumnixCompletionsLLM(
            base_url=base_url,
            model=model_id,
            seed=seed,
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
            timeout=timeout if timeout is not None else PER_CALL_TIMEOUT,
        )

    kwargs: dict = dict(
        base_url=base_url,
        api_key="dummy",
        model=model_id,
        temperature=0.0,
        timeout=timeout if timeout is not None else PER_CALL_TIMEOUT,
        top_p=1.0,
        seed=seed,
    )
    extra: dict = {}
    if halo_ttft_slo is not None:
        extra["halo_ttft_slo"] = halo_ttft_slo
    if halo_tbt_slo is not None:
        extra["halo_tbt_slo"] = halo_tbt_slo
    if halo_e2e_slo is not None:
        extra["halo_e2e_slo"] = halo_e2e_slo
    if halo_bypass:
        extra["halo_bypass"] = True
    if extra:
        kwargs["extra_body"] = extra
    return ChatOpenAI(**kwargs)


# ---------------------------------------------------------------------------
# Llumnix completions adapter (/v1/completions)
# ---------------------------------------------------------------------------
# The Llumnix gateway only serves /v1/completions (chat -> HTTP 400). This
# adapter renders the LangChain message list with the Llama-3 chat template and
# streams SSE from /v1/completions, yielding chunk objects with a `.content`
# attribute so the shared invoke_with_tracking() path is unchanged. Built on
# `requests` (already a dependency) to avoid pulling in the openai SDK.

_LLAMA3_ROLE_MAP = {"system": "system", "human": "user", "ai": "assistant"}


def render_llama3_prompt(messages: list) -> str:
    """Render langchain messages into a Llama-3-Instruct prompt string.

    /v1/completions does NOT apply a chat template, so we build the special-
    token prompt ourselves and end with the assistant header so the model
    continues as the assistant. Role mapping matches `_serialize_messages`.
    """
    parts = ["<|begin_of_text|>"]
    for m in messages:
        mtype = getattr(m, "type", None) or m.__class__.__name__.lower()
        role = _LLAMA3_ROLE_MAP.get(mtype, "user")
        content = getattr(m, "content", str(m))
        parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


_HTTP_SESSION = None
_HTTP_SESSION_LOCK = threading.Lock()


def _llumnix_http_session():
    """Process-shared requests.Session with a large connection pool.

    Under high concurrency (multiprocess load gen with many threads/worker), a
    fresh `requests.post` per call opens a new TCP connection each time →
    connection churn / ephemeral-port exhaustion / "Max retries exceeded". A
    pooled keep-alive Session with a big pool reuses connections and sustains
    high load. One session per process (each MP worker gets its own).
    """
    global _HTTP_SESSION
    if _HTTP_SESSION is None:
        with _HTTP_SESSION_LOCK:
            if _HTTP_SESSION is None:
                from requests.adapters import HTTPAdapter
                s = requests.Session()
                adapter = HTTPAdapter(pool_connections=64, pool_maxsize=1024, max_retries=0)
                s.mount("http://", adapter)
                s.mount("https://", adapter)
                _HTTP_SESSION = s
    return _HTTP_SESSION


class LlumnixRejectedError(RuntimeError):
    """Llumnix gateway/scheduler admission rejection.

    Raised when the gateway answers 503 `no available inference worker`
    (scheduler returned 429 no-available-endpoint, e.g. the EXP-07
    `--admission-kv-usage-threshold` hard filter) or 429 `rate limit
    exceeded`. Recognized by `_detect_admission_rejection`, so the call is
    recorded as is_rejected=True instead of a generic error, and the
    non-streaming fallback retry is skipped (no double submission).
    """

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(
            f"llumnix admission rejected (HTTP {status_code}): {body[:200]}")


def _raise_if_llumnix_rejected(resp) -> None:
    if resp.status_code in (429, 503):
        body = resp.text
        if ("no available inference worker" in body
                or "rate limit exceeded" in body):
            raise LlumnixRejectedError(resp.status_code, body)


class LlumnixCompletionsLLM:
    """Minimal /v1/completions client with a ChatOpenAI-compatible surface.

    Exposes `.stream(messages)` (generator of objects with `.content`) and
    `.invoke(messages)` (returns an object with `.content`) so it drops into
    `invoke_with_tracking` unchanged. TTFT/TBT/token counting all operate on
    `.content`, and metadata access degrades to {} (no admission gate in
    Llumnix), so no rejection is ever falsely detected.
    """

    # Halt generation at the Llama-3 turn boundary (we bypassed the chat
    # template, so the server won't add these stops automatically).
    STOP = ["<|eot_id|>", "<|end_of_text|>"]

    def __init__(self, base_url, model, seed=42, temperature=0.0, top_p=1.0,
                 max_tokens=DEFAULT_MAX_TOKENS, timeout=PER_CALL_TIMEOUT):
        # base_url already includes the /v1 suffix (added at the workload
        # call site), matching ChatOpenAI's convention.
        self.completions_url = base_url.rstrip("/") + "/completions"
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        # Server-assigned request id of the most recent call ("cmpl-<uuid>").
        # The Llumnix scheduler logs the same uuid in its dispatch line
        # ("[Schedule] dispatch request <uuid> to neutral instance <id>"), so
        # capturing it here is what makes per-request -> engine attribution
        # possible (see experiments/DEV_request-engine-attribution.md).
        # One LLM object is built per request in the request-level workloads
        # and used sequentially within a chain, so this is not shared across
        # concurrent calls.
        self.last_request_id = None

    def _payload(self, messages: list, stream: bool) -> dict:
        return {
            "model": self.model,
            "prompt": render_llama3_prompt(messages),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "stop": self.STOP,
            "stream": stream,
        }

    def stream(self, messages: list):
        self.last_request_id = None
        resp = _llumnix_http_session().post(
            self.completions_url,
            json=self._payload(messages, stream=True),
            stream=True,
            timeout=self.timeout,
        )
        _raise_if_llumnix_rejected(resp)
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if raw.startswith("data:"):
                data = raw[len("data:"):].strip()
            else:
                continue
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
                text = obj["choices"][0].get("text", "")
            except (ValueError, KeyError, IndexError):
                continue
            if self.last_request_id is None:
                rid = obj.get("id")
                if rid:
                    self.last_request_id = rid
            yield SimpleNamespace(content=text)

    def invoke(self, messages: list):
        self.last_request_id = None
        resp = _llumnix_http_session().post(
            self.completions_url,
            json=self._payload(messages, stream=False),
            timeout=self.timeout,
        )
        _raise_if_llumnix_rejected(resp)
        resp.raise_for_status()
        obj = resp.json()
        self.last_request_id = obj.get("id")
        text = obj["choices"][0].get("text", "")
        return SimpleNamespace(content=text)


# ---------------------------------------------------------------------------
# Stage sequence builder
# ---------------------------------------------------------------------------
def build_stage_sequence(chain_length: int, rng: random.Random) -> List[str]:
    """Build the stage sequence for a given chain length.

    The sequence follows the pattern:
    1. UNDERSTAND: always 1 call
    2. LOCATE: 1-4 calls (reading multiple files)
    3. PLAN: 1 call
    4. IMPLEMENT: 1-3 calls
    5. VERIFY: 1 call
    6. Debug loop fills remaining: DEBUG -> IMPLEMENT -> VERIFY cycle

    Args:
        chain_length: Total number of calls to fill.
        rng: Random instance for reproducibility.

    Returns:
        List of AgentStage values, one per call.
    """
    if chain_length <= 0:
        return []

    stages: List[str] = []
    idx = 0

    # 1. UNDERSTAND: always 1 call
    stages.append(AgentStage.UNDERSTAND)
    idx += 1
    if idx >= chain_length:
        return stages

    # 2. LOCATE: 1-4 calls
    n_locate = rng.randint(1, min(4, chain_length - idx))
    for _ in range(n_locate):
        stages.append(AgentStage.LOCATE)
        idx += 1
        if idx >= chain_length:
            return stages

    # 3. PLAN: 1 call
    stages.append(AgentStage.PLAN)
    idx += 1
    if idx >= chain_length:
        return stages

    # 4. IMPLEMENT: 1-3 calls
    n_impl = rng.randint(1, min(3, chain_length - idx))
    for _ in range(n_impl):
        stages.append(AgentStage.IMPLEMENT)
        idx += 1
        if idx >= chain_length:
            return stages

    # 5. VERIFY: 1 call
    stages.append(AgentStage.VERIFY)
    idx += 1
    if idx >= chain_length:
        return stages

    # 6. Debug loop fills remaining: DEBUG -> IMPLEMENT -> VERIFY
    debug_cycle = [AgentStage.DEBUG, AgentStage.IMPLEMENT, AgentStage.VERIFY]
    cycle_idx = 0
    while idx < chain_length:
        stages.append(debug_cycle[cycle_idx % len(debug_cycle)])
        cycle_idx += 1
        idx += 1

    return stages


# ---------------------------------------------------------------------------
# Tool results builder (variety-aware)
# ---------------------------------------------------------------------------
def build_tool_results(
    stage_sequence: List[str], rng: random.Random
) -> List[Optional[str]]:
    """Pre-compute the tool result to inject for each stage call.

    Each LOCATE call gets a DIFFERENT file (rotating sequentially).
    Each VERIFY call gets appropriate pass/fail (first=pass, debug=fail, then pass).

    This ensures that even with temperature=0.0, repeated stages produce
    genuinely different outputs because the input tool results differ.

    Args:
        stage_sequence: List of AgentStage values from build_stage_sequence.
        rng: Random instance (kept for API compatibility with create_chain_state,
             but not used for file/test selection anymore).

    Returns:
        List of tool result strings (or None), one per stage call.
    """
    locate_idx = 0
    verify_idx = 0
    results: List[Optional[str]] = []

    for stage in stage_sequence:
        if stage == AgentStage.UNDERSTAND:
            results.append(SIMULATED_SEARCH_RESULT)

        elif stage == AgentStage.LOCATE:
            results.append(get_file_content_for_locate(locate_idx))
            locate_idx += 1

        elif stage == AgentStage.VERIFY:
            results.append(get_test_result_for_verify(verify_idx))
            verify_idx += 1

        elif stage == AgentStage.DEBUG:
            # Debug stage gets both file content and test output
            file_content = get_file_content_for_locate(locate_idx)
            # Debug stage implies we're in a failure loop -- show the failure
            test_output = SIMULATED_TEST_OUTPUT_FAIL
            results.append(f"{file_content}\n\n{test_output}")
            locate_idx += 1

        elif stage in (AgentStage.PLAN, AgentStage.IMPLEMENT):
            results.append(None)

        else:
            results.append(None)

    return results


# ---------------------------------------------------------------------------
# ChainState TypedDict
# ---------------------------------------------------------------------------
class ChainState(TypedDict):
    job_id: str
    problem_statement: str          # from SWE-bench, used as initial context
    chain_length: int               # target number of calls (N)
    call_index: int                 # current call (1-based)
    accumulated_context: str        # grows with each call
    nonce: str                      # unique per job+replay
    stage_sequence: List[str]       # pre-computed stage for each call
    tool_results: List[Optional[str]]  # pre-computed tool result per call
    log_level: str
    metrics_tracker: Optional[object]
    agent_logger: Optional[object]
    console_write: Optional[Callable]
    llm: Optional[Any]              # per-job ChatOpenAI instance
    last_call_output: str           # output from the previous call
    job_completed: bool             # True only if ALL N calls succeed
    error_msg: str                  # non-empty if job failed
    rng: Optional[random.Random]    # per-job RNG for reproducibility
    job_start_time: float           # epoch seconds when the job started
    job_timeout_sec: float          # baseline_latency × τ
    is_job_timeout: bool            # True if job exceeded τ timeout
    is_server_terminated: bool      # True if server was terminated mid-job
    is_rejected: bool               # True if SGLang admission control rejected a call
    rejection_reason: str           # Admission rejection reason, e.g. TBT_RATIO
    last_call_error_msg: str        # Client-visible failure for the latest call
    total_input_tokens: int         # cumulative input tokens across calls
    total_output_tokens: int        # cumulative output tokens across calls
    server_terminated_event: Optional[threading.Event]  # set when server is killed
    tool_call_delays: List[float]   # delay before each call, indexed by call_index - 1
    tool_delay_total_s: float       # cumulative application-side tool delay
    # Transcript recording (literal-replay capture). When set, each
    # successful call's full prompt + solo timings are appended here.
    transcript_record_path: Optional[str]
    # Client-side abort thresholds (seconds). None disables that abort —
    # invoke_with_tracking reads these with PER_CALL_TIMEOUT / IDLE_TIMEOUT
    # as the fallback. Set by create_chain_state from --disable-timeouts.
    per_call_timeout: Optional[float]
    idle_timeout: Optional[float]


# ---------------------------------------------------------------------------
# Streaming invocation with TBT measurement and per-call timeout
# ---------------------------------------------------------------------------
def _metadata_dict(obj: Any) -> dict:
    """Collect LangChain/OpenAI metadata without assuming one provider shape."""
    metadata = {}
    for attr in ("response_metadata", "additional_kwargs", "generation_info"):
        value = getattr(obj, attr, None)
        if isinstance(value, dict):
            metadata.update(value)
    return metadata


def _serialize_messages(messages: list) -> List[dict]:
    """Serialize langchain messages to {role, content} dicts.

    Used by the transcript recorder so the request-level workload can
    replay a recorded agent call's prompt verbatim.
    """
    role_map = {"system": "system", "human": "user", "ai": "assistant"}
    out: List[dict] = []
    for m in messages:
        mtype = getattr(m, "type", None) or m.__class__.__name__.lower()
        role = role_map.get(mtype, "user")
        out.append({"role": role, "content": getattr(m, "content", str(m))})
    return out


def _detect_admission_rejection(obj: Any) -> tuple[bool, str]:
    """Detect server-side admission rejection in a chunk / exception / response.

    Handles both:
      - Halo request-level admission gate: HTTP 400. The scheduler emits
        an abort whose finish info carries
        `{"type":"abort","status_code":400,"halo_reason":"HALO_*",
          "message":"Halo admission gate rejected this request (reason=...)"}`.
        Reasons: HALO_KV_CAP / HALO_VSS_PREDICTED / HALO_ADMISSION_PREDICTED
        (and future policy reasons).
      - Legacy admission_control (Mooncake-style predictive): HTTP 429 +
        `admission_reason` (still present in sglang but dormant).

    Returns (rejected, reason). reason is the upstream reason string when
    available, else "HALO_REJECTED" / "ADMISSION_REJECTED" fallback.

    For exceptions raised by the langchain/openai SDK on HTTP 4xx, the
    relevant info is usually only in `str(exception)` (response metadata
    fields are empty), so we also scan that as a fallback.
    """
    metadata = _metadata_dict(obj)
    finish_reason = metadata.get("finish_reason") or metadata.get("finish_details")
    reason = (
        metadata.get("halo_reason")
        or metadata.get("admission_reason")
        or metadata.get("reason")
        or ""
    )
    message = metadata.get("message") or ""
    status_code = metadata.get("status_code")

    if isinstance(finish_reason, dict):
        status_code = finish_reason.get("status_code", status_code)
        reason = (
            finish_reason.get("halo_reason")
            or finish_reason.get("admission_reason")
            or reason
        )
        message = finish_reason.get("message", message)
        finish_type = str(finish_reason.get("type", "")).lower()
    else:
        finish_type = str(finish_reason or "").lower()

    # Include str(obj) so exception messages from the openai SDK (e.g.
    # `openai.BadRequestError`) — which don't surface fields via
    # _metadata_dict — still expose 'halo_reason' / 'HALO_*' / the
    # rejection phrasing. Without this the HTTP 400 rejection path is
    # invisible and metrics.csv never marks the call as is_rejected.
    text = " ".join(
        str(part)
        for part in (finish_reason, reason, message, metadata, obj)
        if part is not None
    ).lower()

    # Halo request-level path: HTTP 400 + the admission-gate phrasing or
    # any halo_reason / HALO_* token.
    halo_rejected = (
        "halo_reason" in text
        or "admission gate rejected" in text
        or "halo_kv_cap" in text
        or "halo_vss_predicted" in text
        or "halo_admission_predicted" in text
        or (str(status_code) == "400" and "halo" in text)
    )
    # Legacy admission_control path: HTTP 429 + "admission rejected".
    admission_rejected = (
        "admission rejected" in text
        or "admission_reason" in text
        or (str(status_code) == "429" and "abort" in finish_type)
    )
    # Llumnix admission path (EXP-07 KV-threshold filter): scheduler answers
    # 429 no-available-endpoint, gateway surfaces 503
    # {"error":{"code":503,"message":"no available inference worker"}} (or
    # 429 "rate limit exceeded"), and LlumnixCompletionsLLM wraps it in
    # LlumnixRejectedError whose str() lands in `text`.
    llumnix_rejected = (
        "llumnix admission rejected" in text
        or "no available inference worker" in text
    )

    if not (halo_rejected or admission_rejected or llumnix_rejected):
        return False, ""

    if not reason and isinstance(message, str):
        if "reason=" in message:
            reason = message.split("reason=", 1)[1].split(")", 1)[0].strip()
        elif "Admission rejected:" in message:
            reason = message.split("Admission rejected:", 1)[1].split("(", 1)[0].strip()

    if not reason:
        if halo_rejected:
            reason = "HALO_REJECTED"
        elif llumnix_rejected:
            reason = "KV_THRESHOLD"
        else:
            reason = "ADMISSION_REJECTED"
    return True, str(reason)


def sleep_with_abort_checks(state: ChainState, delay_s: float, call_index: int) -> Optional[dict]:
    """Sleep in small chunks while honoring job timeout and server termination."""
    if delay_s <= 0:
        return None

    last_time = time.time()
    deadline = time.time() + delay_s
    while True:
        now = time.time()
        state["tool_delay_total_s"] = state.get("tool_delay_total_s", 0.0) + max(0.0, now - last_time)
        last_time = now
        if now >= deadline:
            return None

        job_timeout_sec = state.get("job_timeout_sec", 0)
        job_start_time = state.get("job_start_time", 0)
        if job_timeout_sec > 0 and job_start_time > 0 and now - job_start_time > job_timeout_sec:
            state["is_job_timeout"] = True
            return {
                "job_completed": False,
                "error_msg": f"Job exceeded {job_timeout_sec:.0f}s (τ timeout) during tool delay before call {call_index}",
                "call_index": call_index,
                "is_job_timeout": True,
                "is_server_terminated": state.get("is_server_terminated", False),
                "tool_delay_total_s": state.get("tool_delay_total_s", 0.0),
            }

        server_event = state.get("server_terminated_event")
        if server_event is not None and server_event.is_set():
            state["is_server_terminated"] = True
            return {
                "job_completed": False,
                "error_msg": f"Server terminated during tool delay before call {call_index}",
                "call_index": call_index,
                "is_job_timeout": state.get("is_job_timeout", False),
                "is_server_terminated": True,
                "tool_delay_total_s": state.get("tool_delay_total_s", 0.0),
            }

        sleep_s = min(0.1, deadline - now)
        time.sleep(sleep_s)


def invoke_with_tracking(
    messages: list,
    call_index: int,
    state: ChainState,
    current_stage: str = "",
    agent_label: Optional[str] = None,
) -> Optional[str]:
    """
    Invoke LLM with streaming, TBT measurement, timeout, and metrics recording.

    Args:
        messages: list of langchain message objects
        call_index: 1-based index of this call within the chain
        state: current ChainState
        current_stage: stage label used in the default agent name
        agent_label: when set, overrides the metrics `agent` column
            verbatim (e.g. "request" for the request-level workload).

    Returns:
        Response content string on success, None on timeout/error.
    """
    llm = state.get("llm")
    if llm is None:
        llm = make_llm()
        state["llm"] = llm

    start_time = time.time()
    first_token_time = None
    last_stream_chunk_time = None

    # Compute input token count from all messages
    full_input = " ".join(
        m.content if hasattr(m, "content") else str(m) for m in messages
    )
    input_tokens = count_tokens(full_input)

    stream_fallback_used = False
    response_chunks: List[str] = []
    chunk_events = []
    tbt_values_ms: List[float] = []
    streamed_output_tokens_est = 0
    first_chunk_tokens_est = 0
    content_chunk_idx = 0
    is_timeout = False
    is_error = False
    is_rejected = False
    is_job_timeout = False
    is_server_terminated = False
    error_msg = ""
    rejection_reason = ""
    response_content = ""

    job_timeout_sec = state.get("job_timeout_sec", 0)
    job_start_time = state.get("job_start_time", 0)
    # Client-side abort thresholds. A workload may set these to None to
    # disable the TTFT / idle aborts (e.g. the request-level workload
    # under --disable-request-timeouts). Absent keys keep the defaults.
    per_call_timeout = state.get("per_call_timeout", PER_CALL_TIMEOUT)
    idle_timeout = state.get("idle_timeout", IDLE_TIMEOUT)

    try:
        for chunk in llm.stream(messages):
            chunk_arrival_time = time.time()

            rejected, detected_reason = _detect_admission_rejection(chunk)
            if rejected:
                is_rejected = True
                is_error = True
                rejection_reason = detected_reason
                error_msg = f"Call {call_index}: admission rejected ({rejection_reason})"
                state["is_rejected"] = True
                state["rejection_reason"] = rejection_reason
                state["last_call_error_msg"] = error_msg
                break

            # Job-level τ timeout check
            if job_timeout_sec > 0 and job_start_time > 0:
                if chunk_arrival_time - job_start_time > job_timeout_sec:
                    is_job_timeout = True
                    error_msg = f"Call {call_index}: job exceeded {job_timeout_sec:.0f}s (τ timeout)"
                    # Propagate to state so call_chain_node can see it
                    state["is_job_timeout"] = True
                    break

            # Server terminated event check
            server_event = state.get("server_terminated_event")
            if server_event is not None and server_event.is_set():
                is_server_terminated = True
                error_msg = f"Call {call_index}: server terminated during streaming"
                # Propagate to state so call_chain_node can see it
                state["is_server_terminated"] = True
                break

            # Call-level timeout check: TTFT or idle (skipped when the
            # corresponding threshold is None — disabled by the workload).
            if first_token_time is None:
                # No first token yet — TTFT timeout
                if per_call_timeout is not None and chunk_arrival_time - start_time > per_call_timeout:
                    is_timeout = True
                    error_msg = f"Call {call_index}: no response within {per_call_timeout}s (TTFT timeout)"
                    break
            else:
                # Streaming in progress — idle timeout
                if (
                    idle_timeout is not None
                    and last_stream_chunk_time is not None
                    and chunk_arrival_time - last_stream_chunk_time > idle_timeout
                ):
                    is_timeout = True
                    error_msg = f"Call {call_index}: no chunk for {idle_timeout}s (idle timeout)"
                    break

            if hasattr(chunk, "content") and chunk.content is not None:
                chunk_text = chunk.content
                if chunk_text:
                    if first_token_time is None:
                        first_token_time = chunk_arrival_time
                    response_chunks.append(chunk_text)
                    chunk_tokens_est = max(count_tokens(chunk_text), 1)
                    streamed_output_tokens_est += chunk_tokens_est

                    if last_stream_chunk_time is None:
                        first_chunk_tokens_est = chunk_tokens_est
                        inter_arrival_ms = None
                    else:
                        inter_arrival_ms = (chunk_arrival_time - last_stream_chunk_time) * 1000.0
                        per_token_tbt_ms = inter_arrival_ms / chunk_tokens_est
                        tbt_values_ms.extend([per_token_tbt_ms] * chunk_tokens_est)

                    chunk_events.append({
                        "chunk_idx": content_chunk_idx,
                        "arrival_offset_ms": round((chunk_arrival_time - start_time) * 1000.0, 4),
                        "delta_chars": len(chunk_text),
                        "delta_tokens_est": chunk_tokens_est,
                        "inter_arrival_ms": round(inter_arrival_ms, 4) if inter_arrival_ms is not None else None,
                    })
                    last_stream_chunk_time = chunk_arrival_time
                    content_chunk_idx += 1

    except Exception as e:
        err_str = str(e).lower()
        rejected, detected_reason = _detect_admission_rejection(e)
        # HALO path: server returns HTTP 400 with body
        #   {"error":{"message":"Halo rejected: ...","halo_reason":"HALO_*"}}
        # langchain/openai wraps that into an exception whose str()
        # contains those tokens. We catch them here in addition to the
        # admission_control 429 path.
        halo_text_hit = (
            "halo rejected" in err_str
            or "halo_reason" in err_str
            or "halo_no_job_id" in err_str
            or "halo_program_not_registered" in err_str
        )
        if (
            rejected
            or "admission rejected" in err_str
            or "429" in err_str
            or halo_text_hit
        ):
            is_rejected = True
            is_error = True
            if not detected_reason and halo_text_hit:
                detected_reason = "HALO_REJECTED"
            rejection_reason = detected_reason or "ADMISSION_REJECTED"
            error_msg = f"Call {call_index}: admission rejected ({rejection_reason})"
            state["is_rejected"] = True
            state["rejection_reason"] = rejection_reason
            state["last_call_error_msg"] = error_msg
        # Detect server termination (connection reset, broken pipe, etc.)
        elif any(kw in err_str for kw in ["connectionreset", "brokenpipe", "connectionaborted",
                                           "connection refused", "eof occurred", "server disconnected"]):
            is_server_terminated = True
            error_msg = f"Call {call_index}: server terminated ({type(e).__name__})"
            # Propagate to state so call_chain_node can see it
            state["is_server_terminated"] = True
        else:
            stream_fallback_used = True
            try:
                response = llm.invoke(messages)
                rejected, detected_reason = _detect_admission_rejection(response)
                if rejected:
                    is_rejected = True
                    is_error = True
                    rejection_reason = detected_reason
                    error_msg = f"Call {call_index}: admission rejected ({rejection_reason})"
                    state["is_rejected"] = True
                    state["rejection_reason"] = rejection_reason
                    state["last_call_error_msg"] = error_msg
                else:
                    response_content = response.content
                    first_token_time = start_time + (time.time() - start_time) * 0.1
            except Exception as e2:
                rejected, detected_reason = _detect_admission_rejection(e2)
                is_error = True
                err2 = str(e2).lower()
                halo_hit2 = (
                    "halo rejected" in err2
                    or "halo_reason" in err2
                    or "halo_no_job_id" in err2
                    or "halo_program_not_registered" in err2
                )
                if (
                    rejected
                    or "admission rejected" in err2
                    or "429" in err2
                    or halo_hit2
                ):
                    is_rejected = True
                    if not detected_reason and halo_hit2:
                        detected_reason = "HALO_REJECTED"
                    rejection_reason = detected_reason or "ADMISSION_REJECTED"
                    error_msg = f"Call {call_index}: admission rejected ({rejection_reason})"
                    state["is_rejected"] = True
                    state["rejection_reason"] = rejection_reason
                else:
                    error_msg = f"Call {call_index} error: {e2}"
                state["last_call_error_msg"] = error_msg

    end_time = time.time()

    if error_msg:
        state["last_call_error_msg"] = error_msg

    if not is_timeout and not is_error and not response_content:
        response_content = "".join(response_chunks)

    output_tokens = count_tokens(response_content) if response_content else 0

    # Compute TBT summary using the shared utility
    tbt_summary = summarize_tbt_ms(tbt_values_ms)
    tbt_summary.update({
        "stream_chunks": len(chunk_events),
        "streamed_output_tokens_est": streamed_output_tokens_est,
        "first_chunk_tokens_est": first_chunk_tokens_est,
    })
    tbt_detail = {
        "nonce": state.get("nonce", ""),
        "call_index": call_index,
        "start_time": start_time,
        "end_time": end_time,
        "latency_s": round(end_time - start_time, 6),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "chunk_count": len(chunk_events),
        "chunk_events": chunk_events,
    }

    # ---- Metrics recording ----
    if state.get("metrics_tracker"):
        state["metrics_tracker"].record_chain_call(
            agent_name=agent_label
            or (f"chain_call_{current_stage}" if current_stage else "chain_call"),
            start_time=start_time,
            end_time=end_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            first_token_time=first_token_time,
            call_index=call_index,
            total_calls_expected=state["chain_length"],
            is_timeout=is_timeout,
            is_error=is_error,
            is_rejected=is_rejected,
            rejection_reason=rejection_reason,
            is_job_timeout=is_job_timeout,
            job_timeout_sec=job_timeout_sec if job_timeout_sec > 0 else None,
            is_server_terminated=is_server_terminated,
            job_submit_time=state.get("job_start_time"),
            job_completed=None,
            concurrency_level=None,
            success=not (is_timeout or is_error or is_job_timeout or is_server_terminated),
            error_msg=error_msg,
            tokenizer_mode="tiktoken:gpt-3.5-turbo",
            stream_fallback_used=stream_fallback_used,
            tbt_summary=tbt_summary,
            tbt_detail=tbt_detail,
        )

    # ---- Request-id sidecar (per-request -> engine attribution) ----
    # The Llumnix scheduler logs "[Schedule] dispatch request <uuid> to
    # neutral instance <instance_id>", so pairing our task_id with the
    # server-assigned request id lets the analysis join a request to the
    # engine that served it. Written to a SIDECAR file (never into
    # metrics.csv) so the run output schema stays unchanged; the path is
    # derived from the tracker's csv path so shards (metrics.p<k>.csv ->
    # request_ids.p<k>.jsonl) work without threading a new path through
    # RunContext and every workload. See
    # experiments/DEV_request-engine-attribution.md.
    tracker = state.get("metrics_tracker")
    req_id = getattr(state.get("llm"), "last_request_id", None)
    if tracker is not None and req_id:
        csv_path = getattr(tracker, "csv_path", None)
        if csv_path:
            from metrics_tracker import MetricsTracker

            base = os.path.basename(csv_path)
            # "metrics.csv" -> "request_ids.jsonl";
            # "metrics.p3.csv" -> "request_ids.p3.jsonl"
            suffix = base[len("metrics"):-len(".csv")] if base.startswith("metrics") else ""
            sidecar = os.path.join(
                os.path.dirname(csv_path) or ".", f"request_ids{suffix}.jsonl"
            )
            MetricsTracker._get_jsonl_writer(sidecar).write({
                "task_id": tracker.current_task_id,
                "call_index": call_index,
                "request_id": req_id,
                "start_time": start_time,
                "end_time": end_time,
            })

    # ---- Transcript recording (literal-replay capture) ----
    # On a concurrency-1 baseline run with --record-transcript, append
    # this call's full prompt + solo timings to a JSONL. The
    # codingagent_request_level_poisson workload replays these verbatim
    # as independent Poisson-arriving requests. Only successful calls are
    # recorded so every transcript entry has a valid solo baseline.
    transcript_path = state.get("transcript_record_path")
    if (
        transcript_path
        and not (is_timeout or is_error or is_job_timeout or is_server_terminated)
        and response_content
    ):
        from metrics_tracker import MetricsTracker

        ttft_s = (first_token_time - start_time) if first_token_time else None
        MetricsTracker._get_jsonl_writer(transcript_path).write({
            "request_id": f"{state.get('job_id', 'job')}__call{call_index:02d}",
            "src_job_id": state.get("job_id", ""),
            "call_index": call_index,
            "stage": current_stage,
            "nonce": state.get("nonce", ""),
            "messages": _serialize_messages(messages),
            "baseline_ttft_s": round(ttft_s, 6) if ttft_s is not None else None,
            "baseline_tbt_mean_ms": tbt_summary.get("mean_ms"),
            "baseline_e2e_s": round(end_time - start_time, 6),
            "recorded_input_tokens": input_tokens,
            "recorded_output_tokens": output_tokens,
        })

    # ---- Agent logging ----
    if state.get("agent_logger"):
        state["agent_logger"].log_agent_call(
            agent_name=f"chain_call_{call_index}",
            input_prompt=full_input[:2000],  # truncate for log readability
            output_response=response_content[:2000] if response_content else error_msg,
            latency=end_time - start_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    # ---- Console output ----
    if state.get("console_write"):
        status = "REJECTED" if is_rejected else ("TIMEOUT" if is_timeout else ("ERROR" if is_error else "OK"))
        state["console_write"](
            f"  [Call {call_index}/{state['chain_length']}] "
            f"status={status} in_tokens={input_tokens} out_tokens={output_tokens} "
            f"latency={end_time - start_time:.2f}s"
        )

    if is_timeout or is_error or is_job_timeout or is_server_terminated:
        return None

    return response_content


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------
def call_chain_node(state: ChainState) -> dict:
    """
    Execute one LLM call in the chain.

    Builds messages with system prompt + accumulated conversation history +
    current user message. The current user message is determined by the
    stage for this call. If a pre-computed tool result exists, it is
    appended to the user message to simulate realistic context growth.

    On timeout/error/job-timeout/server-terminated, marks the job as
    incomplete and stops immediately.
    """
    call_index = state["call_index"]
    chain_length = state["chain_length"]
    nonce = state["nonce"]
    last_output = state.get("last_call_output", "")
    problem_statement = state.get("problem_statement", "")
    accumulated = state.get("accumulated_context", "")
    stage_sequence = state.get("stage_sequence", [])
    tool_results = state.get("tool_results", [])
    tool_call_delays = state.get("tool_call_delays", [])
    job_start_time = state.get("job_start_time", 0)
    job_timeout_sec = state.get("job_timeout_sec", 0)

    # -- Pre-check: job-level τ timeout --
    if job_timeout_sec > 0 and job_start_time > 0:
        if time.time() - job_start_time > job_timeout_sec:
            if state.get("console_write"):
                state["console_write"](
                    f"[Job {state['job_id']}] ABORTED: job exceeded {job_timeout_sec:.0f}s (τ timeout) "
                    f"before call {call_index}"
                )
            return {
                "job_completed": False,
                "error_msg": f"Job exceeded {job_timeout_sec:.0f}s (τ timeout) at call {call_index}",
                "call_index": call_index,
                "is_job_timeout": True,
                "is_server_terminated": state.get("is_server_terminated", False),
                "tool_delay_total_s": state.get("tool_delay_total_s", 0.0),
            }

    # -- Pre-check: server terminated --
    server_event = state.get("server_terminated_event")
    if server_event is not None and server_event.is_set():
        if state.get("console_write"):
            state["console_write"](
                f"[Job {state['job_id']}] ABORTED: server terminated before call {call_index}"
            )
        return {
            "job_completed": False,
            "error_msg": f"Server terminated before call {call_index}",
            "call_index": call_index,
            "is_job_timeout": state.get("is_job_timeout", False),
            "is_server_terminated": True,
            "tool_delay_total_s": state.get("tool_delay_total_s", 0.0),
        }

    # Determine the current stage
    stage_idx = call_index - 1  # 0-based index into stage_sequence
    current_stage = stage_sequence[stage_idx] if stage_idx < len(stage_sequence) else AgentStage.DEBUG
    tool_result = tool_results[stage_idx] if stage_idx < len(tool_results) else None
    tool_delay_s = (
        float(tool_call_delays[stage_idx])
        if call_index > 1 and tool_result and stage_idx < len(tool_call_delays)
        else 0.0
    )
    if tool_delay_s > 0:
        if state.get("console_write"):
            state["console_write"](
                f"[Job {state['job_id']}] Simulated tool delay before call {call_index}: {tool_delay_s:.3f}s"
            )
        abort_result = sleep_with_abort_checks(state, tool_delay_s, call_index)
        if abort_result is not None:
            return abort_result

    # -- Build messages --
    # 1) System message (~15K tokens, IDENTICAL across ALL jobs and calls)
    #    This enables cross-request prefix KV cache hits in SGLang.
    system_content = SYSTEM_PROMPT

    messages = [SystemMessage(content=system_content)]

    # 2) Conversation history (progressive context growth)
    #    accumulated_context stores "USER: ...\nASSISTANT: ...\n" pairs
    if accumulated:
        # Parse accumulated context back into messages
        # Format: "=== USER ===\n...\n=== ASSISTANT ===\n..."
        segments = accumulated.split("=== USER ===")
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            if "=== ASSISTANT ===" in seg:
                parts = seg.split("=== ASSISTANT ===", 1)
                user_text = parts[0].strip()
                assistant_text = parts[1].strip() if len(parts) > 1 else ""
                if user_text:
                    messages.append(HumanMessage(content=user_text))
                if assistant_text:
                    messages.append(AIMessage(content=assistant_text))
            else:
                if seg:
                    messages.append(HumanMessage(content=seg))

    # 3) Current user message (stage-specific)
    #    Nonce is embedded in the FIRST user message to prevent KV cache reuse
    #    across replays while keeping the system prompt identical.
    stage_prompt = STAGE_PROMPTS.get(current_stage, STAGE_PROMPTS[AgentStage.DEBUG])

    if call_index == 1:
        user_content = (
            f"[Run ID: {nonce}]\n\n"
            f"ISSUE:\n{problem_statement}\n\n"
            f"{stage_prompt}"
        )
    else:
        # Build context from previous steps
        context_parts = [f"Previous step output:\n{last_output}"]

        # Inject pre-computed tool result for this stage (KEY for context growth)
        if tool_result:
            context_parts.append(f"\nTool result:\n{tool_result}")

        user_content = (
            f"{chr(10).join(context_parts)}\n\n"
            f"{stage_prompt}"
        )

    messages.append(HumanMessage(content=user_content))

    # -- Invoke LLM with tracking --
    if state.get("console_write"):
        state["console_write"](
            f"[Job {state['job_id']}] Starting call {call_index}/{chain_length} "
            f"(stage={current_stage})"
        )

    response = invoke_with_tracking(messages, call_index, state, current_stage)

    # -- Handle failure (timeout/error/job-timeout/server-terminated) --
    if response is None:
        is_job_timeout = state.get("is_job_timeout", False)
        is_server_terminated = state.get("is_server_terminated", False)
        error_msg = state.get("last_call_error_msg") or f"Call {call_index} failed or timed out"
        return {
            "job_completed": False,
            "error_msg": error_msg,
            "call_index": call_index,  # do NOT increment
            "is_job_timeout": is_job_timeout,
            "is_server_terminated": is_server_terminated,
            "is_rejected": state.get("is_rejected", False),
            "rejection_reason": state.get("rejection_reason", ""),
            "tool_delay_total_s": state.get("tool_delay_total_s", 0.0),
        }

    # -- Success: update state --
    new_accumulated = accumulated
    new_accumulated += f"\n=== USER ===\n{user_content}\n=== ASSISTANT ===\n{response}\n"

    # Track cumulative tokens
    full_input = " ".join(
        m.content if hasattr(m, "content") else str(m) for m in messages
    )
    call_input_tokens = count_tokens(full_input)
    call_output_tokens = count_tokens(response)
    new_total_input = state.get("total_input_tokens", 0) + call_input_tokens
    new_total_output = state.get("total_output_tokens", 0) + call_output_tokens

    return {
        "call_index": call_index + 1,
        "last_call_output": response,
        "accumulated_context": new_accumulated,
        "job_completed": (call_index + 1 > chain_length),
        "error_msg": "",
        "total_input_tokens": new_total_input,
        "total_output_tokens": new_total_output,
        "is_job_timeout": False,
        "is_server_terminated": False,
        "is_rejected": False,
        "rejection_reason": "",
        "last_call_error_msg": "",
        "tool_delay_total_s": state.get("tool_delay_total_s", 0.0),
    }


def should_continue(state: ChainState) -> str:
    """
    Determine whether to continue the chain or end.

    - If server terminated: job aborted -> END
    - If error: job failed -> END
    - If call_index >= chain_length: job completed successfully -> END
    - Otherwise: continue to next call
    """
    if state.get("is_server_terminated"):
        if state.get("console_write"):
            state["console_write"](
                f"[Job {state['job_id']}] SERVER TERMINATED: {state.get('error_msg', '')}"
            )
        return END

    if state.get("error_msg"):
        if state.get("console_write"):
            state["console_write"](
                f"[Job {state['job_id']}] FAILED: {state['error_msg']}"
            )
        return END

    if state.get("job_completed"):
        if state.get("console_write"):
            state["console_write"](
                f"[Job {state['job_id']}] COMPLETED all {state['chain_length']} calls"
            )
        return END

    if state["call_index"] > state["chain_length"]:
        return END

    # Continue to next call
    if state.get("metrics_tracker"):
        state["metrics_tracker"].next_iteration()

    return "call_chain"


# ---------------------------------------------------------------------------
# LangGraph workflow
# ---------------------------------------------------------------------------
workflow = StateGraph(ChainState)
workflow.add_node("call_chain", call_chain_node)

workflow.set_entry_point("call_chain")
workflow.add_conditional_edges(
    "call_chain",
    should_continue,
    {"call_chain": "call_chain", END: END},
)

agent = workflow.compile()


# ---------------------------------------------------------------------------
# Helper: create initial state for a job
# ---------------------------------------------------------------------------
def create_chain_state(
    job_id: str,
    problem_statement: str,
    chain_length: int,
    nonce: Optional[str] = None,
    metrics_tracker: Optional[object] = None,
    agent_logger: Optional[object] = None,
    console_write: Optional[Callable] = None,
    llm: Optional[ChatOpenAI] = None,
    log_level: str = "INFO",
    job_timeout_sec: float = 0,
    job_start_time: float = 0,
    tool_call_delays: Optional[List[float]] = None,
    transcript_record_path: Optional[str] = None,
    disable_timeouts: bool = False,
) -> ChainState:
    """
    Create the initial ChainState for a synthetic chain job.

    Seeds an RNG from the nonce hash for reproducibility, then builds the
    stage sequence and pre-computes tool results for each call.

    Args:
        job_id: Unique identifier for this job.
        problem_statement: The SWE-bench problem statement used as initial context.
        chain_length: Number of sequential LLM calls (N in {5, 10, 15, 20, 30}).
        nonce: Unique string per job+replay to prevent KV cache reuse.
               If not provided, a UUID is generated. Also seeds the RNG.
        metrics_tracker: MetricsTracker instance for recording call metrics.
        agent_logger: AgentLogger instance for recording agent I/O.
        console_write: Callable for console output (defaults to print).
        llm: ChatOpenAI instance. If not provided, one is created on first call.
        log_level: Logging level string.
        job_timeout_sec: Per-job timeout in seconds (baseline_latency × τ).
                         0 means no job-level timeout.
        job_start_time: Epoch seconds when the job started. 0 means not set.
        tool_call_delays: Optional delay before each call, indexed by call_index - 1.

    Returns:
        Initial ChainState dict ready for agent.invoke().
    """
    if nonce is None:
        nonce = uuid.uuid4().hex[:12]

    # Seed RNG from nonce hash for reproducibility
    seed = int(hashlib.sha256(nonce.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    # Build stage sequence and tool results
    stage_sequence = build_stage_sequence(chain_length, rng)
    tool_results = build_tool_results(stage_sequence, rng)
    if tool_call_delays is None:
        tool_call_delays = [0.0] * chain_length

    return ChainState(
        job_id=job_id,
        problem_statement=problem_statement,
        chain_length=chain_length,
        call_index=1,  # 1-based
        accumulated_context="",
        nonce=nonce,
        stage_sequence=stage_sequence,
        tool_results=tool_results,
        log_level=log_level,
        metrics_tracker=metrics_tracker,
        agent_logger=agent_logger,
        console_write=console_write or print,
        llm=llm,
        last_call_output="",
        job_completed=False,
        error_msg="",
        rng=rng,
        job_start_time=job_start_time,
        job_timeout_sec=job_timeout_sec,
        is_job_timeout=False,
        is_server_terminated=False,
        is_rejected=False,
        rejection_reason="",
        last_call_error_msg="",
        total_input_tokens=0,
        total_output_tokens=0,
        server_terminated_event=None,
        tool_call_delays=tool_call_delays,
        tool_delay_total_s=0.0,
        transcript_record_path=transcript_record_path,
        per_call_timeout=None if disable_timeouts else PER_CALL_TIMEOUT,
        idle_timeout=None if disable_timeouts else IDLE_TIMEOUT,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Synthetic Stage-Based Chain Agent - Self-Test")
    print("=" * 70)

    # 1. Verify system prompt token count
    sp_tokens = count_tokens(SYSTEM_PROMPT)
    print(f"\n[System Prompt]")
    print(f"  Token count: {sp_tokens}")
    print(f"  Character count: {len(SYSTEM_PROMPT)}")

    # 2. Verify stage sequence generation for various chain lengths
    print(f"\n[Stage Sequence Generation]")
    for chain_len in [5, 10, 15, 20, 25, 30]:
        rng = random.Random(42)
        seq = build_stage_sequence(chain_len, rng)
        # Count stages
        from collections import Counter
        counts = Counter(seq)
        print(f"  chain_length={chain_len:2d}: {len(seq)} calls")
        print(f"    stages: {dict(counts)}")

    # 3. Verify tool result sizes
    print(f"\n[Simulated Tool Result Sizes]")
    print(f"  SIMULATED_SEARCH_RESULT: {count_tokens(SIMULATED_SEARCH_RESULT)} tokens, {len(SIMULATED_SEARCH_RESULT)} chars")
    for fname in FILE_ORDER:
        content = SIMULATED_FILE_CONTENTS[fname]
        print(f"  {fname}: {count_tokens(content)} tokens, {len(content)} chars")
    print(f"  SIMULATED_TEST_OUTPUT_PASS: {count_tokens(SIMULATED_TEST_OUTPUT_PASS)} tokens, {len(SIMULATED_TEST_OUTPUT_PASS)} chars")
    print(f"  SIMULATED_TEST_OUTPUT_FAIL: {count_tokens(SIMULATED_TEST_OUTPUT_FAIL)} tokens, {len(SIMULATED_TEST_OUTPUT_FAIL)} chars")

    # 4. Verify full state creation
    print(f"\n[Full State Creation]")
    init_state = create_chain_state(
        job_id="test_001",
        problem_statement=(
            "There is a bug in the authentication middleware that causes "
            "intermittent 403 errors for users with multiple active sessions. "
            "The issue appears to be a race condition in the session token "
            "validation logic."
        ),
        chain_length=15,
        nonce="test_nonce_42",
        metrics_tracker=None,
        agent_logger=None,
    )
    print(f"  job_id: {init_state['job_id']}")
    print(f"  chain_length: {init_state['chain_length']}")
    print(f"  stage_sequence ({len(init_state['stage_sequence'])} stages):")
    for i, stage in enumerate(init_state["stage_sequence"], 1):
        tr = init_state["tool_results"][i - 1]
        tr_info = f"{count_tokens(tr)} tokens" if tr else "None"
        print(f"    {i:2d}. {stage:12s}  tool_result={tr_info}")

    # 5. Estimate context growth across calls
    print(f"\n[Context Growth Estimate]")
    # Estimate input token count for each call
    system_tokens = sp_tokens
    problem_tokens = count_tokens(init_state["problem_statement"])
    print(f"  System prompt: {system_tokens} tokens")
    print(f"  Problem statement: {problem_tokens} tokens")
    print(f"  Estimated input token growth per call (rough):")
    for i in range(init_state["chain_length"]):
        stage = init_state["stage_sequence"][i]
        tool_result = init_state["tool_results"][i]
        tool_tokens = count_tokens(tool_result) if tool_result else 0
        # Rough estimate: system + problem + i * ~300 (prev turn) + tool_tokens
        est = system_tokens + problem_tokens + (i * 300) + tool_tokens
        print(f"    Call {i+1:2d} ({stage:12s}): ~{est:6d} tokens (tool_result adds ~{tool_tokens:4d})")

    # 6. Verify variety: consecutive LOCATE stages should get different files
    print(f"\n[Variety Check: LOCATE stages get different files]")
    rng_test = random.Random(999)
    test_seq = [AgentStage.LOCATE] * 5
    test_results = build_tool_results(test_seq, rng_test)
    for i, tr in enumerate(test_results):
        # Extract the filename from the tool result header
        first_line = tr.split("\n")[0] if tr else "None"
        print(f"  LOCATE #{i}: {first_line}")

    print(f"\n  Base URL: {BASE_URL}")
    print(f"  Model: {MODEL_ID}")
    print(f"  Per-call timeout: {PER_CALL_TIMEOUT}s")
    print(f"\nSelf-test complete.")
