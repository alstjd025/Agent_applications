"""
Enhanced metrics collection and CSV persistence module for the motivation experiment.

Extends the original MetricsTracker with:
- call_index / total_calls_expected for per-call ordering within a job
- job_submit_time / job_end_time / job_completed for job-level lifecycle tracking
- is_timeout / is_error for failure classification
- concurrency_level for the observed concurrency at call time
- record_chain_call() for the synthetic chain agent
- record_job_summary() for per-job aggregated rows
- Thread-safe CSV writes via a shared threading.Lock
"""

import csv
import json
import os
import queue
import requests
import re
import statistics
import threading
import time
import atexit
from datetime import datetime
from typing import Optional, Dict, Any

try:
    import GPUtil
except Exception:
    GPUtil = None


# ---------------------------------------------------------------------------
# KVCacheMonitor
# ---------------------------------------------------------------------------

class KVCacheMonitor:
    """Server /metrics endpoint based KV cache usage monitor.

    Notes:
    - Currently parses vLLM Prometheus metric names first.
    - Falls back to block-based calculation if the percentage metric is absent.
    - Always returns None on any failure (network, parsing, etc.).
    """

    def __init__(self, base_url: str = "http://localhost:30000", enabled: bool = True):
        self.metrics_url = f"{base_url}/metrics"
        self.enabled = enabled

    def get_kv_cache_usage(self) -> Optional[float]:
        """Return KV cache usage as a percentage (0-100), or None on failure."""
        if not self.enabled:
            return None
        try:
            response = requests.get(self.metrics_url, timeout=2)
            response.raise_for_status()

            metrics_text = response.text

            # Direct percentage metric
            match = re.search(r'vllm:gpu_cache_usage_perc\s+([\d.]+)', metrics_text)
            if match:
                return float(match.group(1)) * 100  # 0-1 -> 0-100%

            # Fallback: block-based calculation
            blocks_used_match = re.search(
                r'vllm:gpu_cache_usage_blocks\s+([\d.]+)', metrics_text
            )
            blocks_total_match = re.search(
                r'vllm:gpu_cache_total_blocks\s+([\d.]+)', metrics_text
            )

            if blocks_used_match and blocks_total_match:
                blocks_used = float(blocks_used_match.group(1))
                blocks_total = float(blocks_total_match.group(1))
                if blocks_total > 0:
                    return (blocks_used / blocks_total) * 100

            return None

        except Exception:
            return None


# ---------------------------------------------------------------------------
# Module-level lock shared by all MetricsTracker instances for CSV writes
# ---------------------------------------------------------------------------

_csv_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# MetricsTracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    """Agent execution metrics tracker with CSV persistence.

    Thread safety:
        All CSV writes are serialised through a module-level threading.Lock so
        that multiple concurrent jobs can safely append to the same file.
    """

    _jsonl_writers: Dict[str, "AsyncJSONLWriter"] = {}
    _jsonl_writers_lock = threading.Lock()
    _atexit_registered = False

    # Complete field list (in order)
    FIELDNAMES = [
        'task_id',
        'iteration',
        'agent',
        'call_index',
        'total_calls_expected',
        'start_time',
        'end_time',
        'latency',
        'input_tokens',
        'output_tokens',
        'first_token_latency',
        'decode_speed_tps',
        'gpu_memory_mb',
        'kv_cache_usage_pct',
        'transition_time',
        'tokenizer_mode',
        'stream_fallback_used',
        'tbt_available',
        'stream_chunks',
        'streamed_output_tokens_est',
        'first_chunk_tokens_est',
        'tbt_mean_ms',
        'tbt_p50_ms',
        'tbt_p75_ms',
        'tbt_p80_ms',
        'tbt_p85_ms',
        'tbt_p90_ms',
        'tbt_p95_ms',
        'tbt_max_ms',
        'tbt_sample_count',
        'is_timeout',
        'is_error',
        'is_job_timeout',
        'job_timeout_sec',
        'is_server_terminated',
        'job_submit_time',
        'job_end_time',
        'job_completed',
        'concurrency_level',
        'success',
        'error_msg',
        'timestamp',
    ]

    def __init__(
        self,
        csv_path: str,
        server_base_url: str = "http://localhost:30000",
        enable_server_metrics: bool = True,
        tbt_jsonl_path: Optional[str] = None,
    ):
        """
        Args:
            csv_path: Path to the CSV file for metrics persistence.
            server_base_url: Inference server base URL.
            enable_server_metrics: Whether to query /metrics for server-side stats.
                Defaults to True; the KVCacheMonitor returns None gracefully when
                the endpoint is unavailable.
            tbt_jsonl_path: Optional path for per-chunk TBT detail JSONL sidecar.
        """
        self.csv_path = csv_path
        self.current_task_id = None
        self.current_iteration = 0
        self.last_agent_end_time = None
        self.tbt_jsonl_path = tbt_jsonl_path

        # KV cache monitor (enabled by default, returns None on failure)
        self.kv_monitor = KVCacheMonitor(server_base_url, enabled=enable_server_metrics)

        # CSV file initialisation (write header if file does not exist)
        if not os.path.exists(self.csv_path):
            os.makedirs(os.path.dirname(self.csv_path) or '.', exist_ok=True)
            with _csv_write_lock:
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                    writer.writeheader()

        self._register_atexit()
        self.tbt_writer = (
            self._get_jsonl_writer(self.tbt_jsonl_path)
            if self.tbt_jsonl_path
            else None
        )

    # ------------------------------------------------------------------
    # Atexit / JSONL writer management
    # ------------------------------------------------------------------

    @classmethod
    def _register_atexit(cls):
        if cls._atexit_registered:
            return
        atexit.register(cls.shutdown_all_writers)
        cls._atexit_registered = True

    @classmethod
    def _get_jsonl_writer(cls, path: str) -> "AsyncJSONLWriter":
        with cls._jsonl_writers_lock:
            writer = cls._jsonl_writers.get(path)
            if writer is None:
                writer = AsyncJSONLWriter(path)
                cls._jsonl_writers[path] = writer
            return writer

    @classmethod
    def shutdown_all_writers(cls):
        with cls._jsonl_writers_lock:
            writers = list(cls._jsonl_writers.values())
            cls._jsonl_writers = {}
        for writer in writers:
            writer.close()

    # ------------------------------------------------------------------
    # Task / iteration bookkeeping
    # ------------------------------------------------------------------

    def start_task(self, task_id: str):
        """Start a new task (resets iteration counter and transition time)."""
        self.current_task_id = task_id
        self.current_iteration = 0
        self.last_agent_end_time = None

    def next_iteration(self):
        """Advance to the next iteration."""
        self.current_iteration += 1
        self.last_agent_end_time = None

    # ------------------------------------------------------------------
    # Core recording helpers
    # ------------------------------------------------------------------

    def _write_csv_row(self, record: Dict[str, Any]):
        """Append a single row to the CSV under the module-level lock."""
        with _csv_write_lock:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writerow(record)

    def _compute_common_fields(
        self,
        start_time: float,
        end_time: float,
        input_tokens: int,
        output_tokens: int,
        first_token_time: Optional[float],
    ) -> Dict[str, Any]:
        """Compute latency, first-token-latency, decode speed, GPU mem, KV cache, transition."""
        latency = end_time - start_time

        first_token_latency = None
        if first_token_time is not None:
            first_token_latency = first_token_time - start_time

        decode_speed_tps = None
        if first_token_time is not None and output_tokens > 0:
            decode_time = end_time - first_token_time
            if decode_time > 0:
                decode_speed_tps = output_tokens / decode_time

        gpu_memory_mb = self._get_gpu_memory()
        kv_cache_usage_pct = self.kv_monitor.get_kv_cache_usage()

        transition_time = None
        if self.last_agent_end_time is not None:
            transition_time = start_time - self.last_agent_end_time

        # Store for next transition calculation
        self.last_agent_end_time = end_time

        return {
            'start_time': start_time,
            'end_time': end_time,
            'latency': round(latency, 4),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'first_token_latency': (
                round(first_token_latency, 4) if first_token_latency else None
            ),
            'decode_speed_tps': (
                round(decode_speed_tps, 2) if decode_speed_tps else None
            ),
            'gpu_memory_mb': gpu_memory_mb,
            'kv_cache_usage_pct': (
                round(kv_cache_usage_pct, 2)
                if kv_cache_usage_pct is not None
                else None
            ),
            'transition_time': (
                round(transition_time, 4) if transition_time else None
            ),
        }

    # ------------------------------------------------------------------
    # record_chain_call  (NEW - primary method for synthetic chain agent)
    # ------------------------------------------------------------------

    def record_chain_call(
        self,
        agent_name: str,  # typically "chain_call"
        start_time: float,
        end_time: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        first_token_time: Optional[float] = None,
        call_index: int = 0,
        total_calls_expected: int = 0,
        is_timeout: Optional[bool] = None,
        is_error: Optional[bool] = None,
        is_job_timeout: Optional[bool] = None,
        job_timeout_sec: Optional[float] = None,
        is_server_terminated: Optional[bool] = None,
        job_submit_time: Optional[float] = None,
        job_end_time: Optional[float] = None,
        job_completed: Optional[bool] = None,
        concurrency_level: Optional[int] = None,
        success: Optional[bool] = None,
        error_msg: str = "",
        tokenizer_mode: Optional[str] = None,
        stream_fallback_used: bool = False,
        tbt_summary: Optional[Dict[str, Any]] = None,
        tbt_detail: Optional[Dict[str, Any]] = None,
    ):
        """Record a single call within a synthetic chain job.

        Args:
            agent_name: Agent identifier (e.g. ``"chain_call"``).
            start_time: Epoch seconds when the call started.
            end_time: Epoch seconds when the call ended.
            input_tokens: Prompt token count.
            output_tokens: Generated token count.
            first_token_time: Epoch seconds when the first token was received.
            call_index: 1-based index of this call within the job.
            total_calls_expected: Expected total number of calls for this job.
            is_timeout: True if this call failed due to call-level timeout.
            is_error: True if this call failed due to an error.
            is_job_timeout: True if this call was aborted due to job-level τ timeout.
            job_timeout_sec: The τ-based job timeout in seconds (baseline_latency × τ).
            is_server_terminated: True if this call was aborted because the server was terminated.
            job_submit_time: Epoch seconds when the job was submitted.
            job_end_time: Epoch seconds when the job ended.
            job_completed: Whether the job completed all calls successfully.
            concurrency_level: Observed concurrency level at call time.
            success: Whether this call succeeded.
            error_msg: Error message (empty string if none).
            tokenizer_mode: Tokenizer mode used for this call.
            stream_fallback_used: Whether streaming fell back to non-streaming.
            tbt_summary: Summary dict from ``summarize_tbt_ms()``.
            tbt_detail: Raw TBT detail dict (written to JSONL sidecar).
        """
        common = self._compute_common_fields(
            start_time, end_time, input_tokens, output_tokens, first_token_time
        )

        tbt_summary = tbt_summary or {}

        record = {
            'task_id': self.current_task_id,
            'iteration': self.current_iteration,
            'agent': agent_name,
            'call_index': call_index,
            'total_calls_expected': total_calls_expected,
            **common,
            'tokenizer_mode': tokenizer_mode,
            'stream_fallback_used': stream_fallback_used,
            'tbt_available': tbt_summary.get('available'),
            'stream_chunks': tbt_summary.get('stream_chunks'),
            'streamed_output_tokens_est': tbt_summary.get('streamed_output_tokens_est'),
            'first_chunk_tokens_est': tbt_summary.get('first_chunk_tokens_est'),
            'tbt_mean_ms': (
                round(tbt_summary['mean_ms'], 4)
                if tbt_summary.get('mean_ms') is not None
                else None
            ),
            'tbt_p50_ms': (
                round(tbt_summary['p50_ms'], 4)
                if tbt_summary.get('p50_ms') is not None
                else None
            ),
            'tbt_p75_ms': (
                round(tbt_summary['p75_ms'], 4)
                if tbt_summary.get('p75_ms') is not None
                else None
            ),
            'tbt_p80_ms': (
                round(tbt_summary['p80_ms'], 4)
                if tbt_summary.get('p80_ms') is not None
                else None
            ),
            'tbt_p85_ms': (
                round(tbt_summary['p85_ms'], 4)
                if tbt_summary.get('p85_ms') is not None
                else None
            ),
            'tbt_p90_ms': (
                round(tbt_summary['p90_ms'], 4)
                if tbt_summary.get('p90_ms') is not None
                else None
            ),
            'tbt_p95_ms': (
                round(tbt_summary['p95_ms'], 4)
                if tbt_summary.get('p95_ms') is not None
                else None
            ),
            'tbt_max_ms': (
                round(tbt_summary['max_ms'], 4)
                if tbt_summary.get('max_ms') is not None
                else None
            ),
            'tbt_sample_count': tbt_summary.get('sample_count'),
            'is_timeout': is_timeout,
            'is_error': is_error,
            'is_job_timeout': is_job_timeout,
            'job_timeout_sec': job_timeout_sec,
            'is_server_terminated': is_server_terminated,
            'job_submit_time': job_submit_time,
            'job_end_time': job_end_time,
            'job_completed': job_completed,
            'concurrency_level': concurrency_level,
            'success': success,
            'error_msg': error_msg,
            'timestamp': datetime.now().isoformat(),
        }

        self._write_csv_row(record)

        # TBT detail sidecar
        if self.tbt_writer and tbt_detail is not None:
            detail_record = {
                'task_id': self.current_task_id,
                'iteration': self.current_iteration,
                'agent': agent_name,
                'call_index': call_index,
                'total_calls_expected': total_calls_expected,
                'tokenizer_mode': tokenizer_mode,
                'stream_fallback_used': stream_fallback_used,
                'recorded_at': datetime.now().isoformat(),
                **tbt_detail,
            }
            self.tbt_writer.write(detail_record)

        return record

    # ------------------------------------------------------------------
    # record_agent_call  (KEPT from original, with new optional fields)
    # ------------------------------------------------------------------

    def record_agent_call(
        self,
        agent_name: str,
        start_time: float,
        end_time: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        first_token_time: Optional[float] = None,
        success: Optional[bool] = None,
        error_msg: str = "",
        tokenizer_mode: Optional[str] = None,
        stream_fallback_used: bool = False,
        tbt_summary: Optional[Dict[str, Any]] = None,
        tbt_detail: Optional[Dict[str, Any]] = None,
        # --- new optional fields ---
        call_index: int = 0,
        total_calls_expected: int = 0,
        is_timeout: Optional[bool] = None,
        is_error: Optional[bool] = None,
        is_job_timeout: Optional[bool] = None,
        job_timeout_sec: Optional[float] = None,
        is_server_terminated: Optional[bool] = None,
        job_submit_time: Optional[float] = None,
        job_end_time: Optional[float] = None,
        job_completed: Optional[bool] = None,
        concurrency_level: Optional[int] = None,
    ):
        """Record an agent call (backward-compatible with original signature).

        All new fields default to None / 0 / False so that existing callers
        continue to work without modification.
        """
        return self.record_chain_call(
            agent_name=agent_name,
            start_time=start_time,
            end_time=end_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            first_token_time=first_token_time,
            call_index=call_index,
            total_calls_expected=total_calls_expected,
            is_timeout=is_timeout,
            is_error=is_error,
            is_job_timeout=is_job_timeout,
            job_timeout_sec=job_timeout_sec,
            is_server_terminated=is_server_terminated,
            job_submit_time=job_submit_time,
            job_end_time=job_end_time,
            job_completed=job_completed,
            concurrency_level=concurrency_level,
            success=success,
            error_msg=error_msg,
            tokenizer_mode=tokenizer_mode,
            stream_fallback_used=stream_fallback_used,
            tbt_summary=tbt_summary,
            tbt_detail=tbt_detail,
        )

    # ------------------------------------------------------------------
    # record_job_summary  (NEW)
    # ------------------------------------------------------------------

    def record_job_summary(
        self,
        job_id: str,
        chain_length: int,
        calls_completed: int,
        job_completed: bool,
        job_submit_time: float,
        job_end_time: float,
        total_input_tokens: int = 0,
        total_output_tokens: int = 0,
        wasted_input_tokens: int = 0,
        wasted_output_tokens: int = 0,
        concurrency_level: Optional[int] = None,
        error_msg: str = "",
        is_job_timeout: Optional[bool] = None,
        job_timeout_sec: Optional[float] = None,
        is_server_terminated: Optional[bool] = None,
    ):
        """Record a job-level summary row after a job completes or fails.

        The row has ``agent="job_summary"`` and carries aggregated metrics for
        the entire job lifecycle.

        Args:
            job_id: Unique job identifier (written to ``task_id`` column).
            chain_length: Expected number of calls (``total_calls_expected``).
            calls_completed: Number of calls that finished successfully.
            job_completed: Whether all calls in the job completed.
            job_submit_time: Epoch seconds when the job was submitted.
            job_end_time: Epoch seconds when the job ended.
            total_input_tokens: Sum of input tokens across all calls.
            total_output_tokens: Sum of output tokens across all calls.
            wasted_input_tokens: Input tokens from calls in an incomplete job.
            wasted_output_tokens: Output tokens from calls in an incomplete job.
            concurrency_level: Concurrency level during this job.
            error_msg: Error message if the job failed.
            is_job_timeout: True if the job was aborted due to τ timeout.
            job_timeout_sec: The τ-based job timeout in seconds.
            is_server_terminated: True if the job was aborted because the server was terminated.
        """
        jct = job_end_time - job_submit_time if job_end_time and job_submit_time else None

        record = {
            'task_id': job_id,
            'iteration': self.current_iteration,
            'agent': 'job_summary',
            'call_index': calls_completed,
            'total_calls_expected': chain_length,
            'start_time': job_submit_time,
            'end_time': job_end_time,
            'latency': round(jct, 4) if jct is not None else None,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'first_token_latency': None,
            'decode_speed_tps': None,
            'gpu_memory_mb': None,
            'kv_cache_usage_pct': None,
            'transition_time': None,
            'tokenizer_mode': None,
            'stream_fallback_used': False,
            'tbt_available': None,
            'stream_chunks': None,
            'streamed_output_tokens_est': None,
            'first_chunk_tokens_est': None,
            'tbt_mean_ms': None,
            'tbt_p50_ms': None,
            'tbt_p75_ms': None,
            'tbt_p80_ms': None,
            'tbt_p85_ms': None,
            'tbt_p90_ms': None,
            'tbt_p95_ms': None,
            'tbt_max_ms': None,
            'tbt_sample_count': None,
            'is_timeout': None,
            'is_error': None if job_completed else (not job_completed or None),
            'is_job_timeout': is_job_timeout,
            'job_timeout_sec': job_timeout_sec,
            'is_server_terminated': is_server_terminated,
            'job_submit_time': job_submit_time,
            'job_end_time': job_end_time,
            'job_completed': job_completed,
            'concurrency_level': concurrency_level,
            'success': job_completed,
            'error_msg': error_msg,
            'timestamp': datetime.now().isoformat(),
        }

        self._write_csv_row(record)
        return record

    # ------------------------------------------------------------------
    # GPU memory helper
    # ------------------------------------------------------------------

    def _get_gpu_memory(self) -> Optional[int]:
        """Return current GPU memory usage in MB, or None on failure.

        Note: vLLM pre-allocates ``--gpu-memory-utilization`` at startup, so
        this value rarely changes.  Real variation is in the internal KV cache
        usage which is captured via ``kv_cache_usage_pct``.
        """
        if GPUtil is None:
            return None
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return int(gpus[0].memoryUsed)
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Load completed tasks (for restart-skip)
    # ------------------------------------------------------------------

    @staticmethod
    def load_completed_tasks(csv_path: str) -> set:
        """Load the set of task_ids that have already completed.

        A task is considered completed if a row with ``agent='debugging'``
        and ``success='True'`` exists, or if a ``job_summary`` row has
        ``job_completed='True'``.
        """
        completed = set()
        if not os.path.exists(csv_path):
            return completed

        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    agent = row.get('agent', '')
                    if agent == 'debugging' and row.get('success') == 'True':
                        completed.add(row['task_id'])
                    elif agent == 'job_summary' and row.get('job_completed') == 'True':
                        completed.add(row['task_id'])
        except Exception as e:
            print(f"Warning: Failed to load completed tasks: {e}")

        return completed


# ---------------------------------------------------------------------------
# StreamingTokenTracker
# ---------------------------------------------------------------------------

class StreamingTokenTracker:
    """Track first-token time and total token count from a streaming response."""

    def __init__(self):
        self.start_time = None
        self.first_token_time = None
        self.token_count = 0
        self.full_response = ""

    def reset(self):
        """Reset for a new request."""
        self.start_time = time.time()
        self.first_token_time = None
        self.token_count = 0
        self.full_response = ""

    def on_token(self, token: str):
        """Called when a token is received."""
        if self.first_token_time is None:
            self.first_token_time = time.time()

        self.token_count += 1
        self.full_response += token

    def get_metrics(self) -> Dict[str, Any]:
        """Return collected metrics."""
        return {
            'first_token_time': self.first_token_time,
            'tokens_generated': self.token_count,
            'full_response': self.full_response,
        }


# ---------------------------------------------------------------------------
# AsyncJSONLWriter
# ---------------------------------------------------------------------------

class AsyncJSONLWriter:
    """Write JSONL records asynchronously from a background thread."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def write(self, record: Dict[str, Any]):
        self._queue.put(record)

    def _run(self):
        with open(self.path, 'a', encoding='utf-8') as f:
            while True:
                item = self._queue.get()
                if item is None:
                    self._queue.task_done()
                    break
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                f.flush()
                self._queue.task_done()

    def close(self):
        self._queue.put(None)
        self._queue.join()
        self._thread.join()


# ---------------------------------------------------------------------------
# summarize_tbt_ms
# ---------------------------------------------------------------------------

def summarize_tbt_ms(values_ms):
    """Compute summary statistics for a list of inter-token latency values (ms)."""
    if not values_ms:
        return {
            'available': False,
            'mean_ms': None,
            'p50_ms': None,
            'p75_ms': None,
            'p80_ms': None,
            'p85_ms': None,
            'p90_ms': None,
            'p95_ms': None,
            'max_ms': None,
            'sample_count': 0,
        }

    sorted_vals = sorted(values_ms)

    def percentile_ms(pct: float):
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        position = (pct / 100.0) * (len(sorted_vals) - 1)
        lower_idx = int(position)
        upper_idx = min(lower_idx + 1, len(sorted_vals) - 1)
        fraction = position - lower_idx
        lower = sorted_vals[lower_idx]
        upper = sorted_vals[upper_idx]
        return lower + (upper - lower) * fraction

    return {
        'available': True,
        'mean_ms': statistics.fmean(sorted_vals),
        'p50_ms': percentile_ms(50),
        'p75_ms': percentile_ms(75),
        'p80_ms': percentile_ms(80),
        'p85_ms': percentile_ms(85),
        'p90_ms': percentile_ms(90),
        'p95_ms': percentile_ms(95),
        'max_ms': sorted_vals[-1],
        'sample_count': len(sorted_vals),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    temp_csv = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
    temp_csv.close()

    tracker = MetricsTracker(temp_csv.name, enable_server_metrics=False)

    # --- Test record_chain_call ---
    tracker.start_task("chain_job_1")

    job_submit = time.time()
    start = time.time()
    time.sleep(0.05)
    first_token = time.time()
    time.sleep(0.05)
    end = time.time()

    tracker.record_chain_call(
        agent_name="chain_call",
        start_time=start,
        end_time=end,
        input_tokens=200,
        output_tokens=80,
        first_token_time=first_token,
        call_index=1,
        total_calls_expected=5,
        job_submit_time=job_submit,
        concurrency_level=8,
        success=True,
    )

    # --- Test record_agent_call (backward compat) ---
    tracker.start_task("legacy_task_1")
    tracker.record_agent_call(
        agent_name="planning",
        start_time=start,
        end_time=end,
        input_tokens=150,
        output_tokens=100,
        first_token_time=first_token,
    )

    # --- Test record_job_summary ---
    job_end = time.time()
    tracker.record_job_summary(
        job_id="chain_job_1",
        chain_length=5,
        calls_completed=5,
        job_completed=True,
        job_submit_time=job_submit,
        job_end_time=job_end,
        total_input_tokens=1000,
        total_output_tokens=400,
        concurrency_level=8,
    )

    print(f"Test metrics saved to: {temp_csv.name}")

    # Verify contents
    with open(temp_csv.name, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"Rows written: {len(rows)}")
        for i, row in enumerate(rows):
            print(f"  Row {i}: agent={row['agent']}, task_id={row['task_id']}, "
                  f"call_index={row['call_index']}, job_completed={row['job_completed']}")

    # Test load_completed_tasks
    completed = MetricsTracker.load_completed_tasks(temp_csv.name)
    print(f"Completed tasks: {completed}")

    os.unlink(temp_csv.name)
