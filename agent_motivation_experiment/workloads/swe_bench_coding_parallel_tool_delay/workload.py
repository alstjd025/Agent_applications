"""SWE-bench coding workload with deterministic parallel execution rounds."""

import csv
import hashlib
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from metrics_tracker import MetricsTracker
from workloads.base import JobResult, RunContext
from workloads.swe_bench_coding.agent import (
    MODEL_ID,
    SYSTEM_PROMPT,
    build_stage_sequence,
    build_tool_results,
    count_tokens,
    invoke_with_tracking,
    make_llm,
)
from workloads.swe_bench_coding.prompts.stage_prompts import AgentStage, STAGE_PROMPTS
from workloads.swe_bench_coding.workload import (
    DEFAULT_DATASET,
    DEFAULT_SPLIT,
    CyclingTaskPool,
    Workload as BaseSWEBenchWorkload,
)
from workloads.swe_bench_coding_tool_delay.workload import (
    TOOL_DELAY_BETA_ALPHA,
    TOOL_DELAY_BETA_BETA,
    TOOL_DELAY_MAX_S,
    TOOL_DELAY_MEAN_S,
    TOOL_DELAY_MIN_S,
    TOOL_DELAY_VERSION,
    build_tool_call_delays,
)


DEFAULT_MAX_PARALLEL_WIDTH = 4
PARALLEL_POLICY_VERSION = "parallel_locate_rounds_v1"

PARALLEL_FIELDNAMES = [
    "task_id",
    "base_task_id",
    "call_index",
    "total_calls_expected",
    "stage",
    "execution_round",
    "parallel_group_id",
    "round_size",
    "depends_on_call_indices",
    "tool_delay_before_round_s",
    "tool_delay_before_call_s",
    "is_round_leader",
    "is_parallel_call",
    "round_start_time",
    "round_end_time",
    "call_recorded_at",
]
_parallel_csv_lock = threading.Lock()


def build_execution_rounds(stage_sequence: list[str], max_parallel_width: int) -> list[list[int]]:
    """Group consecutive locate calls into parallel execution rounds."""
    rounds: list[list[int]] = []
    idx = 0
    while idx < len(stage_sequence):
        call_index = idx + 1
        stage = stage_sequence[idx]
        if stage == AgentStage.LOCATE:
            group = []
            while (
                idx < len(stage_sequence)
                and stage_sequence[idx] == AgentStage.LOCATE
                and len(group) < max_parallel_width
            ):
                group.append(idx + 1)
                idx += 1
            rounds.append(group)
        else:
            rounds.append([call_index])
            idx += 1
    return rounds


def _append_parallel_call_row(path: Optional[str], row: dict) -> None:
    if not path:
        return
    with _parallel_csv_lock:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PARALLEL_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow({key: row.get(key) for key in PARALLEL_FIELDNAMES})


def _format_depends_on(call_indices: list[int]) -> str:
    return ",".join(str(idx) for idx in call_indices)


def _parse_accumulated_context(accumulated: str) -> list:
    messages = []
    if not accumulated:
        return messages
    segments = accumulated.split("=== USER ===")
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if "=== ASSISTANT ===" in seg:
            user_text, assistant_text = seg.split("=== ASSISTANT ===", 1)
            user_text = user_text.strip()
            assistant_text = assistant_text.strip()
            if user_text:
                messages.append(HumanMessage(content=user_text))
            if assistant_text:
                messages.append(AIMessage(content=assistant_text))
        else:
            messages.append(HumanMessage(content=seg))
    return messages


def _build_user_content(
    call_index: int,
    nonce: str,
    problem_statement: str,
    stage: str,
    dependency_output: str,
    tool_result: Optional[str],
) -> str:
    stage_prompt = STAGE_PROMPTS.get(stage, STAGE_PROMPTS[AgentStage.DEBUG])
    if call_index == 1:
        return (
            f"[Run ID: {nonce}]\n\n"
            f"ISSUE:\n{problem_statement}\n\n"
            f"{stage_prompt}"
        )

    context_parts = [f"Previous step output:\n{dependency_output}"]
    if tool_result:
        context_parts.append(f"\nTool result:\n{tool_result}")
    return f"{chr(10).join(context_parts)}\n\n{stage_prompt}"


def _build_messages(accumulated: str, user_content: str) -> list:
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(_parse_accumulated_context(accumulated))
    messages.append(HumanMessage(content=user_content))
    return messages


def _job_status_abort(
    job_start_time: float,
    job_timeout_sec: float,
    server_terminated_event,
) -> tuple[bool, bool, str]:
    if server_terminated_event is not None and server_terminated_event.is_set():
        return False, True, "Server terminated before execution round"
    if job_timeout_sec > 0 and time.time() - job_start_time > job_timeout_sec:
        return True, False, f"Job exceeded {job_timeout_sec:.0f}s (tau timeout)"
    return False, False, ""


def _sleep_round_delay(
    delay_s: float,
    job_start_time: float,
    job_timeout_sec: float,
    server_terminated_event,
) -> tuple[float, bool, bool, str]:
    if delay_s <= 0:
        return 0.0, False, False, ""

    slept = 0.0
    last_time = time.time()
    deadline = last_time + delay_s
    while True:
        now = time.time()
        slept += max(0.0, now - last_time)
        last_time = now

        is_job_timeout, is_server_terminated, error_msg = _job_status_abort(
            job_start_time,
            job_timeout_sec,
            server_terminated_event,
        )
        if is_job_timeout or is_server_terminated:
            return slept, is_job_timeout, is_server_terminated, error_msg
        if now >= deadline:
            return slept, False, False, ""
        time.sleep(min(0.1, deadline - now))


class Workload(BaseSWEBenchWorkload):
    name = "swe_bench_coding_parallel_tool_delay"

    def __init__(self):
        self.max_parallel_width = DEFAULT_MAX_PARALLEL_WIDTH

    def create_task_pool(
        self,
        dataset,
        baseline_latencies: dict[str, float],
        rng: random.Random,
        args,
        workload_config: dict,
    ) -> CyclingTaskPool:
        self.max_parallel_width = int(
            workload_config.get("max_parallel_width", DEFAULT_MAX_PARALLEL_WIDTH)
        )
        return CyclingTaskPool(
            dataset=dataset,
            baseline_latencies=baseline_latencies,
            rng=rng,
            chain_min=args.chain_min,
            chain_max=args.chain_max,
            tau=args.tau,
        )

    def run_job(self, task: dict, context: RunContext) -> JobResult:
        job_id = task["instance_id"]
        job_submit_time = context.job_start_time
        job_timeout_sec = task.get("job_timeout_sec", 0)
        chain_length = int(task["chain_length"])
        nonce = task["nonce"]
        max_parallel_width = int(getattr(self, "max_parallel_width", DEFAULT_MAX_PARALLEL_WIDTH))

        stage_rng = random.Random(int(hashlib.sha256(nonce.encode()).hexdigest()[:8], 16))
        stage_sequence = build_stage_sequence(chain_length, stage_rng)
        tool_results = build_tool_results(stage_sequence, stage_rng)
        tool_call_delays = build_tool_call_delays(task)
        rounds = build_execution_rounds(stage_sequence, max_parallel_width)

        # HALO: pre-register the job. Parallel workload knows its full
        # stage_sequence + an execution-round structure, so we can pass
        # richer optional info to the server. Phase 1 stores it without
        # using it; Phase 2 admission will look at total_calls.
        if context.halo_enabled:
            from workloads.halo_helpers import register_halo_program

            register_halo_program(
                context.server_base_url,
                job_id=job_id,
                slo=context.halo_slo,
                total_calls=chain_length,
                stage_sequence=list(stage_sequence),
                dag={
                    "type": "parallel_rounds",
                    "rounds": [
                        {"leader": r[0], "members": list(r)} for r in rounds
                    ],
                },
            )

        accumulated_context = ""
        outputs_by_call: dict[int, str] = {}
        user_content_by_call: dict[int, str] = {}
        total_input_tokens = 0
        total_output_tokens = 0
        successful_calls = 0
        total_tool_delay_s = 0.0
        error_msg = ""
        is_job_timeout = False
        is_server_terminated = False
        is_rejected = False
        rejection_reason = ""

        def run_one_call(
            call_index: int,
            execution_round: int,
            round_size: int,
            round_leader: int,
            depends_on: list[int],
            dependency_output: str,
            round_delay_s: float,
            round_start_time: float,
        ) -> dict:
            stage_idx = call_index - 1
            stage = stage_sequence[stage_idx]
            tool_result = tool_results[stage_idx] if stage_idx < len(tool_results) else None
            user_content = _build_user_content(
                call_index=call_index,
                nonce=nonce,
                problem_statement=task["problem_statement"],
                stage=stage,
                dependency_output=dependency_output,
                tool_result=tool_result,
            )
            messages = _build_messages(accumulated_context, user_content)
            call_tracker = MetricsTracker(
                context.metrics_tracker.csv_path,
                server_base_url=context.server_base_url,
                enable_server_metrics=True,
                tbt_jsonl_path=context.metrics_tracker.tbt_jsonl_path,
            )
            call_tracker.start_task(job_id)
            call_tracker.current_iteration = call_index - 1
            # HALO: build the normal "llm" + an optional "halo_done_llm"
            # used by invoke_with_tracking for the chain's final call.
            # In parallel_tool_delay the final stage is always a
            # singleton round, so identifying "call_index == chain_length"
            # works the same as in the linear workloads.
            llm = make_llm(
                base_url=f"{context.server_base_url}/v1",
                model_id=MODEL_ID,
                seed=context.seed,
                halo_job_id=job_id if context.halo_enabled else None,
                halo_slo=context.halo_slo if context.halo_enabled else None,
            )
            halo_done_llm = None
            if context.halo_enabled:
                halo_done_llm = make_llm(
                    base_url=f"{context.server_base_url}/v1",
                    model_id=MODEL_ID,
                    seed=context.seed,
                    halo_job_id=job_id,
                    halo_slo=context.halo_slo,
                    halo_job_done=True,
                )
            state = {
                "job_id": job_id,
                "chain_length": chain_length,
                "nonce": nonce,
                "metrics_tracker": call_tracker,
                "agent_logger": context.agent_logger,
                "console_write": context.console_write,
                "llm": llm,
                "halo_done_llm": halo_done_llm,
                "job_timeout_sec": job_timeout_sec,
                "job_start_time": job_submit_time,
                "server_terminated_event": context.server_terminated_event,
                "is_job_timeout": False,
                "is_server_terminated": False,
                "is_rejected": False,
                "rejection_reason": "",
                "last_call_error_msg": "",
            }
            if context.console_write:
                context.console_write(
                    f"[Job {job_id}] Starting call {call_index}/{chain_length} "
                    f"(stage={stage}, round={execution_round}, round_size={round_size})"
                )
            response = invoke_with_tracking(messages, call_index, state, stage)
            full_input = " ".join(
                m.content if hasattr(m, "content") else str(m) for m in messages
            )
            input_tokens = count_tokens(full_input)
            output_tokens = count_tokens(response) if response else 0
            round_end_time = time.time()
            _append_parallel_call_row(
                context.parallel_calls_path,
                {
                    "task_id": job_id,
                    "base_task_id": task.get("base_instance_id", job_id),
                    "call_index": call_index,
                    "total_calls_expected": chain_length,
                    "stage": stage,
                    "execution_round": execution_round,
                    "parallel_group_id": f"{job_id}__round{execution_round:02d}",
                    "round_size": round_size,
                    "depends_on_call_indices": _format_depends_on(depends_on),
                    "tool_delay_before_round_s": round(round_delay_s, 4),
                    "tool_delay_before_call_s": round(tool_call_delays[stage_idx], 4),
                    "is_round_leader": call_index == round_leader,
                    "is_parallel_call": round_size > 1,
                    "round_start_time": round_start_time,
                    "round_end_time": round_end_time,
                    "call_recorded_at": time.time(),
                },
            )
            return {
                "call_index": call_index,
                "stage": stage,
                "response": response,
                "user_content": user_content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "is_job_timeout": bool(state.get("is_job_timeout", False)),
                "is_server_terminated": bool(state.get("is_server_terminated", False)),
                "is_rejected": bool(state.get("is_rejected", False)),
                "rejection_reason": state.get("rejection_reason", ""),
                "error_msg": state.get("last_call_error_msg", ""),
            }

        for execution_round, round_calls in enumerate(rounds, start=1):
            is_job_timeout, is_server_terminated, error_msg = _job_status_abort(
                job_submit_time,
                job_timeout_sec,
                context.server_terminated_event,
            )
            if is_job_timeout or is_server_terminated:
                break

            depends_on = rounds[execution_round - 2] if execution_round > 1 else []
            if depends_on:
                dependency_output = "\n\n".join(
                    f"Call {idx} output:\n{outputs_by_call.get(idx, '')}" for idx in depends_on
                )
            else:
                dependency_output = ""

            round_delay_s = 0.0
            for call_index in round_calls:
                stage_idx = call_index - 1
                tool_result = tool_results[stage_idx] if stage_idx < len(tool_results) else None
                if call_index > 1 and tool_result:
                    round_delay_s = max(round_delay_s, float(tool_call_delays[stage_idx]))

            if round_delay_s > 0 and context.console_write:
                context.console_write(
                    f"[Job {job_id}] Simulated tool delay before round {execution_round}: "
                    f"{round_delay_s:.3f}s"
                )
            slept, is_job_timeout, is_server_terminated, error_msg = _sleep_round_delay(
                round_delay_s,
                job_submit_time,
                job_timeout_sec,
                context.server_terminated_event,
            )
            total_tool_delay_s += slept
            if is_job_timeout or is_server_terminated:
                break

            round_start_time = time.time()
            round_results = []
            with ThreadPoolExecutor(max_workers=len(round_calls)) as executor:
                futures = [
                    executor.submit(
                        run_one_call,
                        call_index,
                        execution_round,
                        len(round_calls),
                        min(round_calls),
                        depends_on,
                        dependency_output,
                        round_delay_s,
                        round_start_time,
                    )
                    for call_index in round_calls
                ]
                for future in as_completed(futures):
                    round_results.append(future.result())

            round_results.sort(key=lambda item: item["call_index"])
            failed = [item for item in round_results if item["response"] is None]
            for item in round_results:
                total_input_tokens += item["input_tokens"]
                total_output_tokens += item["output_tokens"]
                if item["response"] is not None:
                    successful_calls += 1
                    outputs_by_call[item["call_index"]] = item["response"]
                    user_content_by_call[item["call_index"]] = item["user_content"]

            for item in round_results:
                if item["response"] is None:
                    is_job_timeout = is_job_timeout or item["is_job_timeout"]
                    is_server_terminated = is_server_terminated or item["is_server_terminated"]
                    is_rejected = is_rejected or item.get("is_rejected", False)
                    if item.get("rejection_reason") and not rejection_reason:
                        rejection_reason = item["rejection_reason"]

            if failed:
                first_failed = failed[0]
                error_msg = first_failed.get("error_msg") or f"Call {first_failed['call_index']} failed or timed out"
                break

            for item in round_results:
                accumulated_context += (
                    f"\n=== USER ===\n{item['user_content']}\n"
                    f"=== ASSISTANT ===\n{item['response']}\n"
                )

        job_end_time = time.time()
        job_completed = successful_calls == chain_length and not error_msg
        context.agent_logger.log_final_result(
            success=job_completed,
            total_time=job_end_time - job_submit_time,
            iterations=successful_calls,
        )

        return JobResult(
            job_id=job_id,
            success=job_completed,
            total_time=job_end_time - job_submit_time,
            calls_completed=successful_calls,
            chain_length=chain_length,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            error=error_msg if error_msg else None,
            is_rejected=is_rejected,
            rejection_reason=rejection_reason,
            is_job_timeout=is_job_timeout,
            is_server_terminated=is_server_terminated,
            job_timeout_sec=job_timeout_sec if job_timeout_sec > 0 else None,
            transition_time=total_tool_delay_s,
        )

    def metadata(self, args, workload_config: dict) -> dict:
        metadata = super().metadata(args, workload_config)
        metadata.update(
            {
                "name": self.name,
                "base_workload": "swe_bench_coding_tool_delay",
                "parallel_execution": {
                    "enabled": True,
                    "unit": "execution_round",
                    "policy": "group consecutive locate calls into one parallel round",
                    "max_parallel_width": int(
                        workload_config.get("max_parallel_width", DEFAULT_MAX_PARALLEL_WIDTH)
                    ),
                    "version": PARALLEL_POLICY_VERSION,
                },
                "tool_delay": {
                    "enabled": True,
                    "applies_to": "rounds containing calls whose prompt includes a simulated tool result",
                    "round_delay": "max deterministic call delay within the execution round",
                    "min_s": TOOL_DELAY_MIN_S,
                    "max_s": TOOL_DELAY_MAX_S,
                    "mean_s": TOOL_DELAY_MEAN_S,
                    "distribution": "scaled_beta",
                    "beta_alpha": TOOL_DELAY_BETA_ALPHA,
                    "beta_beta": TOOL_DELAY_BETA_BETA,
                    "version": TOOL_DELAY_VERSION,
                },
            }
        )
        return metadata

    def reproducibility_config(self, args, workload_config: dict) -> dict:
        config = super().reproducibility_config(args, workload_config)
        config.update(
            {
                "parallel_policy_seed": "deterministic from stage sequence; no extra RNG",
                "parallel_policy": PARALLEL_POLICY_VERSION,
                "max_parallel_width": int(
                    workload_config.get("max_parallel_width", DEFAULT_MAX_PARALLEL_WIDTH)
                ),
                "tool_delay_seed": (
                    "sha256(tool_delay_beta_v1|base_instance_id|replay_index|boundary_call_index)"
                ),
                "tool_delay_distribution": {
                    "min_s": TOOL_DELAY_MIN_S,
                    "max_s": TOOL_DELAY_MAX_S,
                    "mean_s": TOOL_DELAY_MEAN_S,
                    "beta_alpha": TOOL_DELAY_BETA_ALPHA,
                    "beta_beta": TOOL_DELAY_BETA_BETA,
                },
                "dataset": workload_config.get("dataset", DEFAULT_DATASET),
                "split": workload_config.get("split", DEFAULT_SPLIT),
            }
        )
        return config
