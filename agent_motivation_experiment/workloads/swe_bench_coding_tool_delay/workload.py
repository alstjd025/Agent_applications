"""SWE-bench coding workload with deterministic simulated tool-call delays."""

import hashlib
import random
import time

from workloads.base import JobResult, RunContext
from workloads.swe_bench_coding.agent import (
    MODEL_ID,
    agent,
    create_chain_state,
    make_llm,
)
from workloads.swe_bench_coding.workload import (
    DEFAULT_DATASET,
    DEFAULT_SPLIT,
    CyclingTaskPool,
    Workload as BaseSWEBenchWorkload,
)


TOOL_DELAY_MIN_S = 0.1
TOOL_DELAY_MAX_S = 10.0
TOOL_DELAY_MEAN_S = 3.0
TOOL_DELAY_BETA_ALPHA = 2.0
TOOL_DELAY_BETA_BETA = (
    TOOL_DELAY_BETA_ALPHA
    * (1.0 - ((TOOL_DELAY_MEAN_S - TOOL_DELAY_MIN_S) / (TOOL_DELAY_MAX_S - TOOL_DELAY_MIN_S)))
    / ((TOOL_DELAY_MEAN_S - TOOL_DELAY_MIN_S) / (TOOL_DELAY_MAX_S - TOOL_DELAY_MIN_S))
)
TOOL_DELAY_VERSION = "tool_delay_beta_v1"


def deterministic_tool_delay_s(task: dict, call_index: int) -> float:
    """Return deterministic delay before `call_index` for this logical task.

    `call_index` is 1-based. The delay before call 1 is always zero because
    no prior model response could have requested a tool.
    """
    if call_index <= 1:
        return 0.0

    key = "|".join(
        [
            TOOL_DELAY_VERSION,
            str(task.get("base_instance_id") or task.get("instance_id", "")),
            str(task.get("replay_index", "")),
            str(call_index - 1),
        ]
    )
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    unit = rng.betavariate(TOOL_DELAY_BETA_ALPHA, TOOL_DELAY_BETA_BETA)
    return TOOL_DELAY_MIN_S + (TOOL_DELAY_MAX_S - TOOL_DELAY_MIN_S) * unit


def build_tool_call_delays(task: dict) -> list[float]:
    """Build delay list indexed by call_index - 1."""
    chain_length = int(task["chain_length"])
    return [deterministic_tool_delay_s(task, call_index) for call_index in range(1, chain_length + 1)]


class Workload(BaseSWEBenchWorkload):
    name = "swe_bench_coding_tool_delay"

    def create_task_pool(
        self,
        dataset,
        baseline_latencies: dict[str, float],
        rng: random.Random,
        args,
        workload_config: dict,
    ) -> CyclingTaskPool:
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

        # HALO: pre-register the job before any LLM call. Same pattern
        # as the base swe_bench_coding workload — see that file and
        # workloads/halo_helpers.py.
        if context.halo_enabled:
            from workloads.halo_helpers import register_halo_program

            register_halo_program(
                context.server_base_url,
                job_id=job_id,
                slo=context.halo_slo,
                total_calls=task["chain_length"],
            )

        llm = make_llm(
            base_url=f"{context.server_base_url}/v1",
            model_id=MODEL_ID,
            seed=context.seed,
            halo_job_id=job_id if context.halo_enabled else None,
            halo_slo=context.halo_slo if context.halo_enabled else None,
        )
        initial_state = create_chain_state(
            job_id=job_id,
            problem_statement=task["problem_statement"],
            chain_length=task["chain_length"],
            nonce=task["nonce"],
            metrics_tracker=context.metrics_tracker,
            agent_logger=context.agent_logger,
            console_write=context.console_write,
            llm=llm,
            log_level=context.log_level,
            job_timeout_sec=job_timeout_sec,
            job_start_time=job_submit_time,
            tool_call_delays=build_tool_call_delays(task),
        )
        initial_state["server_terminated_event"] = context.server_terminated_event

        result = agent.invoke(initial_state)
        job_end_time = time.time()

        job_completed = result.get("job_completed", False)
        call_index = result.get("call_index", 0)
        error_msg = result.get("error_msg", "")
        tool_delay_total_s = float(result.get("tool_delay_total_s", 0.0) or 0.0)

        context.agent_logger.log_final_result(
            success=job_completed,
            total_time=job_end_time - job_submit_time,
            iterations=call_index,
        )

        return JobResult(
            job_id=job_id,
            success=job_completed,
            total_time=job_end_time - job_submit_time,
            calls_completed=call_index - 1 if not job_completed and call_index > 0 else call_index,
            chain_length=task["chain_length"],
            total_input_tokens=result.get("total_input_tokens", 0),
            total_output_tokens=result.get("total_output_tokens", 0),
            error=error_msg if error_msg else None,
            is_rejected=result.get("is_rejected", False),
            rejection_reason=result.get("rejection_reason", ""),
            is_job_timeout=result.get("is_job_timeout", False),
            is_server_terminated=result.get("is_server_terminated", False),
            job_timeout_sec=job_timeout_sec if job_timeout_sec > 0 else None,
            transition_time=tool_delay_total_s,
        )

    def metadata(self, args, workload_config: dict) -> dict:
        metadata = super().metadata(args, workload_config)
        metadata.update(
            {
                "name": self.name,
                "base_workload": "swe_bench_coding",
                "tool_delay": {
                    "enabled": True,
                    "applies_to": "calls whose prompt includes a simulated tool result",
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
