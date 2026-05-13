"""SWE-bench synthetic coding workload adapter."""

import random
import threading
import time
from typing import Optional

from datasets import load_dataset

from workloads.base import JobResult, RunContext, TaskLogInfo
from workloads.swe_bench_coding.agent import (
    MODEL_ID,
    SYSTEM_PROMPT,
    agent,
    count_tokens,
    create_chain_state,
    make_llm,
)


DEFAULT_DATASET = "princeton-nlp/SWE-bench_Lite"
DEFAULT_SPLIT = "test"


def sample_chain_length(rng: random.Random, chain_min: int, chain_max: int) -> int:
    """Sample a chain length uniformly from [chain_min, chain_max]."""
    return rng.randint(chain_min, chain_max)


def base_task_id(task_id: str) -> str:
    """Strip replay suffix from logical task IDs."""
    return task_id.split("__replay")[0] if "__replay" in task_id else task_id


class CyclingTaskPool:
    """Generate SWE-bench tasks on-the-fly from a deterministic cycling pool."""

    def __init__(
        self,
        dataset,
        baseline_latencies: dict[str, float],
        rng: random.Random,
        chain_min: int,
        chain_max: int,
        tau: float = 2.0,
    ):
        self.dataset = dataset
        self.baseline_latencies = baseline_latencies
        self.rng = rng
        self.chain_min = chain_min
        self.chain_max = chain_max
        self.tau = tau
        self._replay_counter = 0
        self._lock = threading.Lock()

    def next_task(self) -> Optional[dict]:
        with self._lock:
            base_idx = self._replay_counter % len(self.dataset)
            replay_num = self._replay_counter // len(self.dataset) + 1
            chain_length = sample_chain_length(self.rng, self.chain_min, self.chain_max)
            self._replay_counter += 1

        task = self.dataset[base_idx]
        base_id = task["instance_id"]
        logical_id = f"{base_id}__replay{replay_num:02d}"
        nonce = f"replay{replay_num:02d}-idx{base_idx:04d}-{base_id}"
        baseline_latency = self.baseline_latencies.get(base_id)
        job_timeout_sec = baseline_latency * self.tau if baseline_latency else 0

        return {
            "instance_id": logical_id,
            "base_instance_id": base_id,
            "problem_statement": task["problem_statement"],
            "repo": task.get("repo", ""),
            "replay_index": replay_num,
            "logical_index": base_idx,
            "nonce": nonce,
            "chain_length": chain_length,
            "job_timeout_sec": job_timeout_sec,
            "baseline_latency": baseline_latency,
        }


class Workload:
    name = "swe_bench_coding"

    def load_dataset(self, args, workload_config: dict):
        dataset_name = workload_config.get("dataset", DEFAULT_DATASET)
        split = workload_config.get("split", DEFAULT_SPLIT)
        print(f"Loading workload dataset: {dataset_name} split={split}")
        return load_dataset(dataset_name, split=split)

    def build_baseline_tasks(
        self,
        dataset,
        replay_count: int,
        rng: random.Random,
        args,
        workload_config: dict,
    ) -> list[dict]:
        tasks = []
        base_size = len(dataset)
        for replay_idx in range(replay_count):
            replay_num = replay_idx + 1
            for base_idx, task in enumerate(dataset):
                base_id = task["instance_id"]
                logical_id = f"{base_id}__replay{replay_num:02d}"
                logical_position = replay_idx * base_size + base_idx
                nonce = f"replay{replay_num:02d}-idx{logical_position:04d}-{base_id}"

                tasks.append({
                    "instance_id": logical_id,
                    "base_instance_id": base_id,
                    "problem_statement": task["problem_statement"],
                    "repo": task.get("repo", ""),
                    "replay_index": replay_num,
                    "logical_index": logical_position,
                    "nonce": nonce,
                    "chain_length": sample_chain_length(rng, args.chain_min, args.chain_max),
                    "job_timeout_sec": 0,
                    "baseline_latency": None,
                })
        return tasks

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

        # HALO: Pre-register the job before the first LLM call (Option A).
        # When --halo-enabled is on, server strict mode rejects every
        # chat.completions whose halo_job_id wasn't previously registered.
        # Failure here is a misconfiguration → abort the whole run
        # (workloads/halo_helpers.py raises HaloRegisterError).
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
        # HALO: second instance only differs by halo_job_done=True. Used
        # by invoke_with_tracking for the chain's last call so the
        # server marks the Halo job COMPLETE on that request's finish.
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
        initial_state = create_chain_state(
            job_id=job_id,
            problem_statement=task["problem_statement"],
            chain_length=task["chain_length"],
            nonce=task["nonce"],
            metrics_tracker=context.metrics_tracker,
            agent_logger=context.agent_logger,
            console_write=context.console_write,
            llm=llm,
            halo_done_llm=halo_done_llm,
            log_level=context.log_level,
            job_timeout_sec=job_timeout_sec,
            job_start_time=job_submit_time,
        )
        initial_state["server_terminated_event"] = context.server_terminated_event

        result = agent.invoke(initial_state)
        job_end_time = time.time()

        job_completed = result.get("job_completed", False)
        call_index = result.get("call_index", 0)
        error_msg = result.get("error_msg", "")

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
        )

    def task_log_info(self, task: dict) -> TaskLogInfo:
        return TaskLogInfo(
            task_id=task["instance_id"],
            problem_statement=task.get("problem_statement", ""),
            repo=task.get("repo", ""),
        )

    def metadata(self, args, workload_config: dict) -> dict:
        return {
            "name": self.name,
            "dataset": workload_config.get("dataset", DEFAULT_DATASET),
            "split": workload_config.get("split", DEFAULT_SPLIT),
            "system_prompt_tokens": count_tokens(SYSTEM_PROMPT),
            "chain_range": [args.chain_min, args.chain_max],
        }

    def reproducibility_config(self, args, workload_config: dict) -> dict:
        return {
            "client_seed": args.seed,
            "llm_request_seed": args.seed,
            "temperature": 0.0,
            "top_p": 1.0,
            "chain_length_seed": args.seed,
            "task_order": "SWE-bench dataset order with deterministic replay IDs",
            "dataset": workload_config.get("dataset", DEFAULT_DATASET),
            "split": workload_config.get("split", DEFAULT_SPLIT),
            "duration_sweeps": (
                "Each rate/lambda condition creates a fresh task pool with the same "
                "seed so workload order and chain lengths do not depend on previous "
                "conditions in the same sweep."
            ),
            "server_requirement": (
                "For bitwise-identical LLM responses, start SGLang with deterministic "
                "settings such as --random-seed matching this seed and keep model, "
                "parallelism, and scheduler settings fixed."
            ),
        }
