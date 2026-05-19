# Workload Notes

Workloads are adapters loaded by `run_experiment.py` with `--workload <name>`. They hide dataset loading, task generation, per-job execution, and workload-specific reproducibility metadata behind a common interface.

When working inside `workloads/`, read this file first and then read the nearest workload-specific `AGENTS.md` if one exists.

## Adapter Contract

Each workload module should expose a `Workload` class with:

```text
name
load_dataset(args, workload_config)
build_baseline_tasks(dataset, replay_count, rng, args, workload_config)
create_task_pool(dataset, baseline_latencies, rng, args, workload_config)
run_job(task, context)
task_log_info(task)
metadata(args, workload_config)
reproducibility_config(args, workload_config)
```

The protocol is defined in `workloads/base.py`.

## Runtime Context

`run_job(task, context)` receives `RunContext` with:

| Field | Purpose |
|---|---|
| `server_base_url` | OpenAI-compatible SGLang endpoint |
| `seed` | Experiment seed |
| `log_level` | `quiet`, `info`, or `debug` |
| `metrics_tracker` | Write per-call metrics |
| `agent_logger` | Write per-job prompt/response logs |
| `console_write` | Thread-safe status output |
| `server_terminated_event` | Set when runner ends a duration run |
| `job_start_time` | Used for job-level timeout checks |
| `parallel_calls_path` | Optional raw CSV path for workloads that record dependency/round structure |
| `halo_enabled` | True when client runs with `--halo-enabled`. Gates the per-request SLO `extra_body` wiring. Defaults to False |
| `halo_ttft_slo` / `halo_tbt_slo` / `halo_e2e_slo` | Per-request Halo SLO values attached to every chat.completions body. Each defaults to `--tau` |
| `transcript_record_path` | When set, every LLM call's full prompt + solo timings are appended to this JSONL (literal-replay capture). Defaults to None |

## Task Dictionaries

Keep task dict fields stable once analysis depends on them. The default SWE-bench workload uses:

```text
instance_id
base_instance_id
problem_statement
repo
replay_index
logical_index
nonce
chain_length
job_timeout_sec
baseline_latency
```

## Metrics

Workloads are responsible for recording each LLM call through `MetricsTracker.record_chain_call()`. The common runner records `job_summary` after `run_job()` returns.

Important invariants:

- `task_id` should include replay suffixes when replayed.
- `base_task_id` should strip replay suffixes during analysis.
- `call_index` is 1-based.
- `total_calls_expected` is the sampled chain length.
- `is_job_timeout` and `is_server_terminated` should propagate from call failure to job result.
- SGLang admission-control rejects must be recorded explicitly as `is_rejected=True`, `rejection_reason=<reason>`, `success=False`, and `is_error=True` in call rows.
- Job results that stop because of a rejected call must propagate `is_rejected` and `rejection_reason` to the `job_summary` row.
- Prefer the shared invocation path in `swe_bench_coding.agent.invoke_with_tracking()` for OpenAI-compatible LLM calls so rejection handling stays consistent across workloads.
- HALO rejects (HTTP 400, `halo_reason` in `HALO_KV_CAP` / `HALO_VSS_PREDICTED` / `HALO_ADMISSION_PREDICTED` / …) are recognized by `_detect_admission_rejection()` in `agent.py`. `is_rejected=True` + `rejection_reason=<HALO_*>` should be propagated to call rows and `job_summary`.

## Default Workload

`swe_bench_coding/` implements a synthetic SWE-bench Lite coding agent. It runs stage prompts over a fixed sequence:

```text
Understand -> Locate -> Plan -> Implement -> Verify -> Debug -> ...
```

Prompt files live in `swe_bench_coding/prompts/`. Keep prompt edits intentional because they change token counts, latency, cache reuse, and reproducibility.

## Tool-Delay Workload

`swe_bench_coding_tool_delay/` reuses the default SWE-bench coding agent and task pool, but supplies deterministic delays before calls whose prompt includes a simulated `Tool result:` block.

Key invariants:

- Delay applies only to `call_index > 1` and only when the current stage has a simulated tool result.
- Delay is sampled from a scaled beta distribution in `[0.1s, 10.0s]` with mean `3.0s`.
- The seed is `sha256(tool_delay_beta_v1|base_instance_id|replay_index|boundary_call_index)`.
- The same task/replay/boundary must get the same delay across all runs.
- `JobResult.transition_time` should be the total slept tool-delay time for the job.
- Analysis writes transition-adjusted timing to separate CSVs instead of changing existing application job CSVs.

## Parallel Tool-Delay Workload

`swe_bench_coding_parallel_tool_delay/` keeps the same task pool, chain length, and stage sequence as the default SWE-bench workload, then groups consecutive `Locate` calls into one `execution_round`.

Key invariants:

- Use `execution_round`, not `wave`, for dependency-barrier terminology.
- Total call count must remain identical to the base workload for the same task/replay/seed.
- Call-level latency, token, TBT, and goodput source of truth remains `metrics.csv`.
- Dependency and round structure is recorded in `parallel_calls.csv`.
- `parse_application_metrics.py` turns that raw file into `application_parallel_calls.csv` and `application_parallel_rounds.csv`.
- If multiple tool-result calls are in one round, the round sleeps for `max(call_tool_delays)`, modeling parallel tool work.
- `Plan` and later singleton calls depend on all calls from the previous execution round.

## Request-level Poisson Workload

`codingagent_request_level_poisson/` is **not** a job/chain workload.
Each task is one independent LLM request; the runner submits them as a
Poisson process (λ = requests/sec). It replays recorded agent calls
**verbatim** ("literal replay") from a transcript JSONL.

Key invariants:

- Input is a transcript file produced by a concurrency-1
  `swe_bench_coding --mode baseline --record-transcript <path>` run.
  Each line carries the full prompt of one agent call plus that call's
  solo TTFT/TBT/e2e — the per-request goodput baseline.
- No job concept, no `--baseline-dir` — the baseline lives in the
  transcript. `run_job` issues exactly one `chat.completions` request
  via the shared `invoke_with_tracking` (with `agent_label="request"`,
  `chain_length=1`).
- `metrics.csv` rows for this workload have `agent == "request"`.
  `analysis_scripts/parse_request_metrics.py` turns them into
  per-request e2e/TTFT/TBT goodput.
- Prompt bytes are replayed unchanged across pool cycles, so cross-
  request prefix-cache behavior is realistic.

## Halo-compatible Workloads

Project Halo (request-level since 2026-05-19) admits/rejects **each
request independently** — there is no job pre-registration. When a
client runs `run_experiment.py --halo-enabled`, a workload's `run_job`
only has to:

1. **Attach the per-request SLO fields.** Build the LLM with
   `make_llm(..., halo_ttft_slo=, halo_tbt_slo=, halo_e2e_slo=)` when
   `context.halo_enabled` is True; `make_llm` wires them into
   `extra_body` so LangChain forwards them on every `chat.completions`.
   Pass `None` for all three when Halo is off.

2. **Reuse `invoke_with_tracking`** for the call. `_detect_admission_rejection`
   recognizes the HTTP-400 Halo reject (`halo_reason` in `HALO_KV_CAP` /
   `HALO_VSS_PREDICTED` / `HALO_ADMISSION_PREDICTED` / …) and the legacy
   429 admission_control path. It also scans `str(exception)` because
   the openai SDK wraps 4xx into a `BadRequestError` with empty
   `.response_metadata`. Rejections propagate to `metrics.csv` as
   `is_rejected=True, rejection_reason=HALO_*` for free.

There is no `halo_done_llm`, no `register_halo_program`, and no
`halo_job_id` anymore (all removed in the request-level refactor).

See [halo_helpers.py](halo_helpers.py) for the startup probe and
`ms_dev/halo_dev/halo_api_reference.md` in the sglang repo for the full
server-side API spec.

## Adding A Workload

1. Create `workloads/<name>/workload.py`.
2. Export `Workload` from `workloads/<name>/__init__.py`.
3. Ensure `workloads/__init__.py` can load the name.
4. Add metadata and reproducibility fields so `run_config.json` explains the run.
5. Verify with a tiny baseline or single run before launching sweeps.
6. **Halo support**: inside `run_job(task, context)`, when
   `context.halo_enabled` is True, pass `context.halo_ttft_slo` /
   `halo_tbt_slo` / `halo_e2e_slo` into `make_llm(...)`. Reuse
   `invoke_with_tracking` so HTTP-400 reject detection works. See
   "Halo-compatible Workloads" above.
