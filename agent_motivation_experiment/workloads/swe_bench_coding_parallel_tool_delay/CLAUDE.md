# Parallel Tool-Delay SWE-Bench Workload Notes

This workload models modern agent execution where independent information-gathering calls can run concurrently.

## Semantics

- The workload name is `swe_bench_coding_parallel_tool_delay`.
- It keeps the same SWE-bench task pool, chain length sampling, stage sequence, prompts, and total call count as `swe_bench_coding`.
- Consecutive `Locate` stages are grouped into one `execution_round` and submitted concurrently.
- Singleton stages such as `Understand`, `Plan`, `Implement`, `Verify`, and `Debug` run as one-call rounds.
- A round depends on all calls in the previous round. `Plan` therefore sees all parallel `Locate` outputs.
- Use `execution_round`, not `wave`, in code, CSVs, and docs.

## Tool Delay

- Delay values are the same deterministic per-boundary values used by `swe_bench_coding_tool_delay`.
- If a round contains multiple tool-result calls, the round sleeps for the maximum call delay in that round.
- `JobResult.transition_time` is the total actual round-delay sleep time.

## Metrics

- `metrics.csv` remains the source of truth for per-call latency, tokens, TBT, success, and goodput.
- `parallel_calls.csv` is a raw sidecar containing per-call dependency and execution-round metadata.
- `analysis/application_parallel_calls.csv` joins `parallel_calls.csv` with call metrics.
- `analysis/application_parallel_rounds.csv` summarizes each round's wall time, summed call latency, and overlap savings.

## Editing Guidance

- Keep total call count stable when changing parallel grouping.
- Prefer adding fields to the parallel sidecar over changing `metrics.csv`.
- Keep `parallel_group_id` stable as `<task_id>__roundNN`.
- Update root/workload README and AGENTS/CLAUDE notes if round semantics change.
