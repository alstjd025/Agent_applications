# Tool-Delay SWE-Bench Workload Notes

This workload wraps `workloads/swe_bench_coding` and keeps the same task generation, prompts, stage sequence, LLM settings, and metrics path. Its only behavioral difference is deterministic application-side sleep before calls that include a simulated `Tool result:` block.

## Delay Rules

- Delay applies only when `call_index > 1` and the current call prompt will include `Tool result:`.
- No delay is applied before calls without tool results, such as plan/implement-only boundaries.
- Delay values are sampled from a scaled beta distribution in `[0.1, 10.0]` seconds with mean `3.0` seconds.
- Delay is deterministic by task/replay/boundary using `sha256(tool_delay_beta_v1|base_instance_id|replay_index|boundary_call_index)`.
- Do not use Python's process-global RNG for these intervals; reproducibility must not depend on run order or arrival schedule.

## Metrics

- Per-call `transition_time` in `metrics.csv` is still measured by `MetricsTracker` as the wall-clock gap from the previous call end to this call start.
- Per-job `transition_time` is `JobResult.transition_time`, which this workload sets to the total simulated tool-delay sleep time.
- `latency` remains wall-clock job duration including tool delay.
- Transition-adjusted job timing belongs in separate analysis CSVs:

```text
analysis/application_job_transition_adjusted.csv
analysis/application_job_transition_adjusted_summary.csv
```

## Editing Guidance

- Keep this workload thin. Reuse `swe_bench_coding.agent` instead of copying prompts or graph logic.
- If the delay distribution changes, update `TOOL_DELAY_VERSION` so old and new experiments can be distinguished.
- Update README, root `AGENTS.md`, and `workloads/AGENTS.md` when changing delay semantics.
