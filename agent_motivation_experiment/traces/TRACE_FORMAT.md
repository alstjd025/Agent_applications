# Arrival-Trace Format (canonical)

`run_experiment.py --mode trace-replay --trace-file <path>` drives request
**arrival timing** from a trace file in the canonical format below. This is the
**contract** between the `traces/` preprocessing area and the experiment runner:

- **`traces/` (preprocessing).** Public traces (BurstGPT, Azure LLM Inference,
  Mooncake, …) come in different schemas/units. Conversion/scaling/windowing
  scripts live here and emit a file in the canonical format. All trace editing
  (time-scaling, rate-normalizing, windowing, sub-sampling) happens here, **not**
  in the runner.
- **Runner (consumption).** The runner reads only the canonical format and fires
  one request/job per row at the row's `arrival_s`. It does **not** know which
  public trace the file came from.

The runner reads timing **verbatim** (no scaling). If you want a different load
intensity, produce a different canonical file from your `traces/` scripts.

---

## Schema

CSV with a header row. One row = one arrival.

| Column | Required | Type | Meaning |
|---|---|---|---|
| `arrival_s` | **yes** | float, seconds | Arrival time relative to trace start, ascending. Row order is the arrival order. |
| `request_id` | no | string | Opaque id from the source trace. Currently **ignored** by the runner (timing-only); reserved for future trace-driven content selection. |
| `input_tokens` | no | int | Source-trace context/prompt token count. **Ignored** (request content comes from the workload, not the trace). |
| `output_tokens` | no | int | Source-trace generated token count. **Ignored.** |

Any extra columns are allowed and ignored.

### Example

```csv
arrival_s,request_id,input_tokens,output_tokens
0.000,r0,2048,28
0.000,r1,1469,13
0.142,r2,1020,129
0.589,r3,4083,451
1.203,r4,256,14
```

---

## Semantics (what the runner does)

- **Timing only.** The trace decides *when* each arrival fires. *What* is sent is
  produced by the selected `--workload` via `task_pool.next_task()`, in pool
  order. So trace-replay composes with **any** workload (job-level
  `swe_bench_coding*` or request-level `*_request_level_poisson`).
- **Raw timestamps.** `arrival_s` is used as-is. The runner subtracts the minimum
  `arrival_s` so the run starts immediately (no leading idle), but otherwise does
  not rescale. Do all intensity shaping upstream in `traces/`.
- **Open-loop / catch-up.** Arrivals fire by wall clock. If the server can't keep
  up, arrivals whose time has passed are submitted back-to-back (offered load is
  not throttled) — same behavior as the Poisson/rate drivers. Traces with
  near-zero inter-arrival times (e.g. Mooncake median 0 ms) therefore produce
  instantaneous bursts. That is intended.
- **Run length.** The run ends when the last arrival has been submitted; in-flight
  jobs then drain and `server_terminated` is signaled (so run-boundary goodput
  cutoffs work normally). `--trace-duration-min <m>` optionally caps the run by
  dropping arrivals with `arrival_s > m*60`.
- **Baseline / goodput unchanged.** trace-replay only changes arrival timing.
  Job-level workloads still need `--baseline-dir`; request-level transcript
  carries its own baseline; ShareGPT uses absolute SLOs. `metrics.csv` schema is
  unchanged, so existing `analysis_scripts/` parsers apply as-is.

---

## Producing a canonical file (guidance for `traces/` scripts)

Minimum job of a converter: read a public trace, derive a per-request arrival
time in seconds, sort ascending, write `arrival_s` (plus any optional columns).

Source-specific notes:

- **Azure LLM Inference 2023/2024** (`TIMESTAMP,ContextTokens,GeneratedTokens`):
  parse `TIMESTAMP` to epoch seconds, `arrival_s = ts - ts.min()`. Map
  `ContextTokens`/`GeneratedTokens` to the optional `input_tokens`/`output_tokens`.
- **BurstGPT** (`Timestamp,...,Request tokens,Response tokens,...`): `Timestamp`
  is relative seconds; `arrival_s = Timestamp - Timestamp.min()`. Optionally
  filter by `Model` / `Log Type` first.
- **Mooncake** (`timestamp` in ms): `arrival_s = (timestamp - timestamp.min())/1000`.

Intensity shaping (all upstream, optional):
- time-scale: `arrival_s *= s` (s<1 → faster/denser, s>1 → slower).
- rate-normalize to target mean λ: `arrival_s *= (observed_lambda / target_lambda)`.
- window: keep rows in `[start_s, end_s]`, then re-zero with `-= start_s`.
