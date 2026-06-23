# ShareGPT Request-level Poisson Workload

Request-level (flat, open-loop) workload: each task is **one independent
LLM request**, submitted as a Poisson process (λ = requests/sec). No
job/chain concept. The data source is the ShareGPT multi-turn chat
dataset.

## Single-step / direct — no transcript, no baseline

Unlike `codingagent_request_level_poisson`, this workload is **direct**:
`load_dataset` downloads ShareGPT from HuggingFace and turns it into
requests that are sent straight away. There is **no record step, no
transcript file, and no per-request solo baseline**.

Goodput is evaluated against **absolute SLO thresholds** (post-hoc by the
analysis script), not `baseline × tau`, so:

- `tau` is **unused** by this workload.
- There is **no `baseline × tau` job timeout**.
- All client-side aborts are disabled (`job_timeout_sec=0`,
  `per_call_timeout=None`, `idle_timeout=None`) — every request runs to
  completion and is measured, not killed. The HTTP client timeout
  (`_HTTP_SAFETY_TIMEOUT_S`) is only a dead-connection safety net.

Files:
- `sharegpt.py` — dataset download + conversation flattening (unit-testable).
- `workload.py` — the adapter.

## Conversation flattening

- Source: HuggingFace `anon8231489123/ShareGPT_Vicuna_unfiltered`,
  file `ShareGPT_V3_unfiltered_cleaned_split.json` (override via
  `--workload-config` keys `hf_repo` / `hf_data_file`).
- Each conversation is coerced to strict `user/assistant` alternation
  starting at a user turn; the first turn that breaks alternation or is
  empty truncates the conversation. **No system prompt is injected** —
  ShareGPT is replayed as-is.
- Conversation `c` with human turns `1..K` → `K` requests. Request `k`'s
  prompt is the conversation prefix
  `[u_1, a_1, ..., u_{k-1}, a_{k-1}, u_k]`, where the `a_*` are the
  recorded ShareGPT gpt turns (deterministic context → realistic
  prefix-cache behavior).
- `request_id = sg-{conv:05d}-t{turn:02d}`; the runner task `instance_id`
  appends a `__rNN` replay-cycle suffix.

## `--workload-config` keys

| Key | Default | Meaning |
|---|---|---|
| `hf_repo` | `anon8231489123/ShareGPT_Vicuna_unfiltered` | Hub dataset repo |
| `hf_data_file` | `ShareGPT_V3_unfiltered_cleaned_split.json` | JSON dump in the repo |
| `hf_revision` | `192ab2185289094fc556ec8ce5ce1e8e587154ca` | pinned dataset commit (reproducible download) |
| `num_conversations` | 200 | conversations sampled |
| `min_human_turns` | 1 | drop conversations with fewer human turns |
| `max_human_turns` | none | truncate each conversation to this many human turns |
| `sample_seed` | `--seed` | deterministic conversation sampling |

## Invariants

- `metrics.csv` rows have `agent == "request"` (set via
  `invoke_with_tracking(agent_label="request")`).
- `build_baseline_tasks` still works (a concurrency-1 pass whose
  `metrics.csv` is useful for picking absolute SLO thresholds) but writes
  **no transcript**.
- It is in `run_experiment.py`'s `NO_BASELINE_DIR_WORKLOADS`, so runs do
  not need `--baseline-dir`.

## Analysis

Per-run raw-load stats:

```bash
python analysis_scripts/request_level/parse_request_summary.py results/<run>
```

Writes `analysis/request_summary.csv` (one-row aggregate: counts,
throughput, latency/TTFT/TBT percentiles) and `analysis/request_metrics.csv`
(cleaned per-request rows).

Cross-λ sweep table:

```bash
python analysis_scripts/request_level/summarize_lambda_sweep.py \
  --run-dirs results/*sharegpt_sweep_lambda_* \
  --output-dir results/aggregate_analysis/sharegpt_lambda_summary \
  --print-markdown --plot-png
```

Cross-λ server decode throughput vs time (one line per λ):

```bash
python analysis_scripts/parse_server_logs.py results/<run>          # per run, once
python analysis_scripts/request_level/plot_server_throughput_overlay.py \
  --run-dirs results/*sharegpt_sweep_lambda_* \
  --output-path results/aggregate_analysis/sharegpt_lambda_summary/server_throughput_overlay.png \
  --smooth-window 30
```

Cross-λ TBT-p90 attainment curves (X = threshold ms, Y = % of successful
requests with `tbt_p90_ms ≤ T`):

```bash
python analysis_scripts/request_level/plot_tbt_p90_threshold.py \
  --run-dirs results/*sharegpt_sweep_lambda_* \
  --output-path results/aggregate_analysis/sharegpt_lambda_summary/tbt_p90_threshold.png \
  --csv-output  results/aggregate_analysis/sharegpt_lambda_summary/tbt_p90_threshold.csv
```

**Absolute-SLO goodput (TODO):** the parser writes raw load stats only.
Goodput against fixed e2e/TTFT/TBT thresholds is intentionally deferred
— add when thresholds are decided. Use `parse_request_metrics.py`
(sister script) only for the `codingagent_request_level_poisson`
transcript+τ flavor; it does not apply here.

## Halo

`run_job` forwards the per-request `halo_*_slo` fields into `make_llm`
when `context.halo_enabled` is True and reuses `invoke_with_tracking`, so
HTTP-400 reject detection and `metrics.csv` propagation work for free.
With absolute SLOs the `--halo-*-slo` flags should carry absolute
thresholds (the server's `slo_mode` decides interpretation).
