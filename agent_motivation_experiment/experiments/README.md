# Experiment Log — Llumnix migration / throughput / goodput

This folder is the **running record** of every experiment driven against the
Llumnix serving stack (NXC13 k3s). Each experiment has its own `EXP-NN_*.md`
capturing **why** it was run, **what we expected to see**, the **exact settings**,
**how** it was executed, and the **observed result**. Write the file *before/while*
running, not after, so the intent is on record.

Result artifacts (metrics.csv, server_metrics/, analysis CSVs) live under
`../results/<run>/`; these MD files hold the human-readable rationale + findings
and point at the relevant run dirs.

## ⚠️ Ground rule: every condition restarts the whole engine

**Each experimental condition MUST cold-restart the entire serving stack before
it runs** (engine pod `neutral-0` deleted → LWS recreates; scheduler + gateway
rollout-restarted). This guarantees every condition starts from an identical cold
state — empty KV/prefix cache, zeroed Prometheus counters — so results are
**completely independent**. Concretely: in a λ-sweep, **every λ gets its own fresh
restart** (not just once at the start).

- This is enforced by `--restart-per-condition` (env `RESTART_PER_CONDITION=1` in
  `k8s/runner-job.yaml`). **All experiments here run with it ON.** ~150s restart
  cost per condition is accepted as the price of independence.
- Never reuse a warm stack across conditions for a recorded experiment.

## Standard setup (unless an experiment says otherwise)

- **Runner**: in-cluster pod (`k8s/runner-job.yaml`), reaches gateway/scheduler/
  engines via k8s DNS (no port-forward bottleneck). Launch:
  `kubectl -n llumnix delete job bench-runner --ignore-not-found && kubectl apply -f k8s/runner-job.yaml`.
- **Serving**: neutral, 4 instances × TP=2 (Llama-3-8B-Instruct), migration ON
  (`backend:"migration"`, `--disable-custom-all-reduce`).
- **Default dispatch policy**: `load-balance` (scheduler). Some experiments change
  it to `flood`/`round-robin` via a scheduler config edit (documented per-exp).
- **Rescheduling (migration)**: `neutral_load,neutral_failover`, colocated loop
  500ms, neutral-load-threshold 0.003, load-balance-threshold 0.1 (deployment
  values, deliberately low so migration is observable).

## Metrics & analysis

- Per-request: `results/<run>/metrics.csv` (TTFT, TBT, e2e latency, tokens, success).
- Server-side time series: `results/<run>/server_metrics/*.jsonl` →
  `analysis_scripts/parse_llumnix_metrics.py` → `analysis/llumnix_server_metrics{,_summary}.csv`.
- Application throughput/latency: `analysis_scripts/request_level/parse_request_summary.py`.
- Cross-λ: `analysis_scripts/request_level/summarize_lambda_sweep.py`.
- Migration ground truth: `results/<run>/server_metrics/migration_events.log` +
  `scheduler_rescheduling_total`. (Note: the counter is only exposed once a
  rescheduling has actually occurred.)

## Index

| # | File | Status | One-line |
|---|---|---|---|
| 00 | [EXP-00_migration-mechanism.md](EXP-00_migration-mechanism.md) | in progress | Prove the migration pipeline fires and (via flood) actually transfers KV — no code change |
| 01 | EXP-01_slo-calibration.md | planned | Solo-latency baseline to set absolute SLO thresholds for goodput |
| 02 | EXP-02_throughput-goodput-lambda-sweep.md | planned | λ-sweep on the reference config: where does goodput collapse? |
| 03 | EXP-03_dispatch-policy-comparison.md | planned | load-balance vs round-robin vs flood; migration ON vs OFF — migration's effect on goodput |
| 04 | EXP-04_request-gen-comparison.md | planned | Poisson vs fixed-rate vs Azure trace-replay at matched mean load |
| 05 | EXP-05_workload-variety.md | planned | sharegpt vs codingagent transcript vs swe_bench job-chains |
