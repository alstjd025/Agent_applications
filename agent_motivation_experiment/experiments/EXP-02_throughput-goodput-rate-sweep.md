# EXP-02 — Throughput / goodput rate-sweep to fleet overload

**Status**: in progress · **Date**: 2026-07-06 · **Dispatch**: load-balance (reference)

## Why
Characterize how the fleet (4×TP2 B200, Llama-3-8B) responds as **offered request
rate** rises: where does achieved throughput saturate, and where does latency /
goodput collapse? Goal per user: push until the **whole fleet is severely
overloaded**.

## Independence rule
Every rate condition **cold-restarts the whole engine + control plane** before it
runs (`RESTART_PER_CONDITION=1`). Each rate is measured from an identical cold
state — no carryover.

## Calibration finding (before the sweep)
- A single Python load client (the in-cluster runner) tops out at **~17 req/s**
  (λ=50 offered → only 17.6 achieved; fleet KV was ~2%, running≈11 → the **client**,
  not the fleet, is the bottleneck).
- A 256-concurrent closed-loop blast reached only 22 req/s with fleet running=11,
  waiting=0 → the fleet is far from saturated; the Python client saturates first.
- **Implication**: one runner cannot overload this fleet. This sweep characterizes
  the single-runner curve + client ceiling; **EXP-02b will escalate to multiple
  parallel runner pods** to actually overload the fleet.

## Setup
- Runner: in-cluster (`k8s/runner-job.yaml`), `--engine llumnix --in-cluster`.
- Workload: `sharegpt_request_level_poisson`, **mode `rate-sweep`** (fixed
  inter-arrival, cleaner saturation than Poisson).
- **Rates (rpm)**: `300,600,1200,2400,4800,9600` = **5,10,20,40,80,160 req/s**
  (spanning below → at → far above the ~17 req/s client ceiling).
- Duration: **5 min / condition**; NUM_CONV 1000; `RESTART_PER_CONDITION=1`.
- Session: `exp02_ratesweep`.

## What to observe
- Achieved throughput (req/s, tokens/s) vs offered rate → plateau = saturation.
- Latency p50/p90/p99 (TTFT, e2e) → knee/explosion point.
- Server-side (`server_metrics/`): engine running/waiting/KV per instance, gateway
  pending/current → is it the **fleet** saturating (waiting-queue grows, KV high)
  or the **client** (fleet idle while achieved plateaus)?
- Any migration (`rescheduling_total`, migration_events.log) — expected ~none
  (dispatch balances; KV stays low).

## Result
_(filled in after the run)_

Run dirs: `results/*exp02_ratesweep_rate_*/`.
Analysis: `parse_request_summary.py` (per run) + `summarize_lambda_sweep.py`
(cross-rate) + `parse_llumnix_metrics.py` (server-side).
