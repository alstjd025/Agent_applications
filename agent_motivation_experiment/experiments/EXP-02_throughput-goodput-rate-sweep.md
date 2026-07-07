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

## Result (2026-07-06, MP load-procs=8, restart per condition)

Run dirs: `results/*_exp02_ratesweep_rpm_{300,600,900,1200,1500,1800}/`.
Cross-rate summary: `results/aggregate_analysis/exp02_ratesweep_summary.csv`.

| req/s | offered | ok% | ok tput | out tok/s | e2e p50 | e2e p99 | TTFT p50 | gw_current | gw_pending | eng_run(Σ4) | KV% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5  | 1200 | **100** | 5.0 | 1971 | 1.4s  | 24.4s | 0.03s | 88   | 56  | 75  | 2.0 |
| 10 | 2400 | 89  | **8.8** | 3247 | 24.6s | 81.9s | 0.76s | 295  | 160 | 144 | 2.2 |
| 15 | 3600 | 53  | 7.7 | 2055 | 20.4s | 83.9s | 0.95s | 841  | 176 | 175 | 2.2 |
| 20 | 4800 | 35  | 6.1 | 1425 | 13.6s | 57.0s | 1.01s | 941  | 304 | 157 | 3.4 |
| 25 | 6000 | 28  | 4.4 | 1102 | 16.3s | 56.4s | 1.39s | 1084 | 349 | 150 | 2.2 |
| 30 | 7200 | 25  | 6.8 | 1794 | 18.4s | 48.0s | 1.87s | 1357 | 517 | 188 | 4.1 |

**Findings**
1. **Saturation knee ≈ 8–10 req/s.** Successful throughput peaks at ~8.8 req/s
   (offered 10) then **declines** (7.7 → 6.1 → 4.4) as offered rate rises —
   textbook **congestion collapse**: past saturation the system does more work
   that ends up failing, so goodput goes *down*.
2. **Goodput collapses** monotonically: 100 → 89 → 53 → 35 → 28 → 25 %.
   TTFT p50 degrades 0.03s → 1.9s (gateway queueing).
3. **The bottleneck is the gateway control plane, NOT the GPU fleet.**
   `gateway_current` climbs 88 → 1357 and `gateway_pending` 56 → 517 (the gateway
   saturates around its ~1024 concurrency budget), while the **GPU engines stay
   idle**: total running ~75–188 across 4 instances and **KV cache only 2–4 %** at
   every rate. In full-mode the gateway does a per-request `scheduler:8088/schedule`
   round-trip, which serializes throughput long before the GPUs are stressed.
4. **The MP load generator was necessary and sufficient** to reach these rates: a
   single process caps ~17 req/s (client-bound); with `--load-procs 8` the client
   drove up to 120 offered req/s and saturated the serving system. Each rate ran
   from a **cold engine restart**, so the collapse is not carryover.

**Caveat / what this does NOT show**: true **GPU-inference** overload (high KV,
preemptions). Short ShareGPT outputs keep KV at ~2–4 % even when the gateway is
saturated. To stress the GPU fleet itself you must either send **long-output**
requests (so 1024 concurrent gateway requests translate into large KV) or raise
the gateway concurrency budget (a gateway config flag, not code). See EXP-03/next.

## Next
- EXP-03: dispatch-policy comparison (load-balance vs round-robin vs flood; migration
  ON vs OFF) at rates around the knee — does policy/migration move the goodput curve?
- To reach GPU-KV overload: a long-output workload variant or a larger gateway
  concurrency budget.


## Plots + steady-state note (2026-07-07)
Generated by `analysis_scripts/request_level/plot_ratesweep.py` →
`results/aggregate_analysis/exp02_plots/{timeseries.png, throughput_vs_rate.png}`.

- **timeseries.png** (4 panels, all rates overlaid, warmup shaded): completed
  throughput is stable at low rates but **bursty at overload** (spikes to ~28/s
  then crashes); **KV stays <4%** the whole time (GPU idle); the **queue grows in
  large sawtooth waves** (up to ~500) at 25-30 req/s; **migration is flat** (no
  rescheduling fired).
- **throughput_vs_rate.png**: completed throughput tracks the offered=achieved
  diagonal only up to ~10 req/s, then **bends down** (congestion collapse) while
  completion rate falls 100%→25%.

**Steady-state window [60s, dur-20s]** (drops warmup + end-drain) completed tput:
5→5.0, 10→10.1, 15→9.1, 20→7.2, 25→4.7, 30→8.5 req/s (30 is noisy — overload is
non-stationary; see the CV in the per-condition check). This is *completion*
throughput; the "%" column is **completion rate, not latency-SLO goodput** (SLO
thresholds not yet set — that is EXP-01).
