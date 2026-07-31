# EXP-41 — the two control planes on an hour of moving load

Started 2026-07-31 12:12 KST. `full` finished 14:31, `azcode` running, expected
16:45 KST. Written while running.

## 1. Why, and why two traces

Every comparison up to EXP-40 held the arrival rate fixed for eight minutes at a
time. A rate that never moves cannot show an adaptation cost, and the whole
claim about admission is about what happens when the load changes. Two traces,
deliberately different in what each can answer:

| variant | trace | what it can show | what it cannot |
|---|---|---|---|
| `full` | `dyn60_short_m123` | four peaks, so the load crosses the knee four times **and comes back down four times** — recovery is visible | rate and mix move together, so a segment difference cannot be attributed to one of them |
| `azcode` | `azcode_w60_m1` | a **real** 60-minute window replayed verbatim, no compression or rescaling, **mix fixed at m1** — so only the rate moves | one crossing, no return; recovery is not visible |

Their shapes are in `results/aggregate_analysis/trace_shapes.png`. `full` is
Azure conv+code minute counts, four days compressed 96x and rank-mapped to
25–75 req/s, mix stepping m1→m2→m3→m1 every 15 minutes, mean 50.1 req/s.
`azcode` is minute 5,087 of the Azure code 2024 trace (2024-05-13 12:47 UTC),
125,864 arrivals, mean 35.0 req/s, per-minute rate rising from the mid-20s to
the high 40s.

Both arms are **stock FIFO**; the engine scheduler is not a variable here.

## 2. What had to be built

`mix_dyn60_short_m123.json` declares the agent class as `tbt_ms 25`, which the
Llumnix SLO filter reads literally and which made it reject 98% of that class in
EXP-28. The static sweep solves this with `m1f`; the dynamic configs had no
equivalent. Added, for both traces:

- `workload_configs/mix_dyn60_short_m123_slofair.json`
- `workload_configs/mix_azcode_w60_m1.json` (verbatim trace, mix fixed m1)
- `workload_configs/mix_azcode_w60_m1_slofair.json`

Agent class restated as (2500, 52), exactly as `mix_short_m1_slofair.json` does.
Scoring is unaffected: the analysis judges that class end to end whatever the
config says. `run_exp30_dynamic.sh` now picks the fair config **by arm**, the
same way `run_exp27_mixsweep.sh` picks m1 or m1f, and gained the `azcode`
variant, an `ARMS` override and an overridable session prefix.

**The pre-flight check earned its keep again.** The first launch aborted with
`engine has SCHED_EXTRA_ARGS='--scheduling-policy priority --scheduler-cls
deadline_sched.DeadlineScheduler'` — EXP-40 had left the engine on QoServe. Run
as it was, three conditions labelled FIFO would have been QoServe. The engine
was reset and **the reset is now part of the driver** rather than something
remembered between runs.

## 3. Result — `full`, one run each, both healthy

| | offered | admitted | rej% | goodput | total tok/s | chat / dr / agent |
|---|---|---|---|---|---|---|
| Llumnix SLO | 38.5 | 66.7 | 42.1 | 11,384 | 16,143 | 26.7 / **100.0** / 58.5 |
| **FluidServe** | **59.7** | **84.3** | **29.1** | **15,643** | **18,383** | **59.3** / 83.7 / 35.1 |

**+21.2 points offered, 37% more token goodput, 13 points less rejected, and 14%
more total throughput.**

By 15-minute segment — the advantage holds through every mix:

| segment | mix | rate | SLO | FluidServe | diff |
|---|---|---|---|---|---|
| 0–15 | m1 | 43.3 | 43.8 | **71.2** | **+27.4** |
| 15–30 | m2 | 47.2 | 67.5 | **89.9** | +22.3 |
| 30–45 | m3 | 49.1 | 29.3 | **52.8** | +23.5 |
| 45–60 | m1 | 58.3 | 19.2 | **32.1** | +12.9 |

By rate band, pooling the hour by each request's 30-second arrival rate:

| band | n | SLO | FluidServe | diff |
|---|---|---|---|---|
| 0–40 req/s | 35,439 | 75.2 | 93.2 | +18.0 |
| **40–50** | 30,694 | 49.4 | **75.9** | **+26.5** |
| **50–60** | 46,518 | 31.7 | **60.1** | **+28.3** |
| 60+ | 66,767 | 19.0 | 34.4 | +15.4 |

**The same shape the static sweep found** — largest in the 40–60 band, smaller
at both ends — reproduced under load that never sits still. EXP-38 measured
+35.6 at a static 45 req/s and +8.3 at 60; here the 40–50 band reads +26.5 and
60+ reads +15.4.

### The timeline, and one thing only a timeline shows

`results/aggregate_analysis/exp41/exp41_full_timeline.png`, six panels on one
time axis, 90-second sliding windows anchored on arrival.

**Llumnix SLO stops recovering.** Through the first half both policies fall at
each peak and climb back between them. After minute 45 FluidServe returns to
about 60% in the troughs and **the SLO arm stays pinned near 20%** even when the
offered rate drops. Its KV occupancy (panel F) climbs over the same stretch
while its decode batch does not, which is the candidate explanation and is not
yet established — separating it needs the transition-only scoring in
`exp30_dynamic.py`.

**It rejects more and scores less.** At the peaks the SLO arm rejects up to 80%
against FluidServe's 50%, and its attainment is lower anyway.

**Deep research sits at 100% for both policies for the whole hour** (panel E)
while chat and the agent class collapse at every peak. That is §34.6's finding
— the class using half its budget is making the batch heavy for the class on its
budget line — holding for sixty minutes of moving load rather than for one
eight-minute condition.

## 4. Caveats

- **One run per condition.** The differences, 12.9 to 28.3 points, are well
  above the repeat spread measured elsewhere (1.2–4.2 points, up to 11.4 at the
  knee), but they are not repeats.
- **`full` moves rate and mix together.** Both arms see the identical trace, so
  the arm difference is valid; "why is the 15–30 segment better" is not
  answerable from this run. The flat-rate ablation
  (`dyn09_short_mixcycle_flat`, rate fixed at 40, mix cycling only) is what
  separates them, and `azcode` avoids the confound entirely by holding the mix.
- Recovery is described from the timeline, not yet measured. The per-segment and
  transition-only breakdown is the measurement.

## 5. Result — `azcode`

(running, expected 16:45 KST 2026-07-31)
