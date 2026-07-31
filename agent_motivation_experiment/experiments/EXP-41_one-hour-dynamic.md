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

(running, expected ~17:00 KST 2026-07-31)

## 6. Two failure modes an eight-minute condition cannot produce

Both were found by asking a question about the timeline rather than the whole-run
mean, and neither appears anywhere in EXP-38 or EXP-40. They are at opposite
ends of the hour and have nothing to do with each other.

### 6.1 Minutes 50–56: the hold turns into a loss, and the arm difference goes negative

This is the only stretch of the hour in which Llumnix SLO scores higher than
FluidServe.

| min | rate (req/s) | SLO | FluidServe | diff |
|---|---|---|---|---|
| 50 | 68.3 | 17.1 | 15.0 | −2.2 |
| 51 | 72.7 | 17.3 | 14.8 | −2.5 |
| 52 | 73.0 | 17.3 | 13.7 | **−3.6** |
| 53 | 74.5 | 17.5 | 14.7 | −3.1 |
| 54 | 73.0 | 17.3 | 14.2 | −3.1 |
| 55 | 69.6 | 17.1 | 15.2 | −1.9 |
| 56 | 58.8 | 17.9 | 17.6 | −0.4 |
| 57 | 56.6 | 18.3 | **31.5** | +13.2 |

**It is the only sustained extreme overload in the trace.** Counting stretches
whose 30-second rate stays above 65 req/s: minute 6.5 for 1.0 min, minute 8.0
for 2.5 min, minute 21.0 for 1.0 min, minute 36.0 for 1.0 min, and **minute 50.0
for 6.0 min** (peak 77.1 req/s at 53.5). Measured SLO capacity is about 44
req/s, so this is 1.6–1.7x capacity held for six minutes. The difference returns
to +13.2 in the first minute the rate drops below 60.

Decomposed over minutes 50–57 by what each class contributes to the offered
score (attainment x that class's share of arrivals):

| class | share | SLO att | FS att | SLO contrib | FS contrib | diff |
|---|---|---|---|---|---|---|
| chat | 76.9% | 0.0 | 1.5 | 0.0 | 1.2 | +1.2 |
| deepresearch | 15.4% | 100.0 | 80.2 | 15.4 | 12.3 | **−3.1** |
| swe | 7.7% | 25.7 | 18.5 | 2.0 | 1.4 | −0.6 |

**The Llumnix SLO arm's 17.4 is arithmetic, not adaptation.** It rejects
**100%** of chat (22,604 of 22,604) and **0%** of deep research, so it collects
that class's 15.4 points in full and nothing else. Because the offered
denominator counts a rejection as a violation, refusing a class costs exactly
what that class could have contributed — which at 1.6x capacity is nothing that
either policy can collect anyway.

**Deep research is not failing on latency for FluidServe; it is being shed by
FluidServe.** Of 4,517 arrivals it sheds 829 (18.4%), and of the 3,688 it admits
only 64 (1.7%) miss a rule. The whole 3.1-point loss is its own admission
decision. The mechanism is the hold:

| | dr TTFT p50 | dr TTFT p90 | budget |
|---|---|---|---|
| Llumnix SLO | 0.45 s | 0.97 s | 10 s |
| FluidServe | **8.91 s** | **9.59 s** | 10 s |

`canWait` sets the hold deadline at `ttftSlo − prefillMs − placementDelayBound −
recheckMs`, so FluidServe spends about nine of deep research's ten TTFT seconds
waiting for an instance that can meet the deadline. At 1.6x capacity sustained
for six minutes that instance never appears, so 18.4% reach the deadline with no
feasible target and are shed, and the rest are dispatched with under a second of
margin left. **This is the same hold that produces the advantage at 40–60 req/s**
and that section 35 credited for the result being insensitive to the engine
scheduler; past the capacity point it converts a class that would have passed on
immediate dispatch into an 18.4% loss.

Where the capacity went: FluidServe admits 3,965 chat requests (17.5% of chat
arrivals) of which **91.2% fail, 3,546 of them on TBT** at an ITL p50 of 55.1 ms
against a 50 ms budget. Those requests produce output that scores nothing.

| | total output | goodput | share that scored |
|---|---|---|---|
| Llumnix SLO | 12,115 tok/s | **11,132** | 92% |
| FluidServe | **13,747** | 9,035 | 66% |

The engine is not the constraint and FluidServe is not producing a slower fleet:
its decode batch is 797.6 against 772.5, its mean KV is 68.7% against 78.1%, and
its ITL is 55 ms against 60 ms. It is spending a faster fleet on a class that
cannot meet 50 ms at this load.

**Leading explanation, not yet measured on this run.** Section 33 measured
`gate_allowance_ms` at exactly 50.0 on all four instances at all times, because
an instance's gate is the minimum `nominalMs` over the requests live on it and
chat's budget is the smallest. The pace in this window is 55 ms. A deep research
request therefore cannot ROUTE — it is judged against chat's 50 ms gate rather
than its own 100 ms budget — falls through to PEND, and ends in a shed. Scraping
`gate_allowance_ms` over minutes 50–56 is what would confirm it.

Two fixes follow without needing class priorities, which remain future work:
judge the gate against **the arriving request's own budget** rather than the
instance minimum, so a class with large slack is not held behind a class with
none; and **shorten the hold deadline when recent holds have been ending in
sheds**, so the nine seconds are not spent before the request is discarded
anyway.

### 6.2 Minutes 6–16: FluidServe overloads one engine and pays 1,471 recomputes

`exp41_engine_view.py`, two figures in `results/aggregate_analysis/exp41/`.
EXP-38 and EXP-40 recorded **zero preemptions in every condition**; an hour of
moving load is where they appear, and they appear on one engine of one arm.

| arm | port | batch | KV mean | KV max | queue | preemptions | prefix hit |
|---|---|---|---|---|---|---|---|
| SLO | 8000 | 196 | 54.2 | 92 | 0.4 | 0 | 78.8% |
| SLO | 8001 | 191 | 54.7 | 95 | 0.4 | 0 | 78.4% |
| SLO | 8002 | 203 | 54.9 | 92 | 0.4 | 0 | 78.3% |
| SLO | 8003 | 199 | 54.8 | 95 | 0.4 | 0 | 78.7% |
| FS | 8000 | 183 | 41.4 | 82 | 0.4 | 0 | 79.8% |
| FS | 8001 | 177 | 46.1 | 83 | 0.4 | 0 | 79.3% |
| FS | 8002 | 231 | 35.7 | 87 | 0.6 | 0 | 81.1% |
| **FS** | **8003** | **220** | **55.9** | **100** | **2.6** | **1,471** | 80.0% |

Busiest-over-least-busy decode batch, averaged per window: **1.25x for the SLO
arm, 2.70x for FluidServe.** The preemptions are confined to minutes 7–15 (149 /
158 / 174 / 177 / 113 / 417 / — / 254 / 29), during which engine 8003's batch
reaches 540, its KV touches 100%, and its engine queue reaches 27 while the
other three stay at 0. vLLM V1 preempts by recompute, so each one discards a
completed prefill and pays for it again, and no request-level metric attributes
that cost to anything.

Minutes 7–15 are the first peak, and they are where FluidServe's per-minute
advantage is at its most erratic in the whole first half: +11.6 at minute 7 and
**+1.4 at minute 9**, against +34 to +61 in the surrounding minutes. The two
observations are consistent in time; attributing one to the other needs the
repeat, since a single run cannot separate them from the peak itself.

**The dispatch side** (`analysis/request_engine.csv`, every admitted request
matched to the engine the scheduler dispatched it to, 100% of 103,801 and
127,266 respectively):

| arm | port | requests | attainment (admitted) | chat% | dr% | swe% | chat ITL |
|---|---|---|---|---|---|---|---|
| SLO | 8000 | 25,867 | 65.7 | 63.6 | 20.6 | 15.7 | 47.6 |
| SLO | 8001 | 25,406 | 69.2 | 63.2 | 20.7 | 16.1 | 44.3 |
| SLO | 8002 | 26,241 | 64.2 | 65.8 | 19.7 | 14.5 | 47.7 |
| SLO | 8003 | 26,287 | 67.7 | 64.3 | 20.5 | 15.2 | 44.7 |
| FS | 8000 | 31,149 | 84.1 | 79.9 | 12.3 | 7.8 | 43.6 |
| FS | 8001 | 26,908 | 77.8 | 70.9 | 17.3 | 11.9 | 45.0 |
| FS | 8002 | **45,118** | **91.7** | **93.0** | 5.1 | 1.9 | **41.2** |
| FS | 8003 | 24,091 | 77.7 | 60.2 | **32.9** | 6.9 | 44.8 |

The Llumnix SLO arm gives all four engines the same request count to within 3%
and the same class mix to within 3 points, which is what a policy with no class
concept produces. FluidServe does separate the classes without being asked to —
engine 8002 receives 93.0% chat and 45,118 requests, engine 8003 receives 32.9%
deep research — and its per-engine attainment tracks that separation, with the
chat-concentrated engine scoring highest (91.7%) at the lowest chat ITL (41.2 ms).

**The prefix cache hit rate does not read out this separation here**, contrary to
what it did for PolyServe (94.5% on the chat engine against 66.7% on the agent
engines). All eight engines report 78–81%. The separation is partial rather than
a partition, and the three classes do not have disjoint enough prompt prefixes on
this workload for a 93/60 split in class share to move the counter. **The
engine-reported hit rate is therefore not a reliable read-out of routing on this
workload; the dispatch log is.**

### 6.3 What is measured and what is not

- Both subsections are one run per arm. In 6.1 the −2.5-point crossing is inside
  the repeat spread seen elsewhere (up to 4.2 points, 11.4 at the knee), so **the
  crossing as a number is not established**; the mechanism is — an 18.4% versus
  0% shed rate on deep research, and 91.2% failure among admitted chat, are far
  outside any measured spread.
- In 6.2 the preemption count, the imbalance, and the class shares are direct
  counter readings, not estimates. The link from the preemption burst to the
  narrowed advantage at minute 9 is temporal co-occurrence only.
- The engine attainment column uses the **admitted** denominator and is not
  comparable with the offered numbers in sections 3 and 6.1. A rejected request
  is never dispatched, so it has no engine to be attributed to.
