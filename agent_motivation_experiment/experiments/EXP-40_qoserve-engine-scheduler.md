# EXP-40 — does FluidServe's advantage survive a deadline-aware engine?

Started 2026-07-30 15:17 UTC. Written while running.

## 1. Why

EXP-38 put FluidServe 35.6 points ahead of the Llumnix SLO arm on the offered
denominator at 45 req/s, with every engine on stock FIFO ordering. The objection
that follows is that the advantage is an artifact of a weak engine: give the
engine a deadline-aware scheduler and the control plane's contribution might
disappear. This crosses the two control planes with two engine schedulers in one
session so the objection is answered by measurement rather than argument.

There is also a mechanism reason. implementation.md §34 traced the failure at
60 req/s to a 4.5 ms rise in per-token time of which 72% is the residual between
the measurement and the decode-only law, booked as prefill duty. The one Niyama
unit that acts on that term is dynamic prefill chunk sizing. The port was made
faithful to the original first; that work is in
`ms_dev/notes/qoserve-niyama-fidelity.md`.

## 2. Arms and settings

Four arms, one session, arm inside and repeat outside, two repeats.

| arm | control plane | engine |
|---|---|---|
| `slofifo` | Llumnix SLO filter, m1f | stock vLLM ordering |
| `sloqoserve` | Llumnix SLO filter, m1f | `deadline_sched.DeadlineScheduler` |
| `fluidservefifo` | FluidServe, m1 | stock vLLM ordering |
| `fluidserveqoserve` | FluidServe, m1 | `deadline_sched.DeadlineScheduler` |

Rates 1800 / 2700 / 3600 rpm = 30 / 45 / 60 req/s, 8 min per condition. 15 req/s
is omitted: every arm reads 100 there and the condition costs 14 minutes to
confirm it again.

Only `SCHED_EXTRA_ARGS` on the LeaderWorkerSet differs between an arm and its
pair. The client sends `--priority-mode deadline` in every condition, which
stock vLLM ignores because `priority` does nothing without
`--scheduling-policy priority`, so the two FIFO arms are the EXP-38
configuration exactly.

The FIFO arms are re-measured rather than taken from EXP-38 because
cross-session movement on this workload reaches 4.6 points and the differences
at issue are of that order. **Measured, the control reproduces far better than
that**: `slofifo` against EXP-38's mean of two repeats reads 99.9 vs 99.9 at
30 req/s, 52.8 vs 52.9 at 45, and 26.1 vs 26.9 at 60, all on the offered
denominator.

Driver `/home/nxclab/tools/exp40_qoserve.sh`. Scheduler binary unchanged from
EXP-38 (`scheduler-exp07-v29fluxcount`, md5 `cbbefe34`).

## 3. Runs to exclude

**The first launch aborted after its first arm and the relaunch reused the same
session prefix**, so `exp40r1_slofifo` matches three directories from the aborted
session and three from the live one. They are distinguishable only by timestamp.

Excluded — aborted session, 2026-07-30 15:17 UTC and earlier:

```
results/260730_0724_exp40r1_slofifo_m1f_rpm_1800
results/260730_0743_exp40r1_slofifo_m1f_rpm_2700
results/260730_0756_exp40r1_slofifo_m1f_rpm_3600
```

The live session starts at `260730_0818`. These three are kept rather than
deleted: they are healthy measurements of the FIFO arm and they are what
established that the control reproduces EXP-38 to within 0.8 points.

**Why the first launch aborted.** `run_exp27_mixsweep.sh`'s `check_stack`
refuses to run when `SCHED_EXTRA_ARGS` is non-empty, a rule added so that an
engine-side scheduler could never silently confound a control-plane comparison.
EXP-40 makes the engine scheduler the variable under test. The check was not
removed — it now takes `EXPECT_SCHED_EXTRA_ARGS` and compares against it, so it
still catches an engine left in the previous arm's state, which is the failure
it exists to prevent, and an unset variable keeps the old behaviour exactly.

A second defect in the driver: `verify_engine` ended with a grep for the
per-request `[deadline] req=` lines, so a condition whose lines had already
scrolled out of the log reported a verification failure. Those lines are
advisory; the start-up line is the verdict.

## 4. Judgement rules, fixed before the run

1. **Primary.** FluidServe minus the SLO arm, per request, offered denominator,
   under QoServe against the same difference under FIFO. **The claim is refuted
   if the gap narrows by more than the repeat spread**, which would mean the
   advantage was partly the engine's weakness.
2. **Mechanism.** Chat's mean inter-token latency at 60 req/s, 48.7 ms against a
   50 ms budget in EXP-38, where §34 measures roughly 10 points of chat
   attainment per millisecond. **If it moves less than 1 ms the run says nothing
   about dynamic chunking**, only about the two control planes.
3. **The cost side.** Chat's TTFT-only failure rate, 7.0% at 60 req/s. Dynamic
   chunking buys time between tokens by deferring prefill, which is paid for in
   time to first token. If TTFT failures rise while TBT failures fall, unit 4 has
   moved failures between the two halves of a conjunctive rule and bought
   nothing.
4. **Capacity guard.** Total output tokens per second, 20,244 at 60 req/s under
   FluidServe. A fall means attainment was bought with throughput.
5. **Inertness check.** Engine queue depth and preemption counts. Both were zero
   for both control planes in EXP-38. If they stay zero, units 2, 3 and 5 did
   nothing and the result belongs to unit 4 alone.

### The prior is not that QoServe helps FluidServe more

Dynamic chunking spends time to first token to buy time between tokens, and
FluidServe has already spent that budget at the gateway — it holds each chat
request until its computed deadline, so at 60 req/s its chat TTFT median is
4.52 s against the SLO arm's 1.71 s, with a 5 s budget. On that reasoning the
SLO arm has more to gain. Against it, FluidServe's `canWait` subtracts a
*measured* placement-delay bound, so it shortens its own hold when the engine
starts deferring prefill, and a fixed 5 s window cannot. Which dominates is the
measurement.

## 5. Two asymmetries to state when reporting

**Only FluidServe adapts to a changed engine.** The Llumnix SLO filter predicts
from the static profile table and has no online correction; FluidServe has two
measured feedback paths, `capacity_correction` on the step time and
`PlacementDelayBound` on the queueing delay. The profile was fitted against FIFO
and is stale for both arms under QoServe. It is deliberately not re-fitted: the
profile is a deployment artifact an operator fits once, re-fitting would change
two things at once, and the online correction is part of what FluidServe is. A
follow-up with a re-fitted table is how "our correction adapts" would be
separated from "our policy is better".

**FluidServe's capacity model is chunk-invariant, so unit 4 does not corrupt it.**
The model reads the chunk from the engine's static `MaxNumBatchedTokens`, 8192,
while unit 4 caps the real step at 2552. The chunk enters as
`prefillSteps × (t_pre(chunk) − c0)` = `(pending / chunk) × (t_pre(chunk) − c0)`,
and the measured prefill profile is linear above about 1024 tokens —
`(t_pre − c0)/chunk` is 0.0589 at 2048, 0.0592 at 2552 and 0.0614 at 8192 — so
the chunk cancels to within 4%. The prefill term is about 6.65 ms of the 50 ms
prediction at 60 req/s, so the mismatch costs about **0.24 ms, half a percent**.
The same cancellation holds in `carriedPrefillTokens` and `maxKvForAllowance`.

## 6. Pre-flight verification, 15:25 UTC

All four checks the design named as void-if-absent passed on the first QoServe
condition.

- **The scheduler loaded**, with the fidelity pass live:
  `[deadline] relegation=True dynamic_chunk=True grid=True grid_top=2552
  hybrid_ms_per_tok=8.000 c_pctx=1.226e-03 tbt_s=0.050 max_batched=8192`.
  All four engines report `scheduler_cls: deadline_sched.DeadlineScheduler`
  and `scheduling_policy: priority` in their non-default args.
- **Priority reaches the engine and unpacks correctly.**
  `[deadline] req=... slo_ms=5000 tbt_ms=50 ttft_deadline=+5.000s` — the chat
  class's own budget, not the 30,000 ms fallback that would have made every
  request look identical and the arm measure something else.
- **Health.** Every completed condition: four engine metric files all
  advancing, offered rate within 0.2% of target.
- **The FIFO control reproduces EXP-38** to within 0.8 points offered.

At 30 req/s the two engines are indistinguishable, which is expected: chat's
inter-token latency is 29.9 ms under FIFO against 29.6 under QoServe, both far
inside the 50 ms budget, and offered attainment is 99.9 for both. Unit 4 only
binds when a running decode's slack is short, and at 30 req/s there is slack
everywhere. The comparison lives at 45 and 60.

## 7. Result

(pending)
