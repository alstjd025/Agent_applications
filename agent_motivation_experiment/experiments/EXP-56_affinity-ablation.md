# EXP-56 — turning the class preference off: is a fully mixed fleet worse?

## 1. Why

The system's headline claim has two halves and only one is measured.

**A static partition is worse** — established. PolyServe carries a 48x imbalance
between its busiest and least busy engine, runs chat at 96.4 and 101.0 ms per
token against a 50 ms budget on two engines while a third sits at batch 42, and
returns 3,192 goodput tokens per second against FluidServe's 17,967 (EXP-54).

**A fully mixed fleet is worse** — **not established.** Llumnix SLO mixes evenly
(each class 25–27% concentrated, against 25% for a perfectly even spread) and
scores 39.0 against 69.6, but it differs from FluidServe in several other ways at
once: no per-request conversion of the budget into a per-token allowance, a
different admission rule, no holding at the gateway.

`--fluidserve-enable-affinity=false` isolates exactly one difference. It removes
both places a class preference acts — the ordering of the feasible set by
`classShare`, and the class term in the damage estimate — and the ordering falls
back to free space, which is a load-balancing rule. Everything else is identical:
same binary, same budgets, same capacity model, same admission ladder.

EXP-54 could not answer this because **both of its repeats separated.** The
concentration of deep research on one engine reached 99–100% in the early windows
of both, so there was no unseparated run to compare against.

## 2. What the answer decides

Section 58.3 of `fluidserve-implementation.md` attributes the score to the
separation:

> chat만 남은 세 엔진에서 chat의 토큰당 시간이 40~42 ms로 예산 안이다.
> **분리가 chat 페이스를 예산 안에 두는 기전이다.**

Section 3 of `fluidserve-how-it-works.md` §6 gives a different mechanism, from
the gate arithmetic:

```
chat이 있는 엔진:  게이트 = min(50, 61.9, 100) × 0.9 = 45.0 ms
chat이 없는 엔진:  게이트 = min(61.9, 100)      × 0.9 = 55.7 ms
```

and the fleet at saturation delivers 55.6 ms. Under that reading the separation
is not about a homogeneous batch being faster; it is about **keeping a place
where loose-budget work is admissible at all.**

The two readings predict different things when the preference is removed, and
the experiment is designed to tell them apart.

## 3. Hypotheses and judgement rules — written before the run

### H1. A fully mixed fleet is worse

**Prediction**: at 45 req/s the ablated arm loses **5 points or more** of offered
attainment against the paired baseline in the same session.

**Refutation**: a loss under 2 points, or a gain. That result would say the class
preference is worth nothing on this workload, and §6 of the how-it-works document
would have to be rewritten — the separation would be an artefact rather than a
mechanism.

The threshold is 5 because the within-session repeat spread at 45 req/s is up to
4.2 points on this workload, and a difference has to clear that to be read.

### H2. The mechanism is the gate, not batch homogeneity

**Prediction if the gate reading is right**: with the preference off,
`gate_allowance_ms` reads 50.0 on all four engines essentially all the time (the
"chat-free engine" fraction goes to near zero), and the route share falls. Chat's
per-token time need not move much.

**Prediction if the homogeneity reading is right**: the gate readings are similar
to the baseline and **chat's per-token time rises** while the route share does not
collapse.

**Neither**: the score falls without either signature. That would mean the
mechanism is not understood, and the next step is instrumentation rather than
another arm.

These are pre-registered as mutually exclusive so the result cannot be read both
ways afterwards.

### H3. The separation is what makes the 45 req/s outcome bistable

**Prediction**: with the preference off there is **no bimodality** — the three
repeats at 45 req/s land close together, because the positive feedback that
amplifies an arbitrary initial imbalance is gone. The mean may be low.

**Refutation**: the ablated arm is itself bimodal. That would locate the
bistability somewhere other than the class preference.

## 4. Design

| | |
|---|---|
| arms | `fluidserve` (baseline, v0.1.1 defaults) and `fsnoaff` (`--fluidserve-enable-affinity=false`), **interleaved in one session** so the pair shares any session offset |
| static | 45 and 55 req/s, **3 repeats**, 8 minutes per condition, engine cold-restarted per condition. 12 conditions |
| dynamic | `dyn60_short_m123`, one hour, **1 repeat per arm**. 2 conditions |
| fixed | migration off, stock vLLM FIFO engine, no engine-side admission, KV admission threshold 0 |
| binary | `f88e9430f21dc74a410a7091ebdea218` — the same one EXP-53 and EXP-54 ran |
| ablations pinned | `FS_CLASS_HARM=false` in both arms. Not because false is the intended setting but because every FluidServe condition it is being read against ran that way; turning it on is EXP-43 |

Rates 45 and 55 are chosen because the gate arithmetic binds in a narrow band:
below it the 45 ms gate passes anyway and above it nothing passes, so an effect
that exists only in the band would be invisible at 20 or 80. EXP-54's minute-level
series puts that band around a delivered pace of 45–56 ms, which is 45–60 req/s.

Estimated 5 hours.

## 5. What is being measured

Beyond the standard set:

- `scheduler_fluidserve_gate_allowance_ms` per instance — the fraction of samples
  above 50.5 ms is "an engine with no chat resident", which is the state variable
  section 52 found predicts the 45 req/s outcome 24 times out of 24.
- decision mix (route / pend / shed / force).
- per-window class concentration **and which engine holds it** — pooled over a
  run this statistic cannot see a moving assignment (§59.3).
- chat's per-token time per engine.
- preemption per engine and fleet total.

## 6. Result

*(to be filled in)*

## 7. Note — a false alarm that was raised and withdrawn before this ran

While preparing this experiment the scheduler's start-up line was found to report
`classharm=false` in all 50 FluidServe conditions of EXP-51 through EXP-54, while
the source default, the deployed binary's own `--help`, the deploy script's own
output and the running pod's arguments all said `true`. It was recorded as an
unexplained contradiction that might mean the start-up line could not be trusted.

**It was none of those.** The driver pins `FS_CLASS_HARM=false` on every FluidServe
arm, and the comment above that line says why: every FluidServe condition from
2026-07-28 through EXP-46 ran with it false because the flag stuck in the
deployment spec, so the baseline is pinned to false to stay comparable with
everything recorded against it. Passing the variable explicitly in both
directions was tested here and the start-up line follows it exactly.

**What was missed is written in CLAUDE.md already**: the measurement path is not
only the binary and the workload but everything a running sweep calls, and the
driver's arm definitions are part of it. Four sources were checked and the one
that sets the value was not.
