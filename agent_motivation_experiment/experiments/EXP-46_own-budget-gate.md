# EXP-46 — candidate C: judge the arriving request against its own budget

Written before the run, 2026-08-01 16:45 KST, while EXP-45 is still on its last
arm. Background: `fluidserve-implementation.md` §44, which decomposed the one
window where FluidServe loses to Llumnix SLO and priced the fix.

## 1. The change

One expression, in `evaluate` (fluidserve.go:1260):

```go
gate := math.Min(f.gateAllowance, req.nominalMs)   // before
gate := req.nominalMs                              // with --fluidserve-own-budget-gate
c.gateAfter = gate * fsAllowanceUtilisation
```

**It deletes a redundancy rather than loosening a constraint.** Two things are
protected in the feasibility test:

- the **arriving** request must be able to run at the pace its class was
  promised — that is `req.nominalMs`;
- the **incumbents** must not be pushed past what they can still meet — that is
  `c.meanAfter <= f.tightestAllowance` on the next line, which uses each
  incumbent's **remaining** budget.

`f.gateAllowance` protects the incumbents a second time using their **nominal**
budgets, and being a minimum over every class on the instance it becomes chat's
50 ms within seconds of a run starting, because chat is 76.9% of arrivals.

**No constant is added and one term is removed.** `f.gateAllowance` is left in
place for `capKv`, which is a per-instance reporting and tie-break quantity, not
part of the feasibility test — changing that too would be a second change.

## 2. Why, with the measurement it comes from

Over minutes 50–56 of EXP-41's `full` trace, from the scheduler's own gauges:
`gate_allowance_ms` read **exactly 50.0 on all four instances** while the
delivered step was **55.6 ms** and `tightest_allowance_ms` was 68.3–72.7. So a
deep research request, whose budget is 100 ms, was tested against
`55.6 ≤ 45.0`, failed everywhere, was held for 8.85 s of its 10 s budget and
then shed — while the fleet it could not enter was running at 55.6 ms, well
inside its budget. Llumnix SLO dispatches the same requests at 0.44 s and scores
100 on that class.

§44 decomposed the resulting gap in that window. Against Llumnix SLO's 17.4,
FluidServe with candidate A reads 16.7, and **the whole difference is deep
research**: chat 2.5 vs 0.0 is worth +1.9 to us, swe ties at 2.0, deep research
83.0 vs 100.0 costs −2.6. Of the 3,810 deep research requests admitted there,
**61 miss a rule**; the loss is the 15.7% shed.

Deep research at 100 in that window takes the arm from 16.7 to **19.3**, past
Llumnix SLO's 17.4.

## 3. Design

Two parts, the fast regression gate first.

| | part 1 | part 2 |
|---|---|---|
| what | static sweep | the `full` hour |
| rates | 45 and 60 req/s | the trace |
| arms | `fsa` (A only) and `fsac` (A + C) | same two |
| repeats | 2 | 1 |
| duration | ~55 min | ~2.3 h |

Both arms carry candidate A, accepted in EXP-42, so the single difference is C.
`--fluidserve-class-harm` stays pinned false on both, as in every FluidServe
condition since 2026-07-28. One binary for all conditions,
`scheduler-exp46-C` md5 `550036c2907365654f61825e71985460`, with C behind a flag
that is off by default.

**Part 1 first** because the risk in §4 is a regression at static rates, and an
hour is cheaper than finding it after three.

## 4. What is expected and what would refute it

**Expected.** Deep research routes instead of holding: at saturation
`55.6 ≤ 90` and `55.6 ≤ 68.3` both pass. Chat still fails `55.6 ≤ 45.0` and does
not route, which is the correct outcome — that instance cannot serve chat at
50 ms. swe is unaffected: its `nominalMs` is about 57.7 and `57.7 × 0.9 = 51.9`
is below 55.6, so it does not route either way.

**The risk, stated in advance.** `tightestAllowance` excludes incumbents already
past their budgets, so the remaining protection weakens exactly when a class has
begun to miss. If deep research routes onto instances full of failing chat, it
can make that chat worse. Chat is 76.9% of requests, so a small loss there
outweighs the whole deep research gain.

**Second risk.** Deep research routing again means it collects on one engine
again, and that is the state that produced 1,471 preemptions in EXP-41 and 1,852
in EXP-44's baseline — a failure whose cause has been open since §40. EXP-44
showed candidate A holds that count at zero because the fleet carries 8% less
work, so C starts from a lighter fleet than EXP-41 did, but this is a reason to
watch the counter, not a reason to expect nothing.

**Acceptance conditions, in the order they will be checked.**

1. **The mechanism fired.** Deep research's shed rate at 60 req/s and in minutes
   50–56 must fall. If it does not, check the start-up line for
   `ownbudgetgate=true` first; without that the run is void, not negative.
2. **No regression at static rates.** `fsac` at 45 and 60 req/s must not fall
   more than 4.2 points below `fsa` — EXP-42 read 99.0 and 49.3 for that arm.
   **This is checked before part 2 runs and a failure stops the experiment.**
3. **Chat does not get worse.** In minutes 50–56, chat must not fall below the
   2.5 candidate A reads there, and at static 60 req/s chat must not fall below
   41.7. This is the risk in §4 and it is the condition most likely to fail.
4. **The window is won.** Minutes 50–56 offered must exceed **17.4**, which is
   what Llumnix SLO scores there, from candidate A's 16.7. Deep research in that
   window must reach 95 or better from 83.0.
5. **Preemptions are recorded whatever the score does.** Zero on candidate A in
   EXP-44. Any non-zero count is a result; above 500 the change is rejected
   regardless of the score, because the cause of that failure mode is open.

A result that passes 1 and 4 but fails 3 is the informative failure: it would
mean the instance-minimum gate was doing real work for chat and the redundancy
argument in §1 is wrong.

## 5. Result

(to be filled in)
