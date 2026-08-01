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

## 5. Part 1 — the static gate, passed by a wide margin

Finished 2026-08-01 19:41 KST. Eight conditions, two repeats of each arm at each
rate, all healthy: 4/4 engines, rate within 0.2% of target, no flags.

| rate | arm | offered | chat | dr | swe | rej% | goodput | route% | dr shed | chat ITL | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 45 | fsa | 95.9 | 96.3 | 100.0 | 84.5 | 3.2 | 21,476 | 65.2 | 0.0 | 37.0 | 0 |
| 45 | **fsac** | **99.6** | **99.9** | 97.8 | **100.0** | **0.0** | 21,840 | **99.9** | 0.0 | **32.2** | 0 |
| 60 | fsa | 48.7 | 40.8 | 92.9 | 43.9 | 47.1 | 16,895 | 6.3 | 6.3 | 45.2 | 0 |
| 60 | **fsac** | **72.4** | **66.5** | **99.9** | **82.2** | **24.8** | **22,334** | **22.7** | **0.0** | **40.6** | **112** |

**+23.7 points at 60 req/s**, larger than candidate A's +13.1, with all three
classes rising: chat 40.8 → 66.5, deep research 92.9 → 99.9, swe 43.9 → 82.2.
Token goodput 16,895 → 22,334, which is **higher than the 21,840 this arm
produces at 45 req/s** and higher than any figure recorded in this project at
any rate.

### Two things came out opposite to what was written down

**Chat improved rather than degraded.** §4 named condition 3 as the one most
likely to fail, on the grounds that `tightestAllowance` excludes incumbents
already past their budgets so deep research could land on instances full of
failing chat. Chat went 40.8 → 66.5 instead.

The mechanism is the rejection rate: 47.1% → 24.8%. Once deep research can route,
the requests that were being held at the gateway for 8.85 s each leave
immediately — `route` rises 6.3% → 22.7% and deep research's shed rate goes
6.3% → 0.0%. **Holding deep research was costing chat as well**: chat's median
inter-token latency falls 45.2 → 40.6 ms. That is not what the redundancy
argument in §1 predicted; it predicted no effect on chat.

**The bimodality at 45 req/s disappeared.** `fsa` split into its two operating
states again — route 93.2% scoring 99.1 and route 37.2% scoring 92.7 — while
`fsac` read 99.9% routing in both repeats, for 99.3 and 99.9. Candidate A did
not remove that split (EXP-42 §4 saw both repeats in the routing regime, but
EXP-46's `fsa` shows it can still fall out). Two repeats, so this is not
established.

### The one thing that got worse

**Preemptions at a static rate, for the first time since EXP-38.** `fsac` at 60
req/s recorded 20 and 92 across its two repeats; every static condition of
EXP-38, EXP-40, EXP-42 and EXP-43 recorded zero, and so does `fsa` here.

112 is far below the 500 at which §4 says to reject, and far below the
1,471–1,852 that EXP-41, EXP-44 and EXP-45 recorded on the dynamic trace. But it
is the signal §4 said to watch for: **deep research is collecting on an engine
again**, which is the mechanism whose cause has been open since §40. Recorded as
a result, not as noise.

### Against the pre-registered conditions

1. **PASS** — deep research shed 6.3% → 0.0%, routing 6.3% → 22.7%.
2. **PASS** — no regression at either static rate; both improved.
3. **PASS**, and opposite to the stated expectation — chat 40.8 → 66.5.
4. Part 2.
5. **Recorded**: 0 → 112. Below the rejection threshold, above zero.

Part 2 proceeds.
