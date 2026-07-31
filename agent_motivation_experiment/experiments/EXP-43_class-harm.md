# EXP-43 — candidate E: turn the class term in the damage estimate back on

Written before the run, 2026-08-01 04:20 KST. Second change, one at a time.
Background: `fluidserve-implementation.md` §38 (why this was never measured) and
§37.4 (why it is the mechanism the saturated regime is missing).

## 1. What is being changed and why

No code changes. One flag: `--fluidserve-class-harm`, whose compiled default is
`true` and which has been `false` in every deployment since 2026-07-28 10:28.

The term is the last line of `harmToIncumbents` (fluidserve.go:1400):

```go
if p.cfg.enableAffinity && p.cfg.classHarm && len(f.live) > 0 {
    harm += (1 - classShare(f, tier)) * fsHarmCap
}
```

When no instance can take a request at its promised pace, the candidates are
ordered by `harm` ascending. Without this term `harm` is the sum over the
requests already on the instance that can still meet their budgets, and requests
already past their budgets contribute nothing — so an instance whose own class
has begun to miss reads as costing nothing and keeps attracting work from other
classes. The term charges a candidate for the share of it that belongs to other
classes, which is what makes an instance already full of one class the place
that class keeps going even when nothing is feasible.

**Why this matters here.** §37.1 measured how often the policy is in that
regime: the ROUTE path, which is where the class-affinity ordering in
`sortCandidates` operates, takes 99.7% of decisions at 15 req/s, 11.2% at 45 and
**1.5% at 60**. §37.4 concluded that at saturation there is no path that can
build class separation, and §38 established why: the one path designed for that
regime was switched off. This turns it on and measures it.

## 2. What is expected, on what grounds, and what would refute it

**Expected.** At 60 req/s the class mix per engine becomes less uniform, chat
concentrates on fewer engines, and chat's inter-token latency on those engines
falls below its 50 ms budget the way it did on `azcode` in EXP-41 §5.1, where a
chat-only engine ran at 26.6–32.6 ms against a fleet average of 44.3.

**Grounds, and the reason for doubt.** EXP-41 showed the separation is worth a
great deal when it forms — chat went from 45.3 to 85.4 offered — but it also
showed what it costs: the engine that collected deep research reached 100% KV
occupancy and preempted 5,513 times, because `capMem` is computed from a
prompt-sharing ratio measured on the current residents and no allowance is made
for the tokens they have yet to generate, which are never shared (§36.3). This
term makes the concentration stronger. **It could therefore improve the score
and produce the failure mode that candidate D exists to fix, at the same time.**

**Refutation conditions, in the order they will be checked.**

1. **The mechanism did not fire.** If the Herfindahl index of the class mix per
   engine at 60 req/s does not rise, the term did not change any ordering and the
   result is a null, not a negative. Check the start-up line for
   `classharm=true` first. A null here would itself be informative: it would mean
   the incumbent sum dominates the ordering and the class term is too small to
   decide anything, which is a statement about `fsHarmCap`.
2. **It fired and did not help.** Offered attainment at 60 req/s must rise above
   the within-session baseline. The baseline for this experiment is candidate A
   with class-harm off, not the shipped policy.
3. **It cost something at the lower rates.** Offered attainment at 15, 30 and 45
   req/s must not fall by more than the within-session repeat spread, or by 4.2
   points where only one repeat is available. At 15 and 30 the ROUTE path takes
   99.9% of decisions, so this term should never be reached; a difference there
   means something other than the flag changed.
4. **It made the memory problem worse.** Preemptions were **zero in every static
   condition of EXP-38 and EXP-40**. If any condition here records a non-zero
   preemption count, that is a result in itself and is recorded whatever happens
   to the score: it would mean the concentration this term produces is already
   past what `capMem` can account for at a static rate, which EXP-41 only saw
   under a moving load.
5. **It only concentrates.** If the separation index rises and neither the score
   nor chat's inter-token latency improves, the term is producing concentration
   without the capacity gain it is supposed to buy.

## 3. Design

| | |
|---|---|
| arms | `fsah` (`--fluidserve-class-harm=true`) and `fsa` (`=false`) |
| **both arms** | `--fluidserve-force-margin=true` — candidate A is in the baseline now |
| rates | 15, 30, 45, 60 req/s |
| repeats | 2, outer loop |
| mix | m1 |
| engine | stock FIFO, four engines, migration off, engine admission off |
| binary | same as EXP-42, `scheduler-exp42-A` md5 `d24861df81d89257f5f16dbbb70bb295` |
| session | one session for all conditions |

**The baseline moves to candidate A.** EXP-42 accepted A on all five of its
pre-registered conditions, so leaving A off here would measure class-harm on top
of a policy that is no longer the one being developed, and turning both on in a
single arm would make the result unattributable. Both arms carry A; the single
difference is class-harm.

**Why 15 and 30 req/s are still run.** They cost an hour and they are the check
that the harness did not change: at those rates the changed line is never
reached, so any difference is a fault rather than an effect.

**45 req/s is bistable and this must be read with that in mind.** EXP-38's
condition routed 11.2% of decisions and scored 88.5; EXP-42's baseline routed
90.6% and scored 99.1. The same code and the same rate produce two operating
regimes, and which one a run lands in persists for the whole eight minutes. A
difference at 45 req/s between two arms of this experiment is therefore only
readable if both arms landed in the same regime, which the route share reports.

## 4. Result

(to be filled in)
