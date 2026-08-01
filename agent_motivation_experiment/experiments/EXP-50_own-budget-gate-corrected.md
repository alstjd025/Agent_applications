# EXP-50 — candidate C again, on the corrected length profile and with the binding term now known

Written before the run, 2026-08-02 08:10 KST.

## 1. Why this is being run a second time

EXP-46 measured candidate C as **the largest static gain on record** — offered
48.7 → 72.4 at 60 req/s, all three classes up, token goodput 22,334 — and
**rejected it on the hour** for 6,583 preemptions against a threshold of 500.

Two things have changed since, and both bear on that rejection.

**The profile it was measured on was wrong.** EXP-48 found the deep research
class's output-length distribution stated a mean of 282 for a class producing
985, which over-predicted the release term of the KV projection by a factor of
3.4 on the class holding roughly 77% of resident KV. C's mechanism is to let deep
research route where the instance minimum had been refusing it, so **C is the
candidate whose behaviour depends most on that class's model being right**, and
the 6,583 was produced with it wrong.

**The binding constraint has now been measured rather than inferred.** The
counter added for EXP-49 counts which of the four conditions in `feasible`
refused each placement. Over the hour:

| | gate | memory | incumbents | unpredictable |
|---|---|---|---|---|
| `fluidserve` | **90.8%** | 7.2% | 2.0% | 0.0% |
| `fskv` (H2) | 82.3% | 7.8% | 9.9% | 0.0% |

**The pace gate is what refuses placements, and candidate C is the only proposal
that changes it.** EXP-49 rejected H2, which changes the projection; the
projection enters `feasible` through the memory condition, which accounts for
7.2%.

## 2. What is changed

One line in `evaluate`, behind `--fluidserve-own-budget-gate`:

```go
gate := math.Min(f.gateAllowance, req.nominalMs)
if p.cfg.ownBudgetGate {
    gate = req.nominalMs        // the arriving request's own class budget
}
c.gateAfter = gate * fsAllowanceUtilisation
```

`f.gateAllowance` is the minimum nominal budget over every request live on the
instance, so within seconds of a run starting it is chat's 50 ms everywhere —
chat is 76.9% of arrivals. A deep research request with a 100 ms budget is
therefore tested against 45.0 ms and fails on every instance, while the fleet it
cannot enter is running at 55.6 ms, inside its own budget. The incumbents are
still protected on the next line by `f.tightestAllowance`, which uses what each
of them has left rather than what its class was promised.

**Candidate A is off in both arms**, unlike EXP-46 where both arms carried it.
This makes the difference from the `fluidserve` arm one change.

## 3. Design

| | |
|---|---|
| arms | `fluidserve` and `fsc` (`--fluidserve-own-budget-gate=true`), A off and `FS_CLASS_HARM=false` in both |
| part 1 | static m1, 45 and 60 req/s, 2 repeats |
| part 2 | `full` hour, 1 repeat per arm |
| binary | `scheduler-exp49-H2` md5 `c3c6a3518092ea80eb7bb4dc1725ae5b`, one for both arms — it carries C behind its flag, H2 behind a flag that stays off, and the refusal counter |
| profile | the corrected `classes[]` from EXP-48 |

The baseline arm is re-run rather than read from EXP-48 or EXP-49, even though
those two agree to within 0.3 points on the hour (70.2 and 69.9 offered, 2,776
and 2,724 preemptions). It costs one condition per rate and it is what makes the
difference attributable.

## 4. What is expected

**Static, at 60 req/s.** EXP-46 measured C at 72.4 against its baseline's 48.7 on
the old profile. The corrected profile has already moved the baseline from 36.6
to 60.2, so most of that headroom is spent: **the expectation is a gain of a few
points, not twenty**, and a result near the baseline would not be surprising.
What would be surprising is a loss.

**On the hour, the preemption count is the whole question.** Baselines on the
corrected profile: 2,776 (EXP-48) and 2,724 (EXP-49), both on one or two engines
reaching 100% KV. C's mechanism sends more deep research to engines that were
refusing it, which is what produced 6,583 on the old profile.

## 5. Judgement rules, written before any result

1. **Static offered at 60 req/s at least 4.2 points above the `fluidserve` arm of
   the same session**, 4.2 being EXP-38's largest measured repeat spread on this
   workload. Below that, C no longer buys anything now that the profile is
   correct, and the honest conclusion is that EXP-46's +23.7 was mostly the
   profile defect and not the gate.
2. **Preemptions on the hour no worse than the baseline's 2,724–2,776 plus its
   own spread.** The two baseline runs differ by 52, so anything above about
   3,000 is a real rise and C is rejected again — this time on a correct input,
   which settles it.
3. **Rejection rising while offered does not rise is a failure** (§33.3),
   reported with token goodput beside both denominators.
4. **The refusal counter is read whatever the score does.** If C works, the gate
   share should fall from 90.8% and something else should become binding; if the
   gate share does not move, C did not do what it is supposed to do and the run
   is void rather than negative.
5. **Deep research is watched separately.** C exists to let that class route. If
   its offered attainment does not rise, the mechanism did not fire.

## 6. Result

(to be filled in)
