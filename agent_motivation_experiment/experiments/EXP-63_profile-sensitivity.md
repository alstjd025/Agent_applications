# EXP-63 — how much does the policy lose when its length profile is wrong?

Written before the run. 2026-08-07 01:30 KST.

## 1. Why

The policy takes each class's output-length distribution as an input. It uses it
in three places: the expected remaining tokens `E[L−j | L>j]` that the
feasibility test multiplies by the pace, the completion probability over a
horizon, and the KV growth projection.

**That input is the strongest objection to the work.** A reviewer will say the
profile was built from the same workload it is evaluated on, so the system is
being given something a deployment would not have. The answer cannot be "we do
not look at the future", because we do look at a distribution estimated from
past traffic; the answer has to be **"an estimate this far off costs this
much"**, and today we have exactly one point on that curve: when the
deepresearch profile was 3.4× too small — because the workload generator grew
that class's output from 282 to 985 tokens and the profile was not regenerated —
static 60 req/s scored 36.6 where the corrected profile scores 56.8, a loss of
**20.2 points** (`fluidserve-implementation.md` §49).

**Everything between ±0% and −71% is unmeasured.** A 3.4× error is a mistake, not
a normal estimation error; the interesting question is what a routine ±30% costs.

**This is not a train/test split question** (asked 2026-08-03 and answered in
`fluidserve-v0.1.1.md` §6.6). The profile was fitted on 1.78M chat, 410K
deepresearch and 209K swe requests, so a hold-out half has an almost identical
survival function and would report "no difference", which only says the sample
is large. The risk is not sample size; it is **the distribution moving while the
profile stays**, which is what happened on 2026-07-29.

## 2. Design

| | |
|---|---|
| what varies | the deepresearch class's length distribution, scaled by **0.5 / 0.7 / 1.0 / 1.5 / 2.0** |
| how | `ms_dev/scripts/scale_class_profile.py` multiplies that class's `grid` by the factor and leaves `survival` alone, which is the same shape s times longer; mean/p50/p90/p99/max are scaled to match |
| what does not change | `decode_step_law`, `prefill_step_law` (properties of the engine, not the workload), and the other two classes |
| rate | 60 req/s (3600 rpm) — where the one existing point was measured |
| repeats | 2 |
| conditions | 10, eight minutes each, engines cold-restarted per condition |
| arm | `fluidserve`, m1, stock engine ordering, migration off |
| code change | none. The scaled file replaces `deploy/profiling/.../fluidserve.json`, the driver uploads it as the ConfigMap, and the chain restores the original at exit |

**deepresearch is the class to vary** because it is the one that actually moved
in production of this workload, because its outputs are the longest so the
absolute error in tokens is largest, and because the existing data point is on
that class.

**×1.0 is its own control inside this session**, which matters because
FluidServe at 60 req/s has a wide repeat spread historically (EXP-53: 55.0 to
66.7). EXP-62 also measures `fluidserve` at 60 req/s with two repeats in the
session immediately before, giving an external anchor for the same point.

## 3. Hypotheses and judgement rules

### H1 — the cost is asymmetric, and under-estimating is the damaging direction

**Prediction**: ×0.5 loses at least **8 points** of offered attainment against
×1.0, and ×2.0 loses less than ×0.5 does.

**Grounds**: an under-estimate tells the policy that resident requests will
finish sooner than they will, so the feasibility test believes an instance has
room it does not have and the KV projection under-shoots. Over-estimating makes
the policy conservative, which costs admissions rather than violations. The one
existing point is on the under-estimating side and it is large.

**Refutation**: ×2.0 loses as much as or more than ×0.5. That would mean the
conservative direction is equally damaging, and the defence in §8.5 of
`motivation.md` would have to be stated as a two-sided tolerance rather than a
one-sided one.

### H2 — ±30% is affordable

**Prediction**: ×0.7 and ×1.5 are both within **5 points** of ×1.0, which is
inside the repeat spread this arm shows at this rate.

**Refutation, and the more consequential outcome**: either loses more than 8
points. Then the profile is not a mild input but a tight requirement, and that
belongs in the paper as a stated limitation with this number attached, not in a
footnote. **Record it as a finding.** It would also raise the priority of an
online re-estimator, which is currently not in the design at all.

### H3 — the mechanism separates the two directions

**Prediction**: the rejection rate rises monotonically with the scale factor
(a longer assumed output means fewer requests look feasible), while the
attainment among admitted requests falls as the factor drops (the policy admits
work it cannot serve).

This distinguishes "the policy is wrong about capacity" from "the policy is
merely conservative", and the two failures need different fixes.

**Refutation**: rejection does not move with the factor. That would mean the
profile is not reaching admission at all, which would need explaining before any
score is read.

## 4. Checks that must pass before any number is read

- **The ConfigMap the cluster holds, not the file on disk, is the authority.**
  The chain reads back `configmap/llumnix-profiling` after every condition and
  prints `verified: configmap deepresearch mean=…`. A condition without that
  line is invalid. This exists because `GetLatencyPredictor` reads the profile
  once under `sync.Once`, so a scheduler that did not restart would silently
  keep the previous profile.
- The scheduler start-up line reports `fluidserve` with the same ablation flags
  as EXP-59 and EXP-62, and `set_scheduler_profiling.py` prints its `verified:`
  line.
- `metrics.csv` has data rather than only a header; if only a header, open
  `shards/` before recording the condition as failed.
- The original profile is restored at exit and `git diff` on that file is empty.

## 5. What this cannot answer

- **One class, one rate.** Scaling chat instead would be a different experiment
  and chat is 76.9% of arrivals, so its profile error probably matters more per
  unit; that is a follow-up, not this run.
- **Scaling is not the same as a shape error.** The real 2026-07-29 error changed
  the shape as well as the mean (a 4-section report became a 7-section one). This
  measures a pure location error.
- **It does not say what an online estimator would achieve**, only what the cost
  of a given error is. Whether per-request length knowledge would buy anything
  beyond a correct class distribution is EXP-64.

## 6. Result

To be filled in when the run finishes.
