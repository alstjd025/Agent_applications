# EXP-49 — candidate H2: project from the rate the occupancy is observed to be moving at

Written before the run, 2026-08-02 01:20 KST, while EXP-48 part 1 is still
running. **One change at a time**: this is the only difference from EXP-48's
condition, and it runs on EXP-48's corrected length profile.

## 1. What is being changed

```go
f.inflow  = f.nDecode * float64(p.cfg.horizonSteps)
f.outflow = p.expectedOutflow(f.live, p.cfg.horizonSteps)
f.proj    = f.kvLogical + f.inflow - f.outflow
```

becomes, behind `--fluidserve-kv-slope-projection`,

```go
delta  = kvSlopeOf(instance) * horizonMs        // tokens/ms, EWMA over status intervals
f.proj = f.kvLogical + delta
```

`kvSlope` is maintained in `observeInstance`, which already holds both ends of a
status interval, with the same smoothing constant `fsPrefillDutyAlpha = 0.1` the
prefill duty cycle uses. `inflow` and `outflow` are still published as the
positive and negative part of the same movement so the existing series keep a
meaning. Before any interval has been measured the slope is zero and the
projection is the occupancy itself, which is the permissive direction.

## 2. Why — the modelled balance has three terms where the quantity has four

Occupancy one horizon ahead is

```
current occupancy
  + the growth of the requests already resident
  − what the completions release
  + the footprint of the requests the scheduler places during the horizon
```

and FluidServe carries the first three. The fourth is absent deliberately: the
projection is meant to describe the instance *before* this one request is added.
At saturation that is not a small omission — EXP-47 measured **5.24 placements
per 500 ms status interval per engine**, so a 4.6 s horizon carries about 48 of
them, and at a mix-weighted footprint of some 1,732 logical tokens that is
**about 83,000 tokens** the projection does not contain.

**Measured (§48.2).** Pairing each published `projected_kv_tokens` with the
occupancy the same engine reported one horizon later:

| run | predictor | mean error | MAE | under |
|---|---|---|---|---|
| `full` hour, 14,676 pairs | occupancy only | +537 | 29,380 | 51.4% |
| | **shipped** | **−101,633** | 106,416 | **88.5%** |
| | **slope (this change)** | **+1,393** | **29,106** | 50.3% |
| static 60 req/s, 2,160 pairs | occupancy only | +1,730 | 35,841 | 49.7% |
| | **shipped** | **−103,411** | 112,358 | **87.0%** |
| | **slope** | +5,491 | 38,073 | 46.8% |

The shipped projection is worse than making no projection at all, and its error
is one-sided, so it does not average out over many decisions. The slope
projection is unbiased and has about a third of the absolute error. On the run
where the engines were driven hardest (`fsac`, 6,583 preemptions) it is also
better than making no projection: MAE 38,767 against 43,756.

## 3. Why this is not the offered-rate projection that was removed in v19

§15.3 and §15.4 removed an arrivals term for three reasons, and each is checked
against this one.

| §15's objection to `offeredRate` | this change |
|---|---|
| the value is a fleet total divided by instance count, so it is identical on every instance and cannot order candidates — its only effect is to tighten every gate at once | measured per instance from that instance's own status pair. §48.2's per-occupancy table shows the error, and therefore the correction, is several times larger on a full engine than on an empty one |
| it charges a 674-token chat request the same background load as a 22k-token swe request, which is how 44% of chat came to be rejected at 3000 rpm | the projection is a property of the instance; what each request is charged is still `costOf`, which is per request and unchanged |
| positive feedback through the rejection rate: `offered = served/(1−s)`, so s rising raises the projection, tightens the gate, and raises s again, with a fixed point at s = 54.6% | the slope counts requests that were **placed**, not requests that **arrived**. Placing more raises the slope, which lowers the headroom, which places less. The loop is negative and settles, the same construction `prefillDutyOf` uses |

## 4. What is given up

The slope says how fast occupancy is moving without saying why, so
`completionProb` and the class length profile leave the projection entirely.
Two consequences worth stating before the run:

- **A burst is extrapolated.** The slope is measured over about 500 ms and
  applied over 4.6 s. The EWMA window is roughly ten intervals, close to the
  horizon itself, but a step change in arrivals is still projected forward for a
  few seconds after it stops. That is the conservative direction on a rise and
  the permissive one on a fall.
- **§13's argument for the release term is not answered, only bypassed.** That
  argument was that charging every resident request its class mean over-predicts
  release for a heavy-tailed length distribution. It remains correct; the slope
  simply does not use a length distribution. If H2 is accepted, the class
  profile still drives `costOf` and `missesOwnBudget`, so it does not become
  dead input.

## 5. Design

| | |
|---|---|
| arms | `fluidserve` (EXP-48's condition, shipped projection) and `fskv` (`--fluidserve-kv-slope-projection=true`), both with A and C off and `FS_CLASS_HARM=false` |
| part 1 | static m1, 45 and 60 req/s, 2 repeats |
| part 2 | `full` hour, 1 repeat per arm |
| binary | `scheduler-exp49-H2` md5 `c3c6a3518092ea80eb7bb4dc1725ae5b`, one binary for both arms |
| profile | EXP-48's corrected `classes[]` in both arms |

The baseline arm is re-run rather than read from EXP-48 for part 1, because the
binary changes between the two experiments and one condition of a baseline is
cheap next to the risk of attributing a binary difference to the flag. For part
2, EXP-48's `full` run is the baseline if it has been made by then.

## 6. Judgement rules, written before any result

**Rule 0 — the mechanism check, and it comes first.** Under this flag the
published `projected_kv_tokens` **is** the slope projection, so scoring the
`fskv` run with `exp48_projection_error.py` measures the deployed projection
directly rather than an offline reconstruction. Required: **mean error within
±20,000 tokens and the under-prediction fraction between 40% and 60%.** If the
deployed projection is still biased, the offline result did not transfer and the
reason has to be found before any score is read, whatever the score says.

1. **Preemptions on `full` below 500**, against a baseline of 1,471 / 1,852 /
   1,605 across three runs of the shipped projection with the old profile, and
   against EXP-48's number with the new one. This is the failure H2 is built to
   remove.
2. **Static offered at 60 req/s not more than 4.2 points below the `fluidserve`
   arm of the same session.** 4.2 is EXP-38's largest measured repeat spread on
   this workload.
3. **Rejection rising while offered does not rise is a failure** (§33.3),
   reported with token goodput beside both denominators.
4. **Refutation of the premise.** If rule 0 passes — the projection is now
   unbiased — and preemptions do not fall and the score does not move, then the
   projection was never what limited admission, and the next thing to measure is
   which term of `feasible` actually refuses placements at saturation.
   `c.feasible` is a conjunction of four conditions and no run has yet recorded
   which of them binds. **This binary answers that**: a counter
   `scheduler_fluidserve_infeasible_total{reason}` is incremented for each
   condition that failed, separately rather than only for the first, so two
   failing together is visible as such. It changes no decision, and it is in
   both arms.

## 7. Result

(to be filled in)
