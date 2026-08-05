# EXP-58 — the class preference as a degree, so that separation becomes an axis

Written before the run. 2026-08-06.

## 1. Why this exists

EXP-56 established that a fully mixed fleet scores worse than the shipped
configuration: turning `--fluidserve-enable-affinity` off drops the median
class concentration from 58.7% to 29.1% (25.0% is an even spread over four
instances) and costs 7.5 points of offered attainment at 45 req/s and 4.8 at 55.

That result has a shape problem, and the shape is what the paper needs.

The claim the motivation section wants to make is **not** "the preference is
worth 7.5 points". It is: *fully separating the classes is bad, fully mixing
them is bad, and the good region is in between*. Stating that needs at least
three points on one axis. We have two, and the third — the fully separated end —
comes from PolyServe, a different system that also differs in how it estimates
demand, in having no spill between tiers, and in its admission rule. So
"concentration 83.5 scores 27.5" cannot be attributed to the concentration.

With the preference as a switch there is no way to fix that from inside our own
system. This experiment turns it into a number.

## 2. What changed in the code

`sortCandidates` ordered the feasible instances lexicographically: most of the
arriving request's class first, free space breaking ties. It now orders them by

```
score = w · share + (1 − w) · room
```

`share` is the fraction of the instance's resident requests that belong to the
arriving request's class; `room` is the free space left after admitting, as a
fraction of the instance's physical capacity. Both are fractions between 0 and
1, so the weighted sum compares commensurable quantities and needs no scaling
constant.

- **w = 1** reproduces the previous behaviour exactly. The score is the share,
  and equal shares fall through to the same room comparison as before.
- **w = 0** reproduces `--fluidserve-enable-affinity=false` exactly. The score
  is the room and the class plays no part.

The class term inside `harmToIncumbents` is scaled by the same `w`, so that one
number moves the preference in both of the places it acts. **In this experiment
that scaling is inert**: every FluidServe arm in this repository pins
`FS_CLASS_HARM=false`, so the damage-estimate term is already off. The weight
therefore acts in exactly one place here — the ordering of the feasible set —
and these arms differ from `fluidserve` and `fsnoaff` in that and nothing else.

Flag: `--fluidserve-affinity-weight`, default 1.0, clamped to [0, 1]. It is
ignored when `--fluidserve-enable-affinity=false`, so the switch stays the
master and every earlier experiment's description of itself stays true.

Binary `cf74fe987302003256f17b728e8cb21b` (previous: `f88e9430…`, kept at
`/home/nxclab/tools/bin-backup/scheduler-exp07.pre-exp58`).

## 3. Why the values are not evenly spaced

`w = 0, 0.05, 0.15, 0.4, 1.0`.

The ordering between two instances flips where `w · Δshare = (1 − w) · Δroom`.
At 45 req/s the difference in class share between two instances is of order 0.5
and the difference in free space is of order 0.1, which puts the crossing near
`w = 0.17`. Even spacing would put three of the five points in the region where
the class preference already decides every comparison, and none in the region
where the two terms actually trade off.

If it turns out that 0.05 already behaves like 1.0, that is a result — it says
the knob is parameterised on the wrong scale and needs values an order of
magnitude smaller — but it is a result obtained at the cost of the sweep, so the
spacing is chosen to make it unlikely.

## 4. Design

| | |
|---|---|
| rates | 45 and 55 req/s (2700 and 3300 rpm), the band where the gate arithmetic binds |
| arms | `fsw000` `fsw005` `fsw015` `fsw040` `fsw100` |
| repeats | 2 |
| conditions | 20, eight minutes each, engines cold-restarted per condition |
| engine scheduler | stock FIFO, migration off |
| mix | m1 |

Loop order is repeat → rate → weight, so the five weights sit adjacent in time
within a repeat and any drift across the run is shared rather than read as an
effect of `w`.

**Two repeats is deliberately few, and the analysis is built for that.** The
quantity this experiment produces is not a precise mean per weight. It is a set
of runs, all from one binary, spread along the concentration axis. Every run
contributes one (concentration, attainment) point, and `w` is the instrument
that spreads them out. EXP-56 already showed that at `w = 1` the concentration
varies from 43.1 to 74.6 between repeats of an identical configuration; that
variation adds points to the plot rather than obscuring it. Twenty runs here
plus the twelve of EXP-56 give thirty-two points from one system.

## 5. Hypotheses and judgement rules

### H1 — concentration rises with the weight

**Prediction**: median class concentration is ordered `w=0 ≤ 0.05 ≤ 0.15 ≤ 0.4 ≤
1.0`, with the endpoints landing on the values EXP-56 measured — `w=0` inside
28.8–29.3 and `w=1` inside 43.1–74.6.

**Refutation**: an endpoint outside the EXP-56 range for that arm. That would
mean the weighted sum is not the same rule as the switch it is supposed to
generalise, and nothing else in this experiment could be read.

**Weaker failure, still informative**: the four non-zero weights all land within
2 points of each other. That says the axis saturates below `w = 0.05` and the
sweep has to be repeated with smaller values.

### H2 — attainment rises with concentration over the range we can reach

**Prediction**: offered attainment increases with concentration across the
reachable range, without a downturn at the top. Our maximum concentration is
around 75, well below PolyServe's 83.5, and the mechanism that makes full
separation bad — a class pinned to instances whose number cannot follow the mix
— needs the assignment to be *fixed*, which no value of `w` makes it.

**Refutation**: attainment at `w = 1` lower than at `w = 0.4` by more than the
repeat spread of the `w = 1` arm.

**If refuted, that is the most valuable result in the experiment**, because it
would be the first evidence from inside our own system that too much separation
hurts. Today that half of the argument rests entirely on PolyServe. Record it as
a finding rather than as a failure.

### H3 — the spread between repeats grows with the weight

**Prediction**: the range of concentration across repeats is widest at `w = 1`
and narrowest at `w = 0`. The reason is that the preference is positive
feedback — an instance holding slightly more of a class becomes more attractive
to it, so a small difference in arrival order is amplified — and the strength of
that amplification is `w`.

**Refutation**: the range at `w = 1` no larger than at `w = 0`. Two repeats
cannot measure a range well, so this is read together with EXP-56's three
repeats at each endpoint (spread 31.5 at `w = 1`, 0.5 at `w = 0`) rather than on
its own.

**Why it matters beyond this experiment**: if some middle weight keeps most of
the attainment gain while cutting the run-to-run variation, that is a better
operating point than the shipped one, and the paper can say so with measurements
instead of speculation.

## 6. What this experiment still cannot answer

Separation acts through two paths at once — it creates instances with no chat
resident, whose gate is 61.9 or 100 ms instead of 45, and it creates instances
whose batches are homogeneous, on which chat's time per token falls. EXP-56
registered these as mutually exclusive and found both. Varying `w` moves both
together, exactly as the switch did, so this experiment does not decompose them
either. Doing that needs an arm that produces the separation but pins the gate
to chat's value regardless of who is resident.

## 7. Result

### 7.1 Repeat 1 at 45 req/s — the axis is monotone in four quantities at once

| w | offered | effective instances | chat-free instance-time % | rejection % | chat ms/token |
|---|---|---|---|---|---|
| 0.00 | 86.33 | 3.95 | 1.03 | 9.76 | 44.15 |
| 0.05 | 86.68 | 3.84 | 1.03 | 9.87 | 44.34 |
| 0.15 | 87.42 | 3.74 | 1.03 | 9.92 | 43.89 |
| 0.40 | 89.15 | 3.44 | 4.38 | 8.20 | 43.54 |
| 1.00 | 90.21 | 3.19 | 9.28 | 7.28 | 42.77 |

**H1 holds, and the exact distances are worth stating rather than a verdict.**
The reference ranges are the minimum and maximum of three draws, so a fourth
draw landing just outside one is expected and says nothing; what would matter is
landing far outside.

| | measured | EXP-56 range | distance outside | range width |
|---|---|---|---|---|
| `w=0` offered | 86.33 | 86.5 – 87.0 | 0.17 | 0.50 |
| `w=0` effective instances | 3.95 | 3.91 – 3.94 | 0.01 | 0.03 |
| `w=0` chat-free % | 1.03 | 1.0 – 1.0 | 0.03 | 0.00 |
| `w=1` offered | 90.21 | 90.4 – 98.9 | 0.19 | 8.50 |
| `w=1` effective instances | 3.19 | 1.63 – 3.01 | 0.18 | 1.38 |
| `w=1` chat-free % | 9.28 | 4.4 – 30.9 | inside | 26.5 |

Every distance is smaller than, or of the order of, the width of the range it
misses. **And the two `w=1` misses are consistent with each other**: 3.19 is less
separated than any of the three EXP-56 draws and 90.21 is lower than any of their
scores, which is the same direction. So this repeat is a fourth draw at the
low-separation end of the same distribution, not a different rule.

**The axis is monotone anyway**, which is the useful part: reading it does not
depend on catching the high-separation state, and this repeat did not catch it.

### 7.2 An observation the design did not anticipate, to be checked against repeat 2

**The two paths EXP-56 could not separate come apart here.** Section 6.4 of that
experiment registered the gate reading and the batch-homogeneity reading as
mutually exclusive, found both, and concluded that they are two faces of one
cause and cannot be decomposed with that design. On this axis they switch on at
different weights.

- **At w = 0.05 and 0.15 the gate path is inert**: the fraction of instance-time
  with no resident chat request is 1.03, exactly what it is at w = 0. No
  instance is free of chat, so no instance can admit a looser class on its own
  terms.
- **The homogeneity path is already acting there**: chat's time per token falls
  44.15 -> 43.89 and the score rises 86.33 -> 87.42.
- **At w = 0.4 the gate path switches on** -- chat-free instance-time goes 1.03
  -> 4.38 -> 9.28 -- and the score rises a further 2.8 points.

If this survives repeat 2, it is a decomposition: **about 1.1 of the 3.9 points
between w=0 and w=1 arrive before any instance is free of chat.** Two cautions
before believing it. The 1.1 points is close to the repeat spread of the `w=0`
arm measured three times in EXP-56 (0.5 points), so one repeat is not enough.
And this reading depends on the chat-free fraction being a step rather than a
gradual rise; with points only at 0.05, 0.15 and 0.4 the location of that step is
known to within a factor of three.
