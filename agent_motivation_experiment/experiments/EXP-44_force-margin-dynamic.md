# EXP-44 — candidate A on an hour of moving load

Written before the run, 2026-08-01 10:40 KST. EXP-42 accepted candidate A on
static conditions; this asks whether the gain transfers to load that moves, and
what it does to the two failure modes EXP-41 recorded.

## 1. Design

| | |
|---|---|
| arms | `fluidserve` (A off) and `fsa` (A on) |
| trace | `full` = `dyn60_short_m123`, the same hour EXP-41 ran |
| repeats | 1 (about 2.3 hours); a second is a separate decision |
| engine | stock FIFO, four engines, migration off, engine admission off |
| binary | `scheduler-exp42-A` md5 `d24861df81d89257f5f16dbbb70bb295`, one for both arms |
| `--fluidserve-class-harm` | **false in both arms, set explicitly** |

**The Llumnix SLO arm is dropped** at the user's direction. The question here is
what A did, and both arms answer it inside one session. The FluidServe-versus-
Llumnix comparison already exists from EXP-41 and can be re-run once A is
settled; **no claim about that gap may be made from this run**, because its
baseline would be in another session.

**Why class-harm is pinned false in both arms.** EXP-41's FluidServe ran that way
(§38), and `set_scheduler_profiling.py` now returns unset ablations to their
compiled default, which for this flag is `true`. Without pinning, the
`fluidserve` arm would differ from EXP-41 in two ways instead of none.

## 2. Predictions, written before the run

**Whole-hour offered attainment for `fsa`: about 68**, against EXP-41's 59.7 for
the same trace and policy without A. Derived by carrying the static gains of
EXP-42 onto EXP-41's rate bands:

| band | requests | EXP-41 FluidServe | predicted with A |
|---|---|---|---|
| under 40 req/s | 35,439 | 93.2 | ~95 |
| 40–50 | 30,694 | 75.9 | ~85 |
| 50–60 | 46,518 | 60.1 | ~70 |
| 60+ | 66,767 | 34.4 | ~45 |

Weighted by request count that is **68.2**. The same arithmetic reproduces
EXP-41's measured 59.7 from its own bands, which is the check that the weighting
is right.

**This is an extrapolation.** The static conditions went to 60 req/s; this trace
reaches 77 in a 30-second bin and holds above 65 for six minutes. Nothing was
measured there.

### What should be fixed

**EXP-41 §6.1, the loss at minutes 50–56.** FluidServe scored 14.9 against
Llumnix SLO's 17.4 in the only stretch above 65 req/s for more than 2.5 minutes.
The mechanism was that it admitted 3,965 chat requests of which **91.2% missed**,
3,546 of them on time-between-tokens. A rejects exactly those placements: at a
static 60 req/s it took admitted attainment from 55.5 to 93.6. **Prediction:
`fsa` scores higher than `fluidserve` across minutes 50–56.**

### What may get worse, and this is the prediction worth writing down

**EXP-41 §6.2, the 1,471 preemptions on engine 8003.** A sheds chat harder and
deep research less — measured at a static 60 req/s, chat 37.1% → 56.1% and deep
research 13.0% → **7.3%** — and deep research is what filled that engine. Its
share of resident KV went 70.8% → **73.5%** with A.

**Prediction: preemptions do not fall, and may rise.** Static conditions recorded
zero on both arms, so this trace is the only place the question can be asked.

If preemptions rise, that is not a reason to reject A — its score gain is
measured and large — but it promotes candidate B (putting the request's KV
occupancy into the shed decision, the user's proposal) from "well motivated" to
"the next thing to fix", because A moved the class balance in the direction B
corrects.

## 3. Judgement rules

1. **`fsa` whole-hour offered at least 5 points above `fluidserve`.** Below that,
   the static gain did not transfer and the reason has to be found before
   anything else is built on A.
2. **`fsa` above `fluidserve` across minutes 50–56.** This is the specific
   failure A was expected to fix.
3. **Preemptions recorded whatever the score does**, per engine, against
   EXP-41's 1,471 on 8003.
4. **No loss in the low band.** `fsa` under 40 req/s must not fall below
   EXP-41's 93.2 by more than the repeat spread; the closest available figure
   for that spread is the 0.4 points EXP-42 measured between candidate A's two
   repeats at 45 req/s, so a fall of more than about 2 points is a real loss.
5. **One run per arm.** A difference smaller than EXP-38's measured repeat
   spread on this workload (up to 4.2 points, and 10.2 at 45 req/s on the
   shipped policy) is not established by this run, whatever its sign.

## 4. Result

(to be filled in)
