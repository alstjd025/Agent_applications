# EXP-42 — candidate A: the same allowance margin on the forced-placement test

Written before the run, 2026-08-01 02:10 KST. This is the first of the four
candidates in `fluidserve-implementation.md` §37.6. **One change at a time**, at
the user's instruction, so that a result can be attributed.

## 1. What is being changed and why

Two places in the policy compare the same predicted quantity — the mean time to
produce one token on the instance after this request is placed there
(`c.meanAfter`) — against the same request's budget, and until now they used
different thresholds.

| path | condition | threshold for chat, whose budget is 50 ms |
|---|---|---|
| ROUTE (`evaluate`, fluidserve.go:1259) | `meanAfter ≤ min(gateAllowance, nominalMs) × fsAllowanceUtilisation` | **45.0 ms** |
| FORCE (`missesOwnBudget`, fluidserve.go:1322) | reject if `meanAfter > nominalMs`, otherwise place | **50.0 ms** |

A request whose predicted pace falls between 45.0 and 50.0 ms is therefore
refused a routed placement, held at the gateway until it can wait no longer, and
then placed anyway.

That band is where the failures are. Measured at 60 req/s (EXP-38, two
repeats pooled): chat's realised mean inter-token latency has a median of
**49.9 ms** against its 50 ms budget, and **62% of admitted chat requests miss on
it**. The prediction is not what is wrong — over minutes 50–56 of EXP-41 the
scheduler's predicted step and the engine's observed step agree to within 0.1 ms
on all four instances. What differs between a placement that succeeds and one
that does not is which of the two thresholds it was tested against.

The change makes `missesOwnBudget` use the same margin the routing test already
uses:

```go
budget := req.nominalMs
if p.cfg.forceMargin {
    budget *= fsAllowanceUtilisation      // 0.90, the constant already in use
}
return req.nominalMs > 0 && c.meanAfter > budget
```

**No new constant.** `fsAllowanceUtilisation` is already 0.90 and already applied
on the routing path; this applies it consistently. Only the branch for classes
judged on time-between-tokens is touched. The end-to-end branch (the swe class)
and the time-to-first-token branch are left alone, because changing them is a
different change and would make this result unattributable.

**Why it should not cost anything directly.** Under the offered denominator a
rejection and a miss both score zero, so a request that is going to miss scores
the same whether it is rejected or placed. Placing it is not free: it occupies a
decode slot and its KV for its whole life. Rejecting it instead returns that
capacity to the requests around it.

## 2. What is expected, on what grounds, and what would refute it

**Expected.** At 60 req/s, requests predicted to land between 45 and 50 ms are
rejected instead of forced. Estimating with the refit decode law
(`t = 14.792 + 2.266e-5·M + 0.06759·n`, per engine, from 14,428 EXP-38 per-tick
samples):

- about 62% of admitted chat stops being placed, which is roughly 341 requests
  across the fleet, 85 per engine → `0.06759 × 85` = **−5.75 ms**
- those requests were holding roughly 850 logical tokens each, 72,250 per engine
  → `2.266e-5 × 72,250` = **−1.64 ms**
- total **about −7.4 ms**, taking the realised pace from 49.9 to about 42.5 ms

42.5 ms is below both thresholds, so the requests that remain should meet the
rule and the policy should return to routing rather than forcing.

**These are calculations, not measurements.** They assume the decode law holds
at the new operating point and that the requests removed are the ones the
estimate names.

**Refutation conditions, in the order they will be checked.**

1. **The mechanism did not fire.** If the share of decisions that are `shed` does
   not rise at 60 req/s, the flag did not take effect and the run is void, not
   negative. Check the scheduler's start-up line for `forcemargin=true` first.
2. **It fired and did not help.** If chat's median inter-token latency at 60
   req/s does not fall below **47 ms**, the estimate above is wrong and the
   change is rejected. 47 rather than 42.5 because the estimate is coarse; the
   direction is what is being tested.
3. **It helped the pace and not the score.** If offered attainment at 60 req/s
   does not rise above **45** from the current 35.2, the freed capacity did not
   convert into requests meeting their rule. §33.3 already records one case where
   rejecting earlier moved the numerator and the denominator together and the
   offered score did not move, so this is a real possibility and not a formality.
4. **It cost something elsewhere.** If offered attainment at 45 req/s falls below
   **88.5 − 4.2 = 84.3** (the current value less the largest repeat spread ever
   measured on this arm), the change is rejected regardless of what it does at
   60. Same for 15 and 30 req/s, which are at 100.0 and must stay there.
5. **It only rejects more.** If the rejection rate rises and token goodput at 60
   req/s does not rise above the current 12,359 tokens/s, the change is rejected.

A result that passes 1–2 but fails 3 is still worth recording: it would mean the
pace is not the binding constraint at that load, which contradicts §34.

## 3. Design

| | |
|---|---|
| arms | `fsA` (`--fluidserve-force-margin=true`) and `fsBase` (flag absent, shipped behaviour) |
| rates | 15, 30, 45, 60 req/s (900/1800/2700/3600 rpm) |
| repeats | 2, as the outer loop |
| mix | m1, the same workload as EXP-38 |
| engine | stock FIFO, four engines, migration off, engine admission off |
| binary | one binary for both arms, `scheduler-exp42-A` md5 `d24861df81d89257f5f16dbbb70bb295` |
| session | one session for all conditions |
| `--fluidserve-class-harm` | **false in both arms, set explicitly** (see below) |

**Why class-harm is pinned to false.** While verifying that the new flag reached
the scheduler, its start-up line reported `classharm=false`, which is not the
compiled default and which this run had not asked for. Reading the 120 archived
deployment specs in time order shows the flag was written by one ablation arm on
2026-07-28 10:28 and stayed in the deployment for the 61 conditions that
followed, because `set_scheduler_profiling.py` kept ablation flags it was not
asked to set. EXP-27 pass 2 onward, EXP-28 to EXP-38, EXP-40 and EXP-41 all ran
that way. The script now removes ablations it does not set, and EXP-42 pins this
one to false **in both arms** — not because false is right, but because a
baseline that silently differed from EXP-38 would make this result incomparable
with the experiments it is being read against. Turning it on is EXP-43.
Full account: `fluidserve-implementation.md` §38.

**Both arms run from the same binary**, differing only in the flag, so the
comparison cannot pick up a build difference. The flag is off in the shipped
default, so `fsBase` needs no environment variable and is the same code path
that produced EXP-38's numbers.

15 and 30 req/s are included even though nothing is expected to change there,
because refutation condition 4 needs them and because at those rates the policy
routes 99.7% of the time and never reaches the changed line at all — so a
difference there would mean something is wrong with the harness rather than with
the change.

**Go boolean flags must be written `--flag=value` as one token.** Written
separated, pflag sets the flag true and treats `"false"` as a positional
argument, which is how EXP-25's ablation arm ran four hours with the setting it
was supposed to have turned off. `set_scheduler_profiling.py` writes the joined
form and reads the scheduler's start-up line back to confirm; the run is void
without that confirmation.

## 4. Result

(to be filled in)
