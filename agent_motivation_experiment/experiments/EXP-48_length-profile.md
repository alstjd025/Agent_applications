# EXP-48 — the deep research length profile was three and a half times short

Written before the run, 2026-08-02 01:05 KST. This is **an input correction, not
a policy change**: no decision rule is edited and no flag is added. What changes
is one file the policy reads.

## 1. What was wrong

`deploy/profiling/llama31-70b-b200-tp2/fluidserve.json` states the empirical
output-length distribution of each class. For deep research it said **mean 282,
p50 256**. The workload has been producing **mean 985, p50 973** since
2026-07-29.

The profile was built by `d4e8250` (2026-07-26) from the EXP-21 metrics, and the
deep research mean measured in those runs is 277.8, so **it was correct when it
was written**. `5fa82f8` in `Agent_applications` (2026-07-29 16:35 KST) then
changed `searcharena.py`'s report structure from four sections to seven and gave
each request notes from its own conversation; that commit's own message records
the effect it measured, median output 249 → 942. The profile was not rebuilt.
Run by run the transition is one day wide: 07-27 reads 149–300, 07-28 reads
274–903, and every run from 07-29 onward reads 947–984.

## 2. Why it matters — it is the release term of the KV projection

```
outflow = Σ_live  completionProb(j_i, 100) × kvTokens_i  −  z·σ
proj    = kvLogical + nDecode×100 − outflow
```

`completionProb(j, k)` is `[S(j) − S(j+k)] / S(j)`, read off the survival curve
the profile carries. With the 07-26 curve, a deep research request that has
already produced 300 tokens is given a **0.602** chance of finishing within the
next 100 iterations. It has about 650 tokens left, so the true value is near
0.1. Averaged over a resident set spread across its own lifetime the completion
rate per 100 iterations is `100 / mean length`:

| class | 07-26 profile | measured | ratio |
|---|---|---|---|
| chat | 0.237 | 0.240 | 1.0 |
| swe | 0.192 | 0.206 | 0.9 |
| **deepresearch** | **0.355** | **0.105** | **3.4** |

and deep research holds an estimated **77% of resident KV** (arrival share ×
residency: chat 77.9%×12.6 s, dr 14.3%×72.7 s, swe 7.8%×13.9 s, times per-request
footprints of 878 / 4,851 / 6,462 logical tokens). So the term that says how much
KV is about to be released is over-predicted on the class that dominates it.

**Measured consequence, §48.2 of `fluidserve-implementation.md`:** the projection
was compared against what the engines actually held one horizon (4.6 s) later,
14,676 paired samples from `260731_2313_exp45r1_fluidserve_full`. The shipped
projection under-predicts **88.5%** of the time, mean error **−101,633** tokens,
mean absolute error 106,416 — against 29,380 for using the current occupancy and
making no projection at all. The bias grows with occupancy, from −7.1% of the
actual future value on a near-empty engine to **−22.7%** on a near-full one.

## 3. What exactly changed in the file

**`classes[]` only.** `decode_step_law`, `prefill_step_law` and
`mixed_step_validation` are byte-identical to the 07-26 document, verified by
comparison after writing. They describe the engine, were fitted on the EXP-16
per-step dumps, and rebuilding them would make this two changes instead of one.
(§34.4's separate finding that the deployed `c_kv` is about 1/1.6 of the measured
value is left alone here.)

| class | mean before → after | n (07-30 … 08-01 runs) |
|---|---|---|
| chat | 422 → **428** | 1,780,160 |
| swe | 520 → **494** | 209,282 |
| deepresearch | 282 → **985** | 410,195 |

Conditional completion probabilities over the 100-iteration horizon:

| class | cp(0,100) | cp(300,100) | cp(600,100) | cp(900,100) |
|---|---|---|---|---|
| chat | 0.114 → 0.150 | 0.258 → 0.234 | 0.423 → 0.396 | 0.374 → 0.330 |
| swe | 0.000 → 0.000 | 0.151 → 0.175 | 0.619 → 0.719 | 0.558 → 0.473 |
| **deepresearch** | 0.006 → 0.005 | **0.602 → 0.001** | **0.567 → 0.058** | 0.061 → **0.317** |

The 07-26 profile is kept beside the new one as
`fluidserve-classes-20260726.json` so a reader who sees the old numbers can find
the document they came from.

### Everything in the policy this reaches

`expectedToks` is `expectedRemaining(0)`, the class mean, and it enters the
decision path **only through the end-to-end branch**, which is swe alone:

| use | chat | deepresearch | swe |
|---|---|---|---|
| `outflow` via `completionProb` | small | **large** | small |
| `costOf` growth | none — clamped at `min(100, expectedToks)` and both values exceed 100 | none | none |
| `missesOwnBudget` end-to-end test, `deadline` | not used | not used | 520 → 494, slightly less demanding |
| `requestBudget` nominal pace for an end-to-end class | not used | not used | 30000/520 = 57.7 ms → 30000/494 = 60.7 ms |

## 4. Design

| | |
|---|---|
| arm | `fluidserve` — the shipped policy, candidates A and C both off, `FS_CLASS_HARM=false` pinned |
| part 1 | static m1, 45 and 60 req/s, 2 repeats, 8 minutes each (~55 min) |
| part 2 | `full` = `dyn60_short_m123`, 1 repeat (~70 min) |
| binary | `scheduler-exp47-probe` md5 `809b823b540629f6c0fc32adb3b87fdb`, **unchanged** |
| profile | the new `fluidserve.json`; `set_scheduler_profiling.py` recreates the ConfigMap per condition from the repository file |

**Only one profile is measured here.** The runner reinstalls the ConfigMap for
every condition from the repository file, so putting both profiles in one session
would mean editing a file on the measurement path while the sweep runs, which is
the defect that made EXP-42 lose a condition. The comparison is therefore against
recorded values from earlier sessions, which the 2026-08-01 rule allows provided
each quantity is read against its own repeat spread. The baselines available:

| quantity | old-profile `fluidserve` | n | spread |
|---|---|---|---|
| static 45 req/s offered | 99.1, 88.9 (EXP-42) | 2 | 10.2 |
| static 60 req/s offered | 35.2 (EXP-38), 36.1 (EXP-42) | 2 | 0.9 |
| static 60 req/s goodput | 12,359, 12,401, 12,638 | 3 | 279 |
| `full` whole-hour offered | 59.7 (41), 59.2 (44), 60.3 (45) | 3 | 1.1 |
| `full` preemptions | 1,471 (41), 1,852 (44), 1,605 (45) | 3 | 381 (26%) |

**Verification before the numbers are used.** The scheduler reads the profile
once at start-up (`sync.Once` in `GetLatencyPredictor`), so the ConfigMap being
right is not the same as the running process having it. After the first
condition's rollout, the mounted file inside the scheduler pod is checksummed
against the repository file. A mismatch voids the run.

## 5. What is expected, and what would refute it

**The correction ships whatever the score does.** The deployed value states 282
for a class that measures 985; that is wrong independently of what it buys, and
leaving it in place means every later candidate is judged on an input known to be
false. What this run decides is the size and direction of the effect, and which
of the two causes in §48.4 dominates.

**Prediction.** `outflow` falls by an estimated 59%, so `proj` rises, headroom
falls, and admission tightens on the engines holding deep research. Static
conditions have never recorded a preemption on this policy, so at 45 and 60 req/s
the change should be visible only as **fewer admissions**: rejection up, offered
attainment flat or slightly down. On the hour it should be visible as **fewer
preemptions**.

1. **Part 1 is a regression gate and it stops the experiment on failure.**
   Offered at 60 req/s must not fall more than **4.2 points** below 36.1 — that
   is EXP-38's largest measured repeat spread on this workload, so a larger fall
   is a real cost. At 45 req/s the same rule cannot be applied, because the two
   recorded repeats are 99.1 and 88.9; that rate is reported and not gated.
2. **Rejection rising while offered does not rise is a failure** (§33.3). Token
   goodput is reported beside both denominators for exactly this reason.
3. **Part 2, the direct test of §48.3.** Preemptions **below 1,000** confirm that
   the over-predicted release term was the dominant cause of engines filling.
   **Above 1,400** — inside the baseline's own 1,471–1,852 range — means it was
   not, and that the missing arrival term of §48.4 is what matters, which
   promotes H2 (projecting from the engine's observed rate of change of KV
   occupancy) to the next change.
4. **The refutation of §48.3 as a mechanism at all**: preemptions unchanged *and*
   offered, rejection and goodput all unchanged at both static rates. That would
   mean `outflow` never binds on a decision, and §48.2's bias, though correct
   arithmetic, does not reach admission. In that case the next thing to measure
   is which term of `feasible` actually refuses placements at saturation, not
   another projection change.

## 6. Result

(to be filled in)
