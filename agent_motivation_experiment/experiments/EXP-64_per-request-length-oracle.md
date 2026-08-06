# EXP-64 — what would per-request length knowledge buy?

Written before the run. 2026-08-07 01:50 KST.

## 1. The question

Our policy knows each class's output-length distribution and nothing about the
individual request. Two of the systems this work is compared against know more:
Scorpio classifies each request into one of 100 length bins, AdaGen predicts its
decode length with a DistillBERT model at 76.4% accuracy. SLOs-Serve assumes the
decode length is known exactly.

So a reviewer can ask two things and we can answer neither today:

1. **Would a per-request predictor help us?** If it would help a lot, the design
   is leaving something on the table and the paper should say so.
2. **Is our class-conditional estimate the reason we win?** If per-request truth
   adds nothing, then the answer is no, and the comparison against systems that
   do use predictors is not an information-advantage comparison.

**Building a predictor answers neither cleanly**, because a negative result would
be attributable to the predictor being bad. **Giving the policy the answer does.**
Whatever the gain from near-perfect per-request length knowledge turns out to be,
no predictor can beat it, so it is an upper bound on the whole question.

EXP-63 measures the cost of the class-level estimate being *biased*. This
measures the value of removing its *per-request variance*. They are the two
halves of "how much does the length input matter".

## 2. What "oracle" means here, exactly

The hint is **the output length the same request produced in an earlier run of
the same condition**, joined on `(task_id, call_index)`.

**It is not perfect and the paper must say by how much.** The same input
produces a different output length between runs even with a fixed seed, because
continuous batching changes the reduction order and flips near-ties: over 9,660
matched requests the inputs were 100% identical and the outputs were identical
**65.7%** of the time (chat 72.4 / deepresearch 44.2 / swe 40.9), with
`|difference|` p50 of 0 to 4 tokens and p90 of 28 to 122
(`fluidserve-implementation.md`, trap D).

So this is **a predictor about as accurate as a predictor could plausibly be**,
not an oracle in the strict sense. That is the honest description and it is also
the more useful one: it bounds what a real predictor could achieve.

**`(task_id, call_index)` is unique only in the eight-minute static conditions.**
On the one-hour trace 95,558 of 106,116 rows share that pair because tasks are
replayed hundreds of times, so this experiment is static-only.

## 3. How the hint reaches the policy — and why the workload does not change

The OpenAI completion API's **`user`** field is a free-form string that the
gateway already declares and re-marshals, and that vLLM accepts and ignores for
generation. The client packs `len:<tokens>` into it.

**This matters more than it sounds.** The obvious alternative, `max_tokens`,
would truncate generation, which changes the workload and makes the hint an
upper bound rather than a value. With `user` the two arms emit **byte-identical
requests** and differ only in whether the scheduler reads the field, so the
comparison has no workload confound at all.

| component | change |
|---|---|
| `pkg/types/scheduling_request.go` | `PredictedOutputTokens int` and a parser for the `len:<n>` encoding, rejecting anything else rather than guessing |
| `pkg/gateway/load-balancer/scheduler_client.go` | populate it from `cr.User` |
| `cmd/config/config.go` | `--fluidserve-oracle-length`, default **false** |
| `pkg/scheduler/policy/fluidserve.go` | when the flag is on and the request carries a positive hint, the request's expected remaining tokens and completion probability come from the hint instead of the class distribution — **for resident requests as well as the arriving one**, since the feasibility test is about what the incumbents still have to produce |
| workload | write `user="len:<n>"` from a lookup table built from a prior run |
| analysis | `ms_dev/scripts/build_length_oracle.py` builds the table from a result directory |

**Default off means the control arm is the same policy every experiment since
EXP-27 measured**, and a request without a parseable hint falls back to the class
distribution, so a missing table entry degrades to today's behaviour rather than
to zero.

## 4. Design

| | |
|---|---|
| rates | 45 and 60 req/s |
| arms | `fluidserve` (control, class distribution) and `fsoracle` (per-request hint) |
| repeats | 2 |
| conditions | 8, eight minutes each |
| mix | m1, stock engine ordering, migration off |
| oracle source | a prior `fluidserve` run at the same rate and mix, named in the experiment record |
| coverage check | the fraction of requests that carried a parseable hint, printed per condition; below 95% the condition is invalid |

Both arms send the `user` field. Only the flag differs. That way a mistake in
building the table cannot make the two arms differ in what they sent.

## 5. Hypotheses and judgement rules

### H1 — the gain is small

**Prediction**: `fsoracle` is within **3 points** of the control at both rates.

**Grounds**: the policy uses the length estimate through `E[L−j | L>j]`
multiplied by a pace, and the classes are chosen so that the within-class spread
is much smaller than the between-class spread (chat p50 389 / p90 766,
deepresearch p50 973 / p90 1246). If that is right, the class label already
carries most of the information and per-request truth adds little.

**Refutation, and the more consequential outcome**: the gain exceeds **5 points**
at either rate. Then per-request length prediction is a real lever we are not
pulling, the comparison against Scorpio and AdaGen becomes an
information-disadvantage comparison rather than the reverse, and the paper has to
say so. **Record it as a finding, not a failure** — it would point at a concrete
next version of the design.

### H2 — if there is a gain, it is concentrated in the class with the widest spread

**Prediction**: whatever gain appears, deepresearch shows more of it than chat,
because its p90/p50 ratio is the one the class mean serves worst in absolute
tokens.

### H3 — the mechanism is admission, not placement

**Prediction**: the rejection rate changes more than the per-engine class
composition. A better length estimate sharpens the feasibility test, which
decides whether a request can be served at all; it does not change which
instances a class prefers.

**Refutation**: the effective instances per class move by more than 0.3 while the
rejection rate does not move. That would mean the hint is acting somewhere other
than where it was wired in and the wiring must be re-read.

## 6. What this cannot answer

- **It is not a predictor.** A deployed system would have to predict the length
  from the prompt, and the accuracy achievable there is 76.4% in AdaGen's
  measurement, well below the 65.7%-exact / few-token-error hint used here.
  **The result bounds the value; it does not deliver it.**
- **It says nothing about the profile being biased**, which is EXP-63.
- **Static only**, for the join-key reason in §2.

## 7. Result

To be filled in when the run finishes.
