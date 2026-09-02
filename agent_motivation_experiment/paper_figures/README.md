# paper_figures

Figures built for the paper. Each is drawn at its final physical size so that
LaTeX includes it at a scale factor of 1.0 and the type lands on the page at the
size it was set at. **Do not re-scale these in LaTeX** — `scale=`, `\resizebox`,
or a `width=` other than the one named below multiplies every font size by the
same factor.

| Script | Output | Size | Include as |
|---|---|---|---|
| `fig_exp27_pass3.py` | `exp27_pass3_attainment_goodput_offered.pdf` | 3.335 × 2.22 in | `figure`, `width=\columnwidth` |
| | `exp27_pass3_attainment_goodput_admitted.pdf` | 3.335 × 1.95 in | `figure`, `width=\columnwidth` |
| `fig_exp53_policies.py` | `exp53_attainment_goodput_offered.pdf` | 3.335 × 2.22 in | `figure`, `width=\columnwidth` |
| | `exp53_attainment_goodput_admitted.pdf` | 3.335 × 1.95 in | `figure`, `width=\columnwidth` |
| `fig_exp50_hour.py` | `exp50_hour_attainment_goodput.pdf` | 7.000 × 1.60 in | **`figure*`**, `width=\textwidth` |
| `fig_workload_lengths.py` | `workload_lengths.pdf` | 3.335 × 1.95 in | `figure`, `width=\columnwidth` |
| `fig_azure_trace_shape.py` | `azure_trace_shape.pdf` | 3.335 × 1.60 in | `figure`, `width=\columnwidth` |
| `fig_exp54_hour.py` | `exp54_hour_offered.pdf` | 7.000 × 1.60 in | **`figure*`**, `width=\textwidth` |
| | `exp54_hour_admitted.pdf` | 7.000 × 1.60 in | **`figure*`**, `width=\textwidth` |

`paper_style.py` holds the width, the rcParams, the arm colours and markers, the
`k`-suffix tick formatter, and `save()`. Import from it rather than copying the
constants; `save()` in particular exists so that no figure here reintroduces
`bbox_inches="tight"`.

3.335 in is the USENIX single column: `usenix2019_v3.sty` sets `\textwidth=7in`
and `\columnsep=0.33in`. All type is 8 pt except the panel-internal annotations.
Fonts are embedded as Type 42.

---

## `fig_exp27_pass3.py` — attainment and token goodput vs offered rate

Two side-by-side panels: (left) SLO attainment, (right) output-token goodput,
both against offered request rate. Two versions, differing only in whether the
offered-denominator curves are drawn.

### Runs

Twenty run directories, all `m1` balanced mix, four engines, 8 minutes per
condition, `--restart-per-condition` so each condition is independent.

| Arm | Legend | Colour / marker | Runs | n per rate | Session |
|---|---|---|---|---|---|
| `polyserve` | PolyServe | `#d62728` ○ | `results/260728_00*_exp27p3r{1,2}_polyserve_m1_rpm_*` | 2 | 2026-07-28 00:08–02:45 |
| `slo` | Llumnix | `#2ca02c` △ | `results/260728_0[89]*_exp27p5r1_slo_m1f_rpm_*` | **1** | 2026-07-28 08:38–09:21 |
| `fluidserve` | FluidServe | `#1f77b4` □ | `results/260728_01*_exp27p3r{1,2}_fluidserve_m1_rpm_*` | 2 | 2026-07-28 01:06–03:37 |

Rates: 1200 / 2400 / 3600 / 4800 rpm = **20 / 40 / 60 / 80 req/s**.

The glob is `RUNS` in the script: `results/*exp27p3*` and `results/*exp27p5*`.

**Pass 4 is deliberately excluded.** `exp27p4*` is the same Llumnix baseline at
the 25 ms/token setting, which judged the agent class 2.3× tighter than
FluidServe judged it and rejected 98% of it; that curve belongs to a policy
answering a different question. This matches the exclusion in
`analysis_scripts/request_level/exp27_figures.py`, which is the authority on the
run selection.

The baseline runs are tagged `m1f`, not `m1`. `mix_of` folds the `f` variant
onto `m1` — same three workloads, same ratio, same arrival process and seed;
only the `(ttft_ms, tbt_ms)` pair the baseline is configured with differs.

### Metrics

Computed by `analysis_scripts/request_level/exp22_fluidserve.py`, which the
figure script imports rather than reimplements, via `exp27_figures.collect`.

- **Analysis window** — arrivals from 60 s after the first request to 20 s
  before the last completion. Requests still in flight at the end are excluded
  from both denominators; their outcome was never determined.
- **Violation** — per class: chat TTFT > 5 s or ITL > 50 ms; deep research
  TTFT > 10 s or ITL > 100 ms; agent (swe) end-to-end > 30 s. A request that
  produced no first token is a violation under any rule.
- **SLO attainment, admitted** (solid) — denominator is the requests the system
  accepted. A rejected request leaves the population.
- **SLO attainment, offered** (dotted) — denominator is every request that
  arrived; a rejection counts as a violation.
- **Goodput token (t/s)** — output tokens from requests that met their rule on
  the offered denominator, divided by the window. Whole-request: a request that
  missed contributes none of its tokens. Input tokens never count.
- Aggregation is **per request**, every request counting once. Not the
  class-equal average.

**Inter-token latency is the corrected quantity.** It is derived as
`(e2e − ttft) / (output_tokens − 1)`, not read from the recorded `tbt_mean_ms`
column, which is about half the true value on every run collected before
2026-07-30 (`fluidserve-implementation.md` §32). These runs predate that, so
they score very differently under the two bases:

| | uncorrected | corrected (drawn) |
|---|---|---|
| FluidServe @ 80 req/s, admitted | 87.0% | **48.1%** |
| Llumnix @ 60 req/s, admitted | 93.7% | **28.6%** |

Any table or figure of these runs quoted from before 2026-07-30 is on the
uncorrected basis. `FS_LEGACY_TBT=1` reproduces it.

The defect is in what the client recorded, so it is bounded by run date, not by
analysis date. Ratio of the recorded column to the corrected value, chat median,
one run per experiment:

| Runs | recorded / corrected |
|---|---|
| EXP-27 pass 3 and pass 5 (2026-07-28) | **0.52** — affected |
| EXP-41 (2026-07-30 20:13 on) | 1.04 |
| EXP-42 (2026-07-31) | 1.01 |

So only this figure's runs are moved by the correction. EXP-41 and EXP-42 were
collected after the client was fixed and read the same either way.

### Per-class attainment, admitted denominator

The aggregate is per request, and chat is 76.9% of the requests, so the
aggregate tracks chat. Reporting it alone would hide that the three classes fail
at completely different rates.

| Arm | req/s | chat | deep research | agent (swe) |
|---|---|---|---|---|
| PolyServe | 20 | 100.0 | 100.0 | 99.7 |
| PolyServe | 40 | **16.6** | 100.0 | 100.0 |
| PolyServe | 60 | **6.2** | 100.0 | 99.5 |
| PolyServe | 80 | **3.0** | 99.9 | 98.6 |
| Llumnix | 20 | 100.0 | 100.0 | 100.0 |
| Llumnix | 40 | 99.8 | 100.0 | 100.0 |
| Llumnix | 60 | **10.8** | 100.0 | 52.0 |
| Llumnix | 80 | **5.2** | 100.0 | 40.0 |
| FluidServe | 20 | 100.0 | 100.0 | 100.0 |
| FluidServe | 40 | 99.9 | 100.0 | 100.0 |
| FluidServe | 60 | 96.5 | 99.9 | 91.9 |
| FluidServe | 80 | 38.2 | 94.5 | 65.3 |

Every collapse in this figure is the chat class missing its 50 ms inter-token
budget. Deep research, whose budget is 100 ms, is at or near 100% for all three
arms at every rate; the agent class, judged end to end, only moves for Llumnix
and for FluidServe at 80 req/s. This is the reason the correction matters so
much here: it is the per-token half of the rule that was being judged against
twice its budget, and chat is the class with the tightest per-token budget and
the largest share of the requests.

### Values drawn

Mean over repeats; range is min..max.

| Arm | req/s | n | Attain. admitted | range | Attain. offered | range | Goodput t/s | range | Rejected |
|---|---|---|---|---|---|---|---|---|---|
| PolyServe | 20 | 2 | 100.0 | 99.9–100.0 | 100.0 | 99.9–100.0 | 8,528 | 8,510–8,546 | 0.0% |
| PolyServe | 40 | 2 | 39.3 | 38.6–40.0 | 39.3 | 38.6–40.0 | 5,004 | 4,908–5,101 | 0.0% |
| PolyServe | 60 | 2 | 32.5 | 32.0–33.0 | 32.5 | 32.0–33.0 | 5,836 | 5,739–5,932 | 0.0% |
| PolyServe | 80 | 2 | 28.4 | 28.2–28.6 | 28.4 | 28.2–28.6 | 7,258 | 7,208–7,309 | 0.0% |
| Llumnix | 20 | 1 | 100.0 | — | 100.0 | — | 8,619 | — | 0.0% |
| Llumnix | 40 | 1 | 99.9 | — | 99.9 | — | 16,525 | — | 0.0% |
| Llumnix | 60 | 1 | 28.6 | — | 27.1 | — | 4,763 | — | 5.3% |
| Llumnix | 80 | 1 | 32.5 | — | 20.6 | — | 4,674 | — | 36.1% |
| FluidServe | 20 | 2 | 100.0 | 99.9–100.0 | 100.0 | 99.9–100.0 | 8,573 | 8,562–8,584 | 0.0% |
| FluidServe | 40 | 2 | 99.9 | 99.9–99.9 | 99.9 | 99.8–99.9 | 16,546 | 16,521–16,571 | 0.1% |
| FluidServe | 60 | 2 | 96.8 | 96.7–97.0 | 87.8 | 87.2–88.3 | 21,371 | 21,345–21,398 | 9.3% |
| FluidServe | 80 | 2 | 48.1 | **40.1–56.2** | 33.6 | 28.1–39.1 | 9,861 | 7,719–12,002 | 29.6% |

### What the caption has to carry

The drawing cannot say these, and two of them change how the figure reads.

1. **The rejection rates.** Attainment on an admitted denominator is not
   interpretable without them — a policy that refuses everything scores 100%.
   Llumnix rejects 5.3% at 60 req/s and 36.1% at 80; FluidServe 9.3% and 29.6%;
   PolyServe never rejects. In the `_admitted` version this is the only place
   the reader can learn it.
2. **The baseline is one run per rate**, so it has no bars. A point with no bar
   beside points with bars reads as the more precise measurement, which is the
   opposite of the truth.
3. **The baseline ran in a separate session** (08:38–09:21) from the other two
   arms (00:08–03:37). The size of session-to-session movement on this workload
   was measured as 0.1 attainment points at 20 req/s and 4.6 at 80, but that was
   measured on the uncorrected basis and has not been recomputed.
4. **The n=2 spread is not uniformly small.** Under 0.5 points at 20–60 req/s on
   every arm, but **16.1 points on FluidServe at 80 req/s** (40.1 vs 56.2). No
   claim should rest on that point.

### Regenerating

```bash
cd agent_motivation_experiment
python3 paper_figures/fig_exp27_pass3.py
```

Both PDFs are rewritten. The script reads the run directories directly; it holds
no cached numbers, so a change in `exp22_fluidserve.load_run` changes the figure
silently. Re-run it after any change to the loader and compare against the table
above.

---

## Why EXP-42's FluidServe cannot be substituted into this figure

Asked 2026-08-01. It cannot, for two independent reasons.

**The rates do not line up.** EXP-42 measured 900 / 1800 / 2700 / 3600 rpm =
15 / 30 / 45 / 60 req/s. This figure is at 20 / 40 / 60 / 80. Only 60 req/s is
shared, so three of four x positions would have no EXP-42 measurement and the
other two arms would have none at 15 / 30 / 45.

**The workload changed.** Same workload name (`mixed_request_level_poisson`),
same mix weights (chat 10 : deep research 2 : swe 1), same transcript file, and
the same SLO block — but the deep-research class was reworked between them. Per
class, over completed requests only (so rejections do not distort it):

| Class | EXP-27 pass 3, output tokens mean / p50 | EXP-42, mean / p50 |
|---|---|---|
| chat | 423 / 389 | 415 / 381 — unchanged |
| agent (swe) | 490 / 487 | 487 / 487 — unchanged |
| **deep research** | **284 / 250** | **990 / 976 — 3.5×** |

EXP-42's `run_config.json` records the cause: "deep research moved 4,055 → 4,639
when its system prompt grew and its notes became the question's own". Measured
input tokens agree (4,122 → 4,363 for deep research; chat and swe unchanged).
`max_tokens` also went from unset to 4096, but EXP-27's deep-research p95 output
was 535 tokens, so nothing was near a cap and the cap is not the cause.

Weighting by arrival shares (chat 76.9%, deep research 15.4%, swe 7.7%), the
mean decode tokens per request goes from 406 to 509, so **the same nominal
request rate is about 25% more decode work in EXP-42**. That is why FluidServe
at the one shared rate reads 96.8% admitted here and 56.2% in EXP-42, and it is
consistent with EXP-42 having moved its sweep down to 15–60 req/s.

Putting a series from each into one figure would place three arms on three
different workloads. An EXP-42 figure has to be built from EXP-42 arms only.

---

## `fig_exp53_policies.py` — four control planes on one static rate sweep

Same two panels and same geometry as the EXP-27 figure, four arms instead of
three.

### Runs

Thirty-two conditions, `results/*exp53r*`, m1 balanced mix, 8 minutes each,
`--restart-per-condition`. Rates 900–4200 rpm = **15, 25, 35, 45, 50, 55, 60,
70 req/s**. Run selection and the per-condition aggregation come from
`analysis_scripts/request_level/exp53_compare.py`, which this script imports.

| Arm | Legend | Colour / marker |
|---|---|---|
| `fluidserve` | FluidServe | `#1f77b4` □ |
| `polyserve` | PolyServe | `#d62728` ○ |
| `slo` | Llumnix SLO | `#2ca02c` △ |
| `loadbalance` | Llumnix | `#9467bd` ▽ |

**Every cell is n=1.** All 32 conditions are single runs, so no point on this
figure carries an error bar and none of the spread numbers that exist for
EXP-27 apply here. This is the single most important thing the caption has to
say, because the reader's default reading of an unbarred point is that it is
precise.

### Values drawn

`off` = offered denominator, `adm` = admitted, `gp` = goodput in output tokens/s.

| req/s | FluidServe off / adm / rej / gp | PolyServe off=adm / gp | Llumnix SLO off / adm / rej / gp | Llumnix off=adm / gp |
|---|---|---|---|---|
| 15 | 100.0 / 100.0 / 0% / 8,049 | 100.0 / 7,883 | 100.0 / 100.0 / 0% / 7,846 | 100.0 / 8,023 |
| 25 | 100.0 / 100.0 / 0% / 13,419 | 99.7 / 13,214 | 99.9 / 99.9 / 0% / 13,387 | 100.0 / 13,357 |
| 35 | 100.0 / 100.0 / 0% / 17,879 | 40.5 / 8,379 | 99.9 / 99.9 / 0% / 18,163 | 99.7 / 18,093 |
| 45 | 90.0 / 97.6 / 7.6% / 20,426 | 27.6 / 5,519 | 51.9 / 56.7 / 8.4% / 12,390 | 32.2 / 9,098 |
| 50 | 77.9 / 93.0 / 16.1% / 20,072 | 26.8 / 5,571 | 34.7 / 44.0 / 20.7% / 10,760 | 14.3 / 3,727 |
| 55 | 66.3 / 88.4 / 24.3% / 19,019 | 18.1 / 4,123 | 30.7 / 48.6 / 36.1% / 11,072 | 9.8 / 2,438 |
| 60 | 66.7 / 97.7 / 31.0% / 20,775 | 17.0 / 4,122 | 26.4 / 54.0 / 50.3% / 11,099 | 7.3 / 1,749 |
| 70 | 51.5 / 94.0 / 44.3% / 19,454 | 13.7 / 3,937 | 21.1 / 69.8 / 68.9% / 11,659 | 4.0 / 842 |

### Why the offered version is the one to use

This sweep is the clearest case in the project of the admitted denominator
misleading. **Llumnix SLO's admitted attainment RISES from 44.0% at 50 req/s to
69.8% at 70 req/s while its rejection rate goes 20.7% → 68.9%.** It is not
recovering; it is refusing more than two thirds of arrivals and scoring itself
on the third it kept. On the offered denominator the same arm falls 34.7 → 21.1.
FluidServe's admitted curve has the same shape for the same reason, though less
extremely (31.0% → 44.3% rejected).

The admitted-only PDF is for the case where the surrounding text already gives
the rejection rates. If it does not, use the offered one.

### Open discrepancy — migration on PolyServe

`exp53_compare.py` prints a standing note on every figure: *"Llumnix arms run
with migration enabled; FluidServe and PolyServe do not use it and run
without."* Its own counter, reading `server_metrics/migration_events.log`,
disagrees for PolyServe:

| Arm | non-zero rescheduling pairs, summed over its conditions |
|---|---|
| FluidServe | 0 |
| **PolyServe** | **225** (8 at 45 req/s, 27 at 50, 40 at 55, 60 at 60, 90 at 70) |
| Llumnix SLO | 0 |
| Llumnix | 13 |

So the arm the note says ran without migration is the one with by far the most
rescheduling pairs, and one of the two arms the note says had it enabled
produced none. **This is unresolved and the note is not reproduced on the paper
figure until it is.** Either the note is wrong about how PolyServe was
configured, or `rescheduling_pairs` counts something other than what the note
means by migration. Whichever it is, the asymmetry between the arms is not
currently established in the direction the note claims.

### Regenerating

```bash
cd agent_motivation_experiment
python3 paper_figures/fig_exp53_policies.py
```

---

## `fig_exp50_hour.py` — the hour, FluidServe against Llumnix SLO

Two side-by-side panels against time: (left) SLO attainment on the **admitted**
denominator, (right) output token goodput. **This is the only figure here drawn
at the full text width (7.0 in), so it goes in a `figure*` at
`width=\textwidth`** — an hour at 30 s steps is ~119 points per line, which is
unreadable in a 1.3 in panel.

It is panels D and G of `results/aggregate_analysis/exp50/hour_timeline.png`,
reduced from four arms to the two being compared and redrawn at paper size.

### Runs

One hour of moving load, the same trace for both arms, **one run each, from
different experiments a day apart**.

| Arm | Legend | Colour / style | Run |
|---|---|---|---|
| FluidServe, corrected length profile | FluidServe | `#1f77b4` solid | `results/260801_1808_exp50p2r1_fluidserve_full` |
| Llumnix SLO | Llumnix SLO | `#2ca02c` dashed | `results/260731_2203_exp45r1_slo_full` |

The trace steps its class mix every 15 minutes — **m1, m2, m3, m1, fifteen
minutes each** — marked with grey vertical guides at 15, 30 and 45 min. The
segment tags are no longer drawn on the figure, so **the caption has to name
them**, otherwise the three guides are unexplained lines. Both curves move at
those instants, which is what makes the shape attributable to the workload
rather than to the policy. That the two arms saw the same trace is verified by
panel A of the source figure, which draws the offered rate per arm and gets one
curve.

### Metrics

Imported from `exp41_dynamic_timeline.py` and `exp22_fluidserve.py`, not
reimplemented.

- **Window** — 90 s wide, stepped every 30 s, anchored on **arrival**. A point
  at minute *t* is "of the requests that arrived around *t*, this is what
  happened to them"; a request arriving at *t* and finishing at *t*+30 s is
  scored at *t*. Windows holding fewer than 30 requests are dropped.
- **SLO attainment, admitted** — rejections leave the denominator entirely. This
  is the quality of the work the policy chose to do, not the fraction of offered
  load that was served.
- **Goodput token (t/s)** — output tokens/s from requests that met their rule,
  judged whole-request. A rejected request produced no tokens and contributes
  none; that is a fact about tokens, not a choice of denominator, so this panel
  is the same under either.

### Whole-hour figures

| Arm | Attainment, admitted | Attainment, offered | Rejected |
|---|---|---|---|
| FluidServe | **95.5%** | 69.7% | 27.0% |
| Llumnix SLO | 66.3% | 38.3% | 42.1% |

### What the caption has to carry

1. **The rejection rates above.** The admitted denominator is the flattering one
   for both arms and neither rejection rate appears anywhere on this figure.
   FluidServe's 95.5% admitted is 69.7% offered; Llumnix SLO's 66.3% is 38.3%.
   Without those numbers the left panel overstates both arms, and it overstates
   the baseline more, because the baseline rejects more.
2. **One run per arm.** No point on either line is an average and no wiggle is
   noise-bounded. Every feature is one realisation.
3. **The two runs are from different experiments a day apart** — EXP-45 for the
   baseline, EXP-50 part 2 for FluidServe. Same trace, different session.
4. **What the three vertical guides mark**: the class mix steps m1 → m2 → m3 →
   m1 at 15, 30 and 45 minutes. Nothing on the figure says so any more.

### Regenerating

```bash
cd agent_motivation_experiment
python3 paper_figures/fig_exp50_hour.py
```

It prints the window count and the whole-hour figures for each arm; check them
against the table above.

---

## The workload, for the paper's setup section

Three request classes drawn from three sources, arriving as one request-level
Poisson stream with classes drawn in shuffled fixed-composition blocks. Model:
`meta-llama/Meta-Llama-3.1-70B-Instruct`, four engines, `max_tokens` 4096.

| Class | Source | Dataset |
|---|---|---|
| chat | `sharegpt_request_level_poisson` | ShareGPT — HF `anon8231489123/ShareGPT_Vicuna_unfiltered`, `ShareGPT_V3_unfiltered_cleaned_split.json`, rev `192ab218`, 1,000 conversations → 3,354 requests, seed 42. Multi-turn chat turns sent as independent requests |
| deep research | `searcharena_request_level_poisson` | Search Arena — HF `lmarena-ai/search-arena-24k`, `data/search-arena-chat-24k.parquet`, rev `fac8dcf8`, English, 12,603 questions and 33,152 notes → 60,000 requests, K notes per request with K ∈ [2, 12], seed 42. Each request is a synthesis over search-grounded notes |
| agent (swe) | `codingagent_request_level_poisson` | SWE-bench Lite — literal replay of a recorded concurrency-1 transcript, `transcript_swe_short7k_mix1500.jsonl`. Prompts are replayed verbatim, so the input distribution is fixed by the recording |

Mix on `m1`: request counts chat 10 : deep research 2 : agent 1, i.e. **76.9 /
15.4 / 7.7 %** of requests.

### Length distributions

Tokens, measured at the gateway. Output is over **completed requests only** (not
rejected, errored, or cut off at run end), pooled over the four arms of EXP-53
at **15 req/s** — the lowest rate in that sweep, where all four arms hold 100%
attainment and reject nothing, so the lengths are set by the workload and not
truncated by saturation. The four arms agree to within 2% on every class median,
which is the check that this is a workload property rather than a policy one.

| Class | n | Input median | Input p90 | Input std | Output median | Output p90 | Output std |
|---|---|---|---|---|---|---|---|
| chat | 21,178 | 556 | 1,580 | 589 | 384 | 775 | 395 |
| deep research | 4,225 | 3,993 | 7,519 | 2,411 | 971 | 1,233 | 274 |
| agent (swe) | 2,157 | 5,725 | 7,787 | 1,416 | 517 | 648 | 119 |

Means, where they are wanted alongside: input 677 / 4,328 / 5,788, output 445 /
973 / 531.

The three classes are deliberately spread on both axes: chat is short in and
short out with a long tail (input std larger than its median), deep research is
long in and long out with the tightest output distribution, and the agent class
is the longest input with a short, tight output. That spread is what makes the
per-class SLO budgets bind at different rates.

### SLO rules

There are two SLO specifications and they are not the same object. Do not quote
one for the other.

**1. What attainment is scored against** — `SLO_RULES` in
`analysis_scripts/request_level/exp22_fluidserve.py`. A request is a violation
if it breaks its class rule, or if it produced no first token at all.

| Class | Rule |
|---|---|
| chat | TTFT ≤ 5 s **and** inter-token ≤ 50 ms |
| deep research | TTFT ≤ 10 s **and** inter-token ≤ 100 ms |
| agent (swe) | end-to-end ≤ 30 s |

These are **absolute thresholds**, not multiples of a solo baseline; `--tau` is
unused on these workloads. Inter-token latency is the corrected quantity,
`(e2e − ttft) / (output_tokens − 1)`.

**2. What is sent to the engine** — the `slo` block of the run config, packed
into the OpenAI `priority` field as `ttft_ms * 1000 + tbt_ms`, where the `tbt`
half doubles as PolyServe's tier key.

| Class | `ttft_ms` | `tbt_ms` | `out_len` |
|---|---|---|---|
| chat | 5,000 | 50 | 386 |
| deep research | 10,000 | 100 | 275 |
| agent (swe) | 11,800 | 25 | 728 |

The two agree for chat and deep research. For the agent class they are the same
budget written two ways: the class is scored end to end at 30 s, and the engine
needs a first-token deadline, so the 30 s is decomposed as
`ttft = 30,000 − out_len × tbt = 30,000 − 728 × 25 = 11,800 ms`. That is the
TTLT-to-first-token conversion in `workloads/swe_bench_coding/agent.py`.

Two stale numbers in that block, neither of which invalidates a measurement:

- **deep research `out_len` 275** against a measured median of 971. It is
  **inert**: `agent.py` only reads `out_len` on the e2e-only branch of the
  deadline calculation, and all three classes set `ttft_ms`, so that branch
  never runs.
- **agent `out_len` 728** against a measured median of 517. This one *was* used,
  once, to compute the 11,800 ms above. At 517 tokens the decomposition would
  give `30,000 − 517 × 25 = 17,075 ms`, so the engine is handed a first-token
  deadline about 5.3 s tighter than the 30 s end-to-end budget requires. That is
  conservative rather than wrong — the class is scored on the 30 s, which is
  unaffected — but it means the agent class is prioritised slightly harder than
  its own SLO demands.

### Note on output-length reproducibility

The same input does not always produce the same output length. Matched on
`(task_id, iteration)` over 9,660 requests, inputs were 100% identical while
outputs matched on only 65.7% (chat 72.4, deep research 44.2, agent 40.9%).
Continuous batching puts a request in a different batch each run, which changes
the floating-point reduction order and flips the choice at near-tied tokens.
Differences are usually small (|Δ| p50 = 0–4 tokens) but the tail is long
(p90 28–122, max 2,904). This is a property of batched serving, not a
misconfiguration, and it is one reason a single run per condition is not enough.

---

## `fig_workload_lengths.py` — length distributions of the three classes

`workload_lengths.pdf`, 3.335 × 1.95 in, `figure`, `width=\columnwidth`. Two
panels — (left) input tokens, (right) output tokens — each carrying one
probability density per class. This is the distribution behind the median / p90 /
std table above: the table gives three numbers per class, the density shows the
shape those numbers summarise.

Same population as the table — the four arms of EXP-53 at 15 req/s, pooled,
completed requests only, 27,560 requests.

**Reading it.**

- **chat** is the only broad distribution on either axis: input spans three
  decades with a secondary mode near 20 tokens (very short turns) and a main
  mode near 1k; output is a single wide mode near 400. Its input std (589)
  exceeding its median (556) is this shape.
- **deep research** is narrow on both axes and is the only class whose output
  mode sits above 1k tokens.
- **agent** has the highest input mode and a narrow output near 500. Its input
  is a small number of recurring prompt shapes — the transcript is replayed
  verbatim — which is why its curve is the spikiest.
- The two panels share one x range, so the horizontal distance between a class's
  input mode and its output mode is its expansion ratio and can be compared
  across classes by eye.

**How the density is computed.** Gaussian KDE over `log10(tokens)`, drawn
against a log x axis, because the data spans 2 to 15,334 tokens: a kernel wide
enough to smooth the high end erases the low end, and one narrow enough for the
low end leaves the high end a comb of spikes. Consequences:

- The curve is a density **per decade**, not per token. Area between two x values
  is the share of that class's requests in that range; height is not comparable
  to a linear-axis density. The y axis carries no numbers for that reason.
- **Each class integrates to 1 over its own requests**, not over the traffic. The
  curves answer "given a request of this class, how long is it". They do *not*
  show that chat is 77% of requests — the shares are in the legend.
- Heights are comparable between curves within a panel, not across the two
  panels.

---

## `fig_azure_trace_shape.py` — how much the source load varies

`azure_trace_shape.pdf`, 3.335 × 1.60 in, `figure`, `width=\columnwidth`.
Arrival rate over the four-day Azure window our dynamic trace is built from,
normalised to its own peak. The claim is the vertical extent.

### Why the SOURCE and not our replayed trace

Our trace is **not** a linearly rescaled Azure trace, and the difference decides
what this figure is allowed to say. `traces/dynamic/build_dynamic_mix_trace.py`
applies a **quantile (rank) transform** onto a chosen band — 10 to 50 req/s —
which keeps the temporal order and autocorrelation of the source (when Azure is
busy, we are busy) but replaces the distribution of rates with a uniform one
over the band. Its own docstring says to describe the result as "Azure-shaped,
rescaled to our cluster", not "an Azure trace".

So **the replayed trace's peak-to-trough ratio is 50/10 = 5.0× by
construction** — a parameter chosen so the run sweeps the band the fleet
resolves. Drawing that as evidence that real serving load varies would be
circular. The variation claim has to come from the source series, which is what
this figure draws.

A second consequence: time spent at each rate is uniform in the replay and is
not uniform in the source. Horizontal extents on this figure are not the
replay's.

### Window and numbers

`traces/azure/plots/_minute_{conv,code}2024.csv` — per-minute request counts of
the Azure LLM Inference 2024 conversation and code traces, summed index-wise —
restricted to `day_window=[0,4]` from `start_hour=6`, i.e. minutes 360–6120,
5,760 minutes, four days. The script reads those parameters from
`traces/dynamic/canonical/dyn60_azure4d.plan.json` rather than hardcoding them,
so regenerating the trace with a different window cannot leave this figure
describing the old one.

| | requests/min | fraction of peak |
|---|---|---|
| max | 9,049 | 1.000 |
| p95 | 7,351 | 0.812 |
| median | 3,887 | 0.430 |
| p5 | 2,181 | 0.241 |
| min | 1,568 | 0.173 |

**max / min = 5.77×**, **p95 / p5 = 3.37×**. Quote whichever matches the claim
being made and say which it is — the first is the extreme-to-extreme range and
is sensitive to single minutes, the second is the robust one. The generator's
docstring quotes ~3.6× for p95/p5, computed over a different window than the
four days used here.

The conv and code traces cover **different calendar weeks** and are summed by
index, which aligns hour-of-day phase but not calendar date. That is the
generator's documented caveat, carried here for completeness; it does not affect
the diurnal shape.

---

## `fig_exp54_hour.py` — three control planes on the hour-long dynamic trace

Two side-by-side panels against time: (left) SLO attainment, (right) output
token goodput. Full text width, `figure*`. Two versions differing only in the
attainment denominator. The dynamic counterpart of the EXP-53 static sweep, and
the four-policy counterpart of the EXP-50 hour figure.

### Runs

`dyn60_short_m123`, mean offered 50.1 req/s, about 179,000 requests. Arrival
rate follows an Azure production trace; the class mix steps m1 → m2 → m3 → m1 at
15-minute boundaries, marked with grey guides. **The guides are not labelled on
the figure, so the caption has to name them.** Repeat 1, 2026-08-03.

| Arm | Legend | Colour / style | Run |
|---|---|---|---|
| FluidServe | FluidServe | `#1f77b4` solid | `results/260803_1751_exp54r1_fluidserve_full` |
| PolyServe | PolyServe | `#d62728` dash-dot | `results/260803_1905_exp54r1_polyserve_full` |
| Llumnix SLO | Llumnix SLO | `#2ca02c` dashed | `results/260803_2117_exp54r1_slo_full` |

**One run per arm.** Repeat 2 was still running when this was drawn and covers
only FluidServe. Do not add that one arm's repeat by itself: an arm carrying a
band beside two arms without one reads as the better-measured arm rather than
the only repeated one.

### The load-balance arm is excluded, and not because it lost

`260803_2229_exp54r1_loadbalance_full` was stopped at 87% of its tasks and its
`metrics.csv` holds one row. It rejects nothing, so from minute 40 all four
engines saturated, long-queued requests had their streams cut, and the client
began failing to obtain source ports: **62,114 `[Errno 99] Cannot assign
requested address`, against zero in every other arm.** Its last twenty minutes
measure the load generator, not the policy. Drawing it as a fourth curve would
report a client defect as a policy result. State the exclusion and the reason.

### Which version to use

**The offered one, unless the surrounding text already gives the rejection
rates.** The three arms reject at completely different rates, so the admitted
denominator flatters them by completely different amounts:

| | rejected | offered | admitted | goodput tok/s | throughput tok/s |
|---|---|---|---|---|---|
| FluidServe | 27.0% | **70.0** | **95.9** | **18,014** | 18,944 |
| Llumnix SLO | 42.1% | 38.7 | 67.0 | 11,451 | 16,163 |
| PolyServe | 0.0% | 13.9 | 13.9 | 3,198 | 18,150 |

PolyServe's two attainment numbers are identical because it never rejects; the
other two gain 26 and 28 points from having refused work. The headline of this
figure is the last two columns together: **PolyServe produces 18,150 output
tokens/s against FluidServe's 18,944, a 4% difference, and 3,198 of them versus
18,014 land inside a latency rule.** Same fleet, same token production, 5.6× the
useful output.

### One value disagrees with the experiment write-up

`experiments/EXP-54_four-policy-hour.md` §2 gives PolyServe **17.6** on both
denominators. The runs give **13.9**, and that write-up's own per-class table
implies 13.9: 3.7 / 8.8 / 95.3 on chat / deep research / agent over 140,304 /
21,161 / 17,978 requests weights out to 13.9, not 17.6. Every other cell of that
table reproduces here exactly, including PolyServe's goodput (3,198) and
throughput (18,150), so this is one value in that document rather than a
difference of window or scoring. **Resolve it there before quoting either
number.** The figures here use 13.9.

### The x axis stops at 57.8 minutes, not 60

A request still in flight when the run ends has an unknown outcome, so it leaves
both denominators instead of counting as a violation. At the end of a backlogged
run that removes precisely the slow requests. **PolyServe's final window holds
3,678 arrivals of which 3,396 — 92.3% — never finished**; the 282 that did were
the fast ones, and attainment reads 99.6% against 9.1% two minutes earlier.
Drawn untrimmed, the figure shows the static partition recovering at the end,
which is the opposite of what happened.

Windows above 20% in-flight-at-end are dropped and **all arms are cut at the
same time**, which lands at 57.8 min. Maximum in-flight-at-end share per arm:
FluidServe 4.5%, Llumnix SLO 5.8%, PolyServe 92.3% — the artifact exists only on
the arm with a backlog, and it flatters that arm. The script prints the trim and
the per-arm maxima when it runs.

The EXP-50 hour figure was checked against the same threshold and needs no trim:
its two arms peak at 4.4% and 6.3%.

### Regenerating

```bash
cd agent_motivation_experiment
python3 paper_figures/fig_exp54_hour.py
```

## `intro_capacity.pdf` / `intro_capacity_curves.pdf` (2026-08-13 갱신, EXP-80)

**주장**: 도착한 요청의 90% 이상이 자기 지연 규칙을 지키는 최대 도착률이
**FluidServe v0.2 28.0 req/s, llm-d 19.9 req/s로 1.41배**, 가장 낮은 PolyServe(15.8)에
대해서는 **1.78배**다. 거절은 위반으로 세므로 전부 거절해서 이 값을 살 수 없다.
`paper_figures/fig_intro_capacity.py`.

**데이터**: 2026-08-08 워크로드 수정 **이후**의 정적 조건만, **다섯 arm × 여덟 도착률
(10/15/20/25/35/45/55/70 req/s) × 2반복 = 80 run**. 출처는 EXP-68·69·70(FluidServe·llm-d의
1반복), EXP-72(PolyServe·Llumnix SLO의 1반복), EXP-77(vLLM router 2반복),
**EXP-80(모자란 칸 23개를 채운 2반복)**.

**run 목록은 `paper_experiment/static_sweep_2026-08/manifest.tsv`에 고정돼 있다.** 그림을
다시 그리기 전에 `python3 paper_experiment/verify.py static_sweep_2026-08`를 돌린다 — glob에
새로 걸리는 run, `metrics.csv`의 변경, 채점의 변경 셋을 따로 검사한다.

| 포화 기준 | vLLM router | PolyServe | Llumnix SLO | llm-d | **FluidServe** | FluidServe/llm-d |
|---|---|---|---|---|---|---|
| 95% | 20.6 | 15.4 | 20.0 | 15.3 | **25.2** | 1.64 |
| **90%** | **21.4** | **15.8** | **20.4** | **19.9** | **28.0** | **1.41** |
| 80% | 22.9 | 16.6 | 21.3 | 22.1 | **33.7** | 1.53 |
| 70% | 24.5 | 17.4 | 22.1 | 24.2 | **39.2** | 1.62 |

⚠ **이 표는 2026-08-13에 바뀌었다. 그 전 값은 반복 하나 위에 있었다.**

| | 반복 1회 | 반복 2회 | 움직임 |
|---|---|---|---|
| llm-d 90% | 18.7 | **19.9** | **+1.2** |
| llm-d 95% | 12.2 | **15.3** | **+3.1** |
| llm-d 80% / 70% | 22.7 / 25.8 | 22.1 / 24.2 | −0.6 / −1.6 |
| FluidServe 90% | 28.1 | 28.0 | −0.1 |
| vLLM router · Llumnix SLO · PolyServe 90% | 21.4 / 20.4 / 15.8 | 21.4 / 20.4 / 15.8 | **0.0** |

**움직인 것은 llm-d 하나다.** 그 arm이 90% 선을 얕게 지나가기 때문이다 — 20 req/s에서 두
반복 평균이 89.9%로 선에 거의 정확히 걸쳐 있고 그 칸의 반복 폭이 2.6점이다. 나머지 셋은 선을
가파르게 지나가서(PolyServe는 15 req/s 99.9%에서 20 req/s 37.0%로 떨어진다) 반복이 늘어도
교차점이 그대로다. **교차점이 반복에 민감한지는 그 arm이 기준선을 어떻게 지나가느냐로 정해지고,
곡선을 보지 않으면 알 수 없다.**

⚠ **그래서 "기준선을 어디에 그어도 비가 1.49~1.52"라고 쓸 수 없게 됐다.** 지금은
**1.41~1.64**이고, 95%에서 가장 크고 90%에서 가장 작다. 쓸 수 있는 것은 **"네 기준 전부에서
FluidServe가 1.4배 이상 앞선다"**까지다.

⚠ **llm-d의 반복 폭이 포화 구간에서 유별나게 크다** — 25 req/s에서 12.0점, 35 req/s에서
15.1점이다(같은 구간 FluidServe는 1.1과 3.9, PolyServe·Llumnix SLO는 0.1 이하).
**그 구간의 llm-d 값을 한 번 재고 인용하면 안 된다.**

**캡션에 반드시 들어가야 하는 것 넷** (스크립트 docstring이 정본):

1. **arm이 둘이다.** PolyServe와 Llumnix SLO는 이 워크로드의 정적 조건이 없다. 5 baseline
   sweep이 채운다(격자는 EXP-70이 정했다).
2. **10~25 req/s가 1반복**이고 90% 통과가 FluidServe는 25~35, llm-d는 20~25 사이에서
   보간되므로 **두 값 다 1반복 점을 가로지른다.**
3. **swe를 llm-d만 m1f로 채점**한다(전체 시간 예산을 표현 못 해서). 채점 규칙 자체는 양쪽 다
   전체 30초다.
4. **예측선이 없다.** `motivation_capacity_is_a_policy.png`의 점선은 EXP-55에서 오는데
   그것은 수정 이전 워크로드다. 되살리려면 EXP-55 재측정이 필요하다.

⚠ **`motivation_capacity_is_a_policy_llmd.png`(수정 이전 워크로드)의 45.1 / 45.8과 같은 표에
넣으면 안 된다.** 거기서는 llm-d가 위인데, 그것은 다른 워크로드다.

---

## `motivation_four_panels.pdf` — 네 컨트롤플레인의 throughput / goodput / 달성률

**스크립트**: `fig_motivation_four_panels.py`. `--no-ours`를 주면 FluidServe를 빼고
`motivation_four_panels_noours.pdf`를 쓴다.
**크기**: 7.0 × 1.62 in, `figure*`에 `width=\textwidth`로 넣는다(스케일 1.0).

**무엇을 주장하는 그림인가.** 같은 네 엔진에서 컨트롤플레인만 바꾸면 **엔진이 만드는 토큰의
양보다 그중 규칙 안에 도착하는 양이 훨씬 크게 갈린다.** 45 req/s에서 총 출력은
6,847~13,077 tok/s(1.9배 폭)이고 goodput은 1,325~11,836(**8.9배 폭**)이다.

**데이터** (수정 후 워크로드, 정적 sweep 8 rate, 총 40조건):

| arm | 글롭 | 반복 |
|---|---|---|
| FluidServe (`fspfx`) | `*exp68s*`, `*exp68r*`, `*exp69*`, `*exp70*`의 `_fspfx_m1_rpm_*` | 35~70은 **2**, 10~25는 **1** |
| llm-d (`llmdslo`) | `*exp68s*`, `*exp68r*`, `*exp70*`의 `_llmdslo_m1f_rpm_*` | 35~70은 **2**, 10~25는 **1** |
| Llumnix SLO (`slo`) | `*exp72r1_slo_m1f_rpm_*` | 전 구간 **1** |
| PolyServe | `*exp72r1_polyserve_m1_rpm_*` | 전 구간 **1** |

**그려지는 값** (평균; 괄호는 두 반복이 있는 칸의 폭):

| req/s | | FluidServe | llm-d | Llumnix SLO | PolyServe |
|---|---|---|---|---|---|
| 20 | 총 출력 / goodput / offered | 9,095 / 9,040 / 99.6 | 8,762 / 8,383 / 88.7 | 9,106 / 8,994 / 95.4 | 9,017 / 4,159 / 39.9 |
| 45 | | 13,077 / 11,836 / 59.4 | 6,847 / 6,298 / 24.9 | 8,045 / 2,132 / 6.4 | 10,548 / 1,325 / 9.6 |
| 70 | | 13,952 / 11,897 / 38.2 | 7,102 / 6,663 / 17.7 | 8,825 / 747 / 1.3 | 8,443 / 424 / 1.2 |

⚠ **축 이름에 `variance`를 쓰면 안 된다.** 원래 통계의 이름이 total **variation** 거리라서
줄여 쓴 것처럼 보이는데, **이 값은 분산이 아니다** — 2차 모멘트가 아니고 단위가 요청 비율이지
비율의 제곱이 아니다. 그렇게 적으면 차이를 아는 독자가 "무엇의, 무엇에 대한 분산이냐"에서
멈춘다. **`deviation`은 기준에서 벗어난 정도를 가리키는 평범한 말이라 되묻지 않는다.**

⚠ **(a)의 축 이름이 `(normalized)`이므로 무엇으로 정규화했는지는 캡션이 말해야 한다** —
그 창의 최대값이다. 그림 안에서는 위쪽 점선이 1.0이라는 것이 그 말을 대신한다.

**캡션이 반드시 담아야 하는 것 넷.**

1. **음영이 없는 곡선이 더 정확한 것이 아니다.** 음영은 두 반복의 min..max이고, PolyServe와
   Llumnix SLO는 전 구간 1반복이라 음영이 아예 없다. **덜 정확한 쪽에 띠가 없다.**
2. **Llumnix SLO와 llm-d는 `m1f` 워크로드 설정을 받는다** — 두 정책 모두 전체 시간 예산을
   표현할 수 없어서다. 채점은 넷 다 전체 30초다.
3. **(a)를 "throughput은 정책과 무관하게 유지된다"로 읽으면 안 된다.** PolyServe는 35 req/s의
   10,628에서 70의 8,443으로 내려가고, llm-d는 25의 8,985에서 45의 6,847로 내려간다(그때
   거절률 71.5%). FluidServe만 단조 증가한다.
4. **goodput의 측정 구간은 0부터 마지막 도착까지다.** `exp23_rate_sweep.py`는 첫 도착부터
   재므로 이 trace에서 약 1.65% 크게 나온다.

**다시 만들려면**: `python3 paper_figures/fig_motivation_four_panels.py`

⚠ **`motivation_throughput_vs_goodput.pdf`와 같은 자리에 쓰지 않는다.** 그것은 수정 이전
워크로드의 세 정책이고, 그 워크로드는 부하 생성기가 모든 프롬프트를 정확히 12번씩 보내서
엔진 prefix hit rate가 83~86%로 부풀어 있던 상태다(고친 뒤 28.9%).

---

## `three_way_split.pdf` — 도착한 요청에 무슨 일이 있었나

**스크립트**: `fig_three_way_split.py`. **크기**: 3.335 × 1.75 in, 단일 열,
`width=\columnwidth`.

**무엇을 주장하는 그림인가.** 달성률과 거절률을 따로 실으면 읽는 사람이 뺄셈을 해야 하고,
**그 뺄셈이 두 실패를 가르는 자리다.** 막대 하나가 결과를 아는 도착 요청 전부이고, 셋으로
나뉜다 — 규칙 안에 완료 / 받았지만 못 지킴 / 거절. 첫 조각이 offered 달성률이라 논문의 다른
모든 달성률 수치와 맞는다.

| | 규칙 안 / 받았지만 못 지킴 / 거절 | 결과를 모름 | n |
|---|---|---|---|
| **한 시간 trace** | | | |
| FluidServe | 81.0 / **3.2** / 15.8 | 0.1% | 2 |
| llm-d | 56.7 / **4.6** / 38.7 | 0.1% | 2 |
| Llumnix SLO | 37.6 / **31.1** / 31.3 | 0.1% | 2 |
| PolyServe | 11.2 / **88.8** / **0.0** | **7.4%** | 2 |
| **정적 45 req/s** | | | |
| FluidServe | 59.4 / 4.5 / 36.0 | 1.7% | 2 |
| llm-d | 24.9 / 3.1 / 71.9 | 0.5% | 2 |
| Llumnix SLO | 6.4 / **20.9** / 72.8 | 2.5% | 1 |
| PolyServe | 9.6 / **90.4** / 0.0 | **35.9%** | 1 |

**캡션이 반드시 담아야 하는 것 셋.**

1. **막대에서 빠진 것이 있다** — run이 끝날 때 아직 처리 중이던 요청은 결과를 알 수 없어
   `attain()`이 두 분모에서 모두 빼고, 이 그림도 같게 뺀다. 그 비율이 한 시간에서 셋은
   0.1%인데 PolyServe만 7.4%이고, **정적 45 req/s의 PolyServe에서는 35.9%다.** 그 요청들은
   깊은 큐 뒤에 있던 것이므로 **빼는 것이 그 arm에 유리하다.**
2. **반복 횟수가 고르지 않다** — 한 시간은 네 arm 모두 2반복, 정적 45는 FluidServe와 llm-d가
   2반복이고 나머지 둘이 1반복이다.
3. **Llumnix SLO와 llm-d는 `m1f` 워크로드 설정을 받는다.** 채점은 넷 다 swe에 대해 전체
   30초다.

**다시 만들려면**: `python3 paper_figures/fig_three_way_split.py`

---

## `azure_rate_and_mix.pdf` — 도착률과 구성이 둘 다 움직이고, 서로를 예측하지 못한다

**스크립트**: `fig_azure_rate_and_mix.py`. **크기**: 3.335 × 2.45 in, 단일 열,
`width=\columnwidth`.

**무엇을 주장하는 그림인가.** `azure_trace_shape.pdf`는 도착률만 보여 주는데, motivation의
주장은 **도착률이 변하고 그 구성도 변한다**는 두 부분이다. 이 그림이 같은 나흘 창에 둘을
같이 놓는다. 구성이 변한다는 것이 **정적 클래스 파티션을 틀리게 만드는 것**이다 — 마루
도착률에 맞춰 함대를 준비해도 그때의 구성이 골에서의 구성이 아니다.

| 패널 | 무엇 | 실측 (Azure LLM Inference 2024, 나흘 창, 10분 구간) |
|---|---|---|
| (a) | 도착률 (정규화 — 그 창의 최대값으로 나눈다) | 골이 마루의 **17.3%**, 마루/골 **5.8배** |
| (b) | **Request mixture deviation** — 도착 요청 중 평균 구성과 어긋난 비율 | 중앙값 **11.5%**, p95 34.0%, **최대 40.2%** |

### (b)가 정확히 무엇인가

**그 구간의 구성과 창 평균 구성 사이의 total variation 거리**이고, 퍼센트로 적는다.
total variation에는 평범한 읽는 법이 하나 있고 그것이 이 값을 그릴 이유다 —
**지금 도착한 요청 중 몇 %가 클래스를 바꿔야 평균 구성과 같아지는가**이다(클래스별로 보면
지금 과잉인 클래스들의 초과분 합이다). **40%라고 읽히면 다섯 중 둘이 평균 구성에는 자리가
없는 클래스라는 뜻이고, 그 다섯 중 둘이 곧 평균에 맞춰 크기를 정한 파티션이 잘못 놓는
요청이다.**

기준을 평균으로 삼는 이유는 **고정된 구성을 하나 골라야 하는 배포가 고를 자연스러운 값**이기
때문이다. ⚠ **오차를 최소로 만드는 고정 구성이라고 주장하지는 않는다** — 그것은 다른
통계이고 이 주장에는 필요 없다.

### 구성 분해가 아니라 이 값을 그리는 이유 셋

1. **정적 파티션이 틀리는 양이 바로 이 값이다.** 평균 구성에 맞춰 크기를 정한 파티션은 매
   순간 정확히 이만큼의 요청을 잘못 놓는다. **y축이 trace의 성질이 아니라 그 설계가 지는
   오차다.**
2. **클래스 개수에 안 걸린다.** Azure 공개본은 요청 유형이 **둘**(대화형, 코드)이고 **우리
   워크로드는 셋**이다. 둘로 나눈 그림을 그리면 읽는 사람이 그것을 우리 셋에 대응시키려
   하는데, **그 대응은 존재하지 않는다** — 그 공개본에 deepresearch에 해당하는 것이 없다.
   바꿔야 하는 요청의 비율은 양쪽에서 같은 뜻이다.
3. **주장이 "믹스가 움직인다"이지 "코드 쪽으로 움직인다"가 아니다.** 이 파일의 앞 판은 반대로
   적었다 — 지표가 방향을 잃고 방향이 파티션 크기를 정하는 데 중요하다고. **그 반론은 다른
   주장에 대한 것이다.** 방향은 **어느** 파티션을 들 것인가를 물을 때 중요하고, 이 그림은
   **어떤 고정 파티션도 맞지 않는다**를 세우는 자리라 이 비율이 내용 전부다.

비율 자체를 잃지는 않는다 — 스크립트가 출력하고 아래에 적어 둔다. 코드 요청이 도착의
**3.9% ~ 68.0%**(p5 5.9 / 중앙값 23.2 / p95 61.8, **p95/p5 10.54배**)이고 **나머지 전부가
대화형**이다. 유효 클래스 수 `1/Σs²`는 최소 1.08 / 중앙값 1.55 / 최대 2.00이다.

**캡션이 반드시 담아야 하는 것 넷.**

1. **구성은 부하 수준의 함수가 아니다.** 두 계열의 상관이 **lag 0에서 +0.57**, ±12시간 안의
   어느 lag에서도 최대 **+0.63**이다. **부하 수준이 구성 분산의 약 40%만 설명한다.**
   ⚠ **"구성이 도착률보다 늦게 정점"이라고 쓰면 안 된다** — 재는 방법 둘이 어긋난다(하루별
   최대끼리는 구성이 0.8~8.3시간 늦고, 교차상관은 구성이 1.5시간 앞선다). 도착률의 하루별
   최대는 뾰족한 1분이라 안정적인 통계가 아니다.
2. **두 패널 다 원본 trace이고 우리가 재생하는 trace가 아니다.** 재생 trace의 도착률은 분위수
   변환을 우리가 고른 대역에 씌운 것이라 마루/골 비가 우리가 정한 값이고, 재생 trace의 구성은
   **세 클래스짜리 합성 스케줄**이다. **둘 다 그리면 순환이다.**
3. **Azure에는 요청 유형이 둘뿐이고 deepresearch에 해당하는 것이 없다.** 그러므로 이 그림은
   **구성이 움직인다는 근거이지, 우리가 고른 수준(10:2:1)이나 세 클래스 분해의 근거가
   아니다**(`motivation.md` §7.3.1의 한계와 같다).
4. **(b)의 값은 이 나흘 창의 평균 구성을 기준으로 한 것이다.** 다른 창을 잡으면 기준이 바뀌고
   값도 바뀐다. `traces/azure/mix_over_time.txt`의 168시간 통계와 같은 수가 아니다.

**색.** (b)는 채도를 낮춘 테라코타 `#b56349`이고, **가까운 두 색을 일부러 피한 것**이다.
`paper_style.ARM_COLOR`가 `#d62728`을 PolyServe에, `#ff7f0e`를 ablation arm 하나에 묶어
두었고 `exp27_figures`는 주황을 **deepresearch 클래스**에 쓴다. 둘 중 아무거나 쓰면 이 패널이
정책이나 클래스를 그린 것처럼 읽히는데 **이 그림은 워크로드의 성질이다.** 그리고 채도를 낮춘
이유는 (a)의 파랑 옆에 놓기 위해서다 — 처음에 PolyServe의 빨강을 그대로 썼더니 아래 패널이
위 패널에서 눈을 빼앗았다.

**다시 만들려면**: `python3 paper_figures/fig_azure_rate_and_mix.py`
(창은 `traces/dynamic/canonical/dyn60_azure4d.plan.json`에서 읽으므로 trace를 다시 만들면
그림도 따라간다.)

---

## `mix_shift.pdf` — 믹스 이동은 도착률이 보고하지 않는 용량 변화다

**주장**: 요청 개수 기준 클래스 비율이 chat 93%에서 균등으로 옮겨가면, **도착률이 그대로여도**
같은 도착률에서의 SLO 달성률이 FluidServe 34.6점, llm-d 48.8점 내려간다.

**데이터**: EXP-93, 4 run = 2 arm(`fspfx`, `llmdslo`) × **2반복**, trace
`traces/dynamic/canonical/dyn60_shift_m2Am1B_b1045`. 전부 2026-08-23, 바이너리 `89d9593b`,
엔진 stock, 게이트웨이 `GOMAXPROCS=16`. **한 세션이다.**
채점은 `analysis_scripts/request_level/exp93_mix_shift.py`를 import해서 쓴다(재구현 아님) —
로더 수정이 그림에 도달하게 하기 위해서다. 표 전문 `results/aggregate_analysis/exp93/report.md`.

**패널**

- **(a)** 한 시간 시계열. 회색 면이 **도착률**이고, 세로 점선이 믹스 구간 경계, 위 숫자가
  그 구간의 chat 비중이다. 도착률을 같이 그린 이유는 그것이 **통제**이기 때문이다 — 두 arm과
  두 trace의 도착 시각이 동일하므로, **구간 경계에서 수준이 이동하는 것은 믹스이고 구간 안에서
  출렁이는 것은 도착률**이라는 것을 읽는 사람이 직접 확인할 수 있다. 곡선은 60초 창을 세 칸
  이동평균한 것이다.
- **(b)** 같은 run을 rate sweep으로 다시 읽은 것. 창 하나가 그 창이 본 도착률에서의 관측
  하나이므로 **chat 93% 구간과 균등 구간을 같은 도착률에서 비교**할 수 있다. 실선과 점선의
  세로 간격이 믹스가 옮긴 용량이다. **경계에 걸친 창은 양쪽에서 제외**했다 — 두 믹스를 담은
  창은 어느 쪽의 관측도 아니다.

**표와 대조했다.** (b)의 여섯 점이 `report.md`의 H1 표와 일치한다(FluidServe 균등
83.0 / 75.8 / 69.1 / 59.5 / 50.5 / 47.3, llm-d 균등 65.4 / 52.0 / 41.7 / 35.6 / 29.6 / 27.7).
(a)의 첫 구간에서 FluidServe가 93까지 내려가는 것이 창별 최솟값 91.7과 맞는다.
⚠ **작게 렌더해서 읽으면 (a)를 잘못 읽는다** — y축 상한이 118이라 100 눈금이 축 위쪽에 있어서,
FluidServe의 93이 70대로 보인다. 300 dpi로 왼쪽 패널만 잘라서 확인했다.

**캡션에 반드시 들어가야 하는 것 넷**

1. **arm당 반복 2회**, 한 세션. `fspfx` 반복 폭 0.4점, `llmdslo` 0.9점.
2. **offered 분모** — 도착한 모든 요청이 분모이고 거절은 위반이다. 거절률은 FluidServe 18.9%,
   llm-d 45.6%.
3. **stress test이지 현실적 워크로드가 아니다.** 믹스 이동 폭이 Azure 2024가 뒷받침하는
   범위(conv/code 비 4.71배)를 넘는다.
4. **균등 구간에서 swe의 prompt 재사용이 부풀려져 있다** — transcript가 1,500건이라 그 구간에서
   사본 간격이 약 2.7분이고 스케줄러 prefix 색인 보유 시간(약 2.8분) 안이다. 이것은 그 구간을
   **실제보다 좋게** 만들므로 **그림이 보이는 격차는 참값의 하한이다.**

**이 그림이 말하지 않는 것**

- **s0와 s3의 무릎(약 48, 약 20 req/s)은 측정값이 아니다** — m1의 측정된 무릎 28.0 req/s를
  요청당 입력 토큰 수에 반비례로 환산한 추정치다.
- **(a)의 마지막 구간은 믹스와 도착률이 함께 나빠진다** — chat 60%이면서 그 구간의 도착률
  중앙값이 35.8 req/s로 네 구간 중 가장 높다. **그 구간의 하락을 믹스만으로 돌리면 안 된다.**
  믹스만 분리해서 보는 것은 (b)이고, 그것이 (b)가 있는 이유다.
- **선호(class affinity)를 끈 조건은 이 그림에 없다.** 네 run 모두 배포 기본값(켬)이다.

**다시 만들려면**: `python3 paper_figures/fig_mix_shift.py`

---

## `motivation_throughput_vs_goodput_4panel.pdf` — 같은 측정을 네 패널로

`fig_motivation_throughput_goodput.py`의 `build_four()`가 그린다. 세 패널판
(`motivation_throughput_vs_goodput.pdf`)과 **같은 `collect()` 통과분에서 나오므로 채점이
같다** — `EXP22.load_run`과 `EXP22.per_request`가 같은 파일을 같은 규칙으로 읽고, 이 함수는
모아 둔 다섯 양 중 넷을 골라 순서를 정할 뿐이다. 같은 arm·같은 도착률의 값을 두 그림에서
읽으면 같은 수가 나온다.

| 패널 | 무엇 | y축 |
|---|---|---|
| (a) Goodput Tokens | 자기 규칙 안에서 완료된 요청이 만든 출력 토큰 | Tokens/s (b와 공유) |
| (b) Throughput | 엔진이 만든 출력 토큰 전부 | Tokens/s (a와 공유) |
| (c) Request SLO (admitted) | 정책이 **받아들인** 요청을 분모로 한 달성률, 요청 단위 | % (d와 공유) |
| (d) Rejection rate | 도착한 요청 중 거절된 비율 | % (c와 공유) |

**왜 이 순서인가.** (a)가 이 그림이 주장하는 양이고 (b)가 바로 옆에서 같은 y축을 쓰므로,
엔진이 만든 토큰의 양은 정책 간에 비슷한데 그중 값을 하는 부분은 자릿수가 다르다는 것이
축을 다시 재지 않고 보인다. (c)와 (d)는 같은 질문을 토큰이 아니라 **요청 수**로 센 것이고,
둘 다 같은 도착 집합에 대한 백분율이라 역시 축을 공유한다.

**(c)와 (d)는 반드시 같이 읽는다.** (c)는 받아들인 것만 분모로 하므로 **거의 아무것도
받지 않는 정책에 최고점을 준다.** (d)가 그 옆에 있는 이유가 그것이고, 세 패널판이 못 하는
짝이다 — 거기서는 둘째 분모가 점선 겹침으로 실려서 거절량을 두 곡선의 간격으로 추론해야 한다.

⚠ **(d)에 곡선이 둘뿐인데 범례는 넷이다.** PolyServe와 vLLM router는 admission control이
없어서 여덟 도착률 전부에서 거절률이 0.0%다. 둘을 그리면 x축 위에서 서로 겹쳐 마지막에
그린 것만 색이 보이고, 읽는 사람은 다른 arm이 0인지 빠진 것인지 구별할 수 없다. 그래서
그리지 않는다 — **캡션이 "그 둘은 어느 도착률에서도 거절하지 않는다"를 반드시 적어야 하고,
그것은 데이터가 없는 것이 아니라 정책의 성질이다.**

**x축 눈금은 실제로 측정한 도착률이다.** 네 패널을 7.0인치에 놓으면 패널당 그리는 폭이 약
1.4인치라 8 pt로 일곱 개 라벨이 안 들어간다. 라벨을 다는 것은 10 / 25 / 45 / 70이고 나머지
측정 도착률(15 / 20 / 35 / 55)은 라벨 없는 minor tick이다. 세 패널판은 10부터 70까지 10
간격으로 라벨을 다는데 그중 넷(30 / 40 / 50 / 60)은 **측정한 적이 없는 값**이고, 이 그림은
그것을 만들지 않는다.

### ⚠ 이 그림만 pinned 사본에서 그린다 — 선택이 형제 그림들과 다르다

**`results/`가 아니라 `paper_experiment/static_sweep_clean_2026-08/data/`에서 읽는다.**
2026-08-26 디스크 정리가 manifest가 참조하는 run만 남기고 나머지를 비웠는데, EXP-86 조건들은
**`results/` 쪽 사본이 비워지고 pinned 사본만 남았다** — 디렉토리 24개가 아직 있고 안이
비어 있다. 그래서 `results/`를 glob하면 대부분의 도착률에서 반복이 둘이 아니라 하나가 되고,
**세 칸은 아예 사라진다**(vLLM 45 req/s, PolyServe 70, Llumnix SLO 55). 빠지는 것이 조용해서
곡선이 짧아진 것으로만 보인다. llm-d는 EXP-82라 정리 대상이 아니었고 영향이 없다.

`pinned_arms()`가 arm 이름·색·마커·순서를 그대로 두고 **run glob만 바꾼다. 채점은 하나도
바뀌지 않는다** — 같은 `load_run`이 같은 파일을 아직 갖고 있는 디렉토리에서 읽는다.

⚠ **그러므로 이 그림과 세 패널 형제들은 지금 같은 집합에서 그려지지 않았다.** 형제들은
아직 `results/` 선택을 쓴다. **한쪽에서 읽은 수를 다른 쪽에서 읽은 수 옆에 인용하면 안
된다.** 형제들의 선택도 같이 pinned로 옮기면 이 경고는 없어진다.

### 데이터

`paper_experiment/static_sweep_clean_2026-08/manifest.tsv`가 정본이다 — arm마다 **16 run**
(8 도착률 × **2반복**), 조건당 8분, 엔진 넷, 각 `metrics.csv`의 sha256이 함께 적혀 있다.
네 arm 전부 8/8 칸이 n=2다. **오차막대는 그리지 않았다** — 값은 두 반복의 평균이고, 이
워크로드의 세션 간 이동은 최대 4.6점으로 측정돼 있다.

swe는 설정이 둘이고 채점은 하나다: llm-d와 Llumnix SLO는 전체 시간 예산을 표현할 수단이
없어 `m1f` 워크로드 설정을 쓰고, PolyServe와 vLLM router는 `m1`을 쓰며, **네 arm 모두 전체
시간 30초 규칙으로 채점된다.**

⚠ **EXP-108이 끝나면 이 그림은 그대로 쓸 수 없다.** 그 실험이 swe의 약속을 전체 시간 30초에서
per-token(첫토큰 7초 + 토큰당 75 ms)으로 바꿔 네 arm을 다시 재고 있으므로, **t75로 채점한
어떤 열도 위 표와 같은 표에 못 들어간다.**

### 그려진 값 (평균, n=2)

| arm | req/s | 10 | 15 | 20 | 25 | 35 | 45 | 55 | 70 |
|---|---|---|---|---|---|---|---|---|---|
| vLLM | goodput (tok/s) | 5170 | 7717 | 10045 | 9000 | 1371 | 778 | 762 | 579 |
| PolyServe | goodput (tok/s) | 5153 | 7689 | 4711 | 4647 | 2269 | 1516 | 592 | 549 |
| Llumnix SLO | goodput (tok/s) | 5137 | 7664 | 9904 | 4602 | 3030 | 2489 | 1538 | 795 |
| llm-d | goodput (tok/s) | 4819 | 7355 | 9448 | 9148 | 8311 | 6601 | 6949 | 7565 |
| vLLM | throughput (tok/s) | 5218 | 7829 | 10310 | 12691 | 14291 | 14303 | 13981 | 13191 |
| PolyServe | throughput (tok/s) | 5218 | 7825 | 10220 | 11194 | 11876 | 11750 | 11119 | 9609 |
| Llumnix SLO | throughput (tok/s) | 5201 | 7825 | 10293 | 12528 | 11738 | 9033 | 9123 | 9979 |
| llm-d | throughput (tok/s) | 4918 | 7774 | 10092 | 10156 | 9178 | 7259 | 7857 | 8322 |
| vLLM | admitted (%) | 100.0 | 99.8 | 98.9 | 70.9 | 7.9 | 3.8 | 3.4 | 2.6 |
| PolyServe | admitted (%) | 100.0 | 99.9 | 41.8 | 30.7 | 16.1 | 9.3 | 2.0 | 1.5 |
| Llumnix SLO | admitted (%) | 99.6 | 98.0 | 94.8 | 33.2 | 20.4 | 25.0 | 15.5 | 7.5 |
| llm-d | admitted (%) | 98.7 | 95.0 | 93.3 | 89.5 | 89.3 | 89.4 | 86.9 | 91.2 |
| vLLM | rejected (%) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PolyServe | rejected (%) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Llumnix SLO | rejected (%) | 0.0 | 0.0 | 0.0 | 0.0 | 39.4 | 71.1 | 78.4 | 81.9 |
| llm-d | rejected (%) | 6.1 | 0.2 | 1.9 | 23.2 | 52.5 | 76.5 | 77.7 | 81.9 |

### 캡션이 반드시 적어야 하는 것

- **(c)는 admitted 분모다.** (d)와 함께 읽어야 뜻이 있다.
- **PolyServe와 vLLM router는 (d)에 없고, 그것은 두 정책이 어느 도착률에서도 거절하지 않기
  때문이다.**
- **"vLLM"은 PyPI `vllm-router` 패키지의 기본 `cache_aware` 정책**(SGLang model gateway의
  fork)이고 vllm-project/production-stack이 아니다. 같은 이름의 물건이 둘이라 축의 짧은
  라벨이 그 구별을 못 실으므로 캡션이 어느 쪽인지 적어야 한다.
- **거절하지 않는 두 arm은 다른 종류의 사라진 요청을 만들고 그것은 그림에 없다.** admission
  control이 없는 정책은 요청을 거절하는 대신 측정 창이 닫힐 때 미완으로 남기고, 미완 요청은
  결과가 없어 `attain()`이 **두 분모 모두에서** 뺀다. 그 비율이 35 / 45 / 70 req/s에서 vLLM
  router 21.8 / 39.6 / 61.6%, PolyServe 37.9 / 35.9 / 35.7%인 반면 llm-d 1.0 / 0.5 / 0.3%,
  Llumnix SLO 3.8 / 2.5 / 2.2%다. **즉 이 제외는 거절하지 않는 두 arm에 유리하다.**
- FluidServe는 이 그림에 **일부러 없다.** 논문 자신의 시스템이 있어야 성립하는 motivation
  그림은 motivation 그림이 아니다.

**다시 만들려면**: `python3 paper_figures/fig_motivation_throughput_goodput.py`
(이 파일 하나가 형제 그림들도 같이 다시 그린다)

## `motivation_throughput_vs_goodput_4panel_t75.pdf` / `_t75_withfs.pdf` — 다섯 컨트롤플레인, 토큰 단위 누적 deadline 채점

**스크립트**: `fig_motivation_tg_4panel.py`. 한 번 실행하면 둘 다 쓴다 — FluidServe를 뺀 판
(motivation용)과 넣은 판. `--rule`로 여덟 채점 규칙 중 하나를 고를 수 있고 기본은 `q90`.
**크기**: 7.0 × 1.75 in, `figure*`에 `width=\textwidth`(스케일 1.0). 페이지 크기 504 × 126 pt 확인.

| 패널 | 무엇 | y축 |
|---|---|---|
| (a) Goodput tokens | 자기 규칙 안에서 완료된 요청이 만든 출력 토큰 | Tokens/s (b와 공유) |
| (b) Throughput | 엔진이 만든 출력 토큰 전부 | Tokens/s (a와 공유) |
| (c) Request SLO (admitted) | 정책이 **받아들인** 요청을 분모로 한 달성률, 요청 단위 | % |
| (d) Rejection rate | 도착한 요청 중 거절된 비율 | % |

**(c)와 (d)는 반드시 같이 읽는다.** (c)는 받아들인 것만 분모로 하므로 거의 아무것도 받지 않는
정책에 최고점을 준다. 모든 도착을 분모로 한 곡선은 그리지 않는다 — (c)와 (d)가 그것을 이미
담고 있어서 세 번째 표현이 된다.

**x축은 선형이고 라벨만 등간격이다 (2026-09-01 사용자 결정).** 도착률은 양이므로 10에서 15까지의
거리가 55에서 70까지의 거리의 3분의 1로 그려진다 — 등간격 범주형 축은 그 두 단계가 같은 크기라고
주장하게 된다. **라벨은 10 req/s 간격(10·20·…·70)으로 놓는데**, 측정한 도착률
(10·15·20·25·35·45·55·70)은 단계 크기가 셋(5·10·15)이라 그것으로 라벨을 달면 넷이 축 앞 3분의 1에
몰리기 때문이다.
⚠ **그 일곱 라벨 중 넷(30·40·50·60)은 측정한 적이 없는 값이다.** 등간격 축을 위해 치르는 실제
비용이고, 그래서 **측정한 도착률에는 라벨 없는 minor tick을 남기고** 모든 곡선이 측정 지점마다
마커를 갖는다 — 어디가 측정이고 어디가 그 사이를 이은 선인지는 그것으로 읽는다.

⚠ **이름의 `t75`는 장식이 아니다.** `fig_motivation_throughput_goodput.py`가 이미
`motivation_throughput_vs_goodput_4panel.pdf`를 쓰는데, 그것은 이전 정적 sweep의 pinned 사본에서
그려지고 **swe를 전체 시간 30초로 채점**한다. 두 그림의 수치는 서로 옆에 놓을 수 없으므로 파일
이름을 공유하면 안 된다.

### 채점 규칙

토큰당 절반을 **누적 deadline + 관용**으로 판정한다. `a_i`는 i번째 출력 토큰이 제출 시각으로부터
도착한 시각, `T`는 클래스의 첫 토큰 예산, `P`는 토큰당 예산이다.

```
a_1 <= T   그리고   |{i : a_i <= T + (i-1)*P}| / N >= 0.90
```

첫 토큰 조건을 별도 논리곱으로 남기는 이유: 스케줄이 도착 시각에서 시작하면 `T`를 안 쓴 만큼이
디코드로 이월되어, 엔진이 `P`보다 빠르면 요청 하나가 약 `N × (P − 실제 토큰당 시간)`의 첫 토큰
위반을 흡수한다. 누적 deadline 자체는 PolyServe·QoServe·JITServe의 규칙이고 **90% 관용은 그
아홉 편 어디에도 없는 이 프로젝트의 추가**다. 여덟 규칙 전부의 표는
`results/aggregate_analysis/token_deadline_2026-08-31/README.md` §8에 있다.

클래스 예산: chat (5 s, 50 ms) · deepresearch (10 s, 100 ms) · **swe (7 s, 75 ms)**.

### 데이터

| arm | 결과 디렉토리 | 반복 | 정책이 받은 swe 약속 |
|---|---|---|---|
| vLLM-router (`vllmcache`) | `results/*exp77r[12]_vllmcache_m1_rpm_*` | 70 req/s만 **1**, 나머지 2 | 읽지 않는다 |
| PolyServe (`polyservept75`) | `results/*exp108r[12]_polyservept75_t75fair_rpm_*` | 전 구간 **2** | `t75fair`의 `slo.swe.tbt_ms=75` |
| Llumnix SLO (`slot75`) | `results/*exp108r[12]_slot75_t75fair_rpm_*` | 전 구간 **2** | 같음 |
| llm-d (`llmdslot75`) | `results/*exp108r[12]_llmdslot75_t75fair_rpm_*` | 전 구간 **2** | 같음 |
| FluidServe (`fsv3capgnofrct75`) | `results/*exp108r[12]_fsv3capgnofrct75_t75_rpm_*` | 전 구간 **2** | `--fluidserve-class-budgets 25:decode:75` |

띠는 두 반복의 min..max다. **띠가 없는 칸은 반복이 하나**(vLLM-router 70 req/s)이거나 두 반복이
일치하는 칸이다.

⚠ **vLLM-router만 다른 세션이고, 그것은 실제 결합이다.** EXP-108의 arm이 넷이고 거기 없으므로
EXP-77(2026-08-10)에서 가져왔다. **약속을 읽지 않는 유일한 arm이라 다른 약속으로 재채점하는 것이
정당하다** — 라우팅이 prefix cache affinity뿐이라 워크로드 파일의 swe 예산이 무엇이든 그 arm이 본
요청은 같다. 그러나 세션의 나머지는 같지 않다: 3주 앞이고, 같이 그려지는 arm들의 스케줄러
바이너리가 다르며, **그 사이 2026-08-28에 장비가 초기화되고 복원됐다.** 이 워크로드의 세션 간
이동은 최대 4.6점으로 측정돼 있는데 **그 측정은 복원을 사이에 두지 않는다.**

### 그려진 값 (반복 평균, 규칙 `q90`)

⚠ **정본은 그림 옆의 CSV 다** (`motivation_throughput_vs_goodput_4panel_t75[_withfs].csv`,
그림을 그리는 같은 실행이 같은 집계에서 쓴다). 아래 표는 손으로 옮긴 것이라 그림을 다시
그리면 낡는다 — 실제로 2026-09-02 에 llm-d 의 10 req/s 네 칸이 낡은 채로 남아 있었다.
값을 인용할 때는 CSV 를 연다.

| arm | 양 | 10 | 15 | 20 | 25 | 35 | 45 | 55 | 70 |
|---|---|---|---|---|---|---|---|---|---|
| vLLM-router | goodput (tok/s) | 5,168 | 7,704 | 10,073 | 11,826 | 1,946 | 943 | 702 | 225 |
| PolyServe | goodput (tok/s) | 4,890 | 7,207 | 7,785 | 7,890 | 5,654 | 4,907 | 4,764 | 4,362 |
| Llumnix SLO | goodput (tok/s) | 5,150 | 7,693 | 10,038 | 7,681 | 4,095 | 1,158 | 411 | 273 |
| llm-d | goodput (tok/s) | 5,040 | 7,531 | 9,755 | 8,686 | 8,524 | 9,223 | 7,762 | 7,092 |
| FluidServe | goodput (tok/s) | 5,132 | 7,528 | 9,904 | 11,940 | 13,336 | 15,099 | 14,399 | 14,875 |
| vLLM-router | throughput (tok/s) | 5,223 | 7,812 | 10,292 | 12,682 | 14,374 | 14,299 | 14,105 | 13,436 |
| PolyServe | throughput (tok/s) | 5,115 | 7,718 | 10,220 | 12,525 | 14,773 | 15,084 | 15,219 | 14,780 |
| Llumnix SLO | throughput (tok/s) | 5,217 | 7,803 | 10,287 | 12,565 | 11,724 | 8,981 | 9,034 | 9,884 |
| llm-d | throughput (tok/s) | 5,172 | 7,783 | 10,053 | 9,260 | 8,927 | 9,708 | 8,148 | 7,683 |
| FluidServe | throughput (tok/s) | 5,204 | 7,744 | 10,213 | 12,369 | 13,881 | 15,607 | 14,956 | 15,457 |
| vLLM-router | admitted (%) | 100.0 | 100.0 | 100.0 | 98.6 | 14.0 | 6.3 | 4.3 | 1.4 |
| PolyServe | admitted (%) | 99.2 | 95.5 | 86.1 | 74.8 | 52.2 | 44.3 | 42.4 | 37.7 |
| Llumnix SLO | admitted (%) | 100.0 | 100.0 | 100.0 | 72.7 | 35.5 | 13.8 | 5.8 | 3.2 |
| llm-d | admitted (%) | 99.9 | 98.7 | 99.3 | 96.6 | 98.4 | 97.9 | 98.5 | 93.7 |
| FluidServe | admitted (%) | 100.0 | 100.0 | 100.0 | 100.0 | 99.9 | 99.9 | 99.9 | 99.8 |
| vLLM-router | rejected (%) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PolyServe | rejected (%) | 0.0 | 0.0 | 0.0 | 0.4 | 10.5 | 27.6 | 40.5 | 54.7 |
| Llumnix SLO | rejected (%) | 0.0 | 0.0 | 0.0 | 0.0 | 39.2 | 70.8 | 78.5 | 81.9 |
| llm-d | rejected (%) | 0.0 | 0.0 | 2.4 | 32.7 | 54.6 | 61.0 | 75.3 | 82.6 |
| FluidServe | rejected (%) | 0.0 | 0.1 | 0.2 | 2.6 | 22.2 | 30.0 | 46.9 | 56.1 |

### 캡션이 반드시 적어야 하는 것

- **swe가 per-token 형태(첫 토큰 7초 + 토큰당 75 ms)로 채점된다.** 전체 시간 30초로 채점한 어떤
  수치와도 같은 표·같은 문장에 넣지 않는다.
- **(c)는 받아들인 요청만 분모로 한다.** (d)와 함께 읽어야 뜻이 있다.
- **vLLM-router는 어느 도착률에서도 거절하지 않는다.** (d)에서 x축 위에 놓이는 것은 데이터가
  없어서가 아니라 admission control이 없기 때문이고, 그래서 그 arm의 (c)는 모든 도착을 분모로
  한 값과 같다.
- **vLLM-router만 EXP-77(2026-08-10)이고 나머지 넷은 EXP-108(2026-08-31)이다.** 그 사이에 장비
  초기화가 있었고 그 결합은 측정되지 않았다.
- **llm-d의 10 req/s 칸은 두 반복이 갈린다**(거절 0.0% 대 42.0%). 둘 다 `run_health.py`를
  통과했으므로 인프라 고장이 아니라 조건마다 새로 학습되는 예측기의 성질이다. **그 칸의 넓은
  띠가 그것이고 평균선 하나로 읽으면 안 된다.**
- **무릎 아래(10~20 req/s)에서는 다섯 arm이 전부 100 부근이다.** 그 구간의 차이는 순위가 아니라
  천장이다.
- **90% 관용은 FluidServe에 +0.0점, PolyServe에 +2.6점(최대 5.2)을 준다** — 관용이 우리에게
  유리한 잣대라는 반론은 측정으로 닫힌다(같은 README §8.2).

### ⚠ llm-d 의 10 req/s 칸은 EXP-108 의 두 반복이 아니다 (2026-09-02)

**그린 표는 `exp108_paper_per_run.csv` 이고, 그것은 `exp108_per_run.csv` 와 한 칸이 다르다.**
EXP-108 의 llm-d 10 req/s 두 반복이 **거절률 0.0% 와 42.0%** 로 갈렸다 — 같은 워크로드, 도착
요청 수 4,618 대 4,611, 엔진 4/4 인데 offered 달성률이 40 점 넘게 벌어졌다. 나머지 일곱 도착률의
반복 간 편차는 0.1~4.0 점이므로 그 칸에만 있는 현상이다.

**EXP-110 이 그 도착률만 세 번 더 쟀고 세 번 다 거절 0.0% 였다** (offered 97.8 / 99.5 / 98.6,
누적 규칙 99.5 / 99.9 / 100.0, goodput 4,970~5,159 tok/s). 다섯 run 중 넷이 거절 0% 이므로
**42.0% 반복은 다섯 번에 한 번 일어난 outlier** 이고, 원인은 밝히지 못했다.

→ **이 칸은 EXP-110 반복 3 과 5 로 그린다**(사용자 결정, 2026-09-02). 다른 모든 칸과 같이
n=2 를 유지한다. 바뀐 값은 그 칸 하나뿐이고 나머지 31 칸은 바이트 단위로 동일하다:

| | 이전 (EXP-108 반복 1·2) | 지금 (EXP-110 반복 3·5) |
|---|---|---|
| goodput | 3,981 (밴드 2,895~5,067) | **5,040** (밴드 4,990~5,091) |
| throughput | 4,077 (2,975~5,180) | **5,172** (5,169~5,175) |
| admitted | 99.48 (98.95~100.0) | 99.90 (99.80~100.0) |
| 거절률 | 21.0 (0.0~42.0) | **0.0** |

**말하지 않는 것**: 왜 그 한 번이 42% 를 거절했는가. EXP-108 은 학습 서버 로그를 마지막 2,000 줄
(= 약 80 초)만 저장해서 거절이 일어난 1~6 분 구간의 기록이 없고, EXP-110 이 그 계측을 고쳤지만
세 번 다 정상 run 이라 비교 대상이 없다. 전체 경위와 반증 조건은
`experiments/EXP-110_llmd-lowrate-split.md`. 이전 표를 보려면
`fig_motivation_tg_4panel.py --csv <...>/exp108_per_run.csv`.

**다시 만들려면**: `python3 paper_figures/fig_motivation_tg_4panel.py`
(채점 표가 먼저 있어야 한다: `rescore_token_deadline.py --tag exp108 --runs <64개>`,
`--tag exp110_llmd10 --runs <EXP-110 반복 3·5>` 뒤에 두 표를 합쳐
`exp108_paper_per_run.csv` 를 만든다, `--tag vllm77 --runs <EXP-77 vllmcache 15개>`)
