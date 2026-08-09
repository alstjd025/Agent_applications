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

## `intro_capacity.pdf` / `intro_capacity_curves.pdf` (2026-08-09)

**주장**: 도착한 요청의 90% 이상이 자기 지연 규칙을 지키는 최대 도착률이
**FluidServe v0.2 28.1 req/s, llm-d 18.7 req/s로 1.50배**다. 거절은 위반으로 세므로
전부 거절해서 이 값을 살 수 없다. `paper_figures/fig_intro_capacity.py`.

**데이터**: 2026-08-08 워크로드 수정 **이후**의 정적 조건만. arm당 8 rate
(10/15/20/25/35/45/55/70 req/s). EXP-70(10~25, 1반복) + EXP-68·69(35~70, 2반복).

| 포화 기준 | FluidServe v0.2 | llm-d | 비 |
|---|---|---|---|
| 95% | 25.4 | 12.2 | 2.08 |
| **90%** | **28.1** | **18.7** | **1.50** |
| 80% | 33.7 | 22.7 | 1.49 |
| 70% | 39.2 | 25.8 | 1.52 |

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
