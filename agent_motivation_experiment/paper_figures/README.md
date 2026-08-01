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
