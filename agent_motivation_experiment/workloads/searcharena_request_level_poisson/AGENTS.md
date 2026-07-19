# Search Arena Deep-Research Request-level Poisson Workload

Request-level (flat, open-loop) workload: each task is **one independent
LLM request**, submitted as a Poisson process (λ = requests/sec). No
job/chain concept. The request content is a reconstructed **deep-research
synthesis request** ("K research notes + question → comprehensive
answer") built from the LM-Arena Search Arena dataset.

**Role in the workload suite: the mid-length workload.** Measured input
lengths (Llama-3.1 tokenizer):

| Workload | input mean | input p50/p95 | output mean |
|---|---|---|---|
| chat (`sharegpt_...`) | 674 | — | 404 |
| **this (deep-research)** | **3,230** | **2,636 / 7,089** | model-determined (notes are ~620) |
| SWE (`codingagent_...`) | 21,789 | — | 538 |

(This-workload numbers measured on the actual implementation: 400
assembled requests sampled from the default 60k-spec pool, Llama-3.1
tokenizer; p99 10,197 / max 11,386.)

It fills the length axis between chat and SWE **without a chain
structure** — structurally identical to the chat workload (single-turn,
independent requests), so mix experiments can isolate the input-length
variable from the chain-feedback variable (which the SWE workload
carries).

## Dataset: `lmarena-ai/search-arena-24k` (CC-BY-4.0)

24,069 LM-Arena "search battle" conversations (2026 release; pinned
revision `fac8dcf8`). Each row: user question(s) + two answers from
search-augmented models (Perplexity sonar family, gpt-4o-search-preview,
gemini-2.x grounding) as `messages_a`/`messages_b`, plus per-side
metadata (`llm_trace`, `web_search_trace`, `llm_config`), battle outcome
(`winner`/`judge`), `primary_intent`, `languages`.

Facts established by inspection (2026-07-18/19):

- **78% single-turn** (18,674 rows turn=1); multi-turn drops fast.
- **Multilingual**: English 12,849 (53%), Russian 2,604, Chinese 1,150,
  Korean 592, Japanese 290, … → this workload filters
  `languages == ["English"]` only.
- **No retrieved web-page bodies**: `web_search_trace` holds
  `[citation_number, URL]` pairs only; `llm_trace` (300 rows inspected)
  has no system prompts and no injected search context — just
  user/assistant text. So the dataset **cannot be replayed as served
  requests as-is**: flattened ShareGPT-style, inputs measure mean 699 /
  p50 36 tok — chat-sized, not mid-length.
- What it DOES have: the assistant answers are real **search-grounded
  syntheses with inline citations** (English pool: 33,182 answers >200
  chars from both sides; mean 619 / p50 489 / p90 1,239 tok). These are
  exactly the "research note" texts a deep-research pipeline feeds its
  synthesis step.

## Reconstruction (what a request is)

Since the served-request form of deep research is "question + retrieved
material → synthesize" and the retrieved material is absent, we rebuild
it with the dataset's own search-grounded answers standing in for the
retrieved notes:

```text
[system]  fixed research-synthesis instruction (~80 tok, SYSTEM_PROMPT
          in searcharena.py: answer ONLY from the notes, cite [note n],
          flag conflicts/missing info)
[user]    Research notes:

          [note 1]
          <search-grounded assistant answer from another conversation>
          ...
          [note K]
          <...>

          ---
          Question: <real user question from the dataset>
```

- **K ~ log-uniform int in [k_min, k_max]** (default [2, 12], measured
  E[K]=5.37) — the single length knob. Default measured assembly
  (English pools, actual implementation, n=400):
  **mean 3,230 / p10 1,173 / p50 2,636 / p90 6,166 / p95 7,089 /
  p99 10,197 tok.**
- Notes are sampled **without replacement**, excluding notes from the
  question's own conversation (a question never ships with its own
  recorded answer).
- Question pool: first user turn of `messages_a` (user turns are
  identical across battle sides): 12,849 English questions (mean 105 /
  p50 17 tok).
- **Determinism**: request spec `i` is drawn from
  `random.Random(sample_seed*1_000_003 + i)`; pools are built in parquet
  row order at a pinned revision → the same `request_id` is
  byte-identical across replays, shards, and runs.
- Combination space is `12,849 × C(33,182, K)` → every emitted request
  is unique within any practical run (no prefix-cache aliasing between
  requests; only cycled replays repeat bytes, as in sharegpt).

### Why this is defensible for publication

1. **Precedent**: serving papers construct workloads from datasets as a
   rule (vLLM: ShareGPT/Alpaca sampling; DistServe/Sarathi: length
   distributions). **JitServe (NSDI'26) builds its deep-research
   workload from this same Search Arena dataset** — and since the
   dataset contains no retrieved bodies, their served requests (single
   mean 1,911 / P95 7,573; compound mean 12,223 tok) are necessarily
   reconstructions too.
2. **Length anchor**: our default p95 (7,089) matches JitServe's
   single-request P95 (7,573), and our mean (3.2k) sits inside their
   single→compound range (1.9k–12.2k), near the geometric middle of our
   own chat (0.7k) and SWE (21.8k) workloads (~3.8k).
3. **Content is 100% real data** (real questions, real search-grounded
   answers with citations) — natural token statistics, not synthetic
   filler.
4. **Disclosed limitations**: (a) notes are search-augmented *answers*
   standing in for retrieved *web documents* (more uniform style —
   irrelevant to the engine, which sees token streams); (b) notes are
   sampled across topics, so a request's notes are not semantically
   about its question (also engine-irrelevant; set config
   `language`/intent filters if semantic coherence ever matters);
   (c) this models the *synthesis stage* of deep research, not the full
   multi-round pipeline (a chained variant would overlap the SWE
   workload's role and is intentionally out of scope).

## Single-step / direct — no transcript, no baseline

Same contract as `sharegpt_request_level_poisson`:

- `tau` is **unused**; goodput is judged post-hoc against **absolute SLO
  thresholds**.
- No record step, no transcript, no per-request solo baseline.
- All client-side aborts disabled (`job_timeout_sec=0`,
  `per_call_timeout=None`, `idle_timeout=None`); the HTTP client timeout
  is only a dead-connection safety net.
- In `run_experiment.py`'s `NO_BASELINE_DIR_WORKLOADS`.
- `metrics.csv` rows have `agent == "request"`.

Files:
- `searcharena.py` — dataset download, pool building, spec sampling,
  message assembly (unit-testable pure functions).
- `workload.py` — the adapter.

## Memory/MP design (differs from sharegpt)

Tasks are lightweight **index specs** (`question_idx`, `note_idxs`), not
materialized text. The text pools (~90 MB for English) are loaded once
per load process by `load_dataset`; `run_job` assembles the prompt on
demand via `assemble_messages`. This keeps the 24-process load generator
at ~90 MB/process instead of ~1 GB/process for a fully materialized
60k-request pool. The spec list is what gets sharded
(`dataset[shard_idx::n_shards]`) across load processes — specs are
self-seeded by index, so sharding does not perturb determinism.

## `--workload-config` keys

| Key | Default | Meaning |
|---|---|---|
| `hf_repo` | `lmarena-ai/search-arena-24k` | Hub dataset repo |
| `hf_data_file` | `data/search-arena-chat-24k.parquet` | parquet in the repo |
| `hf_revision` | `fac8dcf86146c8773ef020095c5694c9b2bc98d7` | pinned dataset commit |
| `language` | `English` | keep rows with `languages == [language]` |
| `num_requests` | 60000 | unique request specs before cycling |
| `k_min` / `k_max` | 2 / 12 | notes per request, log-uniform int |
| `note_min_chars` | 200 | drop shorter assistant answers |
| `note_max_chars` | 20000 | drop freak-long answers (~5k tok cap) |
| `question_max_chars` | 4000 | drop freak-long pasted questions |
| `sample_seed` | `--seed` | deterministic spec sampling |

Length tuning: mean input scales ≈ `80 + 105 + E[K]·619` tok; with
log-uniform K, `E[K] ≈ (k_max+1-k_min)/ln((k_max+1)/k_min)`. To match
JitServe's single-request mean (~1.9k) instead of the mid-length default,
use `k_min=1, k_max=6`.

## Invariants

- Requests are single-turn `[system, user]`; output length is
  model-determined (cap via `--max-tokens`).
- `request_id = sa-{index:06d}`; runner task `instance_id` appends a
  `__rNN` replay-cycle suffix.
- Keep `SYSTEM_PROMPT` stable — editing it changes every request's token
  count and breaks cross-run comparability.
- `build_baseline_tasks` works (concurrency-1 pass for absolute-SLO
  threshold picking) but writes no transcript.

## Analysis

Identical pipeline to the sharegpt flavor:

```bash
python analysis_scripts/request_level/parse_request_summary.py results/<run>
python analysis_scripts/request_level/summarize_lambda_sweep.py \
  --run-dirs results/*searcharena_sweep_lambda_* \
  --output-dir results/aggregate_analysis/searcharena_lambda_summary \
  --print-markdown --plot-png
```

Cross-experiment comparisons use the standard arrival-anchored [60s,
340s] window (`plot_slo_vs_throughput.py --steady-anchor arrival
--steady-max-s 360`); see `experiments/README.md` §"Cross-experiment
analysis standard".

## Halo

`run_job` forwards the per-request `halo_*_slo` fields into `make_llm`
when `context.halo_enabled` is True and reuses `invoke_with_tracking`,
so HTTP-400 reject detection and `metrics.csv` propagation work for
free.
