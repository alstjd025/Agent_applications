"""Search Arena dataset loading + deep-research request assembly.

Kept separate from `workload.py` so the pool-building / assembly logic is
easy to unit test without constructing a runner.

Dataset: `lmarena-ai/search-arena-24k` (CC-BY-4.0) — 24,069 LM-Arena
"search battle" conversations answered by search-augmented models
(Perplexity sonar family, gpt-4o-search-preview, gemini-grounding). Each
row has the user question(s) and two search-grounded assistant answers
(`messages_a` / `messages_b`) with inline citations. The dataset does
**NOT** contain the retrieved web-page bodies (`web_search_trace` holds
citation URLs only), so it cannot be replayed as served requests as-is —
flattened, its inputs are chat-sized (mean ~0.7k tok, p50 36).

Reconstruction (this module): a deep-research *synthesis* request is
rebuilt as

    system  (fixed deep-research analyst instruction, ~910 tok)
    user    "Research notes: [note 1..K]  ---  Question: <q>"

where the K "research notes" are search-grounded assistant answers drawn
from *other* conversations in the same dataset (real search-augmented
text with citations, mean ~620 tok each) and <q> is a real user
question. K is sampled log-uniform in [k_min, k_max]; with the defaults
(2..12) the assembled input measures mean 4,055 / p95 7,914 tok on the
Llama-3.1 tokenizer — between the chat (0.7k) and SWE (21.8k) workloads,
and matching the JitServe (NSDI'26) deep-research request scale (single
mean 1,911 / P95 7,573; compound mean 12,223), which also builds its
deep-research workload from Search Arena.

The ~910-tok `SYSTEM_PROMPT` is a realistic fixed deep-research analyst
instruction block (role, grounding rules, report structure). It is the
SAME bytes on every request, so vLLM's prefix cache serves it as a
shared hit; only the per-request "notes + question" tail (mean ~3,145
tok) is new prefill. This mirrors real deep-research serving ("large
fixed system block cached + unique retrieved material re-prefilled")
instead of the earlier tiny-prompt shape where nearly nothing was
cacheable.

Everything is deterministic given (dataset revision, filters,
sample_seed): pools are built in parquet row order and each request spec
is drawn from `random.Random(sample_seed * 1_000_003 + index)`.
"""

import math
import random
from typing import Optional

DEFAULT_HF_REPO = "lmarena-ai/search-arena-24k"
DEFAULT_HF_DATA_FILE = "data/search-arena-chat-24k.parquet"
# Pinned dataset revision (reproducible download across machines).
DEFAULT_HF_REVISION = "fac8dcf86146c8773ef020095c5694c9b2bc98d7"

DEFAULT_LANGUAGE = "English"
DEFAULT_NUM_REQUESTS = 60000
DEFAULT_K_MIN = 2
DEFAULT_K_MAX = 12
DEFAULT_NOTE_MIN_CHARS = 200
DEFAULT_NOTE_MAX_CHARS = 20000
DEFAULT_QUESTION_MAX_CHARS = 4000

# Fixed deep-research analyst instruction (~910 tok). It is byte-identical
# on every request, so the engine serves it from the prefix cache (shared
# hit) and only the per-request notes+question tail is new prefill — a
# realistic deep-research serving profile. Changing it changes every
# request's token count AND the cached-prefix size — keep edits intentional.
SYSTEM_PROMPT = (
    "You are Atlas, a senior deep-research analyst. You produce rigorous, "
    "well-sourced research reports for a demanding professional audience "
    "(analysts, engineers, decision-makers) who rely on your synthesis to "
    "act. Your defining trait is intellectual honesty: you never overstate "
    "what the evidence supports, and you are explicit about uncertainty.\n"
    "\n"
    "OPERATING CONTEXT\n"
    "A retrieval subsystem has already run one or more web searches for the "
    "user's question and collected the results as a set of numbered research "
    "notes, provided in the user message under 'Research notes'. Each note is "
    "an independent excerpt gathered from a distinct source; notes may "
    "overlap, may be written from different viewpoints, may be dated, and may "
    "occasionally contradict one another. The notes are your ONLY admissible "
    "evidence. You have no other tools available for this turn and cannot "
    "issue further searches; work strictly from what the notes contain.\n"
    "\n"
    "GROUNDING RULES\n"
    "1. Use ONLY information found in the research notes. Do not introduce "
    "outside facts, figures, dates, names, or events, even if you believe you "
    "know them. If a needed fact is absent, say so rather than filling the "
    "gap.\n"
    "2. Attribute every substantive claim to its supporting note(s) with an "
    "inline citation of the form [note 3], or [note 2; note 5] when several "
    "notes agree. Place the citation immediately after the sentence or clause "
    "it supports.\n"
    "3. When notes conflict, do not silently pick a side. Surface the "
    "disagreement, attribute each position to its note(s), and, where the "
    "notes give enough basis (recency, specificity, source type), briefly "
    "explain which is more credible and why. Otherwise present both and label "
    "the point unresolved.\n"
    "4. Distinguish established fact from speculation, forecast, or opinion "
    "expressed in the notes, and carry that distinction into your report.\n"
    "5. Never fabricate citations. A [note N] marker must correspond to a "
    "note that genuinely supports the claim.\n"
    "6. Treat quantities carefully. Report numbers, units, dates, and ranges "
    "exactly as the supporting note states them; do not round, convert, "
    "extrapolate, or aggregate figures across notes unless the notes "
    "themselves provide the basis for doing so, and show your reasoning when "
    "you do.\n"
    "7. Be alert to the provenance and freshness of each note. If a note is "
    "clearly time-sensitive (prices, standings, versions, ongoing events) or "
    "appears to reflect a particular vantage point, weigh it accordingly and "
    "flag that context to the reader rather than presenting it as timeless "
    "fact.\n"
    "\n"
    "REASONING PROCESS (internal)\n"
    "Before writing, work through the notes methodically: (a) identify which "
    "notes bear on the question and which are tangential; (b) cluster notes "
    "that address the same sub-topic; (c) within each cluster, check for "
    "agreement, partial overlap, or contradiction; (d) determine what the "
    "combined evidence does and does not settle. Do NOT narrate this process "
    "or expose scratch work in the output — the reader sees only the finished "
    "report described below. Keep any chain-of-thought to yourself.\n"
    "\n"
    "CITATION EXAMPLE\n"
    "Good: 'The framework's throughput scaled roughly linearly up to eight "
    "workers before plateauing [note 4], though a separate evaluation reports "
    "diminishing returns past four workers on memory-bound workloads "
    "[note 7].' This shows attribution, an explicit contrast between sources, "
    "and no invented detail. Avoid: unsourced assertions, a citation that "
    "does not match the note, or blending two notes' numbers into a single "
    "figure they never state.\n"
    "\n"
    "REPORT STRUCTURE\n"
    "Write a comprehensive, well-organized report in Markdown with these "
    "sections:\n"
    "- '## Summary': 3-6 sentences giving the direct, decision-relevant "
    "answer to the question up front.\n"
    "- '## Key Findings': the main substantiated points, as a structured "
    "list or short thematic subsections, each claim cited. Group related "
    "evidence; do not merely restate notes one by one.\n"
    "- '## Analysis': synthesize across notes — reconcile or contrast them, "
    "draw out implications, and explain the reasoning that connects the "
    "evidence to the answer.\n"
    "- '## Limitations and Open Questions': what the notes do NOT establish, "
    "where coverage is thin or dated, unresolved conflicts, and what "
    "additional evidence would strengthen the conclusion.\n"
    "\n"
    "STYLE\n"
    "Be thorough but precise; prefer specific, evidence-anchored statements "
    "over vague generalities. Use a neutral, professional register. Do not "
    "pad with filler, and do not repeat the question back to the user. Begin "
    "your response directly with the '## Summary' heading."
)


def download_search_arena(
    repo: str, data_file: str, revision: Optional[str] = None
) -> str:
    """Fetch the Search Arena parquet from the Hub, returning a local path."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo, filename=data_file, repo_type="dataset", revision=revision
    )


def build_pools(
    parquet_path: str,
    language: str = DEFAULT_LANGUAGE,
    note_min_chars: int = DEFAULT_NOTE_MIN_CHARS,
    note_max_chars: int = DEFAULT_NOTE_MAX_CHARS,
    question_max_chars: int = DEFAULT_QUESTION_MAX_CHARS,
) -> tuple[list[dict], list[dict]]:
    """Build the (questions, notes) pools from the raw parquet.

    - Rows are kept only when `languages == [language]` (single-language
      conversations; mixed-language rows are dropped).
    - Question pool: first user turn of `messages_a` (the user turns are
      identical across the a/b sides of a battle), non-empty and at most
      `question_max_chars`.
    - Notes pool: every assistant turn from BOTH `messages_a` and
      `messages_b` whose length is in (note_min_chars, note_max_chars] —
      each is a complete search-grounded answer with inline citations.

    Both pools are built in parquet row order, so they are deterministic
    for a fixed dataset revision + filters. Each entry carries its source
    `conv_index` (row position) so assembly can exclude notes that come
    from the same conversation as the question.
    """
    import pandas as pd

    df = pd.read_parquet(
        parquet_path, columns=["messages_a", "messages_b", "languages"]
    )

    questions: list[dict] = []
    notes: list[dict] = []
    for conv_index, row in enumerate(df.itertuples(index=False)):
        if list(row.languages) != [language]:
            continue
        for side in (row.messages_a, row.messages_b):
            for m in side:
                if m.get("role") != "assistant":
                    continue
                text = (m.get("content") or "").strip()
                if note_min_chars < len(text) <= note_max_chars:
                    notes.append({"conv_index": conv_index, "text": text})
        for m in row.messages_a:
            if m.get("role") == "user":
                text = (m.get("content") or "").strip()
                if text and len(text) <= question_max_chars:
                    questions.append({"conv_index": conv_index, "text": text})
                break

    if not questions or not notes:
        raise ValueError(
            f"Empty Search Arena pools after filtering (language={language!r}, "
            f"questions={len(questions)}, notes={len(notes)})."
        )
    return questions, notes


def _sample_k(rng: random.Random, k_min: int, k_max: int) -> int:
    """Log-uniform integer in [k_min, k_max]."""
    k = int(math.exp(rng.uniform(math.log(k_min), math.log(k_max + 1))))
    return min(max(k, k_min), k_max)


def sample_request_specs(
    questions: list[dict],
    notes: list[dict],
    num_requests: int,
    k_min: int,
    k_max: int,
    sample_seed: int,
) -> list[dict]:
    """Draw `num_requests` lightweight request specs (indices only).

    Spec `i` is drawn from its own `random.Random(sample_seed*1_000_003+i)`
    so the mapping index -> request is deterministic and independent of
    how the spec list is later sharded across load processes. Notes are
    sampled without replacement, excluding notes that come from the same
    conversation as the question (a question must never be accompanied by
    its own recorded answer).

    Specs deliberately do NOT carry text — the full pools stay resident
    once per process and `assemble_messages` concatenates on demand,
    keeping the sharded task lists tiny.
    """
    specs: list[dict] = []
    n_notes = len(notes)
    for i in range(num_requests):
        rng = random.Random(sample_seed * 1_000_003 + i)
        k = _sample_k(rng, k_min, k_max)
        q_idx = rng.randrange(len(questions))
        q_conv = questions[q_idx]["conv_index"]
        note_idxs: list[int] = []
        seen: set[int] = set()
        while len(note_idxs) < k and len(seen) < n_notes:
            j = rng.randrange(n_notes)
            if j in seen:
                continue
            seen.add(j)
            if notes[j]["conv_index"] == q_conv:
                continue
            note_idxs.append(j)
        specs.append(
            {
                "request_id": f"sa-{i:06d}",
                "question_idx": q_idx,
                "note_idxs": note_idxs,
            }
        )
    return specs


def assemble_messages(
    spec: dict, questions: list[dict], notes: list[dict]
) -> list[dict]:
    """Materialize one spec into `[system, user]` chat messages.

    Pure function of (spec, pools): the same spec always produces the
    same bytes, so cycled replays and re-runs are literal replays.
    """
    parts = ["Research notes:"]
    for pos, j in enumerate(spec["note_idxs"], start=1):
        parts.append(f"[note {pos}]\n{notes[j]['text']}")
    body = "\n\n".join(parts)
    question = questions[spec["question_idx"]]["text"]
    user = f"{body}\n\n---\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
