"""Derive a short-input variant of a recorded SWE agent transcript.

Why this exists
---------------
The recorded SWE transcript averages 22,474 input tokens per request, against
674 for chat and 4,055 for deep research. In a mix drawn at equal request
counts that puts 82.6% of all input tokens in one class, and the consequence is
not a matter of degree: the per-request server cost of SWE is about 15x chat, so
every partition PolyServe can compute over four servers comes out the same
(2 SWE / 1 chat / 1 deep research) for every mix from 1:1:1 to 5% SWE. A static
partition that never has to move cannot be shown to be worse than one that
moves, so the workload has to stop being dominated by a single class before the
comparison means anything.

Shortening the class is also the more realistic setting. A 14,481-token system
prompt is at the top of the range real coding agents use; 4,000 tokens of role,
tool descriptions and output format is squarely inside it.

What is preserved, and why each one matters
-------------------------------------------
1. **Prefix structure.** Every transformation is a pure function of the message
   content, so two records that shared a message prefix before still share the
   identical token prefix afterwards. The engine's radix cache therefore sees
   the same sharing pattern, only at a smaller scale. The measured structural
   prefix share goes from 81.5% to about 78% -- deliberately not to zero,
   because an agent workload with no prefix reuse is the unrealistic case.
2. **Message count and roles.** The conversation shape (how many turns, who
   speaks) is untouched, so the intra-job prefix relations between call 1, 2, 3
   of the same agent job survive.
3. **The end of every message.** Long messages are cut in the middle, not at
   the end, because the stage instruction the agent is responding to is the
   last thing in the last user message. Cutting the tail would change what the
   model is asked to do, and therefore how much it generates.
4. **Relative length spread.** Non-system messages are scaled by one ratio, so
   the distribution of per-request length keeps its shape.

What is NOT preserved
---------------------
`baseline_ttft_s` / `baseline_tbt_mean_ms` / `baseline_e2e_s` are the solo
timings of the ORIGINAL long prompt and are left in place unchanged. They are
wrong for the short prompt. They are unused in the Llumnix mix runs, which pass
`--disable-timeouts` (so `job_timeout_sec` is forced to 0) and score against
absolute SLO thresholds from the workload config rather than against a recorded
baseline. The fields are kept rather than deleted so the record schema stays
identical; `baseline_valid: false` is stamped on every record so anything that
does reach for them can tell.

Usage
-----
    python workloads/codingagent_request_level_poisson/build_short_transcript.py \
        --in  workloads/codingagent_request_level_poisson/data/transcript_swe_calls_mix1500.jsonl \
        --out workloads/codingagent_request_level_poisson/data/transcript_swe_short7k_mix1500.jsonl \
        --target-mean-tokens 7000
"""

import argparse
import hashlib
import json
import re
import statistics
import sys

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")
_TOK_CACHE: dict[str, int] = {}

TRUNCATION_MARKER = "\n\n...[truncated for length]...\n\n"

# Sections of the coding-agent system prompt kept in the short variant, in the
# order they appear. The choice is not arbitrary trimming: a real agent system
# prompt states the role, describes the tools, and fixes the output format. The
# dropped sections are the encyclopedic parts -- style guides, security
# guidance, five worked examples, advanced patterns -- which are the ones a real
# deployment would put in retrieved context rather than in every request.
KEEP_SECTIONS = [
    "## Role Definition and Core Capabilities",
    "### General Principles",
    "## Tool Usage Instructions",
    "## Output Format Specifications",
]


def ntok(text: str) -> int:
    key = hashlib.md5(text.encode("utf-8")).hexdigest()
    n = _TOK_CACHE.get(key)
    if n is None:
        n = len(_ENC.encode(text))
        _TOK_CACHE[key] = n
    return n


def _section_spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, heading) per markdown section, plus a preamble.

    A section runs to the next heading of the SAME OR HIGHER level, so naming a
    level-2 heading selects it together with its level-3 subsections. Ending it
    at the next heading of any level instead would select the bare heading line:
    "## Tool Usage Instructions" is 254 characters on its own and 6,434 with the
    per-tool subsections that are the actual content.
    """
    heads = [(m.start(), len(m.group(1)), m.group().strip())
             for m in re.finditer(r"(?m)^(#{2,6}) .*$", text)]
    if not heads:
        return [(0, len(text), "")]
    spans: list[tuple[int, int, str]] = [(0, heads[0][0], "__preamble__")]
    for i, (pos, level, head) in enumerate(heads):
        end = len(text)
        for npos, nlevel, _ in heads[i + 1:]:
            if nlevel <= level:
                end = npos
                break
        spans.append((pos, end, head))
    return spans


def shorten_system_prompt(text: str, tail_chars: int = 1200) -> str:
    """Keep a coherent subset of sections instead of cutting at a token offset.

    A system prompt is the one message a reader of the workload will actually
    look at, and a mid-sentence cut in it would make the workload look broken
    rather than smaller. Section selection produces a prompt that still reads as
    a complete instruction set.
    """
    spans = _section_spans(text)
    by_head = {head: (a, b) for a, b, head in spans}
    parts: list[str] = []
    pre = by_head.get("__preamble__")
    if pre:
        parts.append(text[pre[0]:pre[1]].rstrip())
    for head in KEEP_SECTIONS:
        span = by_head.get(head)
        if span is None:
            raise ValueError(
                f"system prompt has no section {head!r}; the prompt changed and "
                f"KEEP_SECTIONS needs revisiting. Headings present: "
                f"{sorted(by_head)}"
            )
        parts.append(text[span[0]:span[1]].rstrip())
    # The closing paragraph restates the operating rules and is what the model
    # sees last before the conversation; dropping it would change the framing.
    parts.append(text[-tail_chars:].lstrip())
    return "\n\n".join(parts) + "\n"


def shorten_message(text: str, ratio: float, min_keep_tokens: int = 64,
                    head_share: float = 0.55) -> str:
    """Cut the middle out of a message, at line boundaries, deterministically.

    Head and tail are both kept because the two ends carry different things: the
    head says what the block is (a file, a diff, a tool result) and the tail
    carries the instruction the model answers. Cutting at line boundaries is
    what a real agent harness does when it elides a long tool output, so the
    result still reads as a plausible prompt.
    """
    n = ntok(text)
    target = max(min_keep_tokens, int(round(n * ratio)))
    if target >= n:
        return text
    lines = text.splitlines(keepends=True)
    if len(lines) < 4:
        # Not enough line structure to cut cleanly; fall back to a token cut.
        toks = _ENC.encode(text)
        head_n = int(target * head_share)
        head = _ENC.decode(toks[:head_n])
        tail = _ENC.decode(toks[len(toks) - (target - head_n):])
        return head + TRUNCATION_MARKER + tail

    head_budget = int(target * head_share)
    tail_budget = target - head_budget

    head_lines, used = [], 0
    for ln in lines:
        c = ntok(ln)
        if used + c > head_budget and head_lines:
            break
        head_lines.append(ln)
        used += c

    tail_lines, used = [], 0
    for ln in reversed(lines[len(head_lines):]):
        c = ntok(ln)
        if used + c > tail_budget and tail_lines:
            break
        tail_lines.append(ln)
        used += c
    tail_lines.reverse()

    if not tail_lines:
        tail_lines = lines[-1:]
    return "".join(head_lines).rstrip() + TRUNCATION_MARKER + "".join(tail_lines).lstrip()


def structural_prefix_share(records: list[dict]) -> float:
    """Fraction of input tokens that lie in a prefix shared with another record.

    Message-granularity, which is where the boundaries actually are: two records
    share a token prefix exactly as far as they share whole leading messages.
    This is the ceiling a radix cache could reach with unlimited capacity, so it
    measures the workload rather than the engine's eviction policy.
    """
    from collections import Counter

    paths, tokens = [], []
    for r in records:
        p, t = [], []
        for m in r["messages"]:
            h = hashlib.md5(m["content"].encode("utf-8")).hexdigest()
            p.append((m.get("role", "user"), h))
            t.append(ntok(m["content"]))
        paths.append(p)
        tokens.append(t)

    seen: Counter = Counter()
    for p in paths:
        for k in range(1, len(p) + 1):
            seen[tuple(p[:k])] += 1

    shared = total = 0
    for p, t in zip(paths, tokens):
        total += sum(t)
        for k in range(1, len(p) + 1):
            if seen[tuple(p[:k])] >= 2:
                shared += t[k - 1]
            else:
                break
    return shared / max(total, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    # Pinning the ratio is what makes a larger build a SUPERSET of a smaller one
    # rather than a different workload. Everything else here is a pure function
    # of message content: shorten_message is keyed on (content, ratio) and the
    # system prompt on --system-tail-chars, so two builds that share a ratio
    # produce byte-identical output for every record they share. The ratio
    # itself is the one global, solved from the mean input length of whatever
    # file is passed in, so rebuilding from a larger transcript moves it -- 1,500
    # records give 0.369435 and the full 13,218 give 0.373289, a 1.04%
    # difference that would silently rewrite every record already measured.
    #
    # Pass the earlier build's ratio to extend a transcript. The mean then lands
    # near the target rather than on it (6,970 instead of 7,000 for the full
    # file at 0.369435), which is the right trade: 0.4% off a target that was
    # itself chosen round, against a workload that stays comparable.
    ap.add_argument("--ratio", type=float, default=None,
                    help="Override the solved conversation ratio. Use the "
                         "ratio an earlier build printed to make this build a "
                         "superset of it; the mean will then land near "
                         "--target-mean-tokens rather than on it.")
    ap.add_argument("--target-mean-tokens", type=float, default=7000.0,
                    help="target mean input tokens per request after shortening")
    ap.add_argument("--system-tail-chars", type=int, default=1200)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.src, encoding="utf-8") if l.strip()]
    if not records:
        print(f"no records in {args.src}", file=sys.stderr)
        return 1

    before_mean = statistics.mean(r["recorded_input_tokens"] for r in records)
    before_share = structural_prefix_share(records)

    # One shared system prompt across the whole transcript, verified rather than
    # assumed: if some records carried a different one, shortening only the
    # first would silently split the shared prefix into two.
    system_texts = {r["messages"][0]["content"] for r in records
                    if r["messages"] and r["messages"][0].get("role") == "system"}
    if len(system_texts) != 1:
        print(f"expected exactly one distinct system prompt, found "
              f"{len(system_texts)}", file=sys.stderr)
        return 1
    long_system = system_texts.pop()
    short_system = shorten_system_prompt(long_system, args.system_tail_chars)
    sys_before, sys_after = ntok(long_system), ntok(short_system)

    # The ratio for the conversation is solved for, not guessed: the system
    # prompt is now a fixed cost on every request, so the rest has to absorb
    # whatever the target leaves over.
    tail_before = before_mean - sys_before
    tail_target = args.target_mean_tokens - sys_after
    if tail_target <= 0:
        print(f"target {args.target_mean_tokens} is below the shortened system "
              f"prompt alone ({sys_after} tokens)", file=sys.stderr)
        return 1
    ratio = min(1.0, tail_target / tail_before)
    if args.ratio is not None:
        print(f"ratio overridden: solved {ratio:.6f} -> given {args.ratio:.6f}")
        ratio = args.ratio

    cache: dict[str, str] = {}
    out_records = []
    for r in records:
        msgs = []
        for i, m in enumerate(r["messages"]):
            content = m["content"]
            if i == 0 and m.get("role") == "system":
                new = short_system
            else:
                new = cache.get(content)
                if new is None:
                    new = shorten_message(content, ratio)
                    cache[content] = new
            msgs.append({"role": m.get("role", "user"), "content": new})
        rec = dict(r)
        rec["messages"] = msgs
        rec["recorded_input_tokens"] = sum(ntok(m["content"]) for m in msgs)
        rec["baseline_valid"] = False
        rec["derived_from"] = args.src
        out_records.append(rec)

    with open(args.dst, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    after_mean = statistics.mean(r["recorded_input_tokens"] for r in out_records)
    after_share = structural_prefix_share(out_records)
    ins = [r["recorded_input_tokens"] for r in out_records]
    ins.sort()

    def pct(p):
        return ins[min(len(ins) - 1, int(len(ins) * p / 100))]

    print(f"records                 {len(out_records)}")
    print(f"system prompt tokens    {sys_before} -> {sys_after}")
    print(f"conversation ratio      {ratio:.4f}")
    print(f"mean input tokens       {before_mean:.0f} -> {after_mean:.0f}")
    print(f"input p10/p50/p90       {pct(10)} / {pct(50)} / {pct(90)}")
    print(f"structural prefix share {before_share*100:.1f}% -> {after_share*100:.1f}%")
    print(f"wrote {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
