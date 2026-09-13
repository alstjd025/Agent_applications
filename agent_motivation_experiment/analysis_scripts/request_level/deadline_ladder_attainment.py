#!/usr/bin/env python3
"""Score a run under a CUMULATIVE per-token deadline instead of two means.

The rule this implements (2026-09-01, user's definition):

    token i of a request is on time if it arrives within
        TTFT_SLO + i * TBT_SLO
    of the moment the request was sent, counting the first token as i = 0 so
    that its deadline is exactly TTFT_SLO;

    the request is on time if at least 95% of its tokens are;

    and token goodput is the count of tokens that met their own deadline,
    divided by the measurement window -- per token, not gated on the request.

How this differs from the rule the other tables use. That one asks two separate
questions of a request -- was the time to first token inside its budget, and was
the MEAN time between tokens inside its budget -- and the two budgets cannot pay
for each other. This rule adds them: the deadline for token i is the whole
allowance accumulated up to i, so a request that got its first token in 0.7 s
against a 5 s budget carries 4.3 s of slack forward and may run slower than
TBT_SLO for a long stretch before any token is late. It is therefore MORE
forgiving of a request whose start was fast, and less forgiving of one that was
on pace on average but stalled: a mean hides a stall, a ladder of deadlines does
not. The 95% allowance keeps a handful of late tokens from failing a request
that was otherwise served as promised.

Where the numbers come from. tbt_events.jsonl records, per request, one entry
per streamed chunk with its arrival offset in milliseconds from the send. The
engine emits one token per chunk on this workload -- chunks per token measured
0.9956 on the condition this was written against -- so chunk index is token
index. The 0.44% of tokens that arrive bundled with another shift later tokens
to a slightly smaller index and therefore a slightly later deadline, which makes
this scoring marginally lenient rather than strict; it is stated here rather
than corrected because the correction would require token-level timing the
client does not record.

Verified before use, on 2,367 requests of one condition: the first chunk's
arrival offset equals the client's own first_token_latency to a median of
0.0 ms, so the two quantities share an origin, and the last chunk's offset
matches the recorded end-to-end latency to -0.3 ms.

Denominators follow exp22_fluidserve.py exactly, so the two scorings can be read
side by side: offered counts every arrival and a rejection is a violation,
admitted drops rejections from the population, and a request cut off at the run
boundary leaves both because its outcome is unknown. The same 60 s warm-up and
20 s drain are trimmed.

Usage
-----
  python3 deadline_ladder_attainment.py --runs 'results/*exp108r1_*_rpm_*' \
      [--frac 0.95] [--out table.csv]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The ladder rule needs a rate for every class, so swe is always scored in the
# per-token form v0.4 adopted; an end-to-end budget has no expression here at
# all. `setdefault` and not an assignment, so a caller that has already asked
# for a different swe budget keeps it. These two lines must run BEFORE
# exp22_fluidserve is imported, because it reads the environment at import.
os.environ.setdefault("FS_SWE_TBT_MS", "75")
os.environ.setdefault("FS_SWE_TTFT_S", "7")
from exp22_fluidserve import (  # noqa: E402
    WARMUP_S, DRAIN_S, class_of, truthy, arm_of, SLO_RULES,
)

# The per-class budgets, as (TTFT seconds, per-token milliseconds), TAKEN FROM
# THE SAME PLACE THE OTHER SCORER TAKES THEM (2026-09-11). They were written
# here as six constants, which is the shape CLAUDE.md records as "the same
# quantity in two places and only one of them updated": EXP-121 and EXP-123 made
# all six settable through the environment on the exp22 side -- so that the
# policy flag and the scorer move together -- and this file kept scoring at the
# standard six whatever those variables said.
#
# ⚠ WITH NO ENVIRONMENT SET THE VALUES ARE UNCHANGED: chat (5 s, 50 ms),
# deepresearch (10 s, 100 ms), swe (7 s, 75 ms). Every run scored before this
# change re-scores to the same verdicts, which was checked on one run of each
# arm before the change was kept.
BUDGETS = {c: (r["ttft"], r["tbt"]) for c, r in SLO_RULES.items() if "tbt" in r}


def ladder_ok(chunk_events, ttft_s, tbt_ms, frac):
    """Did at least `frac` of this request's tokens meet their own deadline."""
    n = len(chunk_events)
    if n == 0:
        return False, 0, 0
    late = 0
    for c in chunk_events:
        off = c.get("arrival_offset_ms")
        if off is None:
            late += 1
            continue
        if off > (ttft_s * 1000.0 + c["chunk_idx"] * tbt_ms):
            late += 1
    return (n - late) >= frac * n, n, late


def score_run(run_dir, frac, dump_dir=None):
    mpath = os.path.join(run_dir, "metrics.csv")
    epath = os.path.join(run_dir, "tbt_events.jsonl")
    if not os.path.exists(epath):
        return None
    m = pd.read_csv(mpath, low_memory=False)
    m = m[m["agent"] != "job_summary"].copy()
    if m.empty:
        return None
    t0 = m["start_time"].min()
    m["rel"] = m["start_time"] - t0
    end = min(m["end_time"].max() - t0, m["rel"].max())
    m = m[(m["rel"] >= WARMUP_S) & (m["rel"] < end - DRAIN_S)].copy()
    if m.empty:
        return None
    m["class"] = m["task_id"].map(class_of)
    m["rejected"] = truthy(m, "is_rejected")
    m["errored"] = truthy(m, "is_error") | truthy(m, "is_timeout")
    m["cutoff"] = truthy(m, "is_server_terminated") & ~m["rejected"] & ~m["errored"]
    keys = set(zip(m["task_id"], m["call_index"], m["iteration"]))

    # Stream the events once; hold only the verdict per request.
    verdict, ntok = {}, {}
    with open(epath) as f:
        for line in f:
            if '"agent": "request"' not in line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            k = (r.get("task_id"), r.get("call_index"), r.get("iteration"))
            if k not in keys:
                continue
            cls = class_of(r.get("task_id"))
            if cls not in BUDGETS:
                continue
            ttft_s, tbt_ms = BUDGETS[cls]
            ok, n, late = ladder_ok(r.get("chunk_events") or [], ttft_s, tbt_ms, frac)
            verdict[k] = ok
            ntok[k] = (n, late)

    m["k"] = list(zip(m["task_id"], m["call_index"], m["iteration"]))
    # A request with no event record produced nothing observable. That is a miss
    # under any rule, the same way exp22 treats a missing first token.
    m["ladder_ok"] = m["k"].map(verdict).fillna(False).astype(bool)
    m["miss"] = ~m["ladder_ok"]
    m["violate_served"] = m["miss"]
    m["violate_offered"] = m["miss"] | m["rejected"] | m["errored"]
    m["_ntok"] = m["k"].map(lambda k: ntok.get(k, (0, 0))[0])
    m["_nlate"] = m["k"].map(lambda k: ntok.get(k, (0, 0))[1])

    live = m[~m["cutoff"]]
    served = live[~live["rejected"]]
    window = (m["rel"].max() - m["rel"].min()) or np.nan
    # Goodput is counted PER TOKEN, not per request: every token the fleet
    # actually delivered is checked against its own deadline and the ones that
    # met it are the goodput. A request that missed the 95% bar still
    # contributes the tokens of its that were on time, and a request that met it
    # does not contribute the ones that were late. Tokens from a request cut off
    # at the run boundary are included because they were really produced and
    # each one's arrival time is known -- the reason cutoffs leave the
    # attainment denominators is that the REQUEST's outcome is unknown, which
    # says nothing about the tokens already delivered.
    tok_ok = float(m["_ntok"].sum() - m["_nlate"].sum())
    tok_all = float(m["_ntok"].sum())

    if dump_dir:
        # The per-request verdict, so that a figure needing a timeline or an
        # outcome split reads THIS rule's judgement instead of reimplementing
        # it. One file per run, keyed the way the scorer keys requests.
        os.makedirs(dump_dir, exist_ok=True)
        m[["task_id", "call_index", "iteration", "class", "ladder_ok",
           "_ntok", "_nlate", "rejected", "errored", "cutoff", "rel"]].rename(
            columns={"_ntok": "n_tokens", "_nlate": "n_late"}).to_csv(
            os.path.join(dump_dir, os.path.basename(run_dir) + ".csv"),
            index=False)

    out = dict(
        run=os.path.basename(run_dir),
        arm=arm_of(run_dir),
        n=len(live),
        offered=100.0 * (~live["violate_offered"]).mean() if len(live) else np.nan,
        admitted=100.0 * (~served["violate_served"]).mean() if len(served) else np.nan,
        rejected_pct=100.0 * live["rejected"].mean() if len(live) else np.nan,
        goodput_tok_s=tok_ok / window if window else np.nan,
        throughput_tok_s=tok_all / window if window else np.nan,
        tokens_ontime_pct=100.0 * tok_ok / max(tok_all, 1.0),
    )
    for c in BUDGETS:
        sub = live[live["class"] == c]
        out[f"{c}_offered"] = 100.0 * (~sub["violate_offered"]).mean() if len(sub) else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="glob over run directories")
    ap.add_argument("--frac", type=float, default=0.95,
                    help="share of a request's tokens that must be on time")
    ap.add_argument("--out")
    ap.add_argument("--dump-verdicts", metavar="DIR",
                    help="write one CSV per run holding the per-request "
                         "verdict, so figures consume this rule rather than "
                         "reimplementing it")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    dirs = sorted(d for d in glob.glob(a.runs) if os.path.isdir(d) and "PRERUN" not in d)
    if not dirs:
        sys.exit(f"no run directories match {a.runs}")
    rows = []
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for d, r in zip(dirs, ex.map(partial(score_run, frac=a.frac,
                                                 dump_dir=a.dump_verdicts),
                                         dirs)):
                if r is None:
                    print(f"  skipped (no tbt_events.jsonl): "
                          f"{os.path.basename(d)}", file=sys.stderr)
                    continue
                rows.append(r)
                print(f"  scored {r['run']}", file=sys.stderr, flush=True)
    else:
        for d in dirs:
            r = score_run(d, a.frac, a.dump_verdicts)
            if r is None:
                print(f"  skipped (no tbt_events.jsonl): {os.path.basename(d)}", file=sys.stderr)
                continue
            rows.append(r)
            print(f"  scored {r['run']}", file=sys.stderr)
    t = pd.DataFrame(rows)
    print(f"\ncumulative per-token deadline, >= {100*a.frac:.0f}% of a request's tokens on time")
    print("budgets: chat (5 s, 50 ms)  deepresearch (10 s, 100 ms)  swe (7 s, 75 ms)\n")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(t.to_string(index=False, float_format=lambda x: f"{x:8.1f}"))
    if a.out:
        t.to_csv(a.out, index=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
