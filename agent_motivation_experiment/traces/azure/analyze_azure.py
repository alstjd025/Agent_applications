#!/usr/bin/env python3
"""Characterize an Azure LLM Inference trace (2023 or 2024, code or conv).

Reports request count, time span, per-request token fields, arrival rate,
and burstiness — answering "what data, how long, what per-request info".

Memory-light: streams the file once, keeping per-hour arrival buckets and
bounded-integer histograms for the token columns (no full row buffering),
so it handles the multi-GB 2024 files.

Schema: TIMESTAMP,ContextTokens,GeneratedTokens  (stdlib only).
"""

import argparse
import csv
import re
from datetime import datetime

# Trailing tz offset (+00:00 / +0000 / Z). Uniform across a trace, so it is
# safe to drop for relative-timing analysis. 2024 traces carry it; 2023 don't.
_TZ_OFFSET = re.compile(r"(?:[+-]\d{2}:?\d{2}|Z)$")


def _parse_ts(s: str) -> datetime:
    s = _TZ_OFFSET.sub("", s.strip()).strip()
    if "." in s:
        head, frac = s.split(".", 1)
        # %f handles at most 6 fractional digits; 2023 uses 7 (100 ns ticks).
        s = f"{head}.{frac[:6].ljust(6, '0')}"
        fmt = "%Y-%m-%d %H:%M:%S.%f"
    else:
        fmt = "%Y-%m-%d %H:%M:%S"
    return datetime.strptime(s, fmt)


def _quantiles(hist: dict, qs):
    """Exact quantiles from an int->count histogram."""
    total = sum(hist.values())
    keys = sorted(hist)
    out = {}
    targets = {q: q * total for q in qs}
    cum = 0
    qi = 0
    sqs = sorted(qs)
    for k in keys:
        cum += hist[k]
        while qi < len(sqs) and cum >= targets[sqs[qi]]:
            out[sqs[qi]] = k
            qi += 1
    while qi < len(sqs):
        out[sqs[qi]] = keys[-1]
        qi += 1
    return out


def _summ(hist: dict, name: str):
    total = sum(hist.values())
    ssum = sum(k * c for k, c in hist.items())
    q = _quantiles(hist, [0.5, 0.95, 0.99])
    print(f"  {name:15s} mean={ssum/total:8.1f}  median={q[0.5]:7d}  "
          f"p95={q[0.95]:7d}  p99={q[0.99]:7d}  max={max(hist):7d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="Azure trace CSV")
    args = ap.parse_args()

    n = 0
    first = last = None
    ctx_hist, gen_hist = {}, {}
    hour_buckets = {}   # hour index -> count
    minute_buckets = {} # minute index -> count
    t0 = None

    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = _parse_ts(row["TIMESTAMP"])
            if first is None:
                first = dt
                t0 = dt
            last = dt
            n += 1
            ctx = int(row.get("ContextTokens", 0) or 0)
            gen = int(row.get("GeneratedTokens", 0) or 0)
            ctx_hist[ctx] = ctx_hist.get(ctx, 0) + 1
            gen_hist[gen] = gen_hist.get(gen, 0) + 1
            rel = (dt - t0).total_seconds()
            hour_buckets[int(rel // 3600)] = hour_buckets.get(int(rel // 3600), 0) + 1
            minute_buckets[int(rel // 60)] = minute_buckets.get(int(rel // 60), 0) + 1

    span_s = (last - first).total_seconds()
    hcounts = list(hour_buckets.values())
    mcounts = list(minute_buckets.values())
    mean_h = sum(hcounts) / len(hcounts)
    mean_m = sum(mcounts) / len(mcounts)

    def pstdev(xs, mu):
        return (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5

    print(f"file              : {args.input}")
    print(f"requests          : {n:,}")
    print(f"first / last       : {first}  ->  {last}")
    print(f"span              : {span_s/3600:.1f} h ({span_s/86400:.2f} days)")
    print(f"mean rate         : {n/span_s:.2f} req/s")
    print(f"per-hour rate     : mean={mean_h:,.0f}  peak={max(hcounts):,}  "
          f"min={min(hcounts):,}  CV={pstdev(hcounts, mean_h)/mean_h:.2f}")
    print(f"per-min  rate     : mean={mean_m:,.0f}  peak={max(mcounts):,}  "
          f"peak/mean={max(mcounts)/mean_m:.1f}x  CV={pstdev(mcounts, mean_m)/mean_m:.2f}")
    print("token distributions (per request):")
    _summ(ctx_hist, "ContextTokens")
    _summ(gen_hist, "GeneratedTokens")


if __name__ == "__main__":
    main()
