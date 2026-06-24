#!/usr/bin/env python3
"""Convert an Azure LLM Inference trace to the canonical arrival format.

Source schema (Azure LLM Inference 2023/2024,
github.com/Azure/AzurePublicDataset):

    TIMESTAMP,ContextTokens,GeneratedTokens
    2023-11-16 18:17:03.9799600,4808,10
    ...

`TIMESTAMP` is an absolute datetime with 7-digit (100 ns) fractional
seconds. This emits the canonical format consumed by
`run_experiment.py --mode trace-replay` (see TRACE_FORMAT.md):

    arrival_s,request_id,input_tokens,output_tokens

All intensity shaping lives here (upstream of the runner), not in the
runner: --time-scale, --window-min, --target-lambda. The runner replays
the resulting arrival_s verbatim.

Stdlib only (no pandas).
"""

import argparse
import csv
from datetime import datetime


def _parse_ts(s: str) -> float:
    """Parse 'YYYY-MM-DD HH:MM:SS.fffffff' to epoch seconds.

    Python's %f handles at most 6 fractional digits; Azure uses 7
    (100 ns ticks), so truncate the fractional part to microseconds.
    """
    s = s.strip()
    if "." in s:
        head, frac = s.split(".", 1)
        frac = frac[:6].ljust(6, "0")
        s = f"{head}.{frac}"
        fmt = "%Y-%m-%d %H:%M:%S.%f"
    else:
        fmt = "%Y-%m-%d %H:%M:%S"
    return datetime.strptime(s, fmt).timestamp()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="Azure trace CSV (TIMESTAMP,ContextTokens,GeneratedTokens)")
    ap.add_argument("output", help="canonical arrival CSV to write")
    ap.add_argument(
        "--time-scale", type=float, default=1.0,
        help="Multiply arrival_s by this (s<1 = denser/faster, s>1 = sparser). "
             "Applied after windowing, before --target-lambda.",
    )
    ap.add_argument(
        "--window-min", type=float, nargs=2, metavar=("START", "END"), default=None,
        help="Keep only arrivals in [START, END] minutes (pre-scale clock), then re-zero.",
    )
    ap.add_argument(
        "--target-lambda", type=float, default=None,
        help="Rescale so the mean arrival rate equals this (requests/sec). "
             "Overrides --time-scale's effect on mean rate; burst shape is preserved.",
    )
    args = ap.parse_args()

    rows = []  # (arrival_s, ctx, gen)
    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = _parse_ts(row["TIMESTAMP"])
            ctx = int(row.get("ContextTokens", 0) or 0)
            gen = int(row.get("GeneratedTokens", 0) or 0)
            rows.append([ts, ctx, gen])

    if not rows:
        raise SystemExit(f"No rows parsed from {args.input!r}")

    rows.sort(key=lambda r: r[0])
    t0 = rows[0][0]
    for r in rows:
        r[0] -= t0  # seconds from start

    if args.window_min is not None:
        lo, hi = args.window_min[0] * 60.0, args.window_min[1] * 60.0
        rows = [r for r in rows if lo <= r[0] <= hi]
        if not rows:
            raise SystemExit(f"--window-min {args.window_min} kept no arrivals")
        base = rows[0][0]
        for r in rows:
            r[0] -= base

    if args.time_scale != 1.0:
        for r in rows:
            r[0] *= args.time_scale

    if args.target_lambda is not None:
        span = rows[-1][0]
        if span <= 0:
            raise SystemExit("Cannot --target-lambda: zero time span")
        observed_lambda = len(rows) / span
        factor = observed_lambda / args.target_lambda
        for r in rows:
            r[0] *= factor

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arrival_s", "request_id", "input_tokens", "output_tokens"])
        for i, (a, ctx, gen) in enumerate(rows):
            w.writerow([f"{a:.6f}", f"az{i}", ctx, gen])

    span = rows[-1][0]
    print(
        f"Wrote {len(rows)} arrivals to {args.output} | "
        f"span={span:.1f}s ({span / 60.0:.1f} min) | "
        f"mean rate={len(rows) / span:.3f} req/s"
    )


if __name__ == "__main__":
    main()
