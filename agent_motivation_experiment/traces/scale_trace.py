#!/usr/bin/env python3
"""Scale a canonical arrival trace up or down, preserving temporal pattern.

Operates on our canonical `arrival_s` (traces/TRACE_FORMAT.md), source-
agnostic. One knob, `--factor f` (offered-load multiplier):

  f >= 1  UPSCALE  — TraceUpscaler overlay (EuroSys '24, Python port of
          github.com/smsajal/TraceUpscaler `getNewArrivalTimes`). Walking
          the original arrivals in order, emit floor(f) copies of each real
          timestamp (+1 w.p. the fractional part) until N (= original count)
          exist; truncate to N. Result: SAME request count (token dist
          preserved), uses the first ~1/f of the source by duration.
          => rate x f, duration / f, real burst shape within the window.

  f < 1   DOWNSCALE — TraceUpscaler is upscale-only; use a standard inverse:
          --down-method thin (default): keep each arrival independently
              w.p. f (Bernoulli point-process thinning). Timeline and burst
              LOCATIONS unchanged; only fewer requests.
              => rate x f, duration UNCHANGED, count ~ f*N.
          --down-method stretch: multiply every timestamp by 1/f (keep all
              requests, dilate time).
              => rate x f, duration / f (longer), count preserved.

Pick down-method by intent: `thin` to keep the real wall-clock window of a
big-cluster trace while matching a smaller server's capacity (drops some
requests); `stretch` to keep every request and its token sizes (dilates the
window). Both preserve the relative temporal pattern.

The runner ignores optional columns (request content comes from the
workload); they are carried through 1:1 with their row for faithfulness.
"""

import argparse
import csv
import random


def upscale_overlay(rows, factor, rng):
    """f>=1: TraceUpscaler overlay. rows: [(t, opt)] sorted, zeroed."""
    n = len(rows)
    whole = int(factor)
    frac = factor - whole
    times = []
    for t, _ in rows:
        cnt = whole + (1 if rng.random() < frac else 0)
        times.extend([t] * cnt)
        if len(times) >= n:
            break
    times = times[:n]
    # Pair the i-th upscaled timestamp with original row i's optional cols
    # (data/timing decoupled: full token distribution kept, timing scaled).
    return [(times[i], rows[i][1]) for i in range(len(times))]


def downscale_thin(rows, factor, rng):
    """f<1: Bernoulli thinning. Keep each arrival w.p. factor; same timeline."""
    return [(t, opt) for t, opt in rows if rng.random() < factor]


def downscale_stretch(rows, factor):
    """f<1: dilate time by 1/factor; keep every request."""
    return [(t / factor, opt) for t, opt in rows]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="canonical arrival CSV (arrival_s required)")
    ap.add_argument("output", help="scaled canonical CSV to write")
    ap.add_argument("--factor", type=float, required=True,
                    help="offered-load multiplier (>1 upscale, <1 downscale)")
    ap.add_argument("--down-method", choices=["thin", "stretch"], default="thin",
                    help="downscale method for factor<1 (default: thin)")
    ap.add_argument("--window-min", type=float, nargs=2, metavar=("START", "END"),
                    default=None, help="pre-slice source to [START,END] min, re-zero, then scale.")
    ap.add_argument("--jitter-ms", type=float, default=0.0,
                    help="spread duplicate timestamps by uniform[0,jitter_ms] (break hard simultaneity).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.factor <= 0:
        raise SystemExit("--factor must be > 0")

    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "arrival_s" not in reader.fieldnames:
            raise SystemExit(f"{args.input!r} missing 'arrival_s' column")
        opt_fields = [c for c in reader.fieldnames if c != "arrival_s"]
        rows = []
        for r in reader:
            raw = (r.get("arrival_s") or "").strip()
            if raw == "":
                continue
            rows.append((float(raw), {c: r.get(c, "") for c in opt_fields}))

    if not rows:
        raise SystemExit(f"{args.input!r} has no arrivals")

    rows.sort(key=lambda x: x[0])
    base = rows[0][0]
    rows = [(t - base, opt) for t, opt in rows]

    if args.window_min is not None:
        lo, hi = args.window_min[0] * 60.0, args.window_min[1] * 60.0
        rows = [r for r in rows if lo <= r[0] <= hi]
        if not rows:
            raise SystemExit(f"--window-min {args.window_min} kept no arrivals")
        wbase = rows[0][0]
        rows = [(t - wbase, opt) for t, opt in rows]

    rng = random.Random(args.seed)
    n_in = len(rows)
    span_in = rows[-1][0] or 1e-9

    if args.factor >= 1.0:
        method = "upscale/overlay"
        out = upscale_overlay(rows, args.factor, rng)
    elif args.down_method == "thin":
        method = "downscale/thin"
        out = downscale_thin(rows, args.factor, rng)
    else:
        method = "downscale/stretch"
        out = downscale_stretch(rows, args.factor)

    if not out:
        raise SystemExit("scaling produced no arrivals (factor too small?)")

    if args.jitter_ms > 0:
        jit = args.jitter_ms / 1000.0
        out = [(t + rng.uniform(0.0, jit), opt) for t, opt in out]
    out.sort(key=lambda x: x[0])
    z = out[0][0]
    out = [(t - z, opt) for t, opt in out]

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arrival_s"] + opt_fields)
        for t, opt in out:
            w.writerow([f"{t:.6f}"] + [opt[c] for c in opt_fields])

    span_out = out[-1][0] or 1e-9
    print(
        f"[{method}] factor={args.factor} | "
        f"in: {n_in:,} reqs, {span_in/60:.1f} min, {n_in/span_in:.1f} req/s "
        f"-> out: {len(out):,} reqs, {span_out/60:.1f} min, {len(out)/span_out:.1f} req/s"
    )


if __name__ == "__main__":
    main()
