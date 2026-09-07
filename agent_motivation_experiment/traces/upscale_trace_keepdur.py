#!/usr/bin/env python3
"""Upscale a canonical arrival trace WITHOUT shortening it.

WHY THIS EXISTS ALONGSIDE scale_trace.py. That tool's upscale path is the
TraceUpscaler overlay: it emits floor(f) copies of each arrival but then
truncates to the ORIGINAL request count, so the rate rises by f and the duration
falls to 1/f. For the Qwen work that did not matter -- the factor was 0.644, a
downscale, and `thin` keeps the wall clock. Here the eight-instance Llama-3.1-8B
fleet needs roughly six times the arrival rate of the Llama-70B hour, and
scale_trace.py would turn a 61-minute trace into a 10-minute one. An hour trace
that is not an hour measures something else: the mix schedule changes every 15
minutes, and the fleet's KV pool takes about 150 seconds to fill.

WHAT THIS DOES INSTEAD. Overlay: emit floor(f) copies of every arrival plus one
more with probability frac(f), keep the full timeline, and jitter each copy
inside a typical inter-arrival gap so the copies do not land on identical
timestamps. Rate x f, duration unchanged, count ~ f x N. Each copy inherits its
row's class and segment, so the mix schedule and the burst locations are
preserved minute by minute.

WHAT IT DOES NOT PRESERVE. The request count changes, so anything derived from
"the trace has N requests" has to be recomputed. And the copies are not new
requests from the source trace -- they are the same arrival pattern played
several times over, which is the same assumption TraceUpscaler makes.

  python3 upscale_trace_keepdur.py in.csv out.csv --factor 6.2 [--seed 42]
"""
import argparse, pandas as pd, numpy as np, os, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("output")
    ap.add_argument("--factor", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.factor < 1.0:
        sys.exit("factor < 1 is a downscale; use scale_trace.py --down-method thin")
    if os.path.exists(a.output):
        sys.exit(f"refusing to overwrite {a.output}")
    d = pd.read_csv(a.input)
    if "arrival_s" not in d.columns:
        sys.exit("input has no arrival_s column")
    rng = np.random.default_rng(a.seed)
    k, frac = int(np.floor(a.factor)), a.factor - np.floor(a.factor)
    t = d["arrival_s"].to_numpy()
    gap = float(np.median(np.diff(np.sort(t))))      # typical original gap
    parts = []
    for c in range(k):
        parts.append(d.copy())
    if frac > 0:
        take = rng.random(len(d)) < frac
        parts.append(d[take].copy())
    out = pd.concat(parts, ignore_index=True)
    # Jitter inside one typical gap so copies do not share a timestamp; the
    # minute-by-minute rate and the class mix are untouched by a shift this small.
    out["arrival_s"] = np.clip(out["arrival_s"].to_numpy()
                               + rng.uniform(-gap/2, gap/2, len(out)),
                               0.0, float(t.max()))
    out = out.sort_values("arrival_s", kind="mergesort").reset_index(drop=True)
    out.to_csv(a.output, index=False)

    def band(x):
        per = [((x >= 60*i) & (x < 60*(i+1))).sum()/60 for i in range(int(x.max()//60)+1)]
        per = [p for p in per if p > 0]
        return min(per), sum(per)/len(per), max(per)
    lo0, me0, hi0 = band(pd.Series(t))
    lo1, me1, hi1 = band(out["arrival_s"])
    print(f"  in : {len(d):,} arrivals, {t.max()/60:.1f} min, {lo0:.1f} / {me0:.1f} / {hi0:.1f} req/s (min/mean/max per minute)")
    print(f"  out: {len(out):,} arrivals, {out['arrival_s'].max()/60:.1f} min, {lo1:.1f} / {me1:.1f} / {hi1:.1f} req/s")
    print(f"  realised factor: count {len(out)/len(d):.3f}, mean rate {me1/me0:.3f}  (asked {a.factor})")
    for col in ("class", "segment"):
        if col in d.columns:
            a0 = d[col].value_counts(normalize=True).sort_index()
            a1 = out[col].value_counts(normalize=True).sort_index()
            drift = (a1 - a0).abs().max()
            print(f"  {col} mix preserved to {drift*100:.3f} percentage points")

if __name__ == "__main__":
    main()
