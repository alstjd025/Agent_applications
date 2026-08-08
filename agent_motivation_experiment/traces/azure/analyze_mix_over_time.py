#!/usr/bin/env python3
"""Does the conv:code ratio in the Azure LLM Inference 2024 traces move over time?

Why this is asked. Our workload assumes the class mix is not stationary, and the
dynamic trace we built cycles it through three hand-chosen ratios. That schedule
is synthetic. The arrival RATE in the same trace is Azure-shaped, so if the two
Azure traces also disagree with each other over time, the mix has a measured
source too rather than a chosen one.

What is and is not claimed. Azure gives two request streams, conv and code. They
are not our three classes: conv is closest to chat and code to swe, and nothing
in Azure corresponds to deep research. So the most this can support is "the
proportion between two request populations of a real service moves by X over a
day", not "our mix is Azure's mix".

Alignment caveat, already recorded in build_dynamic_mix_trace.py: the two files
cover different calendar weeks (conv 2024-05-12..18, code 2024-05-10..16). Hour
of day lines up because both start at 00:00 UTC; the dates do not. Both views
are printed: elapsed-time-from-each-file's-own-start, which is how the trace
builder sums them, and hour-of-day averaged over days, which is the one that can
be read as a diurnal pattern.

Streams both files; no pandas.
"""
import collections
import os
import sys

RAW = "traces/azure/raw/2024"
FILES = {"conv": "AzureLLMInferenceTrace_conv_2024.csv",
         "code": "AzureLLMInferenceTrace_code_2024.csv"}
BIN_MIN = 10          # minutes per bin


def scan(path):
    """(bin index from file start) -> [requests, input tokens], plus by hour of day."""
    by_bin = collections.defaultdict(lambda: [0, 0])
    by_hour = collections.defaultdict(lambda: [0, 0])
    t0 = None
    n = 0
    with open(path, "r", buffering=1 << 22) as f:
        f.readline()                                   # header
        for line in f:
            # `2024-05-10 00:00:00.009930+00:00,2162,5` -- the timestamp holds no
            # comma, so splitting from the right is safe and much cheaper than csv.
            try:
                ts, ctx, _gen = line.rsplit(",", 2)
            except ValueError:
                continue
            try:
                ctx = int(ctx)
            except ValueError:
                continue
            day = int(ts[8:10])
            hh = int(ts[11:13])
            mm = int(ts[14:16])
            minute = (day * 24 + hh) * 60 + mm
            if t0 is None:
                t0 = minute
            b = (minute - t0) // BIN_MIN
            rec = by_bin[b]
            rec[0] += 1
            rec[1] += ctx
            rec = by_hour[hh]
            rec[0] += 1
            rec[1] += ctx
            n += 1
    return by_bin, by_hour, n


def pct(x):
    return f"{100.0 * x:5.1f}%"


def main():
    os.chdir(sys.argv[1] if len(sys.argv) > 1 else ".")
    out = {}
    for name, fn in FILES.items():
        p = os.path.join(RAW, fn)
        print(f"scanning {fn} ...", flush=True)
        out[name] = scan(p)
        print(f"  {out[name][2]:,} rows", flush=True)

    cb, ch, cn = out["conv"]
    kb, kh, kn = out["code"]

    print(f"\n=== totals")
    print(f"  conv {cn:>12,} requests, {sum(v[1] for v in cb.values()):>15,} input tokens, "
          f"mean {sum(v[1] for v in cb.values())/cn:,.0f}")
    print(f"  code {kn:>12,} requests, {sum(v[1] for v in kb.values()):>15,} input tokens, "
          f"mean {sum(v[1] for v in kb.values())/kn:,.0f}")
    print(f"  code share overall: requests {pct(kn/(cn+kn))}, "
          f"input tokens {pct(sum(v[1] for v in kb.values()) / (sum(v[1] for v in kb.values())+sum(v[1] for v in cb.values())))}")

    # --- view 1: summed by index, which is what the trace builder does
    common = sorted(set(cb) & set(kb))
    shares_r, shares_t = [], []
    for b in common:
        cr, ct = cb[b]
        kr, kt = kb[b]
        if cr + kr == 0 or ct + kt == 0:
            continue
        shares_r.append(kr / (cr + kr))
        shares_t.append(kt / (ct + kt))
    shares_r.sort()
    shares_t.sort()

    def q(v, p):
        return v[int(p * (len(v) - 1))]

    print(f"\n=== view 1: {BIN_MIN}-minute bins, aligned on each file's own start "
          f"({len(common)} bins = {len(common)*BIN_MIN/60:.0f} h)")
    print("    this is the alignment build_dynamic_mix_trace.py uses")
    for label, v in (("code share of REQUESTS", shares_r),
                     ("code share of INPUT TOKENS", shares_t)):
        print(f"  {label:26s} p5 {pct(q(v,0.05))}  p50 {pct(q(v,0.5))}  "
              f"p95 {pct(q(v,0.95))}  min {pct(v[0])}  max {pct(v[-1])}"
              f"   p95/p5 = {q(v,0.95)/q(v,0.05):.2f}x")

    # --- view 2: hour of day, averaged over the days each file covers
    print(f"\n=== view 2: hour of day (UTC), averaged over all days in each file")
    print(f"  {'hour':>4}{'conv req/h':>12}{'code req/h':>12}{'code share req':>16}{'code share tok':>16}")
    hr_r, hr_t = [], []
    for h in range(24):
        cr, ct = ch.get(h, [0, 0])
        kr, kt = kh.get(h, [0, 0])
        if cr + kr == 0:
            continue
        sr = kr / (cr + kr)
        st = kt / (ct + kt) if ct + kt else float("nan")
        hr_r.append(sr)
        hr_t.append(st)
        print(f"  {h:>4}{cr:>12,}{kr:>12,}{pct(sr):>16}{pct(st):>16}")
    if hr_r:
        print(f"\n  hour-of-day range: requests {pct(min(hr_r))} .. {pct(max(hr_r))} "
              f"({max(hr_r)/min(hr_r):.2f}x),  input tokens {pct(min(hr_t))} .. {pct(max(hr_t))} "
              f"({max(hr_t)/min(hr_t):.2f}x)")


if __name__ == "__main__":
    main()
