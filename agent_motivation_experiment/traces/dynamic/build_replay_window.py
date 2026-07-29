#!/usr/bin/env python3
"""Extract a contiguous window of a real trace and replay it at its own rate.

Why this exists next to build_dynamic_mix_trace.py, which already produces a
dynamic trace. That script compresses N source days into one hour and rank-maps
the rate onto a chosen band. Both steps are deliberate and both destroy the
thing this experiment needs: its own docstring records that at 4 days into 3600
bins "Azure's minute-level jitter is aggregated away", and the rank transform
replaces the real rate distribution with a uniform one. Measured on the trace it
produced, the arrival rate's autocorrelation is 0.78 at one second and 0.54 at
sixty, so the load moves far more slowly than the fleet responds and a snapshot
policy tracks it without difficulty.

This script does no compression and no rescaling. It takes a window of the
source trace and emits its arrival times verbatim, so every timescale in the
output is the timescale that was recorded. The only choice made is which window,
and that choice is stated: a window is selected by its mean rate, so that the
segment lands in the band where this cluster's policies differ.

What is NOT taken from the source: request content. The trace supplies arrival
timing only; prompts and outputs come from the three application workloads, as
they do for every other experiment here. Say "arrival timing from a 60-minute
segment of the Azure LLM Inference 2024 code trace, replayed at its recorded
rate", never "we replay the Azure trace".

Class labels are drawn by the same `build_class_sequence` the static-mix
experiments use, so the realised ratio is produced by the identical code path.
The mix is held FIXED here: EXP-30 measured that mix motion costs nothing on
two-minute segments, which is the timing most favourable to the claim, so
varying it alongside the rate would only confound the rate effect.

Example, the window this was written for:

  python3 traces/dynamic/build_replay_window.py \
      --source traces/azure/raw/2024/AzureLLMInferenceTrace_code_2024.csv \
      --start-min 5087 --duration-min 60 --mix m1 \
      --out traces/dynamic/canonical/azcode_w60_m1
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from workloads.mixed_request_level_poisson.mixplan import (  # noqa: E402
    build_class_sequence,
)

# Request-count ratios of the static mixes, as used by workload_configs/.
MIXES = {
    "m1": {"chat": 10, "deepresearch": 2, "swe": 1},
    "m2": {"chat": 40, "deepresearch": 2, "swe": 1},
    "m3": {"chat": 6, "deepresearch": 1, "swe": 2},
}


def load_window(path, start_min, duration_min, scan_rows):
    """Arrival times, in seconds from the window start, of one contiguous window.

    Reads only the timestamp column, and only far enough to cover the window.
    """
    need_s = (start_min + duration_min) * 60.0
    ts = pd.to_datetime(
        pd.read_csv(path, usecols=["TIMESTAMP"], nrows=scan_rows)["TIMESTAMP"],
        format="mixed", utc=True)
    t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
    if t.max() < need_s:
        raise SystemExit(
            f"scanned {scan_rows} rows covering {t.max()/60:.0f} min, which does "
            f"not reach minute {start_min + duration_min}; raise --scan-rows")
    lo, hi = start_min * 60.0, (start_min + duration_min) * 60.0
    w = t[(t >= lo) & (t < hi)] - lo
    return np.sort(w), str(ts.iloc[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--start-min", type=float, required=True,
                    help="window start, minutes from the first row of --source")
    ap.add_argument("--duration-min", type=float, default=60.0)
    ap.add_argument("--mix", default="m1", choices=sorted(MIXES))
    ap.add_argument("--warmup-sec", type=float, default=120.0,
                    help="lead-in drawn from the window's own first minute, so "
                         "the fleet is not cold at t=0; tagged phase=warmup and "
                         "excluded from analysis")
    ap.add_argument("--scan-rows", type=int, default=6_000_000)
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--out", required=True, help="output stem, no extension")
    a = ap.parse_args()

    w, src_t0 = load_window(a.source, a.start_min, a.duration_min, a.scan_rows)
    if len(w) < 100:
        raise SystemExit(f"window holds only {len(w)} arrivals")

    # Warm-up is built by repeating the window's OWN first minute, rather than
    # by a constant-rate Poisson draw, so the lead-in has the same character as
    # what follows it and the fleet reaches steady state on the real process.
    first_min = w[w < 60.0]
    if a.warmup_sec > 0 and len(first_min) > 10:
        reps = int(np.ceil(a.warmup_sec / 60.0))
        warm = np.concatenate([first_min + 60.0 * k for k in range(reps)])
        warm = warm[warm < a.warmup_sec]
    else:
        warm = np.array([])
    arrivals = np.concatenate([warm, w + a.warmup_sec])
    phases = ["warmup"] * len(warm) + ["measure"] * len(w)

    # Strictly ascending is an invariant of the format: the runner sorts by
    # arrival_s and the workload pairs `class` by row index, so a tie that the
    # sort reorders would silently pair a class with a different arrival.
    order = np.argsort(arrivals, kind="stable")
    arrivals, phases = arrivals[order], [phases[i] for i in order]
    for i in range(1, len(arrivals)):
        if arrivals[i] <= arrivals[i - 1]:
            arrivals[i] = arrivals[i - 1] + 1e-6

    classes = build_class_sequence(MIXES[a.mix], len(arrivals), a.seed)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["arrival_s", "class", "phase", "segment"])
        for t, c, p in zip(arrivals, classes, phases):
            wr.writerow([f"{t:.4f}", c, p, a.mix])

    meas = w
    per_min = np.histogram(meas, bins=np.arange(0, meas.max() + 60, 60))[0] / 60.0
    plan = {
        "source": os.path.basename(a.source),
        "source_first_timestamp": src_t0,
        "window_start_min": a.start_min,
        "duration_min": a.duration_min,
        "replay": "verbatim: no time compression, no rate rescaling",
        "mix": a.mix, "mix_ratio": MIXES[a.mix],
        "warmup_sec": a.warmup_sec,
        "measure_arrivals": int(len(meas)),
        "measure_mean_rate": float(len(meas) / meas.max()),
        "per_minute_rate": [round(float(x), 1) for x in per_min],
        "rate_min": float(per_min.min()), "rate_max": float(per_min.max()),
        "t_range_s": [a.warmup_sec, float(a.warmup_sec + meas.max())],
    }
    with open(a.out + ".plan.json", "w") as f:
        json.dump(plan, f, indent=1)

    print(f"wrote {a.out}.csv  {len(arrivals)} rows "
          f"({len(warm)} warmup + {len(meas)} measured)")
    print(f"  mean {plan['measure_mean_rate']:.1f} req/s, "
          f"per-minute {plan['rate_min']:.0f}..{plan['rate_max']:.0f} req/s")
    print("  per-minute:", " ".join(f"{x:.0f}" for x in per_min))


if __name__ == "__main__":
    main()
