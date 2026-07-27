#!/usr/bin/env python3
"""Check that a generated arrival trace still reproduces the Azure shape.

Why this is a question at all: the rate range the fleet is driven at has to
change when the workload changes -- shortening the agent class moved the fleet's
capacity, so the rates that put it under, at, and over capacity moved with it --
and the experiment's claim is that the arrival pattern is Azure's, scaled. That
claim has to survive the rescaling or it is not a claim about Azure any more.

It does survive, and the reason is structural rather than empirical:
build_dynamic_mix_trace.py maps the Azure minute series onto [rate_min,
rate_max] by a QUANTILE (rank) transform, which is strictly monotone. A
monotone map cannot change the order of the minutes, so every peak, trough and
their relative timing are preserved exactly whatever range is chosen. Measured
on the canonical 1-hour trace the rank correlation is 1.0000 to four places.

What the rescaling DOES change is the amplitude distribution. Mapping ranks onto
a linear range spreads the quantiles evenly, which flattens Azure's heavy tail:
the source's coefficient of variation over the window is 0.309 and the generated
series' is 0.385. That is worth stating rather than hiding -- the trace is
Azure's temporal pattern with a slightly wider relative spread, not Azure's
distribution.

    python3 traces/dynamic/verify_azure_shape.py \
        --plan traces/dynamic/canonical/dyn60_azure4d.plan.json \
        --out traces/dynamic/canonical/dyn60_azure4d.shape.png
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER_STYLE = {
    "font.family": "serif", "font.size": 8, "axes.labelsize": 9,
    "axes.titlesize": 9, "axes.linewidth": 0.75, "legend.fontsize": 8,
    "legend.frameon": False, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.direction": "in", "ytick.direction": "in", "lines.linewidth": 1.2,
}


def load_minutes(path):
    rows = list(csv.DictReader(open(path)))
    key = [c for c in rows[0] if c != "minute"][0]
    return np.array([float(r[key]) for r in rows])


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--root", default=os.getcwd(),
                    help="directory the plan's source paths are relative to")
    a = ap.parse_args()

    plan = json.load(open(a.plan))
    lam = np.array(plan["rate"]["lambda_series"], float)
    src = plan["source"]
    total = (load_minutes(os.path.join(a.root, src["conv_minutes"]))
             + load_minutes(os.path.join(a.root, src["code_minutes"])))
    start = int(src["start_hour"]) * 60
    days = int(src["day_window"][1]) - int(src["day_window"][0])
    window = total[start:start + days * 24 * 60]
    if len(window) == 0:
        sys.exit("the plan's day window selects nothing from the source series")

    # Bin the source down to the generated series' resolution so the two are
    # compared at the same granularity rather than through an interpolation.
    n = len(lam)
    edges = (np.arange(n + 1) * len(window) / n).astype(int)
    binned = np.array([window[i:j].mean() if j > i else window[min(i, len(window) - 1)]
                       for i, j in zip(edges[:-1], edges[1:])])

    rho = spearman(binned, lam)
    r = float(np.corrcoef(binned, lam)[0, 1])
    print(f"plan          {a.plan}")
    print(f"source window {len(window)} minutes, compressed {src['compression_x']}x")
    print(f"rank correlation (shape preserved?)  {rho:.4f}")
    print(f"linear correlation                   {r:.4f}")
    for name, v in (("azure (window)", binned), ("generated lambda", lam)):
        print(f"{name:<18} mean {v.mean():9.1f}  cv {v.std()/v.mean():.3f}  "
              f"p5/p50/p95 {np.percentile(v,5):.1f}/{np.percentile(v,50):.1f}/"
              f"{np.percentile(v,95):.1f}")
    if rho < 0.99:
        print("\nWARNING: the rank correlation is not ~1. The generator is supposed "
              "to use a monotone quantile map, so anything below 1 means the shape "
              "was altered, not merely rescaled.")

    out = a.out or os.path.splitext(a.plan)[0].replace(".plan", "") + ".shape.png"
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(2, 1, figsize=(6.5, 4.0), sharex=True)
        t = np.arange(n) * (plan["totals"]["span_s"] / n) / 60.0
        # Both on their own scale, normalised by their own mean: the question is
        # whether the SHAPE matches, and plotting raw values on one axis would
        # only show that one is 140x the other.
        ax[0].plot(t, binned / binned.mean(), color="#888888",
                   label="Azure (conv+code, windowed and binned)")
        ax[0].plot(t, lam / lam.mean(), color="#1f77b4",
                   label="generated arrival rate")
        ax[0].set_ylabel("rate / its own mean")
        ax[0].legend(loc="upper right", ncol=1)
        ax[0].set_title(f"shape, rank correlation {rho:.4f}")
        ax[1].plot(t, lam, color="#1f77b4")
        ax[1].set_ylabel("arrival rate (req/s)")
        ax[1].set_xlabel("trace time (minutes)")
        for x in ax:
            x.grid(axis="y", ls=":", lw=0.7, alpha=0.6)
        fig.tight_layout()
        fig.savefig(out, dpi=300)
        plt.close(fig)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
