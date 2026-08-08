#!/usr/bin/env python3
"""Draw what a dynamic trace actually offers: arrival rate and class mix over time.

Why this exists as its own script. build_dynamic_mix_trace.py emits a figure of
the rate alone, and the mix is only in the plan JSON as target ratios per
segment. The two knobs are what the trace is for, so they belong on one time
axis, and the mix has to be drawn from the arrivals themselves rather than from
the targets -- a segment's realised ratio differs from its target by the
sampling, and the mix panel is the one a reader checks the schedule against.

The knee lines are the point of the rate panel. A trace whose rate never crosses
the knee cannot show a policy adapting to load, whatever its mix does, and that
was true of the 25-75 req/s band once the workload got heavier: 93% of its
minutes sat above FluidServe's knee. Pass --knee to mark them.

Three panels share the x axis:

  1. offered rate per minute, with knee lines and segment boundaries
  2. class mix as a share of REQUESTS, stacked
  3. class mix as a share of INPUT TOKENS, stacked -- not the same picture,
     because the mean input length differs by a factor of ten across classes,
     and it is the token share that sets what the engines have to compute

Usage
-----
    python3 traces/dynamic/plot_trace_shape.py \\
        traces/dynamic/canonical/dyn60_short_m123_b1045.csv \\
        --knee 28.1 "FluidServe v0.2" --knee 18.2 "llm-d" \\
        --out results/aggregate_analysis/exp71/trace_shape.png
"""
import argparse
import collections
import csv
import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASSES = ["chat", "deepresearch", "swe"]
CLASS_COLORS = {"chat": "#1f77b4", "deepresearch": "#ff7f0e", "swe": "#d62728"}
PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "axes.linewidth": 0.75, "legend.fontsize": 8, "legend.frameon": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.4, "lines.markersize": 4.0,
}


def load(path, bin_s):
    """Per-bin request counts and input tokens per class, plus segment bounds.

    Mean input tokens come from the plan JSON when it is there, because the CSV
    for a class-plan trace carries the class but not the prompt: the prompt is
    drawn from the pool at replay time. Falling back to equal weights would make
    the token panel a copy of the request panel, which is the one thing it must
    not be, so the fallback is to skip that panel instead.
    """
    rows = collections.defaultdict(lambda: collections.Counter())
    segs = []
    last_seg, seg_start = None, 0.0
    tmax = 0.0
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("phase") == "warmup":
                continue
            t = float(r["arrival_s"])
            tmax = max(tmax, t)
            rows[int(t // bin_s)][r["class"]] += 1
            seg = r.get("segment")
            if seg != last_seg:
                if last_seg is not None:
                    segs.append((seg_start, t, last_seg))
                last_seg, seg_start = seg, t
    if last_seg is not None:
        segs.append((seg_start, tmax, last_seg))

    means = None
    plan = path.rsplit(".csv", 1)[0] + ".plan.json"
    if os.path.exists(plan):
        d = json.load(open(plan))
        means = (d.get("workload") or {}).get("mean_input_tokens")
    return rows, segs, means, tmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--bin-sec", type=float, default=60.0)
    ap.add_argument("--knee", nargs=2, action="append", default=[],
                    metavar=("REQ_PER_S", "LABEL"))
    ap.add_argument("--mean-input", nargs=3, type=float, default=None,
                    metavar=("CHAT", "DEEPRESEARCH", "SWE"),
                    help="mean input tokens per class, when the plan JSON has none")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows, segs, means, tmax = load(a.trace, a.bin_sec)
    if a.mean_input:
        means = dict(zip(CLASSES, a.mean_input))
    if not rows:
        sys.exit(f"no measured arrivals in {a.trace}")

    bins = sorted(rows)
    x = [b * a.bin_sec / 60.0 for b in bins]
    total = [sum(rows[b].values()) for b in bins]
    rate = [t / a.bin_sec for t in total]
    req_share = {c: [rows[b][c] / max(1, sum(rows[b].values())) for b in bins]
                 for c in CLASSES}
    tok_share = None
    if means:
        tok = {c: [rows[b][c] * means[c] for b in bins] for c in CLASSES}
        tot = [sum(tok[c][i] for c in CLASSES) for i in range(len(bins))]
        tok_share = {c: [tok[c][i] / max(1e-9, tot[i]) for i in range(len(bins))]
                     for c in CLASSES}

    npanel = 3 if tok_share else 2
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(npanel, 1, figsize=(6.6, 2.1 * npanel), sharex=True)

        ax[0].plot(x, rate, color="#333333")
        ax[0].set_ylabel("offered rate\n(requests/s)")
        ax[0].set_ylim(0, max(rate) * 1.15)
        for k, lab in a.knee:
            k = float(k)
            ax[0].axhline(k, ls="--", lw=0.9, color="#666666")
            over = 100.0 * sum(1 for r in rate if r > k) / len(rate)
            ax[0].text(x[-1], k, f"  {lab} knee {k:g} ({over:.0f}% of minutes above)",
                       va="center", ha="right", fontsize=7, color="#444444",
                       bbox=dict(fc="white", ec="none", pad=0.8))

        for p, (share, name) in enumerate(
                [(req_share, "share of requests")] +
                ([(tok_share, "share of input tokens")] if tok_share else []), start=1):
            bottom = [0.0] * len(bins)
            for c in CLASSES:
                ax[p].fill_between(x, bottom, [bottom[i] + share[c][i] for i in range(len(bins))],
                                   color=CLASS_COLORS[c], alpha=0.85, lw=0, label=c)
                bottom = [bottom[i] + share[c][i] for i in range(len(bins))]
            ax[p].set_ylabel(name.replace(" of ", " of\n"))
            ax[p].set_ylim(0, 1)

        # Segment boundaries on every panel: the mix schedule is what they mark,
        # and the rate panel needs them to show that the two are NOT aligned.
        for s0, s1, name in segs:
            for axi in ax:
                axi.axvline(s0 / 60.0, color="#999999", lw=0.6, ls=":")
            ax[0].text((s0 + s1) / 120.0, max(rate) * 1.05, name, ha="center",
                       fontsize=7, color="#444444")

        ax[-1].set_xlabel("time (minutes)")
        ax[-1].set_xlim(0, tmax / 60.0)
        ax[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=7)
        for axi in ax:
            axi.grid(axis="y", ls=":", lw=0.7, alpha=0.6)
        if a.title:
            ax[0].set_title(textwrap.fill(a.title, 82), fontsize=8)
        fig.tight_layout()
        fig.savefig(a.out, dpi=300)
        plt.close(fig)
    print(f"wrote {a.out}")
    print(f"  {len(bins)} bins of {a.bin_sec:g}s, rate {min(rate):.1f}..{max(rate):.1f} req/s")
    for c in CLASSES:
        v = req_share[c]
        line = f"  {c:14s} requests {100*min(v):5.1f}..{100*max(v):5.1f}%"
        if tok_share:
            w = tok_share[c]
            line += f"   input tokens {100*min(w):5.1f}..{100*max(w):5.1f}%"
        print(line)


if __name__ == "__main__":
    sys.exit(main())
