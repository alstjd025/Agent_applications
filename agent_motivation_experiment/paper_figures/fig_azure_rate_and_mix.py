#!/usr/bin/env python3
"""Paper figure: in the source trace the arrival rate and the class composition both move, and they move apart.

  azure_rate_and_mix.pdf    3.335 x 2.45 in, `figure`, width=\\columnwidth

Two panels over the SAME four-day window of the Azure LLM Inference 2024 traces.

  (a) arrival rate, normalised to its own peak
  (b) the share of arrivals that are code requests, in 10-minute bins

This exists because the motivation makes a two-part claim -- the offered load
varies AND what it is made of varies -- and `azure_trace_shape.pdf` only shows
the first half. The second half is what makes a fixed class partition wrong: a
fleet could be provisioned for the peak rate and still be holding the wrong
allocation, because the mix at the peak is not the mix at the trough.

WHY BOTH PANELS ARE THE SOURCE TRACE AND NOT OUR REPLAY. Neither axis of our
replayed trace can be used as evidence for the claim it is built to exercise:

  rate  `build_dynamic_mix_trace.py` applies a quantile (rank) transform onto a
        chosen band, so the replay's peak-to-trough ratio is a parameter we
        picked. Plotting it would be circular. This is the same reason
        `azure_trace_shape.pdf` plots the source.
  mix   the replay's class schedule is synthetic and has THREE classes; Azure
        has two, conversation and code, with nothing corresponding to deep
        research. So panel (b) is evidence that composition MOVES on a real
        deployment, and it is NOT evidence for the levels we chose (10:2:1) or
        for a three-class decomposition. That limit is already recorded in
        motivation.md section 7.3.1 and must stay in the caption.

WHAT THE FIGURE IS FOR, precisely: the composition is not a function of the load
level. Over this window the correlation between the binned arrival rate and the
code share is +0.57 at zero lag and +0.63 at its best lag anywhere within twelve
hours either way, so the load level accounts for about 40% of the variance in
the composition and a policy that infers one from the other is wrong the rest of
the time. That number belongs in the caption.

DO NOT SAY THE SHARE PEAKS LATER THAN THE RATE. It was written that way here
before it was measured, and the two ways of measuring it disagree: comparing
per-day maxima puts the share 0.8 to 8.3 hours after the rate, while the
cross-correlation is maximised with the share 1.5 hours AHEAD. The per-day
maximum of the rate is a single spiky minute and is not a stable statistic. The
claim that survives both is the one above -- they are only moderately related.

BINNING. Panel (b) is 10-minute bins, which is the alignment
`analyze_mix_over_time.py` and the trace generator both use; at one-minute
resolution the share is dominated by counting noise in the low-rate hours.
Panel (a) stays at one minute, as in the existing figure.

SOURCE WINDOW. Read from `traces/dynamic/canonical/dyn60_azure4d.plan.json`, so
this figure and the trace cannot describe different windows. The two source
traces cover different calendar weeks and are summed by index, which aligns
hour-of-day phase but not calendar date -- the generator's documented caveat.

    python3 paper_figures/fig_azure_rate_and_mix.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from paper_style import COL_W, STYLE, GRID, save  # noqa: E402

PLAN = "traces/dynamic/canonical/dyn60_azure4d.plan.json"
CONV = "traces/azure/plots/_minute_conv2024.csv"
CODE = "traces/azure/plots/_minute_code2024.csv"
FIG_H = 2.45
C_RATE = "#1f77b4"
C_MIX = "#d62728"
BIN = 10          # minutes per bin in panel (b)


def source_window():
    """Per-minute conversation and code counts over the trace's own window."""
    plan = json.load(open(os.path.join(ROOT, PLAN)))
    src = plan["source"]
    day0, day1 = src["day_window"]
    start_hour = src["start_hour"]

    conv = pd.read_csv(os.path.join(ROOT, CONV))["count"].to_numpy(float)
    code = pd.read_csv(os.path.join(ROOT, CODE))["count"].to_numpy(float)
    n = min(len(conv), len(code))
    lo = day0 * 1440 + start_hour * 60
    hi = lo + (day1 - day0) * 1440
    if hi > n:
        sys.exit(f"window {lo}..{hi} exceeds {n} minutes of source")
    return conv[lo:hi], code[lo:hi], plan


def main():
    conv, code, plan = source_window()
    total = conv + code
    hours = np.arange(len(total)) / 60.0

    nb = len(total) // BIN
    cb = conv[:nb * BIN].reshape(nb, BIN).sum(1)
    kb = code[:nb * BIN].reshape(nb, BIN).sum(1)
    share = 100.0 * kb / (cb + kb)
    hb = (np.arange(nb) * BIN + BIN / 2) / 60.0
    # The rate on the same bins, so the correlation compares like with like
    # rather than a one-minute series against a ten-minute one.
    rate_b = (cb + kb) / BIN

    peak = total.max()
    r = np.corrcoef(rate_b, share)[0, 1]
    print(f"{len(total)} minutes ({len(total)/1440:.1f} days), {nb} bins of {BIN} min")
    print(f"  rate      trough {total.min()/peak:.3f} of peak, peak/trough "
          f"{peak/total.min():.1f}x")
    print(f"  code share  min {share.min():.1f}%  p5 {np.percentile(share,5):.1f}%  "
          f"median {np.median(share):.1f}%  p95 {np.percentile(share,95):.1f}%  "
          f"max {share.max():.1f}%   p95/p5 {np.percentile(share,95)/np.percentile(share,5):.2f}x")
    print(f"  correlation between binned rate and code share: {r:+.3f}")

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(2, 1, figsize=(COL_W, FIG_H), sharex=True)

        ax[0].plot(hours, total / peak, color=C_RATE, lw=0.5)
        ax[0].fill_between(hours, total / peak, color=C_RATE, alpha=0.15, lw=0)
        ax[0].axhline(1.0, color="#555555", lw=0.6, ls="--")
        ax[0].axhline(total.min() / peak, color="#555555", lw=0.6, ls="--")
        ax[0].annotate(f"peak / trough = {peak/total.min():.1f}$\\times$",
                       (0.5, 0.58), xycoords="axes fraction", ha="center",
                       fontsize=7, color="#333333")
        ax[0].set_ylabel("Arrival rate\n(fraction of peak)")
        ax[0].set_ylim(0, 1.12)
        ax[0].set_yticks([0, 0.5, 1.0])

        ax[1].plot(hb, share, color=C_MIX, lw=0.6)
        ax[1].fill_between(hb, share, color=C_MIX, alpha=0.15, lw=0)
        # The band the share moves through, which is the claim, rather than the
        # instants at which the extremes occur.
        ax[1].axhline(share.max(), color="#555555", lw=0.6, ls="--")
        ax[1].axhline(share.min(), color="#555555", lw=0.6, ls="--")
        ax[1].annotate(f"{share.min():.0f}% – {share.max():.0f}% of arrivals",
                       (0.5, 0.12), xycoords="axes fraction", ha="center",
                       fontsize=7, color="#333333")
        ax[1].set_ylabel("Code requests\n(% of arrivals)")
        ax[1].set_ylim(0, 100)
        ax[1].set_yticks([0, 50, 100])

        for a in ax:
            a.grid(axis="both", **GRID)
            a.set_axisbelow(True)
            a.set_xlim(0, hours[-1])
            a.set_xticks(np.arange(0, hours[-1] + 1, 24))
        ax[1].set_xlabel("Time (hours)")

        fig.tight_layout(pad=0.35)
        fig.subplots_adjust(hspace=0.18)
        save(fig, os.path.join(HERE, "azure_rate_and_mix.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
