#!/usr/bin/env python3
"""Paper figure: in the source trace the arrival rate and the class composition both move, and they move apart.

  azure_rate_and_mix.pdf    3.335 x 2.45 in, `figure`, width=\\columnwidth

Two panels over the SAME four-day window of the Azure LLM Inference 2024 traces.

  (a) arrival rate, divided by the window's own peak
  (b) what share of those arrivals is off the window's average composition, in
      10-minute bins

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

WHAT PANEL (b) PLOTS. The total-variation distance between the composition in
that bin and the mean composition of the whole window, expressed as a percentage.

Total variation has one plain reading and it is the reason this quantity is worth
drawing: it is **the share of the arrivals that would have to change class for
the current mix to equal the average mix**. Equivalently, summed over classes, it
is the excess share of the classes that are over-represented right now. So a
reading of 40% means two in five arrivals are of a class the average composition
does not have room for -- and that is the same two in five a partition sized for
the average would put in the wrong place.

The average is used as the reference because it is the natural fixed choice: a
deployment that must pick one composition and hold it picks the typical one. The
panel is therefore the error such a choice carries, moment by moment, rather than
a property of the trace that has to be argued into relevance. (It is not claimed
to be the error-minimising fixed choice; that would be a different statistic and
is not needed for the point.)

WHY NOT A COMPOSITION BREAKDOWN. Two further reasons.

  1. IT DOES NOT DEPEND ON THE NUMBER OF CLASSES. Azure publishes two request
     types; our workload has three. Drawing Azure's two-way split invites the
     reader to map it onto our three classes, which is exactly the mapping that
     does not exist -- there is no deep-research analogue in that release. A
     share-to-reassign is comparable across both.
  2. THE CLAIM IS THAT THE MIX MOVES, not that it moves toward code. An earlier
     version of this file argued the opposite -- that an index loses the
     direction and the direction is what matters for sizing a partition. That
     objection is about a different claim. Direction matters when asking WHICH
     partition to hold; this figure is establishing that NO fixed one is right,
     and for that the share-to-reassign is the whole content.

The shares themselves are not lost: they are printed by this script and recorded
in the README, so a sentence needing "code runs from 3.9% to 68.0% of arrivals"
still has its source.

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
# Deliberately NOT one of the arm colours in paper_style.ARM_COLOR. Those are
# bound to control planes, and this figure is a property of the workload; reusing
# PolyServe's red here made the panel read as if a policy were being shown, quite
# apart from being too saturated to sit next to panel (a).
C_MIX = "#55606e"
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
    # The effective number of classes, 1/sum(s^2). Reported rather than drawn --
    # see the docstring. With two classes it is symmetric about a 50/50 split, so
    # it cannot be inverted back to a composition.
    f = share / 100.0
    neff = 1.0 / (f ** 2 + (1 - f) ** 2)
    print(f"{len(total)} minutes ({len(total)/1440:.1f} days), {nb} bins of {BIN} min")
    print(f"  rate      trough {total.min()/peak:.3f} of peak, peak/trough "
          f"{peak/total.min():.1f}x")
    print(f"  code share  min {share.min():.1f}%  p5 {np.percentile(share,5):.1f}%  "
          f"median {np.median(share):.1f}%  p95 {np.percentile(share,95):.1f}%  "
          f"max {share.max():.1f}%   p95/p5 {np.percentile(share,95)/np.percentile(share,5):.2f}x")
    print(f"  correlation between binned rate and code share: {r:+.3f}")
    print(f"  effective number of classes 1/sum(s^2): min {neff.min():.2f}  "
          f"median {np.median(neff):.2f}  max {neff.max():.2f}  (2 classes, so 1.00-2.00)")
    # Total variation between each bin's composition and the window mean. With
    # two classes 0.5*(|d| + |-d|) reduces to |d|, and the expression below is
    # the general one so a three-class source needs no change here.
    comp = np.column_stack([1.0 - f, f])
    tv = 100.0 * 0.5 * np.abs(comp - comp.mean(axis=0)).sum(axis=1)
    print(f"  arrivals off the average mix: median {np.median(tv):.1f}%  "
          f"p95 {np.percentile(tv,95):.1f}%  max {tv.max():.1f}%"
          f"   (the share that would have to change class to match the average)")

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(2, 1, figsize=(COL_W, FIG_H), sharex=True)

        ax[0].plot(hours, total / peak, color=C_RATE, lw=0.5)
        ax[0].fill_between(hours, total / peak, color=C_RATE, alpha=0.15, lw=0)
        ax[0].axhline(1.0, color="#555555", lw=0.6, ls="--")
        ax[0].axhline(total.min() / peak, color="#555555", lw=0.6, ls="--")
        ax[0].annotate(f"peak / trough = {peak/total.min():.1f}$\\times$",
                       (0.5, 0.58), xycoords="axes fraction", ha="center",
                       fontsize=7, color="#333333")
        ax[0].set_ylabel("Arrival rate\n/ peak")
        ax[0].set_ylim(0, 1.12)
        ax[0].set_yticks([0, 0.5, 1.0])

        ax[1].plot(hb, tv, color=C_MIX, lw=0.6)
        ax[1].fill_between(hb, tv, color=C_MIX, alpha=0.18, lw=0)
        ax[1].axhline(tv.max(), color="#333333", lw=0.6, ls="--")
        ax[1].annotate(f"up to {tv.max():.0f}% of arrivals",
                       (0.5, 0.63), xycoords="axes fraction", ha="center",
                       fontsize=7, color="#333333")
        ax[1].set_ylabel("Arrivals off the\naverage mix (%)")
        ax[1].set_ylim(0, max(50.0, tv.max() * 1.25))
        ax[1].set_yticks([0, 20, 40])

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
