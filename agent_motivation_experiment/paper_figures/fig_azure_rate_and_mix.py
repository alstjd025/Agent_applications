#!/usr/bin/env python3
"""Paper figure: in the source trace the arrival rate and the class composition both move, and they move apart.

  azure_rate_and_mix.pdf     3.335 x 2.45 in, `figure`, width=\\columnwidth
  azure_rate_and_mix_lr.pdf  3.335 x 1.42 in, the same two panels side by side
  azure_mix_only.pdf         3.335 x 1.40 in, panel (b) alone

Two panels over the SAME four-day window of the Azure LLM Inference 2024 traces.

  (a) arrival rate, normalized -- divided by the window's own peak, so the
      upper dashed rule is 1.0 by construction
  (b) what share of those arrivals is off the window's average composition, in
      10-minute bins

`azure_mix_only.pdf` IS PANEL (b) ALONE, for a place in the text that makes only
the composition half of the claim. It is the same series over the same window, so
⚠ EVERY CAVEAT BELOW STILL APPLIES, and one more: with the rate panel gone the
figure no longer shows that this window's load moves by 5.8x, so a caption that
needs the two-part claim has to say the rate range in words or use the stacked
version. The panel also cannot be read as "the mix moves when the load does" --
that comparison is what the stacked layout exists for.

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

THE AXIS IS NOT LABELLED "VARIANCE", and the near-miss is worth stating because
the source statistic is called total VARIATION. This is not a variance: it is not
a second moment, and its units are a share of arrivals rather than the square of
one. Labelling it variance would read as a contraction of "total variation" and
would stop any reader who knows the difference, asking the variance of what over
what. "Deviation" is the ordinary word for a departure from a reference and needs
no such question.

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
FIG_H_SHORT = 1.95
# The mixture panel on its own. One panel carries the x axis and its label that
# the stacked pair shares, so it cannot simply be half of 2.45.
# ⚠ THE HEIGHT IS SET BY THE ROTATED Y LABEL AND NOT BY THE CURVE. "Workload
# mixture / deviation (%)" is 1.01 in of ink measured along the axis, and a
# rotated label is centred on the axes and NOT clipped to the canvas: at 1.30 in
# the axes are 0.91 in and the label's first line ran 0.01 in off the top of the
# page, silently. 1.40 in puts the axes at 1.01 in and the label's top at 1.36
# against a 1.40 in canvas. `check_ylabel` measures it on every build so a
# future height change says so instead of shipping a cut label.
FIG_H_MIX = 1.40
C_RATE = "#1f77b4"
# A muted terracotta, and the two nearby colours it deliberately is not.
# paper_style.ARM_COLOR binds #d62728 to PolyServe and #ff7f0e to an ablation arm,
# and exp27_figures uses orange for the deep-research CLASS, so either of those
# read here as if a policy or a class were being shown. This figure is a property
# of the workload. Desaturated so it sits beside panel (a)'s blue rather than
# competing with it: the first attempt used PolyServe's red straight and the
# panel drew the eye away from the one above it.
C_MIX = "#b56349"
# `azure_mix_only.pdf` ONLY (2026-09-11, at the author's request): an orange,
# #e08214 -- ColorBrewer PuOr, the same one the author picked on the same day for
# the attainment curve in `load_engine_tbt.pdf`.
#
# NOT #ff7f0e AND NOT #d62728. The note above this one records why: paper_style
# binds #d62728 to PolyServe and #ff7f0e to an ablation arm, and exp27_figures
# gives orange to the deep research CLASS, so either of those reads here as if a
# policy or a class were being drawn. This panel is a property of the workload.
#
# TWO QUANTITIES NOW SHARE #e08214 -- this panel and that attainment curve. They
# are in different figures and nothing else connects them, but a page carrying
# both should not let the colour suggest otherwise. #fe9929 (ColorBrewer YlOrBr)
# is the nearest orange that keeps them apart, if that ever matters.
#
# THE AREA IS THE LINE'S OWN COLOUR AT AN ALPHA, not a second step of the ramp.
# Two steps were tried -- #ffffd4 and then #a1dab4 under the line -- and both
# read as two quantities rather than one quantity and the area under it. One hue
# at two strengths is what panel (a) does with C_RATE, and it is what the
# terracotta version of this panel did before any of this.
# The other three outputs keep C_MIX, because there the mixture panel sits
# beside the rate panel and the terracotta was chosen to sit beside that panel's
# blue -- see the note above. So the same panel is two colours depending on
# which file it is in, and a caption that shows both has to say so.
#
# THE AREA IS OPAQUE, not C_MIX's alpha 0.20, because the colour asked for is a
# value the area should actually take. An opaque area would hide the dotted grid
# underneath it, so the area alone is dropped below the gridlines (zorder 0
# against the 0.5 `set_axisbelow(True)` gives them); the line stays above both.
# Env-overridable only so a variant can be rendered during a review
# without editing the file; the default here is what the figure ships.
C_MIX_ONLY = os.environ.get("FS_MIX_COLOR", "#e08214")
# The one number to turn if the area is too faint or too heavy. Panel (a) fills
# at 0.15 and the terracotta mixture panel at 0.20.
C_MIX_ONLY_ALPHA = float(os.environ.get("FS_MIX_ALPHA", 0.22))
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

    draw(hours, total, peak, hb, tv, "stack",
         os.path.join(HERE, "azure_rate_and_mix.pdf"))
    draw(hours, total, peak, hb, tv, "stack",
         os.path.join(HERE, "azure_rate_and_mix_short.pdf"), height=FIG_H_SHORT)
    draw(hours, total, peak, hb, tv, "side",
         os.path.join(HERE, "azure_rate_and_mix_lr.pdf"))
    # The mixture panel alone, for the places that make only the composition
    # claim. ⚠ IT IS THE SAME PANEL AND THE SAME DATA -- the deviation is
    # measured against the mean composition of this window, so a figure that
    # drops the rate panel still describes a window whose load moves by
    # 15.6x, and the caption has to say which window it is.
    draw(hours, total, peak, hb, tv, "mix",
         os.path.join(HERE, "azure_mix_only.pdf"))
    return 0


def span(ax, x, y, lo, hi, label):
    """A double-headed arrow between two horizontal rules, labelled beside it.

    WHERE IT IS PUT IS COMPUTED, NOT CHOSEN. The arrow spans the full height
    between the two rules, so it crosses the series wherever it is placed; what
    can be avoided is placing it where the series is HIGH, which is where the
    label would sit on top of a peak. The x is therefore the point of the
    series' own minimum within the middle half of the window -- middle half so
    the label, which is drawn to the RIGHT of the arrow, does not run off the
    axis, and the series' own minimum so the crossing happens where the curve
    is thin and flat.

    WHICH SIDE THE LABEL GOES ON follows from that x: to the right of the arrow
    in the first half of the window and to the left in the second. Both minima
    here land past the middle, and "up to 40% of arrivals" placed to the right
    of hour 62 runs past the axis -- matplotlib does not clip a text artist to
    its axes, so it would simply be drawn over the frame and, with
    `bbox_inches` off, off the canvas.

    THE LABEL SITS NEAR THE TOP OF THE ARROW, NOT AT ITS MIDDLE, AND HAS NO
    BACKGROUND. An opaque background would keep the glyphs readable over the
    filled area, but it ERASES what it covers: at the arrow's midpoint it was
    cutting the tops off two peaks of the mixture series, which is worse than
    an unreadable label because the reader cannot tell anything is missing. At
    82% of the way up, both series are below the label everywhere it reaches,
    so no background is needed.

    ⚠ 82% IS A PROPERTY OF THESE TWO SERIES AT THESE PANEL HEIGHTS, not a rule.
    A label's height in DATA units grows as the panel shrinks, so a shorter
    figure can push the label down onto a peak that it cleared before. That is
    why `check_overlap` measures the drawn text against the series and prints
    what it finds, rather than the placement being trusted because it looked
    right once.
    """
    n = len(x)
    seg = slice(n // 4, 3 * n // 4)
    xi = x[seg][int(np.argmin(np.asarray(y)[seg]))]
    ax.annotate("", xy=(xi, hi), xytext=(xi, lo),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="#333333",
                                shrinkA=0, shrinkB=0, mutation_scale=6))
    right = xi < 0.5 * x[-1]
    return ax.text(xi + (0.025 if right else -0.025) * x[-1],
                   lo + 0.82 * (hi - lo), label,
                   ha="left" if right else "right", va="center", fontsize=7,
                   color="#333333")


def check_overlap(fig, ax, txt, x, y, name):
    """Does the drawn label cross the series it is drawn over?

    Called after `tight_layout`, because that is what fixes the axes box and
    therefore what the text's extent in DATA coordinates depends on. The text
    box is converted to data coordinates and compared against the series over
    the same x range; anything that reaches into the box is reported with the
    margin, so a placement that stops working at a new figure height says so
    instead of being found by eye later, or not at all.
    """
    box = txt.get_window_extent(fig.canvas.get_renderer())
    (x0, y0), (x1, y1) = ax.transData.inverted().transform(
        [[box.x0, box.y0], [box.x1, box.y1]])
    x, y = np.asarray(x), np.asarray(y)
    m = (x >= x0) & (x <= x1)
    if not m.any():
        return
    top = float(y[m].max())
    if top >= y0:
        print(f"  ⚠ {name}: the label overlaps the series by "
              f"{top - y0:.3g} (label bottom {y0:.3g}, series top {top:.3g} "
              f"over {x0:.1f}..{x1:.1f})")
    else:
        print(f"  {name}: label clears the series by {y0 - top:.3g} "
              f"(label bottom {y0:.3g}, series top {top:.3g})")


def check_ylabel(fig, ax, name):
    """Does the rotated y label fit on the canvas?

    A y label is centred on its axes and drawn outside them, and matplotlib
    clips neither to the figure. With `bbox_inches` off -- which this directory
    requires, so that the PDF is the size it was drawn at -- a label taller than
    the axes simply loses its ends off the page and the script says nothing.
    """
    fig.canvas.draw()
    b = ax.yaxis.label.get_window_extent(fig.canvas.get_renderer())
    h = fig.get_size_inches()[1] * fig.dpi
    if b.y0 < 0 or b.y1 > h:
        print(f"  ⚠ {name}: the y label runs off the canvas by "
              f"{max(-b.y0, b.y1 - h) / fig.dpi:.3f} in "
              f"(label {b.height / fig.dpi:.2f} in tall)")
    else:
        print(f"  {name}: y label fits with "
              f"{min(b.y0, h - b.y1) / fig.dpi:.3f} in to spare")


def draw(hours, total, peak, hb, tv, layout, out, height=None):
    """The same two panels stacked (one above the other) or side by side.

    BOTH FIT IN ONE COLUMN. Stacked, the two panels share one x axis and one
    set of hour labels, which is why that version can afford 24-hour ticks and
    2.45 in of height. Side by side, each panel is about 1.35 in wide and needs
    its own x axis, so the ticks go to 48 hours -- five hour labels under 1.35
    in of axes run together -- and the annotations are shortened for the same
    reason. THE SIDE-BY-SIDE VERSION THEREFORE SHOWS THE SAME DATA AT LOWER
    TIME RESOLUTION IN THE LABELLING, not in the data: every minute and every
    10-minute bin is still drawn.

    ⚠ THE STACKED VERSION IS THE ONE THAT SUPPORTS READING THE TWO PANELS
    AGAINST EACH OTHER IN TIME. Vertically aligned panels put the same hour at
    the same horizontal position, so a reader can drop a line from a peak in
    the rate to the mixture below it. Side by side, that comparison requires
    measuring from two different origins, and the figure's claim is precisely
    that the two do NOT track each other (correlation +0.57). If the caption
    makes that claim, the stacked version is the one to include.
    """
    side = layout == "side"
    mix_only = layout == "mix"
    tight = not side and not mix_only and (height or FIG_H) < FIG_H
    with plt.rc_context(STYLE):
        if mix_only:
            # `ax[0]` is left absent rather than made empty: everything the
            # rate panel draws is then skipped by the same test, and nothing
            # downstream can quietly draw into a panel that is not there.
            fig, a1 = plt.subplots(figsize=(COL_W, height or FIG_H_MIX))
            ax = [None, a1]
        elif side:
            fig, ax = plt.subplots(1, 2, figsize=(COL_W, height or 1.42))
        else:
            fig, ax = plt.subplots(2, 1, figsize=(COL_W, height or FIG_H),
                                   sharex=True)

        if not mix_only:
            ax[0].plot(hours, total / peak, color=C_RATE, lw=0.5)
            ax[0].fill_between(hours, total / peak, color=C_RATE, alpha=0.15, lw=0)
            ax[0].axhline(1.0, color="#555555", lw=0.6, ls="--")
            ax[0].axhline(total.min() / peak, color="#555555", lw=0.6, ls="--")
            t0 = span(ax[0], hours, total / peak, total.min() / peak, 1.0,
                      (f"{peak/total.min():.1f}$\\times$" if side else
                       f"peak / trough = {peak/total.min():.1f}$\\times$"))
            # ONE LINE WHERE IT FITS AND TWO WHERE IT DOES NOT, same words either
            # way. A rotated axis label is not clipped to its panel, and this one
            # is 1.11 in of ink against a panel that is 0.95 in at the tall stacked
            # height and 0.71 in at the short one; the tall version already spends
            # its whole top margin on it (0.013 in left) and every height from 2.35
            # down was measured with the label's ink at row 0, that is, cut off. At
            # 8-9 pt the only way to keep one line at the short height would be to
            # take the type below the paper's floor, so the short version breaks
            # the line instead.
            ax[0].set_ylabel("Arrival rate (norm.)" if not tight
                             else "Arrival rate\n(norm.)")
            ax[0].set_ylim(0, 1.12)
            ax[0].set_yticks([0, 0.5, 1.0])

        # One hue for both; only the strength differs. A translucent area lets
        # the dotted grid through on its own, so the zorder juggling an opaque
        # one needed is gone with it.
        mix_c = C_MIX_ONLY if mix_only else C_MIX
        mix_a = C_MIX_ONLY_ALPHA if mix_only else 0.20
        ax[1].plot(hb, tv, color=mix_c, lw=0.6)
        ax[1].fill_between(hb, tv, color=mix_c, alpha=mix_a, lw=0)
        ax[1].axhline(tv.max(), color="#333333", lw=0.6, ls="--")
        # The lower end is 0, not tv.min(): the claim the arrow carries is the
        # RANGE the deviation covers, and the trace does come within 0.4 points
        # of zero, so drawing from the observed minimum would put the arrow's
        # tail on a value no reader can see and read as if it started above the
        # axis. ⚠ THE ARROW THEREFORE MEANS "up to", not "between x and y".
        t1 = span(ax[1], hb, tv, 0.0, tv.max(),
                  (f"up to {tv.max():.0f}%" if side else
                   f"up to {tv.max():.0f}% of arrivals"))
        ax[1].set_ylabel("Workload mixture\ndeviation (%)")
        ax[1].set_ylim(0, max(50.0, tv.max() * 1.25))
        ax[1].set_yticks([0, 20, 40])

        for a in ax:
            if a is None:
                continue
            a.grid(axis="both", **GRID)
            a.set_axisbelow(True)
            a.set_xlim(0, hours[-1])
            a.set_xticks(np.arange(0, hours[-1] + 1, 48 if side else 24))
        if side:
            for a in ax:
                a.set_xlabel("Time (hours)", labelpad=1.5)
        else:
            ax[1].set_xlabel("Time (hours)")

        fig.tight_layout(pad=0.35, **({"w_pad": 1.2} if side else {}))
        if not side and not mix_only:
            fig.subplots_adjust(hspace=0.18)
        print(f"{os.path.basename(out)}")
        if not mix_only:
            check_overlap(fig, ax[0], t0, hours, total / peak, "arrival rate")
        check_overlap(fig, ax[1], t1, hb, tv, "mixture deviation")
        if mix_only:
            check_ylabel(fig, ax[1], "mixture deviation")
        save(fig, out)


if __name__ == "__main__":
    sys.exit(main())
