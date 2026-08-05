#!/usr/bin/env python3
"""Paper figure: three policies, comparable tokens produced, 15.2x apart in useful ones.

  motivation_throughput_vs_goodput.pdf   7.0 x 1.62 in, `figure*`, width=\\textwidth

The paper version of `results/aggregate_analysis/motivation/`'s figure of the
same name. Same data and same panels; every annotation is gone — the title
paragraph, the shaded "all three the same" bands, the pointing arrows, the
in-bar "thrown away" labels and the numbers under the bars. What those said
belongs in the caption, where a reader can disagree with it.

  (a) output tokens per second the engines produced
  (b) how much of that belonged to a request that finished inside its rule
  (c) the same question counted in REQUESTS, on both denominators

(a) AND (b) SHARE A Y AXIS ON PURPOSE. The claim is that the first is nearly the
same across policies and the second is not; rescaling either would hide it. At
70 req/s the three produce 20,057 / 20,212 / 13,563 output tokens per second and
turn 710 / 9,064 / 10,811 of them into requests that met their rule.

THE CORRECTED POLYSERVE WEAKENS ONE HALF OF THIS AND STRENGTHENS THE OTHER, so
read the panels before reusing an older sentence about them. On (a) the arm now
tracks the load balancer to within 1% at the top of the range instead of falling
5,000 tokens/s below it, which makes "the engines are equally busy" a stronger
statement than it was. On (b) it retains about 45% of what it produces rather
than 20%, so it is no longer an example of a policy whose output is almost
entirely wasted; the only arm of which that remains true is the load balancer,
at 3.5%. The 15.2x spread is now between the load balancer and Llumnix SLO.

(c) IS NOT A RESTATEMENT OF (b). (b) weights each request by the tokens it
produced, so a policy that keeps long requests and refuses short ones scores
well on it; (c) counts each request once. At 70 req/s the spread across the
three policies is 15.2x on (b) and 6.2x on (c), and the two panels do not even
rank the policies the same way: Llumnix SLO produces the most useful tokens
(10,811/s against PolyServe's 9,064) while PolyServe returns the larger share of
the requests that were sent (24.0% against 21.4%). Llumnix SLO rejects 69.2%
there, so its tokens come from under a third of the arrivals.

(c) therefore carries both denominators: solid counts every request that
arrived, so a rejection is a violation; dotted counts only the requests the
policy accepted. Only Llumnix SLO rejects, so only its accepted line is drawn
and named -- the other two never reject and their two lines coincide, so a
second curve on top of the first would assert a distinction the data does not
contain. Llumnix SLO's dotted line RISES from 56.8% to 71.3% between 45
and 70 req/s while its solid line falls from 52.4% to 21.4%. The other two never
reject, so their two lines coincide.

FluidServe is deliberately absent. A motivation figure that needs the paper's
own system to make its point is not a motivation figure.

WHAT THE CAPTION HAS TO CARRY, because the annotations that used to say it are
gone: at 25 req/s and below the three policies are within 0.2 points of each
other on all three panels, and the first rate at which they separate is 35,
where throughput still spans only 6% and goodput already spans 2.4x; at 70 req/s
Llumnix (load balance) produces 20,057 output tokens/s of which 710 are useful;
Llumnix SLO rejects 69.2%, which is the whole distance between its two lines in
(c).

DATA. Both repeats of every condition, stock vLLM FIFO engine, migration off for
PolyServe and on for the two Llumnix arms. The Llumnix SLO arm uses the `m1f`
workload configuration, which splits the agent class's 30 s end-to-end budget
into the (first-token, per-token) pair that policy requires.

The two Llumnix arms come from EXP-53 and PolyServe from EXP-57, which
re-measured that arm alone after `--polyserve-tier-decode-tokens` was corrected
on 2026-08-05: it configured deep research at 275 expected output tokens against
a measured 985, a factor of 3.58, and that number feeds both the admission limit
and the repartitioner. The correction is large — at 70 req/s the arm goes from
14.4% to 24.0% of requests inside their rule and from 3,965 to 10,220 tokens/s
of goodput — and it changes which policy this figure shows highest on (c), so
any earlier version of this figure is superseded rather than approximate.
Joining the two sessions was checked: Llumnix SLO, unchanged, was re-run in the
EXP-57 session at 45 and 70 req/s and read 52.3 and 21.7 against EXP-53's 52.4
and 21.4.

    python3 paper_figures/fig_motivation_throughput_goodput.py
"""
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis_scripts", "request_level"))

from paper_style import TEXT_W, STYLE, GRID, kfmt, save  # noqa: E402
from motivation_fig1 import collect, ARMS, _supersede  # noqa: E402

# Legend text, overriding the names `motivation_fig1` uses. The load-balancing
# arm is written plainly as "Llumnix" here. NOTE that this leaves two entries
# beginning with the same word, so the caption has to say that "Llumnix" is the
# shipped load-balancing policy and "Llumnix SLO" its SLO-aware one.
LABEL = {"loadbalance": "Llumnix"}

FIG_H = 1.62
# Bold, and written as mathtext because the panel title is the second line of
# the x label (see below) and a Text artist carries one weight for the whole
# string -- there is no way to bold half of it. `\mathbf` bolds just this line
# while the rate above it stays regular. The interword spaces have to be
# written `\ ` because mathtext discards ordinary spaces.
TITLES = [r"$\mathbf{(a)\ Throughput}$",
          r"$\mathbf{(b)\ Goodput\ Tokens}$",
          r"$\mathbf{(c)\ Request\ SLO}$"]
# Without this mathtext would set the bold line in DejaVu Sans while the rest
# of the figure is DejaVu Serif, so the three titles would be in a different
# typeface from every other word.
MATH_SERIF = {"mathtext.fontset": "dejavuserif"}


def main():
    os.chdir(ROOT)
    # `collect()`'s default glob is `results/*exp53*_rpm_*`, which also matches
    # `exp53smoker1_loadbalance_m1_rpm_2700` -- a smoke condition, pooled with
    # the two real repeats and making that one cell an n=3 average of two
    # measurements and a shakedown. Excluded here; the exploratory figure in
    # results/aggregate_analysis/motivation still includes it.
    # The two repeats named explicitly, rather than `collect()`'s default glob
    # `results/*exp53*_rpm_*`. That default also matches
    # `exp53smoker1_loadbalance_m1_rpm_2700` -- a smoke condition -- which it
    # pools with the two real repeats, making that one cell an n=3 average of
    # two measurements and a shakedown. `exp53r1*` also picks up the `exp53r1r1`
    # top-up that refilled four repeat-1 cells lost to an engine that did not
    # come up.
    # EXP-57 is the third pattern, and the supersede below is what makes it
    # replace rather than join. PolyServe's tier length table was corrected on
    # 2026-08-05 and that arm alone re-measured; `collect()` tags each row with
    # the session it came from and `_supersede` drops the EXP-53 PolyServe rows
    # when EXP-57 ones are present, printing what it dropped. It runs inside
    # each `collect()` call as well, where it is a no-op because no single
    # pattern here contains both sessions -- it has to be applied again to the
    # concatenation, which is the only frame that does.
    df = _supersede(pd.concat([collect("results/*exp53r1*_rpm_*"),
                               collect("results/*exp53p2*_rpm_*"),
                               collect("results/*exp57r*_rpm_*")],
                              ignore_index=True))
    if df.empty:
        print("no EXP-53 runs matched", file=sys.stderr)
        return 1
    g = df.groupby(["arm", "rate"]).agg(
        thru=("thru", "mean"), thruLo=("thru", "min"), thruHi=("thru", "max"),
        good=("good", "mean"), goodLo=("good", "min"), goodHi=("good", "max"),
        off=("att_off", "mean"), adm=("att_adm", "mean"),
        rej=("rej", "mean"),
    ).reset_index()
    print(f"{len(df)} conditions, repeats per cell: "
          f"{sorted(set(df.groupby(['arm', 'rate']).size()))}")

    with plt.rc_context({**STYLE, **MATH_SERIF}):
        fig, ax = plt.subplots(1, 3, figsize=(TEXT_W, FIG_H))
        handles, labels = [], []
        adm_key = None

        for key, lab, col in ARMS:
            s = g[g.arm == key].sort_values("rate")
            mk = dict(marker="o", ms=2.8, mec="white", mew=0.4)
            h, = ax[0].plot(s.rate, s.thru, color=col, lw=1.2, **mk)
            ax[0].fill_between(s.rate, s.thruLo, s.thruHi, color=col,
                               alpha=0.18, lw=0)
            ax[1].plot(s.rate, s.good, color=col, lw=1.2, **mk)
            ax[1].fill_between(s.rate, s.goodLo, s.goodHi, color=col,
                               alpha=0.18, lw=0)
            ax[2].plot(s.rate, s.off, color=col, lw=1.2, **mk)
            # The accepted-set denominator, drawn ONLY for an arm that
            # actually rejects. Llumnix and PolyServe reject nothing at any
            # rate here, so their accepted line is their offered line; drawing
            # it would put a second curve exactly on top of the first and
            # assert a distinction the data does not contain.
            adm_handle = None
            if float(s.rej.max()) > 0:
                adm_handle, = ax[2].plot(s.rate, s.adm, color=col, ls=":",
                                         lw=1.1, marker="x", ms=3.2, mew=0.9)
            handles.append(h)
            labels.append(LABEL.get(key, lab))
            # NOT registered in the figure-level legend. The top legend names
            # the three policies; this curve is not a fourth policy but the
            # same policy counted on a second denominator, and it exists in
            # panel (c) only. It gets its own key inside that panel instead.
            if adm_handle is not None:
                adm_key = (adm_handle, "Admitted")

        # (b) is read against (a), so they share a y axis.
        # Both keep their tick numbers and their name: side by side as separate
        # rectangles, a panel with no y axis of its own reads as a continuation
        # of the one left of it rather than a second measurement.
        ax[1].sharey(ax[0])

        for i in (0, 1, 2):
            # The panel title is the SECOND LINE of the x label, not a text
            # box placed under the axes: `tight_layout` reserves room for an
            # axis label and does not know about an artist positioned in axes
            # coordinates, so the hand-placed version left the panels crushed
            # into the top of the canvas with the titles overlapping them.
            ax[i].set_xlabel(f"Offered rate (req/s)\n{TITLES[i]}",
                             labelpad=1.5, linespacing=1.6)
            ax[i].set_xlim(13, 72)
            ax[i].set_xticks([20, 30, 40, 50, 60, 70])
            ax[i].grid(axis="both", **GRID)
            ax[i].set_axisbelow(True)
        ax[0].set_ylim(0, 22000)
        ax[0].set_yticks([0, 5000, 10000, 15000, 20000])
        ax[0].yaxis.set_major_formatter(kfmt())
        ax[0].set_ylabel("Tokens/s")
        ax[1].set_ylabel("Tokens/s")
        ax[2].set_ylim(0, 105)
        ax[2].set_yticks([0, 25, 50, 75, 100])
        ax[2].set_ylabel("SLO attainment (%)")

        # The second denominator is kept out of the figure legend, which names
        # the three POLICIES; this is not a fourth policy but one of those
        # three counted a second way, and it exists in (c) only. It sits just
        # above (c)'s frame, right-aligned over the panel it belongs to, in
        # smaller type so it reads as a qualifier rather than a fourth series.
        # The word alone is the label: which arm it qualifies is unambiguous
        # from the colour, and the caption states that the solid curves count
        # every arrival while this one counts only accepted requests.
        if adm_key is not None:
            ax[2].legend([adm_key[0]], [adm_key[1]], loc="lower right",
                         bbox_to_anchor=(1.0, 1.0), fontsize=6.5,
                         frameon=False, handlelength=1.6, handletextpad=0.35,
                         borderaxespad=0.0, borderpad=0.0)

        # One row. Four entries wrap to two at ncol=3 and the upper row
        # lands outside the canvas.
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 0.866), frameon=False,
                   columnspacing=1.4, handlelength=1.8, handletextpad=0.4)
        # w_pad is large on purpose: the canvas is 12% shorter than it was,
        # and widening the gaps narrows the panels by about the same
        # proportion, so each keeps roughly the shape it had rather than
        # being squashed into a letterbox.
        # The canvas height is fixed, so the panels are made taller by taking
        # back the margin above the legend and below the x label rather than
        # by growing the figure. `rect`'s top rises from 0.845 to 0.872 and
        # `pad` (the border the layout keeps clear at the canvas edge) drops
        # from 0.35 to 0.25 of a font size. The legend anchor moves the OTHER
        # way, from 0.855 to 0.866, because raising it with the panels pushed
        # the capitals of the legend text off the top of the canvas.
        #
        # `rect`'s bottom is BELOW the canvas at -0.03 because the bold titles
        # are mathtext, and mathtext reports a box taller than the glyphs it
        # actually draws; laid out inside the canvas the panels lost 15 px of
        # the height this change had just won while 17 px at the bottom stayed
        # empty. The measured result is 47.7% -> 51.5% of the canvas height,
        # about 8% taller, with the lowest ink 5 px clear of the bottom edge
        # at 250 dpi.
        fig.tight_layout(rect=(0, -0.030, 1, 0.872), w_pad=3.0, pad=0.25)
        save(fig, os.path.join(HERE, "motivation_throughput_vs_goodput.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
