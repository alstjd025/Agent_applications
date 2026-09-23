#!/usr/bin/env python3
"""Paper figure: four control planes on the static rate sweep, scored under the
token-level cumulative deadline.

  motivation_throughput_vs_goodput_4panel_t75.pdf         baselines only
  motivation_throughput_vs_goodput_4panel_t75_withfs.pdf  the same plus FluidServe
  7.0 x 1.75 in, `figure*`, width=\\textwidth

  Each PDF is written with a CSV of the same basename holding EXACTLY the values
  drawn in it -- one row per arm and arrival rate, the mean that is the line and
  the min and max that are the band, the number of repeats behind each cell, and
  the scoring rule the numbers were produced under. A figure whose numbers live
  only in a prose table cannot be checked against its own source, and the prose
  table goes stale the first time the figure is redrawn.

⚠ THE `t75` IN THE FILE NAME IS NOT DECORATION. `fig_motivation_throughput_goodput.py`
already writes a four-panel figure called
`motivation_throughput_vs_goodput_4panel.pdf`, drawn from the pinned copy of the
EARLIER static sweep and scored with the agent class on its end-to-end 30 s
budget. Nothing in that figure may sit beside anything in this one, so the two
must not share a file name; `t75` records the form of the agent class's promise
and therefore which of the two a number came from.

  (a) how much of the output belonged to a request that finished inside its rule
  (b) output tokens per second the engines produced
  (c) the same question counted in REQUESTS, over the requests the policy ACCEPTED
  (d) what share of arrivals the policy rejected

WHY THIS PANEL ORDER. (a) is the quantity the figure claims and (b) sits beside
it on the same y axis, so that "the engines produce comparable numbers of tokens
and the share of them worth anything is not comparable" is visible without
rescaling either. (c) and (d) ask the same question counted in requests rather
than tokens, and they are a PAIR: (c)'s denominator is the requests the policy
accepted, which read alone hands the best score to a policy that accepts almost
nothing, and (d) is what makes it readable. The offered denominator is not drawn
because (c) and (d) together carry it -- a curve for it would be the third
presentation of one number.

X AXIS IS LINEAR AND ITS LABELS ARE EVENLY SPACED. The arrival rate is a
quantity, so the distance between 10 and 15 req/s is drawn at half the distance
between 55 and 70; an equal-width categorical axis would assert that those two
steps are the same size. The LABELS are placed at a uniform 10 req/s interval
(10, 20, ..., 70) rather than on the measured rates, because the measured set
(10, 15, 20, 25, 35, 45, 55, 70) has three different step sizes and labelling it
puts four labels in the first third of the axis.
⚠ FOUR OF THOSE SEVEN LABELS ARE AT RATES THAT WERE NEVER MEASURED (30, 40, 50,
60). That is a real cost and it is paid deliberately for an evenly spaced axis:
the measured rates keep an unlabelled MINOR tick, and every curve carries a
marker at each measured rate, so where the data actually is stays visible.

THE SCORING RULE IS NEW AND IT IS THE POINT OF THIS FIGURE. Until now the
per-token half of a class rule was scored as `mean inter-token time over the
request <= budget`, which is Scorpio's rule and, among the nine systems whose
definitions were read for ms_dev/notes/slo-definitions.md, the rule only Scorpio
uses. Here it is a TOKEN-LEVEL CUMULATIVE DEADLINE WITH A TOLERANCE: writing
a_i for the arrival of the i-th output token measured from submission, T for the
class time-to-first-token budget and P for its per-token budget,

    a_1 <= T                                        AND
    |{i : a_i <= T + (i-1) * P}| / N >= 0.90

The first conjunct is kept separate on purpose: with the deadline schedule
anchored at submission, a request that beat T banks the unused time, and a
request whose engine decodes faster than P can absorb a first-token violation of
about N * (P - actual per-token time) before missing any deadline. Keeping
`a_1 <= T` as its own condition removes that.

The cumulative deadline itself is PolyServe's, QoServe's and JITServe's rule.
The 90% tolerance is not in any of those nine papers and is this project's
addition; `--rule` draws the figure under any of the five columns the scorer
produces (mean, cumt, q90, q90end, ft90) and the README records all of them, so
the relaxation is a column difference and not an assertion.

⚠ swe IS SCORED ON A PER-TOKEN RULE HERE FOR THE FIRST TIME. Its promise was an
end-to-end 30 s budget with no per-token term; FluidServe v0.4 restated it as
TTFT 7 s + 75 ms per token, and EXP-108 is the sweep in which every arm was TOLD
that form -- FluidServe through `--fluidserve-class-budgets 25:decode:75`, the
other three through `mix_short_m1_t75fair.json`, whose `slo.swe.tbt_ms` is 75.
NOTHING IN THIS FIGURE MAY SIT BESIDE A NUMBER SCORED AGAINST THE 30 s
END-TO-END RULE, and that includes every earlier version of the motivation
figure.

DATA. EXP-108, the static rate sweep of 2026-08-31: four arms, eight arrival
rates (10, 15, 20, 25, 35, 45, 55, 70 req/s), 8 minutes per condition, two
repeats, four engines, the engine cold-restarted between conditions. 64 result
directories, all of which carry per-token arrival events. The band is min..max
over the two repeats; a cell with no visible band has two repeats that agree,
not one repeat.

⚠ THE vLLM ROUTER COMES FROM A DIFFERENT SESSION, AND THAT IS A REAL JOIN.
EXP-108 has four arms and this is not one of them, so its curve is EXP-77
(2026-08-10): eight arrival rates, two repeats at all but 70 req/s, where the
second repeat's per-token events are gone. It is the one arm that can be
re-scored under a promise it did not run with, because it reads no latency
promise at all -- its routing is prefix-cache affinity and nothing else, so the
requests it saw are identical whichever agent-class budget the workload file
states. What is NOT identical is everything else about the session: three weeks
earlier, a different scheduler binary for the arms it is drawn beside, and the
machine was wiped and restored on 2026-08-28 between the two. Session-to-session
movement on this workload has been measured at up to 4.6 points and that
measurement does not span the restore. THE CAPTION HAS TO SAY THIS.

    python3 paper_figures/fig_motivation_tg_4panel.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from paper_style import (COL_W, TEXT_W, STYLE, GRID, ARM_COLOR, kfmt, save,
                         final, legend_items, KEY_FS)  # noqa: E402

# ⚠ THE SCORING RULE IS FIXED AND THIS SCRIPT DOES NOT OWN IT. Since
# 2026-09-03 every figure in this directory is scored by
# `analysis_scripts/request_level/deadline_ladder_attainment.py`: token i of a
# request is on time if it arrives within TTFT_SLO + i * TBT_SLO of the send,
# counting the first token as i = 0 so its deadline is exactly TTFT_SLO; the
# request is on time if at least 95% of its tokens are; and goodput is counted
# PER TOKEN, so a request that missed the 95% bar still contributes the tokens
# of its own that were on time. This script reads that scorer's table and
# recomputes nothing.
#
# The runs the paper draws are not exactly the runs EXP-108 produced.
# EXP-108's llm-d cell at 10 req/s holds two repeats that disagree by 42 points
# of rejection rate (0.0% and 42.0%) with no cause found; EXP-110 re-measured
# that one arrival rate three more times and got 0.0% every time, so the 42.0%
# repeat is an outlier that occurred once in five runs. The cell is drawn from
# EXP-110 repeats 3 and 5 instead, which keeps n=2 as in every other cell here.
# The full account is in experiments/EXP-110_llmd-lowrate-split.md. The
# substitution is made by `build_paper_ladder_table.py`, not here, so that the
# static figures cannot disagree about which runs the paper shows;
# `exp108_ladder95.csv` beside it holds the unmodified scoring of every EXP-108
# run.
LADDER = os.path.join(ROOT, "results", "aggregate_analysis", "ladder95")
CSV = os.path.join(LADDER, "exp108_paper_ladder95.csv")

# EXP-108's arm directory names, in the order they are drawn and listed. The
# name in the second field is the one the paper uses; the `t75` suffix records
# the form of the agent class's promise and belongs in the README, not on the
# figure, because every arm here carries it.
# Colour AND marker per arm: two channels rather than one, because a reader
# printing in greyscale, or looking at two curves where they cross, has only the
# shape left. The last field is the marker size, which is not uniform on purpose.
#
# ⚠ vLLM-router IS A PENTAGON, NOT THE HEXAGON `fig_intro_capacity.py` GIVES IT.
# A hexagon at this size is a circle whose corners nobody can see and PolyServe
# is the circle, so the pentagon is drawn a point larger than the circle: its
# flat top and single apex are what separate the two, and they only show if the
# glyph is bigger than the circle it sits beside. (A star and then a cross were
# tried; the star blurs once its size comes down and the cross was not wanted.)
# The other four keep that figure's assignment.
#
# The sizes are not uniform. A triangle of side s carries about half the ink of
# a square of side s, and the pentagon needs the extra size to be told from the
# circle at all, so both are drawn larger than the three that need no help.
ARMS = [
    ("vllmcache",        "vLLM-router", ARM_COLOR["vllmrouter"], "p", 2.6),
    ("polyservept75",    "PolyServe",   ARM_COLOR["polyserve"],  "o", 1.8),
    # ⚠ NAMED "Llumnix" AND NOT "Llumnix SLO" (2026-09-11, at the author's
    # request). CLAUDE.md gives that name to the LOAD-BALANCE policy and calls
    # this one "Llumnix SLO"; the short name is unambiguous only while the
    # load-balance arm is absent, which it is from every figure this file
    # writes. The caption has to say `--scheduling-policy slo`.
    ("slot75",           "Llumnix",     ARM_COLOR["slo"],        "^", 2.4),
    ("llmdslot75",       "llm-d",       ARM_COLOR["llmd"],       "D", 1.6),
    ("fsv3capgnofrct75", "FluidServe",  ARM_COLOR["fluidserve"], "s", 1.8),
]
# The vLLM router is not an EXP-108 arm; its rows come from the EXP-77 scoring
# under the same rule. Two files rather than one because they are two sessions,
# and a single file would let that be forgotten.
CSV_VLLM = os.path.join(LADDER, "vllm77_ladder95.csv")
OURS = "FluidServe"
FIG_H = 1.75
# The wide variant. The canvas width is fixed at the text width, so the only
# way to make each panel less square is to give it less height and less gutter:
# at 1.75 in and w_pad 2.0 the four axes boxes are about 1.29 x 1.14 in, which
# reads as square; at 1.52 in and w_pad 0.9 they are about 1.43 x 0.95 in. The
# legend band and the two-line x label below each panel are reserved in INCHES
# in both, so they stay the same physical size rather than shrinking with the
# canvas and colliding with the axes.
FIG_H_WIDE = 1.52
# The one-column variant: the same four panels stacked two by two. At 3.335 in
# the four cannot sit in a row -- each panel would be 0.6 in wide and its y
# label taller than the panel -- so the row becomes a square block, (a) beside
# (b) on the top row so they can still be read against each other on their
# shared y axis, and (c) beside (d) below.
FIG_H_2X2 = 2.85
# `--ramp`: ColorBrewer's five-class PuBuGn in the ARMS order, the same ramp and
# the same order `fig_two_models_hour.py --ramp` uses, so the two figures name
# the arms with one set of colours.
# ⚠ SEQUENTIAL RAMP, UNORDERED ARMS -- the five steps differ mainly in lightness
# and the eye reads an order the arms do not have. And the palest step is all
# but white: the markers and lines here are 0.9 pt, so with the ramp they carry
# a thin grey stroke underneath, which keeps the fill colour exactly as asked
# and keeps the pale end of the ramp on the page.
# ColorBrewer YlGnBu, five classes (2026-09-11, at the author's request; it was
# PuBuGn). Same idea and the same warnings: the steps differ mainly in lightness
# so the eye reads an order the arms do not have, and the pale end is a yellow
# that a 0.9 pt line cannot carry on white paper -- which is what the grey
# stroke under each line is for. FluidServe takes the dark end, #253494.
# ⚠ llm-d IS NOT A STEP OF THE RAMP ANY MORE (2026-09-17, at the author's
# request): it is #c2a5cf, a purple, where the ramp had #2c7fb8. The four
# remaining arms keep their YlGnBu steps.
RAMP = ["#ffffcc", "#a1dab4", "#41b6c4", "#c2a5cf", "#253494"]
W_PAD = 2.0
W_PAD_WIDE = 0.9
LEGEND_BAND_IN = 0.2135          # 1.75 * (1 - 0.878)
RULE = "ladder95"               # the fixed rule; see LADDER above
RATES = [10, 15, 20, 25, 35, 45, 55, 70]        # what was measured
LABEL_TICKS = [10, 20, 30, 40, 50, 60, 70]      # what the axis is labelled with


def collect(_rule=None, _prefix=None):
    """label -> rate -> the five plotted quantities, mean over repeats, with the
    min and max of each so the band is the measured spread.

    Reads the ladder scorer's per-run table. The arrival rate is not a column
    there -- that table is keyed by run -- so it is parsed from the directory
    name, and a run whose name carries no rate is dropped loudly rather than
    silently landing in one bucket with the others."""
    frames = [pd.read_csv(CSV)]
    if os.path.exists(CSV_VLLM):
        frames.append(pd.read_csv(CSV_VLLM))
    else:
        print(f"!! {CSV_VLLM} missing; the vLLM router arm will be absent",
              file=sys.stderr)
    d = pd.concat(frames, ignore_index=True)
    rate = d["run"].str.extract(r"_rpm_(\d+)")[0]
    if rate.isna().any():
        for r in d.loc[rate.isna(), "run"]:
            print(f"!! no _rpm_ in {r}; dropped", file=sys.stderr)
    d = d[rate.notna()].copy()
    d["rate"] = rate[rate.notna()].astype(float) / 60.0
    cols = {"thru": "throughput_tok_s", "good": "goodput_tok_s",
            "off": "offered", "adm": "admitted", "rej": "rejected_pct"}
    out, counts = {}, {}
    for arm, label, _, _, _ in ARMS:
        sub = d[d["arm"] == arm]
        if sub.empty:
            print(f"!! no runs for arm {arm}", file=sys.stderr)
            continue
        per = {}
        for rate_v, g in sub.groupby("rate"):
            per[float(rate_v)] = {k: (g[c].mean(), g[c].min(), g[c].max())
                                  for k, c in cols.items()}
            counts[(label, float(rate_v))] = len(g)
        out[label] = per
    return out, counts


def report(data, counts, rule):
    print(f"rule = {rule}\n")
    print(f"{'arm':13s} {'req/s':>5s} {'n':>2s} {'thru':>7s} {'goodput':>8s} "
          f"{'offered':>8s} {'admitted':>9s} {'rejected':>9s}")
    for _, label, _, _, _ in ARMS:
        if label not in data:
            continue
        for rate in sorted(data[label]):
            v = data[label][rate]
            print(f"{label:13s} {rate:5.0f} {counts[(label, rate)]:2d} "
                  f"{v['thru'][0]:7.0f} {v['good'][0]:8.0f} {v['off'][0]:8.1f} "
                  f"{v['adm'][0]:9.1f} {v['rej'][0]:9.1f}")


# The quantity each panel draws, in panel order, with the column name it takes
# in the CSV written beside the figure. The units are in the names because a
# column called `goodput` alone does not say whether it is tokens or requests.
PANEL_COLS = [("good", "goodput_tok_s"), ("thru", "throughput_tok_s"),
              ("adm", "admitted_pct"), ("rej", "rejected_pct")]


def write_csv(data, labels, counts, rule, out_path):
    """The values the figure draws, beside the figure, under its own basename.

    Only what is drawn: the line (mean over repeats), the band (min and max),
    the number of repeats behind the cell, and the identity of the scoring rule.
    Everything else about these runs is in the per-run scoring table this reads
    from, and duplicating it here would create a second place for it to go
    stale."""
    rows = []
    for _, label, _, _, _ in ARMS:
        if label not in labels or label not in data:
            continue
        for rate in sorted(data[label]):
            v = data[label][rate]
            row = {"arm": label, "req_per_s": rate,
                   "n_repeats": counts[(label, rate)], "rule": rule}
            for key, name in PANEL_COLS:
                row[name], row[f"{name}_min"], row[f"{name}_max"] = v[key]
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"wrote {out_path}  ({len(df)} rows = "
          f"{df['arm'].nunique()} arms x {df['req_per_s'].nunique()} rates)")


def build(data, labels, out_path, legend=True, height=FIG_H, w_pad=W_PAD,
          ylab_c="SLO attainment (%)", grid2x2=False, ramp=False, shrink=0.0):
    """Four panels at equal x spacing. `labels` selects which arms are drawn;
    the legend row is reserved either way so the with- and without-FluidServe
    versions have axes of exactly the same size and can be read against each
    other. `height` and `w_pad` are the only two things the wide variant
    changes -- same data, same styling, same y limits."""
    style = dict(STYLE)
    if shrink:
        for k in ("axes.labelsize", "xtick.labelsize", "ytick.labelsize"):
            style[k] = max(4.0, style[k] - shrink)
    colours = ({lab: c for (_, lab, _, _, _), c in zip(ARMS, RAMP)}
               if ramp else {lab: c for _, lab, c, _, _ in ARMS})
    # Black, not grey, since 2026-09-17 (at the author's request; the key's
    # samples were blackened first and the panels now match). The outline is
    # what keeps the pale end of the ramp -- the vLLM router's #ffffcc -- on
    # white paper at 0.9 pt.
    stroke = ([pe.Stroke(linewidth=1.7, foreground="#000000"), pe.Normal()]
              if ramp else None)
    with plt.rc_context(style):
        if grid2x2:
            fig, axes = plt.subplots(2, 2, figsize=(COL_W, height))
            ax = axes.ravel()
        else:
            fig, ax = plt.subplots(1, 4, figsize=(TEXT_W, height))
        handles, drawn, no_reject = [], [], []
        for _, label, col, marker, msz in ARMS:
            col = colours[label]
            if label not in labels or label not in data:
                continue
            rates = sorted(data[label])
            x = rates
            # No white keyline around the marker: at this size the halo eats
            # into the glyph and a triangle or a star loses the outline that
            # identifies it.
            mk = dict(marker=marker, ms=msz)

            def series(key):
                return ([data[label][r][key][0] for r in rates],
                        [data[label][r][key][1] for r in rates],
                        [data[label][r][key][2] for r in rates])

            for i, key in enumerate(("good", "thru", "adm", "rej")):
                # The line only. The min..max band over the two repeats was
                # drawn here until 2026-09-03 and is not: at four panels across
                # seven inches five bands overlap in the region the figure is
                # read for, and the spread they carried is in the CSV beside
                # this file, as `*_min` and `*_max` on every plotted quantity.
                m, _, _ = series(key)
                if key == "rej" and max(m) == 0.0:
                    # An arm that refuses nothing at any rate lies on the x axis
                    # for the whole panel, where it is a second axis line rather
                    # than a series, and it hides whichever arm is drawn under
                    # it. It is left out of THIS panel only -- it keeps its
                    # curves in the other three and its entry in the legend --
                    # and the name is printed so the omission is never silent.
                    # ⚠ THE CAPTION MUST THEN SAY that the arm missing from (d)
                    # rejects nothing at any rate: absence there is a property
                    # of the policy, not missing data.
                    no_reject.append(label)
                    continue
                h, = ax[i].plot(x, m, color=col, lw=0.9,
                                path_effects=stroke if stroke else None, **mk)
                if i == 0:
                    # ⚠ THE KEY'S LINE IS A COPY, NOT THE DRAWN LINE (2026-09-17,
                    # at the author's request). On the page each curve carries a
                    # grey stroke under it so the pale ramp steps stay visible
                    # against white; in the key the sample sits in its own white
                    # box, where a grey outline reads as a smudge. The copy has
                    # the same colour, width and marker with a BLACK outline.
                    handles.append(Line2D(
                        [], [], color=col, lw=0.9,
                        path_effects=[pe.Stroke(linewidth=1.7,
                                                foreground="#000000"),
                                      pe.Normal()], **mk))
                    drawn.append(label)

        # (a) is read against (b): the claim is that the first spans much more
        # than the second, and rescaling either would hide it.
        ax[1].sharey(ax[0])
        titles = ["(a) Goodput tokens", "(b) Throughput",
                  "(c) Request SLO (admitted)", "(d) Rejection rate"]
        for i in range(4):
            ax[i].set_xlabel(f"Offered rate (req/s)\n{titles[i]}",
                             labelpad=1.5, linespacing=1.6)
            ax[i].set_xlim(7, 73)
            # Labels every 10 req/s; the rates that were actually measured keep
            # an unlabelled minor tick so the reader can see which points on the
            # curve are measurements and which are the line between them.
            ax[i].set_xticks(LABEL_TICKS)
            ax[i].set_xticks([r for r in RATES if r not in LABEL_TICKS],
                             minor=True)
            ax[i].grid(axis="both", **GRID)
            ax[i].set_axisbelow(True)
            # ⚠ ALL FOUR SPINES (2026-09-17, at the author's request). The top
            # and right were dropped here for years and are back, so panel (d)
            # and (c) now have their 100% ceiling drawn as well as ticked.
        # Over EVERY arm, not only the drawn ones, so that the with- and
        # without-FluidServe versions share a y axis and a value read off one
        # sits at the same height on the other.
        top = max(v["thru"][2] for lab in data for v in data[lab].values())
        ax[0].set_ylim(0, 1.05 * top)
        # The ticks are stated rather than left to the automatic locator, which
        # chooses how many to place from how tall the axis is: the same data on
        # the shorter canvas of the wide variant came out with 0 and 10k alone,
        # so the two variants of one figure disagreed about how finely the same
        # axis was ruled. Every 5k, up to the highest multiple the limit holds.
        ax[0].set_yticks([t for t in range(0, int(1.05 * top) + 1, 5000)])
        ax[0].yaxis.set_major_formatter(kfmt())
        ax[0].set_ylabel("Tokens/s")
        ax[1].set_ylabel("Tokens/s")
        for i in (2, 3):
            ax[i].set_ylim(0, 105)
            ax[i].set_yticks([0, 25, 50, 75, 100])
        # ⚠ A ROTATED Y LABEL LONGER THAN THE PANEL IS TALL IS CLIPPED, and the
        # wide variant's panels are 0.82 in high against about 1.05 in of set
        # type for "SLO attainment (%)". The text is shortened there rather than
        # the font: what the panel measures is already named in its own caption
        # underneath, "(c) Request SLO (admitted)".
        ax[2].set_ylabel(ylab_c)
        ax[3].set_ylabel("Rejected (%)")

        # Reserved in inches, so a shorter canvas keeps the same gap above the
        # panels instead of scaling it down with the figure.
        # the key is ordered FluidServe first and the two starred arms renamed;
        # the curves keep the order they were drawn in
        handles, drawn = legend_items(handles, drawn)
        band = 1.0 - (LEGEND_BAND_IN * (2.0 if grid2x2 else 1.0)) / height
        if legend and grid2x2:
            # Five names do not fit across 3.335 in on one row at any size that
            # can be read, so the one-column variant takes two rows and the band
            # reserved above the panels is doubled to hold them.
            #
            # TWO LEGENDS, NOT ONE WITH ncol=3. A single legend fills its
            # columns top to bottom, so the last row carries two entries under
            # the first two columns and the block reads as left-aligned with a
            # hole in it. Two legends, each anchored at the centre of the
            # figure, put 3 over 2 with both rows centred on the same axis.
            # One point larger when both rows still fit the canvas width
            # (2026-09-18, at the author's request); measured, not assumed.
            key_fs = KEY_FS
            for cand in (KEY_FS + 1.0, KEY_FS):
                trial = [fig.legend(handles[3:], drawn[3:],
                                    ncol=max(1, len(drawn) - 3),
                                    bbox_to_anchor=(0.5, 0.0),
                                    loc="lower center", frameon=False,
                                    fontsize=cand, columnspacing=1.0,
                                    handlelength=1.5, handletextpad=0.3),
                         fig.legend(handles[:3], drawn[:3],
                                    ncol=min(3, len(drawn)),
                                    bbox_to_anchor=(0.5, 0.0),
                                    loc="lower center", frameon=False,
                                    fontsize=cand, columnspacing=1.0,
                                    handlelength=1.5, handletextpad=0.3)]
                fig.canvas.draw()
                w_in = max(t.get_window_extent().width for t in trial) / fig.dpi
                for t in trial:
                    t.remove()
                W_in = fig.get_size_inches()[0]
                print(f"    key at {cand:.1f} pt: widest row {w_in:.2f} in on a "
                      f"{W_in:.2f} in canvas"
                      + ("  -> too wide" if w_in > W_in - 0.02 else "  -> used"))
                if w_in <= W_in - 0.02:
                    key_fs = cand
                    break
            rowh = (0.115 * key_fs / KEY_FS) / height   # one key row, in inches
            kw = dict(loc="lower center", frameon=False, fontsize=key_fs,
                      columnspacing=1.0, handlelength=1.5, handletextpad=0.3)
            fig.legend(handles[3:], drawn[3:], ncol=max(1, len(drawn) - 3),
                       bbox_to_anchor=(0.5, band - 0.003), **kw)
            fig.legend(handles[:3], drawn[:3], ncol=min(3, len(drawn)),
                       bbox_to_anchor=(0.5, band - 0.003 + rowh), **kw)
        elif legend:
            fig.legend(handles, drawn, loc="lower center", ncol=len(drawn),
                       bbox_to_anchor=(0.5, band - 0.003), frameon=False,
                       columnspacing=1.0, handlelength=1.5, handletextpad=0.3)
        fig.tight_layout(rect=(0, 0, 1, band), w_pad=w_pad,
                         h_pad=0.9 if grid2x2 else None, pad=0.25)
        if grid2x2 and legend:
            # THE BAND ABOVE THE PANELS IS A GUESS; THE KEY IS THE MEASUREMENT.
            # `LEGEND_BAND_IN` reserves 0.427 in for two 6.5 pt rows that need
            # about 0.31, and the difference was a white strip at the top of the
            # canvas. ⚠ THE PANELS MUST NOT PAY FOR IT: the axes region keeps
            # its height in INCHES and only the band shrinks, so the canvas
            # loses exactly the strip. Shrinking the canvas by the gap while
            # holding `band` as a fraction takes the difference out of the
            # panels instead, which is what a first attempt did (2.85 -> 2.18 in
            # with the panels squashed).
            fig.canvas.draw()
            h_now = fig.get_size_inches()[1]
            axes_in = band * h_now
            tops = [l.get_window_extent() for l in fig.legends]
            key_in = (max(t.ymax for t in tops) - min(t.ymin for t in tops)) / fig.dpi
            # The first band from `key_in` lands within a hundredth of an
            # inch, and the residual is corrected rather than accepted: the two
            # rows are placed from a nominal 0.115 in row height, and the real
            # one at 6.5 pt is a little larger, so without this pass the key
            # hangs 0.017 in off the top of the page.
            new_band_in = key_in + 0.02
            gap = None
            for _ in range(5):
                new_h = axes_in + new_band_in
                band = axes_in / new_h
                rowh = (0.115 * key_fs / KEY_FS) / new_h
                fig.set_size_inches(fig.get_size_inches()[0], new_h)
                for l in list(fig.legends):
                    l.remove()
                fig.legend(handles[3:], drawn[3:], ncol=max(1, len(drawn) - 3),
                           bbox_to_anchor=(0.5, band - 0.003), **kw)
                fig.legend(handles[:3], drawn[:3], ncol=min(3, len(drawn)),
                           bbox_to_anchor=(0.5, band - 0.003 + rowh), **kw)
                fig.tight_layout(rect=(0, 0, 1, band), w_pad=w_pad, h_pad=0.9,
                                 pad=0.25)
                fig.canvas.draw()
                gap = new_h - max(l.get_window_extent().ymax
                                  for l in fig.legends) / fig.dpi
                if 0.0 <= gap <= 0.012:
                    break
                new_band_in += 0.012 - gap
            print(f"    key needs {key_in:.3f} in; canvas {h_now:.2f} -> "
                  f"{new_h:.2f} in, top gap {gap:+.3f} in")
        # The axes box in inches, printed because "the panels are too square" is
        # a statement about this number and about nothing else in the file.
        w, h = fig.get_size_inches()
        bb = ax[0].get_position()
        print(f"    panel axes box {bb.width * w:.2f} x {bb.height * h:.2f} in "
              f"(aspect {bb.width * w / (bb.height * h):.2f})")
        save(fig, out_path)
        if no_reject:
            print(f"    (d) omits {', '.join(sorted(set(no_reject)))}: rejects "
                  f"nothing at any rate, so the caption has to say so")
        return drawn


def main():
    global CSV
    ap = argparse.ArgumentParser()
    # There is no --rule any more. The rule is fixed and lives in
    # deadline_ladder_attainment.py; a figure that could be drawn under several
    # rules is a figure whose numbers cannot be quoted without naming one.
    ap.add_argument("--csv", default=CSV,
                    help="per-run scoring table to draw from; the default is "
                         "the paper's table, see the note beside CSV")
    args = ap.parse_args()
    CSV = args.csv

    data, counts = collect()
    report(data, counts, RULE)

    base = final("motivation_throughput_vs_goodput_4panel_t75")
    baselines = [l for _, l, _, _, _ in ARMS if l != OURS]
    drawn_b = build(data, baselines, base + ".pdf")
    write_csv(data, baselines, counts, RULE, base + ".csv")
    drawn_a = build(data, [l for _, l, _, _, _ in ARMS], base + "_withfs.pdf")
    write_csv(data, [l for _, l, _, _, _ in ARMS], counts, RULE,
              base + "_withfs.csv")

    # The wide pair: the same four panels drawn less square, for a layout where
    # the figure can be shorter. No CSV of its own -- it draws the values
    # already written beside the standard pair, and a second copy of one table
    # is a second place for it to go stale.
    build(data, baselines, base + "_wide.pdf",
          height=FIG_H_WIDE, w_pad=W_PAD_WIDE, ylab_c="Attainment (%)")
    build(data, [l for _, l, _, _, _ in ARMS], base + "_withfs_wide.pdf",
          height=FIG_H_WIDE, w_pad=W_PAD_WIDE, ylab_c="Attainment (%)")
    # The one-column square block, in the PuBuGn ramp and with the panels' type
    # two points down, to match `two_models_hour_reqgoodput.pdf`.
    build(data, [l for _, l, _, _, _ in ARMS], base + "_withfs_col.pdf",
          height=FIG_H_2X2, w_pad=1.0, ylab_c="Attainment (%)",
          grid2x2=True, ramp=True, shrink=2.0)
    print("    the two _wide files draw the values in "
          f"{os.path.basename(base)}[_withfs].csv")
    print(f"\narms drawn: baselines {drawn_b}\n            with ours  {drawn_a}")

    # Caption numbers are computed from the same table the panels are drawn
    # from, never written in by hand: a re-measurement moves them and a literal
    # would not follow.
    for rate in (45.0, 70.0):
        if not all(rate in data[l] for l in data):
            continue
        thru = {l: data[l][rate]["thru"][0] for l in data}
        good = {l: data[l][rate]["good"][0] for l in data}
        print(f"\nat {rate:.0f} req/s: throughput "
              f"{min(thru.values()):,.0f}-{max(thru.values()):,.0f} tok/s "
              f"({max(thru.values()) / max(min(thru.values()), 1e-9):.1f}x), "
              f"goodput {min(good.values()):,.0f}-{max(good.values()):,.0f} "
              f"({max(good.values()) / max(min(good.values()), 1e-9):.1f}x)")


if __name__ == "__main__":
    main()
