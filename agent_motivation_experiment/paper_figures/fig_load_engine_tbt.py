#!/usr/bin/env python3
"""Paper figure: the load moves, the instance's state moves, and so does the
pace that instance delivers.

  load_engine_tbt.pdf   3.335 x 2.70 in, one column, width=\\columnwidth
  load_engine_tbt.csv   exactly the values drawn

  (a) the arrival rate of a production trace over four days, normalised by its
      own peak
  (b) one instance over twelve minutes: how full its KV pool was and how many
      requests were in its running batch, each scaled over the window shown
  (c) the SAME instance over the SAME twelve minutes: the per-token time it
      actually delivered, in milliseconds

This is `load_and_engine.pdf` with the third panel added underneath, drawn the
full width because it shares panel (b)'s clock and the eye should be able to
drop a line from one to the other.

⚠ (b) AND (c) ARE THE SAME RUN, THE SAME ENGINE AND THE SAME MINUTES -- EXP-109
repeat 1 of the Llumnix SLO arm, engine 8002, minutes 35-47 of the hour -- so a
feature in (b) and a feature in (c) at the same x are the same moment. Panel (a)
is a different trace on a different clock, four days against twelve minutes, and
nothing in (b) or (c) is a consequence of the interval drawn in (a).

⚠ (b) IS SCALED OVER ITS OWN WINDOW and (c) IS NOT. Each series in (b) runs
between its own minimum and maximum there, so that panel shows the SHAPE of the
two and not how large either is; (c) is in milliseconds on an axis that starts
at zero, because the quantity it draws is one the class budgets are stated in
and a normalised pace could not be read against them.

⚠ THE PER-TOKEN AXIS IS CUT AT 150 ms AND THE CUT IS COUNTED. A handful of
seconds in this window are three to five times the rest; letting the axis reach
them puts the band the engine actually lives in into a seventh of the panel. The
seconds above the cut are printed and belong in the caption.

WHAT THE THIRD PANEL IS FOR. (a) says the work a cluster is asked to do is not
stationary. (b) says the state a placement decision reads is not stationary
either. (c) says the thing the SLO is written about -- the time between one
token and the next -- moves with them, on the same instance and in the same
minutes as (b): it is the consequence the first two panels are drawn to set up.

DATA. Panel (a): `fig_azure_rate_and_mix.py`'s source window, imported rather
than recomputed. Panels (b) and (c): `fig_engine_window.py`'s gauge reader, also
imported. The gauges are the engine's own, as scraped each second; the
per-token time is its histogram counters differenced per second and smoothed
over 3 s before the division, which is what that reader does.

    python3 paper_figures/fig_load_engine_tbt.py
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import paper_style as ps  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AZ = _load("azfig", os.path.join(HERE, "fig_azure_rate_and_mix.py"))
EW = _load("engwin", os.path.join(HERE, "fig_engine_window.py"))
LE = _load("loadeng", os.path.join(HERE, "fig_load_and_engine.py"))

ARM_KEY, PORT = "slo", 8002
T_LO, T_HI = 35.0, 47.0
FIG_H = 2.88          # 2.70 + a row for panel (c)'s own key
# The arm's own hue, as in `latency_two_arms.pdf` where this curve comes from,
# so the same line is the same colour in both figures.
TBT_C = "#1b7837"
# Orange for the attainment curve (2026-09-11, at the author's request; it was
# #54278f, the dark purple `latency_three_arms.pdf` gives the same quantity).
# ⚠ THIS FIGURE AND THAT ONE NOW GIVE ONE QUANTITY TWO COLOURS, so they should
# not be shown on one page without the caption saying which is which.
# #e08214 and not a yellow: a yellow line 1 pt wide has almost no contrast
# against white paper, and the marker would be carrying the curve on its own.
# It is also not #ff7f0e or #d62728, which this directory binds to the deep
# research CLASS and to PolyServe.
ATT_C = "#e08214"
# A marker at every attainment sample, because that curve is 48 points on a
# 15 s grid while the TBT curve beside it is 720 points at 1 s: without them the
# two read as the same kind of series, and a straight stretch of the attainment
# line is an interpolation between two windows rather than a measurement.
ATT_MARK, ATT_MS = "o", 2.6
# The three series in (a) and (b) came from the modules this figure imports its
# readers from, and those modules are still on matplotlib's tab10: a saturated
# blue for both the arrival rate and the KV pool, and a brown for the batch. On
# one page with panel (c), whose two colours were chosen from ColorBrewer, the
# figure reads as two figures. These three are therefore overridden HERE, to the
# PuBuGn steps `class_mix_hour_3arms_notbt.pdf` uses, so that the hour-trace
# figures share one palette.
#
# OVERRIDDEN LOCALLY, NOT IN THE MODULES THEY COME FROM. The same three
# constants are read by fig_azure_rate_and_mix.py, fig_engine_window.py and
# fig_load_and_engine.py, so editing them there would recolour three other
# figures without anyone asking for it. If the whole set is meant to move, move
# it in those modules and delete these three lines.
RATE_C = "#1c9099"     # PuBuGn teal, the anchor of the palette
KV_C = "#016c59"       # its dark end: the fuller quantity takes the darker step
BATCH_C = "#67a9cf"    # its mid blue, in place of the tab10 brown
#
# PANEL (c) IS LEFT AS IT IS. TBT_C is bound to `latency_two_arms.pdf` by the
# note above -- the same curve is meant to be the same colour there -- and ATT_C
# was set on 2026-09-11 at the author's request. Changing either here would undo
# a decision made elsewhere, so the green and the orange stay, and they sit
# beside the teal rather than inside it.
TBT_TOP = 150.0
ATT_WIN = 15.0


def engine_series():
    EW.PORT, EW.T_LO, EW.T_HI = PORT, T_LO, T_HI
    EW.SMOOTH_S = 1
    run = os.path.join(ROOT, "results", EW.RUNS[ARM_KEY][1])
    t0 = float(pd.read_csv(os.path.join(run, "metrics.csv"),
                           usecols=["start_time"])["start_time"].min())
    EW.WIN, EW.STEP = ATT_WIN, ATT_WIN
    m, kv, bat, _wait, itl = EW.gauges(run, t0)
    att = EW.attainment(run, t0)
    return m, kv, bat, itl, att


def write_csv(hours, rate, m, kv, bat, itl, att, path):
    rows = [{"panel": "a", "series": "arrival_rate_norm", "x_hours": float(a),
             "value": float(b)} for a, b in zip(hours, rate)]
    rows += [{"panel": "b", "series": "kv_cache_usage_pct", "x_minutes": float(a),
              "value": float(b)} for a, b in zip(m, kv)]
    rows += [{"panel": "b", "series": "running_batch_requests",
              "x_minutes": float(a), "value": float(b)} for a, b in zip(m, bat)]
    rows += [{"panel": "c", "series": "inter_token_latency_ms",
              "x_minutes": float(a), "value": float(b)}
             for a, b in zip(m, itl) if b == b]
    rows += [{"panel": "c", "series": "slo_attainment_admitted_pct",
              "x_minutes": float(a), "value": float(b), "n_requests": int(c),
              "attainment_window_s": ATT_WIN}
             for a, b, c in zip(*att)]
    df = pd.DataFrame(rows)
    df["panel_bc_run"] = EW.RUNS[ARM_KEY][1]
    df["panel_bc_engine_port"] = PORT
    df.to_csv(path, index=False, float_format="%.5f")
    print(f"wrote {path}  ({len(df)} rows)")


def key_pair(fig, ax, keys_a, keys_b, sizes=(6.4, 6.0, 5.6, 5.2, 4.8, 4.4)):
    """One key above each top panel, centred on that panel but anchored to the
    figure, at the largest size at which the two do not overlap.

    Returns the size, so the third panel's key can be set to the same one.
    """
    def draw(keys, axis, size):
        hs = [Line2D([], [], color=c, lw=0.9) for c, _ in keys]
        bb = axis.get_position()
        return fig.legend(hs, [l for _, l in keys], loc="lower center",
                          bbox_to_anchor=(0.5 * (bb.x0 + bb.x1), bb.y1 + 0.008),
                          ncol=len(keys), frameon=False, fontsize=size,
                          columnspacing=0.7, handlelength=1.1,
                          handletextpad=0.3, borderaxespad=0.0)

    for size in sizes:
        la, lb = draw(keys_a, ax[0], size), draw(keys_b, ax[1], size)
        fig.canvas.draw()
        ea, eb = la.get_window_extent(), lb.get_window_extent()
        w = fig.get_size_inches()[0] * fig.dpi
        if ea.x1 + 4 < eb.x0 and ea.x0 >= 0 and eb.x1 <= w:
            return size
        la.remove(); lb.remove()
    draw(keys_a, ax[0], sizes[-1]); draw(keys_b, ax[1], sizes[-1])
    return sizes[-1]


def build(hours, total, peak, m, kv, bat, itl, att, out,
          width=ps.COL_W, height=FIG_H):
    def mm(v):
        return 100.0 * (v - v.min()) / max(v.max() - v.min(), 1e-9)

    with plt.rc_context({**ps.STYLE, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "axes.labelsize": 6}):
        fig = plt.figure(figsize=(width, height))
        # Two panels over one. The bottom row is shorter than the top: it holds
        # one curve and one y label, where the top row holds two of each.
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.78])
        ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
        axc = fig.add_subplot(gs[1, :])
        rate = total / peak

        ax[0].plot(hours, rate, color=RATE_C, lw=0.5)
        ax[0].fill_between(hours, rate, color=RATE_C, alpha=0.15, lw=0)
        ax[0].axhline(1.0, color="#555555", lw=0.6, ls="--")
        ax[0].axhline(rate.min(), color="#555555", lw=0.6, ls="--")
        t0 = AZ.span(ax[0], hours, rate, rate.min(), 1.0,
                     f"{1 / rate.min():.1f}$\\times$")
        ax[0].set_ylabel("Arrival Rate (norm.)", labelpad=1.5)
        ax[0].set_ylim(0, 1.0)
        ax[0].set_yticks([0, 0.5, 1.0])
        ax[0].set_xlim(0, hours[-1])
        ax[0].set_xticks(np.arange(0, hours[-1] + 1, 48))
        ax[0].set_xlabel("Time (hours)\n(a) Request Arrivals", labelpad=1.5,
                         linespacing=1.5)

        ax[1].plot(m, mm(kv), color=KV_C, lw=0.9)
        ax[1].plot(m, mm(bat), color=BATCH_C, lw=0.9)
        ax[1].set_ylabel("Norm. (%)", labelpad=1.5)
        ax[1].set_ylim(0, 100)
        ax[1].set_yticks([0, 25, 50, 75, 100])
        ax[1].set_xlim(T_LO, T_HI)
        ax[1].set_xticks(np.arange(T_LO, T_HI + 1e-9, 3))
        ax[1].set_xlabel("Time (min.)\n(b) Instance State", labelpad=1.5,
                         linespacing=1.5)

        ok = ~np.isnan(itl)
        axc.plot(m, itl, color=TBT_C, lw=0.9, zorder=3)
        axc.set_ylim(0, TBT_TOP)
        axc.set_yticks([0, 50, 100, 150])
        axc.set_xlim(T_LO, T_HI)
        axc.set_xticks(np.arange(T_LO, T_HI + 1e-9, 1))
        axc.set_ylabel("TBT (ms)", labelpad=1.5)
        axc.set_xlabel("Time (min.)\n(c) Measured TBT and Request SLO "
                       "Attainment of an Instance",
                       labelpad=1.5, linespacing=1.5)
        # The share of the requests sent to THIS engine that met their rule, on
        # the right. ⚠ ADMITTED DENOMINATOR: it says nothing about what the arm
        # refused, and over these twelve minutes Llumnix SLO refuses 28.3% of
        # arrivals, so the caption has to carry that number or a high curve here
        # reads as "nearly everything was served".
        axd = axc.twinx()
        axd.plot(att[0], att[1], color=ATT_C, lw=1.0, zorder=2,
                 marker=ATT_MARK, ms=ATT_MS, mew=0.0)
        axd.set_ylim(0, 100)
        axd.set_yticks([0, 50, 100])
        axd.set_ylabel("Request SLO (%)", labelpad=2.0)
        axd.tick_params(axis="y", length=2.0)
        axd.spines["top"].set_visible(False)
        # BOTH AXES IN BLACK and a key of its own (2026-09-11, at the author's
        # request). Colouring the two y labels was the alternative to a key on a
        # twin axis; with the key present the coloured labels would be the same
        # information twice, and black is what the other panels' labels are.
        # The row for it comes from the extra 0.18 in of canvas: put in the gap
        # between the rows, it sits under the top row's panel names instead of
        # on them, and `check_key_gap` measures that it does.


        keys_a = [(RATE_C, "Request Arrivals")]
        keys_b = [(KV_C, "KV-cache Occupancy"), (BATCH_C, "Batch Size")]
        # ⚠ THE TOP KEYS ARE ANCHORED TO THE FIGURE, NOT TO THEIR PANELS.
        # `LE.legend_row` fits each key inside its own panel, and at 1.2 in wide
        # a panel cannot hold "KV-cache occupancy" and "Batch size" on one row
        # above about 4.4 pt -- which, since one size is used for every key on
        # the figure, took all three keys down with it. Centred on its panel but
        # free to reach into the gutter and the margin, the same key fits at
        # 5.6 pt. The two are then measured against each other and the size is
        # the largest at which they do not touch, so a key can never be drawn
        # over its neighbour.
        size = key_pair(fig, ax, keys_a, keys_b)
        # (c) is twice as wide as either top panel, so its two-entry key fits at
        # the same size; it is set from `size` and not fitted again, because two
        # keys set differently on one figure read as two kinds of key.
        # (c)'s key is built here rather than through `LE.legend_row`, which
        # draws line handles only: the attainment entry has to carry the marker
        # or the key would describe a curve the panel does not contain.
        hs = [Line2D([], [], color=TBT_C, lw=0.9),
              Line2D([], [], color=ATT_C, lw=1.0, marker=ATT_MARK,
                     ms=ATT_MS, mew=0.0)]
        axc.legend(hs, ["TBT", "Request SLO Attainment"], loc="lower center",
                   bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False,
                   fontsize=size, columnspacing=0.7, handlelength=1.1,
                   handletextpad=0.3, borderaxespad=0.0)
        print(f"    legend type {size:.1f} pt on all three panels")

        for a in list(ax) + [axc]:
            a.grid(axis="both", **ps.GRID)
            a.set_axisbelow(True)
            a.xaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(3))
            a.yaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
            a.tick_params(which="minor", length=1.2)

        # The band reserved above the panels is a guess and the key is the
        # measurement, so the residual is driven to nothing IN BOTH DIRECTIONS:
        # a positive gap is a white strip at the top of the canvas, and a
        # NEGATIVE one is the key's ink hanging off the page, which is what a
        # one-directional fit left behind (the top row of type came out 0.028 in
        # short). The top keys are anchored to the axes' positions at the moment
        # they are created, so they are rebuilt after every layout pass.
        rect_top = 1 - 0.24 / height
        fig.tight_layout(rect=(0, 0, 1, rect_top), pad=0.3, w_pad=1.4,
                         h_pad=1.0)
        gap = None
        for _ in range(6):
            fig.canvas.draw()
            gap = height - max(l.get_window_extent().ymax
                               for l in fig.legends) / fig.dpi
            if abs(gap) <= 0.005:
                break
            rect_top = min(0.999, rect_top + gap / height)
            fig.tight_layout(rect=(0, 0, 1, rect_top), pad=0.3, w_pad=1.4,
                             h_pad=1.0)
            for l in list(fig.legends):
                l.remove()
            key_pair(fig, ax, keys_a, keys_b, sizes=(size,))
        AZ.check_overlap(fig, ax[0], t0, hours, rate, "arrival rate")
        w, h = fig.get_size_inches()
        bb = axc.get_position()
        print(f"    (c) axes box {bb.width * w:.2f} x {bb.height * h:.2f} in, "
              f"top gap {gap:.3f} in")
        print(f"    (c) TBT p50 {np.percentile(itl[ok], 50):.1f} ms, "
              f"mean {itl[ok].mean():.1f}, min {itl[ok].min():.1f}, "
              f"max {itl[ok].max():.0f}; "
              f"{int((itl[ok] > TBT_TOP).sum())} of {int(ok.sum())} s above the "
              f"{TBT_TOP:.0f} ms axis")
        ps.save(fig, out)


def main():
    conv, code, _plan = AZ.source_window()
    total = conv + code
    hours = np.arange(len(total)) / 60.0
    peak = total.max()
    m, kv, bat, itl, att = engine_series()
    print(f"panel (a): {len(total)/1440:.1f} days, peak/trough "
          f"{peak/total.min():.1f}x")
    print(f"panel (b): KV {kv.min():.0f}-{kv.max():.0f}%, batch "
          f"{bat.min():.0f}-{bat.max():.0f} over {T_LO:.0f}-{T_HI:.0f} min")
    pdf = os.path.join(HERE, "load_engine_tbt.pdf")
    build(hours, total, peak, m, kv, bat, itl, att, pdf)
    write_csv(hours, total / peak, m, kv, bat, itl, att, pdf[:-4] + ".csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
