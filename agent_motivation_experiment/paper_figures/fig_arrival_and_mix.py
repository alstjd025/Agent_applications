#!/usr/bin/env python3
"""Paper figure: the load a cluster is asked to serve moves, and so does what
that load is made of.

  arrival_and_mix.pdf          3.335 x 1.45 in, one column, width=\\columnwidth
  arrival_and_mix_stacked.pdf  3.335 x 2.10 in, the same two panels one above
                               the other
  arrival_and_mix.csv          exactly the values drawn

  (a) the arrival rate of a production trace over four days, normalised by its
      own peak -- panel (a) of `load_engine_tbt.pdf`, unchanged
  (b) how far the class composition of those arrivals sits from the window's
      own average composition -- `azure_mix_only.pdf`, unchanged

The two panels were separate figures and are joined here because they make one
argument in two steps: the SIZE of the offered load moves, and so does its
COMPOSITION. Nothing about either series changed in the move; what changed is
that they now carry (a) and (b) and share one canvas.

⚠ BOTH PANELS ARE THE SAME FOUR DAYS AND THE SAME CLOCK, which is why the
stacked version exists: vertically aligned panels put the same hour at the same
horizontal position, so a reader can drop a line from a peak in the rate to the
composition under it. ⚠ AND THE FIGURE'S POINT IS THAT THE TWO DO NOT TRACK
EACH OTHER -- the correlation between the binned rate and the code share is
+0.57, printed by `fig_azure_rate_and_mix.py`. If the caption makes that claim,
include the stacked version; the side-by-side one asks the reader to compare
from two origins.

⚠ (b) IS A DEVIATION, NOT A MIXTURE. It is the share of arrivals that would have
to change class for that ten-minute bin to have the window's average
composition, so 0 means "the average mix" and not "one class". The dashed rule
is its maximum over the window and the arrow reads "up to", not "between".

⚠ THIS IS THE SOURCE TRACE, NOT THE ONE WE REPLAY. Our hour-long trace takes
this window's temporal ORDER through a quantile (rank) transform onto a chosen
rate band, so the peak-to-trough ratio of what we run is a parameter we picked
and not this figure's 15.6x. `fig_azure_trace_shape.py` carries the full
argument.

DATA. `fig_azure_rate_and_mix.py`'s window and its ten-minute bins, imported
rather than recomputed, and `fig_load_engine_tbt.py`'s colour and styling for
panel (a) so that the panel is the same panel in both figures.

    python3 paper_figures/fig_arrival_and_mix.py
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
LET = _load("loadengtbt", os.path.join(HERE, "fig_load_engine_tbt.py"))

# 2026-09-16, at the author's request: the blues of the class-mix figures
# (chat #0868ac, deep research #43a2ca) instead of the teal and terracotta the
# two source figures use, so this figure sits in the same tone as those. Was
# LET.RATE_C and AZ.C_MIX_ONLY / AZ.C_MIX_ONLY_ALPHA.
RATE_C = "#0868ac"
MIX_C = "#43a2ca"
MIX_A = 0.22
FIG_W = ps.COL_W
FIG_H_SIDE, FIG_H_STACK = 1.45, 2.10


def series():
    """(hours, per-minute total, peak, bin centres, mixture deviation %)."""
    conv, code, _plan = AZ.source_window()
    total = conv + code
    hours = np.arange(len(total)) / 60.0
    nb = len(total) // AZ.BIN
    cb = conv[:nb * AZ.BIN].reshape(nb, AZ.BIN).sum(1)
    kb = code[:nb * AZ.BIN].reshape(nb, AZ.BIN).sum(1)
    f = kb / (cb + kb)
    hb = (np.arange(nb) * AZ.BIN + AZ.BIN / 2) / 60.0
    comp = np.column_stack([1.0 - f, f])
    tv = 100.0 * 0.5 * np.abs(comp - comp.mean(axis=0)).sum(axis=1)
    return hours, total, float(total.max()), hb, tv


def write_csv(hours, rate, hb, tv, path):
    rows = [{"panel": "a", "series": "arrival_rate_norm", "x_hours": float(a),
             "value": float(b)} for a, b in zip(hours, rate)]
    rows += [{"panel": "b", "series": "mixture_deviation_pct",
              "x_hours": float(a), "value": float(b),
              "bin_minutes": AZ.BIN} for a, b in zip(hb, tv)]
    df = pd.DataFrame(rows)
    df["source"] = "Azure LLM Inference 2024 conversation + code, the window in "
    df["source"] += os.path.basename(AZ.PLAN)
    df.to_csv(path, index=False, float_format="%.5f")
    print(f"wrote {path}  ({len(df)} rows)")


def build(hours, total, peak, hb, tv, out, side=True):
    rate = total / peak
    with plt.rc_context({**ps.STYLE, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "axes.labelsize": 6}):
        h = FIG_H_SIDE if side else FIG_H_STACK
        if side:
            fig, ax = plt.subplots(1, 2, figsize=(FIG_W, h))
        else:
            fig, ax = plt.subplots(2, 1, figsize=(FIG_W, h))

        ax[0].plot(hours, rate, color=RATE_C, lw=0.5)
        ax[0].fill_between(hours, rate, color=RATE_C, alpha=0.15, lw=0)
        ax[0].axhline(1.0, color="#555555", lw=0.6, ls="--")
        ax[0].axhline(rate.min(), color="#555555", lw=0.6, ls="--")
        # The same arrow and the same words as in `load_engine_tbt.pdf`: short
        # in the side-by-side version, where the panel is 1.35 in wide, and the
        # full sentence where there is room for it.
        t0 = AZ.span(ax[0], hours, rate, rate.min(), 1.0,
                     (f"{1 / rate.min():.1f}$\\times$" if side else
                      f"peak / trough = {1 / rate.min():.1f}$\\times$"))
        ax[0].set_ylabel("Arrival Rate (norm.)", labelpad=1.5)
        ax[0].set_ylim(0, 1.0)
        ax[0].set_yticks([0, 0.5, 1.0])
        ax[0].set_xlabel("Time (hours)\n(a) Request Arrivals", labelpad=1.5,
                         linespacing=1.5)

        ax[1].plot(hb, tv, color=MIX_C, lw=0.6)
        ax[1].fill_between(hb, tv, color=MIX_C, alpha=MIX_A, lw=0)
        ax[1].axhline(tv.max(), color="#333333", lw=0.6, ls="--")
        t1 = AZ.span(ax[1], hb, tv, 0.0, tv.max(),
                     (f"up to {tv.max():.0f}%" if side else
                      f"up to {tv.max():.0f}% of arrivals"))
        # ⚠ TWO LINES STACKED, ONE SIDE BY SIDE. A rotated y label is not
        # clipped to its panel: "Workload Mixture Deviation (%)" is 1.5 in of
        # ink against a stacked panel 0.75 in tall, so on one line it ran over
        # the panel above and off the canvas. Side by side the panel is taller
        # and the short form fits on one line.
        ax[1].set_ylabel("Mixture Dev. (%)" if side else
                         "Workload Mixture\nDeviation (%)", labelpad=1.5)
        ax[1].set_ylim(0, max(50.0, tv.max() * 1.25))
        ax[1].set_yticks([0, 20, 40])
        ax[1].set_xlabel("Time (hours)\n(b) Workload Mixture", labelpad=1.5,
                         linespacing=1.5)

        for a in ax:
            a.grid(axis="both", **ps.GRID)
            a.set_axisbelow(True)
            a.set_xlim(0, hours[-1])
            # 48-hour ticks side by side, where five hour labels under 1.35 in
            # of axes would run together; 24 stacked, where the panel is the
            # full column.
            a.set_xticks(np.arange(0, hours[-1] + 1, 48 if side else 24))
            a.xaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(3))
            a.yaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
            a.tick_params(which="minor", length=1.2)

        # One key over each panel, fitted to the largest size at which the two
        # do not touch -- `fig_load_engine_tbt.key_pair`, so that a key on this
        # figure is the same kind of object at the same size as on that one.
        rect_top = 1 - 0.20 / h
        fig.tight_layout(rect=(0, 0, 1, rect_top), pad=0.3,
                         **({"w_pad": 1.4} if side else {"h_pad": 1.2}))
        # ⚠ `key_pair` IS FOR TWO PANELS SIDE BY SIDE. It fits the largest type
        # at which the two keys do not overlap HORIZONTALLY, and stacked panels
        # share an x centre, so every size it tries "overlaps" and it falls all
        # the way to its floor -- 4.4 pt, against 6.4 for the same two keys on
        # the same canvas. Stacked, each key is anchored over its own panel and
        # the two can never meet, so the size is simply set.
        size, gap = None, float("nan")
        for _ in range(6 if side else 1):
            for l in list(fig.legends):
                l.remove()
            if not side:
                size = 6.4
                for a, (c, lab) in zip(ax, [(RATE_C, "Request Arrivals"),
                                            (MIX_C, "Mixture Deviation")]):
                    bb = a.get_position()
                    fig.legend([Line2D([], [], color=c, lw=0.9)], [lab],
                               loc="lower center",
                               bbox_to_anchor=(0.5 * (bb.x0 + bb.x1),
                                               bb.y1 + 0.008),
                               ncol=1, frameon=False, fontsize=size,
                               columnspacing=0.7, handlelength=1.1,
                               handletextpad=0.3, borderaxespad=0.0)
                break
            size = LET.key_pair(fig, ax,
                                [(RATE_C, "Request Arrivals")],
                                [(MIX_C, "Mixture Deviation")],
                                sizes=(size,) if size else
                                (6.4, 6.0, 5.6, 5.2, 4.8, 4.4))
            fig.canvas.draw()
            # The band above the panels is a guess and the key is the
            # measurement; the residual is driven to nothing IN BOTH
            # DIRECTIONS, because a negative gap is the key hanging off the
            # page and a positive one is a white strip.
            fig.canvas.draw()
            gap = h - max(l.get_window_extent().ymax
                          for l in fig.legends) / fig.dpi
            if abs(gap) <= 0.005:
                break
            rect_top = min(0.999, rect_top + gap / h)
            fig.tight_layout(rect=(0, 0, 1, rect_top), pad=0.3,
                             **({"w_pad": 1.4} if side else {"h_pad": 1.2}))
        print(f"    legend type {size:.1f} pt, top gap {gap:+.3f} in")
        AZ.check_overlap(fig, ax[0], t0, hours, rate, "arrival rate")
        AZ.check_overlap(fig, ax[1], t1, hb, tv, "mixture deviation")
        ps.save(fig, out)


def main():
    hours, total, peak, hb, tv = series()
    print(f"{len(total)} minutes ({len(total) / 1440:.1f} days); "
          f"rate trough {total.min() / peak:.3f} of peak, peak/trough "
          f"{peak / total.min():.1f}x; mixture deviation median "
          f"{np.median(tv):.1f}%, max {tv.max():.1f}%")
    build(hours, total, peak, hb, tv,
          ps.final("arrival_and_mix.pdf"), side=True)
    build(hours, total, peak, hb, tv,
          ps.final("arrival_and_mix_stacked.pdf"), side=False)
    write_csv(hours, total / peak, hb, tv,
              ps.final("arrival_and_mix.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
