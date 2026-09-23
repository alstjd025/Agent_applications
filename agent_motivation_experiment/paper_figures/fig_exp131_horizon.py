#!/usr/bin/env python3
"""Paper figure: FluidServe's projection horizon, swept.

  exp131_horizon_goodput.pdf   3.335 in wide (one column), no scaling
  exp131_horizon_goodput.csv   the values drawn, one row per horizon

Panel (b) of `results/aggregate_analysis/exp131_ablation/exp131_ablation_and_horizon.png`
drawn on its own, with token goodput added as bars (2026-09-14, at the author's
request).

  bars (left axis)   token goodput -- output tokens of the requests that met
                     their SLO, per second of the run
  lines (right axis) SLO attainment over all arrivals, and over admitted requests

The x axis is CATEGORICAL: the seven horizons are drawn evenly spaced. The source
panel drew them on a log scale, which is right for a line but gives bars of
unequal width; the tick labels carry the values.

DATA. The per-run values `analysis_scripts/request_level/exp131_figures.py`
wrote beside its PNG, read unchanged. Llama-3.1-70B, 4 instances at TP=2,
35 req/s, 8 min, agent class promised TTFT 7 s + 75 ms/token, 2 repeats per
horizon. h = 100 is the deployed value and its two runs are the EXP-108
control at the same rate; the other six horizons are EXP-131. h = 0 is absent
because the scheduler refuses it by design.

⚠ THE SCORING RULE IS NOT THE ONE THE OTHER PAPER FIGURES USE. These values come
from `all_arrivals_attainment.one_run`, which applies `exp22_fluidserve`'s rule:
a request meets its SLO if its time to first token AND its MEAN time between
tokens are inside the class budget, and token goodput counts all output tokens
of such requests. The hour-trace and static-sweep figures use the per-token
cumulative deadline with a 95% tolerance (ladder95) and count on-time tokens.
The two give different numbers for the same run (the control here reads 13,246
tok/s; ladder95 gives 13,932 for its repeat 1), so these numbers must not sit in
one table with ladder95 numbers.

    python3 paper_figures/fig_exp131_horizon.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import paper_style as ps  # noqa: E402

SRC = os.path.join(ROOT, "results", "aggregate_analysis", "exp131_ablation",
                   "exp131_ablation_and_horizon.csv")
DEPLOYED = 100
# Colours from ColorBrewer YlGnBu, the family `motivation_throughput_vs_goodput_4panel_t75_withfs_col.pdf`
# and the hour figures are drawn in (2026-09-14, at the author's request). Both
# lines are FluidServe, so the darker one takes FluidServe's #253494 there; the
# bars take #c7e9b4, a step of the same ramp that names no arm in any figure.
BAR_C = "#bdc9e1"   # at the author's request (2026-09-14)
ALL_C, ADM_C = "#253494", "#1d91c0"
STROKE = None           # set in main() once patheffects is imported
FIG_W, FIG_H = ps.COL_W, 2.05
# Panel height as a fraction of its width: 0.8 (2026-09-14, at the author's
# request; it was square).
ASPECT = 0.8
SHRINK = 2.0            # the panels' type, points below the paper style, as there
# The panels. (b) is reserved for the same sweep on a second model and is drawn
# as an empty panel with the same axes until those runs exist.
# (panel, model, per-run table). (b) is EXP-136 (2026-09-14): the same sweep on
# Qwen2.5-72B, same fleet (4 instances at TP=2), same budgets, at 1,344 rpm
# (22.4 req/s) -- the same position relative to that model's knee as 35 req/s is
# for Llama (1.22x against 1.23x). Its per-run values were scored with the same
# `all_arrivals_attainment.one_run` rule as (a) and written to
# exp136_qwen_horizon.csv. h = 100 there is the deployed arm run in the same
# session, not a borrowed control.
SRC_QWEN = os.path.join(ROOT, "results", "aggregate_analysis", "exp131_ablation",
                        "exp136_qwen_horizon.csv")
PANELS = [("a", "Llama-3.1-70B", SRC), ("b", "Qwen2.5-72B", SRC_QWEN)]


def collect(src=None):
    d = pd.read_csv(src or SRC)
    d = d[d["panel"] == "horizon"].copy()
    d["h"] = d["arm"].str.replace("h=", "", regex=False).astype(int)
    g = d.groupby("h").agg(
        n=("run", "count"),
        goodput=("goodput_tok_s", "mean"), goodput_min=("goodput_tok_s", "min"),
        goodput_max=("goodput_tok_s", "max"),
        all_arrivals=("all_arrivals", "mean"), all_min=("all_arrivals", "min"),
        all_max=("all_arrivals", "max"),
        admitted=("admitted", "mean"), adm_min=("admitted", "min"),
        adm_max=("admitted", "max"),
        rejected_pct=("rejected_pct", "mean"),
        runs=("run", lambda s: ";".join(s))).reset_index()
    # h = 1 is left out of the figure (2026-09-14, at the author's request). It
    # is in the source CSV and in the table this script writes; at 35 req/s it
    # rejects 65.1% of arrivals and delivers 6,149 tok/s.
    g = g.sort_values("h")
    return g[g["h"] != 1].reset_index(drop=True), g


def main():
    import matplotlib.patheffects as pe
    stroke = [pe.Stroke(linewidth=1.7, foreground="#9a9a9a"), pe.Normal()]
    per = {pid: collect(src) for pid, _m, src in PANELS}
    g, g_all = per["a"]
    if (g["n"] < 2).any():
        print("!! a horizon with fewer than two runs:",
              g.loc[g["n"] < 2, "h"].tolist(), file=sys.stderr)
    x = np.arange(len(g))
    style = dict(ps.STYLE)
    for k in ("axes.labelsize", "xtick.labelsize", "ytick.labelsize"):
        style[k] = max(4.0, style[k] - SHRINK)
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
        twins = []
        for i, (pid, model, _src) in enumerate(PANELS):
            ax = axes[i]
            g = per[pid][0]
            drawn = len(g) > 0
            ax2 = ax.twinx()
            twins.append(ax2)
            if drawn:
                ax.bar(x, g["goodput"], width=0.62, color=BAR_C,
                       edgecolor="#000000", linewidth=0.5, zorder=2)
                # ⚠ NO RANGE MARK (2026-09-18, at the author's request: the
                # marks were too small to read). The per-repeat minimum and
                # maximum stay in exp131_horizon_goodput.csv, and the caption
                # has to give n and the spread in words.
                # The all-arrivals attainment line was removed (2026-09-14, at
                # the author's request). ⚠ WHAT IS LEFT IS THE ADMITTED
                # DENOMINATOR ALONE, which does not charge a horizon for the
                # requests it refused -- the caption has to give the rejection
                # rate (15.1-35.5% across the drawn horizons, in the CSV).
                ax2.fill_between(x, g["adm_min"], g["adm_max"], color=ADM_C,
                                 alpha=0.18, lw=0, zorder=4)
                # no grey stroke under the dashed line: the stroke fills the
                # gaps of the dash and it reads as a solid grey line
                ax2.plot(x, g["admitted"], "s--", color=ADM_C, ms=2.4,
                         lw=0.9, mec="#9a9a9a", mew=0.4, zorder=5)
                # No mark on the deployed horizon (h = 100): the rule and its
                # label were both removed at the author's request, so the
                # caption has to say which horizon is deployed.
            else:
                ax.text(0.5, 0.5, "pending", transform=ax.transAxes,
                        ha="center", va="center", fontsize=6, color="#999999")
            # 16,000 tok/s and 100% share one height and both axes keep 5% of
            # headroom, so the left grid falls on the right ticks. The two
            # panels share both scales, so only the outer sides are named.
            ax.set_ylim(0, 16800)
            ax.set_yticks([0, 4000, 8000, 12000, 16000])
            ax.yaxis.set_major_formatter(ps.kfmt())
            ax2.set_ylim(0, 105)
            ax2.set_yticks([0, 25, 50, 75, 100])
            ax.set_xticks(x)
            # Rotated: seven labels under a 1.1 in panel run together flat
            # ("100250500"), and the values are what the axis is read for.
            ax.set_xticklabels([str(h) for h in g["h"]], rotation=45,
                               ha="right", rotation_mode="anchor")
            ax.set_xlim(-0.6, len(g) - 0.4)
            ax.set_xlabel("Projection Horizon (iterations)", labelpad=1.5)
            ax.grid(axis="both", **ps.GRID)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", length=2.0, pad=1.5)
            ax2.tick_params(axis="y", length=2.0, pad=1.5)
            # ⚠ ALL FOUR SPINES (2026-09-18, at the author's request), as in
            # every other figure in final/. The top was hidden here before.
        axes[0].set_ylabel("Token Goodput (t/s)")
        twins[1].set_ylabel("SLO Attainment (%)")

        handles = [Patch(facecolor=BAR_C, edgecolor="#000000", linewidth=0.5),
                   Line2D([], [], color=ADM_C, marker="s", ms=2.4, lw=0.9,
                          ls="--", mec="#9a9a9a", mew=0.4)]
        labels = ["Token Goodput", "SLO Attainment (admitted)"]
        top_band, bot_band = 0.20, 0.16
        h_fig = FIG_H

        # A SQUARE SWATCH FOR THE BARS, a line for the attainment (2026-09-14,
        # at the author's request). One `handlelength` serves every entry, so
        # setting it equal to `handleheight` would also crush the dashed line
        # and its marker; instead the patch entry draws its own square inside
        # the handle box, as tall as the box, centred.
        from matplotlib.legend_handler import HandlerPatch
        from matplotlib.patches import Rectangle

        class SquareHandler(HandlerPatch):
            def create_artists(self, legend, orig, xd, yd, width, height,
                               fontsize, trans):
                side = height
                r = Rectangle((xd + (width - side) / 2.0, yd), side, side,
                              facecolor=orig.get_facecolor(),
                              edgecolor=orig.get_edgecolor(),
                              linewidth=orig.get_linewidth(), transform=trans)
                return [r]

        def lay():
            """Layout at the current canvas height, key band fitted to the key."""
            nonlocal_key = None
            band = top_band
            for _ in range(6):
                if nonlocal_key is not None:
                    nonlocal_key.remove()
                fig.tight_layout(rect=(0, bot_band / h_fig, 1,
                                       1 - band / h_fig), pad=0.3, w_pad=0.6)
                nonlocal_key = fig.legend(
                    handles, labels, loc="lower center", ncol=len(labels),
                    bbox_to_anchor=(0.5, 1 - band / h_fig), frameon=False,
                    fontsize=7, columnspacing=1.5, handlelength=1.8,
                    handleheight=0.9, handletextpad=0.4, borderaxespad=0.0,
                    handler_map={handles[0]: SquareHandler()})
                fig.canvas.draw()
                gap = h_fig - nonlocal_key.get_window_extent().y1 / fig.dpi
                if abs(gap) <= 0.005:
                    break
                band = max(0.05, band - gap)
            return nonlocal_key

        # SQUARE PANELS (2026-09-14, at the author's request). The width is
        # fixed by the column, so the canvas HEIGHT is solved for: lay out,
        # measure the axes box, and move the canvas height by the difference
        # between the box's height and width until the two agree. Nothing is
        # squeezed sideways, and no white band is left above or below.
        key = None
        for _ in range(10):
            if key is not None:
                key.remove()
            fig.set_size_inches(FIG_W, h_fig)
            key = lay()
            p = axes[0].get_position()
            w_in, h_in = p.width * FIG_W, p.height * h_fig
            if abs(h_in - ASPECT * w_in) <= 0.005:
                break
            h_fig += (ASPECT * w_in - h_in)
        print(f"  panels {w_in:.3f} x {h_in:.3f} in on a {FIG_W:.3f} x "
              f"{h_fig:.3f} in canvas")
        caps = []
        for i, (pid, model, _src) in enumerate(PANELS):
            p = axes[i].get_position()
            caps.append(fig.text(0.5 * (p.x0 + p.x1), 0.012 * FIG_H / h_fig,
                                 f"({pid}) {model}",
                                 ha="center", va="bottom", fontsize=7))
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        W, H = fig.get_size_inches() * fig.dpi
        ink_top = max(a.get_tightbbox(rend).y1 for a in list(axes) + twins)
        ink_bot = min(a.get_tightbbox(rend).y0 for a in list(axes) + twins)
        print(f"  key clears the panels by "
              f"{(key.get_window_extent(rend).y0 - ink_top) / fig.dpi:+.3f} in; "
              f"captions clear them by "
              f"{(ink_bot - max(c.get_window_extent(rend).y1 for c in caps)) / fig.dpi:+.3f} in")
        gap_ab = (axes[1].get_tightbbox(rend).x0 - twins[0].get_tightbbox(rend).x1) / fig.dpi
        print(f"  panel (a)'s right ticks clear panel (b)'s left ticks by {gap_ab:+.3f} in"
              + ("  ⚠ OVERLAP" if gap_ab < 0 else ""))
        for art in [key] + caps + [axes[0].yaxis.label, twins[1].yaxis.label]:
            e = art.get_window_extent(rend)
            if e.x0 < -0.5 or e.x1 > W + 0.5 or e.y0 < -0.5 or e.y1 > H + 0.5:
                print("  ⚠ off the canvas")
        ps.save(fig, ps.final("exp131_horizon_goodput.pdf"))

    out = ps.final("exp131_horizon_goodput.csv")
    tabs = []
    for pid, model, _src in PANELS:
        ga = per[pid][1]
        tabs.append(ga.assign(panel=pid, model=model, drawn=ga["h"] != 1))
        print(f"({pid}) {model}")
        print(ga[["h", "n", "goodput", "goodput_min", "goodput_max",
                  "all_arrivals", "admitted", "rejected_pct"]].round(1)
              .to_string(index=False))
    pd.concat(tabs).assign(
        rule="exp22 mean-TBT rule (not ladder95)",
        goodput_definition="output tokens of requests that met the rule, per second"
    ).to_csv(out, index=False, float_format="%.4f")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
