#!/usr/bin/env python3
"""Paper figure: per class, how fast each control plane served, and what became
of every arrival, on the Llama-3.1-70B hour trace.

  class_latency_outcome_rows.pdf   3.335 x 3.70 in, `figure`, width=\\columnwidth
  class_latency_outcome_rows.csv   exactly the values drawn, with repeat ranges

  (a) time to first token, per class, five arms per class
  (b) mean time between tokens, per class, five arms per class
  (c) every arrival of that class, as met / missed / rejected / unfinished

GROUPED BY CLASS, NOT BY ARM. The question each panel answers is "within one
class, which arm served it how", so the five arms of one class sit side by side
and the class budget is one rule across its group.

⚠ THE WHISKER IS P95, NOT AN ERROR BAR. The bar is the median (P50) of the
finished requests and the whisker runs from it up to their 95th percentile.
Both are quantiles of one distribution of requests; nothing here is a spread
over repeats, and the caption must say so because the glyph suggests it.
The repeat spread is in the CSV and is 0.0-1.4 ms and 0.0-0.6 s for every arm
but the vLLM router.

⚠ (a) AND (b) ARE MEASURED ON THE REQUESTS THAT FINISHED, so their population
is different for every arm, and (c) is what shows how different. An arm that
refuses half of a class is timed on the half it kept and looks fast for that
reason alone. The two latency panels are only readable with (c) under them.

⚠ THE RULE IS THE NOMINAL PROMISE, NOT THE SCORING RULE. Requests are scored in
(c) with the cumulative ladder -- token i is on time if it arrives by
TTFT + i x TBT, and a request passes if 95% of its tokens do -- which lets a
request bank early tokens against late ones. A P95 above the rule is therefore
not the same as 5% of requests failing.

⚠ THE AXES ARE CUT AND WHAT CROSSES THE CUT IS WRITTEN ABOVE THE PANEL. Only
the vLLM router crosses: it has no admission control, a queue that grows for
the whole hour, and first-token times of minutes. An axis that reached it would
draw every other bar as a line on the floor.

⚠ FOUR BANDS IN (c) AND NOT THREE. The vLLM router leaves about half of its
arrivals still running when the run ends, and those have no outcome. Leaving
the band out would rescale its bar and hide that.

DATA. EXP-109, the five arms of `two_models_hour_reqgoodput.pdf`, BOTH repeats;
every drawn value is the mean of the two. Latencies are the client's record
over `exp22_fluidserve.load_run`'s analysis window, per-token time derived as
(end-to-end - first token) / (output tokens - 1). Outcomes are the ladder95
verdict files, standard budgets (chat 5 s / 50 ms, agent 7 s / 75 ms, deep
research 10 s / 100 ms).

    python3 paper_figures/fig_class_latency_outcome_rows.py
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
sys.path.insert(0, os.path.join(ROOT, "analysis_scripts", "request_level"))
import paper_style as ps  # noqa: E402

os.environ.setdefault("FS_SWE_TBT_MS", "75")
os.environ.setdefault("FS_SWE_TTFT_S", "7")
from exp22_fluidserve import load_run  # noqa: E402

VERDICTS = os.path.join(ROOT, "results", "aggregate_analysis", "ladder95",
                        "verdicts")
# (key, label shown, colour). The order and the colours are the ones
# `two_models_hour_reqgoodput.pdf` uses (YlGnBu, FluidServe darkest), so a
# colour names the same arm in both figures.
# ⚠ "Llumnix" HERE IS LLUMNIX'S SLO-AWARE POLICY (`--scheduling-policy slo`),
# shortened at the author's request as in the other hour figures; the caption
# has to say which policy it is.
ARMS = [
    ("vLLM", "vLLM", "#ffffcc",
     ("260901_2137_exp109r1_vllmcachet75_shift",
      "260901_2010_exp109r2_vllmcachet75_shift")),
    ("Llumnix", "Llumnix", "#a1dab4",
     ("260831_2346_exp109r1_slot75_shift",
      "260901_1856_exp109r2_slot75_shift")),
    ("PolyServe", "PolyServe", "#41b6c4",
     ("260831_2232_exp109r1_polyservept75_shift",
      "260901_1742_exp109r2_polyservept75_shift")),
    ("llm-d", "llm-d", "#c2a5cf",
     ("260831_2128_exp109r1_llmdslot75_shift",
      "260901_1637_exp109r2_llmdslot75_shift")),
    ("FluidServe", "FluidServe", "#253494",
     ("260831_2015_exp109r1_fsv3capgnofrct75_shift",
      "260901_1514_exp109r2_fsv3capgnofrct75_shift")),
]
CLASSES = [("chat", "Chat"), ("swe", "Agent"), ("deepresearch", "Deep Research")]
# Panel (c) only: a pooled group first, at the author's request (2026-09-13).
# (a) and (b) keep the three classes -- a pooled latency percentile moves with
# the class mix each arm finished, which differs by arm, and would not be
# comparable across arms the way the pooled outcome shares are.
C_GROUPS = [("all", "Total")] + CLASSES
BUDGET = {"chat": (5.0, 50.0), "swe": (7.0, 75.0), "deepresearch": (10.0, 100.0)}
# YlGnBu steps, at the author's request (2026-09-13, replacing an RdYlBu
# trio): the darkest, blue, for the requests that met their SLO, the middle
# teal for those served late, the palest green for those refused -- so the
# bands keep the order they were read in, lightest = nothing was served.
# ⚠ THESE ARE EXACTLY THREE OF THE ARM COLOURS IN (a) AND (b): #a1dab4 is
# Llumnix, #41b6c4 PolyServe, #2c7fb8 llm-d. In (c) colour is an outcome and
# the arm is the tick label, but on one figure the same hex naming an arm in
# one panel and an outcome in the next invites the wrong match; the caption
# has to say that (c)'s colours are outcomes.
SEG = [("met", "SLO Attained", "#2c7fb8", None),
       ("missed", "SLO Missed", "#41b6c4", None),
       ("rejected", "Rejected", "#a1dab4", None),
       ("unfinished", "Unfinished", "#ffffff", "////")]
# 10 s, at the author's request (2026-09-13). ⚠ IT IS ALSO DEEP RESEARCH'S
# BUDGET, so that class's SLO rule lies on the top frame, and it cuts two
# Llumnix bars besides the vLLM router's (agent P95 14.7 s, deep research P50
# 17.2 / P95 23.9 s); those are written above the panel like the rest.
TTFT_TOP = 10.0
TBT_TOP = 120.0     # ms: holds every P95 but the vLLM router's (the next is 109)
FIG_W, FIG_H = ps.COL_W, 3.70
FIG_H_GRID = 2.75
LABEL = 6.0
TICK = 5.5


def latency(run):
    r = load_run(os.path.join(ROOT, "results", run))
    done = r[~r["rejected"] & ~r["errored"] & ~r["cutoff"]
             & r["first_token_latency"].notna()]
    out = {}
    for c, _ in CLASSES:
        g = done[done["class"] == c]
        tt = pd.to_numeric(g["first_token_latency"], errors="coerce").dropna()
        it = pd.to_numeric(g["itl_ms"], errors="coerce").dropna()
        out[c] = dict(ttft_p50=tt.quantile(.5), ttft_p95=tt.quantile(.95),
                      tbt_p50=it.quantile(.5), tbt_p95=it.quantile(.95),
                      n_done=len(g))
    return out


def outcome(run):
    v = pd.read_csv(os.path.join(VERDICTS, run + ".csv"))
    out = {}
    for c, _ in C_GROUPS:
        # "all" pools every class PER REQUEST: each arrival counts once, so a
        # class weighs what its share of the arrivals is (chat is about two
        # thirds). It is not the mean of the three class bars.
        g = v if c == "all" else v[v["class"] == c]
        rej = g["rejected"].astype(bool)
        cut = g["cutoff"].astype(bool) & ~rej
        met = g["ladder_ok"].astype(bool) & ~rej & ~cut
        n = float(len(g))
        out[c] = dict(met=100 * met.sum() / n, rejected=100 * rej.sum() / n,
                      unfinished=100 * cut.sum() / n,
                      missed=100 * (n - met.sum() - rej.sum() - cut.sum()) / n,
                      arrivals=int(n))
    return out


def collect():
    rows = []
    for key, label, _c, runs in ARMS:
        for rep, run in enumerate(runs, 1):
            print(f"  {label:11s} repeat {rep}  {run}", flush=True)
            lat, out = latency(run), outcome(run)
            for c, _ in C_GROUPS:
                rows.append(dict(arm=label, repeat=rep, run=run, cls=c,
                                 **lat.get(c, {}), **out[c]))
    return pd.DataFrame(rows)


def build(df, out_pdf, layout="rows"):
    """`rows`: (a), (b), (c) one above the other. `grid`: (a) and (b) side by
    side on top, (c) the full width underneath (2026-09-13, at the author's
    request), with the same bars made narrower by the narrower panels."""
    grid = layout == "grid"
    mean = df.groupby(["arm", "cls"]).mean(numeric_only=True)
    n_arm = len(ARMS)
    gap = 1.2                           # a group is n_arm slots plus this gap
    xpos, centres = {}, []
    for i, (c, _) in enumerate(CLASSES):
        base = i * (n_arm + gap)
        centres.append(base + (n_arm - 1) / 2.0)
        for j, (_k, label, _col, _r) in enumerate(ARMS):
            xpos[(c, label)] = base + j
    xmin, xmax = -0.7, (len(CLASSES) - 1) * (n_arm + gap) + n_arm - 0.3

    style = {**ps.STYLE, "xtick.labelsize": TICK, "ytick.labelsize": TICK,
             "axes.labelsize": LABEL}
    with plt.rc_context(style):
        fig_h = FIG_H_GRID if grid else FIG_H
        if grid:
            fig = plt.figure(figsize=(FIG_W, fig_h))
            gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
            ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                  fig.add_subplot(gs[1, :])]
        else:
            fig, ax = plt.subplots(3, 1, figsize=(FIG_W, fig_h),
                                   gridspec_kw=dict(height_ratios=[1, 1, 1.15]))
        over_notes, over_txt = [], []
        for pi, (a, stat, top, unit) in enumerate(
                [(ax[0], "ttft", TTFT_TOP, "s"), (ax[1], "tbt", TBT_TOP, "ms")]):
            for c, _ in CLASSES:
                for _k, label, col, _r in ARMS:
                    x = xpos[(c, label)]
                    p50 = float(mean.loc[(label, c), f"{stat}_p50"])
                    p95 = float(mean.loc[(label, c), f"{stat}_p95"])
                    # P50 is the bar and P95 is a WHISKER from the bar's top
                    # (2026-09-13, at the author's request; it was an open bar
                    # behind). ⚠ IT LOOKS LIKE AN ERROR BAR AND IS NOT ONE: it
                    # is the 95th percentile of the same requests, not a spread
                    # over repeats, so the key names it P95 and the caption has
                    # to say so. A whisker that crosses the cut stops at the
                    # frame and its value is written above the panel.
                    a.bar(x, min(p50, top), width=0.78, facecolor=col,
                          edgecolor="#000000", linewidth=0.4, zorder=3)
                    if p50 < top:
                        a.errorbar([x], [p50], yerr=[[0.0], [min(p95, top) - p50]],
                                   fmt="none", ecolor="#000000", elinewidth=0.6,
                                   capsize=1.6 if p95 <= top else 0.0,
                                   capthick=0.6, zorder=4)
                    over = [(q, v) for q, v in (("P50", p50), ("P95", p95))
                            if v > top]
                    if over:
                        # HORIZONTAL, ONE VALUE PER LINE, P95 ON TOP (2026-09-13,
                        # at the author's request; they were rotated). A bar slot
                        # is about 0.16 in, so "393/1,716" on one line runs into
                        # the neighbouring label; stacked, with thousands as
                        # "1.7k", each line is at most four characters. The top
                        # line is the higher quantile, matching the whisker
                        # above the bar. Their spacing is checked below.
                        def short(v):
                            return (f"{v / 1000:.1f}k" if v >= 1000 else
                                    f"{v:.0f}" if v >= 100 else f"{v:.1f}")
                        txt = "\n".join(short(v) for _q, v in reversed(over))
                        # An ANNOTATION offset in points, not a text at a data
                        # height: a label that would sit on its neighbour's is
                        # lifted after the layout (see below), and an offset in
                        # points survives the layout being redone.
                        over_txt.append(a.annotate(
                            txt, xy=(x, top), xytext=(0, 1.2),
                            textcoords="offset points", ha="center",
                            va="bottom", fontsize=4.0 if grid else 4.4,
                            linespacing=1.0, annotation_clip=False, zorder=6))
                        over_notes.append((stat, c, label, over))
                # the class's nominal promise, one rule across its group
                b = BUDGET[c][0 if stat == "ttft" else 1]
                x0 = xpos[(c, ARMS[0][1])] - 0.45
                x1 = xpos[(c, ARMS[-1][1])] + 0.45
                a.plot([x0, x1], [b, b], color="#d62728", lw=0.8, ls="--",
                       zorder=5)
            a.set_ylim(0, top)
            a.set_xlim(xmin, xmax)
            a.set_xticks(centres)
            # side by side a panel is about 1.3 in wide and "Deep Research" on
            # one line is wider than a group, so it breaks there
            a.set_xticklabels([(l.replace(" ", "\n") if grid else l)
                               for _, l in CLASSES],
                              fontsize=5.4 if grid else LABEL,
                              linespacing=1.0, fontstyle="italic")
            a.tick_params(axis="x", length=0)
            a.grid(axis="y", **ps.GRID)
            a.set_axisbelow(True)
        ax[0].set_ylabel("TTFT P50/P95 (s)" if grid else "TTFT (s)")
        ax[0].set_yticks([0, 2.5, 5, 7.5, 10])
        ax[1].set_ylabel("TBT P50/P95 (ms)" if grid else "TBT (ms)")
        ax[1].set_yticks([0, 25, 50, 75, 100])

        # (c): one stacked bar per arm per group, share of that group's
        # arrivals. It has FOUR groups (Total first), so its positions are its
        # own and not (a)'s and (b)'s.
        xpos_c, centres_c = {}, []
        for i, (c, _) in enumerate(C_GROUPS):
            base = i * (n_arm + gap)
            centres_c.append(base + (n_arm - 1) / 2.0)
            for j, (_k, label, _col, _r) in enumerate(ARMS):
                xpos_c[(c, label)] = base + j
        xmax_c = (len(C_GROUPS) - 1) * (n_arm + gap) + n_arm - 0.3
        a = ax[2]
        for c, _ in C_GROUPS:
            for _k, label, col, _r in ARMS:
                x = xpos_c[(c, label)]
                bottom = 0.0
                for key, _lab, fc, hatch in SEG:
                    v = float(mean.loc[(label, c), key])
                    a.bar(x, v, bottom=bottom, width=0.78, facecolor=fc,
                          edgecolor="#000000", linewidth=0.4, hatch=hatch,
                          zorder=3)
                    bottom += v
        # a hairline between the pooled group and the classes, so Total does
        # not read as a fourth class
        xs = 0.5 * (xpos_c[("all", ARMS[-1][1])] + xpos_c[(CLASSES[0][0], ARMS[0][1])])
        a.axvline(xs, color="#888888", lw=0.5, ls="-", zorder=1)
        a.set_ylim(0, 100)
        a.set_yticks([0, 25, 50, 75, 100])
        # "Arrived Requests", not "Arrivals": the denominator is every request
        # that ARRIVED in that class, rejected and unfinished included, and the
        # label says so without the caption.
        a.set_ylabel("Arrived Requests (%)")
        a.set_xlim(xmin, xmax_c)
        # The arm under every bar, since colour in (c) is the outcome and not
        # the arm; the class under each group of five.
        a.set_xticks([xpos_c[(c, l)] for c, _ in C_GROUPS for _k, l, _cc, _r in ARMS])
        a.set_xticklabels([ps.LEGEND_NAME.get(l, l)
                           for _ in C_GROUPS for _k, l, _cc, _r in ARMS],
                          rotation=50, ha="right", rotation_mode="anchor",
                          fontsize=4.8)
        a.tick_params(axis="x", length=1.5, pad=1.0)
        a.grid(axis="y", **ps.GRID)
        a.set_axisbelow(True)

        # Side by side the captions lose "(bar P50, whisker P95)", which is
        # 1.9 in of type under a 1.3 in panel; the y labels carry P50/P95
        # there instead, and the key names the whisker.
        cap = {0: "(a) Time to First Token" if grid else
                  "(a) Time to First Token (bar P50, whisker P95)",
               1: "(b) Time between Tokens" if grid else
                  "(b) Time between Tokens (bar P50, whisker P95)",
               2: "(c) Request SLO Attainment"}
        # The captions are the x labels, so `tight_layout` reserves their room
        # and a caption can never sit on the panel below. Panel (c)'s label
        # starts with an EMPTY line: its ticks carry the arm names, and the
        # class names are drawn into that reserved line after the layout, at
        # the centre of each group.
        ax[0].set_xlabel(cap[0], fontsize=LABEL, labelpad=2)
        ax[1].set_xlabel(cap[1], fontsize=LABEL, labelpad=2)
        ax[2].set_xlabel("\n" + cap[2], fontsize=LABEL, labelpad=2,
                         linespacing=1.35)

        arm_h = [Patch(facecolor=col, edgecolor="#000000", linewidth=0.4)
                 for _k, _l, col, _r in ARMS]
        arm_l = [l for _k, l, _c, _r in ARMS]
        # the key's P95 entry is a real errorbar container drawn off-axes, so
        # the key shows the same glyph the panels do
        p95_h = ax[1].errorbar([np.nan], [np.nan], yerr=[[0.0], [1.0]],
                               fmt="none", ecolor="#000000", elinewidth=0.6,
                               capsize=1.6, capthick=0.6)
        extra_h = [p95_h, Line2D([], [], color="#d62728", lw=0.8, ls="--")]
        extra_l = ["P95", "SLO"]
        seg_h = [Patch(facecolor=fc, edgecolor="#000000", linewidth=0.4,
                       hatch=h) for _k, _l, fc, h in SEG]
        seg_l = [l for _k, l, _f, _h in SEG]
        ax[2].legend(seg_h, seg_l, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                     ncol=4, frameon=False, fontsize=ps.KEY_FS,
                     columnspacing=0.8, borderaxespad=0.15,
                     handler_map=ps.square_handler(seg_h), **ps.KEY_SQUARE)

        top_band = 0.16

        def lay():
            fig.tight_layout(rect=(0, 0, 1, 1 - top_band / fig_h), pad=0.3,
                             h_pad=0.5, w_pad=0.8)

        lay()
        # ⚠ SPREAD LABELS THAT WOULD SIT ON EACH OTHER SIDEWAYS, IN ONE ROW
        # (2026-09-13, at the author's request). Two neighbouring bars can both
        # cross the cut -- the vLLM router and Llumnix in Agent and Deep
        # Research -- and with narrow bars their labels overlap. Lifting one
        # above the other cleared them but stacked four lines of type between
        # panel (a) and the key. Instead the two are pushed apart by exactly the
        # overlap plus a hairline, half each way, so both stay at the same
        # height and each stays as close to its own bar as the other allows.
        # The space they move into is empty: left of the first bar of a group is
        # the gap between groups, and above the next arm's bar there is no
        # label, because that bar is inside the axis.
        PAD_PX = 2.0
        for axis in (ax[0], ax[1]):
            mine = sorted([t for t in over_txt if t.axes is axis],
                          key=lambda t: t.xy[0])
            for _ in range(8):
                fig.canvas.draw()
                rend0 = fig.canvas.get_renderer()
                moved = False
                for t1, t2 in zip(mine, mine[1:]):
                    e1, e2 = t1.get_window_extent(rend0), t2.get_window_extent(rend0)
                    need = e1.x1 + PAD_PX - e2.x0
                    if need > 0:
                        half_pt = need / 2.0 / fig.dpi * 72.0
                        t1.xyann = (t1.xyann[0] - half_pt, t1.xyann[1])
                        t2.xyann = (t2.xyann[0] + half_pt, t2.xyann[1])
                        moved = True
                if not moved:
                    break
        lay()
        p0 = ax[0].get_position()
        arm_h, arm_l = ps.legend_items(arm_h, arm_l)
        # The key sits as high as it can without leaving the page. At the
        # shared 6.5 pt size it is taller than it was at 5.4, so the anchor is
        # not a constant any more: it is lowered by whatever overflows, which
        # is measured. The two layouts have different canvas heights and would
        # need two constants otherwise.
        anchor_y = 0.998
        # The three-row layout centres the key over panel (a); at the shared
        # 6.5 pt the key is 3.16 in and that anchor pushes it off the right
        # edge, so it is centred on the figure instead. The grid layout already
        # centres on the figure.
        anchor_x = 0.5 if grid else 0.5 * (p0.x0 + p0.x1)
        # One point larger than the shared size WHEN IT FITS (2026-09-18, at
        # the author's request): the larger size is tried first and kept only
        # if the key stays inside the canvas, so the figure never grows.
        key = None
        for key_fs in (ps.KEY_FS + 1.0, ps.KEY_FS):
            if key is not None:
                key.remove()
                key = None
            trial = fig.legend(
                arm_h + extra_h, arm_l + extra_l, loc="upper center",
                bbox_to_anchor=(0.5, anchor_y), ncol=len(arm_l) + 2,
                frameon=False, fontsize=key_fs, columnspacing=0.6,
                borderaxespad=0.0, handler_map=ps.square_handler(arm_h),
                **ps.KEY_SQUARE)
            fig.canvas.draw()
            w_in = trial.get_window_extent(
                fig.canvas.get_renderer()).width / fig.dpi
            trial.remove()
            print(f"  key at {key_fs:.1f} pt would be {w_in:.2f} in wide "
                  f"(canvas {FIG_W:.3f} in)"
                  + ("  -> too wide, not used" if w_in > FIG_W - 0.02 else
                     "  -> used"))
            if w_in <= FIG_W - 0.02:
                break
        for _ in range(5):
            if key is not None:
                key.remove()
            key = fig.legend(
                arm_h + extra_h, arm_l + extra_l, loc="upper center",
                bbox_to_anchor=(anchor_x, anchor_y),
                ncol=len(arm_l) + 2, frameon=False, fontsize=key_fs,
                columnspacing=0.6, borderaxespad=0.0,
                handler_map=ps.square_handler(arm_h), **ps.KEY_SQUARE)
            fig.canvas.draw()
            kb = key.get_window_extent(fig.canvas.get_renderer())
            W_px = fig.get_size_inches()[0] * fig.dpi
            if kb.x1 > W_px - 1.0 or kb.x0 < 1.0:
                anchor_x = 0.5
                continue
            over = (kb.y1 - fig.get_size_inches()[1] * fig.dpi + 1.0)
            if over <= 0:
                break
            anchor_y -= over / (fig.get_size_inches()[1] * fig.dpi)
        print(f"  arm key: {kb.width / fig.dpi:.2f} in wide on a "
              f"{FIG_W:.3f} in canvas, anchored at {anchor_y:.4f}"
              + ("  ⚠ WIDER THAN THE CANVAS" if kb.width / fig.dpi > FIG_W
                 else ""))

        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        # class names into the empty first line of (c)'s x label
        lab = ax[2].xaxis.label.get_window_extent(rend)
        line_h = (lab.y1 - lab.y0) / 2.0
        yc = inv.transform((0, lab.y1 - 0.5 * line_h))[1]
        for (c, name), xc in zip(C_GROUPS, centres_c):
            xf = inv.transform(ax[2].transData.transform((xc, 0)))[0]
            # Class names in italic, "Total" upright (2026-09-13, at the
            # author's request): the three classes are one kind of group and
            # the pooled one is another, and the type says so as well as the
            # hairline between them.
            fig.text(xf, yc, name, ha="center", va="center", fontsize=LABEL,
                     fontstyle="normal" if c == "all" else "italic")

        # checks: nothing off the canvas; the top key clears (a)'s cut labels
        fig.canvas.draw()
        W, H = fig.get_size_inches() * fig.dpi
        # ⚠ THE LAYOUT MOVES AFTER THE KEY IS PLACED (the class names under
        # panel (c) are added and the figure is drawn again), so the key is
        # measured once more here and lowered if it now leaves the page.
        H_in = fig.get_size_inches()[1]
        for _ in range(4):
            e = key.get_window_extent(rend)
            over = e.y1 - H_in * fig.dpi + 1.0
            if over <= 0:
                break
            x_anchor = key.get_bbox_to_anchor()
            anchor_y -= over / (H_in * fig.dpi)
            key.set_bbox_to_anchor((anchor_x, anchor_y))
            fig.canvas.draw()
        arts = list(fig.legends) + [a_.xaxis.label for a_ in ax] + list(fig.texts)
        for art in arts:
            e = art.get_window_extent(rend)
            if e.x0 < -0.5 or e.x1 > W + 0.5 or e.y0 < -0.5 or e.y1 > H + 0.5:
                print(f"  ⚠ off the canvas: "
                      f"{getattr(art, 'get_text', lambda: 'key')()!r} "
                      f"x {e.x0:.1f}..{e.x1:.1f} of {W:.1f}, "
                      f"y {e.y0:.1f}..{e.y1:.1f} of {H:.1f}")
        n_hit = 0
        for i, t1 in enumerate(over_txt):
            for t2 in over_txt[i + 1:]:
                if t1.axes is not t2.axes:
                    continue
                e1, e2 = t1.get_window_extent(rend), t2.get_window_extent(rend)
                if e1.x0 < e2.x1 and e2.x0 < e1.x1 and e1.y0 < e2.y1 and e2.y0 < e1.y1:
                    n_hit += 1
        lifted = sum(1 for t in over_txt if abs(t.xyann[0]) > 0.05)
        print(f"  labels above the panels: {len(over_txt)}, moved sideways {lifted}, "
              f"overlapping pairs {n_hit}" + ("  ⚠ OVERLAP" if n_hit else ""))
        key_y0 = fig.legends[0].get_window_extent(rend).y0
        ink_a = ax[0].get_tightbbox(rend).y1
        print(f"  top key clears panel (a)'s ink by {(key_y0 - ink_a) / fig.dpi:+.3f} in")
        # Which pairs can collide depends on the layout: stacked, each panel
        # against the one below; side by side, (a) against (b) across, and
        # both against (c) below.
        tb = [a_.get_tightbbox(rend) for a_ in ax]
        pairs = ([("a", "b", (tb[1].x0 - tb[0].x1)), ("a", "c", tb[0].y0 - tb[2].y1),
                  ("b", "c", tb[1].y0 - tb[2].y1)] if grid else
                 [("a", "b", tb[0].y0 - tb[1].y1), ("b", "c", tb[1].y0 - tb[2].y1)])
        for n1, n2, g_ in pairs:
            print(f"  panel ({n1}) clears panel ({n2}) by {g_ / fig.dpi:+.3f} in"
                  + ("  ⚠ OVERLAP" if g_ < 0 else ""))
        for stat, c, label, over in over_notes:
            print(f"  cut: {stat} {c:12s} {label:10s} " + ", ".join(
                f"{q} {v:,.1f}" for q, v in over))
        ps.save(fig, out_pdf)


def main():
    df = collect()
    out = ps.final("class_latency_outcome_rows")
    spread = df.groupby(["arm", "cls"]).agg(lambda s: s.max() - s.min()
                                            if s.dtype.kind in "fi" else s.iloc[0])
    build(df, out + ".pdf")
    build(df, ps.final("class_latency_outcome_grid.pdf"), layout="grid")
    df.to_csv(out + ".csv", index=False, float_format="%.4f")
    print(f"wrote {out}.csv  ({len(df)} rows, one per arm x repeat x class)")
    s = spread[["ttft_p50", "ttft_p95", "tbt_p50", "tbt_p95", "met",
                "rejected"]]
    print("repeat spread (max - min):")
    print(s.round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
