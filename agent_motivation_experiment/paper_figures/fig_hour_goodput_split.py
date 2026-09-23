#!/usr/bin/env python3
"""Paper figures: the three hour-trace goodput results, split by quantity.

  hour_token_goodput_3panel.pdf     7.0 in wide (`figure*`)  token goodput over the hour
  hour_request_goodput_3panel.pdf   3.335 in wide (`figure`) request goodput per arm
  hour_goodput_split.csv            the bar values, one row per model x arm x run

The two figures take apart `two_models_hour_reqgoodput.pdf` (Llama-3.1-70B and
Qwen2.5-72B) and `exp126_two_models_hour_reqgoodput.pdf` (Llama-3.1-8B), at the
author's request (2026-09-14): each of the three pairs was a timeline beside a
bar panel, and here the timelines form one figure and the bars another.

  (a) Llama-3.1-70B, 4 instances, EXP-109, standard budgets
  (b) Qwen2.5-72B,   4 instances, EXP-113, standard budgets
  (c) Llama-3.1-8B,  8 instances, EXP-126, HALVED budgets

NOTHING IS RECOMPUTED DIFFERENTLY. The readers, the drawn repeat, the span and
the colours are imported from the two source scripts, so a panel here draws
exactly what the same panel there draws.

⚠ (c) IS NOT THE SAME EXPERIMENT AS (a) AND (b), and the caption has to say so.
It is a different model on eight instances, the trace scaled x6.20, and every
budget halved (chat 2.5 s / 25 ms, deep research 5 s / 50 ms, agent 3.5 s /
38 ms) against the standard ones (5 s / 50 ms, 10 s / 100 ms, 7 s / 75 ms).
Its axes are its own.

⚠ THE SPANS DIFFER. (a) is scored to 59.25 min and (b) to 58.75 min -- the
earliest minute at which an arm with admission control has more than 20% of a
window still running when its run ends. (c) is scored over the full 60 min,
as its source figure is: there one arm's drain cut the span to 56.75 min, and
the author chose the whole hour instead.

The line and the bar are repeat 2 of each arm; the thin mark on each bar spans
both repeats. In (a) and (b) the Llumnix arm is Llumnix's SLO-aware policy
(`--scheduling-policy slo`).

    python3 paper_figures/fig_hour_goodput_split.py
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_style as ps  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TM = _load("tm", os.path.join(HERE, "fig_two_models_hour.py"))
E126 = _load("e126", os.path.join(HERE, "fig_two_models_hour_exp126.py"))
TM.apply_ramp()          # the YlGnBu ramp both source figures are drawn in

# (label shown, key in the two-model script, label in the EXP-126 script)
ARMS = [("vLLM", "vLLM", "vLLM-router"),
        ("Llumnix", "Llumnix SLO", "Llumnix"),
        ("PolyServe", "PolyServe", "PolyServe"),
        ("llm-d", "llm-d", "llm-d"),
        ("FluidServe", "FluidServe", "FluidServe")]
PANELS = [("a", "Llama-3.1-70B"), ("b", "Qwen2.5-72B"), ("c", "Llama-3.1-8B")]
# Bar tops: (a) as its source draws it (10% over the tallest mark), (b) and (c)
# the fixed tops the author set on the source figures.
BAR_TOP = {"a": None, "b": 15.0, "c": 125.0}
BAR_TICKS = {"a": None, "b": [0, 5, 10, 15], "c": [0, 25, 50, 75, 100, 125]}
STROKE = [pe.Stroke(linewidth=1.5, foreground="#9a9a9a"), pe.Normal()]
SHRINK = 2.0
TOK_W, TOK_H = ps.TEXT_W, 1.62
REQ_W, REQ_H = ps.COL_W, 1.52


def collect():
    """panel -> dict(dur, arms: label -> (x, gp, bar value, lo, hi, runs))."""
    out = {}
    two = TM.collect()
    for pid, model in PANELS[:2]:
        per = two[model]
        dur = TM.cut_minute(per)
        arms = {}
        for label, key, _ in ARMS:
            d = per[key]
            g2 = TM.split_counts(d["v2"], dur)["met"] / (dur * 60.0)
            g1 = TM.split_counts(d["v1"], dur)["met"] / (dur * 60.0)
            keep = d["x"] <= dur
            arms[label] = dict(x=d["x"][keep], gp=d["gp"][keep], bar=g2,
                               lo=min(g1, g2), hi=max(g1, g2),
                               runs=[d["run2"], d["run1"]], vals=[g2, g1])
        out[pid] = dict(dur=dur, arms=arms)
    per = E126.collect()
    dur = E126.HOUR_MIN            # the whole hour, as the source figure draws it
    arms = {}
    for label, _, key in ARMS:
        recs = per[key]
        d = recs[-1]               # the newest run (repeat 2) draws the line and bar
        gs = [E126.met_per_s(r["v"], dur)[0] for r in recs]
        keep = d["x"] <= dur
        arms[label] = dict(x=d["x"][keep], gp=d["gp"][keep], bar=gs[-1],
                           lo=min(gs), hi=max(gs),
                           runs=[r["run"] for r in recs][::-1], vals=gs[::-1])
    out["c"] = dict(dur=dur, arms=arms)
    return out


def colour_style(label):
    key = dict((l, k) for l, k, _ in ARMS)[label]
    return TM.STYLE_OF[key]


def fit_key(fig, handles, labels, width_in, ncol, sizes, anchor_y, **kw):
    """Largest type size at which the key fits the canvas width."""
    for fs in sizes:
        key = fig.legend(handles, labels, loc="upper center",
                         bbox_to_anchor=(0.5, anchor_y), ncol=ncol,
                         frameon=False, fontsize=fs, borderaxespad=0.0, **kw)
        fig.canvas.draw()
        e = key.get_window_extent()
        if e.x0 >= 1.0 and e.x1 <= width_in * fig.dpi - 1.0:
            return key, fs
        key.remove()
    return key, sizes[-1]


def token_figure(data, out):
    style = dict(ps.STYLE)
    for k in ("axes.labelsize", "xtick.labelsize", "ytick.labelsize"):
        style[k] = max(4.0, style[k] - SHRINK)
    with plt.rc_context(style):
        fig, ax = plt.subplots(1, 3, figsize=(TOK_W, TOK_H))
        handles, labels = [], []
        tops = {}
        for i, (pid, model) in enumerate(PANELS):
            a = ax[i]
            for label, _, _ in ARMS:
                d = data[pid]["arms"][label]
                col, ls = colour_style(label)
                line, = a.plot(d["x"], d["gp"], color=col, ls=ls, lw=0.8,
                               path_effects=STROKE)
                if i == 0:
                    handles.append(line)
                    labels.append(label)
            tops[pid] = max(float(d["gp"].max()) for d in data[pid]["arms"].values())
            a.set_xlim(0, 60)
            a.set_xticks([0, 15, 30, 45, 60])
            a.yaxis.set_major_formatter(ps.kfmt())
            a.set_ylabel("Token Goodput (t/s)")
            a.set_xlabel(f"Time (minutes)\n({pid}) {model}", labelpad=1.5,
                         linespacing=1.6, fontsize=style["axes.labelsize"])
            a.grid(axis="both", **ps.GRID)
            a.set_axisbelow(True)
        # (a) and (b) share one scale, as they did side by side in their source
        # figure; (c) is a different fleet and model and keeps its own.
        ab = max(tops["a"], tops["b"]) * 1.08
        ax[0].set_ylim(0, ab)
        ax[1].set_ylim(0, ab)
        ax[2].set_ylim(0, tops["c"] * 1.08)
        # the caption line of each x label at the paper's 8 pt, the minutes line
        # at the panels' size: set the label as two texts is not possible, so
        # the whole label is at the panel size and the caption is redrawn below
        for i, (pid, model) in enumerate(PANELS):
            ax[i].set_xlabel("Time (minutes)", labelpad=1.5)

        top_band = 0.24
        bot_band = 0.19
        fig.tight_layout(rect=(0, bot_band / TOK_H, 1, 1 - top_band / TOK_H),
                         w_pad=1.6, pad=0.3)
        handles, labels = ps.legend_items(handles, labels)
        key, fs = fit_key(fig, handles, labels, TOK_W, len(labels),
                          (8.0, 7.5, 7.0), 0.998, columnspacing=1.2,
                          handlelength=1.9, handletextpad=0.4)
        caps = []
        for i, (pid, model) in enumerate(PANELS):
            p = ax[i].get_position()
            caps.append(fig.text(0.5 * (p.x0 + p.x1), 0.012,
                                 f"({pid}) {model}",
                                 ha="center", va="bottom", fontsize=8))
        check(fig, ax, key, caps, "token")
        ps.save(fig, out)


def request_figure(data, out):
    style = dict(ps.STYLE)
    for k in ("axes.labelsize", "xtick.labelsize", "ytick.labelsize"):
        style[k] = max(4.0, style[k] - SHRINK)
    rows = []
    with plt.rc_context(style):
        fig, ax = plt.subplots(1, 3, figsize=(REQ_W, REQ_H))
        for i, (pid, model) in enumerate(PANELS):
            a = ax[i]
            bar_hi = 0.0
            for j, (label, _, _) in enumerate(ARMS):
                d = data[pid]["arms"][label]
                col, _ls = colour_style(label)
                a.bar([j], [d["bar"]], width=0.72, color=col,
                      edgecolor="#000000", linewidth=0.5)
                # ⚠ NO REPEAT RANGE MARK (2026-09-18, at the author's request:
                # the two repeats are close enough that the mark is not
                # legible at this size). THE SPREAD IS STILL IN THE CSV,
                # and the caption has to give n and the spread in words.
                bar_hi = max(bar_hi, d["hi"])
                for run, val in zip(d["runs"], d["vals"]):
                    rows.append(dict(panel=pid, model=model, arm=label,
                                     run=run, req_goodput_s=val,
                                     drawn=(val == d["bar"] and run == d["runs"][0]),
                                     span_min=data[pid]["dur"],
                                     budgets=("halved" if pid == "c"
                                              else "standard")))
            a.set_xlim(-0.7, len(ARMS) - 0.3)
            a.set_xticks([])
            a.set_ylim(0, BAR_TOP[pid] or bar_hi * 1.10)
            if BAR_TICKS[pid]:
                a.set_yticks(BAR_TICKS[pid])
            a.grid(axis="y", **ps.GRID)
            a.set_axisbelow(True)
        ax[0].set_ylabel("Request Goodput (r/s)")

        handles = [Patch(facecolor=colour_style(l)[0], edgecolor="#000000",
                         linewidth=0.5) for l, _, _ in ARMS]
        labels = [l for l, _, _ in ARMS]
        handles, labels = ps.legend_items(handles, labels)
        top_band = 0.20
        bot_band = 0.17
        fig.tight_layout(rect=(0, bot_band / REQ_H, 1, 1 - top_band / REQ_H),
                         w_pad=0.9, pad=0.3)
        # ⚠ FIXED AT 8 pt, NOT FITTED (2026-09-18, at the author's request:
        # the same size as the key of `two_models_hour_reqgoodput.pdf`, which
        # takes the project default). The ladder is kept as a single rung so a
        # later change of names cannot silently shrink the key instead of
        # reporting that it no longer fits.
        key, fs = fit_key(fig, handles, labels, REQ_W, len(labels),
                          (8.0,), 0.998,
                          columnspacing=0.7, handlelength=1.0,
                          handleheight=1.0, handletextpad=0.3)
        caps = []
        for i, (pid, model) in enumerate(PANELS):
            p = ax[i].get_position()
            caps.append(fig.text(0.5 * (p.x0 + p.x1), 0.012,
                                 f"({pid}) {model}", ha="center",
                                 va="bottom", fontsize=7))
        print(f"  request figure key at {fs:.1f} pt")
        check(fig, ax, key, caps, "request")
        ps.save(fig, out)
    return pd.DataFrame(rows)


def check(fig, ax, key, caps, name):
    """Nothing off the canvas, no caption on another, key clear of the panels."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W, H = fig.get_size_inches() * fig.dpi
    for art in [key] + caps:
        e = art.get_window_extent(rend)
        if e.x0 < -0.5 or e.x1 > W + 0.5 or e.y0 < -0.5 or e.y1 > H + 0.5:
            print(f"  ⚠ {name}: off the canvas: "
                  f"{getattr(art, 'get_text', lambda: 'key')()!r}")
    ce = [c.get_window_extent(rend) for c in caps]
    for c1, c2 in zip(ce, ce[1:]):
        if c1.x1 > c2.x0:
            print(f"  ⚠ {name}: two captions overlap")
    ink_top = max(a.get_tightbbox(rend).y1 for a in ax)
    ink_bot = min(a.get_tightbbox(rend).y0 for a in ax)
    print(f"  {name}: key clears the panels by "
          f"{(key.get_window_extent(rend).y0 - ink_top) / fig.dpi:+.3f} in, "
          f"captions clear them by "
          f"{(ink_bot - max(e.y1 for e in ce)) / fig.dpi:+.3f} in")


def main():
    data = collect()
    for pid, model in PANELS:
        d = data[pid]
        print(f"({pid}) {model}: span 0-{d['dur']:.2f} min; request goodput "
              + ", ".join(f"{l} {d['arms'][l]['bar']:.2f}" for l, _, _ in ARMS))
    token_figure(data, ps.final("hour_token_goodput_3panel.pdf"))
    df = request_figure(data, ps.final("hour_request_goodput_3panel.pdf"))
    out = ps.final("hour_goodput_split.csv")
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"wrote {out}  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
