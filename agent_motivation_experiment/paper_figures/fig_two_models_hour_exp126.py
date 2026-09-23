#!/usr/bin/env python3
"""Paper figure: the hour trace at the HALVED budgets, three control planes.

  exp126_two_models_hour_reqgoodput.pdf   7.0 x 1.59 in, `figure*`, width=\\textwidth
  exp126_two_models_hour_reqgoodput.csv   exactly the values drawn

This is `two_models_hour_reqgoodput.pdf` rebuilt on EXP-126: same shape, same
two units (tokens per second on the left of a pair, requests per second on its
right), same PuBuGn ramp, same type sizes.

⚠ NOTHING HERE MAY SIT BESIDE A STANDARD-BUDGET NUMBER. EXP-126 halves all six
budgets -- chat 5 s / 50 ms becomes 2.5 s / 25 ms, deep research 10 s / 100 ms
becomes 5 s / 50 ms, agent 7 s / 75 ms becomes 3.5 s / 38 ms -- and both the
POLICY and the SCORER are given the halved set, so a request that met its rule
in EXP-109 can miss it here without anything about the system changing.
PolyServe's tier boundaries move with them (chat 25 / agent 38 / deep research
50), so its arm differs from `polyservept75` in how it PARTITIONS and not only
in how it is scored.

⚠ THE SECOND PAIR IS EMPTY ON PURPOSE. It is reserved for the smaller Qwen
model, whose runs do not exist yet; the axes are drawn so the layout is the one
the finished figure will have, and the word "pending" is printed in the panel so
that an empty pair is never mistaken for a measured zero. That word has to be
gone before this figure goes in the paper.

⚠ ONE REPEAT PER ARM SO FAR, so no cell carries a range mark. The script draws
one as soon as a second repeat of that arm exists: it globs the results
directory rather than naming runs, and every complete run of an arm goes into
the bar's range while the newest one draws the timeline. A bar with no mark is
therefore a cell measured once, not a cell whose repeats agree -- the count is
printed per arm and belongs in the caption.

SCORING is the fixed ladder rule of this directory -- token i of a request is on
time if it arrives within TTFT_SLO + i x TBT_SLO of the send, the request is on
time if at least 95% of its tokens are, goodput counts tokens that met their own
deadline -- evaluated at the halved budgets. Its verdicts live in a SEPARATE
directory, `results/aggregate_analysis/ladder95_halved/verdicts`, because a
halved-budget verdict file dropped in beside the standard ones would be picked
up silently by every other figure in this directory.

  request goodput = requests that met their rule in the drawn span / that span

DATA. EXP-126 (2026-09-10), the one-hour mixture-shift trace at x6.20
(67-279 req/s), 8 x Llama-3.1-8B-Instruct, TP=1.

    python3 paper_figures/fig_two_models_hour_exp126.py
"""
import argparse
import glob
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis_scripts", "request_level"))

from paper_style import TEXT_W, STYLE, GRID, kfmt, save  # noqa: E402
from exp41_dynamic_timeline import windows, WIN  # noqa: E402

VERDICTS = os.path.join(ROOT, "results", "aggregate_analysis",
                        "ladder95_halved", "verdicts")
RESULTS = os.path.join(ROOT, "results")
GLOB = "*exp126*_shift62"
MIN_IN_WINDOW = 30
MAX_CUTOFF = 0.20
HOUR_MIN = 60.0

# (key in the directory name, label, colour, dash). The two FluidServe arms
# differ by one named flag and nothing else, so they are two tones of one hue
# and PolyServe is the third step of the ramp.
# ⚠ THE `fsv3ah1` ARM IS SCORED BUT NOT DRAWN (2026-09-11, at the author's
# request). It is the same policy with `--fluidserve-arrival-horizons` and
# nothing else, and on this trace at these budgets the two are indistinguishable
# -- 123.6 against 123.7 requests per second, a twentieth of the 1.9-point read
# floor EXP-125 measured for this workload -- so drawing both spends a third of
# the panel saying that. Its verdicts stay in
# `results/aggregate_analysis/ladder95_halved/verdicts`, and putting the line
# back is uncommenting the row below.
# ⚠ ONE COLOUR AND ONE DASH PER ARM ACROSS THE TWO HOUR FIGURES (2026-09-11).
# `two_models_hour_reqgoodput.pdf` paints the five arms with ColorBrewer PuBuGn
# in the paper's order, and this figure now takes the same entries for the four
# it draws, so a reader moving between the two never has to relearn a colour.
# The vLLM router's step of that ramp (#f6eff7) is simply unused here.
ARMS = [
    ("vllmcache",      "vLLM-router",         "#ffffcc", (0, (4, 1, 1, 1, 1, 1))),
    ("slot",           "Llumnix",             "#a1dab4", "--"),
    ("polyservep",     "PolyServe",           "#41b6c4", "-."),
    ("llmdslot",       "llm-d",               "#2c7fb8", (0, (6, 1.5))),
    ("fsv3capgnofrc",  "FluidServe",          "#253494", "-"),
    # ("fsv3ah1",      "FluidServe +KV arr.", "#7fcdbb", (0, (4, 1.5))),
]
# ⚠ THE FLEET SIZE IS NOT IN THE PANEL NAME (2026-09-11, at the author's
# request), so the caption has to carry it: eight instances of Llama-3.1-8B at
# TP=1. The same model on four instances is a different capacity and this figure
# does not say which it is.
PAIRS = [("Llama-3.1-8B", True), ("Qwen (small)", False)]
SEG_STROKE = [pe.Stroke(linewidth=1.5, foreground="#9a9a9a"), pe.Normal()]
# ⚠ THE CANVAS LOST THE KEY'S BAND (2026-09-11). With no key on this figure the
# 0.225 in reserved above the axes is empty, so the height comes down by the
# 0.185 in that is not needed as clearance for the topmost tick label. The
# panels themselves are unchanged, which is what lets this figure sit directly
# under `two_models_hour_reqgoodput.pdf` at the same panel size.
TOP_BAND = 0.040
FIG_H = 1.405
SHRINK = 2.0
# The bar axis is fixed at 0..125 requests per second with a tick every 25, on
# BOTH pairs: the tallest bar is 123.7 and a round ceiling just above it keeps
# the grid readable, and the empty pair reserved for the second model gets the
# same scale so the two halves can be read against each other when it is filled.
BAR_MAX = [125.0, 125.0]
BAR_TICK = 25.0


def runs_of(key):
    """Every complete EXP-126 run of one arm, oldest first.

    Complete means the verdict file exists: the scorer writes it only after a
    run is read end to end, so a run still being measured -- there is usually
    one -- cannot reach the figure through this path.
    """
    out = []
    for d in sorted(glob.glob(os.path.join(RESULTS, GLOB))):
        name = os.path.basename(d)
        if f"_{key}" not in name.replace("exp126h62r1_", "_") \
                .replace("exp126h62r2_", "_").replace("exp126h62r3_", "_"):
            continue
        if os.path.exists(os.path.join(VERDICTS, name + ".csv")):
            out.append(name)
    return out


def verdicts(run):
    return pd.read_csv(os.path.join(VERDICTS, run + ".csv"))


def goodput_series(v):
    dur = v["rel"].max()
    x, gp, cut = [], [], []
    for t, g in windows(v, dur):
        if len(g) < MIN_IN_WINDOW:
            continue
        x.append(t)
        gp.append(float((g["n_tokens"] - g["n_late"]).sum()) / WIN)
        cut.append(float(g["cutoff"].mean()))
    return np.array(x), np.array(gp), np.array(cut)


def met_per_s(v, minute):
    """Requests that met their rule per second, over minutes 0..`minute`."""
    w = v[v["rel"] <= minute * 60.0]
    met = int((w["ladder_ok"] & ~w["rejected"] & ~w["errored"]
               & ~w["cutoff"]).sum())
    return met / (minute * 60.0), met, len(w)


def collect():
    per = {}
    for key, label, _c, _ls in ARMS:
        rs = runs_of(key)
        if not rs:
            print(f"!! {label}: no scored run yet", file=sys.stderr)
            continue
        recs = []
        for r in rs:
            v = verdicts(r)
            x, gp, cut = goodput_series(v)
            recs.append(dict(run=r, v=v, x=x, gp=gp, cut=cut))
        per[label] = recs
    return per


def cut_minute(per):
    ends = []
    for recs in per.values():
        for d in recs:
            ok = d["cut"] <= MAX_CUTOFF
            ends.append(d["x"][ok].max() if ok.any() else d["x"].min())
    return min(ends) if ends else HOUR_MIN


def build(per, dur, out):
    style = dict(STYLE)
    for k in ("axes.labelsize", "xtick.labelsize", "ytick.labelsize"):
        style[k] = max(4.0, style[k] - SHRINK)
    rows = []
    with plt.rc_context(style):
        fig, ax = plt.subplots(1, 4, figsize=(TEXT_W, FIG_H),
                               gridspec_kw=dict(width_ratios=[1.72, 0.532,
                                                              1.72, 0.532]))
        handles, labels = [], []
        gp_ax, bar_ax = ax[0], ax[1]
        xs, vals, lo, hi, cols = [], [], [], [], []
        top = 0.0
        for j, (key, label, col, ls) in enumerate(ARMS):
            if label not in per:
                continue
            recs = per[label]
            d = recs[-1]                      # the newest run draws the line
            keep = d["x"] <= dur
            line, = gp_ax.plot(d["x"][keep], d["gp"][keep], color=col, ls=ls,
                               lw=0.8, path_effects=SEG_STROKE)
            top = max(top, float(d["gp"][keep].max()))
            handles.append(line); labels.append(label)
            gs = [met_per_s(r["v"], dur) for r in recs]
            xs.append(j); vals.append(gs[-1][0])
            lo.append(min(g[0] for g in gs)); hi.append(max(g[0] for g in gs))
            cols.append(col)
            for r, g in zip(recs, gs):
                rows.append(dict(model=PAIRS[0][0], arm=label, run=r["run"],
                                 cut_minute=dur, req_goodput_s=g[0],
                                 met_requests=g[1], arrivals=g[2],
                                 drawn=(r["run"] == d["run"]),
                                 repeats=len(recs)))
            print(f"    {label:22s} {len(recs)} run(s)  "
                  f"{gs[-1][0]:6.2f} req/s met  ({gs[-1][1]:,} of {gs[-1][2]:,} "
                  f"arrivals in {dur:.2f} min)")
        bar_ax.bar(xs, vals, width=0.72, color=cols, edgecolor="#000000",
                   linewidth=0.5)
        bar_ax.vlines(xs, lo, hi, color="#333333", lw=0.7)
        bar_ax.set_xticks([])
        bar_ax.set_xlim(-0.7, len(ARMS) - 0.3)
        bar_top = max(hi) * 1.15
        # The empty pair mirrors the drawn one, so the reserved layout is the
        # one the finished figure will have.
        for i, (name, drawn) in enumerate(PAIRS):
            g, b = ax[2 * i], ax[2 * i + 1]
            g.set_xlim(0, HOUR_MIN)
            g.set_xticks(list(range(0, int(HOUR_MIN) + 1, 15)))
            g.set_ylim(0, top * 1.08)
            g.yaxis.set_major_formatter(kfmt())
            g.set_ylabel("Token Goodput (t/s)")
            g.set_xlabel("Time (minutes)", labelpad=1.5)
            b.set_ylim(0, BAR_MAX[i] or bar_top)
            if BAR_MAX[i]:
                b.set_yticks(list(np.arange(0, BAR_MAX[i] + 1e-9, BAR_TICK)))
            b.set_ylabel("Request Goodput (r/s)")
            b.set_xticks([])
            b.set_xlim(-0.7, len(ARMS) - 0.3)
            for a in (g, b):
                a.grid(axis="both" if a is g else "y", **GRID)
                a.set_axisbelow(True)
                for side in ("top", "right"):
                    a.spines[side].set_visible(False)
            if not drawn:
                for a in (g, b):
                    a.text(0.5, 0.5, "pending", transform=a.transAxes,
                           ha="center", va="center", fontsize=6,
                           color="#999999")
        # ⚠ NO `fontsize` HERE ON PURPOSE. The figure this one sits under
        # (`two_models_hour_reqgoodput.pdf`) leaves the key at the project's
        # 8 pt, and a key one size smaller directly below it reads as a
        # different KIND of key rather than the same one with fewer entries.
        # The panels' type is shrunk by `SHRINK` and the key is not: the key is
        # read once for the whole figure, the tick numbers beside their own ink.
        # ⚠ NO KEY ON THIS FIGURE (2026-09-11, at the author's request). It is
        # meant to sit under `two_models_hour_reqgoodput.pdf`, whose key names
        # the same five arms in the same colours and the same order, so one key
        # serves both. Shown on its own, this figure has nothing that names an
        # arm and the caption must do it.
        if False:
            fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                       bbox_to_anchor=(0.5, 1.0 - TOP_BAND / FIG_H), frameon=False,
                       columnspacing=0.8, handlelength=1.6,
                       handletextpad=0.3, borderaxespad=0.0)
        fig.tight_layout(rect=(0, 0.140 / FIG_H, 0.988, 1.0 - TOP_BAND / FIG_H),
                         w_pad=1.2, pad=0.3)
        for i, (name, _d) in enumerate(PAIRS):
            b0 = ax[2 * i].get_position(); b1 = ax[2 * i + 1].get_position()
            # (c) and (d), not (a) and (b): in the paper this pair follows two
            # panels that come from another figure, so the letters continue that
            # sequence rather than restarting. ⚠ If this figure is ever shown on
            # its own, the letters have to go back to (a) and (b).
            fig.text(0.5 * (b0.x0 + b1.x1), 0.012,
                     f"({'cd'[i]}) Token and Request Goodput - {name}",
                     ha="center", va="bottom", fontsize=8)
        save(fig, out)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-hour", action="store_true",
                    help="draw and score the whole hour instead of cutting at "
                         "the last window every arm finished")
    a = ap.parse_args()
    per = collect()
    if not per:
        sys.exit("no scored EXP-126 runs; score them into "
                 f"{VERDICTS} first")
    # ⚠ THE CUT IS NOT A PROPERTY OF THE TRACE, IT IS THE SLOWEST ARM'S DRAIN.
    # The default stops at the last window in which every arm with admission
    # control still finished at least 80% of what arrived in it, and on this
    # experiment one arm crosses that at 58 min while the other three are at
    # 0.0-0.6% unfinished right to the end, so the whole figure lost its last
    # 3.25 minutes to that one arm. `--full-hour` keeps all sixty minutes; the
    # price is that the slow arm's last minutes are scored with a fifth to a
    # third of their arrivals having no outcome at all, and the share is
    # printed per arm so the caption can carry it.
    dur = HOUR_MIN if a.full_hour else cut_minute(per)
    print(f"span drawn: {dur:.2f} min"
          + ("  (whole hour, not cut)" if a.full_hour else "  (cut)"))
    for label, recs in per.items():
        v = recs[-1]["v"]
        w = v[v["rel"] <= dur * 60.0]
        tail = w[w["rel"] >= (dur - 3.0) * 60.0]
        print(f"    {label:22s} unfinished at the end of the span: "
              f"{100.0 * w['cutoff'].mean():5.2f}% of the span, "
              f"{100.0 * tail['cutoff'].mean():5.2f}% of its last three minutes")
    pdf = os.path.join(HERE, "exp126_two_models_hour_reqgoodput.pdf")
    df = build(per, dur, pdf)
    df["budgets"] = "chat 2.5s/25ms, deepresearch 5s/50ms, swe 3.5s/38ms"
    df["rule"] = "ladder95, halved budgets"
    out = pdf[:-4] + ".csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"wrote {out}  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
