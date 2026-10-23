#!/usr/bin/env python3
"""Figure 1 companion panel: the two ways a scheduler loses capacity, and the
one arm that avoids both.

  goodput_vs_attainment.pdf   3.335 x 1.92 in, one column, no scaling
  goodput_vs_attainment.csv   the plotted values, one row per arm and repeat

One point per scheduler.
  x = token goodput: output tokens per second that arrived on time
      (ladder95 rule), over the drawn span of the hour.
  y = SLO attainment of the requests the scheduler ADMITTED:
      met / (arrivals - rejected). A request still running when the run ends
      counts as admitted and not met (only the arm without admission control
      has any: its queue grows all hour and those are the requests stuck in
      it). The alternative that drops them from the denominator is printed.

Reading. A scheduler that cannot see capacity fails in one of two ways:
  (i)  it refuses work the cluster could have served: attainment stays high
       but goodput is low                                   -> upper left
  (ii) it admits work the instance cannot serve within SLO: attainment drops,
       and the late tokens are not goodput either           -> lower part
Only an arm that judges feasibility correctly sits at the upper right.

DATA. EXP-109 (Llama-3.1-70B, 4 instances, one-hour shifting trace, 99,242
arrivals), the per-request ladder95 verdict tables of
../two_models_hour_reqgoodput_a/, five arms x two repeats, the same span
(0-59.25 min) as the hour figure so the numbers agree with it. The marker is
repeat 2; repeat 1 is not drawn (it is in the CSV; only llm-d moves visibly,
8,491 -> 7,837 tok/s).

    python3.12 fig_goodput_vs_attainment.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
import glob  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paper_style import COL_W, STYLE, GRID, kfmt, save  # noqa: E402

DATA = os.path.join(os.path.dirname(HERE), "data", "verdicts")
CUT_MIN = 59.25                      # the hour figure's span; see its README
# (label, colour, marker, repeat 2, repeat 1). Colours: the paper's five-step
# ramp, FluidServe at the dark end (hour, motivation and capacity-loss figures).
# Markers: the motivation figure's, one per arm across the paper
# (vLLM p, Llumnix ^, PolyServe o, llm-d D, FluidServe s).
# llm-d: Figure 10 of the paper draws it in a muted lavender, not the #2c7fb8
# of the hour script in this repo. The hex below is READ OFF THE RENDERED
# FIGURE, not taken from its script; replace it with that script's value.
C_LLMD = "#b3a2c9"
ARMS = [
    ("vLLM",       "#ffffcc", "p", "260901_2010_exp109r2_vllmcachet75_shift",
                                    "260901_2137_exp109r1_vllmcachet75_shift"),
    ("Llumnix",    "#a1dab4", "^", "260901_1856_exp109r2_slot75_shift",
                                    "260831_2346_exp109r1_slot75_shift"),
    ("PolyServe",  "#41b6c4", "o", "260901_1742_exp109r2_polyservept75_shift",
                                    "260831_2232_exp109r1_polyservept75_shift"),
    ("llm-d",      C_LLMD,    "D", "260901_1637_exp109r2_llmdslot75_shift",
                                    "260831_2128_exp109r1_llmdslot75_shift"),
    ("FluidServe", "#253494", "s", "260901_1514_exp109r2_fsv3capgnofrct75_shift",
                                    "260831_2015_exp109r1_fsv3capgnofrct75_shift"),
]
# One full column, drawn at final size: include with width=\\columnwidth and
# nothing is rescaled (paper_style's rule). Kept low so it reads as a small panel.
# SCALE enlarges everything drawn (type, markers, line widths, axes box) by one
# factor; the canvas width cannot grow past the column, so only the height and
# the axes box do. At 1.2 the axis type is 7.8 pt, i.e. the paper's full-size
# 8 pt rather than the compact 6.5 pt of the other one-column figures.
SCALE = 1.2
FIG_W, FIG_H = COL_W, 1.6 * SCALE
FS = (STYLE["font.size"] - 1.5) * SCALE
ANNOT = (STYLE["font.size"] - 3.0) * SCALE
# Second line under each bold name. {rej} is filled with the measured share of
# arrivals the arm rejected in the drawn repeat; replace a string with the
# arm's design in a few words if the paper prefers that (as conceptual
# scatter plots do: "Sarathi-Serve / stall-free batching").
DESCR = {
    "vLLM":       "(RR: {rej:.0f}%)",     # no admission control, so 0%
    "Llumnix":    "(RR: {rej:.0f}%)",
    "PolyServe":  "(RR: {rej:.0f}%)",
    "llm-d":      "(RR: {rej:.0f}%)",
    "FluidServe": "(RR: {rej:.0f}%)",
}
RR_NOTE = "*RR: rejection ratio"          # defined once, as a footnote inside the frame
# --chat-rr: second line "(RR: 21% · Chat 17%)", i.e. also the rejection ratio of
# the tightest class, written to goodput_vs_attainment_chatrr.*
CHAT_RR = "--chat-rr" in sys.argv
if CHAT_RR:
    DESCR = {k: "(RR: {rej:.0f}% \u00b7 Chat {chat:.0f}%)" for k in DESCR}
    RR_NOTE = "*RR: rejection ratio, all / Chat"
OUT = "goodput_vs_attainment" + ("_chatrr" if CHAT_RR else "")

# Type: Nimbus Roman No9 L (URW's Times), the paper's body face, from the
# user's font folder; matplotlib's cache does not list it, so register it.
# Its heavier cut is "Medium", so that is what "bold" maps to below.
NIMBUS = sorted(glob.glob(os.path.join(os.path.dirname(HERE), "fonts", "NimbusRomNo9L-*.otf"))
                or glob.glob(os.path.expanduser("~/Library/Fonts/NimbusRomNo9L-*.otf")))
for _f in NIMBUS:
    fm.fontManager.addfont(_f)
FONT = fm.FontProperties(fname=NIMBUS[0]).get_name() if NIMBUS else "Times New Roman"
BOLD = "medium" if NIMBUS else "bold"


def point(run):
    v = pd.read_csv(os.path.join(DATA, run + ".csv"))
    w = v[v["rel"] <= CUT_MIN * 60.0]
    c = w[w["class"] == "chat"]
    chat_rej_pct = 100.0 * c["rejected"].sum() / len(c)
    n = len(w)
    rej = int(w["rejected"].sum())
    cut = int(w["cutoff"].sum())
    met = int((w["ladder_ok"] & ~w["rejected"] & ~w["errored"] & ~w["cutoff"]).sum())
    ok = ~w["rejected"] & ~w["errored"]
    good_tok = float((w.loc[ok, "n_tokens"] - w.loc[ok, "n_late"]).sum())
    return dict(token_goodput=good_tok / (CUT_MIN * 60.0),
                attain_admitted=met / (n - rej),
                attain_admitted_finished=met / max(1, n - rej - cut),
                met_pct=100.0 * met / n, rejected_pct=100.0 * rej / n,
                chat_rejected_pct=chat_rej_pct,
                unfinished_pct=100.0 * cut / n, arrivals=n)


def main():
    rows = []
    for lab, col, mk, r2, r1 in ARMS:
        for rep, run in (("r2", r2), ("r1", r1)):
            rows.append(dict(arm=lab, repeat=rep, **point(run)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, OUT + ".csv"), index=False, float_format="%.4f")
    print(df.round(3).to_string(index=False))

    st = dict(STYLE)
    for k in ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize", "ytick.labelsize"):
        st[k] = STYLE[k] - 1.5
    st["font.family"] = "serif"
    st["font.serif"] = [FONT] + list(STYLE["font.serif"])
    with plt.rc_context(st):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        # The canvas stays one column wide (scale 1.0 at \\columnwidth); the axes
        # are narrower than the canvas and centred, so the points sit close
        # together instead of being spread over an empty 3.3 in.
        AX_W = 2.05 * SCALE                           # axes width, inches
        x0 = (FIG_W - AX_W) / 2 + 0.12                # nudged right: the y label is on the left
        fig.subplots_adjust(left=x0 / FIG_W, right=(x0 + AX_W) / FIG_W, top=0.96, bottom=0.215)
        for lab, col, mk, _r2, _r1 in ARMS:
            a = df[(df.arm == lab) & (df.repeat == "r2")].iloc[0]
            b = df[(df.arm == lab) & (df.repeat == "r1")].iloc[0]
            # No repeat-to-repeat mark: four arms repeat to within their marker,
            # so the only visible line was llm-d's (8,491 vs 7,837 tok/s) and it
            # read as a stray stroke. Both repeats are in the CSV and the README.
            # Shapes by the author's choice: a star for FluidServe, circles for the
            # baselines (this panel does NOT use the per-arm markers of ARMS; the
            # colour ramp and the labels identify the arms).
            star = lab == "FluidServe"
            ax.scatter([a.token_goodput], [100 * a.attain_admitted],
                       marker="*" if star else "o",
                       s=(95 if star else 34) * SCALE ** 2,
                       facecolor=col, edgecolor="black", linewidths=0.6 * SCALE, zorder=3)
            # bold name, plain second line, both to the right of the marker
            x0, y0 = a.token_goodput, 100 * a.attain_admitted
            # llm-d: FluidServe's star is on its right; Llumnix: PolyServe's label is
            left = lab in ("llm-d", "Llumnix")
            dx = (-6 if left else (8 if star else 6)) * SCALE
            ha = "right" if left else "left"
            ax.annotate(lab, (x0, y0), xytext=(dx, 1.5 * SCALE), textcoords="offset points",
                        ha=ha, va="bottom", fontsize=FS,
                        fontweight=BOLD if star else "normal", zorder=4)
            ax.annotate(DESCR[lab].format(rej=a.rejected_pct, chat=a.chat_rejected_pct), (x0, y0),
                        xytext=(dx, -1.5 * SCALE), textcoords="offset points",
                        ha=ha, va="top", fontsize=ANNOT + 0.5, zorder=4)
        # No failure-mode notes on the figure: the positions and the "rejects"
        # lines carry them, and Llumnix belongs to both, so the caption says it.
        ax.set_xlim(0, 17500)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        # axis type at the other figures' 6.5 pt, so the panels match side by side
        ax.set_xlabel("Token goodput (tokens/s)", labelpad=1.5, fontsize=FS)
        ax.set_ylabel("SLO attainment of\nadmitted requests (%)", labelpad=2, linespacing=1.15,
                      fontsize=FS)
        ax.tick_params(labelsize=FS)
        ax.xaxis.set_major_formatter(kfmt())
        ax.set_xticks([0, 4000, 8000, 12000, 16000])
        ax.text(0.975, 0.05, RR_NOTE, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=ANNOT + 0.5, color="#444444", zorder=4)
        # closed frame and dotted grid, as the paper's Figure 10 panels
        for side in ("left", "bottom", "top", "right"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(STYLE["axes.linewidth"] * SCALE)
        ax.tick_params(length=STYLE["xtick.major.size"] * SCALE,
                       width=STYLE["xtick.major.width"] * SCALE)
        ax.grid(**GRID)
        ax.set_axisbelow(True)
        fig.savefig(os.path.join(HERE, OUT + ".png"), dpi=200)
        save(fig, os.path.join(HERE, OUT + ".pdf"))


if __name__ == "__main__":
    main()
