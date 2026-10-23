#!/usr/bin/env python3
"""goodput_vs_attainment_byclass.pdf: the Figure 1 scatter twice, (a) over all
requests and (b) over the tightest class alone (Chat, 5 s / 50 ms), so that a
reader can see the rejection ratio is not bought by refusing the hard class.

Same data, axes and reading as fig_goodput_vs_attainment.py; (b) restricts
every quantity (goodput, attainment, RR) to Chat arrivals.

    python3.12 fig_goodput_vs_attainment_byclass.py
"""
import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paper_style import COL_W, STYLE, GRID, kfmt, save  # noqa: E402
import fig_goodput_vs_attainment as M  # noqa: E402  (ARMS, DATA, CUT_MIN, fonts, DESCR)

PANELS = [("(a) All requests", None), ("(b) Chat only (5 s / 50 ms)", "chat")]
FS = STYLE["font.size"] - 1.5          # 6.5 pt, the one-column size
ANNOT = FS - 1.0                       # 5.5 pt second lines


def point(run, cls):
    v = pd.read_csv(os.path.join(M.DATA, run + ".csv"))
    w = v[v["rel"] <= M.CUT_MIN * 60.0]
    if cls is not None:
        w = w[w["class"] == cls]
    n = len(w)
    rej = int(w["rejected"].sum())
    met = int((w["ladder_ok"] & ~w["rejected"] & ~w["errored"] & ~w["cutoff"]).sum())
    ok = ~w["rejected"] & ~w["errored"]
    good_tok = float((w.loc[ok, "n_tokens"] - w.loc[ok, "n_late"]).sum())
    return dict(token_goodput=good_tok / (M.CUT_MIN * 60.0), attain=met / (n - rej),
                rejected_pct=100.0 * rej / n)


def main():
    st = dict(STYLE)
    for k in ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize", "ytick.labelsize"):
        st[k] = STYLE[k] - 1.5
    st["font.family"] = "serif"
    st["font.serif"] = [M.FONT] + list(STYLE["font.serif"])
    rows = []
    with plt.rc_context(st):
        fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.75), sharey=True,
                                 gridspec_kw=dict(wspace=0.12))
        fig.subplots_adjust(left=0.135, right=0.985, top=0.88, bottom=0.22)
        for ax, (title, cls) in zip(axes, PANELS):
            pts = {lab: point(r2, cls) for lab, _c, _m, r2, _r1 in M.ARMS}
            xmax = max(p["token_goodput"] for p in pts.values())
            for lab, col, _mk, _r2, _r1 in M.ARMS:
                p = pts[lab]
                rows.append(dict(panel=title, arm=lab, **p))
                star = lab == "FluidServe"
                ax.scatter([p["token_goodput"]], [100 * p["attain"]],
                           marker="*" if star else "o", s=95 if star else 34,
                           facecolor=col, edgecolor="black", linewidths=0.6, zorder=3)
                xy = (p["token_goodput"], 100 * p["attain"])
                if lab == "Llumnix" and cls is not None:
                    # Chat panel: PolyServe sits just to its right and the frame
                    # to its left, so name above and RR below the marker
                    up, dn = (0, 6), (0, -6)
                    ha = "center"
                elif lab == "vLLM" and cls is not None:
                    # Chat panel: both lines below-right, under Llumnix's RR line
                    up, dn, ha = (5, -2.5), (5, -10), "left"
                else:
                    left = lab in ("llm-d", "Llumnix")
                    dx = -5 if left else (7 if star else 5)
                    up, dn, ha = (dx, 1.5), (dx, -1.5), ("right" if left else "left")
                va_up = "top" if (lab == "vLLM" and cls is not None) else "bottom"
                ax.annotate(lab, xy, xytext=up, textcoords="offset points", ha=ha,
                            va=va_up, fontsize=FS, fontweight=M.BOLD if star else "normal",
                            zorder=4)
                ax.annotate(f"(RR: {p['rejected_pct']:.0f}%)", xy, xytext=dn,
                            textcoords="offset points", ha=ha, va="top", fontsize=ANNOT,
                            zorder=4)
            ax.set_title(title, pad=3)
            ax.set_xlim(0, xmax * (1.55 if cls is None else 1.8))
            ax.set_ylim(0, 112)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.xaxis.set_major_formatter(kfmt())
            ax.set_xticks([0, 4000, 8000, 12000] if xmax > 9000 else [0, 3000, 6000, 9000])
            ax.grid(**GRID)
            ax.set_axisbelow(True)
            if cls is None:                    # the footnote once, in the first panel
                ax.text(0.975, 0.05, M.RR_NOTE, transform=ax.transAxes, ha="right",
                        va="bottom", fontsize=ANNOT - 0.5, color="#444444", zorder=4)
            ax.set_xlabel("Token goodput (tokens/s)", labelpad=1.5)
        axes[0].set_ylabel("SLO attainment of\nadmitted requests (%)", labelpad=3, linespacing=1.2)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(HERE, "goodput_vs_attainment_byclass.csv"), index=False,
                  float_format="%.4f")
        print(df.round(3).to_string(index=False))
        fig.savefig(os.path.join(HERE, "goodput_vs_attainment_byclass.png"), dpi=200)
        save(fig, os.path.join(HERE, "goodput_vs_attainment_byclass.pdf"))


if __name__ == "__main__":
    main()
