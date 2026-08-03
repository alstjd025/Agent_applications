#!/usr/bin/env python3
"""Paper figure: length distributions of the three request classes.

  workload_lengths.pdf     3.335 x 1.95 in, `figure`, width=\\columnwidth

Two panels — (left) input tokens, (right) output tokens — each carrying one
probability density per class. This is the distribution behind the median / p90 /
std table in README.md: the table gives three numbers per class and the density
shows the shape they summarise, which for chat is a long right tail and for the
other two is a single narrow mode.

POPULATION. The four arms of EXP-53 at 15 req/s, pooled, completed requests only
(not rejected, errored, or cut off at run end). 15 req/s is the lowest rate in
that sweep and every arm holds 100% attainment and rejects nothing there, so the
lengths are set by the workload rather than truncated by saturation. The four
arms agree to within 2% on every class median, which is what makes pooling them
legitimate: this is a workload property, not a policy one. 27,560 requests.

DENSITY IN LOG SPACE, AND WHAT THAT MEANS FOR THE Y AXIS. Input spans 2 to
15,334 tokens and output 2 to 7,855, so the kernel density is estimated over
log10(tokens) and drawn against a log x axis. The curve is therefore a density
per decade, not per token: the AREA under a curve between two x values is the
share of that class's requests in that range, but the HEIGHT cannot be compared
against a linear-axis density. The y axis carries no numbers for that reason —
heights are comparable between the curves on this figure and meaningless off it.

EACH CLASS IS NORMALISED TO ITS OWN AREA, not to its share of the traffic. The
three curves therefore answer "given a request of this class, how long is it",
which is the question the table answers too. They do NOT show that chat is 77%
of the requests; the class shares are in the legend.

    python3 paper_figures/fig_workload_lengths.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis_scripts", "request_level"))

from paper_style import COL_W, STYLE, GRID, save  # noqa: E402
from exp22_fluidserve import load_run, CLASSES, CLASS_COLORS  # noqa: E402

RUNS = "results/*exp53r1_*_m1*_rpm_900"
LABEL = {"chat": "chat", "deepresearch": "deep research", "swe": "agent"}
FIG_H = 1.95
# One range for both panels, so "this class outputs about a tenth of what
# it takes in" is readable by moving the eye across rather than by
# reading two differently-scaled axes. Covers both maxima (input 15,334,
# output 7,855).
XLIM = (1.5, 2.0e4)


def population():
    frames = []
    for d in sorted(glob.glob(os.path.join(ROOT, RUNS))):
        r = load_run(d)
        if r is None or r.empty:
            continue
        frames.append(r[(~r["rejected"]) & (~r["errored"]) & (~r["cutoff"])])
    if not frames:
        return None
    a = pd.concat(frames)
    a["inp"] = pd.to_numeric(a["input_tokens"], errors="coerce")
    a["out"] = pd.to_numeric(a["output_tokens"], errors="coerce")
    a = a.dropna(subset=["inp", "out"])
    return a[(a["inp"] > 0) & (a["out"] > 0)]


def density(v, lo, hi):
    """Kernel density over log10(v), evaluated on a grid in the same space.

    Estimated in log space rather than by transforming a linear-space estimate:
    a Gaussian kernel wide enough to smooth the 15,000-token end would erase
    the 2-token end entirely, and one narrow enough for the low end leaves the
    high end as a comb of spikes.
    """
    x = np.log10(np.asarray(v, dtype=float))
    grid = np.linspace(np.log10(lo), np.log10(hi), 512)
    return 10.0 ** grid, gaussian_kde(x)(grid)


def main():
    a = population()
    if a is None:
        print(f"no runs matched {RUNS}", file=sys.stderr)
        return 1
    share = a["class"].value_counts(normalize=True) * 100

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(COL_W, FIG_H))
        handles, labels = [], []

        for ax, col in zip(axes, ("inp", "out")):
            lo, hi = XLIM
            for c in CLASSES:
                v = a.loc[a["class"] == c, col]
                if len(v) < 10:
                    continue
                x, y = density(v, lo, hi)
                ax.plot(x, y, color=CLASS_COLORS[c], lw=1.1)
                # A light fill separates the three modes where two curves cross;
                # at 0.13 it does not hide the curve underneath it.
                ax.fill_between(x, y, color=CLASS_COLORS[c], alpha=0.13, lw=0)
                if ax is axes[0]:
                    handles.append(plt.Line2D([], [], color=CLASS_COLORS[c],
                                              lw=1.4))
                    labels.append(f"{LABEL[c]} ({share[c]:.0f}%)")
            ax.set_xscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylim(0, None)
            # The height is a density per decade of tokens, which is not a
            # quantity a reader should read off an axis. The shape and the area
            # carry the information, so the tick labels go.
            ax.set_yticks([])
            ax.grid(axis="x", **GRID)
            ax.set_axisbelow(True)

        axes[0].set_xlabel("Input tokens")
        axes[1].set_xlabel("Output tokens")
        # Each curve integrates to 1 over its own class, so heights are
        # comparable within a panel and not across panels.
        axes[0].set_ylabel("Density")

        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 0.845), frameon=False,
                   columnspacing=0.8, handlelength=1.2, handletextpad=0.4,
                   borderaxespad=0.0)
        fig.tight_layout(rect=(0, 0, 1, 0.835), w_pad=1.0, pad=0.35)
        save(fig, os.path.join(HERE, "workload_lengths.pdf"))

    print(f"{len(a)} requests: " + ", ".join(
        f"{LABEL[c]} {len(a[a['class'] == c])}" for c in CLASSES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
