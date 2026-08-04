#!/usr/bin/env python3
"""Paper figure: TTFT and inter-token latency distributions at 50 req/s.

  exp53_latency_cdf.pdf     7.0 x 2.90 in, `figure*`, width=\\textwidth

Six panels: two rows (time to first token, inter-token latency) by three columns
(the request classes), four control planes per panel, from EXP-53 at 3000 rpm =
50 req/s. Vertical rules mark the budget the class is scored against on that
axis. This is the latency view behind the attainment number: attainment says what
fraction cleared the rule, these say by how much and where the distribution sits.

THE Y AXIS IS THE FRACTION OF ARRIVALS, NOT OF COMPLETIONS, AND THAT IS THE
WHOLE POINT OF DRAWING IT THIS WAY. A latency CDF can only contain requests that
produced tokens, and at this rate the four arms lose very different shares of
their arrivals before that, for very different reasons:

                rejected   cut off at run end   in the CDF
  FluidServe      16.1%           1.4%             82.5%
  Llumnix SLO     20.7-22.8%      1.9-2.2%         75-77%
  PolyServe        0.0%          24.5%             75.5%
  Llumnix          0.0%          13.9%             86.1%

Normalised to completions, every curve would end at 1.0 and each arm would be
showing the latency of a different three quarters of the load — and the quarter
each one drops is the slow quarter. PolyServe rejects nothing but never finishes
24.5% of what it accepts; those are its most backlogged requests, exactly the
ones that belong in the tail. Normalising them away would hand the arm with the
worst backlog the best-looking tail. Against arrivals, each curve instead tops
out at the share of arrivals it actually served, so the missing mass is on the
figure rather than divided out of it. **The height each curve reaches is as much
of the result as its shape.**

This is the same defect as the end-of-run spike in the EXP-54 timeline, in
distribution form: dropping requests whose outcome is unknown removes the slow
ones, and only the backlogged arm has many.

THE AGENT CLASS HAS NO RULE ON EITHER OF THESE AXES. It is scored end to end at
30 s, so its two panels carry no budget rule and are descriptive only. chat and
deep research are scored on TTFT *and* inter-token latency, so both of their
panels carry one.

Both repeats are pooled (repeat 1 `exp53r1*`, repeat 2 `exp53p2*`), about 46,000
arrivals per arm.

    python3 paper_figures/fig_exp53_latency_cdf.py
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis_scripts", "request_level"))

from paper_style import TEXT_W, STYLE, GRID, ARM_COLOR  # noqa: E402
from paper_style import save  # noqa: E402
from exp22_fluidserve import load_run, CLASSES, SLO_RULES  # noqa: E402

RUNS = ["results/*exp53r1*_rpm_3000", "results/*exp53p2*_rpm_3000"]
ARM_RE = re.compile(r"_(fluidserve|polyserve|slo|loadbalance)_m1f?_rpm_")
ORDER = ["fluidserve", "polyserve", "slo", "loadbalance"]
LABEL = {"fluidserve": "FluidServe", "polyserve": "PolyServe",
         "slo": "Llumnix SLO", "loadbalance": "Llumnix"}
STYLE_LS = {"fluidserve": "-", "polyserve": "-.", "slo": "--",
            "loadbalance": ":"}
CLASS_TITLE = {"chat": "chat", "deepresearch": "deep research", "swe": "agent"}

FIG_H = 2.90
TTFT_XLIM = (0.02, 200.0)     # seconds, log
ITL_XLIM = (0.0, 200.0)       # ms, linear


def population():
    """Arrivals per arm, pooled over both repeats, with the served subset."""
    out = {}
    for pat in RUNS:
        for d in sorted(glob.glob(os.path.join(ROOT, pat))):
            m = ARM_RE.search(os.path.basename(d))
            if not m:
                continue
            r = load_run(d)
            if r is None or r.empty:
                continue
            out.setdefault(m.group(1), []).append(r)
    return {a: pd.concat(v) for a, v in out.items()}


def cdf_of_arrivals(values, n_arrivals):
    """x, y for a CDF whose y is the share of ARRIVALS below x.

    The curve therefore ends at len(values)/n_arrivals rather than at 1.0, and
    that endpoint is the share of arrivals the arm actually served.
    """
    v = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(v) + 1) / float(n_arrivals)
    return v, y


def main():
    data = population()
    if not data:
        print("no EXP-53 3000 rpm conditions matched", file=sys.stderr)
        return 1

    arms = [a for a in ORDER if a in data]
    print(f"{'arm':12s} {'arrivals':>9s} {'rej%':>6s} {'cut%':>6s} {'in CDF':>7s}")
    for a in arms:
        r = data[a]
        ok = r[(~r["rejected"]) & (~r["errored"]) & (~r["cutoff"])]
        print(f"{a:12s} {len(r):9d} {100*r['rejected'].mean():6.1f} "
              f"{100*r['cutoff'].mean():6.1f} {100*len(ok)/len(r):6.1f}%")

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, len(CLASSES), figsize=(TEXT_W, FIG_H),
                                 sharey=True)
        handles, labels = [], []

        for col, cls in enumerate(CLASSES):
            rule = SLO_RULES[cls]
            for row, (field, budget, xlim) in enumerate([
                    ("first_token_latency", rule.get("ttft"), TTFT_XLIM),
                    ("itl_ms", rule.get("tbt"), ITL_XLIM)]):
                ax = axes[row][col]
                for a in arms:
                    r = data[a]
                    sub = r[r["class"] == cls]
                    n_arr = len(sub)
                    ok = sub[(~sub["rejected"]) & (~sub["errored"])
                             & (~sub["cutoff"])]
                    v = pd.to_numeric(ok[field], errors="coerce").dropna()
                    if not len(v) or not n_arr:
                        continue
                    x, y = cdf_of_arrivals(v, n_arr)
                    h, = ax.plot(x, y, color=ARM_COLOR[a], ls=STYLE_LS[a],
                                 lw=1.1)
                    if row == 0 and col == 0:
                        handles.append(h)
                        labels.append(LABEL[a])
                if budget is not None:
                    ax.axvline(budget, color="#555555", lw=0.7, ls="--",
                               zorder=0)
                if row == 0:
                    ax.set_xscale("log")
                ax.set_xlim(*xlim)
                ax.set_ylim(0, 1.0)
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.grid(axis="both", **GRID)
                ax.set_axisbelow(True)
            axes[0][col].set_title(CLASS_TITLE[cls], fontsize=8, pad=3)
            axes[0][col].set_xlabel("Time to first token (s)")
            axes[1][col].set_xlabel("Inter-token latency (ms)")

        axes[0][0].set_ylabel("Fraction of arrivals")
        axes[1][0].set_ylabel("Fraction of arrivals")

        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 0.935), frameon=False,
                   columnspacing=1.4, handlelength=2.0, handletextpad=0.5,
                   borderaxespad=0.0)
        fig.tight_layout(rect=(0, 0, 1, 0.925), w_pad=1.0, h_pad=1.0, pad=0.35)
        save(fig, os.path.join(HERE, "exp53_latency_cdf.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
