#!/usr/bin/env python3
"""Motivation figure 1: the fleet produces the same tokens and almost none count.

Three existing answers, on the same four engines, at the same offered rates:

  Llumnix (load balance)  no latency awareness at all -- spread the load
  Llumnix SLO             latency aware, no class structure
  PolyServe               class aware, by assigning engines to classes

FluidServe is deliberately absent. A motivation figure that needs the paper's own
system to make its point is not a motivation figure.

Left panel is what the engines produced. Right panel is how much of it belonged
to a request that finished inside its latency rule. The two panels share a y
axis, because the claim is that the first is nearly the same across policies
while the second is not, and a rescaled right panel would hide exactly that.

Below about 35 req/s the three policies are indistinguishable on both panels:
the routing decision does not matter until the fleet is loaded enough for it to.

Data: EXP-53, two repeats per cell, stock vLLM FIFO engine, migration off for
PolyServe and on for the two Llumnix arms. The Llumnix SLO arm uses the `m1f`
workload configuration, which splits swe's 30-second end-to-end budget into the
(first-token, per-token) pair that policy requires; the alternative left it
rejecting 98% of that class and answering a different question.

WARNING -- the PolyServe line is not yet publishable. Its
`--polyserve-tier-decode-tokens` is stale in exactly the way FluidServe's own
length profile was before EXP-48 fixed it: deep research is configured at 275
expected output tokens against a measured 985, a factor of 3.58. So the figure
currently compares a system whose length profile was corrected against a
baseline whose was not, which is an asymmetry in our favour. See
ms_dev/notes/polyserve-fidelity.md section 2 and paper-outline.md section 0. The
fix is one string and the re-run is about three hours.

  python3 motivation_fig1.py <out-dir>
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp22_fluidserve import PAPER_STYLE, load_run  # noqa: E402

ARMS = [("loadbalance", "Llumnix (load balance)", "#9467bd"),
        ("slo", "Llumnix SLO", "#2ca02c"),
        ("polyserve", "PolyServe", "#d62728")]


def collect(pattern="results/*exp53*_rpm_*"):
    rows = []
    for d in sorted(glob.glob(pattern)):
        m = re.search(r"_(polyserve|slo|loadbalance)_m1f?_rpm_(\d+)$",
                      os.path.basename(d))
        if not m:
            continue
        r = load_run(d)
        if r is None or r.empty:
            continue
        rej = r["rejected"] if "rejected" in r else pd.Series(False, index=r.index)
        served = r[~rej]
        dur = r["rel"].max()
        met = served[~served["violate_served"]]
        rows.append(dict(arm=m.group(1), rate=int(m.group(2)) / 60.0,
                         thru=served["output_tokens"].sum() / dur,
                         good=met["output_tokens"].sum() / dur))
    return pd.DataFrame(rows)


def main(out):
    df = collect()
    if df.empty:
        sys.exit("no EXP-53 runs matched")
    g = df.groupby(["arm", "rate"])
    agg = g.agg(thru=("thru", "mean"), good=("good", "mean"),
                thru_lo=("thru", "min"), thru_hi=("thru", "max"),
                good_lo=("good", "min"), good_hi=("good", "max"),
                n=("thru", "size")).reset_index()

    print("mean over repeats, output tokens per second")
    for q in ("thru", "good"):
        p = agg.pivot(index="rate", columns="arm", values=q)
        cols = [k for k, _, _ in ARMS if k in p.columns]
        print(f"\n  {'total produced' if q == 'thru' else 'met its rule (goodput)'}")
        print(p[cols].to_string(float_format=lambda v: f"{v:,.0f}"))
        print("  max/min across the three: " +
              " ".join(f"{r:.0f}:{v:.1f}x"
                       for r, v in (p[cols].max(axis=1) / p[cols].min(axis=1)).items()))
    print("\nrepeats per cell: " + str(sorted(agg["n"].unique())))

    os.makedirs(out, exist_ok=True)
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9),
                               gridspec_kw={"width_ratios": [1, 1, 0.62]})
        ax[1].sharey(ax[0])
        for key, lab, col in ARMS:
            s = agg[agg.arm == key].sort_values("rate")
            if s.empty:
                continue
            for i, (m_, lo, hi) in enumerate((("thru", "thru_lo", "thru_hi"),
                                              ("good", "good_lo", "good_hi"))):
                ax[i].plot(s.rate, s[m_], color=col, marker="o", ms=3.5,
                           mec="white", mew=0.5, label=lab)
                ax[i].fill_between(s.rate, s[lo], s[hi], color=col, alpha=0.18,
                                   lw=0)

        # Where the policies start to differ. Read off the goodput panel, where
        # the three are within 1% up to 25 req/s and 2.1x apart by 35.
        for a_ in ax[:2]:
            a_.axvspan(15, 35, color="#888888", alpha=0.06, lw=0)
        ax[0].annotate("all three the same\n(the decision does not\nmatter yet)",
                       (25, 3200), fontsize=6.5, color="#666666", ha="center")
        ax[1].annotate("all three the same", (25, 3200), fontsize=6.5,
                       color="#666666", ha="center")

        ax[1].annotate("20,057 produced\n→ 710 useful",
                       (70, 710), xytext=(47, 5600), fontsize=7, color="#9467bd",
                       arrowprops=dict(arrowstyle="->", color="#9467bd", lw=0.8))

        # Panel C says the same thing at one rate, where the split is legible.
        # The bar height is what the engines produced; the solid part is what a
        # client could use and the hatched part is what was thrown away.
        HL = 70.0
        xs = np.arange(len(ARMS))
        for i, (key, lab, col) in enumerate(ARMS):
            s_ = agg[(agg.arm == key) & (agg.rate == HL)]
            if s_.empty:
                continue
            good, thru = float(s_["good"].iloc[0]), float(s_["thru"].iloc[0])
            ax[2].bar(i, good, color=col, width=0.62)
            ax[2].bar(i, thru - good, bottom=good, color=col, width=0.62,
                      alpha=0.22, hatch="////", edgecolor=col, lw=0)
            ax[2].annotate(f"{good:,.0f}", (i, good / 2), ha="center", va="center",
                           fontsize=7, color="white", weight="bold")
            ax[2].annotate(f"{100 * (thru - good) / thru:.0f}%\nthrown\naway",
                           (i, good + (thru - good) / 2), ha="center", va="center",
                           fontsize=6.5, color="#444444")
        ax[2].set_xlim(-0.62, len(ARMS) - 0.38)
        ax[2].set_xticks(xs)
        ax[2].set_xticklabels(["Llumnix\n(load bal.)", "Llumnix\nSLO", "Poly\nServe"],
                              fontsize=6.5)
        ax[2].set_ylim(0, 22000)
        ax[2].set_title(f"the same at {HL:.0f} req/s:\nbar height is production", fontsize=8)
        ax[2].grid(axis="y", ls=":", lw=0.7, alpha=0.6)

        ax[0].set_title("what the engines produced", fontsize=8)
        ax[1].set_title("how much of it met its latency rule", fontsize=8)
        ax[0].set_ylabel("output tokens per second")
        for a_ in ax[:2]:
            a_.set_xlabel("offered request rate (req/s)")
            a_.set_xlim(10, 75)
            a_.set_ylim(0, 22000)
            a_.grid(axis="y", ls=":", lw=0.7, alpha=0.6)
        ax[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.22), ncol=3,
                     fontsize=7, columnspacing=1.4)
        fig.suptitle(
            "Three existing policies, the same four engines, the same load. "
            "They produce within 1.5x of each other\nand differ by 15x in how "
            "much of that production a client could use.\nShaded band is the spread over repeats; it is invisible because the repeats agree.",
            fontsize=8, y=1.06)
        p = os.path.join(out, "motivation_throughput_vs_goodput.png")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "results/aggregate_analysis/motivation")
