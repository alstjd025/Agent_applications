#!/usr/bin/env python3
"""Motivation figure 1: the fleet produces the same tokens and almost none count.

Three existing answers, on the same four engines, at the same offered rates:

  Llumnix (load balance)  no latency awareness at all -- spread the load
  Llumnix SLO             latency aware, no class structure
  PolyServe               class aware, by assigning engines to classes

FluidServe is deliberately absent. A motivation figure that needs the paper's own
system to make its point is not a motivation figure.

Four panels.

  A  what the engines produced, in output tokens per second
  B  how much of that belonged to a request that finished inside its latency
     rule. A and B share a y axis, because the claim is that the first is nearly
     the same across policies while the second is not, and rescaling either
     would hide exactly that
  C  the same question counted in REQUESTS rather than tokens
  D  A and B again at one rate, where the split is legible

C is not a restatement of B. B weights every request by how many tokens it
produced, so a policy that keeps the long requests and refuses the short ones
scores well on it; C counts each request once. The two disagree by more than
their shapes suggest -- at 70 req/s the spread across the three policies is 15.2x
on B and 5.5x on C -- and the reason is Llumnix SLO, which rejects 69.2% of
arrivals there and serves the survivors well. It is the best of the three on
goodput and returns 21.4% of what was sent to it.

C therefore carries BOTH denominators, and the distance between them is the
point rather than a caveat. Solid counts every request that arrived, so a
rejection is a violation. Dotted counts only the requests the policy accepted.
Llumnix SLO's dotted line RISES from 56.8% to 71.3% between 45 and 70 req/s
while its solid line falls from 52.4% to 21.4%: read on the accepted set alone
it looks like a policy improving under load, and what it is doing is refusing
more. The other two never reject, so their lines coincide.

Below about 35 req/s the three policies are indistinguishable on every panel:
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
from exp22_fluidserve import PAPER_STYLE, load_run, attain  # noqa: E402

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
                         good=met["output_tokens"].sum() / dur,
                         att_off=attain(r, "violate_offered"),
                         att_adm=attain(served, "violate_served"),
                         rej=100.0 * rej.mean()))
    return pd.DataFrame(rows)


def main(out):
    df = collect()
    if df.empty:
        sys.exit("no EXP-53 runs matched")
    g = df.groupby(["arm", "rate"])
    agg = g.agg(thru=("thru", "mean"), good=("good", "mean"),
                thru_lo=("thru", "min"), thru_hi=("thru", "max"),
                good_lo=("good", "min"), good_hi=("good", "max"),
                att_off=("att_off", "mean"), att_adm=("att_adm", "mean"),
                att_off_lo=("att_off", "min"), att_off_hi=("att_off", "max"),
                rej=("rej", "mean"),
                n=("thru", "size")).reset_index()

    for q, lab in (("thru", "total produced (output tokens/s)"),
                   ("good", "met its rule -- goodput (output tokens/s)"),
                   ("att_off", "SLO attainment (%), offered denominator"),
                   ("att_adm", "SLO attainment (%), admitted denominator"),
                   ("rej", "rejected (%)")):
        p = agg.pivot(index="rate", columns="arm", values=q)
        cols = [k for k, _, _ in ARMS if k in p.columns]
        print(f"\n  {lab}")
        print(p[cols].to_string(float_format=lambda v: f"{v:,.1f}"))
        if q != "rej":
            print("  max/min across the three: " +
                  " ".join(f"{r:.0f}:{v:.1f}x" for r, v in
                           (p[cols].max(axis=1)
                            / p[cols].min(axis=1).replace(0, np.nan)).items()))
    print("\nrepeats per cell: " + str(sorted(agg["n"].unique())))

    os.makedirs(out, exist_ok=True)
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(1, 4, figsize=(12.8, 2.95),
                               gridspec_kw={"width_ratios": [1, 1, 1, 0.78]})
        # Only the first two share an axis. They are the pair that carries the
        # claim -- same units, one nearly flat across policies and one not --
        # and rescaling either would hide exactly that.
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
            # Panel C counts requests rather than tokens, and carries BOTH
            # denominators. Solid counts every request that arrived, so a
            # rejection is a violation; dotted counts only what the policy
            # accepted. The distance between them is what that policy bought by
            # refusing work, and reporting either one alone is what makes a
            # policy that rejects most of its load look either broken or
            # excellent depending on which was picked.
            ax[2].plot(s.rate, s.att_off, color=col, marker="o", ms=3.5,
                       mec="white", mew=0.5, ls="-")
            ax[2].plot(s.rate, s.att_adm, color=col, ls=":", lw=1.1)
            ax[2].fill_between(s.rate, s.att_off, s.att_adm, color=col,
                               alpha=0.10, lw=0)

        # Where the policies start to differ. Read off the goodput panel, where
        # the three are within 1% up to 25 req/s and 2.1x apart by 35.
        for a_ in ax[:3]:
            a_.axvspan(15, 35, color="#888888", alpha=0.06, lw=0)
        ax[0].annotate("all three the same\n(the decision does not\nmatter yet)",
                       (25, 3200), fontsize=6.5, color="#666666", ha="center")
        ax[1].annotate("all three the same", (25, 3200), fontsize=6.5,
                       color="#666666", ha="center")

        ax[1].annotate("20,057 produced\n→ 710 useful",
                       (70, 710), xytext=(47, 5600), fontsize=7, color="#9467bd",
                       arrowprops=dict(arrowstyle="->", color="#9467bd", lw=0.8))

        # The gap on the arm that creates it. Llumnix SLO rejects 69.2% at 70
        # req/s, which is the whole distance between its two lines.
        s_ = agg[(agg.arm == "slo") & (agg.rate == 70.0)]
        if not s_.empty:
            off = float(s_["att_off"].iloc[0])
            adm = float(s_["att_adm"].iloc[0])
            rej = float(s_["rej"].iloc[0])
            ax[2].annotate("", xy=(70, adm), xytext=(70, off),
                           arrowprops=dict(arrowstyle="<->", color="#2ca02c",
                                           lw=0.9))
            # Under the flat stretch, which is the only empty region: every
            # line sits at 100 up to 35 req/s, so nothing is covered there.
            ax[2].annotate(f"Llumnix SLO rejects {rej:.0f}% at this rate,\n"
                           f"and that is the whole distance\nbetween its two lines",
                           (70, (off + adm) / 2), xytext=(11, 14), fontsize=6.5,
                           color="#2ca02c", va="center",
                           arrowprops=dict(arrowstyle="->", color="#2ca02c",
                                           lw=0.7,
                                           connectionstyle="arc3,rad=0.20"))

        # Panel D says the same thing at one rate, where the split is legible.
        # The bar height is what the engines produced; the solid part is what a
        # client could use and the hatched part is what was thrown away.
        HL = 70.0
        xs = np.arange(len(ARMS))
        usable = []
        for i, (key, lab, col) in enumerate(ARMS):
            s_ = agg[(agg.arm == key) & (agg.rate == HL)]
            if s_.empty:
                continue
            good, thru = float(s_["good"].iloc[0]), float(s_["thru"].iloc[0])
            ax[3].bar(i, good, color=col, width=0.62)
            ax[3].bar(i, thru - good, bottom=good, color=col, width=0.62,
                      alpha=0.22, hatch="////", edgecolor=col, lw=0)
            ax[3].annotate(f"{100 * (thru - good) / thru:.0f}%\nthrown\naway",
                           (i, good + (thru - good) / 2), ha="center", va="center",
                           fontsize=6.5, color="#444444")
            usable.append(f"{good:,.0f}")
        ax[3].set_xlim(-0.62, len(ARMS) - 0.38)
        ax[3].set_xticks(xs)
        names = ["Llumnix\n(load bal.)", "Llumnix\nSLO", "Poly\nServe"]
        ax[3].set_xticklabels([f"{n}\n{u}" for n, u in zip(names, usable)],
                              fontsize=6.3)
        ax[3].set_ylim(0, 22000)
        ax[3].set_title(f"D. the same at {HL:.0f} req/s: bar height is\n"
                        f"production, solid part is usable", fontsize=8)
        ax[3].grid(axis="y", ls=":", lw=0.7, alpha=0.6)

        ax[0].set_title("A. what the engines produced", fontsize=8)
        ax[1].set_title("B. how much of it met its latency rule", fontsize=8)
        ax[2].set_title("C. and how many REQUESTS were met", fontsize=8)
        ax[0].set_ylabel("output tokens per second")
        ax[2].set_ylabel("SLO attainment (%), per request")
        for a_ in ax[:3]:
            a_.set_xlabel("offered request rate (req/s)")
            a_.set_xlim(10, 75)
            a_.grid(axis="y", ls=":", lw=0.7, alpha=0.6)
        for a_ in ax[:2]:
            a_.set_ylim(0, 22000)
        ax[2].set_ylim(0, 105)

        # Two legends. Colour is the policy and line style is the denominator,
        # and a single combined legend reads as though the styles were more
        # policies. Built from what was drawn rather than from a fixed list.
        h, l = ax[0].get_legend_handles_labels()
        ax[0].legend(h, l, loc="upper center", bbox_to_anchor=(1.1, -0.22),
                     ncol=3, fontsize=7, columnspacing=1.4)
        den = [plt.Line2D([], [], color="#444444", ls="-", lw=1.2),
               plt.Line2D([], [], color="#444444", ls=":", lw=1.2)]
        ax[2].legend(den,
                     ["every request that arrived\n(a rejection is a violation)",
                      "only the requests it accepted"],
                     loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
                     fontsize=6.3, columnspacing=1.0, handlelength=2.0)

        fig.suptitle(
            "Three existing policies, the same four engines, the same load. They "
            "produce within 1.5x of each other, differ by 15x in how much of that\n"
            "production a client could use, and the best of them returns 21% of "
            "the requests that were sent to it inside the latency it promised.\n"
            "Shaded band in A and B is the spread over repeats; it is invisible "
            "because the repeats agree.",
            fontsize=8, y=1.10)
        p = os.path.join(out, "motivation_throughput_vs_goodput.png")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "results/aggregate_analysis/motivation")
