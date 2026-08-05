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
  D  what became of every request that arrived, at one rate

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

Rejecting cannot raise the offered number, because a rejection is a violation
there. Llumnix SLO's offered attainment falls with load like everyone else's, 52.4
to 21.4 between 45 and 70 req/s. What rises is the admitted number and the
distance to the policies that never reject, 1.6x at 45 req/s and 5.5x at 70. Panel
D says why that distance opens, by splitting every arrival three ways at 70 req/s:

                        met its rule   served, too late   refused
  Llumnix (load bal.)            3.9               96.1       0.0
  PolyServe                     14.4               85.6       0.0
  Llumnix SLO                   21.4                8.6      70.0

A refusal and a late answer both score zero on the offered denominator, so
Llumnix SLO converted 70 of the 96 requests load balancing served late into
refusals and got 17.5 more requests met in exchange. Those 17.5 are work that
succeeded on capacity the refusals freed, not a smaller denominator.

So what separates the three is not whether they reject. It is how much of the
fleet's work lands in neither column: 96.1%, 85.6%, 8.6%. FluidServe rejects too
-- 45.1% at this rate, with 51.9% met and 3.0% late -- so the claim to make is
that it rejects less than Llumnix SLO and meets more, never that rejecting is
itself the fault.

The split drops requests still running when the run ended, because their outcome
is unknown rather than bad, which is the same denominator attain() uses. Without
that exclusion load balance reads 30.9% met against the 3.9% its attainment
reports, and that mismatch is how the omission was caught.

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
        # Panel D splits arrivals three ways, on the SAME denominator attain()
        # uses: requests still running when the run ended are dropped from both,
        # because their outcome is unknown rather than bad. Computing the split
        # without that exclusion put load balance at 30.9% met against the 3.9%
        # its attainment reads, which is how the mismatch was caught.
        k = r[~r["cutoff"]]
        nk = max(len(k), 1)
        k_rej = k["rejected"]
        k_err = k["errored"] & ~k_rej
        k_srv = k[~k_rej & ~k["errored"]]
        k_met = k_srv[~k_srv["violate_served"]]
        rows.append(dict(arm=m.group(1), rate=int(m.group(2)) / 60.0,
                         thru=served["output_tokens"].sum() / dur,
                         good=met["output_tokens"].sum() / dur,
                         att_off=attain(r, "violate_offered"),
                         att_adm=attain(served, "violate_served"),
                         rej=100.0 * rej.mean(),
                         o_met=100.0 * len(k_met) / nk,
                         o_late=100.0 * (len(k_srv) - len(k_met)) / nk,
                         o_rej=100.0 * k_rej.sum() / nk,
                         o_err=100.0 * k_err.sum() / nk))
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
                rej=("rej", "mean"), o_met=("o_met", "mean"),
                o_late=("o_late", "mean"), o_rej=("o_rej", "mean"),
                o_err=("o_err", "mean"),
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
            rej = float(s_["o_rej"].iloc[0])
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

        # Panel D: what became of every request that arrived, at one rate.
        # Tokens are already covered across all rates by A and B, so restating
        # them here would be redundant; what is NOT anywhere else is how much of
        # each policy's failure is work done and wasted versus work refused.
        # Those two score identically -- both are violations on the offered
        # denominator -- and separating them is what makes the rejecting policy
        # readable instead of suspicious.
        HL = 70.0
        xs = np.arange(len(ARMS))
        met_lab = []
        for i, (key, lab, col) in enumerate(ARMS):
            s_ = agg[(agg.arm == key) & (agg.rate == HL)]
            if s_.empty:
                continue
            met = float(s_["o_met"].iloc[0])
            late = float(s_["o_late"].iloc[0])
            rj = float(s_["o_rej"].iloc[0])
            ax[3].bar(i, met, color=col, width=0.62)
            ax[3].bar(i, late, bottom=met, color=col, width=0.62, alpha=0.22,
                      hatch="////", edgecolor=col, lw=0)
            ax[3].bar(i, rj, bottom=met + late, color="#bbbbbb", width=0.62,
                      alpha=0.55)
            if met > 6:
                ax[3].annotate(f"{met:.0f}", (i, met / 2), ha="center",
                               va="center", fontsize=7, color="white",
                               weight="bold")
            if late > 9:
                ax[3].annotate(f"{late:.0f}", (i, met + late / 2), ha="center",
                               va="center", fontsize=7, color="#444444")
            if rj > 9:
                ax[3].annotate(f"{rj:.0f}", (i, met + late + rj / 2),
                               ha="center", va="center", fontsize=7,
                               color="#333333")
            met_lab.append(f"{met:.0f} met")
        ax[3].set_xlim(-0.62, len(ARMS) - 0.38)
        ax[3].set_xticks(xs)
        names = ["Llumnix\n(load bal.)", "Llumnix\nSLO", "Poly\nServe"]
        ax[3].set_xticklabels([f"{n}\n{m}" for n, m in zip(names, met_lab)],
                              fontsize=6.3)
        ax[3].set_ylim(0, 100)
        ax[3].set_ylabel("% of arrivals")
        ax[3].set_title(f"D. what became of every request\nat {HL:.0f} req/s",
                        fontsize=8)
        ax[3].grid(axis="y", ls=":", lw=0.7, alpha=0.6)
        d_leg = [plt.Rectangle((0, 0), 1, 1, fc="#666666"),
                 plt.Rectangle((0, 0), 1, 1, fc="#666666", alpha=0.22,
                               hatch="////", ec="#666666"),
                 plt.Rectangle((0, 0), 1, 1, fc="#bbbbbb", alpha=0.55)]
        ax[3].legend(d_leg, ["met its rule", "served, too late", "refused"],
                     loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1,
                     fontsize=6.3, handlelength=1.4, labelspacing=0.25)

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
            "produce within 1.5x of each other and differ by 15x in how much of "
            "that production a client could use.\n"
            "The best of them returns 21% of the requests sent to it inside the "
            "latency it promised, and gets there by refusing 70% of them: on "
            "panel D a refusal and a late answer\nboth score zero, so what "
            "separates the policies is how much of the fleet's work ends up in "
            "neither column. "
            "Shaded band in A and B is the spread over repeats.",
            fontsize=8, y=1.13)
        p = os.path.join(out, "motivation_throughput_vs_goodput.png")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "results/aggregate_analysis/motivation")
