#!/usr/bin/env python3
"""Paper figure (evaluation): both tails, and what the tail actually costs.

  eval_tails.pdf   7.0 x 2.10 in, FULL TEXT WIDTH, three panels

**The objection this answers.** "You buy your capacity by sacrificing tail
latency, against llm-d in particular." The word "tail" covers two different
quantities here and the answer differs between them, so both are drawn, plus the
panel that prices the one we lose.

  (a) ACROSS-REQUEST p90: the p90 over requests of each request's own mean
      inter-token latency -- the quantity the literature reports as P90 TPOT.
      FluidServe is the only arm inside chat's 50 ms budget at all eight rates
      (45-49 ms at 25-70 req/s). llm-d is at 51-111. This tail is not
      sacrificed; it is where we are strongest.

  (b) WITHIN-REQUEST p90: the p90 of one request's own token gaps, median over
      requests -- the Mooncake-style quantity no paper in our comparison set
      uses as an attainment criterion. llm-d leads at every rate. FluidServe is
      second from 25 req/s and is the only arm that stays flat with load
      (62-72 ms from 20 to 70 req/s) while the other three run to 196-523 ms.
      So "sacrificed" is wrong against three of the four baselines and correct
      only against llm-d.

  (c) The frontier: total goodput against within-request p90, one point per
      (arm, rate), lines join increasing load. This is the panel that prices
      the llm-d gap. llm-d holds 31-44 ms but its goodput peaks at 9.4k tok/s
      at 20 req/s and FALLS to 7.6k at 70 -- above 25 req/s it rejects 55-93%
      of chat -- while FluidServe rises monotonically to 13.9k at 62-63 ms. On
      this plot llm-d's curve folds back to the left as load rises; ours extends
      right. The choice is between a smoother stream and 1.5-1.8x the delivered
      work, not between good and bad.

**At matched work, stated honestly.** At 10-20 req/s nobody rejects and all five
arms serve the same requests. There the within-request gap to llm-d is real:
43-45 vs 29-31 ms at 15 req/s, 63-69 vs 34-37 at 20 -- and 63-69 exceeds chat's
budget on this metric. The measured cause is class concentration: the arms that
do not concentrate classes (vLLM router, Llumnix SLO) sit at 28-41 ms there too,
and concentration is the same mechanism that buys the capacity advantage at high
load. EXP-85 measured that flattening the engine's prefill chunks (QoServe
unit 4) removes the spikes but raises the floor above the budget, so a
within-request p90 of 50 ms is not reachable at these loads by any placement of
this work.

**Data.** 31_tails_all_arms_clean.csv -- the pinned clean set (EXP-82 + EXP-86),
five arms, eight rates, two repeats, GOMAXPROCS=16 everywhere. Chat class for
panels (a)/(b); ALL-class goodput for (c).

**CAVEATS for the caption.**

1. Panels (a)/(b) are conditioned on SERVED requests, and the arms serve very
   different fractions above 25 req/s (llm-d rejects 55-93% of chat). Panel (c)
   exists precisely because of that: it puts the served volume back on the axis.
2. Within-request p99 cannot be computed -- the client records p50-p95 and max.
3. Two repeats; bands are min..max.
4. The scoring rules of the paper (per-request mean, cumulative deadline) use
   neither of these percentiles; this figure exists to answer the tail question,
   not to re-score attainment.

Source: analysis_scripts/request_level/tails_all_arms.py wrote the CSV.
"""
import collections
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_style as ps  # noqa: E402

CSV = os.path.abspath(os.path.join(
    HERE, "..", "results", "aggregate_analysis", "tail_2026-08-16",
    "31_tails_all_arms_clean.csv"))

ARMS = [("FluidServe", ps.ARM_COLOR["fluidserve"], "s", 5),
        ("llm-d", ps.ARM_COLOR["llmd"], "D", 4),
        ("vLLM router", ps.ARM_COLOR["vllmrouter"], "h", 2),
        ("Llumnix SLO", ps.ARM_COLOR["slo"], "^", 2),
        ("PolyServe", ps.ARM_COLOR["polyserve"], "o", 2)]
RATES = [10, 15, 20, 25, 35, 45, 55, 70]


def main():
    rows = list(csv.DictReader(open(CSV)))
    chat = collections.defaultdict(lambda: collections.defaultdict(list))
    allc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["cls"] not in ("chat", "ALL"):
            continue
        dest = chat if r["cls"] == "chat" else allc
        d = {k: float(r[k]) for k in ("rej", "goodput", "w_p90", "a_p90")}
        dest[r["arm"]][float(r["rate"])].append(d)

    plt.rcParams.update(ps.STYLE)
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(ps.TEXT_W, 2.10))
    x = np.arange(len(RATES))

    def band(axis, vals, colour, marker, label, z):
        m = [np.mean(v) for v in vals]
        lo = [min(v) for v in vals]
        hi = [max(v) for v in vals]
        axis.plot(x, m, color=colour, marker=marker, ms=3.0, lw=1.3, label=label,
                  zorder=z)
        axis.fill_between(x, lo, hi, color=colour, alpha=0.16, lw=0, zorder=z - 1)

    for axis, key, title, ylim in (
            (ax, "a_p90", "(a) across-request p90, chat", 130),
            (bx, "w_p90", "(b) within-request p90, chat", 260)):
        for arm, colour, marker, z in ARMS:
            band(axis, [[q[key] for q in chat[arm][r]] for r in RATES],
                 colour, marker, arm, z)
        axis.axhline(50, color="black", lw=0.7, ls="--", zorder=1)
        axis.text(0.02, 50, " budget 50 ms", fontsize=6, va="bottom",
                  transform=axis.get_yaxis_transform())
        axis.set_xticks(x)
        axis.set_xticklabels(["%g" % r for r in RATES], fontsize=7)
        axis.set_xlabel("offered load (req/s)")
        axis.set_ylim(0, ylim)
        axis.grid(axis="y", **ps.GRID)
        axis.set_title(title, pad=3)
    ax.set_ylabel("ms")
    ax.legend(loc="upper left", handlelength=1.2, borderpad=0.2,
              labelspacing=0.2, handletextpad=0.4, fontsize=6)

    # (c) the frontier: total goodput vs within-request p90 (chat)
    for arm, colour, marker, z in ARMS:
        gx = [np.mean([q["goodput"] for q in allc[arm][r]]) / ps.KTOK for r in RATES]
        gy = [np.mean([q["w_p90"] for q in chat[arm][r]]) for r in RATES]
        cx.plot(gx, gy, color=colour, marker=marker, ms=3.0, lw=1.1, zorder=z)
        cx.annotate("70", (gx[-1], gy[-1]), fontsize=5.5, color=colour,
                    textcoords="offset points", xytext=(3, 2))
    cx.axhline(50, color="black", lw=0.7, ls="--", zorder=1)
    cx.set_xlabel("total goodput (k tok/s)")
    cx.set_ylabel("within-request p90 (ms)")
    cx.set_ylim(0, 260)
    cx.grid(axis="y", **ps.GRID)
    cx.set_title("(c) what the tail buys: goodput", pad=3)

    fig.tight_layout(rect=(0, 0, 1, 1), w_pad=1.3)
    ps.save(fig, os.path.join(HERE, "eval_tails.pdf"))

    print("\nchat, min..max over 2 repeats")
    for key, name in (("a_p90", "across-request p90"), ("w_p90", "within-request p90")):
        print("\n%s (ms)" % name)
        print("req/s".rjust(6) + "".join(a[:12].rjust(15) for a, _, _, _ in ARMS))
        for r in RATES:
            line = ("%g" % r).rjust(6)
            for arm, _, _, _ in ARMS:
                v = [q[key] for q in chat[arm][r]]
                line += ("%.0f-%.0f" % (min(v), max(v))).rjust(15)
            print(line)
    print("\ntotal goodput at 70 req/s (k tok/s): " + "  ".join(
        "%s %.1f" % (a, np.mean([q["goodput"] for q in allc[a][70]]) / 1000)
        for a, _, _, _ in ARMS))


if __name__ == "__main__":
    main()
