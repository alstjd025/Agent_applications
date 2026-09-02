#!/usr/bin/env python3
"""Paper figure: four control planes on the static rate sweep, scored under the
token-level cumulative deadline.

  motivation_throughput_vs_goodput_4panel_t75.pdf         baselines only
  motivation_throughput_vs_goodput_4panel_t75_withfs.pdf  the same plus FluidServe
  7.0 x 1.75 in, `figure*`, width=\\textwidth

  Each PDF is written with a CSV of the same basename holding EXACTLY the values
  drawn in it -- one row per arm and arrival rate, the mean that is the line and
  the min and max that are the band, the number of repeats behind each cell, and
  the scoring rule the numbers were produced under. A figure whose numbers live
  only in a prose table cannot be checked against its own source, and the prose
  table goes stale the first time the figure is redrawn.

⚠ THE `t75` IN THE FILE NAME IS NOT DECORATION. `fig_motivation_throughput_goodput.py`
already writes a four-panel figure called
`motivation_throughput_vs_goodput_4panel.pdf`, drawn from the pinned copy of the
EARLIER static sweep and scored with the agent class on its end-to-end 30 s
budget. Nothing in that figure may sit beside anything in this one, so the two
must not share a file name; `t75` records the form of the agent class's promise
and therefore which of the two a number came from.

  (a) how much of the output belonged to a request that finished inside its rule
  (b) output tokens per second the engines produced
  (c) the same question counted in REQUESTS, over the requests the policy ACCEPTED
  (d) what share of arrivals the policy rejected

WHY THIS PANEL ORDER. (a) is the quantity the figure claims and (b) sits beside
it on the same y axis, so that "the engines produce comparable numbers of tokens
and the share of them worth anything is not comparable" is visible without
rescaling either. (c) and (d) ask the same question counted in requests rather
than tokens, and they are a PAIR: (c)'s denominator is the requests the policy
accepted, which read alone hands the best score to a policy that accepts almost
nothing, and (d) is what makes it readable. The offered denominator is not drawn
because (c) and (d) together carry it -- a curve for it would be the third
presentation of one number.

X AXIS IS LINEAR AND ITS LABELS ARE EVENLY SPACED. The arrival rate is a
quantity, so the distance between 10 and 15 req/s is drawn at half the distance
between 55 and 70; an equal-width categorical axis would assert that those two
steps are the same size. The LABELS are placed at a uniform 10 req/s interval
(10, 20, ..., 70) rather than on the measured rates, because the measured set
(10, 15, 20, 25, 35, 45, 55, 70) has three different step sizes and labelling it
puts four labels in the first third of the axis.
⚠ FOUR OF THOSE SEVEN LABELS ARE AT RATES THAT WERE NEVER MEASURED (30, 40, 50,
60). That is a real cost and it is paid deliberately for an evenly spaced axis:
the measured rates keep an unlabelled MINOR tick, and every curve carries a
marker at each measured rate, so where the data actually is stays visible.

THE SCORING RULE IS NEW AND IT IS THE POINT OF THIS FIGURE. Until now the
per-token half of a class rule was scored as `mean inter-token time over the
request <= budget`, which is Scorpio's rule and, among the nine systems whose
definitions were read for ms_dev/notes/slo-definitions.md, the rule only Scorpio
uses. Here it is a TOKEN-LEVEL CUMULATIVE DEADLINE WITH A TOLERANCE: writing
a_i for the arrival of the i-th output token measured from submission, T for the
class time-to-first-token budget and P for its per-token budget,

    a_1 <= T                                        AND
    |{i : a_i <= T + (i-1) * P}| / N >= 0.90

The first conjunct is kept separate on purpose: with the deadline schedule
anchored at submission, a request that beat T banks the unused time, and a
request whose engine decodes faster than P can absorb a first-token violation of
about N * (P - actual per-token time) before missing any deadline. Keeping
`a_1 <= T` as its own condition removes that.

The cumulative deadline itself is PolyServe's, QoServe's and JITServe's rule.
The 90% tolerance is not in any of those nine papers and is this project's
addition; `--rule` draws the figure under any of the five columns the scorer
produces (mean, cumt, q90, q90end, ft90) and the README records all of them, so
the relaxation is a column difference and not an assertion.

⚠ swe IS SCORED ON A PER-TOKEN RULE HERE FOR THE FIRST TIME. Its promise was an
end-to-end 30 s budget with no per-token term; FluidServe v0.4 restated it as
TTFT 7 s + 75 ms per token, and EXP-108 is the sweep in which every arm was TOLD
that form -- FluidServe through `--fluidserve-class-budgets 25:decode:75`, the
other three through `mix_short_m1_t75fair.json`, whose `slo.swe.tbt_ms` is 75.
NOTHING IN THIS FIGURE MAY SIT BESIDE A NUMBER SCORED AGAINST THE 30 s
END-TO-END RULE, and that includes every earlier version of the motivation
figure.

DATA. EXP-108, the static rate sweep of 2026-08-31: four arms, eight arrival
rates (10, 15, 20, 25, 35, 45, 55, 70 req/s), 8 minutes per condition, two
repeats, four engines, the engine cold-restarted between conditions. 64 result
directories, all of which carry per-token arrival events. The band is min..max
over the two repeats; a cell with no visible band has two repeats that agree,
not one repeat.

⚠ THE vLLM ROUTER COMES FROM A DIFFERENT SESSION, AND THAT IS A REAL JOIN.
EXP-108 has four arms and this is not one of them, so its curve is EXP-77
(2026-08-10): eight arrival rates, two repeats at all but 70 req/s, where the
second repeat's per-token events are gone. It is the one arm that can be
re-scored under a promise it did not run with, because it reads no latency
promise at all -- its routing is prefix-cache affinity and nothing else, so the
requests it saw are identical whichever agent-class budget the workload file
states. What is NOT identical is everything else about the session: three weeks
earlier, a different scheduler binary for the arms it is drawn beside, and the
machine was wiped and restored on 2026-08-28 between the two. Session-to-session
movement on this workload has been measured at up to 4.6 points and that
measurement does not span the restore. THE CAPTION HAS TO SAY THIS.

    python3 paper_figures/fig_motivation_tg_4panel.py [--rule q90]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from paper_style import TEXT_W, STYLE, GRID, ARM_COLOR, kfmt, save  # noqa: E402

# The runs the paper draws, which are not exactly the runs EXP-108 produced.
# EXP-108's llm-d cell at 10 req/s holds two repeats that disagree by 42 points
# of rejection rate (0.0% and 42.0%) with no cause found; EXP-110 re-measured
# that one arrival rate three more times and got 0.0% every time, so the 42.0%
# repeat is an outlier that occurred once in five runs. The cell is drawn from
# EXP-110 repeats 3 and 5 instead, which keeps n=2 as in every other cell here.
# The full account, including what the outlier was and why it is not explained,
# is in experiments/EXP-110_llmd-lowrate-split.md. exp108_per_run.csv still holds
# the unmodified scoring of every EXP-108 run and is what --csv should point at
# to see it.
CSV_DEFAULT = os.path.join(ROOT, "results", "aggregate_analysis",
                           "token_deadline_2026-08-31",
                           "exp108_paper_per_run.csv")
CSV = CSV_DEFAULT

# EXP-108's arm directory names, in the order they are drawn and listed. The
# name in the second field is the one the paper uses; the `t75` suffix records
# the form of the agent class's promise and belongs in the README, not on the
# figure, because every arm here carries it.
ARMS = [
    ("vllmcache",        "vLLM-router", ARM_COLOR["vllmrouter"]),
    ("polyservept75",    "PolyServe",   ARM_COLOR["polyserve"]),
    ("slot75",           "Llumnix SLO", ARM_COLOR["slo"]),
    ("llmdslot75",       "llm-d",       ARM_COLOR["llmd"]),
    ("fsv3capgnofrct75", "FluidServe",  ARM_COLOR["fluidserve"]),
]
# The vLLM router is not an EXP-108 arm; its rows come from the EXP-77 scoring
# of the same rule. Two files rather than one because they are two sessions, and
# a single file would let that be forgotten.
CSV_VLLM = os.path.join(ROOT, "results", "aggregate_analysis",
                        "token_deadline_2026-08-31", "vllm77_per_run.csv")
OURS = "FluidServe"
FIG_H = 1.75
RATES = [10, 15, 20, 25, 35, 45, 55, 70]        # what was measured
LABEL_TICKS = [10, 20, 30, 40, 50, 60, 70]      # what the axis is labelled with


def collect(rule, prefix="raw"):
    """label -> rate -> dict of the five plotted quantities, mean over repeats,
    with the min and max of each kept so the band is the measured spread."""
    frames = [pd.read_csv(CSV)]
    if os.path.exists(CSV_VLLM):
        frames.append(pd.read_csv(CSV_VLLM))
    else:
        print(f"!! {CSV_VLLM} missing; the vLLM router arm will be absent",
              file=sys.stderr)
    d = pd.concat(frames, ignore_index=True)
    d = d[d["prefix"] == prefix]
    if d.empty:
        sys.exit(f"{CSV} has no rows with prefix={prefix}")
    cols = {"thru": "throughput_tok_s", "good": f"{rule}_goodput",
            "off": f"{rule}_offered", "adm": f"{rule}_admitted",
            "rej": "rejected_pct"}
    missing = [c for c in cols.values() if c not in d.columns]
    if missing:
        sys.exit(f"{CSV} is missing {missing}; re-run rescore_token_deadline.py")
    out, counts = {}, {}
    for arm, label, _ in ARMS:
        sub = d[d["arm"] == arm]
        if sub.empty:
            print(f"!! no runs for arm {arm}", file=sys.stderr)
            continue
        per = {}
        for rate, g in sub.groupby("rate"):
            per[float(rate)] = {k: (g[c].mean(), g[c].min(), g[c].max())
                                for k, c in cols.items()}
            counts[(label, float(rate))] = len(g)
        out[label] = per
    return out, counts


def report(data, counts, rule):
    print(f"rule = {rule}\n")
    print(f"{'arm':13s} {'req/s':>5s} {'n':>2s} {'thru':>7s} {'goodput':>8s} "
          f"{'offered':>8s} {'admitted':>9s} {'rejected':>9s}")
    for _, label, _ in ARMS:
        if label not in data:
            continue
        for rate in sorted(data[label]):
            v = data[label][rate]
            print(f"{label:13s} {rate:5.0f} {counts[(label, rate)]:2d} "
                  f"{v['thru'][0]:7.0f} {v['good'][0]:8.0f} {v['off'][0]:8.1f} "
                  f"{v['adm'][0]:9.1f} {v['rej'][0]:9.1f}")


# The quantity each panel draws, in panel order, with the column name it takes
# in the CSV written beside the figure. The units are in the names because a
# column called `goodput` alone does not say whether it is tokens or requests.
PANEL_COLS = [("good", "goodput_tok_s"), ("thru", "throughput_tok_s"),
              ("adm", "admitted_pct"), ("rej", "rejected_pct")]


def write_csv(data, labels, counts, rule, out_path):
    """The values the figure draws, beside the figure, under its own basename.

    Only what is drawn: the line (mean over repeats), the band (min and max),
    the number of repeats behind the cell, and the identity of the scoring rule.
    Everything else about these runs is in the per-run scoring table this reads
    from, and duplicating it here would create a second place for it to go
    stale."""
    rows = []
    for _, label, _ in ARMS:
        if label not in labels or label not in data:
            continue
        for rate in sorted(data[label]):
            v = data[label][rate]
            row = {"arm": label, "req_per_s": rate,
                   "n_repeats": counts[(label, rate)], "rule": rule}
            for key, name in PANEL_COLS:
                row[name], row[f"{name}_min"], row[f"{name}_max"] = v[key]
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"wrote {out_path}  ({len(df)} rows = "
          f"{df['arm'].nunique()} arms x {df['req_per_s'].nunique()} rates)")


def build(data, labels, out_path, legend=True):
    """Four panels at equal x spacing. `labels` selects which arms are drawn;
    the legend row is reserved either way so the with- and without-FluidServe
    versions have axes of exactly the same size and can be read against each
    other."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 4, figsize=(TEXT_W, FIG_H))
        handles, drawn = [], []
        for _, label, col in ARMS:
            if label not in labels or label not in data:
                continue
            rates = sorted(data[label])
            x = rates
            mk = dict(marker="o", ms=2.6, mec="white", mew=0.35)

            def series(key):
                return ([data[label][r][key][0] for r in rates],
                        [data[label][r][key][1] for r in rates],
                        [data[label][r][key][2] for r in rates])

            for i, key in enumerate(("good", "thru", "adm", "rej")):
                m, lo, hi = series(key)
                ax[i].fill_between(x, lo, hi, color=col, alpha=0.16, lw=0)
                h, = ax[i].plot(x, m, color=col, lw=1.2, **mk)
                if i == 0:
                    handles.append(h)
                    drawn.append(label)

        # (a) is read against (b): the claim is that the first spans much more
        # than the second, and rescaling either would hide it.
        ax[1].sharey(ax[0])
        titles = ["(a) Goodput tokens", "(b) Throughput",
                  "(c) Request SLO (admitted)", "(d) Rejection rate"]
        for i in range(4):
            ax[i].set_xlabel(f"Offered rate (req/s)\n{titles[i]}",
                             labelpad=1.5, linespacing=1.6)
            ax[i].set_xlim(7, 73)
            # Labels every 10 req/s; the rates that were actually measured keep
            # an unlabelled minor tick so the reader can see which points on the
            # curve are measurements and which are the line between them.
            ax[i].set_xticks(LABEL_TICKS)
            ax[i].set_xticks([r for r in RATES if r not in LABEL_TICKS],
                             minor=True)
            ax[i].grid(axis="both", **GRID)
            ax[i].set_axisbelow(True)
        # Over EVERY arm, not only the drawn ones, so that the with- and
        # without-FluidServe versions share a y axis and a value read off one
        # sits at the same height on the other.
        top = max(v["thru"][2] for lab in data for v in data[lab].values())
        ax[0].set_ylim(0, 1.05 * top)
        ax[0].yaxis.set_major_formatter(kfmt())
        ax[0].set_ylabel("Tokens/s")
        ax[1].set_ylabel("Tokens/s")
        for i in (2, 3):
            ax[i].set_ylim(0, 105)
            ax[i].set_yticks([0, 25, 50, 75, 100])
        ax[2].set_ylabel("SLO attainment (%)")
        ax[3].set_ylabel("Rejected (%)")

        if legend:
            fig.legend(handles, drawn, loc="lower center", ncol=len(drawn),
                       bbox_to_anchor=(0.5, 0.875), frameon=False,
                       columnspacing=1.0, handlelength=1.5, handletextpad=0.3)
        fig.tight_layout(rect=(0, 0, 1, 0.878), w_pad=2.0, pad=0.25)
        save(fig, out_path)
        return drawn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="q90",
                    choices=["mean", "cum", "cumt", "q90", "q95", "q99",
                             "q90end", "ft90"])
    ap.add_argument("--prefix", default="raw", choices=["raw", "deb"])
    ap.add_argument("--csv", default=CSV_DEFAULT,
                    help="per-run scoring table to draw from; the default is "
                         "the paper's table, see the note beside CSV_DEFAULT")
    args = ap.parse_args()
    global CSV
    CSV = args.csv

    data, counts = collect(args.rule, args.prefix)
    report(data, counts, args.rule)

    base = os.path.join(HERE, "motivation_throughput_vs_goodput_4panel_t75")
    baselines = [l for _, l, _ in ARMS if l != OURS]
    drawn_b = build(data, baselines, base + ".pdf")
    write_csv(data, baselines, counts, args.rule, base + ".csv")
    drawn_a = build(data, [l for _, l, _ in ARMS], base + "_withfs.pdf")
    write_csv(data, [l for _, l, _ in ARMS], counts, args.rule,
              base + "_withfs.csv")
    print(f"\narms drawn: baselines {drawn_b}\n            with ours  {drawn_a}")

    # Caption numbers are computed from the same table the panels are drawn
    # from, never written in by hand: a re-measurement moves them and a literal
    # would not follow.
    for rate in (45.0, 70.0):
        if not all(rate in data[l] for l in data):
            continue
        thru = {l: data[l][rate]["thru"][0] for l in data}
        good = {l: data[l][rate]["good"][0] for l in data}
        print(f"\nat {rate:.0f} req/s: throughput "
              f"{min(thru.values()):,.0f}-{max(thru.values()):,.0f} tok/s "
              f"({max(thru.values()) / max(min(thru.values()), 1e-9):.1f}x), "
              f"goodput {min(good.values()):,.0f}-{max(good.values()):,.0f} "
              f"({max(good.values()) / max(min(good.values()), 1e-9):.1f}x)")


if __name__ == "__main__":
    main()
