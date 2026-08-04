#!/usr/bin/env python3
"""Paper figure: EXP-55 — each class saturates at a different rate, in a
different engine state.

  exp55_class_knees.pdf     7.0 x 1.90 in, `figure*`, width=\\textwidth

Four panels, one curve per class, from the single-class sweep: only one class is
offered at a time, so no routing decision can differentiate anything and the
four engines are four replicas of the same experiment.

THE X AXIS IS OFFERED RATE DIVIDED BY THAT CLASS'S OWN KNEE. Drawn against
absolute rate the three classes occupy three disjoint stretches — chat 40-90,
agent 12-48, deep research 8-30 req/s — so no two curves share an x value and
each hangs in its own third of the panel with nothing to compare against.
Normalised, all three span roughly 0.4 to 1.7 and lie on top of one another,
which is what turns the figure from three separate sweeps into one comparison.

The absolute knees are the fact that normalising would otherwise throw away, so
they are printed in the legend. 1.0 is marked in every panel.

  (a) SLO attainment      the outcome. Normalised, the three nearly overlay:
                          every class collapses the same way across its own knee
  (b) distance to rule    each class against ITS OWN rule, 1.0 = the rule
  (c) KV occupancy        the engine state at that moment -- and here the three
                          do NOT overlay, which is the point of the figure
  (d) engine queue        the other engine state, on a symlog axis

(a) and (b) collapsing onto one curve while (c) and (d) stay three curves apart
IS the result: the classes fail identically relative to their own limit, and the
engine is in a completely different state each time they do.

The claim the figure exists for is the pair (c) and (d) read at the rate where
(b) crosses 1.0. All three classes fail when they reach their own rule, and the
engine looks completely different at each of those moments:

              knee     KV at knee   queue at knee
  chat        70 req/s     27%           2
  agent       28 req/s     64%           1
  deep res.   18 req/s    100%         140

**Chat's collapse is invisible in both (c) and (d).** At the rate where its
attainment falls from 94.9% to 53.3%, the KV pool is 27% used and two requests
are queued. A policy watching memory, or queue depth, sees nothing wrong at the
moment chat is already missing its rule. That is the argument for modelling
per-token pace per class rather than governing by a single occupancy threshold.

PANEL (b) IS NORMALISED PER CLASS BECAUSE THE THREE ARE NOT SCORED ON THE SAME
AXIS. chat and deep research are scored on time-to-first-token AND inter-token
latency; the agent class is scored end to end at 30 s and has no rule on either
of the other two. Plotting raw inter-token latency for all three would compare
a quantity that is the rule for two classes against one that is not the rule for
the third. The panel therefore plots, per request, how close it came to its own
rule — `max(ITL/budget, TTFT/budget)` for chat and deep research, `e2e/30 s` for
the agent class — and draws the median per condition. 1.0 is the rule for every
class, so the three curves cross it at their own knees.

WHICH HALF OF THE RULE ACTUALLY BREAKS, MEASURED, IS NOT THE SAME FOR ALL THREE:

  chat          inter-token only. TTFT violations are 0.0% at every rate and
                TTFT p95 is 1.12 s against a 5 s budget.
  agent         end-to-end. TTFT p50 is 0.49 s, so of the 32.8 s median at the
                knee essentially all is decode; the budget is a total but the
                quantity that fills it is per-token time.
  deep research BOTH, and this is the only class where queueing enters the rule.
                At the knee 22.1% of requests miss both halves and 3.0% miss
                TTFT alone; one rate higher TTFT p50 jumps from 0.69 s to
                11.21 s as the queue goes from 140 to 338.

ONE RUN PER CONDITION, 24 conditions, no repeats. And the experiment's own
caveat: bursts are spread over four engines, so each knee sits very slightly
higher than a single engine would show.

    python3 paper_figures/fig_exp55_class_knees.py
"""
import glob
import json
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

from paper_style import TEXT_W, STYLE, GRID, save  # noqa: E402
from exp22_fluidserve import load_run, per_request, CLASS_COLORS  # noqa: E402

RUNS = "results/*exp55r1_loadbalance_*"
TAG_RE = re.compile(r"_loadbalance_(schat|sdr|sswe)_rpm_(\d+)$")
# tag -> (label, colour, line style, the rule it is scored on)
CLS = {
    "schat": ("chat", CLASS_COLORS["chat"], "-", dict(ttft=5.0, tbt=50.0)),
    "sdr": ("deep research", CLASS_COLORS["deepresearch"], "--",
            dict(ttft=10.0, tbt=100.0)),
    "sswe": ("agent", CLASS_COLORS["swe"], "-.", dict(e2e=30.0)),
}
ORDER = ["schat", "sswe", "sdr"]
FIG_H = 1.90


def engine_series(run, prefix):
    """Every sample of one engine gauge, pooled over the four engines."""
    out = []
    for f in glob.glob(os.path.join(run, "server_metrics", "engine_*.jsonl")):
        for line in open(f):
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if not o.get("ok"):
                continue
            for k, v in o.items():
                if k.startswith(prefix) and isinstance(v, (int, float)):
                    out.append(float(v))
    return np.array(out)


def rule_distance(r, rule):
    """Per request, how close it came to ITS OWN rule. 1.0 is the rule.

    Two classes are scored on a pair of rules and one on a total, so the only
    quantity that can share an axis across all three is each request's distance
    to whichever rule applies to it. For the pair, the binding half is whichever
    is closer, hence the max.
    """
    if "e2e" in rule:
        return pd.to_numeric(r["latency"], errors="coerce") / rule["e2e"]
    ttft = pd.to_numeric(r["first_token_latency"], errors="coerce") / rule["ttft"]
    itl = pd.to_numeric(r["itl_ms"], errors="coerce") / rule["tbt"]
    return pd.concat([ttft, itl], axis=1).max(axis=1)


def collect():
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, RUNS))):
        m = TAG_RE.search(os.path.basename(d))
        if not m:
            continue
        tag, rpm = m.group(1), int(m.group(2))
        r = load_run(d)
        if r is None or r.empty:
            continue
        served = r[~r["cutoff"]]
        kv = engine_series(d, "vllm:kv_cache_usage_perc")
        q = engine_series(d, "vllm:num_requests_waiting")
        rows.append(dict(
            tag=tag, rate=rpm / 60.0,
            attain=per_request(r, "violate_offered"),
            dist=float(rule_distance(served, CLS[tag][3]).median()),
            kv=float(np.percentile(kv, 90) * 100) if len(kv) else np.nan,
            q=float(np.percentile(q, 90)) if len(q) else np.nan,
        ))
    return pd.DataFrame(rows).sort_values(["tag", "rate"])


def main():
    df = collect()
    if df.empty:
        print(f"no runs matched {RUNS}", file=sys.stderr)
        return 1

    knees = {}
    for tag in ORDER:
        d = df[df.tag == tag]
        # The knee: the first rate at which offered attainment falls below 60%.
        # This is the definition EXP-55 uses, and it is not the same as "the
        # median request crosses its rule" -- that one lands a step later for
        # chat and deep research, because the median crossing 1.0 IS the 50%
        # attainment point. Reporting both would invite quoting whichever is
        # convenient, so one is chosen and named.
        under = d[d.attain < 60.0]
        knee = under.rate.min() if len(under) else np.nan
        knees[tag] = knee
        at = d[d.rate == knee]
        if len(at):
            a = at.iloc[0]
            print(f"{CLS[tag][0]:14s} knee {knee:5.1f} req/s  "
                  f"attainment {a.attain:5.1f}%  KV p90 {a.kv:5.1f}%  "
                  f"queue p90 {a.q:7.1f}")

    # Normalise the x axis per class now that the knees are known.
    df["x"] = df.apply(lambda r: r.rate / knees[r.tag], axis=1)

    panels = [("attain", "SLO attainment (%)", None),
              ("dist", "Distance to rule\n(1.0 = the rule)", 1.0),
              ("kv", "KV occupancy p90 (%)", None),
              ("q", "Engine queue p90", None)]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(TEXT_W, FIG_H))
        handles, labels = [], []

        for ax, (col, ylab, hline) in zip(axes, panels):
            for tag in ORDER:
                d = df[df.tag == tag]
                lab, c, ls, _ = CLS[tag]
                # Lines only. Eight points per class and three classes now
                # overlapping in x, so markers add clutter without adding a
                # reading: the curves are dense enough to follow unaided.
                h, = ax.plot(d.x, d[col], color=c, ls=ls, lw=1.2)
                if col == "attain":
                    handles.append(h)
                    labels.append(f"{lab} ({knees[tag]:.0f} req/s)")
            if hline is not None:
                ax.axhline(hline, color="#555555", lw=0.7, ls=":", zorder=0)
            ax.set_xlim(0.4, 1.75)
            ax.set_xticks([0.5, 1.0, 1.5])
            ax.axvline(1.0, color="#555555", lw=0.7, ls=":", zorder=0)
            ax.set_xlabel("Offered rate / knee")
            ax.set_ylabel(ylab)
            ax.grid(axis="both", **GRID)
            ax.set_axisbelow(True)

        axes[0].set_ylim(0, 105)
        axes[0].set_yticks([0, 50, 100])
        # Log, because the quantity is a ratio and everything the panel is for
        # happens between 0.3 and 1.5; on a linear axis the deep research tail
        # at 6.8 compresses all three crossings into the bottom fifth.
        axes[1].set_yscale("log")
        axes[1].set_ylim(0.2, 10)
        axes[1].set_yticks([0.25, 0.5, 1, 2, 4, 8])
        axes[1].set_yticklabels(["0.25", "0.5", "1", "2", "4", "8"])
        axes[2].set_ylim(0, 105)
        axes[2].set_yticks([0, 50, 100])
        # Queue spans 0 to 2,666, and zero is a value the chat class holds all
        # the way through, so a plain log axis would drop it. symlog keeps 0.
        axes[3].set_yscale("symlog", linthresh=1.0)
        axes[3].set_ylim(0, 4000)

        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 0.905), frameon=False,
                   columnspacing=1.4, handlelength=2.0, handletextpad=0.5,
                   borderaxespad=0.0)
        fig.tight_layout(rect=(0, 0, 1, 0.895), w_pad=0.9, pad=0.35)
        save(fig, os.path.join(HERE, "exp55_class_knees.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
