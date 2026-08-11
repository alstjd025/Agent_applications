#!/usr/bin/env python3
"""Paper figure: what the engines were doing underneath the scores — evictions
on the left, how full the KV pool was on the right.

  preemption_kv.pdf       3.335 x 1.75 in, one column, width=\\columnwidth
  preemption_kv_wide.pdf  7.0 x 1.62 in, `figure*`, width=\\textwidth

  (a) Preemptions per 8-minute condition, summed over the four engines
  (b) KV cache occupancy, mean over the four engines and over all samples

FLUIDSERVE IS ABSENT, as in the other motivation figures: the point is what the
deployed and published control planes do to the engines, and it does not need
our system to be made.

WHAT A PREEMPTION COSTS, AND WHY IT GETS ITS OWN PANEL. vLLM V1 preempts by
recompute: the prefill already done for the evicted request is discarded and
paid for again when it is rescheduled. That engine time produces no tokens and
no request-level metric attributes it to anything, so it is invisible in every
other figure in this paper.

⚠ THE TWO PANELS CANNOT BE READ SEPARATELY, BECAUSE ZERO PREEMPTIONS HAPPENS FOR
TWO OPPOSITE REASONS. llm-d sits at zero at every rate, and not because it is
healthy: it refuses 52-80% of arrivals from 35 req/s on and its KV pool holds
28-32%, so its engines are short of work and there is nothing to evict. Its
token throughput is the lowest of the four. The right panel is what separates
starved from comfortable from thrashing, and the caption has to say that the
left panel alone ranks nothing.

⚠ AND PREEMPTIONS FALLING IS NOT AN IMPROVEMENT EITHER. PolyServe drops from
3,175 at 35 req/s to 1,902 at 70. At 70 one of its engines holds a waiting queue
of 7,738 requests, so most arrivals never enter the running batch at all and
cannot be evicted from it. Preemption counts the requests that got in and were
then pushed out, so a queue deep enough to block entry drives the count DOWN.
Said plainly: the arm with the fewest evictions at 70 req/s among the three that
evict at all is the arm doing worst.

HOW THE NUMBERS ARE MADE. The preemption count is the difference between the
last and the first sample of `vllm:num_preemptions_total` in each engine's own
metrics file, summed over the four engines, per 8-minute condition. KV occupancy
is the mean of `vllm:kv_cache_usage_perc` over the four engines and every sample
in the condition -- a mean, not a peak, so a pool that is full half the time
reads near 50 rather than near 100. Error bars are min..max over repeats and a
point with no bar is one run, which is not the same as a precise one.

DATA. The same runs and the same `ARMS` table as `fig_intro_capacity.py`, which
this imports rather than copying: PolyServe and Llumnix SLO from EXP-72 (one
repeat at every rate), llm-d from EXP-68/70 (two repeats at 35-70, one at 10-25),
the vLLM router from EXP-77 (two repeats at every rate). Eight rates, the
workload as fixed on 2026-08-08. The exploratory version of this figure, with
FluidServe and with both panels stacked, is
`analysis_scripts/request_level/preemption_vs_rate.py`.

    python3 paper_figures/fig_preemption_kv.py
"""
import collections
import glob
import importlib.util
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import paper_style as ps  # noqa: E402


def _load_module(path, name="m"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CAP = _load_module(os.path.join(HERE, "fig_intro_capacity.py"), "capfig")
OURS = "FluidServe"
ARMS = [a for a in CAP.ARMS if a[0] != OURS]

PRE = "vllm:num_preemptions_total"
KV = "vllm:kv_cache_usage_perc"
QUEUE = "vllm:num_requests_waiting"
BATCH = "vllm:num_requests_running"
PCT = 90
TITLES = ["(a) Preemptions", "(b) KV occupancy"]
XTICKS = [10, 30, 50, 70]


def series(run):
    """One condition: preemptions, and three engine gauges.

    Returns (preemptions, KV mean %, KV p90 %, queue p90, batch p90).

    Read from the engine's own Prometheus series rather than from anything the
    load generator recorded, so nothing here depends on the client.

    EVERY PERCENTILE IS TAKEN PER ENGINE FIRST AND THEN AVERAGED OVER THE FOUR,
    not pooled across engines. Pooling would let one saturated engine and three
    idle ones read as a fleet at the middle, which is exactly PolyServe's state.
    The busiest-engine variant is in
    `analysis_scripts/request_level/engine_state_p90.py`, and for PolyServe the
    two differ a great deal -- queue p90 2,794 averaged against 7,640 on the
    engine that holds the class carrying 77% of the requests.
    """
    total, seen = 0.0, 0
    per = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(run, "server_metrics",
                                           "engine_*.jsonl"))):
        first = last = None
        vals = collections.defaultdict(list)
        with open(f) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                for k, v in d.items():
                    if v is None:
                        continue
                    if k.startswith(PRE):
                        if first is None:
                            first = v
                        last = v
                    elif k.startswith(KV):
                        vals["kv"].append(v)
                    elif k.startswith(QUEUE):
                        vals["queue"].append(v)
                    elif k.startswith(BATCH):
                        vals["batch"].append(v)
        # A counter, so the condition's total is last minus first. Clamped at
        # zero because an engine restarted mid-condition would otherwise
        # contribute a negative count.
        if first is not None and last is not None:
            total += max(0.0, last - first)
            seen += 1
        for name, xs in vals.items():
            if xs:
                sc = 100.0 if name == "kv" else 1.0
                per[name].append((sc * float(np.percentile(xs, PCT)),
                                  sc * float(np.mean(xs))))
    if not seen:
        return None

    def fleet(name, idx):
        xs = per.get(name)
        return float(np.mean([v[idx] for v in xs])) if xs else np.nan

    return (total, fleet("kv", 1), fleet("kv", 0),
            fleet("queue", 0), fleet("batch", 0))


def collect():
    out = {}
    for name, _, _, pats in ARMS:
        acc = collections.defaultdict(list)
        seen = set()
        for pat in pats:
            for d in sorted(glob.glob(os.path.join(ROOT, pat))):
                if "PRERUN" in d or d in seen:
                    continue
                seen.add(d)
                s = series(d)
                if s is None:
                    continue
                acc[int(re.search(r"rpm_(\d+)", d).group(1)) / 60.0].append(s)
        out[name] = dict(acc)
    return out


def build(data, arms, out, width, height):
    with plt.rc_context(ps.STYLE):
        fig, ax = plt.subplots(1, 2, figsize=(width, height))
        handles, labels = [], []

        for name, col, mk, _ in arms:
            x = sorted(data[name])
            for i in (0, 1):
                y = np.array([np.mean([v[i] for v in data[name][r]]) for r in x])
                lo = y - np.array([min(v[i] for v in data[name][r]) for r in x])
                hi = np.array([max(v[i] for v in data[name][r]) for r in x]) - y
                h = ax[i].errorbar(x, y, yerr=[lo, hi], color=col, marker=mk,
                                   ms=2.8, lw=1.2, capsize=1.5,
                                   mec="white", mew=0.4)
                if i == 0:
                    handles.append(h)
                    labels.append(name)

        for i in (0, 1):
            ax[i].set_xlabel(f"Offered rate (req/s)\n{TITLES[i]}",
                             labelpad=1.5, linespacing=1.6)
            ax[i].set_xlim(7, 73)
            ax[i].set_xticks(XTICKS)
            ax[i].grid(axis="both", **ps.GRID)
            ax[i].set_axisbelow(True)
        ax[0].set_ylabel("Preemptions")
        ax[0].set_ylim(0, None)
        ax[0].yaxis.set_major_formatter(ps.kfmt())
        ax[1].set_ylabel("KV occupancy (%)")
        ax[1].set_ylim(0, 105)
        ax[1].set_yticks([0, 25, 50, 75, 100])

        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 0.872), frameon=False, fontsize=6.5,
                   columnspacing=0.6, handlelength=1.2, handletextpad=0.3,
                   borderaxespad=0.0)
        fig.tight_layout(rect=(0, 0, 1, 0.878), w_pad=1.0, pad=0.3)
        ps.save(fig, out)


# The 2x2 grid: (index into the tuple `series` returns, title, y label).
GRID_PANELS = [
    # "Counts", not "requests": `vllm:num_preemptions_total` counts preemption
    # OCCURRENCES, and vLLM V1 returns an evicted request to the waiting queue,
    # so one request preempted twice counts twice. The number is therefore at
    # least the number of distinct requests evicted and usually more. (c) and (d)
    # are gauges of requests waiting and running, where "requests" IS the unit.
    # ⚠ THE AXIS NO LONGER SAYS "per condition", so the caption must: this is a
    # total accumulated over one 8-minute condition, not a rate, and it is not
    # comparable with a run of a different length.
    (0, "(a) Preemptions", "Counts"),
    (2, "(b) KV occupancy", "p90 (%)"),
    (3, "(c) Waiting queue", "p90 (requests)"),
    (4, "(d) Running batch", "p90 (requests)"),
]
GRID_H = 3.05


def build_grid(data, arms, out):
    """Four engine-layer quantities in a 2x2 grid, one column wide.

    (b) IS THE 90TH PERCENTILE HERE AND THE MEAN IN THE TWO-PANEL FIGURE, and the
    difference is not cosmetic. A preemption fires when the pool runs out, so
    what predicts one is the top of the KV distribution rather than its centre.
    On the mean, Llumnix SLO reads 70-76% at 35-70 req/s and looks like it has
    room; on p90 it reads 98-100% and does not. The mean answers "how much memory
    was in use over the run", which is a different question and is what the
    two-panel figure asks.

    THE QUEUE PANEL IS SYMLOG. Its values run from 0, which llm-d holds at every
    rate, to 4,657, and a plain log axis would drop the zeros while a linear one
    would flatten everything below about 200 -- which is all of llm-d and all of
    Llumnix SLO -- onto the axis.
    """
    with plt.rc_context(ps.STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(ps.COL_W, GRID_H), sharex=True)
        ax = axes.ravel()
        handles, labels = [], []

        for name, col, mk, _ in arms:
            x = sorted(data[name])
            for i, (key, _, _) in enumerate(GRID_PANELS):
                y = np.array([np.mean([v[key] for v in data[name][r]])
                              for r in x])
                lo = y - np.array([min(v[key] for v in data[name][r])
                                   for r in x])
                hi = np.array([max(v[key] for v in data[name][r])
                               for r in x]) - y
                h = ax[i].errorbar(x, y, yerr=[lo, hi], color=col, marker=mk,
                                   ms=2.6, lw=1.1, capsize=1.3,
                                   mec="white", mew=0.4)
                if i == 0:
                    handles.append(h)
                    labels.append(name)

        for i, (_, title, ylab) in enumerate(GRID_PANELS):
            ax[i].set_title(title, fontsize=8, pad=2)
            ax[i].set_ylabel(ylab, labelpad=1.5)
            ax[i].set_xlim(7, 73)
            ax[i].set_xticks(XTICKS)
            ax[i].grid(axis="both", **ps.GRID)
            ax[i].set_axisbelow(True)
            # Tick NUMBERS on all four, the axis NAME only on the bottom row:
            # every panel has its own scale so a reader needs the values in each
            # cell, while the x quantity is the same in all four.
            ax[i].tick_params(labelbottom=True)
        for i in (2, 3):
            ax[i].set_xlabel("Offered rate (req/s)", labelpad=1.5)

        ax[0].set_ylim(0, None)
        ax[0].yaxis.set_major_formatter(ps.kfmt())
        ax[1].set_ylim(0, 105)
        ax[1].set_yticks([0, 25, 50, 75, 100])
        ax[2].set_yscale("symlog", linthresh=1.0)
        # Top just above the largest value (4,657) so the highest labelled
        # decade is 10^3 and the curves are not pushed into the lower half of
        # the panel by empty headroom.
        ax[2].set_ylim(0, 8000)
        ax[3].set_ylim(0, None)

        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 0.935), frameon=False, fontsize=6.5,
                   columnspacing=0.6, handlelength=1.2, handletextpad=0.3,
                   borderaxespad=0.0)
        fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.0, h_pad=0.9, pad=0.3)
        ps.save(fig, out)


def main():
    data = collect()
    arms = [a for a in ARMS if data[a[0]]]
    missing = [n for n, _, _, _ in ARMS if not data[n]]
    if missing:
        print(f"  NOT DRAWN: {', '.join(missing)} has no conditions")

    print(f"{'arm':13s} {'rate':>5s} {'preempt':>9s} {'KVmean':>7s} "
          f"{'KVp90':>6s} {'Qp90':>7s} {'Bp90':>6s} {'n':>2s}")
    for name, _, _, _ in arms:
        for r in sorted(data[name]):
            v = data[name][r]
            m = [np.mean([x[i] for x in v]) for i in range(5)]
            print(f"{name:13s} {r:5.0f} {m[0]:9,.0f} {m[1]:7.1f} {m[2]:6.1f} "
                  f"{m[3]:7,.0f} {m[4]:6,.0f} {len(v):2d}")

    build(data, arms, os.path.join(HERE, "preemption_kv.pdf"), ps.COL_W, 1.75)
    build(data, arms, os.path.join(HERE, "preemption_kv_wide.pdf"),
          ps.TEXT_W, 1.62)
    build_grid(data, arms, os.path.join(HERE, "engine_state_2x2.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
