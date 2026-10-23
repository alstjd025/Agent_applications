#!/usr/bin/env python3
"""Figure 1 candidate, MEASURED: a tighter resident shrinks the admissible
batch of an instance.

  operating_space_data.pdf        3.335 x 1.75 in, one column, no scaling
  operating_space_data_split.pdf  the same envelope in two panels: (a) Deep
                                  Research residents, whole range admissible;
                                  (b) plus one Chat resident, admissible up to
                                  the 50 ms crossing, the rest marked lost
  (and a 200 dpi PNG of each)

x = how many requests were decoding on the same instance while a request
ran (its mean concurrency); y = that request's mean time-between-tokens.
The black line is the median over requests at that concurrency, the band
the 10th-90th percentile: the instance's iteration-latency envelope,
measured. The dashed lines are the per-class TBT budgets of the paper
(Chat 50, Agent 75, Deep Research 100 ms). Where a budget meets the
envelope is the largest batch at which a resident of that class still
meets its budget, i.e. the instance's capacity while such a request is
resident. The budget lines are drawn to the 90th-percentile curve
(conservative: 9 in 10 requests at that batch meet it); the median
crossing is printed for the caption.

DATA. EXP-109 request traces (Llama-3.1-70B, 4 instances, one-hour trace,
first 12-18 min of each run survive in the trace files), the arms WITHOUT
capacity-aware admission so the instances are driven across the whole
batch range: vLLM router (repeats 1 and 2) and Llumnix (repeat 2). Per
request: mean_tbt_ms from the scorer; concurrency = the time-average number
of requests on its engine between its first token and its last, from the
first_token_ts / end_ts of every request on that engine. Requests with
under 0.5 s of decode are dropped (their mean is too noisy). Binned by 10
concurrent requests, bins with fewer than MIN_N requests dropped.

    python3.12 fig_operating_space_data.py
"""
import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paper_style import COL_W, STYLE, GRID, save  # noqa: E402

TRACE_DIR = os.path.join(os.path.dirname(HERE), "data", "trace")
RUNS = ["260901_2010_exp109r2_vllmcachet75_shift",
        "260901_2137_exp109r1_vllmcachet75_shift",
        "260901_1856_exp109r2_slot75_shift"]
SLO = [("chat", "Chat", 50, "#3a8f8a"), ("agent", "Agent", 75, "#b07cc6"),
       ("deepresearch", "Deep Research", 100, "#4c72b0")]
BIN = 10
MIN_N = 30
MIN_DECODE_S = 0.5
FS = STYLE["font.size"] - 1.5
ANNOT = FS - 1.5


def concurrency(d):
    """Time-average number of decoding requests on the request's engine over
    its own decode interval, from every request's first_token_ts / end_ts."""
    out = []
    for _, g in d.groupby("engine_port"):
        ev = np.concatenate([np.c_[g.first_token_ts.values, np.ones(len(g))],
                             np.c_[g.end_ts.values, -np.ones(len(g))]])
        ev = ev[np.argsort(ev[:, 0], kind="stable")]
        t, cnt = ev[:, 0], np.cumsum(ev[:, 1])
        integ = np.concatenate([[0.0], np.cumsum(cnt[:-1] * np.diff(t))])

        def I(x):
            i = np.clip(np.searchsorted(t, x, side="right") - 1, 0, len(t) - 1)
            return integ[i] + cnt[i] * (x - t[i])
        a, b = g.first_token_ts.values, g.end_ts.values
        dur = b - a
        ok = dur > MIN_DECODE_S
        gg = g.copy()
        gg["conc"] = np.where(ok, (I(b) - I(a)) / np.where(ok, dur, 1.0), np.nan)
        out.append(gg)
    return pd.concat(out).dropna(subset=["conc"])


def load():
    parts = []
    for run in RUNS:
        p = os.path.join(TRACE_DIR, run + "_trace.csv")
        d = pd.read_csv(p).dropna(subset=["first_token_ts", "end_ts", "engine_port", "mean_tbt_ms"])
        d = concurrency(d)
        d["run"] = run.split("_")[2] + "/" + run.split("_")[3]
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def envelope(r):
    edges = np.arange(0, r.conc.max() + BIN, BIN)
    b = pd.cut(r.conc, edges)
    g = r.groupby(b, observed=True).mean_tbt_ms
    tab = pd.DataFrame({"n": g.size(), "p10": g.quantile(0.1), "p50": g.median(),
                        "p90": g.quantile(0.9)})
    tab = tab[tab.n >= MIN_N]
    tab["x"] = [iv.mid for iv in tab.index]
    return tab.reset_index(drop=True)


def crossing(x, y, lvl):
    """First x at which the (piecewise-linear) curve y reaches lvl, or None."""
    for i in range(1, len(x)):
        if y[i - 1] < lvl <= y[i]:
            f = (lvl - y[i - 1]) / (y[i] - y[i - 1])
            return x[i - 1] + f * (x[i] - x[i - 1])
    return None


def main():
    r = load()
    print(f"requests {len(r)}; per run: {r.run.value_counts().to_dict()}; "
          f"max concurrency {r.conc.max():.0f}")
    tab = envelope(r)
    x, p50, p90 = tab.x.values, tab.p50.values, tab.p90.values
    xmax_obs = x[-1] + BIN / 2
    caps = {}
    for k, name, ms, _ in SLO:
        c90, c50 = crossing(x, p90, ms), crossing(x, p50, ms)
        caps[k] = c90
        print(f"{name:14s} {ms:3d} ms: p90 crosses at {c90 if c90 is None else round(c90)}, "
              f"median crosses at {c50 if c50 is None else round(c50)}")
    for run, g in r.groupby("run"):
        t = envelope(g)
        print(f"  {run}: chat cap (p90) {crossing(t.x.values, t.p90.values, 50)}, "
              f"largest bin {t.x.values[-1] + BIN / 2:.0f}")
    tight = caps["chat"]
    loose_end = caps["deepresearch"] if caps["deepresearch"] is not None else xmax_obs

    st = dict(STYLE)
    for k in ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize", "ytick.labelsize"):
        st[k] = STYLE[k] - 1.5
    with plt.rc_context(st):
        fig, ax = plt.subplots(figsize=(COL_W, 1.75))
        fig.subplots_adjust(left=0.15, right=0.985, top=0.90, bottom=0.21)
        ymax = 112
        # the two regimes as light vertical spans: admissible with a Chat
        # resident, and the range only looser residents allow
        ax.axvspan(0, tight, color="#3a8f8a", alpha=0.10, linewidth=0, zorder=0)
        ax.axvspan(tight, loose_end, color="#000000", alpha=0.05, linewidth=0, zorder=0)
        # regime names as a header row above the axes
        ax.text(tight / 2, 1.02, "admissible with a Chat resident", ha="center", va="bottom",
                fontsize=ANNOT, color="#2c6e69", transform=ax.get_xaxis_transform())
        ax.text((tight + loose_end) / 2, 1.02, "admissible only without one", ha="center",
                va="bottom", fontsize=ANNOT, color="#555555", transform=ax.get_xaxis_transform())
        # measured envelope
        ax.fill_between(x, tab.p10, tab.p90, color="#b5b5b5", alpha=0.5, linewidth=0, zorder=2,
                        label="p10–p90")
        ax.plot(x, p50, color="black", lw=1.0, zorder=3, label="median")
        # budgets: dashed to their crossing, labelled once at the left
        for k, name, ms, col in SLO:
            c = caps[k]
            xe = c if c is not None else loose_end
            ax.plot([0, xe], [ms, ms], color=col, lw=0.8, ls=(0, (3, 1.5)), zorder=4)
            ax.text(4, ms + 1.2, f"{name} budget ({ms} ms)", ha="left", va="bottom",
                    fontsize=ANNOT, color=col, zorder=5)
            if c is not None:
                ax.plot([c, c], [0, ms], color=col, lw=0.6, ls=(0, (1, 1.5)), zorder=4)
                ax.plot(c, ms, "o", ms=3.0, color=col, markerfacecolor="white",
                        markeredgewidth=0.9, zorder=6)
                ax.text(c + 4, 2, f"{c:.0f}", ha="left", va="bottom", fontsize=ANNOT,
                        color=col, zorder=7)
        # axes: major ticks every 100; the crossings as labelled minor ticks
        ax.set_xlim(0, loose_end)
        ax.set_ylim(0, ymax)
        ax.set_xticks([0, 100, 200, 300])
        ax.set_xticks([50, 150, 250, 350], minor=True)
        ax.set_yticks([0, 50, 100])
        ax.set_xlabel("Concurrent decoding requests on the instance", labelpad=2)
        ax.set_ylabel("Mean TBT (ms)", labelpad=3)
        ax.grid(axis="y", **GRID)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="lower left", bbox_to_anchor=(0.50, 0.02), frameon=False,
                  fontsize=STYLE["font.size"] - 2.5, handlelength=1.6, handletextpad=0.4,
                  borderaxespad=0.0, labelspacing=0.3)
        fig.savefig(os.path.join(HERE, "operating_space_data.png"), dpi=200)
        save(fig, os.path.join(HERE, "operating_space_data.pdf"))

    # ---- split version: the same envelope twice, one resident mix per panel ----
    with plt.rc_context(st):
        fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.7), sharey=True,
                                 gridspec_kw=dict(wspace=0.10))
        fig.subplots_adjust(left=0.15, right=0.985, top=0.80, bottom=0.21)
        panels = [("(a) Deep Research residents", "deepresearch"),
                  ("(b) plus one Chat resident", "chat")]
        for ax, (title, k) in zip(axes, panels):
            name, ms, col = next((n, m, c) for kk, n, m, c in SLO if kk == k)
            c = caps[k]
            cap_x = c if c is not None else loose_end
            ax.axvspan(0, cap_x, color="#3a8f8a", alpha=0.10, linewidth=0, zorder=0)
            if c is not None:
                ax.axvspan(cap_x, loose_end, color="#000000", alpha=0.06, linewidth=0, zorder=0)
                ax.text((cap_x + loose_end) / 2, 8, "lost", ha="center", va="bottom",
                        fontsize=ANNOT, color="#555555")
            ax.fill_between(x, tab.p10, tab.p90, color="#b5b5b5", alpha=0.5, linewidth=0, zorder=2)
            ax.plot(x, p50, color="black", lw=1.0, zorder=3)
            ax.plot([0, cap_x], [ms, ms], color=col, lw=0.8, ls=(0, (3, 1.5)), zorder=4)
            ax.text(4, ms + 1.2, f"{ms} ms budget", ha="left", va="bottom",
                    fontsize=ANNOT, color=col, zorder=5)
            if c is not None:
                ax.plot([c, c], [0, ms], color=col, lw=0.6, ls=(0, (1, 1.5)), zorder=4)
                ax.plot(c, ms, "o", ms=3.0, color=col, markerfacecolor="white",
                        markeredgewidth=0.9, zorder=6)
            label = f"admissible batch {c:.0f}" if c is not None else \
                f"admissible batch \u2265 {loose_end:.0f}"
            ax.set_title(f"{title}\n{label}", pad=3, linespacing=1.15)
            ax.set_xlim(0, loose_end)
            ax.set_ylim(0, ymax)
            ax.set_xticks([0, 100, 200, 300])
            ax.set_xticks([50, 150, 250, 350], minor=True)
            ax.set_yticks([0, 50, 100])
            ax.grid(axis="y", **GRID)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        axes[0].set_ylabel("Mean TBT (ms)", labelpad=3)
        fig.text(0.57, 0.02, "Concurrent decoding requests on the instance", ha="center",
                 va="bottom", fontsize=st["axes.labelsize"])
        fig.savefig(os.path.join(HERE, "operating_space_data_split.png"), dpi=200)
        save(fig, os.path.join(HERE, "operating_space_data_split.pdf"))


if __name__ == "__main__":
    main()
