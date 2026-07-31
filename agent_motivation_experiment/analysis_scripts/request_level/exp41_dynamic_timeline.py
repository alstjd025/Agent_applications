#!/usr/bin/env python3
"""EXP-41: the hour, as a timeline, for two policies on one trace.

A whole-run mean of a run whose load moves cannot answer what the run was for.
These panels put every quantity on the same time axis so a difference can be
read against the load that produced it.

  A  offered rate, the context every other panel is read against
  B  SLO attainment, offered denominator, in a sliding window
  C  token goodput, output tokens/s from requests that met their rule
  D  rejection rate, which is the gap between B and the admitted view
  E  attainment per class, one line style per policy
  F  what the four engines were holding, summed per policy

Requests are anchored on ARRIVAL, so a point at minute t is "of the requests
that arrived around t, what fraction met their rule" -- the queueing-theory
convention, and the one that makes B comparable with the rate in A. A request
that arrives at t and finishes at t+30s is scored at t.

  python3 exp41_dynamic_timeline.py --variant full --out-dir <dir>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp22_fluidserve import (  # noqa: E402
    PAPER_STYLE, CLASSES, CLASS_COLORS, load_run, attain,
)

ARMS = {"slo": ("Llumnix SLO", "#2ca02c", "--"),
        "fluidserve": ("FluidServe", "#1f77b4", "-")}
WIN, STEP = 90.0, 30.0


def windows(r, dur):
    """Sliding windows anchored on arrival: (centre minute, rows)."""
    t = WIN / 2.0
    while t + WIN / 2.0 <= dur:
        yield t / 60.0, r[(r["rel"] >= t - WIN / 2) & (r["rel"] < t + WIN / 2)]
        t += STEP


def engine_total(run):
    """Decode batch and KV occupancy summed / averaged over the four engines."""
    ts, bt, kv = [], [], []
    for f in sorted(glob.glob(os.path.join(run, "server_metrics", "engine_*.jsonl"))):
        t, b, k = [], [], []
        for line in open(f):
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if not o.get("ok"):
                continue

            def pick(pref):
                for kk, vv in o.items():
                    if kk.startswith(pref) and isinstance(vv, (int, float)):
                        return float(vv)
                return np.nan
            t.append(o["t"]); b.append(pick("vllm:num_requests_running"))
            k.append(pick("vllm:kv_cache_usage_perc") * 100)
        if t:
            t = np.array(t) - t[0]
            ts.append(t); bt.append(np.array(b)); kv.append(np.array(k))
    if not ts:
        return None
    grid = np.arange(0, min(x.max() for x in ts), 10.0)
    B = sum(np.interp(grid, x, y) for x, y in zip(ts, bt))
    K = sum(np.interp(grid, x, y) for x, y in zip(ts, kv)) / len(ts)
    return grid / 60.0, B, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full")
    ap.add_argument("--pattern", default="results/*exp41r1_{arm}_{variant}")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    runs = {}
    for arm in ARMS:
        hits = glob.glob(a.pattern.format(arm=arm, variant=a.variant))
        if hits:
            runs[arm] = sorted(hits)[-1]
    if not runs:
        sys.exit(f"no runs for variant {a.variant}")
    print(f"variant {a.variant}: {list(runs)}")

    data = {arm: load_run(d) for arm, d in runs.items()}
    dur = min(r["rel"].max() for r in data.values())

    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(3, 2, figsize=(11.0, 8.4), sharex=True)
        ax = ax.ravel()

        for arm, r in data.items():
            lab, c, ls = ARMS[arm]
            x, att, gp, rej = [], [], [], []
            per = {cl: [] for cl in CLASSES}
            for t, g in windows(r, dur):
                if len(g) < 30:
                    continue
                x.append(t)
                att.append(attain(g, "violate_offered"))
                ok = g[(~g["violate_offered"]) & (~g["cutoff"])]
                gp.append(pd.to_numeric(ok.get("output_tokens"), errors="coerce")
                          .fillna(0).sum() / WIN)
                rej.append(100.0 * g["rejected"].mean())
                for cl in CLASSES:
                    per[cl].append(attain(g[g["class"] == cl], "violate_offered"))
            # A: the load. Drawn once per arm; they see the same trace, so the
            # two lines lying on top of each other is the check that they did.
            b = (r["rel"] // STEP).astype(int)
            ax[0].plot(np.array(sorted(b.unique())) * STEP / 60.0,
                       b.value_counts().sort_index().values / STEP,
                       color=c, ls=ls, lw=0.7, alpha=0.8, label=lab)
            ax[1].plot(x, att, color=c, ls=ls, label=lab)
            ax[2].plot(x, gp, color=c, ls=ls, label=lab)
            ax[3].plot(x, rej, color=c, ls=ls, label=lab)
            for cl in CLASSES:
                ax[4].plot(x, per[cl], color=CLASS_COLORS[cl], ls=ls, lw=1.1)
            eng = engine_total(runs[arm])
            if eng:
                ax[5].plot(eng[0], eng[1], color=c, ls=ls, label=f"{lab} batch")
                ax[5].plot(eng[0], eng[2] * 20, color=c, ls=":", lw=0.8, alpha=0.6)

        titles = ["A. offered rate (30 s bins) — both arms see the same trace",
                  f"B. SLO attainment, offered denominator ({WIN:.0f} s window)",
                  "C. token goodput — output tokens/s from requests that met their rule",
                  "D. rejection rate",
                  "E. attainment per class (colour = class, style = policy)",
                  "F. fleet decode batch (solid) and mean KV % x20 (dotted)"]
        ylabs = ["req/s", "attainment (%)", "tokens/s", "rejected (%)",
                 "attainment (%)", "requests"]
        for i, (t, y) in enumerate(zip(titles, ylabs)):
            ax[i].set_title(t, fontsize=8)
            ax[i].set_ylabel(y)
            ax[i].grid(axis="y", ls=":", lw=0.7, alpha=0.6)
            ax[i].set_xlim(0, dur / 60.0)
        for i in (1, 4):
            ax[i].set_ylim(0, 105)
        for i in (4, 5):
            ax[i].set_xlabel("time (minutes)")
        ax[1].legend(fontsize=7, loc="lower left")
        h = [plt.Line2D([], [], color=CLASS_COLORS[c], lw=1.1) for c in CLASSES]
        h += [plt.Line2D([], [], color="#666666", ls=ARMS[k][2], lw=1.1) for k in data]
        ax[4].legend(h, list(CLASSES) + [ARMS[k][0] for k in data],
                     fontsize=6, ncol=2, loc="upper right")
        # The mix steps every 15 minutes on the compressed trace; the verbatim
        # one holds m1 throughout, so the guides would be meaningless there.
        if a.variant == "full":
            for i in range(6):
                for m in (15, 30, 45):
                    ax[i].axvline(m, color="#999999", lw=0.5, ls=":")
            ax[0].annotate("mix m1 | m2 | m3 | m1", (0.5, 0.92), xycoords="axes fraction",
                           ha="center", fontsize=6.5, color="#666666")
        fig.suptitle(f"EXP-41 {a.variant} — one hour of moving load, two control planes, "
                     f"both on stock FIFO (one run each)", fontsize=9, y=1.01)
        fig.tight_layout()
        p = os.path.join(a.out_dir, f"exp41_{a.variant}_timeline.png")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
