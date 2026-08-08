#!/usr/bin/env python3
"""Three policies side by side at one offered rate, engine layer.

`plot_ratesweep_split.py` draws six panels for ONE condition, which is the right
shape for reading a single run in detail and the wrong shape for the question
this experiment asks, which is what three policies do differently with the same
four engines. This puts one policy per column and one quantity per row, on
shared y axes, so a difference between columns is a difference between policies
and not between scales.

Four rows, chosen because each answers something the request-level table cannot:

  running batch   whether the fleet is being used evenly. PolyServe's tier
                  partition shows up here directly: at 60 req/s two of its
                  engines carry about 1,000 and 700 requests while the other two
                  carry about 20.
  engine queue    work the engine has accepted but not started. A policy with no
                  admission control puts its backlog here, where it costs the
                  requests already running; a policy that holds at the gateway
                  does not.
  KV occupancy    how close each engine is to the memory bound, which is what
                  triggers preemption and recompute.
  decode tok/s    the fleet's actual output rate, summed. Attainment says how
                  many requests met their rule; this says how much work the
                  fleet did to get there.

Prefix hit rate and queueing time are deliberately left to the per-condition
figure: they are per engine and would need a fifth and sixth row that duplicate
what the first two already show, at a third of the height.

  python3 exp38_policy_compare.py --runs 'results/*exp38r1*' \
      --out-dir results/aggregate_analysis/exp38_engines/compare
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp22_fluidserve import PAPER_STYLE, arm_of  # noqa: E402

# An arm missing from ARM_ORDER is dropped from the figure and the script still
# writes a plausible-looking PNG with the remaining columns, so adding an arm
# here is part of running a new experiment. On 2026-08-08 an EXP-68 comparison
# was drawn with only the llm-d column because `fspfx` was not listed; the
# `arms=[...]` line the script prints at the end is what caught it, and the
# check below turns that from something to notice into something that fails.
ARM_ORDER = ["polyserve", "slo", "loadbalance", "llmdslo",
             "fluidserve", "fspfx", "fspfxb"]
ARM_LABEL = {"polyserve": "PolyServe", "slo": "Llumnix SLO",
             "loadbalance": "Llumnix", "llmdslo": "llm-d",
             "fluidserve": "FluidServe",
             "fspfx": "FluidServe\n(prefix-aware)",
             "fspfxb": "FluidServe\n(prefix-aware, calib. fixed)"}
ENGINE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

E_RUN = "vllm:num_requests_running"
E_WAIT = "vllm:num_requests_waiting"
E_KV = "vllm:kv_cache_usage_perc"
E_GEN = "vllm:generation_tokens_total"


def series(run):
    """Per-engine (t, running, waiting, kv%) plus the fleet decode rate.

    The generation counter is cumulative, so the rate is its difference over the
    sampling interval rather than the value itself. Ticks where the collector
    failed carry ok=false and are skipped; a skipped tick widens the interval it
    sits in rather than producing a spike, because the difference is divided by
    the actual elapsed time.
    """
    eng, gen_t, gen_v = {}, [], []
    for path in sorted(glob.glob(os.path.join(run, "server_metrics", "engine_*.jsonl"))):
        t, running, wait, kv, gen = [], [], [], [], []
        for line in open(path):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not rec.get("ok"):
                continue

            def pick(pref):
                for k, v in rec.items():
                    if k.startswith(pref) and isinstance(v, (int, float)):
                        return float(v)
                return np.nan

            t.append(rec["t"])
            running.append(pick(E_RUN))
            wait.append(pick(E_WAIT))
            kv.append(pick(E_KV) * 100)
            gen.append(pick(E_GEN))
        if not t:
            continue
        name = os.path.basename(path).replace("engine_", "").replace(".jsonl", "")
        t = np.array(t, dtype=float)
        eng[name] = (t - t[0], np.array(running, dtype=float),
                     np.array(wait, dtype=float), np.array(kv, dtype=float))
        gen_t.append(t - t[0])
        gen_v.append(np.array(gen, dtype=float))
    if not eng:
        return None
    # Fleet decode rate: interpolate each engine's cumulative counter onto a
    # common grid before differencing, because the four collectors do not tick
    # together and differencing per engine then summing would alias.
    grid = np.arange(0, min(x.max() for x in gen_t), 2.0)
    tot = np.zeros_like(grid)
    for tt, vv in zip(gen_t, gen_v):
        ok = np.isfinite(vv)
        if ok.sum() < 2:
            continue
        tot += np.interp(grid, tt[ok], vv[ok])
    rate = np.diff(tot) / np.diff(grid)
    return eng, grid[1:], rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["results/*exp38r1*"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--title", default="EXP-38")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    cells = {}
    for pat in a.runs:
        for d in sorted(glob.glob(pat)):
            # A rate sweep names the rate in the directory; a dynamic trace has
            # no single rate, so all its arms go into one cell keyed by the
            # variant. Without this branch every hour-long run was skipped and
            # the side-by-side figure existed only for sweeps.
            m = re.search(r"_rpm_(\d+)", os.path.basename(d))
            key = int(m.group(1)) if m else os.path.basename(d).rsplit("_", 1)[-1]
            cells.setdefault(key, {})[arm_of(d)] = d

    for rpm in sorted(cells, key=str):
        arms = [x for x in ARM_ORDER if x in cells[rpm]]
        unknown = sorted(set(cells[rpm]) - set(ARM_ORDER))
        if unknown:
            sys.exit(f"{rpm}: these arms have runs but are not in ARM_ORDER, so "
                     f"they would be dropped from the figure without a message: "
                     f"{unknown}. Add them to ARM_ORDER and ARM_LABEL.")
        if not arms:
            continue
        loaded = {x: series(cells[rpm][x]) for x in arms}
        arms = [x for x in arms if loaded[x] is not None]
        if not arms:
            print(f"{rpm} rpm: no engine metrics")
            continue
        with plt.rc_context(PAPER_STYLE):
            fig, ax = plt.subplots(4, len(arms), figsize=(3.6 * len(arms), 8.4),
                                   sharex=True)
            ax = np.atleast_2d(ax)
            if ax.shape[0] != 4:
                ax = ax.T
            # Shared limits per row, set from the widest arm, so a column that
            # looks calm is calm and not merely rescaled.
            lim = [0.0, 0.0, 105.0, 0.0]
            for x in arms:
                eng, gt, gr = loaded[x]
                for _, (_, rn, wt, _) in eng.items():
                    lim[0] = max(lim[0], np.nanmax(rn))
                    lim[1] = max(lim[1], np.nanmax(wt))
                lim[3] = max(lim[3], np.nanmax(gr) if len(gr) else 0.0)
            for j, x in enumerate(arms):
                eng, gt, gr = loaded[x]
                for i, (name, (t, rn, wt, kv)) in enumerate(sorted(eng.items())):
                    c = ENGINE_COLORS[i % len(ENGINE_COLORS)]
                    ax[0][j].plot(t, rn, color=c, lw=0.8, label=f"engine {name}")
                    ax[1][j].plot(t, wt, color=c, lw=0.8)
                    ax[2][j].plot(t, kv, color=c, lw=0.8)
                ax[3][j].plot(gt, gr, color="#2ca02c", lw=0.8)
                ax[0][j].set_title(ARM_LABEL[x])
                for i, hi in enumerate(lim):
                    ax[i][j].set_ylim(0, hi * 1.05 if hi else 1)
                    ax[i][j].grid(axis="y", ls=":", lw=0.7, alpha=0.6)
                ax[3][j].set_xlabel("time (s)")
            for i, lab in enumerate(["running batch\n(requests)",
                                     "engine queue\n(requests)",
                                     "KV occupancy (%)",
                                     "fleet decode\n(tokens/s)"]):
                ax[i][0].set_ylabel(lab)
            ax[0][0].legend(fontsize=6, loc="upper left")
            where = (f"at {rpm/60:.0f} req/s" if isinstance(rpm, int)
                     else f"on the hour-long dynamic trace ({rpm})")
            fig.suptitle(f"{a.title} engine layer {where} — "
                         f"same four engines, {len(arms)} policies\n"
                         f"rows share a y axis across columns", fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            p = os.path.join(a.out_dir, f"compare_rpm_{rpm}.png")
            fig.savefig(p, dpi=300)
            plt.close(fig)
            print(f"wrote {p}  arms={arms}")


if __name__ == "__main__":
    main()
