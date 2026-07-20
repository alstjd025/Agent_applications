#!/usr/bin/env python3
"""True per-engine (and per-class) SLO attainment vs offered rate.

Unlike `plot_per_engine_slo_signals.py` (which could only show each engine's
*mean* TTFT/ITL because requests could not be attributed to engines), this
computes the real attainment **percentage** per engine, using the
per-request -> engine map produced by `build_request_engine_map.py`
(see experiments/DEV_request-engine-attribution.md).

SLO rule and analysis window are imported from `plot_slo_vs_throughput` so the
definitions are identical to the fleet-level figure:
    violation = TTFT > 5s OR meanTBT > 50ms   (errors/timeouts/run-end-cut
    excluded, admission rejects excluded from the served view)
    window    = arrival-anchored [60s, min(last_arrival, steady_max) - 20s]

`--by engine`  one line per engine port (load-balance / straggler view)
`--by class`   one line per workload class, from the task_id prefix
               (sg-=chat, sa-=deepresearch, else swe) — the per-class fairness
               view for mixed runs (EXP-14).
`--by both`    per-class attainment faceted per engine.

Requires each run to have `analysis/request_engine.csv` (build it first with
build_request_engine_map.py) when `--by engine|both`.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_slo_vs_throughput import (  # noqa: E402
    TTFT_SLO_S, TBT_SLO_MS, STEADY_LO, DRAIN_S, _slo_gw_timeout,
)

PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "axes.linewidth": 0.75, "legend.fontsize": 8, "legend.frameon": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.4, "lines.markersize": 4.5,
}
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def class_of(task_id: str) -> str:
    """Workload class from the task_id prefix (mixed runs)."""
    t = str(task_id)
    if t.startswith("sg-"):
        return "chat"
    if t.startswith("sa-"):
        return "deepresearch"
    return "swe"


def served_rows(run_dir, steady_max_s=360.0):
    """Rows that count toward attainment, with a `violate` flag + window mask.

    Mirrors plot_slo_vs_throughput.condition_stats' classification exactly.
    """
    df = pd.read_csv(os.path.join(run_dir, "metrics.csv"), low_memory=False)
    r = df[df.agent != "job_summary"].copy()
    r = r[pd.to_numeric(r.get("start_time"), errors="coerce").notna()]
    if r.empty:
        return None
    t0 = r["start_time"].min()
    r["rel"] = r["start_time"] - t0
    dur = min(r["end_time"].max() - t0, r["rel"].max())
    if steady_max_s is not None:
        dur = min(dur, steady_max_s)

    bl = lambda c: (r[c].fillna(False).astype(bool)
                    if c in r.columns else pd.Series(False, index=r.index))
    rejected = bl("is_rejected")
    gw = _slo_gw_timeout(r, bl)
    cls = r[(~(bl("is_error") | bl("is_timeout") | bl("is_server_terminated")) | gw)
            & ~rejected].copy()
    tbt = pd.to_numeric(cls["tbt_mean_ms"], errors="coerce")
    cls["violate"] = (
        (pd.to_numeric(cls["first_token_latency"], errors="coerce") > TTFT_SLO_S)
        | (tbt > TBT_SLO_MS) | gw.reindex(cls.index, fill_value=False))
    hi = dur - DRAIN_S
    return cls[(cls["rel"] >= STEADY_LO) & (cls["rel"] < hi)].copy()


def attach_engine(rows, run_dir):
    """Left-join engine_port from analysis/request_engine.csv on task_id."""
    p = os.path.join(run_dir, "analysis", "request_engine.csv")
    if not os.path.isfile(p):
        return None
    m = pd.read_csv(p)
    m = m[["task_id", "engine_port", "migrated"]].drop_duplicates("task_id")
    return rows.merge(m, on="task_id", how="left")


def collect(run_dirs, rate_key, rate_div, steady_max_s, by, drop_migrated):
    out = []
    for d in sorted(run_dirs):
        mm = re.search(rf"{rate_key}(\d+)", os.path.basename(d))
        if not mm:
            continue
        rate = int(mm.group(1)) / rate_div
        rows = served_rows(d, steady_max_s)
        if rows is None or rows.empty:
            continue
        rows["class"] = rows["task_id"].map(class_of)
        if by in ("engine", "both"):
            merged = attach_engine(rows, d)
            if merged is None:
                print(f"  [skip engine view] no request_engine.csv in {d}")
                continue
            rows = merged
            if drop_migrated and "migrated" in rows:
                rows = rows[rows["migrated"].astype(str).str.lower() != "true"]
        out.append((rate, rows))
    out.sort(key=lambda x: x[0])
    return out


def attain(sub):
    return 100.0 * (~sub["violate"]).mean() if len(sub) else np.nan


def plot_lines(data, groups, group_col, out_path, title, rate_unit):
    rates = [r for r, _ in data]
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(5.2, 3.5))
        for g, c in zip(groups, COLORS):
            ys = [attain(rows[rows[group_col] == g]) for _, rows in data]
            ax.plot(rates, ys, "-o", color=c, label=str(g), mec="white", mew=0.5)
        ax.set_xlabel(f"Offered rate ({rate_unit})")
        ax.set_ylabel("SLO attainment (%)")
        ax.set_ylim(-3, 105)
        ax.grid(axis="y", ls=":", lw=0.7, alpha=0.6)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
                  ncol=min(len(groups), 4))
        ax.set_title(title, pad=26)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        print("wrote:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rate-key", default="rpm_")
    ap.add_argument("--rate-div", type=float, default=60.0)
    ap.add_argument("--rate-unit", default="req/s")
    ap.add_argument("--steady-max-s", type=float, default=360.0)
    ap.add_argument("--by", choices=["engine", "class", "both"], default="engine")
    ap.add_argument("--drop-migrated", action="store_true",
                    help="exclude requests flagged as migrated (attribution is "
                         "the INITIAL engine only)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    data = collect(sorted(glob.glob(a.glob)), a.rate_key, a.rate_div,
                   a.steady_max_s, a.by, a.drop_migrated)
    if not data:
        raise SystemExit("no runs matched (or no request_engine.csv)")
    tag = f"_{a.tag}" if a.tag else ""

    if a.by in ("engine", "both"):
        ports = sorted({p for _, rows in data
                        for p in rows["engine_port"].dropna().unique()})
        print(f"{'rate':>6} | " + " | ".join(f"e{int(p)}" for p in ports) + " | fleet")
        for rate, rows in data:
            cells = " | ".join(f"{attain(rows[rows['engine_port'] == p]):5.1f}"
                               for p in ports)
            print(f"{rate:6.1f} | {cells} | {attain(rows):5.1f}")
        plot_lines(data, ports, "engine_port",
                   os.path.join(a.out_dir, f"per_engine_attainment{tag}.png"),
                   "Per-engine SLO attainment", a.rate_unit)

    if a.by in ("class", "both"):
        classes = [c for c in ("chat", "deepresearch", "swe")
                   if any((rows["class"] == c).any() for _, rows in data)]
        print(f"{'rate':>6} | " + " | ".join(f"{c:>12}" for c in classes) + " |  fleet")
        for rate, rows in data:
            cells = " | ".join(f"{attain(rows[rows['class'] == c]):12.1f}"
                               for c in classes)
            print(f"{rate:6.1f} | {cells} | {attain(rows):6.1f}")
        plot_lines(data, classes, "class",
                   os.path.join(a.out_dir, f"per_class_attainment{tag}.png"),
                   "Per-class SLO attainment", a.rate_unit)


if __name__ == "__main__":
    main()
