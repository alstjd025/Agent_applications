#!/usr/bin/env python3
"""EXP-27 figures: the rate sweep, what each engine was doing, and the latencies.

Three figures, each answering one question that the others cannot.

  sweep    attainment and token goodput against offered rate, one panel per mix.
           Two y axes on purpose: a policy can raise attainment by refusing work,
           and goodput is the quantity that does not let it. Putting them on one
           panel means no reading of the attainment curve is possible without the
           throughput curve in view.
  engines  per-engine decode batch, queue depth and KV occupancy over time. The
           request-level numbers cannot show a fleet that is idle in three places
           and saturated in the fourth, which is exactly the failure the static
           partition produces.
  latency  TTFT and inter-token latency distributions per class, against the
           budgets they are scored on. This is what separates "the engine is too
           slow" from "the request waited": if ITL is inside budget and TTFT is
           not, nothing was overloaded, the request was queued.

    python3 analysis_scripts/request_level/exp27_figures.py \
        --pass1 'results/*exp27r1_*' --pass2 'results/*exp27p2*' \
        --out-dir results/aggregate_analysis/exp27
"""
import argparse
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp22_fluidserve import (  # noqa: E402
    CLASSES, CLASS_COLORS, PAPER_STYLE, load_run, per_request, goodput_tokens,
    arm_of,
)

ARM_C = {"polyserve": "#d62728", "fluidserve": "#1f77b4",
         "fluidserveflat": "#2ca02c"}
ARM_L = {"polyserve": "PolyServe", "fluidserve": "FluidServe",
         "fluidserveflat": "FluidServe (v20 off)"}
MIX_TITLE = {
    "m1": "m1 balanced\n31/37/31% of input tokens",
    "m2": "m2 chat-heavy\n64/19/16%",
    "m3": "m3 agent-heavy\n19/19/63%",
}
# The rules attainment is scored against, drawn as reference lines.
TTFT_BUDGET = {"chat": 5.0, "deepresearch": 10.0, "swe": None}
ITL_BUDGET = {"chat": 50.0, "deepresearch": 100.0, "swe": None}


def rpm_of(d):
    m = re.search(r"_rpm_(\d+)", os.path.basename(d))
    return int(m.group(1)) if m else None


def mix_of(d):
    m = re.search(r"_(m[123])_rpm_", os.path.basename(d))
    return m.group(1) if m else None


def collect(patterns):
    rows = []
    for p in patterns:
        for d in sorted(glob.glob(p)):
            r = load_run(d)
            if r is None or r.empty:
                continue
            w = r["rel"].max() - r["rel"].min()
            if w <= 0:
                continue
            rows.append({
                "dir": d, "arm": arm_of(d), "mix": mix_of(d), "rpm": rpm_of(d),
                "attain": per_request(r, "violate_served"),
                "attain_off": per_request(r, "violate_offered"),
                "goodput": goodput_tokens(r, w),
                "rejected": 100.0 * r["rejected"].mean(),
            })
    return pd.DataFrame([r for r in rows if r["rpm"] and r["mix"]])


# ---------------------------------------------------------------- sweep figure
def fig_sweep(df, out, title):
    mixes = [m for m in ("m1", "m2", "m3") if m in set(df["mix"])]
    arms = [a for a in ARM_C if a in set(df["arm"])]
    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(1, len(mixes), figsize=(3.6 * len(mixes), 3.4),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        for ax, mix in zip(axes, mixes):
            ax2 = ax.twinx()
            for a in arms:
                # Column names avoid `at`, `iat`, `loc`: those are DataFrame
                # indexer attributes, so `d.at` silently returns the indexer
                # rather than the column and the arithmetic fails far from here.
                d = df[(df.arm == a) & (df.mix == mix)].groupby("rpm").agg(
                    sloA=("attain", "mean"), sloLo=("attain", "min"),
                    sloHi=("attain", "max"), gp=("goodput", "mean"),
                    gpLo=("goodput", "min"), gpHi=("goodput", "max"),
                ).reset_index().sort_values("rpm")
                if d.empty:
                    continue
                c = ARM_C[a]
                ax.errorbar(d.rpm, d.sloA,
                            yerr=[d.sloA - d.sloLo, d.sloHi - d.sloA],
                            color=c, ls="-", marker="o", capsize=2)
                ax2.errorbar(d.rpm, d.gp,
                             yerr=[d.gp - d.gpLo, d.gpHi - d.gp],
                             color=c, ls="--", marker="s", ms=3.5, alpha=0.75,
                             capsize=2)
            ax.set_title(MIX_TITLE.get(mix, mix))
            ax.set_xlabel("offered rate (rpm)")
            ax.set_xticks(sorted(df.rpm.unique()))
            ax.set_ylim(0, 105)
            ax2.set_ylim(0, 19000)
            ax.grid(axis="y", ls=":", lw=0.7, alpha=0.6)
            if ax is axes[0]:
                ax.set_ylabel("SLO attainment (%), admitted, per request")
            if ax is axes[-1]:
                ax2.set_ylabel("goodput (output tokens/s)")
            else:
                ax2.set_yticklabels([])
        handles, labels = [], []
        for a in arms:
            handles.append(plt.Line2D([], [], color=ARM_C[a], ls="-", marker="o"))
            labels.append(f"{ARM_L[a]} — attainment (left)")
            handles.append(plt.Line2D([], [], color=ARM_C[a], ls="--", marker="s",
                                      ms=3.5, alpha=0.75))
            labels.append(f"{ARM_L[a]} — goodput (right)")
        fig.legend(handles, labels, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, 1.0), frameon=False)
        fig.suptitle(title, y=1.16, fontsize=9)
        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------- engine figure
E_RUN = "vllm:num_requests_running"
E_WAIT = "vllm:num_requests_waiting"
E_KV = "vllm:kv_cache_usage_perc"


def engine_series(run):
    out = {}
    for path in sorted(glob.glob(os.path.join(run, "server_metrics", "engine_*.jsonl"))):
        t, run_, wait, kv = [], [], [], []
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
                return 0.0
            t.append(rec["t"]); run_.append(pick(E_RUN))
            wait.append(pick(E_WAIT)); kv.append(pick(E_KV) * 100)
        if t:
            name = os.path.basename(path).replace("engine_", "").replace(".jsonl", "")
            t0 = t[0]
            out[name] = (np.array(t) - t0, np.array(run_), np.array(wait), np.array(kv))
    return out


def fig_engines(runs, out, title):
    """runs: list of (label, run_dir). One column per run, three rows."""
    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(3, len(runs), figsize=(4.2 * len(runs), 6.2),
                                 sharex=True)
        axes = np.atleast_2d(axes)
        if axes.shape[0] != 3:
            axes = axes.T
        cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for j, (label, run) in enumerate(runs):
            eng = engine_series(run)
            for i, (name, ser) in enumerate(sorted(eng.items())):
                t, r, w, kv = ser
                c = cols[i % len(cols)]
                axes[0][j].plot(t / 60, r, color=c, lw=1.0, label=f"engine {name}")
                axes[1][j].plot(t / 60, w, color=c, lw=1.0)
                axes[2][j].plot(t / 60, kv, color=c, lw=1.0)
            axes[0][j].set_title(label)
            axes[2][j].set_xlabel("time in condition (min)")
            for ax in axes[:, j]:
                ax.grid(axis="y", ls=":", lw=0.7, alpha=0.6)
        axes[0][0].set_ylabel("requests decoding")
        axes[1][0].set_ylabel("requests queued")
        axes[2][0].set_ylabel("KV pool used (%)")
        axes[2][0].set_ylim(0, 105)
        # A shared y range per row makes the columns comparable at a glance;
        # without it the idle arm is autoscaled up and looks as busy as the
        # saturated one.
        for row in range(3):
            lo = min(ax.get_ylim()[0] for ax in axes[row])
            hi = max(ax.get_ylim()[1] for ax in axes[row])
            for ax in axes[row]:
                ax.set_ylim(lo, hi)
        axes[0][0].legend(fontsize=6, ncol=2, loc="upper left")
        fig.suptitle(title, y=1.0, fontsize=9)
        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {out}")


# -------------------------------------------------------------- latency figure
def fig_latency(cells, out, title):
    """cells: list of (row_label, {arm: run_dir}). TTFT and ITL CDFs per class."""
    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(len(cells), 2, figsize=(7.6, 3.0 * len(cells)))
        axes = np.atleast_2d(axes)
        for i, (row_label, arms) in enumerate(cells):
            for arm, run in arms.items():
                r = load_run(run)
                if r is None:
                    continue
                # Admitted requests only: a rejected one has no latency to plot,
                # and counting it as infinite would hide the shape of what was
                # actually served. The rejection rate is in the sweep figure.
                r = r[(~r["cutoff"]) & (~r["rejected"])]
                ls = "-" if arm == "fluidserve" else "--"
                for c in CLASSES:
                    s = r[r["class"] == c]
                    for j, (col, budget) in enumerate(
                            [("first_token_latency", TTFT_BUDGET[c]),
                             ("tbt_mean_ms", ITL_BUDGET[c])]):
                        v = pd.to_numeric(s[col], errors="coerce").dropna()
                        if v.empty:
                            continue
                        v = np.sort(v.values)
                        y = np.arange(1, len(v) + 1) / len(v) * 100
                        axes[i][j].plot(v, y, color=CLASS_COLORS[c], ls=ls, lw=1.2)
                        if budget:
                            axes[i][j].axvline(budget, color=CLASS_COLORS[c],
                                               lw=0.6, alpha=0.45)
            axes[i][0].set_xscale("log")
            axes[i][0].set_xlabel("time to first token (s)")
            axes[i][1].set_xlabel("mean time between tokens (ms)")
            axes[i][1].set_xlim(0, 120)
            for ax in axes[i]:
                ax.set_ylabel(f"{row_label}\npercentile")
                ax.set_ylim(0, 100)
                ax.grid(ls=":", lw=0.7, alpha=0.6)
        handles = [plt.Line2D([], [], color=CLASS_COLORS[c], lw=1.2) for c in CLASSES]
        labels = list(CLASSES)
        handles += [plt.Line2D([], [], color="#666666", ls="-", lw=1.2),
                    plt.Line2D([], [], color="#666666", ls="--", lw=1.2),
                    plt.Line2D([], [], color="#666666", lw=0.6, alpha=0.45)]
        labels += ["FluidServe", "PolyServe", "SLO budget"]
        fig.legend(handles, labels, loc="lower center", ncol=6,
                   bbox_to_anchor=(0.5, 1.0), frameon=False)
        fig.suptitle(title, y=1.06, fontsize=9)
        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {out}")


def find(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass1", nargs="+", default=["results/*exp27r1_*"])
    ap.add_argument("--pass2", nargs="+", default=["results/*exp27p2*"])
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    d1 = collect(a.pass1)
    fig_sweep(d1, os.path.join(a.out_dir, "exp27_sweep_pass1.png"),
              "EXP-27 pass 1 (v19+v20, one run per condition): "
              "three mixes, four engines, 8 min per condition")

    d2 = collect(a.pass2)
    if not d2.empty:
        fig_sweep(d2[d2.arm != "fluidserveflat"],
                  os.path.join(a.out_dir, "exp27_sweep_pass2_m1.png"),
                  "EXP-27 pass 2 (v21, two repeats, bars = min..max): m1 balanced")

    p = {arm: find(f"results/*exp27r1_{arm}_m1_rpm_2400") for arm in
         ("polyserve", "fluidserve")}
    q = {arm: find(f"results/*exp27r1_{arm}_m1_rpm_4800") for arm in
         ("polyserve", "fluidserve")}
    if all(p.values()):
        fig_engines([("PolyServe, m1, 2400 rpm", p["polyserve"]),
                     ("FluidServe, m1, 2400 rpm", p["fluidserve"])],
                    os.path.join(a.out_dir, "exp27_engines_m1_2400.png"),
                    "What each of the four engines was doing (m1, 2400 rpm)")
    if all(p.values()) and all(q.values()):
        fig_latency([("m1, 2400 rpm", p), ("m1, 4800 rpm", q)],
                    os.path.join(a.out_dir, "exp27_latency_m1.png"),
                    "Latency of ADMITTED requests against the budgets they are "
                    "scored on")
    return 0


if __name__ == "__main__":
    sys.exit(main())
