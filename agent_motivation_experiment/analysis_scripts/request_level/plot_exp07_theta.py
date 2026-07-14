#!/usr/bin/env python3
"""EXP-07: KV-occupancy-threshold admission — per-rate views over theta.

Conditions are results/*exp07_kvadm_th<TTTT>_rpm_<RPM>/ where theta =
TTTT/1000 (0 = admission off). SLO and steady-window definitions follow the
exp04/05 analysis (TTFT<=5s arrival-anchored, meanTBT<=50ms, steady window
[60s, dur-20s]); admission rejects are a separate category counted as
violations in the OFFERED view and excluded from the ADMITTED view.

Outputs (out-dir):
  kvadm_summary.csv
  kvadm_goodput_vs_theta.png   attain_offered / attain_admitted / reject%  per rate
  kvadm_kv_vs_theta.png        steady KV usage mean + p10-p90 band per rate
  kvadm_tbt_vs_theta.png       TBT p50/p99 (admitted, steady) per rate
"""

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TTFT_SLO_S, TBT_SLO_MS = 5.0, 50.0
STEADY_LO, DRAIN_S = 60.0, 20.0
ENGINE_PORTS = (8000, 8001, 8002, 8003)

PAPER = {"font.family": "serif", "font.size": 9, "axes.labelsize": 10,
         "axes.titlesize": 9.5, "legend.fontsize": 8, "legend.frameon": False,
         "lines.linewidth": 1.4, "lines.markersize": 5}
RATE_COLOR = {50: "#2ca02c", 60: "#1f77b4", 90: "#d62728"}


def kv_series(run_dir):
    per = []
    for p in ENGINE_PORTS:
        f = os.path.join(run_dir, "server_metrics", f"engine_{p}.jsonl")
        if not os.path.isfile(f):
            continue
        ts, vs, t0 = [], [], None
        for line in open(f):
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            v = [x for k, x in rec.items()
                 if k.split("|")[0] == "vllm:kv_cache_usage_perc" and isinstance(x, (int, float))]
            if not v:
                continue
            if t0 is None:
                t0 = rec["t"]
            ts.append(rec["t"] - t0); vs.append(sum(v))
        if ts:
            per.append((np.array(ts), np.array(vs)))
    if not per:
        return np.array([]), np.array([])
    n = min(len(v) for _, v in per)
    return per[0][0][:n], np.mean([v[:n] for _, v in per], axis=0) * 100.0


def condition_stats(run_dir):
    df = pd.read_csv(os.path.join(run_dir, "metrics.csv"), low_memory=False)
    r = df[df.agent != "job_summary"].copy()
    r = r[pd.to_numeric(r.get("start_time"), errors="coerce").notna()]
    t0 = r["start_time"].min()
    r["rel"] = r["start_time"] - t0
    dur = r["end_time"].max() - t0
    hi = dur - DRAIN_S

    bl = lambda c: r[c].fillna(False).astype(bool) if c in r.columns else pd.Series(False, index=r.index)
    rej = bl("is_rejected")
    excl = (bl("is_error") | bl("is_timeout") | bl("is_server_terminated")) & ~rej
    cls = r[~excl & ~rej].copy()
    tbt = pd.to_numeric(cls["tbt_mean_ms"], errors="coerce")
    cls["violate"] = (pd.to_numeric(cls["first_token_latency"], errors="coerce") > TTFT_SLO_S) | (tbt > TBT_SLO_MS)

    sw = cls[(cls["rel"] >= STEADY_LO) & (cls["rel"] < hi)]
    rj = r[rej & (r["rel"] >= STEADY_LO) & (r["rel"] < hi)]
    n_att = int((~sw["violate"]).sum())
    n_off = len(sw) + len(rj)
    swt = pd.to_numeric(sw["tbt_mean_ms"], errors="coerce").dropna()

    ok = r[r["success"].astype(bool)].copy()
    ok["rel_end"] = ok["end_time"] - t0
    win = ok[(ok["rel_end"] >= STEADY_LO) & (ok["rel_end"] <= hi)]
    tokps = pd.to_numeric(win["output_tokens"], errors="coerce").sum() / max(1e-9, hi - STEADY_LO)

    kt, kv = kv_series(run_dir)
    m = (kt >= STEADY_LO) & (kt <= (kt.max() if len(kt) else 0) - DRAIN_S)
    kvs = kv[m] if m.any() else np.array([np.nan])

    return dict(
        classified=len(sw), rejected=len(rj), attain=n_att,
        attain_admitted=100.0 * n_att / max(1, len(sw)),
        attain_offered=100.0 * n_att / max(1, n_off),
        reject_pct=100.0 * len(rj) / max(1, n_off),
        tokps=tokps,
        tbt_p50=float(swt.quantile(0.5)) if len(swt) else np.nan,
        tbt_p99=float(swt.quantile(0.99)) if len(swt) else np.nan,
        kv_mean=float(np.nanmean(kvs)), kv_p10=float(np.nanpercentile(kvs, 10)),
        kv_p90=float(np.nanpercentile(kvs, 90)),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="results/*exp07_kvadm_th*_rpm_*")
    ap.add_argument("--out-dir", default="results/aggregate_analysis/exp07")
    ap.add_argument("--exclude-rpm", default="300", help="comma list of rpm to skip (smoke)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    skip = {int(x) for x in args.exclude_rpm.split(",") if x}

    rows = []
    for d in sorted(glob.glob(args.glob)):
        m = re.search(r"th(\d{4})_rpm_(\d+)$", d)
        if not m or int(m.group(2)) in skip:
            continue
        theta, rpm = int(m.group(1)) / 1000.0, int(m.group(2))
        s = condition_stats(d)
        s.update(rate=rpm / 60.0, theta=theta, run=os.path.basename(d))
        rows.append(s)
        print(f"{s['rate']:3.0f} req/s th={theta if theta else 'off':>5}: "
              f"offered={s['attain_offered']:5.1f}% admitted={s['attain_admitted']:5.1f}% "
              f"rej={s['reject_pct']:5.1f}% tok/s={s['tokps']:6.0f} KVμ={s['kv_mean']:5.1f}% "
              f"TBTp50={s['tbt_p50']:5.1f} p99={s['tbt_p99']:6.1f}")
    t = pd.DataFrame(rows).sort_values(["rate", "theta"])
    t.to_csv(os.path.join(args.out_dir, "kvadm_summary.csv"), index=False)

    # theta axis: real thetas ascending, 'off'(0) plotted at the right edge
    def axis_pos(th):
        return 0.95 if th == 0 else th

    def per_rate(ax, col, fmt, label_fn=None, band=None):
        for rate, g in t.groupby("rate"):
            g = g.copy()
            g["x"] = g["theta"].map(axis_pos)
            g = g.sort_values("x")
            c = RATE_COLOR.get(int(rate), "0.4")
            ax.plot(g["x"], g[col], fmt, color=c,
                    label=(label_fn(rate) if label_fn else f"{rate:.0f} req/s"))
            if band:
                ax.fill_between(g["x"], g[band[0]], g[band[1]], color=c, alpha=0.15)
        xs = sorted({axis_pos(th) for th in t["theta"]})
        ax.set_xticks(xs, [("off" if x == 0.95 else f"{x:g}") for x in xs])
        ax.set_xlabel("admission threshold θ (hot KV usage ratio)")
        ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)

    with plt.rc_context(PAPER):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
        per_rate(axes[0], "attain_offered", "o-")
        axes[0].set_ylabel("SLO attainment, offered (%)"); axes[0].set_ylim(-3, 105)
        axes[0].set_title("Offered goodput (rejects = violations)"); axes[0].legend()
        per_rate(axes[1], "attain_admitted", "s-")
        axes[1].set_ylabel("SLO attainment, admitted (%)"); axes[1].set_ylim(-3, 105)
        axes[1].set_title("Admitted-only attainment")
        per_rate(axes[2], "reject_pct", "x--")
        axes[2].set_ylabel("Rejection rate (%)"); axes[2].set_ylim(-3, 105)
        axes[2].set_title("Rejection rate")
        fig.suptitle("EXP-07 — KV-occupancy threshold admission (chat, steady window)", y=1.02)
        fig.tight_layout()
        out = os.path.join(args.out_dir, "kvadm_goodput_vs_theta.png")
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("wrote:", out)

        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        per_rate(ax, "kv_mean", "o-", band=("kv_p10", "kv_p90"))
        ax.set_ylabel("hot KV usage (%)"); ax.set_ylim(0, 105); ax.legend()
        ax.set_title("Steady KV usage vs θ (band = p10–p90 over time)")
        fig.tight_layout()
        out = os.path.join(args.out_dir, "kvadm_kv_vs_theta.png")
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("wrote:", out)

        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        per_rate(ax, "tbt_p99", "^-")
        per_rate(ax, "tbt_p50", "v--", label_fn=lambda r: None)
        ax.axhline(TBT_SLO_MS, color="0.4", ls=":", label=f"TBT SLO {TBT_SLO_MS:.0f}ms")
        ax.set_ylabel("mean-TBT per request (ms)"); ax.legend()
        ax.set_title("Admitted TBT p99 (solid) / p50 (dashed) vs θ")
        fig.tight_layout()
        out = os.path.join(args.out_dir, "kvadm_tbt_vs_theta.png")
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("wrote:", out)


if __name__ == "__main__":
    main()
