#!/usr/bin/env python3
"""SLO attain/violate analysis with a 1-min sliding window (arrival-anchored).

Definitions (EXP-04 final, agreed 2026-07-09):
  - TTFT SLO = 5 s, measured from request ARRIVAL (client send time; the
    `first_token_latency` column is exactly send->first-token, so gateway
    queueing is included).
  - TBT SLO = 50 ms on the per-request MEAN (`tbt_mean_ms`).
  - violation = (TTFT > 5 s) OR (tbt_mean_ms > 50 ms). attain = passes both.
    Requests with no TBT samples (tbt_available False / NaN, e.g. 1-token
    outputs) are judged on TTFT alone.
  - EXCLUDED from classification (reported separately): is_error, is_timeout,
    is_server_terminated (run-boundary cut). Per user decision these are not
    counted as violations.
  - Sliding window: 60 s window, 10 s step, requests assigned by ARRIVAL time
    (start_time - first arrival). Counts are also given as rates (/s).

Outputs:
  <out>/slo_summary.csv                  per-condition totals
  <out>/slo_windows_rpm_<rpm>.csv        per-window counts
  <out>/slo_sliding_grid.png             2x4 grid: attain/s vs violate/s over time
  <out>/slo_summary_bars.png             stacked attain/violate per offered rate
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TTFT_SLO_S = 5.0
TBT_SLO_MS = 50.0
WIN_S = 60.0
STEP_S = 10.0

PAPER = {"font.family": "serif", "font.size": 9, "axes.labelsize": 10,
         "axes.titlesize": 9.5, "legend.fontsize": 8, "legend.frameon": False,
         "lines.linewidth": 1.4}


# The stock gateway aborts any request whose first SSE byte hasn't arrived
# within 300s (forwarder ReadTimeout, compile-time constant) and returns 400;
# the client's non-stream fallback then eats another 300s, so these surface
# as errors at ~600s with output_tokens=0. Such a request *waited out* the
# gateway — it is a TTFT violation in the offered view, not run-boundary
# noise. Latency floor 295s keeps Halo 400-rejects / real errors excluded.
GW_TIMEOUT_MIN_S = 295.0


def gw_timeout_mask(r, boolcol):
    msg = r["error_msg"].fillna("") if "error_msg" in r.columns else pd.Series("", index=r.index)
    lat = pd.to_numeric(r.get("latency"), errors="coerce").fillna(0)
    return (boolcol("is_error") & ~boolcol("is_rejected") & ~boolcol("is_timeout")
            & ~boolcol("is_server_terminated")
            & msg.str.contains("400 Client Error", regex=False)
            & (lat >= GW_TIMEOUT_MIN_S))


def classify(run_dir):
    df = pd.read_csv(os.path.join(run_dir, "metrics.csv"))
    r = df[df.agent != "job_summary"].copy()   # request/chain_call rows
    t0 = r["start_time"].min()
    r["rel"] = r["start_time"] - t0

    boolcol = lambda c: r[c].fillna(False).astype(bool) if c in r.columns else pd.Series(False, index=r.index)
    rejected_mask = boolcol("is_rejected")
    gw_mask = gw_timeout_mask(r, boolcol)
    # admission rejects (is_rejected also sets is_error per the workload
    # invariant) are their own category — counted as violations in the
    # offered-goodput view, never silently excluded as errors. Gateway
    # 300s-timeout kills are reclassified as TTFT violations (see above).
    excluded_mask = ((boolcol("is_error") | boolcol("is_timeout") | boolcol("is_server_terminated"))
                     & ~rejected_mask & ~gw_mask)
    excl_detail = {
        "error": int((boolcol("is_error") & ~rejected_mask & ~gw_mask).sum()),
        "timeout": int(boolcol("is_timeout").sum()),
        "server_terminated": int((boolcol("is_server_terminated") & ~boolcol("is_error") & ~boolcol("is_timeout")).sum()),
    }
    n_rejected = int(rejected_mask.sum())
    cls = r[~excluded_mask & ~rejected_mask].copy()

    ttft_viol = (cls["first_token_latency"] > TTFT_SLO_S) | gw_timeout_mask(cls, (
        lambda c: cls[c].fillna(False).astype(bool) if c in cls.columns else pd.Series(False, index=cls.index)))
    tbt = pd.to_numeric(cls["tbt_mean_ms"], errors="coerce")
    tbt_viol = tbt > TBT_SLO_MS          # NaN -> False (TTFT-only judgement)
    cls["violate"] = ttft_viol | tbt_viol
    detail = {
        "ttft_only": int((ttft_viol & ~tbt_viol).sum()),
        "tbt_only": int((~ttft_viol & tbt_viol).sum()),
        "both": int((ttft_viol & tbt_viol).sum()),
        "gw_timeout": int(gw_mask.sum()),
    }
    return cls, excluded_mask.sum(), excl_detail, detail, r["rel"].max(), n_rejected


def windows(cls, dur):
    rows = []
    t = 0.0
    while t + WIN_S <= dur + STEP_S:
        m = (cls["rel"] >= t) & (cls["rel"] < t + WIN_S)
        w = cls[m]
        rows.append(dict(t_mid=t + WIN_S / 2,
                         attain=int((~w["violate"]).sum()),
                         violate=int(w["violate"].sum())))
        t += STEP_S
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="results/exp04_final_8192conc/*rpm_*")
    ap.add_argument("--out-dir", default="results/aggregate_analysis/exp04_slo")
    ap.add_argument("--rate-key", default="rpm_",
                    help="dirname token preceding the rate value (e.g. 'lambda_')")
    ap.add_argument("--rate-div", type=float, default=60.0,
                    help="divide the parsed value by this to get req/s "
                         "(60 for rpm dirs, 1 for lambda dirs)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dirs = [d for d in sorted(glob.glob(args.glob),
                              key=lambda x: float(x.split(args.rate_key)[1]))
            if os.path.isdir(d)]

    summary, per_win = [], {}
    for d in dirs:
        rpm = float(d.split(args.rate_key)[1]); reqps = rpm / args.rate_div
        cls, n_excl, excl_detail, viol_detail, dur, n_rej = classify(d)
        w = windows(cls, dur)
        per_win[reqps] = w
        w.to_csv(os.path.join(args.out_dir,
                              f"slo_windows_{args.rate_key}{rpm:g}.csv"), index=False)
        n_at, n_vi = int((~cls["violate"]).sum()), int(cls["violate"].sum())
        summary.append(dict(offered_reqps=reqps, classified=len(cls),
                            attain=n_at, violate=n_vi,
                            attain_pct=100 * n_at / max(1, len(cls)),
                            rejected=n_rej,
                            attain_pct_offered=100 * n_at / max(1, len(cls) + n_rej),
                            excluded=int(n_excl), **{f"excl_{k}": v for k, v in excl_detail.items()},
                            **{f"viol_{k}": v for k, v in viol_detail.items()}))
    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(args.out_dir, "slo_summary.csv"), index=False)
    print(sm.to_string(index=False))

    with plt.rc_context(PAPER):
        # grid of sliding-window series
        n = len(per_win)
        ncol = 4; nrow = (n + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow), squeeze=False)
        for i, (reqps_i, w) in enumerate(sorted(per_win.items())):
            ax = axes[i // ncol][i % ncol]
            ax.plot(w["t_mid"], w["attain"] / WIN_S, color="#1f77b4", label="SLO attain (/s)")
            ax.plot(w["t_mid"], w["violate"] / WIN_S, color="#d62728", label="SLO violate (/s)")
            ax.set_title(f"{reqps_i:g} req/s offered")
            ax.set_xlabel("arrival time (s)"); ax.grid(axis="y", ls=":", lw=0.5, alpha=0.5)
            if i % ncol == 0:
                ax.set_ylabel("requests/s")
            if i == 0:
                ax.legend()
        for j in range(n, nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle(f"SLO attain vs violate — 60s sliding window (step 10s), arrival-anchored; "
                     f"TTFT≤{TTFT_SLO_S:.0f}s & meanTBT≤{TBT_SLO_MS:.0f}ms "
                     f"(errors/timeouts/run-end excluded)", y=1.0)
        fig.tight_layout()
        f1 = os.path.join(args.out_dir, "slo_sliding_grid.png")
        fig.savefig(f1, dpi=150, bbox_inches="tight"); plt.close(fig)

        # summary bars
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        x = np.arange(len(sm))
        ax.bar(x, sm["attain"], color="#1f77b4", label="SLO attain")
        ax.bar(x, sm["violate"], bottom=sm["attain"], color="#d62728", label="SLO violate")
        ax.bar(x, sm["excluded"], bottom=sm["attain"] + sm["violate"], color="0.75",
               label="excluded (error/timeout/run-end)")
        for i, row in sm.iterrows():
            ax.text(i, row["attain"] / 2, f"{row['attain_pct']:.0f}%", ha="center",
                    va="center", fontsize=8, color="white")
        ax.set_xticks(x); ax.set_xticklabels([f"{v:.0f}" for v in sm["offered_reqps"]])
        ax.set_xlabel("offered rate (req/s)"); ax.set_ylabel("requests (5-min run)")
        ax.legend(); ax.grid(axis="y", ls=":", lw=0.5, alpha=0.5)
        ax.set_title("SLO attainment per offered rate (label = attain % of classified)")
        fig.tight_layout()
        f2 = os.path.join(args.out_dir, "slo_summary_bars.png")
        fig.savefig(f2, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote:", f1, "\n      ", f2)


if __name__ == "__main__":
    main()
