#!/usr/bin/env python3
"""x = offered rate; left y = SLO attainment (%); right y = output token throughput.

SLO attainment (same rules as slo_sliding_window.py):
  violation = TTFT>5s (arrival-anchored) OR meanTBT>50ms; errors/timeouts/run-end
  cut requests excluded from the denominator. Two variants plotted:
    - full-run   : all classified arrivals (includes the cold-start herd window)
    - steady     : arrivals in [60s, 280s) only (drops warmup herd + tail window)
Throughput: steady-state output tokens/s (successful requests completing in
  [60s, dur-20s], same definition as the sweep summaries).
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TTFT_SLO_S, TBT_SLO_MS = 5.0, 50.0
STEADY_LO, STEADY_HI = 60.0, 280.0
WARM, DRAIN = 60.0, 20.0

PAPER = {"font.family": "serif", "font.size": 9, "axes.labelsize": 10,
         "axes.titlesize": 10, "legend.fontsize": 8, "legend.frameon": False,
         "lines.linewidth": 1.6, "lines.markersize": 6}


def condition_stats(run_dir):
    df = pd.read_csv(os.path.join(run_dir, "metrics.csv"))
    r = df[df.agent == "request"].copy()
    t0 = r["start_time"].min()
    r["rel"] = r["start_time"] - t0
    bl = lambda c: r[c].fillna(False).astype(bool) if c in r.columns else pd.Series(False, index=r.index)
    cls = r[~(bl("is_error") | bl("is_timeout") | bl("is_server_terminated"))].copy()
    tbt = pd.to_numeric(cls["tbt_mean_ms"], errors="coerce")
    cls["violate"] = (cls["first_token_latency"] > TTFT_SLO_S) | (tbt > TBT_SLO_MS)

    full = 100.0 * (~cls["violate"]).mean() if len(cls) else float("nan")
    sw = cls[(cls["rel"] >= STEADY_LO) & (cls["rel"] < STEADY_HI)]
    steady = 100.0 * (~sw["violate"]).mean() if len(sw) else float("nan")

    ok = r[r["success"].astype(bool)].copy()
    ok["rel_end"] = ok["end_time"] - t0
    dur = r["end_time"].max() - t0
    lo, hi = WARM, dur - DRAIN
    win = ok[(ok["rel_end"] >= lo) & (ok["rel_end"] <= hi)]
    tokps = win["output_tokens"].sum() / max(1e-9, hi - lo)
    return full, steady, tokps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="results/exp04_final_8192conc/*rpm_*")
    ap.add_argument("--out", default="results/aggregate_analysis/exp04_slo/slo_vs_throughput.png")
    args = ap.parse_args()
    dirs = [d for d in sorted(glob.glob(args.glob), key=lambda x: int(x.split("rpm_")[1]))
            if os.path.isdir(d)]
    rows = []
    for d in dirs:
        rpm = int(d.split("rpm_")[1])
        full, steady, tokps = condition_stats(d)
        rows.append((rpm / 60.0, full, steady, tokps))
        print(f"{rpm/60:6.0f} req/s: attain_full={full:5.1f}%  attain_steady={steady:5.1f}%  out_tok/s={tokps:8.0f}")
    x, full, steady, tok = zip(*rows)

    with plt.rc_context(PAPER):
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        ax.plot(x, steady, "o-", color="#1f77b4", label="SLO attainment (steady, arrivals 60-280s)")
        ax.plot(x, full, "s--", color="#1f77b4", alpha=0.5,
                label="SLO attainment (full run, incl. cold-start herd)")
        ax.set_xlabel("Offered rate (req/s)")
        ax.set_ylabel("SLO attainment (%)", color="#1f77b4")
        ax.set_ylim(-3, 105)
        ax.tick_params(axis="y", colors="#1f77b4")
        ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)

        ax2 = ax.twinx()
        ax2.plot(x, tok, "^-", color="#d62728", label="Output token throughput (steady)")
        ax2.set_ylabel("Output tokens/s", color="#d62728")
        ax2.tick_params(axis="y", colors="#d62728")
        ax2.set_ylim(0, max(tok) * 1.15)

        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], loc="center left", fontsize=7.5)
        ax.set_title("SLO attainment vs output throughput — 70B, 8192-conc sweep\n"
                     f"(SLO: TTFT≤{TTFT_SLO_S:.0f}s & meanTBT≤{TBT_SLO_MS:.0f}ms; "
                     "errors/timeouts/run-end-cut excluded)")
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
