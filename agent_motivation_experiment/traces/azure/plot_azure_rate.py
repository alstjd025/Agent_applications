#!/usr/bin/env python3
"""Plot hourly request-count curves for Azure LLM Inference traces.

Buckets arrivals by the timestamp string prefix (`YYYY-MM-DD HH`) — no
datetime parsing of all 40M+ rows — and caches per-hour counts next to the
output so re-plotting is instant. Pass one or more `label=path` raw Azure
CSVs (TIMESTAMP,ContextTokens,GeneratedTokens).

Example:
  python plot_azure_rate.py \
    code2024=raw/2024/AzureLLMInferenceTrace_code_2024.csv \
    conv2024=raw/2024/AzureLLMInferenceTrace_conv_2024.csv \
    --out plots/azure_2024_hourly_rate.png
"""

import argparse
import csv
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "axes.linewidth": 0.75, "legend.fontsize": 8, "legend.frameon": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.4,
}
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def hourly_counts(path: str) -> dict:
    """Count requests per 'YYYY-MM-DD HH' bucket via cheap string slicing."""
    counts = {}
    with open(path, newline="") as f:
        next(f, None)  # header
        for line in f:
            key = line[:13]  # 'YYYY-MM-DD HH'
            if len(key) == 13:
                counts[key] = counts.get(key, 0) + 1
    return counts


def load_or_build(path: str, cache: str) -> dict:
    if os.path.exists(cache):
        out = {}
        with open(cache, newline="") as f:
            for row in csv.DictReader(f):
                out[row["hour"]] = int(row["count"])
        return out
    counts = hourly_counts(path)
    with open(cache, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hour", "count"])
        for k in sorted(counts):
            w.writerow([k, counts[k]])
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("series", nargs="+", help="label=path raw Azure CSVs")
    ap.add_argument("--out", required=True, help="output PNG")
    ap.add_argument("--unit", choices=["hour", "day"], default="day",
                    help="x-axis unit (elapsed time from each trace's own start)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(6.0, 2.8))
        for i, spec in enumerate(args.series):
            label, path = spec.split("=", 1)
            cache = os.path.join(os.path.dirname(args.out), f"_hourly_{label}.csv")
            counts = load_or_build(path, cache)
            keys = sorted(counts)
            t0 = datetime.strptime(keys[0], "%Y-%m-%d %H")
            xs, ys = [], []
            for k in keys:
                dt = datetime.strptime(k, "%Y-%m-%d %H")
                elapsed_h = (dt - t0).total_seconds() / 3600.0
                xs.append(elapsed_h / 24.0 if args.unit == "day" else elapsed_h)
                ys.append(counts[k] / 1000.0)  # k requests/hour
            ax.plot(xs, ys, color=COLORS[i % len(COLORS)], label=label)

        ax.set_xlabel("Elapsed time (days)" if args.unit == "day" else "Elapsed time (hours)")
        ax.set_ylabel("Requests / hour (×1000)")
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
                  ncol=len(args.series))
        fig.tight_layout()
        fig.savefig(args.out, dpi=300)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
