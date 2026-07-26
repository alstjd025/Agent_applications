#!/usr/bin/env python3
"""Build a 1-hour dynamic trace: Azure-shaped arrival rate + time-varying class mix.

Produces one canonical arrival trace (see ../TRACE_FORMAT.md) in which BOTH
knobs move over the hour:

  * **offered rate** follows the real Azure LLM Inference 2024 diurnal shape,
    time-compressed so N source days land in one hour (N peaks), and
    quantile-mapped onto our cluster's usable band (default 10-50 req/s);
  * **class mix** (chat : deepresearch : swe) steps through a schedule of the
    EXP-14 ratios A/B/C, with segment boundaries deliberately NOT aligned to
    the rate peaks so the rate effect and the mix effect stay separable.

Why quantile mapping and not a plain rescale: Azure conv+code has a p95/p5
dynamic range of ~3.6x, but the band we want to sweep is 5x (10->50 req/s).
A linear stretch would either clip the band or exaggerate the noise floor.
The rank transform keeps the temporal ORDER and autocorrelation of the real
trace -- when Azure is busy, we are busy -- while letting us choose the band
our fleet actually resolves. Time spent at each rate becomes uniform over the
band, which also makes the rate-binned analysis (recover an attainment-vs-rate
curve from a single run) well conditioned. What is NOT preserved is the shape
of the rate *distribution*; say "Azure-shaped, rescaled to our cluster", not
"an Azure trace".

Source alignment caveat: the conv and code 2024 traces cover different calendar
weeks (conv May 12-18, code May 10-16). They are summed by index, so hour-of-day
phase lines up (both start at 00:00 UTC) but the calendar dates do not. That is
fine for a shape donor and is recorded in the plan JSON.

Output (three files sharing a stem):
  <out>.csv        canonical trace. `arrival_s` is what the runner reads;
                   `class` is what the mixed workload reads (extra columns are
                   explicitly ignored by arrival_trace.load_arrival_trace).
  <out>.plan.json  ground truth: lambda(t), segment boundaries, realised ratios.
  <out>.png        the curve, for eyeballing before spending an hour of cluster.

INVARIANT (row order == class order): the runner sorts `arrival_s` and drops
every other column, then the workload reads `class` from this same file in file
order and pairs them by index. So the file MUST be strictly ascending in
`arrival_s`; the generator enforces that by nudging exact ties.

Example:
  python3 traces/dynamic/build_dynamic_mix_trace.py \
      --out traces/dynamic/canonical/dyn60_azure4d
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

# build_class_sequence lives with the workload; reuse it verbatim so the
# realised per-segment ratio is produced by the exact same code path the
# static-mix experiments used.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from workloads.mixed_request_level_poisson.mixplan import (  # noqa: E402
    build_class_sequence,
)

# EXP-14 ratios (chat : deepresearch : swe), by REQUEST count.
MIXES = {
    "A": {"chat": 1, "deepresearch": 1, "swe": 1},   # 33/33/33%  - balanced
    "B": {"chat": 6, "deepresearch": 3, "swe": 1},   # 60/30/10%  - light-heavy
    "C": {"chat": 1, "deepresearch": 1, "swe": 3},   # 20/20/60%  - heavy-heavy
}

DEF_CONV = "traces/azure/plots/_minute_conv2024.csv"
DEF_CODE = "traces/azure/plots/_minute_code2024.csv"


def load_minutes(path: str) -> np.ndarray:
    with open(path, newline="") as f:
        return np.array([int(r["count"]) for r in csv.DictReader(f)], dtype=float)


def azure_shape(conv_path: str, code_path: str, day0: int, ndays: int,
                n_bins: int, start_hour: int = 6) -> np.ndarray:
    """Return the source rate shape resampled onto `n_bins` equal bins.

    The window starts at `start_hour` (default 06:00, the observed daily
    minimum of the combined series) rather than midnight, so the compressed
    hour BEGINS and ENDS at a diurnal trough. Starting at midnight puts the
    last daily peak ~3 min before the end of the run, which would leave the
    fleet draining a backlog exactly where the measurement stops.

    Sums conv+code per source minute over the window, then averages the
    source minutes falling into each output bin. With the default 4 days -> 3600
    bins each bin averages 1.6 Azure minutes, so Azure's minute-level jitter is
    aggregated away -- deliberately: after 96x compression it would land below
    one second, far under any timescale the fleet responds to, and the Poisson
    draw already supplies sub-second randomness. What survives the compression
    is the diurnal and multi-hour structure, which is the point.
    """
    conv, code = load_minutes(conv_path), load_minutes(code_path)
    n = min(len(conv), len(code))
    total = conv[:n] + code[:n]
    lo = day0 * 1440 + start_hour * 60
    hi = lo + ndays * 1440
    if hi > n:
        raise ValueError(f"window day {day0}+{start_hour}h .. +{ndays}d exceeds "
                         f"the {n/1440:.1f} days available")
    win = total[lo:hi]
    edges = np.linspace(0, len(win), n_bins + 1)
    return np.array([win[int(edges[i]):max(int(edges[i]) + 1, int(edges[i + 1]))].mean()
                     for i in range(n_bins)])


def quantile_map(shape: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Rank-transform `shape` onto [lo, hi] (ties get their average rank)."""
    order = np.argsort(shape, kind="stable")
    ranks = np.empty(len(shape), dtype=float)
    ranks[order] = np.arange(len(shape), dtype=float)
    # average rank within tied groups, so flat stretches stay flat
    uniq, inv = np.unique(shape, return_inverse=True)
    for g in range(len(uniq)):
        m = inv == g
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return lo + (hi - lo) * ranks / (len(shape) - 1)


def minmax_map(shape: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linear stretch (kept for the comparison panel in the figure)."""
    s0, s1 = shape.min(), shape.max()
    return lo + (hi - lo) * (shape - s0) / (s1 - s0)


def poisson_arrivals(lam_per_bin: np.ndarray, bin_s: float, t0: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Non-homogeneous Poisson arrivals: per-bin count, then uniform within."""
    counts = rng.poisson(lam_per_bin * bin_s)
    out = []
    for i, c in enumerate(counts):
        if c:
            out.append(t0 + bin_s * (i + rng.random(c)))
    return np.sort(np.concatenate(out)) if out else np.array([])


def strictly_increasing(t: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Nudge ties so file order and sorted order can never disagree."""
    out = t.copy()
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + eps
    return out


def segment_of(t: float, bounds: list) -> int:
    for i, (lo, hi, _) in enumerate(bounds):
        if lo <= t < hi:
            return i
    return len(bounds) - 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conv-minutes", default=DEF_CONV)
    ap.add_argument("--code-minutes", default=DEF_CODE)
    ap.add_argument("--source-day0", type=int, default=0,
                    help="first source day of the donor window")
    ap.add_argument("--source-days", type=int, default=4,
                    help="donor days compressed into the run == number of peaks")
    ap.add_argument("--source-start-hour", type=int, default=6,
                    help="hour-of-day the donor window starts at; 6 is the "
                         "combined series' daily minimum, so the run both "
                         "starts and ends at a trough")
    ap.add_argument("--duration-min", type=float, default=60.0)
    ap.add_argument("--bin-sec", type=float, default=1.0,
                    help="resolution of the piecewise-constant rate function")
    ap.add_argument("--rate-min", type=float, default=10.0, help="req/s at trough")
    ap.add_argument("--rate-max", type=float, default=50.0, help="req/s at peak")
    ap.add_argument("--mix-schedule", default="A,C,B,A",
                    help="comma-separated mix names, one per equal-length segment")
    ap.add_argument("--warmup-sec", type=float, default=60.0,
                    help="lead-in at --rate-min so the fleet is not cold at t=0; "
                         "tagged phase=warmup and excluded from analysis")
    ap.add_argument("--warmup-mix", default="A")
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--out", required=True, help="output stem (no extension)")
    args = ap.parse_args()

    names = [s.strip() for s in args.mix_schedule.split(",") if s.strip()]
    unknown = set(names) | {args.warmup_mix}
    unknown -= set(MIXES)
    if unknown:
        raise SystemExit(f"unknown mix name(s): {sorted(unknown)}; have {sorted(MIXES)}")

    rng = np.random.default_rng(args.seed)
    dur_s = args.duration_min * 60.0
    n_bins = int(round(dur_s / args.bin_sec))

    shape = azure_shape(args.conv_minutes, args.code_minutes,
                        args.source_day0, args.source_days, n_bins,
                        args.source_start_hour)
    lam = quantile_map(shape, args.rate_min, args.rate_max)

    # ---- arrivals: warmup lead-in, then the measured hour -------------------
    warm_t = np.array([])
    if args.warmup_sec > 0:
        n_warm = int(round(args.warmup_sec / args.bin_sec))
        warm_t = poisson_arrivals(np.full(n_warm, args.rate_min),
                                  args.bin_sec, 0.0, rng)
    main_t = poisson_arrivals(lam, args.bin_sec, args.warmup_sec, rng)
    t = strictly_increasing(np.concatenate([warm_t, main_t]))

    # ---- class plan ---------------------------------------------------------
    seg_len = dur_s / len(names)
    bounds = [(args.warmup_sec + i * seg_len, args.warmup_sec + (i + 1) * seg_len,
               names[i]) for i in range(len(names))]

    classes = [""] * len(t)
    phases = [""] * len(t)
    segs = [""] * len(t)
    n_warm_arr = len(warm_t)
    # Per segment: hand its own arrival count to build_class_sequence so the
    # realised ratio is exact within each segment (the same block-shuffle the
    # static-mix runs used), rather than approximately right on average.
    groups = {}
    for i in range(len(t)):
        if i < n_warm_arr:
            groups.setdefault(-1, []).append(i)
        else:
            groups.setdefault(segment_of(t[i], bounds), []).append(i)
    for gi, idxs in groups.items():
        name = args.warmup_mix if gi < 0 else bounds[gi][2]
        seq = build_class_sequence(MIXES[name], len(idxs), args.seed + 1000 + gi)
        for j, i in enumerate(idxs):
            classes[i] = seq[j]
            phases[i] = "warmup" if gi < 0 else "measure"
            segs[i] = f"warmup_{name}" if gi < 0 else f"s{gi}_{name}"

    # ---- write --------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    csv_path, json_path, png_path = (f"{args.out}.csv", f"{args.out}.plan.json",
                                     f"{args.out}.png")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arrival_s", "class", "phase", "segment"])
        for i in range(len(t)):
            w.writerow([f"{t[i]:.4f}", classes[i], phases[i], segs[i]])

    realised = {}
    for gi, idxs in sorted(groups.items()):
        key = "warmup" if gi < 0 else f"s{gi}_{bounds[gi][2]}"
        cnt = {}
        for i in idxs:
            cnt[classes[i]] = cnt.get(classes[i], 0) + 1
        n = len(idxs) or 1
        realised[key] = {
            "target_mix": MIXES[args.warmup_mix if gi < 0 else bounds[gi][2]],
            "arrivals": len(idxs),
            "realised_ratio": {k: round(v / n, 4) for k, v in sorted(cnt.items())},
            "t_range_s": ([0.0, args.warmup_sec] if gi < 0
                          else [round(bounds[gi][0], 1), round(bounds[gi][1], 1)]),
        }

    plan = {
        "generator": os.path.basename(__file__),
        "seed": args.seed,
        "source": {
            "conv_minutes": args.conv_minutes, "code_minutes": args.code_minutes,
            "day_window": [args.source_day0, args.source_day0 + args.source_days],
            "start_hour": args.source_start_hour,
            "compression_x": round(args.source_days * 1440 * 60.0 / dur_s, 1),
            "caveat": "conv and code cover different calendar weeks; summed by "
                      "index so hour-of-day phase aligns, calendar date does not",
        },
        "rate": {
            "mapping": "quantile (rank) transform onto [rate_min, rate_max]",
            "rate_min": args.rate_min, "rate_max": args.rate_max,
            "bin_sec": args.bin_sec,
            "warmup_sec": args.warmup_sec, "warmup_rate": args.rate_min,
            "lambda_series": [round(float(x), 3) for x in lam],
        },
        "mix": {"schedule": names, "segment_sec": seg_len,
                "warmup_mix": args.warmup_mix, "definitions": MIXES},
        "totals": {
            "arrivals": len(t), "warmup_arrivals": int(n_warm_arr),
            "measured_arrivals": int(len(t) - n_warm_arr),
            "span_s": round(float(t[-1]), 1),
            "mean_measured_rate": round(float(len(t) - n_warm_arr) / dur_s, 2),
        },
        "segments": realised,
    }
    with open(json_path, "w") as f:
        json.dump(plan, f, indent=2)

    _plot(png_path, lam, shape, args, bounds, t, n_warm_arr, classes)

    print(f"[dyn] {csv_path}: {len(t)} arrivals "
          f"({n_warm_arr} warmup + {len(t)-n_warm_arr} measured), "
          f"span {t[-1]/60:.1f} min, mean {plan['totals']['mean_measured_rate']} req/s")
    for k, v in realised.items():
        print(f"[dyn]   {k:14s} {v['arrivals']:6d} arrivals  {v['realised_ratio']}")
    print(f"[dyn] plan -> {json_path}\n[dyn] figure -> {png_path}")


def _plot(path, lam, shape, args, bounds, t, n_warm, classes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"font.family": "serif", "font.size": 8, "axes.labelsize": 9,
             "axes.titlesize": 9, "legend.fontsize": 7.5, "legend.frameon": False,
             "xtick.direction": "in", "ytick.direction": "in", "lines.linewidth": 1.1}
    seg_color = {"A": "#1f77b4", "B": "#2ca02c", "C": "#d62728"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=False)

        ax = axes[0]
        tt = (args.warmup_sec + np.arange(len(lam)) * args.bin_sec) / 60.0
        ax.plot(tt, lam, color="#1f77b4", label="target rate (quantile map)")
        ax.plot(tt, minmax_map(shape, args.rate_min, args.rate_max), color="#999999",
                lw=0.8, ls="--", label="linear stretch (not used, for comparison)")
        for lo, hi, name in bounds:
            ax.axvspan(lo / 60, hi / 60, color=seg_color[name], alpha=0.07)
            ax.text((lo + hi) / 120, args.rate_max * 1.03, f"mix {name}",
                    ha="center", va="bottom", fontsize=8, color=seg_color[name])
        ax.axvspan(0, args.warmup_sec / 60, color="#888888", alpha=0.18)
        ax.set_ylabel("offered rate (req/s)")
        ax.set_xlim(0, (args.warmup_sec + args.duration_min * 60) / 60)
        ax.set_ylim(0, args.rate_max * 1.18)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2)
        ax.grid(axis="y", ls=":", lw=0.7, alpha=0.6)

        ax = axes[1]
        edges = np.arange(0, t[-1] + 10, 10.0)
        ax.plot(edges[:-1] / 60, np.histogram(t, bins=edges)[0] / 10.0,
                color="#d62728", label="realised arrivals (10 s bins)")
        ax.plot(tt, lam, color="#1f77b4", lw=0.8, alpha=0.7, label="target")
        ax.set_ylabel("req/s")
        ax.set_xlim(0, (args.warmup_sec + args.duration_min * 60) / 60)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
        ax.grid(axis="y", ls=":", lw=0.7, alpha=0.6)

        ax = axes[2]
        cls_arr = np.array(classes)
        bottom = np.zeros(len(edges) - 1)
        for name, col in [("chat", "#1f77b4"), ("deepresearch", "#2ca02c"),
                          ("swe", "#d62728")]:
            h = np.histogram(t[cls_arr == name], bins=edges)[0].astype(float)
            tot = np.histogram(t, bins=edges)[0].astype(float)
            frac = np.divide(h, tot, out=np.zeros_like(h), where=tot > 0)
            ax.fill_between(edges[:-1] / 60, bottom, bottom + frac, color=col,
                            alpha=0.85, step="post", label=name)
            bottom += frac
        ax.set_ylabel("request share")
        ax.set_xlabel("time (min)")
        ax.set_xlim(0, (args.warmup_sec + args.duration_min * 60) / 60)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)

        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
