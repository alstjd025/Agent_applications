#!/usr/bin/env python3
"""EXP-23 - the two admission layers, as a function of offered rate.

A single load point cannot separate two admission policies, because at a load
either of them can carry, neither of them rejects anything. Measured: PolyServe
turned away 20 requests out of 23,080 on the dynamic trace. Raising the rate
until the fleet cannot carry the offer is what makes the admission decision the
thing being compared rather than an unused code path.

What is being compared:

  PolyServe   rejects when an instance's occupancy crosses a threshold. Whether
              the request would have met its SLO does not enter the decision.
  FluidServe  rejects when the placement it is willing to make would still miss
              the request's own budget. The capacity that request would have
              consumed then goes to the requests that can still meet theirs.

Scoring uses the offered denominator throughout: a rejected, errored or
unanswered request is a violation. Excluding rejections would reward a policy
for refusing the requests that were going to miss, which is exactly the axis
under test.

Reads every run directory named ..._<arm>_..._rpm_<N> and reports, per arm and
rate: equal-weight attainment across classes, each class separately, token
goodput, the rejection rate, and the arrival count -- the last because a policy
that holds requests can change how many arrive when any part of the client is
closed-loop, and a comparison at unequal offer is not a comparison at the same
load.

Usage
-----
  python3 exp23_rate_sweep.py --runs 'results/*exp23_*' --out-dir results/aggregate_analysis/exp23
"""
import argparse
import glob
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
    ARM_STYLE, CLASSES, CLASS_COLORS, PAPER_STYLE, WARMUP_S, DRAIN_S,
    load_run, equal_mix, attain, goodput_tokens, arm_of,
)


def rpm_of(run_dir):
    m = re.search(r"_rpm_(\d+)", os.path.basename(run_dir))
    return int(m.group(1)) if m else None


def summarise(run_dir):
    rows = load_run(run_dir)
    if rows is None or rows.empty:
        return None
    window = rows["rel"].max() - rows["rel"].min()
    if window <= 0:
        return None
    out = {
        "arm": arm_of(run_dir),
        "rpm": rpm_of(run_dir),
        "run": os.path.basename(run_dir),
        "n": len(rows),
        "arrivals_per_s": len(rows) / window,
        "eqmix": equal_mix(rows, "violate_offered"),
        "eqmix_served": equal_mix(rows, "violate_served"),
        "goodput": goodput_tokens(rows, window),
        "rejected_pct": 100.0 * rows["rejected"].mean(),
        "errored_pct": 100.0 * rows["errored"].mean(),
    }
    for c in CLASSES:
        sub = rows[rows["class"] == c]
        out[f"attain_{c}"] = attain(sub, "violate_offered")
        out[f"served_{c}"] = attain(sub, "violate_served")
        out[f"rejected_{c}"] = 100.0 * sub["rejected"].mean() if len(sub) else np.nan
    return out


def figures(df, out_dir):
    """Four panels against offered rate: the headline, the classes, throughput,
    and how much each policy turned away to get there."""
    arms = [a for a in ARM_STYLE if a in set(df["arm"])]
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.0))
        for a in arms:
            d = df[df.arm == a].sort_values("rpm")
            st = dict(ARM_STYLE[a])
            lbl = st.pop("label")
            # Solid: the admitted denominator, which is the headline. Faint: the
            # same arm on the offered denominator, drawn in the same panel so the
            # gap between the two lines IS the rejection rate's cost and cannot
            # be presented without it.
            ax[0].plot(d.rpm, d.eqmix_served, label=lbl, **st)
            ax[0].plot(d.rpm, d.eqmix, alpha=0.35, lw=1.0,
                       color=st.get("color"), ls=st.get("ls", "-"), marker="",
                       label=f"{lbl} (offered)")
            ax[2].plot(d.rpm, d.goodput, label=lbl, **st)
            ax[3].plot(d.rpm, d.rejected_pct, label=lbl, **st)
        # Per class, one line style per arm and one colour per class, so the
        # question "which class is the difference in" is answerable at a glance.
        for a in arms:
            d = df[df.arm == a].sort_values("rpm")
            for c in CLASSES:
                ax[1].plot(d.rpm, d[f"served_{c}"], color=CLASS_COLORS[c],
                           ls=ARM_STYLE[a]["ls"], marker=ARM_STYLE[a]["marker"],
                           label=f"{c} / {ARM_STYLE[a]['label']}")
        titles = ["equal-weight attainment\n(solid admitted, faint offered)",
                  "per class (admitted)", "token goodput", "rejected"]
        ylabels = ["SLO attainment (%)", "SLO attainment (%)",
                   "output tokens/s from requests that met their SLO",
                   "requests rejected (%)"]
        for i, (t, y) in enumerate(zip(titles, ylabels)):
            ax[i].set_title(t)
            ax[i].set_xlabel("offered rate (rpm)")
            ax[i].set_ylabel(y)
            ax[i].grid(axis="y", ls=":", lw=0.7, alpha=0.6)
        ax[0].legend(loc="lower left")
        ax[1].legend(fontsize=5.5, ncol=2, loc="lower left")
        fig.tight_layout()
        p = os.path.join(out_dir, "exp23_rate_sweep.png")
        fig.savefig(p, dpi=300)
        plt.close(fig)
        print(f"\nwrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    dirs = []
    for pattern in a.runs:
        dirs.extend(sorted(glob.glob(pattern)) or [pattern])
    rows = [r for r in (summarise(d) for d in dirs) if r and r["rpm"]]
    if not rows:
        sys.exit("no usable runs")
    df = pd.DataFrame(rows).sort_values(["arm", "rpm"])
    os.makedirs(a.out_dir, exist_ok=True)
    csv = os.path.join(a.out_dir, "exp23_rate_sweep.csv")
    df.to_csv(csv, index=False)

    # Both denominators, side by side, because neither answers the question on
    # its own. Attainment among ADMITTED requests is what a request that got in
    # can expect, and it is the number an admission controller is entitled to be
    # judged on -- but read alone it rewards refusing everything, so the
    # rejection rate is printed in the same row and token goodput beside it.
    # Attainment over EVERYTHING OFFERED has the opposite bias: it charges every
    # rejection as a violation even when the request was going to miss anyway.
    print("\nSLO attainment, equal weight across classes.")
    print("  admitted = denominator is the requests the system accepted")
    print("  offered  = denominator is every arriving request; a reject is a miss\n")
    hdr = (f"{'arm':<12}{'rpm':>6}{'arrived/s':>11}{'admitted':>10}{'offered':>9}"
           f"{'rej%':>7}{'goodput':>10}   per class (admitted) chat/dr/swe")
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        print(f"{r['arm']:<12}{r['rpm']:>6}{r['arrivals_per_s']:>11.1f}"
              f"{r['eqmix_served']:>10.1f}{r['eqmix']:>9.1f}"
              f"{r['rejected_pct']:>7.1f}{r['goodput']:>10.0f}   "
              f"{r['served_chat']:>5.1f}/{r['served_deepresearch']:>5.1f}/"
              f"{r['served_swe']:>5.1f}")

    print("\nRejection rate by class (%)\n")
    hdr = f"{'arm':<12}{'rpm':>6}{'chat':>8}{'dr':>8}{'swe':>8}"
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        print(f"{r['arm']:<12}{r['rpm']:>6}{r['rejected_chat']:>8.1f}"
              f"{r['rejected_deepresearch']:>8.1f}{r['rejected_swe']:>8.1f}")

    # The comparison itself, stated per rate rather than left to the reader.
    arms = sorted(set(df["arm"]))
    if len(arms) == 2:
        a0, a1 = arms
        print(f"\n{a1} minus {a0}, per rate")
        print(f"{'rpm':>6}{'admitted':>10}{'offered':>9}{'goodput':>10}")
        for rpm in sorted(set(df["rpm"])):
            x = df[(df.arm == a0) & (df.rpm == rpm)]
            y = df[(df.arm == a1) & (df.rpm == rpm)]
            if x.empty or y.empty:
                continue
            print(f"{rpm:>6}"
                  f"{y.eqmix_served.iloc[0] - x.eqmix_served.iloc[0]:>+10.1f}"
                  f"{y.eqmix.iloc[0] - x.eqmix.iloc[0]:>+9.1f}"
                  f"{y.goodput.iloc[0] - x.goodput.iloc[0]:>+10.0f}")

    figures(df, a.out_dir)
    print(f"wrote {csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
