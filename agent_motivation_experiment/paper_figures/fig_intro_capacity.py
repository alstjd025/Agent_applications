#!/usr/bin/env python3
"""Paper figure (intro): the load these four engines sustain is set by the router.

  intro_capacity.pdf        3.335 x 2.05 in, single column, width=\\columnwidth
  intro_capacity_curves.pdf 3.335 x 2.20 in, the same claim with its derivation

**The claim.** Capacity is defined here as the highest offered rate at which at
least 90% of ARRIVING requests still meet their own latency rule -- rejections
counted as misses, so a policy cannot buy the number by refusing work. On the
same four engines, the same request mix and the same rule, FluidServe v0.2
sustains 28.1 req/s and llm-d 18.7, a factor of 1.50.

**Why two files.** The bar chart is the compact statement. The curve version
shows where each bar comes from: attainment against offered rate with a rule at
90% and the crossings dropped to the axis. For an introduction the curves are
usually the better figure, because a reader who has not yet been told what
"capacity" means here can see it being measured; the bars are for when space is
short or the same quantity has already been introduced.

**How the number is computed.** Linear interpolation between the two measured
conditions that bracket 90%, which is the definition motivation_fig3.py uses so
the two agree. Eight rates per arm: 10, 15, 20, 25, 35, 45, 55, 70 req/s.

**Robustness.** The ratio does not depend on where the line is drawn:

      counted as saturated   FluidServe   llm-d   ratio
             95%                25.4       12.2    2.08
             90%                28.1       18.7    1.50
             80%                33.7       22.7    1.49
             70%                39.2       25.8    1.52

**CAVEATS that belong in the caption.**

1. **Two arms, not five.** PolyServe and Llumnix SLO have no static conditions
   on this workload; their sweeps predate the 2026-08-08 load-generator fix and
   cannot be placed beside these. The five-baseline sweep is what fills the
   figure out, on the grid EXP-70 chose.
2. **One repeat at 10, 15, 20 and 25 req/s** (EXP-70, whose purpose was to locate
   the knee), two at 35 through 70 (EXP-68/69). The crossing at 90% falls between
   25 and 35 for FluidServe and between 20 and 25 for llm-d, so **both bars are
   interpolated across a single-repeat point** and the figure says so.
3. **swe is scored from a different workload config for llm-d** (m1f, 2,500 ms /
   52 ms per token) than for FluidServe (m1 with a 25:e2e:30000 override),
   because llm-d cannot express an end-to-end budget. Both are judged against the
   same 30 s end-to-end rule when scored.
4. **No prediction line.** The dashed additive prediction in
   motivation_capacity_is_a_policy.png comes from EXP-55, which ran each class
   alone on the pre-fix workload. Drawing it here would put a pre-fix line
   against post-fix bars. Re-measuring EXP-55 is what brings it back.

  python3 paper_figures/fig_intro_capacity.py
"""
import collections
import glob
import importlib.util
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import paper_style as ps  # noqa: E402


def _load_module(rel):
    spec = importlib.util.spec_from_file_location("m", os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Import the scoring rather than reimplementing it: a correction to load_run has
# to reach the paper, and a second copy of the attainment definition is how the
# two drift apart.
EXP22 = _load_module("analysis_scripts/request_level/exp22_fluidserve.py")

LEVEL = 90.0
CRITERIA = [95.0, 90.0, 80.0, 70.0]

# Post-fix static conditions only. The globs are written out rather than
# widened, because the pre-fix sweeps match any looser pattern and averaging
# them in would silently mix two workloads.
ARMS = [
    ("FluidServe v0.2", "#17becf", "s",
     ["results/*exp68s*_fspfx_*_rpm_*", "results/*exp69*_fspfx_*_rpm_*",
      "results/*exp70*_fspfx_*_rpm_*", "results/*exp68r*_fspfx_*_rpm_*"]),
    ("llm-d", "#8c564b", "D",
     ["results/*exp68s*_llmdslo_*_rpm_*", "results/*exp70*_llmdslo_*_rpm_*",
      "results/*exp68r*_llmdslo_*_rpm_*"]),
]


def sweep(patterns):
    """rate (req/s) -> list of offered attainment, one entry per repeat."""
    pts = collections.defaultdict(list)
    seen = set()
    for pat in patterns:
        for d in sorted(glob.glob(os.path.join(ROOT, pat))):
            if "PRERUN" in d or d in seen:
                continue
            seen.add(d)
            r = EXP22.load_run(d)
            if r is None or r.empty:
                continue
            rate = int(re.search(r"rpm_(\d+)", d).group(1)) / 60.0
            pts[rate].append(EXP22.per_request(r, "violate_offered"))
    return pts


def crossing(pts, level=LEVEL):
    """Offered rate at which mean attainment falls through `level`."""
    x = sorted(pts)
    y = [float(np.mean(pts[k])) for k in x]
    for i in range(len(x)):
        if y[i] < level:
            if i == 0:
                return x[0]
            return x[i - 1] + (y[i - 1] - level) * (x[i] - x[i - 1]) / (y[i - 1] - y[i])
    return float("nan")


def fig_bars(data, caps, out):
    with plt.rc_context(ps.STYLE):
        fig, ax = plt.subplots(figsize=(ps.COL_W, 2.05))
        names = [n for n, _, _, _ in ARMS]
        for i, (name, col, _, _) in enumerate(ARMS):
            ax.bar(i, caps[name], color=col, width=0.55,
                   edgecolor="white", linewidth=0.4)
            ax.annotate(f"{caps[name]:.1f}", (i, caps[name]), ha="center",
                        va="bottom", fontsize=8, color=col, weight="bold",
                        xytext=(0, 1.5), textcoords="offset points")
        hi, lo = caps[names[0]], caps[names[-1]]
        ax.annotate(f"{hi / lo:.2f}x", xy=(0.5, max(caps.values()) * 0.55),
                    ha="center", fontsize=8, color="#333333")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_ylabel("sustained rate (req/s)")
        ax.set_ylim(0, max(caps.values()) * 1.22)
        ax.grid(axis="y", **ps.GRID)
        ax.set_axisbelow(True)
        fig.tight_layout(rect=(0, 0, 1, 1))
        ps.save(fig, out)


def fig_curves(data, caps, out):
    with plt.rc_context(ps.STYLE):
        fig, ax = plt.subplots(figsize=(ps.COL_W, 2.20))
        for name, col, mk, _ in ARMS:
            pts = data[name]
            x = sorted(pts)
            y = [float(np.mean(pts[k])) for k in x]
            ax.plot(x, y, color=col, marker=mk, label=name)
            c = caps[name]
            # Drop the crossing to the axis so the bar's number is visibly the
            # x coordinate where the curve meets the rule, not a separate claim.
            ax.plot([c, c], [0, LEVEL], color=col, ls="--", lw=0.7)
            ax.annotate(f"{c:.1f}", (c, 2), color=col, fontsize=8,
                        weight="bold", ha="center", va="bottom",
                        bbox=dict(fc="white", ec="none", pad=0.6))
        ax.axhline(LEVEL, color="#333333", lw=0.7, ls=":")
        ax.annotate("90% of arriving requests meet their rule", (71, LEVEL),
                    ha="right", va="bottom", fontsize=7, color="#333333")
        ax.set_xlabel("offered rate (requests/s)")
        ax.set_ylabel("SLO attainment (%)")
        ax.set_xlim(5, 73)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", **ps.GRID)
        ax.set_axisbelow(True)
        # Left of centre, not lower left: the crossing labels sit just above
        # the x axis at the two dashed rules, and a lower-left legend lands on
        # top of the lower of them.
        ax.legend(loc="center left", handlelength=1.4)
        fig.tight_layout(rect=(0, 0, 1, 1))
        ps.save(fig, out)


def main():
    data = {name: sweep(pats) for name, _, _, pats in ARMS}
    for name, pts in data.items():
        if not pts:
            sys.exit(f"no post-fix static runs matched for {name}")
    caps = {name: crossing(pts) for name, pts in data.items()}

    print(f"capacity at >= {LEVEL:.0f}% offered attainment "
          f"(post-2026-08-08 workload, rejections count as misses)")
    for name, _, _, _ in ARMS:
        pts = data[name]
        reps = ", ".join(f"{k:g}:{len(v)}" for k, v in sorted(pts.items()))
        print(f"  {name:18s} {caps[name]:5.1f} req/s     repeats per rate  {reps}")
    names = [n for n, _, _, _ in ARMS]
    print(f"  ratio {names[0]} / {names[-1]} = "
          f"{caps[names[0]] / caps[names[-1]]:.2f}x")
    print("\nsensitivity to where 'saturated' is drawn")
    for lv in CRITERIA:
        v = {n: crossing(data[n], lv) for n in names}
        print(f"  {lv:.0f}%  " + "  ".join(f"{n} {v[n]:5.1f}" for n in names)
              + f"   ratio {v[names[0]] / v[names[-1]]:.2f}x")

    fig_bars(data, caps, os.path.join(HERE, "intro_capacity.pdf"))
    fig_curves(data, caps, os.path.join(HERE, "intro_capacity_curves.pdf"))


if __name__ == "__main__":
    sys.exit(main())
