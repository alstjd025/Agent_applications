#!/usr/bin/env python3
"""Paper figure (intro): the load these four engines sustain is set by the router.

  intro_capacity.pdf        3.335 x 2.05 in, single column, width=\\columnwidth
  intro_capacity_curves.pdf 3.335 x 2.20 in, the same claim with its derivation

**The claim.** Capacity is defined here as the highest offered rate at which at
least 90% of ARRIVING requests still meet their own latency rule -- rejections
counted as misses, so a policy cannot buy the number by refusing work. On the
same four engines, the same request mix and the same rule:

      FluidServe v0.2   28.1 req/s
      Llumnix SLO       20.4
      llm-d             18.7
      PolyServe         15.8

**Why two files.** The bar chart is the compact statement. The curve version
shows where each bar comes from: attainment against offered rate with a rule at
90% and the crossings dropped to the axis. For an introduction the curves are
usually the better figure, because a reader who has not yet been told what
"capacity" means here can see it being measured; the bars are for when space is
short or the same quantity has already been introduced.

**How the number is computed.** Linear interpolation between the two measured
conditions that bracket 90%, which is the definition motivation_fig3.py uses so
the two agree. Eight rates per arm: 10, 15, 20, 25, 35, 45, 55, 70 req/s.

**Robustness, and the part of it that does NOT hold.**

      counted as saturated   PolyServe  Llumnix SLO   llm-d   FluidServe
             95%                15.4        20.0       12.2      25.4
             90%                15.8        20.4       18.7      28.1
             80%                16.7        21.3       22.7      33.7
             70%                17.5        22.1       25.8      39.2

FluidServe is first at every threshold, so "this policy sustains the highest
rate" does not depend on where the line is drawn. **The ordering among the three
baselines does.** llm-d is LAST at 95%, third at 90%, and second at 80% and 70%,
because it degrades gradually while the other two fall off a cliff -- Llumnix SLO
holds 95.4% at 20 req/s and drops to 34.4% at 25, and PolyServe holds 99.9% at 15
and drops to 39.9% at 20. A single capacity number therefore ranks our policy
against the baselines robustly and ranks the baselines against each other only at
the threshold it was computed for. Say so in the caption; the curve figure shows
it directly and is the better choice where the ranking of baselines matters.

**CAVEATS that belong in the caption.**

1. **Four arms, not five.** vLLM's production stack router is not ported, so it
   is absent. The other four are all on the post-2026-08-08 workload.
2. **Repeats are uneven, and the crossings sit in the thin part.** FluidServe and
   llm-d have two repeats at 35-70 req/s and one at 10-25 (EXP-68/69/70);
   PolyServe and Llumnix SLO have **one repeat at every rate** (EXP-72). Every
   one of the four crossings falls in a single-repeat region. A second repeat of
   EXP-72, and of the low rates for the other two, is what this figure needs
   before it goes in the paper.
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
# Ordered worst to best so the bars read left to right as an argument. An arm
# with no post-fix runs yet is skipped with a message rather than silently
# dropped, and rather than being filled in from its pre-fix sweep -- EXP-53/57
# and EXP-66 are a different workload and putting them on this axis is the
# same-name-two-quantities failure this repository keeps hitting.
ARMS = [
    ("PolyServe", ps.ARM_COLOR["polyserve"], "o",
     ["results/*exp72*_polyserve_*_rpm_*"]),
    ("Llumnix SLO", ps.ARM_COLOR["slo"], "^",
     ["results/*exp72*_slo_*_rpm_*"]),
    ("llm-d", ps.ARM_COLOR["llmd"], "D",
     ["results/*exp68s*_llmdslo_*_rpm_*", "results/*exp70*_llmdslo_*_rpm_*",
      "results/*exp68r*_llmdslo_*_rpm_*"]),
    # Named "FluidServe" and drawn in the shared blue, matching every other
    # paper figure and `fig_exp71_hour.py`. The arm is FluidServe v0.2 and the
    # version is stated in the docstring and the caption instead of on the axis:
    # no other version of ours appears in this paper, so "v0.2" on a tick label
    # asks the reader to hold a distinction the figure never uses. The cyan this
    # carried came from the EXP-71 analysis script, where it separates v0.2 from
    # the earlier arm; here that separation does not exist.
    ("FluidServe", ps.ARM_COLOR["fluidserve"], "s",
     ["results/*exp68s*_fspfx_*_rpm_*", "results/*exp69*_fspfx_*_rpm_*",
      "results/*exp70*_fspfx_*_rpm_*", "results/*exp68r*_fspfx_*_rpm_*"]),
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


# Narrower than one column, so this is the one figure here that must NOT be
# included with `width=\columnwidth`: that would scale 2.90 in up to 3.335 and
# multiply every glyph by 1.15. Include it at its natural size --
# `\includegraphics{intro_capacity}` or `width=2.9in` -- and it lands at the
# 8 pt everything else in the paper is set at, leaving 0.44 in of column beside
# it. The bars are 0.40 wide rather than 0.55 for the same reason the canvas is
# narrower: at four bars the default reads as a block of colour.
# 2.90, not 2.60. At 2.60 the four arm names at 8 pt run into each other:
# "PolyServe" ends exactly where "Llumnix" begins. Rendering the tick row at
# 2.60 / 2.90 / 3.10 in and reading it back, 2.90 is where they separate. The
# alternative was to keep 2.60 and set this axis at 7 pt, which is what the
# figure did before and is the reason it did not match the other figures.
BAR_W = 2.90
BAR_H = 1.85
# The combined figure: bars and curves side by side in ONE COLUMN.
PAIR_H = 1.75
# Tick labels only; the legend of the curve figure has room for the full name.
TICK_BREAK = {"Llumnix SLO": "Llumnix\nSLO"}


def fig_bars(data, caps, out):
    with plt.rc_context(ps.STYLE):
        fig, ax = plt.subplots(figsize=(BAR_W, BAR_H))
        draw_bars(ax, caps)
        fig.tight_layout(rect=(0, 0, 1, 1))
        ps.save(fig, out)


def draw_bars(ax, caps, tick_fontsize=8, names_on_axis=True, ylabel=None):
    """The bar panel, on an axes handed in, so the standalone figure and the
    combined one cannot drift apart.

    `names_on_axis=False` drops the arm names from the x axis. That is only
    legitimate where a legend on the same figure names the same four arms in the
    same colours and the same left-to-right order, which is the case in the
    one-column pair; a bar chart whose bars are identified nowhere is not.
    """
    names = [n for n, _, _, _ in ARMS]
    for i, (name, col, _, _) in enumerate(ARMS):
        ax.bar(i, caps[name], color=col, width=0.40,
               edgecolor="white", linewidth=0.4)
        ax.annotate(f"{caps[name]:.1f}", (i, caps[name]), ha="center",
                    va="bottom", fontsize=8, color=col, weight="bold",
                    xytext=(0, 1.5), textcoords="offset points")
    ax.set_xticks(range(len(names)))
    if names_on_axis:
        ax.set_xticklabels([TICK_BREAK.get(n, n) for n in names],
                           fontsize=tick_fontsize)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    ax.set_ylabel(ylabel or "Maximum capacity (req/s)")
    ax.set_ylim(0, max(caps.values()) * 1.12)
    ax.grid(axis="y", **ps.GRID)
    ax.set_axisbelow(True)


def draw_curves(ax, data, caps, label_gap=0.09, row_step=7.5,
                crossing_labels=True):
    """The curve panel. Returns the legend handles in ARMS order.

    `crossing_labels=False` keeps the dashed drop lines but not the numbers on
    them. Only for a figure where the bar panel prints the same four numbers:
    the drop line still shows that the bar's value is where the curve meets the
    rule, and four stacked labels do not fit in a 1.8 in panel.
    """
    crossings, handles = [], []
    for name, col, mk, _ in ARMS:
        pts = data[name]
        x = sorted(pts)
        y = [float(np.mean(pts[k])) for k in x]
        h, = ax.plot(x, y, color=col, marker=mk, label=name)
        handles.append(h)
        c = caps[name]
        # Drop the crossing to the axis so the bar's number is visibly the x
        # coordinate where the curve meets the rule, not a separate claim.
        ax.plot([c, c], [0, LEVEL], color=col, ls="--", lw=0.7)
        crossings.append((c, name, col))
    ax.set_xlim(5, 73)
    ax.set_ylim(0, 105)
    # The crossing labels are placed after all the curves are drawn, because
    # with four arms three of them land within 5 req/s of each other and collide
    # into an unreadable run of digits. Stack them instead: sort by x, and step
    # the y position for any label whose neighbour is closer than `gap` on the x
    # axis. Alternating heights would still collide when three cluster, so this
    # counts the run.
    crossings.sort()
    gap = label_gap * (ax.get_xlim()[1] - ax.get_xlim()[0])
    row = 0
    for i, (c, name, col) in enumerate(crossings if crossing_labels else []):
        if i and c - crossings[i - 1][0] < gap:
            row += 1
        else:
            row = 0
        # `row_step` is in data units and the panel is 105 units tall, so how
        # many POINTS one step is worth depends on how tall the axes are: 7.5
        # units is 10 pt in the tall standalone figure and 6.4 pt in the short
        # combined one, where an 8 pt label then overlaps the row below it.
        # The caller sets it from its own panel height.
        ax.annotate(f"{c:.1f}", (c, 2 + row_step * row), color=col, fontsize=8,
                    weight="bold", ha="center", va="bottom",
                    bbox=dict(fc="white", ec="none", pad=0.6))
    ax.axhline(LEVEL, color="#333333", lw=0.7, ls=":")
    ax.set_xlabel("Offered rate (req/s)")
    ax.set_ylabel("SLO attainment (%)")
    ax.grid(axis="y", **ps.GRID)
    ax.set_axisbelow(True)
    return handles


def fig_pair(data, caps, out):
    """Both panels in one file: the capacity number on the left, the measurement
    it comes from on the right. ONE COLUMN, `width=\\columnwidth`.

    3.335 in split 1 : 1.45 leaves the bars about 1.1 in of drawing area and the
    curves about 1.8. Three things are dropped to fit, each because the same
    information is already on the figure:

      - THE ARM NAMES ON THE BAR AXIS. The legend above names the four arms in
        the same colours and the same left-to-right order the bars are in.
        Four names at 8 pt need about 2.0 in and there is 1.1.
      - THE CROSSING NUMBERS ON THE CURVES. The bars print the same four values.
        The dashed drop lines stay, so the reader can still see that each bar's
        height is the rate at which that curve meets the rule.
      - THE SENTENCE ON THE RULE LINE. "90% of arriving requests meet their
        rule" is 2.2 in at 7 pt. Shortened to "90%" against the line; the
        caption has to carry what the rule is and that rejections count as
        misses.

    The curve panel is the wider of the two because it carries thirty-two
    points, eight x ticks and a rule line against the bars' four values.
    """
    with plt.rc_context(ps.STYLE):
        fig, (ax_b, ax_c) = plt.subplots(
            1, 2, figsize=(ps.COL_W, PAIR_H),
            gridspec_kw=dict(width_ratios=[1.0, 1.45]))
        # "Maximum capacity (req/s)" rotated is 1.15 in and the panel is about
        # 1.0 in tall here, so it would be clipped. The word the axis cannot
        # afford is the one the caption can carry.
        draw_bars(ax_b, caps, names_on_axis=False, ylabel="Capacity (req/s)")
        handles = draw_curves(ax_c, data, caps, crossing_labels=False)
        # Right end of the rule line, not the left: past 25 req/s every curve
        # is below 90%, so that corner is the only empty stretch of the line.
        ax_c.annotate("90%", (72, LEVEL), ha="right", va="bottom", fontsize=7,
                      color="#333333")
        ax_c.set_xticks([10, 30, 50, 70])
        # Spanning BOTH panels, not just the curve one: with the names gone from
        # the bar axis this legend is the only thing identifying the bars, so it
        # has to sit over them too.
        fig.legend(handles, [n for n, _, _, _ in ARMS], loc="lower center",
                   bbox_to_anchor=(0.5, 0.885), ncol=4, fontsize=7,
                   frameon=False, handlelength=1.2, columnspacing=0.7,
                   handletextpad=0.3, borderaxespad=0.0)
        fig.tight_layout(rect=(0, 0, 1, 0.885), w_pad=0.8, pad=0.3)
        ps.save(fig, out)


def fig_curves(data, caps, out):
    with plt.rc_context(ps.STYLE):
        fig, ax = plt.subplots(figsize=(ps.COL_W, 2.20))
        draw_curves(ax, data, caps)
        ax.annotate("90% of arriving requests meet their rule", (71, LEVEL),
                    ha="right", va="bottom", fontsize=7, color="#333333")
        # Left of centre, not lower left: the crossing labels sit just above the
        # x axis at the dashed rules, and a lower-left legend lands on top of
        # the lower of them.
        ax.legend(loc="center left", handlelength=1.4)
        fig.tight_layout(rect=(0, 0, 1, 1))
        ps.save(fig, out)


def main():
    global ARMS
    data = {name: sweep(pats) for name, _, _, pats in ARMS}
    absent = [n for n, pts in data.items() if not pts]
    for n in absent:
        print(f"  NOT DRAWN: {n} has no static conditions on the post-fix "
              f"workload yet")
    ARMS = [a for a in ARMS if data[a[0]]]
    data = {n: p for n, p in data.items() if p}
    if len(ARMS) < 2:
        sys.exit("fewer than two arms have post-fix runs")
    caps = {name: crossing(pts) for name, pts in data.items()}

    print(f"capacity at >= {LEVEL:.0f}% offered attainment "
          f"(post-2026-08-08 workload, rejections count as misses)")
    for name, _, _, _ in ARMS:
        pts = data[name]
        reps = ", ".join(f"{k:g}:{len(v)}" for k, v in sorted(pts.items()))
        print(f"  {name:18s} {caps[name]:5.1f} req/s     repeats per rate  {reps}")
    names = [n for n, _, _, _ in ARMS]
    # Highest over lowest, named. Taking the first and last of the list gave
    # 0.67x the moment the list was reordered, which is the same number upside
    # down and reads as a loss.
    best, worst = max(caps, key=caps.get), min(caps, key=caps.get)
    print(f"  ratio {best} / {worst} = {caps[best] / caps[worst]:.2f}x")
    print("\nsensitivity to where 'saturated' is drawn")
    for lv in CRITERIA:
        v = {n: crossing(data[n], lv) for n in names}
        b2, w2 = max(v, key=v.get), min(v, key=v.get)
        print(f"  {lv:.0f}%  " + "  ".join(f"{n} {v[n]:5.1f}" for n in names)
              + f"   {b2}/{w2} = {v[b2] / v[w2]:.2f}x")

    fig_bars(data, caps, os.path.join(HERE, "intro_capacity.pdf"))
    fig_pair(data, caps, os.path.join(HERE, "intro_capacity_pair.pdf"))
    fig_curves(data, caps, os.path.join(HERE, "intro_capacity_curves.pdf"))


if __name__ == "__main__":
    sys.exit(main())
