#!/usr/bin/env python3
"""Paper figure: the same hour under two models, four panels.

  two_models_hour.pdf           7.0 x 1.85 in, `figure*`, width=\\textwidth
  two_models_hour_stacked.pdf   7.0 x 2.25 in, the same with the bar panels
                                split into what happened to every arrival
  two_models_hour_reqgoodput.pdf 7.0 x 1.59 in, the same with the bar panels
                                holding REQUEST GOODPUT, requests that met the
                                rule per second (`--req-goodput`)
  a `.csv` of the same basename beside each

  (a) token goodput over the hour, Llama-3.1-70B      wide, a timeline
  (b) request SLO on the admitted denominator, Llama  narrow, one bar per arm
  (c) token goodput over the hour, Qwen2.5-72B        wide, a timeline
  (d) request SLO on the admitted denominator, Qwen   narrow, one bar per arm

WHY THE TWO SHAPES SIT SIDE BY SIDE. Goodput is a rate and its shape over the
hour is the thing worth seeing -- where each policy loses the fleet and whether
it comes back. Attainment on the accepted set is, for most of these arms, a line
that sits near its own ceiling for most of the hour, and a timeline of it spends
a panel's width saying one number. That number is the bar. Each pair therefore
reads as "how much useful work, and of what was accepted how much was kept",
once per model.

⚠ THE BAR IS COMPUTED OVER EXACTLY THE SPAN ITS NEIGHBOUR DRAWS, not over the
whole run. Both panels of a pair are cut at the same minute, so a reader can put
a finger on the timeline and know the bar is the same requests.

⚠ THE TWO MODELS ARE NOT A CONTROLLED COMPARISON AND THIS FIGURE IS NOT ONE.
Qwen2.5-72B is the slower model on this hardware, so its hour runs a THINNED
arrival trace -- the same hour with 64% of the arrivals, 63,818 against 99,242,
a mean of 17.4 against 27.1 req/s -- while every class budget stays where it
was. Model and offered load therefore differ at once between (a,b) and (c,d),
and no difference between the halves can be attributed to either alone. The
figure puts two settings side by side; it does not measure the effect of the
model.

SCORING is the fixed rule of this directory: token i of a request is on time if
it arrives within TTFT_SLO + i * TBT_SLO of the send, counting the first token
as i = 0; the request is on time if at least 95% of its tokens are; goodput
counts tokens that met their own deadline. Budgets chat (5 s, 50 ms),
deepresearch (10 s, 100 ms), swe (7 s, 75 ms), unchanged between the models.

DATA. Repeat 2 of two for both models -- EXP-109 (2026-09-01/02, Llama) and
EXP-113 (2026-09-03, Qwen) -- because a timeline is one run and the second pass
is the convention the hour figures in this directory already follow. The bars
carry a thin range mark spanning repeat 1 and repeat 2 of the same cell,
computed the same way over the same span, so a bar whose mark is invisible is a
cell whose two repeats agree rather than a cell measured once.

`--stacked` REPLACES THE BAR WITH THE WHOLE POPULATION. The default bar is one
number, attainment among the requests the policy accepted, and it cannot say
what the policy did with the rest. The stacked version divides every arrival in
the same span four ways -- met its rule, accepted and missed, still unfinished
when the span ended, rejected -- so the bar is 100% of arrivals and the accepted
set is the first two bands together.

⚠ IT HAS TO BE FOUR BANDS AND NOT THREE. "Offered attainment" drops the
unfinished from its denominator, which is right for a ratio and wrong for a
stack: a bar of met / missed / rejected does not sum to the arrivals, and the
gap is largest exactly where it matters -- the arm without admission control
leaves about half of its arrivals unfinished on the Llama hour. Leaving that
band out would rescale that arm's bar and flatter it.

⚠ THE SEGMENT COLOURS ARE THE OUTCOME'S, NOT THE ARM'S, so the arm is named
under its bar instead of by colour, and the key carries both: the five arms for
the timelines, the four outcomes for the bars.

`--req-goodput` PUTS A RATE IN THE BAR INSTEAD OF A SHARE. The bar is the
number of requests that met their rule in the drawn span divided by that span,
so the pair becomes the same quantity in two units: tokens that met their
deadline per second on the left, requests that met theirs per second on the
right. A share cannot be read against another arm's share when the arms accept
different numbers of requests, and this is the version to use when the question
is how much each control plane delivered rather than what it did with what it
accepted. The other repeat is drawn as a range mark, as in the default bars.

⚠ THE vLLM ROUTER WAS NOT SENT THE SAME ARRIVALS. It refuses nothing, the load
generator fell behind against it, and in the drawn span it received 87,335
arrivals on the Llama hour and 56,073 on the Qwen hour against 96,68x and
61,50x for every other arm -- 9-10% fewer. Its bar is therefore a rate over a
thinner stream than its neighbours', which flatters it, and the caption has to
say so. The four-band figure does not have this problem because each of its
bars is a share of that arm's own arrivals.

⚠ THE TWO HALVES SHARE THE BAR AXIS AND STILL ARE NOT COMPARABLE. The Qwen hour
replays a thinned trace -- 17.45 against 27.20 arrivals per second -- so its
bars are lower for a reason that has nothing to do with the model. The shared
scale is there so the five bars of one half can be read against each other and
against that half's arrival rate.

    python3 paper_figures/fig_two_models_hour.py [--stacked] [--req-goodput]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis_scripts", "request_level"))

from paper_style import TEXT_W, STYLE, GRID, ARM_COLOR, kfmt, save  # noqa: E402
from exp41_dynamic_timeline import windows, WIN  # noqa: E402

VERDICTS = os.path.join(ROOT, "results", "aggregate_analysis", "ladder95",
                        "verdicts")
MIN_IN_WINDOW = 30
MAX_CUTOFF = 0.20
HOUR_MIN = 60.0
NO_ADMISSION = "vLLM"
# Panel heights were cut to 85% on 2026-09-04 and the canvas with them. The
# bands above and below the axes -- the key, and the paired panel names -- hold
# TEXT, whose size is absolute, so they must keep their inches while the axes
# lose theirs. `_rect` turns those two inch bands back into the figure
# fractions `tight_layout` wants, which is why the numbers below are inches and
# not the fractions they used to be.
# Cut twice: to 85% of the original panel height on 2026-09-04 and by a further
# 10% the same day. Only the axes lose the height -- the two text bands below
# keep their inches -- so the canvas falls by the same amount the panels do.
FIG_H = 1.59            # 1.85 -> 1.69 -> 1.59
FIG_H_STACKED = 1.92    # 2.25 -> 2.04 -> 1.92; the arm names need the extra
TOP_BAND_IN = 0.225     # the key
BOT_BAND_IN = 0.140     # the paired panel names


def _rect(height, right=0.988):
    return (0, BOT_BAND_IN / height, right, 1.0 - TOP_BAND_IN / height)

def _desat(hex_colour, f):
    """The same hue at lower saturation: blend the colour with its own grey."""
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    grey = 0.299 * r + 0.587 * g + 0.114 * b
    m = [c * f + grey * (1 - f) for c in (r, g, b)]
    return "#%02x%02x%02x" % tuple(int(round(255 * c)) for c in m)


def _pastel(hex_colour, f):
    """The same hue, lighter: blend the colour toward white.

    Lightening and desaturating are two different moves and this figure uses
    both, in that order, for the filled bands: taking chroma out first keeps a
    band from turning into a tint of its own hue, and lightening after is what
    makes it pastel. ⚠ THE HUE DOES NOT MOVE in either step, which is what lets
    an arm stay recognisable here as the same colour the rest of the paper gives
    it -- this figure draws them lighter and nowhere else does.
    """
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(
        int(round(255 * (c + (1.0 - c) * f))) for c in (r, g, b))


# How far the arm colours are lightened for the timelines. A filled area can
# take much more of this than a 0.8 pt line, which disappears into the page
# before the eye reads it as pastel, so the lines get a fifth and the bands
# well over a third.
LINE_PASTEL = 0.20


# The four outcome bands, in the same colours as `outcome_split_sweep_t75.pdf`
# and `pace_and_outcome_35.pdf`: one colour means one outcome across the paper.
# Bottom to top, and the key reads in the same order: what the fleet delivered
# inside the rule, what it delivered outside it, what it refused, and what has
# no outcome at all. `Unfinished` sits on top because it is the one band that is
# not a result -- the request was still running when the span ended.
# Chosen by the user on 2026-09-04, after a lighter pair (hsl(205,50%,91%) and
# hsl(14,41%,90%)) washed out and a blue one (#92c5de) collided with the
# FluidServe LINE in the panel beside it -- one figure cannot have blue mean a
# policy on its left and an outcome on its right. The teal and the salmon sit
# at nearly the same lightness, which is what makes them read as two bands
# rather than as one band and its shadow, and neither is any arm's colour.
#
# ⚠ THE GREY IS LIGHTER THAN BOTH ON PURPOSE. `Rejected` is the band where
# nothing was served, and the eye should not land on it first; at #cfcfcf it is
# the palest filled band and the two outcomes that did consume engine time are
# the ones that carry colour.
SEG = [("met", "SLO Attained", "#80cdc1", None),
       ("missed", "SLO Missed", "#f4a582", None),
       ("rejected", "Rejected", "#cfcfcf", None),
       ("unfinished", "Unfinished", "#ffffff", "////")]

STYLE_OF = {
    "vLLM": (_pastel(ARM_COLOR["vllmrouter"], LINE_PASTEL),
             (0, (4, 1, 1, 1, 1, 1))),
    "PolyServe": (_pastel(ARM_COLOR["polyserve"], LINE_PASTEL), "-."),
    "Llumnix SLO": (_pastel(ARM_COLOR["slo"], LINE_PASTEL), "--"),
    "llm-d": (_pastel(ARM_COLOR["llmd"], LINE_PASTEL), (0, (6, 1.5))),
    "FluidServe": (_pastel(ARM_COLOR["fluidserve"], LINE_PASTEL), "-"),
}
# Llumnix SLO and PolyServe swapped on 2026-09-11 at the author's request.
# ⚠ THE RAMP FOLLOWS THE ORDER, so the two also swap colours: Llumnix SLO takes
# #bdc9e1 and PolyServe #67a9cf. That keeps the key reading pale-to-dark from
# left to right, and it means a colour from a figure drawn before this date
# names a different arm.
ORDER = ["vLLM", "Llumnix SLO", "PolyServe", "llm-d", "FluidServe"]

# `--ramp`: ColorBrewer's five-class PuBuGn, in the order above. Chosen by the
# author on 2026-09-10 so that this figure and the class-mix ones share one
# family of colour.
# ⚠ IT IS A SEQUENTIAL RAMP AND THE ARMS ARE NOT ORDERED. Its five steps differ
# mainly in lightness, so the eye reads them as a scale from pale to dark; the
# five control planes are not five levels of one thing, and the caption must not
# lean on the order. What the ramp does buy is that the five are separable in
# greyscale, which the arm palette is not.
# ⚠ THE FIRST STEP IS #f6eff7, WHICH IS ALL BUT WHITE. As a 0.8 pt line on a
# white page it disappears, so the timelines are drawn with a thin grey stroke
# UNDER the line: the fill colour stays exactly the one asked for and the pale
# end of the ramp stays on the page. The bars get a black keyline for the same
# reason -- a #f6eff7 rectangle with no edge is not a rectangle anyone can see.
# ⚠ REPAINTED 2026-09-11 FROM PuBuGn TO ColorBrewer YlGnBu, at the author's
# request, so that this figure, its EXP-126 companion and the four-panel
# motivation figure all carry one ramp. FluidServe takes #253494 and the vLLM
# router #ffffcc in every one of the three. A colour read off a copy of this
# figure dated before today names a different arm.
# ⚠ llm-d IS NOT A STEP OF THE RAMP ANY MORE (2026-09-17, at the author's
# request): it is #c2a5cf, a purple, where the ramp had #2c7fb8. The four
# remaining arms keep their YlGnBu steps.
RAMP = ["#ffffcc", "#a1dab4", "#41b6c4", "#c2a5cf", "#253494"]

# The key says `Llumnix`, the code says `Llumnix SLO` (2026-09-11, at the
# author's request). The arm is Llumnix's SLO-aware policy
# (`--scheduling-policy slo`), and the internal name stays that because it is
# also the key into RUNS and STYLE_OF; only what the reader sees changes.
LEGEND_NAME = {"Llumnix SLO": "Llumnix"}


def apply_ramp():
    """Repaint the arms with RAMP, keeping each arm's dash pattern."""
    for arm, colour in zip(ORDER, RAMP):
        STYLE_OF[arm] = (colour, STYLE_OF[arm][1])

RUNS = {
    "Llama-3.1-70B": {
        "vLLM": ("260901_2010_exp109r2_vllmcachet75_shift",
                 "260901_2137_exp109r1_vllmcachet75_shift"),
        "PolyServe": ("260901_1742_exp109r2_polyservept75_shift",
                      "260831_2232_exp109r1_polyservept75_shift"),
        "Llumnix SLO": ("260901_1856_exp109r2_slot75_shift",
                        "260831_2346_exp109r1_slot75_shift"),
        "llm-d": ("260901_1637_exp109r2_llmdslot75_shift",
                  "260831_2128_exp109r1_llmdslot75_shift"),
        "FluidServe": ("260901_1514_exp109r2_fsv3capgnofrct75_shift",
                       "260831_2015_exp109r1_fsv3capgnofrct75_shift"),
    },
    "Qwen2.5-72B": {
        "vLLM": ("260903_1406_exp113r2_vllmcachet75_shiftq",
                 "260903_0702_exp113r1_vllmcachet75_shiftq"),
        "PolyServe": ("260903_1139_exp113r2_polyservept75_shiftq",
                      "260903_0436_exp113r1_polyservept75_shiftq"),
        "Llumnix SLO": ("260903_1252_exp113r2_slot75_shiftq",
                        "260903_0548_exp113r1_slot75_shiftq"),
        "llm-d": ("260903_1537_exp113r2_llmdslot75_shiftq",
                  "260903_0834_exp113r1_llmdslot75_shiftq"),
        "FluidServe": ("260903_1031_exp113r2_fsv3capgnofrct75_shiftq",
                       "260903_0302_exp113r1_fsv3capgnofrct75_shiftq"),
    },
}


def verdicts(run):
    p = os.path.join(VERDICTS, run + ".csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def goodput_series(v):
    dur = v["rel"].max()
    x, gp, cut = [], [], []
    for t, g in windows(v, dur):
        if len(g) < MIN_IN_WINDOW:
            continue
        x.append(t)
        gp.append(float((g["n_tokens"] - g["n_late"]).sum()) / WIN)
        cut.append(float(g["cutoff"].mean()))
    return np.array(x), np.array(gp), np.array(cut)


def admitted_upto(v, minute):
    w = v[v["rel"] <= minute * 60.0]
    served = w[~w["cutoff"] & ~w["rejected"]]
    return 100.0 * float(served["ladder_ok"].mean()) if len(served) else np.nan


def split_counts(v, minute):
    """The four COUNTS behind `split_upto`, in requests."""
    w = v[v["rel"] <= minute * 60.0]
    n = len(w)
    rej = int(w["rejected"].sum())
    cut = int(w["cutoff"].sum())
    met = int((w["ladder_ok"] & ~w["rejected"] & ~w["errored"]
               & ~w["cutoff"]).sum())
    return dict(met=met, missed=n - met - rej - cut, unfinished=cut,
                rejected=rej, arrivals=n)


def split_upto(v, minute):
    """The four shares of every arrival in the span, which sum to 100."""
    w = v[v["rel"] <= minute * 60.0]
    n = len(w)
    if not n:
        return {k: np.nan for k, _, _, _ in SEG}
    rej = int(w["rejected"].sum())
    cut = int(w["cutoff"].sum())
    met = int((w["ladder_ok"] & ~w["rejected"] & ~w["errored"]
               & ~w["cutoff"]).sum())
    missed = n - met - rej - cut          # client errors fold in here
    return dict(met=100.0 * met / n, missed=100.0 * missed / n,
                unfinished=100.0 * cut / n, rejected=100.0 * rej / n)


def collect():
    data, missing = {}, []
    for model, arms in RUNS.items():
        per = {}
        for arm in ORDER:
            r2, r1 = arms[arm]
            v2 = verdicts(r2)
            if v2 is None:
                missing.append(f"{model}/{arm}: no verdicts for {r2}")
                continue
            x, gp, cut = goodput_series(v2)
            per[arm] = dict(x=x, gp=gp, cut=cut, v2=v2, v1=verdicts(r1),
                            run2=r2, run1=r1)
        data[model] = per
    for m in missing:
        print(f"!! {m} -- skipped", file=sys.stderr)
    return data


def cut_minute(per):
    """The rule the hour figures use: the earliest minute at which an arm WITH
    admission control crosses MAX_CUTOFF in flight at the end."""
    ends = []
    for arm, d in per.items():
        if arm == NO_ADMISSION:
            continue
        ok = d["cut"] <= MAX_CUTOFF
        ends.append(d["x"][ok].max() if ok.any() else d["x"].min())
    return min(ends) if ends else HOUR_MIN


def build(data, out_pdf, stacked=False, norm_to=None, req_goodput=False,
          ramp=False, shrink=0.0, bar_max=None):
    models = list(RUNS)
    # A 0.6 pt grey outline drawn beneath each line. Only with the ramp: the arm
    # palette has no colour pale enough to need it, and an outline on a
    # saturated line reads as a heavier line.
    stroke = ([pe.Stroke(linewidth=1.5, foreground="#9a9a9a"), pe.Normal()]
              if ramp else None)
    cuts = {m: cut_minute(data[m]) for m in models}
    rows, top, bar_top = [], 0.0, 0.0
    # `shrink` takes points off the PANELS' type only -- the tick numbers and
    # the two axis names. The key above the panels and the pair captions below
    # them are figure-level text and keep their size, because they are read at
    # arm's length while the tick numbers are read next to the ink they label.
    style = dict(STYLE)
    if shrink:
        for k in ("axes.labelsize", "xtick.labelsize", "ytick.labelsize"):
            style[k] = max(4.0, style[k] - shrink)
    with plt.rc_context(style):
        fig, ax = plt.subplots(
            1, 4, figsize=(TEXT_W, FIG_H_STACKED if stacked else FIG_H),
            # The bar panels were cut a further 5% on 2026-09-04. The bar
            # rectangles are 0.72 of a category slot, so a narrower panel
            # narrows the bars with it.
            gridspec_kw=dict(width_ratios=[1.72, 0.532, 1.72, 0.532]))
        handles, labels, seg_handles = [], [], []
        for mi, model in enumerate(models):
            per, dur = data[model], cuts[model]
            gp_ax, bar_ax = ax[2 * mi], ax[2 * mi + 1]
            xs, vals, lo, hi, cols = [], [], [], [], []
            gvals, glo, ghi = [], [], []
            drawn_arms, splits, counts = [], {}, {}
            for j, arm in enumerate(ORDER):
                if arm not in per:
                    continue
                d = per[arm]
                col, ls = STYLE_OF[arm]
                keep = d["x"] <= dur
                line, = gp_ax.plot(d["x"][keep], d["gp"][keep], color=col,
                                   ls=ls, lw=0.8,
                                   path_effects=stroke if stroke else None)
                top = max(top, float(d["gp"][keep].max()))
                if mi == 0:
                    handles.append(line)
                    labels.append(LEGEND_NAME.get(arm, arm))
                a2 = admitted_upto(d["v2"], dur)
                a1 = admitted_upto(d["v1"], dur) if d["v1"] is not None else np.nan
                # Request goodput: the requests that met the rule, over the
                # span the panel beside it draws. The denominator is TIME, not
                # arrivals, so this is a rate and not a share -- an arm that is
                # sent fewer requests cannot make it up by keeping a higher
                # fraction of them.
                c2 = split_counts(d["v2"], dur)
                g2 = c2["met"] / (dur * 60.0)
                c1 = split_counts(d["v1"], dur) if d["v1"] is not None else None
                g1 = c1["met"] / (dur * 60.0) if c1 is not None else np.nan
                xs.append(j)
                vals.append(a2)
                lo.append(min(a2, a1) if a1 == a1 else a2)
                hi.append(max(a2, a1) if a1 == a1 else a2)
                gvals.append(g2)
                glo.append(min(g2, g1) if g1 == g1 else g2)
                ghi.append(max(g2, g1) if g1 == g1 else g2)
                cols.append(col)
                drawn_arms.append(arm)
                splits[arm] = split_upto(d["v2"], dur)
                counts[arm] = split_counts(d["v2"], dur)
                rows.append(dict(model=model, arm=arm, cut_minute=dur,
                                 admitted_pct=a2, admitted_pct_repeat1=a1,
                                 met_requests=c2["met"],
                                 met_requests_repeat1=(c1["met"] if c1 else np.nan),
                                 arrivals=c2["arrivals"],
                                 req_goodput_s=g2, req_goodput_s_repeat1=g1,
                                 run_drawn=d["run2"], run_other=d["run1"],
                                 **{f"{k}_pct": splits[arm][k]
                                    for k, _, _, _ in SEG}))
            # ⚠ NORMALISING TO FLUIDSERVE CHANGES WHAT A BAR IS. Without it
            # every bar is 100% of that arm's own arrivals, so the panel
            # compares COMPOSITION and says nothing about amount -- which is
            # right for "what happened to what arrived" and wrong for "who
            # delivered more". With it every segment is divided by FluidServe's
            # count of the chosen base, so a green segment of 71 means "71% as
            # many requests met their deadline as under FluidServe". The bars
            # then run past 100 and the axis says so.
            if stacked and norm_to:
                base = max(counts["FluidServe"][norm_to], 1)
                for arm in drawn_arms:
                    splits[arm] = {k: 100.0 * counts[arm][k] / base
                                   for k, _, _, _ in SEG}
                # The rows were written from the un-normalised shares; the CSV
                # has to hold what the figure DRAWS, so the drawn values are
                # added beside them with the base named.
                for row in rows:
                    if row["model"] != model:
                        continue
                    for k, _, _, _ in SEG:
                        row[f"{k}_norm_fs_pct"] = splits[row["arm"]][k]
                    row["norm_base"] = f"FluidServe {norm_to} = {base}"
            if stacked:
                bottom = np.zeros(len(xs))
                for key, _, colour, hatch in SEG:
                    v = np.array([splits[a][key] for a in drawn_arms])
                    b = bar_ax.bar(xs, v, bottom=bottom, width=0.72,
                                   color=colour, edgecolor="#555555",
                                   linewidth=0.35, hatch=hatch)
                    bottom += v
                    if mi == 0:
                        seg_handles.append(b[0])
                # The arm is named under its bar because in this version the
                # colour names the OUTCOME. Rotated because five names do not
                # fit side by side under a panel this narrow at any size that
                # can be read.
                bar_ax.set_xticks(xs)
                bar_ax.set_xticklabels(drawn_arms, fontsize=5.2, rotation=40,
                                       ha="right", rotation_mode="anchor")
            else:
                # `req_goodput` swaps what the bar holds and nothing else: the
                # arm still owns the colour, so the key that names the lines
                # names the bars, and the bars keep the order of the key.
                bv = gvals if req_goodput else vals
                blo, bhi = (glo, ghi) if req_goodput else (lo, hi)
                bar_ax.bar(xs, bv, width=0.72, color=cols,
                           edgecolor="#000000" if ramp else "#444444",
                           linewidth=0.5 if ramp else 0.35)
                # The other repeat as a RANGE mark, not an error bar: two runs
                # give a range, and a symmetric bar with a cap would suggest a
                # standard error that two points cannot support.
                bar_ax.vlines(xs, blo, bhi, color="#333333", lw=0.7)
                bar_ax.set_xticks([])
                bar_top = max(bar_top, max(bhi))
            bar_ax.set_xlim(-0.7, len(ORDER) - 0.3)
            # Exactly 0 to 100: the bars are shares of one population, so the
            # top of the panel IS 100% and a headroom of five points would
            # leave a gap that means nothing.
            if stacked and norm_to:
                topb = max(sum(splits[a][k] for k, _, _, _ in SEG)
                           for a in drawn_arms)
                bar_ax.set_ylim(0, max(105.0, topb * 1.05))
            elif req_goodput:
                pass          # set below, once both models are known
            else:
                bar_ax.set_ylim(0, 100)
                bar_ax.set_yticks([0, 25, 50, 75, 100])
            bar_ax.grid(axis="y", **GRID)
            bar_ax.set_axisbelow(True)
            gp_ax.set_xlim(0, HOUR_MIN)
            gp_ax.set_xticks(list(range(0, int(HOUR_MIN) + 1, 15)))
            gp_ax.grid(axis="both", **GRID)
            gp_ax.set_axisbelow(True)
            gp_ax.set_xlabel("Time (minutes)", labelpad=1.5)
            # Only the two spines that are axes, on every panel. The project
            # style draws all four; this figure drops the top and right on the
            # timelines because they box a curve that ends where it ends, and
            # on the bars because the 100% ceiling they used to mark is already
            # named by the topmost y tick. ⚠ The bars now END at the top of
            # their panel with no line above them, so the y axis is the only
            # thing saying the stack is a share of 100 -- the caption has to
            # say it too.
            for side in ("top", "right"):
                gp_ax.spines[side].set_visible(False)
                bar_ax.spines[side].set_visible(False)

        ax[2].sharey(ax[0])
        # ⚠ THE TWO BAR PANELS DO NOT SHARE A SCALE WHEN A TOP IS GIVEN. They
        # share it by default, which is what lets a reader put the two halves
        # side by side; a per-half top breaks that, and the caption then has to
        # say the two bar axes differ. It is defensible here only because the
        # halves are already not comparable -- the Qwen hour replays a thinned
        # trace -- so the bars are read within a half either way.
        if not any(bar_max or []):
            ax[3].sharey(ax[1])
        if req_goodput:
            # ⚠ ONE SCALE FOR BOTH MODELS, and it is the absolute rate. The two
            # halves are not a controlled comparison -- the Qwen hour replays a
            # thinned arrival trace -- so the shared axis is there to let the
            # five bars of one half be read against each other and against the
            # arrival rate stated in the caption, not to license reading the
            # halves against each other.
            # ⚠ NOT NAMED `top`: that name holds the timelines' maximum a few
            # lines below and shadowing it set the goodput panels to the bar
            # axis, which drew every curve as a flat line on the floor.
            for mi, i in enumerate((1, 3)):
                cap = (bar_max or [None, None])[mi] if bar_max else None
                ax[i].set_ylim(0, cap if cap else bar_top * 1.10)
        for i in (0, 2):
            ax[i].set_ylim(0, top * 1.08)
            ax[i].yaxis.set_major_formatter(kfmt())
        # Every panel carries its own y name. (c) and (d) share their limits
        # with (a) and (b) -- which is what makes the two halves comparable --
        # but a shared limit is not a shared label, and a panel without one
        # sends the reader back across the figure to find what it plots.
        if stacked:
            bar_name = "Request Arrivals (%)"
        elif req_goodput:
            # Named to rhyme with the timeline beside it: the pair is the same
            # quantity counted in two units, tokens that met their deadline per
            # second and requests that met theirs per second.
            bar_name = "Request Goodput (r/s)"
        else:
            bar_name = "SLO attain. (%)"
        ax[0].set_ylabel("Token Goodput (t/s)")
        ax[2].set_ylabel("Token Goodput (t/s)")
        ax[1].set_ylabel(bar_name)
        ax[3].set_ylabel(bar_name)

        if stacked:
            # One row, nine entries: five arms naming the timelines and four
            # outcomes naming the bands. They are different KINDS of key -- a
            # line and a filled area -- which is what keeps them apart in one
            # row without a separator.
            # Square swatches for the four outcomes: a legend patch is drawn
            # `handlelength` wide and `handleheight` tall, so setting the two
            # equal is what makes it a square rather than the wide rectangle
            # matplotlib gives by default. The arms stay LINES -- their dash
            # pattern is how they are told apart on the timelines, and a square
            # cannot carry it.
            seg_keys = [Patch(facecolor=c, edgecolor="#555555", linewidth=0.35,
                              hatch=h) for _, _, c, h in SEG]
            fig.legend(handles + seg_keys,
                       labels + [t for _, t, _, _ in SEG],
                       loc="lower center", ncol=len(labels) + len(SEG),
                       bbox_to_anchor=(0.5, 1.0 - TOP_BAND_IN / FIG_H_STACKED),
                       frameon=False, fontsize=6.4,
                       columnspacing=0.6, handlelength=1.25, handleheight=1.25,
                       handletextpad=0.3, borderaxespad=0.0)
            fig.tight_layout(rect=_rect(FIG_H_STACKED), w_pad=1.2, pad=0.3)
        else:
            fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                       bbox_to_anchor=(0.5, 1.0 - TOP_BAND_IN / FIG_H),
                       frameon=False,
                       columnspacing=1.2, handlelength=1.9, handletextpad=0.4,
                       borderaxespad=0.0)
            fig.tight_layout(rect=_rect(FIG_H, right=0.995), w_pad=1.2,
                             pad=0.3)
        # The panel names sit at one height in canvas coordinates: the bar
        # panels have no x tick labels and the timelines do, so names attached
        # to the axes would come out at two heights.
        # The bar panels' names are short because their panels are: a name
        # centred on a 0.9 in panel runs off the canvas at the width the four
        # of these together can afford, and the model is already named by the
        # timeline each bar panel sits beside.
        # ONE NAME PER PAIR, centred under the two panels it covers. The pair
        # is the unit of this figure -- a rate over the hour and what became of
        # the arrivals behind it, for one model -- and four separate names made
        # a reader match them up by position.
        #
        # ⚠ The model names are Llama-3.1-70B and Qwen2.5-72B. Neither is 80B;
        # the parameter count is part of the model's name and a figure that
        # states it wrongly is wrong about its own setup.
        # One name for the pair when both panels hold goodput, two names when
        # they hold different quantities: "Token and Request Goodput" says the
        # pair is one measurement in two units, which "Token goodput, Request
        # SLO" cannot say because those are two measurements.
        name = ("Token and Request Goodput" if req_goodput
                else "Token goodput, Request SLO")
        pairs = [(0, 1, f"(a) {name} - Llama-3.1-70B"),
                 (2, 3, f"(b) {name} - Qwen2.5-72B")]
        for i, j, name in pairs:
            b0, b1 = ax[i].get_position(), ax[j].get_position()
            fig.text(0.5 * (b0.x0 + b1.x1), 0.012, name, ha="center",
                     va="bottom", fontsize=8)
        save(fig, out_pdf)
    return pd.DataFrame(rows), cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stacked", action="store_true",
                    help="bar panels show what happened to every arrival")
    ap.add_argument("--norm-to-fs", choices=["met", "arrivals"], default=None,
                    help="divide every segment by FluidServe's count of this, "
                         "so the bars compare AMOUNT and not only composition")
    ap.add_argument("--bar-max", default=None,
                    help="top of each pair's bar axis, comma separated and in "
                         "the pairs' order; an empty entry keeps the automatic "
                         "top. Giving either one stops the two bar panels "
                         "sharing a scale, e.g. ',15'")
    ap.add_argument("--shrink-labels", type=float, default=0.0,
                    help="take this many points off the panels' tick and axis "
                         "label size (the key and the captions keep theirs)")
    ap.add_argument("--ramp", action="store_true",
                    help="colour the five arms with the PuBuGn ramp instead of "
                         "their usual palette")
    ap.add_argument("--req-goodput", action="store_true",
                    help="bar panels show requests that met the rule per "
                         "second instead of a share of the arrivals")
    args = ap.parse_args()
    if args.norm_to_fs and not args.stacked:
        sys.exit("--norm-to-fs applies to the stacked bars; add --stacked")
    if args.req_goodput and args.stacked:
        sys.exit("--req-goodput replaces the stacked bars; drop --stacked")
    stem = ("two_models_hour" + ("_stacked" if args.stacked else "")
            + (f"_normfs_{args.norm_to_fs}" if args.norm_to_fs else "")
            + ("_reqgoodput" if args.req_goodput else ""))
    if args.ramp:
        apply_ramp()
    data = collect()
    df, cuts = build(data, os.path.join(HERE, stem + ".pdf"),
                     stacked=args.stacked, norm_to=args.norm_to_fs,
                     req_goodput=args.req_goodput, ramp=args.ramp,
                     shrink=args.shrink_labels,
                     bar_max=([float(v) if v.strip() else None
                               for v in args.bar_max.split(",")]
                              if args.bar_max else None))
    out = os.path.join(HERE, stem + ".csv")
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"wrote {out}  ({len(df)} rows)")
    print("\ncut minute per model:", {k: round(v, 1) for k, v in cuts.items()})
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
