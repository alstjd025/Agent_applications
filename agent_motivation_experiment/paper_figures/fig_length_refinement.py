#!/usr/bin/env python3
"""Paper figure: how well a request's length can be known, and when.

  length_refinement.pdf   3.335 x 1.95 in, `figure`, width=\\columnwidth
  length_refinement.csv   exactly the values drawn

  (a) the median error of estimating a request's FINAL output length, against
      how many tokens it has already produced, with and without knowing which
      dataset it came from
  (b) the length distributions themselves, pooled and per dataset, as kernel
      densities over log10(tokens)

WHAT IT MEASURES. At a point where a request has produced j tokens and has not
finished, estimate its final length as j plus the median remaining length of the
requests that were still running at j, and score the estimate against what the
request actually produced: |estimate - actual| / actual, median over the
requests still running. Two estimators are drawn in (a): one that pools the
whole mixture and one that uses the request's own class. Nothing here is a
prediction of the future from a model -- both estimators are the empirical
distribution of this workload, so the curves are the information that is THERE,
not the accuracy of any particular predictor.

THE TWO WAYS OF KNOWING MORE, AND THEIR SIZES.

    knowing the class, at admission      43.8% -> 35.6%   (8.2 points)
    watching 400 tokens go by, pooled    43.8% -> 26.2%   (17.6 points)
    both                                 43.8% -> 16.9%   (26.9 points)

Reading down a column of the table in the CSV gives the first, reading along a
row gives the second. THE SECOND IS THE LARGER OF THE TWO, and it costs nothing
to obtain: a control plane that revisits a request while it runs is already
holding the measurement that produces it.

⚠ THE CURVES ARE OVER SURVIVORS, AND THEY STOP WHEN THE SURVIVORS DO. Every
point is computed on the requests still running at that j, a smaller and
longer-tailed population as j grows -- 6,393 of 34,598 at j = 800 pooled, but
only 52 of the 2,745 agent requests. A series is drawn only while at least 200
of its requests are still running: the agent class ends at j = 600 for that
reason, and the reason is arithmetic rather than editorial, since its 5.9% at
j = 700 and 11.8% at j = 800 have bootstrap intervals of [4.1, 8.2] and
[6.2, 14.1] and do not separate. The counts behind every point are in the CSV.

⚠ THIS IS NOT OUR IMPLEMENTATION. FluidServe's decision uses the class's
distribution through the expected remaining tokens and the probability of
finishing inside a horizon, and it re-evaluates a held request every 500 ms; the
figure says what such a re-evaluation has available, not what our code does with
it. The end-to-end value of the finest possible version of the first axis was
measured separately and is small: giving the policy each request's exact length
moved the score by -0.2 points at 35 req/s (EXP-64).

POPULATION. EXP-108 at 10 req/s, four arms and both repeats pooled, completed
requests only, 34,598 requests. No arm rejects at that rate, so the lengths are
the workload's rather than what saturation left behind.

    python3 paper_figures/fig_length_refinement.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis_scripts", "request_level"))

from paper_style import COL_W, STYLE, GRID, save  # noqa: E402
from exp22_fluidserve import load_run, CLASSES, CLASS_COLORS  # noqa: E402

RUNS = "results/*exp108r?_*_rpm_600"
# The dataset each class replays, not the class name: a reader who wants to know
# what "deep research" is has to be told where its requests come from, and the
# three sources are what makes the length distributions what they are.
#   chat          ShareGPT (anon8231489123/ShareGPT_Vicuna_unfiltered), multi-turn
#                 conversation turns replayed as independent requests
#   deepresearch  Search Arena (lmarena-ai/search-arena-24k), a question plus K
#                 search-grounded notes, synthesised into a report
#   swe           SWE-bench Lite, recorded coding-agent calls replayed verbatim
LABEL = {"chat": "ShareGPT", "deepresearch": "Search Arena", "swe": "SWE-bench"}
POOL_C, CLS_C = "#444444", "#1f77b4"
# ⚠ PANEL (b) DOES NOT USE THE PROJECT'S CLASS COLOURS. Elsewhere the three
# classes are blue / orange / red (`exp22_fluidserve.CLASS_COLORS`); here the
# agent class is green and deep research a lighter orange, on request
# (2026-09-09), because the panel sits beside a two-line plot in grey and blue
# and the saturated red pulled the eye to the smallest of the three datasets.
# The consequence has to be stated: a colour in this panel does not mean the
# same class as the same colour in `class_mix_hour.pdf` or
# `workload_lengths.pdf`.
DS_COLORS = {"chat": "#1f77b4", "deepresearch": "#fdae61", "swe": "#4daf4a"}
# ⚠ WHERE THE CURVE STOPS IS A PROPERTY OF THE WORKLOAD, NOT A CHOICE OF RANGE.
# Panel (b)'s axis reaches 3,000 tokens because a few requests are that long;
# panel (a) asks a different question -- how well the length of a request that
# has produced j tokens can be estimated -- and it can only be asked of the
# requests still running at j. That population thins fast: 18.5% of requests are
# still running at 800 tokens, 9.4% at 1,000, 3.6% at 1,200 and 1.2% at 1,500,
# where the estimate swings to 18% and 35% on 430 requests. The grid stops at
# 1,200, the last point where every series clears MIN_ALIVE.
GRID_J = [0, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 1000]
# ⚠ A SERIES STOPS WHERE ITS SURVIVORS RUN OUT (2026-09-09). Each point is
# computed on the requests still running at that j, and for the agent class that
# is 2,031 at j = 400 but 156 at 700 and 52 at 800. Drawn to 800 its curve fell
# to 5.9% and then rose to 11.8%, which reads as a real reversal and is not one:
# the 95% bootstrap interval at j = 800 is [6.2, 14.1] and at 700 is [4.1, 8.2],
# so the two points do not separate. Every series is therefore cut at the last j
# where at least MIN_ALIVE of its requests were still running, and the counts
# are in the CSV.
MIN_ALIVE = 200
FIG_H = 1.60


def population():
    frames = []
    for d in sorted(glob.glob(os.path.join(ROOT, RUNS))):
        if "PRERUN" in d:
            continue
        r = load_run(d)
        if r is None or r.empty:
            continue
        frames.append(r[(~r["rejected"]) & (~r["errored"]) & (~r["cutoff"])])
    a = pd.concat(frames)
    a["out"] = pd.to_numeric(a["output_tokens"], errors="coerce")
    a = a.dropna(subset=["out"])
    return a[a["out"] > 0]


def density(v, lo, hi):
    """Kernel density over log10(v), evaluated on a grid in the same space.

    In log space because the lengths run from 2 to about 8,000 tokens: one
    Gaussian kernel cannot serve both ends on a linear axis. The estimate
    integrates to 1 against d(log10 x), so its HEIGHT is a density per decade
    -- not a number to read off an axis, which is why the y axis carries none --
    and the AREA under a curve between two x values, read on the log axis as
    drawn, is the share of that population's requests in the range.
    """
    x = np.log10(np.asarray(v, dtype=float))
    grid = np.linspace(np.log10(lo), np.log10(hi), 512)
    return 10.0 ** grid, gaussian_kde(x)(grid)


def err_at(target, pool, j, boot=1000, seed=0):
    """Median |estimate - actual| / actual among requests alive at j, with a
    bootstrap interval.

    `pool` supplies the estimator (the empirical remaining distribution) and
    `target` the requests being scored; passing the same array for both is the
    per-dataset case, passing the whole mixture as `pool` is the aggregated one.

    ⚠ THE INTERVAL IS A RESAMPLING INTERVAL OVER THE REQUESTS, not a spread over
    repeats. It says how much of the curve is the finite number of requests
    still running at that point -- which is what makes the far right of a short
    dataset's curve unreadable -- and it does not cover run-to-run variation,
    which this figure does not measure because the lengths are a property of the
    workload rather than of a run.
    """
    alive = target[target > j]
    ref = pool[pool > j]
    if len(alive) < MIN_ALIVE or len(ref) < MIN_ALIVE:
        return np.nan, np.nan, np.nan, len(alive)
    est = j + float(np.median(ref - j))
    e = float(np.median(np.abs(est - alive) / alive))
    rng = np.random.default_rng(seed + j)
    bs = np.empty(boot)
    for b in range(boot):
        s_ = rng.choice(alive, len(alive), replace=True)
        bs[b] = np.median(np.abs((j + np.median(s_ - j)) - s_) / s_)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return e, float(lo), float(hi), len(alive)


def main():
    weighted = "--weighted" in sys.argv
    a = population()
    allL = a["out"].to_numpy()
    byc = {c: a.loc[a["class"] == c, "out"].to_numpy() for c in CLASSES}

    rows = []
    for j in GRID_J:
        e_pool, lo_pool, hi_pool, n_pool = err_at(allL, allL, j)
        per = {c: err_at(v, v, j) for c, v in byc.items()}
        # The per-dataset estimator scored over the SAME population as the
        # aggregated one: each dataset's error weighted by how many of its
        # requests are still running at j, not by how many it has in total. The
        # interval is combined the same way, which understates it slightly
        # because it ignores the covariance between the three.
        num = [(e, lo, hi, n) for e, lo, hi, n in per.values() if n and e == e]
        tot = sum(n for *_, n in num)
        e_cls = sum(e * n for e, _, _, n in num) / tot if tot else np.nan
        lo_cls = sum(lo * n for _, lo, _, n in num) / tot if tot else np.nan
        hi_cls = sum(hi * n for _, _, hi, n in num) / tot if tot else np.nan
        row = dict(tokens_produced=j, err_pooled_pct=100 * e_pool,
                   err_pooled_lo=100 * lo_pool, err_pooled_hi=100 * hi_pool,
                   err_class_pct=100 * e_cls, err_class_lo=100 * lo_cls,
                   err_class_hi=100 * hi_cls, n_alive=n_pool)
        for c in CLASSES:
            row[f"err_{c}_pct"] = 100 * per[c][0]
            row[f"n_alive_{c}"] = per[c][3]
        rows.append(row)
    tab = pd.DataFrame(rows)

    with plt.rc_context({**STYLE, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "axes.labelsize": 7}):
        fig, ax = plt.subplots(1, 2, figsize=(COL_W, FIG_H))
        x = tab["tokens_produced"]
        for key, col, mk in (("pooled", POOL_C, "o"), ("class", CLS_C, "s")):
            y = tab[f"err_{key}_pct"]
            lo = y - tab[f"err_{key}_lo"]
            hi = tab[f"err_{key}_hi"] - y
            ax[0].errorbar(x, y, yerr=[lo, hi], color=col, lw=1.1, marker=mk,
                           ms=2.2, elinewidth=0.6, capsize=1.2)
        h0 = [plt.Line2D([], [], color=POOL_C, lw=1.1, marker="o", ms=2.6),
              plt.Line2D([], [], color=CLS_C, lw=1.1, marker="s", ms=2.6)]
        ax[0].legend(h0, ["Aggregated", "Workload-level"], loc="lower left",
                     frameon=False, fontsize=5.8, handlelength=1.3,
                     handletextpad=0.35, borderaxespad=0.2)
        # (b) is the distribution the left panel is about: where the lengths of
        # each dataset actually sit. The pooled curve is grey because it is not
        # a dataset.
        lo, hi = 30.0, 4.0e3
        xa, ya = density(a["out"].to_numpy(), lo, hi)
        ax[1].plot(xa, ya, color=POOL_C, lw=1.0)
        ax[1].fill_between(xa, ya, color=POOL_C, alpha=0.12, lw=0)
        h1 = [plt.Line2D([], [], color=POOL_C, lw=1.1)]
        lab1 = ["Aggregated"]
        for c in CLASSES:
            v = a.loc[a["class"] == c, "out"].to_numpy()
            xc, yc = density(v, lo, hi)
            if weighted:
                # Each curve scaled by its share of the arrivals, so the three
                # SUM to the grey one and every height and area on the panel is
                # an amount: the reader's instinct that bulk means quantity is
                # then correct, which it is not when each class integrates to
                # one on its own.
                yc = yc * len(v) / len(a)
            ax[1].plot(xc, yc, color=DS_COLORS[c], lw=1.0)
            ax[1].fill_between(xc, yc, color=DS_COLORS[c], alpha=0.12, lw=0)
            h1.append(plt.Line2D([], [], color=DS_COLORS[c], lw=1.1))
            lab1.append(LABEL[c])
        ax[1].legend(h1, lab1, loc="upper left", frameon=False, fontsize=5.4,
                     handlelength=1.2, handletextpad=0.3, borderaxespad=0.2)
        ax[1].set_xscale("log")
        ax[1].set_xlim(lo, hi)
        # Ticks across the whole span, not the two decades matplotlib picks:
        # the axis runs from 30 to 4,000 and a reader should be able to place
        # both ends. Thousands abbreviated so five labels fit under 1.4 in.
        xt = [30, 100, 300, 1000, 3000]
        ax[1].set_xticks(xt)
        ax[1].set_xticklabels(["30", "100", "300", "1k", "3k"])
        ax[1].xaxis.set_minor_locator(plt.matplotlib.ticker.NullLocator())
        ax[1].set_ylim(0, None)
        # A density per decade is not a quantity to read off an axis; the shape
        # and the area carry it.
        ax[1].set_yticks([])
        ax[1].set_ylabel("Density", labelpad=1.5)
        ax[1].set_xlabel("Output Tokens", labelpad=1.5)

        ax[0].set_xlim(0, GRID_J[-1])
        ax[0].set_xticks([0, 250, 500, 750, 1000])
        ax[0].set_ylim(0, 50)
        ax[0].set_yticks([0, 10, 20, 30, 40, 50])
        ax[0].set_xlabel("Produced Tokens", labelpad=1.5)
        ax[0].set_ylabel("Prediction Error (%)", labelpad=1.5)
        for a_ in ax:
            a_.grid(axis="both" if a_ is ax[0] else "x", **GRID)
            a_.set_axisbelow(True)
        # The two captions are placed by the FIGURE, not as a second line of
        # each x label: (b)'s tick labels are set in mathtext and are taller
        # than (a)'s plain numerals, so labels attached to the axes come out at
        # two different heights. One y for both, centred on each panel.
        fig.tight_layout(rect=(0, 0.10, 1, 1), pad=0.35, w_pad=1.0)
        for a_, cap in zip(ax, ("(a) Prediction Error", "(b) Length Distribution")):
            bb = a_.get_position()
            fig.text((bb.x0 + bb.x1) / 2, 0.012, cap, ha="center", va="bottom",
                     fontsize=7)
        save(fig, os.path.join(HERE, "length_refinement"
                               + ("_weighted" if weighted else "") + ".pdf"))

    print(tab.round(1).to_string(index=False))
    # Panel (b)'s summary beside panel (a)'s curve, in one file: the CSV holds
    # what each panel draws, and a distribution is quoted by its percentiles.
    dist = [dict(group="Aggregated", n=len(a),
                 **{f"p{q}": float(np.percentile(a["out"], q))
                    for q in (10, 50, 90)})]
    for c in CLASSES:
        v = a.loc[a["class"] == c, "out"]
        dist.append(dict(group=LABEL[c], n=len(v),
                         **{f"p{q}": float(np.percentile(v, q))
                            for q in (10, 50, 90)}))
    dist = pd.DataFrame(dist)
    dist["p90_over_p10"] = dist["p90"] / dist["p10"]
    print(dist.round(1).to_string(index=False))
    out = os.path.join(HERE, "length_refinement.csv")
    tab.to_csv(out, index=False, float_format="%.3f")
    dist.to_csv(out.replace(".csv", "_distribution.csv"), index=False,
                float_format="%.3f")
    print(f"wrote {out}  ({len(tab)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
