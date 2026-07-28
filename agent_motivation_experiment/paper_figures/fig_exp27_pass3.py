#!/usr/bin/env python3
"""Paper figure: EXP-27 pass 3 (v22) rate sweep — attainment and token goodput.

Two PDFs, each two side-by-side panels, sized for ONE COLUMN of a two-column
USENIX paper. They differ only in whether the offered-denominator curve is
drawn; see `build()` and the "Content" note below.

  exp27_pass3_attainment_goodput_offered.pdf    with FluidServe's offered curve
  exp27_pass3_attainment_goodput_admitted.pdf   admitted denominator only

Sizing. `usenix2019_v3.sty` sets `\\textwidth=7in` and `\\columnsep=0.33in`, so a
single column is 3.335 in wide. The figure is drawn at exactly that width, which
makes `\\includegraphics[width=\\columnwidth]` inside a plain `figure` a scale
factor of 1.0, so the 8 pt type set here is 8 pt on the page. Any re-scaling in
LaTeX (`scale=`, `width=` something other than `\\columnwidth`, `\\resizebox`)
multiplies the effective font size by the same factor and breaks that; the
earlier 7 in version of this figure rendered at roughly 3.8 pt when placed in one
column for exactly that reason. If the figure is ever moved into a `figure*`
spanning both columns, redraw it at COL_W = 7.0 rather than stretching it.

Side by side inside 3.335 in leaves about 1.3 in of plotting area per panel,
which the default labelling does not fit into. Three things buy that room back
and are the reason this figure is not just the wide version rescaled:

  - one x axis label centred under both panels rather than one per panel;
  - goodput tick labels carry a k suffix (6k, 12k, 18k) instead of being
    written out (6000, 12000, 18000), which is three characters less of axis
    width and lets both y axis labels stay on a single line;
  - the rejection annotations set at 6 pt rather than the 8 pt body size.

Everything an axis is read by — tick labels, axis labels, legend — stays at 8 pt.

Content. The two panels have to stay together: a routing/admission policy can
raise SLO attainment by refusing work, so the attainment panel alone is not a
claim about capacity, and the goodput panel is the quantity that rejections
cannot inflate. For the same reason the attainment panel carries both
denominators for the arm that rejects:

  solid   admitted — the denominator is the requests the system accepted, so a
          rejected request leaves the population entirely.
  dotted  offered  — every request that arrived is in the denominator and a
          rejection counts as a violation.

PolyServe rejected zero requests at every rate in this pass, so its offered
curve is numerically identical to its admitted curve and is not drawn in either
version: two coincident lines and a second legend entry would suggest a
distinction that the data does not contain. FluidServe's two curves separate at
60 and 80 req/s, and the gap between them is exactly what its rejections cost.

The admitted-only version drops that curve and the rejection annotations with
it, so nothing in that figure records that FluidServe refused 9% of arrivals at
60 req/s and 30% at 80 req/s. Attainment over an admitted denominator is not
interpretable without that number — a policy that refuses everything scores
100% — so if this is the version that goes in the paper, THE CAPTION OR THE
SURROUNDING TEXT HAS TO STATE THE REJECTION RATES. The goodput panel is a
partial guard, since rejected work produces no tokens, but it reports a rate
rather than the fraction of arrivals that were turned away.

Data, metric definitions and run selection are unchanged from the exploratory
figures (`analysis_scripts/request_level/exp27_figures.py`, `fig_split`), which
this script imports rather than reimplements — only the styling differs.

    python3 paper_figures/fig_exp27_pass3.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis_scripts", "request_level"))

from exp27_figures import collect, rps, ARM_C, ARM_L  # noqa: E402

# m1 balanced mix, four engines, 8 min per condition, BOTH repeats of pass 3
# (r1 and r2), so every point is n=2 and the bars are min..max. The exploratory
# PNGs this figure replaces were drawn from r1 alone, before r2 finished; the
# two repeats agree to within 3 points of attainment at every rate, so the
# reading is unchanged, but a single measurement per condition is not a basis
# for a published claim when the session-to-session spread on this workload is
# larger than most of the differences being compared.
RUNS = ["results/*exp27p3*"]

COL_W = 3.335   # USENIX single column, inches. See the sizing note above.
FIG_H = 1.95
KTOK = 1000.0   # goodput tick labels are written in thousands (6k, 12k, ...)

# Same rcParams as the project-wide PAPER_STYLE, with the two sizes that it
# raises to 9 pt (axes labels and titles) pulled back to 8 pt, and markers and
# line widths trimmed for the smaller drawing area.
STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "axes.linewidth": 0.7, "legend.fontsize": 8, "legend.frameon": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.2, "lines.markersize": 3.5,
    # Type 42 (TrueType) rather than the default Type 3, which several
    # camera-ready checkers reject.
    "pdf.fonttype": 42, "ps.fonttype": 42,
}
GRID = dict(ls=":", lw=0.5, alpha=0.6)
XLABEL = "Offered rate (request/s)"


def agg(df, arm):
    """Per-rate mean and min/max over the repeats of one arm.

    Two repeats per condition, so the bars span the two observations rather
    than estimating a confidence interval; they are the observed spread and
    should be described that way in the caption.
    """
    return df[df.arm == arm].groupby("rpm").agg(
        sloA=("attain", "mean"), sloALo=("attain", "min"), sloAHi=("attain", "max"),
        sloO=("attain_off", "mean"), sloOLo=("attain_off", "min"),
        sloOHi=("attain_off", "max"),
        gp=("goodput", "mean"), gpLo=("goodput", "min"), gpHi=("goodput", "max"),
        rej=("rejected", "mean"),
    ).reset_index().sort_values("rpm")


def ktick(v, _pos):
    """Thousands with a k suffix, so the axis label can stay 'Goodput token'.

    Zero is written plain: '0k' is not a quantity anyone writes, and the tick
    is the axis origin rather than a value being compared.
    """
    return "0" if v == 0 else f"{v / KTOK:g}k"


def build(df, out, show_offered):
    """One figure. `show_offered` adds the second attainment denominator.

    It also controls the rejection annotations, which belong to the same
    statement: with `show_offered=False` the figure reads as "FluidServe holds
    87% attainment at 80 req/s" and says nothing about the 30% of arrivals it
    refused to admit. That is a deliberate choice for readability, not an
    oversight, and it moves the obligation to report the rejection rate into
    the caption. See the module docstring.
    """
    arms = [a for a in ARM_C if a in set(df["arm"])]
    xticks = rps(sorted(df.rpm.unique()))

    with plt.rc_context(STYLE):
        fig, (ax_a, ax_g) = plt.subplots(1, 2, figsize=(COL_W, FIG_H))

        handles, labels = [], []
        for arm in arms:
            d, c = agg(df, arm), ARM_C[arm]
            # Circle for PolyServe, square for FluidServe, and the same marker
            # for that arm in both panels, so one legend describes both.
            mk = "s" if arm == "fluidserve" else "o"

            h = ax_a.errorbar(rps(d.rpm), d.sloA,
                              yerr=[d.sloA - d.sloALo, d.sloAHi - d.sloA],
                              color=c, ls="-", marker=mk, capsize=2,
                              mec="white", mew=0.5)
            handles.append(h)
            labels.append(ARM_L[arm])

            # Only for an arm that actually rejects: with zero rejections the
            # offered curve lies exactly on the admitted one.
            rejects = bool((d.rej > 0).any())
            if show_offered and rejects:
                h = ax_a.errorbar(rps(d.rpm), d.sloO,
                                  yerr=[d.sloO - d.sloOLo, d.sloOHi - d.sloO],
                                  color=c, ls=":", marker="^", ms=3.0, alpha=0.6,
                                  capsize=2, mec="white", mew=0.5)
                handles.append(h)
                labels.append(f"{ARM_L[arm]}, offered")
                for _, r in d.iterrows():
                    if r.rej >= 1.0:
                        # Below and to the left of the offered point, which is
                        # empty space. Not drawn in the admitted-only version:
                        # see the note in this function's docstring about what
                        # the caption then has to carry.
                        ax_a.annotate(f"{r.rej:.0f}% rej.", (r.rpm / 60.0, r.sloO),
                                      fontsize=6, color=c, ha="right", va="top",
                                      xytext=(-2, -1), textcoords="offset points")

            ax_g.errorbar(rps(d.rpm), d.gp,
                          yerr=[d.gp - d.gpLo, d.gpHi - d.gp],
                          color=c, ls="-", marker=mk, capsize=2,
                          mec="white", mew=0.5)

        # Single-line y labels. The unit lives in the tick labels on the right
        # panel (6k, 12k, ...) rather than in the axis label, which keeps both
        # labels to one line and gives the panels the width back.
        ax_a.set_ylabel("SLO attainment (%)")
        ax_a.set_ylim(0, 108)
        ax_a.set_yticks([0, 25, 50, 75, 100])
        ax_g.set_ylabel("Goodput token (t/s)")
        # Ticks every 5k up to 20k, with the top of the axis above FluidServe's
        # 60 req/s peak (21.5k). The automatic locator stopped at 18k and left
        # that peak between the last gridline and the frame, where its value
        # cannot be read off the axis; with a 20k tick it reads as "just over
        # 20k". Five ticks here also match the five on the left panel.
        ax_g.set_ylim(0, 23000)
        ax_g.set_yticks([0, 5000, 10000, 15000, 20000])
        ax_g.yaxis.set_major_formatter(FuncFormatter(ktick))

        for ax in (ax_a, ax_g):
            ax.set_xticks(xticks)
            ax.set_xlim(xticks.min() - 4, xticks.max() + 4)
            ax.grid(axis="both", **GRID)
            ax.set_axisbelow(True)

        # One row. Three entries fit only with the handles and the column gap
        # trimmed; at the matplotlib defaults they overrun the 3.335 in canvas.
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 0.845), frameon=False,
                   columnspacing=0.8, handlelength=1.5, handletextpad=0.4,
                   borderaxespad=0.0)
        # No bbox_inches="tight" on the save: it crops the canvas to the ink,
        # which makes the PDF narrower than COL_W, and \includegraphics then
        # scales it back UP to \columnwidth and multiplies every font size by
        # the same factor. The canvas has to stay exactly COL_W wide, so the
        # layout is fitted inside it with rect instead.
        fig.tight_layout(rect=(0, 0.10, 1, 0.835), w_pad=0.8, pad=0.35)
        # One x label centred under both panels: repeating it costs a line of
        # figure height and reads as two different quantities.
        fig.text(0.5, 0.005, XLABEL, ha="center", va="bottom", fontsize=8)

        fig.savefig(out)
        w, h = fig.get_size_inches()
        plt.close(fig)

    print(f"wrote {out}  ({w:.3f} x {h:.2f} in; include with "
          f"width=\\columnwidth, no scaling)")


def main():
    df = collect([os.path.join(ROOT, p) for p in RUNS])
    if df.empty:
        print("no runs matched; run from the experiment root", file=sys.stderr)
        return 1
    # Two versions of the same figure. Which one belongs in the paper depends on
    # whether the surrounding text has already established that FluidServe
    # rejects and PolyServe does not; the offered curve is the honest version
    # and the admitted-only one is the readable version.
    build(df, os.path.join(HERE, "exp27_pass3_attainment_goodput_offered.pdf"),
          show_offered=True)
    build(df, os.path.join(HERE, "exp27_pass3_attainment_goodput_admitted.pdf"),
          show_offered=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
