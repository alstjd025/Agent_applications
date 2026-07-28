#!/usr/bin/env python3
"""Paper figure: EXP-27 pass 3 (v22) rate sweep — attainment and token goodput.

One PDF, two side-by-side panels, sized for ONE COLUMN of a two-column USENIX
paper.

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
  - goodput plotted in THOUSANDS of tokens/s, so its y tick labels are two
    characters instead of five and the axis costs ~0.2 in less width;
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
curve is numerically identical to its admitted curve and is not drawn: two
coincident lines and a second legend entry would suggest a distinction that the
data does not contain. FluidServe's two curves separate at 60 and 80 req/s, and
the gap between them is exactly what its rejections cost; the rejection rate is
annotated on those points.

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
from matplotlib.ticker import MaxNLocator  # noqa: E402

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
KTOK = 1000.0   # goodput is drawn in thousands of output tokens/s

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


def main():
    df = collect([os.path.join(ROOT, p) for p in RUNS])
    if df.empty:
        print("no runs matched; run from the experiment root", file=sys.stderr)
        return 1
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
            if (d.rej > 0).any():
                h = ax_a.errorbar(rps(d.rpm), d.sloO,
                                  yerr=[d.sloO - d.sloOLo, d.sloOHi - d.sloO],
                                  color=c, ls=":", marker="^", ms=3.0, alpha=0.6,
                                  capsize=2, mec="white", mew=0.5)
                handles.append(h)
                labels.append(f"{ARM_L[arm]}, offered")
                for _, r in d.iterrows():
                    if r.rej >= 1.0:
                        # Below and to the left of the offered point. The
                        # midpoint between the two curves, which reads as
                        # "this is the gap", is already occupied by the
                        # admitted curve at both of these rates.
                        ax_a.annotate(f"{r.rej:.0f}% rej.",
                                      (r.rpm / 60.0, r.sloO),
                                      fontsize=6, color=c, ha="right", va="top",
                                      xytext=(-2, -1), textcoords="offset points")

            ax_g.errorbar(rps(d.rpm), d.gp / KTOK,
                          yerr=[(d.gp - d.gpLo) / KTOK, (d.gpHi - d.gp) / KTOK],
                          color=c, ls="-", marker=mk, capsize=2,
                          mec="white", mew=0.5)

        # Two-line y labels: at 8 pt the one-line forms are longer than the
        # panel is tall and get clipped.
        ax_a.set_ylabel("SLO attainment\n(%), admitted")
        ax_a.set_ylim(0, 108)
        ax_a.set_yticks([0, 25, 50, 75, 100])
        ax_g.set_ylabel("Output token\ngoodput (k/s)")
        ax_g.set_ylim(0, None)
        ax_g.yaxis.set_major_locator(MaxNLocator(4))

        for ax, cap in ((ax_a, "(a)"), (ax_g, "(b)")):
            ax.set_xticks(xticks)
            ax.set_xlim(xticks.min() - 4, xticks.max() + 4)
            ax.grid(axis="both", **GRID)
            ax.set_axisbelow(True)
            # Panel label inside the axes, bottom right: the space above each
            # panel is taken by the legend, and the bottom-right corner is
            # empty in both panels.
            ax.text(0.97, 0.05, cap, transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=8)

        # Left column of the legend is the two arms, right column the second
        # denominator. Three entries on one row need 3.28 in of the 3.335 in
        # available and leave no margin for a font substitution.
        fig.legend(handles, labels, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, 0.815), frameon=False,
                   columnspacing=1.0, handlelength=1.8, handletextpad=0.5,
                   labelspacing=0.25, borderaxespad=0.0)
        # No bbox_inches="tight" on the save: it crops the canvas to the ink,
        # which makes the PDF narrower than COL_W, and \includegraphics then
        # scales it back UP to \columnwidth and multiplies every font size by
        # the same factor. The canvas has to stay exactly COL_W wide, so the
        # layout is fitted inside it with rect instead.
        fig.tight_layout(rect=(0, 0.10, 1, 0.805), w_pad=0.8, pad=0.35)
        # One x label centred under both panels: repeating it costs a line of
        # figure height and reads as two different quantities.
        fig.text(0.5, 0.005, XLABEL, ha="center", va="bottom", fontsize=8)

        out = os.path.join(HERE, "exp27_pass3_attainment_goodput.pdf")
        fig.savefig(out)
        plt.close(fig)

    w, h = fig.get_size_inches()
    print(f"wrote {out}  ({w:.3f} x {h:.2f} in before the tight bbox; "
          f"include with width=\\columnwidth, no scaling)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
