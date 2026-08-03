#!/usr/bin/env python3
"""Shared style and geometry for the paper figures in this directory.

The one thing every figure here has in common is that it is drawn at its FINAL
PHYSICAL SIZE, so that `\\includegraphics` applies a scale factor of 1.0 and the
type lands on the page at the size it was set at. Everything below exists to
make that hold.

Do not add plotting logic here. This is constants plus two helpers that encode
the two mistakes that cost the most time; the figures themselves stay readable
as single files.
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# USENIX single column. `usenix2019_v3.sty` sets \textwidth=7in and
# \columnsep=0.33in, so a column is (7 - 0.33) / 2. ACM sigconf is 3.33in, close
# enough that the same figure serves both; the full text width is 7.0.
COL_W = 7.0 / 2 - 0.33 / 2      # 3.335
TEXT_W = 7.0

KTOK = 1000.0

# The project-wide PAPER_STYLE with the two sizes it raises to 9 pt (axes labels
# and titles) pulled back to 8 pt, because at COL_W the figure is not scaled and
# 8 pt here is 8 pt on the page. Markers and line widths are trimmed for the
# smaller drawing area.
STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "axes.linewidth": 0.7, "legend.fontsize": 8, "legend.frameon": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.3,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.2, "lines.markersize": 3.5,
    # Type 42 (TrueType) rather than the default Type 3, which several
    # camera-ready checkers reject.
    "pdf.fonttype": 42, "ps.fonttype": 42,
}

GRID = dict(ls=":", lw=0.5, alpha=0.6)

# Arm colours, identical to the analysis scripts so a colour means the same
# policy in an exploratory figure and in the paper.
ARM_COLOR = {
    "fluidserve": "#1f77b4",
    "polyserve": "#d62728",
    "slo": "#2ca02c",
    "loadbalance": "#9467bd",
    "fluidserveflat": "#ff7f0e",
}
# One marker per arm, reused (dotted, faded) for that arm's offered curve so the
# two denominators of one policy read as the same policy.
ARM_MARKER = {
    "fluidserve": "s", "polyserve": "o", "slo": "^", "loadbalance": "v",
    "fluidserveflat": "D",
}


def ktick(v, _pos):
    """Thousands with a k suffix: 6000 -> '6k'.

    Zero is written plain. '0k' is not a quantity anyone writes, and that tick
    is the axis origin rather than a value being compared.
    """
    return "0" if v == 0 else f"{v / KTOK:g}k"


def kfmt():
    return FuncFormatter(ktick)


def save(fig, path):
    """Write the PDF without cropping the canvas.

    NEVER `bbox_inches="tight"` here. It crops to the ink, which makes the PDF
    narrower than the width it was designed for; `\\includegraphics` then scales
    it back UP to \\columnwidth and multiplies every font size by the same
    factor. The canvas has to stay exactly the width it was created at, so each
    figure fits its layout inside that canvas with `tight_layout(rect=...)`.
    """
    fig.savefig(path)
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"wrote {path}  ({w:.3f} x {h:.2f} in; include at that width, "
          f"no scaling)")
