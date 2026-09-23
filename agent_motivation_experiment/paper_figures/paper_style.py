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
import os
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
    # Nimbus Roman first (2026-09-18, at the author's request): it is URW's
    # Times, metrically the same as the body face of the USENIX and ACM
    # templates, so type in a figure matches type in the text. DejaVu Serif,
    # matplotlib's own, is wider and was what every figure carried before.
    "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif",
                   "DejaVu Serif"],
    # math in the same face as the text: "$5.8\\times$" beside "Arrival Rate"
    "mathtext.fontset": "custom",
    "mathtext.rm": "Nimbus Roman", "mathtext.it": "Nimbus Roman:italic",
    "mathtext.bf": "Nimbus Roman:bold",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "axes.linewidth": 0.5, "legend.fontsize": 8, "legend.frameon": False,
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

# The grey is set directly and the line is drawn at full opacity, rather than
# fading matplotlib's default #b0b0b0 with an alpha. Both routes reach the same
# printed colour on white, but alpha also fades the grid against anything drawn
# UNDER it and makes the printed value depend on the background, so the colour
# is the honest control. #909090 is one step darker than the default; the grid
# stays lighter than the 0.5 pt frame so it does not read as a fourth curve.
GRID = dict(ls=":", lw=0.5, color="#909090", alpha=1.0)

# Arm colours, identical to the analysis scripts so a colour means the same
# policy in an exploratory figure and in the paper.
ARM_COLOR = {
    "fluidserve": "#1f77b4",
    "polyserve": "#d62728",
    "slo": "#2ca02c",
    "loadbalance": "#9467bd",
    "fluidserveflat": "#ff7f0e",
    # llm-d, brown, as `redraw_hour_trace_exp71_four.sh` assigns it. NOTE that
    # the same script gives FluidServe v0.2 cyan (#17becf) rather than the blue
    # used here and in every other paper figure; the paper figures keep blue so
    # that one colour means one policy across the whole paper, and v0.2 is not
    # a different policy from the one those figures already call FluidServe.
    "llmd": "#8c564b",
    # The vLLM router's default cache-aware policy, as `exp22_fluidserve.py`
    # assigns it. Distinct from `loadbalance`'s #9467bd, which is a different
    # purple; the two never appear on the same figure.
    "vllmrouter": "#7b3294",
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


# Key order and key names, one place for every paper figure (2026-09-17, at the
# author's request). The KEY is ordered FluidServe first; the bars and lines
# keep the order each figure draws them in, which is the ramp's order.
# The star marks the two arms whose caption carries a qualification: the vLLM
# router has no admission control, and "Llumnix" here is Llumnix's SLO-aware
# policy (`--scheduling-policy slo`), not its load balancer.
# One key size for every paper figure: the size class_mix_hour draws its key at
# (label_size 5 + 1.5), chosen 2026-09-18 by the author as the common one.
KEY_FS = 6.5
# Square swatches: matplotlib's default key patch is a wide rectangle, and a
# square reads as a colour sample rather than as a bar.
KEY_SQUARE = dict(handlelength=1.0, handleheight=1.0, handletextpad=0.35)


def square_handler(handles):
    """handler_map that draws every Patch handle as a square."""
    from matplotlib.legend_handler import HandlerPatch
    from matplotlib.patches import Rectangle, Patch

    class _Sq(HandlerPatch):
        def create_artists(self, legend, orig, xd, yd, width, height,
                           fontsize, trans):
            # ⚠ SIDE FROM THE TYPE, NOT FROM THE HANDLE BOX (2026-09-18). The
            # box is taller than the letters, so a square filling it sat below
            # the text's midline and read as misaligned. The side is the
            # capital height of the key's type and the square is centred on the
            # box, which puts it on the same midline as the words beside it.
            side = 0.70 * fontsize
            # ⚠ CENTRED ON THE TEXT, NOT ON THE HANDLE BOX (2026-09-18). The
            # box's middle sits about a fifth of the type size above the middle
            # of the words beside it, which read as the swatch floating high;
            # the offset below was measured off a 600 dpi render of
            # class_latency_outcome_grid.pdf and drives that difference to
            # nothing.
            yc = yd + (height - side) / 2.0 - 0.21 * fontsize
            return [Rectangle((xd + (width - side) / 2.0, yc), side, side,
                              facecolor=orig.get_facecolor(),
                              edgecolor=orig.get_edgecolor(),
                              linewidth=orig.get_linewidth(),
                              hatch=orig.get_hatch(), transform=trans)]

    return {h: _Sq() for h in handles if isinstance(h, Patch)}


LEGEND_ORDER = ["FluidServe", "llm-d", "PolyServe", "Llumnix", "vLLM"]
LEGEND_NAME = {"Llumnix": "Llumnix*", "Llumnix SLO": "Llumnix*",
               "vLLM": "vLLM*", "vLLM-router": "vLLM*"}


def legend_items(handles, labels):
    """(handles, labels) reordered for the key and renamed.

    A label not in LEGEND_ORDER sorts with the arm whose name it starts with
    ("FluidServe w/o Affinity" follows FluidServe) and keeps its own name.
    """
    def rank(i_lab):
        i, lab = i_lab
        for r, name in enumerate(LEGEND_ORDER):
            if lab == name or lab.startswith(name):
                return (r, i)
        return (len(LEGEND_ORDER), i)

    order = sorted(range(len(labels)), key=lambda i: rank((i, labels[i])))
    return ([handles[i] for i in order],
            [LEGEND_NAME.get(labels[i], labels[i]) for i in order])


FINAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final")


def final(name):
    """Path of an output inside `paper_figures/final/`, created on demand.

    The figures the paper actually uses are written there (2026-09-17, at the
    author's request) so that they are not mixed with the exploratory ones in
    this directory. Scripts pass the FILE NAME, never a directory.
    """
    os.makedirs(FINAL, exist_ok=True)
    return os.path.join(FINAL, name)


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
