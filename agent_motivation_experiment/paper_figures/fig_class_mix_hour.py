#!/usr/bin/env python3
"""Paper figure: how each control plane mixed the three classes across the four
engines, over the hour-long trace whose arrival rate and class mixture both move.

  class_mix_hour.pdf       7.0 x 3.10 in, `figure*`, width=\\textwidth
      each engine's composition over the hour, as a share of what that engine
      held, so the picture does not depend on how much work each arm accepted
  class_mix_hour_abs.pdf   the same in requests rather than shares
  class_mix_hour.csv       exactly the values drawn in both
  class_mix_hour_abs_tbt.pdf  `--tbt`: the absolute version with a second axis
      per panel, that instance's mean inter-token latency over the same hour
  class_mix_hour_abs_tbt.csv  the drawn latency series

  rows    the four engines, ordered by how much chat they held over the whole
          hour, so ROW 1 IS ONE ENGINE FOR THE WHOLE PANEL and a reassignment
          appears as that row changing colour partway through
  columns the five control planes, in the paper's order
  x       minutes into the run; the band at time t is the mean number of
          requests of that class concurrently resident on that engine in the
          60 s window starting at t

`--tbt` PAIRS WHAT AN INSTANCE HELD WITH HOW FAST IT RAN. The bands say which
classes were resident; the line says the mean time between tokens the engine
itself reported in the same minute, smoothed with a 60 s box filter, on a shared
logarithmic right axis. What it is for: the pace an instance can offer is set by
what is on it, and this puts the two side by side per instance rather than in two
figures a reader has to align by eye.

⚠ THE LINE IS ONE NUMBER FOR ALL THE CLASSES ON THAT INSTANCE, so it must not be
read against a single class's budget. It is the token-weighted mean over every
token the engine emitted in that second, chat and deep research and agent
together, and the three budgets are 50, 100 and 75 ms. A panel whose line sits at
70 ms is not thereby violating the chat budget, and a panel at 45 ms is not
thereby meeting it for every class on it. Per-class pace is a different
measurement and this figure does not carry it.

THIS IS THE FIGURE THE STATIC ONE CANNOT BE. `class_mix_static.pdf` asks how the
mixture changes with the arrival rate, and its conditions hold one class mixture
fixed for eight minutes each. Here the class mixture moves on fifteen-minute
segments while the arrival rate follows a production trace, so the question is
whether an assignment made once stays right -- and an assignment that was made
once and never revisited shows up as four bands that never change.

WHY A WHOLE-HOUR BAR WOULD NOT DO. Summing an hour into one stacked bar per
engine spreads a concentration that MOVED and reads it as no concentration at
all: this is how an hour in which deep research sat at 99-100% on one engine at
every instant was once recorded as 32.6%. The numbers are in
`hour_summary.csv` beside the figure -- for each class, the effective number of
instances measured per 60 s window, the same measured by pooling the hour, and
how many times the engine holding the most of that class changed. FluidServe's
chat reads 1.64 per window against 2.16 pooled with four changes; llm-d's reads
3.45 against 3.96 with thirty. Neither arm is described by its pooled number.

⚠ THE DENOMINATOR IS ADMITTED WORK, and the arms differ enormously in how much
of the hour they accepted: rejection over this run is FluidServe 19.9%,
PolyServe 22.5%, Llumnix SLO 39.4%, llm-d 49.0%, vLLM router 0.0%. The share
panels are therefore the ones that can be compared straight across; in the
absolute panels the vLLM router column peaks at about 7,000 concurrent requests
on one engine against 200-430 for every other arm, because it accepts everything
and the excess waits, SO THAT COLUMN HAS ITS OWN Y SCALE and the scale is
printed in the panel.

⚠ RESIDENT MEANS ASSIGNED TO THAT ENGINE, NOT RUNNING ON IT. A request counts
from when the client sent it to when it finished, so time spent waiting in that
engine's queue is counted. That is the intended reading -- the request is that
engine's work either way -- but it is why an arm that never rejects shows
thousands of resident requests rather than a batch-sized number.

⚠ THE vLLM ROUTER COLUMN IS DRAWN ON 74.9% OF ITS ADMITTED REQUESTS. Engine
attribution goes through the scheduler's own dispatch log, whose lines are
dropped under high load; the other four arms are at 99.9-100%. Its column is
labelled with that number.

DATA. EXP-109 (2026-08-31), repeat 1 of each of the five arms, through
`build_class_mix_tables.py`. Repeat 2 is in the same tables and the summary
records that the picture reproduces. The window is 60 s and the analysis window
is `load_run`'s, so 60 s of warmup and 20 s of drain are cut.

    python3 paper_figures/fig_class_mix_hour.py
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import paper_style as ps  # noqa: E402

RL = os.path.join(ROOT, "analysis_scripts", "request_level")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ECO = _load("eco", os.path.join(RL, "engine_class_occupancy.py"))
DKG = _load("dkg", os.path.join(RL, "decode_kv_growth.py"))
EV = _load("exp41ev", os.path.join(RL, "exp41_engine_view.py"))
EW = _load("engwin", os.path.join(HERE, "fig_engine_window.py"))

MIX = os.path.join(ROOT, "results", "aggregate_analysis", "class_mix")
CSV_MIX = os.path.join(MIX, "hour_engine_mix.csv")
CSV_SUM = os.path.join(MIX, "hour_summary.csv")
REPEAT = "exp109r1"

ARMS = [("vllmcachet75", "vLLM-router"), ("polyservept75", "PolyServe"),
        ("slot75", "Llumnix SLO"), ("llmdslot75", "llm-d"),
        ("fsv3capgnofrct75", "FluidServe")]
# ⚠ THESE ARE NOT THE PROJECT-WIDE CLASS COLOURS. Every other figure uses
# `exp22_fluidserve.CLASS_COLORS` (chat #1f77b4, deep research #ff7f0e, agent
# #d62728); this figure was given the three-class RdYlBu ramp instead, on
# 2026-09-07, because a stacked area fills large blocks of colour and the
# saturated tab10 trio reads as three warnings rather than three categories.
# The consequence has to be stated rather than absorbed: a colour does NOT mean
# the same class here as in `class_mix_static.pdf` beside it, so either that
# figure moves to this ramp too or the two are never shown on one page.
CLASS_COLOR = {"chat": "#fc8d59", "deepresearch": "#ffffbf", "swe": "#91bfdb"}
# `--muted`: the same three hues with the chroma taken out, for the one-column
# version where the three bands sit in a 0.85 in panel and the saturated ramp
# reads as three signals rather than three categories. Each entry is its RdYlBu
# counterpart blended 65/35 with its own grey, so the HUE ORDER IS UNCHANGED --
# chat warm, deep research sand, agent cool -- and a reader moving between the
# two versions is not asked to relearn which colour is which class. Deep
# research is the exception: desaturating #ffffbf gives #fcfcd3, which is
# invisible on white, so it is darkened to a sand instead of only desaturated.
# ⚠ CHAT AND AGENT SIT AT SIMILAR GREYSCALE LIGHTNESS (168 and 181 of 255), so
# this palette separates by hue and a greyscale print of it does not separate
# those two bands. The saturated version does no better on that axis (168, 181
# as well); neither is safe for greyscale and the caption should not claim it.
# Softened once more on 2026-09-10 at the author's request. The first muted set
# was #df9675 / #e3d9a6 / #9ebbce -- 65/35 with grey; this one takes the chroma
# down a further quarter and then lifts each colour an eighth of the way to
# white, so the three read as tints of one another rather than as three inks.
# ⚠ WHAT LIMITS HOW FAR THIS CAN GO IS THE BOUNDARY BETWEEN TWO BANDS, not the
# contrast against the page: the bands are stacked with no outline, so where
# chat ends and deep research begins is a colour step and nothing else. The
# three greyscale lightnesses are 179, 218 and 189 of 255, so the two boundaries
# the stacking order usually shows -- chat|deep and deep|agent -- are 39 and 29
# apart. ⚠ CHAT AND AGENT ARE ONLY 10 APART and they touch wherever deep
# research is absent, which happens in several panels, so this palette separates
# those two by HUE alone and a greyscale print loses that boundary. The
# saturated ramp is no better there (168 against 181).
CLASS_COLOR_MUTED = {"chat": "#d7a791", "deepresearch": "#e3dcba",
                     "swe": "#afc1cf"}
# `--palette pubugn`, chosen by the author on 2026-09-10: ColorBrewer's
# three-class PuBuGn.
# ⚠ THIS IS A SEQUENTIAL RAMP AND THE OTHER TWO ARE NOT. Its three steps differ
# mainly in lightness (233, 187, 137 of 255), so the eye reads them as an ORDER
# -- pale, mid, dark -- and a reader can take that as "agent is the big one" or
# "the classes are three levels of something". The three classes are categories
# with no order, and the caption should not lean on the ramp. What it buys is
# that the bands separate in greyscale, which neither of the other two palettes
# does: the boundaries chat|deep and deep|agent are 46 and 50 apart in
# lightness, and chat|agent, which touches wherever deep research is absent, is
# 96 apart.
# chat and agent swapped on 2026-09-11 at the author's request: chat now takes
# the dark end of the ramp and agent the pale one.
# ⚠ WHAT THIS COSTS: agent is the THINNEST band in almost every panel (7.7% of
# the requests) and it is now the palest colour, so where it sits on top of the
# stack it is nearly the white of the page. chat is the thickest band (76.9%)
# and now carries the most ink. If the agent band becomes hard to find, the
# middle step (#a6bddb) is the one to give it.
CLASS_COLOR_PUBUGN = {"chat": "#1c9099", "deepresearch": "#a6bddb",
                      "swe": "#ece2f0"}
# GnBu picks (2026-09-16, at the author's request): chat darkest, deep research
# middle, agent lightest -- the same order of darkness as PuBuGn.
CLASS_COLOR_GNBU = {"chat": "#0868ac", "deepresearch": "#43a2ca",
                    "swe": "#bae4bc"}
# Chat in a lighter blue and deep research in the purple the arm figures now
# give llm-d (2026-09-17, at the author's request). Chat was #0868ac here, and
# deep research went #c00000, #f1a6b2, #f4a582 before this purple.
CLASS_COLOR_GNBU_RED = {"chat": "#4393c3", "deepresearch": "#c2a5cf",
                        "swe": "#bae4bc"}
# The ATTAINMENT overlay (`--attain`), added 2026-09-13 at the author's request
# as a second thing a panel's right axis can carry.
#
# ⚠ IT IS THE SAME QUANTITY THE HOUR GOODPUT FIGURES SCORE, read from the same
# verdict files: ladder95, in which a token is on time if it arrives within
# `TTFT budget + i x TBT budget` and a request is on time if at least 95% of
# its tokens are. Recomputing it here with a different rule would put two
# numbers called "attainment" in one paper.
#
# ⚠ THE DENOMINATOR IS ADMITTED, AND IT CANNOT BE ANYTHING ELSE HERE. A
# rejected request was never dispatched, so it belongs to no instance and
# cannot appear on a per-instance curve; run-boundary cutoffs leave the
# denominator too, because their outcome was never determined. A panel
# therefore says "of the requests this instance was given, what share met the
# rule" -- NOT what the policy owes for the requests it refused. The rejection
# rate belongs in the caption beside it.
#
# #e08214 as in `load_and_engine.pdf`, which draws the same quantity: not a
# yellow (no contrast at 0.8 pt on white), and not #ff7f0e or #d62728, which
# this directory binds to the deep research class and to PolyServe.
# The REJECTION overlay (`--reject`), added 2026-09-13 at the author's request.
#
# ⚠ IT IS A FLEET-WIDE QUANTITY DRAWN ON A PER-INSTANCE PANEL, and that is not
# a drawing choice, it is what the quantity is. A request is refused BEFORE any
# instance is chosen, so it belongs to no instance and there is no per-instance
# rejection rate to draw. The same curve is therefore repeated in all four
# panels of a column: it is the arm's rejection rate in that minute, put beside
# each instance so the bands can be read with it, NOT four different series.
#
# ⚠ IT SHARES THE LEFT AXIS WITH A COUNT, so the axis carries two scales: the
# bands are requests and the curve is a percentage, mapped so the TOP OF THE
# PANEL IS 100%. The caption has to say so -- a reader who takes the curve for
# a count will read a 40% rejection as 180 requests. The mapping is written
# into the CSV with every row.
#
# #c51b7d: distinct from the attainment orange on the right axis, from the
# three class fills, and from the red this directory binds to PolyServe.
REJ_COLOR = "#c51b7d"
REJ_STYLE = (0, (1.2, 1.2))
REJ_WIN = 60.0
# ⚠ A THREE-MINUTE COLUMN CANNOT BE DRAWN FROM THE HOUR TABLE. That table is
# built in 60 s windows, so three minutes of it is THREE POINTS per band and
# three points of rejection -- a panel that looks like a drawing of the data
# but is a drawing of the sampling. When a column is zoomed, all three series
# are recomputed at the resolutions below: residency straight from
# `engine_class_occupancy.occupancy` at 10 s, attainment in 30 s windows every
# 10 s, rejection in 10 s bins. At this hour's rates a 10 s bin still holds
# about 270 arrivals and a 30 s attainment window about 120 scored requests per
# instance, so the finer grid is measurement and not noise.
#
# ⚠ THE ROW IS STILL THE ENGINE THE HOUR TABLE RANKED. The rank comes from the
# whole hour's chat residency, not from the three minutes on screen, so
# "Instance 1" means the same engine in the zoomed column as in the full one.
ZOOM_RES_WIN = 10.0
ZOOM_ATT_WIN, ZOOM_ATT_STEP = 30.0, 10.0
ZOOM_REJ_WIN = 10.0
ATT_COLOR = "#e08214"
ATT_STYLE = (0, (3, 1.5))
# 90 s of requests every 30 s, as in the other attainment-over-time figures, and
# a window with fewer than ten scored requests is left blank rather than drawn:
# on four instances the quiet ones hold a handful of requests a minute and one
# miss would move the curve 20 points.
ATT_WIN, ATT_STEP, ATT_MIN_N = 90.0, 30.0, 10
VERDICTS = os.path.join(ROOT, "results", "aggregate_analysis", "ladder95",
                        "verdicts")
PALETTES = {"rdylbu": None, "muted": CLASS_COLOR_MUTED,
            "pubugn": CLASS_COLOR_PUBUGN, "gnbu": CLASS_COLOR_GNBU,
            "gnbured": CLASS_COLOR_GNBU_RED}
CLASS_LABEL = {"chat": "Chat", "deepresearch": "Deep Research", "swe": "Agent"}
CLASSES = ("chat", "deepresearch", "swe")
RANKS = [1, 2, 3, 4]
# The key's name for the attainment curve drawn on the left axis. None keeps the
# width-dependent default; --att-label sets it.
ATT_LABEL = None
# The key's name for the red time-between-tokens curve (--tbt without
# --attain). None leaves it out of the key, as before 2026-09-16.
TBT_LABEL = None
GRID_ON = False       # --grid


def read_tables(paths):
    """One table, or several concatenated (a preset can draw runs whose engine
    mix was written into different files). A run present in two of them -- the
    EXP-109 FluidServe repeat is in both the EXP-109 and the EXP-138 table --
    is kept once; the rows are identical, so dropping duplicates is exact."""
    if isinstance(paths, str):
        return pd.read_csv(paths)
    return pd.concat([pd.read_csv(p) for p in paths],
                     ignore_index=True).drop_duplicates()
XMAX = 60.0
FIG_H = 3.10
# The overlay line. A dark neutral and not a sixth hue: the panel already spends
# three colours on the classes, and the right axis is a different KIND of
# quantity -- what the engine did to the work, not what work it held.
# Red and dashed since 2026-09-10, at the author's request. ⚠ THE PROJECT BINDS
# #d62728 TO POLYSERVE elsewhere; here the arm is named by the column title and
# the classes are blue and teal, so nothing else on the panel is red, but a page
# that puts this figure beside an arm-coloured one has two meanings for the same
# ink.
TBT_COLOR = "#d62728"
TBT_STYLE = (0, (3, 1.5))
# 60 -> 120 s on 2026-09-10. The bands are 60 s windows, so 60 s was the
# resolution-matching choice; at 120 s the minute-to-minute steps that come from
# a few slow tokens in a quiet second flatten out and what is left is the level
# each instance sits at. ⚠ IT IS A BOX FILTER CENTRED ON THE SAMPLE, so a step
# in the true series is drawn as a 2-minute ramp and the first and last minute
# of the run are averaged over fewer samples than the middle.
TBT_SMOOTH_S = 120


def nice_ceil(v):
    """The smallest of 1, 2, 2.5, 5 x 10^k that is >= v."""
    if v <= 0:
        return 1.0
    e = 10.0 ** np.floor(np.log10(v))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if m * e >= v - 1e-9:
            return m * e
    return 10.0 * e


def origin(run_dir):
    """The time the residency table calls zero, in client seconds.

    ⚠ IT IS NOT THE RUN'S FIRST ARRIVAL. `engine_class_occupancy.occupancy`
    reads `exp22_fluidserve.load_run`, which drops the first 60 s of the trace
    (its lead-in) before anything is computed, and anchors its windows at the
    first arrival that is LEFT -- 60.0-60.3 s after the first one on EXP-109.
    Every series drawn against the bands has to use this origin. A series
    anchored at `metrics.csv`'s first `start_time` is one minute out of step
    with them, and on 2026-09-13 that alone made the drawn residency read as
    1.95x the engine's own batch on a ramping instance (1.03x once aligned).
    """
    r = EV.load_run(run_dir)
    return float(pd.to_numeric(r["start_time"], errors="coerce").min())


# ⚠ THE CLASS SPLIT NEEDS THE DISPATCH LOG AND THE BATCH DOES NOT. The engine
# reports its own running batch for the whole run, but which classes that batch
# held is known only for the requests that could be tied to an instance, and on
# some runs the log capture lost lines under load (EXP-126: Llumnix keeps 18.6%
# of its admitted requests, almost none after minute 20). A window in which
# fewer than COVER_MIN of the admitted arrivals were tied to an instance has no
# trustworthy split, so its batch is drawn as one grey "class unknown" band and
# its attainment point is left blank (2026-09-14).
COVER_MIN = 0.90
UNKNOWN = "unknown"
UNKNOWN_COLOR = "#bdbdbd"
_ATTR = {}


def attributed(run_dir):
    """(client rows, attributed rows, admitted count), computed once per run."""
    if run_dir not in _ATTR:
        r = EV.load_run(run_dir)
        j, n = EV.attribute_engines(run_dir, r)
        _ATTR[run_dir] = (r, j, n)
    return _ATTR[run_dir]


def coverage(run_dir, t0, lo_s, hi_s):
    """Share of admitted arrivals in [lo_s, hi_s) (s from t0) tied to an instance."""
    r, j, _n = attributed(run_dir)
    adm = pd.to_numeric(r.loc[~r["rejected"], "start_time"], errors="coerce") - t0
    att = pd.to_numeric(j["start_time"], errors="coerce") - t0
    na = int(((adm >= lo_s) & (adm < hi_s)).sum())
    nj = int(((att >= lo_s) & (att < hi_s)).sum())
    return (nj / na) if na else np.nan


def to_engine_batch(piv, run_dir, port, t0, win_s, tag):
    """Rescale one instance's class bands so each window sums to the ENGINE'S
    OWN running batch (`vllm:num_requests_running`, averaged over the window),
    keeping the class split the residency gives.

    ⚠ ONLY THE TOTAL IS MEASURED BY THE ENGINE. The engine does not report
    which classes its batch holds, so the split inside a bar is the residency's
    split applied to the engine's total -- an estimate, and the caption has to
    say so. A window in which the residency is empty but the engine reports a
    batch cannot be split at all; it is drawn empty and counted.
    """
    # ⚠ `fig_engine_window.gauges` CROPS TO ITS OWN FIGURE'S INTERVAL (the
    # module globals T_LO, T_HI = 33, 43 min). Left at their values, every
    # window outside those ten minutes had no gauge and was drawn EMPTY -- the
    # first hour-long draw on 2026-09-13 scaled 10 of 60 windows and zeroed the
    # rest. The zoomed figure escaped only because 33-36 sits inside 33-43.
    EW.PORT, EW.SMOOTH_S = int(port), 1
    EW.T_LO, EW.T_HI = -1e9, 1e9
    m, _kv, bat, _wait, _itl = EW.gauges(run_dir, t0)
    sec = m * 60.0
    g = np.array([np.nanmean(bat[(sec >= w) & (sec < w + win_s)])
                  if ((sec >= w) & (sec < w + win_s)).any() else np.nan
                  for w in piv.index])
    tot = piv.sum(axis=1).to_numpy(float)
    ok = tot > 0
    lost = int(((~ok) & (np.nan_to_num(g) > 0.5)).sum())
    no_gauge = int((ok & ~np.isfinite(g)).sum())
    if no_gauge:
        print(f"    ⚠ {tag} port {port}: {no_gauge} window(s) with residency but "
              f"no engine scrape, drawn empty")
    factor = np.where(ok, np.nan_to_num(g) / np.where(ok, tot, 1.0), 0.0)
    both = ok & np.isfinite(g)
    ratio = (np.nansum(g[both]) / tot[both].sum()) if both.any() else np.nan
    cov = np.array([coverage(run_dir, t0, w, w + win_s) for w in piv.index])
    unknown = (~ok | (np.nan_to_num(cov, nan=1.0) < COVER_MIN)) & (np.nan_to_num(g) > 0)
    out = piv.mul(factor, axis=0)
    out.loc[unknown, list(CLASSES)] = 0.0
    out[UNKNOWN] = np.where(unknown, np.nan_to_num(g), 0.0)
    print(f"    {tag} port {port}: engine batch / residency {ratio:.3f} over "
          f"{int(both.sum())} windows; class unknown in {int(unknown.sum())} of "
          f"{len(piv)} windows")
    return out


def collect(arms=None, zoom=None, height="resident"):
    d = read_tables(CSV_MIX)
    d = d[d["run"].str.contains(REPEAT)]
    s = read_tables(CSV_SUM)
    s = s[s["run"].str.contains(REPEAT)]
    out, cover = {}, {}
    for arm, label in (arms or ARMS):
        sub = d[d["arm"] == arm]
        if sub.empty:
            print(f"!! no rows for {label}", file=sys.stderr)
            continue
        cover[label] = float(s[s["arm"] == arm]["attributed_pct"].mean())
        if zoom and arm in zoom:
            lo, hi = zoom[arm]
            run = sub["run"].iloc[0]
            rank_of = sub.groupby("engine_port")["engine_rank"].first().to_dict()
            tab, frac = ECO.occupancy(os.path.join(ROOT, "results", run),
                                      ZOOM_RES_WIN)
            t_origin = (origin(os.path.join(ROOT, "results", run))
                        if height == "batch" else None)
            cover[label] = 100.0 * frac
            # ⚠ ONE WINDOW PAST THE RIGHT EDGE, on purpose. The x of a band
            # point is the window's START, so a grid that stops at `hi` draws
            # its last value at `hi - 10 s` and the fill ends short of the
            # frame with a white wedge -- which reads as "nothing was resident
            # here" rather than "the grid ended". The extra point is the
            # MEASURED window that starts at `hi`; nothing is extrapolated.
            grid = np.arange(lo * 60.0, hi * 60.0 + ZOOM_RES_WIN, ZOOM_RES_WIN)
            print(f"  {label:13s} zoomed to {lo:.0f}-{hi:.0f} min, "
                  f"residency recomputed in {ZOOM_RES_WIN:.0f} s windows "
                  f"({len(grid)} per panel, was {(hi - lo):.0f})")
            for port, g in tab.groupby("engine_port"):
                port = int(port)
                if port not in rank_of:
                    continue
                piv = g.pivot_table(index="win_start_s", columns="class",
                                    values="resident", aggfunc="sum",
                                    fill_value=0.0)
                for c in CLASSES:
                    if c not in piv:
                        piv[c] = 0.0
                piv = piv[list(CLASSES)].reindex(grid, fill_value=0.0)
                if height == "batch":
                    piv = to_engine_batch(piv, os.path.join(ROOT, "results", run),
                                          port, t_origin, ZOOM_RES_WIN, label)
                out[(label, int(rank_of[port]))] = piv
            continue
        for rank in RANKS:
            r = sub[sub["engine_rank"] == rank]
            piv = r.pivot_table(index="win_start_s", columns="class",
                                values="resident", aggfunc="sum", fill_value=0.0)
            for c in CLASSES:
                if c not in piv:
                    piv[c] = 0.0
            # Every window of the run, so a gap is drawn as a gap rather than
            # closed up by the neighbouring windows.
            grid = np.arange(0.0, sub["win_start_s"].max() + 60.0, 60.0)
            if height == "batch":
                # the engine reports the whole hour even where the dispatch log
                # does not, so the grid is the hour and not the table's windows
                grid = np.arange(0.0, XMAX * 60.0, 60.0)
            piv = piv.reindex(grid, fill_value=0.0)[list(CLASSES)]
            if height == "batch":
                run = sub["run"].iloc[0]
                if rank == RANKS[0]:
                    t_origin = origin(os.path.join(ROOT, "results", run))
                port_of = sub.groupby("engine_rank")["engine_port"].first()
                piv = to_engine_batch(piv, os.path.join(ROOT, "results", run),
                                      int(port_of[rank]), t_origin, 60.0, label)
            out[(label, rank)] = piv
    return out, cover


def check_side_labels(fig, labels):
    """Do the rotated figure-level labels fit on the canvas?

    They are drawn outside every axes and are clipped by neither the axes nor
    the figure, so one taller than the page loses its ends and the script says
    nothing. This is the same failure `fig_azure_rate_and_mix.check_ylabel`
    exists for, one level up: there it was an axis label, here a figure label.
    """
    fig.canvas.draw()
    h = fig.get_size_inches()[1] * fig.dpi
    for side, art in labels:
        if art is None:
            continue
        b = art.get_window_extent(fig.canvas.get_renderer())
        if b.y0 < 0 or b.y1 > h:
            print(f"  ⚠ {side} label runs off the canvas by "
                  f"{max(-b.y0, b.y1 - h) / fig.dpi:.3f} in "
                  f"({b.height / fig.dpi:.2f} in of ink, canvas "
                  f"{h / fig.dpi:.2f} in)")
        else:
            print(f"  {side} label fits with "
                  f"{min(b.y0, h - b.y1) / fig.dpi:.3f} in to spare")


def tbt_series(arms, smooth_s=TBT_SMOOTH_S):
    """(arm label, engine rank) -> (minutes, mean inter-token latency in ms).

    The engines' OWN counters, not the client's: per engine, the increment of
    `vllm:inter_token_latency_seconds_sum` over the increment of the matching
    `_count` between scrapes, which is the token-weighted mean over the tokens
    that engine emitted in that second. Averaging per-token is what makes it
    comparable with a per-token budget; a mean over requests would weigh a
    two-token request like a two-thousand-token one.

    ⚠ THE ROW OF THE FIGURE IS AN ENGINE RANK AND THE METRIC FILE IS A PORT, so
    the two are joined through the rank the mix table already assigned to each
    port for the whole run. If that ranking changes the rows move together and
    the overlay moves with them.

    ⚠ TIME IS ANCHORED THE WAY THE MIX TABLE ANCHORS IT -- the first arrival of
    the run as the loader's analysis window sees it -- and not at the moment the
    metric collector started, which is earlier and differs per run.
    """
    d = read_tables(CSV_MIX)
    d = d[d["run"].str.contains(REPEAT)]
    k = max(1, int(smooth_s))
    out = {}
    for arm, label in arms:
        sub = d[d["arm"] == arm]
        if sub.empty:
            continue
        run = sub["run"].iloc[0]
        rank_of = sub.groupby("engine_port")["engine_rank"].first().to_dict()
        run_dir = os.path.join(ROOT, "results", run)
        r = EV.load_run(run_dir)
        if r is None or r.empty:
            print(f"!! {label}: no client rows, no TBT overlay", file=sys.stderr)
            continue
        t0 = float(pd.to_numeric(r["start_time"], errors="coerce").min())
        per = DKG.itl_per_engine(run_dir)
        if not per:
            print(f"!! {label}: no engine ITL counters", file=sys.stderr)
            continue
        for name, (grid, ds, dc) in per.items():
            port = int(name)
            if port not in rank_of:
                continue
            sd = DKG.smooth(ds, k)
            cd = DKG.smooth(dc, k)
            v = np.divide(sd, cd, out=np.full_like(sd, np.nan),
                          where=cd > 0) * 1000.0
            out[(label, int(rank_of[port]))] = ((grid - t0) / 60.0, v, port)
    return out


def attain_series(arms, win_s=ATT_WIN, step_s=ATT_STEP, min_n=ATT_MIN_N,
                  zoom=None):
    """(arm label, engine rank) -> (minutes, admitted attainment %, port).

    Each request is tied to the instance that served it through the same join
    the residency bands use, and to its ladder95 verdict through
    `(task_id, call_index, iteration)`. The share of the join that finds a
    verdict is printed per arm: a request the scorer dropped cannot be counted
    either way, and a curve drawn from 60% of an instance's requests is not the
    same statement as one drawn from all of them.
    """
    d = read_tables(CSV_MIX)
    d = d[d["run"].str.contains(REPEAT)]
    keys = ["task_id", "call_index", "iteration"]
    out = {}
    for arm, label in arms:
        sub = d[d["arm"] == arm]
        if sub.empty:
            continue
        run = sub["run"].iloc[0]
        vp = os.path.join(VERDICTS, run + ".csv")
        if not os.path.isfile(vp):
            print(f"!! {label}: no ladder95 verdicts at {vp}", file=sys.stderr)
            continue
        rank_of = sub.groupby("engine_port")["engine_rank"].first().to_dict()
        run_dir = os.path.join(ROOT, "results", run)
        r, j, n_adm = attributed(run_dir)
        if r is None or r.empty:
            print(f"!! {label}: no client rows, no attainment overlay",
                  file=sys.stderr)
            continue
        v = pd.read_csv(vp)[keys + ["ladder_ok", "cutoff", "rejected"]].rename(
            columns={"cutoff": "v_cutoff", "rejected": "v_rejected"})
        j = j.merge(v, on=keys, how="left")
        got = float(j["ladder_ok"].notna().mean()) if len(j) else 0.0
        print(f"  {label:13s} {len(j):7d} requests tied to an instance "
              f"({100.0 * len(j) / max(n_adm, 1):5.1f}% of admitted), "
              f"{100.0 * got:5.1f}% of those carry a verdict")
        j = j[j["ladder_ok"].notna()].copy()
        t0 = float(pd.to_numeric(r["start_time"], errors="coerce").min())
        j["rel_min"] = (pd.to_numeric(j["start_time"], errors="coerce") - t0) / 60.0
        w_s, s_s, t_lo, t_hi = win_s, step_s, 0.0, XMAX
        if zoom and arm in zoom:
            w_s, s_s = ZOOM_ATT_WIN, ZOOM_ATT_STEP
            t_lo, t_hi = zoom[arm]
        half = w_s / 120.0
        for port, g0 in j.groupby("engine_port"):
            port = int(port)
            if port not in rank_of:
                continue
            # ⚠ THE WINDOWS ARE CENTRED ON THE DRAWN INTERVAL AND CROPPED TO
            # IT, not built inside it. A window centred at `t_lo` reaches
            # half a window before it, which is measurement that exists; a
            # curve that starts at `t_lo + half` instead leaves the first and
            # last 15 s of the panel empty for no reason. Only at the run's own
            # boundaries is there nothing to reach for, and `served` is empty
            # there, so the point is dropped by the min_n test.
            xs, ys = [], []
            c = t_lo
            while c <= t_hi + 1e-9:
                g = g0[(g0["rel_min"] >= c - half) & (g0["rel_min"] < c + half)]
                served = g[~g["v_cutoff"].astype(bool)
                           & ~g["v_rejected"].astype(bool)]
                xs.append(c)
                cv = coverage(run_dir, t0, (c - half) * 60.0, (c + half) * 60.0)
                ys.append(100.0 * float(served["ladder_ok"].astype(bool).mean())
                          if len(served) >= min_n and not (cv < COVER_MIN)
                          else np.nan)
                c += s_s / 60.0
            out[(label, int(rank_of[port]))] = (np.array(xs), np.array(ys), port)
    return out


def write_tbt_csv(tbt, path):
    rows = []
    for (label, rank), (m, v, port) in sorted(tbt.items()):
        ok = np.isfinite(v) & (m >= 0) & (m <= XMAX)
        for mm, vv in zip(m[ok], v[ok]):
            rows.append({"arm": label, "engine_rank": rank, "engine_port": port,
                         "minute": round(float(mm), 4),
                         "tbt_ms": round(float(vv), 4),
                         "smoothing_s": TBT_SMOOTH_S, "run": REPEAT,
                         "source": "vllm:inter_token_latency_seconds_"
                                   "{sum,count}, per engine, per second"})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path}  ({len(rows)} rows)")


def report_tbt(tbt):
    print(f"{'arm':13s} {'rank':>4s} {'port':>5s} {'p10':>6s} {'p50':>6s} "
          f"{'p90':>6s}   (mean TBT, ms, minutes 0-60)")
    for (label, rank), (m, v, port) in sorted(tbt.items()):
        ok = np.isfinite(v) & (m >= 0) & (m <= XMAX) & (v > 0)
        if not ok.any():
            continue
        vv = v[ok]
        print(f"{label:13s} {rank:4d} {port:5d} {np.percentile(vv,10):6.1f} "
              f"{np.percentile(vv,50):6.1f} {np.percentile(vv,90):6.1f}")


def reject_series(arms, win_s=REJ_WIN, zoom=None):
    """arm label -> (minutes, % of that minute's arrivals that were refused).

    From the client's own `metrics.csv`, which is where a rejection is visible:
    the scheduler refuses the request and the gateway returns it, so it never
    reaches an engine and appears in no dispatch log. Run-boundary cutoffs are
    not excluded here because the denominator is ARRIVALS -- every request that
    was offered in that minute -- and a request that arrived was either refused
    or it was not.
    """
    d = read_tables(CSV_MIX)
    d = d[d["run"].str.contains(REPEAT)]
    out = {}
    for arm, label in arms:
        sub = d[d["arm"] == arm]
        if sub.empty:
            continue
        run = sub["run"].iloc[0]
        m = pd.read_csv(os.path.join(ROOT, "results", run, "metrics.csv"),
                        usecols=["agent", "start_time", "is_rejected"],
                        low_memory=False)
        m = m[m["agent"] != "job_summary"]
        t = pd.to_numeric(m["start_time"], errors="coerce")
        rej = m["is_rejected"].astype(str).str.lower().isin(["true", "1"])
        # ⚠ THE BANDS' ORIGIN, NOT `t.min()` (fixed 2026-09-13). The two are
        # 60 s apart -- see `origin` -- and every rejection figure drawn before
        # this line was changed puts the curve one minute LATE against the
        # bands and the attainment curve beside it.
        rel = (t - origin(os.path.join(ROOT, "results", run))).to_numpy(float)
        w_s, t_lo, t_hi = win_s, 0.0, XMAX
        if zoom and arm in zoom:
            w_s = ZOOM_REJ_WIN
            t_lo, t_hi = zoom[arm]
        # Bins CENTRED on a grid that includes both edges, for the same
        # reason as the bands above: with edges starting at `t_lo` the first
        # value is plotted half a bin inside the panel and the curve is short
        # at both ends.
        centres = np.arange(t_lo * 60.0, t_hi * 60.0 + w_s / 2.0, w_s)
        edges = np.append(centres - w_s / 2.0, centres[-1] + w_s / 2.0)
        n = np.histogram(rel, bins=edges)[0]
        nr = np.histogram(rel[rej.to_numpy()], bins=edges)[0]
        pct = 100.0 * np.divide(nr, n, out=np.full(len(n), np.nan),
                                where=n > 0)
        out[label] = (centres / 60.0, pct)
        print(f"  {label:13s} refused {100.0 * rej.mean():5.1f}% of "
              f"{len(m)} arrivals  (per minute: p10 "
              f"{np.nanpercentile(pct, 10):.1f}, p50 "
              f"{np.nanpercentile(pct, 50):.1f}, p90 "
              f"{np.nanpercentile(pct, 90):.1f})")
    return out


def write_attain_csv(att, path, win_s, step_s, rej=None):
    rows = []
    if rej:
        for label, (m, v) in sorted(rej.items()):
            for mm, vv in zip(m, v):
                rows.append({"arm": label, "engine_rank": "fleet",
                             "engine_port": "", "minute": round(float(mm), 4),
                             "attainment_pct": None,
                             "rejected_pct": (None if not np.isfinite(vv)
                                              else round(float(vv), 4)),
                             "window_s": REJ_WIN, "step_s": REJ_WIN,
                             "run": REPEAT,
                             "denominator": "ALL arrivals in that minute, "
                                            "fleet-wide; drawn on the left "
                                            "axis with its top as 100%",
                             "rule": "is_rejected in metrics.csv"})
    for (label, rank), (m, v, port) in sorted(att.items()):
        for mm, vv in zip(m, v):
            rows.append({"arm": label, "engine_rank": rank, "engine_port": port,
                         "minute": round(float(mm), 4),
                         "attainment_pct": (None if not np.isfinite(vv)
                                            else round(float(vv), 4)),
                         "window_s": win_s, "step_s": step_s, "run": REPEAT,
                         "denominator": "admitted requests dispatched to this "
                                        "instance, cutoffs excluded",
                         "rule": "ladder95 (results/aggregate_analysis/"
                                 "ladder95/verdicts)"})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path}  ({len(rows)} rows)")


def report_attain(att):
    print(f"{'arm':13s} {'rank':>4s} {'port':>5s} {'p10':>6s} {'p50':>6s} "
          f"{'p90':>6s} {'blank':>6s}   (attainment %, admitted, minutes 0-60)")
    for (label, rank), (m, v, port) in sorted(att.items()):
        ok = np.isfinite(v)
        if not ok.any():
            continue
        vv = v[ok]
        print(f"{label:13s} {rank:4d} {port:5d} {np.percentile(vv,10):6.1f} "
              f"{np.percentile(vv,50):6.1f} {np.percentile(vv,90):6.1f} "
              f"{100.0 * (~ok).mean():5.1f}%")


def write_csv(data, cover, path):
    rows = []
    for (label, rank), piv in data.items():
        tot = piv.sum(axis=1)
        for w, row in piv.iterrows():
            t = float(tot.loc[w])
            rec = {"arm": label, "engine_rank": rank, "minute": w / 60.0,
                   "attributed_pct": round(cover[label], 2)}
            for c in CLASSES:
                rec[f"resident_{c}"] = float(row[c])
                rec[f"share_{c}_pct"] = (100.0 * float(row[c]) / t) if t > 0 else np.nan
            rec["resident_total"] = t
            rec["unit"] = "mean concurrently resident requests in a 60 s window"
            rec["run"] = REPEAT
            rows.append(rec)
    df = pd.DataFrame(rows).sort_values(["arm", "engine_rank", "minute"])
    df.to_csv(path, index=False)
    print(f"wrote {path}  ({len(df)} rows)")


def build(data, cover, out, share=True, arms=None, width=None, height=None,
          label_size=None, colors=None, tbt=None, dense_y=False,
          tbt_max=None, overlay="tbt", reject=None, zoom=None, req_max=None, req_ticks=None, req_skip=(),
          row_tags=False, panel_titles=False, column_rules=False,
          grow=False, height_kind="resident", req_nticks=None, att_left=None):
    arms = arms or ARMS
    h_now = height or FIG_H
    colors = colors or CLASS_COLOR
    labels = [l for _, l in arms if any((l, r) in data for r in RANKS)]
    style = dict(ps.STYLE)
    if label_size:
        style.update({"xtick.labelsize": label_size,
                      "ytick.labelsize": label_size,
                      "axes.labelsize": label_size})
    twins = {}
    any_unknown = [False]
    # ⚠ ONE SIZE FOR THE KEY, THE AXIS NAMES AND THE PANEL TITLES (2026-09-13,
    # at the author's request). They are the figure's LABELS -- read once each,
    # from across the page -- as opposed to the tick numbers, which are read
    # beside their own ink and stay at `label_size`. Before this they were at
    # three sizes (7 pt key, 8 pt side labels, `label_size` x label) for no
    # reason other than the order they were added in.
    title_size = (label_size or 8) + 1.5
    # ⚠ NARROW COLUMNS (2026-09-16, four arms in one column: 0.42 in per panel).
    # At the sizes above the "Instance k" titles and the arm names run into
    # the next column and 15-minute ticks collide, so below 1.0 in of canvas per
    # column the titles and column names come down to the tick size + 1, the
    # x ticks thin to 0/30/60, and an arm name may carry a line break. The
    # threshold is canvas per column: 0.83 in for four arms in 3.335 in, 1.67
    # for the two-arm figure, which keeps its sizes.
    narrow = (width or ps.TEXT_W) / max(len(labels), 1) < 1.0
    inst_size = (label_size or 8) + 1.0 if narrow else title_size
    name_size = (label_size or 8) + (1.5 if narrow else 3.0)
    name_lines = max(l.count("\n") + 1 for l in labels) if labels else 1
    with plt.rc_context(style):
        fig, axes = plt.subplots(len(RANKS), len(labels),
                                 figsize=(width or ps.TEXT_W,
                                          height or FIG_H), # ⚠ `sharex="col"`, NOT `True`. With one shared x axis for the whole grid the
                                 # LAST column drawn sets the limits for every
                                 # panel, so a per-column interval (`--zoom`)
                                 # silently moved both columns to the same three
                                 # minutes and emptied the other one. Sharing
                                 # within a column keeps the inner tick labels
                                 # hidden, which is what the sharing was for.
                                 sharex="col")
        # In the absolute figure each COLUMN has its own y scale, because the
        # arm that never rejects holds twenty times what the others do and one
        # shared scale would draw four of the five columns as a flat line. The
        # scale is printed in the top panel of the column so the difference is
        # stated rather than hidden.
        # ONE RIGHT-HAND SCALE FOR THE WHOLE FIGURE, unlike the left one in the
        # absolute version. The left axis is per column because the arms differ
        # by twenty times in how much they hold; the right axis is a LATENCY the
        # same budgets apply to everywhere, so a panel-by-panel scale would hide
        # exactly the comparison it is drawn for -- that one engine sits at 40 ms
        # while another sits at 70.
        tbt_lo, tbt_hi = 1e9, 0.0
        if tbt:
            for (label, rank), (m, v, _p) in tbt.items():
                ok = np.isfinite(v) & (m >= 0) & (m <= XMAX) & (v > 0)
                if ok.any():
                    tbt_lo = min(tbt_lo, float(np.percentile(v[ok], 1)))
                    tbt_hi = max(tbt_hi, float(np.percentile(v[ok], 99.5)))
        tops = {}
        for label in labels:
            tops[label] = max(float(data[(label, r)].sum(axis=1).max())
                              for r in RANKS if (label, r) in data)
        for i, rank in enumerate(RANKS):
            for j, label in enumerate(labels):
                ax = axes[i][j]
                if GRID_ON:
                    # dotted grey guides at the ticks of both axes, the style
                    # of kv_batch_attain_4arms (2026-09-16, at the author's
                    # request); behind the bands, so they show where a panel
                    # is empty and not across the stacked areas
                    ax.grid(axis="both", **ps.GRID)
                    ax.set_axisbelow(True)
                # ⚠ EACH COLUMN CAN CARRY ITS OWN INTERVAL (`--zoom`), and
                # then the columns are NOT on one time axis. Two columns showing
                # different three-minute stretches cannot be read across, and
                # the caption has to name the interval of each; the panels
                # themselves say it only through their tick numbers.
                lo, hi = zoom.get(label, (0.0, XMAX)) if zoom else (0.0, XMAX)
                ax.set_xlim(lo, hi)
                if (hi - lo) >= 30.0:
                    ax.set_xticks([0, 30, 60] if narrow else [0, 15, 30, 45, 60])
                else:
                    # One tick per whole minute in the interval: on three
                    # minutes the hour figure's 15-minute ticks would leave the
                    # panel with none at all, or with one.
                    ax.set_xticks(list(np.arange(np.ceil(lo), hi + 1e-9, 1.0)))
                if (label, rank) not in data:
                    continue
                piv = data[(label, rank)]
                x = piv.index.to_numpy() / 60.0
                vals = [piv[c].to_numpy(dtype=float) for c in CLASSES]
                if share:
                    tot = np.sum(vals, axis=0)
                    vals = [np.where(tot > 0, 100.0 * v / np.where(tot > 0, tot, 1.0), 0.0)
                            for v in vals]
                # No outline on the bands: the boundary between two of them is
                # where one colour ends and the next begins, and a drawn line
                # there is a fourth thing on a panel that has three. The legend
                # swatches keep theirs, because a #ffffbf square on white paper
                # with no edge is not a square anyone can see.
                cols_ = [colors[c] for c in CLASSES]
                if UNKNOWN in piv and not share:
                    vals.append(piv[UNKNOWN].to_numpy(dtype=float))
                    cols_.append(UNKNOWN_COLOR)
                    if piv[UNKNOWN].to_numpy(dtype=float).max() > 0:
                        any_unknown[0] = True
                ax.stackplot(x, *vals, colors=cols_, linewidth=0.0)
                ax.set_ylim(0, 100 if share else
                            (req_max if (req_max and label not in req_skip)
                             else tops[label] * 1.05))
                if share:
                    ax.set_yticks([0, 50, 100])
                else:
                    # `dense_y` puts a value at the middle of the left axis as
                    # well as at its ends. Two ticks are enough to say what the
                    # column's scale is; three are needed to read a level off
                    # the band, which is what the panel is for once a second
                    # quantity is drawn beside it.
                    # `req_max` replaces the per-column top with one number
                    # for every panel. ⚠ THAT MAKES THE COLUMNS COMPARABLE AND
                    # THE QUIET ONES FLAT: the arm holding a fifth of what the
                    # busiest holds is drawn a fifth as tall, which is the
                    # point, but its composition -- the thing the bands are
                    # for -- becomes harder to read. Anything above the top is
                    # cut, and the count of cut windows is printed.
                    # An arm in `req_skip` keeps its own top: one column can
                    # hold twenty times what the others do, and forcing it onto
                    # their axis would draw it as a line along the ceiling that
                    # says nothing about its shape. ⚠ THE FIGURE THEN HAS TWO
                    # KINDS OF COLUMN and the axis name has to keep saying so.
                    fixed = req_max and label not in req_skip
                    hi = req_max if fixed else tops[label]
                    if fixed and req_ticks:
                        t = list(req_ticks)
                    elif req_nticks:
                        # `--req-nticks N` (2026-09-13, at the author's
                        # request): N evenly spaced values on every panel. On a
                        # fixed axis they divide the given top; on a column's
                        # own axis the top is raised to the next round number
                        # that N-1 equal steps reach, so the labels are round
                        # and the tallest band still fits.
                        k = req_nticks - 1
                        if fixed:
                            step = hi / k
                        else:
                            step = nice_ceil(hi / k)
                            ax.set_ylim(0, step * k)
                        t = [step * i for i in range(req_nticks)]
                    else:
                        t = [0, hi / 2, hi] if dense_y else [0, hi]
                    ax.set_yticks(t)
                    ax.set_yticklabels([f"{x:,.0f}" for x in t])
                if panel_titles:
                    # The instance is named over its own panel, so the row
                    # identity is read where the panel is rather than carried in
                    # from the margin. The arm then has to move: it becomes the
                    # second line of the bottom row's x label, which is where
                    # every other figure in this directory names a column.
                    ax.set_title(f"Instance {rank}",
                                 fontsize=inst_size, pad=2)
                if i == 0 and not panel_titles:
                    # ⚠ THE COVERAGE IS NO LONGER PRINTED ON THE FIGURE (removed
                    # 2026-09-07 at the author's request), so the CAPTION now
                    # carries the whole of it: the vLLM router column is drawn
                    # from 74.9% of its admitted requests, and that shortfall is
                    # concentrated in the second half of the hour (99.7% over
                    # minutes 0-20, 43.6% after minute 40) and is not
                    # class-neutral. The script still prints it per arm.
                    ax.set_title(label, fontsize=8, pad=3)
                if j == 0 and not row_tags and not panel_titles:
                    ax.set_ylabel(f"Instance {rank}", labelpad=1.5)
                elif j == 0 and row_tags and not panel_titles:
                    # ⚠ THE ROW IDENTITY IS NOT AN AXIS NAME. Written where a y
                    # label goes -- rotated, beside the ticks -- "Instance 3"
                    # reads as the quantity the axis measures, and the reader
                    # then has to work out whether the panel counts requests or
                    # instances. As a HORIZONTAL tag in the left margin it
                    # cannot be an axis name: axis names in this figure are
                    # rotated, and the only rotated text on the left is the one
                    # that names the quantity for every panel.
                    ax.text(-0.30, 0.5, f"#{rank}", transform=ax.transAxes,
                            ha="right", va="center",
                            fontsize=(label_size or 8) + 0.5, color="#444444")
                if i == len(RANKS) - 1:
                    # With `panel_titles` the arm name is NOT the second line
                    # of this label: it is drawn after the layout at its own
                    # size, because the x label carries the panels' type size
                    # (5 pt here) and the column name is read from across the
                    # page rather than beside its ticks.
                    ax.set_xlabel("Time (min.)", labelpad=1.5,
                                  fontsize=inst_size)
                if att_left and (label, rank) in att_left:
                    # ⚠ ON THE LEFT AXIS, MAPPED SO THE TOP OF THE PANEL IS 100%
                    # (2026-09-14): the right axis already carries the time
                    # between tokens, and one axis cannot carry both. The key
                    # names the curve; the caption has to give the mapping.
                    am, av, _p = att_left[(label, rank)]
                    keep = (am >= lo) & (am <= hi)
                    # NaN is KEPT so the line breaks across a blank window; drawing
                    # only the finite points joined the two sides of a 30-minute
                    # gap with a straight line that looked like a measurement.
                    ax.plot(am[keep], av[keep] / 100.0 * ax.get_ylim()[1],
                            color=ATT_COLOR, lw=0.7, ls=ATT_STYLE, zorder=6)
                if reject and label in reject:
                    # Drawn LAST inside the panel so it sits over the bands,
                    # and mapped onto the left axis by its top: `top` is the
                    # number of requests the axis ends at, so 100% lands on the
                    # frame. It is the same curve in every panel of this column
                    # -- see REJ_COLOR above for why it cannot be otherwise.
                    rm, rp = reject[label]
                    ok = np.isfinite(rp) & (rm >= lo) & (rm <= hi)
                    ax.plot(rm[ok], rp[ok] / 100.0 * ax.get_ylim()[1],
                            color=REJ_COLOR, lw=0.7, ls=REJ_STYLE, zorder=6)
                if tbt and (label, rank) in tbt:
                    m, v, _p = tbt[(label, rank)]
                    ok = np.isfinite(v) & (m >= lo) & (m <= hi)
                    ax2 = ax.twinx()
                    twins.setdefault(j, []).append(ax2)
                    ax2.plot(m[ok], v[ok],
                             color=ATT_COLOR if overlay == "attain" else TBT_COLOR,
                             lw=0.8,
                             ls=ATT_STYLE if overlay == "attain" else TBT_STYLE,
                             zorder=5)
                    # ⚠ THE RIGHT AXIS IS LOGARITHMIC AND THE READER HAS TO BE
                    # TOLD. One scale for the whole figure was the right choice
                    # -- a latency is measured against budgets that do not
                    # change from panel to panel -- but the arm without
                    # admission control runs at 400 ms of median on one engine
                    # while every other panel lives between 20 and 90, so a
                    # linear axis holding both draws four columns as a flat line
                    # on the floor. On a log axis the 40-vs-70 ms difference
                    # between two FluidServe engines is still legible and the
                    # 400 ms engine is on the same panel grid.
                    if overlay == "attain":
                        # A PERCENTAGE HAS ITS OWN CEILING and it is the same on
                        # every panel, so the axis is fixed at 0..100 whatever
                        # the data does. 50 and 100 only: the left axis already
                        # carries a 0 at the same height, and a third number
                        # between them costs ink that the 0.85 in panel does not
                        # have.
                        ax2.set_ylim(0, 103)
                        ax2.set_yticks([50, 100])
                    elif tbt_max:
                        # A FIXED LINEAR CEILING, given on the command line.
                        # Whatever runs past it is drawn up to the top of the
                        # panel and no further, so the script counts what it
                        # cuts and prints it -- a clipped line reads as a line
                        # that touched the ceiling and stayed there.
                        ax2.set_ylim(0, tbt_max)
                        # From 50, not from 0: the left axis already carries a
                        # 0 at the same height, and two of them on one panel is
                        # the same information twice (2026-09-11, at the
                        # author's request). The axis still STARTS at 0 -- only
                        # the label for it is gone.
                        ax2.set_yticks(list(np.arange(50, tbt_max + 1, 50)))
                    else:
                        ax2.set_yscale("log")
                        ax2.set_ylim(tbt_lo * 0.85, tbt_hi * 1.20)
                        ax2.set_yticks([30, 100, 300])
                        ax2.set_yticklabels(["30", "100", "300"])
                        ax2.minorticks_off()
                    ax2.tick_params(length=2.0, pad=1.5)
                    # ⚠ THE NUMBERS ARE ON EVERY PANEL WHEN `dense_y` IS SET
                    # (2026-09-10, at the author's request). With five columns
                    # they were drawn once per row, because five copies of one
                    # scale cost five times the ink for one piece of
                    # information; with three columns there is room, and a
                    # reader comparing one panel with the one below it no longer
                    # has to carry the scale across the figure.
                    if not dense_y and j != len(labels) - 1:
                        ax2.set_yticklabels([])

        # Square swatches: `handlelength` and `handleheight` are both in font
        # units, so setting them equal makes the key a square rather than the
        # default wide rectangle. The thin grey edge is not decoration -- the
        # deep research fill is #ffffbf, which has almost no contrast against the
        # white page, and without an outline that entry reads as an empty gap.
        # Black keylines on the swatches (2026-09-10, was #666666 at 0.4 pt).
        # The palest class is #ece2f0, which has almost no edge against white
        # paper, and a grey outline on a pale fill reads as a smudge rather than
        # as a square; black is the only edge that says "this is a colour
        # sample" for every fill in every palette this script can draw.
        handles = [Patch(facecolor=colors[c], label=CLASS_LABEL[c],
                         edgecolor="#000000", linewidth=0.5)
                   for c in CLASSES]
        names = [CLASS_LABEL[c] for c in CLASSES]
        # ⚠ THE REJECTION CURVE HAS TO BE NAMED IN THE KEY because it is the
        # only line on the figure with no axis of its own: the attainment curve
        # is named by the right-hand axis label, and the bands by their
        # swatches. What the key CANNOT carry is the mapping (the top of the
        # panel is 100%) -- that is the caption's job.
        if any_unknown[0]:
            handles.append(Patch(facecolor=UNKNOWN_COLOR, edgecolor="#000000",
                                 linewidth=0.5))
            names.append("Class unknown")
        if att_left:
            handles.append(Line2D([], [], color=ATT_COLOR, lw=0.9,
                                  ls=ATT_STYLE))
            # the short form on a one-column canvas: the long one made the key
            # 3.80 in wide on a 3.335 in figure
            names.append(ATT_LABEL if ATT_LABEL else
                         "Request SLO Attainment (top = 100%)"
                         if (width or ps.TEXT_W) > 4.0 else "Attainment (top = 100%)")
        if tbt and overlay == "tbt" and TBT_LABEL:
            handles.append(Line2D([], [], color=TBT_COLOR, lw=0.9,
                                  ls=TBT_STYLE))
            names.append(TBT_LABEL)
        if tbt and overlay == "attain":
            # ⚠ NAMED IN THE KEY AS WELL AS BY ITS AXIS (2026-09-13, at the
            # author's request). The right-hand axis carries the units and the
            # denominator, which the key cannot; what the key adds is that the
            # reader does not have to turn the page sideways to learn which of
            # the two dashed curves is which.
            handles.append(Line2D([], [], color=ATT_COLOR, lw=0.9,
                                  ls=ATT_STYLE))
            names.append("Request SLO Attainment")
        if reject:
            handles.append(Line2D([], [], color=REJ_COLOR, lw=0.9,
                                  ls=REJ_STYLE))
            names.append("Rejected")
        # ⚠ THE KEY MAY BE ONE SIZE ABOVE THE OTHER LABELS (2026-09-18, at the
        # author's request: the key at exp131_horizon_goodput's 7 pt where the
        # row still fits). Tried, measured, and dropped back to `title_size`
        # when it does not; the figure never widens for it.
        key_fs = max(title_size, 7.0)
        for _try in range(2):
            probe = fig.legend(handles, names, loc="lower center",
                               ncol=len(names), bbox_to_anchor=(0.5, -1.0),
                               frameon=False, fontsize=key_fs,
                               columnspacing=1.0, handlelength=1.0,
                               handleheight=1.0, handletextpad=0.35,
                               borderaxespad=0.0)
            fig.canvas.draw()
            w_in = probe.get_window_extent().width / fig.dpi
            probe.remove()
            w_fig = fig.get_size_inches()[0]
            print(f"  key at {key_fs:.1f} pt would be {w_in:.2f} in wide on a "
                  f"{w_fig:.2f} in canvas"
                  + ("  -> too wide" if w_in > w_fig - 0.02 else "  -> used"))
            if w_in <= w_fig - 0.02 or key_fs <= title_size:
                break
            key_fs = title_size
        key = fig.legend(handles, names, loc="lower center",
                   # ⚠ THE ACTUAL CANVAS HEIGHT, NOT THE MODULE DEFAULT. Both
                   # of these bands are an amount of INK -- 0.175 in for the key
                   # and 0.155 in of clearance -- expressed as a fraction of the
                   # figure, so dividing by `FIG_H` (3.10) while drawing on a
                   # 1.49 in canvas asks for a band less than half the size and
                   # the key is cut off by the top of the page. It survived
                   # unnoticed at 2.59 in, where the error is 16%.
                   ncol=len(names), bbox_to_anchor=(0.5, 1 - 0.175 / h_now),
                   frameon=False,
                   fontsize=key_fs, columnspacing=1.0, handlelength=1.0,
                   handleheight=1.0, handletextpad=0.35, borderaxespad=0.0)
        # ⚠ THE SIDE LABELS ARE SET BY THE CANVAS HEIGHT, NOT BY TASTE. A
        # rotated label is centred on the figure and clipped by nothing, so one
        # longer than the page is silently cut at both ends. "Requests resident
        # per instance (own scale per column)" is 3.4 in of ink and fits the
        # 3.10 in text-width figure only because the ends fall in its margins;
        # at the one-column height of 2.60 in it was cut, so the parenthesis
        # moves to the caption there. `check_side_labels` measures both and
        # prints what it finds.
        compact = bool(label_size)
        if share:
            unit = "Class share rate per instances (%)"
        else:
            # Renamed on 2026-09-10: "Number of Requests per Instance" says
            # the same thing in the words the rest of the paper uses for a
            # count. The long form keeps the parenthesis that the compact one
            # has to drop for want of height.
            # The parenthesis stays whenever the columns do NOT all share one
            # top: with an exempt column they still do not.
            same = bool(req_max) and not req_skip
            unit = ("Number of Requests in Instance" if compact or same else
                    "Number of Requests in Instance (own scale per column)")
            # ⚠ THE LABEL IS NOT THE PLACE TO SAY "ONE CLUSTER". It was
            # extended to "Number of Requests, per Instance of One Cluster
            # (#1-#4)" and that is 3.4 in of rotated ink against a 2.6 in
            # canvas, so it lost both ends -- the failure this file's
            # `check_side_labels` exists for. What the rows are belongs in the
            # caption: "the four rows are the four instances of one cluster,
            # over the same hour".
        # ⚠ A ROTATED SIDE LABEL IS MEASURED AGAINST THE CANVAS HEIGHT, and a
        # figure with fewer rows is a SHORTER canvas -- the same label that fits
        # four rows runs off two. `check_side_labels` catches it, and the fix is
        # a shorter label rather than a taller canvas: below about 2 in the long
        # forms ("Number of Requests in Instance" is 1.60 in of ink, "Request
        # SLO attainment (%), admitted" is 1.92) cannot fit whatever the type
        # size. What is dropped moves to the caption, and `admitted` in
        # particular MUST be there -- it is the denominator.
        short = (height or FIG_H) < 2.0
        if short and not share:
            unit = "Requests in Instance"
        if height_kind == "batch" and not share:
            # "Decode Batch Size", at the author's request (2026-09-13).
            # ⚠ `vllm:num_requests_running` counts every request the engine
            # has scheduled, so a request still in its chunked prefill is in
            # it too; the height is the running batch, and the caption should
            # not claim it excludes prefill.
            unit = "Decode Batch Size of an Instance"
        sy = fig.supylabel(unit, fontsize=title_size, x=0.006)
        right = 1.0 if not tbt else 0.965
        fig.tight_layout(rect=(0.022, 0, right, 1 - 0.155 / h_now),
                         w_pad=0.7, h_pad=0.5, pad=0.3)
        ry = None
        if tbt:
            name = (("Attainment (%), admitted" if short else
                     "Request SLO attainment (%), admitted")
                    if overlay == "attain" else
                    ("Mean time between tokens (ms)" if tbt_max
                     else "Mean time between tokens (ms, log)"))
            if compact:
                name = name.replace("Mean time between tokens",
                                    "Mean time between tokens")
            ry = fig.text(0.998, 0.5, name, fontsize=title_size,
                          rotation=270, ha="right", va="center")
        if panel_titles:
            # One name per column, under its own column, at the size a column
            # name is read at. Reserved in inches so the canvas keeps it.
            #
            # WHAT THE RESERVE HAS TO HOLD. The name is drawn at the very bottom
            # of the canvas and the reserve pushes the axes UP, so the gap
            # between the name and the "Time (min.)" label above it is the
            # reserve MINUS the name's own height. At 0.13 in against a name
            # that is (label_size + 3) pt tall -- 0.153 in at the default 8 --
            # the difference was negative and the two touched. It is therefore
            # the name's height plus a fixed 0.09 in of air (2026-09-11), which
            # keeps the gap the same at every font size instead of closing as
            # the type grows.
            name_pt = (label_size or 8) + 3.0
            fig.subplots_adjust(bottom=fig.subplotpars.bottom
                                + (name_pt / 72.0 + 0.09) / (height or FIG_H))
        if grow:
            # GROW THE PANELS INTO THE SPARE ROOM, measured rather than guessed.
            # `tight_layout` fits the axes so that every label lands on the
            # canvas and then stops, which leaves air between the panels and the
            # things around them.
            #
            # ⚠ THE SPARE ROOM IS NOT AT THE CANVAS EDGE. Both rotated side
            # labels are pinned to the edges (x = 0.006 and 0.998) and the
            # column names sit at y = 0.008, so the figure's tight bbox already
            # touches all four sides and measuring against it returns zero. What
            # is actually free is the GAP BETWEEN the axes block and each of
            # those neighbours, which is what is measured here.
            #
            # ⚠ IT RUNS AFTER THE BOTTOM RESERVE AND BEFORE THE RULES AND THE
            # COLUMN NAMES. `subplots_adjust` rebuilds every panel position from
            # the gridspec, so a growth applied before it is silently thrown
            # away; the rules and the names are placed FROM the positions, so a
            # growth applied after them leaves both pointing at where the panels
            # used to be.
            fig.canvas.draw()
            rend = fig.canvas.get_renderer()
            W, H = fig.get_size_inches()
            inv = fig.transFigure.inverted()
            allax = [a for row in axes for a in row]
            allax += [a for v in twins.values() for a in v]
            pos = {a: a.get_position() for a in allax}

            def ink(a):
                bb = a.get_tightbbox(rend)
                p0 = inv.transform((bb.x0, bb.y0)); p1 = inv.transform((bb.x1, bb.y1))
                return p0[0] * W, p1[0] * W, p0[1] * H, p1[1] * H

            ib = [ink(a) for a in allax]
            ink_x0, ink_x1 = min(v[0] for v in ib), max(v[1] for v in ib)
            ink_y0, ink_y1 = min(v[2] for v in ib), max(v[3] for v in ib)

            def edge(art, which):
                if art is None:
                    return None
                bb = art.get_window_extent(rend)
                p0 = inv.transform((bb.x0, bb.y0)); p1 = inv.transform((bb.x1, bb.y1))
                return dict(x0=p0[0] * W, x1=p1[0] * W,
                            y0=p0[1] * H, y1=p1[1] * H)[which]

            SAFE = 0.02
            name_h = ((name_lines * name_size * 1.2 / 72.0 + 0.09)
                      if panel_titles else 0.0)
            free = {
                "left": ink_x0 - (edge(sy, "x1") or 0.0) - SAFE,
                "right": (edge(ry, "x0") or W) - ink_x1 - SAFE,
                "top": (edge(key, "y0") or H) - ink_y1 - SAFE,
                "bottom": ink_y0 - name_h - SAFE,
            }
            free = {k: max(0.0, v) for k, v in free.items()}
            print("  spare room (in): " + ", ".join(
                f"{k} {v:.3f}" for k, v in sorted(free.items())))
            x0b = min(p_.x0 for p_ in pos.values())
            x1b = max(p_.x1 for p_ in pos.values())
            y0b = min(p_.y0 for p_ in pos.values())
            y1b = max(p_.y1 for p_ in pos.values())
            nx0, nx1 = x0b - free["left"] / W, x1b + free["right"] / W
            ny0, ny1 = y0b - free["bottom"] / H, y1b + free["top"] / H
            sfx = (nx1 - nx0) / (x1b - x0b)
            sfy = (ny1 - ny0) / (y1b - y0b)
            for a, p_ in pos.items():
                a.set_position([nx0 + (p_.x0 - x0b) * sfx,
                                ny0 + (p_.y0 - y0b) * sfy,
                                p_.width * sfx, p_.height * sfy])
            print(f"  grown by x{sfx:.3f} across, x{sfy:.3f} down")
        if column_rules and len(labels) > 1:
            # A hairline in each gutter, running the height of the block. The
            # columns are three separate runs of the same trace, and without a
            # divider the twelve panels read as one grid in which a row might
            # be comparable across columns -- which it is not, since each arm
            # ranks its own instances. A rule is the cheapest thing that says
            # "these four belong together"; a tint would sit under the palest
            # class band and change its colour.
            fig.canvas.draw()
            tops = [axes[0][j].get_position() for j in range(len(labels))]
            bots = [axes[-1][j].get_position() for j in range(len(labels))]
            y0 = min(b.y0 for b in bots) - 0.055
            y1 = max(t.y1 for t in tops) + 0.018
            # ⚠ THE MIDDLE OF THE GUTTER IS NOT EMPTY. The next column's tick
            # numbers hang to the LEFT of its axes, so a rule at the midpoint
            # lands on them. The rule goes midway between this column's right
            # edge and the leftmost ink of the next one, which is measured
            # rather than guessed.
            rend = fig.canvas.get_renderer()
            inv = fig.transFigure.inverted()
            for j in range(len(labels) - 1):
                left = min(inv.transform(
                    (axes[r][j + 1].get_tightbbox(rend).x0, 0))[0]
                    for r in range(len(RANKS)))
                # ⚠ THIS COLUMN HAS INK TO THE RIGHT OF ITS AXES TOO. With an
                # overlay the right-hand twin carries tick numbers OUTSIDE the
                # frame, and the old midpoint was taken from the FRAME, so the
                # rule landed on those numbers. Both sides of the gutter are
                # measured now: the rightmost ink of this column against the
                # leftmost ink of the next one.
                own = [axes[r][j] for r in range(len(RANKS))] + twins.get(j, [])
                rightmost = max(inv.transform(
                    (a.get_tightbbox(rend).x1, 0))[0] for a in own)
                x = 0.5 * (rightmost + max(left, rightmost))
                w = fig.get_size_inches()[0]
                print(f"  column rule {j}: gutter "
                      f"{rightmost * w:.3f}..{left * w:.3f} in, rule at "
                      f"{x * w:.3f} in (axes frame ends {tops[j].x1 * w:.3f})")
                fig.add_artist(plt.Line2D([x, x], [max(y0, 0.0), min(y1, 1.0)],
                                          color="#bbbbbb", lw=0.5,
                                          transform=fig.transFigure))
            for j, lab in enumerate(labels):
                bb = axes[-1][j].get_position()
                fig.text(0.5 * (bb.x0 + bb.x1), 0.008, lab, ha="center",
                         va="bottom", fontsize=name_size, linespacing=1.0)
        fig.canvas.draw()
        kb = key.get_window_extent(fig.canvas.get_renderer())
        W_px = fig.get_size_inches()[0] * fig.dpi
        print(f"  key: {kb.width / fig.dpi:.2f} in wide, "
              f"{min(kb.x0, W_px - kb.x1) / fig.dpi:+.3f} in from the nearer edge"
              + ("  ⚠ OFF THE CANVAS" if kb.x0 < 0 or kb.x1 > W_px else ""))
        # The drawn panel size, printed so that a figure with fewer rows can be
        # given the SAME panel as the one it is derived from: the canvas height
        # to ask for is this height times the new row count plus whatever the
        # key, the x label and the arm names take, which is what the difference
        # between two of these lines tells you.
        pb = axes[0][0].get_position()
        print(f"  panel: {pb.width * fig.get_size_inches()[0]:.3f} x "
              f"{pb.height * fig.get_size_inches()[1]:.3f} in "
              f"({len(RANKS)} rows on a {fig.get_size_inches()[1]:.2f} in canvas)")
        check_side_labels(fig, [("left", sy), ("right", ry)])
        ps.save(fig, out)


def report(data, cover):
    print(f"{'arm':13s} {'attr%':>6s} {'rank':>4s} "
          f"{'chat%':>6s} {'deep%':>6s} {'agent%':>6s} {'peak req':>9s}")
    for _, label in ARMS:
        for rank in RANKS:
            if (label, rank) not in data:
                continue
            piv = data[(label, rank)]
            tot = piv.sum(axis=1)
            sh = piv.sum() / tot.sum() * 100.0 if tot.sum() > 0 else piv.sum() * 0
            print(f"{label:13s} {cover.get(label, np.nan):6.1f} {rank:4d} "
                  f"{sh['chat']:6.1f} {sh['deepresearch']:6.1f} {sh['swe']:6.1f} "
                  f"{tot.max():9,.0f}")


def out_dir(a):
    """Where this run writes: `paper_figures/final/` with --final, here
    otherwise. The paper's own copies are kept apart from the exploratory
    ones (2026-09-17, at the author's request)."""
    return ps.final if a.final else (lambda n: os.path.join(HERE, n))


def main():
    global CSV_MIX, CSV_SUM, REPEAT, VERDICTS, RANKS, ARMS, ATT_LABEL, TBT_LABEL, GRID_ON, ATT_COLOR, TBT_COLOR
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None,
                    help="arm keys to draw, in the order given")
    ap.add_argument("--col", action="store_true",
                    help="draw at one column instead of the text width")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--muted", action="store_true",
                    help="alias for --palette muted")
    ap.add_argument("--palette", choices=sorted(PALETTES), default=None,
                    help="class colours: rdylbu (default), muted, pubugn")
    ap.add_argument("--fig-width", type=float, default=None,
                    help="canvas width in inches; overrides --col's "
                         "3.335 and the 7.0 default")
    ap.add_argument("--fig-height", type=float, default=None,
                    help="canvas height in inches")
    ap.add_argument("--column-rules", action="store_true",
                    help="draw a hairline in each gutter so a column reads as "
                         "one block")
    ap.add_argument("--panel-titles", action="store_true",
                    help="name every panel \"Instance k\" above itself and move "
                         "the arm name under the bottom row")
    ap.add_argument("--row-tags", action="store_true",
                    help="name the rows #1..#N in the left margin instead of "
                         "putting \"Instance k\" where the y axis name goes")
    ap.add_argument("--relabel", default=None,
                    help="rename arms in the titles, key=Label,key=Label")
    ap.add_argument("--dense-y", action="store_true",
                    help="put the numbers on BOTH y axes of every panel; the "
                         "default does that only when three columns or fewer "
                         "leave room for it")
    ap.add_argument("--label-size", type=float, default=None,
                    help="point size of the panels' tick and axis labels; the "
                         "one-column default is 6")
    ap.add_argument("--req-max", type=float, default=None,
                    help="fix the requests axis of EVERY panel at 0..N instead "
                         "of giving each column its own top")
    ap.add_argument("--req-ticks", default=None,
                    help="tick values for the fixed requests axis, comma "
                         "separated, e.g. 100,200,300,430")
    ap.add_argument("--req-max-skip", default=None,
                    help="arm KEYS whose column keeps its own requests top "
                         "despite --req-max, comma separated")
    ap.add_argument("--tbt-max", type=float, default=None,
                    help="fix the latency axis at 0..N ms, linear, ticks every "
                         "50; without it the axis is logarithmic")
    ap.add_argument("--tbt", action="store_true",
                    help="add a right-hand axis per panel: that instance's mean "
                         "inter-token latency, from its own counters")
    ap.add_argument("--req-nticks", type=int, default=None,
                    help="evenly spaced value labels on every panel's left axis")
    ap.add_argument("--height", choices=["resident", "batch"], default="resident",
                    help="what a band's total height is: requests the client "
                         "recorded as on that instance (default), or the "
                         "engine's own running batch with the class split "
                         "taken from the residency")
    ap.add_argument("--grow", action="store_true",
                    help="after the layout, expand the panels into whatever "
                         "margin is left on each side; the amount is measured "
                         "from the figure's own ink and printed")
    ap.add_argument("--ranks", default=None,
                    help="which instance rows to draw, comma separated, e.g. "
                         "1,2. The rank is fixed by the whole hour's chat "
                         "residency, so dropping rows changes what is SHOWN "
                         "and not which engine a row is")
    ap.add_argument("--zoom", default=None,
                    help="draw a column over its own interval instead of the "
                         "whole hour, as KEY=LO:HI in minutes, comma "
                         "separated, e.g. llmdslot75=14:17,polyservept75=33:36."
                         " All three series of that column are recomputed on a "
                         "finer grid; the columns are then NOT comparable "
                         "across time and the caption must say so")
    ap.add_argument("--reject", action="store_true",
                    help="overlay the arm's rejection rate per minute on the "
                         "LEFT axis, mapped so the top of the panel is 100%%; "
                         "it is fleet-wide, so the same curve is repeated in "
                         "every panel of a column")
    ap.add_argument("--attain-left", action="store_true",
                    help="with --tbt: also draw each instance's attainment, on "
                         "the LEFT axis with the top of the panel as 100%%")
    ap.add_argument("--preset", choices=["exp109", "exp126", "exp138", "exp138all"], default="exp109",
                    help="which experiment's tables, runs and instance count")
    ap.add_argument("--final", action="store_true",
                    help="write into paper_figures/final/ instead of here")
    ap.add_argument("--tbt-color", default=None,
                    help="colour of the time-between-tokens curve (default #d62728)")
    ap.add_argument("--att-color", default=None,
                    help="colour of the attainment curve (default #e08214)")
    ap.add_argument("--grid", action="store_true",
                    help="dotted grey guides at the x and y ticks of every panel")
    ap.add_argument("--tbt-label", default=None,
                    help="key name of the red time-between-tokens curve")
    ap.add_argument("--att-label", default=None,
                    help="key name of the left-axis attainment curve")
    ap.add_argument("--attain", action="store_true",
                    help="add a right-hand axis per panel: the share of the "
                         "requests THAT INSTANCE was given which met the "
                         "ladder95 rule, in 90 s windows every 30 s")
    ap.add_argument("--att-win", type=float, default=ATT_WIN,
                    help="attainment window in seconds")
    ap.add_argument("--att-step", type=float, default=ATT_STEP,
                    help="spacing between attainment windows in seconds")
    ap.add_argument("--att-min-n", type=int, default=ATT_MIN_N,
                    help="a window with fewer scored requests is left blank")
    a = ap.parse_args()
    if a.preset == "exp126":
        # EXP-126: 8 x Llama-3.1-8B, halved budgets, repeat 1 of each arm. The
        # verdicts are the halved-budget scoring; the process must be started
        # with the halved FS_* budgets so the loader agrees with them.
        CSV_MIX = os.path.join(MIX, "hour_engine_mix_exp126.csv")
        CSV_SUM = os.path.join(MIX, "hour_summary_exp126.csv")
        REPEAT = "exp126h62r1"
        VERDICTS = os.path.join(ROOT, "results", "aggregate_analysis",
                                "ladder95_halved", "verdicts")
        RANKS = list(range(1, 9))
        ARMS = [("vllmcache", "vLLM-router"), ("polyservep", "PolyServe"),
                ("slot", "Llumnix SLO"), ("llmdslot", "llm-d"),
                ("fsv3capgnofrc", "FluidServe")]
    if a.preset == "exp138":
        # FluidServe as deployed (EXP-109 repeat 1) beside FluidServe with class
        # affinity AND the instance cap both turned off (EXP-138 repeat 1), on
        # the same one-hour trace and the same fleet. The table holds only
        # these two runs and was written apart from the EXP-109 one.
        CSV_MIX = os.path.join(MIX, "hour_engine_mix_exp138.csv")
        CSV_SUM = os.path.join(MIX, "hour_summary_exp138.csv")
        REPEAT = "exp109r1_fsv3capgnofrct75|exp138r1_noaffnocaphour"
        ARMS = [("fsv3capgnofrct75", "FluidServe"),
                ("noaffnocaphour", "FluidServe w/o Affinity")]
        # ⚠ "w/o Affinity" names an arm that ALSO has the instance cap off
        # (2026-09-15, the label shortened at the author's request); the
        # caption has to say both are off.
    if a.preset == "exp138all":
        # the two EXP-138 columns plus llm-d and PolyServe from EXP-109 repeat
        # 1, the same one-hour trace and fleet as the FluidServe column
        CSV_MIX = [os.path.join(MIX, "hour_engine_mix.csv"),
                   os.path.join(MIX, "hour_engine_mix_exp138.csv")]
        CSV_SUM = [os.path.join(MIX, "hour_summary.csv"),
                   os.path.join(MIX, "hour_summary_exp138.csv")]
        REPEAT = ("exp109r1_fsv3capgnofrct75|exp138r1_noaffnocaphour"
                  "|exp109r1_llmdslot75|exp109r1_polyservept75")
        ARMS = [("fsv3capgnofrct75", "FluidServe"),
                ("noaffnocaphour", "FluidServe\nw/o Affinity"),
                ("llmdslot75", "llm-d"), ("polyservept75", "PolyServe")]
    ATT_LABEL = a.att_label
    TBT_LABEL = a.tbt_label
    GRID_ON = a.grid
    if a.tbt_color:
        TBT_COLOR = a.tbt_color
    if a.att_color:
        # per figure, because the attainment curve has to stay apart from the
        # class colours and those differ between figures (2026-09-17)
        ATT_COLOR = a.att_color
    arms = ARMS if not a.arms else [(k, dict(ARMS)[k]) for k in a.arms]
    # ⚠ RENAMING AN ARM IN A FIGURE DOES NOT RENAME IT IN THE PROJECT.
    # CLAUDE.md binds "Llumnix" to the load-balance policy and "Llumnix SLO"
    # to the SLO-aware one; calling the second one "Llumnix" is unambiguous
    # only while the first is absent from the figure, and the caption has to
    # say which policy it is.
    if a.relabel:
        ren = dict(kv.split("=", 1) for kv in a.relabel.split(",") if "=" in kv)
        arms = [(k, ren.get(k, l)) for k, l in arms]
        print(f"  relabelled: {ren}")

    if a.ranks:
        RANKS = [int(v) for v in a.ranks.split(",") if v.strip()]
        bad = [r for r in RANKS if r < 1 or r > 4]
        if bad:
            sys.exit(f"--ranks: {bad} is not among the four instances")
        print(f"  drawing instance rows {RANKS} of 1,2,3,4")
    zoom = None
    if a.zoom:
        keys = dict(arms)
        zoom = {}
        for part in a.zoom.split(","):
            if "=" not in part:
                sys.exit(f"--zoom wants KEY=LO:HI, got {part!r}")
            k, span = part.split("=", 1)
            if k not in keys:
                sys.exit(f"--zoom names {k!r}, which is not a drawn arm "
                         f"({', '.join(keys)})")
            lo, hi = (float(v) for v in span.split(":"))
            if not 0.0 <= lo < hi <= XMAX:
                sys.exit(f"--zoom {k}: {lo}-{hi} is not inside 0-{XMAX:.0f} min")
            zoom[k] = (lo, hi)
        # the drawing is keyed by LABEL, the data by arm KEY
        zoom_lab = {keys[k]: v for k, v in zoom.items()}
    data, cover = collect(arms, zoom, a.height)
    report(data, cover)
    if a.tbt or a.attain:
        # ⚠ ONLY THE ABSOLUTE FIGURE GETS THE OVERLAY, and it is written under a
        # new name. The share version answers "what was this instance made of",
        # where a latency has no partner quantity on the left; and overwriting
        # the existing pair would change two figures that are already in use.
        overlay = "attain" if a.attain else "tbt"
        att_left = None
        if a.attain_left:
            if not a.tbt:
                sys.exit("--attain-left draws beside --tbt; add --tbt")
            att_left = attain_series(arms, a.att_win, a.att_step, a.att_min_n,
                                     zoom)
            report_attain(att_left)
        if a.attain:
            if a.tbt:
                sys.exit("--tbt and --attain both draw the right-hand axis; "
                         "pick one, or use --tbt --attain-left")
            tbt = attain_series(arms, a.att_win, a.att_step, a.att_min_n,
                                zoom)
            report_attain(tbt)
        else:
            tbt = tbt_series(arms)
            report_tbt(tbt)
        rej = reject_series(arms, zoom=zoom) if a.reject else None
        geo = (dict(width=ps.COL_W, height=2.60,
                    label_size=a.label_size or 6) if a.col
               else ({"label_size": a.label_size} if a.label_size else {}))
        if a.fig_width:
            geo["width"] = a.fig_width
        if a.fig_height:
            geo["height"] = a.fig_height
        pal = PALETTES[a.palette] if a.palette else (CLASS_COLOR_MUTED
                                                     if a.muted else None)
        if pal:
            geo["colors"] = pal
        if a.tbt_max and not a.attain:
            n = sum(int(((v > a.tbt_max) & np.isfinite(v) & (m >= 0)
                         & (m <= XMAX)).sum()) for m, v, _ in tbt.values())
            tot = sum(int((np.isfinite(v) & (m >= 0) & (m <= XMAX)).sum())
                      for m, v, _ in tbt.values())
            print(f"  latency axis fixed at 0..{a.tbt_max:.0f} ms: {n} of {tot} "
                  f"drawn seconds ({100.0 * n / max(tot, 1):.3f}%) are above it "
                  f"and are cut at the top of their panel")
        skip = {dict(ARMS)[k] for k in (a.req_max_skip or "").split(",") if k}
        ticks = ([float(v) for v in a.req_ticks.split(",")]
                 if a.req_ticks else None)
        if a.req_max:
            n = tot = 0
            for (lab, rank), piv in data.items():
                if lab in skip:
                    continue
                v = piv.sum(axis=1).to_numpy()
                n += int((v > a.req_max).sum()); tot += len(v)
            print(f"  requests axis fixed at 0..{a.req_max:.0f}: {n} of {tot} "
                  f"drawn windows ({100.0 * n / max(tot, 1):.3f}%) are above it "
                  f"and are cut at the top of their panel")
        tag = (("_att" if a.attain else "_tbt") + ("_att" if a.attain_left else "")
               + ("_rej" if a.reject else ""))
        pdf = out_dir(a)(f"class_mix_hour_abs{a.suffix}{tag}.pdf")
        build(data, cover, pdf, share=False, arms=arms, tbt=tbt,
              overlay=overlay, reject=rej,
              zoom=(zoom_lab if zoom else None),
              dense_y=a.dense_y or len(arms) <= 3, tbt_max=a.tbt_max,
              req_max=a.req_max, req_ticks=ticks, req_skip=skip,
              row_tags=a.row_tags, panel_titles=a.panel_titles, column_rules=a.column_rules,
              grow=a.grow, height_kind=a.height, req_nticks=a.req_nticks,
              att_left=att_left, **geo)
        if a.attain:
            write_attain_csv(tbt, pdf[:-4] + ".csv", a.att_win, a.att_step,
                             rej)
        else:
            write_tbt_csv(tbt, pdf[:-4] + ".csv")
        print("  the bands are the same table as class_mix_hour_abs.pdf "
              "(this arm subset); only the right-hand axis is new")
        return 0
    # One column with three arms leaves each panel about 0.85 in across, so the
    # type on the axes comes down; the five-arm figure at the text width keeps
    # the paper's 8 pt.
    geo = (dict(width=ps.COL_W, height=2.60,
                label_size=a.label_size or 6) if a.col
           else ({"label_size": a.label_size} if a.label_size else {}))
    if a.fig_width:
        geo["width"] = a.fig_width
    if a.fig_height:
        geo["height"] = a.fig_height
    pal = PALETTES[a.palette] if a.palette else (CLASS_COLOR_MUTED
                                                 if a.muted else None)
    if pal:
        geo["colors"] = pal
    # The requests axis takes the same options here as it does with --tbt, so a
    # figure drawn without the latency overlay can still be put beside one drawn
    # with it and read at the same height.
    skip = {dict(ARMS)[k] for k in (a.req_max_skip or "").split(",") if k}
    ticks = ([float(v) for v in a.req_ticks.split(",")]
             if a.req_ticks else None)
    dense = a.dense_y or len(arms) <= 3
    pdf = out_dir(a)(f"class_mix_hour{a.suffix}.pdf")
    build(data, cover, pdf, share=True, arms=arms, dense_y=dense,
          row_tags=a.row_tags, panel_titles=a.panel_titles, column_rules=a.column_rules, **geo)
    write_csv(data, cover, pdf[:-4] + ".csv")
    build(data, cover, out_dir(a)(f"class_mix_hour_abs{a.suffix}.pdf"),
          share=False, arms=arms, dense_y=dense, req_max=a.req_max,
          req_ticks=ticks, req_skip=skip, row_tags=a.row_tags,
          panel_titles=a.panel_titles, column_rules=a.column_rules, **geo)
    print(f"  class_mix_hour_abs{a.suffix}.pdf is the same table in requests; "
          f"both are in class_mix_hour{a.suffix}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
