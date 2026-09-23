#!/usr/bin/env python3
"""Paper figure: one engine, one quarter of an hour, under Llumnix SLO.

  engine_window_<arm>_<port>.pdf        3.335 x 2.55 in, one column
  engine_window_<arm>_<port>_wide.pdf   7.0 x 2.10 in, `figure*`
  engine_window_<arm>_<port>.csv        exactly the values drawn

  `--arm {slo,polyserve,fluidserve,llmd,vllm} --port 8000 --t-lo 33 --t-hi 43`

  (a) how full the engine's KV pool was, and how many requests it was running
  (b) the inter-token latency that engine reported, and the share of the
      requests it was given that met their deadline

WHY ONE ENGINE AND WHY THESE FIFTEEN MINUTES. The fleet average describes no
engine when a control plane makes the engines unequal, and Llumnix SLO does:
this is instance 8002 between minutes 33 and 43 of the hour-long mix-shift
trace: the window in which its KV pool refills, its running batch grows from
about 110 requests to over 300, and the attainment of the work it was given
falls back. Everything is drawn on one x axis so a change in the pool or the batch
can be read against the latency and the attainment in the same minute.

WHAT EACH LINE IS.
  KV occupancy      `vllm:kv_cache_usage_perc`, this engine's own gauge, as
                    scraped at 1 s and not smoothed.
  Running batch     `vllm:num_requests_running`, the same. It is a gauge of the
                    requests in the running batch, not of arrivals.
  Inter-token time  the per-second increments of this engine's
                    `vllm:inter_token_latency_seconds_sum` divided by the
                    increments of its `_count`, which is the token-weighted mean
                    over the tokens the engine produced in that second. Both
                    increments are smoothed over 3 s BEFORE the division,
                    because a second in which few tokens finished gives a ratio
                    that moves without the latency moving.
  SLO attainment    the requests the scheduler DISPATCHED TO THIS ENGINE,
                    windowed by arrival over 90 s every 30 s, scored by the
                    token-level cumulative deadline and counted on the ADMITTED
                    denominator: of the requests this engine was given and that
                    finished, the share that met the rule.

⚠ THE TWO HALVES HAVE DIFFERENT DENOMINATORS AND DIFFERENT ORIGINS. The gauges
are the engine's, per second; the attainment is the client's records joined to
the scheduler's dispatch log, per 90 s window. A request is placed in the window
of its ARRIVAL, so a miss appears at the moment the request arrived and not at
the moment it was declared late. Reading (b) as if it moved with (a) within a
few seconds is therefore wrong; the correct reading is over minutes.

⚠ ADMITTED, SO REJECTION IS NOT IN IT. Llumnix SLO refuses 39.4% of arrivals
over this run. Those requests never reach an engine and are absent from both
panels, so a rise in (b) can mean the engine served its work better or that the
control plane sent it less of the work it would have failed.

DATA. EXP-109 (2026-08-31), repeat 1 of the Llumnix SLO arm
(`260831_2346_exp109r1_slot75_shift`), engine 8002, minutes 25-40 measured from
the first arrival. Attribution covers 100% of this run's admitted requests.

    python3 paper_figures/fig_engine8002_window.py
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, FixedLocator  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis_scripts", "request_level"))
import paper_style as ps  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ECO = _load("eco", os.path.join(ROOT, "analysis_scripts", "request_level",
                                "engine_class_occupancy.py"))

# Any arm of the hour, any engine, any window: which engine is worth drawing is
# a property of the arm, not a constant. PolyServe puts chat on 8000 and 8001,
# the agent class on 8002 and deep research on 8003; Llumnix SLO mixes, so its
# engines differ only in load.
RUNS = {
    "slo": ("Llumnix SLO", "260831_2346_exp109r1_slot75_shift"),
    "polyserve": ("PolyServe", "260831_2232_exp109r1_polyservept75_shift"),
    "fluidserve": ("FluidServe", "260831_2015_exp109r1_fsv3capgnofrct75_shift"),
    "llmd": ("llm-d", "260831_2128_exp109r1_llmdslot75_shift"),
    "vllm": ("vLLM-router", "260901_2137_exp109r1_vllmcachet75_shift"),
}
ARM_KEY, PORT = "slo", 8002
RUN = RUNS[ARM_KEY][1]
ARM = RUNS[ARM_KEY][0]
T_LO, T_HI = 33.0, 43.0          # minutes from the first arrival
# ⚠ ALMOST NO SMOOTHING (2026-09-08). The gauges are drawn as scraped, at 1 s,
# so that a spike is the engine's and not the filter's; a 10 s box filter was
# used until now and it rounded the corners of exactly the transitions this
# figure is about. The inter-token latency keeps a 3 s filter for a different
# reason: it is a RATIO of two per-second increments, and in a second where the
# engine finished few tokens the denominator is small enough that the ratio
# swings without the latency having changed.
SMOOTH_S = 1
SMOOTH_ITL_S = 3
WIN, STEP = 90.0, 30.0           # the attainment window, as in the hour figures
MIN_IN_WINDOW = 10
VERDICTS = os.path.join(ROOT, "results", "aggregate_analysis", "ladder95",
                        "verdicts")
KV_C, BATCH_C, QUEUE_C = "#1f77b4", "#8c564b", "#9467bd"
# Orange rather than red for the per-token time (2026-09-08): red reads as an
# alarm beside a green line that is a score, and neither panel is about failure.
ITL_C, ATT_C = "#ff7f0e", "#2ca02c"
# The comparison arm gets its own two colours rather than the same two dashed:
# at this panel size a dash pattern is the first thing that disappears, and the
# two arms are the comparison the panel exists to make.
ITL_C2, ATT_C2 = "#7b3294", "#17becf"
FIG_H = 2.55


def smooth(x, k):
    return np.convolve(x, np.ones(k) / k, mode="same") if k > 1 else x


def gauges(run, t0):
    """minutes, KV %, running batch, waiting queue, inter-token latency ms.

    The waiting queue is `vllm:num_requests_waiting`: requests the engine has
    accepted and not started. It is the third thing the placement decision
    changes and it moves on its own -- an engine can hold a full batch with an
    empty queue and the reverse.
    """
    path = os.path.join(run, "server_metrics", f"engine_{PORT}.jsonl")
    ts, kv, run_q, wait_q, isum, icnt = [], [], [], [], [], []
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not r.get("ok", True):
                continue

            def pick(prefix):
                return next((v for k, v in r.items()
                             if k.startswith(prefix) and v is not None), None)

            a = pick("vllm:kv_cache_usage_perc")
            b = pick("vllm:num_requests_running")
            w = pick("vllm:num_requests_waiting")
            c = pick("vllm:inter_token_latency_seconds_sum")
            d = pick("vllm:inter_token_latency_seconds_count")
            if a is None or b is None:
                continue
            ts.append(r["t"]); kv.append(a); run_q.append(b)
            wait_q.append(0.0 if w is None else w)
            isum.append(np.nan if c is None else c)
            icnt.append(np.nan if d is None else d)
    t = np.array(ts)
    grid = np.arange(t[0], t[-1], 1.0)
    kvp = smooth(np.interp(grid, t, np.array(kv, float)) * 100.0, SMOOTH_S)
    bat = smooth(np.interp(grid, t, np.array(run_q, float)), SMOOTH_S)
    wait = smooth(np.interp(grid, t, np.array(wait_q, float)), SMOOTH_S)
    ds = np.diff(np.interp(grid, t, np.nan_to_num(np.array(isum, float))),
                 prepend=np.nan)
    dc = np.diff(np.interp(grid, t, np.nan_to_num(np.array(icnt, float))),
                 prepend=np.nan)
    ds, dc = np.nan_to_num(ds), np.nan_to_num(dc)
    ds[ds < 0] = 0; dc[dc < 0] = 0
    ds, dc = smooth(ds, SMOOTH_ITL_S), smooth(dc, SMOOTH_ITL_S)
    itl = np.divide(ds, dc, out=np.full_like(ds, np.nan), where=dc > 0) * 1000.0
    m = (grid - t0) / 60.0
    keep = (m >= T_LO) & (m <= T_HI)
    return m[keep], kvp[keep], bat[keep], wait[keep], itl[keep]


def attainment(run_dir, t0):
    """minutes, admitted attainment (%) among the requests sent to this engine."""
    r = ECO.load_run(run_dir)
    j, n = ECO.attribute_engines(run_dir, r)
    j = j[j["engine_port"] == PORT].copy()
    v = pd.read_csv(os.path.join(VERDICTS, os.path.basename(run_dir) + ".csv"))
    keys = ["task_id", "call_index", "iteration"]
    # The verdict columns are renamed before the join: `load_run` already
    # publishes `cutoff`, `rejected` and `errored` of its own, and merging on
    # top of them silently produces `_x`/`_y` pairs where the wrong one is easy
    # to pick up. The scorer's are the ones the rule was applied to.
    v = v[keys + ["ladder_ok", "cutoff", "rejected"]].rename(
        columns={"cutoff": "v_cutoff", "rejected": "v_rejected"})
    j = j.merge(v, on=keys, how="left")
    j = j[j["ladder_ok"].notna()]
    j["rel_min"] = (pd.to_numeric(j["start_time"], errors="coerce") - t0) / 60.0
    # ⚠ THE WINDOWS ARE CUT FROM THE WHOLE RUN AND THEN CROPPED, not built
    # inside the drawn interval. Building them inside it would leave the series
    # short of both edges by half a window (45 s each side) for no reason: those
    # windows exist, they are simply centred near the boundary.
    xs, ys, ns = [], [], []
    c = WIN / 120.0
    while c <= float(j["rel_min"].max()) - WIN / 120.0 + 1e-9:
        if c < T_LO - 1e-9 or c > T_HI + 1e-9:
            c += STEP / 60.0
            continue
        g = j[(j["rel_min"] >= c - WIN / 120.0) & (j["rel_min"] < c + WIN / 120.0)]
        served = g[~g["v_cutoff"].astype(bool) & ~g["v_rejected"].astype(bool)]
        if len(served) >= MIN_IN_WINDOW:
            xs.append(c)
            ys.append(100.0 * float(served["ladder_ok"].astype(bool).mean()))
            ns.append(len(served))
        c += STEP / 60.0
    return np.array(xs), np.array(ys), np.array(ns)


def write_csv(g, att, path, with_queue=False):
    m, kv, bat, wait, itl = g
    rows = [{"series": "kv_cache_usage_pct", "minute": float(a), "value": float(b)}
            for a, b in zip(m, kv)]
    rows += [{"series": "running_batch_requests", "minute": float(a),
              "value": float(b)} for a, b in zip(m, bat)]
    if with_queue:
        rows += [{"series": "waiting_queue_requests", "minute": float(a),
                  "value": float(b)} for a, b in zip(m, wait)]
    rows += [{"series": "inter_token_latency_ms", "minute": float(a),
              "value": float(b)} for a, b in zip(m, itl)]
    rows += [{"series": "slo_attainment_admitted_pct", "minute": float(a),
              "value": float(b), "n_requests": int(c)}
             for a, b, c in zip(*att)]
    df = pd.DataFrame(rows)
    df["run"], df["engine_port"] = RUN, PORT
    df["gauge_smoothing_s"] = SMOOTH_S
    df["itl_smoothing_s"] = SMOOTH_ITL_S
    df["attainment_window_s"] = WIN
    df.to_csv(path, index=False, float_format="%.4f")
    print(f"wrote {path}  ({len(df)} rows)")


def build(g, att, out, width, height, kv_floor=0.0, batch_floor=0.0,
          normalize=False, over_markers=True, side=False, notes=True,
          frame=False, tight_ylim=False, fit_ylim=False, label_size=None,
          dense_ticks=False, with_queue=False, trim_top=True, itl2=None,
          arm2=None, att2=None):
    """`kv_floor`/`batch_floor` cut the bottom off the two gauge axes;
    `normalize` divides each of them by its own peak in the window and puts
    both on one axis.

    ⚠ BOTH OF THOSE MAKE A SERIES LOOK MORE VARIABLE THAN IT IS. An axis that
    starts at 30% turns a 47-to-91 swing into a curve that reaches the floor,
    and dividing by the peak hides that one series moved between 40 and 100
    while the other moved between 20 and 100 of quantities that are not
    comparable to begin with. The unzoomed, unnormalised figure is the one to
    show unless the point being made is about the SHAPE of the two curves, and
    the caption has to say which one it is.
    """
    m, kv, bat, wait, itl = g
    # Smaller type on the axes buys plotting area, and only there: the legend
    # and the panel captions keep their size, because they are read once while
    # the curves are read continuously. The axes box in inches is printed so the
    # trade is a measured one.
    style = dict(ps.STYLE)
    if label_size:
        style.update({"xtick.labelsize": label_size,
                      "ytick.labelsize": label_size,
                      "axes.labelsize": label_size})
    with plt.rc_context(style):
        # Side by side or stacked. Stacked shares one x axis and is the shape
        # for reading the two panels against each other minute by minute; side
        # by side gives each panel twice the width and half the height, which
        # suits a short window and a figure that has to sit in one row.
        fig, ax = plt.subplots(1 if side else 2, 2 if side else 1,
                               figsize=(width, height), sharex=not side)
        a1, a2 = ax
        b1, b2 = a1.twinx(), a2.twinx()

        if normalize:
            # Both scalings are computed INSIDE THE DRAWN WINDOW, never over the
            # hour: the question is how the two move against each other here.
            #   peak    x / max(x)          -- keeps the distance from zero, so
            #           a series that never falls below half its peak still sits
            #           in the top half.
            #   minmax  (x - min) / (max - min) -- spends the whole axis on the
            #           range each series actually covered in these minutes,
            #           which is what shows the SHAPE and what destroys any
            #           sense of how large the movement was.
            def mm(v):
                return 100.0 * (v - v.min()) / max(v.max() - v.min(), 1e-9)

            if normalize == "minmax":
                fk, fb, fw = mm(kv), mm(bat), mm(wait)
                lab = "Window min-max (%)"
                note = (f"KV {kv.min():.0f}-{kv.max():.0f}%, "
                        f"batch {bat.min():.0f}-{bat.max():.0f}")
            else:
                fk, fb = 100.0 * kv / kv.max(), 100.0 * bat / bat.max()
                fw = 100.0 * wait / max(wait.max(), 1e-9)
                lab = "Share of peak (%)"
                note = f"peak KV {kv.max():.0f}%, batch {bat.max():.0f}"
            if with_queue:
                note += (f", queue {wait.min():.0f}-{wait.max():.0f}"
                         if normalize == "minmax"
                         else f", queue {wait.max():.0f}")
            # Two axes even though both series are already on 0-100: the left
            # one is the KV curve's and the right one the batch curve's, so the
            # panel names each quantity beside its own colour instead of asking
            # the reader to carry the mapping over from the legend. The scale is
            # the same on both, which is the point of normalising.
            a1.plot(m, fk, color=KV_C, lw=0.9)
            b1.plot(m, fb, color=BATCH_C, lw=0.9)
            # The queue is OFF BY DEFAULT (`--with-queue`). On this engine it
            # is 0 at the median and 30 at its maximum over the window, and
            # min-max scaling stretches that 30 to the full axis, which reads as
            # a series swinging as widely as the other two. It is in the run's
            # metrics either way; the figure does not need to carry it to make
            # its point.
            if with_queue:
                a1.plot(m, fw, color=QUEUE_C, lw=0.9)
            # ⚠ NORMALISED, SO BOTH CURVES ARE ALREADY ON ONE SCALE. On the
            # wide canvas each keeps its own axis, which names the quantity
            # beside its colour; on one column that costs 0.3 in of the 1 in of
            # axis available, so the right-hand axis is dropped and the legend
            # is what identifies the two curves.
            if side and width < 4.0:
                a1.set_ylabel("Norm. (%)", labelpad=1.5)
                b1.set_yticks([])
            else:
                a1.set_ylabel("KV-cache (norm.)", color=KV_C, labelpad=1.5)
                b1.set_ylabel("Batch (norm.)", color=BATCH_C, labelpad=2.0)
            cap = 100 if tight_ylim else 105
            a1.set_ylim(0, cap)
            b1.set_ylim(0, cap)
            a1.set_yticks([0, 50, 100])
            # ⚠ AFTER the branch above, not before it: setting the ticks here
            # unconditionally is what put a second copy of 0/50/100 back on the
            # narrow panel whose right axis had just been cleared.
            b1.set_yticks([] if (side and width < 4.0) else [0, 50, 100])
            # The scaling constants, in the panel: without them the reader
            # cannot get back to the quantities.
            # The scaling constants stay in the panel: normalised curves cannot
            # be read back to quantities without them, and `lab` no longer has a
            # place on either axis.
            # ⚠ WITHOUT THIS NOTE THE PANEL CANNOT BE READ BACK TO QUANTITIES.
            # `--no-notes` removes it, and then the scaling constants have to be
            # in the caption instead; the script prints them.
            if notes:
                a1.text(0.985, 0.06, f"{lab}: {note}", transform=a1.transAxes,
                        ha="right", va="bottom", fontsize=5.6, color="#404040")
            print(f"    panel (a) scaling: {lab}: {note}")
        else:
            a1.plot(m, kv, color=KV_C, lw=0.9)
            a1.set_ylabel("KV (%)", color=KV_C, labelpad=1.5)
            b1.plot(m, bat, color=BATCH_C, lw=0.9)
            b1.set_ylabel("Batch (reqs)", color=BATCH_C, labelpad=2.0)
            if fit_ylim:
                # Each axis spans exactly what its own series covered in the
                # window shown. The curves then fill the panel the way the
                # min-max normalised version does, and the tick labels are still
                # the quantity: 23-100% of the pool, 59-392 requests. ⚠ NEITHER
                # AXIS STARTS AT ZERO, so vertical distance on this panel is not
                # proportional to the quantity and the two curves crossing means
                # nothing. It shows shape, and the caption has to say so.
                for ax_, v in ((a1, kv), (b1, bat)):
                    lo, hi = float(np.min(v)), float(np.max(v))
                    ax_.set_ylim(lo, hi)
                    ax_.set_yticks([lo, (lo + hi) / 2, hi])
                    ax_.set_yticklabels([f"{lo:.0f}", f"{(lo + hi) / 2:.0f}",
                                         f"{hi:.0f}"])
            else:
                a1.set_ylim(kv_floor, 100 if tight_ylim else 105)
                a1.set_yticks([t for t in (0, 25, 50, 75, 100) if t >= kv_floor])
                b1.set_ylim(batch_floor, None)

        a2.plot(m, itl, color=ITL_C, lw=0.9)
        # A second arm's per-token time, same engine index, same minutes, drawn
        # dashed in the same colour: the COLOUR is the quantity and the STYLE is
        # the control plane, so the panel does not need a second axis or a
        # second hue for a quantity it already shows.
        if itl2 is not None:
            a2.plot(m, itl2, color=ITL_C2, lw=0.9)
        a2.set_ylabel("TBT (ms)", labelpad=1.5)
        # ⚠ THE AXIS IS CUT AT THE 99TH PERCENTILE AND THE SECONDS ABOVE IT ARE
        # DRAWN ON THE TOP EDGE, not dropped. Unsmoothed, this series has a few
        # seconds three to five times the rest -- 2 of 600 above 200 ms, one at
        # 379 -- and letting the axis reach them spends two thirds of the panel
        # on two samples and flattens the 40-110 ms band the figure is read for.
        # The count and the maximum are printed in the panel so the cut is
        # visible rather than implied, and the CSV holds every value.
        ok = ~np.isnan(itl)
        pool = itl[ok] if itl2 is None else np.concatenate(
            [itl[ok], itl2[~np.isnan(itl2)]])
        top = max(120.0, float(np.ceil(np.percentile(pool, 99) / 20.0) * 20))
        over = ok & (itl > top)
        if over.any():
            # The markers can be switched off; the COUNT AND MAXIMUM CANNOT.
            # Without the note the cut axis silently drops the largest seconds
            # in the series.
            if over_markers:
                a2.plot(m[over], np.full(over.sum(), top), ls="none", marker="^",
                        ms=2.0, color=ITL_C, clip_on=False)
            # BOTTOM left. The top of this panel is taken by whichever of the
            # two series is high -- the latency spikes on one arm, the
            # attainment line on another -- and the floor is the one place that
            # is empty on both, because neither series approaches zero.
            if notes:
                a2.text(0.015, 0.03,
                        f"{over.sum()} s above, max {itl[ok].max():.0f} ms",
                        transform=a2.transAxes, ha="left", va="bottom",
                        fontsize=5.6, color=ITL_C)
            print(f"    latency axis cut at {top:.0f} ms: {over.sum()} s above, "
                  f"max {itl[ok].max():.0f} ms")
        a2.set_ylim(0, top)
        # Three labels, as on the other three axes of this figure: matplotlib
        # picks two here because the limit is a cut percentile rather than a
        # round number, and one axis labelled differently from its neighbours
        # reads as a different kind of axis.
        a2.set_yticks([0, round(top / 2), round(top)])
        b2.plot(att[0], att[1], color=ATT_C, lw=0.9)
        if att2 is not None:
            b2.plot(att2[0], att2[1], color=ATT_C2, lw=0.9)
        b2.set_ylabel("Request SLO (%)", labelpad=2.0)
        b2.set_ylim(0, 100 if tight_ylim else 105)
        b2.set_yticks([0, 50, 100])

        narrow = side and width < 4.0
        # The right panel's axes are black: two coloured axes there would say
        # that the axis belongs to one curve, and the panel's own legend already
        # says which curve is which.
        pairs = ((a1, b1, "#000000" if (narrow and normalize) else KV_C,
                  "#000000" if (narrow and normalize) else BATCH_C),
                 (a2, b2, "#000000", "#000000"))
        for a, b, lc, rc in pairs:
            a.set_xlim(T_LO, T_HI)
            # Ticks every whole minute where the window is short enough for
            # that to be readable, every five where it is not, so the same
            # script serves a 7-minute window and a 15-minute one.
            span = T_HI - T_LO
            step = 1 if span <= 8 else (2 if span <= 16 else 5)
            # Two panels on one column leave about an inch of axis each, and a
            # tick every minute then sets five labels in that inch. The
            # interval doubles until the labels stop colliding.
            if side and width < 4.0:
                step = max(step * 2, 2)
            if dense_ticks:
                # Five labelled ticks: the largest whole-minute interval that
                # still fits five of them inside the window, so the labels stay
                # integers instead of falling on 38.5 and 45.5. The right edge
                # can then be past the last label, and it carries a minor tick.
                step = 1
                for cand in range(1, int(T_HI - T_LO) + 1):
                    if int((T_HI - T_LO) // cand) + 1 >= 5:
                        step = cand
                    else:
                        break
            a.set_xticks(np.arange(np.ceil(T_LO), T_HI + 1e-9, step))
            if dense_ticks:
                a.xaxis.set_minor_locator(AutoMinorLocator(step))
                a.tick_params(axis="x", which="minor", length=1.2)
            a.grid(axis="both", **ps.GRID)
            a.set_axisbelow(True)
            a.tick_params(axis="y", colors=lc)
            b.tick_params(axis="y", colors=rc)
            # With `frame` the panel is boxed: the top spine then also draws
            # the ceiling of an axis whose highest tick is its limit, which is
            # what `tight_ylim` sets up.
            for sp in ("top",):
                a.spines[sp].set_visible(frame)
                b.spines[sp].set_visible(frame)
            a.spines["left"].set_color(lc)
            a.spines["right"].set_color(rc)
            b.spines["left"].set_color(lc)
            b.spines["right"].set_color(rc)
        # Side by side, each panel carries its own caption under its own x
        # label; stacked, the two panels share one axis and one caption would
        # have to serve both, so they keep the plain label.
        xlab = "Time (min.)" if side else "Time (minutes)"
        if side:
            a1.set_xlabel(f"{xlab}\n(a) Engine Signals", labelpad=1.5,
                          linespacing=1.5)
            a2.set_xlabel(f"{xlab}\n(b) Latency and SLO Attainment",
                          labelpad=1.5, linespacing=1.5)
        else:
            a2.set_xlabel(xlab, labelpad=1.5)

        keys = [(KV_C, "KV cache"), (BATCH_C, "Running batch")]
        if with_queue:
            keys.append((QUEUE_C, "Waiting queue"))
        n_a = len(keys)
        if itl2 is None:
            keys += [(ITL_C, "Time between Token (TBT)"),
                     (ATT_C, "SLO attainment (admitted)")]
        else:
            keys += [(ITL_C, f"TBT, {ARM}"), (ITL_C2, f"TBT, {arm2}"),
                     (ATT_C, f"SLO, {ARM}")]
            if att2 is not None:
                keys.append((ATT_C2, f"SLO, {arm2}"))
        handles = [Line2D([], [], color=c, lw=0.9, label=l) for c, l in keys]
        if side:
            # One key per panel, above that panel: with the panels side by side
            # a single legend at the top of the canvas leaves the reader
            # matching four names against two panels.
            for ax_, pair in ((a1, keys[:n_a]), (a2, keys[n_a:])):
                hs = [Line2D([], [], color=c, lw=0.9, label=l)
                      for c, l in pair]
                # Four keys in one column is four rows of type above a one-inch
                # panel; two columns keeps it to two rows.
                ncol_ = (2 if len(pair) > 2 else 1) if narrow else 2
                ax_.legend(hs, [l for _, l in pair], loc="lower center",
                           bbox_to_anchor=(0.5, 1.01), ncol=ncol_,
                           frameon=False, fontsize=5.4 if narrow else 6.4,
                           columnspacing=0.7, handlelength=1.1,
                           handletextpad=0.3, borderaxespad=0.0)
        else:
            fig.legend(handles, [l for _, l in keys], loc="lower center", ncol=2,
                       bbox_to_anchor=(0.5, 1 - 0.30 / height), frameon=False,
                       fontsize=5.6 if narrow else 6.4,
                       columnspacing=0.7 if narrow else 0.9, handlelength=1.1,
                       handletextpad=0.3, borderaxespad=0.0)
        # Side by side, the left panel's RIGHT label and the right panel's LEFT
        # label meet in the middle of the canvas, so the gutter has to hold two
        # rotated labels rather than one.
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.285 / height), h_pad=0.5,
                         w_pad=(1.2 if narrow else 2.6) if side else 1.0,
                         pad=0.3)
        # ⚠ THE RESERVED TOP BAND IS A GUESS AND THE LEGEND IS THE MEASUREMENT.
        # The band above the panels is set in inches before anything is drawn,
        # so whatever the per-panel legends do not use is left as white space at
        # the top of the canvas. The residual gap between the top of the legends
        # and the top of the canvas is measured and the layout redone with the
        # rect raised by it; two or three passes drive it to nothing. The canvas
        # keeps the height the page expects and the panels take the space.
        if trim_top and side:
            rect_top = 1 - 0.285 / height
            for _ in range(4):
                fig.canvas.draw()
                tops = [ax_.get_legend().get_window_extent().ymax / fig.dpi
                        for ax_ in (a1, a2) if ax_.get_legend() is not None]
                if not tops:
                    break
                gap = height - max(tops)
                if abs(gap) <= 0.01:
                    break
                rect_top = min(0.999, rect_top + gap / height)
                fig.tight_layout(rect=(0, 0, 1, rect_top), h_pad=0.5,
                                 w_pad=(1.2 if narrow else 2.6) if side else 1.0,
                                 pad=0.3)
            print(f"    top gap after fitting: {gap:.3f} in")
        if dense_ticks:
            # Five labels on every y axis too, and a minor tick between each
            # pair. `b1` is skipped where it carries no scale of its own.
            for ax_ in (a1, b1, a2, b2):
                lo, hi = ax_.get_ylim()
                if not len(ax_.get_yticks()):
                    continue
                ticks = np.linspace(lo, hi, 5)
                ax_.yaxis.set_major_locator(FixedLocator(ticks))
                ax_.set_yticklabels([f"{t:.0f}" for t in ticks])
                ax_.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax_.tick_params(axis="y", which="minor", length=1.2)
        # Measured after everything that can move the axes has run, and
        # returned so that `--square` can solve for the height.
        w, h = fig.get_size_inches()
        bb = a1.get_position()
        box = (bb.width * w, bb.height * h)
        print(f"    panel axes box {box[0]:.2f} x {box[1]:.2f} in")
        ps.save(fig, out)
        return box


def report(g, att):
    m, kv, bat, wait, itl = g
    ok = ~np.isnan(itl)
    print(f"engine {PORT}, minutes {T_LO:.0f}-{T_HI:.0f} of {RUN}")
    print(f"  KV occupancy   p50 {np.percentile(kv, 50):5.1f}%  "
          f"p90 {np.percentile(kv, 90):5.1f}%  max {kv.max():5.1f}%")
    print(f"  running batch  p50 {np.percentile(bat, 50):5.0f}   "
          f"p90 {np.percentile(bat, 90):5.0f}    max {bat.max():5.0f}")
    print(f"  waiting queue  p50 {np.percentile(wait, 50):5.0f}   "
          f"p90 {np.percentile(wait, 90):5.0f}    max {wait.max():5.0f}")
    print(f"  TBT            p50 {np.percentile(itl[ok], 50):5.1f} ms "
          f"p90 {np.percentile(itl[ok], 90):5.1f} ms")
    print(f"  SLO (admitted) min {att[1].min():5.1f}%  "
          f"mean {att[1].mean():5.1f}%  max {att[1].max():5.1f}%  "
          f"({att[2].sum():,} requests in {len(att[0])} windows)")


def main():
    global ARM_KEY, ARM, RUN, PORT, T_LO, T_HI, WIN, STEP
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(RUNS), default=ARM_KEY)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--t-lo", type=float, default=T_LO)
    ap.add_argument("--t-hi", type=float, default=T_HI)
    ap.add_argument("--kv-floor", type=float, default=0.0)
    ap.add_argument("--batch-floor", type=float, default=0.0)
    ap.add_argument("--normalize", nargs="?", const="peak",
                    choices=["peak", "minmax"], default=None,
                    help="panel (a) scaled inside the drawn window: by each "
                         "series' peak, or over its min..max range")
    ap.add_argument("--att-win", type=float, default=WIN,
                    help="attainment window in seconds (default 90)")
    ap.add_argument("--att-step", type=float, default=STEP)
    ap.add_argument("--compare-arm", choices=sorted(RUNS), default=None,
                    help="draw a second arm's per-token time in panel (b), "
                         "same engine and same minutes")
    ap.add_argument("--square", action="store_true",
                    help="solve for the canvas height that makes each panel's "
                         "axes box square")
    ap.add_argument("--with-queue", action="store_true",
                    help="draw the engine's waiting queue in panel (a)")
    ap.add_argument("--dense-ticks", action="store_true",
                    help="five labelled ticks on every axis, with minor ticks "
                         "between them")
    ap.add_argument("--height", type=float, default=None,
                    help="canvas height in inches for the column figure")
    ap.add_argument("--label-size", type=float, default=None,
                    help="font size for the tick and axis labels; smaller type "
                         "there enlarges the plotting area")
    ap.add_argument("--fit-ylim", action="store_true",
                    help="panel (a): each axis spans its own series' min..max "
                         "over the window, in units, instead of starting at 0")
    ap.add_argument("--frame", action="store_true",
                    help="box each panel (all four spines) instead of drawing "
                         "only the two that carry an axis")
    ap.add_argument("--tight-ylim", action="store_true",
                    help="the top of each y axis IS the top of the panel: no "
                         "headroom above the highest tick")
    ap.add_argument("--side", action="store_true",
                    help="the two panels in one row instead of stacked")
    ap.add_argument("--no-notes", action="store_true",
                    help="drop the in-panel scaling and cut-axis notes; the "
                         "script prints them for the caption")
    ap.add_argument("--no-over-markers", action="store_true",
                    help="drop the triangles on the cut latency axis; the count "
                         "and maximum stay in the panel either way")
    ap.add_argument("--suffix", default="",
                    help="appended to the file name, e.g. _zoom")
    a = ap.parse_args()
    ARM_KEY, PORT, T_LO, T_HI = a.arm, a.port, a.t_lo, a.t_hi
    WIN, STEP = a.att_win, a.att_step
    ARM, RUN = RUNS[ARM_KEY]

    run_dir = os.path.join(ROOT, "results", RUN)
    t0 = float(pd.read_csv(os.path.join(run_dir, "metrics.csv"),
                           usecols=["start_time"])["start_time"].min())
    g = gauges(run_dir, t0)
    att = attainment(run_dir, t0)

    # The comparison arm: the same engine index over the same minutes, measured
    # from ITS OWN first arrival, because the two runs did not start together.
    itl2, arm2, att2 = None, None, None
    if a.compare_arm:
        arm2, run2 = RUNS[a.compare_arm]
        d2 = os.path.join(ROOT, "results", run2)
        t02 = float(pd.read_csv(os.path.join(d2, "metrics.csv"),
                                usecols=["start_time"])["start_time"].min())
        g2 = gauges(d2, t02)
        itl2 = np.interp(g[0], g2[0], g2[4], left=np.nan, right=np.nan)
        ok2 = ~np.isnan(itl2)
        att2 = attainment(d2, t02)
        print(f"  comparison arm {arm2} ({run2}), engine {PORT}: TBT p50 "
              f"{np.percentile(itl2[ok2], 50):.1f} ms  p90 "
              f"{np.percentile(itl2[ok2], 90):.1f} ms, SLO(admitted) mean "
              f"{att2[1].mean():.1f}%")
    report(g, att)
    # The arm and the engine are in the file name: the same figure drawn for a
    # different arm or a different engine says something else entirely, and a
    # name that carries neither is a name that will be cited for the wrong one.
    stem = f"engine_window_{ARM_KEY}_{PORT}{a.suffix}"
    pdf = os.path.join(HERE, stem + ".pdf")
    common = dict(itl2=itl2, arm2=arm2, att2=att2, kv_floor=a.kv_floor, batch_floor=a.batch_floor,
                  normalize=a.normalize, over_markers=not a.no_over_markers,
                  side=a.side, notes=not a.no_notes, frame=a.frame,
                  tight_ylim=a.tight_ylim, fit_ylim=a.fit_ylim,
                  label_size=a.label_size, dense_ticks=a.dense_ticks,
                  with_queue=a.with_queue)
    if a.side:
        # Both canvases. At 3.335 in the two panels are about 1 in across each,
        # so the labelling is trimmed rather than the figure widened: see the
        # `side and width < 4` branches in `build`.
        # --square: the canvas height is solved for rather than guessed. The
        # panel's WIDTH is fixed by the column and by how much the labels take,
        # so the only free variable is the height, and each pass moves it by the
        # difference between the two sides of the axes box.
        h = a.height or 1.50
        for _ in range(5 if a.square else 1):
            box = build(g, att, os.path.join(HERE, stem + ".pdf"), ps.COL_W, h,
                        **common)
            if not a.square or box is None or abs(box[0] - box[1]) < 0.015:
                break
            h += box[0] - box[1]
        build(g, att, os.path.join(HERE, stem + "_wide.pdf"), ps.TEXT_W, 1.62,
              **common)
    else:
        build(g, att, pdf, ps.COL_W, FIG_H, **common)
        build(g, att, os.path.join(HERE, stem + "_wide.pdf"), ps.TEXT_W, 2.10,
              **common)
    write_csv(g, att, pdf[:-4] + ".csv", a.with_queue)
    return 0


if __name__ == "__main__":
    sys.exit(main())
