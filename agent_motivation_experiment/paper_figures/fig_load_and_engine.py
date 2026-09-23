#!/usr/bin/env python3
"""Paper figure: the load a cluster is given moves, and so does one instance.

  load_and_engine.pdf   3.335 x 1.75 in, one column, width=\\columnwidth
  load_and_engine.csv   exactly the values drawn

  (a) the arrival rate of a production trace over four days, normalised by its
      own peak: the request rate a cluster is asked to serve varies 5.8x
      between its trough and its peak.
  (b) one instance of our fleet over twelve minutes of one hour: how full its
      KV pool was and how many requests were in its running batch, each scaled
      over the window shown.

THE TWO PANELS ARE AT DIFFERENT SCALES ON PURPOSE, four days beside twelve
minutes, because the claim is that the same kind of movement exists at both:
the cluster's input is not stationary and neither is the state of a single
instance inside it. A placement decision made from an instance's state is made
from a quantity that looks like panel (b), and a capacity plan made from the
offered rate is made from one that looks like panel (a).

⚠ THEY ARE NOT THE SAME RUN AND NOT THE SAME CLOCK. Panel (a) is the Azure
production trace this project's arrivals are derived from, in hours; panel (b)
is EXP-109 repeat 1 of the Llumnix SLO arm, engine 8002, minutes 35-47 of the
hour, in minutes. Nothing in (b) is a consequence of the interval drawn in (a).

⚠ PANEL (b) IS SCALED OVER ITS OWN WINDOW, each series between its own minimum
and maximum there (KV 23-100% of the pool, running batch 59-392 requests), so
the panel shows the SHAPE of the two and not how large either is. Vertical
distance is not proportional to either quantity and the two curves crossing
means nothing.

DATA. Panel (a): `fig_azure_rate_and_mix.py`'s source window, imported rather
than recomputed. Panel (b): `fig_engine_window.py`'s gauge reader, also
imported; the gauges are the engine's own, as scraped each second and not
smoothed.

    python3 paper_figures/fig_load_and_engine.py
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import paper_style as ps  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AZ = _load("azfig", os.path.join(HERE, "fig_azure_rate_and_mix.py"))
EW = _load("engwin", os.path.join(HERE, "fig_engine_window.py"))

ARM_KEY, PORT = "slo", 8002
T_LO, T_HI = 35.0, 47.0
FIG_H = 1.75


def engine_series():
    EW.PORT, EW.T_LO, EW.T_HI = PORT, T_LO, T_HI
    EW.SMOOTH_S = 1
    run = os.path.join(ROOT, "results", EW.RUNS[ARM_KEY][1])
    t0 = float(pd.read_csv(os.path.join(run, "metrics.csv"),
                           usecols=["start_time"])["start_time"].min())
    m, kv, bat, _wait, _itl = EW.gauges(run, t0)
    return m, kv, bat


def write_csv(hours, rate, m, kv, bat, path):
    rows = [{"panel": "a", "series": "arrival_rate_norm", "x_hours": float(a),
             "value": float(b)} for a, b in zip(hours, rate)]
    rows += [{"panel": "b", "series": "kv_cache_usage_pct", "x_minutes": float(a),
              "value": float(b)} for a, b in zip(m, kv)]
    rows += [{"panel": "b", "series": "running_batch_requests",
              "x_minutes": float(a), "value": float(b)} for a, b in zip(m, bat)]
    df = pd.DataFrame(rows)
    df["panel_b_run"] = EW.RUNS[ARM_KEY][1]
    df["panel_b_engine_port"] = PORT
    df.to_csv(path, index=False, float_format="%.5f")
    print(f"wrote {path}  ({len(df)} rows)")


def legend_row(fig, ax, keys, sizes=(6.4, 6.0, 5.6, 5.2, 4.8, 4.4)):
    """One row of keys above `ax`, at the largest size that fits its width."""
    handles = [Line2D([], [], color=c, lw=0.9) for c, _ in keys]
    labels = [l for _, l in keys]
    for size in sizes:
        lg = ax.legend(handles, labels, loc="lower center",
                       bbox_to_anchor=(0.5, 1.01), ncol=len(keys),
                       frameon=False, fontsize=size, columnspacing=0.7,
                       handlelength=1.1, handletextpad=0.3, borderaxespad=0.0)
        fig.canvas.draw()
        if lg.get_window_extent().width <= ax.get_window_extent().width * 1.02:
            return size
    return sizes[-1]


def build(hours, total, peak, m, kv, bat, out, width=ps.COL_W, height=FIG_H):
    def mm(v):
        return 100.0 * (v - v.min()) / max(v.max() - v.min(), 1e-9)

    with plt.rc_context({**ps.STYLE, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "axes.labelsize": 6}):
        fig, ax = plt.subplots(1, 2, figsize=(width, height))
        rate = total / peak

        ax[0].plot(hours, rate, color=AZ.C_RATE, lw=0.5)
        ax[0].fill_between(hours, rate, color=AZ.C_RATE, alpha=0.15, lw=0)
        ax[0].axhline(1.0, color="#555555", lw=0.6, ls="--")
        ax[0].axhline(rate.min(), color="#555555", lw=0.6, ls="--")
        t0 = AZ.span(ax[0], hours, rate, rate.min(), 1.0,
                     f"{1 / rate.min():.1f}$\\times$")
        ax[0].set_ylabel("Arrival rate (norm.)", labelpad=1.5)
        ax[0].set_ylim(0, 1.0)
        ax[0].set_yticks([0, 0.5, 1.0])
        ax[0].set_xlim(0, hours[-1])
        ax[0].set_xticks(np.arange(0, hours[-1] + 1, 48))
        ax[0].set_xlabel("Time (hours)\n(a) Cluster Load", labelpad=1.5,
                         linespacing=1.5)

        ax[1].plot(m, mm(kv), color=EW.KV_C, lw=0.9)
        ax[1].plot(m, mm(bat), color=EW.BATCH_C, lw=0.9)
        ax[1].set_ylabel("Norm. (%)", labelpad=1.5)
        ax[1].set_ylim(0, 100)
        ax[1].set_yticks([0, 25, 50, 75, 100])
        ax[1].set_xlim(T_LO, T_HI)
        ax[1].set_xticks(np.arange(T_LO, T_HI + 1e-9, 3))
        ax[1].set_xlabel("Time (min.)\n(b) Engine Signals", labelpad=1.5,
                         linespacing=1.5)
        # A key on BOTH panels, each in ONE ROW above its own panel. The size
        # is the largest that fits the panel's width, found by drawing the
        # legend and measuring it: matplotlib will let a one-row legend run past
        # the panel without saying so.
        # ⚠ ONE SIZE FOR BOTH KEYS. Fitting each panel separately gives the
        # one-entry key a larger type than the two-entry one, and two keys set
        # differently on one figure read as two kinds of key. The size is the
        # smaller of the two fits, applied to both.
        keys_a = [(AZ.C_RATE, "Request arrival")]
        keys_b = [(EW.KV_C, "KV cache"), (EW.BATCH_C, "Running batch")]
        size = min(legend_row(fig, ax[0], keys_a),
                   legend_row(fig, ax[1], keys_b))
        legend_row(fig, ax[0], keys_a, sizes=(size,))
        legend_row(fig, ax[1], keys_b, sizes=(size,))
        print(f"    legend type {size:.1f} pt on both panels")

        for a in ax:
            a.grid(axis="both", **ps.GRID)
            a.set_axisbelow(True)
            a.xaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(3))
            a.yaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
            a.tick_params(which="minor", length=1.2)

        # The reserved band above the panels is a guess and the legend is the
        # measurement; the residual gap is driven to nothing so the canvas
        # carries no white strip at the top.
        rect_top = 1 - 0.24 / height
        fig.tight_layout(rect=(0, 0, 1, rect_top), pad=0.3, w_pad=1.4)
        for _ in range(4):
            fig.canvas.draw()
            gap = height - max(a.get_legend().get_window_extent().ymax
                               for a in ax if a.get_legend()) / fig.dpi
            if abs(gap) <= 0.01:
                break
            rect_top = min(0.999, rect_top + gap / height)
            fig.tight_layout(rect=(0, 0, 1, rect_top), pad=0.3, w_pad=1.4)
        AZ.check_overlap(fig, ax[0], t0, hours, rate, "arrival rate")
        w, h = fig.get_size_inches()
        bb = ax[1].get_position()
        print(f"    panel axes box {bb.width * w:.2f} x {bb.height * h:.2f} in, "
              f"top gap {gap:.3f} in")
        ps.save(fig, out)


def main():
    conv, code, _plan = AZ.source_window()
    total = conv + code
    hours = np.arange(len(total)) / 60.0
    peak = total.max()
    m, kv, bat = engine_series()
    print(f"panel (a): {len(total)/1440:.1f} days, peak/trough "
          f"{peak/total.min():.1f}x")
    print(f"panel (b): KV {kv.min():.0f}-{kv.max():.0f}%, batch "
          f"{bat.min():.0f}-{bat.max():.0f} over {T_LO:.0f}-{T_HI:.0f} min")
    pdf = os.path.join(HERE, "load_and_engine.pdf")
    build(hours, total, peak, m, kv, bat, pdf)
    write_csv(hours, total / peak, m, kv, bat, pdf[:-4] + ".csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
