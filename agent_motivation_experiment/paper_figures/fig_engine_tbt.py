#!/usr/bin/env python3
"""Paper figure: what one instance held, and the pace it delivered, over the
same twelve minutes.

  engine_tbt.pdf   3.335 x 2.30 in, one column, width=\\columnwidth
  engine_tbt.csv   exactly the values drawn

  (a) one instance over twelve minutes: how full its KV pool was and how many
      requests were in its running batch, each scaled over the window shown
  (b) the SAME instance over the SAME minutes: the per-token time it actually
      delivered, and the share of the requests sent to it that met their rule

This is `load_engine_tbt.pdf` WITHOUT its arrival panel, with the instance-state
panel drawn the full width instead of half. ⚠ THE PANELS ARE THEREFORE
RE-LETTERED: what was (b) and (c) there is (a) and (b) here. A caption written
for the three-panel figure names the wrong panels on this one.

The arrival panel left because it is a different trace on a different clock --
four days against twelve minutes -- so nothing in these two panels is a
consequence of the interval it drew. It now opens `arrival_and_mix.pdf`
together with the mixture panel, where both halves share one clock.

⚠ BOTH PANELS ARE ONE RUN, ONE ENGINE AND THE SAME MINUTES -- EXP-109 repeat 1
of the Llumnix SLO arm, engine 8002, minutes 35-47 of the hour -- so a feature
in (a) and a feature in (b) at the same x are the same moment. That is what the
full width buys: the eye can drop a line from one panel to the other.

⚠ (a) IS SCALED OVER ITS OWN WINDOW AND (b) IS NOT. Each series in (a) runs
between its own minimum and maximum there, so that panel shows the SHAPE of the
two and not how large either is; (b) is in milliseconds on an axis that starts
at zero, because the class budgets are stated in that unit and a normalised pace
could not be read against them.

⚠ THE PER-TOKEN AXIS IS CUT AT 150 ms AND THE CUT IS COUNTED. The seconds above
it are printed and belong in the caption.

⚠ THE ATTAINMENT CURVE HAS AN ADMITTED DENOMINATOR. It says nothing about what
the arm refused, and over these twelve minutes Llumnix SLO refuses 28.3% of
arrivals, so the caption has to carry that number or a high curve reads as
"nearly everything was served".

DATA. `fig_load_engine_tbt.py`'s readers, colours and constants, imported rather
than restated, so the two figures cannot drift apart.

    python3 paper_figures/fig_engine_tbt.py
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


LET = _load("loadengtbt", os.path.join(HERE, "fig_load_engine_tbt.py"))
EW = LET.EW

FIG_W, FIG_H = ps.COL_W, 2.30
# Vertical air between the two rows, in font-size units, as `tight_layout` takes
# it. 1.6 was inherited from the three-panel figure, where the top row held two
# panels whose keys were anchored to the FIGURE and needed the room; here (b)'s
# key belongs to its own axes and travels with it, so the rows can sit as close
# as the ink between them allows. The clearance that results is measured and
# printed rather than assumed.
H_PAD = 0.4


def write_csv(m, kv, bat, itl, att, path):
    rows = [{"panel": "a", "series": "kv_cache_usage_pct", "x_minutes": float(a),
             "value": float(b)} for a, b in zip(m, kv)]
    rows += [{"panel": "a", "series": "running_batch_requests",
              "x_minutes": float(a), "value": float(b)} for a, b in zip(m, bat)]
    rows += [{"panel": "b", "series": "inter_token_latency_ms",
              "x_minutes": float(a), "value": float(b)}
             for a, b in zip(m, itl) if b == b]
    rows += [{"panel": "b", "series": "slo_attainment_admitted_pct",
              "x_minutes": float(a), "value": float(b), "n_requests": int(c),
              "attainment_window_s": LET.ATT_WIN}
             for a, b, c in zip(*att)]
    df = pd.DataFrame(rows)
    df["run"] = EW.RUNS[LET.ARM_KEY][1]
    df["engine_port"] = LET.PORT
    df.to_csv(path, index=False, float_format="%.5f")
    print(f"wrote {path}  ({len(df)} rows)")


def build(m, kv, bat, itl, att, out, width=FIG_W, height=FIG_H):
    def mm(v):
        return 100.0 * (v - v.min()) / max(v.max() - v.min(), 1e-9)

    with plt.rc_context({**ps.STYLE, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "axes.labelsize": 6}):
        fig = plt.figure(figsize=(width, height))
        # Equal rows. In the three-panel figure the bottom row was shorter
        # because the top row held two panels side by side and needed the
        # height for two y labels; here both rows are one panel of the same
        # width, and the two quantities are read against each other, so neither
        # has a claim on more of the canvas than the other.
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0])
        axa = fig.add_subplot(gs[0, 0])
        axb = fig.add_subplot(gs[1, 0])

        axa.plot(m, mm(kv), color=LET.KV_C, lw=0.9)
        axa.plot(m, mm(bat), color=LET.BATCH_C, lw=0.9)
        axa.set_ylabel("Norm. (%)", labelpad=1.5)
        axa.set_ylim(0, 100)
        axa.set_yticks([0, 25, 50, 75, 100])
        axa.set_xlabel("Time (min.)\n(a) Observed Engine State of an Instance",
                       labelpad=1.5, linespacing=1.5)

        ok = ~np.isnan(itl)
        axb.plot(m, itl, color=LET.TBT_C, lw=0.9, zorder=3)
        axb.set_ylim(0, LET.TBT_TOP)
        axb.set_yticks([0, 50, 100, 150])
        axb.set_ylabel("TBT (ms)", labelpad=1.5)
        axb.set_xlabel("Time (min.)\n(b) Measured TBT and Request SLO "
                       "Attainment of an Instance",
                       labelpad=1.5, linespacing=1.5)
        axd = axb.twinx()
        axd.plot(att[0], att[1], color=LET.ATT_C, lw=1.0, zorder=2,
                 marker=LET.ATT_MARK, ms=LET.ATT_MS, mew=0.0)
        axd.set_ylim(0, 100)
        axd.set_yticks([0, 50, 100])
        axd.set_ylabel("Request SLO (%)", labelpad=2.0)
        axd.tick_params(axis="y", length=2.0)
        axd.spines["top"].set_visible(False)

        for a in (axa, axb):
            a.set_xlim(LET.T_LO, LET.T_HI)
            # ONE TICK PER MINUTE ON BOTH, which the state panel could not have
            # at half the width: it is what lets a reader carry a moment from
            # one panel to the other without measuring.
            a.set_xticks(np.arange(LET.T_LO, LET.T_HI + 1e-9, 1))
            a.grid(axis="both", **ps.GRID)
            a.set_axisbelow(True)
            a.xaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(3))
            a.yaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
            a.tick_params(which="minor", length=1.2)

        # One key over each panel. Both panels are the full width, so the two
        # keys are on different rows and cannot meet: the size is set rather
        # than fitted against each other, and it is the size the three-panel
        # figure settles on so the two figures' keys match.
        size = 6.4
        ka = [(LET.KV_C, "KV-cache Occupancy"), (LET.BATCH_C, "Batch Size")]
        hb = [Line2D([], [], color=LET.TBT_C, lw=0.9),
              Line2D([], [], color=LET.ATT_C, lw=1.0, marker=LET.ATT_MARK,
                     ms=LET.ATT_MS, mew=0.0)]
        rect_top = 1 - 0.20 / height
        gap = float("nan")
        for _ in range(6):
            for l in list(fig.legends):
                l.remove()
            # ⚠ THE ROWS ARE SET AS CLOSE AS THE INK BETWEEN THEM ALLOWS.
            # What sits in that gap is (a)'s two-line x label and (b)'s own
            # key, so `h_pad` cannot simply be taken to zero: the measured
            # clearance between those two is printed below, and a negative one
            # is the caption and the key overlapping.
            fig.tight_layout(rect=(0, 0, 1, rect_top), pad=0.3, h_pad=H_PAD)
            bba = axa.get_position()
            fig.legend([Line2D([], [], color=c, lw=0.9) for c, _ in ka],
                       [l for _, l in ka], loc="lower center",
                       bbox_to_anchor=(0.5 * (bba.x0 + bba.x1), bba.y1 + 0.008),
                       ncol=len(ka), frameon=False, fontsize=size,
                       columnspacing=0.7, handlelength=1.1,
                       handletextpad=0.3, borderaxespad=0.0)
            axb.legend(hb, ["TBT", "Request SLO Attainment"],
                       loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
                       frameon=False, fontsize=size, columnspacing=0.7,
                       handlelength=1.1, handletextpad=0.3, borderaxespad=0.0)
            fig.canvas.draw()
            # The band reserved above the panels is a guess and the key is the
            # measurement, so the residual is driven to nothing IN BOTH
            # DIRECTIONS: a positive gap is a white strip at the top of the
            # canvas, a negative one is the key's ink hanging off the page.
            gap = height - max(l.get_window_extent().ymax
                               for l in fig.legends) / fig.dpi
            if abs(gap) <= 0.005:
                break
            rect_top = min(0.999, rect_top + gap / height)
        w, h = fig.get_size_inches()
        rend = fig.canvas.get_renderer()
        cap_bottom = axa.xaxis.get_tightbbox(rend).y0 / fig.dpi
        key_b = axb.get_legend()
        key_top = key_b.get_window_extent(rend).y1 / fig.dpi
        clear = cap_bottom - key_top
        print(f"    clearance between (a)'s caption and (b)'s key: "
              f"{clear:+.3f} in" + ("  ⚠ THEY OVERLAP" if clear < 0 else ""))
        for nm, a in (("a", axa), ("b", axb)):
            bb = a.get_position()
            print(f"    ({nm}) axes box {bb.width * w:.2f} x "
                  f"{bb.height * h:.2f} in")
        print(f"    legend type {size:.1f} pt, top gap {gap:+.3f} in")
        print(f"    (b) TBT p50 {np.percentile(itl[ok], 50):.1f} ms, "
              f"mean {itl[ok].mean():.1f}, min {itl[ok].min():.1f}, "
              f"max {itl[ok].max():.0f}; "
              f"{int((itl[ok] > LET.TBT_TOP).sum())} of {int(ok.sum())} s above "
              f"the {LET.TBT_TOP:.0f} ms axis")
        ps.save(fig, out)


def main():
    m, kv, bat, itl, att = LET.engine_series()
    print(f"engine {LET.PORT}, minutes {LET.T_LO:.0f}-{LET.T_HI:.0f} of "
          f"{EW.RUNS[LET.ARM_KEY][1]}")
    print(f"panel (a): KV {kv.min():.0f}-{kv.max():.0f}%, batch "
          f"{bat.min():.0f}-{bat.max():.0f} requests")
    build(m, kv, bat, itl, att, os.path.join(HERE, "engine_tbt.pdf"))
    write_csv(m, kv, bat, itl, att, os.path.join(HERE, "engine_tbt.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
