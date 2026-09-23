#!/usr/bin/env python3
"""Paper figure: what the engines held, and what the requests got, per arm.

  kv_batch_attain_4arms.pdf   3.335 in wide (one column), no scaling
  kv_batch_attain_4arms.csv   one row per run, plus per-engine medians

  (a) median KV-cache occupancy (%)
  (b) median batch size (running requests)
  (c) Request SLO attainment over admitted requests (%)

Four arms on the one-hour mixture-shift trace, Llama-3.1-70B on 4 instances at
TP=2, standard budgets (agent class TTFT 7 s + 75 ms/token):

  FluidServe                EXP-109, repeats 1 and 2   (deployed configuration)
  FluidServe w/o Affinity   EXP-138, repeat 1 only     (class affinity AND instance cap off)
  llm-d                     EXP-109, repeats 1 and 2
  PolyServe                 EXP-109, repeats 1 and 2

MEASUREMENT.
  KV and batch   each engine's own Prometheus gauges, `vllm:kv_cache_usage_perc`
                 and `vllm:num_requests_running`, put on whole seconds. At each
                 second the FLEET value is formed -- batch = the four engines'
                 running requests added; KV = used blocks over the four pools'
                 capacity, i.e. the mean of the four percentages -- and the
                 median is taken over the seconds of the scorer's analysis
                 window (60 s after the first arrival to 20 s before the last).
  batch          `num_requests_running` counts every request the engine has
                 scheduled, including one still in its chunked prefill.
  attainment     ladder95 verdicts: token i is on time if it arrives within
                 TTFT budget + i x TBT budget, a request meets its SLO if 95% of
                 its tokens are; denominator = admitted requests that finished
                 (rejected and run-boundary cutoffs excluded).

The bar is the mean of the repeats' values and the thin mark spans the
repeats. ⚠ FluidServe w/o Affinity has ONE repeat and therefore no mark; the
caption has to say so.

⚠ ADMITTED ATTAINMENT DOES NOT CHARGE REJECTION. The rejection rate per arm is
printed and written to the CSV and belongs in the caption.

    python3 paper_figures/fig_kv_batch_attain.py
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import paper_style as ps  # noqa: E402

VERDICTS = os.path.join(ROOT, "results", "aggregate_analysis", "ladder95",
                        "verdicts")
PORTS = [8000, 8001, 8002, 8003]
WARMUP_S, DRAIN_S = 60.0, 20.0
# (label, colour, hatch, runs). Colours are the YlGnBu steps the other hour
# figures give these arms; the variant keeps FluidServe's colour and adds a
# hatch, so it reads as FluidServe changed rather than as a fifth system.
ARMS = [
    ("FluidServe", "#253494", None,
     ["260831_2015_exp109r1_fsv3capgnofrct75_shift",
      "260901_1514_exp109r2_fsv3capgnofrct75_shift"]),
    # ⚠ BOTH class affinity AND the instance cap off (EXP-138), at the author's
    # request (2026-09-14). The first draw used EXP-132, which turns off
    # affinity alone; the label is kept as the author named it, and the caption
    # has to say that the instance cap is off too.
    ("FluidServe w/o Affinity", "#253494", "////",
     ["260914_0539_exp138r1_noaffnocaphour_shift"]),
    ("llm-d", "#c2a5cf", None,
     ["260831_2128_exp109r1_llmdslot75_shift",
      "260901_1637_exp109r2_llmdslot75_shift"]),
    ("PolyServe", "#41b6c4", None,
     ["260831_2232_exp109r1_polyservept75_shift",
      "260901_1742_exp109r2_polyservept75_shift"]),
]
FIG_W, FIG_H = ps.COL_W, 1.55
SHRINK = 2.0


def engine_samples(rd, port, lo, hi):
    """Per-scrape (kv %, running, t) of one engine with t in [lo, hi) epoch s."""
    kv, run, ts = [], [], []
    with open(os.path.join(rd, "server_metrics", f"engine_{port}.jsonl")) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            t = r.get("t")
            if t is None or not (lo <= t < hi) or not r.get("ok", True):
                continue
            a = next((v for k, v in r.items()
                      if k.startswith("vllm:kv_cache_usage_perc") and v is not None), None)
            b = next((v for k, v in r.items()
                      if k.startswith("vllm:num_requests_running") and v is not None), None)
            if a is None or b is None:
                continue
            kv.append(100.0 * float(a))
            run.append(float(b))
            ts.append(float(t))
    return np.array(kv), np.array(run), np.array(ts)


def one_run(run):
    rd = os.path.join(ROOT, "results", run)
    m = pd.read_csv(os.path.join(rd, "metrics.csv"),
                    usecols=["agent", "start_time"], low_memory=False)
    m = m[m["agent"] != "job_summary"]
    st = pd.to_numeric(m["start_time"], errors="coerce").dropna()
    lo, hi = st.min() + WARMUP_S, st.max() - DRAIN_S
    # FLEET TOTALS PER SECOND, THEN THE MEDIAN OVER THE HOUR (2026-09-14, at the
    # author's request; the first version pooled every engine's samples and took
    # one median, which is the value of a TYPICAL ENGINE and not of the fleet).
    # Each engine's scrapes are put on whole seconds; a second enters only when
    # all four engines reported in it, so a missing scrape cannot read as an
    # empty instance.
    #   batch  the four engines' running requests ADDED
    #   KV     the four engines' used KV blocks over the four pools' capacity.
    #          The pools are the same size, so this is the mean of the four
    #          percentages at that second (the sum of the four, divided by 4).
    kv_s, rq_s, per = [], [], {}
    for p in PORTS:
        kv, rq, sec = engine_samples(rd, p, lo, hi)
        d = pd.DataFrame({"s": sec.astype(int), "kv": kv, "rq": rq}).groupby("s").last()
        kv_s.append(d["kv"].rename(p))
        rq_s.append(d["rq"].rename(p))
        per[p] = (float(np.median(kv)) if len(kv) else np.nan,
                  float(np.median(rq)) if len(rq) else np.nan, len(kv))
    kvm = pd.concat(kv_s, axis=1, join="inner")
    rqm = pd.concat(rq_s, axis=1, join="inner")
    kv_all = kvm.sum(axis=1).to_numpy() / len(PORTS)
    run_all = rqm.sum(axis=1).to_numpy()
    vp = os.path.join(VERDICTS, run + ".csv")
    if not os.path.exists(vp):
        sys.exit(f"no ladder95 verdicts for {run}; score it first")
    v = pd.read_csv(vp)
    rej = v["rejected"].astype(bool)
    cut = v["cutoff"].astype(bool)
    served = v[~rej & ~cut]
    out = dict(run=run, kv_p50=float(np.median(kv_all)),
               batch_p50=float(np.median(run_all)),
               kv_p90=float(np.percentile(kv_all, 90)),
               batch_p90=float(np.percentile(run_all, 90)),
               seconds=int(len(kv_all)), window_s=float(hi - lo),
               attain_admitted=100.0 * float(served["ladder_ok"].astype(bool).mean()),
               # Goodput over the same window the gauges are read in. Request
               # goodput counts requests that met their SLO; token goodput counts
               # the output tokens that arrived by their own deadline (ladder95),
               # so a request that missed the 95% bar still contributes its
               # on-time tokens -- the definition the hour goodput figures use.
               req_goodput=float((v["ladder_ok"].astype(bool) & ~rej & ~cut).sum())
                           / float(hi - lo),
               tok_goodput=float((v["n_tokens"] - v["n_late"]).sum()) / float(hi - lo),
               rejected_pct=100.0 * float(rej.mean()), arrivals=int(len(v)))
    for p, (k, b, n) in per.items():
        out[f"kv_p50_{p}"], out[f"batch_p50_{p}"], out[f"samples_{p}"] = k, b, n
    return out


def bars(a, df, col):
    """Four bars on one axes: mean of the repeats, a mark spanning them."""
    top = 0.0
    for j, (label, colour, hatch, _runs) in enumerate(ARMS):
        vals = df.loc[df["arm"] == label, col].to_numpy(float)
        a.bar([j], [vals.mean()], width=0.72, color=colour,
              edgecolor="#000000" if hatch is None else "white",
              linewidth=0.5 if hatch is None else 0.0, hatch=hatch, zorder=2)
        if hatch is not None:
            a.bar([j], [vals.mean()], width=0.72, fill=False,
                  edgecolor="#000000", linewidth=0.5, zorder=3)
        # no repeat range mark (2026-09-18, at the author's request); the
        # per-run values stay in the CSV
        top = max(top, vals.max())
    a.set_xlim(-0.7, len(ARMS) - 0.3)
    a.set_xticks([])
    a.grid(axis="y", **ps.GRID)
    a.set_axisbelow(True)
    return top


def build_goodput(df):
    """The same figure with attainment replaced by the two goodputs, 2 x 2.

      kv_batch_goodput_4arms.pdf   3.335 in wide (one column)

    (a) and (b) as in kv_batch_attain_4arms.pdf; (c) request goodput, requests
    that met their SLO per second; (d) token goodput, on-time output tokens per
    second. Both over the same analysis window as the gauges.
    """
    fig_h = 2.55
    style = dict(ps.STYLE)
    for k in ("axes.labelsize", "xtick.labelsize", "ytick.labelsize"):
        style[k] = max(4.0, style[k] - SHRINK)
    panels = [("kv_p50", "Cluster KV-cache\nOccupancy P50 (%)", "(a) KV-cache",
               (0, 100), [0, 25, 50, 75, 100], None),
              ("batch_p50", "Cluster Batch Size P50", "(b) Batch Size",
               None, None, None),
              ("req_goodput", "Request Goodput (r/s)", "(c) Request Goodput",
               None, None, None),
              ("tok_goodput", "Token Goodput (t/s)", "(d) Token Goodput",
               None, None, ps.kfmt())]
    with plt.rc_context(style):
        fig, axs = plt.subplots(2, 2, figsize=(FIG_W, fig_h))
        axs = axs.ravel()
        for a, (col, ylab, cap, ylim, yt, fmt) in zip(axs, panels):
            top = bars(a, df, col)
            if ylim:
                a.set_ylim(*ylim)
                a.set_yticks(yt)
            else:
                a.set_ylim(0, top * 1.15)
            if fmt is not None:
                a.yaxis.set_major_formatter(fmt)
            a.set_ylabel(ylab, labelpad=1.5)
            # the caption is the x label, so the layout reserves its room under
            # both rows
            a.set_xlabel(cap, fontsize=6.5, labelpad=3)
        handles = [Patch(facecolor=c, hatch=h, edgecolor="white" if h else "#000000",
                         linewidth=0.5) for _l, c, h, _r in ARMS]
        labels = [l for l, _c, _h, _r in ARMS]
        handles, labels = ps.legend_items(handles, labels)
        top_band = 0.18
        key = None
        for _ in range(6):
            if key is not None:
                key.remove()
            fig.tight_layout(rect=(0, 0, 1, 1 - top_band / fig_h), pad=0.3,
                             w_pad=0.9, h_pad=0.6)
            for fs in (7.0, ps.KEY_FS):
                key = fig.legend(handles, labels, loc="lower center",
                                 ncol=len(labels),
                                 bbox_to_anchor=(0.5, 1 - top_band / fig_h),
                                 frameon=False, fontsize=fs, columnspacing=0.8,
                                 borderaxespad=0.0,
                                 handler_map=ps.square_handler(handles),
                                 **ps.KEY_SQUARE)
                fig.canvas.draw()
                e = key.get_window_extent()
                if e.x0 >= 1.0 and e.x1 <= FIG_W * fig.dpi - 1.0:
                    break
                key.remove()
            fig.canvas.draw()
            gap = fig_h - key.get_window_extent().y1 / fig.dpi
            if abs(gap) <= 0.005:
                break
            top_band = max(0.05, top_band - gap)
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        tb = [a.get_tightbbox(rend) for a in axs]
        print(f"  goodput figure: key {fs} pt; gaps across "
              f"{(tb[1].x0 - tb[0].x1) / fig.dpi:+.3f} / {(tb[3].x0 - tb[2].x1) / fig.dpi:+.3f} in, "
              f"between rows {(tb[0].y0 - tb[2].y1) / fig.dpi:+.3f} in")
        W, H = fig.get_size_inches() * fig.dpi
        for b in tb + [key.get_window_extent(rend)]:
            if b.x0 < -0.5 or b.x1 > W + 0.5 or b.y0 < -0.5 or b.y1 > H + 0.5:
                print("  ⚠ goodput figure: ink off the canvas")
        ps.save(fig, ps.final("kv_batch_goodput_4arms.pdf"))


def main():
    rows = []
    for label, _c, _h, runs in ARMS:
        for run in runs:
            r = one_run(run)
            r["arm"] = label
            rows.append(r)
            print(f"  {label:24s} {run}: KV p50 {r['kv_p50']:5.1f}%  batch p50 "
                  f"{r['batch_p50']:5.1f}  attainment {r['attain_admitted']:5.1f}%  "
                  f"rejected {r['rejected_pct']:4.1f}%  ({r['seconds']} seconds, "
                  f"{r['window_s'] / 60:.1f} min)")
    df = pd.DataFrame(rows)

    style = dict(ps.STYLE)
    for k in ("axes.labelsize", "xtick.labelsize", "ytick.labelsize"):
        style[k] = max(4.0, style[k] - SHRINK)
    panels = [("kv_p50", "Cluster KV-cache\nOccupancy P50 (%)", (0, 100), [0, 25, 50, 75, 100]),
              ("batch_p50", "Cluster Batch Size P50", None, None),
              ("attain_admitted", "Attainment (%)", (0, 100), [0, 25, 50, 75, 100])]
    # "(c) Request SLO Attainment" is 1.6 in of 6.5 pt type under a 1.1 in panel:
    # on one line it either leaves the canvas or, pulled back in, runs over
    # "(b)". It breaks after "SLO", and every caption hangs from the same top
    # line so the first lines stay level.
    caps = ["(a) KV-cache", "(b) Batch Size", "(c) Request SLO\nAttainment"]
    with plt.rc_context(style):
        fig, ax = plt.subplots(1, 3, figsize=(FIG_W, FIG_H))
        for i, (col, ylab, ylim, yt) in enumerate(panels):
            a = ax[i]
            top = 0.0
            for j, (label, colour, hatch, _runs) in enumerate(ARMS):
                vals = df.loc[df["arm"] == label, col].to_numpy(float)
                a.bar([j], [vals.mean()], width=0.72, color=colour,
                      edgecolor="#000000" if hatch is None else "white",
                      linewidth=0.5 if hatch is None else 0.0, hatch=hatch,
                      zorder=2)
                if hatch is not None:
                    # the outline drawn separately: a hatched bar needs a white
                    # hatch colour, and matplotlib takes it from the edge colour
                    a.bar([j], [vals.mean()], width=0.72, fill=False,
                          edgecolor="#000000", linewidth=0.5, zorder=3)
                # no repeat range mark (2026-09-18, at the author's request)
                top = max(top, vals.max())
            a.set_xlim(-0.7, len(ARMS) - 0.3)
            a.set_xticks([])
            if ylim:
                a.set_ylim(*ylim)
                a.set_yticks(yt)
            else:
                a.set_ylim(0, top * 1.15)
            a.set_ylabel(ylab, labelpad=1.5)
            a.grid(axis="y", **ps.GRID)
            a.set_axisbelow(True)

        handles = []
        for label, colour, hatch, _r in ARMS:
            handles.append(Patch(facecolor=colour, hatch=hatch,
                                 edgecolor="white" if hatch else "#000000",
                                 linewidth=0.5))
        labels = [l for l, _c, _h, _r in ARMS]
        handles, labels = ps.legend_items(handles, labels)
        top_band, bot_band = 0.18, 0.25
        key = None
        for _ in range(6):
            if key is not None:
                key.remove()
            fig.tight_layout(rect=(0, bot_band / FIG_H, 1, 1 - top_band / FIG_H),
                             pad=0.3, w_pad=0.9)
            # ONE ROW (2026-09-14, at the author's request). Four entries, one
            # of them long, on a 3.335 in canvas: the type is fitted -- the
            # largest size at which the row fits inside the canvas.
            for fs in (7.0, ps.KEY_FS):
                key = fig.legend(handles, labels, loc="lower center",
                                 ncol=len(labels),
                                 bbox_to_anchor=(0.5, 1 - top_band / FIG_H),
                                 frameon=False, fontsize=fs, columnspacing=0.8,
                                 borderaxespad=0.0,
                                 handler_map=ps.square_handler(handles),
                                 **ps.KEY_SQUARE)
                fig.canvas.draw()
                e = key.get_window_extent()
                if e.x0 >= 1.0 and e.x1 <= FIG_W * fig.dpi - 1.0:
                    break
                key.remove()
            key_fs = fs
            fig.canvas.draw()
            gap = FIG_H - key.get_window_extent().y1 / fig.dpi
            if abs(gap) <= 0.005:
                break
            top_band = max(0.05, top_band - gap)
        texts = []
        for i, c in enumerate(caps):
            p = ax[i].get_position()
            texts.append(fig.text(0.5 * (p.x0 + p.x1), (bot_band - 0.03) / FIG_H,
                                  c, ha="center", va="top", fontsize=6.5,
                                  linespacing=1.1, multialignment="center"))
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        W = fig.get_size_inches()[0] * fig.dpi
        # "(c) Request SLO Attainment" is wider than its panel and its panel is
        # the last one, so centred under it the caption runs past the right
        # edge. It is moved left just far enough to end 2 px inside the canvas;
        # the overlap test below still guards the caption beside it.
        for t in texts:
            e = t.get_window_extent(rend)
            if e.x1 > W - 2.0:
                x, y = t.get_position()
                t.set_position((x - (e.x1 - (W - 2.0)) / W, y))
        fig.canvas.draw()
        ce = [t.get_window_extent(rend) for t in texts]
        for e1, e2 in zip(ce, ce[1:]):
            if e1.x1 > e2.x0:
                print("  ⚠ two captions overlap")
        for e in ce + [key.get_window_extent(rend)]:
            if e.x0 < -0.5 or e.x1 > W + 0.5:
                print("  ⚠ text off the canvas")
        gaps = [(ax[i + 1].get_tightbbox(rend).x0 - ax[i].get_tightbbox(rend).x1) / fig.dpi
                for i in range(2)]
        print(f"  gaps between panels: {gaps[0]:+.3f} / {gaps[1]:+.3f} in; key at {key_fs} pt")
        ps.save(fig, ps.final("kv_batch_attain_4arms.pdf"))

    build_goodput(df)

    out = ps.final("kv_batch_attain_4arms.csv")
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"wrote {out}")
    agg = df.groupby("arm", sort=False)[["kv_p50", "batch_p50", "attain_admitted",
                                          "req_goodput", "tok_goodput",
                                          "rejected_pct"]].agg(["mean", "min", "max"])
    print(agg.round(1).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
