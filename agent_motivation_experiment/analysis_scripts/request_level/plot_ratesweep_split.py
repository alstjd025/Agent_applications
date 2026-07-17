#!/usr/bin/env python3
"""Per-condition time series split into TWO images per rate condition:

  llumnix_rpm_<rpm>.png  — Llumnix control-plane layer only:
      A. gateway_current / gateway_pending / being_inferred (=current-pending)
      B. migration count (scheduler_rescheduling_total) + reject/error rate
         (gateway request_total by status). [auto-plots if captured; else a note]

  engine_rpm_<rpm>.png   — engine layer only, per engine (8000-8003):
      A. running (vllm:num_requests_running — execution batch, no queue)
      B. waiting (vllm:num_requests_waiting — engine-side queue)
      C. KV cache usage %
      D. token throughput: prefill (prompt tok/s) + decode (generation tok/s)
      E. KV cache hit rate (prefix_cache_hits/queries).  [if captured]
      F. queueing time (avg request_queue_time). [if captured]

The script auto-detects which series are present, so the same script produces the
partial plots from current runs and the full plots after the collector is
enhanced + the sweep re-run. Missing panels show a "not captured" note.
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ENGINE_PORTS = (8000, 8001, 8002, 8003)
SMOOTH = 3
PAPER = {"font.family": "serif", "font.size": 9, "axes.labelsize": 10,
         "axes.titlesize": 9.5, "legend.fontsize": 7, "legend.frameon": False,
         "lines.linewidth": 1.3}


def _load(path):
    if not os.path.isfile(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def gauge_recs(recs, key, t0=None):
    """(t_rel, summed value) for a gauge key across pre-loaded records."""
    ts, vs = [], []
    base = t0
    for r in recs:
        if not r.get("ok"):
            continue
        vals = [v for k, v in r.items()
                if k.split("|")[0] == key and isinstance(v, (int, float))]
        if not vals:
            continue
        if base is None:
            base = r["t"]
        ts.append(r["t"] - base)
        vs.append(sum(vals))
    return np.array(ts), np.array(vs), base


def rate_recs(recs, key, t0=None, label_filter=None):
    """per-second rate from a cumulative counter (optionally filtered by label substr)."""
    ts, vs, base = [], [], t0
    for r in recs:
        if not r.get("ok"):
            continue
        vals = [v for k, v in r.items()
                if k.split("|")[0] == key and isinstance(v, (int, float))
                and (label_filter is None or label_filter in k)]
        if not vals:
            continue
        if base is None:
            base = r["t"]
        ts.append(r["t"] - base)
        vs.append(sum(vals))
    t, v = np.array(ts), np.array(vs)
    if len(t) < 2:
        return np.array([]), np.array([]), base
    dt = np.diff(t)
    rr = np.clip(np.diff(v) / np.where(dt > 0, dt, 1), 0, None)
    tm = t[:-1] + dt / 2
    if SMOOTH > 1 and len(rr) >= SMOOTH:
        rr = np.convolve(rr, np.ones(SMOOTH) / SMOOTH, mode="same")
    return tm, rr, base


def _note(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
            fontsize=8, color="0.5", style="italic")


def plot_llumnix(run, rpm, out_dir, tag=None, rate_div=60.0):
    reqps = rpm / rate_div
    tag = tag or f"rpm_{rpm:g}"
    gw = _load(os.path.join(run, "server_metrics", "gateway.jsonl"))
    sc = _load(os.path.join(run, "server_metrics", "scheduler.jsonl"))
    with plt.rc_context(PAPER):
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        # A. current / pending / being_inferred
        a = ax[0]
        ct, cv, t0 = gauge_recs(gw, "gateway_current_requests")
        pt, pv, _ = gauge_recs(gw, "gateway_pending_requests", t0)
        a.plot(ct, cv, color="#d62728", lw=1.8, label="gateway_current")
        a.plot(pt, pv, color="#ff7f0e", lw=1.4, ls="--", label="gateway_pending")
        n = min(len(cv), len(pv))
        if n:
            a.plot(ct[:n], np.clip(cv[:n] - pv[:n], 0, None), color="#1f77b4",
                   lw=1.4, ls=":", label="being_inferred (=current-pending)")
        a.set_title("A. Requests in gateway"); a.set_ylabel("requests")
        a.set_xlabel("time (s)"); a.legend(loc="upper right")

        # B. migration + reject/error/drop (gateway request_total by status_code)
        b = ax[1]
        plotted = False
        # completed(200) for context + failure classes: 499 client-closed(drop),
        # 503 reject, 400 bad-request.
        for status, col, lab in (("200", "#2ca02c", "200 completed/s"),
                                  ("499", "#d62728", "499 client-closed/s"),
                                  ("503", "#ff7f0e", "503 reject/s"),
                                  ("400", "#7f7f7f", "400 error/s")):
            rt, rv, _ = rate_recs(gw, "request_total", t0, label_filter=f"status_code={status}")
            if len(rv) and rv.max() > 0:
                b.plot(rt, rv, color=col, lw=1.3, label=lab); plotted = True
        # migration decisions (cumulative) on a twin axis, if captured
        mt, mv, _ = gauge_recs(sc, "scheduler_rescheduling_total")
        if len(mv) and mv.max() > 0:
            bt = b.twinx()
            bt.plot(mt, mv, color="#9467bd", lw=1.6, ls="--", label="migration (cum)")
            bt.set_ylabel("migration count", color="#9467bd")
            bt.legend(loc="upper left"); plotted = True
        b.set_title("B. Completed / reject / error / drop  (+migration)")
        b.set_xlabel("time (s)")
        if plotted:
            b.set_ylabel("requests/s"); b.legend(loc="upper right")
        else:
            _note(b, "request_total/rescheduling not captured\n(needs collector re-run)")
        fig.suptitle(f"Llumnix layer — offered {reqps:.0f} req/s", y=1.0)
        fig.tight_layout()
        out = os.path.join(out_dir, f"llumnix_{tag}.png")
        fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


def plot_engine(run, rpm, out_dir, tag=None, rate_div=60.0):
    reqps = rpm / rate_div
    tag = tag or f"rpm_{rpm:g}"
    recs = {p: _load(os.path.join(run, "server_metrics", f"engine_{p}.jsonl")) for p in ENGINE_PORTS}
    colors = plt.cm.tab10(np.arange(4))
    with plt.rc_context(PAPER):
        fig, ax = plt.subplots(2, 3, figsize=(16, 8))

        def per_engine_gauge(axis, key, ylabel, title, scale=1.0):
            for c, p in zip(colors, ENGINE_PORTS):
                t, v, _ = gauge_recs(recs[p], key)
                if len(t):
                    axis.plot(t, v * scale, color=c, lw=1.0, label=f"eng {p}")
            axis.set_title(title); axis.set_ylabel(ylabel); axis.set_xlabel("time (s)")
            axis.legend(ncol=2, fontsize=6)

        per_engine_gauge(ax[0, 0], "vllm:num_requests_running", "requests",
                         "A. RUNNING (execution batch; no queue)")
        per_engine_gauge(ax[0, 1], "vllm:num_requests_waiting", "requests",
                         "B. WAITING (engine-side queue)")
        per_engine_gauge(ax[0, 2], "vllm:kv_cache_usage_perc", "KV usage (%)",
                         "C. KV cache usage", scale=100.0)

        # D. token throughput prefill + decode (fleet Σ)
        d = ax[1, 0]
        for key, col, lab in (("vllm:prompt_tokens_total", "#9467bd", "prefill (prompt) tok/s"),
                              ("vllm:generation_tokens_total", "#2ca02c", "decode (generation) tok/s")):
            agg_t, agg = None, None
            for p in ENGINE_PORTS:
                t, r, _ = rate_recs(recs[p], key)
                if len(t):
                    if agg is None:
                        agg_t, agg = t, r.astype(float)
                    else:
                        m = min(len(agg), len(r)); agg_t, agg = agg_t[:m], agg[:m] + r[:m]
            if agg is not None:
                d.plot(agg_t, agg, color=col, lw=1.5, label=lab)
        d.set_title("D. Token throughput (fleet Σ)"); d.set_ylabel("tokens/s")
        d.set_xlabel("time (s)"); d.legend(loc="upper right")

        # E. KV cache hit rate (prefix_cache_hits/queries) if captured
        e = ax[1, 1]
        any_hit = False
        for c, p in zip(colors, ENGINE_PORTS):
            ht, hv, _ = rate_recs(recs[p], "vllm:prefix_cache_hits_total")
            qt, qv, _ = rate_recs(recs[p], "vllm:prefix_cache_queries_total")
            if len(hv) and len(qv):
                m = min(len(hv), len(qv))
                ratio = np.divide(hv[:m], np.where(qv[:m] > 0, qv[:m], np.nan)) * 100
                e.plot(ht[:m], ratio, color=c, lw=1.0, label=f"eng {p}")
                any_hit = True
        e.set_title("E. KV cache hit rate"); e.set_xlabel("time (s)")
        if any_hit:
            e.set_ylabel("prefix-cache hit (%)"); e.legend(ncol=2, fontsize=6)
        else:
            _note(e, "prefix_cache_hits/queries not captured\n(needs collector re-run)")

        # F. queueing time (avg = d(queue_time_sum)/d(count)) if captured
        f = ax[1, 2]
        any_q = False
        for c, p in zip(colors, ENGINE_PORTS):
            st, sv, t0 = gauge_recs(recs[p], "vllm:request_queue_time_seconds_sum")
            ntc, nvc, _ = gauge_recs(recs[p], "vllm:request_queue_time_seconds_count", t0)
            if len(sv) > 2 and len(nvc) > 2:
                m = min(len(sv), len(nvc))
                dsum = np.diff(sv[:m]); dcnt = np.diff(nvc[:m])
                avg = np.divide(dsum, np.where(dcnt > 0, dcnt, np.nan)) * 1000
                f.plot(st[:m - 1] if len(st) >= m else ntc[:m-1], avg, color=c, lw=1.0, label=f"eng {p}")
                any_q = True
        f.set_title("F. Queueing time (avg per request)"); f.set_xlabel("time (s)")
        if any_q:
            f.set_ylabel("queue time (ms)"); f.legend(ncol=2, fontsize=6)
        else:
            _note(f, "request_queue_time _count not captured\n(needs collector re-run)")

        for row in ax:
            for p in row:
                p.grid(axis="y", ls=":", lw=0.5, alpha=0.5)
        fig.suptitle(f"Engine layer — offered {reqps:.0f} req/s "
                     f"(per engine 8000-8003)", y=1.0)
        fig.tight_layout()
        out = os.path.join(out_dir, f"engine_{tag}.png")
        fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out



def plot_tokens(run, rpm, out_dir, tag=None, rate_div=60.0):
    """Separate-scale prefill / decode throughput panels (one condition).

    Panel D of the engine figure puts both on one axis, where decode
    (~1e3 tok/s) is visually flattened under prefill (~1e5 tok/s). Here each
    gets its own panel and y-scale. NB vllm:prompt_tokens_total counts ALL
    scheduled prompt tokens including prefix-cache HITS (cheap attach), so
    the prefill panel is volume, not pure compute.
    """
    reqps = rpm / rate_div
    tag = tag or f"rpm_{rpm:g}"
    recs = {p: _load(os.path.join(run, "server_metrics", f"engine_{p}.jsonl"))
            for p in ENGINE_PORTS}
    colors = plt.cm.tab10(np.arange(4))
    with plt.rc_context(PAPER):
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for axis, key, lab in (
                (axes[0], "vllm:prompt_tokens_total",
                 "prefill (prompt) tok/s — incl. prefix-cache hits"),
                (axes[1], "vllm:generation_tokens_total",
                 "decode (generation) tok/s")):
            agg_t, agg = None, None
            for c, p in zip(colors, ENGINE_PORTS):
                t, r, _ = rate_recs(recs[p], key)
                if not len(t):
                    continue
                axis.plot(t, r, color=c, lw=0.9, alpha=0.7, label=f"eng {p}")
                if agg is None:
                    agg_t, agg = t, r.astype(float)
                else:
                    m = min(len(agg), len(r))
                    agg_t, agg = agg_t[:m], agg[:m] + r[:m]
            if agg is not None:
                axis.plot(agg_t, agg, color="k", lw=1.6, label="fleet Σ")
            axis.set_ylabel(lab)
            axis.grid(axis="y", ls=":", lw=0.5, alpha=0.5)
            axis.legend(ncol=3, fontsize=6.5)
        axes[1].set_xlabel("time (s)")
        fig.suptitle(f"Token throughput, separate scales — offered {reqps:g} req/s")
        fig.tight_layout()
        out = os.path.join(out_dir, f"tokens_{tag}.png")
        fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="results/*exp02b_ratesweep_rpm_*")
    ap.add_argument("--out-dir", default="results/aggregate_analysis/exp02b_plots")
    ap.add_argument("--rate-key", default="rpm_",
                    help="dirname token preceding the rate value (e.g. 'lambda_')")
    ap.add_argument("--rate-div", type=float, default=60.0,
                    help="divide the parsed value by this to get req/s")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dirs = sorted(glob.glob(args.glob),
                  key=lambda x: float(x.split(args.rate_key)[1]))
    written = []
    for run in dirs:
        rpm = float(run.split(args.rate_key)[1])
        tag = f"{args.rate_key}{rpm:g}"
        written.append(plot_llumnix(run, rpm, args.out_dir, tag, args.rate_div))
        written.append(plot_engine(run, rpm, args.out_dir, tag, args.rate_div))
        written.append(plot_tokens(run, rpm, args.out_dir, tag, args.rate_div))
    print("wrote:")
    for w in written:
        print(" ", w)


if __name__ == "__main__":
    main()
