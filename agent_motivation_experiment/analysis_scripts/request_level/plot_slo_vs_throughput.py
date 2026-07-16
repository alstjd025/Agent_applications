#!/usr/bin/env python3
"""Rate-summary figures: x = offered rate, right y = output token throughput.

Produces FOUR figures in --out-dir:
  slo_vs_throughput_steady.png : SLO attainment (steady window only; legend just
                                 "SLO attainment") + throughput
  slo_vs_throughput_full.png   : both attainment variants (steady + full-run) +
                                 throughput
  kv_vs_throughput.png         : fleet KV usage (steady-window time-mean, shaded
                                 p10-p90 band across ticks) + throughput
  inflight_vs_throughput.png   : in-flight requests split running / queued
                                 (steady-window mean + p10-p90 bands) + throughput

SLO rules (as agreed): violation = TTFT>5s (arrival-anchored) OR meanTBT>50ms;
errors/timeouts/run-end-cut excluded. Steady window = [60s, last-arrival-20s].

Within-condition time variation: KV and in-flight vary over a run, so each point
is the steady-window TIME-MEAN and the shaded band is the p10-p90 of the per-tick
values — a wide band flags a non-stationary (drifting/overloaded) condition.

Definitions: running = Σ engine vllm:num_requests_running (in execution batch);
queued = gateway_current − running (in system but not in a batch: gateway queue,
scheduling, engine waiting). NB for chained workloads (e.g. swe_bench), jobs
sleeping in a tool-delay have no request in flight and are invisible here.
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TTFT_SLO_S, TBT_SLO_MS = 5.0, 50.0
STEADY_LO = 60.0
DRAIN_S = 20.0

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_slomod", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "slo_sliding_window.py"))
_slomod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_slomod)
_slo_gw_timeout = _slomod.gw_timeout_mask
ENGINE_PORTS = (8000, 8001, 8002, 8003)

PAPER = {"font.family": "serif", "font.size": 9, "axes.labelsize": 10,
         "axes.titlesize": 10, "legend.fontsize": 8, "legend.frameon": False,
         "lines.linewidth": 1.6, "lines.markersize": 6}
BLUE, RED, GREEN, ORANGE = "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"


def _series(run, fname, key):
    """(t_rel, per-tick summed value) for a gauge key in a server_metrics file."""
    p = os.path.join(run, "server_metrics", fname)
    ts, vs, t0 = [], [], None
    if not os.path.isfile(p):
        return np.array([]), np.array([])
    for line in open(p):
        rec = json.loads(line)
        if not rec.get("ok"):
            continue
        vals = [v for k, v in rec.items()
                if k.split("|")[0] == key and isinstance(v, (int, float))]
        if not vals:
            continue
        if t0 is None:
            t0 = rec["t"]
        ts.append(rec["t"] - t0)
        vs.append(sum(vals))
    return np.array(ts), np.array(vs)


def _steady_stats(t, v, lo, hi):
    """(mean, p10, p90) of per-tick values inside the steady window."""
    if len(t) == 0:
        return (np.nan,) * 3
    m = (t >= lo) & (t <= hi)
    if not m.any():
        return (np.nan,) * 3
    w = v[m]
    return float(w.mean()), float(np.percentile(w, 10)), float(np.percentile(w, 90))


def condition_stats(run_dir, steady_anchor="end", steady_max_s=None):
    df = pd.read_csv(os.path.join(run_dir, "metrics.csv"), low_memory=False)
    r = df[df.agent != "job_summary"].copy()   # request/chain_call rows
    r = r[pd.to_numeric(r.get("start_time"), errors="coerce").notna()]
    t0 = r["start_time"].min()
    r["rel"] = r["start_time"] - t0
    dur = r["end_time"].max() - t0
    if steady_anchor == "arrival":
        # End the steady window at the LAST ARRIVAL, not the last completion.
        # When stragglers drain long after submission stops (e.g. exp10's
        # gateway-timeout tail: ~10 min of near-idle abort draining), the
        # completion-anchored window dilutes tok/s and server-state means
        # with the idle drain phase. Arrival anchoring keeps the window on
        # the load phase only. (Default "end" preserves historical numbers.)
        dur = min(dur, r["rel"].max())
    if steady_max_s is not None:
        # cross-experiment standard window: cap at steady_max_s from t0
        dur = min(dur, steady_max_s)

    bl = lambda c: r[c].fillna(False).astype(bool) if c in r.columns else pd.Series(False, index=r.index)
    rejected = bl("is_rejected")
    # admission rejects are NOT run-boundary noise: keep them as a separate
    # category (they count as violations in the offered-goodput view).
    # Gateway 300s-SSE-timeout kills (400 after >=295s wait) are TTFT
    # violations, not exclusions — see slo_sliding_window.gw_timeout_mask.
    gw = _slo_gw_timeout(r, bl)
    cls = r[(~(bl("is_error") | bl("is_timeout") | bl("is_server_terminated")) | gw) & ~rejected].copy()
    rej = r[rejected].copy()
    tbt = pd.to_numeric(cls["tbt_mean_ms"], errors="coerce")
    cls["violate"] = ((pd.to_numeric(cls["first_token_latency"], errors="coerce") > TTFT_SLO_S)
                      | (tbt > TBT_SLO_MS) | gw.reindex(cls.index, fill_value=False))

    hi = dur - DRAIN_S
    full = 100.0 * (~cls["violate"]).mean() if len(cls) else np.nan
    sw = cls[(cls["rel"] >= STEADY_LO) & (cls["rel"] < hi)]
    steady = 100.0 * (~sw["violate"]).mean() if len(sw) else np.nan
    # offered view: rejected requests are violations (they got no service)
    rej_w = rej[(rej["rel"] >= STEADY_LO) & (rej["rel"] < hi)]
    n_off = len(sw) + len(rej_w)
    steady_offered = 100.0 * (~sw["violate"]).sum() / n_off if n_off else np.nan
    reject_pct = 100.0 * len(rej_w) / n_off if n_off else 0.0

    ok = r[r["success"].astype(bool)].copy()
    ok["rel_end"] = ok["end_time"] - t0
    win = ok[(ok["rel_end"] >= STEADY_LO) & (ok["rel_end"] <= hi)]
    tokps = pd.to_numeric(win["output_tokens"], errors="coerce").sum() / max(1e-9, hi - STEADY_LO)

    # server-side series (their own clock; same steady window bounds)
    kv_per = []
    run_t, run_v = None, None
    for pnum in ENGINE_PORTS:
        kt, kvv = _series(run_dir, f"engine_{pnum}.jsonl", "vllm:kv_cache_usage_perc")
        if len(kt):
            kv_per.append((kt, kvv))
        rt, rv = _series(run_dir, f"engine_{pnum}.jsonl", "vllm:num_requests_running")
        if len(rt):
            if run_v is None:
                run_t, run_v = rt, rv.astype(float)
            else:
                n = min(len(run_v), len(rv))
                run_t, run_v = run_t[:n], run_v[:n] + rv[:n]
    # fleet KV = mean of engines per tick
    if kv_per:
        n = min(len(v) for _, v in kv_per)
        kv_t = kv_per[0][0][:n]
        kv_v = np.mean([v[:n] for _, v in kv_per], axis=0) * 100.0
    else:
        kv_t, kv_v = np.array([]), np.array([])
    gt, gv = _series(run_dir, "gateway.jsonl", "gateway_current_requests")

    if steady_anchor == "arrival":
        srv_hi = dur - DRAIN_S      # load phase only (see dur clamp above)
    else:
        srv_hi = max(kv_t.max() if len(kv_t) else 0, dur) - DRAIN_S
    kv = _steady_stats(kv_t, kv_v, STEADY_LO, srv_hi)
    running = _steady_stats(run_t if run_t is not None else np.array([]),
                            run_v if run_v is not None else np.array([]), STEADY_LO, srv_hi)
    # queued = gateway_current - running, aligned by tick index
    if len(gv) and run_v is not None:
        n = min(len(gv), len(run_v))
        queued = _steady_stats(gt[:n], np.clip(gv[:n] - run_v[:n], 0, None), STEADY_LO, srv_hi)
    else:
        queued = (np.nan,) * 3
    return dict(full=full, steady=steady, steady_offered=steady_offered,
                reject_pct=reject_pct, tokps=tokps, kv=kv, running=running, queued=queued)


def _tput_axis(ax, x, tok):
    ax2 = ax.twinx()
    ax2.plot(x, tok, "^-", color=RED, label="Output token throughput (steady)")
    ax2.set_ylabel("Output tokens/s", color=RED)
    ax2.tick_params(axis="y", colors=RED)
    ax2.set_ylim(0, max(tok) * 1.15)
    return ax2


def _finish(fig, ax, ax2, out, title):
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="center left", fontsize=7.5)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote:", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="results/exp04_final_8192conc/*rpm_*")
    ap.add_argument("--out-dir", default="results/aggregate_analysis/exp04_slo")
    ap.add_argument("--rate-unit", default="req/s",
                    help="x-axis unit label (e.g. 'req/s' or 'jobs/s')")
    ap.add_argument("--rate-key", default="rpm_",
                    help="dirname token preceding the rate value (e.g. 'lambda_')")
    ap.add_argument("--rate-div", type=float, default=60.0,
                    help="divide the parsed value by this to get the rate "
                         "(60 for rpm dirs, 1 for lambda dirs)")
    ap.add_argument("--steady-anchor", choices=["end", "arrival"], default="end",
                    help="steady-window end: last completion (historical) or "
                         "last arrival (load phase only; use when a long "
                         "post-submission drain tail exists)")
    ap.add_argument("--steady-max-s", type=float, default=None,
                    help="cap the steady-window end at this many seconds from "
                         "the first arrival (e.g. 360 to recut a 10-min run "
                         "to the exp05-equivalent [60,340] window)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dirs = [d for d in sorted(glob.glob(args.glob),
                              key=lambda x: float(x.split(args.rate_key)[1]))
            if os.path.isdir(d)]
    rows = []
    for d in dirs:
        val = float(d.split(args.rate_key)[1])
        s = condition_stats(d, steady_anchor=args.steady_anchor,
                            steady_max_s=args.steady_max_s)
        s["rate"] = val / args.rate_div
        rows.append(s)
        print(f"{s['rate']:6.2f} {args.rate_unit}: attain_steady={s['steady']:5.1f}% "
              f"full={s['full']:5.1f}%  tok/s={s['tokps']:7.0f}  KVμ={s['kv'][0]:5.1f}%  "
              f"runμ={s['running'][0]:6.0f} queμ={s['queued'][0]:6.0f}"
              + (f"  rej={s['reject_pct']:4.1f}% offered={s['steady_offered']:5.1f}%"
                 if s["reject_pct"] > 0 else ""))
    any_rejects = any(r["reject_pct"] > 0 for r in rows)
    x = [r["rate"] for r in rows]
    tok = [r["tokps"] for r in rows]
    xlabel = f"Offered rate ({args.rate_unit})"
    slo_note = (f"SLO: TTFT≤{TTFT_SLO_S:.0f}s & meanTBT≤{TBT_SLO_MS:.0f}ms; "
                "errors/timeouts/run-end-cut excluded")

    with plt.rc_context(PAPER):
        # 1) steady-only
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        ax.plot(x, [r["steady"] for r in rows], "o-", color=BLUE, label="SLO attainment")
        if any_rejects:
            ax.plot(x, [r["steady_offered"] for r in rows], "D-", color="#9467bd",
                    label="SLO attainment (offered: rejects = violations)")
            ax.plot(x, [r["reject_pct"] for r in rows], "x--", color="#ff7f0e",
                    label="Rejection rate")
        ax.set_xlabel(xlabel); ax.set_xticks(x)
        ax.set_ylabel("SLO attainment (%)", color=BLUE); ax.set_ylim(-3, 105)
        ax.tick_params(axis="y", colors=BLUE); ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
        ax2 = _tput_axis(ax, x, tok)
        _finish(fig, ax, ax2, os.path.join(args.out_dir, "slo_vs_throughput_steady.png"),
                f"SLO attainment vs output throughput\n({slo_note})")

        # 2) both variants
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        ax.plot(x, [r["steady"] for r in rows], "o-", color=BLUE,
                label="SLO attainment (steady)")
        ax.plot(x, [r["full"] for r in rows], "s--", color=BLUE, alpha=0.5,
                label="SLO attainment (full run, incl. warmup)")
        ax.set_xlabel(xlabel); ax.set_xticks(x)
        ax.set_ylabel("SLO attainment (%)", color=BLUE); ax.set_ylim(-3, 105)
        ax.tick_params(axis="y", colors=BLUE); ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
        ax2 = _tput_axis(ax, x, tok)
        _finish(fig, ax, ax2, os.path.join(args.out_dir, "slo_vs_throughput_full.png"),
                f"SLO attainment (steady + full) vs output throughput\n({slo_note})")

        # 3) KV vs throughput (mean + p10-p90 band)
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        kvm = [r["kv"][0] for r in rows]
        ax.plot(x, kvm, "o-", color=GREEN, label="Fleet KV usage (steady mean)")
        ax.fill_between(x, [r["kv"][1] for r in rows], [r["kv"][2] for r in rows],
                        color=GREEN, alpha=0.18, label="KV p10–p90 over time")
        ax.set_xlabel(xlabel); ax.set_xticks(x)
        ax.set_ylabel("KV cache usage (%)", color=GREEN); ax.set_ylim(0, 105)
        ax.tick_params(axis="y", colors=GREEN); ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
        ax2 = _tput_axis(ax, x, tok)
        _finish(fig, ax, ax2, os.path.join(args.out_dir, "kv_vs_throughput.png"),
                "Fleet KV usage vs output throughput\n"
                "(band = within-run p10–p90; wide band ⇒ non-stationary condition)")

        # 4) in-flight (running/queued) vs throughput
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        ax.plot(x, [r["running"][0] for r in rows], "o-", color=BLUE,
                label="running (Σ engine batches, steady mean)")
        ax.fill_between(x, [r["running"][1] for r in rows], [r["running"][2] for r in rows],
                        color=BLUE, alpha=0.15)
        ax.plot(x, [r["queued"][0] for r in rows], "s-", color=ORANGE,
                label="queued (in system − running, steady mean)")
        ax.fill_between(x, [r["queued"][1] for r in rows], [r["queued"][2] for r in rows],
                        color=ORANGE, alpha=0.18)
        ax.set_xlabel(xlabel); ax.set_xticks(x)
        ax.set_ylabel("in-flight requests"); ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
        ax2 = _tput_axis(ax, x, tok)
        _finish(fig, ax, ax2, os.path.join(args.out_dir, "inflight_vs_throughput.png"),
                "In-flight requests (running vs queued) vs output throughput\n"
                "(bands = within-run p10–p90; tool-delay-sleeping jobs not visible here)")


if __name__ == "__main__":
    main()
