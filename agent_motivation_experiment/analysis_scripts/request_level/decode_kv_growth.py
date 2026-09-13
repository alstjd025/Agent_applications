#!/usr/bin/env python3
"""How much KV the decode phase creates over an hour, against the offered rate.

The engines publish `vllm:generation_tokens_total` as a cumulative counter.
Differencing consecutive scrapes gives a rate; the rate times the model's
per-token KV size gives the bytes per second of cache the decode phase brings
into existence. The area under the curve is the volume created over the run.

The offered rate is drawn on the second axis from the client's own record --
one point per arrival, binned per second -- rather than from the trace plan, so
the two curves come from the same run rather than from an input file and a run.

WHAT THE FIGURE SAYS. The offered rate swings by several fold across the hour;
the decode phase creates KV at a rate that barely moves. Decode load is not made
by what is arriving, it is made by what was already admitted, and it persists for
as long as those requests live.

⚠ WHAT IT IS NOT. Not resident occupancy. A request releases its whole footprint
when it finishes, so the hour's total is the volume created and destroyed, not a
level; the physical pool is three orders of magnitude smaller. Write "decode
creates KV at X GB/s", never "X TB accumulates".

Per-token KV = 2 (K and V) x layers x kv_heads x head_dim x dtype_bytes.
"""
import argparse, csv, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "NanumBarunGothic"
plt.rcParams["axes.unicode_minus"] = False

# 80 layers, 8 KV heads (GQA), head_dim 128, bf16.
# ⚠ Qwen2.5-72B is entered as the same shape and has NOT been checked against the
# engine's own config. Verify before this figure goes in the paper.
BYTES_PER_TOKEN = {"llama": 2 * 80 * 8 * 128 * 2, "qwen": 2 * 80 * 8 * 128 * 2}


def counter(run, name):
    """Absolute unix seconds and the fleet-summed cumulative counter."""
    per = []
    for f in sorted(glob.glob(os.path.join(run, "server_metrics", "engine_*.jsonl"))):
        ts, vs = [], []
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            v = next((x for k, x in r.items() if k.startswith(name)), None)
            if v is None:
                continue
            ts.append(r["t"]); vs.append(v)
        if len(ts) > 1:
            per.append((np.array(ts), np.array(vs, float)))
    if not per:
        return None, None
    lo, hi = max(t[0] for t, _ in per), min(t[-1] for t, _ in per)
    grid = np.arange(lo, hi, 1.0)
    tot = np.zeros_like(grid)
    for t, v in per:
        tot += np.interp(grid, t, v)
    return grid, tot


def arrivals(run):
    """Unix-second bins and arrivals per second, from the client's own record.

    `agent != "job_summary"` is the arrival filter the loader uses: the file
    writes a summary row beside every request row, and a third kind, `grace_cut`,
    which is a separate arrival rather than a duplicate.
    """
    ts = []
    with open(os.path.join(run, "metrics.csv")) as fh:
        for row in csv.DictReader(fh):
            if row.get("agent") == "job_summary":
                continue
            try:
                ts.append(float(row["start_time"]))
            except (TypeError, ValueError):
                continue
    if not ts:
        return None, None
    ts = np.array(ts)
    grid = np.arange(ts.min(), ts.max(), 1.0)
    cnt, _ = np.histogram(ts, bins=np.append(grid, grid[-1] + 1))
    return grid, cnt.astype(float)


def itl_per_engine(run):
    """Per-engine mean inter-token latency per second, from each engine's own
    histogram counters.

    Drawn separately rather than pooled because this policy makes the engines
    asymmetric on purpose: measured on EXP-109 repeat 1, one engine produced 40%
    of the fleet's tokens while holding the flattest latency (p10-p90 of
    35.5-45.4 ms) and another sat at 70.5 ms of median. A token-weighted fleet
    mean is pulled toward the busiest engine and describes no engine that ran.
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(run, "server_metrics", "engine_*.jsonl"))):
        ts, sm, ct = [], [], []
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            a = next((x for k, x in r.items()
                      if k.startswith("vllm:inter_token_latency_seconds_sum")), None)
            b = next((x for k, x in r.items()
                      if k.startswith("vllm:inter_token_latency_seconds_count")), None)
            if a is None or b is None:
                continue
            ts.append(r["t"]); sm.append(a); ct.append(b)
        if len(ts) < 2:
            continue
        t = np.array(ts)
        grid = np.arange(t[0], t[-1], 1.0)
        ds = np.diff(np.interp(grid, t, np.array(sm, float)), prepend=np.nan)
        dc = np.diff(np.interp(grid, t, np.array(ct, float)), prepend=np.nan)
        ds = np.nan_to_num(ds); dc = np.nan_to_num(dc)
        ds[ds < 0] = 0; dc[dc < 0] = 0
        name = os.path.basename(f).replace("engine_", "").replace(".jsonl", "")
        out[name] = (grid, ds, dc)
    return out


def itl_ms(run):
    """Fleet mean inter-token latency per second, from the engines' own
    histogram counters.

    Aggregated as (sum of the summed latencies) / (sum of the counts) across
    engines, which is the token-weighted mean. Averaging the four per-engine
    means instead would weigh an idle engine's few tokens as heavily as a busy
    engine's many, and the busy engine is the one the budget is about.
    """
    num = den = None
    grid = None
    for f in sorted(glob.glob(os.path.join(run, "server_metrics", "engine_*.jsonl"))):
        ts, sm, ct = [], [], []
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            a = next((x for k, x in r.items()
                      if k.startswith("vllm:inter_token_latency_seconds_sum")), None)
            b = next((x for k, x in r.items()
                      if k.startswith("vllm:inter_token_latency_seconds_count")), None)
            if a is None or b is None:
                continue
            ts.append(r["t"]); sm.append(a); ct.append(b)
        if len(ts) < 2:
            continue
        t = np.array(ts)
        if grid is None:
            grid = np.arange(t[0], t[-1], 1.0)
            num = np.zeros_like(grid); den = np.zeros_like(grid)
        ds = np.diff(np.interp(grid, t, np.array(sm, float)), prepend=np.nan)
        dc = np.diff(np.interp(grid, t, np.array(ct, float)), prepend=np.nan)
        ds = np.nan_to_num(ds); dc = np.nan_to_num(dc)
        ds[ds < 0] = 0; dc[dc < 0] = 0
        num += ds; den += dc
    if grid is None:
        return None, None
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return grid, out * 1000.0


def smooth(x, k):
    return np.convolve(x, np.ones(k) / k, mode="same") if k > 1 else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True,
                    help="label|model|rep1_dir|rep2_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smooth-s", type=int, default=20)
    ap.add_argument("--budget-ms", type=float, default=50.0,
                    help="horizontal reference: the tightest class's per-token "
                         "budget, 50 ms for chat on this workload")
    ap.add_argument("--overlay", choices=["arrival", "itl", "itl-engine"],
                    default="arrival",
                    help="second axis: offered rate, or the engines' own "
                         "inter-token latency")
    a = ap.parse_args()

    panels = [p.split("|") for p in a.panel]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 3.3),
                             sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)

    for ax, (label, model, *runs) in zip(axes, panels):
        bpt = BYTES_PER_TOKEN[model]
        ax2 = ax.twinx()
        note = []
        for i, run in enumerate(runs):
            if not run:
                continue
            t, cum = counter(run, "vllm:generation_tokens_total")
            if t is None:
                continue
            d = np.diff(cum, prepend=cum[0]); d[d < 0] = 0.0
            gbps = smooth(d * bpt / 1e9, a.smooth_s)
            at, ac = arrivals(run)
            # Align both to the first arrival, so 0 on the x axis is the moment
            # load starts rather than the moment the collector started.
            t0 = at[0] if at is not None else t[0]
            m = (t - t0) / 60.0
            keep = m >= 0
            if i == 0:
                ax.fill_between(m[keep], 0, gbps[keep], color="#1f77b4",
                                alpha=0.35, lw=0)
            ax.plot(m[keep], gbps[keep], color="#1f77b4",
                    lw=1.1 if i == 0 else 0.8, ls="-" if i == 0 else "--",
                    label=f"decode KV 생성, rep{i+1}", zorder=3)
            tot = (cum[-1] - cum[0]) * bpt / 1e9
            if i == 0:
                g = gbps[keep]; g = g[g > 0]
                note.append(f"총 {tot:,.0f} GB")
                if a.overlay == "arrival" and at is not None:
                    ax2.plot((at - t0) / 60.0, smooth(ac, a.smooth_s),
                             color="#d62728", lw=0.9, alpha=0.85,
                             label="도착률", zorder=2)
                    r = smooth(ac, a.smooth_s); r = r[r > 0]
                    note.append(f"도착률 p90/p10 = "
                                f"{np.percentile(r,90)/np.percentile(r,10):.1f}배")
                elif a.overlay == "itl-engine":
                    ENG = ["#d62728", "#ff7f0e", "#8c564b", "#e377c2"]
                    per = itl_per_engine(run)
                    lows, highs = [], []
                    for j, (name, (grid, ds, dc)) in enumerate(sorted(per.items())):
                        sd = smooth(ds, a.smooth_s); cd = smooth(dc, a.smooth_s)
                        v = np.divide(sd, cd, out=np.full_like(sd, np.nan),
                                      where=cd > 0) * 1000.0
                        ok = ~np.isnan(v)
                        ax2.plot((grid[ok] - t0) / 60.0, v[ok], color=ENG[j % 4],
                                 lw=0.8, alpha=0.9, label=f"엔진 {name}", zorder=2)
                        vv = v[ok]; vv = vv[vv > 0]
                        lows.append(np.percentile(vv, 50))
                        highs.append(np.percentile(vv, 90))
                    ax2.axhline(a.budget_ms, color="#555555", lw=0.8, ls="--",
                                zorder=1)
                    ax2.text(61.5, a.budget_ms, f" chat 예산 {a.budget_ms:.0f} ms",
                             fontsize=7, color="#555555", va="bottom", ha="right")
                    note.append(f"엔진별 ITL 중앙값 "
                                f"{min(lows):.0f}~{max(lows):.0f} ms")
                else:
                    it, iv = itl_ms(run)
                    if it is not None:
                        ok = ~np.isnan(iv)
                        sm = smooth(np.nan_to_num(iv), a.smooth_s)
                        ax2.plot((it[ok] - t0) / 60.0, sm[ok], color="#d62728",
                                 lw=0.9, alpha=0.85,
                                 label="토큰간 시간 (엔진 측정)", zorder=2)
                        v = sm[ok]; v = v[v > 0]
                        note.append(f"토큰간 시간 p10/p90 = "
                                    f"{np.percentile(v,10):.0f} / "
                                    f"{np.percentile(v,90):.0f} ms")
                note.append(f"decode 생성 p90/p10 = "
                            f"{np.percentile(g,90)/np.percentile(g,10):.1f}배")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("부하 시작 후 경과 시간 (분)")
        ax.set_xlim(0, 62)
        ax.set_ylim(0, None)
        ax2.set_ylim(0, None)
        ax2.set_ylabel("도착률 (req/s)" if a.overlay == "arrival"
                       else "토큰간 시간 (ms)", color="#333333", fontsize=9)
        ax2.tick_params(axis="y", colors="#333333", labelsize=8)
        ax.grid(axis="both", ls=":", lw=0.5, color="#909090")
        ax.set_axisbelow(True)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc="upper left")
        ax.text(0.985, 0.03, "\n".join(note), transform=ax.transAxes, fontsize=7.5,
                ha="right", va="bottom", color="#333333")
    axes[0].set_ylabel("decode가 만드는 KV (GB/s)\n곡선 아래 면적 = 총량")
    fig.tight_layout()
    fig.savefig(a.out, dpi=200)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
