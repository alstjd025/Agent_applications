"""Per-class performance breakdown for every policy on the four hour traces, and each policy's
engine operating point over time. Exploratory figures (PNG) for the problem analysis of
2026-11-20, not paper figures.

  breakdown_<trace>.png / .csv   rows: TTFT, TBT, outcome; one group per class (outcome adds "all")
      TTFT and TBT: bar = MEAN over the requests that finished, whisker = their P99 (a quantile of
      the same requests, not a spread over repeats). Finished = not rejected, not errored, not cut
      off at the end of the run; TBT only for requests with >= 2 output tokens. Each value is the
      mean of the repeats; the CSV carries min/max over repeats and n_repeats.
      Outcome: every arrival of the class, as met (ladder95) / missed / rejected / unfinished.
  operating_point.png / .csv     per trace, per policy (repeat 1), fleet inter-token latency and
      running requests per engine in 5-minute windows, from the engines' vLLM counters, with the
      class TBT budgets as dashed lines.

Per-request data: the dashboard's scored table (dashboard-cache/runs/<run>/requests.csv.gz), the
same ladder95 verdicts as every score in the records. Arms whose runs are missing are skipped and
named on stdout.

usage: fig_breakdown.py [outdir]
"""
import glob, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
CACHE = "/home/nxclab/scratch/dashboard-cache/runs/"
OUT = sys.argv[1] if len(sys.argv) > 1 else RES + "aggregate_analysis/capacity/"
BUDGET = {"chat": (5.0, 50.0), "swe": (7.0, 75.0), "deepresearch": (10.0, 100.0)}
CLASS_LABEL = {"chat": "Chat", "swe": "Agent (swe)", "deepresearch": "Deep research", "all": "All"}
# (label, colour, run glob per trace)
ARMS = [
    ("FluidServe v0.5", "#9ecae1", {"shift": "*exp161hr*_fsexactprog_shift", "onlychat25": "*exp164hr*_fsexactprog_onlychat25",
                                    "onlyswe": "*exp164hr*_fsexactprog_onlyswe", "onlydr": "*exp164hr*_fsexactprog_onlydr"}),
    ("FluidServe v0.6", "#1f77b4", {"shift": "*exp161hr*_fsg10dl_shift", "onlychat25": "*exp164hr*_fsg10dl_onlychat25",
                                    "onlyswe": "*exp164hr*_fsg10dl_onlyswe", "onlydr": "*exp164hr*_fsg10dl_onlydr"}),
    ("PolyServe", "#d62728", {"shift": "*exp168hr*_polyservept75_shift", "onlychat25": "*exp165hr*_polyservept75_onlychat25",
                              "onlyswe": "*exp165hr*_polyservept75_onlyswe", "onlydr": "*exp165hr*_polyservept75_onlydr"}),
    ("Llumnix SLO", "#98df8a", {"shift": "*exp168hr*_slot75_shift"}),
    ("Llumnix SLO (resident gate)", "#2ca02c", {"shift": "*exp168hr*_slot75inc_shift", "onlychat25": "*exp165hr*_slot75inc_onlychat25",
                                                "onlyswe": "*exp165hr*_slot75inc_onlyswe", "onlydr": "*exp165hr*_slot75inc_onlydr"}),
    ("llm-d (p90, paper)", "#9467bd", {"shift": "*exp167r*_llmdslot75_shift", "onlychat25": "*exp165mr*_llmdslot75_onlychat25",
                                       "onlyswe": "*exp165mr*_llmdslot75_onlyswe", "onlydr": "*exp165mr*_llmdslot75_onlydr"}),
    ("llm-d (p50)", "#c5b0d5", {"shift": "*exp167r*_llmdp50slot75_shift", "onlychat25": "*exp166r*_llmdp50slot75_onlychat25",
                                "onlyswe": "*exp167r*_llmdp50slot75_onlyswe", "onlydr": "*exp167r*_llmdp50slot75_onlydr"}),
]
TRACES = [("shift", "Mixed paper trace (chat-heavy, three classes)", ["chat", "swe", "deepresearch"]),
          ("onlychat25", "Chat only, 2.5x arrival rate", ["chat"]),
          ("onlyswe", "Agent (swe) only", ["swe"]),
          ("onlydr", "Deep research only", ["deepresearch"])]
SEG = [("met", "SLO met", "#2c7fb8"), ("missed", "SLO missed", "#fdae61"),
       ("rejected", "rejected", "#d9d9d9"), ("unfinished", "unfinished at end", "#ffffff")]


def runs_for(pattern):
    return sorted(os.path.basename(p) for p in glob.glob(RES + pattern)
                  if os.path.exists(CACHE + os.path.basename(p) + "/requests.csv.gz"))


def per_run(run, classes):
    q = pd.read_csv(CACHE + run + "/requests.csv.gz")
    out = {}
    for c in classes + ["all"]:
        g = q if c == "all" else q[q["class"] == c]
        rej = g.rejected.astype(bool); cut = g.cutoff.astype(bool) & ~rej; err = g.errored.astype(bool)
        met = g.ladder_ok.astype(bool) & ~rej & ~cut & ~err
        n = float(len(g))
        d = dict(arrivals=int(n), met=100 * met.sum() / n, rejected=100 * rej.sum() / n,
                 unfinished=100 * cut.sum() / n, missed=100 * (n - met.sum() - rej.sum() - cut.sum()) / n)
        fin = g[~rej & ~cut & ~err & g.first_token_latency.notna()]
        tt = fin.first_token_latency.astype(float)
        it = fin[fin.output_tokens >= 2].itl_ms.astype(float)
        d.update(ttft_mean=tt.mean(), ttft_p99=tt.quantile(.99), tbt_mean=it.mean(), tbt_p99=it.quantile(.99), n_finished=len(fin))
        out[c] = d
    return out


def breakdown(trace, title, classes):
    rows, arms = [], []
    for label, col, pats in ARMS:
        rs = runs_for(pats[trace]) if trace in pats else []
        if not rs:
            print(f"  [{trace}] {label}: no scored runs for {pats.get(trace)} -- skipped")
            continue
        arms.append((label, col))
        for i, r in enumerate(rs, 1):
            for c, d in per_run(r, classes).items():
                rows.append(dict(trace=trace, arm=label, repeat=i, run=r, cls=c, **d))
        print(f"  [{trace}] {label}: {len(rs)} run(s) {', '.join(rs)}")
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in ("trace", "arm", "repeat", "run", "cls")]
    agg = df.groupby(["arm", "cls"])[num].agg(["mean", "min", "max"])
    agg.columns = [f"{a}" if b == "mean" else f"{a}_{b}" for a, b in agg.columns]
    agg["n_repeats"] = df.groupby(["arm", "cls"]).size()
    agg = agg.reset_index(); agg.insert(0, "trace", trace)
    agg["latency_stat"] = "bar=mean, whisker=P99 over finished requests; scoring ladder95"
    agg.to_csv(OUT + f"breakdown_{trace}.csv", index=False)

    groups_lat = classes
    groups_out = (["all"] + classes) if len(classes) > 1 else classes
    n = len(arms); gap = 1.0
    fig, ax = plt.subplots(3, 1, figsize=(max(5.5, 1.6 + 1.0 * n * len(groups_out)), 7.2))
    for pi, (stat, unit, idx) in enumerate([("ttft", "s", 0), ("tbt", "ms", 1)]):
        a = ax[pi]; centres = []
        for gi, c in enumerate(groups_lat):
            base = gi * (n + gap); centres.append(base + (n - 1) / 2)
            for j, (label, col) in enumerate(arms):
                r = agg[(agg.arm == label) & (agg.cls == c)]
                if r.empty:
                    continue
                m, p = float(r[f"{stat}_mean"].iloc[0]), float(r[f"{stat}_p99"].iloc[0])
                a.bar(base + j, m, 0.8, color=col, edgecolor="black", lw=0.4, zorder=3)
                a.errorbar([base + j], [m], yerr=[[0], [max(p - m, 0)]], fmt="none", ecolor="black", elinewidth=0.6, capsize=2, zorder=4)
                a.text(base + j, m, f"{m:.2f}" if unit == "s" else f"{m:.0f}", ha="center", va="bottom", fontsize=5.5, zorder=5)
            b = BUDGET[c][idx]
            a.plot([base - 0.5, base + n - 0.5], [b, b], "r--", lw=0.9, zorder=6)
        a.set_xticks(centres); a.set_xticklabels([CLASS_LABEL[c] for c in groups_lat])
        a.set_ylabel(f"{'TTFT' if stat == 'ttft' else 'mean time between tokens'} ({unit})")
        a.grid(axis="y", ls=":", lw=0.5); a.set_axisbelow(True)
    a = ax[2]; centres = []
    for gi, c in enumerate(groups_out):
        base = gi * (n + gap); centres.append(base + (n - 1) / 2)
        for j, (label, col) in enumerate(arms):
            r = agg[(agg.arm == label) & (agg.cls == c)]
            if r.empty:
                continue
            bottom = 0
            for key, _lab, sc in SEG:
                v = float(r[key].iloc[0])
                a.bar(base + j, v, 0.8, bottom=bottom, color=sc, edgecolor=col, lw=1.2, hatch="////" if key == "unfinished" else None)
                bottom += v
            a.text(base + j, 101, f"{float(r['met'].iloc[0]):.0f}", ha="center", va="bottom", fontsize=5.5)
    a.set_xticks(centres); a.set_xticklabels([CLASS_LABEL[c] for c in groups_out])
    a.set_ylim(0, 108); a.set_ylabel("arrivals (%)")
    h_arm = [plt.Rectangle((0, 0), 1, 1, fc=col, ec="black", lw=0.4) for _l, col in arms]
    h_seg = [plt.Rectangle((0, 0), 1, 1, fc=sc, ec="black", lw=0.4, hatch="////" if k == "unfinished" else None) for k, _l, sc in SEG]
    fig.legend(h_arm + h_seg, [l for l, _c in arms] + [l for _k, l, _c in SEG], loc="lower center", ncol=4, fontsize=6.5, frameon=False)
    fig.suptitle(f"{title}: bar = mean, whisker = P99 of finished requests; red dashed = class budget\n"
                 f"outcome per arrival (ladder95); number above = % met; edge colour = policy", fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(OUT + f"breakdown_{trace}.png", dpi=170); plt.close(fig)
    return arms


def engine_windows(run):
    E = [[json.loads(l) for l in open(f)] for f in sorted(glob.glob(RES + run + "/server_metrics/engine_*.jsonl"))]
    t0 = min(e[0]["t"] for e in E)
    g = lambda r, k: sum(v for kk, v in r.items() if kk.split("|")[0] == k and isinstance(v, (int, float)))
    out = []
    for w in range(12):
        s = c = 0.0; run_ = []
        for e in E:
            sel = [x for x in e if w * 300 <= x["t"] - t0 < (w + 1) * 300]
            if len(sel) < 3:
                continue
            s += g(sel[-1], "vllm:inter_token_latency_seconds_sum") - g(sel[0], "vllm:inter_token_latency_seconds_sum")
            c += g(sel[-1], "vllm:inter_token_latency_seconds_count") - g(sel[0], "vllm:inter_token_latency_seconds_count")
            run_.append(np.mean([g(x, "vllm:num_requests_running") for x in sel]))
        out.append((w * 5 + 2.5, 1000 * s / c if c else np.nan, np.mean(run_) if run_ else np.nan))
    return out


def operating_point():
    fig, ax = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True)
    rows = []
    for ti, (trace, title, classes) in enumerate(TRACES):
        for label, col, pats in ARMS:
            rs = runs_for(pats[trace]) if trace in pats else []
            if not rs:
                continue
            w = engine_windows(rs[0])
            x = [a for a, _b, _c in w]
            ax[0, ti].plot(x, [b for _a, b, _c in w], "-o", ms=2.5, color=col, label=label)
            ax[1, ti].plot(x, [c for _a, _b, c in w], "-o", ms=2.5, color=col, label=label)
            rows += [dict(trace=trace, arm=label, run=rs[0], minute=a, fleet_itl_ms=b, running_per_engine=c) for a, b, c in w]
        for c in classes:
            ax[0, ti].axhline(BUDGET[c][1], color="r", ls="--", lw=0.8)
        ax[0, ti].set_title(title, fontsize=8)
        ax[0, ti].set_ylim(0, 130); ax[1, ti].set_xlabel("minute")
        for a in ax[:, ti]:
            a.grid(ls=":", lw=0.5)
    ax[0, 0].set_ylabel("fleet ITL (ms), 5-min window")
    ax[1, 0].set_ylabel("running requests per engine")
    # built from every panel: an arm absent from the first panel (Llumnix SLO on the mixed trace
    # while EXP-168 runs) must still be named.
    seen = {}
    for a in ax[0]:
        for h, l in zip(*a.get_legend_handles_labels()):
            seen.setdefault(l, h)
    order = [l for l, _c, _p in ARMS if l in seen]
    fig.legend([seen[l] for l in order], order, loc="lower center", ncol=7, fontsize=7, frameon=False)
    fig.suptitle("Where each policy runs the engines (repeat 1 of each arm); red dashed = class TBT budgets", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95)); fig.savefig(OUT + "operating_point.png", dpi=150); plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT + "operating_point.csv", index=False)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for t in TRACES:
        breakdown(*t)
    operating_point()
    print("wrote", OUT)
