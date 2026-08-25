#!/usr/bin/env python3
"""EXP-41 at the engine layer: what the four engines did over the hour.

`exp41_dynamic_timeline.py` sums the fleet, which is the right shape for reading
attainment against the offered rate and the wrong shape for two questions that
only the per-engine series can answer:

  is the fleet being used as a fleet   Four engines carrying 180 requests each
      and four engines carrying 320 / 150 / 150 / 150 produce the same fleet
      total. The second one has an engine near its memory bound while three sit
      with headroom, and the requests on the loaded engine pay a pace the fleet
      average never shows.
  where did the recompute go   vLLM V1 preempts by RECOMPUTE, so a preempted
      request discards the prefill already done and pays for it again. It still
      completes, so no request-level number attributes the cost to anything. The
      static sweep measured zero preemptions at every rate; an hour of moving
      load is where they appear, and the counter is per engine.

Two figures.

  exp41_<variant>_engine_timeline.png
      one quantity per row, one policy per column, y shared across the columns
      so a difference between columns is a difference between policies rather
      than between scales -- the same layout as exp38_policy_compare.py. Four
      coloured lines per panel, one per engine.

  exp41_<variant>_engine_requests.png
      the request side of the same question, from analysis/request_engine.csv,
      which is the scheduler's own dispatch log joined to the client ids. Note
      the denominator: a rejected request is never dispatched and so is absent
      from that file entirely, which makes per-engine attainment necessarily an
      ADMITTED-denominator quantity. It is not comparable with the offered
      numbers in the timeline figure and is labelled accordingly.

  python3 exp41_engine_view.py --variant full --out-dir <dir>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp22_fluidserve import (  # noqa: E402
    PAPER_STYLE, CLASSES, CLASS_COLORS, load_run, attain,
)

# Same dict for EXP-41 and EXP-44; an arm whose glob matches nothing is skipped.
# EXP-71. fspfx is the deployed default since v0.2 (prefix-aware prefill
# charging); the arm named `fluidserve` is its prefix-off ablation. llm-d is
# brown because orange is the deep-research class colour. An arm missing from
# this table is dropped from the figure without a message.
ARMS = {"slo": "Llumnix SLO", "fluidserve": "FluidServe (prefix off)",
        "fsa": "FluidServe + margin", "polyserve": "PolyServe",
        "loadbalance": "Llumnix",
        "fspfx": "FluidServe v0.2", "llmdslo": "llm-d",
        # EXP-93/97's class-preference arms. `fscount` ranks the feasible
        # instances by how many of the class each holds, `fsnoaff` has no class
        # term at all. Registered here rather than after the figures were drawn,
        # because an arm missing from this table is dropped in silence: unlike
        # exp38_policy_compare.py this script has no abort on an unknown arm, so
        # a run set of four came out as a figure of two and said nothing.
        "fscount": "FluidServe (pref. by count)",
        "fsnoaff": "FluidServe (class pref. off)",
        # EXP-98.
        "fscorr": "FluidServe (per-inst. corr.)",
        "fspacecap": "FluidServe (pace cap)",
        "fsboth": "FluidServe (corr. + pace cap)"}
# One colour per engine, held across every panel and both figures.
ENG_C = {8000: "#1f77b4", 8001: "#ff7f0e", 8002: "#2ca02c", 8003: "#d62728"}
WIN = 60.0  # seconds per point on the engine series

RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
KVUSED = "vllm:kv_cache_usage_perc"
PREEMPT = "vllm:num_preemptions_total"
PFX_HIT = "vllm:prefix_cache_hits_total"
PFX_Q = "vllm:prefix_cache_queries_total"


def title_slug(t):
    """A filename fragment from a free-text title.

    The title used to go straight into the path with only hyphens stripped, so a
    title naming two experiments -- "EXP-54/57", which is what the hour figures
    are called after PolyServe alone was re-measured -- produced
    'exp54/57_full_engine_timeline.png' and failed on a directory that was never
    meant to exist.

    Hyphens are still dropped rather than replaced, so "EXP-41" slugs to "exp41"
    exactly as before. Replacing them would rename every figure already on disk
    and leave the old one beside the new one, which is the stale-duplicate
    problem one step later.
    """
    out = []
    for c in t:
        if c.isalnum():
            out.append(c.lower())
        elif c != "-":
            out.append("_")
    return "".join(out).strip("_").replace("__", "_")


def pick(rec, prefix):
    """The engine series carry model-name labels, so match on the prefix."""
    for k, v in rec.items():
        if k.startswith(prefix) and isinstance(v, (int, float)):
            return float(v)
    return np.nan


def engine_series(run):
    """{port: DataFrame indexed by minute} from the scraped per-engine jsonl.

    Counters (preemptions, prefix hits/queries) are cumulative, so they are
    differenced inside each window; gauges are averaged, except KV which also
    keeps its window maximum because the memory bound is a peak property and an
    engine that touches 100% for a few seconds preempts for it.
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(run, "server_metrics", "engine_*.jsonl"))):
        port = int(os.path.basename(f).split("_")[1][:4])
        rows = []
        for line in open(f):
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if not o.get("ok"):
                continue
            rows.append((o["t"], pick(o, RUNNING), pick(o, WAITING),
                         pick(o, KVUSED) * 100, pick(o, PREEMPT),
                         pick(o, PFX_HIT), pick(o, PFX_Q)))
        if not rows:
            continue
        d = pd.DataFrame(rows, columns=["t", "run", "wait", "kv", "pre", "ph", "pq"])
        d["t"] -= d["t"].iloc[0]
        # The collector keeps scraping after the load stops; those samples are
        # all zero and would drag every mean down, so drop the drained tail.
        live = d["run"] > 0
        if live.any():
            d = d.loc[:live[::-1].idxmax()]
        g = d.groupby((d["t"] // WIN).astype(int))
        agg = pd.DataFrame({
            "run": g["run"].mean(), "wait": g["wait"].mean(),
            "kv": g["kv"].mean(), "kvmax": g["kv"].max(),
            "pre": g["pre"].max() - g["pre"].min(),
            "hit": 100.0 * (g["ph"].max() - g["ph"].min())
                   / (g["pq"].max() - g["pq"].min()).replace(0, np.nan),
        })
        agg.index = agg.index * WIN / 60.0
        out[port] = agg
    return out


def fig_timeline(data, eng, variant, out_dir, exp_title="EXP-41"):
    rows = [
        ("A", "run", "decode batch per engine", "requests", None),
        ("B", "kv", "KV occupancy per engine", "KV (%)", (0, 105)),
        ("C", "wait", "engine queue per engine — work accepted, not started", "requests", None),
        ("D", "pre", f"preemptions per engine per {WIN:.0f} s — each discards a finished prefill",
         "preemptions", None),
        ("E", "hit", "prefix cache hit rate per engine — the engine's own report on routing",
         "hit rate (%)", (0, 105)),
    ]
    arms = [a for a in ARMS if a in data]
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(len(rows), len(arms), figsize=(4.9 * len(arms), 1.85 * len(rows)),
                               sharex=True, squeeze=False)
        for j, arm in enumerate(arms):
            for i, (tag, col, title, ylab, ylim) in enumerate(rows):
                a = ax[i][j]
                for port, d in sorted(eng[arm].items()):
                    a.plot(d.index, d[col], color=ENG_C[port], lw=0.9,
                           label=f"engine {port}")
                    if col == "kv":
                        a.plot(d.index, d["kvmax"], color=ENG_C[port], lw=0.5,
                               ls=":", alpha=0.7)
                a.grid(axis="y", ls=":", lw=0.7, alpha=0.6)
                if ylim:
                    a.set_ylim(*ylim)
                if j == 0:
                    a.set_ylabel(ylab)
                if i == 0:
                    a.set_title(f"{ARMS[arm]}", fontsize=9)
                a.annotate(f"{tag}. {title}", (0.012, 0.93), xycoords="axes fraction",
                           fontsize=6.4, va="top", color="#333333")
                if variant == "full":
                    for m in (15, 30, 45):
                        a.axvline(m, color="#999999", lw=0.5, ls=":")
            ax[-1][j].set_xlabel("time (minutes)")
        # y shared across the columns of a row, set after both are drawn so the
        # limit is the union rather than whichever column was drawn last.
        for i in range(len(rows)):
            lo = min(a.get_ylim()[0] for a in ax[i])
            hi = max(a.get_ylim()[1] for a in ax[i])
            for a in ax[i]:
                a.set_ylim(lo, hi)
        h = [plt.Line2D([], [], color=ENG_C[p], lw=1.2) for p in sorted(ENG_C)]
        ax[0][0].legend(h, [f"engine {p}" for p in sorted(ENG_C)],
                        fontsize=6.2, ncol=4, loc="lower left")
        fig.suptitle(f"{exp_title} {variant} — the four engines over the hour, "
                     f"{WIN:.0f} s windows (dotted on row B is the window maximum)",
                     fontsize=9, y=1.005)
        fig.tight_layout()
        p = os.path.join(out_dir, f"{title_slug(exp_title)}_{variant}_engine_timeline.png")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {p}")


def attribute_engines(run, r):
    """Join each client request row to the engine the scheduler dispatched it to.

    `analysis/request_engine.csv` is keyed by request_id, and the client's
    metrics.csv has no request_id column, so the join has to go through
    request_ids.jsonl. Its (task_id, call_index) pair is NOT unique on a
    one-hour trace -- the same task is replayed hundreds of times, and on this
    run 95,558 of 106,116 rows share a pair with another row -- so joining on
    that pair alone multiplies the table by an order of magnitude. The
    disambiguator is start_time, matched nearest within 2 s rather than exactly
    because a request that took the non-streaming fallback records a slightly
    different one, which loses 14% of the rows on an exact match.

    Only admitted requests are matched: a rejected request is never dispatched
    and has no id, so allowing it into the left side lets the nearest-match pair
    it with a neighbour's id. Restricted this way the match is exact for every
    admitted request on both arms.
    """
    ids = pd.DataFrame([json.loads(l) for l in
                        open(os.path.join(run, "request_ids.jsonl"))])
    mp = pd.read_csv(os.path.join(run, "analysis", "request_engine.csv"))
    ids = ids.merge(mp[["request_id", "engine_port"]], on="request_id", how="left")
    r = r.copy()
    r["start_time"] = pd.to_numeric(r["start_time"], errors="coerce")
    left = r[~r["rejected"]].dropna(subset=["start_time"]).sort_values("start_time")
    right = ids.dropna(subset=["start_time"]).sort_values("start_time")
    j = pd.merge_asof(left, right[["task_id", "call_index", "start_time", "engine_port"]],
                      on="start_time", by=["task_id", "call_index"],
                      direction="nearest", tolerance=2.0)
    return j[j["engine_port"].notna()], len(left)


def fig_requests(data, runs, variant, out_dir, exp_title="EXP-41"):
    """The dispatch side: how many went where, and how they fared there."""
    per = {}
    for arm, r in data.items():
        f = os.path.join(runs[arm], "analysis", "request_engine.csv")
        if not os.path.exists(f):
            # Skip the arm rather than abandon the figure. Returning here meant
            # one arm without an attribution map cost every other arm its panel,
            # and the message named the missing file without saying the figure
            # had been dropped -- so it read as a warning and was a failure.
            print(f"  no request_engine.csv for {arm}; that arm is OMITTED "
                  f"(run build_request_engine_map.py to include it)")
            continue
        j, n = attribute_engines(runs[arm], r)
        per[arm] = j
        print(f"  {arm}: {len(j)} of {n} admitted requests attributed "
              f"({100.0 * len(j) / max(n, 1):.1f}%)")
    print(f"\n{'arm':>11} {'port':>6} {'reqs':>7} {'att-adm':>8} "
          + " ".join(f"{c[:4]+'%':>7}" for c in CLASSES) + f" {'chatITL':>8}")
    for arm, j in per.items():
        for p in sorted(ENG_C):
            e = j[j["engine_port"] == p]
            if e.empty:
                continue
            mix = " ".join(f"{100.0 * (e['class'] == c).mean():>7.1f}" for c in CLASSES)
            itl = pd.to_numeric(e[e["class"] == "chat"]["itl_ms"], errors="coerce").median()
            print(f"{arm:>11} {p:>6} {len(e):>7} "
                  f"{attain(e, 'violate_served'):>8.1f} {mix} {itl:>8.1f}")
    ports = sorted(ENG_C)
    arms = [a for a in ARMS if a in per]

    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(1, 4, figsize=(13.0, 3.0))
        w = 0.36
        x = np.arange(len(ports))
        for k, arm in enumerate(arms):
            j = per[arm]
            off = (k - (len(arms) - 1) / 2) * w
            n = [(j["engine_port"] == p).sum() for p in ports]
            ax[0].bar(x + off, n, w, color=[ENG_C[p] for p in ports],
                      alpha=1.0 if k else 0.45, edgecolor="white", lw=0.5)
            att = [attain(j[j["engine_port"] == p], "violate_served") for p in ports]
            ax[1].bar(x + off, att, w, color=[ENG_C[p] for p in ports],
                      alpha=1.0 if k else 0.45, edgecolor="white", lw=0.5)
            # class mix per engine: does the routing separate the classes at all
            bot = np.zeros(len(ports))
            for cl in CLASSES:
                sh = [100.0 * ((j["engine_port"] == p) & (j["class"] == cl)).sum()
                      / max((j["engine_port"] == p).sum(), 1) for p in ports]
                ax[2].bar(x + off, sh, w, bottom=bot, color=CLASS_COLORS[cl],
                          alpha=1.0 if k else 0.45, edgecolor="white", lw=0.4)
                bot += np.array(sh)
            itl = [pd.to_numeric(j[(j["engine_port"] == p) & (j["class"] == "chat")]["itl_ms"],
                                 errors="coerce").median() for p in ports]
            ax[3].bar(x + off, itl, w, color=[ENG_C[p] for p in ports],
                      alpha=1.0 if k else 0.45, edgecolor="white", lw=0.5)
        ax[3].axhline(50.0, color="#d62728", lw=0.8, ls=":")
        # Sits below the line, not above: above it runs into the panel title.
        ax[3].annotate("chat budget 50 ms", (-0.45, 50.0), textcoords="offset points",
                       xytext=(2, -8), fontsize=6.5, color="#d62728")
        ax[3].set_ylim(0, 58)
        titles = ["requests dispatched",
                  "attainment on that engine (ADMITTED denom.)",
                  "class mix on that engine",
                  "chat median inter-token latency"]
        ylabs = ["requests", "attainment (%)", "share of engine's requests (%)", "ms"]
        for i, (t, y) in enumerate(zip(titles, ylabs)):
            ax[i].set_title(t, fontsize=8.5)
            ax[i].set_ylabel(y)
            ax[i].set_xticks(x)
            ax[i].set_xticklabels([str(p) for p in ports])
            ax[i].set_xlabel("engine port")
            ax[i].grid(axis="y", ls=":", lw=0.7, alpha=0.6)
        for i in (1, 2):
            ax[i].set_ylim(0, 105)
        h = [plt.Rectangle((0, 0), 1, 1, fc="#666666", alpha=(1.0 if k else 0.45))
             for k in range(len(arms))]
        ax[0].legend(h, [ARMS[a] for a in arms], fontsize=6.5, loc="lower left")
        # A stacked bar fills its panel, so both legends go under the axes
        # rather than on top of the bars they describe.
        hc = [plt.Rectangle((0, 0), 1, 1, fc=CLASS_COLORS[c]) for c in CLASSES]
        ax[2].legend(hc, list(CLASSES), fontsize=6.5, ncol=3, loc="upper center",
                     bbox_to_anchor=(0.5, -0.20))
        fig.suptitle(f"{exp_title} {variant} — the dispatch side: faded bars are "
                     f"{ARMS[arms[0]]}, solid are {ARMS[arms[-1]]}", fontsize=9, y=1.04)
        fig.tight_layout()
        p = os.path.join(out_dir, f"{title_slug(exp_title)}_{variant}_engine_requests.png")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full")
    ap.add_argument("--pattern", default="results/*exp41r1_{arm}_{variant}")
    ap.add_argument("--title", default="EXP-41")
    # An aborted run leaves a directory whose name differs from the live one
    # only by its timestamp, and the arm glob matches both. sorted()[-1] picks
    # the later one, which is right by accident when the abort came first and
    # wrong when it did not, so the exclusion is stated instead of relied on.
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="substrings; any run directory containing one is dropped")
    # Naming a run outright, rather than describing it with a glob the script
    # then resolves. One --pattern cannot address a set whose arms come from
    # different sessions, and where it matches more than one run it takes
    # sorted()[-1]: `*r1_fspfx_shift*` matches exp97r1 AND exp97br1, so asking
    # for repeat 1 quietly drew repeat 2. A caller that has already decided
    # which directory it means should be able to say so.
    ap.add_argument("--run", nargs="*", default=[], metavar="ARM=DIR",
                    help="explicit run per arm; when given, --pattern is unused")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    runs = {}
    if a.run:
        for spec in a.run:
            arm, _, d = spec.partition("=")
            if not d:
                sys.exit(f"--run wants ARM=DIR, got {spec!r}")
            if not os.path.isdir(d):
                sys.exit(f"--run {arm}: no such directory {d}")
            if arm not in ARMS:
                sys.exit(f"--run {arm}: not in ARMS, so it would be dropped from "
                         f"the figure without a message. Add it to ARMS.")
            runs[arm] = d
    else:
        for arm in ARMS:
            hits = [h for h in glob.glob(a.pattern.format(arm=arm, variant=a.variant))
                    if not any(x in h for x in a.exclude)]
            if hits:
                runs[arm] = sorted(hits)[-1]
    if not runs:
        sys.exit(f"no runs for variant {a.variant}")
    print(f"variant {a.variant}: {runs}")

    data = {arm: load_run(d) for arm, d in runs.items()}
    eng = {arm: engine_series(d) for arm, d in runs.items()}

    print("\nper-engine summary")
    print(f"{'arm':>11} {'port':>6} {'batch':>7} {'KV%':>6} {'KVmax':>6} {'queue':>6} "
          f"{'preempt':>8} {'hit%':>6}")
    for arm, e in eng.items():
        for port, d in sorted(e.items()):
            print(f"{arm:>11} {port:>6} {d['run'].mean():>7.0f} {d['kv'].mean():>6.1f} "
                  f"{d['kvmax'].max():>6.0f} {d['wait'].mean():>6.1f} "
                  f"{d['pre'].sum():>8.0f} {d['hit'].mean():>6.1f}")
        b = pd.DataFrame({p: d["run"] for p, d in e.items()})
        print(f"{'':>11} {'imbalance busiest/least, per window, mean':>52} "
              f"{(b.max(axis=1) / b.min(axis=1).replace(0, np.nan)).mean():>6.2f}x")

    fig_timeline(data, eng, a.variant, a.out_dir, a.title)
    fig_requests(data, runs, a.variant, a.out_dir, a.title)


if __name__ == "__main__":
    main()
