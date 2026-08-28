#!/usr/bin/env python3
"""EXP-41: the hour, as a timeline, for two policies on one trace.

A whole-run mean of a run whose load moves cannot answer what the run was for.
These panels put every quantity on the same time axis so a difference can be
read against the load that produced it.

  A  offered rate, the context every other panel is read against
  B  rejection rate, which is what separates the two attainment panels
  C  SLO attainment, OFFERED denominator: every request that arrived is in the
     denominator and a rejection is a violation
  D  SLO attainment, ADMITTED denominator: rejections leave the population
     entirely, so this is the quality of the work the policy chose to do
  E  attainment per class, offered
  F  attainment per class, admitted
  G  token goodput, output tokens/s from requests that met their rule
  H  what the four engines were holding, summed per policy

The two denominators are drawn side by side rather than as one line because
they answer different questions and a policy that can reject can move them in
opposite directions. Read alone, the admitted view rewards refusing everything:
a policy that rejects 80% of arrivals and serves the remainder perfectly scores
100 on D and 20 on C. Neither is the "real" number -- C is what the workload
asked for, D is how well the accepted work was done, and the rejection rate in B
is the exchange rate between them.

Requests are anchored on ARRIVAL, so a point at minute t is "of the requests
that arrived around t, what fraction met their rule" -- the queueing-theory
convention, and the one that makes B comparable with the rate in A. A request
that arrives at t and finishes at t+30s is scored at t.

  python3 exp41_dynamic_timeline.py --variant full --out-dir <dir>
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
from exp41_engine_view import title_slug  # noqa: E402

# One entry per arm that may appear. An arm whose glob matches nothing is
# skipped, so the same dict serves EXP-41 (slo vs fluidserve) and EXP-44
# (fluidserve vs fluidserve+A). Colours are the fixed policy ones where a fixed
# policy is meant; the candidate-A variant takes the same colour it has in
# exp42_figures.py so it means the same thing across experiments.
# EXP-71. fspfx is the deployed default since v0.2 (prefix-aware prefill
# charging); the arm named `fluidserve` is its prefix-off ablation. llm-d is
# brown because orange is the deep-research class colour. An arm missing from
# this table is dropped from the figure without a message.
ARMS = {"slo": ("Llumnix SLO", "#2ca02c", ":"),
        "fluidserve": ("FluidServe", "#1f77b4", "-"),
        # Dashed, not solid. Panels E and F encode the CLASS in the colour and
        # the POLICY in the line style, so two solid arms make those panels
        # unreadable however different their colours are elsewhere.
        "fsa": ("FluidServe + forced-placement margin", "#ff7f0e", "-."),
        "fspfx": ("FluidServe v0.2", "#17becf", "-"),
        "llmdslo": ("llm-d", "#8c564b", "-")}
WIN, STEP = 90.0, 30.0


def windows(r, dur):
    """Sliding windows anchored on arrival: (centre minute, rows)."""
    t = WIN / 2.0
    while t + WIN / 2.0 <= dur:
        yield t / 60.0, r[(r["rel"] >= t - WIN / 2) & (r["rel"] < t + WIN / 2)]
        t += STEP


def engine_total(run):
    """Decode batch and KV occupancy summed / averaged over the four engines."""
    ts, bt, kv = [], [], []
    for f in sorted(glob.glob(os.path.join(run, "server_metrics", "engine_*.jsonl"))):
        t, b, k = [], [], []
        for line in open(f):
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if not o.get("ok"):
                continue

            def pick(pref):
                for kk, vv in o.items():
                    if kk.startswith(pref) and isinstance(vv, (int, float)):
                        return float(vv)
                return np.nan
            t.append(o["t"]); b.append(pick("vllm:num_requests_running"))
            k.append(pick("vllm:kv_cache_usage_perc") * 100)
        if t:
            t = np.array(t) - t[0]
            ts.append(t); bt.append(np.array(b)); kv.append(np.array(k))
    if not ts:
        return None
    grid = np.arange(0, min(x.max() for x in ts), 10.0)
    B = sum(np.interp(grid, x, y) for x, y in zip(ts, bt))
    K = sum(np.interp(grid, x, y) for x, y in zip(ts, kv)) / len(ts)
    return grid / 60.0, B, K


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
    # An explicit series list, for comparisons the arm name cannot express: two
    # runs of the SAME arm that differ in something outside the policy, such as
    # the length profile, or arms drawn from different experiments. Repeatable,
    # as label|colour|linestyle|glob[|swe-rule]. When given it replaces the arm
    # registry. The optional fifth field re-scores THAT series' swe rows under
    # its own promise instead of the global SLO_RULES: "e2e:40" (seconds) or
    # "tok:7:75" (TTFT seconds, mean per-token ms). This exists because arms
    # measured under different swe promise FORMS (EXP-107T) can only share a
    # figure if each is scored by the rule it actually ran under -- and then the
    # swe columns are different quantities, which the labels must say.
    ap.add_argument("--series", nargs="*", default=[],
                    help="label|colour|linestyle|glob[|swe-rule], repeatable")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    arms = dict(ARMS)
    runs = {}
    swe_rules = {}
    if a.series:
        arms = {}
        for spec in a.series:
            parts = spec.split("|")
            if len(parts) not in (4, 5):
                sys.exit(f"bad series spec (want 4 or 5 fields): {spec!r}")
            lab, col, ls, pat = parts[:4]
            hits = [h for h in glob.glob(pat)
                    if not any(x in h for x in a.exclude)]
            if not hits:
                sys.exit(f"series {lab!r} matched nothing: {pat}")
            arms[lab] = (lab, col, ls)
            runs[lab] = sorted(hits)[-1]
            if len(parts) == 5:
                f = parts[4].split(":")
                if f[0] == "e2e" and len(f) == 2:
                    swe_rules[lab] = ("e2e", float(f[1]))
                elif f[0] == "tok" and len(f) == 3:
                    swe_rules[lab] = ("tok", float(f[1]), float(f[2]))
                else:
                    sys.exit(f"bad swe-rule {parts[4]!r} (want e2e:<s> or tok:<s>:<ms>)")
    else:
        for arm in ARMS:
            hits = [h for h in glob.glob(a.pattern.format(arm=arm, variant=a.variant))
                    if not any(x in h for x in a.exclude)]
            if hits:
                runs[arm] = sorted(hits)[-1]
    if not runs:
        sys.exit(f"no runs for variant {a.variant}")
    print(f"variant {a.variant}: {list(runs)}")

    data = {arm: load_run(d) for arm, d in runs.items()}
    # Per-series swe re-score. load_run scored every run with the global
    # SLO_RULES; a series carrying its own rule gets its swe rows re-judged
    # from the same three quantities the global rule used (first-token latency,
    # end-to-end latency, and the corrected mean inter-token time that load_run
    # publishes as itl_ms). Only the two violate_* columns move; chat and
    # deepresearch rows are untouched, so panels C/D/G mix swe rules across
    # arms exactly as the labels state.
    for arm, rule in swe_rules.items():
        r = data[arm]
        m = r["class"] == "swe"
        ttft = pd.to_numeric(r["first_token_latency"], errors="coerce")
        e2e = pd.to_numeric(r["latency"], errors="coerce")
        if rule[0] == "e2e":
            miss = e2e > rule[1]
        else:
            miss = (ttft > rule[1]) | (r["itl_ms"] > rule[2])
        miss = miss | (ttft.isna() & ~r["cutoff"])
        r.loc[m, "violate_served"] = miss[m]
        r.loc[m, "violate_offered"] = (miss[m] | r.loc[m, "rejected"]
                                       | r.loc[m, "errored"])
        print(f"swe re-scored for {arm!r}: {rule}")
    dur = min(r["rel"].max() for r in data.values())

    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(4, 2, figsize=(11.0, 10.6), sharex=True)
        ax = ax.ravel()

        for arm, r in data.items():
            lab, c, ls = arms[arm]
            x, off, adm, gp, rej = [], [], [], [], []
            per_off = {cl: [] for cl in CLASSES}
            per_adm = {cl: [] for cl in CLASSES}
            for t, g in windows(r, dur):
                if len(g) < 30:
                    continue
                x.append(t)
                off.append(attain(g, "violate_offered"))
                adm.append(attain(g, "violate_served"))
                ok = g[(~g["violate_offered"]) & (~g["cutoff"])]
                gp.append(pd.to_numeric(ok.get("output_tokens"), errors="coerce")
                          .fillna(0).sum() / WIN)
                rej.append(100.0 * g["rejected"].mean())
                for cl in CLASSES:
                    sub = g[g["class"] == cl]
                    per_off[cl].append(attain(sub, "violate_offered"))
                    per_adm[cl].append(attain(sub, "violate_served"))
            # A: the load. Drawn once per arm; they see the same trace, so the
            # two lines lying on top of each other is the check that they did.
            b = (r["rel"] // STEP).astype(int)
            # Panels where the arm is already carried by colour are drawn
            # SOLID. The registered line style is used only in E and F, where
            # colour encodes the class and the style is the only thing left to
            # separate the policies. Mixing the two conventions made a reader
            # ask whether a dashed attainment line meant a different quantity.
            ax[0].plot(np.array(sorted(b.unique())) * STEP / 60.0,
                       b.value_counts().sort_index().values / STEP,
                       color=c, lw=0.7, alpha=0.8, label=lab)
            ax[1].plot(x, rej, color=c, label=lab)
            ax[2].plot(x, off, color=c, label=lab)
            ax[3].plot(x, adm, color=c, label=lab)
            for cl in CLASSES:
                ax[4].plot(x, per_off[cl], color=CLASS_COLORS[cl], ls=ls, lw=1.1)
                ax[5].plot(x, per_adm[cl], color=CLASS_COLORS[cl], ls=ls, lw=1.1)
            ax[6].plot(x, gp, color=c, label=lab)
            eng = engine_total(runs[arm])
            if eng:
                ax[7].plot(eng[0], eng[1], color=c, label=f"{lab} batch")
                ax[7].plot(eng[0], eng[2] * 20, color=c, ls=":", lw=0.8, alpha=0.6)

        titles = [f"A. offered rate (30 s bins) — all {len(data)} arms see the same trace",
                  "B. rejection rate — what separates C from D",
                  f"C. attainment, OFFERED denominator: rejection counts as a "
                  f"violation ({WIN:.0f} s window)",
                  "D. attainment, ADMITTED denominator: rejections leave the population",
                  "E. attainment per class, OFFERED (colour = class, style = policy)",
                  "F. attainment per class, ADMITTED",
                  "G. token goodput — output tokens/s from requests that met their rule",
                  "H. fleet decode batch (solid) and mean KV % x20 (dotted)"]
        ylabs = ["req/s", "rejected (%)", "attainment (%)", "attainment (%)",
                 "attainment (%)", "attainment (%)", "tokens/s", "requests"]
        for i, (t, y) in enumerate(zip(titles, ylabs)):
            ax[i].set_title(t, fontsize=8)
            ax[i].set_ylabel(y)
            ax[i].grid(axis="y", ls=":", lw=0.7, alpha=0.6)
            ax[i].set_xlim(0, dur / 60.0)
        # The two denominators, and the two per-class panels, are only readable
        # against each other on one scale.
        for i in (2, 3, 4, 5):
            ax[i].set_ylim(0, 105)
        for i in (6, 7):
            ax[i].set_xlabel("time (minutes)")
        ax[2].legend(fontsize=7, loc="lower left")
        h = [plt.Line2D([], [], color=CLASS_COLORS[c], lw=1.1) for c in CLASSES]
        h += [plt.Line2D([], [], color="#666666", ls=arms[k][2], lw=1.1) for k in data]
        # Five entries over a panel whose lines cover the whole 0-100 range, so
        # this one legend gets a background rather than sitting on the data.
        ax[4].legend(h, list(CLASSES) + [arms[k][0] for k in data],
                     fontsize=6, ncol=3, loc="lower center", frameon=True,
                     framealpha=0.85, edgecolor="none")
        # The mix steps every 15 minutes on the compressed trace; the verbatim
        # one holds m1 throughout, so the guides would be meaningless there.
        if a.variant == "full":
            for i in range(8):
                for m in (15, 30, 45):
                    ax[i].axvline(m, color="#999999", lw=0.5, ls=":")
            ax[0].annotate("mix m1 | m2 | m3 | m1", (0.5, 0.92), xycoords="axes fraction",
                           ha="center", fontsize=6.5, color="#666666")
        # The arms present decide the title: this script now serves EXP-41
        # (two control planes) and EXP-44 (one control plane, one flag).
        who = " vs ".join(arms[k][0] for k in data)
        # Name the session each arm came from. Arms in different sessions is
        # allowed (CLAUDE.md, 2026-08-01) but a reader has to be told, because
        # the spread to judge a difference against depends on it.
        src = ", ".join(f"{arms[k][0]}: {os.path.basename(runs[k])}" for k in data)
        # Wrapped, because bbox_inches="tight" fits the CANVAS to the ink: an
        # unwrapped source line naming seven run directories once widened the
        # saved image to 8,870 px while the axes stayed at a third of that.
        import textwrap
        fig.suptitle(textwrap.fill(
            f"{a.title} {a.variant} — one hour of moving load, {who}, "
            f"stock FIFO (one run each). {src}", 150), fontsize=8.5, y=1.03)
        fig.tight_layout()
        # Shared with exp41_engine_view so the two figures of one run agree on
        # their names. A title naming two experiments -- "EXP-54/57", which is
        # what these are called after PolyServe alone was re-measured -- used to
        # put a slash in the path and fail on a directory nobody created.
        p = os.path.join(a.out_dir, f"{title_slug(a.title)}_"
                         f"{a.variant}_timeline.png")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
