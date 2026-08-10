#!/usr/bin/env python3
"""EXP-22 - FluidServe vs PolyServe.

Both arms run the same workload against the same engines (stock FIFO, migration
off); only the scheduler's routing policy differs. FluidServe can additionally
hold a request at the gateway when no instance can take it within budget, which
changes what "attainment" has to mean.

Denominator
-----------
The standard `served_rows` helper drops rejected requests before computing
attainment. That is the right choice when no policy can reject, which was true
of every arm up to EXP-21, but it silently rewards a policy that can: refusing
the requests that were going to miss raises the reported figure without serving
anyone better. This script therefore reports two figures and headlines the first:

  offered   every request that arrived inside the analysis window is in the
            denominator, and one that was rejected, errored or never answered
            counts as a violation. This is what an admission-controlling policy
            has to be judged by.
  served    the EXP-21 definition, kept so the numbers can be placed alongside
            the published ones.

Aggregation is equal-weight across classes. The workload offers a mix that moves
over the hour, and a fleet-wide mean weights whichever class happened to be
served most, which is exactly the composition effect EXP-21 documented.

Scoring rules are EXP-17/21's, unchanged:
  chat          mean TTFT <= 5s  and mean TBT <= 50ms
  deepresearch  mean TTFT <= 10s and mean TBT <= 100ms
  swe           end-to-end <= 30s

Usage
-----
  python3 exp22_fluidserve.py --runs results/*exp22_* --out-dir results/aggregate_analysis/exp22
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_per_engine_attainment import class_of  # noqa: E402

SLO_RULES = {
    "chat":         {"ttft": 5.0,  "tbt": 50.0},
    "deepresearch": {"ttft": 10.0, "tbt": 100.0},
    "swe":          {"e2e": 30.0},
}
CLASSES = ["chat", "deepresearch", "swe"]
CLASS_COLORS = {"chat": "#1f77b4", "deepresearch": "#ff7f0e", "swe": "#d62728"}
ARM_STYLE = {
    "polyserve":  dict(color="#d62728", ls="--", marker="o", label="PolyServe"),
    "fluidserve": dict(color="#1f77b4", ls="-", marker="s", label="FluidServe"),
    "loadbalance": dict(color="#7f7f7f", ls=":", marker="^", label="Llumnix load-balance"),
    # EXP-40: hue = control plane, lighter shade = deadline-aware engine.
    "slofifo": dict(color="#2ca02c", ls="-", marker="o", label="Llumnix SLO + FIFO"),
    "sloqoserve": dict(color="#98df8a", ls="--", marker="o", label="Llumnix SLO + QoServe"),
    "fluidservefifo": dict(color="#1f77b4", ls="-", marker="s", label="FluidServe + FIFO"),
    "fluidserveqoserve": dict(color="#aec7e8", ls="--", marker="s", label="FluidServe + QoServe"),
    # EXP-66. Brown, not orange: orange is the deep-research class colour.
    "llmdslo": dict(color="#8c564b", ls="-", marker="D", label="llm-d"),
    # EXP-67/68/69. Registering them matters more than the colours do:
    # fig_engines and make_figures build their arm list as
    # [k for k in ARM_STYLE if k in slides], so an arm that is absent here is
    # dropped from every figure without a message and the figure still renders
    # with the remaining arms and a plausible legend.
    #
    # These are the same control plane as `fluidserve` and were first given its
    # blue for that reason, which made the control and the treatment the same
    # colour on the EXP-69 figure that exists precisely to separate them. Cyan
    # and olive read as the same family, do not collide with the class colours
    # (orange is deep research), and are free in this table.
    "fspfx": dict(color="#17becf", ls="--", marker="P",
                  label="FluidServe (prefix-aware)"),
    "fspfxb": dict(color="#bcbd22", ls="-.", marker="X",
                   label="FluidServe (prefix-aware, calibration fixed)"),
    # EXP-73, the attribution ladder. Three ablations of the same v0.2 binary,
    # each one flag away from `fspfx`. Registered here before the run for the
    # reason written above this block: an arm missing from this table is dropped
    # from every figure without a message.
    "fspnopend": dict(color="#9467bd", ls="--", marker="v",
                      label="FluidServe, holding off"),
    "fspnoaff": dict(color="#e377c2", ls=":", marker="^",
                     label="FluidServe, class preference off"),
    "fspslos": dict(color="#7f7f7f", ls="-.", marker="*",
                    label="FluidServe, both off"),
    # The vLLM router's default cache_aware policy. Registered before the arm
    # runs, for the reason in the block above: an arm missing from this table is
    # dropped from every figure without a message.
    "vllmcache": dict(color="#b56349", ls="-", marker="h",
                      label="vLLM router (cache-aware)"),
}
PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "axes.linewidth": 0.75, "legend.fontsize": 8, "legend.frameon": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.4, "lines.markersize": 4.0,
}

WARMUP_S = 60.0    # the trace carries its own lead-in at the trough rate
DRAIN_S = 20.0     # requests arriving this close to the end cannot finish


def truthy(df, col):
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype(str).str.lower().isin(["true", "1", "1.0"])


LEGACY_TBT = os.environ.get("FS_LEGACY_TBT") == "1"


def mean_inter_token_ms(r, ttft, e2e):
    """Mean time between output tokens, in milliseconds, per request.

    Computed as (end-to-end time - time to first token) / (output tokens - 1)
    rather than read from the recorded `tbt_mean_ms` column, because that column
    is about half the true value on every run collected before 2026-07-30.

    How the recorded column goes wrong. The client accumulates, per streamed
    chunk, `chunk_tokens_est = max(count_tokens(chunk_text), 1)` and then charges
    the gap between two chunks as `inter_arrival_ms / chunk_tokens_est` repeated
    that many times. Tokenising a chunk in isolation is not the same as
    tokenising it in context -- a fragment that is one token inside the full
    string usually splits into two on its own, and the `max(..., 1)` floor keeps
    any sub-token chunk at one. Measured over 2,430 requests of one condition,
    the summed estimate is 1.920x the token count of the concatenated response,
    so the divisor is about twice what it should be and the reported per-token
    time about half.

    That the engine emits exactly one token per chunk is what makes the
    correction exact rather than approximate: over the same sample, chunks per
    token is 0.996, so the gap between consecutive chunks IS the gap between
    consecutive tokens, and no per-chunk estimate is needed at all. Checked
    three ways on that sample: the true per-token time from the chunk arrival
    offsets is 50.4 ms, this expression gives 50.5 ms, and the recorded column
    gives 26.1 ms. The stream span is 1.000x (end-to-end minus first token), so
    the end-to-end figure carries no trailing overhead that would inflate this.

    What it changes. The rule is time to first token AND mean time between
    tokens, so every attainment figure recorded before 2026-07-30 judged the
    per-token half of that rule against roughly twice its intended budget: 96 ms
    where chat's rule says 50, and 192 ms where deep research's says 100. The
    agent class is judged end to end and is unaffected. Set FS_LEGACY_TBT=1 to
    reproduce the earlier numbers.
    """
    if LEGACY_TBT:
        return pd.to_numeric(r["tbt_mean_ms"], errors="coerce")
    out = pd.to_numeric(r.get("output_tokens"), errors="coerce")
    span = (e2e - ttft) * 1000.0
    derived = span / (out - 1.0).where(out > 1.0)
    # A request that produced one token or none has no inter-token interval to
    # measure. It is not thereby compliant: it is caught by the missing-first-
    # token test, or it met its budget trivially.
    return derived


def load_run(run_dir):
    """All requests that arrived in the analysis window, with a violation flag.

    A request is a violation if it broke its class rule OR if it never produced
    a usable answer at all -- rejected, errored, or timed out. Requests still in
    flight when the run ended are excluded rather than failed, because their
    outcome was never determined; that exclusion applies identically to both
    arms and is reported.
    """
    p = os.path.join(run_dir, "metrics.csv")
    if not os.path.isfile(p):
        return None
    df = pd.read_csv(p, low_memory=False)
    r = df[df.agent != "job_summary"].copy()
    r = r[pd.to_numeric(r.get("start_time"), errors="coerce").notna()]
    if r.empty:
        return None

    t0 = r["start_time"].min()
    r["rel"] = r["start_time"] - t0
    end = min(r["end_time"].max() - t0, r["rel"].max())
    r = r[(r["rel"] >= WARMUP_S) & (r["rel"] < end - DRAIN_S)].copy()
    if r.empty:
        return None

    r["class"] = r["task_id"].map(class_of)
    r["rejected"] = truthy(r, "is_rejected")
    r["errored"] = truthy(r, "is_error") | truthy(r, "is_timeout")
    r["cutoff"] = truthy(r, "is_server_terminated") & ~r["rejected"] & ~r["errored"]

    ttft = pd.to_numeric(r["first_token_latency"], errors="coerce")
    e2e = pd.to_numeric(r["latency"], errors="coerce")
    tbt = mean_inter_token_ms(r, ttft, e2e)
    miss = pd.Series(False, index=r.index)
    for cname, rule in SLO_RULES.items():
        m = r["class"] == cname
        if "e2e" in rule:
            miss.loc[m] = e2e[m] > rule["e2e"]
        else:
            miss.loc[m] = (ttft[m] > rule["ttft"]) | (tbt[m] > rule["tbt"])
    # A request with no first token recorded produced nothing, which is a miss
    # under any rule rather than a missing value.
    miss = miss | (ttft.isna() & ~r["cutoff"])
    # Published so every downstream figure uses the same corrected quantity
    # instead of re-reading the tbt_mean_ms column, which is half the true
    # inter-token latency on runs collected before 2026-07-30 (see
    # fluidserve-implementation.md 32).
    r["itl_ms"] = tbt
    r["violate_served"] = miss
    r["violate_offered"] = miss | r["rejected"] | r["errored"]
    r["t0"] = t0
    return r


def attain(rows, col):
    """Attainment under one of the two denominators.

    The denominator is part of the definition, not a detail of the filter, and
    the two columns need different ones:

      violate_offered   every request that arrived is in the denominator, and a
                        rejection is a violation. This is what a policy that can
                        reject has to answer for -- refusing work is not the same
                        as doing it.
      violate_served    the denominator is the requests the system ACCEPTED, so
                        a rejection is neither a success nor a failure; it leaves
                        the population. Read on its own it rewards refusing
                        everything, which is why it is only ever reported next to
                        the rejection rate and to token goodput.

    Rejections were previously left in the denominator for BOTH columns, which
    made the served column differ from the offered one only by client errors: a
    rejected request has no first token, so `miss` was true for it either way.
    Every "served" figure recorded before 2026-07-28 is really an offered figure.

    Run-boundary cutoffs leave both denominators: their outcome is unknown.
    """
    rows = rows[~rows["cutoff"]]
    if col == "violate_served":
        rows = rows[~rows["rejected"]]
    return 100.0 * (~rows[col]).mean() if len(rows) else np.nan


def equal_mix(rows, col):
    """Attainment averaged with every CLASS weighted equally.

    Answers "is any service tier being starved". It was also the per-request
    figure for as long as the mix was 1:1:1 by request count, which is why only
    one aggregate was ever reported. It stops being that as soon as the classes
    have different volumes: on the chat-heavy mix chat is 93% of the requests and
    the other two are 4.7% and 2.3%, so this average gives a class carrying 40
    times fewer requests the same say. Report it next to per_request(), not
    instead of it.
    """
    vals = [attain(rows[rows["class"] == c], col) for c in CLASSES]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def per_request(rows, col):
    """Attainment over all requests, each counting once.

    Answers "what share of the traffic got what it was promised". Neither this
    nor equal_mix is the right one on its own: this one lets a dominant class
    hide a starved small one, and equal_mix lets a starved dominant class hide
    behind two healthy small ones. On mixes whose class volumes differ by 40x the
    two can point in opposite directions, so both are reported and any claim has
    to say which it rests on.
    """
    return attain(rows, col)


def goodput_tokens(rows, window_s, col="violate_offered"):
    """Output tokens produced by requests that met their SLO, per second.

    Judged whole-request: a request that met its rule contributes all of its
    output tokens, one that missed contributes none. Input tokens never count.
    This is EXP-17's definition, so the numbers are comparable with the engine
    scheduler arms.
    """
    ok = rows[(~rows[col]) & (~rows["cutoff"])]
    tok = pd.to_numeric(ok.get("output_tokens"), errors="coerce").fillna(0).sum()
    return float(tok) / window_s if window_s > 0 else np.nan


def total_tokens(rows, window_s):
    tok = pd.to_numeric(rows.get("output_tokens"), errors="coerce").fillna(0).sum()
    return float(tok) / window_s if window_s > 0 else np.nan


def routing_concentration(rows, run_dir):
    """How concentrated each class's requests were across the engines.

    0 means the class was spread evenly over the fleet, 1 means every request of
    that class landed on one engine. Computed as the Herfindahl index of the
    per-engine shares, rescaled so the uniform case reads 0 regardless of how
    many engines there are.

    This is the direct measure of whether a separation between classes occurred.
    EXP-21 used the same quantity to show that PolyServe's tier partition was
    actually in force; here nothing is partitioned, so any concentration is an
    outcome of the routing decisions rather than a constraint on them.
    """
    p = os.path.join(run_dir, "analysis", "request_engine.csv")
    if not os.path.isfile(p):
        return None
    try:
        m = pd.read_csv(p)[["task_id", "engine_port"]].drop_duplicates("task_id")
    except Exception:                                            # noqa: BLE001
        return None
    merged = rows.merge(m, on="task_id", how="inner")
    if merged.empty:
        return None
    engines = merged["engine_port"].nunique()
    if engines < 2:
        return None
    out = {}
    for c in CLASSES:
        sub = merged[merged["class"] == c]
        if len(sub) < 20:
            continue
        shares = sub["engine_port"].value_counts(normalize=True)
        h = float((shares ** 2).sum())
        out[c] = (h - 1.0 / engines) / (1.0 - 1.0 / engines)
    return out


def arm_of(run_dir):
    r"""The arm name out of a run directory, or the whole basename if unsure.

    The session tag is anything from `exp21` to `exp24r3` to `exp25r1`, so the
    digits and letters can alternate. An earlier pattern required the tag to end
    in letters and therefore did not match `exp25r1_`; it fell through to the
    basename, which reads as a distinct arm per run and silently turns a
    two-arm comparison into a table of one-run arms. Nothing errors, the numbers
    are all correct, and the grouping is wrong.

    [a-z]* rather than \w* in the arm itself: the latter swallows underscores
    and would match the trailing "rpm".
    """
    m = re.search(r"exp[0-9a-z]*_([a-z][a-z0-9]*)_", os.path.basename(run_dir))
    return m.group(1) if m else os.path.basename(run_dir)


def sliding(rows, key, width=60.0, step=20.0):
    """Attainment and offered rate over a sliding window, anchored on arrival."""
    lo, hi = rows["rel"].min(), rows["rel"].max()
    out = []
    t = lo
    while t + width <= hi:
        w = rows[(rows["rel"] >= t) & (rows["rel"] < t + width)]
        if len(w) >= 10:
            rec = {"t": t + width / 2, "offered_rps": len(w) / width}
            rec["eqmix"] = equal_mix(w, key)
            for c in CLASSES:
                rec[c] = attain(w[w["class"] == c], key)
            rec["goodput_tps"] = goodput_tokens(w, width, key)
            rec["tokens_tps"] = total_tokens(w, width)
            out.append(rec)
        t += step
    return pd.DataFrame(out)


def rate_binned(sl, key="eqmix", edges=(0, 15, 20, 25, 30, 35, 40, 45, 60)):
    """Recover an attainment-versus-rate curve from a single dynamic run.

    Each sliding window is one observation at the offered rate it saw, so the
    hour becomes a sweep. That is what makes a dynamic run comparable with the
    static grids of EXP-14/17/21 without repeating them.
    """
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        w = sl[(sl["offered_rps"] >= lo) & (sl["offered_rps"] < hi)]
        if len(w) < 3:
            continue
        rec = {"rate_lo": lo, "rate_hi": hi, "n_windows": len(w),
               "rate_mid": float(w["offered_rps"].mean())}
        for c in ["eqmix"] + CLASSES + ["goodput_tps", "tokens_tps"]:
            rec[c] = float(w[c].mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    run_dirs = []
    for pattern in a.runs:
        run_dirs.extend(sorted(glob.glob(pattern)))
    if not run_dirs:
        sys.exit("no runs matched")

    summary, slides = [], {}
    for d in run_dirs:
        rows = load_run(d)
        if rows is None or rows.empty:
            print(f"  skip {os.path.basename(d)}: no usable rows")
            continue
        arm = arm_of(d)
        window = rows["rel"].max() - rows["rel"].min()
        rec = {
            "run": os.path.basename(d), "arm": arm, "n": len(rows),
            "window_s": round(window, 1),
            "offered_rps": len(rows) / window,
            "eqmix_offered": equal_mix(rows, "violate_offered"),
            "eqmix_served": equal_mix(rows, "violate_served"),
            "goodput_tps": goodput_tokens(rows, window),
            "tokens_tps": total_tokens(rows, window),
            "rejected": int(rows["rejected"].sum()),
            "errored": int(rows["errored"].sum()),
            "cutoff": int(rows["cutoff"].sum()),
        }
        for c in CLASSES:
            sub = rows[rows["class"] == c]
            rec[f"{c}_offered"] = attain(sub, "violate_offered")
            rec[f"{c}_served"] = attain(sub, "violate_served")
            rec[f"n_{c}"] = len(sub)
        conc = routing_concentration(rows, d)
        if conc:
            for c, v in conc.items():
                rec[f"conc_{c}"] = v
        summary.append(rec)
        slides[arm] = sliding(rows, "violate_offered")
        slides[arm].to_csv(os.path.join(a.out_dir, f"exp22_timeseries_{arm}.csv"),
                           index=False)

    df = pd.DataFrame(summary).sort_values(["arm", "run"])
    df.to_csv(os.path.join(a.out_dir, "exp22_summary.csv"), index=False)

    print("\nSLO attainment, equal weight across classes.")
    print("offered = every arriving request counts, a reject or error is a miss.")
    print("served  = EXP-21 definition, rejects excluded from the denominator.\n")
    hdr = (f"{'arm':<12}{'req/s':>7}{'eqmix_off':>11}{'eqmix_srv':>11}"
           f"{'chat':>8}{'dr':>8}{'swe':>8}{'goodput':>10}{'total':>10}"
           f"{'rej':>7}{'err':>6}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        print(f"{r['arm']:<12}{r['offered_rps']:>7.1f}{r['eqmix_offered']:>11.1f}"
              f"{r['eqmix_served']:>11.1f}{r['chat_offered']:>8.1f}"
              f"{r['deepresearch_offered']:>8.1f}{r['swe_offered']:>8.1f}"
              f"{r['goodput_tps']:>10.0f}{r['tokens_tps']:>10.0f}"
              f"{r['rejected']:>7d}{r['errored']:>6d}")

    if any(c.startswith("conc_") for c in df.columns):
        print("\nRouting concentration (0 = spread evenly over the engines, "
              "1 = confined to one)")
        print(f"{'arm':<12}" + "".join(f"{c[:12]:>14}" for c in CLASSES))
        for _, r in df.iterrows():
            line = f"{r['arm']:<12}"
            for c in CLASSES:
                v = r.get(f"conc_{c}")
                line += f"{v:>14.3f}" if isinstance(v, float) and not np.isnan(v) else f"{'-':>14}"
            print(line)

    for arm, sl in slides.items():
        if sl.empty:
            continue
        rb = rate_binned(sl)
        rb.to_csv(os.path.join(a.out_dir, f"exp22_rate_binned_{arm}.csv"), index=False)
        print(f"\n{arm}: attainment by instantaneous offered rate "
              f"(each row is the mean over 60s windows that saw that rate)")
        print(f"  {'req/s':>8}{'n_win':>7}{'eqmix':>8}{'chat':>8}{'dr':>8}"
              f"{'swe':>8}{'goodput':>10}")
        for _, r in rb.iterrows():
            print(f"  {r['rate_mid']:>8.1f}{int(r['n_windows']):>7d}{r['eqmix']:>8.1f}"
                  f"{r['chat']:>8.1f}{r['deepresearch']:>8.1f}{r['swe']:>8.1f}"
                  f"{r['goodput_tps']:>10.0f}")

    if not a.no_figures and slides:
        make_figures(slides, df, a.out_dir)
    print(f"\nwrote {a.out_dir}")
    return 0


def make_figures(slides, summary, out_dir):
    arms = [k for k in ARM_STYLE if k in slides and not slides[k].empty]
    if not arms:
        return

    with plt.rc_context(PAPER_STYLE):
        # Time course: what each arm did as the offered rate and the mix moved.
        fig, axes = plt.subplots(3, 1, figsize=(6.6, 6.4), sharex=True)
        ax = axes[0]
        for arm in arms:
            sl = slides[arm]
            ax.plot(sl["t"] / 60, sl["offered_rps"], **{**ARM_STYLE[arm], "marker": ""})
        ax.set_ylabel("Offered rate (req/s)")
        ax.grid(axis="y", ls=":", lw=0.7, alpha=0.6)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=len(arms))

        ax = axes[1]
        for arm in arms:
            sl = slides[arm]
            ax.plot(sl["t"] / 60, sl["eqmix"], **{**ARM_STYLE[arm], "marker": ""})
        ax.set_ylabel("SLO attainment (%)")
        ax.set_ylim(-3, 105)
        ax.grid(axis="y", ls=":", lw=0.7, alpha=0.6)

        ax = axes[2]
        for arm in arms:
            sl = slides[arm]
            ax.plot(sl["t"] / 60, sl["goodput_tps"], **{**ARM_STYLE[arm], "marker": ""})
        ax.set_ylabel("Goodput (output tok/s)")
        ax.set_xlabel("Time (min)")
        ax.grid(axis="y", ls=":", lw=0.7, alpha=0.6)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "exp22_time_course.png"), dpi=300)
        plt.close(fig)

        # The hour re-expressed as a sweep, so it can be laid over the static
        # grids of the earlier experiments.
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
        for arm in arms:
            rb = rate_binned(slides[arm])
            if rb.empty:
                continue
            axes[0].plot(rb["rate_mid"], rb["eqmix"], **ARM_STYLE[arm])
            axes[1].plot(rb["rate_mid"], rb["goodput_tps"], **ARM_STYLE[arm])
        axes[0].set_xlabel("Offered rate (req/s)")
        axes[0].set_ylabel("SLO attainment (%)")
        axes[0].set_ylim(-3, 105)
        axes[1].set_xlabel("Offered rate (req/s)")
        axes[1].set_ylabel("Goodput (output tok/s)")
        for ax in axes:
            ax.grid(axis="y", ls=":", lw=0.7, alpha=0.6)
        axes[0].legend(loc="lower center", bbox_to_anchor=(1.05, 1.02), ncol=len(arms))
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "exp22_rate_binned.png"), dpi=300)
        plt.close(fig)

        # Per class, since the whole point is that the classes are not
        # interchangeable.
        fig, axes = plt.subplots(1, len(CLASSES), figsize=(7.2, 2.6), sharey=True)
        for ax, c in zip(axes, CLASSES):
            for arm in arms:
                sl = slides[arm]
                ax.plot(sl["t"] / 60, sl[c], **{**ARM_STYLE[arm], "marker": ""})
            ax.set_title(c, color=CLASS_COLORS[c])
            ax.set_xlabel("Time (min)")
            ax.grid(axis="y", ls=":", lw=0.7, alpha=0.6)
        axes[0].set_ylabel("SLO attainment (%)")
        axes[0].set_ylim(-3, 105)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "exp22_per_class.png"), dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
