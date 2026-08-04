#!/usr/bin/env python3
"""EXP-56: what turning the class preference off costs, and by what mechanism.

`--fluidserve-enable-affinity=false` removes both places a class preference acts
-- the ordering of the feasible set by class share, and the class term in the
damage estimate -- and the ordering falls back to free space, which is a
load-balancing rule. Everything else is the same binary and the same budgets, so
the difference between the two arms is one mechanism rather than one system.

The three hypotheses were written before the run, in
experiments/EXP-56_affinity-ablation.md, and this script computes exactly what
each of them named:

  H1  a fully mixed fleet is worse. Loss of 5 points or more of offered
      attainment at 45 req/s confirms; under 2 points, or a gain, refutes. The
      threshold is 5 because the within-session repeat spread at 45 req/s
      reaches 4.2 points on this workload.

  H2  the mechanism is the gate rather than a homogeneous batch being faster.
      The gate reading predicts that with the preference off gate_allowance_ms
      reads 50.0 on all four instances essentially always -- the "instance with
      no chat resident" fraction goes to near zero -- and the route share falls,
      while chat's time per token need not move much. The homogeneity reading
      predicts the opposite: gate readings similar to the baseline, chat's time
      per token rising, route share not collapsing. They were pre-registered as
      mutually exclusive so the result cannot be read both ways afterwards.

  H3  the class preference is what makes the 45 req/s outcome bistable. With the
      preference off the repeats should land close together, because the
      positive feedback that amplifies an arbitrary initial imbalance is gone.
      The ablated arm being itself bimodal refutes it.

gate_allowance_ms is the smallest budget among the requests resident on an
instance. It reads 50.0 whenever any chat request is there and 61.9 or 100 when
none is, so the fraction of samples above 50.5 is the fraction of the run during
which that instance had a place for loose-budget work at all. That is the state
variable section 52 found predicts the 45 req/s outcome 24 times out of 24.

  python3 exp56_affinity.py [--runs 'results/*exp56*'] [--out <dir>]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp22_fluidserve import load_run, attain  # noqa: E402

ARMS = {"fluidserve": "FluidServe", "fsnoaff": "class preference off"}
GATE = "scheduler_fluidserve_gate_allowance_ms"
DEC = "scheduler_fluidserve_decisions_total"
INF = "scheduler_fluidserve_infeasible_total"
CHAT_BUDGET = 50.0
# Above this an instance is holding no chat request. Chat's budget is exactly
# 50.0 and the next budget up is swe's 61.9, so anything in between is noise.
NO_CHAT_ABOVE = 50.5


def scheduler_series(run):
    """Gate allowance per instance, and the decision and infeasibility counters.

    The counters are cumulative, so the run total is the last sample minus the
    first rather than the last sample: the scheduler is not restarted between
    the warm-up and the measured window on every path.
    """
    p = os.path.join(run, "server_metrics", "scheduler.jsonl")
    if not os.path.exists(p):
        return None
    gate, dec, inf = {}, {}, {}
    first, last = {}, {}
    for line in open(p):
        try:
            o = json.loads(line)
        except ValueError:
            continue
        if not o.get("ok"):
            continue
        for k, v in o.items():
            if not isinstance(v, (int, float)):
                continue
            if k.startswith(GATE + "|"):
                gate.setdefault(k.split("=")[-1], []).append(float(v))
            elif k.startswith(DEC + "|") or k.startswith(INF + "|"):
                first.setdefault(k, float(v))
                last[k] = float(v)
    for k in last:
        tag = k.split("=")[-1]
        (dec if k.startswith(DEC) else inf)[tag] = last[k] - first[k]
    if not gate:
        return None
    return dict(gate=gate, dec=dec, inf=inf)


def summarise(run):
    r = load_run(run)
    if r is None or r.empty:
        return None
    rej = r["rejected"] if "rejected" in r else pd.Series(False, index=r.index)
    served = r[~rej]
    dur = r["rel"].max()
    met = served[~served["violate_served"]]
    row = dict(
        offered=attain(r, "violate_offered"),
        admitted=attain(served, "violate_served"),
        reject=100.0 * rej.mean(),
        goodput=met["output_tokens"].sum() / dur,
    )
    a = served[served["class"] == "chat"]
    row["chat_itl"] = pd.to_numeric(a["itl_ms"], errors="coerce").median()
    row["chat_pace"] = 100.0 * row["chat_itl"] / CHAT_BUDGET

    s = scheduler_series(run)
    if s:
        # The share of gate samples, over all instances, that sit above chat's
        # budget: how much of the run had an instance where loose-budget work
        # was admissible on its own terms.
        allv = np.concatenate([np.asarray(v) for v in s["gate"].values()])
        row["no_chat_pct"] = 100.0 * float((allv > NO_CHAT_ABOVE).mean())
        row["gate_mean"] = float(allv.mean())
        tot = sum(s["dec"].values())
        for d in ("route", "pend", "shed", "force"):
            row[d] = 100.0 * s["dec"].get(d, 0.0) / tot if tot else np.nan
        itot = sum(s["inf"].values())
        for d in ("gate", "memory", "incumbents", "unpredictable"):
            row["inf_" + d] = (100.0 * s["inf"].get(d, 0.0) / itot
                               if itot else np.nan)
    return row


def main(pattern, out):
    rows = []
    for d in sorted(glob.glob(pattern)):
        m = re.search(r"_exp56r(\d)_(fluidserve|fsnoaff)_m1_rpm_(\d+)$",
                      os.path.basename(d))
        if m:
            v = summarise(d)
            if v:
                rows.append(dict(rep=int(m.group(1)), arm=m.group(2),
                                 rate=int(m.group(3)) / 60.0, kind="static", **v))
            continue
        m = re.search(r"_exp56r?\d?_(fluidserve|fsnoaff)_full$",
                      os.path.basename(d))
        if m:
            v = summarise(d)
            if v:
                rows.append(dict(rep=1, arm=m.group(1), rate=np.nan,
                                 kind="hour", **v))
    if not rows:
        sys.exit(f"no EXP-56 runs matched {pattern}")
    df = pd.DataFrame(rows)

    print("=" * 78)
    print("EXP-56 — every condition")
    print("=" * 78)
    cols = ["kind", "rate", "rep", "arm", "offered", "admitted", "reject",
            "goodput", "chat_itl", "no_chat_pct", "route", "pend", "shed",
            "force"]
    have = [c for c in cols if c in df.columns]
    print(df.sort_values(["kind", "rate", "rep", "arm"])[have]
          .to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    print("\n" + "=" * 78)
    print("H1 — is a fully mixed fleet worse? (offered attainment, paired)")
    print("     confirms at a loss >= 5.0 points, refutes under 2.0 or a gain")
    print("=" * 78)
    st = df[df.kind == "static"]
    for rate in sorted(st["rate"].dropna().unique()):
        print(f"\n  {rate:.0f} req/s")
        d = []
        for rep in sorted(st[st.rate == rate]["rep"].unique()):
            g = st[(st.rate == rate) & (st.rep == rep)]
            b = g[g.arm == "fluidserve"]["offered"]
            a = g[g.arm == "fsnoaff"]["offered"]
            if b.empty or a.empty:
                print(f"    repeat {rep}: incomplete pair")
                continue
            d.append(float(b.iloc[0] - a.iloc[0]))
            print(f"    repeat {rep}:  {b.iloc[0]:5.1f}  vs  {a.iloc[0]:5.1f}"
                  f"   -> {d[-1]:+6.1f}")
        if d:
            v = "CONFIRMS" if np.mean(d) >= 5.0 else (
                "REFUTES" if np.mean(d) < 2.0 else "between the thresholds")
            print(f"    mean loss {np.mean(d):+.1f} over {len(d)} pair(s)"
                  f"   -> H1 {v}")

    print("\n" + "=" * 78)
    print("H3 — does the ablated arm stop being bistable?")
    print("     spread over repeats at 45 req/s; the baseline's is the")
    print("     comparison, not an absolute threshold")
    print("=" * 78)
    for rate in sorted(st["rate"].dropna().unique()):
        print(f"\n  {rate:.0f} req/s")
        for arm in ARMS:
            v = st[(st.rate == rate) & (st.arm == arm)]["offered"].to_numpy()
            if len(v) < 2:
                print(f"    {ARMS[arm]:<22} {len(v)} repeat(s), spread needs 2+")
                continue
            print(f"    {ARMS[arm]:<22} " + " ".join(f"{x:5.1f}" for x in v)
                  + f"   spread {v.max()-v.min():5.1f}")

    print("\n" + "=" * 78)
    print("H2 — gate or batch homogeneity? (means over repeats)")
    print("     gate reading:        no-chat share collapses, route share falls,")
    print("                          chat's time per token roughly unchanged")
    print("     homogeneity reading: no-chat share similar, chat's time per")
    print("                          token rises, route share does not collapse")
    print("=" * 78)
    keys = ["no_chat_pct", "gate_mean", "route", "pend", "shed", "force",
            "chat_itl", "inf_gate", "inf_memory", "inf_incumbents"]
    keys = [k for k in keys if k in df.columns]
    for rate in sorted(st["rate"].dropna().unique()):
        print(f"\n  {rate:.0f} req/s")
        print(f"    {'':<22}" + "".join(f"{k[:11]:>13}" for k in keys))
        for arm in ARMS:
            g = st[(st.rate == rate) & (st.arm == arm)]
            if g.empty:
                continue
            print(f"    {ARMS[arm]:<22}"
                  + "".join(f"{g[k].mean():>13.1f}" for k in keys))

    if out:
        os.makedirs(out, exist_ok=True)
        p = os.path.join(out, "exp56_conditions.csv")
        df.to_csv(p, index=False)
        print(f"\nwrote {p}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/*exp56*")
    ap.add_argument("--out", default="results/aggregate_analysis/exp56")
    a = ap.parse_args()
    main(a.runs, a.out)
