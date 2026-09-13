#!/usr/bin/env python3
"""The tables behind the two class-mix figures, built once and read by both.

  results/aggregate_analysis/class_mix/static_rank_mix.csv
      per (run, engine rank, class): the median over 60 s windows of the mean
      number of requests of that class concurrently resident on the engine
      holding the k-th most chat in that window.
  results/aggregate_analysis/class_mix/static_summary.csv
      per run: attributed share, and per class the effective number of
      instances measured per window, the same measured by pooling the whole
      condition, and how many times the engine holding the most of that class
      changed. The three together say whether a whole-condition bar describes
      the condition.
  results/aggregate_analysis/class_mix/hour_engine_mix.csv
      per (run, engine, window, class) for the hour-long trace, with the engine
      also given a fixed rank by its whole-hour chat residency so that a moving
      assignment shows up as a row changing colour rather than as a reordering.

WHY THE STATIC TABLE IS RANKED PER WINDOW AND NOT POOLED. An engine's port
number means nothing: which of the four holds the tightest-budget class is an
outcome, not an identity, and it is not the same port in two repeats or in two
arms. Ranking each window by chat residency makes bar 1 "the engine holding the
most chat AT THAT MOMENT", which is comparable across repeats, arms and rates,
and it also survives the holder moving mid-condition. That movement is real even
in an 8-minute static condition: Llumnix SLO at 55 req/s reads an effective 1.88
instances for the agent class per window and 3.98 pooled, because the two
engines holding it changed six times.

    python3 build_class_mix_tables.py [--static] [--hour] [--workers 8]
"""
import argparse
import collections
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from engine_class_occupancy import (  # noqa: E402
    occupancy, occupancy_split, per_window_neff, pooled_neff, holder_changes,
    CLASSES)

LADDER = os.path.join(ROOT, "results", "aggregate_analysis", "ladder95")
OUT = os.path.join(ROOT, "results", "aggregate_analysis", "class_mix")
WINDOW = 60.0
# The hour-long trace, EXP-109, five arms. Repeat 1 of each; the second repeat
# is scored in every other figure and is used here only to check that the
# picture reproduces, which the summary table records.
HOUR_RUNS = {
    "fsv3capgnofrct75": ["260831_2015_exp109r1_fsv3capgnofrct75_shift",
                         "260901_1514_exp109r2_fsv3capgnofrct75_shift"],
    "llmdslot75": ["260831_2128_exp109r1_llmdslot75_shift",
                   "260901_1637_exp109r2_llmdslot75_shift"],
    "polyservept75": ["260831_2232_exp109r1_polyservept75_shift",
                      "260901_1742_exp109r2_polyservept75_shift"],
    "slot75": ["260831_2346_exp109r1_slot75_shift",
               "260901_1856_exp109r2_slot75_shift"],
    "vllmcachet75": ["260901_2137_exp109r1_vllmcachet75_shift",
                     "260901_2010_exp109r2_vllmcachet75_shift"],
}


def rank_mix(tab):
    """Per window, order engines by chat residency; return rank -> class -> list.

    Ties are broken by total residency so that an arm holding no chat anywhere
    still produces a stable order rather than one that follows the port number.
    """
    piv = tab.pivot_table(index=["win_start_s", "engine_port"], columns="class",
                          values="resident", aggfunc="sum", fill_value=0.0)
    for c in CLASSES:
        if c not in piv:
            piv[c] = 0.0
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for _, g in piv.groupby(level="win_start_s"):
        g = g.assign(_tot=g[list(CLASSES)].sum(axis=1))
        g = g.sort_values(["chat", "_tot"], ascending=False)
        for rank, (_, row) in enumerate(g.iterrows(), start=1):
            for c in CLASSES:
                acc[rank][c].append(float(row[c]))
    return acc


def one_static(spec):
    run, arm = spec
    path = os.path.join(ROOT, "results", run)
    try:
        tab, frac = occupancy(path, WINDOW)
    except Exception as exc:                       # noqa: BLE001
        return run, arm, None, None, f"{type(exc).__name__}: {exc}"
    if tab is None:
        return run, arm, None, None, "no attributable requests"
    rate = int(run.split("_rpm_")[1]) / 60.0
    rows = []
    for rank, per_c in sorted(rank_mix(tab).items()):
        for c in CLASSES:
            v = per_c.get(c, [])
            rows.append({"run": run, "arm": arm, "rate_req_s": rate,
                         "engine_rank": rank, "class": c,
                         "resident_median": float(np.median(v)) if v else 0.0,
                         "resident_mean": float(np.mean(v)) if v else 0.0,
                         "n_windows": len(v)})
    win, pool, ch = per_window_neff(tab), pooled_neff(tab), holder_changes(tab)
    s = {"run": run, "arm": arm, "rate_req_s": rate, "attributed_pct": 100 * frac,
         "windows": int(tab["win_start_s"].nunique())}
    for c in CLASSES:
        s[f"neff_win_{c}"] = win[c][0] if c in win else np.nan
        s[f"neff_pooled_{c}"] = pool.get(c, np.nan)
        s[f"holder_changes_{c}"] = ch.get(c, np.nan)
    return run, arm, pd.DataFrame(rows), pd.DataFrame([s]), None


def one_hour_split(spec):
    """The hour table with each class's residency split on-time / late."""
    run, arm = spec
    path = os.path.join(ROOT, "results", run)
    try:
        tab, frac = occupancy_split(path, WINDOW)
    except Exception as exc:                       # noqa: BLE001
        return run, arm, None, None, f"{type(exc).__name__}: {exc}"
    if tab is None:
        return run, arm, None, None, "no attributable requests"
    chat = tab[tab["class"] == "chat"].groupby("engine_port")["resident"].sum()
    order = list(chat.sort_values(ascending=False).index)
    for p in sorted(tab["engine_port"].unique()):
        if p not in order:
            order.append(p)
    rank = {p: i + 1 for i, p in enumerate(order)}
    t = tab.copy()
    t["run"], t["arm"] = run, arm
    t["engine_rank"] = t["engine_port"].map(rank)
    s = pd.DataFrame([{"run": run, "arm": arm, "attributed_pct": 100 * frac,
                       "late_share_pct": 100 * t["resident_late"].sum()
                       / max(t["resident"].sum(), 1e-9)}])
    return run, arm, t, s, None


def one_hour(spec):
    run, arm = spec
    path = os.path.join(ROOT, "results", run)
    try:
        tab, frac = occupancy(path, WINDOW)
    except Exception as exc:                       # noqa: BLE001
        return run, arm, None, None, f"{type(exc).__name__}: {exc}"
    if tab is None:
        return run, arm, None, None, "no attributable requests"
    # One fixed rank per engine for the whole run, by its chat residency, so a
    # row of the figure is one engine and a reassignment shows as that row
    # changing colour partway through.
    chat = (tab[tab["class"] == "chat"].groupby("engine_port")["resident"].sum())
    order = list(chat.sort_values(ascending=False).index)
    for p in sorted(tab["engine_port"].unique()):
        if p not in order:
            order.append(p)
    rank = {p: i + 1 for i, p in enumerate(order)}
    t = tab.copy()
    t["run"], t["arm"] = run, arm
    t["engine_rank"] = t["engine_port"].map(rank)
    win, pool, ch = per_window_neff(tab), pooled_neff(tab), holder_changes(tab)
    s = {"run": run, "arm": arm, "attributed_pct": 100 * frac,
         "windows": int(tab["win_start_s"].nunique())}
    for c in CLASSES:
        s[f"neff_win_{c}"] = win[c][0] if c in win else np.nan
        s[f"neff_pooled_{c}"] = pool.get(c, np.nan)
        s[f"holder_changes_{c}"] = ch.get(c, np.nan)
    return run, arm, t, pd.DataFrame([s]), None


def run_all(specs, fn, workers):
    mix, summ, bad = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for run, arm, m, s, err in ex.map(fn, specs):
            if err:
                bad.append((run, err))
                print(f"  !! {run}: {err}")
                continue
            mix.append(m)
            summ.append(s)
            print(f"  {run}  ok")
    return mix, summ, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--hour", action="store_true")
    ap.add_argument("--split", action="store_true",
                    help="the hour table with residency split on-time / late")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if not (a.static or a.hour or a.split):
        a.static = a.hour = a.split = True
    os.makedirs(OUT, exist_ok=True)

    if a.static:
        specs = []
        for f in ("exp108_paper_ladder95.csv", "vllm77_ladder95.csv"):
            d = pd.read_csv(os.path.join(LADDER, f))
            specs += list(zip(d["run"], d["arm"]))
        print(f"static: {len(specs)} runs")
        mix, summ, bad = run_all(specs, one_static, a.workers)
        pd.concat(mix, ignore_index=True).to_csv(
            os.path.join(OUT, "static_rank_mix.csv"), index=False)
        pd.concat(summ, ignore_index=True).to_csv(
            os.path.join(OUT, "static_summary.csv"), index=False)
        print(f"static: wrote {len(mix)} runs, {len(bad)} unusable")

    if a.split:
        specs = [(rs[0], arm) for arm, rs in HOUR_RUNS.items()]
        print(f"hour split: {len(specs)} runs")
        mix, summ, bad = run_all(specs, one_hour_split, a.workers)
        pd.concat(mix, ignore_index=True).to_csv(
            os.path.join(OUT, "hour_engine_mix_split.csv"), index=False)
        pd.concat(summ, ignore_index=True).to_csv(
            os.path.join(OUT, "hour_split_summary.csv"), index=False)
        print(f"hour split: wrote {len(mix)} runs, {len(bad)} unusable")

    if a.hour:
        specs = [(r, arm) for arm, rs in HOUR_RUNS.items() for r in rs]
        print(f"hour: {len(specs)} runs")
        mix, summ, bad = run_all(specs, one_hour, a.workers)
        pd.concat(mix, ignore_index=True).to_csv(
            os.path.join(OUT, "hour_engine_mix.csv"), index=False)
        pd.concat(summ, ignore_index=True).to_csv(
            os.path.join(OUT, "hour_summary.csv"), index=False)
        print(f"hour: wrote {len(mix)} runs, {len(bad)} unusable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
