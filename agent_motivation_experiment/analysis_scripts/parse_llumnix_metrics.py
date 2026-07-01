#!/usr/bin/env python3
"""Fold per-run Llumnix ``server_metrics/*.jsonl`` into analysis CSVs.

This is the Llumnix counterpart of ``parse_server_logs.py`` (which parses
SGLang stderr). The Llumnix collector (``llumnix_metrics.py``) writes one JSONL
time-series file per scrape target under ``<run>/server_metrics/``:

    server_metrics/engine_8000.jsonl … engine_8003.jsonl   (vllm:* per engine)
    server_metrics/scheduler.jsonl                          (scheduler_* incl.
                                                             rescheduling counter)
    server_metrics/gateway.jsonl                            (request_*/gateway_*/
                                                             instance_lrs_*/cms_*)

Outputs (under ``<run>/analysis/``):
    llumnix_server_metrics.csv          long format: target,t,rel_t,metric,value
    llumnix_server_metrics_summary.csv  per (target,metric): first/last/min/max/mean

Also prints a short human summary highlighting the migration signal
(scheduler_rescheduling_total delta over the run) and peak engine KV usage.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd


def _load_target_jsonl(path: str) -> pd.DataFrame:
    target = Path(path).stem
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            t = rec.get("t")
            ok = rec.get("ok", True)
            for k, v in rec.items():
                if k in ("t", "ok"):
                    continue
                if not isinstance(v, (int, float)):
                    continue
                rows.append({"target": target, "t": t, "ok": ok, "metric": k, "value": float(v)})
    return pd.DataFrame(rows)


def parse_run_dir(run_dir: str) -> pd.DataFrame:
    sm_dir = os.path.join(run_dir, "server_metrics")
    frames = [
        _load_target_jsonl(p)
        for p in sorted(glob.glob(os.path.join(sm_dir, "*.jsonl")))
    ]
    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    t0 = df["t"].min()
    df["rel_t"] = df["t"] - t0
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(["target", "metric"])["value"]
    summary = pd.DataFrame({
        "first": g.first(),
        "last": g.last(),
        "min": g.min(),
        "max": g.max(),
        "mean": g.mean(),
        "ticks": g.count(),
    }).reset_index()
    # counters (…_total/_sum/_count) report a delta over the run window. Match
    # on the BASE metric name (before the "|labels" suffix), else a labeled
    # series like "request_total|status_code=200" would never be detected.
    base = summary["metric"].str.split("|", n=1).str[0]
    is_counter = base.str.contains(r"(?:_total|_sum|_count)$", regex=True)
    summary["delta"] = pd.NA
    summary.loc[is_counter, "delta"] = (
        summary.loc[is_counter, "last"] - summary.loc[is_counter, "first"]
    )
    return summary


def _print_human_summary(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    if df.empty:
        print("[llumnix-metrics] no server_metrics/*.jsonl rows found")
        return
    span = df["rel_t"].max()
    print(f"[llumnix-metrics] span={span:.0f}s, targets={sorted(df['target'].unique())}")

    # migration signal (decisions counter). The scheduler exposes this metric
    # only once rescheduling has occurred; ground-truth transfers are in
    # server_metrics/migration_events.log regardless.
    resched = summary[summary["metric"].str.startswith("scheduler_rescheduling_total")]
    if not resched.empty:
        for _, r in resched.iterrows():
            d = r["delta"] if pd.notna(r["delta"]) else 0
            print(f"  rescheduling ops (decisions): +{d:.0f} over run "
                  f"(last={r['last']:.0f})")
    elif "scheduler" in set(df["target"]):
        print("  scheduler_rescheduling_total not exposed yet (0 rescheduling "
              "ops so far) — see server_metrics/migration_events.log for the "
              "rescheduling loop / any KV transfers")
    else:
        print("  rescheduling counter: scheduler target not scraped")

    # peak engine KV usage
    kv = summary[summary["metric"].str.contains("kv_cache_usage_perc")]
    for _, r in kv.iterrows():
        print(f"  {r['target']} peak KV usage: {r['max']*100:.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="+", help="Experiment run directories")
    args = ap.parse_args()

    for run_dir in args.run_dirs:
        df = parse_run_dir(run_dir)
        out_dir = Path(run_dir) / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        long_csv = out_dir / "llumnix_server_metrics.csv"
        summ_csv = out_dir / "llumnix_server_metrics_summary.csv"
        summary = build_summary(df)
        df.to_csv(long_csv, index=False)
        summary.to_csv(summ_csv, index=False)
        print(f"\n=== {run_dir} ===")
        print(f"rows: {len(df)} -> {long_csv}")
        print(f"summary -> {summ_csv}")
        _print_human_summary(df, summary)


if __name__ == "__main__":
    main()
