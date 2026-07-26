#!/usr/bin/env python3
"""What each engine was actually doing, from the scraped per-engine series.

This exists because a whole class of failure is invisible in the request-level
numbers. A run can report a plausible attainment and a plausible goodput while
one engine holds a queue of several thousand requests and the rest sit
completely idle: the requests on the idle engines finish fine and are counted,
the ones behind the queue never finish at all and are dropped from the analysis
as "still in flight", and the aggregate looks merely mediocre rather than
broken. Reading `num_requests_running` and `num_requests_waiting` per engine over
time makes it obvious in one table.

Reports, per engine and over time: how many requests were decoding, how many
were queued behind them, and how full the KV pool was. Then three summary
figures that say whether the fleet was used as a fleet:

  peak queue depth   the largest number of requests waiting on any one engine.
                     A healthy run keeps this in the tens; thousands means the
                     router committed work to an engine that could not take it,
                     and since a dispatched request cannot be recalled, that is
                     not recoverable.
  idle-while-queued  seconds in which at least one engine had nothing to decode
                     while another had requests waiting. This is capacity that
                     existed and was not used, and it is the direct measure of a
                     routing failure as opposed to an overload.
  imbalance          the ratio of the busiest to the least busy engine, averaged
                     over the run.

Usage
-----
  python3 engine_occupancy.py --run results/<run_dir> [--step 30]
  python3 engine_occupancy.py --runs 'results/*exp22*' --summary-only
"""
import argparse
import glob
import json
import os
import sys

RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
KVUSED = "vllm:kv_cache_usage_perc"


def pick(rec, prefix):
    """The engine series carry model-name labels, so match on the prefix."""
    for k, v in rec.items():
        if k.startswith(prefix) and isinstance(v, (int, float)):
            return float(v)
    return None


def load_engines(run):
    out = {}
    for path in sorted(glob.glob(os.path.join(run, "server_metrics", "engine_*.jsonl"))):
        rows = []
        with open(path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if not rec.get("ok"):
                    continue
                rows.append((rec["t"], pick(rec, RUNNING) or 0.0,
                             pick(rec, WAITING) or 0.0, pick(rec, KVUSED) or 0.0))
        if rows:
            out[os.path.basename(path).replace("engine_", "").replace(".jsonl", "")] = rows
    return out


def summarise(engines):
    """Peak queue depth, idle-while-queued seconds, and mean imbalance."""
    names = sorted(engines)
    n = min(len(engines[e]) for e in names)
    t0 = engines[names[0]][0][0]
    peak_wait, idle_while_queued, ratios = 0.0, 0, []
    span = 0.0
    for i in range(n):
        run_i = [engines[e][i][1] for e in names]
        wait_i = [engines[e][i][2] for e in names]
        peak_wait = max(peak_wait, max(wait_i))
        # An engine with nothing decoding is free capacity. If any other engine
        # has work queued at the same instant, that capacity was reachable and
        # was not used.
        if min(run_i) == 0 and max(wait_i) > 0:
            idle_while_queued += 1
        if max(run_i) > 0:
            ratios.append(max(run_i) / max(min(run_i), 1.0))
        span = engines[names[0]][i][0] - t0
    # Samples are one per second in this collector; report the count as seconds
    # and the span alongside so the reader can check that assumption.
    return {
        "engines": len(names),
        "samples": n,
        "span_s": round(span, 1),
        "peak_queue_depth": peak_wait,
        "idle_while_queued_samples": idle_while_queued,
        "idle_while_queued_pct": round(100.0 * idle_while_queued / n, 1) if n else 0.0,
        "mean_busiest_over_least_busy": round(sum(ratios) / len(ratios), 1) if ratios else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--runs", help="glob over run directories")
    ap.add_argument("--step", type=int, default=30, help="rows to skip between printed samples")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()

    runs = []
    if a.run:
        runs.append(a.run)
    if a.runs:
        runs.extend(sorted(glob.glob(a.runs)))
    if not runs:
        sys.exit("give --run or --runs")

    for run in runs:
        engines = load_engines(run)
        if not engines:
            print(f"\n=== {os.path.basename(run)} === no per-engine series")
            continue
        names = sorted(engines)
        print(f"\n=== {os.path.basename(run)} ===")
        if not a.summary_only:
            print("   t(s) | " + " | ".join(f"{e:^18}" for e in names))
            print("        | " + " | ".join(f"{'run  wait   kv':^18}" for e in names))
            t0 = engines[names[0]][0][0]
            n = min(len(engines[e]) for e in names)
            for i in range(0, n, a.step):
                cells = []
                for e in names:
                    _, r, w, kv = engines[e][i]
                    cells.append(f"{r:4.0f} {w:5.0f} {100 * kv:4.0f}%")
                print(f"  {engines[names[0]][i][0] - t0:5.0f} | " + " | ".join(cells))
        s = summarise(engines)
        print(f"\n  engines {s['engines']}, {s['samples']} samples over {s['span_s']:.0f}s")
        print(f"  peak queue depth on one engine   {s['peak_queue_depth']:>8,.0f}")
        print(f"  an engine idle while another had work queued  "
              f"{s['idle_while_queued_pct']:>5.1f}% of samples "
              f"({s['idle_while_queued_samples']:,})")
        print(f"  mean busiest / least busy        {s['mean_busiest_over_least_busy']:>8.1f}x")
        if s["peak_queue_depth"] > 500:
            print("  -> a queue this deep is work the router committed to an engine that "
                  "could not take it. A dispatched request cannot be recalled, so this is "
                  "not something later decisions can undo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
