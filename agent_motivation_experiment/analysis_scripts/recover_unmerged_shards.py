#!/usr/bin/env python3
"""Merge the shards of a run whose merge step never ran.

Each load process writes its own metrics/tbt/request_ids/errors/agent_logs shard
and the parent merges them into the run's standard files after the pool
finishes. If the runner dies between the last worker exiting and that merge --
the job being killed, the node evicting the pod -- the run's standard
metrics.csv is left holding only its header while the data sits in shards/,
which reads as "this run produced nothing".

That happened twice and both were recorded as failures before anyone opened the
shard directory: 260803_2229_exp54r1_loadbalance_full (363,262 rows) and
260804_1202_exp56r1_fsnoaff_full (292,854 rows).

This calls the runner's own _merge_load_shards so the result is identical to
what the run would have written itself, rather than a second implementation
that could differ in the header handling or the concatenation order.

It refuses to touch a run whose merged file already has data, so it cannot
overwrite a good result with a partial one.

  python3 recover_unmerged_shards.py [--apply] [runs...]

Without --apply it only reports what it would do.
"""
import argparse
import glob
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from run_experiment import _merge_load_shards  # noqa: E402


def rows_in(path):
    if not os.path.exists(path):
        return -1
    with open(path, encoding="utf-8", errors="replace") as f:
        return max(sum(1 for _ in f) - 1, 0)


def shard_rows(shard_dir):
    n = 0
    for p in sorted(glob.glob(os.path.join(shard_dir, "metrics.p*.csv"))):
        n += rows_in(p)
    return n


def n_shards(shard_dir):
    """The worker count, taken from the highest shard index actually present."""
    idx = []
    for p in glob.glob(os.path.join(shard_dir, "metrics.p*.csv")):
        try:
            idx.append(int(os.path.basename(p)[len("metrics.p"):-len(".csv")]))
        except ValueError:
            pass
    return max(idx) + 1 if idx else 0


def main(apply_, runs):
    if not runs:
        runs = sorted(os.path.dirname(d) for d in glob.glob("results/*/shards"))
    todo = []
    for run in runs:
        sd = os.path.join(run, "shards")
        if not os.path.isdir(sd):
            continue
        have = rows_in(os.path.join(run, "metrics.csv"))
        want = shard_rows(sd)
        if want <= 0:
            continue
        if have > 0:
            continue          # already merged; never overwrite
        todo.append((run, sd, have, want, n_shards(sd)))

    if not todo:
        print("nothing to recover: every run with shards already has a merged "
              "metrics.csv with data")
        return 0

    print(f"{'run':<48}{'merged now':>12}{'in shards':>12}{'workers':>9}")
    for run, _, have, want, n in todo:
        print(f"{os.path.basename(run):<48}{have:>12,}{want:>12,}{n:>9}")

    if not apply_:
        print("\ndry run: pass --apply to write the merged files")
        return 0

    for run, sd, _, want, n in todo:
        _merge_load_shards(sd, n,
                           os.path.join(run, "metrics.csv"),
                           os.path.join(run, "tbt_events.jsonl"),
                           os.path.join(run, "agent_logs"),
                           os.path.join(run, "errors.log"))
        got = rows_in(os.path.join(run, "metrics.csv"))
        flag = "ok" if got == want else "MISMATCH"
        print(f"{flag:>9}  {os.path.basename(run)}: {got:,} rows merged "
              f"(shards held {want:,})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("runs", nargs="*")
    a = ap.parse_args()
    sys.exit(main(a.apply, a.runs))
