#!/usr/bin/env python3
"""The three denominators side by side, for any set of conditions.

EXP-78, EXP-79 and EXP-64 all judge on the same headline and none of them had a
script for it, so each one re-derived it by hand from load_run. That is the shape
of error this repository has hit most often -- one quantity written in several
places and updated in one of them -- so it lives here now.

WHY THREE COLUMNS AND NOT ONE. Every arrival ends in one of four states, and
which of them count as failures is what separates the denominators:

  met                  accepted and finished inside its class rule
  missed               accepted, finished or gave up, outside its rule
  rejected             refused at admission, so it produced nothing
  unfinished           still in flight when the load window closed, so its
                       outcome was never determined

  offered      = met / (met + missed + rejected)          unfinished dropped
  admitted     = met / (met + missed)                     rejected AND unfinished dropped
  all arrivals = met / (met + missed + rejected + unfinished)

`offered` and `admitted` are what exp22_fluidserve.attain computes and this
script calls it, so the two agree by construction rather than by coincidence.

The third exists because the first two disagree about an arm that stops
refusing. Dropping unfinished requests is right in general -- a long request that
cannot finish inside the run is not evidence of a violation -- but it is wrong at
the end of a BACKLOGGED run, where the requests still in flight are precisely the
slow ones. So an arm that refuses nothing converts its rejections into
unfinished requests and both standard denominators stop counting them. EXP-78
pre-registered the third column for exactly that reason: with admission off, the
requests that would have been rejected are still there, and the metric has to see
them.

AND WHY IT IS REPORTED WITH THE OTHER TWO, NOT INSTEAD OF THEM. On the
all-arrivals denominator a policy that refuses 36.5% of arrivals has a ceiling of
63.5%, so the column charges admission for the refusal without crediting it for
what the refusal bought. The admitted column is the one that asks whether the
system kept the promises it made, and at 45 req/s the two answer oppositely:
FluidServe 57.0% of arrivals against 89.8% of what it accepted, the same arm with
rejection off 56.2% against 56.2%. Both are true and neither alone is the result.

  python3 all_arrivals_attainment.py --runs 'results/*exp64r[12]_*'
  python3 all_arrivals_attainment.py --runs 'results/*exp78*' --by rate,arm
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp22_fluidserve import load_run, attain  # noqa: E402

# results/260811_0819_exp64col1_fsroute_m1_rpm_2100
DIRNAME = re.compile(r"^\d{6}_\d{4}_(?P<session>[^_]+)_(?P<arm>[^_]+)_"
                     r"(?P<mix>m1f?|m[23])(?:_rpm_(?P<rpm>\d+))?")


def parse_dir(path):
    m = DIRNAME.match(os.path.basename(path.rstrip("/")))
    if not m:
        # A dynamic trace carries no rate in its name. Falling back to the
        # directory name keeps it in the table instead of raising, which is what
        # silently dropped every hour-long run out of two figure scripts.
        return {"session": os.path.basename(path.rstrip("/")), "arm": "?",
                "mix": "?", "rate": np.nan}
    d = m.groupdict()
    return {"session": d["session"], "arm": d["arm"], "mix": d["mix"],
            "rate": int(d["rpm"]) / 60.0 if d["rpm"] else np.nan}


def one_run(run_dir):
    rows = load_run(run_dir)
    if rows is None or rows.empty:
        return None
    n = len(rows)
    rejected = rows["rejected"]
    cutoff = rows["cutoff"]
    # `violate_offered` is already true for a rejected request, so the met set is
    # simply the requests that violated nothing and were not cut off.
    met = (~rows["violate_offered"]) & (~cutoff)

    out = parse_dir(run_dir)
    out.update({
        "run": os.path.basename(run_dir.rstrip("/")),
        "n": n,
        "offered": attain(rows, "violate_offered"),
        "admitted": attain(rows, "violate_served"),
        "all_arrivals": 100.0 * met.sum() / n,
        "rejected_pct": 100.0 * rejected.mean(),
        "unfinished_pct": 100.0 * cutoff.mean(),
        "met_n": int(met.sum()),
    })
    # Token goodput, because the admitted column alone rewards refusing
    # everything and this is the quantity that does not.
    dur = rows["rel"].max() - rows["rel"].min()
    toks = pd.to_numeric(rows.loc[met, "output_tokens"], errors="coerce").sum()
    out["goodput_tok_s"] = toks / dur if dur > 0 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="glob over run directories")
    ap.add_argument("--by", default="rate,arm",
                    help="comma-separated grouping columns (default rate,arm)")
    ap.add_argument("--out", default=None, help="write the per-run table here")
    ap.add_argument("--since", default=None, metavar="YYMMDD",
                    help="drop runs whose directory name sorts before this. The "
                         "workload changed on 2026-08-08 and runs from either "
                         "side of that must not enter one table, so --since "
                         "260808 is the usual value; without it a glob happily "
                         "averages a policy across a workload change and the "
                         "repeat spread balloons instead of failing")
    a = ap.parse_args()

    dirs = sorted(d for d in glob.glob(a.runs) if os.path.isdir(d))
    if a.since:
        before = [d for d in dirs if os.path.basename(d) < a.since]
        dirs = [d for d in dirs if os.path.basename(d) >= a.since]
        # Loud, because a silent date filter is how a table comes out looking
        # clean while the runs it needed were dropped.
        print(f"--since {a.since}: kept {len(dirs)}, dropped {len(before)}")
    if not dirs:
        sys.exit(f"no run directories match {a.runs}")

    recs = []
    for d in dirs:
        r = one_run(d)
        if r is None:
            print(f"  skipped (no usable rows): {os.path.basename(d)}")
            continue
        recs.append(r)
    if not recs:
        sys.exit("every matched directory was empty")
    df = pd.DataFrame(recs)

    print(f"\n{len(df)} conditions from {a.runs}\n")
    cols = ["run", "n", "offered", "admitted", "all_arrivals",
            "rejected_pct", "unfinished_pct", "goodput_tok_s"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[cols].to_string(index=False,
                                 float_format=lambda v: f"{v:7.1f}"))

    by = [c for c in a.by.split(",") if c in df.columns]
    if by:
        g = df.groupby(by)
        agg = g.agg(reps=("run", "size"),
                    offered=("offered", "mean"),
                    admitted=("admitted", "mean"),
                    all_arrivals=("all_arrivals", "mean"),
                    rejected=("rejected_pct", "mean"),
                    unfinished=("unfinished_pct", "mean"),
                    goodput=("goodput_tok_s", "mean"))
        # The spread of the repeats, because a difference smaller than it is not
        # a difference. With two repeats this is the gap between two numbers and
        # not an estimate of a variance, so it is used as a floor and nothing
        # finer is read off it.
        for c in ("offered", "admitted", "all_arrivals"):
            agg[c + "_spread"] = g[c].max() - g[c].min()
        print(f"\ngrouped by {'/'.join(by)}  "
              f"(spread = max - min over repeats, a floor and not a variance)\n")
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(agg.to_string(float_format=lambda v: f"{v:7.1f}"))

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        df.to_csv(a.out, index=False)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
