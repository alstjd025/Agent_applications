#!/usr/bin/env python3
"""The per-run table the paper's static figures draw, under the fixed rule.

It is EXP-108's ladder scoring with ONE cell substituted, and the substitution
is here rather than in each figure so that three figures cannot disagree about
which runs the paper shows.

THE SUBSTITUTION. EXP-108's llm-d cell at 10 req/s holds two repeats that
disagree by 42 points of rejection rate (0.0% and 42.0%) with no cause found.
EXP-110 re-measured that one arrival rate three more times and got 0.0% every
time, so the 42.0% repeat is an outlier that occurred once in five runs. The
cell is drawn from EXP-110 repeats 3 and 5 instead, which keeps n=2 as in every
other cell. The full account is in
`experiments/EXP-110_llmd-lowrate-split.md`; `exp108_ladder95.csv` still holds
the unmodified scoring of every EXP-108 run.

    python3 build_paper_ladder_table.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LADDER = os.path.join(ROOT, "results", "aggregate_analysis", "ladder95")

SUBSTITUTE_ARM = "llmdslot75"
SUBSTITUTE_RPM = "_rpm_600"
REPLACEMENTS = ["260902_0031_exp110r3_llmdslot75_t75fair_rpm_600",
                "260902_0240_exp110r5_llmdslot75_t75fair_rpm_600"]


def main():
    base = pd.read_csv(os.path.join(LADDER, "exp108_ladder95.csv"))
    repl = pd.read_csv(os.path.join(LADDER, "exp110_ladder95.csv"))
    repl = repl[repl["run"].isin(REPLACEMENTS)]
    if len(repl) != len(REPLACEMENTS):
        sys.exit(f"expected {len(REPLACEMENTS)} replacement rows, found "
                 f"{len(repl)}: {sorted(repl['run'])}")

    drop = base["run"].str.contains(SUBSTITUTE_ARM) & \
        base["run"].str.contains(SUBSTITUTE_RPM)
    if drop.sum() != 2:
        sys.exit(f"expected to drop 2 EXP-108 rows for {SUBSTITUTE_ARM} at "
                 f"{SUBSTITUTE_RPM}, matched {drop.sum()}")
    print("dropped:")
    for r in base.loc[drop, "run"]:
        print(f"    {r}")
    print("added:")
    for r in repl["run"]:
        print(f"    {r}")

    out = pd.concat([base[~drop], repl], ignore_index=True)
    path = os.path.join(LADDER, "exp108_paper_ladder95.csv")
    out.to_csv(path, index=False)
    print(f"\nwrote {path}  ({len(out)} rows, was {len(base)})")


if __name__ == "__main__":
    main()
