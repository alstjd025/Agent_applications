"""EXP-157 agent2: slack_<run>.csv.gz (from slack_stall.py) by 10-minute window: share of instance-snapshots with
min slack < 1 s / < 0.25 s, and share where the stall test would block a median charge of each class; and the share of
snapshots in which every instance with residents blocks (so a decision of that class could only pend)."""
import sys, numpy as np, pandas as pd
for run in sys.argv[1:]:
    d = pd.read_csv(f"slack_{run}.csv.gz"); d["w"] = ((d.t - d.t.min()) // 600 * 10).astype(int)
    print("==", run)
    for w, x in d.groupby("w"):
        snap = x.groupby("t")
        allblk = {c: 100 * snap.apply(lambda s: (s["stall_" + c] > s.minslack).all() and len(s) == 4).mean() for c in ("chat", "deepresearch", "swe")}
        print(f"  {w:2d}-{w+10:2d} min  n={len(x):3d}  slack<1s {100*(x.minslack<1).mean():3.0f}%  <0.25s {100*(x.minslack<.25).mean():3.0f}%  "
              f"block chat/dr/swe {100*(x.stall_chat>x.minslack).mean():3.0f}/{100*(x.stall_deepresearch>x.minslack).mean():3.0f}/{100*(x.stall_swe>x.minslack).mean():3.0f}%  "
              f"all-4-blocked chat/dr/swe {allblk['chat']:.0f}/{allblk['deepresearch']:.0f}/{allblk['swe']:.0f}%")
