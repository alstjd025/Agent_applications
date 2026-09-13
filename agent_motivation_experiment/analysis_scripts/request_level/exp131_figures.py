#!/usr/bin/env python3
"""EXP-131: the ablation bars and the planning-horizon sweep, on one canvas.

Both panels are re-scored from the runs by all_arrivals_attainment.one_run
rather than transcribed from a table, so a change to the scoring reaches the
figure. The CSV written beside the PNG carries exactly what is drawn.

All conditions: Llama-3.1-70B, 4 instances x TP=2, 2100 rpm (35 req/s), 8 min,
swe promised per-token (TTFT 7 s + 75 ms/token). The control is the
fsv3capgnofrct75 cell from EXP-108 at the same rate, 2 repeats -- it is NOT
re-run, and the two EXP-114g4 runs at the same rate are excluded because their
arrival count is 16,464 against 16,860 for every other condition here.
"""
import os, sys, glob, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("FS_SWE_TBT_MS", "75")
os.environ.setdefault("FS_SWE_TTFT_S", "7")
from all_arrivals_attainment import one_run
from exp22_fluidserve import PAPER_STYLE

R = "results"

# Runs that exist and are valid files but must not enter an aggregate. Reading
# the table rather than narrowing the glob by hand: a hand-narrowed pattern
# silently stops excluding the next run that lands in the table.
EXCLUDED = set()
_tsv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "..", "..", "ms_dev", "notes", "excluded_runs.tsv")
if os.path.isfile(_tsv):
    with open(_tsv) as f:
        for line in f:
            name = line.split("\t")[0].strip()
            if name:
                EXCLUDED.add(name)
    print(f"excluded_runs.tsv: {len(EXCLUDED)} runs will be skipped")
else:
    raise SystemExit(f"excluded_runs.tsv not found at {_tsv}")

def collect(pat):
    out = []
    for d in sorted(glob.glob(os.path.join(R, pat))):
        if os.path.basename(d.rstrip("/")) in EXCLUDED:
            print(f"  -- skipped (excluded_runs.tsv): {os.path.basename(d)}")
            continue
        if not os.path.isfile(os.path.join(d, "metrics.csv")):
            continue
        try:
            r = one_run(d)
        except Exception as e:
            print(f"  !! {os.path.basename(d)}: {e}")
            continue
        if r:
            out.append(r)
    return out

# ---- panel (a): leave-one-out, and the two settings that were changed rather
#      than removed. Ordered by effect so the reader does not have to sort.
ABL = [
    ("pace gate margin\n0.90 -> 1.00", "*exp131b*gs111*"),
    ("admission control\n(rejection)",      "*exp131[rf]*fsv3t75norej_t75_rpm_2100"),
    ("class affinity",             "*exp131r[12]_fsv3t75noaff_*"),
    ("deferral (pend)",                   "*exp131r[12]_fsv3t75nopend_*"),
    ("remaining-budget\ncondition",               "*exp131r[12]_fsv3t75noinc_*"),
]
CONTROL = "*exp108r[12]_fsv3capgnofrct75_t75_rpm_2100"

# ---- panel (b): the horizon sweep. 0 is absent because the scheduler panics
#      on it by design; the nearest condition is h=1.
HOR = [(1, "*exp131j*h1_t75*"), (10, "*exp131j*h10_t75*"), (25, "*exp131h*h25_t75*"),
       (50, "*exp131e*h50_t75*"), (100, CONTROL), (250, "*exp131e*h250_t75*"),
       (500, "*exp131e*h500_t75*")]

rows_csv = []
ctl = collect(CONTROL)
assert ctl, "control runs not found"
def band(rs, k): return min(r[k] for r in rs), max(r[k] for r in rs), len(rs)

with plt.rc_context(PAPER_STYLE):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # (a) how much each element is worth, as a loss against the control band
    labels, los, his, ns = [], [], [], []
    c_lo, c_hi, c_n = band(ctl, "all_arrivals")
    for name, pat in ABL:
        rs = collect(pat)
        if not rs:
            print(f"  !! no runs for {name}"); continue
        lo, hi, n = band(rs, "all_arrivals")
        # loss against the control band: the smallest and largest it can be
        labels.append(name); los.append(c_lo - hi); his.append(c_hi - lo); ns.append(n)
        for r in rs:
            rows_csv.append(dict(panel="ablation", arm=name.replace("\n", " "),
                                 run=r["run"], **{k: r[k] for k in
                                 ("offered", "admitted", "all_arrivals", "rejected_pct", "goodput_tok_s")}))
    y = np.arange(len(labels))
    mid = [(a + b) / 2 for a, b in zip(los, his)]
    err = [[m - a for m, a in zip(mid, los)], [b - m for m, b in zip(mid, his)]]
    ax1.barh(y, mid, xerr=err, color="#1f77b4", height=0.6, error_kw=dict(lw=0.9, capsize=2))
    ax1.axvline(0, color="0.3", lw=0.8)
    ax1.set_yticks(y); ax1.set_yticklabels([f"{l}  (n={n})" for l, n in zip(labels, ns)])
    ax1.invert_yaxis()
    ax1.set_xlabel("SLO attainment lost (points, all arrivals)")
    ax1.set_title("(a) attainment lost when the element is removed", pad=6, fontsize=8)
    ax1.grid(axis="x", ls=":", lw=0.5); ax1.set_axisbelow(True)

    # (b) the horizon sweep, two denominators
    xs, a_lo, a_hi, d_lo, d_hi = [], [], [], [], []
    for h, pat in HOR:
        rs = collect(pat)
        if not rs:
            print(f"  !! no runs for h={h}"); continue
        xs.append(h)
        lo, hi, n = band(rs, "all_arrivals"); a_lo.append(lo); a_hi.append(hi)
        lo2, hi2, _ = band(rs, "admitted");   d_lo.append(lo2); d_hi.append(hi2)
        for r in rs:
            rows_csv.append(dict(panel="horizon", arm=f"h={h}", run=r["run"],
                                 **{k: r[k] for k in ("offered", "admitted", "all_arrivals",
                                                      "rejected_pct", "goodput_tok_s")}))
    am = [(a + b) / 2 for a, b in zip(a_lo, a_hi)]
    dm = [(a + b) / 2 for a, b in zip(d_lo, d_hi)]
    ax2.fill_between(xs, a_lo, a_hi, color="#1f77b4", alpha=0.20, lw=0)
    ax2.plot(xs, am, "o-", color="#1f77b4", ms=3.5, lw=1.2, label="all arrivals")
    ax2.fill_between(xs, d_lo, d_hi, color="#d62728", alpha=0.20, lw=0)
    ax2.plot(xs, dm, "s--", color="#d62728", ms=3.2, lw=1.2, label="admitted")
    ax2.axvline(100, color="0.45", lw=0.9, ls="-.")
    # The deployed value, marked on the axis itself so it cannot collide with the
    # legend the way a floating annotation did.
    ax2.text(100, 103, "deployed", fontsize=7, color="0.35", ha="center", va="bottom")
    ax2.set_xscale("log"); ax2.set_xticks(xs)
    ax2.set_xticklabels([str(x) for x in xs])
    ax2.set_xlabel("planning horizon (engine iterations)")
    ax2.set_ylabel("SLO attainment (%)")
    ax2.set_ylim(25, 110)
    ax2.set_title("(b) sweeping the planning horizon", pad=6)
    ax2.grid(axis="y", ls=":", lw=0.5); ax2.set_axisbelow(True)
    ax2.legend(frameon=False, loc="lower right", fontsize=7)

    fig.tight_layout(rect=(0, 0, 1, 1))
    out = "results/aggregate_analysis/exp131_ablation/exp131_ablation_and_horizon.png"
    fig.savefig(out, dpi=300)
    print("wrote", out)

with open(out.replace(".png", ".csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
    w.writeheader(); w.writerows(rows_csv)
print("wrote", out.replace(".png", ".csv"), f"({len(rows_csv)} rows)")
