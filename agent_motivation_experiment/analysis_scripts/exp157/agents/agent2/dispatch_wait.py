"""EXP-157 agent2: from scheduler_dispatch.log fsplacement lines: per tier, route/shed counts,
distribution of `waited` (ms the scheduler held the request before the final decision),
and the TTFT budget left at routing (ttft - waited)."""
import re, sys, numpy as np, collections
RES = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
RUNS = sys.argv[1:] or ["261109_2311_exp157hr1_fsexactprog_shift", "261110_0354_exp157hr2_fsexactprog_shift",
        "261110_0133_exp157hr1_fsc2h100_shift", "261110_0619_exp157hr2_fsc2h100_shift"]
TTFT = {50: 5000, 100: 10000, 25: 7000}
NAME = {50: "chat", 100: "deepresearch", 25: "swe"}
pat = re.compile(r"fsplacement tier=(\d+) waited=(-?\d+) prefillest=([\d.eE+-]+|\+Inf) prefillraw=([\d.]+) prompt=(\d+) decision=(\w+) inst=(\d+)(?: reason=(\w+))? prefillown=([\d.eE+-]+|\+Inf) prefillqueue=([\d.eE+-]+|\+Inf)")
for run in RUNS:
    rows = collections.defaultdict(list)
    for l in open(RES + run + "/server_metrics/scheduler_dispatch.log"):
        m = pat.search(l)
        if not m: continue
        tier, waited = int(m[1]), int(m[2])
        rows[(tier, m[6])].append((waited, float(m[3]), float(m[10]), int(m[5]), m[8]))
    print("==", run)
    for tier in (50, 100, 25):
        r = np.array([x[0] for x in rows[(tier, "route")]]); s = rows[(tier, "shed")]
        sw = np.array([x[0] for x in s]); n = len(r) + len(s)
        left = TTFT[tier] - r
        print(f"  {NAME[tier]:12s} n={n} route={len(r)} shed={len(s)} ({100*len(s)/n:.1f}%) | routed waited>0: {100*(r>0).mean():.1f}%  >1s: {100*(r>1000).mean():.1f}%  "
              f"waited p50/p90 {np.percentile(r,50):.0f}/{np.percentile(r,90):.0f} ms | TTFT left at route p10/p50 {np.percentile(left,10):.0f}/{np.percentile(left,50):.0f} ms"
              f" | of routed-after-wait(>0): TTFT left p50 {np.median(left[r>0]) if (r>0).any() else float('nan'):.0f} ms")
        if s:
            reasons = collections.Counter(x[4] for x in s)
            pq = np.array([x[2] for x in s]); pe = np.array([x[1] for x in s])
            print(f"      shed reasons {dict(reasons)}; shed waited p50 {np.median(sw):.0f}; shed prefillqueue(ms) p50/p90 {np.median(pq):.0f}/{np.percentile(pq,90):.0f}; prefillest p50 {np.median(pe):.0f}")
