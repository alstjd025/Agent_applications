"""EXP-157 agent2: end-of-run cumulative FluidServe counters for the four h100/v0.5 runs."""
import json, sys
REC = "/home/nxclab/scratch/dashboard-cache/recorder/"
RUNS = ["261109_2311_exp157hr1_fsexactprog_shift", "261110_0354_exp157hr2_fsexactprog_shift",
        "261110_0133_exp157hr1_fsc2h100_shift", "261110_0619_exp157hr2_fsc2h100_shift"]
PFX = ("scheduler_fluidserve_decisions_total", "scheduler_fluidserve_infeasible_total",
       "scheduler_fluidserve_infeasible_sole_total", "scheduler_fluidserve_shed_reason_total",
       "scheduler_fluidserve_flux_evaluations_total", "scheduler_fluidserve_topshare")
for r in RUNS:
    last = None
    for l in open(REC + r + "/fluidserve.jsonl"):
        last = l
    d = json.loads(last)
    print("==", r)
    for k in sorted(d):
        if k.startswith(PFX) and d[k]:
            print(f"  {k.replace('scheduler_fluidserve_','')}: {d[k]:.0f}")
