"""EXP-157 agent2: per-instance scheduler gauges (recorder, ~2 s) -> memory-test and stall-test quantities.
For every instance-sample in the measurement window (route counter moving) with live requests:
  - which KV ceiling is lower (cap_kv = pace ceiling, cap_mem = physical ceiling)
  - forecast increment proj - obs_kv, and the 'effective' physical-occupancy limit implied for the
    current KV: safety * (1 - (proj - kv)/capMem) (fraction of the physical pool at which the
    instance can still take a zero-cost request), and the same after a typical class cost
  - stall of the already-queued prefill alone: corr * prefillSteps(q) * (pre(min(q,8192)) - c0)
Writes per-sample CSV to agent2/gauges_<run>.csv.gz."""
import json, sys, numpy as np, pandas as pd
REC = "/home/nxclab/scratch/dashboard-cache/recorder/"
OUT = "/home/nxclab/scratch/exp157/agent2/"
RUNS = {"261109_2311_exp157hr1_fsexactprog_shift": 0.95, "261110_0354_exp157hr2_fsexactprog_shift": 0.95,
        "261110_0133_exp157hr1_fsc2h100_shift": 0.98, "261110_0619_exp157hr2_fsc2h100_shift": 0.98}
prof = json.load(open("/home/nxclab/llumnix_reproduce/deploy/profiling/llama31-70b-b200-tp2/ttft.json"))["results"]
PX = np.array([r["tokens_num"] for r in prof], float); PY = np.array([r["mean"] for r in prof], float)
C0 = 16.361197197677694
def pre(x): return np.interp(x, PX, PY)
def stall(q, corr):
    q = np.asarray(q, float); s = np.where(q <= 0, 0, np.where(q <= 8192, 1, q / 8192))
    return corr * s * (pre(np.minimum(q, 8192)) - C0) * (q > 0)
G = ["obs_kv_tokens", "projected_kv_tokens", "cap_kv_tokens", "cap_mem_tokens", "queued_prefill_tokens",
     "instance_correction", "headroom_tokens", "gate_allowance_ms", "tightest_allowance_ms", "live_requests",
     "obs_decode_batch", "pace_ms", "prefill_duty", "observed_step_ms", "predicted_step_ms", "arriving_prefill_tokens",
     "waiting_residents"]
for run, safety in RUNS.items():
    recs = [json.loads(l) for l in open(REC + run + "/fluidserve.jsonl")]
    route = np.array([r.get("scheduler_fluidserve_decisions_total|decision=route", 0) for r in recs])
    mv = np.nonzero(np.diff(route) > 0)[0]
    lo, hi = mv[0] + 1, mv[-1]
    insts = sorted({k.split("instance=")[1] for r in recs[lo:hi+1] for k in r if "obs_kv_tokens|instance=" in k})
    rows = []
    for r in recs[lo:hi + 1]:
        for i in insts:
            d = {"t": r["t"], "inst": i}
            for g in G:
                d[g] = r.get(f"scheduler_fluidserve_{g}|instance={i}", np.nan)
            rows.append(d)
    df = pd.DataFrame(rows)
    df["min_t"] = (df.t - df.t.min()) / 60
    df["safety"] = safety
    df.to_csv(OUT + f"gauges_{run}.csv.gz", index=False)
    x = df[(df.live_requests > 0) & (df.cap_mem_tokens > 0)].copy()
    ck = x.cap_kv_tokens.where(x.cap_kv_tokens >= 0, np.inf)
    lim = np.minimum(ck, x.cap_mem_tokens)
    inc = x.projected_kv_tokens - x.obs_kv_tokens
    x["eff_phys"] = safety * (1 - inc / x.cap_mem_tokens)            # occupancy (fraction of physical) where memory test blocks a zero-cost request
    x["kv_over_capmem_phys"] = safety * x.obs_kv_tokens / x.cap_mem_tokens  # current physical occupancy fraction
    x["stall_q"] = stall(x.queued_prefill_tokens, x.instance_correction)
    print("==", run, f"samples {len(x)} (instances x 2 s, window {df.min_t.max():.0f} min)")
    print(f"  pace ceiling capKv < capMem: {100*(ck < x.cap_mem_tokens).mean():.1f}% of samples; capKv/capMem p10/p50/p90 "
          f"{np.nanpercentile(ck/x.cap_mem_tokens,10):.2f}/{np.nanpercentile(ck/x.cap_mem_tokens,50):.2f}/{np.nanpercentile(ck/x.cap_mem_tokens,90):.2f}")
    print(f"  capKv/obs_kv p10/p50/p90 {np.nanpercentile(ck/x.obs_kv_tokens,10):.2f}/{np.nanpercentile(ck/x.obs_kv_tokens,50):.2f}/{np.nanpercentile(ck/x.obs_kv_tokens,90):.2f};"
          f" capMem/obs_kv p10/p50/p90 {np.percentile(x.cap_mem_tokens/x.obs_kv_tokens,10):.2f}/{np.percentile(x.cap_mem_tokens/x.obs_kv_tokens,50):.2f}/{np.percentile(x.cap_mem_tokens/x.obs_kv_tokens,90):.2f}")
    print(f"  capMem tokens p50 {x.cap_mem_tokens.median():.0f}; obs_kv p50 {x.obs_kv_tokens.median():.0f}; proj-obs p10/p50/p90 {np.percentile(inc,10):.0f}/{np.percentile(inc,50):.0f}/{np.percentile(inc,90):.0f} tokens"
          f" = {100*np.percentile(inc/x.cap_mem_tokens,50):.1f}% of capMem (p90 {100*np.percentile(inc/x.cap_mem_tokens,90):.1f}%)")
    print(f"  current physical occupancy (safety*kv/capMem) p50/p90 {x.kv_over_capmem_phys.median():.3f}/{x.kv_over_capmem_phys.quantile(.9):.3f};"
          f" effective physical limit for zero-cost request p10/p50/p90 {x.eff_phys.quantile(.1):.3f}/{x.eff_phys.median():.3f}/{x.eff_phys.quantile(.9):.3f}")
    hr = lim - x.projected_kv_tokens
    print(f"  headroom min(capKv,capMem)-proj p10/p50 {np.nanpercentile(hr,10):.0f}/{np.nanpercentile(hr,50):.0f}; <0: {100*(hr<0).mean():.1f}%;"
          f" capMem-proj<0: {100*((x.cap_mem_tokens-x.projected_kv_tokens)<0).mean():.1f}%, <774 (chat cost): {100*((x.cap_mem_tokens-x.projected_kv_tokens)<774).mean():.1f}%, <4155 (dr cost) {100*((x.cap_mem_tokens-x.projected_kv_tokens)<4155).mean():.1f}%, <6912 (swe cost) {100*((x.cap_mem_tokens-x.projected_kv_tokens)<6912).mean():.1f}%")
    print(f"  queued prefill tokens p50/p90 {x.queued_prefill_tokens.median():.0f}/{x.queued_prefill_tokens.quantile(.9):.0f}; stall of queue alone p50/p90 {x.stall_q.median():.0f}/{x.stall_q.quantile(.9):.0f} ms;"
          f" >1000 ms {100*(x.stall_q>1000).mean():.1f}%, >500 ms {100*(x.stall_q>500).mean():.1f}%; corr p50 {x.instance_correction.median():.2f}")
    print(f"  gate_allowance p50 {x.gate_allowance_ms.median():.1f}; tightest_allowance p50 {x.tightest_allowance_ms.median():.1f}; observed step p50/p90 {x.observed_step_ms.median():.1f}/{x.observed_step_ms.quantile(.9):.1f}; decode batch p50 {x.obs_decode_batch.median():.0f}; prefill duty p50 {x.prefill_duty.median():.2f}")
