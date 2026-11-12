"""EXP-157 agent2: reconstruct, at each 30 s scheduler snapshot, the scheduler's minimum non-negative ladder slack per
instance -- INCLUDING requests the engine is still queueing (the scheduler's slackTests does not exclude them,
unlike analysis_scripts/exp157/minslack.py) -- and join the nearest recorder gauge (queued prefill, correction) for
the same instance to evaluate the one-time stall test: stall(q) = corr * prefillSteps(q, 8192) * (pre(min(q,8192)) - c0).
Reports, per run, how often the test blocks (a) for any prompt (queue alone > min slack) and (b) for a typical
prefill charge of each class, and decomposes the min-slack holder's slack:
  slack = [TTFT - waited at scheduler] - [routing -> first token] + [decode surplus j*TBT - (t - first token)]."""
import json, re, sys, glob, numpy as np, pandas as pd, collections
RES = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
CACHE = "/home/nxclab/scratch/dashboard-cache/runs/"
OUT = "/home/nxclab/scratch/exp157/agent2/"
B = {"chat": (5, .050), "deepresearch": (10, .100), "swe": (7, .075)}
prof = json.load(open("/home/nxclab/llumnix_reproduce/deploy/profiling/llama31-70b-b200-tp2/ttft.json"))["results"]
PX = np.array([r["tokens_num"] for r in prof], float); PY = np.array([r["mean"] for r in prof], float)
C0 = 16.361197197677694
def stall(q, corr):
    if q <= 0: return 0.0
    s = 1.0 if q <= 8192 else q / 8192
    return corr * s * (np.interp(min(q, 8192), PX, PY) - C0)
# typical prefill charge (prompt after prefix hit, median and mean of routed requests in the c2 runs)
CHARGE = {"chat": (160, 430), "deepresearch": (2530, 3070), "swe": (1290, 1870)}
snaps = []
for f in glob.glob("/home/nxclab/scratch/exp157/snap/*.jsonl"):
    for l in open(f):
        try: snaps.append(json.loads(l))
        except Exception: pass
pat = re.compile(r"dispatch request (\S+) fsplacement tier=\d+ waited=(-?\d+) .*decision=route")
for run in sys.argv[1:]:
    ids = {}
    for l in open(RES + run + "/request_ids.jsonl"):
        i = json.loads(l); ids[(i["task_id"], i["call_index"])] = re.sub(r"^(chat)?cmpl-", "", i["request_id"])
    req = pd.read_csv(CACHE + run + "/requests.csv.gz", usecols=["task_id", "call_index", "class"])
    cls = dict(zip(zip(req.task_id, req.call_index), req["class"]))
    waited = {}
    for l in open(RES + run + "/server_metrics/scheduler_dispatch.log"):
        m = pat.search(l)
        if m: waited[m[1]] = int(m[2]) / 1000
    info = {}
    for l in open(RES + run + "/tbt_events.jsonl"):
        if '"agent": "request"' not in l: continue
        e = json.loads(l); k = (e["task_id"], e["call_index"]); rid = ids.get(k); c = cls.get(k)
        ev = e.get("chunk_events") or []
        if rid is None or c is None or not ev: continue
        info[rid] = (c, e["start_time"], np.array([e["start_time"] + x["arrival_offset_ms"] / 1000 for x in ev if x.get("arrival_offset_ms") is not None]))
    t0 = min(v[1] for v in info.values()); t1 = max(v[2][-1] for v in info.values() if len(v[2]))
    g = pd.read_csv(OUT + f"gauges_{run}.csv.gz", dtype={"inst": str})
    gi = {i: d.sort_values("t") for i, d in g.groupby("inst")}
    rows = []
    for s in snaps:
        t = s["t_ms"] / 1000
        if not (t0 + 60 <= t <= t1 - 60): continue
        for inst, recs in s["instances"].items():
            best = None; nlive = 0; nwait = 0
            for r in recs:
                v = info.get(r["id"])
                if v is None: continue
                c, st, arr = v; j = int((arr < t).sum())
                if j >= len(arr): continue
                nlive += 1; nwait += bool(r.get("waiting"))
                ttft, tbt = B[c]; sl = st + ttft + j * tbt - t
                if sl < 0: continue
                if best is None or sl < best[0]:
                    w = waited.get(r["id"], np.nan)
                    ft = arr[0] if j > 0 else np.nan
                    A = ttft - w
                    Bq = (ft - (st + w)) if j > 0 else (t - (st + w))
                    C = (j * tbt - (t - ft)) if j > 0 else 0.0
                    best = (sl, c, bool(r.get("waiting")), j, w, A, Bq, C)
            if best is None or inst not in gi: continue
            gd = gi[inst]; k = np.searchsorted(gd.t.values, t)
            k = min(max(k, 0), len(gd) - 1)
            if abs(gd.t.values[k] - t) > 2.5 and k > 0: k -= 1
            q = gd.queued_prefill_tokens.values[k]; corr = gd.instance_correction.values[k]
            if not np.isfinite(q): continue
            if not np.isfinite(corr): corr = 1.0
            d = dict(t=t, inst=inst, minslack=best[0], holder=best[1], holder_waiting=best[2], holder_j=best[3],
                     holder_waited=best[4], A=best[5], Bq=best[6], C=best[7], q=q, corr=corr, nlive=nlive, nwait=nwait,
                     stall_q=stall(q, corr) / 1000)
            for c, (med, mean) in CHARGE.items():
                d["stall_" + c] = stall(q + med, corr) / 1000
            rows.append(d)
    df = pd.DataFrame(rows); df.to_csv(OUT + f"slack_{run}.csv.gz", index=False)
    m = df.minslack
    print(f"== {run}: instance-snapshots {len(df)}")
    print(f"  min non-negative slack incl. engine-queued (s) p10/p50/p90 {m.quantile(.1):.2f}/{m.median():.2f}/{m.quantile(.9):.2f}; <1 s {100*(m<1).mean():.0f}%, <0.25 s {100*(m<.25).mean():.0f}%")
    print(f"  holder class {dict(collections.Counter(df.holder))}; holder still waiting/prefilling (j=0) {100*(df.holder_j==0).mean():.0f}%; engine-waiting {100*df.holder_waiting.mean():.0f}%")
    print(f"  queue alone blocks (stall_q > minslack): {100*(df.stall_q>m).mean():.1f}%")
    for c in CHARGE:
        print(f"  blocks a median {c} charge ({CHARGE[c][0]} tok): {100*(df['stall_'+c]>m).mean():.1f}%")
    w = df.holder_waited
    print(f"  holder waited at scheduler >0.5 s: {100*(w>0.5).mean():.0f}%, >1 s {100*(w>1).mean():.0f}% (of holders with a route line {w.notna().sum()})")
    sm = df[m < 1]
    if len(sm):
        print(f"  among min slack < 1 s (n={len(sm)}): holder j=0 {100*(sm.holder_j==0).mean():.0f}%; median A (TTFT-waited) {sm.A.median():.2f} s, median route->first token {sm[sm.holder_j>0].Bq.median():.2f} s, "
              f"median decode surplus C {sm[sm.holder_j>0].C.median():.2f} s; holder waited>0.5 s {100*(sm.holder_waited>0.5).mean():.0f}%; holder classes {dict(collections.Counter(sm.holder))}")
        dec = sm[sm.holder_j > 0]
        print(f"     decoding holders (n={len(dec)}): TTFT left after first token (A-Bq) p50 {(dec.A-dec.Bq).median():.2f} s; C<0 (ran slower than budget) {100*(dec.C<0).mean():.0f}%, C p10 {dec.C.quantile(.1):.2f} s")
