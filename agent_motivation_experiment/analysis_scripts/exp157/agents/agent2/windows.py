"""EXP-157 agent2: per 5-minute window (from first routed decision): sheds by class (dispatch log), routed,
deltas of the candidate-level sole-block counters by reason (all tiers), and from the gauges the
median effective physical KV limit and the share of instance-samples with capMem - proj < 0."""
import json, re, sys, numpy as np, pandas as pd, collections, datetime
REC = "/home/nxclab/scratch/dashboard-cache/recorder/"
RES = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
OUT = "/home/nxclab/scratch/exp157/agent2/"
pat = re.compile(r"^I(\d{4} \d\d:\d\d:\d\d\.\d+).*fsplacement tier=(\d+) waited=\S+ .*decision=(\w+)")
NAME = {"50": "chat", "100": "dr", "25": "swe"}
for run in sys.argv[1:]:
    recs = [json.loads(l) for l in open(REC + run + "/fluidserve.jsonl")]
    t = np.array([r["t"] for r in recs])
    def ser(key): return np.array([r.get(key, 0.0) for r in recs])
    route = ser("scheduler_fluidserve_decisions_total|decision=route")
    tstart = t[np.nonzero(np.diff(route) > 0)[0][0]]
    year = datetime.datetime.utcfromtimestamp(tstart).year
    ev = []
    for l in open(RES + run + "/server_metrics/scheduler_dispatch.log"):
        m = pat.search(l)
        if not m: continue
        ts = datetime.datetime.strptime(f"{year}{m[1]}", "%Y%m%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp()
        ev.append((ts, NAME[m[2]], m[3]))
    ev = pd.DataFrame(ev, columns=["t", "cls", "dec"]); ev["w"] = ((ev.t - tstart) // 300).astype(int)
    reasons = ["gate", "incumbents", "memory"]
    sole = {rs: sum(ser(f"scheduler_fluidserve_infeasible_sole_total|reason={rs},tier={tr}") for tr in ("25", "50", "100")) for rs in reasons}
    pend = ser("scheduler_fluidserve_decisions_total|decision=pend")
    g = pd.read_csv(OUT + f"gauges_{run}.csv.gz"); g["w"] = ((g.t - tstart) // 300).astype(int)
    g = g[(g.live_requests > 0) & (g.cap_mem_tokens > 0)]
    g["eff"] = g.safety * (1 - (g.projected_kv_tokens - g.obs_kv_tokens) / g.cap_mem_tokens)
    g["phys"] = g.safety * g.obs_kv_tokens / g.cap_mem_tokens
    g["memblk"] = (g.cap_mem_tokens - g.projected_kv_tokens) < 0
    print("==", run)
    print("  win  n    shed%  chat% dr%  swe% | sole-block deltas gate/incumbents/memory (k) | pend(k) | phys KV p90 | eff limit p50 | capMem<proj %")
    for w in range(12):
        e = ev[ev.w == w]
        if not len(e): continue
        sh = lambda c: 100 * ((e.cls == c) & (e.dec == "shed")).sum() / max((e.cls == c).sum(), 1)
        a, b = tstart + 300 * w, tstart + 300 * (w + 1)
        ia, ib = np.searchsorted(t, a), min(np.searchsorted(t, b), len(t) - 1)
        d = [(sole[rs][ib] - sole[rs][ia]) / 1000 for rs in reasons]
        gw = g[g.w == w]
        print(f"  {w*5:3d} {len(e):5d} {100*(e.dec=='shed').mean():5.1f} {sh('chat'):5.1f} {sh('dr'):5.1f} {sh('swe'):5.1f} | {d[0]:6.1f} {d[1]:6.1f} {d[2]:6.1f} | {(pend[ib]-pend[ia])/1000:6.1f} | "
              f"{gw.phys.quantile(.9):.3f} | {gw.eff.median():.3f} | {100*gw.memblk.mean():4.1f}")
