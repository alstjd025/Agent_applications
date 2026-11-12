"""EXP-157 agent3: who holds the instance's smallest non-negative ladder slack -- a request still before its
first token (TTFT slack) or one decoding -- and how long it had waited before its first token.
Same reconstruction as ../minslack.py (client records + 30 s residency snapshots); read-only."""
import json, re, sys, glob, numpy as np, pandas as pd
RES = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
CACHE = "/home/nxclab/scratch/dashboard-cache/runs/"
B = {"chat": (5, .050), "deepresearch": (10, .100), "swe": (7, .075)}
snaps = []
for f in glob.glob("/home/nxclab/scratch/exp157/snap/*.jsonl"):
    for l in open(f):
        try: snaps.append(json.loads(l))
        except Exception: pass
for run in sys.argv[1:]:
    ids = {}
    for l in open(RES + run + "/request_ids.jsonl"):
        i = json.loads(l); ids[(i["task_id"], i["call_index"])] = re.sub(r"^(chat)?cmpl-", "", i["request_id"])
    req = pd.read_csv(CACHE + run + "/requests.csv.gz", usecols=["task_id", "call_index", "class"])
    cls = {(a, b): c for a, b, c in zip(req.task_id, req.call_index, req["class"])}
    info = {}
    for l in open(RES + run + "/tbt_events.jsonl"):
        if '"agent": "request"' not in l: continue
        e = json.loads(l); k = (e["task_id"], e["call_index"]); rid = ids.get(k); c = cls.get(k)
        ev = e.get("chunk_events") or []
        if rid is None or c is None or not ev: continue
        info[rid] = (c, e["start_time"], np.array([e["start_time"] + x["arrival_offset_ms"] / 1000 for x in ev if x.get("arrival_offset_ms") is not None]))
    t0 = min(v[1] for v in info.values()); t1 = max(v[2][-1] for v in info.values() if len(v[2]))
    rows = []
    for s in snaps:
        t = s["t_ms"] / 1000
        if not (t0 + 60 <= t <= t1 - 60): continue
        for recs in s["instances"].values():
            best = None
            for r in recs:
                v = info.get(r["id"])
                if v is None or r.get("waiting"): continue
                c, st, arr = v; j = int((arr < t).sum())
                if j >= len(arr) or len(arr) == 0: continue
                ttft, tbt = B[c]; sl = st + ttft + j * tbt - t
                if sl < 0: continue
                if best is None or sl < best[0]:
                    best = (sl, c, j == 0, arr[0] - st)
            if best: rows.append(best)
    d = pd.DataFrame(rows, columns=["slack", "cls", "prefirst", "ttft"])
    lo = d[d.slack < 1.0]
    print(f"{run}: inst-snapshots {len(d)}, slack<1s {len(lo)} ({100*len(lo)/len(d):.0f}%)")
    print(f"  among slack<1s: before first token {100*lo.prefirst.mean():.0f}% | decoding {100*(~lo.prefirst).mean():.0f}%")
    dec = lo[~lo.prefirst]
    if len(dec):
        print(f"  decoding holders: class {dec.cls.value_counts().to_dict()} | their TTFT median {dec.ttft.median():.2f}s p90 {dec.ttft.quantile(.9):.2f}s")
    print(f"  all holders TTFT median {d.ttft.median():.2f}s; class of slack<1s holders {lo.cls.value_counts().to_dict()}")
