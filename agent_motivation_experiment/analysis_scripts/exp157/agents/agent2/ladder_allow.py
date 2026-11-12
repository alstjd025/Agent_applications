"""EXP-157 agent2: at each 30 s snapshot, the resident ladder allowance the c2 gate uses,
min over residents with slack >= 0 of (nominal + slack / remaining tokens), remaining approximated by the
request's actual remaining output tokens (the scheduler uses the class profile's expected remaining).
Also the share of instance-snapshots where that minimum is within 2 ms of a nominal budget (i.e. the
ladder allowance has collapsed back to the nominal budget), and the joined observed step (gauge)."""
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
    cls = dict(zip(zip(req.task_id, req.call_index), req["class"]))
    info = {}
    for l in open(RES + run + "/tbt_events.jsonl"):
        if '"agent": "request"' not in l: continue
        e = json.loads(l); k = (e["task_id"], e["call_index"]); rid = ids.get(k); c = cls.get(k)
        ev = e.get("chunk_events") or []
        if rid is None or c is None or not ev: continue
        info[rid] = (c, e["start_time"], np.array([e["start_time"] + x["arrival_offset_ms"] / 1000 for x in ev if x.get("arrival_offset_ms") is not None]))
    t0 = min(v[1] for v in info.values()); t1 = max(v[2][-1] for v in info.values() if len(v[2]))
    g = pd.read_csv(f"gauges_{run}.csv.gz", dtype={"inst": str}); gi = {i: d.sort_values("t") for i, d in g.groupby("inst")}
    out = []
    for s in snaps:
        t = s["t_ms"] / 1000
        if not (t0 + 60 <= t <= t1 - 60): continue
        for inst, recs in s["instances"].items():
            best = None
            for r in recs:
                v = info.get(r["id"])
                if v is None: continue
                c, st, arr = v; j = int((arr < t).sum())
                if j >= len(arr): continue
                ttft, tbt = B[c]; sl = st + ttft + j * tbt - t
                if sl < 0: continue
                a = 1000 * (tbt + sl / max(len(arr) - j, 1))
                if best is None or a < best[0]: best = (a, c, 1000 * tbt)
            if best is None or inst not in gi: continue
            gd = gi[inst]; k = min(np.searchsorted(gd.t.values, t), len(gd) - 1)
            out.append((best[0], best[1], best[0] - best[2] < 2, gd.observed_step_ms.values[k]))
    d = pd.DataFrame(out, columns=["allow", "cls", "collapsed", "step"])
    print(f"== {run}: n={len(d)} resident ladder allowance p10/p50/p90 {d.allow.quantile(.1):.1f}/{d.allow.median():.1f}/{d.allow.quantile(.9):.1f} ms;"
          f" within 2 ms of the holder's nominal budget {100*d.collapsed.mean():.0f}%; holder chat {100*(d.cls=='chat').mean():.0f}%;"
          f" observed step p50 {d.step.median():.1f} ms; step > allowance {100*(d.step > d.allow).mean():.0f}%")
