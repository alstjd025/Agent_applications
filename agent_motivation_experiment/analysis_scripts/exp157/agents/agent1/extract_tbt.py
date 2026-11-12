"""Stream tbt_events.jsonl once per run; keep per-request summary for all classes and
chunk arrays (abs arrival time, tokens) for chat. Output: out/<run>_tbt.pkl"""
import json, sys, pickle, re, numpy as np, pandas as pd
RES = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
CACHE = "/home/nxclab/scratch/dashboard-cache/runs/"
run = sys.argv[1]
req = pd.read_csv(CACHE + run + "/requests.csv.gz", usecols=["task_id", "call_index", "class"])
cls = dict(zip(zip(req.task_id, req.call_index), req["class"]))
summ = {}; chunks = {}
with open(RES + run + "/tbt_events.jsonl") as fh:
    for l in fh:
        if '"agent": "request"' not in l: continue
        e = json.loads(l); k = (e["task_id"], e["call_index"]); c = cls.get(k)
        ev = e.get("chunk_events") or []
        t = np.array([x["arrival_offset_ms"] for x in ev if x.get("arrival_offset_ms") is not None], dtype=np.float64)
        tok = np.array([x.get("delta_tokens_est") or 0 for x in ev if x.get("arrival_offset_ms") is not None], dtype=np.int32)
        summ[k] = dict(cls=c, start=e["start_time"], end=e.get("end_time"), inp=e.get("input_tokens"),
                       out=e.get("output_tokens"), first=(t[0] / 1000 if len(t) else np.nan),
                       last=(t[-1] / 1000 if len(t) else np.nan), nchunk=len(t))
        if c == "chat" and len(t):
            chunks[k] = (t.astype(np.float32), tok.astype(np.int16))
pickle.dump(dict(summ=summ, chunks=chunks), open(f"/home/nxclab/scratch/exp157/agent1/out/{run}_tbt.pkl", "wb"))
print(run, len(summ), len(chunks))
