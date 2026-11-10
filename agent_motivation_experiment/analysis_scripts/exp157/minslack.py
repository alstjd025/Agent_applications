"""EXP-157: 인스턴스마다 처리 중 요청의 최소 비음수 ladder 여유(다음 토큰 마감 − snapshot 시각)를 클라이언트 기록으로 재구성한다."""
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
    cls = {(r.task_id, r.call_index): c for (_, r), c in zip(req[["task_id","call_index"]].iterrows(), req["class"])}
    info = {}
    for l in open(RES + run + "/tbt_events.jsonl"):
        if '"agent": "request"' not in l: continue
        e = json.loads(l); k = (e["task_id"], e["call_index"]); rid = ids.get(k); c = cls.get(k)
        ev = e.get("chunk_events") or []
        if rid is None or c is None or not ev: continue
        info[rid] = (c, e["start_time"], np.array([e["start_time"] + x["arrival_offset_ms"] / 1000 for x in ev if x.get("arrival_offset_ms") is not None]))
    t0 = min(v[1] for v in info.values()); t1 = max(v[2][-1] for v in info.values() if len(v[2]))
    mins, neg = [], 0; who = {"chat": 0, "deepresearch": 0, "swe": 0}
    for s in snaps:
        t = s["t_ms"] / 1000
        if not (t0 + 60 <= t <= t1 - 60): continue
        for recs in s["instances"].values():
            best = None
            for r in recs:
                v = info.get(r["id"])
                if v is None or r.get("waiting"): continue
                c, st, arr = v; j = int((arr < t).sum())
                if j >= len(arr): continue
                ttft, tbt = B[c]; sl = st + ttft + j * tbt - t
                if sl < 0: neg += 1; continue
                if best is None or sl < best[0]: best = (sl, c)
            if best: mins.append(best[0]); who[best[1]] += 1
    m = np.array(mins)
    print(f"{run[12:-6]}: 인스턴스-snapshot {len(m)} | 최소 비음수 여유(s) p10 {np.percentile(m,10):.2f} 중앙값 {np.median(m):.2f} p90 {np.percentile(m,90):.2f} | <0.5s {100*(m<0.5).mean():.0f}% <1s {100*(m<1).mean():.0f}% | 최소를 가진 클래스 {who} | 음수 여유 요청 {neg}")
