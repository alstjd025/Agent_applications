"""EXP-147: 남는 기록 = snapshot 시각에 목록에 있는데 클라이언트 기록상 마지막 토큰을 받은 지 2초가 넘은 요청."""
import glob, json, re, sys
import numpy as np
RES = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
SNAP = "/home/nxclab/scratch/exp157/snap/"

def ends_of(run):
    ids = {}
    for l in open(RES + run + "/request_ids.jsonl"):
        i = json.loads(l)
        ids[(i["task_id"], i["call_index"])] = re.sub(r"^(chat)?cmpl-", "", i["request_id"])
    out = {}
    for l in open(RES + run + "/tbt_events.jsonl"):
        e = json.loads(l)
        ev = e.get("chunk_events") or []
        r = ids.get((e["task_id"], e["call_index"]))
        if r and ev:
            out[r] = (e["start_time"], e["start_time"] + ev[-1]["arrival_offset_ms"] / 1000)
    return out

snaps = []
for f in glob.glob(SNAP + "*.jsonl"):
    pod = f.split("/")[-1][:-6]
    for l in open(f):
        try: s = json.loads(l)
        except Exception: continue
        snaps.append((s["t_ms"] / 1000, pod, s))
snaps.sort(key=lambda x: x[0])
for run in sys.argv[1:]:
    E = ends_of(run)
    t0 = min(a for a, _ in E.values()); t1 = max(b for _, b in E.values())
    rows = []
    for t, pod, s in snaps:
        if not (t0 + 60 <= t <= t1 - 60): continue
        reg = [r["id"] for v in s["instances"].values() for r in v]
        stale = [i for i in reg if i in E and E[i][1] + 2 < t]
        rows.append((t - t0, pod, len(reg), len(stale)))
    pods = sorted({p for _, p, _, _ in rows})
    st = np.array([r[3] for r in rows]); rg = np.array([r[2] for r in rows])
    print(f"{run}: snapshot {len(rows)} (pods {pods}) | 목록 크기 중앙값 {np.median(rg):.0f} 최대 {rg.max()} | 남는 기록 최대 {st.max()} 평균 {st.mean():.2f}, 1건 이상인 snapshot {int((st>0).sum())}")
    worst = sorted(rows, key=lambda r: -r[3])[:3]
    print("   가장 많은 시각(측정 시작 후 분):", [(round(r[0]/60,1), r[3]) for r in worst])
