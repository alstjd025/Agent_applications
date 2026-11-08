"""2단계(가): 처리 중인 요청이 ladder95 마감에 대해 실제로 가진 여유(slack).
토큰 i의 마감 = TTFT 예산 + i x 토큰당 예산 (채점기 ladder_ok와 같은 정의, i = chunk_idx).
slack_i = 마감_i − 실제 도착_i. prefill 하나가 지연 Δ를 만들면 그 뒤 토큰이 모두 Δ만큼 밀리므로, 그 시점의 slack이 Δ보다 크면 그 요청은 그 지연을 견딘다.
출력: 클래스별로 '첫 토큰 뒤' 토큰들의 slack 분위수와, 요청 수명 중 최소 slack 분위수."""
import json, sys
import numpy as np, pandas as pd
R = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
C = "/home/nxclab/scratch/dashboard-cache/runs/"
B = {"chat": (5.0, 50.0), "deepresearch": (10.0, 100.0), "swe": (7.0, 75.0)}
for run in sys.argv[1:]:
    req = pd.read_csv(C + run + "/requests.csv.gz", usecols=["task_id", "call_index", "iteration", "class", "rejected", "cutoff"])
    req = req[~req.rejected & ~req.cutoff]
    cls = {(r.task_id, r.call_index, r.iteration): r["class"] for _, r in req.iterrows()}
    s_all = {c: [] for c in B}; s_min = {c: [] for c in B}; s_mid = {c: [] for c in B}
    for l in open(R + run + "/tbt_events.jsonl"):
        if '"agent": "request"' not in l: continue
        e = json.loads(l); c = cls.get((e["task_id"], e["call_index"], e.get("iteration", 0)))
        ev = e.get("chunk_events") or []
        if c is None or len(ev) < 10: continue
        ttft, tbt = B[c]
        sl = np.array([ttft * 1000 + x["chunk_idx"] * tbt - x["arrival_offset_ms"] for x in ev if x.get("arrival_offset_ms") is not None])
        if len(sl) < 10: continue
        s_all[c].extend(sl[1::5].tolist()); s_min[c].append(sl[1:].min()); s_mid[c].append(sl[len(sl)//2])
    print(f"== {run[12:]}  slack(ms) = 마감 − 실제 도착")
    for c in B:
        a = np.array(s_all[c]); m = np.array(s_min[c]); md = np.array(s_mid[c])
        print(f"  {c:12s} 요청 {len(m):6d} | 첫 토큰 뒤 토큰들 slack p1 {np.percentile(a,1):7.0f} p5 {np.percentile(a,5):7.0f} p50 {np.percentile(a,50):7.0f} | "
              f"수명 중 최소 slack p1 {np.percentile(m,1):7.0f} p5 {np.percentile(m,5):7.0f} p50 {np.percentile(m,50):7.0f} | 중간 시점 p5 {np.percentile(md,5):7.0f}")
