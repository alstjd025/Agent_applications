"""(1) 토큰당 시간 판정선: 요청별 '실제 평균 토큰 간격' 대 'ladder로 유도한 허용치'.
허용치 = 토큰당 예산 + (TTFT 예산 − 실제 TTFT) / (토큰 수 − 1)  — 첫 토큰 뒤 토큰 간격을 이 값 이하로 유지하면 마지막 토큰까지 마감 안.
출력: 클래스별 표본(JSON)과 요약: 실제 간격 > 명목 예산 × 0.9 인데 ladder 통과한 비율 등."""
import json, sys, re
import numpy as np, pandas as pd
R = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
C = "/home/nxclab/scratch/dashboard-cache/runs/"
B = {"chat": (5.0, 50.0), "deepresearch": (10.0, 100.0), "swe": (7.0, 75.0)}
out = {}
for run in sys.argv[1:]:
    req = pd.read_csv(C + run + "/requests.csv.gz", usecols=["task_id", "call_index", "iteration", "class", "rejected", "cutoff", "ladder_ok"])
    req = req[~req.rejected & ~req.cutoff]
    meta = {(r.task_id, r.call_index, r.iteration): (r["class"], bool(r.ladder_ok)) for _, r in req.iterrows()}
    rows = []
    for l in open(R + run + "/tbt_events.jsonl"):
        if '"agent": "request"' not in l: continue
        e = json.loads(l); m = meta.get((e["task_id"], e["call_index"], e.get("iteration", 0)))
        ev = [x["arrival_offset_ms"] for x in (e.get("chunk_events") or []) if x.get("arrival_offset_ms") is not None]
        if m is None or len(ev) < 20: continue
        c, ok = m; ttft_b, tbt = B[c]
        n = len(ev); gap = (ev[-1] - ev[0]) / (n - 1); allow = tbt + (ttft_b * 1000 - ev[0]) / (n - 1)
        rows.append((c, gap, allow, ok, ev[0]))
    df = pd.DataFrame(rows, columns=["cls", "gap", "allow", "ok", "ttft"])
    print(f"== {run[12:]}")
    for c, (ttft_b, tbt) in B.items():
        d = df[df.cls == c]
        over_gate = d.gap > 0.9 * tbt
        print(f"  {c:12s} 요청 {len(d):6d}  실제 간격 중앙 {d.gap.median():5.1f} ms (p95 {d.gap.quantile(.95):5.1f})  허용치 중앙 {d.allow.median():6.1f} ms (p5 {d.allow.quantile(.05):6.1f}) | "
              f"간격 > 예산×0.9: {100*over_gate.mean():5.1f}% 중 ladder 통과 {100*d[over_gate].ok.mean() if over_gate.any() else 0:5.1f}% | 간격 > 허용치: {100*(d.gap>d.allow).mean():4.1f}%, 그중 실패 {100*(~d[d.gap>d.allow].ok).mean() if (d.gap>d.allow).any() else 0:5.1f}%")
    smp = df.groupby("cls", group_keys=False).apply(lambda g: g.sample(min(len(g), 1500), random_state=0))
    out[run] = {c: smp[smp.cls == c][["gap", "allow", "ok"]].round(2).values.tolist() for c in B}
json.dump(out, open("/home/nxclab/scratch/exp156/ladder_bound.json", "w"))
