"""EXP-160 사전 확인: TTFT 판정(--fluidserve-deadline-feasible)을 켰다면, 실제로 route된 순간 그 인스턴스가 불가였을까.
조건 = waited + prefillest > TTFT budget (deadlineUsesDelay=false이므로 배치 후 지연 상한은 더하지 않는다). 결정 로그의 route 줄과 채점 결과를 요청 번호로 맞춘다.
마지막 route 시도만 본다(route 줄은 요청마다 한 번). 사용: python3 ttft_check.py <run> ..."""
import json, re, sys, numpy as np, pandas as pd
RES = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
C = "/home/nxclab/scratch/dashboard-cache/runs/"
TTFT = {"50": 5000, "100": 10000, "25": 7000}; CLS = {"50": "chat", "100": "deepresearch", "25": "swe"}
rx = re.compile(r"dispatch request (\S+) fsplacement tier=(\d+) waited=(\d+) prefillest=([\d.]+).*decision=route")
for run in sys.argv[1:]:
    ids = {}
    for l in open(RES + run + "/request_ids.jsonl"):
        i = json.loads(l); ids[re.sub(r"^(chat)?cmpl-", "", i["request_id"])] = (i["task_id"], i["call_index"])
    rows = []
    for l in open(RES + run + "/server_metrics/scheduler_dispatch.log"):
        m = rx.search(l)
        if m and m.group(1) in ids:
            k = ids[m.group(1)]; rows.append((k[0], k[1], m.group(2), int(m.group(3)), float(m.group(4))))
    L = pd.DataFrame(rows, columns=["task_id", "call_index", "tier", "waited", "prefillest"]).drop_duplicates(["task_id", "call_index"], keep="last")
    q = pd.read_csv(C + run + "/requests.csv.gz"); q = q[(q.cutoff == 0) & (q.rejected == 0) & (q.errored == 0)]
    J = L.merge(q[["task_id", "call_index", "first_token_latency", "ladder_ok"]], on=["task_id", "call_index"])
    print(f"== {run[12:-6]} (route 줄과 맞춘 받은 요청 {len(J)})")
    for t, c in CLS.items():
        x = J[J.tier == t]
        if not len(x): continue
        pred = x.waited + x.prefillest; block = pred > TTFT[t]; late = x.first_token_latency * 1000 > TTFT[t]
        err = x.first_token_latency * 1000 - pred
        print(f"  {c:12s} 받음 {len(x):6d} | TTFT 초과 {late.sum():4d}건 중 판정이 막았을 것 {int((late & block).sum()):4d} | "
              f"TTFT 안이었는데 막았을 것 {int((~late & block).sum()):4d} (그중 ladder95 성공 {int((~late & block & (x.ladder_ok == 1)).sum())}) | "
              f"TTFT 초과 요청의 (실제 − 예측) 중앙값 {np.median(err[late]) / 1000 if late.any() else float('nan'):.2f}s, 예측 TTFT 중앙값 {np.median(pred[late]) / 1000 if late.any() else float('nan'):.2f}s")
