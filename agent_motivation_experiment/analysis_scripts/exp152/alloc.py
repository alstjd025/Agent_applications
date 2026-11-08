"""2단계(나): 배분. 5분 창마다 인스턴스 4대 중 허용 속도(gate)가 50 ms(chat이 있음)·75(swe가 가장 빡빡함)·100(deepresearch만)인 인스턴스 수의 평균,
그리고 그 창의 클래스별 거절률."""
import json, sys, collections
import numpy as np, pandas as pd
R = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
C = "/home/nxclab/scratch/dashboard-cache/runs/"
for run in sys.argv[1:]:
    W = collections.defaultdict(lambda: collections.Counter()); n = collections.Counter(); t0 = None
    for l in open(R + run + "/server_metrics/scheduler.jsonl"):
        x = json.loads(l)
        g = [v for k, v in x.items() if k.startswith("scheduler_fluidserve_gate_allowance_ms|") and v is not None]
        if len(g) < 4: continue
        if t0 is None: t0 = x["t"]
        w = int((x["t"] - t0) // 300); n[w] += 1
        for v in g: W[w][int(round(v)) if v > 0 else 0] += 1
    req = pd.read_csv(C + run + "/requests.csv.gz"); req = req[~req.cutoff]; req["w"] = (req.rel // 300).astype(int)
    rej = req.groupby(["w", "class"]).rejected.mean().unstack() * 100
    print(f"== {run[12:]}   창: gate50 / gate75 / gate100 인스턴스 수 | 거절 % chat / dr / swe")
    rows = []
    for w in sorted(n)[:13]:
        c = W[w]; k = n[w]
        rows.append(f"{w:2d}: {c[50]/k:3.1f}/{c[75]/k:3.1f}/{c[100]/k:3.1f} | {rej.loc[w,'chat'] if w in rej.index else float('nan'):3.0f}/{rej.loc[w,'deepresearch'] if w in rej.index else float('nan'):3.0f}/{rej.loc[w,'swe'] if w in rej.index else float('nan'):3.0f}")
    for i in range(0, len(rows), 4): print("   " + "   ".join(rows[i:i+4]))
