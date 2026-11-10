"""(2b) 같은 데이터, 기준을 '다음 5초 동안의 최대 KV 사용률'로: 정점 예측과 비교하려면 정점 기준 곡선이 필요하다."""
import json, sys, glob, collections
import numpy as np
R = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
cnt = collections.Counter(); hit = collections.Counter()
for run in sys.argv[1:]:
    for fn in glob.glob(R + run + "/server_metrics/engine_*.jsonl"):
        ts, kv, pr = [], [], []
        for l in open(fn):
            try: x = json.loads(l)
            except Exception: continue
            u = next((v for k, v in x.items() if k.startswith("vllm:kv_cache_usage_perc|")), None)
            p = next((v for k, v in x.items() if k.startswith("vllm:num_preemptions_total|")), None)
            if u is None or p is None: continue
            ts.append(x["t"]); kv.append(u); pr.append(p)
        ts = np.array(ts); kv = np.array(kv); pr = np.array(pr)
        for i in range(0, len(ts) - 6, 5):
            j = np.searchsorted(ts, ts[i] + 5.0)
            if j >= len(ts): break
            m = kv[i:j + 1].max(); d = pr[j] - pr[i]
            b = round(min(m, 0.999) // 0.01 * 0.01, 2) if m >= 0.90 else 0.0
            cnt[b] += 1; hit[b] += d > 0
res = []
for b in sorted(cnt):
    lab = "<0.90" if b == 0 else f"{b:.2f}"
    print(f"  최대 사용률 {lab:>6s} | 표본 {cnt[b]:6d} | 5초 창 안 preemption {100*hit[b]/cnt[b]:5.1f}%")
    res.append((lab, cnt[b], hit[b] / cnt[b]))
json.dump(res, open("/home/nxclab/scratch/exp156/preempt_peak.json", "w"))
