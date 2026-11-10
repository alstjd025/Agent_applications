"""(2) KV 한도: 엔진 KV 사용률(물리 풀 대비)과 '다음 5초 안에 preemption이 일어날 확률'. 엔진 메트릭 1초 간격, 엔진별."""
import json, sys, glob, collections
import numpy as np
R = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
bins = np.arange(0.80, 1.0001, 0.01)
cnt = collections.Counter(); hit = collections.Counter(); amt = collections.Counter()
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
        j = np.searchsorted(ts, ts + 5.0)
        ok = j < len(ts)
        d = pr[np.minimum(j, len(ts) - 1)] - pr
        for u, dd, o in zip(kv[ok], d[ok], ok[ok]):
            b = min(int((u - 0.80) / 0.01), len(bins) - 2) if u >= 0.80 else -1
            cnt[b] += 1; hit[b] += dd > 0; amt[b] += dd
res = []
print("  KV 사용률 구간 | 표본 | 다음 5초 안 preemption 확률 | 평균 횟수")
for b in sorted(cnt):
    lo = "<0.80" if b < 0 else f"{bins[b]:.2f}-{bins[b]+0.01:.2f}"
    print(f"  {lo:>11s} | {cnt[b]:6d} | {100*hit[b]/cnt[b]:5.1f}% | {amt[b]/cnt[b]:.2f}")
    res.append((lo, cnt[b], hit[b] / cnt[b], amt[b] / cnt[b]))
json.dump(res, open("/home/nxclab/scratch/exp156/preempt_curve.json", "w"))
