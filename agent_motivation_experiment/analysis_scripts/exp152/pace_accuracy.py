"""1단계: 토큰당 시간 추정(meanStepMs)의 분모 h별 예측을, 그 뒤 실제로 걸린 iteration 시간과 비교한다.

입력: run의 server_metrics/scheduler.jsonl (1초 간격, 인스턴스별 게이지).
예측 재계산은 fluidserve_capacity.go meanStepMs와 fluidserve.go buildFlux의 식 그대로:
  arriving = duty * (h * pace) / (pre(chunk) - c0) * chunk ;  eff = max(queued, arriving)
  sp = 1 if 0 < eff <= chunk else eff/chunk ; sp = min(sp, h)
  mean = corr * ((h - sp) * dec + sp * (pre(min(eff, chunk)) + dec - c0)) / h      (sp == 0 -> corr * dec)
dec = decode_law_ms 게이지(같은 상태의 decode 모델 값), corr = instance_correction, pace = pace_ms, duty = prefill_duty.
실제: 시각 t 뒤 T = N x (그 시각 관측 step 시간) 초 동안의 raw_step_ms를 step 수로 가중 평균(같은 측정이 반복 기록된 행은 뺀다).
"""
import json, sys, collections, bisect
import numpy as np
R = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
TT = [(1,21.46),(2,20.62),(4,20.51),(8,20.61),(16,20.24),(32,20.45),(64,20.43),(128,22.34),(256,26.39),(512,40.42),(768,57.08),(1024,80.15),(1536,103.39),(2048,136.07),(3072,200.32),(4096,264.24),(5120,324.32),(6144,392.62),(7168,463.12),(8192,519.69),(12288,745.95),(16384,972.22)]
TX = [a for a, _ in TT]; TY = [b for _, b in TT]
def pre(x): return float(np.interp(x, TX, TY))
C0 = 16.361; CHUNK = 8192.0
def mean_step(dec, corr, queued, duty, pace, h):
    arriving = 0.0
    if duty > 0 and pace > 0:
        per = pre(CHUNK) - C0
        arriving = duty * (h * pace) / per * CHUNK
    eff = max(queued, arriving)
    if eff <= 0: return corr * dec
    sp = 1.0 if eff <= CHUNK else eff / CHUNK
    sp = min(sp, h)
    return corr * ((h - sp) * dec + sp * (pre(min(eff, CHUNK)) + dec - C0)) / h
G = ["decode_law_ms", "instance_correction", "queued_prefill_tokens", "prefill_duty", "pace_ms", "predicted_step_ms",
     "raw_step_ms", "obs_steps", "obs_kv_tokens", "gate_allowance_ms", "live_requests", "observed_step_ms"]
def load(run):
    per = collections.defaultdict(list)
    for l in open(R + run + "/server_metrics/scheduler.jsonl"):
        x = json.loads(l); row = collections.defaultdict(dict)
        for k, v in x.items():
            if k.startswith("scheduler_fluidserve_") and "|instance=" in k:
                n, i = k[21:].split("|instance=")
                if n in G: row[i][n] = v
        for i, d in row.items():
            if all(d.get(g) is not None for g in G): per[i].append((x["t"], d))
    return per
def analyse(run, hrun):
    per = load(run)
    out = collections.defaultdict(list)   # key -> list of (pred_h, realized_N)
    val = []
    for inst, rows in per.items():
        ts = [t for t, _ in rows]
        # 측정 행: raw_step_ms·obs_kv_tokens가 직전 행과 같으면 같은 측정의 반복
        meas = []; prev = None
        for t, d in rows:
            key = (d["raw_step_ms"], d["obs_kv_tokens"], d["obs_steps"])
            if key != prev and d["raw_step_ms"] > 0 and d["obs_steps"] > 0: meas.append((t, d["raw_step_ms"], d["obs_steps"]))
            prev = key
        mt = [m[0] for m in meas]
        for t, d in rows:
            if d["live_requests"] <= 0 or d["observed_step_ms"] <= 0: continue
            p_run = mean_step(d["decode_law_ms"], d["instance_correction"], d["queued_prefill_tokens"], d["prefill_duty"], d["pace_ms"], hrun)
            val.append((p_run, d["predicted_step_ms"]))
            cond = ("q>0" if d["queued_prefill_tokens"] > 0 else "q=0", int(round(d["gate_allowance_ms"])) if d["gate_allowance_ms"] > 0 else -1)
            for N in (25, 50, 100):
                T = N * d["observed_step_ms"] / 1000.0
                a = bisect.bisect_right(mt, t); b = bisect.bisect_right(mt, t + max(T, 1.0))
                if b <= a: continue
                w = sum(m[2] for m in meas[a:b]); real = sum(m[1] * m[2] for m in meas[a:b]) / w
                for h in (25, 50, 100):
                    p = mean_step(d["decode_law_ms"], d["instance_correction"], d["queued_prefill_tokens"], d["prefill_duty"], d["pace_ms"], h)
                    out[(N, h, "all")].append((p, real)); out[(N, h) + cond].append((p, real))
    v = np.array(val)
    print(f"== {run[12:]}  (run의 분모 {hrun})  재계산 검증: |재계산 − 기록| 중앙값 {np.median(np.abs(v[:,0]-v[:,1])):.3f} ms, 95분위 {np.percentile(np.abs(v[:,0]-v[:,1]),95):.2f} ms, 표본 {len(v)}")
    for N in (25, 50, 100):
        line = []
        for h in (25, 50, 100):
            a = np.array(out[(N, h, "all")]); r = a[:,0] / a[:,1]
            line.append(f"h{h}: 예측/실제 중앙값 {np.median(r):.3f} 평균오차 {np.mean(a[:,0]-a[:,1]):+5.1f}ms |오차| {np.mean(np.abs(a[:,0]-a[:,1])):4.1f}")
        print(f"  다음 {N:3d} iteration 실제와:  " + " ; ".join(line))
    print("  조건별 (다음 50 iteration 실제와, 평균오차 ms / 표본)")
    for cond in sorted({k[2:] for k in out if len(k) == 4}):
        line = []
        for h in (25, 50, 100):
            a = np.array(out[(50, h) + cond]) if out.get((50, h) + cond) else None
            if a is None or len(a) < 50: continue
            line.append(f"h{h} {np.mean(a[:,0]-a[:,1]):+6.1f}")
        if line: print(f"    {cond[0]} gate {cond[1]:>4}: n={len(out[(50,25)+cond]):6d}  " + "  ".join(line) + f"   (실제 평균 {np.mean(np.array(out[(50,25)+cond])[:,1]):.1f} ms)")
if __name__ == "__main__":
    for arg in sys.argv[1:]:
        run, h = arg.split(":"); analyse(run, int(h))
