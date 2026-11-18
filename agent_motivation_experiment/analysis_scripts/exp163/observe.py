import os
"""EXP-163 관찰: run 하나(단일 클래스)에 대해 점수, 5분 거절률, 마감 위반 종류, TTFT, prefix 적중, 예측기 정확도(iteration·첫 토큰), 결정 지연, 클라이언트 상태.
사용: python3 observe.py <run 이름>   (대시보드가 채점을 마친 뒤)"""
import json, re, sys, glob, urllib.request, numpy as np, pandas as pd
RES = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
C = "/home/nxclab/scratch/dashboard-cache/runs/"; REC = "/home/nxclab/scratch/dashboard-cache/recorder/"
TR = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/traces/dynamic/canonical/"
B = {"chat": (5, 50), "deepresearch": (10, 100), "swe": (7, 75)}; TIER = {"50": "chat", "100": "deepresearch", "25": "swe"}
run = sys.argv[1]; R = RES + run
api = lambda p: json.load(urllib.request.urlopen("http://localhost:8080/api" + p))
d = api(f"/runs/{run}"); s = d["score"]
print(f"== {run}\n점수: 모든 도착 {s['all_arrivals']:.2f}, admitted {s['admitted']:.2f}, 거절률 {s['rejected_pct']:.2f} %, goodput {s['goodput_tok_s']:.0f} tok/s, preemption {d['preemptions']:.0f}")
for c, v in d["score"]["per_class"].items():
    print(f"  {c}: 모든 도착 {v['all_arrivals']:.2f}, admitted {v['admitted']:.2f}, 거절 {v['rejected_pct']:.2f} %")
L = api(f"/runs/{run}/latency")
for c, v in L["violations"].items():
    print(f"  {c} 받은 {v['admitted']} 마감 위반 {v['deadline_fail']} (첫 토큰 {v['fail_ttft']}, 평균 간격 {v['fail_tbt']}, 몰림 {v['fail_other']}); 평균 간격 > budget {v['tbt_over']} ({100*v['tbt_over']/max(v['admitted'],1):.1f} %)")
q = pd.read_csv(C + run + "/requests.csv.gz"); q = q[q.cutoff == 0]
cls = q["class"].mode()[0]; tt, tb = B[cls]
g = q.groupby((q.rel // 300).astype(int)).rejected.mean() * 100
print("5분 거절률 %:", " ".join(f"{x:.0f}" for x in g.values[:12]))
a = q[(q.rejected == 0) & (q.errored == 0)]
print(f"TTFT p50/p90/p99 {a.first_token_latency.quantile(.5):.2f}/{a.first_token_latency.quantile(.9):.2f}/{a.first_token_latency.quantile(.99):.2f} s (budget {tt} s), TTFT 초과 {int((a.first_token_latency > tt).sum())}; 평균 토큰 간격 p50/p90/p99 {a.itl_ms.quantile(.5):.1f}/{a.itl_ms.quantile(.9):.1f}/{a.itl_ms.quantile(.99):.1f} ms (budget {tb})")
# 결정 로그: prefix 적중, 첫 토큰 예측
ids = {}
for l in open(R + "/request_ids.jsonl"):
    i = json.loads(l); ids[re.sub(r"^(chat)?cmpl-", "", i["request_id"])] = (i["task_id"], i["call_index"])
rx = re.compile(r"dispatch request (\S+) fsplacement tier=(\d+) waited=(\d+) prefillest=([\d.]+) prefillraw=([\d.]+) prompt=(\d+) decision=route")
rows = []
for l in open(R + "/server_metrics/scheduler_dispatch.log"):
    m = rx.search(l)
    if m: rows.append((m.group(1), int(m.group(3)), float(m.group(4)), float(m.group(5)), int(m.group(6))))
D = pd.DataFrame(rows, columns=["id", "waited", "est", "raw", "prompt"])
hit = 1 - D.raw / D.prompt.clip(lower=1)
print(f"prefix 적중(스케줄러 추정): 적중 비율 중앙값 {hit.median():.2f}, 거의 통째 적중(raw < 5 %) {100*(D.raw < 0.05*D.prompt).mean():.1f} %, 적중 토큰 중앙값 {np.median(D.prompt - D.raw):.0f}, 프롬프트 중앙값 {D.prompt.median():.0f}")
D["key"] = D.id.map(ids); D = D.dropna(subset=["key"]); D["task_id"] = D.key.map(lambda k: k[0]); D["call_index"] = D.key.map(lambda k: k[1])
J = D.drop_duplicates(["task_id", "call_index"], keep="last").merge(a[["task_id", "call_index", "first_token_latency"]], on=["task_id", "call_index"])
pred = (J.waited + J.est) / 1000; err = J.first_token_latency - pred
# EXP-165: other policies write no fsplacement line, so there is nothing to compare.
if len(J) == 0: print("첫 토큰 예측: FluidServe 결정 로그가 없다(다른 정책)")
else: print(f"첫 토큰 예측(waited + prefillest) 대 실제 TTFT: 표본 {len(J)}, (실제 − 예측) p10/p50/p90 {np.percentile(err,10):.2f}/{np.median(err):.2f}/{np.percentile(err,90):.2f} s, 실제가 예측보다 긴 비율 {100*(err > 0).mean():.1f} %, 예측 > budget이었는데 실제는 안 {int(((pred > tt) & (J.first_token_latency <= tt)).sum())}건")
# iteration 예측 대 실측
_fj = REC + run + "/fluidserve.jsonl"
rows = [json.loads(l) for l in open(_fj)] if os.path.exists(_fj) else [{"t": 0}]; t0 = rows[0]["t"]; rows = [x for x in rows if 60 < x["t"] - t0 < 3540]
o, p = [], []
for x in rows:
    for k, v in x.items():
        if k.startswith("scheduler_fluidserve_observed_step_ms|instance="):
            pv = x.get(k.replace("observed_step_ms", "predicted_step_ms"))
            if v and v > 0 and pv and pv > 0: o.append(v); p.append(pv)
o, p = np.array(o), np.array(p)
fin = np.isfinite(p); n_inf = int((~fin).sum()); o, p = o[fin], p[fin]
if len(o) == 0: print("iteration 예측 대 실측: FluidServe gauge가 없다(다른 정책)")
else:
  print(f"iteration 예측이 무한대(모델이 값을 내지 못함)인 표본 {n_inf}개 제외")
  print(f"iteration 예측 대 실측(2초 gauge): 실측 평균 {o.mean():.1f} ms, 예측 평균 {p.mean():.1f} ms, 예측/실측 {p.mean()/o.mean():.3f}, (예측 − 실측) p10/p50/p90 {np.percentile(p-o,10):.1f}/{np.median(p-o):.1f}/{np.percentile(p-o,90):.1f} ms")
# 결정 지연, 클라이언트
sj = [json.loads(l) for l in open(R + "/server_metrics/scheduler.jsonl")]
tot = lambda r, k: sum(v for kk, v in r.items() if kk.split("|")[0] == k and isinstance(v, (int, float)))
n = tot(sj[-1], "request_full_mode_schedule_duration_microseconds_count") - tot(sj[0], "request_full_mode_schedule_duration_microseconds_count")
print(f"결정 지연 {(tot(sj[-1],'request_full_mode_schedule_duration_microseconds_sum') - tot(sj[0],'request_full_mode_schedule_duration_microseconds_sum'))/max(n,1):.0f} µs/결정 ({n}회)")
m = pd.read_csv(R + "/metrics.csv", usecols=["task_id", "agent", "start_time", "is_rejected", "is_error", "error_msg"]); m = m[m.agent == "request"]
plan = json.load(open(R + "/run_config.json"))["workload_config"]["class_plan_file"].split("/")[-1]
tr = pd.read_csv(TR + plan); tm = (tr.arrival_s // 60).astype(int).value_counts()
m["min"] = ((m.start_time - m.start_time.min()) // 60).astype(int); gg = m.groupby("min").task_id.nunique()
r_ = np.array([gg[k] / max(tm.get(k, 1), 1) for k in gg.index if 1 <= k <= 58])
e = m[m.is_error.astype(bool) & ~m.is_rejected.astype(bool)]
print(f"클라이언트: 분별 요청/trace {r_.min():.3f}~{r_.max():.3f}, 전체 {m.task_id.nunique()}/{len(tr)}, 거절 아닌 오류 {len(e)} {e.error_msg.fillna('').str.slice(0,50).value_counts().head(2).to_dict()}")
