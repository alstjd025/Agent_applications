"""0단계: KV 투영(projected_kv_tokens = 지금 KV + decode 수 x h − outflow)이 h iteration 뒤 실제 KV와 얼마나, 어느 쪽으로 다른가.

시각 t, 인스턴스 i마다(5초 간격 표본):
  예측  P = projected_kv_tokens(t)          스케줄러가 계산한 'h iteration 뒤 KV' (그 순간 넣으려는 요청은 빼고)
  기간  T = h x observed_step_ms(t)          h iteration이 걸릴 시간
  끝점  E = 실제 KV(t+T),  정점 M = max 실제 KV over (t, t+T]   (실제 KV = obs_kv_tokens: 엔진이 보고한 logical KV)
  도착  A = (t, t+T]에 이 인스턴스로 route된 프롬프트 토큰 합 (결정 로그)
출력(토큰, 그리고 그 인스턴스 실제 KV 대비 %):
  끝점 오차 P − E, 정점 오차 P − M  (음수 = 실제보다 작게 예측 = 낙관)
  도착을 알았다면  P + A − E          (빠진 도착 항이 오차의 얼마인가)
  지금 KV만 썼다면 K(t) − E           (투영이 지금 KV보다 나은가)"""
import json, sys, re, bisect, collections, datetime as dt
import numpy as np
R = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
def analyse(run, h):
    ser = collections.defaultdict(list)
    for l in open(R + run + "/server_metrics/scheduler.jsonl"):
        x = json.loads(l); t = x["t"]; g = collections.defaultdict(dict)
        for k, v in x.items():
            for name in ("projected_kv_tokens", "obs_kv_tokens", "observed_step_ms", "live_requests"):
                if k.startswith("scheduler_fluidserve_" + name + "|instance="):
                    g[k.split("=")[1]][name] = v
        for i, d in g.items():
            if len(d) == 4 and None not in d.values(): ser[i].append((t, d))
    # route된 프롬프트
    t0 = min(s[0][0] for s in ser.values()); day = dt.datetime.fromtimestamp(t0, dt.timezone.utc).date()
    arr = collections.defaultdict(list)
    for l in open(R + run + "/server_metrics/scheduler_dispatch.log", errors="replace"):
        if "fsplacement" not in l or "decision=route" not in l: continue
        m = re.match(r"I(\d\d)(\d\d) (\d\d):(\d\d):(\d\d\.\d+)", l)
        t = dt.datetime(day.year, int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), tzinfo=dt.timezone.utc).timestamp() + float(m.group(5))
        arr[re.search(r"inst=(\d+)", l).group(1)].append((t, int(re.search(r"prompt=(\d+)", l).group(1))))
    out = collections.defaultdict(list)
    for i, rows in ser.items():
        ts = [r[0] for r in rows]; kv = np.array([r[1]["obs_kv_tokens"] for r in rows])
        at = [a[0] for a in sorted(arr.get(i, []))]; ap = np.cumsum([0] + [a[1] for a in sorted(arr.get(i, []))])
        last = -1e9
        for n, (t, d) in enumerate(rows):
            if t - last < 5 or d["live_requests"] <= 0 or d["observed_step_ms"] <= 0: continue
            last = t
            T = h * d["observed_step_ms"] / 1000.0
            b = bisect.bisect_right(ts, t + T)
            if b >= len(ts) or b <= n + 1: continue
            E = kv[b - 1]; M = kv[n + 1:b].max(); P = d["projected_kv_tokens"]; K = d["obs_kv_tokens"]
            A = ap[bisect.bisect_right(at, t + T)] - ap[bisect.bisect_right(at, t)]
            for key, v in (("P-E", P - E), ("P-M", P - M), ("P+A-E", P + A - E), ("K-E", K - E), ("A", A), ("drop", K - P)):
                out[key].append(v)
            out["kv"].append(E)
    o = {k: np.array(v) for k, v in out.items()}
    pct = lambda a: 100 * np.median(a / o["kv"])
    print(f"== h={h:3d}  {run[12:]}  표본 {len(o['kv'])}  기간 중앙값 {np.median([0]):.0f}")
    for k, name in (("drop", "투영이 지금 KV에서 빼는 양 (K − P)"), ("A", "그 기간 실제 도착 프롬프트"), ("P-E", "끝점 오차 P − E"), ("P-M", "정점 오차 P − M"),
                    ("P+A-E", "도착을 더했다면 P + A − E"), ("K-E", "지금 KV만 썼다면 K − E")):
        a = o[k]
        print(f"   {name:34s} 중앙값 {np.median(a)/1e3:8.1f}K  평균 {a.mean()/1e3:8.1f}K  (실제 KV 대비 중앙값 {pct(a):6.1f}%)  |오차| 평균 {np.abs(a).mean()/1e3:7.1f}K")
for arg in sys.argv[1:]:
    run, h = arg.split(":"); analyse(run, int(h))
