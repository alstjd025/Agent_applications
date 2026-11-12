"""EXP-157: 얼마나 빡빡했나 — 클래스별 실측 TBT·TTFT·마감 대비 남은 시간, 스케줄러 예측 대 실측 iteration 시간, KV 사용률."""
import json, sys, collections, numpy as np, pandas as pd
C = "/home/nxclab/scratch/dashboard-cache/"
B = {"chat": (5, 50), "deepresearch": (10, 100), "swe": (7, 75)}
RUNS = {"v0.5": ["261109_2311_exp157hr1_fsexactprog_shift", "261110_0354_exp157hr2_fsexactprog_shift"],
        "h25": ["261110_0020_exp157hr1_fsc2h25_shift", "261110_0507_exp157hr2_fsc2h25_shift"],
        "h100": ["261110_0133_exp157hr1_fsc2h100_shift", "261110_0619_exp157hr2_fsc2h100_shift"],
        "h200": ["261110_0242_exp157hr1_fsc2h200_shift", "261110_0729_exp157hr2_fsc2h200_shift"]}
print("== 클라이언트 실측 (admitted, 오류 없음, 두 반복 합)")
print("arm  class  n  TBT평균 p50 p90 p99 (ms) | 예산 | TTFT p50 p90 (s) | 끝날 때 남은 시간 p10 p50 (s) | 남은시간/허용시간 p50")
for a, rs in RUNS.items():
    d = pd.concat([pd.read_csv(C + "runs/" + r + "/requests.csv.gz") for r in rs])
    d = d[(d.rejected == 0) & (d.errored == 0) & (d.cutoff == 0)]
    for c, (tt, tb) in B.items():
        x = d[(d["class"] == c) & (d.output_tokens > 1)]
        allow = tt + (x.output_tokens - 1) * tb / 1000; spare = allow - x.latency
        print(f"{a:5s} {c:12s} {len(x):6d} {x.itl_ms.mean():5.1f} {x.itl_ms.median():5.1f} {x.itl_ms.quantile(.9):5.1f} {x.itl_ms.quantile(.99):5.1f} | {tb} | "
              f"{x.first_token_latency.median():4.2f} {x.first_token_latency.quantile(.9):4.2f} | {spare.quantile(.1):6.1f} {spare.median():6.1f} | {(spare/allow).median():.2f}")
print("\n== 스케줄러 gauge (인스턴스별 1초 기록을 run 전체에서 평균, 측정 구간 60~3540초)")
print("arm  run  실측 iteration(ms) 예측(ms) 예측/실측 | KV 사용 / cap_mem 평균 p90 | 투영 KV / cap_mem p90 | tightest allowance 중앙값(ms)")
for a, rs in RUNS.items():
    for r in rs:
        rows = [json.loads(l) for l in open(C + "recorder/" + r + "/fluidserve.jsonl")]
        t0 = rows[0]["t"]; rows = [x for x in rows if 60 < x["t"] - t0 < 3540]
        def per(name):
            out = collections.defaultdict(list)
            for x in rows:
                for k, v in x.items():
                    if k.startswith(name + "|instance="): out[k.split("=")[1]].append(v)
            return out
        obs, pre, kv, cm, pj, ta = (per("scheduler_fluidserve_" + n) for n in ("observed_step_ms", "predicted_step_ms", "obs_kv_tokens", "cap_mem_tokens", "projected_kv_tokens", "tightest_allowance_ms"))
        o = np.concatenate([np.array(v) for v in obs.values()]); p = np.concatenate([np.array(v) for v in pre.values()])
        m = (o > 0) & (p > 0) & np.isfinite(p)
        u = []; up = []
        for i in kv:
            k = np.array(kv[i]); c_ = np.array(cm[i]); j = np.array(pj.get(i, [np.nan] * len(k)))
            n = min(len(k), len(c_), len(j)); ok = c_[:n] > 0
            u.append(k[:n][ok] / c_[:n][ok]); up.append(j[:n][ok] / c_[:n][ok])
        u = np.concatenate(u); up = np.concatenate(up); t = np.concatenate([np.array(v) for v in ta.values()]); t = t[np.isfinite(t) & (t > 0) & (t < 1e5)]
        print(f"{a:5s} {r[7:16]} {o[m].mean():6.1f} {p[m].mean():6.1f} {p[m].mean()/o[m].mean():5.2f} | {np.nanmean(u):.2f} {np.nanpercentile(u,90):.2f} | {np.nanpercentile(up,90):.2f} | {np.median(t) if len(t) else float('nan'):.1f}")
