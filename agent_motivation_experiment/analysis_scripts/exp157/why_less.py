"""EXP-157: v0.5 대 v0.6-후보2 h100 — 클래스별 받은 요청 수·제때 만든 토큰, 받은 요청의 출력 길이, chat TTFT 초과 요청의 TTFT, 받은 요청의 TTFT 분포."""
import pandas as pd, numpy as np
C = "/home/nxclab/scratch/dashboard-cache/runs/"
B = {"chat": (5, 50), "deepresearch": (10, 100), "swe": (7, 75)}
RUNS = {"v0.5": ["261109_2311_exp157hr1_fsexactprog_shift", "261110_0354_exp157hr2_fsexactprog_shift"],
        "h100": ["261110_0133_exp157hr1_fsc2h100_shift", "261110_0619_exp157hr2_fsc2h100_shift"]}
W = 3581.6 * 2
for a, rs in RUNS.items():
    d = pd.concat([pd.read_csv(C + r + "/requests.csv.gz") for r in rs]); d = d[(d.cutoff == 0)]
    print(f"== {a} (두 반복 합, 한 시간 기준으로 나눔)")
    for c, (tt, tb) in B.items():
        x = d[d["class"] == c]; adm = x[(x.rejected == 0) & (x.errored == 0)]
        ontime = (adm.n_tokens - adm.n_late).sum() / W
        miss_ttft = adm[adm.first_token_latency > tt]
        print(f"  {c:12s} 도착 {len(x)/2:7.0f}/h 받음 {len(adm)/2:7.0f}/h 제때 토큰 {ontime:7.0f} tok/s | 받은 요청 출력 중앙값 {adm.output_tokens.median():5.0f} 평균 {adm.output_tokens.mean():5.0f} | "
              f"TTFT p50/p99/max {adm.first_token_latency.median():.2f}/{adm.first_token_latency.quantile(.99):.2f}/{adm.first_token_latency.max():.1f} s | TTFT 초과 {len(miss_ttft)}건, 그 TTFT {np.round(np.sort(miss_ttft.first_token_latency.values)[:5],1)}")
