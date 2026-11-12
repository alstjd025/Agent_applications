"""EXP-157: (1) admitted인데 ladder95를 놓친 요청은 어디서 놓쳤나, (2) 거절된 요청은 어떤 요청인가(프롬프트 길이)."""
import pandas as pd, numpy as np
RES = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
C = "/home/nxclab/scratch/dashboard-cache/runs/"
B = {"chat": (5, 50), "deepresearch": (10, 100), "swe": (7, 75)}
RUNS = {"v0.5": ["261109_2311_exp157hr1_fsexactprog_shift", "261110_0354_exp157hr2_fsexactprog_shift"],
        "h100": ["261110_0133_exp157hr1_fsc2h100_shift", "261110_0619_exp157hr2_fsc2h100_shift"]}
for a, rs in RUNS.items():
    parts = []
    for r in rs:
        q = pd.read_csv(C + r + "/requests.csv.gz")
        m = pd.read_csv(RES + r + "/metrics.csv", usecols=["task_id", "iteration", "call_index", "input_tokens"])
        parts.append(q.merge(m.drop_duplicates(["task_id", "iteration", "call_index"]), on=["task_id", "iteration", "call_index"], how="left"))
    d = pd.concat(parts)
    d = d[(d.errored == 0) & (d.cutoff == 0)]
    print(f"===== {a}")
    for c, (tt, tb) in B.items():
        x = d[d["class"] == c]; adm = x[x.rejected == 0]; rej = x[x.rejected == 1]
        miss = adm[adm.ladder_ok == 0]
        ttft_miss = (miss.first_token_latency > tt).sum()
        print(f" {c:12s} admitted {len(adm):6d} 놓침 {len(miss):4d} ({100*len(miss)/max(len(adm),1):.2f}%) | 그중 TTFT 초과 {ttft_miss}, "
              f"늦은 토큰 비율 중앙값 {(miss.n_late/miss.n_tokens).median() if len(miss) else float('nan'):.2f}, TBT 중앙값 {miss.itl_ms.median():.1f} ms, 출력 중앙값 {miss.output_tokens.median():.0f}")
        print(f"    프롬프트 토큰 p50/p90 — 받은 요청 {adm.input_tokens.median():.0f}/{adm.input_tokens.quantile(.9):.0f}, 거절된 요청 {rej.input_tokens.median():.0f}/{rej.input_tokens.quantile(.9):.0f}")
        bins = [0, 2000, 8000, 20000, 1e9]; lab = ["<2k", "2k-8k", "8k-20k", ">20k"]
        g = x.groupby(pd.cut(x.input_tokens, bins, labels=lab), observed=False).rejected.mean() * 100
        print("    프롬프트 길이별 거절률(%):", " ".join(f"{k} {v:.0f}" for k, v in g.items()))
