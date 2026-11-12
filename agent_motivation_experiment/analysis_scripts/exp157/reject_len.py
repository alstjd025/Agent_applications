"""EXP-157: 거절된 요청과 받은 요청의 프롬프트 길이(metrics.csv에서 직접, 클래스는 requests.csv의 task_id로)."""
import pandas as pd
RES = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
C = "/home/nxclab/scratch/dashboard-cache/runs/"
RUNS = {"v0.5": ["261109_2311_exp157hr1_fsexactprog_shift", "261110_0354_exp157hr2_fsexactprog_shift"],
        "h100": ["261110_0133_exp157hr1_fsc2h100_shift", "261110_0619_exp157hr2_fsc2h100_shift"]}
bins = [0, 1000, 2000, 4000, 8000, 1e9]; lab = ["<1k", "1-2k", "2-4k", "4-8k", ">8k"]
for a, rs in RUNS.items():
    ds = []
    for r in rs:
        q = pd.read_csv(C + r + "/requests.csv.gz", usecols=["task_id", "class"]).drop_duplicates("task_id")
        m = pd.read_csv(RES + r + "/metrics.csv", usecols=["task_id", "iteration", "call_index", "input_tokens", "is_rejected", "is_error"])
        m = m.drop_duplicates(["task_id", "iteration", "call_index", "is_rejected"])
        ds.append(m.merge(q, on="task_id"))
    d = pd.concat(ds)
    print(f"== {a}")
    for c in ("chat", "deepresearch", "swe"):
        x = d[d["class"] == c]
        g = x.groupby(pd.cut(x.input_tokens, bins, labels=lab), observed=False).is_rejected.agg(["mean", "size"])
        print(f"  {c:12s} 프롬프트 p50 받음 {x[~x.is_rejected].input_tokens.median():.0f} / 거절 {x[x.is_rejected].input_tokens.median():.0f} | 길이별 거절률: "
              + " ".join(f"{k} {100*v['mean']:.0f}%(n={v['size']})" for k, v in g.iterrows()))
