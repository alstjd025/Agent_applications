import pickle, numpy as np, pandas as pd
OUT = "/home/nxclab/scratch/exp157/agent1/out/"
RUNS = {"v0.5 r1": "261109_2311_exp157hr1_fsexactprog_shift", "v0.5 r2": "261110_0354_exp157hr2_fsexactprog_shift",
        "c2h100 r1": "261110_0133_exp157hr1_fsc2h100_shift", "c2h100 r2": "261110_0619_exp157hr2_fsc2h100_shift"}
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
def load(r):
    C = pickle.load(open(OUT + r + "_chat.pkl", "rb"))
    return C[~C.cutoff.astype(bool) & ~C.errored.astype(bool)].copy()
rows = []
for name, r in RUNS.items():
    C = load(r); o = C[C.tbt_over]; n = C[~C.tbt_over]
    rows.append(dict(run=name, chat_adm=len(C), tbt_over=len(o), pct=100 * len(o) / len(C),
                     itl_p50=C.itl_ms.median(), itl_over_p50=o.itl_ms.median(), itl_over_p90=o.itl_ms.quantile(.9),
                     ttft_p50=C.first_token_latency.median(),
                     mixed_disp_pct=100 * C.mixed_at_disp.mean(), frac_time_mixed=C.frac_time_mixed.mean(),
                     co_dr=C.co_dr.mean(), co_swe=C.co_swe.mean(), co_chat=C.co_chat.mean(), kv_other_share=C.kv_other_share.mean(),
                     eng_running=C.eng_running.mean(), eng_kv=C.eng_kv.mean(), eng_itl=C.eng_itl.mean(),
                     obs_step=C.obs_step.mean(), pred_disp=C.pred_step_disp.mean()))
print("== all admitted chat (cutoff/errored excluded)"); print(pd.DataFrame(rows).round(2).to_string(index=False))
print("\n== TBT-over vs not, by run")
for name, r in RUNS.items():
    C = load(r)
    g = C.groupby("tbt_over")[["itl_ms", "first_token_latency", "n_tokens", "mixed_at_disp", "frac_time_mixed", "co_dr", "co_swe", "co_chat",
                               "kv_other_share", "kv_total", "eng_running", "eng_kv", "eng_itl", "obs_step", "pred_step_disp", "obs_step_disp",
                               "pred_step_life", "own_allow_disp", "preempt"]].mean()
    print(name); print(g.round(3).T.to_string())
