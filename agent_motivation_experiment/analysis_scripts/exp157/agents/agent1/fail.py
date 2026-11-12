import pickle, numpy as np, pandas as pd
from report import RUNS, load, OUT
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
for name in ("c2h100 r1", "c2h100 r2", "v0.5 r2"):
    r = RUNS[name]; C = load(r)
    L = np.sort(C.n_tokens.values.astype(float))
    def resid(j):  # empirical E[L-j | L>j] from this run's admitted chat lengths
        x = L[L > j]; return (x - j).mean() if len(x) else 1
    F = C[(~C.ladder_ok.astype(bool)) & (C.first_token_latency <= 5) & (C.itl_ms > 50)].copy()
    S = C[C.ladder_ok.astype(bool) & C.tbt_over]
    negS = S[S.min_slack < 0]
    print(f"\n##### {name}: failures TTFT<=5s & mean ITL>50ms: {len(F)} (all admitted-chat failures {(~C.ladder_ok.astype(bool)).sum()}); TBT-over successes {len(S)}, of which slack went <0 at some chunk: {len(negS)}")
    cols = ["n_tokens", "n_late", "first_token_latency", "itl_ms", "need_pace", "own_allow_disp", "pred_step_disp", "obs_step", "min_slack", "first_neg_frac",
            "pace_before_neg", "pace_after_neg", "obs_step_before_neg", "obs_step_after_neg", "frac_time_mixed", "kv_other_share", "eng_kv"]
    print(pd.DataFrame({"fail_p50": F[cols].median(), "fail_mean": F[cols].mean(), "TBTover_ok_p50": S[cols].median(), "ok_but_neg_p50": negS[cols].median()}).round(3).to_string())
    print(" failures: n_tokens quantiles", F.n_tokens.quantile([.1, .25, .5, .75, .9]).tolist(), " share n_tokens>1000: %.0f%%; admitted chat with n_tokens>1000: %d, of them failed %d" % (100*(F.n_tokens > 1000).mean(), (C.n_tokens > 1000).sum(), (F.n_tokens > 1000).sum()))
    print(" failures by port:", F.port.value_counts().to_dict(), " by 5-min window:", ((F.t_rel // 300).astype(int)).value_counts().sort_index().to_dict())
    print(" failures whose pace after slack<0 exceeded pace before: %d/%d; instance observed step after > before: %d/%d" % (
        (F.pace_after_neg > F.pace_before_neg).sum(), F.pace_after_neg.notna().sum(), (F.obs_step_after_neg > F.obs_step_before_neg).sum(), F.obs_step_after_neg.notna().sum()))
    print(" failures with client ITL <= own allowance at placement:", (F.itl_ms <= F.own_allow_disp).sum())
    # scheduler-side remaining for a long request: what allowance would the ladder gate grant at j given empirical residual
    for j in (0, 200, 400, 800, 1200):
        print(f"   j={j}: empirical E[L-j|L>j]={resid(j):.0f} tokens -> allowance with 4.2 s slack = {50+4200/resid(j):.1f} ms")
    print(" sample failures (source rows):")
    print(F.sort_values("t_rel")[["task_id" if "task_id" in F else "key", "port", "t_rel", "n_tokens", "n_late", "first_token_latency", "itl_ms", "own_allow_disp", "pred_step_disp", "obs_step", "min_slack", "first_neg_frac", "pace_before_neg", "pace_after_neg"]].head(12).round(2).to_string(index=False))
