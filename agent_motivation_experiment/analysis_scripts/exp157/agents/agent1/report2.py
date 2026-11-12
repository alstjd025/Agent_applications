import pickle, numpy as np, pandas as pd
from report import RUNS, load, OUT
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
for name, r in RUNS.items():
    C = load(r); o = C[C.tbt_over]
    print(f"\n##### {name} ({r}) chat admitted {len(C)}, TBT-over {len(o)}")
    print(" TBT-over with client mean ITL <= own ladder allowance at placement: %d/%d (%.1f%%)" % ((o.itl_ms <= o.own_allow_disp).sum(), len(o), 100*(o.itl_ms <= o.own_allow_disp).mean()))
    print(" TBT-over placed when scheduler's predicted step on that instance (mean of 2 s before dispatch) was >50 ms: %d/%d (%.1f%%)" % ((o.pred_step_disp > 50).sum(), len(o), 100*(o.pred_step_disp > 50).mean()))
    err = o.obs_step - o.pred_step_disp
    print(" TBT-over: observed step over life minus predicted step at dispatch (ms): mean %.2f p50 %.2f p90 %.2f; share >+5 ms %.1f%%" % (err.mean(), err.median(), err.quantile(.9), 100*(err > 5).mean()))
    print(" TBT-over: client ITL minus observed step over life (ms): mean %.2f p50 %.2f" % ((o.itl_ms - o.obs_step).mean(), (o.itl_ms - o.obs_step).median()))
    print(" all chat: client ITL quantiles", C.itl_ms.quantile([.1, .5, .9, .99]).round(1).tolist(), " TBT-over n_tokens p50", o.n_tokens.median(), " not-over n_tokens p50", C[~C.tbt_over].n_tokens.median())
    # by instance
    t = C.groupby("port").agg(n=("tbt_over", "size"), over=("tbt_over", "sum"), obs=("obs_step", "mean"), mixed=("frac_time_mixed", "mean"))
    t["pct"] = 100 * t.over / t.n; print(" by instance port:"); print(t.round(2).to_string())
    # by 5-minute window
    C["win"] = (C.t_rel // 300).astype(int)
    w = C.groupby("win").agg(n=("tbt_over", "size"), over=("tbt_over", "sum"), obs=("obs_step", "mean"))
    w["pct"] = (100 * w.over / w.n).round(1)
    print(" by 5-min window of arrival (pct over):", w.pct.tolist())
    print(" window chat admitted:", w.n.tolist())
    # grid: fraction of instance-seconds with observed step > 50
    G = pickle.load(open(OUT + r + "_grid.pkl", "rb"))["G"]
    for i, g in G.items():
        a = g["observed_step_ms"]; a = a[~np.isnan(a)]
        print(f"  inst {i[-4:]}: observed step p50 {np.median(a):.1f} p90 {np.percentile(a,90):.1f}; share of seconds >50 ms {100*(a>50).mean():.1f}%; mean KV usage {np.nanmean(g['kv_cache_usage_perc']):.3f}; mean running {np.nanmean(g['num_requests_running']):.0f}")
