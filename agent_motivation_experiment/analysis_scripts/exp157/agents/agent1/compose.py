import pickle, numpy as np, pandas as pd
from report import RUNS, load, OUT
for name, r in RUNS.items():
    g = pickle.load(open(OUT + r + "_grid.pkl", "rb")); G, port = g["G"], g["port"]; C = load(r)
    print(f"## {name}")
    for i in sorted(G, key=lambda x: port[x]):
        x = G[i]; tot = x["kv_chat"] + x["kv_deepresearch"] + x["kv_swe"]; m = tot > 0
        sh = {c: (x[f"kv_{c}"][m].sum() / tot[m].sum()) for c in ("chat", "deepresearch", "swe")}
        nchat = (C.port == port[i]).sum()
        st = x["observed_step_ms"]; st = st[~np.isnan(st)]
        print(f"  port {port[i]}: KV share chat {sh['chat']:.2f} dr {sh['deepresearch']:.2f} swe {sh['swe']:.2f} | mean resident n chat {x['n_chat'].mean():.0f} dr {x['n_deepresearch'].mean():.0f} swe {x['n_swe'].mean():.0f} | chat admitted here {nchat} ({100*nchat/len(C):.0f}%) | step p50 {np.median(st):.1f}")
    print(f"  chat requests whose instance had >50% of KV in deepresearch+swe on average over the chat's life: {(C.kv_other_share>0.5).sum()}/{len(C)} ({100*(C.kv_other_share>0.5).mean():.1f}%); TBT-over among them {100*C[C.kv_other_share>0.5].tbt_over.mean():.1f}% vs others {100*C[C.kv_other_share<=0.5].tbt_over.mean():.1f}%")
