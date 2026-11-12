"""EXP-157 chat TBT-over analysis, one run per invocation. Writes out/<run>_chat.pkl (per-chat-request
table) and out/<run>_grid.pkl (per-instance 1 s grid). Prints a short summary."""
import json, sys, re, pickle, datetime as dt, numpy as np, pandas as pd
RES = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
CACHE = "/home/nxclab/scratch/dashboard-cache/"
OUT = "/home/nxclab/scratch/exp157/agent1/out/"
run = sys.argv[1]
B = {"chat": (5.0, .050), "deepresearch": (10.0, .100), "swe": (7.0, .075)}
d = pickle.load(open(OUT + run + "_tbt.pkl", "rb")); summ, chunks = d["summ"], d["chunks"]
ids = {}
for l in open(RES + run + "/request_ids.jsonl"):
    i = json.loads(l); ids[re.sub(r"^(chat)?cmpl-", "", i["request_id"])] = (i["task_id"], i["call_index"])
sc = pd.read_csv(CACHE + "runs/" + run + "/requests.csv.gz")
sc["key"] = list(zip(sc.task_id, sc.call_index))
assert sc.key.is_unique
# ---- dispatch log
def gts(m):
    mo, dd, hh, mi, ss = int(m[1][:2]), int(m[1][2:]), int(m[2]), int(m[3]), float(m[4])
    return dt.datetime(2026, mo, dd, hh, mi, tzinfo=dt.timezone.utc).timestamp() + ss
rx_t = re.compile(r"^I(\d{4}) (\d\d):(\d\d):(\d\d\.\d+)")
rx_route = re.compile(r"dispatch request (\S+) to neutral instance (\d+)")
rx_fs = re.compile(r"dispatch request (\S+) fsplacement (.*)")
rx_cms = re.compile(r"instanceID=(\d+), metadata=.*api_server_port:(\d+)")
route, fsp, port = {}, {}, {}
for l in open(RES + run + "/server_metrics/scheduler_dispatch.log", errors="replace"):
    if "dispatch request" in l:
        m = rx_route.search(l)
        if m:
            route[m[1]] = (gts(rx_t.match(l)), m[2]); continue
        m = rx_fs.search(l)
        if m:
            kv = dict(x.split("=", 1) for x in m[2].split() if "=" in x)
            kv["t"] = gts(rx_t.match(l)); fsp[m[1]] = kv
    elif "api_server_port" in l:
        m = rx_cms.search(l)
        if m: port[m[1]] = int(m[2])
key2rid = {v: k for k, v in ids.items()}
rows = []
for k, s in summ.items():
    rid = key2rid.get(k); r = route.get(rid); f = fsp.get(rid, {})
    rows.append(dict(key=k, rid=rid, cls=s["cls"], start=s["start"], tf=s["first"], tl=s["last"], inp=s["inp"],
                     out=s["out"], t_disp=(r[0] if r else np.nan), inst=(r[1] if r else None),
                     waited=float(f.get("waited", "nan")), prefillest=float(f.get("prefillest", "nan")),
                     prefillqueue=float(f.get("prefillqueue", "nan")), decision=f.get("decision"),
                     prompt=float(f.get("prompt", "nan"))))
R = pd.DataFrame(rows).merge(sc.drop(columns=["task_id", "call_index", "class"]), on="key", how="left")
insts = sorted(port, key=lambda x: port[x])
T0 = int(np.nanmin(R.start)) - 5; T1 = int(np.nanmax(R.start + R.tl.fillna(0))) + 5; NT = T1 - T0 + 1
# ---- per-instance 1 s grid of residents by class (count and approx KV tokens)
cl = ["chat", "deepresearch", "swe"]
G = {i: {f"{w}_{c}": np.zeros(NT) for c in cl for w in ("n", "kv")} for i in insts}
adm = R[(~R.rejected.fillna(True)) & R.inst.notna() & R.tl.notna()]
for r in adm.itertuples():
    a = r.t_disp - T0; e = r.start + r.tl - T0; fi = r.start + r.tf - T0
    ia, ie = int(np.floor(a)), int(np.floor(e))
    if ie < ia: continue
    sec = np.arange(ia, ie + 1) + 0.5
    g = G[r.inst]; g[f"n_{r.cls}"][ia:ie + 1] += 1
    prog = np.clip((sec - fi) / max(e - fi, 1e-3), 0, 1) * (r.out or 0)
    g[f"kv_{r.cls}"][ia:ie + 1] += (r.inp or 0) + prog
# ---- engine per-second
def eng(p):
    recs = []
    for l in open(RES + run + f"/server_metrics/engine_{p}.jsonl"):
        try: e = json.loads(l)
        except Exception: continue
        if not e.get("ok"): continue
        o = {"t": e["t"]}
        for k, v in e.items():
            if "|" not in k: continue
            n = k.split("|")[0]
            if n in ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
                     "vllm:num_preemptions_total", "vllm:inter_token_latency_seconds_sum", "vllm:inter_token_latency_seconds_count",
                     "vllm:generation_tokens_total"):
                o[n.split(":")[1]] = v
        recs.append(o)
    E = pd.DataFrame(recs).sort_values("t")
    E["sec"] = (E.t - T0).astype(int)
    E["itl_ms"] = 1000 * E.inter_token_latency_seconds_sum.diff() / E.inter_token_latency_seconds_count.diff()
    E["preempt"] = E.num_preemptions_total.diff()
    E = E[(E.sec >= 0) & (E.sec < NT)].groupby("sec").last()
    out = {}
    for c in ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc", "itl_ms", "preempt"):
        a = np.full(NT, np.nan); a[E.index.values] = E[c].values; out[c] = a
    return out
for i in insts:
    G[i].update(eng(port[i]))
# ---- recorder gauges, forward-filled onto the grid
for i in insts:
    for n in ("observed_step_ms", "predicted_step_ms", "obs_decode_batch", "obs_kv_tokens", "cap_kv_tokens"):
        G[i][n] = np.full(NT, np.nan)
for l in open(CACHE + "recorder/" + run + "/fluidserve.jsonl"):
    e = json.loads(l); s = int(e["t"] - T0)
    if not (0 <= s < NT): continue
    for i in insts:
        for n in ("observed_step_ms", "predicted_step_ms", "obs_decode_batch", "obs_kv_tokens", "cap_kv_tokens"):
            v = e.get(f"scheduler_fluidserve_{n}|instance={i}")
            if v is not None: G[i][n][s] = v
for i in insts:
    for n in ("observed_step_ms", "predicted_step_ms", "obs_decode_batch", "obs_kv_tokens", "cap_kv_tokens"):
        G[i][n] = pd.Series(G[i][n]).ffill(limit=5).values
# ---- per chat request features
C = adm[adm.cls == "chat"].copy()
Lc = C.out.dropna().values.astype(float)
expTok = Lc.mean()
def mean_over(a, i0, i1):
    seg = a[i0:i1 + 1]
    return np.nan if len(seg) == 0 or np.all(np.isnan(seg)) else np.nanmean(seg)
feat = []
for r in C.itertuples():
    g = G[r.inst]; i0 = int(r.start + r.tf - T0); i1 = int(r.start + r.tl - T0); idp = int(r.t_disp - T0)
    nd = g["n_deepresearch"][i0:i1 + 1]; ns = g["n_swe"][i0:i1 + 1]; nc = g["n_chat"][i0:i1 + 1] - 1
    kvo = (g["kv_deepresearch"] + g["kv_swe"])[i0:i1 + 1]; kvt = (g["kv_deepresearch"] + g["kv_swe"] + g["kv_chat"])[i0:i1 + 1]
    o = dict(key=r.key,
             co_dr=nd.mean(), co_swe=ns.mean(), co_chat=nc.mean(), frac_time_mixed=((nd + ns) > 0).mean(),
             mixed_at_disp=(g["n_deepresearch"][idp] + g["n_swe"][idp]) > 0,
             kv_other_share=np.mean(kvo / np.maximum(kvt, 1)), kv_total=kvt.mean(),
             eng_running=mean_over(g["num_requests_running"], i0, i1), eng_kv=mean_over(g["kv_cache_usage_perc"], i0, i1),
             eng_itl=mean_over(g["itl_ms"], i0, i1), eng_waiting=mean_over(g["num_requests_waiting"], i0, i1),
             preempt=np.nansum(g["preempt"][i0:i1 + 1]),
             obs_step=mean_over(g["observed_step_ms"], i0, i1), pred_step_life=mean_over(g["predicted_step_ms"], i0, i1),
             pred_step_disp=mean_over(g["predicted_step_ms"], max(idp - 2, 0), idp),
             obs_step_disp=mean_over(g["observed_step_ms"], max(idp - 2, 0), idp))
    # ladder slack trajectory from client chunks
    ch = chunks.get(r.key)
    if ch is not None and len(ch[0]) > 1:
        t = r.start + ch[0].astype(float) / 1000
        cum = np.arange(len(t), dtype=float)  # chunk index = token index, as the ladder95 scorer does
        dl = r.start + 5.0 + 0.05 * cum            # deadline of first token in each chunk
        sl = dl - t
        neg = sl < 0
        o.update(min_slack=sl.min(), end_slack=sl[-1], first_neg_frac=(np.argmax(neg) / len(sl) if neg.any() else np.nan),
                 t_first_neg=(t[np.argmax(neg)] if neg.any() else np.nan))
        if neg.any():
            k = np.argmax(neg); tn = t[k]
            pre = (t[k] - t[0]) / max(cum[k] - cum[0], 1) * 1000 if k > 0 else np.nan
            post = (t[-1] - t[k]) / max(cum[-1] - cum[k], 1) * 1000 if cum[-1] > cum[k] else np.nan
            o.update(pace_before_neg=pre, pace_after_neg=post,
                     obs_step_before_neg=mean_over(g["observed_step_ms"], i0, int(tn - T0)),
                     obs_step_after_neg=mean_over(g["observed_step_ms"], int(tn - T0), i1))
    feat.append(o)
C = C.merge(pd.DataFrame(feat), on="key")
left = (5000 - C.waited - C.prefillest).clip(lower=0)
C["own_allow_disp"] = 50 + left / expTok          # arrivingLadderAllowance at placement, profile mean ~ empirical mean
C["exp_tok"] = expTok
C["need_pace"] = (5000 - 1000 * C.first_token_latency + 50 * (C.n_tokens - 1)) / np.maximum(C.n_tokens - 1, 1)
C["tbt_over"] = C.itl_ms > 50
C["t_rel"] = C.start - T0
C["port"] = C.inst.map(port)
pickle.dump(C, open(OUT + run + "_chat.pkl", "wb"))
pickle.dump(dict(G=G, T0=T0, port=port, R=R), open(OUT + run + "_grid.pkl", "wb"))
print(run, "chat admitted", len(C), "tbt_over", int(C.tbt_over.sum()), "expTok", round(expTok, 1))
