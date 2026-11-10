"""EXP-155 1단계(오프라인): KV 정점 예측 버전 비교.

시각 t = 처리 중 목록 snapshot 시각(30초 간격), 인스턴스마다. 앞으로 H = h iteration(= h x 그 순간 iteration 시간) 동안의 실제 최대 KV(M)를 각 버전의 예측과 비교한다.
실제 KV = obs_kv_tokens(1초). 처리 중 요청과 그 진행 토큰 j·프롬프트·클래스는 snapshot의 요청 ID를 클라이언트 기록(tbt_events, request_ids)에 맞춰 구한다.
회수 R(τ) = Σ p_i(τ)·(prompt_i + j_i), p_i(τ) = (S(j) − S(j+τ))/S(j) (클래스 길이 프로파일), 엔진 대기 요청은 뺀다. decode 증가 = n·τ (n = obs_decode_batch).
도착 A = 결정 로그의 route 프롬프트(인스턴스별). 도착 추정은 t 이전 정보만 쓴다.
버전마다 τ ∈ {0, h/4, h/2, 3h/4, h} 중 최대를 정점 예측으로 쓴다(K1은 지금 코드처럼 끝점 하나, z = 1.65).
출력: (예측 − 실제 정점) / 지금 KV 의 중앙값(쏠림), 평균 |오차|, 하위 5 %(실제보다 크게 작게 예측한 꼬리), 표본 수."""
import json, sys, re, glob, bisect, collections, datetime as dt, math
import numpy as np, pandas as pd
RES = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
CACHE = "/home/nxclab/scratch/dashboard-cache/runs/"
PROF = json.load(open("/home/nxclab/llumnix_reproduce/deploy/profiling/llama31-70b-b200-tp2/fluidserve.json"))["classes"]
TIER = {"swe": 25, "chat": 50, "deepresearch": 100}
SURV = {c["name"]: (np.array(c["grid"], float), np.array(c["survival"], float)) for c in PROF}
def S(cls, j):
    g, s = SURV[cls]; return float(np.interp(j, g, s, left=1.0, right=s[-1]))
def pdone(cls, j, k):
    sj = S(cls, j)
    if sj <= 1e-9: return 1.0
    return min(1.0, max(0.0, (sj - S(cls, j + k)) / sj))

def load_run(run, snapdir):
    # client: request id -> (class, prompt, start, chunk arrival abs times)
    ids = {}
    for l in open(RES + run + "/request_ids.jsonl"):
        i = json.loads(l); ids[(i["task_id"], i["call_index"])] = re.sub(r"^(chat)?cmpl-", "", i["request_id"])
    req = pd.read_csv(CACHE + run + "/requests.csv.gz", usecols=["task_id", "call_index", "class"])
    cls = {(r.task_id, r.call_index): r["class"] for _, r in req.iterrows()}
    info = {}
    for l in open(RES + run + "/tbt_events.jsonl"):
        if '"agent": "request"' not in l: continue
        e = json.loads(l); k = (e["task_id"], e["call_index"]); rid = ids.get(k); c = cls.get(k)
        ev = e.get("chunk_events") or []
        if rid is None or c is None or not ev: continue
        st = e["start_time"]
        info[rid] = (c, e.get("input_tokens") or 0, np.array([st + x["arrival_offset_ms"] / 1000 for x in ev if x.get("arrival_offset_ms") is not None]))
    t0 = min(v[2][0] for v in info.values() if len(v[2])); t1 = max(v[2][-1] for v in info.values() if len(v[2]))
    # scheduler gauges
    ser = collections.defaultdict(list)
    for l in open(RES + run + "/server_metrics/scheduler.jsonl"):
        x = json.loads(l); g = collections.defaultdict(dict)
        for k, v in x.items():
            for name in ("obs_kv_tokens", "observed_step_ms", "obs_decode_batch", "projected_kv_tokens"):
                if k.startswith("scheduler_fluidserve_" + name + "|instance="): g[k.split("=")[1]][name] = v
        for i, d in g.items():
            if len(d) == 4 and None not in d.values(): ser[i].append((x["t"], d))
    day = dt.datetime.fromtimestamp(t0, dt.timezone.utc).date()
    arr = collections.defaultdict(list)
    for l in open(RES + run + "/server_metrics/scheduler_dispatch.log", errors="replace"):
        if "fsplacement" not in l or "decision=route" not in l: continue
        m = re.match(r"I(\d\d)(\d\d) (\d\d):(\d\d):(\d\d\.\d+)", l)
        t = dt.datetime(day.year, int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), tzinfo=dt.timezone.utc).timestamp() + float(m.group(5))
        arr[re.search(r"inst=(\d+)", l).group(1)].append((t, int(re.search(r"prompt=(\d+)", l).group(1))))
    snaps = []
    for f in glob.glob(snapdir + "/*.jsonl"):
        for l in open(f):
            try: s = json.loads(l)
            except Exception: continue
            t = s["t_ms"] / 1000
            if t0 + 120 <= t <= t1 - 300 and s["instances"]: snaps.append((t, s["instances"]))
    return info, ser, arr, sorted(snaps, key=lambda x: x[0])

def rate_estimators(times, prompts, t):
    """t 이전 도착(토큰/초) 추정들. 1초 bin."""
    out = {}
    i = bisect.bisect_left(times, t)
    lo = bisect.bisect_left(times, t - 300)
    ts = np.array(times[lo:i]); ps = np.array(prompts[lo:i], float)
    age = t - ts
    for W in (10, 30, 60):
        out[f"K2a_win{W}s"] = ps[age <= W].sum() / W
    for hl in (10, 30):
        w = 0.5 ** (age / hl); out[f"K2b_ewma{hl}s"] = (ps * w).sum() / (hl / math.log(2))
    # 5초 bin 속도 열
    bins = np.zeros(60)
    idx = np.minimum((age // 5).astype(int), 59)
    np.add.at(bins, idx, ps); rates = bins / 5.0          # bins[0] = 가장 최근 5초
    wts = 0.5 ** ((np.arange(60) * 5) / 30.0)
    mu = (rates * wts).sum() / wts.sum(); var = ((rates - mu) ** 2 * wts).sum() / wts.sum()
    out["K2c_ewma30s+2sd"] = mu + 2 * math.sqrt(var)
    out["K2d_max5s_in60s"] = rates[:12].max()
    w60 = 0.5 ** ((np.arange(60) * 5) / 60.0); order = np.argsort(rates); cw = np.cumsum(w60[order]) / w60.sum()
    out["K2e_p90_hl60s"] = rates[order][min(np.searchsorted(cw, 0.9), 59)]
    return out

def analyse(run, snapdir, H_list=None):
    H_list = H_list or (int(run.split("fsc2h")[1].split("_")[0]),)
    info, ser, arr, snaps = load_run(run, snapdir)
    sts = {i: [r[0] for r in rows] for i, rows in ser.items()}
    skv = {i: np.array([r[1]["obs_kv_tokens"] for r in rows]) for i, rows in ser.items()}
    at = {i: [a[0] for a in sorted(v)] for i, v in arr.items()}; ap = {i: [a[1] for a in sorted(v)] for i, v in arr.items()}
    res = collections.defaultdict(list)
    for t, insts in snaps:
        for inst, recs in insts.items():
            if inst not in ser: continue
            k = bisect.bisect_left(sts[inst], t)
            if k >= len(sts[inst]) - 1: continue
            d = ser[inst][k][1]; K = d["obs_kv_tokens"]; n = d["obs_decode_batch"]; step = d["observed_step_ms"] / 1000
            if K <= 0 or step <= 0: continue
            live = []
            for r in recs:
                v = info.get(r["id"])
                if v is None or r.get("waiting"): continue
                c, prompt, arrivals = v; j = int((arrivals < t).sum())
                if j == 0: continue
                live.append((c, prompt + j, j))
            # K 변화율(1초 KV 열)
            lo = bisect.bisect_left(sts[inst], t - 60)
            tt = np.array(sts[inst][lo:k + 1]); kk = skv[inst][lo:k + 1]
            slope = {}
            for hl in (10, 30):
                if len(tt) > 3:
                    w = 0.5 ** ((t - tt[1:]) / hl); dk = np.diff(kk) / np.maximum(np.diff(tt), 1e-3)
                    slope[hl] = (dk * w).sum() / w.sum()
                else: slope[hl] = 0.0
            rates = rate_estimators(at.get(inst, []), ap.get(inst, []), t)
            for H in H_list:
                T = H * step
                b = bisect.bisect_right(sts[inst], t + T)
                if b >= len(sts[inst]): continue
                M = skv[inst][k + 1:b].max() if b > k + 1 else skv[inst][k]
                taus = [0, H / 4, H / 2, 3 * H / 4, H]
                def R(tau, z=0.0):
                    ps = [(pdone(c, j, tau), kv) for c, kv, j in live]
                    mean = sum(p * kv for p, kv in ps); var = sum(p * (1 - p) * kv * kv for p, kv in ps)
                    return max(0.0, mean - z * math.sqrt(var))
                Rm = {tau: R(tau) for tau in taus}
                preds = {"K0_now": K, "K1_now_code(end,z1.65,no arr)": K + n * H - R(H, 1.65),
                         "K1b_peak,no arr": max(K + n * tau - Rm[tau] for tau in taus)}
                for name, rate in rates.items():
                    preds[name] = max(K + n * tau + rate * tau * step - Rm[tau] for tau in taus)
                for hl, sl in slope.items():
                    preds[f"K3_slope{hl}s"] = max(K + sl * tau * step for tau in taus)
                dom = max(("chat","deepresearch","swe"), key=lambda c: sum(kv for cc, kv, _ in live if cc == c)) if live else "none"
                for name, p in preds.items():
                    if name.startswith("K2b_ewma10s"): res[(H, name + ":" + dom)].append((p - M) / K)
    print(f"== {run[12:]}  snapshot {len(snaps)}")
    names = sorted({k[1] for k in res})
    for H in H_list:
        print(f"  h={H}: (예측 − 실제 정점)/지금 KV  — 중앙값 / 평균|오차| / 하위5%   [표본]")
        for nm in names:
            a = np.array(res[(H, nm)])
            if len(a) == 0: continue
            print(f"     {nm:30s} {100*np.median(a):6.1f}% / {100*np.abs(a).mean():5.1f}% / {100*np.percentile(a,5):6.1f}%   [{len(a)}]")
for arg in sys.argv[1:]:
    run, snapdir = arg.split(":"); analyse(run, snapdir)
