"""EXP-158 판정용 집계. 사용: python3 analyse.py <run 이름> [비교할 v0.5 run 이름 ...]

1. 구현 검증: same 변형의 후보 불일치 수(0이어야 함), 아홉 지표(대시보드 채점), 결정 지연(scheduler.jsonl).
2. 결정 단위: 변형별 '가능한 인스턴스가 있었는가'의 실제 대비 뒤집힘, 분리 기회(spread), 고른 인스턴스 변경, T의 pend/shed.
3. 요청 단위(fsshadowv 줄 + 채점 결과): (가) v0.5가 받아 마감을 지킨 요청 가운데 변형이 마지막 시도에서 불가였던 비율,
   (나) v0.5가 거절한 요청 가운데 변형이면 어느 시도에서든 가능했던 비율.
카운터는 체인이 run 끝에 받아 둔 /home/nxclab/scratch/exp158/scheduler_metrics_final.txt에서 읽는다."""
import json, re, sys, collections, urllib.request
import numpy as np, pandas as pd
RES = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
CACHE = "/home/nxclab/scratch/dashboard-cache/runs/"
METRICS = "/home/nxclab/scratch/exp158/scheduler_metrics_final.txt"
VARS = ["same", "G1.0", "U", "U+G1.0", "M0", "M1", "U+M0"]
TIER = {"50": "chat", "100": "deepresearch", "25": "swe"}

def api(path):
    return json.load(urllib.request.urlopen("http://localhost:8080/api" + path))

def nine(run):
    s = api(f"/runs/{run}")["score"]; P = s["per_class"]
    out = {"모든 도착": s["all_arrivals"], "admitted": s["admitted"], "거절률": s["rejected_pct"]}
    for c in ("chat", "deepresearch", "swe"):
        out[c + " admitted"] = P[c]["admitted"]; out[c + " 모든 도착"] = P[c]["all_arrivals"]
    return out

def decision_latency_us(run):
    rows = [json.loads(l) for l in open(RES + run + "/server_metrics/scheduler.jsonl")]
    k_s = "request_full_mode_schedule_duration_microseconds_sum"; k_c = "request_full_mode_schedule_duration_microseconds_count"
    def tot(r, k): return sum(v for kk, v in r.items() if kk.split("|")[0] == k and isinstance(v, (int, float)))
    a, b = rows[0], rows[-1]
    n = tot(b, k_c) - tot(a, k_c)
    return (tot(b, k_s) - tot(a, k_s)) / n if n else float("nan")

def prom(path):
    out = collections.defaultdict(float)
    rx = re.compile(r'^([a-z_]+)\{([^}]*)\}\s+([-\d.e+]+)')
    for l in open(path):
        m = rx.match(l)
        if not m: continue
        lab = dict(x.split("=", 1) for x in re.findall(r'[a-z_]+="[^"]*"', m.group(2)))
        lab = {k: v.strip('"') for k, v in lab.items()}
        out[(m.group(1), tuple(sorted(lab.items())))] += float(m.group(3))
    return out

def counters(name, cnt, **want):
    rows = []
    for (n, lab), v in cnt.items():
        if n != name: continue
        d = dict(lab)
        if all(d.get(k) == w for k, w in want.items()): rows.append((d, v))
    return rows

def main(run, refs):
    print(f"=== {run}")
    # 1. 구현 검증
    me = nine(run)
    print("아홉 지표 (이 run / 비교 run 범위):")
    R = [nine(r) for r in refs]
    for k, v in me.items():
        rng = [r[k] for r in R]
        print(f"  {k:24s} {v:6.2f}   " + (f"{min(rng):6.2f}–{max(rng):6.2f}" if rng else ""))
    print(f"결정 지연(µs/결정): 이 run {decision_latency_us(run):.0f}" + "".join(f", {r[:11]} {decision_latency_us(r):.0f}" for r in refs))
    cnt = prom(METRICS)
    same_flip = sum(v for d, v in counters("scheduler_fluidserve_shadowv_candidate_flips_total", cnt, variant="same"))
    print(f"same 변형의 후보 불일치: {same_flip:.0f} (0이어야 한다)")
    # 2. 결정 단위
    print("\n결정 단위 — 클래스마다: 결정 수, 실제 가능→변형 불가(더 엄격), 실제 불가→변형 가능(더 느슨), 분리 기회(spread), 고른 인스턴스 변경")
    for t, c in TIER.items():
        real_spread = sum(v for d, v in counters("scheduler_fluidserve_shadowv_spread_total", cnt, variant="real", tier=t))
        print(f"  {c} (실제 spread {real_spread:.0f})")
        for vn in VARS:
            rows = counters("scheduler_fluidserve_shadowv_decision_total", cnt, variant=vn, tier=t)
            tot = sum(v for _, v in rows)
            strict = sum(v for d, v in rows if d["real"] == "true" and d["shadow"] == "false")
            loose = sum(v for d, v in rows if d["real"] == "false" and d["shadow"] == "true")
            sp = sum(v for d, v in counters("scheduler_fluidserve_shadowv_spread_total", cnt, variant=vn, tier=t))
            tg = counters("scheduler_fluidserve_shadowv_target_total", cnt, variant=vn, tier=t)
            ch = sum(v for d, v in tg if d["changed"] == "true"); tgt = sum(v for _, v in tg)
            print(f"    {vn:7s} 결정 {tot:8.0f} | 더 엄격 {strict:7.0f} | 더 느슨 {loose:7.0f} | spread {sp:6.0f} | 인스턴스 변경 {100*ch/max(tgt,1):5.1f}%")
    print("T (실제 불가인 결정에서 pend/shed):")
    for t, c in TIER.items():
        rows = counters("scheduler_fluidserve_shadowv_wait_total", cnt, tier=t)
        m = {(d["real"], d["t"]): v for d, v in rows}
        print(f"  {c}: 실제 pend→T pend {m.get(('pend','pend'),0):.0f}, 실제 shed→T pend {m.get(('shed','pend'),0):.0f}, 실제 pend→T shed {m.get(('pend','shed'),0):.0f}, 실제 shed→T shed {m.get(('shed','shed'),0):.0f}")
    mask = counters("scheduler_fluidserve_infeasible_mask_total", cnt)
    print("실제 판정에서 같이 걸린 조건 조합(후보 단위, 상위):")
    agg = collections.Counter()
    for d, v in mask: agg[(TIER.get(d["tier"], d["tier"]), d["mask"])] += v
    for (c, m), v in agg.most_common(12): print(f"  {c:12s} {m:30s} {v:9.0f}")
    # 3. 요청 단위
    rx = re.compile(r"dispatch request (\S+) fsshadowv tier=(\d+) decision=(\w+) waited=(-?\d+) attempts=(\d+) first=(\S+) last=(\S+) tpend=(\d+) tshed=(\d+)")
    recs = []
    for l in open(RES + run + "/server_metrics/scheduler_dispatch.log"):
        m = rx.search(l)
        if not m: continue
        first = dict((k, int(v)) for k, v in (x.split(":") for x in m.group(6).split(",")))
        last = dict((k, int(v)) for k, v in (x.split(":") for x in m.group(7).split(",")))
        recs.append(dict(id=m.group(1), tier=m.group(2), decision=m.group(3), attempts=int(m.group(5)),
                         tpend=int(m.group(8)), tshed=int(m.group(9)), **{"first_" + k: v for k, v in first.items()},
                         **{"last_" + k: v for k, v in last.items()}))
    L = pd.DataFrame(recs)
    ids = {}
    for l in open(RES + run + "/request_ids.jsonl"):
        i = json.loads(l); ids[re.sub(r"^(chat)?cmpl-", "", i["request_id"])] = (i["task_id"], i["call_index"])
    q = pd.read_csv(CACHE + run + "/requests.csv.gz")
    q = q[q.cutoff == 0].drop_duplicates(["task_id", "call_index"])
    L["key"] = L.id.map(ids)
    L = L.dropna(subset=["key"])
    L["task_id"] = L.key.map(lambda k: k[0]); L["call_index"] = L.key.map(lambda k: k[1])
    J = L.merge(q[["task_id", "call_index", "ladder_ok", "rejected"]], on=["task_id", "call_index"], how="inner")
    print(f"\n요청 단위: fsshadowv 줄 {len(recs)}, 채점 결과와 맞춘 것 {len(J)}")
    for t, c in TIER.items():
        x = J[J.tier == t]; met = x[(x.decision == "route") & (x.ladder_ok == 1)]; shed = x[x.decision == "shed"]
        print(f"  {c}: 받아 마감 지킴 {len(met)}, 거절 {len(shed)}")
        for vn in VARS:
            a = 100 * (met["last_" + vn] == 0).mean() if len(met) else float("nan")
            b = 100 * (shed["first_" + vn] >= 0).mean() if len(shed) else float("nan")
            print(f"    {vn:7s} (가) 마지막 시도에서 불가 {a:5.2f}%   (나) 거절된 요청 중 가능했던 것 {b:5.1f}%")
        tp = 100 * (shed.tpend > 0).mean() if len(shed) else float("nan")
        print(f"    T       거절된 요청 중 T였다면 한 번이라도 pend였을 것 {tp:5.1f}%")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
