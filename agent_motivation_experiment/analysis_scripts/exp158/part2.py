"""EXP-158 보충: (나)와 T를 결정 로그의 fsshadowv shed 줄만으로 계산(거절된 요청은 request_ids.jsonl에 없어 채점 결과와 맞출 수 없다),
그리고 결정 단위 카운터를 측정 시작 후 45분까지만 다시 계산(마지막 15분의 갈림을 뺀 확인). 사용: python3 part2.py <run>"""
import json, re, sys, collections
RES = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
REC = "/home/nxclab/scratch/dashboard-cache/recorder/"
VARS = ["same", "G1.0", "U", "U+G1.0", "M0", "M1", "U+M0"]
TIER = {"50": "chat", "100": "deepresearch", "25": "swe"}
run = sys.argv[1]
rx = re.compile(r"fsshadowv tier=(\d+) decision=(\w+) waited=(-?\d+) attempts=(\d+) first=(\S+) last=(\S+) tpend=(\d+) tshed=(\d+)")
sheds = collections.defaultdict(list)
for l in open(RES + run + "/server_metrics/scheduler_dispatch.log"):
    m = rx.search(l)
    if m and m.group(2) == "shed":
        first = dict((k, int(v)) for k, v in (x.split(":") for x in m.group(5).split(",")))
        sheds[m.group(1)].append((first, int(m.group(7))))
print("(나) 거절된 요청 가운데 그 변형이면 어느 시도에서든 가능한 인스턴스가 있었던 비율, 그리고 T였다면 한 번이라도 pend였을 비율")
for t, c in TIER.items():
    x = sheds.get(t, [])
    print(f"  {c:12s} 거절 {len(x):6d} | " + " ".join(f"{v} {100*sum(1 for f,_ in x if f[v]>=0)/max(len(x),1):5.1f}%" for v in VARS)
          + f" | T {100*sum(1 for _,tp in x if tp>0)/max(len(x),1):5.1f}%")
rows = [json.loads(l) for l in open(REC + run + "/fluidserve.jsonl")]
t0 = rows[0]["t"]
def at(sec):
    best = None
    for r in rows:
        if r["t"] - t0 <= sec: best = r
    return best
a, b = at(60), at(45 * 60)
def diff(key):
    return b.get(key, 0) - a.get(key, 0)
print("\n측정 1~45분만: 클래스마다 변형별 더 엄격/더 느슨 결정 수와 spread (실제 spread)")
for t, c in TIER.items():
    real = diff(f"scheduler_fluidserve_shadowv_spread_total|tier={t},variant=real")
    out = []
    for v in VARS:
        st = diff(f"scheduler_fluidserve_shadowv_decision_total|real=true,shadow=false,tier={t},variant={v}")
        lo = diff(f"scheduler_fluidserve_shadowv_decision_total|real=false,shadow=true,tier={t},variant={v}")
        sp = diff(f"scheduler_fluidserve_shadowv_spread_total|tier={t},variant={v}")
        out.append(f"{v}: 엄격 {st:.0f} 느슨 {lo:.0f} spread {sp:.0f}")
    print(f"  {c} (실제 spread {real:.0f})\n    " + "\n    ".join(out))
