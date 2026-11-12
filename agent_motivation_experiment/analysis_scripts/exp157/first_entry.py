"""EXP-157: chat이 퍼지는 경로. chat이 없던 인스턴스(gate_allowance_ms > 50)에 chat이 처음 route된 순간을 모두 찾고,
그때의 인스턴스 iteration 시간과 새 chat의 허용치(50 + (5000 − waited − prefillest) / 428)를 비교한다.
또 시간마다 chat을 가진 인스턴스 수, chat을 가진 상태가 얼마나 오래 유지되는지를 센다.
입력: dashboard 기록기 gauge(2초), scheduler_dispatch.log의 fsplacement route 줄."""
import json, re, sys, bisect, collections, numpy as np, calendar
RES = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
REC = "/home/nxclab/scratch/dashboard-cache/recorder/"
TIER = {"50": "chat", "100": "deepresearch", "25": "swe"}
rx = re.compile(r"^I(\d\d)(\d\d) (\d\d):(\d\d):(\d\d\.\d+).*fsplacement tier=(\d+) waited=(\d+) prefillest=([\d.]+).*decision=route inst=(\d+)")
def run(r):
    year = 2000 + int(r[:2])
    rows = [json.loads(l) for l in open(REC + r + "/fluidserve.jsonl")]
    t0 = rows[0]["t"]
    g = collections.defaultdict(lambda: ([], {}))
    for x in rows:
        for k, v in x.items():
            if "|instance=" in k:
                name, inst = k.split("|instance=")
                if name in ("scheduler_fluidserve_gate_allowance_ms", "scheduler_fluidserve_observed_step_ms"):
                    g[inst][1].setdefault(name, []).append((x["t"], v))
    def at(inst, name, t):
        s = g[inst][1].get(name, [])
        i = bisect.bisect_right([a for a, _ in s], t) - 1
        return s[i][1] if i >= 0 else None
    insts = sorted(g)
    # 시간마다 chat을 가진 인스턴스 수 (측정 구간 60~3540 s)
    ts = [x["t"] for x in rows if 60 < x["t"] - t0 < 3540]
    nchat = [sum(1 for i in insts if (at(i, "scheduler_fluidserve_gate_allowance_ms", t) or 1e9) <= 50) for t in ts[::5]]
    first, ride, other, ride_w = [], 0, 0, []
    for l in open(RES + r + "/server_metrics/scheduler_dispatch.log"):
        m = rx.match(l)
        if not m: continue
        mo, dd, hh, mi, ss, tier, waited, pest, inst = m.groups()
        t = calendar.timegm((year, int(mo), int(dd), int(hh), int(mi), 0)) + float(ss)
        if not (60 < t - t0 < 3540) or tier != "50": continue
        ga = at(inst, "scheduler_fluidserve_gate_allowance_ms", t)
        if ga is None: continue
        if ga > 50:
            step = at(inst, "scheduler_fluidserve_observed_step_ms", t)
            own = 50 + (5000 - int(waited) - float(pest)) / 428
            first.append((t - t0, inst, ga, step, own, int(waited)))
        else:
            ride += 1; ride_w.append(int(waited))
    f = np.array([(a[3] or np.nan, a[4], a[2]) for a in first]) if first else np.zeros((0, 3))
    nc = np.array(nchat)
    print(f"{r[12:-6]:26s} chat route {ride+len(first):6d} | chat 없는 인스턴스로 {len(first):5d} ({100*len(first)/max(ride+len(first),1):.1f}%) "
          f"그 인스턴스 iteration p50 {np.nanmedian(f[:,0]) if len(f) else float('nan'):5.1f} ms, 새 chat 허용치 p50 {np.median(f[:,1]) if len(f) else float('nan'):5.1f} ms, "
          f"iteration>45ms인 것 {int((f[:,0]>45).sum()) if len(f) else 0} | chat 가진 인스턴스 수 분포 1:{(nc==1).mean():.2f} 2:{(nc==2).mean():.2f} 3:{(nc==3).mean():.2f} 4:{(nc==4).mean():.2f}")
    # chat 보유 구간: 인스턴스의 gate_allowance가 50 이하로 바뀐 때부터 50 초과로 돌아간 때까지(2초 gauge)
    dur = collections.defaultdict(list)
    for i in insts:
        s_ = [(a, v) for a, v in g[i][1].get("scheduler_fluidserve_gate_allowance_ms", []) if 60 < a - t0 < 3540]
        on = None
        for a, v in s_:
            if v <= 50 and on is None: on = a
            elif v > 50 and on is not None: dur[i].append(a - on); on = None
        if on is not None: dur[i].append(s_[-1][0] - on)
    fw = np.array([a[5] for a in first]); rw = np.array(ride_w)
    print(f"    첫 진입 chat의 waited: >0인 비율 {100*(fw>0).mean():.0f}%, 중앙값 {np.median(fw):.0f} ms | 이미 chat 있는 인스턴스로 간 chat: >0 {100*(rw>0).mean():.0f}%, 중앙값 {np.median(rw):.0f} ms")
    print("    인스턴스별 chat 보유 구간(초): " + " | ".join(f"{i[-4:]}: {len(v)}회, 합 {sum(v):.0f}s, 최장 {max(v) if v else 0:.0f}s" for i, v in sorted(dur.items())))
    return first
for r in sys.argv[1:]:
    run(r)
