"""2단계(다): 클래스별 route가 그 순간 어떤 인스턴스(gate 50 = chat이 있음 / 75 / 100 / 비어 있음)로 갔나, 그리고 chat 인스턴스로 간 긴 프롬프트의 양."""
import json, sys, re, bisect, collections, datetime as dt
R = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
T = {"25": "swe", "50": "chat", "100": "deepresearch"}
for run in sys.argv[1:]:
    ts, gates = [], []
    for l in open(R + run + "/server_metrics/scheduler.jsonl"):
        x = json.loads(l); g = {}
        for k, v in x.items():
            if k.startswith("scheduler_fluidserve_gate_allowance_ms|instance="): g[k.split("=")[1]] = int(round(v)) if v and v > 0 else 0
        if g: ts.append(x["t"]); gates.append(g)
    day = dt.datetime.utcfromtimestamp(ts[0]).date()
    cnt = collections.defaultdict(collections.Counter); prompt_to_chat = collections.Counter(); n = 0
    for l in open(R + run + "/server_metrics/scheduler_dispatch.log", errors="replace"):
        if "fsplacement" not in l or "decision=route" not in l: continue
        m = re.match(r"I(\d\d)(\d\d) (\d\d):(\d\d):(\d\d\.\d+)", l)
        t = dt.datetime(day.year, int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), tzinfo=dt.timezone.utc).timestamp() + float(m.group(5))
        tier = re.search(r"tier=(\d+)", l).group(1); inst = re.search(r"inst=(\d+)", l).group(1); prompt = int(re.search(r"prompt=(\d+)", l).group(1))
        i = bisect.bisect_left(ts, t)
        if i >= len(ts): continue
        g = gates[i].get(inst, -1)
        cnt[T[tier]][g] += 1
        if g == 50 and tier != "50": prompt_to_chat[T[tier]] += prompt
        n += 1
    print(f"== {run[12:]}  route {n}")
    for c in ("chat", "deepresearch", "swe"):
        tot = sum(cnt[c].values())
        print(f"  {c:12s} route {tot:6d} → gate50 {100*cnt[c][50]/tot:5.1f}%  gate75 {100*cnt[c][75]/tot:5.1f}%  gate100 {100*cnt[c][100]/tot:5.1f}%  빈 인스턴스 {100*cnt[c][0]/tot:4.1f}%" +
              (f"   chat 인스턴스로 간 프롬프트 합 {prompt_to_chat[c]/1e6:.1f}M 토큰" if c != "chat" else ""))
