"""Why a policy stops short of, or runs past, the SLO-feasible operating point: what it predicted or
counted at decision time against what the engines then did. One section per mechanism.

  memory   PolyServe's section 4.5 memory test. Per scheduler tick (1 s), fleet sums of
           logical decode tokens (instance_cms_all_decodes_tokens_num, prefix sharing undone),
           physical KV in use (engine vllm:kv_cache_usage_perc x NumTotalGpuTokens), and the
           growth term (decode batch + 1) x expected output of the tier. The test refuses when
           logical + prompt + growth > capacity on every candidate.
  pace     the step time the policy predicted for the instance it chose (PolyServe
           polyserve_iter_max = steady state with the batch grown, prefill left out;
           Llumnix SLO predicted_tpot = one step after adding the request), per 5-minute
           window, against the engines' measured inter-token latency in the same window.
  engines  per-engine inter-token latency per 5-minute window: how evenly the fleet runs at
           its pace (a mean at the budget with a wide spread puts part of the fleet over it).

usage: root_cause.py memory|pace|engines <run>...
"""
import glob, json, re, sys
import numpy as np

RES = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
CAP = 584928.0          # NumTotalGpuTokens per instance (scheduler_polyserve_kv_capacity_tokens)
EXPECT = {"onlychat25": 428, "onlyswe": 494, "onlydr": 985}


def rows(path):
    return [json.loads(l) for l in open(path)]


def fsum(r, key):
    return sum(v for k, v in r.items() if k.split("|")[0] == key and isinstance(v, (int, float)))


def per_instance(r, key, live=None):
    # instance_id=<id>; the scheduler keeps the ids of instances from before an engine restart,
    # which read 0 for the whole run, so callers pass the set of ids that were ever non-zero.
    out = {k.split("instance_id=")[1].split(",")[0]: v for k, v in r.items()
           if k.split("|")[0] == key and "instance_id=" in k and isinstance(v, (int, float))}
    return {i: v for i, v in out.items() if live is None or i in live}


def engine_rows(run):
    return [rows(f) for f in sorted(glob.glob(RES + run + "/server_metrics/engine_*.jsonl"))]


def memory(run):
    sch = rows(RES + run + "/server_metrics/scheduler.jsonl")
    eng = engine_rows(run)
    t0 = sch[0]["t"]
    et = [np.array([x["t"] for x in e]) for e in eng]
    ek = [np.array([fsum(x, "vllm:kv_cache_usage_perc") for x in e]) for e in eng]
    variant = [v for v in EXPECT if run.endswith(v)]
    exp_out = EXPECT[variant[0]] if variant else 494
    live = set()
    for r in sch:
        live |= {i for i, v in per_instance(r, "instance_cms_all_decodes_tokens_num").items() if v > 0}
    L, P, G, R = [], [], [], []
    for r in sch:
        if not (300 <= r["t"] - t0 <= 3300):
            continue
        logical = per_instance(r, "instance_cms_all_decodes_tokens_num", live)
        batch = per_instance(r, "instance_cms_decode_batch_size", live)
        if len(logical) != 4:
            continue
        phys = sum(CAP * k[min(np.searchsorted(t, r["t"]), len(t) - 1)] for t, k in zip(et, ek))
        lg = sum(logical.values()); gr = sum((b + 1) * exp_out for b in batch.values())
        L.append(lg); P.append(phys); G.append(gr); R.append(max((logical[i] + (batch.get(i, 0) + 1) * exp_out) / CAP for i in logical))
    L, P, G, R = map(np.array, (L, P, G, R))
    print(f"{run}\n  per instance, median over ticks: physical KV {np.median(P)/4/CAP:.1%} of capacity, "
          f"logical decode tokens {np.median(L)/4/CAP:.1%}, growth term (batch+1) x {exp_out} {np.median(G)/4/CAP:.1%}"
          f"\n  logical / physical {np.median(L/P):.2f};  highest instance projection / capacity: median {np.median(R):.2f}, p90 {np.percentile(R,90):.2f}")


def pace(run):
    log = open(RES + run + "/server_metrics/scheduler_dispatch.log").read().splitlines()
    pat = re.compile(r"^I(\d\d)(\d\d) (\d\d):(\d\d):(\d\d)\.\d+ .*dispatch request \S+ prediction (.*) inst=")
    pred = {}
    first = None
    for l in log:
        m = pat.search(l)
        if not m:
            continue
        sec = int(m.group(3)) * 3600 + int(m.group(4)) * 60 + int(m.group(5))
        first = sec if first is None else first
        kv = dict(x.split("=") for x in m.group(6).split())
        key = "polyserve_iter_max" if "polyserve_iter_max" in kv else "predicted_tpot"
        pred.setdefault((sec - first) // 300, []).append(float(kv[key]))
    eng = engine_rows(run); t0 = min(e[0]["t"] for e in eng)
    itl = {}
    for e in eng:
        for w in range(13):
            sel = [x for x in e if w * 300 <= x["t"] - t0 < (w + 1) * 300]
            if len(sel) > 2:
                s = fsum(sel[-1], "vllm:inter_token_latency_seconds_sum") - fsum(sel[0], "vllm:inter_token_latency_seconds_sum")
                c = fsum(sel[-1], "vllm:inter_token_latency_seconds_count") - fsum(sel[0], "vllm:inter_token_latency_seconds_count")
                a = itl.setdefault(w, [0, 0]); a[0] += s; a[1] += c
    print(f"{run}  ({key}, mean over dispatched requests  vs  engine ITL, per 5-min window, ms)")
    print("  " + "  ".join(f"{np.mean(pred[w]):5.1f}/{1000*itl[w][0]/itl[w][1]:5.1f}" for w in sorted(pred)[:12] if w in itl))


def engines(run):
    eng = engine_rows(run); t0 = min(e[0]["t"] for e in eng)
    grid = []
    for e in eng:
        row = []
        for w in range(1, 12):
            sel = [x for x in e if w * 300 <= x["t"] - t0 < (w + 1) * 300]
            s = fsum(sel[-1], "vllm:inter_token_latency_seconds_sum") - fsum(sel[0], "vllm:inter_token_latency_seconds_sum")
            c = fsum(sel[-1], "vllm:inter_token_latency_seconds_count") - fsum(sel[0], "vllm:inter_token_latency_seconds_count")
            row.append(1000 * s / max(c, 1))
        grid.append(row)
    g = np.array(grid)
    print(f"{run}\n  per engine, 5-min windows 1..11, ITL ms:")
    for i, r in enumerate(g):
        print(f"    engine {i}: " + " ".join(f"{x:5.1f}" for x in r))
    print(f"  spread across engines within a window: mean {np.mean(g.max(0)-g.min(0)):.1f} ms, max {np.max(g.max(0)-g.min(0)):.1f} ms")


if __name__ == "__main__":
    fn = {"memory": memory, "pace": pace, "engines": engines}[sys.argv[1]]
    for run in sys.argv[2:]:
        fn(run.rstrip("/").split("/")[-1])
