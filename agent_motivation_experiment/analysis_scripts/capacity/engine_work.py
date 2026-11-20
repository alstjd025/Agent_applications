"""How much of the cluster's decode capacity each run used, and how much of that work was useful.

Per run, from server_metrics/engine_<port>.jsonl (vLLM's own counters, scraped about once a second),
over minutes 5..55 of the hour (warm-up and drain excluded):
  gen_tok_s      output tokens the four engines generated per second (all requests, on time or not)
  prompt_tok_s   prompt tokens prefilled per second (after prefix-cache hits)
  hit            prefix-cache hit ratio
  running        mean requests in the running batch per engine
  waiting        mean requests waiting in the engine queue per engine
  kv_mean/kv_p90 KV-cache usage per engine (fraction of physical blocks)
  itl_ms         engine-measured mean inter-token latency (decode step time seen by a request)
  preempt        preemptions over the whole run (all engines)
and from the run's observe.py output (ladder95): goodput tokens/s, which is output tokens that met
their deadline. useful = goodput / gen_tok_s is the share of the engines' output that counted.

usage: engine_work.py <obs_dir_or_file_glob>... -- <run_dir>...
"""
import glob, json, os, re, sys
import numpy as np

RES = "/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment/results/"
OBS_DIRS = ["/home/nxclab/scratch/" + d for d in ("exp161", "exp164", "exp165", "exp165l", "exp166", "exp167", "exp168", "exp170", "exp172/obs", "capacity")]
LO, HI = 300, 3300


def series(rows, key):
    t, v = [], []
    for r in rows:
        x = sum(val for k, val in r.items() if k.split("|")[0] == key and isinstance(val, (int, float)))
        t.append(r["t"]); v.append(x)
    return np.array(t), np.array(v, dtype=float)


def window(t, v, t0):
    m = (t - t0 >= LO) & (t - t0 <= HI)
    return t[m], v[m]


def rate(t, v):
    return (v[-1] - v[0]) / (t[-1] - t[0]) if len(t) > 1 else float("nan")


def engine_stats(run):
    out = {"gen": 0.0, "prompt": 0.0, "hitq": 0.0, "hitn": 0.0, "itl_s": 0.0, "itl_c": 0.0, "pre": 0.0,
           "running": [], "waiting": [], "kv": []}
    files = sorted(glob.glob(RES + run + "/server_metrics/engine_*.jsonl"))
    t0 = None
    for f in files:
        rows = [json.loads(l) for l in open(f)]
        if t0 is None:
            t0 = rows[0]["t"]
        for key, name in (("vllm:generation_tokens_total", "gen"), ("vllm:prompt_tokens_total", "prompt")):
            t, v = window(*series(rows, key), t0); out[name] += rate(t, v)
        for key, name in (("vllm:prefix_cache_hits_total", "hitn"), ("vllm:prefix_cache_queries_total", "hitq"),
                          ("vllm:inter_token_latency_seconds_sum", "itl_s"), ("vllm:inter_token_latency_seconds_count", "itl_c")):
            t, v = window(*series(rows, key), t0); out[name] += v[-1] - v[0]
        t, v = series(rows, "vllm:num_preemptions_total"); out["pre"] += v[-1] - v[0]
        for key, name in (("vllm:num_requests_running", "running"), ("vllm:num_requests_waiting", "waiting"), ("vllm:kv_cache_usage_perc", "kv")):
            t, v = window(*series(rows, key), t0); out[name].append(v)
    return out, len(files)


def goodput(run):
    for d in OBS_DIRS:
        p = os.path.join(d, f"obs_{run}.txt")
        if os.path.exists(p):
            s = re.search(r"점수: 모든 도착 ([\d.]+), admitted ([\d.]+), 거절률 ([\d.]+) %, goodput (\d+)", open(p).read())
            if s:
                return tuple(float(x) for x in s.groups())
    return None


if __name__ == "__main__":
    runs = [os.path.basename(r.rstrip("/")) for r in sys.argv[1:]]
    print("run\tall\tadm\trej\tgoodput\tgen_tok_s\tuseful\tprompt_tok_s\thit\trunning\twaiting\tkv_mean\tkv_p90\titl_ms\tpreempt")
    for run in runs:
        e, n = engine_stats(run)
        g = goodput(run) or (float("nan"),) * 4
        run_ = np.concatenate(e["running"]); wait = np.concatenate(e["waiting"]); kv = np.concatenate(e["kv"])
        print("\t".join([run, f"{g[0]:.2f}", f"{g[1]:.2f}", f"{g[2]:.1f}", f"{g[3]:.0f}", f"{e['gen']:.0f}",
                         f"{g[3]/e['gen']:.3f}", f"{e['prompt']:.0f}", f"{e['hitn']/max(e['hitq'],1):.3f}",
                         f"{run_.mean():.1f}", f"{wait.mean():.1f}", f"{kv.mean():.3f}", f"{np.percentile(kv,90):.3f}",
                         f"{1000*e['itl_s']/max(e['itl_c'],1):.1f}", f"{e['pre']:.0f}"]))
