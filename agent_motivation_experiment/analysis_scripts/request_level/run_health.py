#!/usr/bin/env python3
"""One line per finished condition, with the things that go wrong flagged.

Written for long sweeps, where a cell can fail in a way that leaves a plausible
looking result directory behind and is only noticed hours later. Each check
below corresponds to something that has actually happened on this cluster:

  rate      the load generator not reaching the target rate, so the condition
            measured a different load than its name says
  engines   fewer than four engine metric files, or one that stopped reporting,
            which means the condition ran on a smaller fleet
  n         a request count far from rate x duration, which is the same failure
            seen from the client side
  attain    exactly 0 or exactly 100 with a large n, which is usually a policy
            that never started rather than a real result
  reject    100%, which is a policy rejecting everything

Usage
-----
  python3 run_health.py 'results/*exp37*'
"""
import glob
import json
import os
import re
import sys


def rate_of(d):
    m = re.search(r"_rpm_(\d+)", os.path.basename(d))
    return int(m.group(1)) / 60.0 if m else None


def engines_ok(d):
    """(count of engine files, count that were still advancing at the end)."""
    files = sorted(glob.glob(os.path.join(d, "server_metrics", "engine_*.jsonl")))
    live = 0
    for f in files:
        rows = [x for x in open(f) if x.strip()]
        if len(rows) < 10:
            continue
        try:
            a, b = json.loads(rows[0]), json.loads(rows[-1])
        except Exception:
            continue
        key = next((k for k in b if isinstance(k, str)
                    and k.startswith("vllm:generation_tokens_total")), None)
        if key and b.get(key, 0) > a.get(key, 0):
            live += 1
    return len(files), live


def main():
    pats = sys.argv[1:] or ["results/*exp37*"]
    dirs = sorted({d for p in pats for d in glob.glob(p) if os.path.isdir(d)})
    if not dirs:
        print("no run directories yet")
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from exp22_fluidserve import load_run, per_request, arm_of

    print(f"{'run':<46}{'arm':<12}{'tgt/s':>7}{'got/s':>7}"
          f"{'n':>8}{'adm':>7}{'rej%':>7}{'eng':>6}  flags")
    print("-" * 108)
    for d in dirs:
        try:
            r = load_run(d)
        except Exception as e:
            print(f"{os.path.basename(d)[:45]:<46}  LOAD FAILED: {e}")
            continue
        if r is None or r.empty:
            print(f"{os.path.basename(d)[:45]:<46}  EMPTY")
            continue
        tgt = rate_of(d)
        span = r["rel"].max() - r["rel"].min()
        got = len(r) / span if span > 0 else 0
        adm = per_request(r, "violate_served")
        rej = 100.0 * r["rejected"].mean()
        nf, nlive = engines_ok(d)

        flags = []
        if tgt and abs(got - tgt) / tgt > 0.10:
            flags.append(f"RATE off by {100*(got-tgt)/tgt:+.0f}%")
        if nf != 4:
            flags.append(f"ENGINE files {nf}")
        elif nlive != 4:
            flags.append(f"ENGINE idle {4-nlive}")
        if len(r) < 200:
            flags.append("N tiny")
        if rej >= 99.9:
            flags.append("REJECT all")
        if len(r) > 1000 and adm in (0.0, 100.0) and rej < 1:
            flags.append(f"ATTAIN exactly {adm:.0f}")

        print(f"{os.path.basename(d)[:45]:<46}{arm_of(d)[:11]:<12}"
              f"{tgt or 0:7.0f}{got:7.1f}{len(r):8d}{adm:7.1f}{rej:7.1f}"
              f"{nlive:3d}/{nf:<2d}  {'; '.join(flags) or 'ok'}")


if __name__ == "__main__":
    main()
