#!/usr/bin/env python3
"""EXP-139 -- the SLO scale sweep at 35 req/s, scored at whatever has finished.

RUN IT AT ANY TIME. It reads only the run directories that exist, prints n per
cell, and names the cells still missing, so a stage can be judged the moment it
completes instead of at the end of the chain.

WHERE THE SCORING RULE COMES FROM, AND WHY NOT FROM A TABLE HERE. Each scale has
six budgets (three classes x TTFT and per-token). They are already written down
in the workload configuration the requests were built from --
`workload_configs/mix_short_m1_k<K>fair.json` -- so this script reads them from
there rather than carrying a second copy that could drift. The `fair` variant is
the right one for BOTH arms: it states the per-token promise literally, whereas
the FluidServe file carries 25/50/100 in that field as the TIER KEY that names
each class in the length profile.

WHY IT RE-EXECS ITSELF. `exp22_fluidserve` reads its budgets into module-level
constants at IMPORT time, so one process can score exactly one scale. The parent
finds which scales are on disk and runs one child per scale with that scale's six
values in the environment; the child imports, scores, and prints. Setting the
variables after the import does nothing and would score every scale at the
default budgets in silence.

  python3 exp139_slo_scale.py
  python3 exp139_slo_scale.py --runs 'results/*exp139k*'
"""
import argparse, glob, json, os, subprocess, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
EXPROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

ARMS = ["fsv3capgnofrct75k", "llmdslot75", "polyservept75", "slot75"]
ARM_LABEL = {"fsv3capgnofrct75k": "FluidServe", "llmdslot75": "llm-d",
             "polyservept75": "PolyServe", "slot75": "Llumnix SLO"}
KTAGS = ["070", "080", "090", "100", "110", "120", "130"]
# Spelled out: "0."+ktag.lstrip("0") gives 0.11 for 110 and 0. for 100.
KLBL = {"070": "0.7", "080": "0.8", "090": "0.9", "100": "1.0",
        "110": "1.1", "120": "1.2", "130": "1.3"}
REPS = ["1", "2"]


def rule_env(ktag):
    """The six budgets for one scale, read out of that scale's workload file."""
    p = os.path.join(EXPROOT, "workload_configs", f"mix_short_m1_k{ktag}fair.json")
    slo = json.load(open(p))["slo"]
    return {
        "FS_CHAT_TTFT_S": f'{slo["chat"]["ttft_ms"]/1000:g}',
        "FS_DR_TTFT_S":   f'{slo["deepresearch"]["ttft_ms"]/1000:g}',
        "FS_SWE_TTFT_S":  f'{slo["swe"]["ttft_ms"]/1000:g}',
        "FS_CHAT_TBT_MS": str(slo["chat"]["tbt_ms"]),
        "FS_DR_TBT_MS":   str(slo["deepresearch"]["tbt_ms"]),
        "FS_SWE_TBT_MS":  str(slo["swe"]["tbt_ms"]),
    }


def find_runs(pattern):
    out = collections.defaultdict(list)          # (ktag, arm) -> [dirs]
    for d in sorted(glob.glob(os.path.join(EXPROOT, pattern))):
        b = os.path.basename(d)
        if "PRERUN" in b:                        # llm-d's discarded warm-up
            continue
        for kt in KTAGS:
            if f"exp139k{kt}r" not in b:
                continue
            for a in ARMS:
                if f"_{a}_" in b:
                    out[(kt, a)].append(d)
            break
    return out


def child(ktag, pattern):
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, ".."))
    # Reuse the repository's own scorer rather than a second copy of the
    # denominators: one_run() returns all three plus the rejection rate and token
    # goodput, and it is the path every other EXP-13x table was built from.
    import all_arrivals_attainment as A                      # noqa: E402
    from exp22_fluidserve import SLO_RULES                    # noqa: E402
    runs = find_runs(pattern)
    print(f"\n### k = {KLBL[ktag]}   rule in force: {SLO_RULES}")
    print(f"  {'arm':<14}{'n':>3}  {'all arrivals':>16}{'admitted':>16}"
          f"{'offered':>16}{'rejected %':>16}{'goodput':>18}")
    for a in ARMS:
        ds = runs.get((ktag, a), [])
        if not ds:
            print(f"  {ARM_LABEL[a]:<14}{0:>3}  {'-- not run --':>16}")
            continue
        cells = [r for r in (A.one_run(d) for d in ds) if r]
        if not cells:
            print(f"  {ARM_LABEL[a]:<14}{0:>3}  {'-- no rows --':>16}")
            continue
        def col(key, fmt="{:.1f}"):
            return "/".join(fmt.format(c[key]) for c in cells)
        print(f"  {ARM_LABEL[a]:<14}{len(cells):>3}  {col('all_arrivals'):>16}"
              f"{col('admitted'):>16}{col('offered'):>16}"
              f"{col('rejected_pct'):>16}{col('goodput_tok_s','{:.0f}'):>18}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/*exp139k*")
    ap.add_argument("--ktag", help="internal: score one scale (set by the parent)")
    a = ap.parse_args()
    if a.ktag:
        child(a.ktag, a.runs)
        return
    runs = find_runs(a.runs)
    have = sorted({kt for kt, _ in runs})
    if not have:
        print(f"no EXP-139 runs matched {a.runs}")
        return
    print(f"EXP-139 -- 35 req/s SLO scale sweep. scales on disk: "
          f"{', '.join(KLBL[k] for k in have)}", flush=True)
    for kt in KTAGS:
        if kt not in have:
            continue
        env = dict(os.environ); env.update(rule_env(kt))
        subprocess.run([sys.executable, os.path.abspath(__file__),
                        "--ktag", kt, "--runs", a.runs], env=env)
    print("\n### still missing")
    # k=1.0 is not part of this chain: EXP-108 measured that cell for all four
    # arms with two repeats at the same 2100 rpm, and those numbers are in
    # EXP-139 section 4.1. Listing it as "missing" would read as a gap in this
    # run rather than a decision.
    # 1.0 comes from EXP-108; 0.8 and 1.2 are a PolyServe-only probe
    # (EXP-139 addendum) and the other three arms were never planned there.
    planned = [k for k in KTAGS if k not in ("100", "080", "120")]
    miss = [f"k={KLBL[kt]} {ARM_LABEL[arm]} (have {len(runs.get((kt,arm),[]))}/2)"
            for kt in planned for arm in ARMS if len(runs.get((kt, arm), [])) < 2]
    print("  " + ("\n  ".join(miss) if miss else "nothing -- all 40 cells present"))


if __name__ == "__main__":
    main()
