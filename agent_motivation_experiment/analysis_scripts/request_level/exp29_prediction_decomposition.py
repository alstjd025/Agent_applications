#!/usr/bin/env python3
"""EXP-29 - where the gap between the predicted and the measured iteration is.

FluidServe gates every placement on a predicted mean iteration time. At 80 req/s
that prediction reads about 8 ms above what the engine actually runs, the gate
sits between the two, and the result is that 84% of decisions become holds while
the fleet runs at 21% KV. Three attempts to name the cause from the two endpoints
alone were wrong, so this decomposes the prediction into its terms instead.

The prediction is assembled as

    meanStep = corr x [ dec + (sp/k) x (pre - c0) ]

and the measurement satisfies, by the definition of the duty cycle that the
prefill attribution uses,

    measured = decodeOnly + duty_instant x measured

where `dec` is the decode law at the batch in the CURRENT status and `decodeOnly`
is the same law at the batch that was running over the interval that was
measured. Because the arriving prefill is built from the duty cycle through the
same constant the model uses in reverse, the prefill term collapses:

    sp/k              = duty x pace / perChunk
    (sp/k)(pre - c0)  = duty x pace

and `pace` is the measured mean. So with dec == decodeOnly and corr == 1 the
prediction equals the measurement identically, and the gap can only come from

    (1) corr    != 1          the multiplicative correction has not settled at 1
    (2) dec     != decodeOnly the two are evaluated at different batches
    (3) pending  > arriving   effectivePrefill takes the queued branch and the
                              collapse above no longer holds

This script measures the size of each, per instance and over the run.

Usage
-----
  python3 exp29_prediction_decomposition.py results/<run-dir> [--from-min 2]
"""
import argparse
import collections
import glob
import json
import os
import sys

# Series this reads. Everything is a gauge sampled once per scheduler scrape.
PER_INSTANCE = [
    "scheduler_fluidserve_observed_step_ms",
    "scheduler_fluidserve_predicted_step_ms",
    "scheduler_fluidserve_decode_only_ms",
    "scheduler_fluidserve_decode_law_ms",
    "scheduler_fluidserve_pace_ms",
    "scheduler_fluidserve_prefill_duty",
    "scheduler_fluidserve_arriving_prefill_tokens",
    "scheduler_fluidserve_queued_prefill_tokens",
    "scheduler_fluidserve_obs_kv_tokens",
    "scheduler_fluidserve_obs_decode_batch",
    "scheduler_fluidserve_gate_allowance_ms",
]
GLOBAL = [
    "scheduler_fluidserve_correction",
    "scheduler_fluidserve_prefill_fraction",
]


def load(run_dir, from_min):
    """Read scheduler.jsonl into {series: {instance: [(t, v)]}}.

    Keys are "<name>|<label>=<value>,..." or the bare name when unlabelled.
    """
    path = os.path.join(run_dir, "server_metrics", "scheduler.jsonl")
    if not os.path.exists(path):
        sys.exit(f"no scheduler metrics in {run_dir}")
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    t0 = None
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        t = r.get("t")
        if t is None:
            continue
        if t0 is None:
            t0 = t
        # Skip the opening minutes: the correction starts at its initial value
        # and the duty cycle at zero, so an average over them describes the
        # transient rather than the state the decisions were made in.
        if (t - t0) < from_min * 60:
            continue
        for k, v in r.items():
            if not isinstance(k, str) or not k.startswith("scheduler_fluidserve_"):
                continue
            if not isinstance(v, (int, float)):
                continue
            name, _, lbl = k.partition("|")
            inst = "-"
            if lbl.startswith("instance="):
                inst = lbl.split("=", 1)[1]
            out[name][inst].append((t, v))
    return out


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def series_mean(d, name, inst=None):
    if name not in d:
        return float("nan")
    if inst is None:
        return mean([v for per in d[name].values() for _, v in per])
    return mean([v for _, v in d[name].get(inst, [])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--from-min", type=float, default=2.0,
                    help="drop this many minutes of transient (default 2)")
    a = ap.parse_args()

    d = load(a.run_dir, a.from_min)
    missing = [s for s in PER_INSTANCE + GLOBAL if s not in d]
    if missing:
        print("MISSING SERIES (binary predates the instrumentation?):")
        for m in missing:
            print("   ", m)
        if "scheduler_fluidserve_decode_only_ms" in missing:
            sys.exit("cannot decompose without decode_only_ms")

    corr = series_mean(d, "scheduler_fluidserve_correction")
    pfrac = series_mean(d, "scheduler_fluidserve_prefill_fraction")
    print(f"\n{os.path.basename(a.run_dir)}   (first {a.from_min:g} min dropped)")
    print(f"correction = {corr:.4f}    prefill_fraction = {pfrac:.4f}\n")

    insts = sorted(d["scheduler_fluidserve_observed_step_ms"].keys())
    hdr = (f"{'instance':>10}{'obs':>8}{'pred':>8}{'gap':>7}"
           f"{'dec':>8}{'decOnly':>9}{'duty':>7}{'pace':>8}"
           f"{'arriv':>9}{'queued':>9}{'gate':>8}")
    print(hdr)
    print("-" * len(hdr))
    agg = collections.defaultdict(list)
    for i in insts:
        row = {n: series_mean(d, n, i) for n in PER_INSTANCE}
        obs = row["scheduler_fluidserve_observed_step_ms"]
        pred = row["scheduler_fluidserve_predicted_step_ms"]
        dec = row["scheduler_fluidserve_decode_law_ms"]
        deco = row["scheduler_fluidserve_decode_only_ms"]
        duty = row["scheduler_fluidserve_prefill_duty"]
        pace = row["scheduler_fluidserve_pace_ms"]
        arr = row["scheduler_fluidserve_arriving_prefill_tokens"]
        que = row["scheduler_fluidserve_queued_prefill_tokens"]
        gate = row["scheduler_fluidserve_gate_allowance_ms"]
        print(f"{i[-8:]:>10}{obs:8.2f}{pred:8.2f}{pred-obs:7.2f}"
              f"{dec:8.2f}{deco:9.2f}{duty:7.3f}{pace:8.2f}"
              f"{arr:9.0f}{que:9.0f}{gate:8.1f}")
        for k, v in (("obs", obs), ("pred", pred), ("dec", dec), ("deco", deco),
                     ("duty", duty), ("pace", pace), ("arr", arr), ("que", que)):
            agg[k].append(v)

    m = {k: mean(v) for k, v in agg.items()}
    gap = m["pred"] - m["obs"]
    # (1) what the multiplicative correction contributes: the prediction is
    #     corr x (the uncorrected assembly), so removing it costs (corr-1)/corr
    #     of the prediction.
    c1 = m["pred"] * (corr - 1) / corr if corr else float("nan")
    # (2) what evaluating the decode law at the current status rather than at the
    #     batch that ran costs. Carried through the same correction.
    c2 = corr * (m["dec"] - m["deco"])
    # (3) what is left of the prefill term after the collapse duty x pace. Zero
    #     when effectivePrefill takes the arriving branch and the duty cycle has
    #     settled; positive when the queued branch wins.
    c3 = gap - c1 - c2

    print(f"\n{'':>10}{'mean':>8}")
    print(f"{'observed':>10}{m['obs']:8.2f}")
    print(f"{'predicted':>10}{m['pred']:8.2f}")
    print(f"{'GAP':>10}{gap:8.2f}\n")
    print("decomposition of the gap")
    print(f"  (1) correction  (corr={corr:.3f})        {c1:7.2f} ms"
          f"   {100*c1/gap if gap else 0:5.1f}%")
    print(f"  (2) batch       (dec-decOnly={m['dec']-m['deco']:.2f})  {c2:7.2f} ms"
          f"   {100*c2/gap if gap else 0:5.1f}%")
    print(f"  (3) prefill term residual              {c3:7.2f} ms"
          f"   {100*c3/gap if gap else 0:5.1f}%")
    print(f"\n  duty={m['duty']:.3f}  pace={m['pace']:.2f}  duty*pace="
          f"{m['duty']*m['pace']:.2f} ms")
    print(f"  arriving={m['arr']:.0f} tok   queued={m['que']:.0f} tok   "
          f"-> effective takes the "
          f"{'QUEUED' if m['que'] > m['arr'] else 'arriving'} branch")

    print("\nverdict (EXP-29 section 4 rule: a component >= 6 ms names the cause)")
    for name, val in (("correction loop", c1), ("evaluation batch", c2),
                      ("prefill term", c3)):
        if val >= 6.0:
            print(f"  -> {name}: {val:.2f} ms")
    if max(c1, c2, c3) < 6.0:
        print("  -> no single component reaches 6 ms; the decomposition is "
              "incomplete and must be rebuilt rather than guessed past")


if __name__ == "__main__":
    main()
