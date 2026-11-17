"""Relabel the classes of an existing dynamic trace to a fixed mix, keeping every arrival time.

EXP-162: the deployed hour trace (dyn60_shift_m2Am1B_b1045.csv) is chat-majority. This writes a trace with the SAME
arrival_s column (byte-identical) and a fixed class mix, so a run on it differs from a run on the source only in the
class of each arrival. Classes are drawn in shuffled fixed-composition blocks (block size = sum of weights; block b
shuffled with random.Random(seed*7919 + b)), so every block has the exact ratio, as the mixed workload's static mix does.
The `phase` column is kept; `segment` becomes `warmup_<name>` / `<name>`.

    python relabel_fixed_mix.py --src canonical/dyn60_shift_m2Am1B_b1045.csv \
        --out canonical/dyn60_mix532_b1045.csv --name mix532 --weights swe=5,deepresearch=3,chat=2 --seed 20261117
"""
import argparse, csv, json, random, hashlib, collections

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--name", required=True); ap.add_argument("--weights", required=True); ap.add_argument("--seed", type=int, required=True)
a = ap.parse_args()
w = [(k, int(v)) for k, v in (x.split("=") for x in a.weights.split(","))]
block = [k for k, n in w for _ in range(n)]
rows = list(csv.DictReader(open(a.src, newline="")))
out = []
for i, r in enumerate(rows):
    b, j = divmod(i, len(block))
    if j == 0:
        cur = block[:]; random.Random(a.seed * 7919 + b).shuffle(cur)
    seg = ("warmup_" + a.name) if r["phase"] == "warmup" else a.name
    out.append({"arrival_s": r["arrival_s"], "class": cur[j], "phase": r["phase"], "segment": seg})
with open(a.out, "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=["arrival_s", "class", "phase", "segment"]); wr.writeheader(); wr.writerows(out)
src_arr = hashlib.md5("".join(r["arrival_s"] + "\n" for r in rows).encode()).hexdigest()
new_arr = hashlib.md5("".join(r["arrival_s"] + "\n" for r in out).encode()).hexdigest()
cnt = collections.Counter(r["class"] for r in out)
plan = {"generator": "relabel_fixed_mix.py", "source_trace": a.src, "name": a.name, "weights": dict(w), "seed": a.seed,
        "rows": len(out), "class_counts": dict(cnt), "arrival_s_md5_source": src_arr, "arrival_s_md5_out": new_arr,
        "note": "arrival_s identical to the source; only class (and segment labels) changed"}
json.dump(plan, open(a.out.replace(".csv", ".plan.json"), "w"), indent=1)
print(json.dumps(plan, indent=1))
