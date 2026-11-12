"""EXP-157 agent2: crude apportionment of the excess rejections (c2 h100 sheds minus v0.5 sheds, same repeat, same 5-min window)
to the tests, using each window's share of candidate-level sole blocks by reason in the c2 run. Parses windows.txt."""
import re, collections
blocks = open("windows.txt").read().split("== ")[1:]
T = {}
for b in blocks:
    lines = b.strip().splitlines(); run = lines[0].strip(); rows = {}
    for l in lines[2:]:
        p = l.replace("|", " ").split()
        rows[int(p[0])] = dict(n=int(p[1]), shed=float(p[2]), g=float(p[6]), i=float(p[7]), m=float(p[8]))
    T[run] = rows
for v5, c2 in [("261110_0354_exp157hr2_fsexactprog_shift", "261110_0619_exp157hr2_fsc2h100_shift"),
               ("261109_2311_exp157hr1_fsexactprog_shift", "261110_0133_exp157hr1_fsc2h100_shift")]:
    acc = collections.Counter(); tot_ex = 0; tot_n = 0
    for w, r in T[c2].items():
        ex = (r["shed"] - T[v5][w]["shed"]) / 100 * r["n"]; tot_ex += ex; tot_n += r["n"]
        s = r["g"] + r["i"] + r["m"]
        if s <= 0: continue
        for k in "gim": acc[k] += ex * r[k] / s
    print(f"{c2} vs {v5}: excess sheds {tot_ex:.0f} of {tot_n} arrivals ({100*tot_ex/tot_n:.1f} points); apportioned gate {acc['g']:.0f} ({100*acc['g']/tot_ex:.0f}%), incumbents {acc['i']:.0f} ({100*acc['i']/tot_ex:.0f}%), memory {acc['m']:.0f} ({100*acc['m']/tot_ex:.0f}%)")
