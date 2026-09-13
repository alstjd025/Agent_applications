#!/usr/bin/env python3
"""Which classes were resident on which engine, per time window, for any run.

This is the data layer under the two paper figures that ask "how did each
control plane mix the classes across the fleet" -- `fig_class_mix_static.py`
(arrival-rate axis) and `fig_class_mix_hour.py` (time axis). Both import from
here so that one definition of residency serves both.

WHAT IS COUNTED, AND WHY IT IS RESIDENCY RATHER THAN DISPATCHES. A dispatch
count answers "where were requests sent"; the mechanism the figures are about is
"what was ON an engine at the same time", because an instance holding one
request of the tightest-budget class is held to that budget for everything else
it might take. So the quantity is TIME-AVERAGED RESIDENCY: for window w and
engine e and class c,

    resident(w, e, c) = sum over requests of |[start, end] cap w| / |w|

which is the mean number of requests of that class concurrently on that engine
during the window. A request that lived 3 s of a 60 s window contributes 0.05,
not 1, so a class of long requests is not read as equal to a class of short ones
that merely arrived as often. The dispatch count is produced beside it in the
same table (`dispatched`), so a figure can ask either question, but the figures
in the paper use `resident`.

⚠ THE DENOMINATOR IS ADMITTED WORK, NECESSARILY. A rejected request is never
dispatched and therefore has no engine; it cannot appear in a per-engine
quantity at all. The bars are what the fleet was made to carry, not what
arrived, and an arm that rejects half its arrivals has lower bars for that
reason as well as any other. Every figure built on this has to say so, and the
rejection rate belongs beside it.

HOW A REQUEST IS TIED TO AN ENGINE. Through `analysis/request_engine.csv`, the
scheduler's own dispatch log joined to the client ids, via the join in
`exp41_engine_view.attribute_engines` -- imported rather than rewritten because
its two corrections are easy to lose: on an hour-long trace `(task_id,
call_index)` is not unique, so the join is disambiguated by start time matched
nearest within 2 s, and rejected requests are excluded from the left side
because a rejected request has no id and would otherwise be paired with a
neighbour's. The attributed fraction is returned and a caller that does not
report it is hiding a coverage hole: on 8-minute static conditions it is 100%,
on the hour-long trace the vLLM router reads about 75% because scheduler
dispatch lines are dropped under high load.

THE ANALYSIS WINDOW IS `load_run`'s, so these numbers sit on the same rows as
every score in the paper: 60 s of warmup and 20 s of drain are cut.

    python3 engine_class_occupancy.py <run_dir> [--window 60] [--out out.csv]
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_EV = _load("exp41ev", os.path.join(HERE, "exp41_engine_view.py"))
load_run = _EV.load_run
attribute_engines = _EV.attribute_engines
CLASSES = ("chat", "deepresearch", "swe")


def occupancy(run_dir, window_s=60.0):
    """(table, attributed_fraction) for one run.

    The table is tidy: one row per (window, engine_port, class) with `resident`
    (mean concurrent requests) and `dispatched` (arrivals in that window).
    Windows are anchored at the first arrival in the analysis window.
    """
    r = load_run(run_dir)
    if r is None or r.empty:
        return None, 0.0
    j, n_admitted = attribute_engines(run_dir, r)
    if j.empty or not n_admitted:
        return None, 0.0
    frac = len(j) / float(n_admitted)

    j = j.copy()
    t0 = float(pd.to_numeric(r["start_time"], errors="coerce").min())
    start = pd.to_numeric(j["start_time"], errors="coerce").to_numpy() - t0
    end = pd.to_numeric(j["end_time"], errors="coerce").to_numpy() - t0
    # A request whose end was never recorded (cut off by the run boundary) is
    # treated as resident to the end of the observed span rather than dropped:
    # it WAS on that engine, and dropping it would empty the last windows of
    # exactly the arms that build a backlog.
    span = float(np.nanmax(np.where(np.isnan(end), start, end)))
    end = np.where(np.isnan(end) | (end < start), span, end)

    edges = np.arange(0.0, span + window_s, window_s)
    port = j["engine_port"].astype("Int64").astype(float).to_numpy()
    cls = j["class"].to_numpy()

    rows = []
    for p in sorted(set(port[~np.isnan(port)])):
        for c in CLASSES:
            m = (port == p) & (cls == c)
            if not m.any():
                continue
            s, e = start[m], end[m]
            for w0 in edges[:-1]:
                w1 = w0 + window_s
                ov = np.minimum(e, w1) - np.maximum(s, w0)
                res = float(np.clip(ov, 0.0, None).sum()) / window_s
                disp = int(((s >= w0) & (s < w1)).sum())
                if res > 0 or disp > 0:
                    rows.append({"win_start_s": float(w0), "engine_port": int(p),
                                 "class": c, "resident": res, "dispatched": disp})
    if not rows:
        return None, frac
    return pd.DataFrame(rows), frac


VERDICTS = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                        "results", "aggregate_analysis", "ladder95", "verdicts")


def occupancy_split(run_dir, window_s=60.0, verdicts=VERDICTS):
    """Residency split into the part that produced on-time tokens and the rest.

    Same quantity as `occupancy`, with each request's contribution divided by
    its own token verdicts: a request whose tokens were all on time counts
    entirely as `resident_ontime`, one that missed a third of its deadlines
    puts a third of its residency in `resident_late`. A request that finished
    with no tokens at all counts wholly as late, because the instance held it
    and it returned nothing.

    ⚠ THE SPLIT IS A SHARE, NOT A TIMELINE. The late tokens of a request are
    not necessarily the ones it produced late in its life, so the hatched part
    of a window is how much of that instance's occupancy belonged to work that
    missed its deadline, not the minutes in which the misses happened.

    ⚠ REJECTED REQUESTS ARE ABSENT, as in `occupancy`: they never reached an
    instance. The split therefore shows work accepted and then delivered late,
    which is a different failure from refusing the work, and a figure drawn on
    it has to carry the rejection rate separately.

    The rule itself is owned by `deadline_ladder_attainment.py`; this reads the
    per-request verdicts it dumps with `--dump-verdicts`.
    """
    r = load_run(run_dir)
    if r is None or r.empty:
        return None, 0.0
    name = os.path.basename(os.path.normpath(run_dir))
    vpath = os.path.join(verdicts, name + ".csv")
    if not os.path.exists(vpath):
        sys.exit(f"no verdict file {vpath}; run deadline_ladder_attainment.py "
                 f"--dump-verdicts first")
    v = pd.read_csv(vpath)
    n_tok = pd.to_numeric(v["n_tokens"], errors="coerce").fillna(0.0)
    n_late = pd.to_numeric(v["n_late"], errors="coerce").fillna(0.0)
    v = v.assign(late_frac=np.where(n_tok > 0, n_late / n_tok.where(n_tok > 0, 1.0),
                                    1.0))

    j, n_admitted = attribute_engines(run_dir, r)
    if j.empty or not n_admitted:
        return None, 0.0
    frac = len(j) / float(n_admitted)
    keys = ["task_id", "call_index", "iteration"]
    j = j.merge(v[keys + ["late_frac"]], on=keys, how="left")
    missing = j["late_frac"].isna().mean()
    if missing > 0.001:
        print(f"!! {name}: {100 * missing:.2f}% of attributed requests have no "
              f"verdict row; they are counted as on time", file=sys.stderr)
    j["late_frac"] = j["late_frac"].fillna(0.0)

    t0 = float(pd.to_numeric(r["start_time"], errors="coerce").min())
    start = pd.to_numeric(j["start_time"], errors="coerce").to_numpy() - t0
    end = pd.to_numeric(j["end_time"], errors="coerce").to_numpy() - t0
    span = float(np.nanmax(np.where(np.isnan(end), start, end)))
    end = np.where(np.isnan(end) | (end < start), span, end)
    late = j["late_frac"].to_numpy(dtype=float)
    port = j["engine_port"].astype("Int64").astype(float).to_numpy()
    cls = j["class"].to_numpy()
    edges = np.arange(0.0, span + window_s, window_s)

    rows = []
    for p in sorted(set(port[~np.isnan(port)])):
        for c in CLASSES:
            m = (port == p) & (cls == c)
            if not m.any():
                continue
            st, en, lt = start[m], end[m], late[m]
            for w0 in edges[:-1]:
                ov = np.clip(np.minimum(en, w0 + window_s) - np.maximum(st, w0),
                             0.0, None)
                tot = float(ov.sum()) / window_s
                bad = float((ov * lt).sum()) / window_s
                if tot > 0:
                    rows.append({"win_start_s": float(w0), "engine_port": int(p),
                                 "class": c, "resident": tot,
                                 "resident_late": bad,
                                 "resident_ontime": tot - bad})
    if not rows:
        return None, frac
    return pd.DataFrame(rows), frac


def _shares(v):
    tot = float(np.sum(v))
    return np.asarray(v, dtype=float) / tot if tot > 0 else None


def neff(v):
    """Effective number of instances: 1 / sum(share^2). 1.0 = one instance."""
    s = _shares(v)
    return float(1.0 / np.sum(s * s)) if s is not None else np.nan


def per_window_neff(tab, value="resident"):
    """class -> (median over windows, per-window series) of the effective
    number of instances that class ran on."""
    out = {}
    for c in CLASSES:
        sub = tab[tab["class"] == c]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="win_start_s", columns="engine_port",
                              values=value, aggfunc="sum", fill_value=0.0)
        vals = [neff(row) for row in piv.to_numpy() if row.sum() > 0]
        if vals:
            out[c] = (float(np.median(vals)), np.asarray(vals))
    return out


def pooled_neff(tab, value="resident"):
    """The same quantity computed on the whole run at once.

    ⚠ Kept so a caller can COMPARE it with the windowed median rather than use
    it: pooling over a run whose concentrated engine moves spreads the
    distribution and reads as no concentration at all, which is how an hour of
    99-100% concentration was once recorded as 32.6%.
    """
    out = {}
    for c in CLASSES:
        sub = tab[tab["class"] == c]
        if sub.empty:
            continue
        out[c] = neff(sub.groupby("engine_port")[value].sum().to_numpy())
    return out


def holder_changes(tab, value="resident"):
    """class -> how many times the engine holding the most of it changed.

    Zero means the assignment stood still, so a whole-run bar describes it. It
    is reported with the windowed and pooled numbers because it is the cause of
    a gap between them.
    """
    out = {}
    for c in CLASSES:
        sub = tab[tab["class"] == c]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="win_start_s", columns="engine_port",
                              values=value, aggfunc="sum", fill_value=0.0)
        piv = piv[piv.sum(axis=1) > 0]
        top = piv.idxmax(axis=1).to_numpy()
        out[c] = int((top[1:] != top[:-1]).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tab, frac = occupancy(a.run_dir, a.window)
    if tab is None:
        sys.exit(f"no attributable requests in {a.run_dir}")
    print(f"{os.path.basename(a.run_dir)}: attributed {100 * frac:.1f}% of "
          f"admitted requests, {tab['win_start_s'].nunique()} windows of "
          f"{a.window:.0f} s")
    win = per_window_neff(tab)
    pool = pooled_neff(tab)
    ch = holder_changes(tab)
    print(f"{'class':14s} {'Neff windowed':>13s} {'Neff pooled':>11s} "
          f"{'holder changes':>14s}")
    for c in CLASSES:
        if c in win:
            print(f"{c:14s} {win[c][0]:13.2f} {pool[c]:11.2f} {ch[c]:14d}")
    if a.out:
        tab.to_csv(a.out, index=False)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
