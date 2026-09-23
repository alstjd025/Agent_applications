#!/usr/bin/env python3
"""Paper figure: token and request goodput across SLO scales at 35 req/s.

  exp139_slo_scale_goodput.pdf   3.335 x 2.00 in, `figure`, width=\\columnwidth
  exp139_slo_scale_goodput.csv   one row per run (every repeat on disk)

x axis: the SLO scale k. All six budgets (three classes x TTFT and per-token) are
multiplied by k, both in what each policy is told and in the scoring rule.
  left axis   token goodput  = output tokens of requests that met their SLO / window
  right axis  request goodput = requests that met their SLO / window
Bars are token goodput, lines are request goodput, one colour per arm.

Data (2026-09-15, at the author's request):
  k = 0.7 / 0.9 / 1.1 / 1.3   EXP-139, `results/*exp139k<K>r<R>_*` (one session)
  k = 1.0                     EXP-108 at 2100 rpm, `results/*exp108r<R>_*_rpm_2100`
                              (2026-08-31 session; EXP-139 did not re-run 1.0)

Scoring is EXP-139's own (`analysis_scripts/request_level/exp139_slo_scale.py`):
`all_arrivals_attainment.one_run` with that scale's six budgets read from
`workload_configs/mix_short_m1_k<K>fair.json`. A request meets its SLO when it
was not rejected, not cut off by the run end, and met the rule; rejected and
unfinished requests count as misses. The window is first to last arrival kept by
`load_run`.

The scorer reads its budgets at import time, so each scale is scored in a child
process with that scale's environment (the same reason exp139_slo_scale.py
re-execs itself).

⚠ CAPTION: k=1.0 comes from a different session; this workload's cross-session
movement is up to 4.6 points of attainment. The llm-d repeats differ by up to 11.7
points (EXP-139 §7.1). Request goodput is attainment over all arrivals x 35 req/s,
so it carries no information the attainment does not.

    python3 paper_figures/fig_exp139_slo_scale.py              # repeat 1
    python3 paper_figures/fig_exp139_slo_scale.py --repeat 2
    python3 paper_figures/fig_exp139_slo_scale.py --rescore    # ignore the CSV cache
    python3 paper_figures/fig_exp139_slo_scale.py --token-only # no request goodput
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RL = os.path.join(ROOT, "analysis_scripts", "request_level")
sys.path.insert(0, HERE)
import paper_style as ps  # noqa: E402

CSV = os.path.join(HERE, "exp139_slo_scale_goodput.csv")
KTAGS = ["070", "090", "100", "110", "130"]
KVAL = {"070": 0.7, "090": 0.9, "100": 1.0, "110": 1.1, "130": 1.3}
# arm token in the directory name -> label; the k=1.0 runs (EXP-108) use the
# same arm names except FluidServe, whose EXP-139 name carries a trailing `k`
ARMS = [("fsv3capgnofrct75", "FluidServe"), ("llmdslot75", "llm-d"),
        ("polyservept75", "PolyServe"), ("slot75", "Llumnix")]
# the YlGnBu ramp of the other paper figures
COLOR = {"FluidServe": "#253494", "llm-d": "#c2a5cf", "PolyServe": "#41b6c4",
         "Llumnix": "#a1dab4"}
FIG_H = 2.0
FIG_H_LRCOL = 1.72   # --panels-lr-col: (a) left of (b) in one column
FIG_H_LR = 1.85      # --panels-lr: (a) left of (b) at the text width
FIG_H_AB = 3.05      # --panels: (a) over (b), captions under each
FIG_H_TOK = 1.65      # --token-only: one legend row, no right axis
ORDER = ["Llumnix", "PolyServe", "llm-d", "FluidServe"]


def run_dirs(ktag):
    """[(arm label, repeat, dir)] for one scale; PRERUN warm-ups dropped."""
    if ktag == "100":
        pat = os.path.join(ROOT, "results", "*exp108r[12]_*_rpm_2100")
    else:
        pat = os.path.join(ROOT, "results", f"*exp139k{ktag}r[12]_*_rpm_2100")
    out = []
    for d in sorted(glob.glob(pat)):
        b = os.path.basename(d)
        if "PRERUN" in b:
            continue
        rep = "1" if ("r1_" in b) else "2"
        for tok, lab in ARMS:
            if f"_{tok}_" in b or f"_{tok}k_" in b:
                out.append((lab, rep, d))
                break
    return out


def child(ktag):
    """Score one scale; the budgets are already in the environment."""
    sys.path.insert(0, RL)
    sys.path.insert(0, os.path.dirname(RL))
    import all_arrivals_attainment as A  # noqa: E402
    from exp22_fluidserve import SLO_RULES  # noqa: E402
    print(f"  k={KVAL[ktag]}: rule in force {SLO_RULES}", file=sys.stderr)
    recs = []
    for lab, rep, d in run_dirs(ktag):
        r = A.one_run(d)
        if not r:
            print(f"    no rows: {os.path.basename(d)}", file=sys.stderr)
            continue
        rows = A.load_run(d)
        dur = rows["rel"].max() - rows["rel"].min()
        recs.append(dict(k=KVAL[ktag], arm=lab, repeat=int(rep),
                         run=os.path.basename(d), n=r["n"],
                         all_arrivals=r["all_arrivals"],
                         admitted=r["admitted"], rejected_pct=r["rejected_pct"],
                         tok_goodput=r["goodput_tok_s"],
                         req_goodput=r["met_n"] / dur, window_s=dur))
    json.dump(recs, sys.stdout)


def score():
    sys.path.insert(0, RL)
    import exp139_slo_scale as E  # noqa: E402
    recs = []
    for kt in KTAGS:
        env = dict(os.environ)
        env.update(E.rule_env(kt))
        p = subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--ktag", kt], env=env, capture_output=True,
                           text=True)
        sys.stderr.write(p.stderr)
        if p.returncode:
            sys.exit(f"scoring k={KVAL[kt]} failed")
        recs += json.loads(p.stdout)
    df = pd.DataFrame(recs).sort_values(["k", "arm", "repeat"])
    df.to_csv(CSV, index=False, float_format="%.4f")
    return df


def draw_panels(sel, ks, repeat, lr=False, col=False):
    """(a) token goodput and (b) request goodput, both as bars.

      exp139_slo_scale_goodput_ab.pdf   3.335 x 3.05 in, (a) over (b), `figure`
      exp139_slo_scale_goodput_lr.pdf   7.0 x 1.85 in, (a) left of (b), `figure*`
      exp139_slo_scale_goodput_lrcol.pdf  3.335 x 1.72 in, (a) left of (b), `figure`
        (lr=True, col=True): smaller type (6.5 pt ticks) and wider bars so the
        twenty bars of each panel stay separable in about 1.3 in

    Side by side is drawn at the full text width: twenty bars per panel in half
    of one column (about 1.3 in) are too thin to read. Token goodput ticks carry a lowercase k
    (paper_style.ktick), the SI prefix and the form every other paper figure uses.
    """
    small = lr and col
    fs = 6.5 if small else 7.0
    style = {**ps.STYLE, "xtick.labelsize": fs, "ytick.labelsize": fs,
             "axes.labelsize": fs + 0.5, "legend.fontsize": fs,
             "xtick.major.pad": 1.5, "ytick.major.pad": 1.5}
    x = np.arange(len(ks))
    w = 0.21 if small else 0.19
    with plt.rc_context(style):
        W, H = ((ps.COL_W, FIG_H_LRCOL) if small else
                (ps.TEXT_W, FIG_H_LR) if lr else (ps.COL_W, FIG_H_AB))
        fig, (a0, a1) = plt.subplots(*((1, 2) if lr else (2, 1)), figsize=(W, H))
        for ax, col, scale in ((a0, "tok_goodput", 1.0), (a1, "req_goodput", 1.0)):
            for i, arm in enumerate(ORDER):
                v = []
                for k in ks:
                    r = sel[(sel["k"] == k) & (sel["arm"] == arm)]
                    v.append(r[col].iloc[0] * scale if not r.empty else np.nan)
                ax.bar(x + (i - 1.5) * w, v, w, color=COLOR[arm],
                       edgecolor="#333333", lw=0.3, zorder=2, label=arm)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{k:.1f}" for k in ks])
            ax.set_xlim(-0.5, len(ks) - 0.5)
            ax.set_xlabel("SLO Scale", labelpad=1.0 if small else 1.5)
            ax.grid(axis="y", **ps.GRID)
            ax.set_axisbelow(True)
        # the column-width side-by-side version writes each label on one line
        # with short units (2026-09-15, at the author's request)
        a0.set_ylabel("Token Goodput (t/s)" if small else "Token Goodput\n(tok/s)")
        a0.set_ylim(0, 16000)
        a0.set_yticks([0, 4000, 8000, 12000, 16000])
        a0.yaxis.set_major_formatter(ps.kfmt())
        a1.set_ylabel("Request Goodput (r/s)" if small else "Request Goodput\n(req/s)")
        a1.set_ylim(0, 32)
        a1.set_yticks([0, 8, 16, 24, 32])
        # ⚠ PROXY PATCHES, NOT THE BAR CONTAINERS (2026-09-18). `bar()` returns
        # a BarContainer, and the square handler is keyed on Patch objects, so
        # with the containers the key fell back to matplotlib's wide default
        # rectangle. These proxies carry the same fill and edge.
        h = [Patch(facecolor=COLOR[arm], edgecolor="#333333", lw=0.3)
             for arm in ORDER]
        l = list(ORDER)
        h, l = ps.legend_items(h, l)
        if lr:
            # one legend centred over both panels
            # the key at exp131_horizon_goodput's 7 pt when it fits, else
            # the shared 6.5 (2026-09-18, at the author's request)
            for key_fs in (7.0, ps.KEY_FS):
                leg = fig.legend(h, l, loc="upper center", fontsize=key_fs,
                                 bbox_to_anchor=(0.5, 1.0), ncol=4,
                                 columnspacing=0.8 if small else 1.2,
                                 borderaxespad=0.2,
                                 handler_map=ps.square_handler(h),
                                 **ps.KEY_SQUARE)
                fig.canvas.draw()
                w_in = leg.get_window_extent().width / fig.dpi
                print(f"  key at {key_fs:.1f} pt is {w_in:.2f} in wide on a "
                      f"{W:.2f} in canvas"
                      + ("  -> too wide" if w_in > W - 0.02 else "  -> used"))
                if w_in <= W - 0.02:
                    break
                leg.remove()
            if small:
                fig.tight_layout(pad=0.25, w_pad=0.9, rect=(0, 0.075, 1, 0.885))
            else:
                fig.tight_layout(pad=0.35, w_pad=2.0, rect=(0, 0.075, 1, 0.90))
        else:
            a0.legend(h, l, loc="lower center",
                      bbox_to_anchor=(0.5, 1.0), ncol=4, handlelength=1.1,
                      handletextpad=0.3, columnspacing=0.8, borderaxespad=0.2)
            # room under each panel for its caption: h_pad between the panels,
            # and the bottom of the rect under panel (b)
            fig.tight_layout(pad=0.35, h_pad=2.2, rect=(0, 0.045, 1, 1))
            fig.align_ylabels([a0, a1])
        # captions under each panel, below its x label
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        for ax, cap in ((a0, "(a) Token Goodput"), (a1, "(b) Request Goodput")):
            y = ax.get_tightbbox(rend).y0 / fig.dpi / H
            fig.text(0.5 * (ax.get_position().x0 + ax.get_position().x1),
                     y - 0.004, cap, ha="center", va="top", fontsize=fs + 0.5)
        if lr:
            gap = (a1.get_tightbbox(rend).x0 - a0.get_tightbbox(rend).x1) / fig.dpi
        else:
            gap = (a0.get_tightbbox(rend).y0 - a1.get_tightbbox(rend).y1) / fig.dpi
        print(f"  gap between panel (a) ink and panel (b) ink: {gap:.3f} in")
        for ax in (a0, a1):
            lab = ax.yaxis.label.get_window_extent(rend).height / fig.dpi
            box = ax.get_window_extent(rend).height / fig.dpi
            print(f"  y label {lab:.2f} in against axes height {box:.2f} in"
                  + ("  ⚠ LONGER THAN THE AXES" if lab > box else ""))
        fig.canvas.draw()
        bot = min(t.get_window_extent(rend).y0 for t in fig.texts) / fig.dpi
        print(f"  lowest caption ink {bot:.3f} in above the canvas bottom")
        stem = "exp139_slo_scale_goodput_" + ("lrcol" if small else "lr" if lr else "ab")
        name = stem + (".pdf" if repeat == 1 else f"_r{repeat}.pdf")
        ps.save(fig, ps.final(name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ktag", help=argparse.SUPPRESS)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--panels", action="store_true",
                    help="(a) token over (b) request -> exp139_slo_scale_goodput_ab.pdf")
    ap.add_argument("--panels-lr", action="store_true",
                    help="(a) left of (b), text width -> exp139_slo_scale_goodput_lr.pdf")
    ap.add_argument("--panels-lr-col", action="store_true",
                    help="(a) left of (b), column width -> exp139_slo_scale_goodput_lrcol.pdf")
    ap.add_argument("--token-only", action="store_true",
                    help="token goodput only -> exp139_slo_scale_tokgoodput.pdf")
    a = ap.parse_args()
    if a.ktag:
        child(a.ktag)
        return 0

    df = score() if (a.rescore or not os.path.exists(CSV)) else pd.read_csv(CSV)
    sel = df[df["repeat"] == a.repeat]
    ks = [KVAL[k] for k in KTAGS]
    missing = [(k, arm) for k in ks for arm in ORDER
               if sel[(sel["k"] == k) & (sel["arm"] == arm)].empty]
    if missing:
        print(f"repeat {a.repeat}: missing cells {missing}")
    print(f"\nrepeat {a.repeat}   token goodput (tok/s) | request goodput (req/s)"
          " | attainment over all arrivals")
    for k in ks:
        line = f"  k={k}: "
        for arm in ORDER:
            r = sel[(sel["k"] == k) & (sel["arm"] == arm)]
            if not r.empty:
                r = r.iloc[0]
                line += (f"{arm} {r.tok_goodput:6.0f} | {r.req_goodput:5.2f} | "
                         f"{r.all_arrivals:4.1f};  ")
        print(line)

    if a.panels or a.panels_lr or a.panels_lr_col:
        draw_panels(sel, ks, a.repeat, lr=a.panels_lr or a.panels_lr_col,
                    col=a.panels_lr_col)
        return 0
    style = {**ps.STYLE, "xtick.labelsize": 7, "ytick.labelsize": 7,
             "axes.labelsize": 7.5, "legend.fontsize": 7}
    both = not a.token_only
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(ps.COL_W, FIG_H if both else FIG_H_TOK))
        ax2 = ax.twinx() if both else None
        x = np.arange(len(ks))
        w = 0.19
        for i, arm in enumerate(ORDER):
            tok, req = [], []
            for k in ks:
                r = sel[(sel["k"] == k) & (sel["arm"] == arm)]
                tok.append(r["tok_goodput"].iloc[0] / 1000 if not r.empty else np.nan)
                req.append(r["req_goodput"].iloc[0] if not r.empty else np.nan)
            ax.bar(x + (i - 1.5) * w, tok, w, color=COLOR[arm],
                   edgecolor="#333333", lw=0.3, zorder=2)
            if both:
                ax2.plot(x + (i - 1.5) * w, req, color=COLOR[arm], lw=1.0,
                         marker="o", ms=3.0, mec="#333333", mew=0.4, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{k:.1f}" for k in ks])
        ax.set_xlim(-0.55, len(ks) - 0.45)
        ax.set_xlabel("SLO Scale", labelpad=1.5)
        ax.set_ylabel("Token Goodput\n(k tok/s)")
        ax.set_ylim(0, 16)
        ax.set_yticks([0, 4, 8, 12, 16])
        if both:
            ax2.set_ylabel("Request Goodput\n(req/s)")
            ax2.set_ylim(0, 32)
            ax2.set_yticks([0, 8, 16, 24, 32])
        ax.grid(axis="y", **ps.GRID)
        ax.set_axisbelow(True)
        arms = [Patch(facecolor=COLOR[arm], edgecolor="#333333", lw=0.3,
                      label=arm) for arm in ORDER[::-1]]
        arms, arm_names = ps.legend_items(arms, [a_.get_label() for a_ in arms])
        for h_, n_ in zip(arms, arm_names):
            h_.set_label(n_)
        if both:
            # two rows: the arms, then what a bar and a line mean. One row of
            # six entries is wider than the column.
            key = [Patch(facecolor="#bdbdbd", edgecolor="#333333", lw=0.3,
                         label="Token goodput (bars, left)"),
                   Line2D([], [], color="#636363", marker="o", ms=3.0, lw=1.0,
                          mec="#333333", mew=0.4,
                          label="Request goodput (lines, right)")]
            leg = ax.legend(handles=arms, loc="lower center",
                            bbox_to_anchor=(0.5, 1.115), ncol=4,
                            handlelength=1.1, handletextpad=0.3,
                            columnspacing=0.8, borderaxespad=0.0)
            ax.add_artist(leg)
            ax.legend(handles=key, loc="lower center",
                      bbox_to_anchor=(0.5, 0.99), ncol=2, handlelength=1.3,
                      handletextpad=0.3, columnspacing=0.8, borderaxespad=0.0)
            # tight_layout does not see the legend added with add_artist, so
            # the two legend rows get their band explicitly
            fig.tight_layout(pad=0.35, rect=(0, 0, 1, 0.885))
        else:
            ax.legend(handles=arms, loc="lower center",
                      bbox_to_anchor=(0.5, 1.0), ncol=4, handlelength=1.1,
                      handletextpad=0.3, columnspacing=0.8, borderaxespad=0.2)
            fig.tight_layout(pad=0.35)
        stem = "exp139_slo_scale_goodput" if both else "exp139_slo_scale_tokgoodput"
        name = stem + (".pdf" if a.repeat == 1 else f"_r{a.repeat}.pdf")
        ps.save(fig, ps.final(name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
