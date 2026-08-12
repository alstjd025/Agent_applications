#!/usr/bin/env python3
"""Schematic (NO MEASURED DATA): what a control plane should do past capacity.

  ideal_overload.pdf   3.335 x 3.30 in, one column, width=\\columnwidth

⚠ EVERY CURVE IN THIS FIGURE IS DRAWN FROM A FORMULA. Nothing here is measured,
the axes carry no units, and the capacity is the symbol C rather than a number.
THE CAPTION MUST SAY SO in its first clause; a schematic that looks like a
result is the worst thing this figure could be. It exists to define what the
paper is aiming at, so that the measured figures can be read against it.

THE SETUP. Offered load is lambda, on a shared x axis in units of C. C is the
highest offered load at which the fleet can still serve every arrival inside its
latency rule -- the same quantity `fig_intro_capacity.py` measures for each real
control plane. Three rows, all functions of lambda:

  Throughput       requests per second the system completes
  SLO attainment   share of ARRIVALS that met their rule; a dropped request is
                   a miss, so this is the offered denominator
  Drop rate        share of arrivals refused at admission

PHASE I, lambda < C. There is nothing to decide: throughput follows the demand
line lambda, attainment sits at the target, the drop rate is zero. Every policy
that is not broken looks the same here, which is why the measured figures also
show the arms on top of each other below about 20 req/s.

PHASE II, lambda > C. This is the whole design problem, and the ideal is a flat
answer in all three rows at once:

  throughput   pinned at C -- the fleet stays busy, and no more
  attainment   pinned at the target -- what was admitted still meets its rule
  drop rate    exactly the excess, 1 - C/lambda, and not a request more

The third curve is the arithmetic consequence of the first two: to hold
throughput at C while lambda arrives, C/lambda of the arrivals can be admitted
and the rest must be refused. It is drawn so that "the drop rate should rise"
does not read as "any drop rate will do" -- there is one correct curve.

THE THREE WAYS TO LEAVE IT, drawn dotted, one colour each:

  no rejection              Everything is accepted. Throughput stays at about C,
                            because the engines are the constraint and they are
                            full either way -- so the top row CANNOT tell this
                            case from the ideal. What gives it away is the
                            middle row: the excess is absorbed as queueing and
                            eviction, so attainment falls away. The drop row is
                            flat at zero.
  mistimed rejection        Requests are refused, but on a signal unrelated to
                            what the engines can currently sustain -- a fixed
                            concurrency cap, a queue-length trigger, a rate
                            limit. BOTH rows fall: work is refused that could
                            have been served, so throughput sits below C, and
                            work is admitted that cannot be served, so
                            attainment drops as well.
  over-rejection            Refusing more than the excess. Attainment holds at
                            the target, because what little is admitted is
                            served well -- so the MIDDLE row cannot tell this
                            case from the ideal. What gives it away is the top
                            row, where throughput falls below C, and the bottom
                            row, where the curve sits above 1 - C/lambda.

Read the three together and each failure is visible in exactly two rows and
invisible in the third. That is the argument for reporting all three in the
measured figures, and it is why attainment on the admitted denominator is not
enough on its own.

    python3 paper_figures/fig_ideal_overload.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_style as ps  # noqa: E402

FIG_W, FIG_H = ps.COL_W, 3.30
XMAX = 2.6          # in units of C
TARGET = 95.0       # the SLO target, in percent; a symbol, not a measurement

# The ideal is drawn in FluidServe's blue, the colour our system carries in
# every measured figure: this schematic states the target the paper aims at, and
# the reader meets the same colour again as the thing that comes closest to it.
IDEAL = ps.ARM_COLOR["fluidserve"]
# Brown and purple were too close to separate at 1.1 pt, so the middle case
# moves to orange, and each deviation also gets its OWN dash pattern -- short,
# long, dotted. Two channels rather than one: at this line width a reader
# following a curve through a crossing needs the rhythm as well as the hue, and
# the figure has to survive being printed in grey.
DEV = {"none": "#d62728",
       "blind": "#ff7f0e",
       "over": "#2ca02c"}
# Short dash, long dash, dash-dot. The third was a fine dot pattern, which at
# 1.2 pt reads as a smudge rather than a line and was weakest exactly where it
# matters most -- lying along the ideal in the attainment row, where it has to
# be legible as a curve to show the two coincide.
DASH = {"none": (0, (2.2, 1.2)),
        "blind": (0, (5.5, 1.6)),
        "over": (0, (3.5, 1.2, 1.0, 1.2))}
# The three names are a parallel set on one axis, how much is refused against
# the excess: none, too little, too much. Earlier drafts called the middle one
# "admission blind to batch" and then "indiscriminate rejection", both of which
# named a mechanism while its neighbours named an amount, so the three did not
# line up.
LABEL = {"none": "No rejection",
         "blind": "Under-rejection",
         "over": "Over-rejection"}


def main():
    x = np.linspace(0.05, XMAX, 600)

    # ONE ADMITTED SHARE AND ONE EFFICIENCY PER CASE, and all three rows are
    # derived from those two, so the rows cannot state different things about
    # the same case -- the failure mode of a schematic drawn curve by curve.
    #
    #   admit(x)  the share of arrivals let in
    #   eta(x)    the share of the engines' work that reaches a completion;
    #             below 1 when admission is mistimed and the engines spend
    #             capacity on requests they later evict and recompute
    #
    #   throughput = min(admit * lambda, C) * eta      drop = 1 - admit
    #
    # `f` is 0 through phase I and rises towards 1 with the overload, so every
    # case is IDENTICAL below C by construction. That is a claim of the figure
    # and it should not depend on three formulas happening to agree.
    over = np.maximum(x, 1.0)
    f = 1.0 - 1.0 / over

    # Over-rejection STARTS REFUSING BEFORE C, at OVER_START, and then refuses
    # on a power of lambda steeper than the excess. That is what makes its
    # throughput turn over slightly to the left of C, at a peak below the
    # ideal's, and its drop rate leave zero while the other three are still at
    # zero -- the signature of a policy that is too eager rather than too late.
    OVER_START, OVER_P = 0.85, 1.8
    # Under-rejection starts refusing WELL AFTER C, which is the contrast with
    # over-rejection starting before it. Between C and here it behaves like the
    # no-rejection case: the queue is already building and nothing is being
    # turned away.
    UNDER_START = 1.25
    admit = {"ideal": 1.0 / over,               # exactly the excess refused
             "none": np.ones_like(x),           # nothing refused, ever
             "blind": 1.0 - 0.45 * np.maximum(0.0, 1.0 - UNDER_START / x),
             "over": np.minimum(1.0, (OVER_START / x) ** OVER_P)}
    # `eta` is what the engines complete out of what they were given. For the
    # two cases that admit more than the fleet can serve it falls with the
    # overload, and it falls fast enough that their throughput PEAKS AT ABOUT C
    # and then declines: past capacity the extra arrivals do not add
    # completions, they add evictions, and every eviction pays for its prefill
    # twice. That decline is what the measured sweeps show as well.
    eta = {"ideal": np.ones_like(x),
           "none": 1.0 - 0.36 * f,
           # UNDER-REJECTION COMPLETES LESS THAN REFUSING NOTHING AT ALL, which
           # is the point of drawing it. It refuses too little to stop the queue
           # building, so it pays nearly the whole eviction cost that the
           # no-rejection case pays; and the refusals it does make land on a
           # coarse signal, so some of them remove work the engines had room
           # for. It gives up completions and buys no relief with them.
           "blind": 1.0 - 0.58 * f,
           "over": np.ones_like(x)}    # what little is admitted is served well

    # A SOFT KNEE, not a corner. Real throughput does not follow the demand
    # line to C and then turn: arrivals are bursty, so queueing costs appear
    # before the mean load reaches capacity and the curve bends into the
    # ceiling. `ceiling` is that bend -- it follows lambda while lambda is well
    # under C, sits at 0.89 C exactly at C, and approaches C from below after
    # it; p sets how sharp the bend is.
    #
    # ⚠ THE BEND IS A PROPERTY OF THE ARRIVAL STREAM AND THE ENGINES, SO IT IS
    # A FUNCTION OF LAMBDA, NOT OF WHAT EACH CASE ADMITTED. Written the other
    # way -- bending `admit * lambda` -- the ideal was the one case whose
    # argument is capped at exactly C, so its own knee held it at 0.89 C while
    # the case that admits everything ran along the ceiling ABOVE it. The
    # figure then said that refusing nothing produces more than the ideal,
    # which is the opposite of its claim.
    #
    # A case completes the smaller of what it let in and what the engines can
    # finish, and then loses the share `1 - eta` to mistimed admission.
    def ceiling(lam, p=6.0):
        return (lam ** -p + 1.0) ** (-1.0 / p)

    thru = {k: np.minimum(admit[k] * x, ceiling(x)) * eta[k] for k in admit}
    drop = {k: 1.0 - a for k, a in admit.items()}
    # Attainment holds at the target for the two cases that never admit more
    # than the engines can serve, and decays for the two that do. The decay
    # constants are chosen only so the two curves are distinguishable; nothing
    # in this figure claims a rate.
    # No rejection decays to the floor rather than towards it: with nothing
    # held back, every arrival past C is queued behind work that is already
    # late, so the share meeting its rule goes to nearly zero rather than
    # settling somewhere. Indiscriminate rejection decays too, but it is holding
    # some load back, so it is still off the floor at the right edge.
    att = {"ideal": np.full_like(x, TARGET),
           "over": np.full_like(x, TARGET),
           "none": TARGET * np.exp(-2.6 * (over - 1.0)),
           "blind": TARGET / (1.0 + 2.2 * (over - 1.0))}

    rows = [("Throughput", thru), ("SLO attainment", att), ("Drop rate", drop)]

    with plt.rc_context(ps.STYLE):
        fig, ax = plt.subplots(3, 1, figsize=(FIG_W, FIG_H), sharex=True)
        handles, labels = [], []

        # WHERE A DEVIATION IS DRAWN FROM. Below its own departure point every
        # case IS the ideal -- exactly, by construction -- and drawing four
        # curves along one line there produced a band of mixed dashes that
        # looked like disagreement where there is none. Each deviation is drawn
        # only from the load at which it leaves the ideal, so phase I carries a
        # single line and the caption says the four coincide there.
        #
        # The one overlap kept on purpose is over-rejection in the attainment
        # row: it holds the target across the whole range, which is the point of
        # that case, so its dots are drawn along the ideal rather than omitted.
        start = {"none": 1.0, "blind": 1.0, "over": OVER_START}

        for i, (_, series) in enumerate(rows):
            # The ideal goes down FIRST and the deviations on top of it. In the
            # attainment row the over-rejection curve is exactly the ideal --
            # that is the point of the case, it holds the target by refusing
            # work -- and drawn underneath it was invisible, so the row looked
            # as though the case were missing. On top, its dashes sit along the
            # blue line and the coincidence is what the reader sees. The caption
            # still has to say the two are equal there rather than close.
            h, = ax[i].plot(x, series["ideal"], color=IDEAL, lw=1.6, zorder=2)
            if i == 0:
                handles.append(h)
                labels.append("Ideal")
            for key in ("none", "blind", "over"):
                m = x >= start[key]
                d, = ax[i].plot(x[m], series[key][m], color=DEV[key], lw=1.2,
                                ls=DASH[key], zorder=3)
                if i == 0:
                    handles.append(d)
                    labels.append(LABEL[key])

        for i in range(3):
            # C divides the two regimes and is the only x value that matters,
            # so it is the only one labelled.
            ax[i].axvline(1.0, color="#999999", lw=0.7, ls=":", zorder=0)
            ax[i].set_xlim(0, XMAX)
            ax[i].set_xticks([1.0])
            ax[i].set_xticklabels(["$C$"])
            ax[i].grid(axis="y", **ps.GRID)
            ax[i].set_axisbelow(True)

        ax[0].set_ylim(0, 1.25)
        ax[0].set_yticks([0, 1.0])
        ax[0].set_yticklabels(["0", "$C$"])
        ax[0].set_ylabel("Throughput", labelpad=1.5)

        ax[1].set_ylim(0, 112)
        # The tick stays so the gridline stays, but it is not labelled: the name
        # goes inside the panel, just above the line and at the left, where it
        # sits on the flat stretch every case shares and cannot be read as a
        # value on the axis.
        ax[1].set_yticks([0, TARGET])
        ax[1].set_yticklabels(["0", ""])
        ax[1].annotate("target", (0.06, TARGET + 3), color="#333333",
                       fontsize=7, ha="left", va="bottom")
        ax[1].set_ylabel("SLO attainment", labelpad=1.5)

        ax[2].set_ylim(0, 1.05)
        ax[2].set_yticks([0, 1.0])
        ax[2].set_yticklabels(["0", "1"])
        ax[2].set_ylabel("Drop rate", labelpad=1.5)
        # The ideal drop curve is 1 - C/lambda. It is NOT written on the figure
        # any more, so the caption has to carry it: "the drop rate should rise"
        # is not the claim, "it should be exactly the excess" is.

        ax[2].set_xlabel(r"Offered load $\lambda$", labelpad=1.5)
        # The two regimes named once, over the top panel.
        ax[0].annotate("Phase I", (0.5, 1.13), color="#777777", fontsize=7,
                       ha="center", va="center")
        ax[0].annotate("Phase II", (1.85, 1.13), color="#777777", fontsize=7,
                       ha="center", va="center")

        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=6.5,
                   bbox_to_anchor=(0.5, 0.905), frameon=False,
                   columnspacing=0.8, handlelength=1.6, handletextpad=0.35,
                   borderaxespad=0.0)
        fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=0.6, pad=0.3)
        ps.save(fig, os.path.join(HERE, "ideal_overload.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
