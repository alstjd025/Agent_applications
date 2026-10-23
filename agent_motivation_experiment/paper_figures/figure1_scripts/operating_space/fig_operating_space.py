#!/usr/bin/env python3
"""Figure 1 candidate: serving capacity as an operating region.

  operating_space.pdf      2D. x = decode batch size, y = decode-iteration
                           latency. One curve, the instance's performance
                           envelope; horizontal lines, the pace each resident
                           SLO class tolerates; where a line meets the curve
                           is the largest admissible batch under that class.
                           The band between the tight and loose intersections
                           is the capacity one tight resident removes. A faint
                           second line shows the constraint moving as
                           residents progress.
  operating_space_3d.pdf   one box: the latency surface over (decode batch
                           size, KV-cache occupancy) and two resident-
                           constraint planes, 50 and 100 ms. The surface is
                           coloured by region: feasible under the tight plane,
                           the wedge between the planes = capacity lost to a
                           tighter resident, above the loose plane infeasible.
  operating_space_3d_pair.pdf  the earlier two-box version (kept for reference).

Both 3.335 in wide, drawn at final size; 200 dpi PNGs too.

MODEL. The envelope is a smooth stand-in, lat(b) = A + B*b + C*b^2 (ms), with
A, B, C chosen so that it crosses 50 / 75 / 100 ms near the caps used by
fig_capacity_cliff.py (170 / 270 / 400); the capacities printed on the
figure are READ OFF THE CURVE, not typed. Replace `lat` with the measured
decode-iteration latency vs. batch size of the served model (at each class's
typical KV footprint) and the figure is a measurement; nothing else changes.

    python3.12 fig_operating_space.py [--fmt=2d|3d|3dmpl|3dpair|all]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paper_style import COL_W, STYLE, GRID, save  # noqa: E402

FMT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--fmt=")), "all")

SLO = [("chat", "Chat", 50, "#3a8f8a"), ("agent", "Agent", 75, "#b07cc6"),
       ("deepresearch", "Deep Research", 100, "#4c72b0")]
B_MAX = 460                     # x range
FIG_H_3D = 2.0                  # height of the 3D figure, inches
KV_LIMIT = 430                  # where the KV cache runs out (batch units)
A, B, C = 22.0, 0.165, 1.0e-4   # envelope coefficients, ms


def lat(b):
    return A + B * b + C * b ** 2


def batch_at(ms):
    """Largest batch with lat(b) <= ms (inverse of the envelope)."""
    return (-B + np.sqrt(B * B - 4 * C * (A - ms))) / (2 * C)


# One-column type, as fig_headroom_scatter's compact variant sets it: body,
# axis labels, titles and ticks 1.5 pt under STYLE (6.5 pt), annotations
# another 1.5 pt under that (5 pt), legends 5.5 pt. Nothing smaller.
FS = STYLE["font.size"] - 1.5
ANNOT = FS - 1.5
LEGEND_FS = 5.5


def small_style():
    st = dict(STYLE)
    for k in ("font.size", "axes.labelsize", "axes.titlesize",
              "xtick.labelsize", "ytick.labelsize"):
        st[k] = STYLE[k] - 1.5
    st["legend.fontsize"] = LEGEND_FS
    return st


def finish(fig, name):
    fig.savefig(os.path.join(HERE, name + ".png"), dpi=200)
    save(fig, os.path.join(HERE, name + ".pdf"))


def fig_2d():
    st = small_style()
    caps = {k: batch_at(ms) for k, _, ms, _ in SLO}
    tight, loose = caps["chat"], caps["deepresearch"]
    print("capacity read off the curve: " +
          ", ".join(f"{n} {ms} ms -> {caps[k]:.0f}" for k, n, ms, _ in SLO) +
          f"; one tight resident removes {loose - tight:.0f}")
    b = np.linspace(0, B_MAX, 300)
    with plt.rc_context(st):
        fig, ax = plt.subplots(figsize=(COL_W, 1.9))
        fig.subplots_adjust(left=0.15, right=0.985, top=0.95, bottom=0.21)
        # regions: feasible under the tight line (solid), the band the tight
        # resident removes (hatched), both under the envelope
        bt = b[b <= tight]
        ax.fill_between(bt, 0, lat(bt), color=to_rgb("#3a8f8a"), alpha=0.16, linewidth=0)
        bl = b[(b >= tight) & (b <= loose)]
        ax.fill_between(bl, 0, lat(bl), facecolor="none", edgecolor="#cfcfcf",
                        hatch="////", linewidth=0)
        # the envelope
        ax.plot(b, lat(b), color="black", lw=1.1, zorder=3)
        ax.text(B_MAX * 0.62, lat(B_MAX * 0.62) + 9, "iteration latency\nof the instance",
                ha="right", va="bottom", fontsize=ANNOT, linespacing=1.1, rotation=0)
        # SLO lines and intersections
        for k, name, ms, col in SLO:
            x = caps[k]
            ax.plot([0, x], [ms, ms], color=col, lw=0.8, ls=(0, (3, 1.5)), zorder=2)
            ax.plot([x, x], [0, ms], color=col, lw=0.5, ls=(0, (1, 1.5)), zorder=2)
            ax.plot(x, ms, "o", ms=3.2, color=col, markerfacecolor="white",
                    markeredgewidth=1.0, zorder=4)
            ax.text(3, ms + 1.5, f"{name} pace, {ms} ms", ha="left", va="bottom",
                    fontsize=ANNOT, color=col)
            ax.text(x, -3, f"{x:.0f}", ha="center", va="top", fontsize=ANNOT, color=col,
                    transform=ax.get_xaxis_transform(), clip_on=False)
        # the constraint moving as residents progress: a faint second tight line
        ms2 = 60
        x2 = batch_at(ms2)
        ax.plot([0, x2], [ms2, ms2], color="#3a8f8a", lw=0.6, ls=(0, (3, 1.5)), alpha=0.45)
        ax.annotate("", xy=(x2 * 0.72, ms2 - 0.8), xytext=(x2 * 0.72, 50 + 0.8),
                    arrowprops=dict(arrowstyle="-|>", lw=0.5, color="#3a8f8a", alpha=0.7,
                                    shrinkA=0, shrinkB=0, mutation_scale=4.5))
        ax.text(x2 + 6, ms2 + 0.5, "constraint moves as\nresidents progress or leave",
                ha="left", va="center", fontsize=ANNOT, color="#3a8f8a", alpha=0.9,
                linespacing=1.1,
                bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85))
        # region labels
        ax.text(tight * 0.45, lat(tight * 0.45) * 0.45, "feasible with\none tight resident",
                ha="center", va="center", fontsize=ANNOT, color="#2c6e69", linespacing=1.1)
        ax.text((tight + loose) / 2, lat((tight + loose) / 2) * 0.42,
                f"capacity the tight\nresident removes\n({loose - tight:.0f} requests)",
                ha="center", va="center", fontsize=ANNOT, color="#555555", linespacing=1.1,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
        # KV limit
        ax.axvline(KV_LIMIT, color="#999999", lw=0.5, ls=":")
        ax.text(KV_LIMIT - 4, 8, "KV-cache\nlimit", ha="right", va="bottom", fontsize=ANNOT,
                color="#777777", linespacing=1.1)
        ax.set_xlim(0, B_MAX)
        ax.set_ylim(0, lat(B_MAX) * 1.06)
        ax.set_xticks([0, 100, 200, 300, 400])
        ax.set_yticks([0, 50, 100])
        ax.set_xlabel("Decode batch size (resident + admitted requests)", labelpad=8)
        ax.set_ylabel("Iteration latency (ms)", labelpad=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        finish(fig, "operating_space")


def build_3d(figsize, elev=18, azim=-68, aspect=(1.6, 0.7, 0.75), zmax=124.0, xmax=420.0,
             pos=(-0.12, 0.05, 1.02, 1.01), zoom=1.15):
    """One box. The latency surface (light grey, shaded), two constraint
    lids drawn only over their footprints, and two closed volumes: the
    feasible block (surface .. 50 ms, teal) and the lost wedge (surface ..
    100 ms over the footprint that only the loose constraint admits, grey).
    Every face goes into ONE Poly3DCollection so the renderer depth-sorts
    them together; separate collections would paint over each other.
    Facets that would rise above zmax are dropped so the box can be cut
    tight at the top (that corner is far above every constraint anyway)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    st = small_style()
    nb, nk = 44, 22
    bb = np.linspace(0, xmax, nb + 1)
    kk = np.linspace(0, 1, nk + 1)
    TIGHT, LOOSE = 50, 100
    kv = lambda k: 1 + 0.12 * k ** 2                  # noqa: E731  KV slows a step
    zf = lambda b, k: lat(b) * kv(k)                  # noqa: E731
    cut = lambda lvl, k: batch_at(lvl / kv(k))        # noqa: E731  batch where surface hits lvl
    teal, blue, grey = np.array(to_rgb("#3a8f8a")), np.array(to_rgb("#4c72b0")), np.array(to_rgb("#8a8a8a"))
    light = np.array([-0.4, -0.6, 0.7]); light /= np.linalg.norm(light)

    faces, colors = [], []

    def add(quad, rgba):
        faces.append(quad); colors.append(rgba)

    # 1. the surface, shaded light grey; teal where feasible, mid grey where lost
    for i in range(nb):
        for j in range(nk):
            q = [(bb[i], kk[j]), (bb[i + 1], kk[j]), (bb[i + 1], kk[j + 1]), (bb[i], kk[j + 1])]
            zs = [zf(x, y) for x, y in q]
            if max(zs) > zmax:
                continue
            zc = np.mean(zs)
            p0, p1, p3 = np.array([q[0][0] / xmax, q[0][1], zs[0] / 100.0]), \
                np.array([q[1][0] / xmax, q[1][1], zs[1] / 100.0]), \
                np.array([q[3][0] / xmax, q[3][1], zs[3] / 100.0])
            n = np.cross(p1 - p0, p3 - p0); n /= np.linalg.norm(n)
            sh = 0.78 + 0.22 * max(0.0, float(np.dot(n, light)))
            if zc <= TIGHT:
                base = 1 - 0.55 * (1 - teal)                     # muted teal
            elif zc <= LOOSE:
                base = np.array([0.80, 0.80, 0.80])
            else:
                base = np.array([0.93, 0.93, 0.93])
            add([(x, y, z) for (x, y), z in zip(q, zs)], (*(base * sh), 1.0))

    # 2. feasible block: lid at 50 over its footprint, walls at b=0, y=0, y=1
    lid_t = (*teal, 0.22)
    wall_t = (*teal, 0.28)
    kf = np.linspace(0, 1, 40)
    add([(0, 0, TIGHT)] + [(cut(TIGHT, k), k, TIGHT) for k in kf] + [(0, 1, TIGHT)], lid_t)
    for j in range(nk):                                   # wall at b = 0
        add([(0, kk[j], zf(0, kk[j])), (0, kk[j + 1], zf(0, kk[j + 1])),
             (0, kk[j + 1], TIGHT), (0, kk[j], TIGHT)], wall_t)
    for y in (0.0, 1.0):                                  # front / back walls
        xs = np.linspace(0, cut(TIGHT, y), 30)
        for i in range(len(xs) - 1):
            add([(xs[i], y, zf(xs[i], y)), (xs[i + 1], y, zf(xs[i + 1], y)),
                 (xs[i + 1], y, TIGHT), (xs[i], y, TIGHT)], wall_t)

    # 3. lost wedge: lid at 100 over the lost footprint, near wall on the
    #    tight cut (50 -> 100), front / back walls, all grey
    lid_g = (*grey, 0.22)
    wall_g = (*grey, 0.26)
    add([(cut(TIGHT, k), k, LOOSE) for k in kf] + [(cut(LOOSE, k), k, LOOSE) for k in kf[::-1]],
        lid_g)
    for j in range(nk):                                   # near wall along the tight cut
        x0, x1 = cut(TIGHT, kk[j]), cut(TIGHT, kk[j + 1])
        add([(x0, kk[j], TIGHT), (x1, kk[j + 1], TIGHT), (x1, kk[j + 1], LOOSE), (x0, kk[j], LOOSE)],
            wall_g)
    for y in (0.0, 1.0):
        xs = np.linspace(cut(TIGHT, y), cut(LOOSE, y), 30)
        for i in range(len(xs) - 1):
            add([(xs[i], y, zf(xs[i], y)), (xs[i + 1], y, zf(xs[i + 1], y)),
                 (xs[i + 1], y, LOOSE), (xs[i], y, LOOSE)], wall_g)

    with plt.rc_context(st):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_proj_type("ortho")
        coll = Poly3DCollection(faces, facecolors=colors, edgecolors="none", linewidths=0)
        coll.set_zsort("average")
        ax.add_collection3d(coll)
        # outlines: the two lids' boundaries (cut curve + box edges)
        for lvl, col, lw in ((TIGHT, "#2f7f79", 0.9), (LOOSE, "#4c72b0", 0.8)):
            xs = np.array([cut(lvl, k) for k in kk])
            ax.plot(xs, kk, [lvl] * len(kk), color=col, lw=lw, zorder=30)
            ax.plot([0, 0], [0, 1], [lvl, lvl], color=col, lw=lw * 0.7, zorder=30)
            ax.plot([0, xs[0]], [0, 0], [lvl, lvl], color=col, lw=lw * 0.7, zorder=30)
            ax.plot([0, xs[-1]], [1, 1], [lvl, lvl], color=col, lw=lw * 0.7, zorder=30)
        # short labels: constraints in the empty back-left corner, regions by
        # their front walls
        ax.text(0, 1.0, LOOSE + 4, "Loose constraint \u00b7 100 ms", fontsize=ANNOT,
                color="#4c72b0", ha="left", va="bottom", zorder=40)
        ax.text(0, 1.0, TIGHT + 4, "Tight constraint \u00b7 50 ms", fontsize=ANNOT,
                color="#2f7f79", ha="left", va="bottom", zorder=40)
        ax.text(cut(TIGHT, 0) * 0.45, -0.02, TIGHT * 0.66, "Feasible", fontsize=ANNOT,
                color="#1f5f5a", ha="center", va="center", zorder=40)
        ax.text((cut(TIGHT, 0) + cut(LOOSE, 0)) / 2 + 20, -0.02, LOOSE * 0.84, "Lost capacity",
                fontsize=ANNOT, color="#4a4a4a", ha="center", va="center", zorder=40)
        # axes
        ax.set_xlim(0, xmax); ax.set_ylim(0, 1); ax.set_zlim(0, zmax)
        ax.set_box_aspect(aspect, zoom=zoom)
        ax.set_xticks([0, 200, 400]); ax.set_yticks([0, 1]); ax.set_zticks([0, 50, 100])
        ax.set_yticklabels(["", "Full"])
        ax.tick_params(labelsize=FS, pad=-3)
        ax.tick_params(axis="z", pad=1)
        # labels follow their axes (mplot3d's default alignment)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_rotate_label(True)
        ax.set_xlabel("Decode batch size", fontsize=FS, labelpad=-3)
        ax.set_ylabel("KV-cache occupancy", fontsize=FS, labelpad=-5)
        ax.set_zlabel("Iteration latency (ms)", fontsize=FS, labelpad=0)
        ax.view_init(elev=elev, azim=azim)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor("#ececec")
            axis.line.set_linewidth(0.5)
            axis.line.set_color("#777777")
            axis._axinfo["tick"]["inward_factor"] = 0.0
            axis._axinfo["tick"]["outward_factor"] = 0.25
        ax.grid(False)
        ax.set_position(pos)
    return fig


def fig_3d_schematic():
    """The operating space drawn as a schematic cube in oblique projection,
    the way conceptual figures in networking papers draw a trade-off space:
    three arrowed axes from one origin, a light cube for the state space,
    thin black lines, flat fills, and level ticks on the vertical axis.
    The content is unchanged: latency surface, the 50 and 100 ms constraint
    lids, the feasible block under the tight lid and the lost wedge between
    the lids. Painted back to front by hand (depth runs along the KV axis
    only, so that order is exact)."""
    from matplotlib.patches import Polygon
    from matplotlib.lines import Line2D
    st = small_style()
    TIGHT, LOOSE = 50, 100
    xmax, zmax = 420.0, 124.0
    kv = lambda k: 1 + 0.12 * k ** 2                  # noqa: E731
    zf = lambda b, k: lat(b) * kv(k)                  # noqa: E731
    cut = lambda lvl, k: batch_at(lvl / kv(k))        # noqa: E731
    # oblique projection: x right, z up, KV (depth) toward the viewer, down-left
    W, H, DX, DY = 1.0, 0.60, 0.13, 0.19
    P = lambda x, y, z: (x / xmax * W - y * DX, z / zmax * H - y * DY)  # noqa: E731
    X0 = -85.0                                        # the axes' origin, left of the cube
    teal, grey = to_rgb("#3a8f8a"), to_rgb("#8a8a8a")
    c_surf, c_lost, c_feas = "#ececec", "#c8c8c8", tuple(1 - 0.5 * (1 - np.array(teal)))
    nb, nk = 60, 24
    bb = np.linspace(0, xmax, nb + 1)
    kk = np.linspace(0, 1, nk + 1)
    kf = np.linspace(0, 1, 60)

    x_lo = P(X0, 1.32, 0)[0] - 0.22                  # room for the KV label
    x_hi = P(xmax * 1.12, 0, 0)[0] + 0.03
    y_lo = P(0, 1.32, 0)[1] - 0.13
    y_hi = P(X0, 0, zmax * 1.14)[1] + 0.05
    fig_h = COL_W * (y_hi - y_lo) / (x_hi - x_lo)
    with plt.rc_context(st):
        fig = plt.figure(figsize=(COL_W, fig_h))
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_aspect("equal")
        ax.axis("off")

        def poly(pts3, fc, alpha=1.0, ec="none", lw=0.0, z=1):
            ax.add_patch(Polygon([P(*p) for p in pts3], closed=True, facecolor=fc,
                                 alpha=alpha, edgecolor=ec, linewidth=lw, zorder=z))

        def line(pts3, **kw):
            xy = np.array([P(*p) for p in pts3])
            ax.add_line(Line2D(xy[:, 0], xy[:, 1], **kw))

        # -- the state cube (back edges first) --
        cube = dict(color="#c4c4c4", lw=0.5, zorder=0)
        line([(0, 0, 0), (xmax, 0, 0), (xmax, 0, zmax), (0, 0, zmax), (0, 0, 0)], **cube)
        line([(0, 0, zmax), (0, 1, zmax)], **cube)
        line([(xmax, 0, zmax), (xmax, 1, zmax)], **cube)
        line([(xmax, 0, 0), (xmax, 1, 0)], **cube)

        # -- back wall (y = 0): feasible part and lost part --
        xs = np.linspace(0, cut(TIGHT, 0), 40)
        poly([(x, 0, zf(x, 0)) for x in xs] + [(xs[-1], 0, TIGHT), (0, 0, TIGHT)],
             teal, alpha=0.30, z=2)
        xs = np.linspace(cut(TIGHT, 0), cut(LOOSE, 0), 40)
        poly([(x, 0, zf(x, 0)) for x in xs] + [(xs[-1], 0, LOOSE), (xs[0], 0, LOOSE)],
             grey, alpha=0.30, z=2)

        # -- the surface, back to front, flat fills by region --
        for j in range(nk):
            for i in range(nb):
                q = [(bb[i], kk[j]), (bb[i + 1], kk[j]), (bb[i + 1], kk[j + 1]), (bb[i], kk[j + 1])]
                zs = [zf(x, y) for x, y in q]
                if max(zs) > zmax:
                    continue
                zc = np.mean(zs)
                fc = c_feas if zc <= TIGHT else (c_lost if zc <= LOOSE else c_surf)
                poly([(x, y, z) for (x, y), z in zip(q, zs)], fc, ec=fc, lw=0.15, z=3)
        # surface silhouette
        edge = dict(color="#555555", lw=0.5, zorder=4)
        for y in (0.0, 1.0):
            xs = np.linspace(0, min(xmax, batch_at(zmax / kv(y))), 80)
            line([(x, y, zf(x, y)) for x in xs], **edge)
        line([(0, k, zf(0, k)) for k in kf], **edge)
        # the two cut curves on the surface (where each lid meets it)
        line([(cut(TIGHT, k), k, TIGHT) for k in kf], color="#2f7f79", lw=0.9, zorder=8)
        line([(cut(LOOSE, k), k, LOOSE) for k in kf], color="#4c72b0", lw=0.8, zorder=8)

        # -- the wedge's near wall, along the tight cut, 50 -> 100 --
        poly([(cut(TIGHT, k), k, TIGHT) for k in kf] + [(cut(TIGHT, k), k, LOOSE) for k in kf[::-1]],
             grey, alpha=0.28, z=5)
        # -- lids --
        poly([(0, 0, TIGHT)] + [(cut(TIGHT, k), k, TIGHT) for k in kf] + [(0, 1, TIGHT)],
             teal, alpha=0.22, z=6)
        poly([(cut(TIGHT, k), k, LOOSE) for k in kf] + [(cut(LOOSE, k), k, LOOSE) for k in kf[::-1]],
             grey, alpha=0.22, z=6)
        # -- front walls (y = 1) and the left wall (x = 0) of the feasible block --
        xs = np.linspace(0, cut(TIGHT, 1), 40)
        poly([(x, 1, zf(x, 1)) for x in xs] + [(xs[-1], 1, TIGHT), (0, 1, TIGHT)],
             teal, alpha=0.30, z=7)
        poly([(0, k, zf(0, k)) for k in kf] + [(0, 1, TIGHT), (0, 0, TIGHT)], teal, alpha=0.25, z=7)
        xs = np.linspace(cut(TIGHT, 1), cut(LOOSE, 1), 40)
        poly([(x, 1, zf(x, 1)) for x in xs] + [(xs[-1], 1, LOOSE), (xs[0], 1, LOOSE)],
             grey, alpha=0.30, z=7)
        # -- lid outlines --
        for lvl, col, lw in ((TIGHT, "#2f7f79", 0.9), (LOOSE, "#4c72b0", 0.8)):
            line([(0, 0, lvl), (0, 1, lvl)], color=col, lw=lw, zorder=9)
            line([(0, 0, lvl), (cut(lvl, 0), 0, lvl)], color=col, lw=lw, zorder=9)
            line([(0, 1, lvl), (cut(lvl, 1), 1, lvl)], color=col, lw=lw, zorder=9)
        # front edges of the cube
        line([(0, 1, 0), (xmax, 1, 0), (xmax, 1, zmax), (0, 1, zmax), (0, 1, 0)], **cube)
        line([(0, 0, 0), (0, 1, 0)], **cube)

        # -- axes: arrows from one origin, left of and behind the cube --
        arr = dict(arrowstyle="-|>", lw=0.8, color="black", shrinkA=0, shrinkB=0,
                   mutation_scale=7)
        O = P(X0, 0, 0)
        ax.annotate("", xy=P(X0, 0, zmax * 1.12), xytext=O, arrowprops=arr, zorder=20)
        ax.annotate("", xy=P(xmax * 1.10, 0, 0), xytext=O, arrowprops=arr, zorder=20)
        ax.annotate("", xy=P(X0, 1.28, 0), xytext=O, arrowprops=arr, zorder=20)
        ax.text(*P(X0, 0, zmax * 1.14), "Iteration latency (ms)", ha="center", va="bottom",
                fontsize=FS, zorder=21)
        ax.text(*P(xmax * 1.10, 0, 1.5), "Decode batch size", ha="right", va="bottom",
                fontsize=FS, zorder=21)
        ang = np.degrees(np.arctan2(DY, DX))          # screen angle of the KV axis
        d = np.array([DX, DY]) / np.hypot(DX, DY)      # unit vector along the axis
        nrm = np.array([d[1], -d[0]])                  # perpendicular, toward the lower right
        mid = np.array(P(X0, 0.80, 0)) - 0.075 * nrm    # outer (upper-left) side
        ax.text(mid[0], mid[1], "KV-cache occupancy", ha="center", va="center",
                rotation=ang, rotation_mode="anchor", fontsize=FS, zorder=21)
        # level ticks on the vertical axis, dashed to the cube edge
        for lvl, col, name in ((TIGHT, "#2f7f79", "Tight constraint"),
                               (LOOSE, "#4c72b0", "Loose constraint")):
            line([(X0, 0, lvl), (0, 0, lvl)], color=col, lw=0.6, ls=(0, (2, 1.5)), zorder=20)
            ax.text(*P(X0 - 8, 0, lvl), f"{lvl} ms", ha="right", va="center", fontsize=FS,
                    color=col, zorder=21)
            # loose: above its lid; tight: just under its lid's back edge, so the
            # label does not sit on the loose lid's front edge
            if lvl == LOOSE:
                ax.text(*P(4, 0, lvl + 4), name, ha="left", va="bottom", fontsize=FS,
                        color=col, zorder=21)
            else:
                ax.text(*P(6, 0.02, lvl - 3), name, ha="left", va="top", fontsize=FS,
                        color=col, zorder=21)
        ax.text(*P(X0 - 8, 0, 0), "0", ha="right", va="center", fontsize=FS, zorder=21)
        # batch ticks along the front-bottom edge
        for v in (0, 200, 400):
            line([(v, 1, 0), (v, 1.06, 0)], color="black", lw=0.5, zorder=20)
            ax.text(*P(v, 1.10, 0), str(v), ha="center", va="top", fontsize=FS, zorder=21)
        line([(X0, 1.0, 0), (0, 1.0, 0)], color="black", lw=0.5, ls=(0, (2, 1.5)), zorder=20)
        # KV ticks on the depth axis
        ax.text(*P(X0 + 6, 1.02, 0), "Full", ha="left", va="top", fontsize=FS, zorder=21)
        # region key, in the empty corner under the batch axis
        from matplotlib.patches import Patch
        ax.legend([Patch(facecolor=c_feas, edgecolor="#2f7f79", lw=0.6),
                   Patch(facecolor=c_lost, edgecolor="#8a8a8a", lw=0.6)],
                  ["Feasible region", "Lost capacity"],
                  loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False,
                  fontsize=LEGEND_FS, handlelength=1.3, handleheight=0.9,
                  handletextpad=0.5, columnspacing=1.2, borderaxespad=0.3)

        # frame
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        fig.savefig(os.path.join(HERE, "operating_space_3d.png"), dpi=200)
        save(fig, os.path.join(HERE, "operating_space_3d.pdf"))


def fig_3d_single(**kw):
    """The earlier mplot3d rendering, kept under operating_space_3d_mpl."""
    fig = build_3d((COL_W, FIG_H_3D), **kw)
    fig.savefig(os.path.join(HERE, "operating_space_3d_mpl.png"), dpi=200)
    save(fig, os.path.join(HERE, "operating_space_3d_mpl.pdf"))


def fig_3d():
    """Two conceptual cubes: latency over (batch, KV occupancy), SLO plane at
    100 ms (loose residents) and at 50 ms (one tight resident)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    st = small_style()
    bb = np.linspace(0, B_MAX, 40)
    kk = np.linspace(0, 1, 40)
    Bg, Kg = np.meshgrid(bb, kk)
    Z = lat(Bg) * (1 + 0.25 * Kg ** 2)        # fuller KV cache slows each step a little
    with plt.rc_context(st):
        fig = plt.figure(figsize=(COL_W, 1.9))
        for i, (title, ms, col) in enumerate((("loose residents (100 ms pace)", 100, "#4c72b0"),
                                              ("one tight resident (50 ms pace)", 50, "#3a8f8a"))):
            ax = fig.add_subplot(1, 2, i + 1, projection="3d")
            feas = Z <= ms
            colors = np.empty(Z.shape + (4,))
            colors[feas] = (*to_rgb(col), 0.85)
            colors[~feas] = (0.85, 0.85, 0.85, 0.35)
            ax.plot_surface(Bg, Kg, Z, facecolors=colors, rstride=1, cstride=1,
                            linewidth=0, antialiased=False, shade=False)
            # SLO plane
            Pb, Pk = np.meshgrid([0, B_MAX], [0, 1])
            ax.plot_surface(Pb, Pk, np.full_like(Pb, ms, dtype=float), color=col,
                            alpha=0.10, linewidth=0)
            ax.plot([0, B_MAX, B_MAX, 0, 0], [0, 0, 1, 1, 0], [ms] * 5, color=col, lw=0.6)
            ax.set_zlim(0, 140)
            ax.set_xlim(0, B_MAX)
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 200, 400])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["0", "full"])
            ax.set_zticks([0, 50, 100])
            ax.tick_params(labelsize=FS, pad=-2)
            ax.tick_params(axis="z", pad=1)
            ax.set_xlabel("batch", fontsize=ANNOT, labelpad=-8)
            ax.set_ylabel("KV cache", fontsize=ANNOT, labelpad=-8)
            ax.set_zlabel("latency (ms)", fontsize=ANNOT, labelpad=-9)
            ax.view_init(elev=22, azim=-58)
            ax.set_title(title, fontsize=ANNOT, pad=-2)
            ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.set_edgecolor("#dddddd")
                axis.line.set_linewidth(0.4)
            ax.grid(False)
        fig.subplots_adjust(left=0.0, right=0.90, top=1.0, bottom=0.08, wspace=0.0)
        fig.text(0.5, 0.015, "coloured: feasible operating region (latency under the resident pace)",
                 ha="center", va="bottom", fontsize=ANNOT, color="#555555")
        finish(fig, "operating_space_3d_pair")


def main():
    todo = ["2d", "3d"] if FMT == "all" else [FMT]
    for f in todo:
        {"2d": fig_2d, "3d": fig_3d_schematic, "3dmpl": fig_3d_single, "3dpair": fig_3d}[f]()


if __name__ == "__main__":
    main()
