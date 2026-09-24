r"""Figures 5-7 for the RESS manuscript: the results added in this revision.

Twenty results were added without a single figure, and the strongest of them -- the
optimal rule, the predictor-comparison dichotomy, the record structure, the
average-versus-maximum split -- were text only.  These three cover them.

Three conventions differ from the earlier figures, all deliberate.

Width.  The house style cropped every figure to its own content, so the authored
figsize never survived and the four earlier figures reached the page 6.22, 6.48, 6.29
and 6.37in wide -- four scale factors under width=\linewidth, hence four different
effective type sizes in one paper.  Passing bbox_inches=None to savefig does not
switch that off: matplotlib resolves None back to the rcParam.  The crop is disabled
in the style itself, and every figure here is authored at 6.48in exactly.

Style.  This module used to put TR_paper on sys.path before importing figstyle, so it
picked up the IEEE style: 7.2pt type, Type 3 fonts and DejaVu Sans mathtext beside the
Times labels.  Elsevier does not accept Type 3 fonts.  The local style is imported
first, and the assertion below fails if the wrong one is ever resolved.

Colour.  Figure 5 compares four predictors but the validated palette holds three hues.
A fourth was searched for; only #177f57 passes all six checks and its CVD separation is
6.1, inside the floor band rather than the target, so it is not used.  The three close
predictors take the three hues and k-nearest neighbours, the one clearly separated
family, is carried by panel (b), where the rows are categorical rather than coloured.

Records.  Panel 6(b) counts record minima and must therefore read the RAW predictor
path.  The prep helper returns paths that are already running minima, against which the
record test is vacuously true and every fleet reports 100 per cent; the raw series is
taken from makefigs6.prep directly instead.
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
# TR_paper ships an IEEE-width style under the same module name, so the local
# one is imported first and checked; only then does TR_paper join the path.
from figstyle import (halo, setup, clean, ring, note, COL, BLUE, ORANGE, VIOLET,
                      INK, INK2, INK3, BAND)                     # noqa: E402
import figstyle as _fs                                           # noqa: E402
# Identity, not width: pinned to 6.48 this fired on the legitimate move to the
# Elsevier 190 mm width rather than on the shadowing it exists to catch.
assert "RESS_paper" in os.path.abspath(_fs.__file__), \
    "IEEE figstyle shadowed the RESS one"

import os.path as _p; sys.path.append(_p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import matplotlib.pyplot as plt                                  # noqa: E402
import makefigs6 as M                                            # noqa: E402
from quota import prep                                           # noqa: E402
from crossing_floor import predictors, envelope_at               # noqa: E402
from two_sided import frontier, interval                         # noqa: E402

setup()


def save(fig, path):
    """Save at the authored size, and check it: a silent crop is the failure."""
    fig.savefig(path)
    w, _ = fig.get_size_inches()
    assert abs(w - COL) < 1e-6, "%s authored %.3fin, not %.3f" % (path, w, COL)
    plt.close(fig)


def anno(ax, fx, fy, s, color=INK2, size=8.8, ha="left", va="top"):
    """A note placed in axes fractions, so it cannot drift onto the data."""
    ax.text(fx, fy, s, transform=ax.transAxes, color=color, fontsize=size,
            ha=ha, va=va, zorder=7, path_effects=halo())


# ----------------------------------------------------------------- figure 5
def fig5():
    seq, life = M.FD002()
    X = M.design(seq)
    R = {i: np.arange(len(s) - 1, -1, -1.0) for i, s in enumerate(seq)}
    n = len(seq)
    p = np.random.default_rng(0).permutation(n)
    cal, val = p[:n // 2], p[n // 2:]
    P = predictors(X, R, cal, val, 0)
    names = list(P)

    chis = np.logspace(0, np.log10(60.0), 60)
    env = {k: np.array([envelope_at(P[k], R, val, c, ngrid=260) for c in chis])
           for k in P}

    rng = np.random.default_rng(7)
    draws = {k: [] for k in names}
    for _ in range(150):
        bs = rng.choice(val, len(val), replace=True)
        for k in names:
            draws[k].append(envelope_at(P[k], R, bs, 9.0, ngrid=400))
    pt = {k: envelope_at(P[k], R, val, 9.0, ngrid=400) for k in names}

    fig, ax = plt.subplots(1, 2, figsize=(COL, 2.78), layout="constrained",
                           gridspec_kw={"width_ratios": [1.1, 1]})
    a, b = ax
    clean(a); clean(b)

    close = [("ridge", BLUE, "ridge"), ("randFourier", ORANGE, "random Fourier"),
             ("network", VIOLET, "network")]
    for k, c, _ in close:
        a.plot(chis, env[k], color=c, lw=1.4, zorder=3)
    for u, v in (("ridge", "randFourier"), ("randFourier", "network"),
                 ("ridge", "network")):
        d = env[u] - env[v]
        for j in np.nonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]:
            ring(a, chis[j], env[u][j], INK2, ms=2.6)
    a.set_xscale("log")
    a.minorticks_off()
    a.set_xticks([1, 3, 10, 30, 60])
    a.set_xticklabels(["1", "3", "10", "30", "60"])
    a.set_xlabel(r"cost ratio $\chi$")
    a.set_ylabel(r"optimal cost $L^{\ast}(\chi)$")
    lo, hi = min(env[k].min() for k, _, _ in close), max(env[k].max()
                                                         for k, _, _ in close)
    a.set_ylim(lo - 0.004, hi + 0.010)
    for k, c, lab in close:
        note(a, chis[-1] * 1.03, env[k][-1], lab, color=INK2, fontsize=8.8)
    a.set_xlim(0.95, 150)
    anno(a, 0.03, 0.97, "rings mark crossings")
    anno(a, 0.03, 0.88, r"$k$-NN lies far above, at %.2f" % pt["kNN15"])
    a.set_title("(a)  three predictors that nearly tie", loc="left", color=INK,
                pad=3)

    pairs = [(names[i], names[j]) for i in range(len(names))
             for j in range(i + 1, len(names))]
    ys = np.arange(len(pairs))[::-1]
    print("   fig5(b) differences at chi=9:")
    for y, (u, v) in zip(ys, pairs):
        d = pt[u] - pt[v]
        se = float(np.std(np.array(draws[u]) - np.array(draws[v]), ddof=1))
        res = abs(d) > 2 * se
        col = BLUE if res else INK3
        b.plot([d - 2 * se, d + 2 * se], [y, y], color=col, lw=1.2,
               solid_capstyle="butt", zorder=3)
        ring(b, d, y, col, ms=3.0)
        print("      %-12s - %-12s %+.4f +- %.4f   %s"
              % (u, v, d, se, "resolved" if res else "inside floor"))
    b.axvline(0, color=INK2, lw=0.6, zorder=1)
    b.set_yticks(ys)
    short = {"randFourier": "rFF", "network": "net", "kNN15": "kNN",
             "ridge": "ridge"}
    b.set_yticklabels(["%s $-$ %s" % (short[u], short[v]) for u, v in pairs],
                      fontsize=8.8)
    b.set_xlabel(r"envelope difference at $\chi=9$   ($\pm2$ s.e.)")
    b.set_ylim(-0.7, len(pairs) - 0.15)
    b.grid(False, axis="y")
    anno(b, 0.03, 0.99, "grey: inside the fleet's floor", color=INK2)
    b.set_title("(b)  which differences a fleet can resolve", loc="left",
                color=INK, pad=3)

    # Not fig5.pdf: the manuscript does not include this figure, and build_submission.py
    # rightly refuses to ship a fig<N>.pdf that nothing references. The computation stays --
    # its printout above is the predictor-family comparison against the floor -- but the
    # plot is named out of the submission's numbering, as fig1_superseded.pdf already is.
    save(fig, "fig5_unused.pdf")
    print("fig5 written (as fig5_unused.pdf; the manuscript does not include it)")


# ----------------------------------------------------------------- figure 6
def fig6():
    fleets = [("FD002", M.FD002, BLUE), ("FD004", M.FD004, ORANGE),
              ("Severson", M.load_severson, VIOLET)]
    fig, ax = plt.subplots(1, 2, figsize=(COL, 2.16), layout="constrained",
                           gridspec_kw={"width_ratios": [1.25, 1]})
    a, b = ax
    clean(a); clean(b)

    seq, life = M.FD002()
    Rr, rm, pa, cal, val, _ = M.prep(seq, life, seed=0)
    i = int(val[3])
    path = np.asarray(pa[i], float)          # RAW predictor path
    run = np.minimum.accumulate(path)
    t = np.arange(len(path))
    rec = path <= run + 1e-12

    a.plot(t, path, color=INK3, lw=0.7, zorder=2)
    a.plot(t, run, color=BLUE, lw=1.4, zorder=3)
    a.plot(t[rec], path[rec], "o", ms=2.2, color=BLUE, zorder=4,
           markeredgewidth=0)
    a.set_xlabel("cycle")
    a.set_ylabel(r"predicted remaining life $\hat R_u$")
    anno(a, 0.03, 0.20, "grey: the raw path", color=INK2)
    anno(a, 0.03, 0.12, "blue: its running minimum and records")
    a.set_title("(a)  one unit, %d of %d cycles are records"
                % (int(rec.sum()), len(path)), loc="left", color=INK, pad=3)

    # (b) one point per unit rather than one bar per fleet.  Three bars carried
    # three numbers over half a figure, and coloured a single series by its own
    # category label; the per-unit cloud carries the same three shares as its
    # slopes and shows the spread they average over.
    print("   fig6(b) visible shares:")
    hi = 0
    for nm, ld, col in fleets:
        seqf, lifef = ld()
        _, _, paf, calf, valf, _ = M.prep(seqf, lifef, seed=0)
        raw = [np.asarray(paf[j], float) for j in range(len(seqf))]
        T = np.array([len(q) for q in raw], float)
        r = np.array([int((q <= np.minimum.accumulate(q) + 1e-12).sum())
                      for q in raw], float)
        vis = 100.0 * r.sum() / T.sum()
        print("      %-9s %5.1f%%  median unit %d cycles, %d records"
              % (nm, vis, int(np.median(T)), int(np.median(r))))
        # Open marks: the two turbofan fleets occupy the same region of this
        # panel, and filled dots let whichever is drawn second bury the other.
        b.plot(T, r, "o", ms=3.0, markerfacecolor="none", markeredgecolor=col,
               markeredgewidth=0.6, linestyle="none", alpha=0.85, zorder=3,
               label="%s, %.0f%%" % (nm, vis))
        hi = max(hi, float(T.max()))
    # Log-log: the battery cells run to 2250 cycles against 200 for the turbofan
    # units, and on a linear axis the two turbofan fleets collapse into one blob
    # in the corner.  A constant record share is a constant offset below the
    # diagonal here, which is the comparison the panel is for.
    lo, lim = 80.0, 1.35 * hi
    b.plot([lo, lim], [lo, lim], color=INK3, lw=0.7, ls=(0, (3, 2)), zorder=2)
    # Above the diagonal, not below it. The line runs at 45 degrees on these log axes, so
    # any offset down-and-left keeps the label on it; the region above it is empty,
    # because no unit reaches the diagonal.
    # Above the diagonal at its top corner: the region above the line is empty, because no
    # unit reaches it. Two other placements were tried and both failed -- partway down the
    # line it abuts the legend at the upper left, and just below the corner the diagonal
    # itself runs through the words. What made the corner work is the shorter panel title
    # below; at the old length the two met in the middle of the panel.
    b.annotate("every cycle a record", (lim, lim), xytext=(-3, 4),
               textcoords="offset points", ha="right", va="bottom",
               color=INK2, fontsize=8.8)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlim(lo, lim)
    b.set_ylim(6, lim)
    b.set_xlabel("monitored cycles")
    b.set_ylabel("of which records")
    b.legend(loc="upper left", ncol=1, handlelength=0.8, handletextpad=0.4,
             labelspacing=0.2)
    b.set_title("(b)  the rest is invisible", loc="left", color=INK, pad=3)

    save(fig, "fig6.pdf")
    print("fig6 written")


# ----------------------------------------------------------------- figure 7
def fig7():
    paths, Rs, lives, lu = prep(M.FD002)
    N = len(paths)
    rng = np.random.default_rng(0)
    sizes = [15, 25, 40, 65, 105]
    e_fix = float(np.quantile(lu, 0.60))

    def coords(sub, ell):
        w, f = [], []
        for i in sub:
            h = np.nonzero(paths[i] <= ell)[0]
            if len(h) == 0 or Rs[i][h[0]] <= 0:
                w.append(0.0); f.append(1.0)
            else:
                w.append(float(Rs[i][h[0]])); f.append(0.0)
        return float(np.mean(w)) / float(lives[sub].mean()), float(np.mean(f))

    allu = np.arange(N)
    om_f, pf_f = coords(allu, e_fix)
    L_full = (1 + 9.0 * pf_f) / (1 - om_f)
    om0_full = coords(allu, lu.max())[0]

    bias_fix, bias_w0 = [], []
    for s in sizes:
        v, w = [], []
        for _ in range(500):
            sub = rng.choice(N, s, replace=False)
            om, pf = coords(sub, e_fix)
            if om < 1:
                v.append((1 + 9.0 * pf) / (1 - om))
            w.append(coords(sub, lu[sub].max())[0])
        bias_fix.append(100 * (np.mean(v) - L_full) / L_full)
        bias_w0.append(100 * (np.mean(w) - om0_full) / om0_full)

    grid = np.linspace(np.quantile(lu, 0.02), lu.max(), 110)
    om, pf = frontier(paths, Rs, lives, allu, grid)
    k = om < 1.0
    om, pf = om[k], pf[k]
    iv = [interval(om, pf, j) for j in range(len(om))]
    live = [(j, a_, b_) for j, (a_, b_) in enumerate(iv) if b_ > a_ + 1e-9]

    fig, ax = plt.subplots(1, 2, figsize=(COL, 2.16), layout="constrained",
                           gridspec_kw={"width_ratios": [1, 1.05]})
    a, b = ax
    clean(a); clean(b)

    a.axhline(0, color=INK2, lw=0.6, zorder=1)
    a.plot(sizes, bias_fix, "-o", color=BLUE, ms=3.0, markeredgewidth=0, zorder=3)
    a.plot(sizes, bias_w0, "-o", color=ORANGE, ms=3.0, markeredgewidth=0, zorder=3)
    a.set_xscale("log")
    a.minorticks_off()
    a.set_xticks(sizes)
    a.set_xticklabels([str(s) for s in sizes])
    a.set_xlabel("fleet size $n$")
    a.set_ylabel("relative bias (%)")
    a.set_ylim(min(bias_w0) - 7, 12)
    anno(a, 0.05, 0.97, "fixed policy: an average", color=INK2)
    anno(a, 0.05, 0.30, r"$\omega_0$: a maximum", color=INK2)
    a.set_title("(a)  what a fleet can estimate", loc="left", color=INK, pad=3)

    for j, lo, hi in live:
        hi_ = min(hi, 60.0)
        if hi_ > max(lo, 0.02):
            b.plot([max(lo, 0.02), hi_], [pf[j], pf[j]], color=BLUE, lw=1.6,
                   solid_capstyle="butt", zorder=3)
    b.set_xscale("log")
    b.minorticks_off()
    b.set_xticks([0.03, 0.1, 0.3, 1, 3, 10, 30])
    b.set_xticklabels(["0.03", "0.1", "0.3", "1", "3", "10", "30"])
    b.set_xlabel(r"cost ratio $\kappa$")
    b.set_ylabel(r"failure probability $P_f$")
    b.set_xlim(0.02, 70)
    b.set_ylim(-0.04, 1.12)
    anno(b, 0.30, 0.99, "each bar is one admissible point,\nover the cost ratios where it is optimal")
    b.set_title("(b)  %d admissible points tile the axis" % len(live),
                loc="left", color=INK, pad=3)

    save(fig, "fig7.pdf")
    print("fig7 written")


if __name__ == "__main__":
    fig6()
    fig7()
    fig5()
