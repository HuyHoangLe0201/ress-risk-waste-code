r"""Figure for the fleet-split reading of a safety requirement.

The result is geometric, so the figure has to carry the mechanism rather than the
number, and drawing it made the mechanism sharper than "the arc is not convex".

What panel (a) shows on FD004, at the paper's own case (iii), is that the achievable
set is a STAIRCASE.  On n units P_f moves in steps of 1/n -- 1/124 here -- so the
family offers failure probabilities 0, 0.0081, 0.0161 and nothing between, while
omega varies continuously along each tread.  A requirement of 0.01 falls strictly
between two steps.  No single rule can meet it with equality: the best one retreats
to the step below and pays for the gap in wasted life, which is where the black point
sits.  The chord between the two rules bracketing the requirement passes below that
corner, and every point of the chord is reachable because mixing is convex
combination in these coordinates -- so the split meets the requirement exactly and
at lower cost.  The shaded gap is the saving, and it is a consequence of the lattice
rather than of any curvature.

Panel (b) is what a regulator sets against what an operator pays: the equivalent
price as a function of the requirement.  It is a step function, one step per envelope
edge, because along an edge the two rules and their tie ratio are fixed and only the
proportion moves.

Panel (c) is the price of insisting on one rule -- the saving the split recovers --
against the requirement, on all three fleets.

Conventions follow makefigs_ress.py: the local style, 6.48in exactly, crop off.
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from figstyle import (setup, clean, ring, COL, BLUE, ORANGE, VIOLET,
                      INK, INK2, INK3, BAND)                     # noqa: E402
import figstyle as _fs                                           # noqa: E402
# Identity, not width: pinned to 6.48 this fired on the legitimate move to the
# Elsevier 190 mm width rather than on the shadowing it exists to catch.
assert "RESS_paper" in os.path.abspath(_fs.__file__), \
    "IEEE figstyle shadowed the RESS one"

import os.path as _p; sys.path.append(_p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import matplotlib.pyplot as plt                                  # noqa: E402
import makefigs6 as M                                            # noqa: E402
from chance_split import family, dual, lower_hull, best_pair, L, TOL  # noqa: E402

setup()
FLEETS = (("FD002", M.FD002, BLUE), ("FD004", M.FD004, ORANGE),
          ("Severson", M.load_severson, VIOLET))
CHI = 0.3


def save(fig, path):
    fig.savefig(path)
    w, _ = fig.get_size_inches()
    assert abs(w - COL) < 1e-6, "%s authored %.3fin, not %.3f" % (path, w, COL)
    print("wrote %s at %.4fin" % (path, w))


def split_at(om, pf, a, b, alpha, chi=CHI):
    """The constrained optimum: the two rules, the proportions, the point."""
    val, i, j, t = best_pair(a, b, chi, alpha)
    q = np.array([1.0 - t, t])
    p = q / (1.0 - om[[i, j]])
    p = p / p.sum()
    return (i, j, p, float(p @ om[[i, j]]), float(p @ pf[[i, j]]), val)


def panel_a(ax, om, pf, a, b, alpha=0.01):
    V = lower_hull(a, b)
    i, j, p, om_m, pf_m, val = split_at(om, pf, a, b, alpha)
    feas = pf <= alpha + TOL
    jd = int(np.argmin(np.where(feas, L(om, pf, CHI), np.inf)))

    # The family is drawn in ITS OWN order, by threshold.  Sorting by P_f instead
    # stacks every threshold beyond the zero-failure boundary at P_f=0, where they
    # differ only in omega, and draws that stack as a spurious vertical tail.
    k0 = int(np.argmax(pf <= 0.0)) + 1 if (pf <= 0.0).any() else len(pf)
    ax.plot(pf[:k0], om[:k0], "-", color=INK3, lw=1.0, zorder=2)

    lo, hi = (i, j) if pf[i] < pf[j] else (j, i)
    ax.plot([pf[lo], pf[hi]], [om[lo], om[hi]], "-", color=ORANGE, lw=2.0, zorder=4)
    m = np.zeros(len(pf), bool)
    m[:k0] = ((pf[:k0] >= pf[lo] - 1e-9) & (pf[:k0] <= pf[hi] + 1e-9))
    ax.fill_between(pf[m], om[m],
                    np.interp(pf[m], [pf[lo], pf[hi]], [om[lo], om[hi]]),
                    color=ORANGE, alpha=0.13, lw=0, zorder=1)

    ax.axvline(alpha, color=INK2, lw=0.9, ls=(0, (4, 2)), zorder=3)
    ring(ax, pf[jd], om[jd], INK, ms=5.0)
    ring(ax, pf_m, om_m, ORANGE, ms=5.0, marker="D")
    for k in (i, j):
        ring(ax, pf[k], om[k], ORANGE, ms=3.6)

    # limits from the points actually drawn: lo and hi order the two rules by P_f,
    # and omega runs the other way, so taking om[hi] for the top collapses the axis
    ys = [om[i], om[j], om[jd], om_m]
    xs = [pf[i], pf[j], pf[jd], pf_m, alpha]
    dy, dx = max(ys) - min(ys), max(xs) - min(xs)
    ax.set_xlim(min(xs) - 0.28 * dx, max(xs) + 0.34 * dx)
    ax.set_ylim(min(ys) - 0.30 * dy, max(ys) + 0.20 * dy)

    ax.annotate("best single rule", xy=(pf[jd], om[jd]),
                xytext=(pf[jd] + 0.0052, om[jd] + 0.0052),
                color=INK, fontsize=6.4, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6,
                                shrinkA=0, shrinkB=3))
    ax.annotate("split $%d/%d$" % (round(100 * p[0]), round(100 * p[1])),
                xy=(pf_m, om_m), xytext=(pf_m - 0.0044, om_m - 0.0052),
                color=ORANGE, fontsize=6.4, ha="center",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.6,
                                shrinkA=0, shrinkB=3))
    ax.text(alpha + 0.0005, max(ys) + 0.10 * dy, "$\\alpha$",
            color=INK2, fontsize=6.8, ha="left")
    xg = 0.5 * (pf[lo] + pf[hi])
    ax.text(xg, 0.5 * (np.interp(xg, [pf[lo], pf[hi]], [om[lo], om[hi]])
                       + np.interp(xg, pf[:k0][::-1], om[:k0][::-1])) - 0.0004,
            "gain", color=ORANGE, fontsize=6.4, ha="center", va="center")

    ax.set_xlabel("$P_f$")
    ax.set_ylabel("$\\omega$")
    ax.set_title("(a) the chord the curve cannot reach", fontsize=7.2, pad=4)
    clean(ax)


def sweep(om, pf, a, b, grid):
    price, save_pc = [], []
    for al in grid:
        i, j, p, om_m, pf_m, val = split_at(om, pf, a, b, al)
        feas = pf <= al + TOL
        det = float(np.min(np.where(feas, L(om, pf, CHI), np.inf)))
        price.append(np.inf if abs(b[j] - b[i]) < TOL
                     else (a[i] - a[j]) / (b[j] - b[i]))
        save_pc.append(100.0 * (det - val) / det)
    return np.array(price), np.array(save_pc)


def main():
    grid = np.linspace(0.004, 0.06, 160)
    data = {}
    for nm, ld, col in FLEETS:
        om, pf, n = family(ld)
        a, b = dual(om, pf)
        data[nm] = (om, pf, a, b, col, n) + sweep(om, pf, a, b, grid)

    fig, axes = plt.subplots(1, 3, figsize=(COL, COL / 3.35))
    om, pf, a, b = data["FD004"][:4]
    panel_a(axes[0], om, pf, a, b)

    ax = axes[1]
    for nm, ld, col in FLEETS:
        price = data[nm][6]
        ax.step(grid, price, where="post", color=col, lw=1.3, label=nm)
    ax.axhline(CHI, color=INK2, lw=0.8, ls=(0, (4, 2)))
    ax.text(0.0575, CHI * 1.13, "$\\chi$ paid", color=INK2, fontsize=6.4, ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("requirement $\\alpha$")
    ax.set_ylabel("equivalent price $\\chi_{ij}$")
    ax.set_title("(b) the price a requirement implies", fontsize=7.2, pad=4)
    ax.legend(frameon=False, fontsize=6.3, handlelength=1.5, borderpad=0.2)
    clean(ax)

    ax = axes[2]
    for nm, ld, col in FLEETS:
        ax.plot(grid, data[nm][7], color=col, lw=1.3)
    ax.set_xlabel("requirement $\\alpha$")
    ax.set_ylabel("saving of the split (\\%)")
    ax.set_title("(c) cost of insisting on one rule", fontsize=7.2, pad=4)
    clean(ax)

    fig.tight_layout(pad=0.30, w_pad=1.5)
    save(fig, "fig14.pdf")


if __name__ == "__main__":
    main()
