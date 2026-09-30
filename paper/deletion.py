r"""Is "no vertex is individually necessary" a property of the object or of the fleet size?

Section 4 deletes each hull vertex in turn, finds the cost rise under the fleet's own
resolution chi/n at every grid, and concludes that the question has to be posed as a
covering.  That conclusion defines the number the section then reports.

The rise and the tolerance need not decay at the same rate.  By the balance of Theorem 6
the vertices sit n^{-1/3} apart, so the depth of one below the chord of its neighbours is
of order c h^2 ~ n^{-2/3}, while the tolerance (chi/n) L* is of order n^{-1}.  If that is
right the ratio grows as n^{1/3} and a large enough fleet makes every vertex necessary.

Two details make the comparison faithful.

Each vertex is optimal at its own cost ratio and must be judged there.  Point-line duality
supplies it: a policy (b, a) is the line a + b*chi, deleting a vertex replaces it by the
better of its two neighbours, and the largest resulting rise is where those neighbour lines
cross, at chi_c = minus the slope of the chord joining them in the dual.  The rise there is
the VERTICAL depth of the vertex below that chord.  That identity is verified below against
a direct recomputation rather than assumed.

And the claim is that NO vertex is necessary, so the statistic is the largest ratio
rise_j / tolerance_j over vertices -- not the largest rise divided by the tolerance that
happens to sit at that vertex.  Those differ, because the deepest vertex tends to lie near
the zero-failure boundary where chi_c, and therefore the tolerance, is largest too.

The window is the difficulty.  The fleets support n = 40 to 260, under one decade, and the
theory is asymptotic; the same simulation is therefore run over a wide range to check the
exponent and then over the fleet's own range to see what that range would report.
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import os.path as _p; sys.path.append(_p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
import hullreal as HR                                            # noqa: E402
import argminrate as AR                                          # noqa: E402
from hullscale import lower_hull                                 # noqa: E402


def dual(pf, om):
    """(P_f, omega) -> the dual point (b, a) whose line is the cost a + b*chi."""
    keep = om < 1.0
    a = 1.0 / (1.0 - om[keep])
    b = pf[keep] / (1.0 - om[keep])
    o = np.argsort(b)
    return b[o], a[o]


def worst_ratio(b, a, n):
    """max over interior hull vertices of (rise from deleting it) / (its tolerance)."""
    h = lower_hull(b, a)
    if len(h) < 3:
        return None
    x, y = b[h], a[h]
    best = 0.0
    for j in range(1, len(x) - 1):
        db = x[j + 1] - x[j - 1]
        if db <= 0:
            continue
        slope = (y[j + 1] - y[j - 1]) / db
        chi_c = -slope
        if chi_c <= 0:
            continue
        rise = y[j - 1] + slope * (x[j] - x[j - 1]) - y[j]
        Ls = float(np.min(a + b * chi_c))
        tol = chi_c / n * Ls
        if tol > 0:
            best = max(best, rise / tol)
    return best if best > 0 else None


def identity_error(b, a):
    """|vertical depth - directly recomputed cost rise|, over every interior vertex."""
    h = lower_hull(b, a)
    x, y = b[h], a[h]
    worst = 0.0
    for j in range(1, len(x) - 1):
        db = x[j + 1] - x[j - 1]
        slope = (y[j + 1] - y[j - 1]) / db
        chi_c = -slope
        rise = y[j - 1] + slope * (x[j] - x[j - 1]) - y[j]
        keep = np.ones(len(x), bool)
        keep[j] = False
        direct = (float(np.min(y[keep] + x[keep] * chi_c))
                  - float(np.min(y + x * chi_c)))
        worst = max(worst, abs(direct - rise))
    return worst


def sim(sizes, reps=200, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for n in sizes:
        v = []
        for _ in range(reps):
            b, a = AR.sample(n, rng)
            r = worst_ratio(b, a, n)
            if r:
                v.append(r)
        out.append(float(np.median(v)))
    sl = float(np.polyfit(np.log(sizes), np.log(out), 1)[0])
    return out, sl


def run(name, ld, seed=0, reps=40):
    tab, lives = HR.prep(ld)
    N = len(tab[0])
    rng = np.random.default_rng(seed)
    sizes = [s for s in (40, 70, 110, 170, 260) if s <= N]
    vals = []
    for s in sizes:
        v = []
        for _ in range(reps):
            pf, om = HR.pareto(tab, lives, rng.choice(N, s, replace=False))
            if pf is None:
                continue
            r = worst_ratio(*dual(pf, om), n=s)
            if r:
                v.append(r)
        vals.append(float(np.median(v)))
    sl = float(np.polyfit(np.log(sizes), np.log(vals), 1)[0])
    print("%-10s %s  exponent %+.2f" %
          (name, " ".join("%7.2f" % v for v in vals), sl))
    return sl


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    b, a = AR.sample(2000, rng)
    print("duality identity, worst |depth - recomputed rise| = %.2e"
          % identity_error(b, a))

    print("=" * 78)
    wide = (300, 1000, 3000, 10000, 30000)
    v, sl = sim(wide)
    print("simulation, wide range %s" % str(wide))
    print("   ratio %s   exponent %+.2f   (theory +0.33)"
          % (" ".join("%7.2f" % t for t in v), sl))

    narrow = (40, 70, 110, 170, 260)
    v2, sl2 = sim(narrow)
    print("simulation, the fleets' own range %s" % str(narrow))
    print("   ratio %s   exponent %+.2f   <- what this window reports"
          % (" ".join("%7.2f" % t for t in v2), sl2))

    print("=" * 78)
    print("%-10s %s  %s" % ("fleet", "  ".join("%5d" % s for s in narrow), "exponent"))
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
