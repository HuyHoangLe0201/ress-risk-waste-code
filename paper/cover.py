"""How many policies does a fleet genuinely distinguish?

Deleting hull vertices one at a time answers nothing: every vertex can go with
negligible loss because its neighbours substitute, so the single-deletion count is
zero at every grid.  That is itself worth knowing -- no policy is individually
necessary -- but the question was posed wrongly.

The right object is a COVERING.  Find the smallest set S of policies such that

    min_{p in S} (a_p + b_p chi)  <=  (1 + eps(chi)) L*(chi)   for all chi,

with the tolerance eps(chi) = chi/n taken from the lattice spacing of equation
(11), so the tolerance is the fleet's own resolution and grows with the cost ratio
exactly as the resolution degrades.  Because L* is a lower envelope of lines, the
greedy sweep that extends each chosen line until it violates the tolerance is
optimal for this covering.

That number -- not the hull size -- is how many policies a fleet distinguishes.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402


def coords(life, idx, R, rm, pa, Tb, ngrid):
    lu = np.array([pa[i][R[i] > 0].min() for i in idx])
    g = np.linspace(np.quantile(lu, .02), lu.max() + .25 * Tb, ngrid)
    C = np.array([[M.unit_cost(i, e, R, rm, life) for i in idx] for e in g])
    om = 1.0 - C[:, :, 1].mean(1) / Tb
    pf = C[:, :, 2].mean(1)
    return 1.0 / (1.0 - om), pf / (1.0 - om), g


def greedy_cover(a, b, chis, eps):
    """Fewest lines whose lower envelope stays within (1+eps) of L* everywhere."""
    Lstar = np.array([np.min(a + b * c) for c in chis])
    tol = Lstar * (1.0 + eps)
    chosen, i = [], 0
    while i < len(chis):
        # among policies optimal-enough at chis[i], take the one covering furthest
        cand = np.nonzero(a + b * chis[i] <= tol[i] + 1e-15)[0]
        if len(cand) == 0:
            cand = np.array([int(np.argmin(a + b * chis[i]))])
        reach = []
        for p in cand:
            ok = a[p] + b[p] * chis <= tol + 1e-15
            j = i
            while j < len(chis) and ok[j]:
                j += 1
            reach.append(j)
        k = int(np.argmax(reach))
        chosen.append(int(cand[k]))
        nxt = reach[k]
        if nxt <= i:
            nxt = i + 1
        i = nxt
    return chosen


def run(name, ld):
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=0)
    pool = np.concatenate([cal, val])
    n = len(pool)
    chis = np.geomspace(0.2, 60.0, 500)
    print("=" * 78)
    print("%-9s n=%d" % (name, n))
    print("   grid   hull   cover at eps=chi/n   cover at eps=0.01   thresholds kept")
    for ng in (100, 200, 300, 400, 800):
        a, b, g = coords(life, pool, R, rm, pa, Tb, ng)
        from ehull import hull as H
        raw = len(H(a, b))
        c1 = greedy_cover(a, b, chis, chis / n)
        c2 = greedy_cover(a, b, chis, 0.01)
        th = sorted(round(g[p] / Tb, 3) for p in c1)
        print("   %4d    %3d          %3d                 %3d          %s"
              % (ng, raw, len(c1), len(c2), th))


if __name__ == "__main__":
    import os.path as _p; sys.path.insert(0, _p.dirname(_p.abspath(__file__)))
    for nm, ld in (("FD002", M.FD002), ("Severson", M.load_severson)):
        run(nm, ld)
