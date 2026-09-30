"""Which hull vertices are real, and which are noise?

The vertex count fluctuates 16 to 21 with the grid, and I explained that as "where
the grid points fall", which is a description rather than a reason.  The reason is
near-collinearity: where the frontier is nearly straight, a vertex contributes
almost nothing, and whether it survives the hull is decided by arithmetic far
below the fleet's resolution.

The paper already has the resolution.  Equation (11) gives the lattice spacing
chi/n: cost differences smaller than that do not correspond to distinct operating
points.  So the statistically meaningful object is not the hull but the
EPSILON-HULL -- keep a vertex only if deleting it would raise the optimal cost, at
the cost ratio where it is best, by more than the noise floor.

Prediction: the filtered count is smaller and stable in the grid, where the raw
count is neither.  If it is not stable either, the notion of "how many policies a
fleet distinguishes" has no well-defined answer and that should be said.
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
    return 1.0 / (1.0 - om), pf / (1.0 - om)


def hull(a, b):
    o = np.lexsort((a, b))
    x, y = b[o], a[o]
    st = []
    for i in range(len(x)):
        while len(st) >= 2:
            p, q = st[-2], st[-1]
            if (x[q]-x[p])*(y[i]-y[p]) - (y[q]-y[p])*(x[i]-x[p]) <= 1e-15:
                st.pop()
            else:
                break
        st.append(i)
    return o[np.array(st)]


def eps_hull(a, b, eps, chis):
    """Vertices whose deletion raises min cost by more than eps somewhere."""
    V = list(hull(a, b))
    base = np.array([np.min(a + b * c) for c in chis])
    keep = []
    for v in V:
        m = np.ones(len(a), bool)
        m[v] = False
        without = np.array([np.min(a[m] + b[m] * c) for c in chis])
        gain = (without - base) / base            # relative cost rise if removed
        if gain.max() > eps:
            keep.append(v)
    return keep


def run(name, ld):
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=0)
    pool = np.concatenate([cal, val])
    n = len(pool)
    chis = np.geomspace(0.1, 100.0, 400)
    print("=" * 76)
    print("%-9s n=%d   noise floor chi/n at chi=9 is %.4f" % (name, n, 9.0 / n))
    print("   grid    raw hull   eps-hull (eps = chi/n, chi taken at each vertex's best)")
    for ng in (100, 200, 300, 400, 800):
        a, b = coords(life, pool, R, rm, pa, Tb, ng)
        raw = len(hull(a, b))
        # noise floor: use the lattice spacing at the cost ratio where the vertex wins
        kept = eps_hull(a, b, 9.0 / n, chis)
        print("   %4d      %3d        %3d" % (ng, raw, len(kept)))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("Severson", M.load_severson)):
        run(nm, ld)
