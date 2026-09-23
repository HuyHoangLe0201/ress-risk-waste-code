"""Operating under an UNKNOWN cost ratio, not merely a random one.

Proposition 15 handles chi with a distribution: the expected cost is the cost at the
expected chi, because the objective is linear in chi.  It says nothing about the case
an engineer actually faces, where chi is known only to lie in a range and no
distribution over it is credible.  That is the minimax-regret problem, and the
geometry already in the paper solves it exactly.

Regret of policy p at chi is R_p(chi) = (a_p + b_p chi) - L*(chi): an affine function
minus a concave one, hence CONVEX in chi.  A convex function on an interval attains
its maximum at an endpoint.  So

    max_{chi in [c1,c2]} R_p(chi) = max( R_p(c1), R_p(c2) ),

the worst case never lies inside the range, and minimax regret is a two-point problem.
Consequences to test:

  (a) convexity, hence endpoint attainment, on a fine grid;
  (b) the minimax policy is a lower-hull vertex lying between the vertices optimal at
      the two endpoints, so it is found by scanning that sub-chain alone;
  (c) minimax regret is zero exactly when one policy is optimal at both endpoints,
      which is what ties this to the interval cover of Section 4.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402


def achievable(ld, n_thr=200):
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=0)
    idx = np.concatenate([cal, val])
    paths = [np.asarray(pa[i], float) for i in idx]
    n = np.array([len(p) for p in paths], float)
    Tb = float(n.mean())
    lo = min(p.min() for p in paths)
    hi = max(p.max() for p in paths)
    A = []
    for e in np.linspace(lo, hi, n_thr):
        w, f = [], []
        for p in paths:
            h = np.nonzero(p <= e)[0]
            if len(h) == 0:
                w.append(0.0); f.append(1.0)
            else:
                w.append(float(len(p) - 1 - h[0])); f.append(0.0)
        A.append((float(np.mean(w)) / Tb, float(np.mean(f))))
    return np.array(A)


def duals(A):
    om, pf = A[:, 0], A[:, 1]
    keep = om < 1.0
    a = 1.0 / (1.0 - om[keep])
    b = pf[keep] / (1.0 - om[keep])
    return a, b


def lower_hull(a, b):
    """Vertices of the lower envelope of lines L(chi)=a+b*chi, by dual lower hull."""
    o = np.lexsort((a, b))
    h = []
    for i in o:
        while len(h) >= 2:
            (b1, a1), (b2, a2) = h[-2], h[-1]
            # drop h[-1] if it is not below the segment h[-2] -> i
            if (b2 - b1) * (a[i] - a1) - (a2 - a1) * (b[i] - b1) <= 0:
                h.pop()
            else:
                break
        h.append((b[i], a[i]))
    return np.array(h)


def Lstar(chi, a, b):
    return np.min(a[None, :] + b[None, :] * np.atleast_1d(chi)[:, None], axis=1)


def run(name, ld):
    A = achievable(ld)
    a, b = duals(A)
    H = lower_hull(a, b)
    print("=" * 78)
    print("%s: %d policies, %d envelope vertices" % (name, len(a), len(H)))

    grid = np.linspace(0.5, 60.0, 4000)
    Lg = Lstar(grid, a, b)

    worst_inside = 0
    rows = []
    for c1, c2 in ((1.0, 5.0), (2.0, 20.0), (3.0, 12.0), (5.0, 50.0), (8.0, 10.0)):
        k = (grid >= c1) & (grid <= c2)
        cg, Lk = grid[k], Lg[k]
        Rall = (a[None, :] + b[None, :] * cg[:, None]) - Lk[:, None]   # regret
        gridmax = Rall.max(axis=0)
        endmax = np.maximum(Rall[0], Rall[-1])
        worst_inside += int(np.sum(gridmax > endmax + 1e-9))

        j = int(np.argmin(endmax))
        # vertices optimal at each endpoint
        j1 = int(np.argmin(a + b * c1))
        j2 = int(np.argmin(a + b * c2))
        lo_b, hi_b = sorted((b[j1], b[j2]))
        between = lo_b - 1e-12 <= b[j] <= hi_b + 1e-12
        on_hull = np.any(np.all(np.isclose(H, [b[j], a[j]], atol=1e-12), axis=1))
        same = (j1 == j2)
        rows.append((c1, c2, endmax[j], Rall[0][j], Rall[-1][j],
                     between, on_hull, same,
                     "lo" if j == j1 else ("hi" if j == j2 else "INTERIOR")))

    print("    policies whose worst case falls strictly inside the range: %d"
          % worst_inside)
    print("%12s %10s %10s %10s %7s %7s %10s" %
          ("range", "minimax R", "R at lo", "R at hi", "betwn", "vertex",
           "which policy"))
    for c1, c2, mr, r1, r2, bt, oh, sm, wh in rows:
        print("[%5.1f,%5.1f] %10.5f %10.5f %10.5f %7s %7s %10s"
              % (c1, c2, mr, r1, r2, bt, oh, wh))
        assert (mr < 1e-9) == sm, "zero-regret characterisation failed"


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
