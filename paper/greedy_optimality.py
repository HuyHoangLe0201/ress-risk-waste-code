"""Is the greedy cover actually optimal?  I asserted it; now check it.

The paper says: "Because L* is a lower envelope of lines, the greedy sweep is
optimal."  The justification would be that each policy covers an INTERVAL of cost
ratios, since covering a line by fewest intervals is the classic greedy problem.
Each policy's covered set is

    { chi : a_p + b_p chi <= (1 + chi/n) L*(chi) }.

If the tolerance were constant this is convex-in-chi and hence an interval, because
L* is concave.  But the tolerance is chi/n, so the right-hand side is a product of
an increasing affine function with a concave one, whose second derivative is
(2/n) L*' + (1 + chi/n) L*'' -- positive on each linear piece of L* and negative at
its kinks.  Neither convex nor concave.  So the covered set need not be an
interval and greedy need not be optimal.

Two checks: are the covered sets intervals in fact, and does greedy match a
brute-force minimum cover?  If either fails the claim in the paper is wrong and
the count of three may be wrong with it.
"""
import itertools
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import os.path as _p; sys.path.insert(0, _p.dirname(_p.abspath(__file__)))
import makefigs6 as M                                            # noqa: E402
from cover import coords, greedy_cover                           # noqa: E402

CHIS = np.geomspace(0.2, 60.0, 400)


def covered_sets(a, b, chis, n):
    Lstar = np.array([np.min(a + b * c) for c in chis])
    tol = Lstar * (1.0 + chis / n)
    return (a[:, None] + b[:, None] * chis[None, :]) <= tol[None, :] + 1e-15


def is_interval(row):
    idx = np.nonzero(row)[0]
    return len(idx) == 0 or (idx[-1] - idx[0] + 1) == len(idx)


def brute_force_min(cov, kmax=4):
    m, T = cov.shape
    useful = [i for i in range(m) if cov[i].any()]
    for k in range(1, kmax + 1):
        for combo in itertools.combinations(useful, k):
            if cov[list(combo)].any(0).all():
                return k, combo
    return None, None


def run(name, ld, ngrid=120):
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=0)
    pool = np.concatenate([cal, val])
    n = len(pool)
    a, b, g = coords(life, pool, R, rm, pa, Tb, ngrid)

    cov = covered_sets(a, b, CHIS, n)
    non_int = [i for i in range(len(a)) if not is_interval(cov[i])]
    print("=" * 74)
    print("%-9s n=%d, %d candidate policies" % (name, n, len(a)))
    print("   covered sets that are NOT intervals : %d of %d"
          % (len(non_int), len(a)))
    if non_int:
        i = non_int[0]
        idx = np.nonzero(cov[i])[0]
        gaps = np.nonzero(np.diff(idx) > 1)[0]
        print("      e.g. policy %d covers %d chi-points in %d runs"
              % (i, len(idx), len(gaps) + 1))

    gr = greedy_cover(a, b, CHIS, CHIS / n)
    k, combo = brute_force_min(cov)
    print("   greedy cover size      : %d" % len(gr))
    print("   brute-force minimum    : %s" % (k if k else ">4"))
    print("   greedy optimal?        : %s" % (k is not None and len(gr) == k))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("Severson", M.load_severson)):
        run(nm, ld)
