"""How does the number of DISTINGUISHABLE policies grow with fleet size?

A 260-unit fleet distinguishes three policies at its own resolution.  The design
question is what a larger fleet buys, and it is a different question from the hull
count: the hull grows like n^{1/3} in exact arithmetic, but the cover of equation
(13) is taken at tolerance chi/n, which SHRINKS with n, so the two effects push
opposite ways and the net is not obvious.

Two rough expectations.  Covering a smooth concave function to relative tolerance
eps needs about eps^{-1/2} lines, which would give n^{1/2}; but L* is already
piecewise linear with about n^{1/3} pieces, which caps the cover there.  So the
growth should be somewhere below n^{1/3}, and the measurement decides.

The answer is the design rule this whole line of work has been heading towards:
how large a fleet must be before it can justify k distinct maintenance policies.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import os.path as _p; sys.path.insert(0, _p.dirname(_p.abspath(__file__)))
import makefigs6 as M                                            # noqa: E402
from cover import coords, greedy_cover                           # noqa: E402
from ehull import hull                                           # noqa: E402

CHIS = np.geomspace(0.2, 60.0, 400)
REPS = 10
NGRID = 300


def run(name, ld, sizes):
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=0)
    pool = np.concatenate([cal, val])
    rng = np.random.default_rng(23)
    print("=" * 74)
    print("%-9s  distinguishable policies vs fleet size" % name)
    print("%6s %10s %12s %12s" % ("n", "hull", "cover(chi/n)", "cover(1%)"))
    ns, cv = [], []
    for n in sizes:
        if n > len(pool):
            continue
        hs, c1s, c2s = [], [], []
        for _ in range(REPS):
            sub = rng.choice(pool, size=n, replace=False)
            a, b, g = coords(life, sub, R, rm, pa, Tb, NGRID)
            hs.append(len(hull(a, b)))
            c1s.append(len(greedy_cover(a, b, CHIS, CHIS / n)))
            c2s.append(len(greedy_cover(a, b, CHIS, 0.01)))
        ns.append(n); cv.append(np.median(c1s))
        print("%6d %10.1f %12.1f %12.1f"
              % (n, np.median(hs), np.median(c1s), np.median(c2s)))
    x, y = np.log(ns), np.log(cv)
    A = np.vstack([x, np.ones_like(x)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ c) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    print("   cover exponent in log n : %+.2f  (R2 %.2f)" % (c[0], r2))
    if c[0] > 1e-6:
        for k in (4, 5, 6):
            need = np.exp((np.log(k) - c[1]) / c[0])
            print("      extrapolated fleet needed for %d policies: n = %.0f" % (k, need))


if __name__ == "__main__":
    run("FD002", M.FD002, (40, 60, 90, 130, 180, 260))
    run("FD004", M.FD004, (40, 60, 90, 130, 180, 249))
