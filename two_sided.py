"""The missing half: when does an operating point beat the points ABOVE it?

Theorem 5 and Proposition 10 are both one-sided.  Each says when a point beats the points
below it -- those admitting more failures -- and neither says anything about the points
above it, those admitting fewer.  For the zero-failure point that is enough, because nothing
lies above it.  For every interior point it is not, and the previous experiment showed the
consequence: a quota point can satisfy its own certificate and still be beaten by the
zero-failure point.

The same inequality supplies the other half.  The alpha-point beats (omega, P_f) when

    omega_alpha - omega <= chi [ P_f (1-omega_alpha) - alpha (1-omega) ].

Below the anchor the bracket is positive and dividing gives a LOWER bound chi*_alpha.
Above the anchor, where P_f < alpha and omega > omega_alpha, both sides are negative and
dividing flips the inequality, giving an UPPER bound.  So the alpha-point is globally
optimal exactly on an interval [chi*_alpha, chi**_alpha], and Theorem 5 is the case where
the upper end is infinite because the branch above is empty.

Checked two ways: that the interval is non-empty exactly for the points that are optimal
somewhere, which is admissibility in the sense of Proposition 2, and that the interval
containing a given chi picks out the true minimiser over the whole frontier.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
from quota import prep                                           # noqa: E402


def frontier(paths, Rs, lives, sub, grid):
    om, pf = [], []
    Tb = float(lives[sub].mean())
    for e in grid:
        w, f = [], []
        for i in sub:
            h = np.nonzero(paths[i] <= e)[0]
            if len(h) == 0 or Rs[i][h[0]] <= 0:
                w.append(0.0); f.append(1.0)
            else:
                w.append(float(Rs[i][h[0]])); f.append(0.0)
        om.append(float(np.mean(w)) / Tb)
        pf.append(float(np.mean(f)))
    return np.array(om), np.array(pf)


def interval(om, pf, j, tol=1e-12):
    """[chi*, chi**] on which point j beats every other point of the frontier."""
    oj, pj = om[j], pf[j]
    num = oj - om
    den = pf * (1.0 - oj) - pj * (1.0 - om)
    lo, hi = 0.0, np.inf
    pos = den > tol
    if pos.any():
        lo = max(lo, float(np.max(num[pos] / den[pos])))
    neg = den < -tol
    if neg.any():
        hi = min(hi, float(np.min(num[neg] / den[neg])))
    return lo, hi


def run(name, ld):
    paths, Rs, lives, lu = prep(ld)
    N = len(paths)
    grid = np.linspace(np.quantile(lu, 0.02), lu.max(), 120)
    om, pf = frontier(paths, Rs, lives, np.arange(N), grid)
    keep = om < 1.0
    om, pf = om[keep], pf[keep]

    iv = [interval(om, pf, j) for j in range(len(om))]
    live = [j for j, (a, b) in enumerate(iv) if b > a + 1e-9]

    chis = np.logspace(-2, 2.5, 900)
    L = (1.0 + chis[:, None] * pf[None, :]) / (1.0 - om[None, :])
    argmin = np.argmin(L, axis=1)

    agree = 0
    for t, c in enumerate(chis):
        holders = [j for j in live if iv[j][0] <= c <= iv[j][1]]
        best = argmin[t]
        # the certificate is satisfied by the true minimiser
        if best in holders:
            agree += 1

    print("=" * 74)
    print("%s   %d frontier points, %d with a non-empty interval"
          % (name, len(om), len(live)))
    print("   cost ratios where the true minimiser holds the certificate: %d/%d"
          % (agree, len(chis)))
    # do the intervals of the live points tile the axis without gaps?
    ends = sorted((iv[j][0], iv[j][1]) for j in live)
    gaps = sum(1 for (a1, b1), (a2, b2) in zip(ends[:-1], ends[1:]) if a2 > b1 + 1e-6)
    print("   gaps between consecutive intervals: %d" % gaps)
    print("%10s %12s %12s %10s" % ("P_f", "chi*", "chi**", "width"))
    for j in live[:8]:
        a, b = iv[j]
        print("%10.4f %12.3f %12.3f %10s"
              % (pf[j], a, b, "inf" if not np.isfinite(b) else "%.3f" % (b - a)))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("Severson", M.load_severson)):
        run(nm, ld)
