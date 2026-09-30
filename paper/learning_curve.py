r"""Is the expected cost of the selection procedure non-increasing in fleet size?

Algorithm 1 works from about 0.63n distinct units, and the manuscript called what it
returns a floor for phi_n, the expected cost of the procedure run on n. That needs
phi_{0.63n} >= phi_n, which is a statement about the learning curve and is not proved
anywhere in the paper. It is also not true in general: expected risk need not decrease
monotonically with sample size.

One inequality does hold with no assumption at all. The procedure deploys some policy, and
phi_0 is the minimum of the cost curve over all policies, so phi_m >= phi_0 for every m.
The interval is therefore a floor with respect to phi_0 whatever the learning curve does.
Against phi_n it is a conservative surrogate unless monotonicity holds here, and that is
what this measures.

phi_m is estimated as the procedure would be deployed: draw m units, split them in half,
fit the predictor and choose the threshold on one half, then score the resulting policy on
the units that were NOT drawn -- the nearest available stand-in for the population.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

DRAWS = 60


def phi_m(seq, life, m, rng):
    """One draw of the procedure at fleet size m, scored off the drawn units."""
    n = len(seq)
    idx = rng.choice(n, size=m, replace=False)
    out = np.setdiff1d(np.arange(n), idx)
    if len(out) < 10:
        return np.nan
    X = M.design(seq)
    R = [np.arange(len(s) - 1, -1, -1.0) for s in seq]
    Tb = life.mean()
    half = max(m // 2, 5)
    cal, val = idx[:half], idx[half:]
    if len(val) < 3:
        return np.nan
    w = M.fit(np.vstack([X[i] for i in cal]),
              np.concatenate([R[i] for i in cal]))
    pa = {i: X[i] @ w for i in range(n)}
    rm = {i: np.minimum.accumulate(pa[i]) for i in pa}
    grid = np.linspace(0, .4 * Tb, 25)

    def cost(ix, e):
        per = np.array([M.unit_cost(i, e, R, rm, life) for i in ix])
        return per[:, 0].mean() / per[:, 1].mean()

    best = grid[int(np.argmin([cost(cal, e) for e in grid]))]
    return cost(out, best) * Tb          # deployed policy, scored off the fleet drawn


def curve(name, loader):
    seq, life = loader()
    n = len(seq)
    rng = np.random.default_rng(0)
    sizes = [int(round(f * n)) for f in (0.30, 0.40, 0.50, 0.63, 0.75, 0.85)]
    sizes = sorted(set(s for s in sizes if s >= 12))
    print("\n%s  (n = %d)" % (name, n))
    prev, mono = None, True
    for m in sizes:
        vals = np.array([phi_m(seq, life, m, rng) for _ in range(DRAWS)])
        vals = vals[np.isfinite(vals)]
        mu, se = vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))
        flag = ""
        if prev is not None and mu > prev + 1e-12:
            flag = "  <-- rises"
            mono = False
        print("   m = %4d (%.2fn)   phi_m = %.4f  +/- %.4f%s"
              % (m, m / n, mu, se, flag))
        prev = mu
    print("   non-increasing across these sizes: %s" % mono)


if __name__ == "__main__":
    curve("FD002", M.FD002)
    curve("FD004", M.FD004)
    curve("Severson", M.load_severson)
