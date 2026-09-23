"""If the zero-failure point cannot be estimated, can a small failure quota be?

Proposition 8 puts omega_0 in the unestimable class because it is anchored at
ell_0 = max_u ell_u, and the previous experiment showed the standard extreme-value repair
is worse than the disease.  There is another way out, and it does not need a better
estimator at all: change the operating point.

At a failure quota alpha > 0 the anchor is the (1-alpha) quantile of {ell_u}, an INTERIOR
order statistic rather than the maximum, and interior quantiles are ordinary
n^{-1/2} objects.  So omega_alpha should be estimable where omega_0 is not, and the ceiling
quoted at a quota, (1 + chi*alpha)/(1 - omega_alpha), should behave like an average.

Two things to measure: whether the bias collapses as alpha moves off zero, and where the
usable range of alpha begins.  It cannot begin below 1/n, since P_f only takes values k/n
on n units -- the lattice sets the smallest quota that means anything.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402


def prep(ld):
    seq, life = ld()
    R, rm, pa, cal, val, _ = M.prep(seq, life, seed=0)
    idx = np.concatenate([cal, val])
    paths = [np.minimum.accumulate(np.asarray(pa[i], float)) for i in idx]
    Rs = [np.asarray(R[i], float) for i in idx]
    lives = np.array([float(len(p)) for p in paths])
    lu = np.array([float(paths[i][Rs[i] > 0].min()) for i in range(len(paths))])
    return paths, Rs, lives, lu


def omega_at(paths, Rs, lives, sub, ell):
    w = []
    for i in sub:
        h = np.nonzero(paths[i] <= ell)[0]
        w.append(float(Rs[i][h[0]]) if len(h) and Rs[i][h[0]] > 0 else 0.0)
    return float(np.mean(w)) / float(lives[sub].mean())


def anchor(lu_sub, alpha):
    """Threshold admitting a fraction alpha of failures: the (1-alpha) quantile."""
    return float(np.quantile(lu_sub, 1.0 - alpha))


def run(name, ld, reps=400, seed=0):
    paths, Rs, lives, lu = prep(ld)
    N = len(paths)
    rng = np.random.default_rng(seed)
    alphas = [0.0, 0.01, 0.02, 0.05, 0.10]
    sizes = [s for s in (30, 60, 100, 160) if s <= N]

    print("=" * 78)
    print("%s   N=%d" % (name, N))
    print("%7s" % "n" + "".join("  alpha=%-6.2f" % a for a in alphas))
    for s in sizes:
        row = []
        for a in alphas:
            full = omega_at(paths, Rs, lives, np.arange(N), anchor(lu, a))
            v = []
            for _ in range(reps):
                sub = rng.choice(N, s, replace=False)
                v.append(omega_at(paths, Rs, lives, sub, anchor(lu[sub], a)))
            rel = 100.0 * (float(np.mean(v)) - full) / full
            # alpha = 0 is always computable -- it is the maximum, and it is the
            # contrast this table exists to draw; only a positive quota below the
            # lattice step is meaningless
            usable = (a == 0.0) or (a >= 1.0 / s)
            row.append("%+8.1f%%  " % rel if usable else "     n/a  ")
        print("%7d" % s + "".join(row))
    print("   entries are relative bias; n/a where the quota is below the "
          "lattice step 1/n")


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
