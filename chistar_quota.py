"""Does the quota fix chi-star as well as omega_0, or only omega_0?

G9 says to quote the ceiling at a failure quota alpha rather than at zero, because the
anchor then becomes an interior quantile.  But Theorem 5's constant is defined AT the
zero-failure point, so the recommendation is incomplete unless chi-star has a quota form
and that form behaves better.

The generalisation is the same secant argument with the anchor moved.  The alpha-quota
point (omega_alpha, alpha) beats a point (omega, P_f) exactly when

    (1 + chi*alpha)/(1 - omega_alpha) <= (1 + chi*P_f)/(1 - omega),

which rearranges to chi >= (omega_alpha - omega) / [P_f (1-omega_alpha) - alpha (1-omega)]
wherever that denominator is positive, so chi*_alpha is the maximum of that ratio over the
branch.  At alpha = 0 it collapses to the expression in the paper.

The open question is empirical.  chi-star's drift with fleet size was shown earlier to come
from the resolution of the failure branch and not from omega_0, so moving the anchor may fix
nothing.  If chi*_alpha still drifts, the honest statement is that the quota repairs the
ceiling but not the constant, and G9 has to say so.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
from quota import prep, omega_at, anchor                         # noqa: E402


def branch(paths, Rs, lives, sub, grid):
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


def chistar_alpha(om, pf, om_a, alpha):
    """Maximum secant against the alpha-quota anchor; alpha=0 gives the paper's chi*."""
    den = pf * (1.0 - om_a) - alpha * (1.0 - om)
    k = (den > 1e-12) & (om < om_a)
    if not k.any():
        return np.nan
    return float(np.max((om_a - om[k]) / den[k]))


def run(name, ld, reps=250, seed=0):
    paths, Rs, lives, lu = prep(ld)
    N = len(paths)
    GRID = np.linspace(np.quantile(lu, 0.02), lu.max(), 200)   # fixed across all n
    rng = np.random.default_rng(seed)
    alphas = [0.0, 0.05, 0.10]
    sizes = [s for s in (30, 60, 100, 160) if s <= N]

    print("=" * 74)
    print("%s   N=%d" % (name, N))
    print("%7s" % "n" + "".join("   chi*_%.2f" % a for a in alphas))
    for s in sizes:
        row = []
        for a in alphas:
            v = []
            for _ in range(reps):
                sub = rng.choice(N, s, replace=False)
                om, pf = branch(paths, Rs, lives, sub, GRID)
                om_a = omega_at(paths, Rs, lives, sub, anchor(lu[sub], a))
                c = chistar_alpha(om, pf, om_a, a)
                if np.isfinite(c):
                    v.append(c)
            row.append("%11.2f" % float(np.mean(v)))
        print("%7d" % s + "".join(row))
    # the full-fleet reference for each alpha
    om, pf = branch(paths, Rs, lives, np.arange(N), GRID)
    ref = ["%11.2f" % chistar_alpha(om, pf,
                                    omega_at(paths, Rs, lives, np.arange(N),
                                             anchor(lu, a)), a) for a in alphas]
    print("%7s" % "full" + "".join(ref))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
