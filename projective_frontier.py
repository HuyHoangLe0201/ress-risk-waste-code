"""I drew the wrong conclusion last round.  The geometry does govern.

Last round found the Pareto set of (cost, availability) has 49 points of which 15
are envelope vertices, and concluded that Proposition 2's geometry does not govern
the bi-objective problem.  That conclusion was too quick.

Both objectives share the denominator D = (1-omega) + delta_p + (delta_f-delta_p)
P_f, so in homogeneous coordinates the map

    (omega, P_f)  |-->  (L, A) = ( (1 + chi P_f)/D , (1-omega)/D )

is a PROJECTIVE transformation of the plane, given by a 3x3 matrix, and it is a
bijection on the half-plane D > 0 where the problem lives.  Projective maps send
lines to lines.  Hence they send conv(A) to a convex set and vertices to vertices,
and the Pareto frontier of the achievable set is the image of a face of conv(A).

What I actually computed was the Pareto set of the FINITE GRID, not of its hull,
and a finite set can have Pareto points strictly inside its own hull.  Those are
precisely the points a MIXTURE dominates.  So the correct statement reverses
Proposition 2(i) in an interesting way: randomising never helps for one
linear-fractional objective, and for two it strictly helps, because it fills in
the hull whose boundary is the real frontier.

Tested: are the non-vertex Pareto points dominated by mixtures of two grid points?
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
import os.path as _p; sys.path.insert(0, _p.dirname(_p.abspath(__file__)))


def objectives(om, pf, chi, dp, df):
    D = (1.0 - om) + dp + (df - dp) * pf
    return (1.0 + chi * pf) / D, (1.0 - om) / D


def pareto(L, A):
    out = []
    for i in range(len(L)):
        d = (L <= L[i] + 1e-12) & (A >= A[i] - 1e-12)
        d[i] = False
        if not ((L[d] < L[i] - 1e-12) | (A[d] > A[i] + 1e-12)).any():
            out.append(i)
    return np.array(out)


def run(name, ld, chi=0.3, dp=0.02, df=0.30, NG=300):
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=0)
    idx = cal
    lu = np.array([pa[i][R[i] > 0].min() for i in idx])
    g = np.linspace(np.quantile(lu, .02), lu.max() + .25 * Tb, NG)
    C = np.array([[M.unit_cost(i, e, R, rm, life) for i in idx] for e in g])
    om = 1.0 - C[:, :, 1].mean(1) / Tb
    pf = C[:, :, 2].mean(1)

    L, A = objectives(om, pf, chi, dp, df)
    P = pareto(L, A)

    # does a mixture of two grid points dominate each Pareto point?
    # a mixture mixes (omega,P_f) linearly; evaluate the objectives after mixing
    lam = np.linspace(0.02, 0.98, 25)
    dominated = 0
    for i in P:
        beat = False
        for a in range(0, NG, 3):
            for b in range(a + 3, NG, 3):
                mo = np.outer(lam, [om[a]]) .ravel() + np.outer(1 - lam, [om[b]]).ravel()
                mp = lam * pf[a] + (1 - lam) * pf[b]
                mo = lam * om[a] + (1 - lam) * om[b]
                ml, ma = objectives(mo, mp, chi, dp, df)
                if ((ml < L[i] - 1e-9) & (ma > A[i] + 1e-9)).any():
                    beat = True
                    break
            if beat:
                break
        dominated += beat
    print("=" * 74)
    print("%s  chi=%.2f  d_p=%.2f  d_f=%.2f" % (name, chi, dp, df))
    print("   finite-grid Pareto points          : %d" % len(P))
    print("   of those, dominated by a MIXTURE   : %d (%.0f%%)"
          % (dominated, 100 * dominated / max(len(P), 1)))
    print("   survivors, i.e. true frontier points: %d" % (len(P) - dominated))


if __name__ == "__main__":
    run("FD002", M.FD002)
