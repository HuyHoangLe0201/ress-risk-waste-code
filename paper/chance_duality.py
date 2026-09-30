"""Are a safety requirement and a cost ratio the same instrument?

Safety-critical maintenance is usually posed as a chance-constrained problem:
minimise cost subject to P_f <= alpha, with alpha handed down by a regulator
rather than priced.  In the plane that is a half-plane cut, and because the
objective is linear-fractional and the cut is linear, the constrained optimum must
be the unconstrained optimum at SOME other cost ratio.  If so:

    every reliability requirement alpha is equivalent to a cost ratio chi(alpha),

so a regulator setting a failure-probability limit is implicitly setting a price
on failure, and the plane says which one.  The map should be computable from the
envelope and monotone decreasing in alpha.

Checked by solving both problems on the real threshold families and matching.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

NGRID = 400


def family(ld, seed=0):
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=seed)
    idx = cal
    lu = np.array([pa[i][R[i] > 0].min() for i in idx])
    g = np.linspace(np.quantile(lu, .02), lu.max() + .25 * Tb, NGRID)
    C = np.array([[M.unit_cost(i, e, R, rm, life) for i in idx] for e in g])
    return 1.0 - C[:, :, 1].mean(1) / Tb, C[:, :, 2].mean(1), len(idx)


def L(om, pf, chi):
    return (1.0 + chi * pf) / (1.0 - om)


def run(name, ld, chi0=9.0):
    om, pf, n = family(ld)
    j_un = int(np.argmin(L(om, pf, chi0)))
    print("=" * 78)
    print("%-9s n=%d, base cost ratio chi=%.0f" % (name, n, chi0))
    print("   alpha      constrained optimum      equivalent chi(alpha)   matches?")
    chis = np.geomspace(0.05, 4000.0, 6000)
    for alpha in (0.20, 0.10, 0.05, 0.02, 0.01, 0.0):
        feas = pf <= alpha + 1e-12
        if not feas.any():
            continue
        j = int(np.argmin(np.where(feas, L(om, pf, chi0), np.inf)))
        binds = "BINDS" if j != j_un else "slack"
        # find the cost ratio at which j is the UNCONSTRAINED optimum
        hit = [c for c in chis if int(np.argmin(L(om, pf, c))) == j]
        if hit:
            lo, hi = min(hit), max(hit)
            ok = "yes"
            eq = "%.2f-%.2f" % (lo, hi) if hi / lo > 1.02 else "%.2f" % lo
        else:
            eq, ok = "none", "NO"
        print("   %-6.2f %s omega=%.3f  P_f=%.4f   %-16s %s"
              % (alpha, binds, om[j], pf[j], eq, ok))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        for c in (0.3, 1.0):
            run(nm, ld, c)
