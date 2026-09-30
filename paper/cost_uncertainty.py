"""What if the cost ratio is not known?

"We do not know the failure cost" is the standard objection to any cost-based
maintenance method, and in this plane it has a three-line answer, because the cost
is LINEAR in chi:

    E_chi[(1 + chi P_f)/(1 - omega)]  =  (1 + E[chi] P_f)/(1 - omega),

so a risk-neutral decision maker who knows only the MEAN of the failure cost loses
nothing at all relative to one who knows its whole distribution.  And since the
cost increases in chi, a decision maker who wants the worst case over an interval
optimises at its upper end.  Both the Bayesian and the robust formulation collapse
to a single effective cost ratio, and the plane says which one.

The identity needs no checking.  What is worth computing is the consequence: for a
decision maker uncertain over a range, which policy do the two criteria select,
and do they differ?  With chi* around 2.5 on FD002 the answer can flip inside quite
ordinary ranges of uncertainty.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

NGRID = 300


def family(ld, seed=0):
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=seed)
    idx = cal
    lu = np.array([pa[i][R[i] > 0].min() for i in idx])
    g = np.linspace(np.quantile(lu, .02), lu.max() + .25 * Tb, NGRID)
    C = np.array([[M.unit_cost(i, e, R, rm, life) for i in idx] for e in g])
    return g, 1.0 - C[:, :, 1].mean(1) / Tb, C[:, :, 2].mean(1), len(idx)


def pick(om, pf, chi):
    return int(np.argmin((1.0 + chi * pf) / (1.0 - om)))


def run(name, ld):
    g, om, pf, n = family(ld)
    z = np.nonzero(pf <= 1e-12)[0]
    kink = int(z.min()) if len(z) else -1
    print("=" * 78)
    print("%-9s n=%d   kink at grid index %d" % (name, n, kink))
    print("   uncertainty range      mean-chi rule        worst-case rule      same?")
    for lo, hi in ((1.0, 4.0), (2.0, 20.0), (1.0, 10.0), (5.0, 50.0), (0.5, 2.0)):
        mean_chi = 0.5 * (lo + hi)                 # uniform on the range
        jm, jw = pick(om, pf, mean_chi), pick(om, pf, hi)
        tag = lambda j: ("kink" if j == kink else "P_f=%.3f" % pf[j])
        print("   chi in [%4.1f,%5.1f]   E[chi]=%5.1f -> %-10s  chi_hi -> %-10s  %s"
              % (lo, hi, mean_chi, tag(jm), tag(jw), "yes" if jm == jw else "NO"))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
