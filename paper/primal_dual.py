"""Are Propositions 2 and 5 the same theorem, seen in the primal and the dual?

The cost of a fixed policy is AFFINE in the cost ratio:

    L(chi) = a + b chi,      a = 1/(1-omega),   b = P_f/(1-omega).

So each policy is a POINT (omega, P_f) in the primal plane and a LINE in the
(chi, L) plane.  Then

    L*(chi) = min_p ( a_p + b_p chi )

is the lower envelope of a line arrangement, and by point-line duality a policy
appears on that envelope exactly when (b_p, a_p) is a vertex of the lower convex
hull of the dual point set.  Proposition 2 says admissible policies are envelope
vertices in the primal; Proposition 5 says L* is concave and piecewise linear in
chi.  If the duality is right these are one statement, the breakpoints of L* are
the indifference cost ratios

    chi_ij = -(a_i - a_j)/(b_i - b_j),

and chi* is simply the abscissa of the LAST breakpoint -- the indifference ratio
between the kink and the final vertex of the failure branch.

Checked by computing the admissible set two ways: sweeping chi, and one convex
hull in the dual.  They must agree exactly.
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
    return 1.0 - C[:, :, 1].mean(1) / Tb, C[:, :, 2].mean(1)


def by_sweep(om, pf, chis):
    a = 1.0 / (1.0 - om)
    b = pf / (1.0 - om)
    return {int(np.argmin(a + b * c)) for c in chis}


def by_dual_hull(om, pf):
    """Lower convex hull of the dual points (b, a); its vertices are admissible."""
    a = 1.0 / (1.0 - om)
    b = pf / (1.0 - om)
    o = np.lexsort((a, b))
    x, y = b[o], a[o]
    st = []
    for i in range(len(x)):
        while len(st) >= 2:
            p, q = st[-2], st[-1]
            cr = (x[q]-x[p])*(y[i]-y[p]) - (y[q]-y[p])*(x[i]-x[p])
            if cr <= 1e-15:
                st.pop()
            else:
                break
        st.append(i)
    return set(o[np.array(st)].tolist()), a, b


def run(name, ld):
    om, pf = family(ld)
    chis = np.geomspace(1e-3, 1e5, 40000)
    S = by_sweep(om, pf, chis)
    H, a, b = by_dual_hull(om, pf)
    # the sweep can only find policies optimal inside the swept range
    print("=" * 74)
    print("%-9s sweep over chi in [1e-3, 1e5] : %d distinct optima" % (name, len(S)))
    print("          dual lower hull vertices    : %d" % len(H))
    print("          sweep set contained in hull : %s" % S.issubset(H))
    # breakpoints = indifference ratios between consecutive hull vertices
    v = sorted(H, key=lambda i: b[i])
    brk = []
    for i, j in zip(v[:-1], v[1:]):
        if abs(b[i] - b[j]) > 1e-15:
            brk.append(-(a[i] - a[j]) / (b[i] - b[j]))
    brk = [c for c in brk if c > 0]
    print("          positive breakpoints        : %d, largest %.3f"
          % (len(brk), max(brk) if brk else float("nan")))
    # chi* from the closed form, for comparison
    z = np.nonzero(pf <= 1e-12)[0]
    if len(z) and z.min() > 0:
        k = int(z.min()); bb = np.arange(k); o0 = om[k]
        chistar = float(((o0 - om[bb]) / (pf[bb] * (1 - o0))).max())
        print("          chi* from eq. (5)           : %.3f   <- compare with the "
              "largest breakpoint" % chistar)


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
