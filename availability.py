"""Maintenance takes time.  Does the plane still work, and what about availability?

The paper assumes repairs are instantaneous.  They are not, and downtime is the
quantity a reliability audience cares about most after cost.  With a preventive
action taking d_p and a failure repair d_f > d_p, the cycle is min(T,tau) plus a
duration, so with delta = d/Tbar

    Tbar g / c_p = (1 + chi P_f) / [ (1-omega) + delta_p + (delta_f - delta_p) P_f ]
    availability = (1-omega) / [ (1-omega) + delta_p + (delta_f - delta_p) P_f ],

both ratios of AFFINE functions of the same two coordinates.  So the plane still
carries the problem, and it now carries two objectives at once.

Two things to check rather than assert.  (a) Is there a real trade-off, or do cost
and availability improve together?  Cost per unit time can fall when failures rise
if failure repairs are long, because downtime dilutes the rate -- a perverse
effect that is exactly why availability is tracked separately.  (b) Is the Pareto
set contained in the envelope vertices of Proposition 2?  A weighted sum of two
linear-fractional objectives is not linear-fractional, so this may well be FALSE,
and it is tested rather than claimed.
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


def envelope_vertices(om, pf):
    o = np.lexsort((pf, om))
    x, y = om[o], pf[o]
    st = []
    for i in range(len(x)):
        while len(st) >= 2:
            a, b = st[-2], st[-1]
            if (x[b]-x[a])*(y[i]-y[a]) - (y[b]-y[a])*(x[i]-x[a]) <= 1e-15:
                st.pop()
            else:
                break
        st.append(i)
    return set(o[np.array(st)].tolist())


def run(name, ld, chi=9.0):
    om, pf = family(ld)
    verts = envelope_vertices(om, pf)
    print("=" * 76)
    print("%-9s envelope vertices: %d of %d grid points" % (name, len(verts), NGRID))
    for dp, df in ((0.02, 0.10), (0.05, 0.30), (0.02, 0.50)):
        D = (1.0 - om) + dp + (df - dp) * pf
        L = (1.0 + chi * pf) / D
        A = (1.0 - om) / D
        # Pareto set: minimise L, maximise A
        par = []
        for i in range(len(L)):
            dom = (L <= L[i] + 1e-12) & (A >= A[i] - 1e-12)
            dom[i] = False
            if not ((L[dom] < L[i] - 1e-12) | (A[dom] > A[i] + 1e-12)).any():
                par.append(i)
        par = set(par)
        jL, jA = int(np.argmin(L)), int(np.argmax(A))
        print("   d_p=%.2f d_f=%.2f Tbar : Pareto set %3d points, %3d of them "
              "envelope vertices (%3.0f%%)"
              % (dp, df, len(par), len(par & verts), 100 * len(par & verts) / max(len(par), 1)))
        print("        cost optimum at P_f=%.4f (vertex %s); availability optimum "
              "at P_f=%.4f (vertex %s);  do they coincide? %s"
              % (pf[jL], jL in verts, pf[jA], jA in verts, jL == jA))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
