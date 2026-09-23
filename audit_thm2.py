"""Audit Theorem 4, the paper's second central reduction.

Theorem 4 states two coordinates for a threshold policy:

    P_f(ell) = Pr(ell_u > ell),
    omega(ell) = (1/Tbar) * int_0^inf F_{Lambda(w)}(ell) dw,

where Lambda_u(w) is the running minimum of Rhat_u at the instant the TRUE
remaining life equals w, set to +infinity for w > T_u.  The first is a tail and is
easy.  The second is the one that has never been checked in this session, and it
is half of the paper's threshold reduction.

It should be a layer-cake identity: 1{Lambda_u(w) <= ell} is the indicator that
the alarm has already fired when the true remaining life is w, so integrating over
w gives exactly the wasted life W_u, and averaging gives omega.  If the identity
is right the integral form and the direct mean must agree to machine precision.
If they do not, either the theorem or the implementation is wrong, and both matter.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402


def run(name, ld, nell=7):
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=0)
    idx = cal
    lu = np.array([pa[i][R[i] > 0].min() for i in idx])
    ells = np.linspace(np.quantile(lu, .10), lu.max() + .15 * Tb, nell)

    print("=" * 78)
    print("%-9s n=%d  Tbar=%.1f" % (name, len(idx), Tb))
    print("%12s %14s %14s %12s %14s" %
          ("ell/Tbar", "omega direct", "omega integral", "P_f direct", "P_f = tail"))
    for e in ells:
        # direct: mean wasted life over the fleet, failures contributing zero
        W, fail = [], []
        for i in idx:
            h = np.nonzero(rm[i] <= e)[0]
            if len(h) == 0 or R[i][h[0]] <= 0:
                W.append(0.0)
                fail.append(1.0)
            else:
                W.append(float(R[i][h[0]]))
                fail.append(0.0)
        om_direct = float(np.mean(W)) / Tb
        pf_direct = float(np.mean(fail))

        # integral form: for each unit, the measure of w with Lambda_u(w) <= ell.
        # Lambda_u(w) is the running min at the cycle whose true remaining life is w,
        # so we sum over cycles with R_u > 0 the indicator that rm has reached ell,
        # each cycle contributing one unit of w.
        acc = 0.0
        for i in idx:
            r = R[i]
            live = r > 0
            reached = rm[i][live] <= e
            acc += float(reached.sum())          # one unit of w per cycle
        om_int = acc / len(idx) / Tb
        pf_tail = float(np.mean(lu > e))

        print("%12.3f %14.6f %14.6f %12.6f %14.6f"
              % (e / Tb, om_direct, om_int, pf_direct, pf_tail))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
