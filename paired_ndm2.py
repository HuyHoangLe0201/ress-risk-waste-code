r"""Where the paired correction works, and where its premise fails instead.

paired_ndm.py applies the numerical delta method to the paired ratio and finds two
things that need explaining before either can be reported.

First, the closed-form derivative disagreed with the numerical one.  That was a fault
in the check, not in either derivative: the closed form was evaluated at the SAMPLE
argmin, which is a singleton, and a singleton argmin makes the derivative linear --
which is exactly the plug-in the correction exists to avoid.  The argmin set has to be
widened to the near-ties the sampling error can reach, K0 = {k : theta_k <= min + c_n}
with c_n vanishing more slowly than n^{-1/2}.  With the set widened the two derivations
agree, and the disagreement at c_n = 0 is a measurement of how far the argmin moves.

Second, the interval width grew with the cost ratio and became unstable in the step:
forty times the percentile width at chi = 30, and swinging between twenty-six and
seventy-six as the step changes.  That is not a defect of the paired extension.  By
Theorem 5 the optimum sits on the zero-failure boundary once chi >= chi*, and there the
underlying estimator is an extreme of the sample rather than an average: its limit is
extreme-value, not Gaussian, so the premise of the numerical delta method is absent --
which is the separate failure the unit-level bootstrap section already characterises,
with its own repair.  So the honest answer to whether the step condition holds
uniformly over the swept cost ratios is that it does not, and the reason has nothing to
do with the step.

This run separates the two regimes by measuring where the argmin actually sits.

Run: python paired_ndm2.py
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import os.path as _p; sys.path.append(_p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
from paired_ndm import (pieces, curves, psi, intervals, NGRID, NBOOT)  # noqa: E402


def widened(theta, h, cn, K=NGRID):
    """psi'(h) with argmin sets widened to the near-ties sampling error can reach."""
    c, a = theta[:K], theta[K:]
    A, B = c.min(), a.min()
    K0 = np.nonzero(c <= A + cn * abs(A))[0]
    J0 = np.nonzero(a <= B + cn * abs(B))[0]
    return (h[:K][K0].min() - (A / B) * h[K:][J0].min()) / B


def run(name, ld, chis=(0.5, 1.0, 1.5, 2.0, 3.0, 9.0), seed=0):
    cyc, fail, acyc, afail, n = pieces(ld, seed)
    rng = np.random.default_rng(7)
    B = rng.integers(0, n, size=(NBOOT, n))
    fbar_all = fail.mean(1)
    print("=" * 100)
    print("%s  n=%d" % (name, n))
    print("   chi    ratio   P_f at the CBM optimum   at boundary?  ndm/pct  "
          "argmin moves  closed form (c_n=n^-1/4)")
    for chi in chis:
        theta = curves(cyc, fail, acyc, afail, chi, np.arange(n))
        star = np.array([curves(cyc, fail, acyc, afail, chi, b) for b in B])
        eps = n ** -0.25
        (pc, nd, p, D, Z) = intervals(theta, star, n, psi, eps)
        k = int(theta[:NGRID].argmin())
        pf = float(fbar_all[k])
        moves = float(np.mean(star[:, :NGRID].argmin(1) != k))
        Dw = np.array([widened(theta, z, n ** -0.25) for z in Z])
        corr = float(np.corrcoef(D, Dw)[0, 1])
        print("   %-6.1f %-7.4f %-24.5f %-13s %-8.2f %-13.3f corr %.4f"
              % (chi, p, pf, "yes" if pf <= 1e-12 else "no",
                 (nd[1] - nd[0]) / (pc[1] - pc[0]), moves, corr))

    print("   agreement of the two derivations as the argmin set is widened, chi=1")
    theta = curves(cyc, fail, acyc, afail, 1.0, np.arange(n))
    star = np.array([curves(cyc, fail, acyc, afail, 1.0, b) for b in B])
    Z = np.sqrt(n) * (star - theta)
    p = psi(theta)
    D = np.array([(psi(theta + (n ** -0.25) * z) - p) / (n ** -0.25) for z in Z])
    print("      c_n        |K0|  |J0|  corr(D, closed form)  mean |D-Dw|/|D|")
    for cn in (0.0, 1e-4, 1e-3, n ** -0.5, n ** -0.25, 0.05):
        Dw = np.array([widened(theta, z, cn) for z in Z])
        K0 = int((theta[:NGRID] <= theta[:NGRID].min()
                  + cn * abs(theta[:NGRID].min())).sum())
        J0 = int((theta[NGRID:] <= theta[NGRID:].min()
                  + cn * abs(theta[NGRID:].min())).sum())
        print("      %-10.5f %-5d %-5d %-21.4f %.4f"
              % (cn, K0, J0, float(np.corrcoef(D, Dw)[0, 1]),
                 float(np.mean(np.abs(D - Dw)) / np.mean(np.abs(D)))))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004)):
        run(nm, ld)
