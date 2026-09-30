r"""Is the instability at high cost ratios the boundary atom, or a coincidence?

paired_ndm2.py shows the paired correction blowing up once the optimum reaches the
zero-failure boundary, and the fraction of resamples in which the optimum MOVES sits at
0.367 on FD002 -- which is 1-(1-1/n)^n to three decimals, the bootstrap atom of the
maximum.  A matching number is not a mechanism, and 0.367 is close enough to several
other quantities to be worth distrusting, so this checks the mechanism directly instead
of the coincidence.

At the boundary the optimal threshold is ell_0 = max_u ell_u, carried by ONE unit.  A
resample either contains that unit or does not.  If the mechanism is the atom, then

    the optimum moves  <==>  the argmax unit is absent from the resample,

resample by resample, not merely in aggregate.  That is a per-draw prediction with no
free constant, and it either matches or it does not.  The aggregate rate is reported
beside it only as a secondary check against (1-1/n)^n.

Run: python paired_ndm3.py
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import os.path as _p; sys.path.append(_p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
from paired_ndm import pieces, curves, NGRID, NBOOT              # noqa: E402


def run(name, ld, chis=(0.5, 3.0, 9.0, 30.0), seed=0):
    cyc, fail, acyc, afail, n = pieces(ld, seed)
    rng = np.random.default_rng(7)
    B = rng.integers(0, n, size=(NBOOT, n))
    fbar = fail.mean(1)
    # the unit that carries the boundary: the last to fail as the threshold rises,
    # i.e. the one whose presence decides whether P_f can reach zero
    present = np.zeros((NBOOT, n), bool)
    for r in range(NBOOT):
        present[r, np.unique(B[r])] = True

    print("=" * 96)
    print("%s  n=%d   predicted atom (1-1/n)^n = %.4f"
          % (name, n, (1 - 1.0 / n) ** n))
    print("   chi    P_f*     optimum moves  argmax unit absent  agree per draw  "
          "|rate - (1-1/n)^n|")
    for chi in chis:
        theta = curves(cyc, fail, acyc, afail, chi, np.arange(n))
        k = int(theta[:NGRID].argmin())
        star_k = np.array([int(curves(cyc, fail, acyc, afail, chi, b)[:NGRID].argmin())
                           for b in B])
        moves = star_k != k
        # which unit fails at the grid point just below the optimum: the carrier
        carrier = None
        if fbar[k] <= 1e-12 and k > 0:
            f_below = fail[k - 1]
            idx = np.nonzero(f_below > 0)[0]
            if len(idx) == 1:
                carrier = int(idx[0])
        if carrier is None:
            print("   %-6.1f %-8.5f %-14.3f %-19s %-15s %s"
                  % (chi, fbar[k], moves.mean(), "no single carrier",
                     "--", "--"))
            continue
        absent = ~present[:, carrier]
        agree = float(np.mean(moves == absent))
        print("   %-6.1f %-8.5f %-14.3f %-19.3f %-15.4f %.4f"
              % (chi, fbar[k], moves.mean(), absent.mean(), agree,
                 abs(moves.mean() - (1 - 1.0 / n) ** n)))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004)):
        run(nm, ld)
