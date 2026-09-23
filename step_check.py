r"""Is the achievable set of the threshold family finite on a finite fleet?

Proposition 4 says L_A is piecewise linear "when A is finite -- as it is on any fleet". The
objection is that a finite fleet does not make A finite: the threshold ell is continuous,
and while P_f(ell) changes only at record values, omega(ell) might drift continuously
between them, giving infinitely many operating points.

Whether it does depends on the construction. The alarm fires at the first t with
M_u(t) <= ell, and M_u is the running minimum of the predicted life: a decreasing STEP
function of t, dropping only at unit u's records. So as ell moves between two consecutive
record levels, no unit's alarm time moves, and neither coordinate moves either -- which
would make A finite after all.

That is a claim about the pipeline as implemented, not about threshold families in general,
so it is measured. Sweep ell on a very fine grid, count distinct (omega, P_f) pairs, and
compare with the number of distinct record values in the fleet. If the two agree, the
coordinates are step functions and the proposition stands as written.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402


def check(name, loader, ngrid=20000):
    seq, life = loader()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=0)

    # every value the running minima actually take on the calibration units: the only
    # places an alarm time can change
    records = np.unique(np.concatenate([np.unique(rm[i]) for i in cal]))
    lo, hi = records.min(), records.max()
    grid = np.linspace(lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo), ngrid)

    pairs = []
    for e in grid:
        per = np.array([M.unit_cost(i, e, R, rm, life) for i in cal])
        pairs.append((round(1 - per[:, 1].mean() / Tb, 12),
                      round(per[:, 2].mean(), 12)))
    distinct = len(set(pairs))

    # how many of the grid steps actually moved a coordinate
    moves = sum(1 for a, b in zip(pairs, pairs[1:]) if a != b)
    print("\n%s  (%d calibration units)" % (name, len(cal)))
    print("   distinct running-minimum values in the fleet : %d" % len(records))
    print("   grid points swept                            : %d" % ngrid)
    print("   distinct (omega, P_f) pairs found            : %d" % distinct)
    print("   grid steps at which a coordinate moved       : %d" % moves)
    print("   -> coordinates are step functions of ell     : %s"
          % (distinct < ngrid / 10))


if __name__ == "__main__":
    check("FD002", M.FD002)
    check("Severson", M.load_severson)
