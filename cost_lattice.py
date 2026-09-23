"""Is there a resolution limit that is not about noise at all?

Every limit in this paper so far is a MEASUREMENT limit: the estimate is noisy,
the interval is wide, the coverage is short.  There may be a different and harder
one underneath.  On a fleet of n units the failure probability can only take the
values k/n, so in the P_f direction the achievable operating points form a
lattice, and by the sensitivity relation the costs they generate are separated by

    Delta L / L  =  chi / (n (1 + chi P_f))  ~  chi/n  near the boundary.

If that holds, then two policies whose costs differ by less than one lattice step
are not hard to tell apart -- they are the same point, or adjacent ones, and the
difference between them does not exist as a distinct operating point on that
fleet.  That is an EXISTENCE limit rather than an estimation limit, it is exact,
and it needs no distributional assumption.

Checked by enumerating the admissible vertices of the threshold family near the
boundary and measuring the cost gaps between consecutive ones.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

NGRID = 400


def run(name, ld, chi=None):
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=0)
    idx = cal
    n = len(idx)
    chi = M.CF - 1.0 if chi is None else chi

    lu = np.array([pa[i][R[i] > 0].min() for i in idx])
    g = np.linspace(np.quantile(lu, .02), lu.max() + .25 * Tb, NGRID)
    C = np.array([[M.unit_cost(i, e, R, rm, life) for i in idx] for e in g])
    om = 1.0 - C[:, :, 1].mean(1) / Tb
    pf = C[:, :, 2].mean(1)
    L = (1.0 + chi * pf) / (1.0 - om)

    # distinct achievable P_f levels, and the cost at the best point of each level
    lev = np.unique(np.round(pf * n).astype(int))
    best = []
    for k in lev:
        m = np.abs(pf * n - k) < 0.5
        if m.any():
            best.append((k, float(L[m].min())))
    best.sort()
    k0 = best[0][0]
    print("=" * 78)
    print("%-9s n=%d  chi=%.0f   distinct P_f levels reachable: %d"
          % (name, n, chi, len(best)))
    print("   k (failures) :" + "".join("%8d" % k for k, _ in best[:7]))
    print("   best cost    :" + "".join("%8.4f" % v for _, v in best[:7]))
    gaps = [(best[i + 1][1] - best[i][1]) / best[i][1] for i in range(min(5, len(best) - 1))]
    print("   relative cost gap between consecutive levels: %s"
          % "".join("%7.3f" % v for v in gaps))
    print("   predicted chi/n = %.4f      median gap = %.4f"
          % (chi / n, float(np.median(np.abs(gaps)))))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
    print()
    print("resolution chi/n for the fleets in the literature table")
    for who, n, chi in (("Mitici et al. [15]", 50, 4.0),
                        ("Lee and Mitici [16]", 130, 1.0),
                        ("Ziyad et al. [17]", 30, 0.14),
                        ("Kamariotis et al. [6]", 20, 9.0),
                        ("Kamariotis et al. [6]", 20, 99.0)):
        print("   %-24s n=%3d  chi=%5.2f  ->  cost resolution %5.1f%%"
              % (who, n, chi, 100 * chi / n))
