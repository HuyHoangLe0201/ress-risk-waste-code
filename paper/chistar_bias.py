"""Does the admissibility test fail safe or fail dangerous?

Guideline G6 tells the reader to compute chi* on calibration data and adopt the
zero-failure boundary when the cost ratio clears it.  chi* is a MAXIMUM over the
failure branch of a ratio whose denominator is P_f, and on n units the failure
branch is resolved only to 1/n.  A smaller fleet therefore offers fewer candidate
secants and a coarser one near the kink, so the maximum it attains is smaller.
That predicts chi_hat* below the population value, and it predicts the direction
of the resulting decision error:

    chi_hat* too small  =>  "chi >= chi_hat*" fires too often
                        =>  the boundary is adopted when it is NOT optimal.

That is the dangerous direction: on a small fleet the test says the shortcut is
safe when it is not.  If instead chi_hat* came out too large the test would merely
be conservative.  Measured here against a population value, with the predictor
held fixed so that only the evaluation fleet varies.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

NGRID = 300
DRAWS = 200


def chistar(life, idx, R, rm, pa, Tb):
    lu = np.array([pa[i][R[i] > 0].min() for i in idx])
    g = np.linspace(np.quantile(lu, .02), lu.max() + .30 * Tb, NGRID)
    C = np.array([[M.unit_cost(i, e, R, rm, life) for i in idx] for e in g])
    om = 1.0 - C[:, :, 1].mean(1) / Tb
    pf = C[:, :, 2].mean(1)
    z = np.nonzero(pf <= 1e-12)[0]
    if not len(z) or z.min() == 0:
        return np.nan
    k = int(z.min())
    b = np.arange(k)
    o0 = om[k]
    return float(((o0 - om[b]) / (pf[b] * (1 - o0))).max())


def run(name, ld):
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=0)
    pool = np.concatenate([cal, val])
    pop = chistar(life, pool, R, rm, pa, Tb)     # the population value
    rng = np.random.default_rng(17)

    print("=" * 78)
    print("%-9s population chi* on all %d units = %.2f" % (name, len(pool), pop))
    print("   n     median chi_hat*   P(chi_hat* < pop)   dangerous-error rate at chi =")
    print("                                              %s"
          % "".join("%7.1f" % c for c in (1.0, 2.0, 5.0, 9.0)))
    for n in (40, 60, 90, 130, min(200, len(pool))):
        if n > len(pool):
            continue
        vals = []
        for _ in range(DRAWS):
            sub = rng.choice(pool, size=n, replace=False)
            v = chistar(life, sub, R, rm, pa, Tb)
            if np.isfinite(v):
                vals.append(v)
        v = np.array(vals)
        if not len(v):
            continue
        under = float((v < pop).mean())
        # dangerous error: test says adopt (chi >= chi_hat*) but truly chi < pop
        errs = []
        for c in (1.0, 2.0, 5.0, 9.0):
            says_adopt = v <= c
            truly_ok = c >= pop
            errs.append(float((says_adopt & (not truly_ok)).mean()))
        print("   %-5d %9.2f       %10.2f        %s"
              % (n, np.median(v), under, "".join("%7.2f" % e for e in errs)))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
