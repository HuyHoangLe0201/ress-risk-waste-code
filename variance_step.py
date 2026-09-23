"""Can the plane PREDICT the variance step, not just describe it?

Section 4 measures a standard deviation that triples across one grid step at the
optimum and explains it by saying the failure indicator enters.  Lemma 1 should
give the size in closed form.  Differentiating L = (1 + chi P_f)/(1 - omega),

    (1/L) dL/dP_f = chi / (1 + chi P_f),      (1/L) dL/domega = 1/(1 - omega),

so at the zero-failure boundary, where P_f = 0, a change in P_f moves the cost by
chi times as much, in relative terms, as the same change in omega moves it by
1/(1-omega).  On n calibration units the smallest non-zero failure probability is
1/n, so crossing the boundary changes the relative cost by

    chi / n

to first order -- a step depending on the cost ratio and the fleet size only, with
no reference to the predictor or to the lifetime law.

Two things to check.  (a) does chi/n match the measured jump?  (b) the design rule
it implies, n >= chi/delta to hold the step under delta, should be compared with
the fleet size the paper reaches by a completely different route, the n^-0.87
half-width scaling that gives n about 480.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

NBOOT = 600
NGRID = 260


def run(name, ld, seed=0):
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=seed)
    idx = val                                   # score on the held-out half
    n = len(idx)
    chi = M.CF - 1.0

    lu = np.array([pa[i][R[i] > 0].min() for i in cal])
    g = np.linspace(np.quantile(lu, .02), lu.max() + .60 * Tb, NGRID)
    C = np.array([[M.unit_cost(i, e, R, rm, life) for i in idx] for e in g])
    num, den, fail = C[:, :, 0], C[:, :, 1], C[:, :, 2]
    pf = fail.mean(1)
    om = 1.0 - den.mean(1) / Tb

    rng = np.random.default_rng(5)
    B = rng.integers(0, n, size=(NBOOT, n))
    boot = num[:, B].mean(2) / den[:, B].mean(2) * Tb
    sd = 100.0 * boot.std(1) / boot.mean(1)

    z = np.nonzero(pf <= 1e-12)[0]
    if not len(z):
        print("=" * 78)
        print("%-9s n=%d  chi=%.0f : the calibration boundary does not reach zero "
              "failures out of sample (min P_f = %.4f = %.1f/n); the step is not "
              "defined here" % (name, n, chi, pf.min(), pf.min() * n))
        return
    k0 = int(z.min())                            # the zero-failure boundary
    k1 = k0 - 1                                  # one grid step onto the failure side
    while k1 > 0 and pf[k1] <= 1e-12:
        k1 -= 1

    sd0, sd1 = sd[k0], sd[k1]
    step = np.sqrt(max(sd1 ** 2 - sd0 ** 2, 0.0))   # the added deviation, in quadrature
    pred = 100.0 * chi / n * (1.0 / (1.0 + chi * pf[k1]))

    print("=" * 78)
    print("%-9s n=%d (validation half)  chi=%.0f  omega_0=%.3f" % (name, n, chi, om[k0]))
    print("   s.d. at the boundary            : %.2f%%" % sd0)
    print("   s.d. one grid step onto failures: %.2f%%   (P_f there = %.4f = %.1f/n)"
          % (sd1, pf[k1], pf[k1] * n))
    print("   added deviation, in quadrature  : %.2f%%" % step)
    print("   predicted by chi/n              : %.2f%%      ratio %.2f"
          % (pred, step / pred if pred else np.nan))
    for delta in (0.02, 0.05):
        print("   design rule n >= chi/delta for a step under %.0f%% : n >= %.0f"
              % (100 * delta, chi / delta))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
