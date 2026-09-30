"""The paired ratio of Section 6, tested against a known population.

Every coverage result so far has been about a single policy's cost.  The paper's
headline claim is a ratio: continuous monitoring attains 0.60-0.70 of the cost of
optimal age replacement.  Ratios can behave far better than their parts, because
a bias shared by numerator and denominator cancels, so the single-policy results
do not settle this and it has to be run.

Population = the empirical distribution of a whole fleet.  Both policies have
their parameter chosen on a calibration sample and are scored on held-out units,
so the estimand is again a procedure value,

    rho_0(m) = E_C[ g_B^P(k_B(C)) ] / E_C[ g_A^P(t_A(C)) ],

the population cost of the threshold the procedure picks, over the population
cost of the replacement age it picks, both from m units.  We test whether the
Algorithm 1 interval covers it, and whether the ratio inherits the conservatism
that the absolute costs showed.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

NGRID = 90
NAGE = 90
REPS = 250
BOOT = 300
RHO_DRAWS = 3000
ALPHA = 0.05
CF = M.CF                                                        # failure cost, c_p = 1


def population(ld):
    """Per-unit numerator and denominator for both policy families, one grid each."""
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=0)
    allu = np.concatenate([cal, val])
    T = life[allu].astype(float)

    lu = np.array([pa[i][R[i] > 0].min() for i in allu])
    gB = np.linspace(np.quantile(lu, .05), np.quantile(lu, .999) + .28 * Tb, NGRID)
    CB = np.array([[M.unit_cost(i, e, R, rm, life) for i in allu] for e in gB])

    # age replacement: preventive at t if the unit survives it, else failure at T
    gA = np.linspace(np.quantile(T, .02), T.max(), NAGE)
    surv = T[None, :] > gA[:, None]
    numA = np.where(surv, 1.0, CF)
    denA = np.where(surv, gA[:, None], T[None, :])
    return CB[:, :, 0], CB[:, :, 1], numA, denA, len(allu)


def rate(num, den, idx):
    return num[:, idx].mean(-1) / den[:, idx].mean(-1)


def rho(numB, denB, numA, denA, N, m, rng):
    """Population ratio delivered by the procedure trained on m units."""
    popB = numB.mean(1) / denB.mean(1)
    popA = numA.mean(1) / denA.mean(1)
    C = rng.integers(0, N, size=(RHO_DRAWS, m))
    kB = rate(numB, denB, C).argmin(0)
    kA = rate(numA, denA, C).argmin(0)
    return float(popB[kB].mean() / popA[kA].mean())


def algo1_ratio(numB, denB, numA, denA, sub, rng):
    n = len(sub)
    draw = sub[rng.integers(0, n, size=n)]
    uniq = np.unique(draw)
    rng.shuffle(uniq)
    half = set(uniq[: len(uniq) // 2].tolist())
    mask = np.array([u in half for u in draw])
    a, b = draw[mask], draw[~mask]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    kB = int(rate(numB, denB, a).argmin())
    kA = int(rate(numA, denA, a).argmin())
    return rate(numB, denB, b)[kB] / rate(numA, denA, b)[kA]


def run(name, ld, nprime):
    numB, denB, numA, denA, N = population(ld)
    popB = numB.mean(1) / denB.mean(1)
    popA = numA.mean(1) / denA.mean(1)
    rho_oracle = float(popB.min() / popA.min())
    rng = np.random.default_rng(31)

    m_rep = nprime // 2
    m_boot = int(round(0.316 * nprime))
    rho_rep = rho(numB, denB, numA, denA, N, m_rep, rng)
    rho_boot = rho(numB, denB, numA, denA, N, m_boot, rng)

    cov = np.zeros(3)
    wid = est = 0.0
    for _ in range(REPS):
        sub = rng.integers(0, N, size=nprime)
        h = np.array([algo1_ratio(numB, denB, numA, denA, sub, rng)
                      for _ in range(BOOT)])
        h = h[np.isfinite(h)]
        lo, hi = np.quantile(h, [ALPHA / 2, 1 - ALPHA / 2])
        for j, tgt in enumerate((rho_oracle, rho_rep, rho_boot)):
            cov[j] += (lo <= tgt <= hi)
        wid += hi - lo
        est += float(np.median(h))
    cov, wid, est = cov / REPS, wid / REPS, est / REPS

    print("=" * 80)
    print("%-9s N=%d  n=%d  %d sub-fleets x %d resamples  nominal 95%%"
          % (name, N, nprime, REPS, BOOT))
    print("   oracle ratio (both optima known)      rho = %.4f" % rho_oracle)
    print("   procedure ratio at m=%-3d (reported)   rho = %.4f" % (m_rep, rho_rep))
    print("   procedure ratio at m=%-3d (resample)   rho = %.4f" % (m_boot, rho_boot))
    print("   Algorithm 1: median %.4f, width %.4f (%.0f%% of rho)"
          % (est, wid, 100 * wid / rho_rep))
    print("   coverage   oracle %5.1f%%   reported-m %5.1f%%   resample-m %5.1f%%"
          % (100 * cov[0], 100 * cov[1], 100 * cov[2]))


if __name__ == "__main__":
    for nm, ld, npr in (("FD002", M.FD002, 100), ("FD004", M.FD004, 100),
                        ("Severson", M.load_severson, 80)):
        run(nm, ld, npr)
