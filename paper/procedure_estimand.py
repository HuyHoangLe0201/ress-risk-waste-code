"""Does Algorithm 1 cover the quantity it actually estimates?

Section 5.5 certified every interval against the population minimum phi_0.  That
is the estimand of the unsplit estimator, not of Algorithm 1.  Algorithm 1
selects a threshold on a calibration sample and scores it on held-out units, so
what it targets is

    psi_0(m) = E_{C ~ P^m} [ theta^P_{ k(C) } ],      k(C) = argmin of the curve on C,

the population cost of the threshold that the procedure picks from m units --- the
value of the procedure, not the value of the best threshold.  psi_0(m) >= phi_0
always, with equality only if selection on m units were perfect.

Two calibration sizes matter and they are not the same number.  On the real data
the protocol selects on n/2 units.  Inside the bootstrap loop a resample holds
about 0.63n distinct units, halved to about 0.32n, so the resampled replicates
imitate a procedure trained on fewer units than the one being reported.  We
compute psi_0 at both and test coverage against each.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

NGRID = 90
REPS = 300
BOOT = 400
PSI_DRAWS = 4000
ALPHA = 0.05


def population(ld):
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=0)
    allu = np.concatenate([cal, val])
    lu = np.array([pa[i][R[i] > 0].min() for i in allu])
    g = np.linspace(np.quantile(lu, .05), np.quantile(lu, .999) + .28 * Tb, NGRID)
    C = np.array([[M.unit_cost(i, e, R, rm, life) for i in allu] for e in g])
    return C[:, :, 0], C[:, :, 1], Tb, len(allu)


def curve(num, den, Tb, idx):
    return num[:, idx].mean(-1) / den[:, idx].mean(-1) * Tb


def psi(num, den, Tb, N, m, rng):
    """Population cost of the threshold the procedure selects from m units."""
    pop = num.mean(1) / den.mean(1) * Tb
    k = curve(num, den, Tb, rng.integers(0, N, size=(PSI_DRAWS, m))).argmin(0)
    return float(pop[k].mean())


def algo1(num, den, Tb, sub, rng):
    """One replicate of Algorithm 1: resample units, split distinct, select, score."""
    n = len(sub)
    draw = sub[rng.integers(0, n, size=n)]
    uniq = np.unique(draw)
    rng.shuffle(uniq)
    half = set(uniq[: len(uniq) // 2].tolist())
    mask = np.array([u in half for u in draw])
    a, b = draw[mask], draw[~mask]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    k = int(curve(num, den, Tb, a).argmin())
    return curve(num, den, Tb, b)[k]


def run(name, ld, nprime):
    num, den, Tb, N = population(ld)
    pop = num.mean(1) / den.mean(1) * Tb
    phi0 = float(pop.min())
    rng = np.random.default_rng(23)

    m_rep = nprime // 2                      # what the reported estimate trains on
    m_boot = int(round(0.316 * nprime))      # what a resample's calibration half holds
    psi_rep = psi(num, den, Tb, N, m_rep, rng)
    psi_boot = psi(num, den, Tb, N, m_boot, rng)

    cov = np.zeros(3)
    wid = est = 0.0
    for _ in range(REPS):
        sub = rng.integers(0, N, size=nprime)
        h = np.array([algo1(num, den, Tb, sub, rng) for _ in range(BOOT)])
        h = h[np.isfinite(h)]
        lo, hi = np.quantile(h, [ALPHA / 2, 1 - ALPHA / 2])
        for j, tgt in enumerate((phi0, psi_rep, psi_boot)):
            cov[j] += (lo <= tgt <= hi)
        wid += hi - lo
        est += float(np.median(h))
    cov, wid, est = cov / REPS, wid / REPS, est / REPS

    print("=" * 80)
    print("%-9s N=%d  sub-fleet n=%d  %d sub-fleets x %d resamples  nominal 95%%"
          % (name, N, nprime, REPS, BOOT))
    print("   population minimum          phi_0        = %.4f" % phi0)
    print("   procedure at m=%-3d (reported) psi_0       = %.4f  (+%.1f%% over phi_0)"
          % (m_rep, psi_rep, 100 * (psi_rep - phi0) / phi0))
    print("   procedure at m=%-3d (resample) psi_0       = %.4f  (+%.1f%% over phi_0)"
          % (m_boot, psi_boot, 100 * (psi_boot - phi0) / phi0))
    print("   Algorithm 1 interval: width %.1f%% of phi_0, median estimate %.4f"
          % (100 * wid / phi0, est))
    print("   coverage of phi_0     %5.1f%%   <- Section 5.5 reported this")
    print("   coverage of psi_0(%-3d) %5.1f%%   <- its own estimand, as reported"
          % (m_rep, 100 * cov[1]))
    print("   coverage of psi_0(%-3d) %5.1f%%   <- its own estimand, as resampled"
          % (m_boot, 100 * cov[2]))
    print("   (phi_0 line: %.1f%%)" % (100 * cov[0]))


if __name__ == "__main__":
    for nm, ld, npr in (("FD002", M.FD002, 100), ("FD004", M.FD004, 100),
                        ("Severson", M.load_severson, 80)):
        run(nm, ld, npr)
